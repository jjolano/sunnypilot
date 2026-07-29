"""Integration test for the CustomLongitudinalStack composition (fakes for car evidence)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import Decision
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceInputs
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import (
  CustomLongitudinalStack,
  LongitudinalStackInputs,
  LongitudinalStackResult,
)

DT = 0.05
LIMITS = (-4.0, 2.0)


def lead(d_rel=30.0, v_lead=12.0, v_rel=0.0, y_rel=0.0, status=True, track_id=3, a_lead=0.0):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=a_lead, yRel=y_rel, radarTrackId=track_id, radar=True,
                         modelProb=0.9, aLeadTau=1.0)


def model_path(xs=(0.0, 30.0, 60.0), ys=(0.0, 0.5, 1.0)):
  return SimpleNamespace(position=SimpleNamespace(x=list(xs), y=list(ys)))


class RaisesOnY:
  x = [0.0, 30.0, 60.0]

  @property
  def y(self):
    raise RuntimeError("bad model path")


def malformed_model_path():
  return SimpleNamespace(position=RaisesOnY())


def circular_arc_path(n=16, step=6.0, radius=100.0, sign=1.0):
  """Constant-curvature arc with curvature sign*1/radius."""
  thetas = [i * step / radius for i in range(n)]
  xs = [radius * math.sin(t) for t in thetas]
  ys = [sign * radius * (1.0 - math.cos(t)) for t in thetas]
  return model_path(xs, ys)


def base(**kw):
  d = dict(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, accel_limits=LIMITS,
           research_actuation_allowed=True)
  d.update(kw)
  return LongitudinalStackInputs(**d)


def test_runs_bounded_over_sequence():
  s = CustomLongitudinalStack()
  rng = np.random.default_rng(20260613)
  for _ in range(800):
    r = s.update(base(
      v_ego=float(rng.uniform(0, 30)), seed_a_target=float(rng.uniform(-2, 1.5)),
      leads=(lead(d_rel=float(rng.uniform(8, 60))) if rng.random() > 0.4 else None, None),
      lead_a_target=float(rng.uniform(-3, 0.5)),
      mode=[LongitudinalMode.ACC, LongitudinalMode.E2E, LongitudinalMode.SCC][int(rng.integers(0, 3))],
    ), DT)
    assert isinstance(r, LongitudinalStackResult)
    assert math.isfinite(r.a_target)
    assert LIMITS[0] - 1e-9 <= r.a_target <= LIMITS[1] + 1e-9


def test_acc_cruises_when_clear():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(5):
    r = s.update(base(seed_a_target=0.4, mode=LongitudinalMode.ACC), DT)
  assert r.a_target == 0.4
  assert r.should_stop is False
  assert r.debug["has_lead"] is False


def test_custom_gap_uses_the_mpc_scheduled_follow_time():
  result = CustomLongitudinalStack().update(base(v_ego=20.0, t_follow=1.2), DT)
  assert result.debug["t_follow"] == pytest.approx(1.2)
  assert result.debug["follow_gap"] == pytest.approx(24.0 + 1.25 + (5.0 - 1.25) / 26.0)


def test_route_282_close_pullaway_survives_alternating_radar_ids():
  """Radar ID churn must still yield one explicit launch verdict. With the 5.0 m stop gap
  the verdict comes once the departing lead clears the follow gap (the speedup guard sees
  no excess inside it; the finalizer's displacement crawl release covers that interim)."""
  stack = CustomLongitudinalStack()
  result = None
  sequence = [
    (4.16, 0.00, 697),
    (4.18, 0.00, 713),
    (4.20, 0.05, 697),
    (4.24, 0.15, 713),
    (4.32, 0.35, 697),
    (4.42, 0.45, 713),
    (4.52, 0.52, 697),
    (4.60, 0.60, 713),
  ]
  for _ in range(3):
    for d_rel, v_lead, track_id in sequence:
      result = stack.update(base(
        v_ego=0.0, v_cruise=12.0, seed_a_target=0.0, lead_a_target=0.0,
        leads=(lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_lead, track_id=track_id), None),
        mode=LongitudinalMode.SCC,
      ), DT)
  assert result is not None
  assert result.debug["follow_gap"] == pytest.approx(5.0)

  # Departing lead clears the follow gap while ids keep alternating: verdict must appear.
  for d_rel, v_lead, track_id in [
    (4.80, 0.70, 697),
    (5.05, 0.80, 713),
    (5.30, 0.90, 697),
    (5.60, 1.00, 713),
  ]:
    result = stack.update(base(
      v_ego=0.0, v_cruise=12.0, seed_a_target=0.0, lead_a_target=0.0,
      leads=(lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_lead, track_id=track_id), None),
      mode=LongitudinalMode.SCC,
    ), DT)
  assert result.standstill_release_allowed is True
  assert result.standstill_release_source in ("lead_pullaway", "lead_standstill_launch")


def test_standstill_release_fields_for_no_lead_launch():
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=False, model_stop_distance=None, model_desired_accel=0.2,
  ), DT)
  assert r.standstill_release_allowed is True
  assert r.standstill_release_source == "no_lead_launch"
  assert r.standstill_release_a_target >= 0.15


def test_no_lead_release_requires_runway_confirmed_clear_path():
  s = CustomLongitudinalStack()
  model_stop = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=True, model_stop_distance=8.0, model_desired_accel=-1.0,
  ), DT)
  assert model_stop.standstill_release_allowed is False

  near_stop = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None), mode=LongitudinalMode.E2E,
    model_should_stop=False, model_stop_distance=8.0, model_desired_accel=0.0,
  ), DT)
  assert near_stop.standstill_release_allowed is False


def test_no_lead_release_blocked_by_shadow_lead():
  s = CustomLongitudinalStack()
  for _ in range(12):
    s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                  leads=(lead(d_rel=6.5, v_lead=0.0), None), mode=LongitudinalMode.ACC), DT)
  lost = s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                       leads=(None, None), mode=LongitudinalMode.ACC), DT)
  assert lost.debug["lead_shadow_active"] is True
  assert lost.standstill_release_allowed is False


def test_standstill_release_blocked_by_physical_hazard_and_brake_seed():
  s = CustomLongitudinalStack()
  haz = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, lead_a_target=-0.2,
    leads=(lead(d_rel=6.0, v_lead=0.0), None), mode=LongitudinalMode.SCC,
  ), DT)
  assert haz.standstill_release_allowed is False
  assert haz.standstill_release_source == ""


def _warm_lead_progress_allowed(s: CustomLongitudinalStack, ld):
  for _ in range(12):
    s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.0,
                  leads=(ld, None), mode=LongitudinalMode.ACC), DT)


def test_lead_release_allows_small_positive_evidence(monkeypatch):
  s = CustomLongitudinalStack()
  ld = lead(d_rel=8.0, v_lead=0.5, v_rel=0.5)
  _warm_lead_progress_allowed(s, ld)

  def fake_decide(candidates, mode, accel_limits, sources=None, previous_intent=""):
    return Decision(a_target=0.08, should_stop=False, selected_intent="lead_pullaway", reason="cruise")
  monkeypatch.setattr("openpilot.sunnypilot.custom.longitudinal.stack.decide", fake_decide)

  r = s.update(base(v_ego=0.0, v_cruise=12.0, leads=(ld, None), mode=LongitudinalMode.ACC), DT)
  assert r.standstill_release_allowed is True
  assert r.standstill_release_source == "lead_pullaway"
  assert r.standstill_release_a_target == pytest.approx(0.15)


def test_no_lead_release_keeps_stronger_evidence_threshold(monkeypatch):
  def fake_decide(candidates, mode, accel_limits, sources=None, previous_intent=""):
    return Decision(a_target=0.08, should_stop=False, selected_intent="no_lead_launch", reason="cruise")
  monkeypatch.setattr("openpilot.sunnypilot.custom.longitudinal.stack.decide", fake_decide)

  s = CustomLongitudinalStack()
  r = s.update(base(v_ego=0.0, v_cruise=12.0, leads=(None, None), mode=LongitudinalMode.E2E,
                    model_should_stop=False, model_stop_distance=None, model_desired_accel=0.08), DT)
  assert r.standstill_release_allowed is False
  assert r.standstill_release_source == ""


def test_lead_release_rejects_tiny_positive_evidence(monkeypatch):
  s = CustomLongitudinalStack()
  ld = lead(d_rel=8.0, v_lead=0.5, v_rel=0.5)
  _warm_lead_progress_allowed(s, ld)

  def fake_decide(candidates, mode, accel_limits, sources=None, previous_intent=""):
    return Decision(a_target=0.04, should_stop=False, selected_intent="lead_pullaway", reason="cruise")
  monkeypatch.setattr("openpilot.sunnypilot.custom.longitudinal.stack.decide", fake_decide)

  r = s.update(base(v_ego=0.0, v_cruise=12.0, leads=(ld, None), mode=LongitudinalMode.ACC), DT)
  assert r.standstill_release_allowed is False


def test_acc_lead_progress_unaffected_by_raw_model_path_geometry():
  """ADR 2026-07-10: raw modelV2 path geometry cannot authorize ACC lead progress.

  A curved model path used to inflate the progress distance via the path-arc shortcut;
  the actuation lead tracker now consumes radar-fused Lead Evidence only, so the same
  scenario with and without the model path produces identical progress and targets.
  """
  ld = lead(d_rel=43.0, v_lead=20.0, v_rel=0.0, y_rel=0.0)
  model = model_path(xs=(0.0, 43.0), ys=(0.0, 20.0))
  s_with, s_without = CustomLongitudinalStack(), CustomLongitudinalStack()
  r_with = r_without = None
  for _ in range(5):
    r_with = s_with.update(base(v_ego=20.0, v_cruise=22.0, seed_a_target=0.0,
                                leads=(ld, None), model_msg=model, mode=LongitudinalMode.ACC), DT)
    r_without = s_without.update(base(v_ego=20.0, v_cruise=22.0, seed_a_target=0.0,
                                      leads=(ld, None), model_msg=None, mode=LongitudinalMode.ACC), DT)

  assert r_with is not None and r_without is not None
  assert r_with.a_target == pytest.approx(r_without.a_target)
  assert r_with.debug["lead_context_progress_allowed"] == r_without.debug["lead_context_progress_allowed"]
  assert r_with.debug["lead_context_gap_excess"] == 0.0
  assert r_without.debug["lead_context_gap_excess"] == 0.0
  assert r_with.debug["actual_primary_lead_gap_shortage"] == pytest.approx(1.0)
  assert r_with.debug["acc_envelope_time_gap"] == pytest.approx(43.0 / 20.0)


def test_apply_mode_actuation_verdicts_independent_of_debug():
  """Turning debug off removes diagnostics only; apply-mode Actuation Verdicts persist."""
  kwargs = dict(
    mode=LongitudinalMode.SCC,
    curve_traffic_advisor_mode="apply_conservative",
    curve_confidence=CurveSpeedConfidenceInputs(vision_active=True, vision_a_target=-0.5),
    model_msg=circular_arc_path(n=24, radius=100.0),
    long_active=True,
    sources=SourceToggles(scc_curve_vision_enabled=True),
    research_actuation_allowed=True,
  )
  r_debug = CustomLongitudinalStack().update(base(**kwargs), DT, collect_debug=True)
  r_quiet = CustomLongitudinalStack().update(base(**kwargs), DT, collect_debug=False)

  assert r_debug.actuation.curve_traffic_advisor is not None
  assert r_quiet.actuation.curve_traffic_advisor == r_debug.actuation.curve_traffic_advisor
  assert r_quiet.actuation.cut_in_brake_assist == r_debug.actuation.cut_in_brake_assist
  assert r_quiet.debug == {}
  assert r_debug.debug


def test_debug_off_without_apply_modes_skips_feature_verdicts():
  r = CustomLongitudinalStack().update(base(mode=LongitudinalMode.SCC), DT, collect_debug=False)
  assert r.actuation.cut_in_brake_assist is None
  assert r.actuation.curve_traffic_advisor is None
  assert r.debug == {}


def test_e2e_model_stop_brakes_acc_does_not():
  stop = dict(model_should_stop=True, model_stop_distance=18.0, model_desired_accel=-2.5, stop_threat=True)
  s_acc, s_e2e = CustomLongitudinalStack(), CustomLongitudinalStack()
  acc = s_acc.update(base(mode=LongitudinalMode.ACC, **stop), DT)
  e2e = s_e2e.update(base(mode=LongitudinalMode.E2E, **stop), DT)
  assert acc.a_target == 0.4 and acc.should_stop is False   # ACC ignores model stop
  assert e2e.a_target < 0.0 and e2e.should_stop is True      # E2E brakes


def test_standstill_is_threaded_into_trusted_model_stop_semantics():
  result = CustomLongitudinalStack().update(base(
    v_ego=0.0, v_cruise=15.0, seed_a_target=0.0, standstill=True, mode=LongitudinalMode.E2E,
    model_should_stop=True, model_stop_distance=8.0, model_desired_accel=-1.0, model_stop_prob=1.0,
  ), DT)
  assert result.should_stop is True


def test_lead_follow_decel_binds():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(6):
    r = s.update(base(leads=(lead(d_rel=15.0), None), lead_a_target=-1.5, mode=LongitudinalMode.ACC), DT)
  assert r.debug["has_lead"] is True
  assert r.a_target <= -1.5 + 1e-9


def test_acc_envelope_gap_caps_remain_telemetry_only():
  raw_stack = CustomLongitudinalStack()
  envelope_stack = CustomLongitudinalStack()
  kwargs = dict(
    v_ego=20.0,
    v_cruise=25.0,
    seed_a_target=0.4,
    leads=(lead(d_rel=20.0, v_lead=15.0, v_rel=-5.0), None),
    lead_a_target=0.4,
    mode=LongitudinalMode.ACC,
  )

  raw = raw_stack.update(base(**kwargs), DT)
  observed = envelope_stack.update(base(**kwargs), DT)

  assert observed.a_target == pytest.approx(raw.a_target)
  assert observed.should_stop == raw.should_stop
  assert observed.debug["acc_envelope_active"] is True
  assert observed.debug["acc_envelope_would_cap"] is True
  assert "inside_time_gap" in observed.debug["acc_envelope_cap_reason"]
  assert observed.debug["acc_envelope_delta_a"] <= 0.0
  assert observed.debug["target_smoothing_applied"] is False


def test_upward_accel_rise_is_slew_limited_when_active():
  s = CustomLongitudinalStack()

  first = s.update(base(seed_a_target=0.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert first.a_target == pytest.approx(0.0)

  second = s.update(base(seed_a_target=1.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert second.a_target == pytest.approx(0.05)
  assert second.debug["target_smoothing_applied"] is True
  assert second.debug["target_smoothing_raw_a_target"] == pytest.approx(1.0)


def test_upward_accel_rise_uses_max_lag_from_stop_hold_target():
  s = CustomLongitudinalStack()

  first = s.update(base(seed_a_target=-0.5, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert first.a_target == pytest.approx(-0.5)

  second = s.update(base(seed_a_target=1.25, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert second.a_target <= 0.75 + 1e-9
  assert second.debug["target_smoothing_applied"] is True


def test_downward_hazard_decel_is_not_slew_limited():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=1.0, mode=LongitudinalMode.ACC, long_active=True), DT)

  r = s.update(base(
    seed_a_target=1.0,
    leads=(lead(d_rel=15.0), None),
    lead_a_target=-1.5,
    mode=LongitudinalMode.ACC,
    long_active=True,
  ), DT)

  assert r.a_target == pytest.approx(-1.5)
  assert r.debug["reason"] == "physical_hazard"
  assert r.debug["target_smoothing_applied"] is False
  assert r.debug["target_smoothing_reason"] == "downward_passthrough"


def test_small_comfort_decel_is_slew_limited_faster_than_accel():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.4, mode=LongitudinalMode.ACC, long_active=True), DT)

  r = s.update(base(seed_a_target=0.2, mode=LongitudinalMode.ACC, long_active=True), DT)

  assert r.a_target == pytest.approx(0.2)  # -4 m/s^3 * 0.05s allows this whole 0.2 step
  assert r.debug["target_smoothing_direction"] == "downward"
  assert r.debug["target_smoothing_applied"] is False


def test_small_comfort_decel_larger_than_one_tick_is_smoothed():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.4, mode=LongitudinalMode.ACC, long_active=True), DT)

  r = s.update(base(seed_a_target=0.15, mode=LongitudinalMode.ACC, long_active=True), DT)

  assert r.a_target == pytest.approx(0.2)
  assert r.debug["target_smoothing_direction"] == "downward"
  assert r.debug["target_smoothing_applied"] is True
  assert r.debug["target_smoothing_reason"] == "downward_slew_limited"


def test_large_or_strong_decel_bypasses_smoothing():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.4, mode=LongitudinalMode.ACC, long_active=True), DT)

  r = s.update(base(seed_a_target=-0.5, mode=LongitudinalMode.ACC, long_active=True), DT)

  assert r.a_target == pytest.approx(-0.5)
  assert r.debug["target_smoothing_direction"] == "downward"
  assert r.debug["target_smoothing_applied"] is False
  assert r.debug["target_smoothing_reason"] == "downward_passthrough"


def test_closing_lead_decel_bypasses_smoothing():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.2, mode=LongitudinalMode.ACC, long_active=True), DT)

  r = s.update(base(
    seed_a_target=0.0,
    leads=(lead(d_rel=80.0, v_lead=19.4, v_rel=-0.6), None),
    lead_a_target=0.0,
    mode=LongitudinalMode.ACC,
    long_active=True,
  ), DT)

  # Cushion coast, relevance-capped for the barely-closing far lead:
  # closing 0.6 over 36 m excess -> -(2.5 * 0.36/72 + 0.1) = -0.1125. Still applied
  # immediately (downward passthrough), which is the mechanism under test.
  assert r.a_target == pytest.approx(-0.1125)
  assert r.debug["target_smoothing_direction"] == "downward"
  assert r.debug["target_smoothing_applied"] is False
  assert r.debug["target_smoothing_reason"] == "downward_passthrough"


def test_selected_lead_two_supplies_policy_kinematics_without_stop_commitment():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(8):
    r = s.update(base(v_ego=20.0, v_cruise=25.0, seed_a_target=0.4,
                      leads=(None, lead(d_rel=18.0, v_lead=15.0, v_rel=-5.0, track_id=22)),
                      lead_a_target=0.4, mode=LongitudinalMode.ACC), DT)
  assert r is not None
  assert r.debug["lead_kinematics_source"] == "physical"
  assert r.debug["lead_kinematics_source_idx"] == 1
  assert r.debug["lead_kinematics_source_track_id"] == 22
  assert r.debug["lead_kinematics_valid"] is True
  assert r.debug["acc_envelope_time_gap"] == pytest.approx(18.0 / 20.0)
  assert r.debug["acc_envelope_ttc"] == pytest.approx(18.0 / 5.0)
  assert r.should_stop is False


@pytest.mark.parametrize(
  ("expected_slot", "expected_track"),
  [
    (0, 11),
    (1, 22),
  ],
)
def test_confidence_snapshot_preserves_selected_lead_slot_track_and_authority(
    expected_slot, expected_track,
):
  stack = CustomLongitudinalStack()
  leads = (
    (lead(d_rel=18.0, v_lead=15.0, v_rel=-5.0, track_id=expected_track) if expected_slot == 0 else None),
    (lead(d_rel=18.0, v_lead=15.0, v_rel=-5.0, track_id=expected_track) if expected_slot == 1 else None),
  )
  result = None
  for _ in range(8):
    result = stack.update(base(
      v_ego=20.0, v_cruise=25.0, seed_a_target=0.0,
      leads=leads, lead_a_target=0.0, mode=LongitudinalMode.ACC,
    ), DT)

  assert result is not None
  expected_slot_name = "leadOne" if expected_slot == 0 else "leadTwo"
  assert result.debug["confidence_selected_lead_slot"] == expected_slot_name
  assert result.debug["confidence_selected_track_id"] == expected_track
  authority = result.debug["lead_kinematics_source_authority"]
  expected_authority = {
    "": "none", "none": "none", "suppress_only": "suppressOnly",
    "physical": "physical", "progress_allowed": "progressAllowed",
  }[authority]
  assert result.debug["confidence_selected_authority"] == expected_authority


def test_confidence_snapshot_keeps_acquisition_and_flicker_timers_distinguishable():
  stack = CustomLongitudinalStack()
  result = None
  for present in (True, False, True, False, True):
    result = stack.update(base(
      v_ego=10.0, v_cruise=10.0, seed_a_target=0.0,
      leads=((lead(d_rel=10.0, v_lead=1.0, v_rel=-9.0, track_id=31) if present else None), None),
      mode=LongitudinalMode.ACC,
    ), DT)

  assert result is not None
  acquisition = result.debug["confidence_acquisition_timer_s"]
  flicker = result.debug["confidence_flicker_guard_timer_s"]
  assert acquisition > 0.0
  assert flicker > acquisition
  assert acquisition != flicker


def test_selected_lead_two_caps_positive_lead_seed_instead_of_synthesizing_progress():
  r = CustomLongitudinalStack().update(base(v_ego=20.0, v_cruise=25.0, seed_a_target=0.4,
                                            leads=(None, lead(d_rel=30.0, v_lead=18.0, v_rel=-2.0, track_id=23)),
                                            lead_a_target=0.4, mode=LongitudinalMode.ACC), DT)
  assert r.debug["lead_kinematics_source_idx"] == 1
  assert r.debug["intent"] == "lead_follow"
  assert r.a_target <= 0.0
  assert r.should_stop is False


def test_selected_lead_two_positive_seed_suppresses_pullaway_progress():
  s = CustomLongitudinalStack()
  r = None
  for _ in range(12):
    r = s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.4,
                      leads=(None, lead(d_rel=8.0, v_lead=3.0, v_rel=3.0, track_id=25)),
                      lead_a_target=0.4, mode=LongitudinalMode.ACC), DT)
  assert r is not None
  assert r.debug["lead_kinematics_source_idx"] == 1
  assert r.debug["lead_context_progress_allowed"] is True
  assert r.debug["lead_progress_allowed"] is False
  assert r.debug["intent"] != "lead_pullaway"
  assert r.standstill_release_allowed is False
  assert r.should_stop is False


def test_selected_lead_two_does_not_inherit_lead0_stop_seed():
  r = CustomLongitudinalStack().update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.0,
                                            leads=(None, lead(d_rel=8.0, v_lead=0.0, v_rel=0.0, track_id=26)),
                                            lead_a_target=0.0, lead_should_stop=True, mode=LongitudinalMode.ACC), DT)
  assert r.debug["lead_kinematics_source_idx"] == 1
  assert r.should_stop is False


def test_downward_smoothing_uses_selected_lead_not_only_lead0():
  s = CustomLongitudinalStack()
  selected_closing_lead = lead(d_rel=80.0, v_lead=19.4, v_rel=-0.6, track_id=24)
  assert s._downward_smoothing_allowed(
    raw=0.0, prev=0.2, inp=base(leads=(None, selected_closing_lead)),
    decision=SimpleNamespace(reason="cruise", should_stop=False),
    acc_envelope_result=SimpleNamespace(cap_reasons=()), selected_lead=selected_closing_lead,
  ) is False


def test_shadow_suppresses_release_without_becoming_kinematic_source():
  s = CustomLongitudinalStack()
  for _ in range(12):
    s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                  leads=(lead(d_rel=6.5, v_lead=0.0), None), mode=LongitudinalMode.ACC), DT)
  lost = s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2,
                       leads=(None, None), mode=LongitudinalMode.ACC), DT)
  assert lost.debug["lead_shadow_active"] is True
  assert lost.debug["lead_kinematics_source"] == "none"
  assert lost.debug["lead_kinematics_source_idx"] == -1
  assert lost.standstill_release_allowed is False
  assert lost.should_stop is False


def test_decel_smoothing_never_raises_above_nonselected_hazard():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.0, mode=LongitudinalMode.SCC, long_active=True), DT)

  r = s.update(base(
    v_ego=20.0,
    v_cruise=22.0,
    seed_a_target=0.0,
    accel_coast=-1.0,
    # Inside the far-lead relevance-cap trust floor (45 m) so the hazard keeps
    # full -0.25 authority for the smoothing-floor clamp under test.
    leads=(lead(d_rel=40.0, v_lead=4.0, v_rel=0.0), None),
    lead_a_target=-0.25,
    speed_limit_active=True,
    speed_limit_v_target=10.0,
    speed_limit_a_target=-0.30,
    mode=LongitudinalMode.SCC,
    long_active=True,
  ), DT)

  assert r.debug["reason"] == "advisory_capped"
  assert r.debug["target_smoothing_applied"] is True
  assert r.debug["target_smoothing_hazard_floor"] == pytest.approx(-0.25)
  assert r.a_target == pytest.approx(-0.25)


def test_target_smoothing_reset_clears_stale_previous():
  s = CustomLongitudinalStack()
  s.update(base(seed_a_target=0.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  capped = s.update(base(seed_a_target=1.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert capped.a_target == pytest.approx(0.05)

  inactive = s.update(base(seed_a_target=1.0, mode=LongitudinalMode.ACC, long_active=False), DT)
  assert inactive.debug["target_smoothing_reason"] == "long_inactive"

  after_reset = s.update(base(seed_a_target=1.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert after_reset.a_target == pytest.approx(1.0)
  assert after_reset.debug["target_smoothing_reason"] == "primed"


def test_brake_release_lag_is_capped_even_when_the_target_stays_negative():
  # Regression, openpilot_lead_decel_3ms2: the lag cap used to require raw >= 0.0, so a
  # release ending at a shallower-but-still-negative target unwound at the envelope's
  # 1 m/s^3 alone — one frame off -3.0 reached only -2.95 and the command stayed ~1.4 m/s^2
  # below the plan for ~1.5 s. That output is the planner's a_desired, so the MPC's own
  # state followed the lag down.
  s = CustomLongitudinalStack()
  primed = s.update(base(v_ego=20.0, v_cruise=10.0, seed_a_target=-3.0,
                         mode=LongitudinalMode.ACC, long_active=True), DT)
  assert primed.a_target == pytest.approx(-3.0)
  r = s.update(base(v_ego=20.0, v_cruise=10.0, seed_a_target=-0.5,
                    mode=LongitudinalMode.ACC, long_active=True), DT)
  assert r.a_target == pytest.approx(-1.0)          # raw - UPWARD_TARGET_SLEW_MAX_LAG
  assert r.a_target_unsmoothed == pytest.approx(-0.5)  # the plan is never lagged


def test_unsmoothed_target_is_the_plan_not_the_smoothed_command():
  s = CustomLongitudinalStack()
  s.update(base(v_ego=20.0, v_cruise=10.0, seed_a_target=-1.0,
                mode=LongitudinalMode.ACC, long_active=True), DT)
  r = s.update(base(v_ego=20.0, v_cruise=10.0, seed_a_target=-0.2,
                    mode=LongitudinalMode.ACC, long_active=True), DT)
  assert r.a_target < r.a_target_unsmoothed          # command lags, plan does not
  assert r.a_target_unsmoothed == pytest.approx(-0.2)


def test_target_smoothing_does_not_block_standstill_release_authorization():
  s = CustomLongitudinalStack()
  primed = s.update(base(v_ego=20.0, v_cruise=10.0, seed_a_target=-1.0, mode=LongitudinalMode.ACC, long_active=True), DT)
  assert primed.a_target == pytest.approx(-1.0)

  r = s.update(base(
    v_ego=0.0,
    v_cruise=12.0,
    seed_a_target=0.4,
    leads=(None, None),
    mode=LongitudinalMode.E2E,
    long_active=True,
    model_should_stop=False,
    model_stop_distance=None,
    model_desired_accel=0.4,
  ), DT)

  assert r.standstill_release_allowed is True
  assert r.standstill_release_a_target >= 0.15
  assert r.debug["target_smoothing_applied"] is True


def test_personality_changes_launch():
  results = {}
  for p in (Personality.RELAXED, Personality.AGGRESSIVE):
    s = CustomLongitudinalStack()
    r = s.update(base(v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, personality=p,
                      model_stop_distance=60.0, model_desired_accel=0.0, mode=LongitudinalMode.ACC), DT)
    results[p] = r.a_target
  assert results[Personality.AGGRESSIVE] > results[Personality.RELAXED]


def test_stack_cushion_active_with_slower_lead():
  # slower lead with enough runway -> the lead-following cushion coasts (advisory) instead of
  # cruising up to the set speed, proving the Phase 5 integration is live in the stack path.
  s = CustomLongitudinalStack()
  r = s.update(base(v_ego=20.0, v_cruise=22.0, seed_a_target=0.3,
                    leads=(lead(d_rel=385.0, v_lead=15.0), None), lead_a_target=0.0,
                    mode=LongitudinalMode.ACC), DT)
  assert r.a_target < 0.3


def test_low_speed_gap_closure_does_not_require_generic_lead_progress_gate():
  stack = CustomLongitudinalStack()
  ld = lead(d_rel=10.3, v_lead=1.2, v_rel=0.0, a_lead=0.0)
  result = None
  for _ in range(8):
    result = stack.update(base(
      v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, lead_a_target=0.0,
      leads=(ld, None), mode=LongitudinalMode.SCC, long_active=True,
    ), DT)

  assert result is not None
  assert result.debug["lead_context_progress_allowed"] is False
  assert result.low_speed_gap_closure is not None
  assert result.low_speed_gap_closure.requested_accel == pytest.approx(0.25)


@pytest.mark.parametrize(
  "overrides",
  [
    {"model_should_stop": True},
    {"curve_active": True, "curve_a_target": -0.2},
    {"gas_pressed": True},
    {"force_slow_decel": True},
    {"seed_a_target": -0.11},
    {"standstill": True},
  ],
)
def test_low_speed_gap_closure_upstream_verdict_is_fail_closed(overrides):
  stack = CustomLongitudinalStack()
  ld = lead(d_rel=10.3, v_lead=1.2, v_rel=0.0, a_lead=0.0)
  result = None
  for _ in range(8):
    values = dict(
      v_ego=1.0, v_cruise=12.0, seed_a_target=0.0, lead_a_target=0.0,
      leads=(ld, None), mode=LongitudinalMode.SCC, long_active=True,
    )
    values.update(overrides)
    if values.get("curve_active"):
      values["sources"] = SourceToggles(scc_curve_vision_enabled=True)
    result = stack.update(base(**values), DT)

  assert result is not None
  assert result.low_speed_gap_closure is None


def test_reset_clears_trackers():
  s = CustomLongitudinalStack()
  for _ in range(10):
    s.update(base(leads=(lead(), None)), DT)
  s.reset()
  r = s.update(base(leads=(None, None), mode=LongitudinalMode.ACC), DT)
  assert r.debug["has_lead"] is False


def test_launch_behind_opening_lead_tracks_off_the_line():
  """Stop-and-go launch through the full stateful stack: stopped ~6.5 m behind a stopped lead,
  then the lead launches. Unlike the policy unit tests (which inject lead_progress_allowed), this
  drives the real lead-confidence tracker, which must *earn* the pull-away authorisation over
  ~0.35 s. Once stable, the stack should pull away off the line — commanding well above a timid
  MPC seed — instead of hanging back (ADR hypermile §1)."""
  from types import SimpleNamespace
  s = CustomLongitudinalStack()
  v_ego = x_ego = 0.0
  v_lead, x_lead = 0.0, 6.5
  seed = 0.15                                   # deliberately timid MPC seed
  launched_a: list[float] = []
  release_seen = False
  for i in range(120):                          # 6 s
    t = i * DT
    a_lead = 1.2 if (1.0 <= t and v_lead < 8.0) else 0.0
    v_lead = min(8.0, v_lead + a_lead * DT)
    x_lead += v_lead * DT
    d_rel = max(0.1, x_lead - x_ego)
    ld = SimpleNamespace(status=True, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead,
                         yRel=0.0, radarTrackId=7, radar=True, modelProb=0.9, aLeadTau=1.0)
    r = s.update(base(v_ego=v_ego, v_cruise=13.4, seed_a_target=seed, lead_a_target=seed,
                      leads=(ld, None), mode=LongitudinalMode.ACC), DT)
    assert LIMITS[0] - 1e-9 <= r.a_target <= LIMITS[1] + 1e-9
    if 1.5 <= t <= 3.0:                         # well into the launch, confidence stabilised
      launched_a.append(r.a_target)
      release_seen = release_seen or r.standstill_release_allowed
    v_ego = max(0.0, v_ego + r.a_target * DT)   # closed-loop: ego driven by the stack
    x_ego += v_ego * DT
  assert launched_a
  assert release_seen, "lead pullaway did not expose standstill release once authorized"
  assert max(launched_a) > seed + 0.3, "hung back at the timid seed instead of following the lead"
  assert min(launched_a) >= 0.0               # never braked during a clean launch


def test_model_stop_distance_closed_loop_comes_to_comfortable_rest():
  """Drive-lab style closed-loop gate for the model_stop_distance path (C): integrate the
  stack's a_target against a fixed stop point and assert the approach comes to rest without
  rolling through the stop, stays within the accel envelope, and keeps jerk comfortable."""
  s = CustomLongitudinalStack()
  v, x, stop_point = 13.0, 0.0, 45.0   # 13 m/s closing on a stop 45 m ahead
  a_prev, max_abs_jerk, min_a, near_zero_while_approaching = None, 0.0, 0.0, 0
  for _ in range(800):
    dist_to_stop = stop_point - x
    r = s.update(base(
      v_ego=v, v_cruise=15.0, seed_a_target=0.0, mode=LongitudinalMode.E2E,
      leads=(None, None),
      model_should_stop=True, model_stop_distance=max(0.0, dist_to_stop),
      model_desired_accel=-2.0, model_stop_prob=1.0,
    ), DT)
    a = r.a_target
    assert LIMITS[0] - 1e-9 <= a <= LIMITS[1] + 1e-9
    if a_prev is not None:                       # skip the cold-start step (a_prev has no real history)
      max_abs_jerk = max(max_abs_jerk, abs(a - a_prev) / DT)
    if a > -0.1 and dist_to_stop > 0.0 and v > 0.5:   # no "stop braking" gaps while still rolling up
      near_zero_while_approaching += 1
    min_a = min(min_a, a)
    a_prev = a
    v = max(0.0, v + a * DT)
    x += v * DT
    if v < 0.05 and dist_to_stop < 1.0:
      break
  assert v < 0.3, f"never came to rest (v={v:.2f})"
  # Comes to rest right at the line: neither rolling through (collision risk) nor stopping far short.
  assert stop_point - 2.0 <= x <= stop_point + 0.5, f"stopped at x={x:.2f}, expected near {stop_point}"
  assert near_zero_while_approaching == 0, "braking relaxed to ~0 mid-approach (coast-horizon gap)"
  assert min_a < -0.5, "never meaningfully braked for the predicted stop"
  assert min_a >= LIMITS[0] - 1e-9          # never demands harder than the decel limit
  # Raw-policy jerk (pre-MPC-smoothing); one comfort-decel candidate step is expected.
  assert max_abs_jerk <= 12.0, f"uncomfortable jerk {max_abs_jerk:.2f} m/s^3"


def test_far_non_closing_lead_raises_pre_mpc_target_via_soft_path():
  # wiring.py sets lead_a_target from seed_a_target when a lead is present, so the seed equals
  # the lead-influenced target. The soft pair should raise the pre-MPC target from -0.3 to -0.05.
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=100.0, v_lead=25.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  ), DT)
  assert r.debug["has_lead"] is True
  assert r.a_target == pytest.approx(-0.05)
  assert r.decision.reason == "advisory_capped"
  assert r.decision.selected_intent == "lead_follow_soft_desire"


def test_lead_softening_mode_admission_unchanged():
  # LEAD evidence is admitted in every mode; the soft path should raise the target in all three.
  for mode in (LongitudinalMode.ACC, LongitudinalMode.E2E, LongitudinalMode.SCC):
    s = CustomLongitudinalStack()
    r = s.update(base(
      v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
      leads=(lead(d_rel=100.0, v_lead=25.0), None),
      lead_a_target=-0.3, mode=mode,
    ), DT)
    assert r.a_target == pytest.approx(-0.05), mode


def test_invalid_raw_live_kinematics_do_not_trigger_softening():
  # Missing vRel on the live lead object makes raw kinematics invalid; _f still provides a
  # fallback value, but softening must reject and the original lead-follow seed stands.
  bad_lead = SimpleNamespace(status=True, dRel=100.0, vLead=25.0, vLeadK=25.0,
                             yRel=0.0, radarTrackId=3, radar=True, modelProb=0.9, aLeadTau=1.0)
  assert not hasattr(bad_lead, "vRel")
  s = CustomLongitudinalStack()
  r = s.update(base(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(bad_lead, None), lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  ), DT)
  assert r.a_target == pytest.approx(-0.3)
  assert "lead_follow_soft" not in r.debug.get("intent", "")


def test_path_shadow_model_offset_does_not_change_stack_actuation():
  raw_stack = CustomLongitudinalStack()
  model_stack = CustomLongitudinalStack()
  common = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=60.0, v_lead=25.0, v_rel=0.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  )

  raw = with_model = None
  for _ in range(12):
    raw = raw_stack.update(base(**common), DT)
    with_model = model_stack.update(base(**common, model_msg=model_path()), DT)
    assert with_model.a_target == pytest.approx(raw.a_target)
    assert with_model.decision.reason == raw.decision.reason
    assert with_model.standstill_release_allowed == raw.standstill_release_allowed

  assert raw is not None and with_model is not None
  assert with_model.debug["actual_primary_lead_path_y_rel"] == pytest.approx(0.0)
  assert with_model.debug["path_shadow_primary_lead_path_y_rel"] == pytest.approx(1.0)
  assert with_model.debug["path_shadow_model_path_available"] is True
  assert with_model.debug["path_shadow_fault"] is False


def test_path_shadow_fault_is_contained_inside_stack_update():
  raw_stack = CustomLongitudinalStack()
  bad_model_stack = CustomLongitudinalStack()
  common = dict(
    v_ego=25.0, v_cruise=25.0, seed_a_target=-0.3,
    leads=(lead(d_rel=60.0, v_lead=25.0, v_rel=0.0), None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC,
  )

  raw = raw_stack.update(base(**common), DT)
  bad = bad_model_stack.update(base(**common, model_msg=malformed_model_path()), DT)

  assert bad.a_target == pytest.approx(raw.a_target)
  assert bad.decision.reason == raw.decision.reason
  assert bad.standstill_release_allowed == raw.standstill_release_allowed
  assert bad.debug["path_shadow_model_path_available"] is False
  assert bad.debug["path_shadow_fault"] is True


def test_acc_polluted_evidence_invariance():
  """ACC must be identical whether or not mode-excluded evidence is injected."""
  common = dict(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.ACC, long_active=False)
  clean = CustomLongitudinalStack().update(base(**common), DT)
  polluted = CustomLongitudinalStack().update(base(
    **common,
    model_should_stop=True, model_stop_distance=10.0, model_desired_accel=-2.0,
    model_stale=True, stop_threat=True,
    speed_limit_active=True, speed_limit_v_target=10.0, speed_limit_a_target=-1.0,
    curve_active=True, curve_a_target=-0.8, curve_source=EvidenceClass.CURVE_VISION,
    sources=SourceToggles(scc_curve_vision_enabled=True, scc_curve_map_enabled=True),
  ), DT)

  assert polluted.a_target == pytest.approx(clean.a_target)
  assert polluted.should_stop == clean.should_stop
  assert polluted.decision.reason == clean.decision.reason
  assert polluted.decision.selected_intent == clean.decision.selected_intent
  assert polluted.standstill_release_allowed == clean.standstill_release_allowed


def test_acc_smoothing_not_contaminated_by_excluded_model_stop():
  """Raw model-stop fields are invisible to ACC smoothing and must not reset it."""
  clean = CustomLongitudinalStack()
  polluted = CustomLongitudinalStack()
  prime = dict(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.ACC, long_active=True)
  clean.update(base(**prime), DT)
  polluted.update(base(**prime), DT)

  step = dict(v_ego=20.0, v_cruise=22.0, seed_a_target=0.15, mode=LongitudinalMode.ACC, long_active=True)
  clean_r = clean.update(base(**step), DT)
  polluted_r = polluted.update(base(
    **step,
    model_should_stop=True, model_stop_distance=10.0, model_desired_accel=-2.0,
    model_stale=True, stop_threat=True,
  ), DT)

  assert polluted_r.a_target == pytest.approx(clean_r.a_target)
  assert polluted_r.should_stop == clean_r.should_stop
  assert polluted_r.decision.reason == clean_r.decision.reason
  assert polluted_r.decision.selected_intent == clean_r.decision.selected_intent
  assert polluted_r.debug["target_smoothing_reason"] != "model_stop"
  assert polluted_r.debug["target_smoothing_reason"] == clean_r.debug["target_smoothing_reason"]


def test_e2e_ignores_speed_limit_and_curve_model_stop_binds():
  """E2E admits model stop but must ignore speed-limit/curve evidence."""
  stop = dict(
    model_should_stop=True, model_stop_distance=18.0, model_desired_accel=-2.5,
    stop_threat=True, model_stop_prob=1.0,
  )
  extras = dict(
    speed_limit_active=True, speed_limit_v_target=10.0, speed_limit_a_target=-1.0,
    curve_active=True, curve_a_target=-1.0, curve_source=EvidenceClass.CURVE_VISION,
  )

  e2e_clean = CustomLongitudinalStack().update(base(mode=LongitudinalMode.E2E, long_active=False, **stop), DT)
  e2e_mixed = CustomLongitudinalStack().update(base(mode=LongitudinalMode.E2E, long_active=False, **stop, **extras), DT)

  assert e2e_clean.should_stop is True
  assert e2e_mixed.should_stop is True
  assert e2e_mixed.a_target == pytest.approx(e2e_clean.a_target)
  assert e2e_mixed.decision.reason == e2e_clean.decision.reason
  assert e2e_mixed.decision.selected_intent == e2e_clean.decision.selected_intent


def test_scc_source_gates_speed_limit_and_curve():
  """SCC admits speed-limit and curve sources only when their toggles are on."""
  r_scc_speed = CustomLongitudinalStack().update(base(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.SCC, long_active=False,
    speed_limit_active=True, speed_limit_v_target=10.0, speed_limit_a_target=-0.3,
  ), DT)
  assert r_scc_speed.a_target == pytest.approx(0.0)
  assert r_scc_speed.decision.reason == "advisory_capped"

  for mode in (LongitudinalMode.ACC, LongitudinalMode.E2E):
    r = CustomLongitudinalStack().update(base(
      v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=mode, long_active=False,
      speed_limit_active=True, speed_limit_v_target=10.0, speed_limit_a_target=-0.3,
    ), DT)
    assert r.a_target == pytest.approx(0.4), mode
    assert r.decision.reason == "cruise"

  # Curve vision toggle
  r_vision_off = CustomLongitudinalStack().update(base(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.SCC, long_active=False,
    curve_active=True, curve_a_target=-1.0, curve_source=EvidenceClass.CURVE_VISION,
    sources=SourceToggles(scc_curve_vision_enabled=False),
  ), DT)
  assert r_vision_off.a_target == pytest.approx(0.4)
  assert r_vision_off.decision.reason == "cruise"

  r_vision_on = CustomLongitudinalStack().update(base(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.SCC, long_active=False,
    curve_active=True, curve_a_target=-1.0, curve_source=EvidenceClass.CURVE_VISION,
    sources=SourceToggles(scc_curve_vision_enabled=True),
  ), DT)
  assert r_vision_on.a_target == pytest.approx(-0.5)
  assert r_vision_on.decision.reason == "advisory_capped"

  # Curve map toggle
  r_map_off = CustomLongitudinalStack().update(base(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.SCC, long_active=False,
    curve_active=True, curve_a_target=-0.9, curve_source=EvidenceClass.CURVE_MAP,
    sources=SourceToggles(scc_curve_map_enabled=False),
  ), DT)
  assert r_map_off.a_target == pytest.approx(0.4)
  assert r_map_off.decision.reason == "cruise"

  r_map_on = CustomLongitudinalStack().update(base(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4, mode=LongitudinalMode.SCC, long_active=False,
    curve_active=True, curve_a_target=-0.9, curve_source=EvidenceClass.CURVE_MAP,
    sources=SourceToggles(scc_curve_map_enabled=True),
  ), DT)
  assert r_map_on.a_target == pytest.approx(-0.5)
  assert r_map_on.decision.reason == "advisory_capped"


def _stable_lead_compression_stack(seed_a: float, lead_d: float, v_lead: float, v_rel: float, n: int = 30):
  s = CustomLongitudinalStack()
  r = None
  for _ in range(n):
    r = s.update(base(
      v_ego=15.0, v_cruise=15.0, seed_a_target=seed_a,
      leads=(lead(d_rel=lead_d, v_lead=v_lead, v_rel=v_rel), None),
      lead_a_target=seed_a,
      mode=LongitudinalMode.ACC,
      long_active=True,
    ), DT)
  return r


def test_lead_gap_compression_not_hardened_by_inside_time_gap():
  # Stable, confident, low-risk inside-gap compression. d_rel is inside the ACC desired gap
  # (1.5 s) but closing is small enough that no binding ttc/closing_decel risk exists.
  # The ACC envelope reports inside_time_gap but the final target stays at the compression target.
  seed = -0.15
  r = _stable_lead_compression_stack(
    seed_a=seed,
    lead_d=22.0,
    v_lead=14.8,
    v_rel=-0.2,
    n=35,
  )

  assert r.debug["intent"] == "lead_gap_compression", f"got intent={r.debug['intent']}"
  assert "inside_time_gap" in r.debug.get("acc_envelope_cap_reason", "")
  assert "closing_decel_high" not in r.debug.get("acc_envelope_cap_reason", "")
  assert "ttc_low" not in r.debug.get("acc_envelope_cap_reason", "")
  # Should stay near the ramped compression target, not harden to raw -required_decel
  assert r.a_target == pytest.approx(seed, abs=0.03)


def test_routine_compression_not_hardened_by_closing_decel_high():
  # Routine compression: desired-gap demand is above the comfort tier but TTC/time-gap
  # are safe and collision-buffer demand is controlled. ACC envelope reports
  # closing_decel_high but must not harden the final target.
  r = _stable_lead_compression_stack(
    seed_a=-1.2,
    lead_d=18.0,
    v_lead=12.5,
    v_rel=-2.5,
    n=35,
  )

  assert r.debug["intent"] == "lead_gap_compression", f"got intent={r.debug['intent']}"
  assert "closing_decel_high" in r.debug.get("acc_envelope_cap_reason", "")
  assert "ttc_low" not in r.debug.get("acc_envelope_cap_reason", "")
  # Final target should sit in the routine range, not snap back to seed -1.2.
  assert -0.85 <= r.a_target <= -0.45


def test_non_compression_lead_hazard_still_hardened_by_inside_time_gap():
  # Inside-time-gap with high closing is not a controlled compression candidate;
  # the raw lead-follow hazard must remain binding and not be softened.
  s = CustomLongitudinalStack()
  r = None
  for _ in range(35):
    r = s.update(base(
      v_ego=20.0, v_cruise=20.0, seed_a_target=0.0,
      leads=(lead(d_rel=22.0, v_lead=17.5, v_rel=-2.5), None),
      lead_a_target=-1.5,
      mode=LongitudinalMode.ACC,
      long_active=True,
    ), DT)

  assert r.debug["intent"] == "lead_follow"
  assert "inside_time_gap" in r.debug.get("acc_envelope_cap_reason", "")
  assert "closing_decel_high" in r.debug.get("acc_envelope_cap_reason", "")
  assert r.a_target <= -1.5


def test_lead_gap_compression_latch_rejects_radar_chatter():
  s = CustomLongitudinalStack()

  def step(v_rel, *, a_lead=0.0, lead_a_target=-0.3, long_active=True, lead_present=True):
    return s.update(base(
      v_ego=15.0, v_cruise=15.0, seed_a_target=-0.3,
      leads=((lead(d_rel=22.0, v_lead=15.0 + v_rel, v_rel=v_rel, a_lead=a_lead)
              if lead_present else None), None),
      lead_a_target=lead_a_target, mode=LongitudinalMode.ACC, long_active=long_active,
    ), DT)

  for _ in range(10):
    step(0.0)

  armed = step(-0.12)
  assert armed.decision.selected_intent == "lead_gap_compression"
  assert armed.decision.a_target == pytest.approx(-0.15)
  assert s._lead_gap_compression_active is True

  for v_rel in (-0.08, -0.11, -0.06, -0.13, -0.04):
    result = step(v_rel)
    assert result.decision.selected_intent == "lead_gap_compression"
    assert result.decision.a_target == pytest.approx(-0.15)

  recovered = step(0.05)
  assert recovered.decision.selected_intent.startswith("lead_gap_recovery")
  assert recovered.decision.a_target == pytest.approx(0.0)
  assert s._lead_gap_compression_active is False

  recovered = step(-0.08)
  assert recovered.decision.selected_intent.startswith("lead_gap_recovery")
  assert recovered.decision.a_target == pytest.approx(0.0)

  rearmed = step(-0.12)
  assert rearmed.decision.selected_intent == "lead_gap_compression"
  assert rearmed.decision.a_target == pytest.approx(-0.15)
  assert s._lead_gap_compression_active is True

  hard = step(-0.12, a_lead=-3.0, lead_a_target=-1.5)
  assert hard.decision.selected_intent == "lead_follow"
  assert hard.decision.a_target == pytest.approx(-1.5)
  assert s._lead_gap_compression_active is False

  rearmed = step(-0.12)
  assert rearmed.decision.selected_intent == "lead_gap_compression"
  assert s._lead_gap_compression_active is True

  inactive = step(-0.12, long_active=False, lead_present=False)
  assert not inactive.decision.selected_intent.startswith("lead_gap_")
  assert s._lead_gap_compression_active is False

  for _ in range(10):
    step(0.0)
  rearmed = step(-0.12)
  assert rearmed.decision.selected_intent == "lead_gap_compression"
  assert s._lead_gap_compression_active is True
  s.reset()
  assert s._lead_gap_compression_active is False
  for _ in range(10):
    step(0.0)
  rearmed = step(-0.12)
  assert rearmed.decision.selected_intent == "lead_gap_compression"
  assert s._lead_gap_compression_active is True
  lost = s.update(base(
    v_ego=15.0, v_cruise=15.0, seed_a_target=-0.3, leads=(None, None),
    lead_a_target=-0.3, mode=LongitudinalMode.ACC, long_active=True,
  ), DT)
  assert not lost.decision.selected_intent.startswith("lead_gap_")
  assert s._lead_gap_compression_active is False


def test_cutout_shadow_blocks_no_lead_launch_then_clears():
  """A stable lead that laterally exits near the lane edge creates a cutout shadow.
  The shadow must suppress no-lead launch/standstill release, then clear after expiry."""
  s = CustomLongitudinalStack()
  v_lead = 2.0
  track_id = 8
  for i in range(15):
    y = min(1.3, 0.0 + i * 0.1)
    ld = SimpleNamespace(status=True, dRel=20.0, vLead=v_lead, vLeadK=v_lead, vRel=v_lead,
                         aLeadK=0.0, yRel=y, radarTrackId=track_id, radar=True,
                         modelProb=0.9, aLeadTau=1.0)
    s.update(base(v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(ld, None),
                  mode=LongitudinalMode.E2E), DT)

  dropped = s.update(base(
    v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None),
    mode=LongitudinalMode.E2E,
    model_should_stop=False, model_stop_distance=None, model_desired_accel=0.2,
  ), DT)
  assert dropped.debug["lead_shadow_active"] is True
  assert dropped.debug.get("actual_shadow_lead_reason") == "cutout_exit"
  assert dropped.standstill_release_allowed is False

  final = dropped
  for _ in range(30):
    final = s.update(base(
      v_ego=0.0, v_cruise=12.0, seed_a_target=0.2, leads=(None, None),
      mode=LongitudinalMode.E2E,
      model_should_stop=False, model_stop_distance=None, model_desired_accel=0.2,
    ), DT)
    if final.debug["lead_shadow_active"]:
      assert final.standstill_release_allowed is False
    else:
      break
  assert final.debug["lead_shadow_active"] is False
  assert final.standstill_release_allowed is True
  assert final.standstill_release_source == "no_lead_launch"
