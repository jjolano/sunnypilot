"""Phase 5A characterization tests for the longitudinal finalizer.

These tests lock the behavior of ``LongitudinalPlannerSP.final_longitudinal_output``
and its helpers before the finalizer arbitration is extracted into a dedicated module.
Only test code is allowed to change in Phase 5A.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import CutInBrakeAssistResult
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceResult
from openpilot.sunnypilot.custom.longitudinal.curve_traffic_advisor import CurveTrafficAdvisorResult
from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.stack import ActuationVerdicts
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def cut_in_verdicts(model_path_available: bool = True, **overrides) -> ActuationVerdicts:
  fields: dict[str, Any] = dict(eligible=True, apply_supported=True, confidence=0.80, path_y_rel=0.3, proposed_cap=-2.0)
  fields.update(overrides)
  return ActuationVerdicts(cut_in_brake_assist=CutInBrakeAssistResult(**fields), model_path_available=model_path_available)


def curve_confidence_verdicts(**overrides) -> ActuationVerdicts:
  fields: dict[str, Any] = dict(eligible=True, apply_supported=True, confidence=0.80, proposed_cap=-1.0)
  fields.update(overrides)
  return ActuationVerdicts(curve_speed_confidence=CurveSpeedConfidenceResult(**fields))


def curve_traffic_verdicts(model_stale: bool = False, **overrides) -> ActuationVerdicts:
  fields: dict[str, Any] = dict(eligible=True, apply_supported=True, confidence=0.50, traffic_block_reason="", a_curve_cap_proposed=-1.0)
  fields.update(overrides)
  return ActuationVerdicts(curve_traffic_advisor=CurveTrafficAdvisorResult(**fields), model_stale=model_stale)


class FakeSM:
  def __init__(self, **services):
    self._services = services
    self.recv_time = services.pop("recv_time", None) or {}

  def get(self, key):
    return self._services.get(key)

  def __getitem__(self, key):
    return self._services[key]


def make_cp(stopping_distance: float = 6.0, v_ego_stopping: float = 0.2, stop_accel: float = -0.5):
  return SimpleNamespace(
    stoppingDistance=stopping_distance,
    vEgoStopping=v_ego_stopping,
    stopAccel=stop_accel,
  )


def make_custom_long(mode: LongitudinalMode = LongitudinalMode.SCC, **overrides):
  defaults = dict(
    enabled=True,
    mode=mode,
    standstill_release_confidence_mode="off",
    curve_speed_confidence_mode="off",
  )
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def make_custom_output(
    a_target: float = 0.0,
    should_stop: bool = False,
    enabled: bool = True,
    mode: LongitudinalMode = LongitudinalMode.SCC,
    selected_intent: str = "cruise",
    reason: str = "cruise",
    standstill_release_allowed: bool = False,
    standstill_release_source: str = "",
    standstill_release_a_target: float = 0.0,
    standstill_release_reason: str = "",
    research_actuation_allowed: bool = True,
    debug: dict[str, Any] | None = None,
    **overrides: Any,
) -> CustomLongitudinalOutput:
  if debug is None:
    debug = {}
  return CustomLongitudinalOutput(
    a_target=a_target,
    should_stop=should_stop,
    enabled=enabled,
    mode=mode,
    selected_intent=selected_intent,
    reason=reason,
    standstill_release_allowed=standstill_release_allowed,
    standstill_release_source=standstill_release_source,
    standstill_release_a_target=standstill_release_a_target,
    standstill_release_reason=standstill_release_reason,
    research_actuation_allowed=research_actuation_allowed,
    debug=debug,
    **overrides,
  )


def make_planner(mode=LongitudinalMode.SCC, custom_long_output=None) -> Any:
  planner: Any = object.__new__(LongitudinalPlannerSP)
  planner.CP = make_cp()
  planner.dt = 0.05
  planner.custom_long_finalizer = CustomLongitudinalFinalizer(planner.CP)
  planner.custom_long = make_custom_long(mode=mode)
  planner.custom_long_output = custom_long_output
  planner._custom_long_output_telemetry = None
  planner._lead_stop_hold_active = False
  planner._lead_stop_hold_gap_increasing_s = 0.0
  planner._lead_stop_hold_missing_s = 0.0
  planner._lead_stop_hold_lead_id = None
  planner._lead_stop_hold_gap_prev_d_rel = None
  planner._lead_stop_hold_gap_baseline_d_rel = None
  planner._stop_hold_release_slew_a_target = None
  planner._stop_hold_release_prep_a_target = None
  planner._stop_hold_release_prep_raw_prev = None
  planner._last_release_block_reason = ""
  return planner


def make_lead(d_rel: float, v_lead: float, v_rel: float, lead_id: int = 1,
              status: bool = True, y_rel: float = 0.0):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    vLead=v_lead,
    vRel=v_rel,
    yRel=y_rel,
    radarTrackId=lead_id,
  )


def make_sm(*, v_ego: float = 0.0, standstill: bool = False, brake_pressed: bool = False,
            gas_pressed: bool = False, force_decel: bool = False, experimental_mode: bool = False,
            model_fresh: bool = True, lead_one=None, lead_two=None) -> FakeSM:
  recv_time = {}
  if model_fresh:
    recv_time["modelV2"] = time.monotonic()
  else:
    recv_time["modelV2"] = time.monotonic() - 1.0
  return FakeSM(
    recv_time=recv_time,
    carState=SimpleNamespace(
      vEgo=v_ego,
      standstill=standstill,
      brakePressed=brake_pressed,
      gasPressed=gas_pressed,
    ),
    controlsState=SimpleNamespace(forceDecel=force_decel),
    selfdriveState=SimpleNamespace(experimentalMode=experimental_mode),
    radarState=SimpleNamespace(leadOne=lead_one, leadTwo=lead_two),
    carControl=SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
    modelV2=SimpleNamespace(),
  )


def test_stopped_close_lead_latches_hold_and_forces_stop():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert e2e_source is False
  assert a_target <= -0.4


def test_stop_hold_selector_keeps_latched_lead_during_pullaway():
  moving_latched = make_lead(d_rel=6.8, v_lead=0.8, v_rel=0.8, lead_id=1)
  stopped_other = make_lead(d_rel=6.0, v_lead=0.0, v_rel=0.0, lead_id=2)
  radar_state = SimpleNamespace(leadOne=moving_latched, leadTwo=stopped_other)

  selected = CustomLongitudinalFinalizer._select_stop_hold_lead(radar_state, latched_id=1)
  assert selected is moving_latched


def test_stopped_close_lead_normalizes_harsh_hold_through_creep_band():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.5, stop_accel=-2.0)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=5.9, v_lead=0.0, v_rel=-0.65)
  sm = make_sm(v_ego=0.65, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.4, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert e2e_source is False
  assert a_target == pytest.approx(-0.5)


def test_authorized_release_clears_mpc_stop_and_returns_bounded_positive_accel():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      standstill_release_allowed=True,
      standstill_release_source="lead_pullaway",
      standstill_release_a_target=0.25,
    ),
  )
  # Pre-latch the stopped lead so the release path runs in a single finalizer tick.
  planner._lead_stop_hold_active = True
  planner._lead_stop_hold_lead_id = 1
  planner._lead_stop_hold_gap_baseline_d_rel = 7.0
  planner._lead_stop_hold_gap_prev_d_rel = 7.0
  planner._lead_stop_hold_gap_increasing_s = 0.5

  lead = make_lead(d_rel=8.0, v_lead=2.0, v_rel=1.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert e2e_source is False
  assert a_target == pytest.approx(0.25)


def test_reset_lead_stop_hold_clears_latch_slew_and_prep_state():
  planner = make_planner()
  planner._lead_stop_hold_active = True
  planner._lead_stop_hold_gap_increasing_s = 0.5
  planner._lead_stop_hold_missing_s = 0.2
  planner._lead_stop_hold_lead_id = 42
  planner._lead_stop_hold_gap_prev_d_rel = 6.0
  planner.custom_long_finalizer.lead_stop_hold_prev_v = 0.2
  planner.custom_long_finalizer.lead_stop_hold_prev_y_rel = 0.1
  planner.custom_long_finalizer.lead_stop_hold_churn_ids = {42, 43}
  planner._lead_stop_hold_gap_baseline_d_rel = 6.0
  planner._stop_hold_release_slew_a_target = 0.1
  planner._stop_hold_release_prep_a_target = -0.2
  planner._stop_hold_release_prep_raw_prev = -0.3

  planner._reset_lead_stop_hold()

  assert planner._lead_stop_hold_active is False
  assert planner._lead_stop_hold_gap_increasing_s == 0.0
  assert planner._lead_stop_hold_missing_s == 0.0
  assert planner._lead_stop_hold_lead_id is None
  assert planner._lead_stop_hold_gap_prev_d_rel is None
  assert planner.custom_long_finalizer.lead_stop_hold_prev_v is None
  assert planner.custom_long_finalizer.lead_stop_hold_prev_y_rel is None
  assert planner.custom_long_finalizer.lead_stop_hold_churn_ids == set()
  assert planner._lead_stop_hold_gap_baseline_d_rel is None
  assert planner._stop_hold_release_slew_a_target is None
  assert planner._stop_hold_release_prep_a_target is None
  assert planner._stop_hold_release_prep_raw_prev is None


def test_internal_stop_hold_reset_uses_planner_monkeypatch_seam():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead),
    mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )
  assert planner._lead_stop_hold_active is True

  original_reset = planner._reset_lead_stop_hold
  reset_snapshots = []

  def reset_hook():
    reset_snapshots.append({
      "active": planner._lead_stop_hold_active,
      "lead_id": planner._lead_stop_hold_lead_id,
      "gap_increasing_s": planner._lead_stop_hold_gap_increasing_s,
    })
    return original_reset()

  planner._reset_lead_stop_hold = reset_hook

  planner.final_longitudinal_output(
    make_sm(v_ego=0.0, gas_pressed=True, lead_one=lead),
    mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert reset_snapshots == [{"active": True, "lead_id": 1, "gap_increasing_s": 0.0}]
  assert planner._lead_stop_hold_active is False


def test_e2e_fresh_model_selects_raw_accel_below_mpc():
  planner = make_planner(
    mode=LongitudinalMode.E2E,
    custom_long_output=make_custom_output(mode=LongitudinalMode.E2E),
  )
  sm = make_sm(v_ego=15.0, model_fresh=True)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.2, mpc_should_stop=False,
    raw_model_a_target=-1.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(-1.0)
  assert a_target < -0.2
  assert e2e_source is True


def test_e2e_stale_model_falls_back_to_mpc_path():
  planner = make_planner(
    mode=LongitudinalMode.E2E,
    custom_long_output=make_custom_output(mode=LongitudinalMode.E2E),
  )
  sm = make_sm(v_ego=15.0, model_fresh=False)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.2, mpc_should_stop=False,
    raw_model_a_target=-1.0, raw_model_should_stop=True,
  )

  assert a_target == pytest.approx(-0.2)
  assert e2e_source is False
  assert should_stop is False


def test_scc_custom_stop_cap_and_curve_confidence_final_cap_apply():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="stop_approach",
      a_target=-0.6,
      actuation=curve_confidence_verdicts(),
    ),
  )
  planner.custom_long.curve_speed_confidence_mode = "apply_conservative"
  sm = make_sm(v_ego=12.0)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert e2e_source is False
  # Custom stop cap pulls 0.0 down to -0.6; curve cap pulls it further to -0.85 floor.
  assert a_target == pytest.approx(-0.85)


def test_scc_cut_in_brake_assist_apply_cap():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise",
      actuation=cut_in_verdicts(),
      # A debug dict alone must never actuate: verdicts are the only actuation interface.
      debug={},
    ),
  )
  planner.custom_long.cut_in_brake_assist_mode = "apply"
  sm = make_sm(v_ego=15.0)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert e2e_source is False
  # Proposed -2.0 is clamped to the gentle -0.6 floor.
  assert a_target == pytest.approx(-0.60)


def test_scc_cut_in_brake_assist_preserves_stronger_braking():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise",
      actuation=cut_in_verdicts(),
    ),
  )
  planner.custom_long.cut_in_brake_assist_mode = "apply"
  sm = make_sm(v_ego=15.0)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-1.5, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  # The cap is restrict-only; stronger existing braking is preserved.
  assert a_target == pytest.approx(-1.5)


@pytest.mark.parametrize(
  "blocker",
  [
    {"model_path_available": False},
    {"eligible": False},
    {"apply_supported": False},
    {"confidence": 0.50},
    {"path_y_rel": None},
    {"path_y_rel": 2.0},
    {"proposed_cap": 0.5},
    {"proposed_cap": float('nan')},
  ],
)
def test_scc_cut_in_brake_assist_apply_noop_cases(blocker):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise", actuation=cut_in_verdicts(**blocker)),
  )
  planner.custom_long.cut_in_brake_assist_mode = "apply"
  sm = make_sm(v_ego=15.0)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.0)


def test_scc_cut_in_brake_assist_apply_blocked_by_driver_override():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise", actuation=cut_in_verdicts()),
  )
  planner.custom_long.cut_in_brake_assist_mode = "apply"

  for flag in ("brake_pressed", "gas_pressed", "force_decel"):
    sm = make_sm(v_ego=15.0, **{flag: True})
    a_target, _, _ = planner.final_longitudinal_output(
      sm, mpc_a_target=0.0, mpc_should_stop=False,
      raw_model_a_target=0.0, raw_model_should_stop=False,
    )
    assert a_target == pytest.approx(0.0), flag


def test_scc_curve_traffic_advisor_apply_cap():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise",
      actuation=curve_traffic_verdicts(),
    ),
  )
  planner.custom_long.curve_traffic_advisor_mode = "apply_conservative"
  sm = make_sm(v_ego=15.0)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert e2e_source is False
  assert a_target == pytest.approx(-0.85)


@pytest.mark.parametrize(
  "blocker",
  [
    {"eligible": False},
    {"apply_supported": False},
    {"confidence": 0.40},
    {"model_stale": True},
    {"traffic_block_reason": "closing_lead"},
    {"a_curve_cap_proposed": 0.1},
    {"a_curve_cap_proposed": float('inf')},
  ],
)
def test_scc_curve_traffic_advisor_apply_noop_cases(blocker):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise", actuation=curve_traffic_verdicts(**blocker)),
  )
  planner.custom_long.curve_traffic_advisor_mode = "apply_conservative"
  sm = make_sm(v_ego=15.0)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.0)


def test_scc_curve_traffic_advisor_apply_blocked_by_driver_override():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise", actuation=curve_traffic_verdicts()),
  )
  planner.custom_long.curve_traffic_advisor_mode = "apply_conservative"

  for flag in ("brake_pressed", "gas_pressed", "force_decel"):
    sm = make_sm(v_ego=15.0, **{flag: True})
    a_target, _, _ = planner.final_longitudinal_output(
      sm, mpc_a_target=0.0, mpc_should_stop=False,
      raw_model_a_target=0.0, raw_model_should_stop=False,
    )
    assert a_target == pytest.approx(0.0), flag


def test_latched_hold_replaces_custom_long_output_telemetry():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise",
      reason="cruise",
      should_stop=False,
    ),
  )
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  telemetry = planner._custom_long_output_telemetry
  assert telemetry is not None
  assert telemetry.should_stop is True
  assert telemetry.selected_intent == "lead_stop_hold"
  assert telemetry.reason == "stopped_lead_latch"


def test_blocked_release_attempt_leaves_persisted_block_reason():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      standstill_release_allowed=True,
      standstill_release_source="lead_pullaway",
      standstill_release_a_target=0.25,
    ),
  )
  planner._lead_stop_hold_active = True
  planner._lead_stop_hold_lead_id = 1
  planner._lead_stop_hold_gap_baseline_d_rel = 7.0
  planner._lead_stop_hold_gap_prev_d_rel = 7.0
  planner._lead_stop_hold_gap_increasing_s = 0.5

  # Lead is authorized by source but not actually moving -> distance gate never reached.
  lead = make_lead(d_rel=8.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "lead_not_moving"


def _arm_stop_hold(planner, d_rel: float = 6.2, lead_id: int = 1, gap_increasing_s: float = 0.30):
  planner._lead_stop_hold_active = True
  planner._lead_stop_hold_lead_id = lead_id
  planner._lead_stop_hold_gap_baseline_d_rel = d_rel
  planner._lead_stop_hold_arm_d_rel = d_rel
  planner._lead_stop_hold_gap_prev_d_rel = d_rel
  planner._lead_stop_hold_gap_increasing_s = gap_increasing_s


def test_sustained_pullaway_prepares_hold_before_absolute_release_distance():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=4.829, lead_id=1, gap_increasing_s=0.15)
  # The lead has moved 7 cm over a sustained opening streak, but is still well short of
  # stoppingDistance + the actual release margin. Prep may unwind the PCM hold; it must
  # not release the latch without the stack's explicit pullaway verdict.
  lead = make_lead(d_rel=4.899, v_lead=0.719, v_rel=0.719, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.328, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target == pytest.approx(-0.2)


def test_crawl_fallback_releases_same_latched_lead_with_invalid_source():
  # The bounded fallback is reserved for a still-stationary lead that has crept the gap open.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.0 < a_target <= planner.custom_long_finalizer._STOP_HOLD_CRAWL_RELEASE_A_MAX


def test_moving_lead_requires_explicit_stack_release_verdict():
  # A moving lead no longer has a second planner-gate authority in the finalizer.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.custom_long.standstill_release_confidence_mode = "gate"
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=6.55, v_lead=0.4, v_rel=0.3, lead_id=1)  # +0.35 m opening
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.1, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.2
  assert planner._last_release_block_reason == "invalid_release_source"


def test_explicit_source_still_requires_physical_opening_margin():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  planner.custom_long.standstill_release_confidence_mode = "gate"
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=6.35, v_lead=0.4, v_rel=0.3, lead_id=1)  # +0.15 m opening
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.1, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert planner._last_release_block_reason == "baseline_opening"


def test_crawl_fallback_rejects_brief_gap_increase():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  # Brief evidence on both axes: streak 0.05 s and displacement 0.4 m (below the 0.5 m
  # creep floor). Cumulative displacement >= 0.5 m now outranks the streak by design.
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.05)
  lead = make_lead(d_rel=6.6, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.2  # prep-softened hold (PCM pre-stage); latch above proves no release


def test_crawl_fallback_rejects_different_lead_id():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=2)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "different_lead_id"


def test_crawl_fallback_rejects_raw_model_stop():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=-0.5, raw_model_should_stop=True,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.4


def test_crawl_fallback_rejects_negative_raw_model_accel():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=-0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.2  # prep-softened hold (PCM pre-stage); latch above proves no release


def test_crawl_fallback_rejects_mpc_brake():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.2, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.4


def test_crawl_fallback_rejects_driver_brake():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  lead = make_lead(d_rel=7.0, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, brake_pressed=True, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True


def test_crawl_fallback_respects_deadband_until_baseline_opens():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1)
  # Small opening below the 0.5 m cumulative baseline threshold; should stay held.
  lead = make_lead(d_rel=6.5, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.2  # prep-softened hold (PCM pre-stage); latch above proves no release


def test_crawl_fallback_large_latched_gap_requires_capped_baseline_opening():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  # Runtime latch baselines are capped at 6m even if the lead is stopped farther away.
  _arm_stop_hold(planner, d_rel=6.0, lead_id=1)
  lead = make_lead(d_rel=6.3, v_lead=0.1, v_rel=0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.2  # prep-softened hold (PCM pre-stage); latch above proves no release


def _make_valid_release_custom_output(a_target: float = 0.25, mode: LongitudinalMode = LongitudinalMode.SCC):
  return make_custom_output(
    mode=mode,
    standstill_release_allowed=True,
    standstill_release_source="lead_pullaway",
    standstill_release_a_target=a_target,
  )


def test_valid_source_releases_before_large_crawl_deadband():
  # Same latched lead moving with only a small baseline opening; valid source should release
  # immediately instead of waiting for the 0.5 m crawl deadband.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target == pytest.approx(0.25)
  assert planner._last_release_block_reason == ""


def test_release_grace_prevents_false_crawl_rearm_but_not_real_stop():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=5.0, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=5.25, v_lead=0.31, v_rel=0.31, lead_id=1)

  released_a, released_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, standstill=True, lead_one=lead),
    mpc_a_target=0.25, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert released_a == pytest.approx(0.25)
  assert released_stop is False
  assert planner._lead_stop_hold_active is False

  crawl = make_lead(d_rel=5.26, v_lead=0.05, v_rel=0.05, lead_id=1)
  crawl_a, crawl_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, standstill=True, lead_one=crawl),
    mpc_a_target=0.10, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert crawl_a >= 0.0
  assert crawl_stop is False
  assert planner._lead_stop_hold_active is False

  stopped = make_lead(d_rel=4.4, v_lead=0.0, v_rel=0.0, lead_id=1)
  stopped_a, stopped_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, standstill=True, lead_one=stopped),
    mpc_a_target=-0.6, mpc_should_stop=True,
    raw_model_a_target=-0.6, raw_model_should_stop=False,
  )
  assert stopped_a <= -0.5
  assert stopped_stop is True
  assert planner._lead_stop_hold_active is True


def test_equal_speed_crawl_does_not_arm_stop_hold():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  lead = make_lead(d_rel=5.5, v_lead=0.3, v_rel=0.0, lead_id=1)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.3, lead_one=lead),
    mpc_a_target=0.1, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert a_target >= 0.0
  assert should_stop is False
  assert planner._lead_stop_hold_active is False


def test_valid_source_blocked_by_driver_brake():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, brake_pressed=True, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert planner._last_release_block_reason == "driver_brake"


def test_valid_source_blocked_by_raw_model_stop():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=True,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert planner._last_release_block_reason == "raw_model_stop"


def test_acc_valid_release_ignores_excluded_raw_model_stop():
  planner = make_planner(
    mode=LongitudinalMode.ACC,
    custom_long_output=_make_valid_release_custom_output(mode=LongitudinalMode.ACC),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=-1.0, raw_model_should_stop=True,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target == pytest.approx(0.25)
  assert planner._last_release_block_reason == ""


def test_route_282_radar_id_churn_keeps_latched_lead_releaseable():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  sequence = [
    (4.16, 0.00, 697),
    (4.18, 0.00, 713),
    (4.20, 0.00, 697),
    (4.22, 0.00, 713),
    (4.32, 0.35, 697),
    (4.42, 0.45, 713),
    (4.52, 0.52, 697),
    (4.62, 0.60, 713),
  ]
  a_target = -0.5
  should_stop = True
  for d_rel, v_lead, lead_id in sequence:
    lead = make_lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_lead, lead_id=lead_id, y_rel=0.1)
    a_target, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead),
      mpc_a_target=0.10 if v_lead >= 0.60 else -0.05, mpc_should_stop=True,
      raw_model_a_target=0.0, raw_model_should_stop=False,
    )

  assert planner.custom_long_finalizer.lead_stop_hold_churn_ids == set()
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target > 0.0
  assert planner._last_release_block_reason == ""


def test_valid_source_blocked_by_mpc_brake():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.2, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert planner._last_release_block_reason == "mpc_brake_veto"


def test_valid_source_blocked_by_weak_opening_rate():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.10, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "lead_not_moving"


def test_valid_source_blocked_until_min_baseline_opening():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.35, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "baseline_opening"


def test_valid_source_blocked_by_stopped_closing_lead():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  # Lead reports zero speed / closing -> not a real pullaway.
  lead = make_lead(d_rel=6.45, v_lead=0.0, v_rel=-0.1, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "lead_not_moving"


def test_settle_hold_arms_low_v_ego_stopping_stationary_lead_near_stop_target():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.1)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=6.2, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.55, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert e2e_source is False
  assert a_target <= -0.4


def test_settle_hold_does_not_rearm_existing_latch():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.1)
  planner.custom_long_finalizer.CP = planner.CP
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.20)
  lead = make_lead(d_rel=6.25, v_lead=0.0, v_rel=0.0, lead_id=1)
  sm = make_sm(v_ego=0.55, lead_one=lead)

  planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner._lead_stop_hold_gap_baseline_d_rel == pytest.approx(6.2)
  assert planner._lead_stop_hold_gap_increasing_s == pytest.approx(0.20 + planner.dt)


@pytest.mark.parametrize("d_rel", [9.0, 10.0])
def test_settle_hold_rejects_far_lead(d_rel):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.1)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=d_rel, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.55, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert e2e_source is False
  assert a_target == pytest.approx(-0.3)


@pytest.mark.parametrize("v_lead,v_rel", [(1.0, 0.5), (0.0, 0.2)])
def test_settle_hold_rejects_moving_or_opening_lead(v_lead, v_rel):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.1)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=6.2, v_lead=v_lead, v_rel=v_rel)
  sm = make_sm(v_ego=0.55, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert e2e_source is False


def test_settle_hold_preserves_legacy_close_stopped_latch():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.1)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert e2e_source is False
  assert a_target <= -0.4


def _release_then_ramp_to_plateau(planner, *, frames: int = 8, mpc_a: float = 1.3):
  # Release via valid source, then ramp mpc frames until the up-jerk slew reaches plateau.
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.45, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)
  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.1, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert planner._lead_stop_hold_active is False
  assert a_target > 0.0
  lead = make_lead(d_rel=9.0, v_lead=2.0, v_rel=1.5, lead_id=1)
  lead.aLeadK = 1.0  # isolate launch-dip damping while the lead is actively pulling away
  sm = make_sm(v_ego=1.5, lead_one=lead)
  for _ in range(frames):
    a_target, _, _ = planner.final_longitudinal_output(
      sm, mpc_a_target=mpc_a, mpc_should_stop=False,
      raw_model_a_target=mpc_a, raw_model_should_stop=False,
    )
  assert a_target >= 1.2
  return sm, a_target


def test_launch_dip_damp_rate_limits_transient_mpc_dip_after_release():
  # Route 282: 1-2 frame mpcA dips (1.3 -> 0.2) during pullaway must not reach the actuator
  # as a step while the radar lead is confirmed departing right after a stop-hold release.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  sm, plateau = _release_then_ramp_to_plateau(planner)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert a_target >= plateau - planner.custom_long_finalizer._APPROACH_DAMP_MAX_JERK * planner.dt - 1e-6
  assert a_target > 1.0


def test_launch_dip_damp_passes_through_real_braking():
  # A genuine braking demand (negative target) right after release is never rate-limited.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  sm, _ = _release_then_ramp_to_plateau(planner)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-1.5, mpc_should_stop=False,
    raw_model_a_target=-1.5, raw_model_should_stop=False,
  )

  assert a_target <= -1.0


def test_lead_catchup_tapers_positive_mpc_accel_from_excess_gap():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise", t_follow=1.45, accel_coast=-0.25,
    ),
  )
  lead = make_lead(d_rel=10.31, v_lead=2.81, v_rel=0.0)
  lead.aLeadK = 0.0
  sm = make_sm(v_ego=2.81, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.48, mpc_should_stop=False,
    raw_model_a_target=0.48, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert 0.15 < a_target < 0.30


def test_lead_catchup_keeps_accel_while_lead_is_still_pulling_away():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="lead_pullaway", t_follow=1.45, accel_coast=-0.25,
    ),
  )
  lead = make_lead(d_rel=10.31, v_lead=2.81, v_rel=0.0)
  lead.aLeadK = 1.0
  sm = make_sm(v_ego=2.81, lead_one=lead)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.48, mpc_should_stop=False,
    raw_model_a_target=0.48, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.48)


def test_lead_catchup_never_weakens_mpc_braking():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise", t_follow=1.45, accel_coast=-0.25,
    ),
  )
  lead = make_lead(d_rel=10.31, v_lead=2.81, v_rel=0.0)
  lead.aLeadK = 0.0
  sm = make_sm(v_ego=3.0, lead_one=lead)

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.4, mpc_should_stop=False,
    raw_model_a_target=-0.4, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# Route 000002ac: slow-crawl lead release (stop-flag cap, carried verdict, sustain)
# ---------------------------------------------------------------------------


def test_crawl_fallback_releases_through_asserted_stop_flags_capped():
  # Route 000002ac t=926: lead crawled 6.4 -> 8.1 m at 0.5 m/s with mpcA +0.68 while the raw
  # model stop and the policy stop verdict stayed asserted; the binary vetoes held -2.0 for
  # 20 s. The stop flags now only cap the released crawl accel.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=6.4, lead_id=1)
  lead = make_lead(d_rel=8.1, v_lead=0.5, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.68, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )

  fin = planner.custom_long_finalizer
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.0 < a_target <= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX + 1e-6
  assert fin.stop_hold_release_sustain_s > 0.0


def test_crawl_fallback_still_blocked_by_negative_model_accel_and_driver_brake():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=6.4, lead_id=1)
  lead = make_lead(d_rel=8.1, v_lead=0.5, v_rel=0.5, lead_id=1)

  # Model actively demanding decel still blocks the fallback.
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.68, mpc_should_stop=False,
    raw_model_a_target=-0.5, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is True
  assert should_stop is True

  # Driver brake still blocks it.
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead, brake_pressed=True), mpc_a_target=0.68, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is True
  assert should_stop is True


def test_carried_release_verdict_survives_model_stop_flicker():
  # Route 000002ac t=252: a lead_pullaway verdict surfaced for one frame (lead still below the
  # 0.30 m/s motion gate), then lapsed when the model stop re-asserted. The carried verdict
  # releases once the live lead-motion gates pass, capped while the flags are asserted.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(a_target=0.40),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.30)

  # Frame 1: verdict live, lead barely moving -> blocked, verdict carried.
  lead = make_lead(d_rel=6.5, v_lead=0.17, v_rel=0.17, lead_id=1)
  planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.3, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert planner._lead_stop_hold_active is True
  assert planner._last_release_block_reason == "lead_not_moving"
  assert planner.custom_long_finalizer.lead_stop_hold_release_carry_s > 0.0

  # Frame 2: verdict lapsed (stop flags re-asserted) but the lead is genuinely moving now.
  # Neutralize the release-slew seed (frame 1's prep hold + ~0 wall-clock dt would clamp
  # the released accel to the prep value; the slew is covered by its own tests).
  planner.custom_long_finalizer.final_a_prev = None
  planner.custom_long_output = make_custom_output(selected_intent="lead_stop_hold", should_stop=True)
  lead = make_lead(d_rel=6.9, v_lead=0.35, v_rel=0.35, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.3, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=True,
  )
  fin = planner.custom_long_finalizer
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.0 < a_target <= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX + 1e-6


def test_sustain_bridges_stack_stop_verdict_until_mpc_asks_to_stop():
  # After a crawl release the stack keeps publishing should_stop for the still-close slow
  # lead; the sustain keeps the release alive while the MPC is comfortable and hands the
  # stop back the moment the MPC demands it.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=6.4, lead_id=1)
  lead = make_lead(d_rel=8.1, v_lead=0.5, v_rel=0.5, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.68, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False and should_stop is False

  # Sustained frame: latch gone, stack verdict still pinned -> release survives, capped.
  lead = make_lead(d_rel=8.3, v_lead=0.5, v_rel=0.4, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.2, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  fin = planner.custom_long_finalizer
  assert should_stop is False
  assert 0.0 < a_target <= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX + 1e-6

  # MPC demands the stop (gap closed) -> sustain ends, stop verdict passes through again.
  lead = make_lead(d_rel=8.3, v_lead=0.5, v_rel=0.4, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.4, lead_one=lead), mpc_a_target=-0.3, mpc_should_stop=True,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert should_stop is True
  assert fin.stop_hold_release_sustain_s == 0.0


def test_release_distance_floor_relaxes_for_sub_floor_latch():
  # Route 000002ac t=1243: latched at 3.3 m; the absolute 4.5 m floor blocked an authorized
  # lead_standstill_launch release at gap 3.6-4.0 m. The floor now never demands more than
  # the latch geometry plus the required opening.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(a_target=0.49),
  )
  _arm_stop_hold(planner, d_rel=3.3, lead_id=1, gap_increasing_s=0.30)
  lead = make_lead(d_rel=3.9, v_lead=0.55, v_rel=0.55, lead_id=1)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.5, mpc_should_stop=False,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target > 0.0


def test_sustain_bridges_stop_approach_advisory_clamp():
  # Route 000002b0 t=915/t=928: after a crawl release the policy sits in stop_approach
  # (model-stop-driven, should_stop False) and scc_custom_stop_cap clamped the creep to its
  # -0.38 approach decel — the car braked at standstill and re-latched. The sustain must
  # bridge the advisory posture exactly like the pinned verdict.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=5.5, lead_id=1)
  lead = make_lead(d_rel=6.2, v_lead=0.35, v_rel=0.35, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
    raw_model_a_target=0.04, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False and should_stop is False

  # Post-release frame in the measured stop_approach posture: must keep creeping, not
  # clamp to the -0.38 approach decel.
  planner.custom_long_output = make_custom_output(
    selected_intent="stop_approach", should_stop=False, a_target=-0.38,
  )
  lead = make_lead(d_rel=6.5, v_lead=0.75, v_rel=0.65, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead), mpc_a_target=0.72, mpc_should_stop=False,
    raw_model_a_target=0.04, raw_model_should_stop=True,
  )
  fin = planner.custom_long_finalizer
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.0 < a_target <= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX + 1e-6
  assert fin.stop_hold_release_sustain_s > 0.0


def test_static_overshoot_release_closes_co_stop_frozen_gap():
  # Route 000002b0 t=948: co-stop settle-freeze parked ego 6.6-7.0 m back with the MPC
  # demanding +0.65 the whole park and the lead never moving. Both at rest + persistent
  # MPC closure demand + gap well past the stop buffer releases into the capped crawl.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=7.0, lead_id=1)
  planner._lead_stop_hold_gap_baseline_d_rel = 5.0  # real arm clamps to _STOP_HOLD_MAX_BASELINE_D_REL
  fin = planner.custom_long_finalizer
  lead = make_lead(d_rel=7.0, v_lead=0.0, v_rel=0.0, lead_id=1)

  # MPC-demand persistence builds over frames; until it is reached the latch must hold.
  for _ in range(fin._STOP_HOLD_MPC_GO_PERSIST_FRAMES - 1):
    _, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
      raw_model_a_target=0.05, raw_model_should_stop=True,
    )
  assert planner._lead_stop_hold_active is True
  assert should_stop is True

  fin.final_a_prev = None  # neutralize the wall-clock release slew seed (covered elsewhere)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.0 < a_target <= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX + 1e-6
  assert fin.stop_hold_release_sustain_s > 0.0


def test_static_overshoot_release_hysteresis_keeps_closed_parks_latched():
  # A park within the overshoot threshold of the stop buffer (the state the closure
  # itself produces) must never re-fire.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=6.5, lead_id=1)
  planner._lead_stop_hold_gap_baseline_d_rel = 5.0
  fin = planner.custom_long_finalizer
  lead = make_lead(d_rel=6.5, v_lead=0.0, v_rel=0.0, lead_id=1)  # overshoot 0.5 < 0.75

  for _ in range(fin._STOP_HOLD_MPC_GO_PERSIST_FRAMES + 5):
    _, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
      raw_model_a_target=0.05, raw_model_should_stop=True,
    )
  assert planner._lead_stop_hold_active is True
  assert should_stop is True


def test_sustain_rides_through_positive_mpc_stop_chatter():
  # Route 000002b2 t=281: three consecutive stop-bit frames with mpcA still positive are
  # launch-transition chatter and must neither cancel the sustain nor allow a re-latch
  # that interrupts an in-progress creep.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  _arm_stop_hold(planner, d_rel=6.4, lead_id=1)
  lead = make_lead(d_rel=8.1, v_lead=0.5, v_rel=0.5, lead_id=1)
  _, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.68, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False and should_stop is False

  fin = planner.custom_long_finalizer
  for _ in range(5):  # stop bit asserted with positive mpcA: chatter, not a demand
    lead = make_lead(d_rel=8.2, v_lead=0.4, v_rel=0.2, lead_id=1)
    a_target, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.4, lead_one=lead), mpc_a_target=0.1, mpc_should_stop=True,
      raw_model_a_target=0.05, raw_model_should_stop=True,
    )
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target > 0.0
  assert fin.stop_hold_release_sustain_s > 0.0


def _launch_release(planner):
  """Arm a hold and release it behind a departing lead; returns the finalizer."""
  _arm_stop_hold(planner, d_rel=5.5, lead_id=1)
  lead = make_lead(d_rel=6.2, v_lead=0.5, v_rel=0.5, lead_id=1)
  _, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.6, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False and should_stop is False
  return planner.custom_long_finalizer


def test_launch_floor_carries_confirmed_departure_through_weak_mpc():
  # Route 000002b2 t=753: after release the policy intent flipped to lead_follow, the
  # verdict lapsed, and the command collapsed to raw mpcA 0.07-0.3 while the lead was
  # clearly going — both driver gas presses landed in that window.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner)

  # Post-release frame: posture cleared, lead departing fast, MPC still ramping.
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.8, v_lead=1.2, v_rel=1.1, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert should_stop is False
  # downstream dampers may shave a frame's worth of jerk off the floored value
  assert fin._STOP_HOLD_LAUNCH_FLOOR_A - 0.05 <= a_target <= fin._STOP_HOLD_LAUNCH_FLOOR_A + 1e-6


def test_launch_floor_stays_off_below_breakout_and_without_grace():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner)

  # Lead only crawling (below the breakout opening): the gentle authorities own it.
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.6, v_lead=0.4, v_rel=0.4, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert a_target == pytest.approx(0.12)

  # No recent release (grace expired): raw mpcA passes through even for a departing lead.
  fin.launch_dip_grace_s = 0.0
  fin.final_a_prev = None
  lead = make_lead(d_rel=8.0, v_lead=1.5, v_rel=1.4, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert a_target == pytest.approx(0.12)


def test_launch_floor_holds_through_the_chase():
  # Route 000002b5 t=1264: ego accelerating after the lead dropped vRel below the breakout
  # entry threshold while the lead was still clearly departing — the floor must hold.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner)

  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  lead = make_lead(d_rel=6.0, v_lead=1.0, v_rel=0.45, lead_id=1)  # chasing: vRel below entry
  # a few frames: the jerk-limited stages climb toward the floored value; the lead-catchup
  # cushion legitimately trims the 0.60 proposal at this tight gap (caps run after the
  # floor by design) — the point is the command holds well above the sagging 0.2 mpcA.
  for _ in range(4):
    a_target, _, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.55, lead_one=lead), mpc_a_target=0.2, mpc_should_stop=False,
      raw_model_a_target=0.3, raw_model_should_stop=False,
    )
  assert 0.4 < a_target <= fin._STOP_HOLD_LAUNCH_FLOOR_A + 1e-6

  # Ego has matched the lead's speed: floor off — the command drops back to the
  # MPC/cushion-owned value instead of being raised.
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.5, v_lead=1.0, v_rel=0.05, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.95, lead_one=lead), mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.3, raw_model_should_stop=False,
  )
  assert a_target <= 0.2 + 1e-6


def _departing_lead(d_rel=8.0, v_lead=1.8, v_rel=0.8, a_lead_k=0.8, lead_id=1):
  lead = make_lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_rel, lead_id=lead_id)
  lead.aLeadK = a_lead_k
  return lead


def test_departing_lead_coast_clamps_gap_restore_braking():
  # Routes 2b5 t=1110 / 2b0 t=338: lead re-accelerates on the green while the MPC keeps
  # braking -0.6..-0.8 for 3-4.5 s to restore the inflated time gap. With the departure
  # sustained, shallow braking clamps to coast.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  fin = planner.custom_long_finalizer
  for _ in range(fin._DEPARTING_LEAD_PERSIST_FRAMES + 6):
    a_target, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=3.3, lead_one=_departing_lead()), mpc_a_target=-0.65, mpc_should_stop=False,
      raw_model_a_target=-0.3, raw_model_should_stop=False,
    )
  assert should_stop is False
  assert a_target > -0.05  # coast, jerk-limited stages settle at ~0


def test_departing_lead_coast_never_reshapes_deep_braking_or_stop_posture():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  fin = planner.custom_long_finalizer

  # Deep demand passes through untouched.
  for _ in range(fin._DEPARTING_LEAD_PERSIST_FRAMES + 3):
    a_target, _, _ = planner.final_longitudinal_output(
      make_sm(v_ego=3.3, lead_one=_departing_lead()), mpc_a_target=-1.5, mpc_should_stop=False,
      raw_model_a_target=-1.5, raw_model_should_stop=False,
    )
  assert a_target == pytest.approx(-1.5)

  # Raw model stop asserted: braking is never clamped.
  for _ in range(fin._DEPARTING_LEAD_PERSIST_FRAMES + 3):
    a_target, _, _ = planner.final_longitudinal_output(
      make_sm(v_ego=3.3, lead_one=_departing_lead()), mpc_a_target=-0.65, mpc_should_stop=False,
      raw_model_a_target=-0.3, raw_model_should_stop=True,
    )
  assert a_target <= -0.5


def test_departing_lead_coast_requires_sustained_lead_acceleration():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  # Lead opening but NOT accelerating (aLeadK ~ 0): the clamp must stay off.
  for _ in range(10):
    a_target, _, _ = planner.final_longitudinal_output(
      make_sm(v_ego=3.3, lead_one=_departing_lead(a_lead_k=0.0)), mpc_a_target=-0.65, mpc_should_stop=False,
      raw_model_a_target=-0.3, raw_model_should_stop=False,
    )
  assert a_target == pytest.approx(-0.65)
