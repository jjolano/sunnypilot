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
