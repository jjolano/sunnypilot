"""Phase 5A characterization tests for the longitudinal finalizer.

These tests lock the behavior of ``LongitudinalPlannerSP.final_longitudinal_output``
and its helpers before the finalizer arbitration is extracted into a dedicated module.
Only test code is allowed to change in Phase 5A.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import CutInBrakeAssistResult
from openpilot.sunnypilot.custom.longitudinal.curve_traffic_advisor import CurveTrafficAdvisorResult
from openpilot.sunnypilot.custom.longitudinal.departure_prediction import (
  DeparturePredictionEvidence,
  PHASE_ARMING,
  PHASE_INACTIVE,
  PHASE_PREDICTED,
  PERSISTENCE_S,
  TIMEOUT_S,
)
from openpilot.sunnypilot.custom.longitudinal.finalizer import (
  CustomLongitudinalFinalizer,
  _FinalArbitration,
  _InputSnapshot,
)
from openpilot.sunnypilot.custom.longitudinal.lead_cushion import LowSpeedGapClosureRequest
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.stack import ActuationVerdicts
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def cut_in_verdicts(model_path_available: bool = True, **overrides) -> ActuationVerdicts:
  fields: dict[str, Any] = dict(eligible=True, apply_supported=True, confidence=0.80, path_y_rel=0.3, proposed_cap=-2.0)
  fields.update(overrides)
  return ActuationVerdicts(cut_in_brake_assist=CutInBrakeAssistResult(**fields), model_path_available=model_path_available)


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
            model_fresh: bool = True, lead_one=None, lead_two=None, long_active: bool = False) -> FakeSM:
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
    carControl=SimpleNamespace(enabled=True, longActive=long_active, cruiseControl=SimpleNamespace(override=False)),
    modelV2=SimpleNamespace(),
  )


def test_stopped_close_lead_latches_hold_and_forces_stop():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.custom_long_finalizer.launch_floor_fade_pending = True
  planner.custom_long_finalizer.approach_damp_a_prev = 0.4
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.3, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert planner.custom_long_finalizer.launch_floor_fade_pending is False
  assert planner.custom_long_finalizer.approach_damp_a_prev is None
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
  planner.custom_long_finalizer.launch_floor_fade_pending = True
  planner.custom_long_finalizer.approach_damp_a_prev = 0.4

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
  assert planner.custom_long_finalizer.launch_floor_fade_pending is False
  assert planner.custom_long_finalizer.approach_damp_a_prev is None


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


@pytest.mark.parametrize("mode", [LongitudinalMode.ACC, LongitudinalMode.E2E])
def test_non_scc_fade_cleanup_preserves_approach_damping(mode):
  planner = make_planner(
    mode=mode,
    custom_long_output=make_custom_output(mode=mode),
  )
  fin = planner.custom_long_finalizer
  fin.launch_floor_fade_pending = True
  fin.approach_damp_a_prev = 0.40

  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, long_active=True, model_fresh=True),
    mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert fin.launch_floor_fade_pending is False
  assert a_target == pytest.approx(0.25)
  assert fin.approach_damp_a_prev == pytest.approx(0.25)


def test_scc_custom_stop_cap_applies():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="stop_approach",
      a_target=-0.6,
      actuation=ActuationVerdicts(),
    ),
  )
  sm = make_sm(v_ego=12.0)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert e2e_source is False
  # Custom stop cap pulls 0.0 down to -0.6. The curve-confidence cap that used to pull this
  # further to the -0.85 floor was deleted 2026-07-24 (0.07% eligibility over 101,741 frames).
  assert a_target == pytest.approx(-0.6)


@pytest.mark.parametrize("mpc_a_target", (-1.0, -1.5))
def test_moving_deep_mpc_stop_demand_passes_through_with_stop_posture(mpc_a_target):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  fin = planner.custom_long_finalizer

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    make_sm(v_ego=15.0), mpc_a_target=mpc_a_target, mpc_should_stop=True,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is False
  assert fin.stop_hold_release_sustain_s == 0.0
  assert fin.stop_hold_release_slew_a_target is None
  assert fin.approach_damp_a_prev is None
  assert a_target == pytest.approx(mpc_a_target)
  assert should_stop is True
  assert e2e_source is False


def test_scc_corroborated_model_stop_cap_survives_nonbinding_intent():
  # Route 2ca t=1220: the pre-MPC seed won arbitration as "cruise", then MPC
  # relaxed it. The still-active corroborated stop posture must carry the custom
  # restriction through finalization independently of the binding intent label.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(
      selected_intent="cruise",
      reason="cruise",
      a_target=-2.18,
      model_stop_corroborated=True,
    ),
  )
  sm = make_sm(v_ego=10.0)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-0.73, mpc_should_stop=False,
    raw_model_a_target=-2.11, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(-2.18)
  assert should_stop is False
  assert e2e_source is False


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


@pytest.mark.parametrize("v_ego, expected", [(0.44, True), (0.46, False)])
def test_release_prep_uses_fork_stop_threshold(v_ego, expected):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=_make_valid_release_custom_output(),
  )
  planner.CP = make_cp(v_ego_stopping=0.5)
  planner.custom_long_finalizer.CP = planner.CP
  _arm_stop_hold(planner, d_rel=6.0, lead_id=1, gap_increasing_s=0.15)
  lead = make_lead(d_rel=6.2, v_lead=0.8, v_rel=0.5, lead_id=1)
  sm = make_sm(v_ego=v_ego, lead_one=lead)
  snapshot = _InputSnapshot.build(
    planner.custom_long_finalizer, sm, planner.custom_long, planner.custom_long_output,
    is_e2e=False, model_stale=False, dt=0.0,
    mpc_a_target=-0.05, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert snapshot.v_ego_stopping == pytest.approx(0.25)
  assert planner.custom_long_finalizer._stop_hold_release_prep_applies(
    sm, lead, planner.custom_long, planner.custom_long_output,
    lead_d_rel=6.2, lead_v=0.8, lead_v_rel=0.5,
    mpc_a_target=-0.05, raw_model_a_target=0.0, raw_model_should_stop=False,
  ) is expected


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


def test_stopped_closing_lead_cannot_release_on_cruise_target():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.30)
  lead = make_lead(d_rel=6.8, v_lead=0.0, v_rel=-0.1, lead_id=1)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead),
    mpc_a_target=0.10, mpc_should_stop=True,
    raw_model_a_target=0.10, raw_model_should_stop=False,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert a_target <= -0.4
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
  sm = make_sm(v_ego=0.0, lead_one=lead, long_active=True)
  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.1, mpc_should_stop=True,
    raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert planner._lead_stop_hold_active is False
  assert a_target > 0.0
  lead = make_lead(d_rel=9.0, v_lead=2.0, v_rel=1.5, lead_id=1)
  lead.aLeadK = 1.0  # isolate launch-dip damping while the lead is actively pulling away
  sm = make_sm(v_ego=1.5, lead_one=lead, long_active=True)
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


def _make_gap_closure_output(requested_accel: float = 0.25, lead_id: int = 1, lead_idx: int = 0,
                             lead_d_rel: float = 10.3, lead_v_lead: float = 1.2,
                             lead_v_rel: float = 0.2, lead_y_rel: float = 0.0, **overrides):
  request = LowSpeedGapClosureRequest(
    requested_accel=requested_accel, desired_closing_speed=1.0, follow_gap=6.28,
    lead_track_id=lead_id, lead_idx=lead_idx, lead_confidence=1.0,
    lead_stable=True, lead_radar=True,
    lead_d_rel=lead_d_rel, lead_v_lead=lead_v_lead, lead_v_rel=lead_v_rel, lead_y_rel=lead_y_rel,
  )
  return make_custom_output(low_speed_gap_closure=request, t_follow=1.5, **overrides)


def _finalize_gap_closure(*, v_ego: float = 1.0, v_lead: float = 1.2, d_rel: float = 10.3,
                          a_lead: float = 0.0, mpc_a: float = 0.0, model_a: float = 0.0,
                          mpc_stop: bool = False, model_stop: bool = False,
                          mode: LongitudinalMode = LongitudinalMode.SCC,
                          **sm_overrides):
  planner = make_planner(mode=mode, custom_long_output=_make_gap_closure_output(mode=mode))
  lead = make_lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_lead - v_ego, lead_id=1)
  lead.radar = True
  lead.aLeadK = a_lead
  sm = make_sm(v_ego=v_ego, lead_one=lead, long_active=True, **sm_overrides)
  return planner, planner.final_longitudinal_output(
    sm, mpc_a_target=mpc_a, mpc_should_stop=mpc_stop,
    raw_model_a_target=model_a, raw_model_should_stop=model_stop,
  )


def _gap_closure_floor(*, base_a_target: float = 0.0, mpc_a: float = 0.0, model_a: float = 0.0,
                       mpc_stop: bool = False, model_stop: bool = False, **sm_overrides):
  planner = make_planner(custom_long_output=_make_gap_closure_output())
  lead = make_lead(d_rel=10.3, v_lead=1.2, v_rel=0.2, lead_id=1)
  lead.radar = True
  lead.aLeadK = 0.0
  sm = make_sm(v_ego=1.0, lead_one=lead, long_active=True, **sm_overrides)
  fin = planner.custom_long_finalizer
  snapshot = _InputSnapshot.build(
    fin, sm, planner.custom_long, planner.custom_long_output,
    is_e2e=False, model_stale=False, dt=0.05,
    mpc_a_target=mpc_a, mpc_should_stop=mpc_stop,
    raw_model_a_target=model_a, raw_model_should_stop=model_stop,
  )
  return _FinalArbitration.scc_low_speed_gap_closure_floor(
    fin, base_a_target, snapshot, should_stop=False, release_mpc_stop=False,
  )


@pytest.mark.parametrize(
  "kwargs, applies",
  [
    ({"base_a_target": -0.100001}, False),
    ({"base_a_target": -0.10}, True),
    ({"base_a_target": -0.099999}, True),
    ({"mpc_a": -0.100001}, False),
    ({"mpc_a": -0.10}, False),
    ({"mpc_a": -0.099999}, True),
    ({"model_a": -0.100001}, False),
    ({"model_a": -0.10}, False),
    ({"model_a": -0.099999}, True),
  ],
)
def test_scc_low_speed_gap_closure_uses_exact_negative_thresholds(kwargs, applies):
  base_a_target = kwargs.get("base_a_target", 0.0)
  a_target = _gap_closure_floor(**kwargs)
  if applies:
    assert a_target > 0.0
  else:
    assert a_target == pytest.approx(base_a_target)


@pytest.mark.parametrize("mpc_a,model_a", [(0.0, 0.0), (-0.05, -0.05)])
def test_scc_low_speed_gap_closure_applies_bounded_route_like_authority(mpc_a, model_a):
  _, (a_target, should_stop, e2e_source) = _finalize_gap_closure(mpc_a=mpc_a, model_a=model_a)

  assert should_stop is False
  assert e2e_source is False
  assert 0.0 < a_target <= 0.25


@pytest.mark.parametrize(
  "kwargs",
  [
    {"v_ego": 2.0, "v_lead": 0.9},
    {"v_lead": 0.1},
    {"d_rel": 6.0},
    {"a_lead": -1.01},
    {"v_ego": 2.1},
  ],
)
def test_scc_low_speed_gap_closure_local_kinematics_remove_request(kwargs):
  _, (a_target, should_stop, e2e_source) = _finalize_gap_closure(**kwargs)

  assert (a_target, should_stop, e2e_source) == (0.0, False, False)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"mpc_stop": True},
    {"model_stop": True},
    {"mpc_a": -0.10},
    {"model_a": -0.10},
    {"gas_pressed": True},
    {"brake_pressed": True},
  ],
)
def test_scc_low_speed_gap_closure_safety_gates_leave_base_unchanged(kwargs):
  _, (a_target, _, e2e_source) = _finalize_gap_closure(**kwargs)

  assert a_target == pytest.approx(kwargs.get("mpc_a", 0.0))
  assert e2e_source is False


def test_scc_low_speed_gap_closure_blocks_a_close_alternate_lead():
  alternate = make_lead(d_rel=8.0, v_lead=0.5, v_rel=-0.5, lead_id=2)
  alternate.radar = True
  alternate.aLeadK = 0.0
  _, (a_target, _, _) = _finalize_gap_closure(lead_two=alternate)

  assert a_target == pytest.approx(0.0)


def test_scc_low_speed_gap_closure_follows_requested_slot_through_track_id_churn():
  planner = make_planner(mode=LongitudinalMode.SCC, custom_long_output=_make_gap_closure_output())
  lead = make_lead(d_rel=10.31, v_lead=1.21, v_rel=0.21, lead_id=2)
  lead.radar = True
  lead.aLeadK = 0.0

  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=1.0, lead_one=lead, long_active=True),
    mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert 0.0 < a_target <= 0.25


def test_scc_low_speed_gap_closure_treats_equivalent_duplicate_slots_as_one_lead():
  planner = make_planner(mode=LongitudinalMode.SCC, custom_long_output=_make_gap_closure_output())
  requested = make_lead(d_rel=10.3, v_lead=1.2, v_rel=0.2, lead_id=1)
  duplicate = make_lead(d_rel=10.35, v_lead=1.18, v_rel=0.18, lead_id=2)
  for lead in (requested, duplicate):
    lead.radar = True
    lead.aLeadK = 0.0

  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=1.0, lead_one=requested, lead_two=duplicate, long_active=True),
    mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert 0.0 < a_target <= 0.25


def test_scc_low_speed_gap_closure_rejects_non_requested_lead_identity():
  planner = make_planner(mode=LongitudinalMode.SCC, custom_long_output=_make_gap_closure_output())
  lead = make_lead(d_rel=8.0, v_lead=0.5, v_rel=-0.5, lead_id=9)
  lead.radar = True
  lead.aLeadK = 0.0

  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=1.0, lead_one=lead, long_active=True),
    mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.0)


def test_scc_low_speed_gap_closure_removal_is_immediate_after_a_gate_drop():
  planner, _ = _finalize_gap_closure()
  lead = make_lead(d_rel=10.3, v_lead=1.2, v_rel=0.2, lead_id=1)
  lead.radar = True
  lead.aLeadK = 0.0
  blocked_sm = make_sm(v_ego=1.0, gas_pressed=True, lead_one=lead, long_active=True)

  a_target, _, _ = planner.final_longitudinal_output(
    blocked_sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.0)


def test_nonbinding_gap_closure_request_preserves_baseline_approach_damping():
  def run(output):
    planner = make_planner(mode=LongitudinalMode.SCC, custom_long_output=output)
    lead = make_lead(d_rel=10.3, v_lead=2.1, v_rel=0.0, lead_id=1)
    lead.radar = True
    lead.aLeadK = 0.0
    sm = make_sm(v_ego=2.1, lead_one=lead, long_active=True)
    planner.final_longitudinal_output(
      sm, mpc_a_target=0.0, mpc_should_stop=False,
      raw_model_a_target=0.0, raw_model_should_stop=False,
    )
    return planner.final_longitudinal_output(
      sm, mpc_a_target=0.2, mpc_should_stop=False,
      raw_model_a_target=0.2, raw_model_should_stop=False,
    )[0]

  baseline = run(make_custom_output())
  nonbinding = run(_make_gap_closure_output())

  assert baseline == pytest.approx(0.15)
  assert nonbinding == pytest.approx(baseline)


@pytest.mark.parametrize("mode", [LongitudinalMode.ACC, LongitudinalMode.E2E])
def test_scc_low_speed_gap_closure_never_applies_outside_scc(mode):
  _, (a_target, _, e2e_source) = _finalize_gap_closure(mode=mode)

  assert a_target == pytest.approx(0.0)
  assert e2e_source is False


def test_scc_low_speed_gap_closure_upward_slew_uses_prior_final_target():
  planner = make_planner(mode=LongitudinalMode.SCC, custom_long_output=make_custom_output())
  lead = make_lead(d_rel=10.3, v_lead=1.2, v_rel=0.2, lead_id=1)
  lead.radar = True
  lead.aLeadK = 0.0
  sm = make_sm(v_ego=1.0, lead_one=lead, long_active=True)
  planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )
  planner.custom_long_output = _make_gap_closure_output()

  a_target, _, _ = planner.final_longitudinal_output(
    sm, mpc_a_target=0.0, mpc_should_stop=False,
    raw_model_a_target=0.0, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.8 * planner.dt)


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
  # Production geometry: STOP_DISTANCE fallback is 5.0, so 7.0 m is a 2.0 m overshoot
  # (>= the 1.25 closer threshold).
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  planner.CP = make_cp(stopping_distance=5.0)
  planner.custom_long_finalizer.CP = planner.CP
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


def test_static_overshoot_5_9m_production_geometry_stays_latched():
  # Route 00000306: the car landed 5.9m (overshoot 0.9 over the production 5.0m stop
  # distance) and the old 0.75 threshold fired the crawl closer -- 5.5s of +0.1 command
  # while the PCM brake held, then a +1.0 m/s^2 lurch when it released, re-stop, repeat.
  # With the threshold at 1.25 a 5-6m stopped gap is acceptable and must stay latched.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  planner.CP = make_cp(stopping_distance=5.0)
  planner.custom_long_finalizer.CP = planner.CP
  _arm_stop_hold(planner, d_rel=5.9, lead_id=1)
  fin = planner.custom_long_finalizer
  lead = make_lead(d_rel=5.9, v_lead=0.0, v_rel=0.0, lead_id=1)

  for _ in range(fin._STOP_HOLD_MPC_GO_PERSIST_FRAMES + 5):
    _, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
      raw_model_a_target=0.05, raw_model_should_stop=True,
    )
  assert planner._lead_stop_hold_active is True
  assert should_stop is True


def test_static_overshoot_6_5m_production_geometry_still_releases_to_breakaway():
  # The closer must still own the genuine co-stop frozen gaps (route 000002b0: 6.6-7.0m)
  # at production geometry, and the accepted release must reach the breakaway region --
  # the generic lead catch-up cap must not cut it below the Toyota brake-release
  # threshold (route 00000306: 0.40 request reduced to ~0.094).
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  planner.CP = make_cp(stopping_distance=5.0)
  planner.custom_long_finalizer.CP = planner.CP
  _arm_stop_hold(planner, d_rel=6.5, lead_id=1)
  planner._lead_stop_hold_gap_baseline_d_rel = 5.0
  fin = planner.custom_long_finalizer
  lead = make_lead(d_rel=6.5, v_lead=0.0, v_rel=0.0, lead_id=1)

  for _ in range(fin._STOP_HOLD_MPC_GO_PERSIST_FRAMES - 1):
    _, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
      raw_model_a_target=0.05, raw_model_should_stop=True,
    )
  assert planner._lead_stop_hold_active is True

  fin.final_a_prev = None  # neutralize the wall-clock release slew seed
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead), mpc_a_target=0.65, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False
  assert should_stop is False
  assert a_target >= fin._STOP_HOLD_CRAWL_MODEL_STOP_A_MAX - 1e-6  # breakaway region, not catch-up-capped


def test_scc_raw_model_stop_does_not_inflate_latched_hold_magnitude():
  # Route 00000306: the standstill normalization was gated on _model_stop_blocks_release,
  # so in SCC a raw model stop exposed Toyota's stopAccel -2.0 as the hold command -- the
  # "very odd stop" slam at 0.1 m/s. Model stop keeps blocking RELEASE, but the hold
  # magnitude of a latched same-lead hold is the calibrated -0.50.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="cruise"),
  )
  planner.CP = make_cp(v_ego_stopping=0.5, stop_accel=-2.0)
  planner.custom_long_finalizer.CP = planner.CP
  lead = make_lead(d_rel=5.9, v_lead=0.0, v_rel=-0.65)
  sm = make_sm(v_ego=0.65, lead_one=lead)

  a_target, should_stop, e2e_source = planner.final_longitudinal_output(
    sm, mpc_a_target=-2.0, mpc_should_stop=True,
    raw_model_a_target=-2.0, raw_model_should_stop=True,
  )

  assert planner._lead_stop_hold_active is True
  assert should_stop is True
  assert e2e_source is False
  assert a_target == pytest.approx(-0.5)


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


def _launch_release(planner, *, long_active: bool = False):
  """Arm a hold and release it behind a departing lead; returns the finalizer."""
  _arm_stop_hold(planner, d_rel=5.5, lead_id=1)
  lead = make_lead(d_rel=6.2, v_lead=0.5, v_rel=0.5, lead_id=1)
  _, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=lead, long_active=long_active), mpc_a_target=0.6, mpc_should_stop=False,
    raw_model_a_target=0.05, raw_model_should_stop=True,
  )
  assert planner._lead_stop_hold_active is False and should_stop is False
  return planner.custom_long_finalizer


def _seed_launch_floor_fade():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner, long_active=True)
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  lead = make_lead(d_rel=30.0, v_lead=8.0, v_rel=1.5, lead_id=1)
  fin.final_a_prev = None
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, lead_one=lead, long_active=True), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.12, raw_model_should_stop=False,
  )
  assert should_stop is False
  assert a_target > 0.12
  assert fin.launch_floor_fade_pending is True
  fin.launch_dip_grace_s = 0.0
  fin.stop_hold_release_sustain_s = 0.0
  fin.stop_hold_release_slew_a_target = None
  fin.approach_damp_a_prev = 0.4
  return planner, fin, lead


def _seed_active_launch_floor_grace():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  fin = planner.custom_long_finalizer
  fin.launch_dip_grace_s = 0.20
  lead = make_lead(d_rel=30.0, v_lead=8.0, v_rel=1.5, lead_id=1)
  return planner, fin, lead


@pytest.mark.parametrize("case", ["disabled", "output_disabled", "output_fault", "faulted", "inactive"])
def test_launch_floor_requires_active_lifecycle_during_grace(case):
  planner, fin, lead = _seed_active_launch_floor_grace()
  sm_kwargs = {"long_active": True}
  if case == "disabled":
    planner.dec = SimpleNamespace(active=lambda: True)
    planner.custom_long.enabled = False
  elif case == "output_disabled":
    planner.custom_long_output = make_custom_output(selected_intent="lead_follow", enabled=False)
  elif case == "output_fault":
    planner.custom_long_output = make_custom_output(selected_intent="lead_follow", fault_class="test_fault")
  elif case == "faulted":
    planner.custom_long.fault_class = "test_fault"
  elif case == "inactive":
    sm_kwargs["long_active"] = False

  assert fin.launch_dip_grace_s > 0.0
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, lead_one=lead, **sm_kwargs),
    mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )

  assert a_target == pytest.approx(0.2)
  assert fin.launch_dip_grace_s == pytest.approx(0.0)
  assert fin.launch_floor_fade_pending is False


def test_launch_floor_grace_does_not_reappear_after_custom_reenable():
  planner, fin, lead = _seed_active_launch_floor_grace()
  planner.dec = SimpleNamespace(active=lambda: True)
  planner.custom_long.enabled = False

  disabled_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, lead_one=lead, long_active=True),
    mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert disabled_target == pytest.approx(0.2)
  assert fin.launch_dip_grace_s == pytest.approx(0.0)
  assert fin.launch_floor_fade_pending is False

  planner.custom_long.enabled = True
  reenabled_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, lead_one=lead, long_active=True),
    mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )

  assert reenabled_target == pytest.approx(0.2)
  assert fin.launch_dip_grace_s == pytest.approx(0.0)
  assert fin.launch_floor_fade_pending is False


def test_launch_floor_fade_converges_at_bounded_positive_jerk_after_grace():
  planner, fin, lead = _seed_launch_floor_fade()
  fin.launch_dip_grace_s = 0.05
  previous = fin.final_a_prev
  outputs = []
  for _ in range(4):
    output, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=2.5, lead_one=lead, long_active=True), mpc_a_target=0.12, mpc_should_stop=False,
      raw_model_a_target=0.12, raw_model_should_stop=False,
    )
    outputs.append(output)
    assert should_stop is False

  assert previous is not None
  assert outputs[-1] == pytest.approx(0.12)
  assert all(abs(current - prior) <= fin._APPROACH_DAMP_MAX_JERK * 0.05 + 1e-6
             for prior, current in zip((previous, *outputs[:-1]), outputs, strict=True))
  assert fin.launch_floor_fade_pending is False


def test_launch_floor_fade_clears_on_eligible_nonbinding_grace_frame():
  planner, fin, lead = _seed_launch_floor_fade()
  fin.launch_dip_grace_s = 1.0

  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.5, lead_one=lead, long_active=True), mpc_a_target=0.7, mpc_should_stop=False,
    raw_model_a_target=0.7, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert a_target == pytest.approx(0.7)
  assert fin.launch_floor_fade_pending is False


def test_launch_floor_fade_runs_before_lead_catchup_cap():
  planner, fin, _ = _seed_launch_floor_fade()
  fin.launch_dip_grace_s = 0.05
  planner.custom_long_output = make_custom_output(
    selected_intent="cruise", t_follow=1.45, accel_coast=-0.25,
  )
  lead = make_lead(d_rel=10.31, v_lead=2.81, v_rel=0.2, lead_id=1)
  previous = fin.final_a_prev
  assert previous is not None
  fade_proposal = max(0.2, previous - fin._APPROACH_DAMP_MAX_JERK * 0.05)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=2.81, lead_one=lead, long_active=True), mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )

  assert should_stop is False
  assert 0.0 < a_target < fade_proposal
  assert fin.launch_floor_fade_pending is True


@pytest.mark.parametrize("case", [
  "nonfinite_mpc", "nonfinite_raw", "no_prior", "nonpositive", "no_downward",
  "negative_mpc", "mpc_stop", "raw_model_stop", "custom_stop",
  "lead_loss", "below_breakout", "lead_closure", "brake", "gas", "force_decel",
  "out_of_regime", "inactive", "non_scc", "disabled", "faulted", "output_disabled",
  "output_fault", "model_stop_corroborated", "stop_approach",
])
def test_launch_floor_fade_cancellation_adds_no_authority_and_clears_pending(case):
  planner, fin, lead = _seed_launch_floor_fade()
  mpc_a_target = 0.2
  raw_model_a_target = 0.2
  mpc_should_stop = False
  raw_model_should_stop = False
  sm_kwargs: dict[str, Any] = {"long_active": True}
  expected = 0.2

  if case == "nonfinite_mpc":
    mpc_a_target = float("nan")
    expected = 0.0
  elif case == "nonfinite_raw":
    raw_model_a_target = float("nan")
  elif case == "no_prior":
    fin.final_a_prev = None
  elif case == "nonpositive":
    mpc_a_target = 0.0
    expected = 0.0
  elif case == "no_downward":
    mpc_a_target = 0.7
    expected = 0.7
  elif case == "negative_mpc":
    mpc_a_target = -0.05
    raw_model_a_target = -0.05
    expected = -0.05
  elif case == "mpc_stop":
    mpc_should_stop = True
  elif case == "raw_model_stop":
    raw_model_should_stop = True
  elif case == "custom_stop":
    planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=True)
  elif case == "lead_loss":
    lead = None
  elif case == "below_breakout":
    lead = make_lead(d_rel=15.0, v_lead=0.4, v_rel=0.4, lead_id=1)
    expected = 0.0
  elif case == "lead_closure":
    lead = make_lead(d_rel=4.0, v_lead=2.0, v_rel=1.5, lead_id=1)
    expected = 0.0
  elif case == "brake":
    sm_kwargs["brake_pressed"] = True
  elif case == "gas":
    sm_kwargs["gas_pressed"] = True
  elif case == "force_decel":
    sm_kwargs["force_decel"] = True
  elif case == "out_of_regime":
    sm_kwargs["v_ego"] = 5.1
    lead = make_lead(d_rel=30.0, v_lead=8.0, v_rel=1.5, lead_id=1)
  elif case == "inactive":
    sm_kwargs["long_active"] = False
    expected = 0.25
  elif case == "non_scc":
    planner.custom_long.mode = LongitudinalMode.ACC
    expected = 0.25
  elif case == "disabled":
    planner.dec = SimpleNamespace(active=lambda: False)
    planner.custom_long.enabled = False
  elif case == "faulted":
    planner.custom_long.fault_class = "test_fault"
  elif case == "output_disabled":
    planner.custom_long_output = make_custom_output(a_target=0.2, selected_intent="lead_follow", enabled=False)
  elif case == "output_fault":
    planner.custom_long_output = make_custom_output(a_target=0.2, selected_intent="lead_follow", fault_class="test_fault")
  elif case == "model_stop_corroborated":
    planner.custom_long_output = make_custom_output(
      a_target=0.2, selected_intent="lead_follow", model_stop_corroborated=True,
    )
  elif case == "stop_approach":
    planner.custom_long_output = make_custom_output(a_target=0.2, selected_intent="stop_approach")

  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=sm_kwargs.pop("v_ego", 2.5), lead_one=lead, **sm_kwargs),
    mpc_a_target=mpc_a_target, mpc_should_stop=mpc_should_stop,
    raw_model_a_target=raw_model_a_target, raw_model_should_stop=raw_model_should_stop,
  )

  assert a_target == pytest.approx(expected)
  assert fin.launch_floor_fade_pending is False


def test_launch_floor_carries_confirmed_departure_through_weak_mpc():
  # Route 000002b2 t=753: after release the policy intent flipped to lead_follow, the
  # verdict lapsed, and the command collapsed to raw mpcA 0.07-0.3 while the lead was
  # clearly going — both driver gas presses landed in that window.
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner, long_active=True)

  # Post-release frame: posture cleared, lead departing fast, MPC still ramping.
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.8, v_lead=1.2, v_rel=1.1, lead_id=1)
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead, long_active=True), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert should_stop is False
  # downstream dampers may shave a frame's worth of jerk off the floored value
  assert fin._STOP_HOLD_LAUNCH_FLOOR_A - 0.05 <= a_target <= fin._STOP_HOLD_LAUNCH_FLOOR_A + 1e-6


@pytest.mark.parametrize(
  "mpc_a_target, stop_posture",
  [(0.12, False), (-0.01, False), (0.12, True)],
)
def test_launch_floor_requires_positive_mpc_and_clear_stop_posture(mpc_a_target, stop_posture):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner, long_active=True)
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=stop_posture)
  if stop_posture:
    fin.stop_hold_release_sustain_s = 0.0
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.8, v_lead=1.2, v_rel=1.1, lead_id=1)

  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead, long_active=True), mpc_a_target=mpc_a_target, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )

  if stop_posture or mpc_a_target < 0.0:
    assert a_target == pytest.approx(mpc_a_target)
  else:
    assert a_target > mpc_a_target
    assert a_target >= fin._STOP_HOLD_LAUNCH_FLOOR_A - 0.05
  assert should_stop is stop_posture


def test_launch_floor_stays_off_below_breakout_and_without_grace():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_stop_hold", should_stop=True),
  )
  fin = _launch_release(planner, long_active=True)

  # Lead only crawling (below the breakout opening): the gentle authorities own it.
  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.6, v_lead=0.4, v_rel=0.4, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead, long_active=True), mpc_a_target=0.12, mpc_should_stop=False,
    raw_model_a_target=0.2, raw_model_should_stop=False,
  )
  assert a_target == pytest.approx(0.12)

  # No recent release (grace expired): raw mpcA passes through even for a departing lead.
  fin.launch_dip_grace_s = 0.0
  fin.final_a_prev = None
  lead = make_lead(d_rel=8.0, v_lead=1.5, v_rel=1.4, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.1, lead_one=lead, long_active=True), mpc_a_target=0.12, mpc_should_stop=False,
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
  fin = _launch_release(planner, long_active=True)

  planner.custom_long_output = make_custom_output(selected_intent="lead_follow", should_stop=False)
  lead = make_lead(d_rel=6.0, v_lead=1.0, v_rel=0.45, lead_id=1)  # chasing: vRel below entry
  # a few frames: the jerk-limited stages climb toward the floored value; the lead-catchup
  # cushion legitimately trims the 0.60 proposal at this tight gap (caps run after the
  # floor by design) — the point is the command holds well above the sagging 0.2 mpcA.
  for _ in range(4):
    a_target, _, _ = planner.final_longitudinal_output(
      make_sm(v_ego=0.55, lead_one=lead, long_active=True), mpc_a_target=0.2, mpc_should_stop=False,
      raw_model_a_target=0.3, raw_model_should_stop=False,
    )
  assert 0.4 < a_target <= fin._STOP_HOLD_LAUNCH_FLOOR_A + 1e-6

  # Ego has matched the lead's speed: floor off — the command drops back to the
  # MPC/cushion-owned value instead of being raised.
  fin.final_a_prev = None
  lead = make_lead(d_rel=6.5, v_lead=1.0, v_rel=0.05, lead_id=1)
  a_target, _, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.95, lead_one=lead, long_active=True), mpc_a_target=0.2, mpc_should_stop=False,
    raw_model_a_target=0.3, raw_model_should_stop=False,
  )
  assert a_target <= 0.2 + 1e-6


def _departing_lead(d_rel=8.0, v_lead=1.8, v_rel=0.8, a_lead_k=0.8, lead_id=1):
  lead = make_lead(d_rel=d_rel, v_lead=v_lead, v_rel=v_rel, lead_id=lead_id)
  lead.aLeadK = a_lead_k
  return lead


@pytest.mark.parametrize("mpc_a_target, expected", [(-1.0, 0.0), (-1.01, -1.01)])
def test_departing_lead_coast_uses_inclusive_depth_boundary(mpc_a_target, expected):
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  fin = planner.custom_long_finalizer
  for _ in range(fin._DEPARTING_LEAD_PERSIST_FRAMES + 3):
    a_target, should_stop, _ = planner.final_longitudinal_output(
      make_sm(v_ego=3.3, lead_one=_departing_lead()), mpc_a_target=mpc_a_target, mpc_should_stop=False,
      raw_model_a_target=mpc_a_target, raw_model_should_stop=False,
    )

  assert should_stop is False
  assert a_target == pytest.approx(expected)


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


def test_departing_lead_coast_never_reshapes_stop_posture():
  planner = make_planner(
    mode=LongitudinalMode.SCC,
    custom_long_output=make_custom_output(selected_intent="lead_follow", should_stop=False),
  )
  fin = planner.custom_long_finalizer

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


def _departure_evidence(*, track_id: int = 1, eligible: bool = True, effective_mode: str = "apply",
                        apply_supported: bool = True, research: bool = True, **overrides):
  fields: dict[str, Any] = dict(
    mode="apply", effective_mode=effective_mode, apply_supported=apply_supported,
    research_actuation_allowed=research, eligible=eligible,
    block_reason="" if eligible else "insufficient_predicted_growth",
    lead_idx=0, track_id=track_id, stable=True, radar=True,
    progress_authorized=True, prediction_valid=True,
    d_rel=6.5, v_lead=0.8, v_rel=0.5, a_lead_k=0.5,
    predicted_gap_1s=7.0, predicted_gap_growth_1s=0.5,
  )
  fields.update(overrides)
  return DeparturePredictionEvidence(**fields)


def _departure_phase_snapshot(planner, output, *, track_id: int = 1, dt: float = 0.05,
                              long_active: bool = True):
  return SimpleNamespace(
    custom_long=planner.custom_long,
    custom_long_output=output,
    long_active=long_active, model_stale=False,
    dt=dt,
    brake_pressed=False,
    gas_pressed=False,
    force_decel=False,
    raw_model_should_stop=False,
    selected_lead=make_lead(d_rel=6.5, v_lead=0.8, v_rel=0.5, lead_id=track_id),
    lead_id=track_id,
  )


def test_departure_prediction_persists_before_the_first_release_frame():
  planner = make_planner(custom_long_output=make_custom_output())
  fin = planner.custom_long_finalizer
  fin.CP.openpilotLongitudinalControl = True
  fin.lead_stop_hold_lead_id = 1
  output = make_custom_output(departure_prediction_evidence=_departure_evidence())
  snapshot = _departure_phase_snapshot(planner, output)

  for _ in range(3):
    fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_phase == PHASE_ARMING
  assert fin.departure_prediction_phase_s == pytest.approx(0.15)
  assert fin.departure_prediction_frame_start_ready is False

  fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_phase == PHASE_PREDICTED
  assert fin.departure_prediction_frame_start_ready is False
  not_ready_fin, not_ready_result = _run_departure_apply(frame_start_predicted=False)
  assert not_ready_result == pytest.approx(-0.1)
  assert not_ready_fin.departure_prediction_trace.block_reason == "prediction_not_persistent"

  fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_phase == PHASE_PREDICTED
  assert fin.departure_prediction_frame_start_ready is True
  assert fin.departure_prediction_trace.evidence_s == pytest.approx(PERSISTENCE_S)


def test_departure_prediction_timeout_locks_one_track_until_evidence_drops_or_track_changes():
  planner = make_planner(custom_long_output=make_custom_output())
  fin = planner.custom_long_finalizer
  fin.CP.openpilotLongitudinalControl = True
  fin.lead_stop_hold_lead_id = 1
  output = make_custom_output(departure_prediction_evidence=_departure_evidence())
  snapshot = _departure_phase_snapshot(planner, output)

  for _ in range(4 + int(TIMEOUT_S / 0.05)):
    fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_phase == PHASE_INACTIVE
  assert fin.departure_prediction_lockout_track_id == 1
  assert fin.departure_prediction_trace.block_reason == "prediction_timeout"

  fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_lockout_track_id == 1
  assert fin.departure_prediction_trace.block_reason == "prediction_timeout"

  dropped = make_custom_output(departure_prediction_evidence=_departure_evidence(eligible=False))
  fin._update_departure_prediction_phase(_departure_phase_snapshot(planner, dropped), pre_hold_active=True)
  assert fin.departure_prediction_lockout_track_id == -1
  assert fin.departure_prediction_phase == PHASE_INACTIVE

  fin.departure_prediction_lockout_track_id = 1
  fin.departure_prediction_track_id = 1
  fin.lead_stop_hold_lead_id = 2
  changed = make_custom_output(departure_prediction_evidence=_departure_evidence(track_id=2))
  fin._update_departure_prediction_phase(_departure_phase_snapshot(planner, changed, track_id=2), pre_hold_active=True)
  assert fin.departure_prediction_lockout_track_id == -1
  assert fin.departure_prediction_phase == PHASE_ARMING


def test_departure_prediction_timeout_lockout_survives_context_loss_and_public_hold_exit():
  planner = make_planner(custom_long_output=make_custom_output())
  fin = planner.custom_long_finalizer
  fin.CP.openpilotLongitudinalControl = True
  fin.lead_stop_hold_lead_id = 1
  output = make_custom_output(departure_prediction_evidence=_departure_evidence())
  snapshot = _departure_phase_snapshot(planner, output)

  for _ in range(4 + int(TIMEOUT_S / 0.05)):
    fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_lockout_track_id == 1

  # Losing the hold, a bad dt, and an inactive longitudinal context are not evidence drops.
  for lost_snapshot, pre_hold, reason in (
    (_departure_phase_snapshot(planner, output, long_active=True), False, "no_stop_hold"),
    (_departure_phase_snapshot(planner, output, dt=0.0), True, "invalid_dt"),
    (_departure_phase_snapshot(planner, output, long_active=False), True, "long_inactive"),
  ):
    fin._update_departure_prediction_phase(lost_snapshot, pre_hold_active=pre_hold)
    assert fin.departure_prediction_lockout_track_id == 1
    assert fin.departure_prediction_trace.block_reason == reason

  # The real public release path must not clear the same-track timeout lockout either.
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)
  planner.custom_long_output = make_custom_output(
    standstill_release_allowed=True,
    standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.25,
    departure_prediction_evidence=_departure_evidence(),
  )
  a_target, should_stop, _ = planner.final_longitudinal_output(
    make_sm(v_ego=0.0, lead_one=make_lead(d_rel=6.7, v_lead=0.8, v_rel=0.5), long_active=True),
    mpc_a_target=0.1, mpc_should_stop=False, raw_model_a_target=0.1, raw_model_should_stop=False,
  )
  assert a_target > 0.0
  assert should_stop is False
  assert fin.lead_stop_hold_active is False
  assert fin.departure_prediction_lockout_track_id == 1

  # Only an actual evidence drop permits a same-track rearm.
  dropped = make_custom_output(departure_prediction_evidence=_departure_evidence(eligible=False))
  fin._update_departure_prediction_phase(_departure_phase_snapshot(planner, dropped), pre_hold_active=True)
  assert fin.departure_prediction_lockout_track_id == -1
  fin.lead_stop_hold_lead_id = 1
  fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_phase == PHASE_ARMING


@pytest.mark.parametrize("reason", ["prediction_timeout", "insufficient_predicted_growth"])
def test_departure_prediction_context_preserves_existing_block_reason(reason):
  planner = make_planner(custom_long_output=make_custom_output())
  fin = planner.custom_long_finalizer
  fin.CP.openpilotLongitudinalControl = True
  fin.lead_stop_hold_lead_id = 1
  evidence = _departure_evidence(
    eligible=reason == "prediction_timeout",
    block_reason="" if reason == "prediction_timeout" else reason,
  )
  snapshot = _departure_phase_snapshot(
    planner, make_custom_output(departure_prediction_evidence=evidence),
  )
  if reason == "prediction_timeout":
    for _ in range(4 + int(TIMEOUT_S / 0.05)):
      fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  else:
    fin._update_departure_prediction_phase(snapshot, pre_hold_active=True)
  assert fin.departure_prediction_trace.block_reason == reason

  fin._record_departure_prediction_context(
    snapshot, -0.5, True, False, True, True, False, 1, "stop_hold_active",
  )

  assert fin.departure_prediction_trace.block_reason == reason


def _departure_apply_snapshot(*, mode=LongitudinalMode.SCC, track_id: int = 1, mpc_a_target: float = 0.1,
                              raw_model_a_target: float = 0.1, mpc_should_stop: bool = False,
                              raw_model_should_stop: bool = False, model_stale: bool = False,
                              brake_pressed: bool = False, gas_pressed: bool = False,
                              force_decel: bool = False, long_active: bool = True,
                              lead_d_rel: float = 6.5, lead_v: float = 0.8, lead_v_rel: float = 0.5,
                              evidence=None, output=None):
  lead = make_lead(d_rel=lead_d_rel, v_lead=lead_v, v_rel=lead_v_rel, lead_id=track_id)
  return SimpleNamespace(
    custom_long=SimpleNamespace(enabled=True, mode=mode, fault_class=""),
    custom_long_output=output or make_custom_output(
      standstill_release_allowed=True, standstill_release_source="lead_pullaway",
      standstill_release_a_target=0.25,
      departure_prediction_evidence=evidence or _departure_evidence(track_id=track_id),
    ),
    long_active=long_active, model_stale=model_stale, brake_pressed=brake_pressed,
    gas_pressed=gas_pressed, force_decel=force_decel,
    raw_model_should_stop=raw_model_should_stop, mpc_should_stop=mpc_should_stop,
    mpc_a_target_valid=math.isfinite(mpc_a_target),
    raw_model_a_target_valid=math.isfinite(raw_model_a_target),
    mpc_a_target=mpc_a_target, raw_model_a_target=raw_model_a_target,
    selected_lead=lead, lead_id=track_id, lead_d_rel=lead_d_rel, lead_v=lead_v, lead_v_rel=lead_v_rel,
    stopping_distance=6.0,
  )


def _run_departure_apply(*, target: float = -0.1, should_stop: bool = False, pre_slew_state=None, post_slew_state=None,
                         post_slew_target=None, snapshot=None, release_mpc_stop=True,
                         pre_hold_active=True, post_hold_active=False, frame_start_predicted=True,
                         openpilot_longitudinal_control: bool = True):
  fin = CustomLongitudinalFinalizer(make_cp())
  fin.CP.openpilotLongitudinalControl = openpilot_longitudinal_control
  fin.lead_stop_hold_lead_id = 1
  snapshot = snapshot or _departure_apply_snapshot()
  post_slew_state = target if post_slew_state is None else post_slew_state
  post_slew_target = target if post_slew_target is None else post_slew_target
  evidence = snapshot.custom_long_output.departure_prediction_evidence
  fin.departure_prediction_trace = fin._departure_prediction_trace_from_evidence(evidence)
  result = fin._apply_departure_prediction(
    snapshot, target, should_stop, release_mpc_stop, pre_hold_active, post_hold_active,
    pre_hold_lead_id=1, frame_start_predicted=frame_start_predicted,
    pre_slew_state=pre_slew_state, pre_slew_input=0.10,
    post_slew_state=post_slew_state, post_slew_target=post_slew_target,
  )
  return fin, result


def _run_public_departure_release(*, raw_model_a_target: float = 0.1, final_evidence=None):
  """Exercise the actual hold-exit, release-slew, and departure-prediction call chain."""
  planner = make_planner()
  fin = planner.custom_long_finalizer
  fin.CP.openpilotLongitudinalControl = True
  _arm_stop_hold(planner, d_rel=6.2, lead_id=1, gap_increasing_s=0.15)

  def finalize(output, lead):
    return fin.finalize(
      make_sm(v_ego=0.0, lead_one=lead, long_active=True), planner.custom_long, output,
      is_e2e=False, model_stale=False, dt=0.05,
      mpc_a_target=0.1, mpc_should_stop=False,
      raw_model_a_target=raw_model_a_target, raw_model_should_stop=False,
      apply_stop_hold_release_slew=lambda sm, target, release, mpc_stop, model_stop, should_stop:
        fin._apply_stop_hold_release_slew(sm, 0.05, target, release, mpc_stop, model_stop, should_stop),
      reset_lead_stop_hold=fin.reset_lead_stop_hold,
    )

  # The predictor is eligible before the lead has crossed the measured-departure gate. Keeping
  # the release permission off holds the car at the real -0.5 command while persistence ages.
  pre_output = make_custom_output(
    standstill_release_allowed=False,
    departure_prediction_evidence=_departure_evidence(v_lead=0.2, v_rel=0.05),
  )
  for _ in range(4):
    finalize(pre_output, make_lead(d_rel=6.5, v_lead=0.2, v_rel=0.05))

  output = make_custom_output(
    standstill_release_allowed=True,
    standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.25,
    departure_prediction_evidence=final_evidence or _departure_evidence(),
  )
  result = finalize(output, make_lead(d_rel=6.7, v_lead=0.8, v_rel=0.5))
  return fin, result


def test_departure_prediction_public_finalize_uses_real_first_slew_provenance():
  fin, result = _run_public_departure_release()
  trace = result.departure_prediction_trace

  assert result.a_target == pytest.approx(0.0)
  assert trace.release_mpc_stop is True
  assert trace.release_slew_provenance is True
  assert fin.stop_hold_release_slew_a_target == pytest.approx(-0.20)
  assert trace.a_target_before == pytest.approx(-0.20)
  assert trace.a_target_proposed == pytest.approx(0.0)
  assert trace.a_target_after == pytest.approx(0.0)
  assert trace.delta_a == pytest.approx(0.20)
  assert trace.measured_departure is True
  assert trace.applied is True


def test_departure_prediction_public_finalize_rejects_invalid_raw_target():
  fin, result = _run_public_departure_release(raw_model_a_target=float("nan"))
  trace = result.departure_prediction_trace

  assert result.a_target == pytest.approx(-0.20)
  assert trace.release_mpc_stop is True
  assert trace.release_slew_provenance is True
  assert trace.applied is False
  assert trace.block_reason == "invalid_raw_target"


def test_departure_prediction_shadow_reports_only_a_fully_qualifying_coast_candidate():
  fin, result = _run_departure_apply(
    snapshot=_departure_apply_snapshot(
      evidence=_departure_evidence(effective_mode="shadow", apply_supported=False, research=False),
    ),
  )
  _, off_result = _run_departure_apply(
    snapshot=_departure_apply_snapshot(evidence=DeparturePredictionEvidence()),
  )

  assert result == pytest.approx(off_result)
  assert fin.departure_prediction_trace.eligible is True
  assert fin.departure_prediction_trace.would_coast is True
  assert fin.departure_prediction_trace.applied is False
  assert fin.departure_prediction_trace.block_reason == "non_actuating_mode"


@pytest.mark.parametrize(
  ("target", "expected", "applied"),
  [(-0.20, 0.0, True), (-0.01, 0.0, True), (-0.21, -0.21, False), (0.0, 0.0, False), (0.2, 0.2, False)],
)
def test_departure_prediction_apply_only_replaces_the_shallow_negative_band(target, expected, applied):
  fin, result = _run_departure_apply(target=target)
  assert result == pytest.approx(expected)
  assert fin.departure_prediction_trace.applied is applied
  assert fin.departure_prediction_trace.a_target_before == pytest.approx(target)


def test_departure_prediction_never_overrides_stop_posture_or_positive_accel():
  fin, result = _run_departure_apply(target=-0.1, should_stop=True)
  assert result == pytest.approx(-0.1)
  assert fin.departure_prediction_trace.applied is False
  assert fin.departure_prediction_trace.block_reason == "should_stop"

  fin, result = _run_departure_apply(target=0.1)
  assert result == pytest.approx(0.1)
  assert fin.departure_prediction_trace.applied is False


@pytest.mark.parametrize(
  "case",
  [
    "negative_mpc", "nonfinite_mpc", "negative_model", "nonfinite_model", "stale_model",
    "mpc_stop", "model_stop", "brake", "gas", "force_decel", "alternate_threat",
    "wrong_track", "acc", "e2e", "not_first_release",
  ],
)
def test_departure_prediction_blockers_leave_the_baseline_target_unchanged(case):
  kwargs: dict[str, Any] = {}
  if case == "negative_mpc":
    kwargs["mpc_a_target"] = -0.1
  elif case == "nonfinite_mpc":
    kwargs["mpc_a_target"] = float("nan")
  elif case == "negative_model":
    kwargs["raw_model_a_target"] = -0.1
  elif case == "nonfinite_model":
    kwargs["raw_model_a_target"] = float("nan")
  elif case == "stale_model":
    kwargs["model_stale"] = True
  elif case == "mpc_stop":
    kwargs["mpc_should_stop"] = True
  elif case == "model_stop":
    kwargs["raw_model_should_stop"] = True
  elif case == "brake":
    kwargs["brake_pressed"] = True
  elif case == "gas":
    kwargs["gas_pressed"] = True
  elif case == "force_decel":
    kwargs["force_decel"] = True
  elif case == "alternate_threat":
    kwargs["output"] = make_custom_output(
      standstill_release_allowed=True, standstill_release_source="lead_pullaway",
      departure_prediction_evidence=_departure_evidence(alternate_threat_active=True),
    )
  elif case == "wrong_track":
    kwargs["track_id"] = 2
  elif case == "acc":
    kwargs["mode"] = LongitudinalMode.ACC
  elif case == "e2e":
    kwargs["mode"] = LongitudinalMode.E2E
  elif case == "not_first_release":
    kwargs["release_mpc_stop"] = False
  release_mpc_stop = kwargs.pop("release_mpc_stop", True)
  snapshot = _departure_apply_snapshot(**kwargs)
  fin, result = _run_departure_apply(snapshot=snapshot, release_mpc_stop=release_mpc_stop)
  assert result == pytest.approx(-0.1)
  assert fin.departure_prediction_trace.applied is False


@pytest.mark.parametrize(
  ("case", "expected_reason"),
  [
    ("long_inactive", "long_inactive"),
    ("openpilot_longitudinal_disabled", "openpilot_longitudinal_disabled"),
    ("effective_shadow", "non_actuating_mode"),
    ("research_disabled", "research_actuation_gate"),
    ("no_release_permission", "no_release_permission"),
    ("invalid_release_source", "invalid_release_source"),
    ("measured_departure", "measured_departure_not_confirmed"),
  ],
)
def test_departure_prediction_apply_gates_block_with_specific_reason(case, expected_reason):
  kwargs: dict[str, Any] = {}
  if case == "long_inactive":
    kwargs["long_active"] = False
  elif case == "openpilot_longitudinal_disabled":
    kwargs["openpilot_longitudinal_control"] = False
  elif case == "effective_shadow":
    kwargs["evidence"] = _departure_evidence(effective_mode="shadow", apply_supported=False, research=False)
  elif case == "research_disabled":
    kwargs["evidence"] = _departure_evidence(research=False)
  elif case == "no_release_permission":
    kwargs["output"] = make_custom_output(
      standstill_release_allowed=False,
      standstill_release_source="lead_pullaway",
      departure_prediction_evidence=_departure_evidence(),
    )
  elif case == "invalid_release_source":
    kwargs["output"] = make_custom_output(
      standstill_release_allowed=True,
      standstill_release_source="unknown",
      departure_prediction_evidence=_departure_evidence(),
    )
  elif case == "measured_departure":
    kwargs.update(lead_v=0.2, lead_v_rel=0.05)

  openpilot_longitudinal_control = kwargs.pop("openpilot_longitudinal_control", True)
  fin, result = _run_departure_apply(
    snapshot=_departure_apply_snapshot(**kwargs),
    openpilot_longitudinal_control=openpilot_longitudinal_control,
  )
  assert result == pytest.approx(-0.1)
  assert fin.departure_prediction_trace.applied is False
  assert fin.departure_prediction_trace.block_reason == expected_reason


@pytest.mark.parametrize("static_release", [False, True])
def test_departure_prediction_is_inert_for_crawl_and_static_release(static_release):
  def run(evidence=None):
    output_kwargs: dict[str, Any] = dict(selected_intent="lead_stop_hold", should_stop=True)
    if evidence is not None:
      output_kwargs["departure_prediction_evidence"] = evidence
    planner = make_planner(custom_long_output=make_custom_output(**output_kwargs))
    planner.CP.openpilotLongitudinalControl = True
    planner.custom_long_finalizer.CP = planner.CP
    fin = planner.custom_long_finalizer
    _arm_stop_hold(planner, d_rel=7.0 if static_release else 6.4, lead_id=1)
    lead = make_lead(
      d_rel=7.0 if static_release else 8.1,
      v_lead=0.0 if static_release else 0.5,
      v_rel=0.0 if static_release else 0.5,
      lead_id=1,
    )
    if static_release:
      planner._lead_stop_hold_gap_baseline_d_rel = 5.0
      for _ in range(fin._STOP_HOLD_MPC_GO_PERSIST_FRAMES - 1):
        planner.final_longitudinal_output(
          make_sm(v_ego=0.0, lead_one=lead, long_active=True),
          mpc_a_target=0.65, mpc_should_stop=False,
          raw_model_a_target=0.05, raw_model_should_stop=True,
        )
      fin.final_a_prev = None
    result = planner.final_longitudinal_output(
      make_sm(v_ego=0.0, lead_one=lead, long_active=True),
      mpc_a_target=0.65 if static_release else 0.68,
      mpc_should_stop=False, raw_model_a_target=0.05, raw_model_should_stop=True,
    )
    return result, fin.departure_prediction_trace

  baseline, _ = run()
  observed, trace = run(_departure_evidence())
  assert observed == pytest.approx(baseline)
  assert trace.applied is False


def test_blocked_departure_trace_retains_release_context():
  snapshot = _departure_apply_snapshot(mpc_a_target=-0.1)
  fin, result = _run_departure_apply(snapshot=snapshot)
  trace = fin.departure_prediction_trace
  assert result == pytest.approx(-0.1)
  assert trace.track_id == 1
  assert trace.predicted_gap_delta == pytest.approx(0.5)
  assert trace.release_source == "lead_pullaway"
  assert trace.release_permission is True
  assert trace.same_track is True
  assert trace.eligible is False
  assert trace.would_coast is False
  assert trace.a_target_before == pytest.approx(-0.1)
  assert trace.block_reason == "mpc_brake_or_stop"
