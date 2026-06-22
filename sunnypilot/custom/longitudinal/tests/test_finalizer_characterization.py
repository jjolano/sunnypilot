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

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


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


def make_lead(d_rel: float, v_lead: float, v_rel: float, lead_id: int = 1, status: bool = True):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    vLead=v_lead,
    vRel=v_rel,
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
      debug={
        "curve_speed_confidence_eligible": True,
        "curve_speed_confidence_apply_supported": True,
        "curve_speed_confidence_confidence": 0.80,
        "curve_speed_confidence_proposed_cap": -1.0,
      },
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
