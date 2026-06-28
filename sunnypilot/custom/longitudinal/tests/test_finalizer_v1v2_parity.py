"""V1/V2 parity characterization for the longitudinal finalizer.

The v2 finalizer is a staged structural rewrite of the v1 implementation. These tests
feed identical frame sequences to both and assert that public results and internal
state evolve identically.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

from openpilot.sunnypilot.custom.longitudinal import finalizer as f_v2
from openpilot.sunnypilot.custom.longitudinal import finalizer_v1 as f_v1
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput


class FakeSM:
  def __init__(self, **services):
    self._services = services

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


def make_lead(d_rel: float, v_lead: float, v_rel: float, lead_id: int = 1, status: bool = True):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    vLead=v_lead,
    vRel=v_rel,
    radarTrackId=lead_id,
  )


def make_sm(*, v_ego: float = 0.0, standstill: bool = False, brake_pressed: bool = False,
            gas_pressed: bool = False, force_decel: bool = False,
            lead_one=None, lead_two=None) -> FakeSM:
  return FakeSM(
    carState=SimpleNamespace(
      vEgo=v_ego,
      standstill=standstill,
      brakePressed=brake_pressed,
      gasPressed=gas_pressed,
    ),
    controlsState=SimpleNamespace(forceDecel=force_decel),
    radarState=SimpleNamespace(leadOne=lead_one, leadTwo=lead_two),
  )


def _float_eq(a, b, *, rel=1e-9, abs_tol=1e-12):
  if a is None or b is None:
    return a is b
  if isinstance(a, float) or isinstance(b, float):
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
      return True
    if math.isinf(a) and math.isinf(b):
      return (a > 0) == (b > 0)
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)
  return a == b


def _assert_value_eq(a: Any, b: Any, path: str = "root") -> None:
  if a is b:
    return
  if a is None or b is None:
    assert a is b, f"{path}: {a!r} != {b!r}"
    return
  if isinstance(a, (str, bool, int)) and type(a) is type(b):
    assert a == b, f"{path}: {a!r} != {b!r}"
    return
  if isinstance(a, float) or isinstance(b, float):
    assert _float_eq(a, b), f"{path}: {a!r} != {b!r}"
    return
  if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
    a_fields = {f.name for f in dataclasses.fields(a)}
    b_fields = {f.name for f in dataclasses.fields(b)}
    assert a_fields == b_fields, f"{path}: dataclass field mismatch {a_fields} vs {b_fields}"
    for field_name in sorted(a_fields):
      _assert_value_eq(getattr(a, field_name), getattr(b, field_name), f"{path}.{field_name}")
    return
  assert a == b, f"{path}: {a!r} != {b!r}"


def _state_attrs() -> list[str]:
  return [
    "lead_stop_hold_active",
    "lead_stop_hold_gap_increasing_s",
    "lead_stop_hold_missing_s",
    "lead_stop_hold_lead_id",
    "lead_stop_hold_gap_prev_d_rel",
    "lead_stop_hold_gap_baseline_d_rel",
    "custom_long_output_telemetry",
    "last_release_block_reason",
    "stop_hold_release_slew_a_target",
    "stop_hold_release_prep_a_target",
    "stop_hold_release_prep_raw_prev",
  ]


def _assert_state_eq(f1, f2):
  for attr in _state_attrs():
    _assert_value_eq(getattr(f1, attr), getattr(f2, attr), f"state.{attr}")


def _make_callbacks(finalizer, dt):
  def reset_lead_stop_hold():
    finalizer.reset_lead_stop_hold()

  def apply_stop_hold_release_slew(sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop):
    return finalizer._apply_stop_hold_release_slew(
      sm, dt, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop
    )

  return apply_stop_hold_release_slew, reset_lead_stop_hold


def _run_parity(frames: list[dict[str, Any]], *, cp: Any = None) -> tuple[f_v1.CustomLongitudinalFinalizer, f_v2.CustomLongitudinalFinalizer]:
  cp = cp if cp is not None else make_cp()
  f1 = f_v1.CustomLongitudinalFinalizer(cp)
  f2 = f_v2.CustomLongitudinalFinalizer(cp)

  for frame in frames:
    dt = frame.get("dt", 0.05)
    apply_slew_1, reset_1 = _make_callbacks(f1, dt)
    apply_slew_2, reset_2 = _make_callbacks(f2, dt)

    result1 = f1.finalize(
      sm=frame["sm"],
      custom_long=frame["custom_long"],
      custom_long_output=frame["custom_long_output"],
      is_e2e=frame.get("is_e2e", False),
      model_stale=frame.get("model_stale", False),
      dt=dt,
      mpc_a_target=frame["mpc_a_target"],
      mpc_should_stop=frame["mpc_should_stop"],
      raw_model_a_target=frame["raw_model_a_target"],
      raw_model_should_stop=frame["raw_model_should_stop"],
      apply_stop_hold_release_slew=apply_slew_1,
      reset_lead_stop_hold=reset_1,
    )
    result2 = f2.finalize(
      sm=frame["sm"],
      custom_long=frame["custom_long"],
      custom_long_output=frame["custom_long_output"],
      is_e2e=frame.get("is_e2e", False),
      model_stale=frame.get("model_stale", False),
      dt=dt,
      mpc_a_target=frame["mpc_a_target"],
      mpc_should_stop=frame["mpc_should_stop"],
      raw_model_a_target=frame["raw_model_a_target"],
      raw_model_should_stop=frame["raw_model_should_stop"],
      apply_stop_hold_release_slew=apply_slew_2,
      reset_lead_stop_hold=reset_2,
    )

    _assert_value_eq(result2.a_target, result1.a_target, "result.a_target")
    _assert_value_eq(result2.should_stop, result1.should_stop, "result.should_stop")
    _assert_value_eq(result2.e2e_source, result1.e2e_source, "result.e2e_source")
    _assert_value_eq(result2.custom_long_output_telemetry, result1.custom_long_output_telemetry, "result.custom_long_output_telemetry")
    _assert_value_eq(result2.last_release_block_reason, result1.last_release_block_reason, "result.last_release_block_reason")
    _assert_state_eq(f1, f2)

  return f1, f2


def _frame(custom_long, custom_long_output,
           sm, mpc_a_target: float = 0.0, mpc_should_stop: bool = False,
           raw_model_a_target: float = 0.0, raw_model_should_stop: bool = False,
           is_e2e: bool = False, model_stale: bool = False, dt: float = 0.05) -> dict[str, Any]:
  return {
    "sm": sm,
    "custom_long": custom_long,
    "custom_long_output": custom_long_output,
    "is_e2e": is_e2e,
    "model_stale": model_stale,
    "dt": dt,
    "mpc_a_target": mpc_a_target,
    "mpc_should_stop": mpc_should_stop,
    "raw_model_a_target": raw_model_a_target,
    "raw_model_should_stop": raw_model_should_stop,
  }


def test_v1_backup_api_matches():
  for name in dir(f_v1):
    if not name.startswith("_"):
      assert hasattr(f_v2, name), f"v2 missing public symbol {name}"
  assert hasattr(f_v1, "CustomLongitudinalFinalizer")
  assert hasattr(f_v2, "CustomLongitudinalFinalizer")
  assert f_v1.FinalizerResult is not f_v2.FinalizerResult


def test_parity_custom_disabled():
  frames = [
    _frame(
      custom_long=make_custom_long(enabled=False),
      custom_long_output=None,
      sm=make_sm(v_ego=15.0),
      mpc_a_target=-0.2,
      mpc_should_stop=False,
      raw_model_a_target=-1.0,
      raw_model_should_stop=False,
      is_e2e=False,
    ),
  ]
  _run_parity(frames)


def test_parity_e2e_fresh_model():
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.E2E),
      custom_long_output=make_custom_output(mode=LongitudinalMode.E2E),
      sm=make_sm(v_ego=15.0),
      mpc_a_target=-0.2,
      mpc_should_stop=False,
      raw_model_a_target=-0.8,
      raw_model_should_stop=False,
      is_e2e=True,
      model_stale=False,
    ),
  ]
  _run_parity(frames)


def test_parity_e2e_stale_model():
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.E2E),
      custom_long_output=make_custom_output(mode=LongitudinalMode.E2E),
      sm=make_sm(v_ego=15.0),
      mpc_a_target=-0.2,
      mpc_should_stop=False,
      raw_model_a_target=-0.8,
      raw_model_should_stop=True,
      is_e2e=True,
      model_stale=True,
    ),
  ]
  _run_parity(frames)


def test_parity_acc_mode():
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.ACC),
      custom_long_output=make_custom_output(mode=LongitudinalMode.ACC),
      sm=make_sm(v_ego=15.0),
      mpc_a_target=-0.2,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=True,
      is_e2e=False,
    ),
  ]
  _run_parity(frames)


def test_parity_stop_hold_latch_arm_and_hold():
  lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=sm,
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  f1, f2 = _run_parity(frames)
  assert f2.lead_stop_hold_active is True


def test_parity_stop_hold_release_allowed():
  # Pre-latch with a stopped lead, then present a clearly opening lead for enough
  # gap-increasing time to satisfy the release gate.
  hold_lead = make_lead(d_rel=7.0, v_lead=0.0, v_rel=0.0)
  opening_frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=hold_lead),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  for d_rel in (7.0, 7.3, 7.6, 7.9, 8.2):
    lead = make_lead(d_rel=d_rel, v_lead=2.0, v_rel=1.0)
    opening_frames.append(_frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(
        standstill_release_allowed=True,
        standstill_release_source="lead_pullaway",
        standstill_release_a_target=0.25,
      ),
      sm=make_sm(v_ego=0.0, lead_one=lead),
      mpc_a_target=-0.05,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ))
  f1, f2 = _run_parity(opening_frames)
  assert f2.lead_stop_hold_active is False
  # Parity helper already compared every frame's result and state, including the
  # release frame where the slew state seeded.


def test_parity_blocked_release_reason():
  lead = make_lead(d_rel=8.0, v_lead=0.0, v_rel=0.0)
  sm = make_sm(v_ego=0.0, lead_one=lead)
  hold_lead = make_lead(d_rel=7.0, v_lead=0.0, v_rel=0.0)
  hold_sm = make_sm(v_ego=0.0, lead_one=hold_lead)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=hold_sm,
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(
        standstill_release_allowed=True,
        standstill_release_source="lead_pullaway",
        standstill_release_a_target=0.25,
      ),
      sm=sm,
      mpc_a_target=-0.05,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  _run_parity(frames)


def test_parity_crawl_fallback():
  hold_lead = make_lead(d_rel=6.2, v_lead=0.0, v_rel=0.0)
  open_lead = make_lead(d_rel=7.0, v_lead=0.4, v_rel=0.2)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=hold_lead),
      mpc_a_target=-0.3,
      mpc_should_stop=True,
      raw_model_a_target=0.1,
      raw_model_should_stop=False,
    ),
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=open_lead),
      mpc_a_target=0.0,
      mpc_should_stop=True,
      raw_model_a_target=0.1,
      raw_model_should_stop=False,
    ),
  ]
  _run_parity(frames)


def test_parity_release_slew():
  # Release then acceleration request must be bounded by slew.
  hold_lead = make_lead(d_rel=7.0, v_lead=0.0, v_rel=0.0)
  open_lead = make_lead(d_rel=8.0, v_lead=2.0, v_rel=1.0)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=hold_lead),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(
        standstill_release_allowed=True,
        standstill_release_source="lead_pullaway",
        standstill_release_a_target=0.25,
      ),
      sm=make_sm(v_ego=0.0, lead_one=open_lead),
      mpc_a_target=-0.05,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(
        standstill_release_allowed=True,
        standstill_release_source="lead_pullaway",
        standstill_release_a_target=0.35,
      ),
      sm=make_sm(v_ego=0.3, lead_one=open_lead),
      mpc_a_target=0.35,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  _run_parity(frames)


def test_parity_curve_confidence_cap():
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC, curve_speed_confidence_mode="apply_conservative"),
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
      sm=make_sm(v_ego=12.0),
      mpc_a_target=0.0,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  f1, f2 = _run_parity(frames)
  assert f2.last_release_block_reason == f1.last_release_block_reason


def test_parity_prep_slew():
  # Same lead pulling away gently while still in hold.
  lead = make_lead(d_rel=6.4, v_lead=0.5, v_rel=0.2)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(
        standstill_release_allowed=True,
        standstill_release_source="lead_pullaway",
      ),
      sm=make_sm(v_ego=0.0, lead_one=lead),
      mpc_a_target=-0.05,
      mpc_should_stop=False,
      raw_model_a_target=0.1,
      raw_model_should_stop=False,
    ),
  ]
  # Pre-latch so hold branch runs.
  hold_lead = make_lead(d_rel=6.2, v_lead=0.0, v_rel=0.0)
  _run_parity([
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=hold_lead),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
    *frames,
  ])


def test_parity_dropout_resets_latch():
  hold_lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  no_lead_sm = make_sm(v_ego=0.0, lead_one=None)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=hold_lead),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  # Exceed the 0.5 s lead-missing grace window with the latch still below vEgoStopping.
  for _ in range(12):
    frames.append(_frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=no_lead_sm,
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
      dt=0.05,
    ))
  f1, f2 = _run_parity(frames)
  assert f2.lead_stop_hold_active is False


def test_parity_nonfinite_fail_closed():
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=15.0),
      mpc_a_target=float('nan'),
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  _run_parity(frames)


def test_parity_transfer_to_new_stopped_lead():
  lead1 = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0, lead_id=1)
  lead2 = make_lead(d_rel=5.5, v_lead=0.0, v_rel=0.0, lead_id=2)
  frames = [
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=lead1),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
    _frame(
      custom_long=make_custom_long(mode=LongitudinalMode.SCC),
      custom_long_output=make_custom_output(selected_intent="cruise"),
      sm=make_sm(v_ego=0.0, lead_one=lead2),
      mpc_a_target=-0.3,
      mpc_should_stop=False,
      raw_model_a_target=0.0,
      raw_model_should_stop=False,
    ),
  ]
  f1, f2 = _run_parity(frames)
  assert f2.lead_stop_hold_active is True
  assert f2.lead_stop_hold_lead_id == 2
