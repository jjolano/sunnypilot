"""V1/V2 parity for the longitudinal planner wrapper cleanup.

The active planner replaces the verbose finalizer-state property block with compact
``_ProxyToFinalizer`` descriptors. These tests verify the external behavior and test
monkeypatch seams remain identical to the v1 backup.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


# Load the v1 backup as an independent module so both class versions coexist.
_V1_PATH = Path(__file__).resolve().parents[3] / "selfdrive/controls/lib/longitudinal_planner_v1.py"
_v1_spec = importlib.util.spec_from_file_location("longitudinal_planner_v1", _V1_PATH)
assert _v1_spec is not None and _v1_spec.loader is not None
_v1_module = importlib.util.module_from_spec(_v1_spec)
_v1_spec.loader.exec_module(_v1_module)
LongitudinalPlannerSP_V1 = _v1_module.LongitudinalPlannerSP


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
    openpilotLongitudinalControl=True,
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
            gas_pressed: bool = False, force_decel: bool = False, experimental_mode: bool = False,
            lead_one=None, lead_two=None, model_fresh: bool = True) -> FakeSM:
  import time
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
      vCruise=123.0,
    ),
    controlsState=SimpleNamespace(forceDecel=force_decel),
    selfdriveState=SimpleNamespace(experimentalMode=experimental_mode),
    radarState=SimpleNamespace(leadOne=lead_one, leadTwo=lead_two),
    carControl=SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
    modelV2=SimpleNamespace(),
  )


def _planner_cls_v1():
  return LongitudinalPlannerSP_V1


def _planner_cls_v2():
  return LongitudinalPlannerSP


def make_planner(cls):
  planner: Any = object.__new__(cls)
  planner.CP = make_cp()
  planner.dt = 0.05
  planner.custom_long_finalizer = CustomLongitudinalFinalizer(planner.CP)
  planner.custom_long = make_custom_long(mode=LongitudinalMode.SCC)
  planner.custom_long_output = make_custom_output()
  planner.events_sp = SimpleNamespace(to_msg=list)
  planner.source = 0
  planner.output_v_target = 0.0
  planner.output_a_target = 0.0
  planner.dec = SimpleNamespace(mode=lambda: "acc", enabled=lambda: False, active=lambda: False)
  planner.scc = SimpleNamespace(
    vision=SimpleNamespace(state=0, output_v_target=0.0, output_a_target=0.0, current_lat_acc=0.0,
                           max_pred_lat_acc=0.0, is_enabled=False, is_active=False),
    map=SimpleNamespace(state=0, output_v_target=0.0, output_a_target=0.0, is_enabled=False, is_active=False),
  )
  planner.resolver = SimpleNamespace(speed_limit=0.0, speed_limit_last=0.0, speed_limit_final=0.0,
                                     speed_limit_final_last=0.0, speed_limit_valid=False,
                                     speed_limit_last_valid=False, speed_limit_offset=0.0,
                                     distance=0.0, source=0)
  planner.sla = SimpleNamespace(state=0, is_enabled=False, is_active=False, output_v_target=0.0,
                                output_a_target=0.0)
  planner.e2e_alerts_helper = SimpleNamespace(green_light_alert=False, lead_depart_alert=False)
  planner.lead_anticipation = SimpleNamespace()
  # Initialize the proxy attrs through their setters.
  planner._lead_stop_hold_active = False
  planner._lead_stop_hold_gap_increasing_s = 0.0
  planner._lead_stop_hold_missing_s = 0.0
  planner._lead_stop_hold_lead_id = None
  planner._lead_stop_hold_gap_prev_d_rel = None
  planner._lead_stop_hold_gap_baseline_d_rel = None
  planner._custom_long_output_telemetry = None
  planner._last_release_block_reason = ""
  planner._stop_hold_release_slew_a_target = None
  planner._stop_hold_release_prep_a_target = None
  planner._stop_hold_release_prep_raw_prev = None
  return planner


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


def _proxy_attrs() -> list[str]:
  return [
    "_lead_stop_hold_active",
    "_lead_stop_hold_gap_increasing_s",
    "_lead_stop_hold_missing_s",
    "_lead_stop_hold_lead_id",
    "_lead_stop_hold_gap_prev_d_rel",
    "_lead_stop_hold_gap_baseline_d_rel",
    "_custom_long_output_telemetry",
    "_last_release_block_reason",
    "_stop_hold_release_slew_a_target",
    "_stop_hold_release_prep_a_target",
    "_stop_hold_release_prep_raw_prev",
  ]


def test_v1_backup_api_matches():
  for name in dir(LongitudinalPlannerSP_V1):
    if not name.startswith("__"):
      assert hasattr(LongitudinalPlannerSP, name), f"v2 missing public symbol {name}"
  assert hasattr(LongitudinalPlannerSP_V1, "_lead_stop_hold_active")
  assert hasattr(LongitudinalPlannerSP, "_lead_stop_hold_active")
  assert LongitudinalPlannerSP_V1 is not LongitudinalPlannerSP


def test_proxy_attrs_read_write_through_to_finalizer():
  p = make_planner(LongitudinalPlannerSP)
  p._lead_stop_hold_active = True
  p._lead_stop_hold_gap_increasing_s = 0.75
  p._lead_stop_hold_missing_s = 0.25
  p._lead_stop_hold_lead_id = 42
  p._lead_stop_hold_gap_prev_d_rel = 5.5
  p._lead_stop_hold_gap_baseline_d_rel = 5.0
  p._last_release_block_reason = "test_reason"
  p._stop_hold_release_slew_a_target = 0.1
  p._stop_hold_release_prep_a_target = -0.2
  p._stop_hold_release_prep_raw_prev = -0.3

  assert p.custom_long_finalizer.lead_stop_hold_active is True
  assert p.custom_long_finalizer.lead_stop_hold_gap_increasing_s == pytest.approx(0.75)
  assert p.custom_long_finalizer.lead_stop_hold_missing_s == pytest.approx(0.25)
  assert p.custom_long_finalizer.lead_stop_hold_lead_id == 42
  assert p.custom_long_finalizer.lead_stop_hold_gap_prev_d_rel == pytest.approx(5.5)
  assert p.custom_long_finalizer.lead_stop_hold_gap_baseline_d_rel == pytest.approx(5.0)
  assert p.custom_long_finalizer.last_release_block_reason == "test_reason"
  assert p.custom_long_finalizer.stop_hold_release_slew_a_target == pytest.approx(0.1)
  assert p.custom_long_finalizer.stop_hold_release_prep_a_target == pytest.approx(-0.2)
  assert p.custom_long_finalizer.stop_hold_release_prep_raw_prev == pytest.approx(-0.3)

  # Read-back through the proxy.
  assert p._lead_stop_hold_active is True
  assert p._last_release_block_reason == "test_reason"


def test_proxy_attrs_parity_with_v1():
  p1 = make_planner(LongitudinalPlannerSP_V1)
  p2 = make_planner(LongitudinalPlannerSP)
  for attr in _proxy_attrs():
    assert type(getattr(p2, attr)) is type(getattr(p1, attr))
  p1._lead_stop_hold_active = True
  p2._lead_stop_hold_active = True
  _assert_value_eq(p1._lead_stop_hold_active, p2._lead_stop_hold_active, "active")


def _run_output_parity(frames: list[dict[str, Any]], *, custom_long_outputs: list[Any] | None = None):
  p1 = make_planner(LongitudinalPlannerSP_V1)
  p2 = make_planner(LongitudinalPlannerSP)
  for i, frame in enumerate(frames):
    if custom_long_outputs is not None:
      p1.custom_long_output = custom_long_outputs[i]
      p2.custom_long_output = custom_long_outputs[i]
    out1 = p1.final_longitudinal_output(**frame)
    out2 = p2.final_longitudinal_output(**frame)
    _assert_value_eq(out2, out1, f"frame[{i}].output")
    for attr in _proxy_attrs():
      _assert_value_eq(getattr(p2, attr), getattr(p1, attr), f"frame[{i}].{attr}")
  return p1, p2


def test_parity_custom_disabled():
  sm = make_sm(v_ego=15.0)
  frames = [{
    "sm": sm,
    "mpc_a_target": -0.2,
    "mpc_should_stop": False,
    "raw_model_a_target": -1.0,
    "raw_model_should_stop": False,
  }]
  custom_outputs = [make_custom_output(enabled=False)]
  _run_output_parity(frames, custom_long_outputs=custom_outputs)


def test_parity_scc_hold_and_release():
  hold_lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  frames = [
    {"sm": make_sm(v_ego=0.0, lead_one=hold_lead), "mpc_a_target": -0.3, "mpc_should_stop": False,
     "raw_model_a_target": 0.0, "raw_model_should_stop": False},
  ]
  custom_outputs = [make_custom_output(selected_intent="cruise")]
  for d_rel in (5.0, 5.3, 5.6, 5.9, 6.2):
    opening_lead = make_lead(d_rel=d_rel, v_lead=2.0, v_rel=1.0)
    frames.append({"sm": make_sm(v_ego=0.0, lead_one=opening_lead), "mpc_a_target": -0.05,
                   "mpc_should_stop": False, "raw_model_a_target": 0.0, "raw_model_should_stop": False})
    custom_outputs.append(make_custom_output(standstill_release_allowed=True,
                                             standstill_release_source="lead_pullaway",
                                             standstill_release_a_target=0.25))
  p1, p2 = _run_output_parity(frames, custom_long_outputs=custom_outputs)
  assert p2._lead_stop_hold_active is False
  assert p2._last_release_block_reason == p1._last_release_block_reason


def test_parity_acc_mode():
  sm = make_sm(v_ego=15.0)
  frames = [{
    "sm": sm,
    "mpc_a_target": -0.2,
    "mpc_should_stop": False,
    "raw_model_a_target": 0.0,
    "raw_model_should_stop": True,
  }]
  custom_outputs = [make_custom_output(mode=LongitudinalMode.ACC)]
  _run_output_parity(frames, custom_long_outputs=custom_outputs)


def test_parity_e2e_fresh():
  sm = make_sm(v_ego=15.0)
  frames = [{
    "sm": sm,
    "mpc_a_target": -0.2,
    "mpc_should_stop": False,
    "raw_model_a_target": -0.8,
    "raw_model_should_stop": False,
  }]
  custom_outputs = [make_custom_output(mode=LongitudinalMode.E2E)]
  p1, p2 = _run_output_parity(frames, custom_long_outputs=custom_outputs)
  assert p2._lead_stop_hold_active is False


def test_parity_e2e_stale():
  sm = make_sm(v_ego=15.0, model_fresh=False)
  frames = [{
    "sm": sm,
    "mpc_a_target": -0.2,
    "mpc_should_stop": False,
    "raw_model_a_target": -0.8,
    "raw_model_should_stop": True,
  }]
  custom_outputs = [make_custom_output(mode=LongitudinalMode.E2E)]
  _run_output_parity(frames, custom_long_outputs=custom_outputs)


def test_parity_telemetry_and_block_reason_reset():
  hold_lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  frames = [{
    "sm": make_sm(v_ego=0.0, lead_one=hold_lead),
    "mpc_a_target": -0.3,
    "mpc_should_stop": False,
    "raw_model_a_target": 0.0,
    "raw_model_should_stop": False,
  }]
  custom_outputs = [make_custom_output(selected_intent="cruise")]
  p1, p2 = _run_output_parity(frames, custom_long_outputs=custom_outputs)
  _assert_value_eq(p1._custom_long_output_telemetry, p2._custom_long_output_telemetry, "telemetry")
  assert p2._custom_long_output_telemetry is not None
  assert p2._custom_long_output_telemetry.should_stop is True
  assert p2._custom_long_output_telemetry.selected_intent == "lead_stop_hold"


def test_parity_monkeypatch_reset_seam():
  p1 = make_planner(LongitudinalPlannerSP_V1)
  p2 = make_planner(LongitudinalPlannerSP)
  hold_lead = make_lead(d_rel=5.0, v_lead=0.0, v_rel=0.0)
  p1.custom_long_output = make_custom_output(selected_intent="cruise")
  p2.custom_long_output = make_custom_output(selected_intent="cruise")

  # Arm the latch first.
  arm_frame = {
    "sm": make_sm(v_ego=0.0, lead_one=hold_lead),
    "mpc_a_target": -0.3,
    "mpc_should_stop": False,
    "raw_model_a_target": 0.0,
    "raw_model_should_stop": False,
  }
  p1.final_longitudinal_output(**arm_frame)
  p2.final_longitudinal_output(**arm_frame)
  assert p1._lead_stop_hold_active is True
  assert p2._lead_stop_hold_active is True

  reset_log_1 = []
  reset_log_2 = []

  def reset_hook_1():
    reset_log_1.append("called")
    return p1.custom_long_finalizer.reset_lead_stop_hold()

  def reset_hook_2():
    reset_log_2.append("called")
    return p2.custom_long_finalizer.reset_lead_stop_hold()

  p1._reset_lead_stop_hold = reset_hook_1
  p2._reset_lead_stop_hold = reset_hook_2

  # Gas pressed triggers reset via the instance seam.
  reset_frame = {
    "sm": make_sm(v_ego=0.0, gas_pressed=True, lead_one=hold_lead),
    "mpc_a_target": -0.3,
    "mpc_should_stop": False,
    "raw_model_a_target": 0.0,
    "raw_model_should_stop": False,
  }
  out1 = p1.final_longitudinal_output(**reset_frame)
  out2 = p2.final_longitudinal_output(**reset_frame)

  _assert_value_eq(out2, out1, "monkeypatch.output")
  assert reset_log_1 == ["called"]
  assert reset_log_2 == ["called"]
  assert p2._lead_stop_hold_active is False


def test_parity_monkeypatch_slew_seam():
  p1 = make_planner(LongitudinalPlannerSP_V1)
  p2 = make_planner(LongitudinalPlannerSP)
  sm = make_sm(v_ego=15.0)
  frame = {
    "sm": sm,
    "mpc_a_target": -0.2,
    "mpc_should_stop": False,
    "raw_model_a_target": 0.3,
    "raw_model_should_stop": False,
  }
  p1.custom_long = make_custom_long(mode=LongitudinalMode.E2E)
  p2.custom_long = make_custom_long(mode=LongitudinalMode.E2E)
  p1.custom_long_output = make_custom_output(mode=LongitudinalMode.E2E)
  p2.custom_long_output = make_custom_output(mode=LongitudinalMode.E2E)

  calls_1 = []
  calls_2 = []

  def slew_hook_1(sm_inner, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop):
    calls_1.append(a_target)
    return a_target

  def slew_hook_2(sm_inner, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop):
    calls_2.append(a_target)
    return a_target

  p1._apply_stop_hold_release_slew = slew_hook_1
  p2._apply_stop_hold_release_slew = slew_hook_2

  out1 = p1.final_longitudinal_output(**frame)
  out2 = p2.final_longitudinal_output(**frame)

  _assert_value_eq(out2, out1, "slew.output")
  assert calls_1 == calls_2
  assert len(calls_1) == 1
