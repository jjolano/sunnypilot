import builtins
from types import SimpleNamespace

import pytest

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.drive_lab import shadow_manual_longitudinal as shadow_cli
from openpilot.tools.drive_lab.shadow_manual_longitudinal import (
  DEFAULT_MIN_INFERRED_CRUISE_KPH,
  MissingModelV2Error,
  ShadowPlannerTargetSample,
  ShadowReplayError,
  ShadowReplayOptions,
  configure_shadow_params,
  default_planner_factory,
  extract_shadow_samples,
  shape_shadow_payload,
  summarize_shadow_agreement,
)
from openpilot.tools.drive_lab.timeline import format_enum
from openpilot.tools.lib.logreader import ReadMode


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakePlanner:
  def __init__(self):
    self.update_calls = []
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.fcw = False
    self.source = "cruise"
    self.longitudinal_plan_source = "cruise"
    self.longitudinal_stack_actuated_stack = "custom-2.0"

  def update(self, sm):
    self.update_calls.append(sm)
    self.output_a_target = 0.7
    self.output_should_stop = True
    self.fcw = True
    self.source = "cruise"
    self.longitudinal_plan_source = "lead0"


def ns(**kwargs):
  return SimpleNamespace(**kwargs)


def msg(kind, t_s, payload=None, **kwargs):
  payload = payload if payload is not None else ns(**kwargs)
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: payload})


def required_shadow_msgs(v_cruise=255.0, gas_pressed=True, brake_pressed=True, model_should_stop=True):
  return [
    msg("carParams", 0.00, carName="toyota", openpilotLongitudinalControl=True, vEgoStopping=0.25),
    msg("carParamsSP", 0.00),
    msg("selfdriveState", 0.01, enabled=False, active=False, experimentalMode=False, personality="standard"),
    msg("carControl", 0.02, enabled=False, longActive=False, cruiseControl=ns(override=True), orientationNED=[]),
    msg("controlsState", 0.03, longControlState="off", forceDecel=True),
    msg("liveParameters", 0.04, angleOffsetDeg=0.0, stiffnessFactor=1.0, steerRatio=15.0, roll=0.0),
    msg("radarState", 0.05, leadOne=ns(status=True, dRel=12.0, vRel=-0.2, yRel=0.0)),
    msg("carStateSP", 0.06),
    msg("liveMapDataSP", 0.07),
    msg("gpsLocation", 0.08),
    msg("gpsLocationExternal", 0.09),
    msg("carState", 0.10, vEgo=10.0, aEgo=0.1, gasPressed=gas_pressed, brakePressed=brake_pressed,
        standstill=False, vCruise=v_cruise, vCruiseCluster=v_cruise, buttonEvents=[]),
    msg("modelV2", 0.11, action=ns(desiredAcceleration=-0.4, shouldStop=model_should_stop)),
  ]


def planner_sample(t=0.0, inferred_cruise=False, model_should_stop=False, v=8.0, plan_a=0.2, a_ego=0.1):
  return ShadowPlannerTargetSample(
    route="route-a--3/rlog.zst",
    route_id="route-a",
    segment=3,
    t=t,
    v_ego=v,
    a_ego=a_ego,
    gas_pressed=False,
    brake_pressed=False,
    standstill=False,
    selfdrive_enabled=True,
    selfdrive_active=True,
    long_active=True,
    long_control_state="pid",
    v_cruise_kph=80.0,
    plan_a_target=plan_a,
    plan_source="cruise",
    plan_should_stop=False,
    plan_fcw=False,
    sp_a_target=plan_a,
    sp_source="cruise",
    sp_stack="customV2",
    lead_status=False,
    lead_d_rel=None,
    lead_v_rel=None,
    model_desired_accel=None,
    model_should_stop=model_should_stop,
    inferred_cruise=inferred_cruise,
  )


def test_missing_model_v2_requires_rlog(monkeypatch):
  monkeypatch.setattr(shadow_cli, "LogReader", lambda route, default_mode, sort_by_time: [msg("carState", 0.0, vEgo=1.0)])

  with pytest.raises(MissingModelV2Error, match="requires rlogs"):
    extract_shadow_samples("route-a--3/qlog.zst", ReadMode.QLOG, ShadowReplayOptions())


def test_missing_model_v2_takes_precedence_over_unset_cruise_option(monkeypatch):
  monkeypatch.setattr(shadow_cli, "LogReader", lambda route, default_mode, sort_by_time: [
    msg("carState", 0.0, vEgo=1.0, vCruise=255.0),
  ])

  with pytest.raises(MissingModelV2Error, match="requires rlogs"):
    extract_shadow_samples("route-a--3/qlog.zst", ReadMode.QLOG, ShadowReplayOptions(fail_on_unset_cruise=True))


def test_extract_shadow_samples_shapes_as_if_engaged_inputs(monkeypatch):
  msgs = required_shadow_msgs()
  planner = FakePlanner()
  monkeypatch.setattr(shadow_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = extract_shadow_samples("route-a--3/rlog.zst", ReadMode.RLOG, ShadowReplayOptions(),
                                   planner_factory=lambda CP, CP_SP, init_v, init_a: planner)

  assert len(samples) == 1
  assert len(planner.update_calls) == 1
  sm = planner.update_calls[0]
  assert sm["selfdriveState"].enabled
  assert sm["selfdriveState"].active
  assert sm["carControl"].enabled
  assert sm["carControl"].longActive
  assert not sm["carControl"].cruiseControl.override
  assert sm["controlsState"].longControlState == "pid"
  assert not sm["controlsState"].forceDecel
  assert not sm["carState"].gasPressed
  assert not sm["carState"].brakePressed
  assert sm.recv_time["modelV2"] > 0.0
  assert sm["carState"].vCruise == pytest.approx(DEFAULT_MIN_INFERRED_CRUISE_KPH)
  assert sm["carState"].vCruiseCluster == pytest.approx(DEFAULT_MIN_INFERRED_CRUISE_KPH)

  sample = samples[0]
  assert sample.inferred_cruise
  assert sample.model_should_stop
  assert sample.plan_a_target == pytest.approx(0.7)
  assert sample.plan_source == "lead0"
  assert sample.sp_stack == "customV2"
  assert sample.gas_pressed
  assert sample.brake_pressed


@pytest.mark.parametrize(("stack", "enabled"), (("custom-2.0", True), ("sunnypilot-current", False)))
def test_configure_shadow_params_materializes_defaults_and_stack_override(stack, enabled):
  with OpenpilotPrefix():
    configure_shadow_params(stack)
    params = Params()

    assert params.get_bool("CustomLongitudinalEnabled") is enabled
    assert params.get("CustomLongitudinalMode") == "scc"
    assert params.get("DynamicFollowGapMode") == "shadow"


def test_preserve_driver_pedals_and_fixed_virtual_cruise(monkeypatch):
  msgs = required_shadow_msgs(v_cruise=255.0, gas_pressed=True, brake_pressed=True)
  planner = FakePlanner()
  monkeypatch.setattr(shadow_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)
  options = ShadowReplayOptions(preserve_driver_pedals=True, fallback_v_cruise_kph=72.0)

  extract_shadow_samples("route-a--3/rlog.zst", ReadMode.RLOG, options,
                         planner_factory=lambda CP, CP_SP, init_v, init_a: planner)

  car_state = planner.update_calls[0]["carState"]
  assert car_state.gasPressed
  assert car_state.brakePressed
  assert car_state.vCruise == pytest.approx(72.0)
  assert car_state.vCruiseCluster == pytest.approx(72.0)


def test_fail_on_unset_cruise_blocks_inference():
  with pytest.raises(ShadowReplayError, match="unset vCruise"):
    shape_shadow_payload("carState", ns(vCruise=255.0, vEgo=5.0), ShadowReplayOptions(fail_on_unset_cruise=True))


def test_shape_shadow_payload_copies_cereal_builders_without_mutating_raw():
  car_state = messaging.new_message("carState").carState
  car_state.vEgo = 10.0
  car_state.vCruise = 255.0
  car_state.vCruiseCluster = 255.0
  car_state.gasPressed = True
  car_state.brakePressed = True

  shadow_car_state, inferred = shape_shadow_payload("carState", car_state, ShadowReplayOptions())

  assert inferred
  assert car_state.vCruise == pytest.approx(255.0)
  assert car_state.gasPressed
  assert car_state.brakePressed
  assert shadow_car_state.vCruise == pytest.approx(DEFAULT_MIN_INFERRED_CRUISE_KPH)
  assert not shadow_car_state.gasPressed
  assert not shadow_car_state.brakePressed

  controls_state = messaging.new_message("controlsState").controlsState
  controls_state.longControlState = "off"
  controls_state.forceDecel = True

  shadow_controls_state, _ = shape_shadow_payload("controlsState", controls_state, ShadowReplayOptions())

  assert format_enum(controls_state.longControlState) == "off"
  assert controls_state.forceDecel
  assert format_enum(shadow_controls_state.longControlState) == "pid"
  assert not shadow_controls_state.forceDecel


def test_summary_counts_inferred_cruise_without_overwriting_model_should_stop():
  samples = [
    planner_sample(t=0.0, inferred_cruise=True, model_should_stop=False),
    planner_sample(t=0.1, inferred_cruise=False, model_should_stop=True),
  ]

  summary = summarize_shadow_agreement({"route-a--3/rlog.zst": samples})

  assert summary.inferred_cruise_sample_count == 1
  assert summary.route_profiles[0].inferred_cruise_samples == 1
  assert samples[0].model_should_stop is False
  assert samples[1].model_should_stop is True


def test_default_planner_factory_reports_generated_mpc_import_failure(monkeypatch):
  real_import = builtins.__import__

  def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "openpilot.selfdrive.controls.lib.longitudinal_planner":
      raise ModuleNotFoundError("No module named 'openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx'")
    return real_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", fake_import)

  with pytest.raises(ShadowReplayError, match="build generated longitudinal MPC modules"):
    default_planner_factory(None, None, 0.0, 0.0)
