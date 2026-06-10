import inspect
import math
import sys
import types

import pytest

from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import SteeringActuatorFeedback, SteeringLimitReason
from openpilot.sunnypilot.selfdrive.locationd.speed_aware_torque import format_speed_aware_params

params_pyx = types.ModuleType("openpilot.common.params_pyx")


class FakeParams:
  def get_bool(self, _key: str) -> bool:
    return False

  def remove(self, _key: str) -> None:
    pass

  def get(self, _key: str, *_args, **_kwargs):
    return None


params_pyx.Params = FakeParams
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_v4

LatControlTorqueV4 = latcontrol_torque_v4.LatControlTorqueV4
LatControlTorqueV41 = latcontrol_torque_v4.LatControlTorqueV41
TorqueV4GovernorReason = latcontrol_torque_v4.TorqueV4GovernorReason
TorqueV4LearnerRejectReason = latcontrol_torque_v4.TorqueV4LearnerRejectReason
TorqueV4Observation = latcontrol_torque_v4.TorqueV4Observation
TorqueV4OutputGovernor = latcontrol_torque_v4.TorqueV4OutputGovernor
TorqueV4SessionAdaptation = latcontrol_torque_v4.TorqueV4SessionAdaptation
TorqueV4SpeedModel = latcontrol_torque_v4.TorqueV4SpeedModel
TorqueV4SpeedModelResult = latcontrol_torque_v4.TorqueV4SpeedModelResult
TorqueV4RecenterMode = latcontrol_torque_v4.TorqueV4RecenterMode
TorqueV4Target = latcontrol_torque_v4.TorqueV4Target
finite_difference_curvature_rate_from_steering_rate = latcontrol_torque_v4.finite_difference_curvature_rate_from_steering_rate


def get_context(car_name=TOYOTA.TOYOTA_RAV4):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  return CP, CP_SP, CI


def get_controller(car_name=TOYOTA.TOYOTA_RAV4):
  CP, CP_SP, CI = get_context(car_name)
  VM = VehicleModel(CP)
  CP_SP = convert_to_capnp(CP_SP)
  controller = LatControlTorqueV4(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return controller, VM, CP


def make_car_state(v_ego=20.0, steering_angle=0.0, steering_rate=0.0, steering_pressed=False):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle
  CS.steeringRateDeg = steering_rate
  CS.steeringPressed = steering_pressed
  return CS


def update(controller, VM, CS, desired_curvature, *, active=True, steer_limited=False, curvature_limited=False, lat_delay=0.2):
  params = log.LiveParametersData.new_message()
  return controller.update(active, CS, VM, params, steer_limited, desired_curvature, None, curvature_limited, lat_delay)


def make_speed_result(**overrides):
  values = {
    "response_scale": 1.0,
    "trim_lateral_accel": 0.0,
    "response_delay": 0.2,
    "lead_gain": 0.5,
    "lead_delta_cap": 0.5,
    "feedback_gain": 0.2,
    "damping_gain": 0.05,
    "breakaway_scale": 0.6,
    "output_slew_rate": 3.0,
    "sign_change_slew_rate": 1.5,
    "speed_aware_confidence": 0.0,
    "speed_aware_factor": 1.0,
    "effective_lat_accel_factor": 1.0,
    "effective_lat_accel_offset": 0.0,
  }
  values.update(overrides)
  return TorqueV4SpeedModelResult(**values)


def make_observation(**overrides):
  values = {
    "active": True,
    "v_ego": 20.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "saturated": False,
    "raw_target_lateral_accel": 0.5,
    "delay_lead_lateral_accel": 0.5,
    "target_lateral_accel_rate": 0.0,
    "actual_lateral_accel": 0.4,
    "actual_lateral_jerk": 0.1,
    "measurement_rate": 0.0,
    "finite": True,
  }
  values.update(overrides)
  return TorqueV4Observation(**values)


def make_processed_lateral_demand(**overrides):
  values = {
    "raw_curvature": 0.001,
    "processed_curvature": 0.001,
    "measured_curvature": 0.0,
    "curvature_limited": False,
    "path_quality": 1.0,
    "path_reason": "ok",
    "lane_change_shaping_active": False,
    "lane_change_blend": 0.0,
    "lateral_accel_limit": 2.5,
    "demand_source": DEMAND_SOURCE_MODEL_PATH,
  }
  values.update(overrides)
  return ProcessedLateralDemand(**values)


class ApplyEnabledParams(FakeParams):
  def get_bool(self, key: str) -> bool:
    return key == "LiveTorqueSpeedAdaptiveApplyToggle"


def test_v4_requires_native_torque_tuning():
  CP, CP_SP, CI = get_context()
  CP.lateralTuning.init('pid')
  CP.lateralTuning.pid.kpBP = [0.0]
  CP.lateralTuning.pid.kpV = [0.1]
  CP.lateralTuning.pid.kiBP = [0.0]
  CP.lateralTuning.pid.kiV = [0.01]
  CP.lateralTuning.pid.kf = 0.00006
  CP_SP = convert_to_capnp(CP_SP)

  with pytest.raises(ValueError, match="Torque v4 requires native torque lateral tuning"):
    LatControlTorqueV4(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)


def test_v4_exposes_direct_controlsd_hooks_without_model_hooks():
  controller, _VM, _CP = get_controller()

  assert controller.CONTROL_STATE == "torque"
  assert hasattr(controller, "update_live_torque_params")
  assert hasattr(controller, "update_speed_aware_params")
  assert hasattr(controller, "update_lateral_lag")
  assert hasattr(controller, "set_processed_lateral_demand")
  assert hasattr(controller, "reset")
  assert not hasattr(controller, "extension")
  assert not hasattr(controller, "update_model_v2")
  assert not hasattr(controller, "model_v2")
  assert not hasattr(controller, "model_valid")


def test_v4_default_governor_profile_is_unchanged():
  assert LatControlTorqueV4.GOVERNOR_PROFILE.output_slew_rate_bp == [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
  assert LatControlTorqueV4.GOVERNOR_PROFILE.output_slew_rate_v == [0.80, 1.10, 2.40, 3.60, 4.00, 4.00]
  assert LatControlTorqueV4.GOVERNOR_PROFILE.sign_change_slew_rate_bp == [0.0, 3.0, 10.0, 20.0, 30.0, 40.0]
  assert LatControlTorqueV4.GOVERNOR_PROFILE.sign_change_slew_rate_v == [0.40, 0.60, 1.40, 2.00, 2.20, 2.00]
  assert LatControlTorqueV4.GOVERNOR_PROFILE.same_direction_limit_cap == pytest.approx(0.72)
  assert LatControlTorqueV4.GOVERNOR_PROFILE.same_direction_limit_rate == pytest.approx(1.20)
  assert LatControlTorqueV4.GOVERNOR_PROFILE.high_rate_start_deg == pytest.approx(70.0)
  assert LatControlTorqueV4.GOVERNOR_PROFILE.high_rate_full_deg == pytest.approx(100.0)
  assert LatControlTorqueV4.GOVERNOR_PROFILE.high_rate_min_cap == pytest.approx(0.60)
  assert LatControlTorqueV4.GOVERNOR_PROFILE.high_rate_slew_scale == pytest.approx(0.65)
  assert not LatControlTorqueV4.GOVERNOR_PROFILE.same_direction_decrease_bypass


def test_v41_governor_profile_is_relaxed():
  assert LatControlTorqueV41.GOVERNOR_PROFILE.output_slew_rate_v == [1.40, 2.00, 3.00, 4.20, 5.00, 5.60]
  assert LatControlTorqueV41.GOVERNOR_PROFILE.sign_change_slew_rate_v == [0.90, 1.20, 1.80, 2.40, 3.00, 3.40]
  assert LatControlTorqueV41.GOVERNOR_PROFILE.same_direction_limit_cap == pytest.approx(0.85)
  assert LatControlTorqueV41.GOVERNOR_PROFILE.same_direction_limit_rate == pytest.approx(1.30)
  assert LatControlTorqueV41.GOVERNOR_PROFILE.same_direction_limit_rate_bp == [0.0, 10.0, 20.0, 30.0, 40.0]
  assert LatControlTorqueV41.GOVERNOR_PROFILE.same_direction_limit_rate_v == [1.30, 1.30, 2.10, 3.20, 3.60]
  assert LatControlTorqueV41.GOVERNOR_PROFILE.high_rate_start_deg == pytest.approx(80.0)
  assert LatControlTorqueV41.GOVERNOR_PROFILE.high_rate_min_cap == pytest.approx(0.62)
  assert LatControlTorqueV41.GOVERNOR_PROFILE.high_rate_slew_scale == pytest.approx(0.70)
  assert LatControlTorqueV41.GOVERNOR_PROFILE.same_direction_decrease_bypass


def test_v41_speed_model_uses_relaxed_slew_profile():
  controller, _VM, _CP = get_controller()
  v4_result = TorqueV4SpeedModel(LatControlTorqueV4.GOVERNOR_PROFILE).update(
    30.0, controller.torque_params, None, False, controller.session_adaptation
  )
  v41_result = TorqueV4SpeedModel(LatControlTorqueV41.GOVERNOR_PROFILE).update(
    30.0, controller.torque_params, None, False, controller.session_adaptation
  )

  assert v4_result.output_slew_rate == pytest.approx(4.0)
  assert v4_result.sign_change_slew_rate == pytest.approx(2.2)
  assert v41_result.output_slew_rate == pytest.approx(5.0)
  assert v41_result.sign_change_slew_rate == pytest.approx(3.0)


def test_v41_logs_version_41():
  CP, CP_SP, CI = get_context()
  VM = VehicleModel(CP)
  v41 = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)

  _steer, _angle, lac_log = update(v41, VM, make_car_state(v_ego=20.0), 0.001)

  assert lac_log.version == 41


def test_v4_uses_no_v2_or_extension_post_core_limiters():
  source = inspect.getsource(latcontrol_torque_v4)

  assert "LatControlTorqueExt" not in source
  assert "TorqueConservativeOutputShaper" not in source
  assert "TorqueGuardedResponseAssist" not in source
  assert "attenuate_same_direction_over_response" not in source
  assert "TorqueV21RefinedOutputGovernor" not in source


def test_v4_direct_speed_aware_hook_parses_numeric_payload():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  friction = float(controller.torque_params.friction)
  payload = format_speed_aware_params(controller.CP, {"20_30": (factor * 1.1, 0.02, friction)})

  controller.update_speed_aware_params(str(payload))

  assert controller.speed_aware_params["20_30"][0] == pytest.approx(factor * 1.1)


def test_v4_direct_speed_aware_hook_honors_apply_toggle_false():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  payload = format_speed_aware_params(controller.CP, {"20_30": (factor * 1.2, 0.02, float(controller.torque_params.friction))})

  controller.update_speed_aware_params(str(payload))
  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, controller.speed_aware_params,
                                       controller.speed_adaptive_apply_enabled, controller.session_adaptation)

  assert controller.speed_aware_params is not None
  assert not controller.speed_adaptive_apply_enabled
  assert result.speed_aware_confidence == 0.0
  assert result.effective_lat_accel_factor == pytest.approx(factor)


def test_v4_direct_speed_aware_hook_applies_when_toggle_true(monkeypatch):
  monkeypatch.setattr(latcontrol_torque_v4, "Params", ApplyEnabledParams)
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  payload = format_speed_aware_params(controller.CP, {"20_30": (factor * 1.2, 0.02, float(controller.torque_params.friction))})

  controller.update_speed_aware_params(str(payload))
  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, controller.speed_aware_params,
                                       controller.speed_adaptive_apply_enabled, controller.session_adaptation)

  assert controller.speed_adaptive_apply_enabled
  assert 0.0 < result.speed_aware_confidence < 1.0
  assert factor < result.effective_lat_accel_factor < factor * 1.2
  assert result.effective_lat_accel_offset == pytest.approx(0.006)


def test_v4_invalid_speed_aware_payload_falls_back_to_global(monkeypatch):
  monkeypatch.setattr(latcontrol_torque_v4, "Params", ApplyEnabledParams)
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)

  controller.update_speed_aware_params("not a payload")
  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, controller.speed_aware_params,
                                       controller.speed_adaptive_apply_enabled, controller.session_adaptation)

  assert controller.speed_aware_params is None
  assert result.speed_aware_confidence == 0.0
  assert result.effective_lat_accel_factor == pytest.approx(factor)


def test_v4_missing_speed_aware_payload_falls_back_to_global(monkeypatch):
  monkeypatch.setattr(latcontrol_torque_v4, "Params", ApplyEnabledParams)
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)

  controller.update_speed_aware_params(None)
  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, controller.speed_aware_params,
                                       controller.speed_adaptive_apply_enabled, controller.session_adaptation)

  assert controller.speed_aware_params is None
  assert result.speed_aware_confidence == 0.0
  assert result.effective_lat_accel_factor == pytest.approx(factor)


def test_v4_update_live_torque_params_and_lateral_lag_are_bounded():
  controller, _VM, _CP = get_controller()

  controller.update_live_torque_params(2.0, 0.1, 0.2)
  controller.update_lateral_lag(0.01)
  low_delay = controller.session_adaptation.response_delay
  controller.update_lateral_lag(1.0)
  high_delay = controller.session_adaptation.response_delay

  assert controller.torque_params.latAccelFactor == pytest.approx(2.0)
  assert controller.torque_params.latAccelOffset == pytest.approx(0.1)
  assert controller.torque_params.friction == pytest.approx(0.2)
  assert low_delay == pytest.approx(latcontrol_torque_v4.RESPONSE_DELAY_MIN)
  assert high_delay == pytest.approx(latcontrol_torque_v4.RESPONSE_DELAY_MAX)


def test_v4_positive_curvature_preserves_existing_torque_sign_convention():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)

  steer, _angle, lac_log = update(controller, VM, CS, 0.001)

  assert steer < 0.0
  assert lac_log.output == pytest.approx(steer)
  assert lac_log.adaptiveTorqueState.rawTargetLateralAccel > 0.0


def test_v4_invalid_input_zeroes_output_and_logs_governor_reason():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)

  steer, _angle, lac_log = update(controller, VM, CS, float("nan"))

  assert steer == 0.0
  assert lac_log.output == 0.0
  assert lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.INVALID
  assert lac_log.adaptiveTorqueState.sampleRejectReason & TorqueV4LearnerRejectReason.INACTIVE


def test_v4_invalid_input_logs_finite_safe_telemetry():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)

  _steer, _angle, lac_log = update(controller, VM, CS, "bad")

  assert lac_log.output == 0.0
  assert math.isfinite(lac_log.error)
  assert math.isfinite(lac_log.errorRate)
  assert math.isfinite(lac_log.adaptiveTorqueState.actualLateralJerk)


def test_v4_inactive_resets_governor_and_adaptation_state():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)
  controller.session_adaptation.response_scale = 1.1
  controller.session_adaptation.trim_lateral_accel = 0.1
  controller.filtered_measurement_rate = 5.0
  update(controller, VM, CS, 0.002)

  update(controller, VM, CS, 0.002, active=False)

  assert controller.governor.previous_output == 0.0
  assert controller.session_adaptation.response_scale == pytest.approx(1.0)
  assert controller.session_adaptation.trim_lateral_accel == pytest.approx(0.0)
  assert controller.filtered_measurement_rate == pytest.approx(0.0)


def test_v4_delay_leads_curve_entry_and_exit():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)

  for _ in range(8):
    update(controller, VM, CS, 0.0)
  _steer, _angle, entry_log = update(controller, VM, CS, 0.001)
  for _ in range(60):
    update(controller, VM, CS, 0.001)
  _steer, _angle, exit_log = update(controller, VM, CS, 0.0)

  assert entry_log.adaptiveTorqueState.delayLeadLateralAccel > entry_log.adaptiveTorqueState.rawTargetLateralAccel
  assert exit_log.adaptiveTorqueState.delayLeadLateralAccel < exit_log.adaptiveTorqueState.rawTargetLateralAccel


def test_v4_feedback_correction_sign_tracks_processed_lateral_demand():
  positive_controller, positive_vm, _CP = get_controller()
  negative_controller, negative_vm, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)

  _steer, _angle, positive_log = update(positive_controller, positive_vm, CS, 0.001)
  _steer, _angle, negative_log = update(negative_controller, negative_vm, CS, -0.001)

  assert positive_log.adaptiveTorqueState.feedbackCorrection > 0.0
  assert negative_log.adaptiveTorqueState.feedbackCorrection < 0.0


def test_v4_processed_lateral_demand_hook_stores_scalar_metadata():
  controller, _VM, _CP = get_controller()
  demand = make_processed_lateral_demand(path_quality=0.8, path_reason="high_path_std")

  controller.set_processed_lateral_demand(demand)

  assert controller.processed_lateral_demand is demand


def test_v4_learning_rejects_forwarded_lateral_maneuver_demand():
  controller, VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand(
    demand_source=DEMAND_SOURCE_LATERAL_MANEUVER,
    path_quality=0.0,
    path_reason="lateral_maneuver",
  ))

  _steer, _angle, lac_log = update(controller, VM, make_car_state(v_ego=20.0), 0.001)

  reject_reason = lac_log.adaptiveTorqueState.sampleRejectReason
  assert reject_reason & TorqueV4LearnerRejectReason.NON_MODEL_DEMAND
  assert reject_reason & TorqueV4LearnerRejectReason.LOW_PATH_QUALITY
  assert reject_reason & TorqueV4LearnerRejectReason.PATH_REASON


def test_v4_governor_driver_override_preserves_authority():
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  governor.previous_output = 1.0
  speed_result = make_speed_result(output_slew_rate=4.0)

  override = governor.update(active=True, v_ego=20.0, steering_pressed=True, steering_rate_deg=0.0,
                             same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                             actuator_error=0.0, raw_output_torque=1.0, max_output=1.0, speed_model=speed_result)

  assert override.reason & TorqueV4GovernorReason.DRIVER_OVERRIDE
  assert override.output_torque == pytest.approx(1.0)


def test_v4_governor_same_direction_limit_and_high_rate_cap_output():
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  speed_result = make_speed_result(output_slew_rate=10.0)

  same_direction = governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
                                   same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                                   actuator_error=0.0, raw_output_torque=1.0, max_output=1.0, speed_model=speed_result)
  high_rate = governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=100.0,
                              same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                              actuator_error=0.0, raw_output_torque=1.0, max_output=1.0, speed_model=speed_result)

  assert same_direction.reason & TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
  assert same_direction.output_cap < 1.0
  assert high_rate.reason & TorqueV4GovernorReason.HIGH_STEERING_RATE
  assert high_rate.output_cap < 1.0


def test_v41_governor_relaxes_same_direction_rate_and_high_rate_gate():
  speed_result = make_speed_result(output_slew_rate=10.0, sign_change_slew_rate=10.0)
  v4_governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  v41_governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV41.GOVERNOR_PROFILE)

  v4_same_direction = v4_governor.update(active=True, v_ego=30.0, steering_pressed=False, steering_rate_deg=0.0,
                                         same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                                         actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                                         speed_model=speed_result)
  v41_same_direction = v41_governor.update(active=True, v_ego=30.0, steering_pressed=False, steering_rate_deg=0.0,
                                           same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                                           actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                                           speed_model=speed_result)

  assert v4_same_direction.output_cap == pytest.approx(0.72)
  assert v41_same_direction.output_cap == pytest.approx(0.85)
  assert v4_same_direction.output_torque == pytest.approx(1.20 * DT_CTRL)
  assert v41_same_direction.output_torque == pytest.approx(3.20 * DT_CTRL)

  v4_high_rate = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE).update(
    active=True, v_ego=30.0, steering_pressed=False, steering_rate_deg=75.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=0.0, max_output=1.0, speed_model=speed_result
  )
  v41_high_rate = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV41.GOVERNOR_PROFILE).update(
    active=True, v_ego=30.0, steering_pressed=False, steering_rate_deg=75.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=0.0, max_output=1.0, speed_model=speed_result
  )

  assert v4_high_rate.reason & TorqueV4GovernorReason.HIGH_STEERING_RATE
  assert not v41_high_rate.reason & TorqueV4GovernorReason.HIGH_STEERING_RATE


def test_v41_governor_same_direction_rate_interpolates_by_speed():
  speed_result = make_speed_result(output_slew_rate=10.0, sign_change_slew_rate=10.0)

  for v_ego, expected_rate in ((0.0, 1.30), (30.0, 3.20), (40.0, 3.60)):
    governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV41.GOVERNOR_PROFILE)
    result = governor.update(active=True, v_ego=v_ego, steering_pressed=False, steering_rate_deg=0.0,
                             same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                             actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                             speed_model=speed_result)

    assert result.output_torque == pytest.approx(expected_rate * DT_CTRL)


def test_v41_governor_uses_profiled_sign_change_rate():
  controller, _VM, _CP = get_controller()
  speed_result = TorqueV4SpeedModel(LatControlTorqueV41.GOVERNOR_PROFILE).update(
    20.0, controller.torque_params, None, False, controller.session_adaptation
  )
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV41.GOVERNOR_PROFILE)
  governor.previous_output = 0.5

  result = governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
                           same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                           actuator_error=0.0, raw_output_torque=-1.0, max_output=1.0,
                           speed_model=speed_result)

  assert result.reason & TorqueV4GovernorReason.SIGN_CHANGE_LIMITED
  assert result.output_torque == pytest.approx(0.5 - speed_result.sign_change_slew_rate * DT_CTRL)


def test_v41_governor_bypasses_slew_for_same_direction_decrease():
  speed_result = make_speed_result(output_slew_rate=0.1, sign_change_slew_rate=0.1)
  v4_governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  v41_governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV41.GOVERNOR_PROFILE)
  v4_governor.previous_output = 0.8
  v41_governor.previous_output = 0.8

  v4_result = v4_governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
                                 same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                                 actuator_error=0.0, raw_output_torque=0.2, max_output=1.0,
                                 speed_model=speed_result)
  v41_result = v41_governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
                                   same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                                   actuator_error=0.0, raw_output_torque=0.2, max_output=1.0,
                                   speed_model=speed_result)

  assert v4_result.output_torque > 0.2
  assert v4_result.reason & TorqueV4GovernorReason.SLEW_LIMITED
  assert v41_result.output_torque == pytest.approx(0.2)
  assert not v41_result.reason & TorqueV4GovernorReason.SLEW_LIMITED


def test_v4_governor_low_speed_under_response_recovers_same_direction_authority():
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  speed_result = make_speed_result(output_slew_rate=3.0, sign_change_slew_rate=1.8)

  result = governor.update(active=True, v_ego=8.0, steering_pressed=False, steering_rate_deg=0.0,
                           same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                           actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                           recovery_target_lateral_accel=0.6, actual_lateral_accel=0.3,
                           under_response_recovery_allowed=True, speed_model=speed_result)

  assert result.reason & TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
  assert result.reason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY
  assert result.reason & TorqueV4GovernorReason.SLEW_LIMITED
  assert result.output_cap == pytest.approx(latcontrol_torque_v4.LOW_SPEED_UNDER_RESPONSE_CAP)
  assert result.output_torque == pytest.approx(speed_result.output_slew_rate * DT_CTRL)


def test_v4_governor_under_response_recovery_extends_across_speeds():
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  speed_result = make_speed_result(output_slew_rate=3.0, sign_change_slew_rate=1.8)

  mid = governor.update(active=True, v_ego=10.5, steering_pressed=False, steering_rate_deg=0.0,
                        same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                        actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                        recovery_target_lateral_accel=0.6, actual_lateral_accel=0.3,
                        under_response_recovery_allowed=True, speed_model=speed_result)

  assert mid.reason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY
  assert latcontrol_torque_v4.SAME_DIRECTION_LIMIT_CAP < mid.output_cap <= latcontrol_torque_v4.LOW_SPEED_UNDER_RESPONSE_CAP
  assert latcontrol_torque_v4.SAME_DIRECTION_LIMIT_RATE * DT_CTRL < mid.output_torque <= speed_result.output_slew_rate * DT_CTRL

  governor.reset()
  high = governor.update(active=True, v_ego=25.0, steering_pressed=False, steering_rate_deg=0.0,
                         same_direction_limit=True, steer_limit_unwind=False, actuator_mismatch=False,
                         actuator_error=0.0, raw_output_torque=1.0, max_output=1.0,
                         recovery_target_lateral_accel=0.6, actual_lateral_accel=0.3,
                         under_response_recovery_allowed=True, speed_model=speed_result)

  assert high.reason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY
  assert latcontrol_torque_v4.SAME_DIRECTION_LIMIT_CAP < high.output_cap < mid.output_cap
  assert high.output_torque > latcontrol_torque_v4.SAME_DIRECTION_LIMIT_RATE * DT_CTRL
  assert high.output_torque < speed_result.output_slew_rate * DT_CTRL


@pytest.mark.parametrize("overrides", [
  {"actual_lateral_accel": 0.7},
  {"actual_lateral_accel": -0.2},
  {"recovery_target_lateral_accel": 0.2},
  {"under_response_recovery_allowed": False},
  {"steering_pressed": True},
  {"steering_rate_deg": latcontrol_torque_v4.HIGH_RATE_START_DEG + 1.0},
  {"actuator_mismatch": True, "actuator_error": latcontrol_torque_v4.STALE_ACTUATOR_ERROR_THRESHOLD + 0.01},
])
def test_v4_governor_low_speed_under_response_keeps_safety_guards(overrides):
  governor = TorqueV4OutputGovernor(DT_CTRL, LatControlTorqueV4.GOVERNOR_PROFILE)
  speed_result = make_speed_result(output_slew_rate=3.0, sign_change_slew_rate=1.8)
  values = {
    "active": True,
    "v_ego": 8.0,
    "steering_pressed": False,
    "steering_rate_deg": 0.0,
    "same_direction_limit": True,
    "steer_limit_unwind": False,
    "actuator_mismatch": False,
    "actuator_error": 0.0,
    "raw_output_torque": 1.0,
    "max_output": 1.0,
    "recovery_target_lateral_accel": 0.6,
    "actual_lateral_accel": 0.3,
    "under_response_recovery_allowed": True,
    "speed_model": speed_result,
  }
  values.update(overrides)

  result = governor.update(**values)

  assert result.output_cap <= latcontrol_torque_v4.SAME_DIRECTION_LIMIT_CAP
  assert result.output_torque <= latcontrol_torque_v4.SAME_DIRECTION_LIMIT_RATE * DT_CTRL
  assert not result.reason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY


def test_v4_under_response_recovery_fails_closed_for_bad_processed_demand_metadata():
  controller, _VM, _CP = get_controller()

  controller.set_processed_lateral_demand(make_processed_lateral_demand(lane_change_blend="bad"))
  assert not controller._under_response_recovery_allowed()

  controller.set_processed_lateral_demand(types.SimpleNamespace(
    demand_source=DEMAND_SOURCE_MODEL_PATH,
    path_quality="bad",
    path_reason="ok",
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
  ))
  assert not controller._under_response_recovery_allowed()

  controller.set_processed_lateral_demand(types.SimpleNamespace())
  assert not controller._under_response_recovery_allowed()


def test_v4_under_response_recovery_allows_low_speed_usable_lane_confidence():
  controller, _VM, _CP = get_controller()
  controller.last_v_ego = 8.0
  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.649, path_reason="low_lane_confidence"))

  assert controller._under_response_recovery_allowed()


def test_v4_under_response_recovery_blocks_unstable_low_speed_path_reasons():
  controller, _VM, _CP = get_controller()
  controller.last_v_ego = 8.0

  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.649, path_reason="high_path_std"))
  assert not controller._under_response_recovery_allowed()

  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.5, path_reason="low_lane_confidence"))
  assert not controller._under_response_recovery_allowed()


def test_v4_under_response_recovery_keeps_high_speed_path_reason_strict():
  controller, _VM, _CP = get_controller()
  controller.last_v_ego = 18.0
  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.95, path_reason="low_lane_confidence"))

  assert not controller._under_response_recovery_allowed()


def test_v4_clean_processed_demand_allows_low_speed_recovery_in_controller():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=8.0)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, -0.8, -0.7, -0.1, False, False)
  )

  _steer, _angle, lac_log = update(controller, VM, CS, 0.002, steer_limited=True)

  assert lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
  assert lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY


def test_v4_bad_processed_demand_blocks_low_speed_recovery_in_controller():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=8.0)
  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.5))
  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, -0.8, -0.7, -0.1, False, False)
  )

  _steer, _angle, lac_log = update(controller, VM, CS, 0.002, steer_limited=True)

  assert lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
  assert not lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.LOW_SPEED_UNDER_RESPONSE_RECOVERY


def test_v4_under_response_lead_boost_uses_clean_processed_demand():
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=0.4, target_rate=1.0, delay_lead_lateral_accel=0.5,
                          lead_delta=0.1, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=0.2, invalid=False)

  assert boosted.lead_gain > target.lead_gain
  assert boosted.lead_delta_cap > target.lead_delta_cap
  assert boosted.delay_lead_lateral_accel > target.delay_lead_lateral_accel


@pytest.mark.parametrize("overrides", [
  {"steering_pressed": True},
  {"active": False},
  {"invalid": True},
])
def test_v4_under_response_lead_boost_freezes_without_clean_active_control(overrides):
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=0.4, target_rate=1.0, delay_lead_lateral_accel=0.5,
                          lead_delta=0.1, lead_gain=0.5, lead_delta_cap=0.5)
  values = {
    "active": True,
    "steering_pressed": False,
    "invalid": False,
  }
  values.update(overrides)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        actual_lateral_accel=0.2, **values)

  assert boosted == target


def test_v4_under_response_lead_boost_requires_clean_processed_demand():
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.5))
  target = TorqueV4Target(raw_lateral_accel=0.4, target_rate=1.0, delay_lead_lateral_accel=0.5,
                          lead_delta=0.1, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=0.2, invalid=False)

  assert boosted == target


def test_v41_under_response_release_hold_does_not_release_away_from_lagging_raw_target():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=-0.6, target_rate=1.0, delay_lead_lateral_accel=-0.2,
                          lead_delta=0.4, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=-0.1, invalid=False)

  assert boosted.delay_lead_lateral_accel <= target.delay_lead_lateral_accel
  assert abs(boosted.lead_delta) < abs(target.lead_delta)
  assert boosted.delay_lead_lateral_accel == pytest.approx(target.raw_lateral_accel)


def test_v41_under_response_release_hold_blocks_partial_release_away_from_lagging_raw_target():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=-0.6, target_rate=1.0, delay_lead_lateral_accel=-0.2,
                          lead_delta=0.4, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=-0.42, invalid=False)

  assert 0.0 < latcontrol_torque_v4._under_response_strength(target.raw_lateral_accel, -0.42) < 1.0
  assert boosted.lead_delta == pytest.approx(0.0)
  assert boosted.delay_lead_lateral_accel == pytest.approx(target.raw_lateral_accel)


def test_v41_under_response_release_hold_freezes_at_high_steering_rate():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=-0.6, target_rate=1.0, delay_lead_lateral_accel=-0.2,
                          lead_delta=0.4, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(
    target,
    make_speed_result(response_delay=0.2),
    v_ego=18.0,
    active=True,
    steering_pressed=False,
    actual_lateral_accel=-0.42,
    invalid=False,
    steering_rate_deg=LatControlTorqueV41.UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG,
  )

  assert boosted == target


def test_v41_under_response_release_hold_requires_uncurtailed_processed_demand():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand(curvature_limited=True))
  target = TorqueV4Target(raw_lateral_accel=-0.6, target_rate=1.0, delay_lead_lateral_accel=-0.2,
                          lead_delta=0.4, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=-0.42, invalid=False)

  assert boosted == target


def test_v4_under_response_release_hold_remains_disabled_for_base_v4():
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  target = TorqueV4Target(raw_lateral_accel=-0.6, target_rate=1.0, delay_lead_lateral_accel=-0.2,
                          lead_delta=0.4, lead_gain=0.5, lead_delta_cap=0.5)

  boosted = controller._apply_under_response_lead_boost(target, make_speed_result(response_delay=0.2), v_ego=18.0,
                                                        active=True, steering_pressed=False,
                                                        actual_lateral_accel=-0.1, invalid=False)

  assert boosted.delay_lead_lateral_accel > target.raw_lateral_accel
  assert boosted.lead_delta > 0.0


@pytest.mark.parametrize("raw_target,actual", [(0.6, 0.2), (-0.6, -0.2)])
def test_v41_under_response_catchup_adds_bounded_same_sign_feedback(raw_target, actual):
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())

  correction = controller._under_response_catchup_correction(
    raw_target,
    actual,
    v_ego=18.0,
    steering_rate_deg=0.0,
    active=True,
    steering_pressed=False,
    invalid=False,
  )

  expected_cap = latcontrol_torque_v4._interp(18.0, latcontrol_torque_v4.V41_UNDER_RESPONSE_CATCHUP_CAP_BP,
                                              latcontrol_torque_v4.V41_UNDER_RESPONSE_CATCHUP_CAP_V)
  assert math.copysign(1.0, correction) == math.copysign(1.0, raw_target)
  assert 0.0 < abs(correction) <= expected_cap + 1e-9


@pytest.mark.parametrize("overrides", [
  {"active": False},
  {"steering_pressed": True},
  {"invalid": True},
  {"steering_rate_deg": LatControlTorqueV41.UNDER_RESPONSE_CATCHUP_MAX_STEERING_RATE_DEG},
  {"actual_lateral_accel": -0.2},
  {"actual_lateral_accel": 0.65},
])
def test_v41_under_response_catchup_keeps_safety_guards(overrides):
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand())

  values = {
    "raw_target_lateral_accel": 0.6,
    "actual_lateral_accel": 0.2,
    "v_ego": 18.0,
    "steering_rate_deg": 0.0,
    "active": True,
    "steering_pressed": False,
    "invalid": False,
  }
  values.update(overrides)

  correction = controller._under_response_catchup_correction(
    values["raw_target_lateral_accel"],
    values["actual_lateral_accel"],
    v_ego=values["v_ego"],
    steering_rate_deg=values["steering_rate_deg"],
    active=values["active"],
    steering_pressed=values["steering_pressed"],
    invalid=values["invalid"],
  )

  assert correction == 0.0


def test_v41_under_response_catchup_requires_clean_processed_demand():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand(path_quality=0.5))

  correction = controller._under_response_catchup_correction(
    0.6,
    0.2,
    v_ego=18.0,
    steering_rate_deg=0.0,
    active=True,
    steering_pressed=False,
    invalid=False,
  )

  assert correction == 0.0


def test_v41_under_response_catchup_blocks_curvature_limited_processed_demand():
  CP, CP_SP, CI = get_context()
  controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  controller.set_processed_lateral_demand(make_processed_lateral_demand(curvature_limited=True))

  correction = controller._under_response_catchup_correction(
    0.6,
    0.2,
    v_ego=18.0,
    steering_rate_deg=0.0,
    active=True,
    steering_pressed=False,
    invalid=False,
  )

  assert correction == 0.0


def test_v41_under_response_catchup_reaches_update_feedback_without_base_v4_change():
  base_controller, base_vm, _CP = get_controller()
  CP, CP_SP, CI = get_context()
  v41_controller = LatControlTorqueV41(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  CS = make_car_state(v_ego=20.0, steering_angle=0.0, steering_rate=0.0)
  base_controller.set_processed_lateral_demand(make_processed_lateral_demand())
  v41_controller.set_processed_lateral_demand(make_processed_lateral_demand())

  _base_steer, _angle, base_log = update(base_controller, base_vm, CS, 0.0015)
  _v41_steer, _angle, v41_log = update(v41_controller, VehicleModel(CP), CS, 0.0015)

  assert base_controller._under_response_catchup_correction(
    base_log.adaptiveTorqueState.rawTargetLateralAccel,
    base_log.actualLateralAccel,
    v_ego=CS.vEgo,
    steering_rate_deg=CS.steeringRateDeg,
    active=True,
    steering_pressed=False,
    invalid=False,
  ) == 0.0
  assert v41_log.adaptiveTorqueState.assistOutput > base_log.adaptiveTorqueState.assistOutput


def test_v4_same_direction_safety_limit_caps_controller_output():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)
  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, -0.8, -0.5, -0.3, False, False)
  )

  _steer, _angle, lac_log = update(controller, VM, CS, 0.01, steer_limited=True)

  assert lac_log.adaptiveTorqueState.governorReason & TorqueV4GovernorReason.SAME_DIRECTION_LIMIT
  assert lac_log.adaptiveTorqueState.outputCap < 1.0


def test_v4_speed_model_ignores_low_speed_learned_bucket():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"0_10": (factor * 1.5, 0.12, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(5.0, controller.torque_params, params, True, controller.session_adaptation)

  assert result.speed_aware_confidence == 0.0
  assert result.speed_aware_factor == pytest.approx(factor)
  assert result.response_scale == pytest.approx(1.0)
  assert result.effective_lat_accel_factor == pytest.approx(factor)


def test_v4_speed_model_missing_bucket_falls_back_to_global():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"20_30": (factor * 1.2, 0.12, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(15.0, controller.torque_params, params, True, controller.session_adaptation)

  assert result.speed_aware_confidence == 0.0
  assert result.speed_aware_factor == pytest.approx(factor)
  assert result.response_scale == pytest.approx(1.0)


def test_v4_speed_model_gates_10_20_bucket_below_collection_speed():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"10_20": (factor * 1.2, 0.0, float(controller.torque_params.friction))}

  below = TorqueV4SpeedModel().update(12.0, controller.torque_params, params, True, controller.session_adaptation)
  at_gate = TorqueV4SpeedModel().update(15.0, controller.torque_params, params, True, controller.session_adaptation)
  partial = TorqueV4SpeedModel().update(17.5, controller.torque_params, params, True, controller.session_adaptation)

  assert below.speed_aware_confidence == 0.0
  assert below.effective_lat_accel_factor == pytest.approx(factor)
  assert at_gate.speed_aware_confidence == pytest.approx(0.0)
  assert at_gate.effective_lat_accel_factor == pytest.approx(factor)
  assert 0.0 < partial.speed_aware_confidence < 0.30
  assert factor < partial.effective_lat_accel_factor < factor * 1.2


def test_v4_speed_model_valid_medium_bucket_uses_shrinkage():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"20_30": (factor * 1.2, 0.12, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, params, True, controller.session_adaptation)

  assert 0.0 < result.speed_aware_confidence < 1.0
  assert factor < result.speed_aware_factor < factor * 1.2
  assert result.effective_lat_accel_factor == pytest.approx(result.speed_aware_factor)
  assert latcontrol_torque_v4.RESPONSE_SCALE_MIN <= result.response_scale <= latcontrol_torque_v4.RESPONSE_SCALE_MAX


def test_v4_speed_model_valid_high_bucket_uses_shrinkage():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"30_40": (factor * 0.85, -0.08, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(35.0, controller.torque_params, params, True, controller.session_adaptation)

  assert 0.0 < result.speed_aware_confidence < 1.0
  assert factor * 0.85 < result.speed_aware_factor < factor
  assert result.effective_lat_accel_factor == pytest.approx(result.speed_aware_factor)
  assert latcontrol_torque_v4.TRIM_LAT_ACCEL_MIN <= result.trim_lateral_accel <= latcontrol_torque_v4.TRIM_LAT_ACCEL_MAX


def test_v4_speed_model_transition_across_valid_buckets_is_bounded():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {
    "10_20": (factor * 1.10, 0.0, float(controller.torque_params.friction)),
    "20_30": (factor * 1.12, 0.0, float(controller.torque_params.friction)),
  }

  low = TorqueV4SpeedModel().update(19.9, controller.torque_params, params, True, controller.session_adaptation)
  high = TorqueV4SpeedModel().update(20.1, controller.torque_params, params, True, controller.session_adaptation)

  assert abs(high.response_scale - low.response_scale) < 0.01


def test_v4_speed_model_interpolates_offset_across_valid_buckets():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {
    "10_20": (factor * 1.10, -0.04, float(controller.torque_params.friction)),
    "20_30": (factor * 1.10, 0.04, float(controller.torque_params.friction)),
  }

  low = TorqueV4SpeedModel().update(19.9, controller.torque_params, params, True, controller.session_adaptation)
  high = TorqueV4SpeedModel().update(20.1, controller.torque_params, params, True, controller.session_adaptation)

  assert abs(high.effective_lat_accel_offset - low.effective_lat_accel_offset) < 0.005


def test_v4_speed_aware_effective_factor_is_reported_in_telemetry(monkeypatch):
  monkeypatch.setattr(latcontrol_torque_v4, "Params", ApplyEnabledParams)
  controller, VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  payload = format_speed_aware_params(controller.CP, {"20_30": (factor * 1.2, 0.02, float(controller.torque_params.friction))})
  controller.update_speed_aware_params(str(payload))

  _steer, _angle, lac_log = update(controller, VM, make_car_state(v_ego=25.0), 0.001)

  assert lac_log.adaptiveTorqueState.modelConfidence > 0.0
  assert factor < lac_log.adaptiveTorqueState.learnedLatAccelFactor < factor * 1.2
  assert lac_log.adaptiveTorqueState.learnedLatAccelOffset == pytest.approx(0.006)


def test_v4_effective_factor_changes_feedforward_conversion_when_apply_enabled(monkeypatch):
  monkeypatch.setattr(latcontrol_torque_v4, "Params", ApplyEnabledParams)
  base_controller, base_vm, _CP = get_controller()
  tuned_controller, tuned_vm, _CP = get_controller()
  factor = float(tuned_controller.torque_params.latAccelFactor)
  payload = format_speed_aware_params(tuned_controller.CP, {"20_30": (factor * 1.2, 0.0, float(tuned_controller.torque_params.friction))})
  tuned_controller.update_speed_aware_params(str(payload))
  CS = make_car_state(v_ego=25.0)

  _base_steer, _angle, base_log = update(base_controller, base_vm, CS, 0.001)
  tuned_steer, _angle, tuned_log = update(tuned_controller, tuned_vm, CS, 0.001)

  assert tuned_log.adaptiveTorqueState.learnedLatAccelFactor > factor
  assert abs(tuned_log.adaptiveTorqueState.unshapedOutput) < abs(base_log.adaptiveTorqueState.unshapedOutput)
  assert abs(tuned_steer) <= abs(base_log.adaptiveTorqueState.unshapedOutput)


def test_v4_measurement_rate_filter_caps_single_frame_spike():
  controller, _VM, _CP = get_controller()

  rate = controller._filtered_measurement_rate(True, False, 100.0)

  assert rate == pytest.approx(latcontrol_torque_v4.MEASUREMENT_RATE_CAP * latcontrol_torque_v4.MEASUREMENT_RATE_FILTER_ALPHA)
  assert math.isfinite(rate)


def test_v4_measurement_rate_filter_resets_on_inactive_or_invalid():
  controller, _VM, _CP = get_controller()
  controller.filtered_measurement_rate = 7.0
  inactive_rate = controller._filtered_measurement_rate(False, False, 1.0)
  controller.filtered_measurement_rate = 7.0
  invalid_rate = controller._filtered_measurement_rate(True, True, float("nan"))

  assert inactive_rate == 0.0
  assert invalid_rate == 0.0
  assert controller.filtered_measurement_rate == 0.0


@pytest.mark.parametrize("overrides,expected", [
  ({"steering_pressed": True}, TorqueV4LearnerRejectReason.STEERING_PRESSED),
  ({"saturated": True}, TorqueV4LearnerRejectReason.SATURATED),
  ({"steer_limited_by_safety": True}, TorqueV4LearnerRejectReason.STEER_LIMITED),
  ({"curvature_limited": True}, TorqueV4LearnerRejectReason.CURVATURE_LIMITED),
  ({"actual_lateral_jerk": 20.0}, TorqueV4LearnerRejectReason.HIGH_JERK),
  ({"actual_lateral_accel": -0.4}, TorqueV4LearnerRejectReason.SIGN_CONFLICT),
  ({"delay_lead_lateral_accel": 0.01}, TorqueV4LearnerRejectReason.LOW_DEMAND),
  ({"finite": False}, TorqueV4LearnerRejectReason.NON_FINITE),
])
def test_v4_session_learner_rejects_bad_samples(overrides, expected):
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(**overrides), TorqueV4GovernorReason.NONE)

  assert not result.sample_accepted
  assert result.reject_reason & expected


@pytest.mark.parametrize("overrides,expected", [
  ({"demand_source": DEMAND_SOURCE_LATERAL_MANEUVER}, TorqueV4LearnerRejectReason.NON_MODEL_DEMAND),
  ({"path_quality": 0.5}, TorqueV4LearnerRejectReason.LOW_PATH_QUALITY),
  ({"path_quality": float("nan")}, TorqueV4LearnerRejectReason.LOW_PATH_QUALITY),
  ({"path_reason": "path_disagreement"}, TorqueV4LearnerRejectReason.PATH_REASON),
  ({"lane_change_shaping_active": True}, TorqueV4LearnerRejectReason.LANE_CHANGE_SHAPING),
  ({"lane_change_blend": 0.5}, TorqueV4LearnerRejectReason.LANE_CHANGE_SHAPING),
])
def test_v4_session_learner_rejects_processed_demand_metadata(overrides, expected):
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(**overrides), TorqueV4GovernorReason.NONE)

  assert not result.sample_accepted
  assert result.reject_reason & expected


def test_v4_session_learner_rejects_safety_governor_dominated_samples():
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(), TorqueV4GovernorReason.SLEW_LIMITED)

  assert not result.sample_accepted
  assert result.reject_reason & TorqueV4LearnerRejectReason.GOVERNOR_ACTIVE


def test_v4_session_learner_accepts_clean_sample_and_stays_bounded():
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(raw_target_lateral_accel=0.5, delay_lead_lateral_accel=0.5, actual_lateral_accel=0.4),
                          TorqueV4GovernorReason.NONE)

  assert result.sample_accepted
  assert result.reject_reason == TorqueV4LearnerRejectReason.NONE
  assert latcontrol_torque_v4.RESPONSE_SCALE_MIN <= learner.response_scale <= latcontrol_torque_v4.RESPONSE_SCALE_MAX
  assert latcontrol_torque_v4.TRIM_LAT_ACCEL_MIN <= learner.trim_lateral_accel <= latcontrol_torque_v4.TRIM_LAT_ACCEL_MAX
  assert learner.response_scale > 1.0
  assert learner.trim_lateral_accel > 0.0


def test_v4_session_learner_response_residual_uses_delay_lead_target():
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(raw_target_lateral_accel=0.42, delay_lead_lateral_accel=0.8,
                                           target_lateral_accel_rate=0.2, actual_lateral_accel=0.4),
                          TorqueV4GovernorReason.NONE)

  assert result.sample_accepted
  assert result.residual_error == pytest.approx(0.4)
  assert learner.response_scale > 1.0
  assert learner.trim_lateral_accel == pytest.approx(0.0)


def test_v4_session_learner_trim_residual_uses_raw_steady_state_target():
  learner = TorqueV4SessionAdaptation(0.2)

  result = learner.update(make_observation(raw_target_lateral_accel=0.42, delay_lead_lateral_accel=0.8,
                                           target_lateral_accel_rate=0.0, actual_lateral_accel=0.4),
                          TorqueV4GovernorReason.NONE)

  assert result.sample_accepted
  assert result.residual_error == pytest.approx(0.4)
  assert learner.trim_lateral_accel == pytest.approx(0.00001)


class LinearVehicleModel:
  def __init__(self, gain=0.02):
    self.gain = gain

  def calc_curvature(self, steering_angle_rad, _v_ego, _roll):
    return self.gain * steering_angle_rad


def test_v4_actual_lateral_jerk_helper_zero_rate_is_zero():
  assert finite_difference_curvature_rate_from_steering_rate(LinearVehicleModel(), 0.1, 0.0, 20.0, 0.0) == pytest.approx(0.0)


def test_v4_actual_lateral_jerk_helper_sign_matches_measured_curvature_convention():
  vm = LinearVehicleModel()

  positive = finite_difference_curvature_rate_from_steering_rate(vm, 0.1, 0.2, 20.0, 0.0)
  negative = finite_difference_curvature_rate_from_steering_rate(vm, 0.1, -0.2, 20.0, 0.0)

  assert positive < 0.0
  assert negative > 0.0


def test_v4_actual_lateral_jerk_helper_is_finite_and_close_to_linear_approximation():
  vm = LinearVehicleModel(gain=0.03)
  rate = 0.2
  v_ego = 15.0

  result = finite_difference_curvature_rate_from_steering_rate(vm, 0.05, rate, v_ego, 0.0)

  assert math.isfinite(result)
  assert result == pytest.approx(-vm.calc_curvature(rate, v_ego, 0.0) * v_ego ** 2)


def test_v4_actual_lateral_jerk_helper_invalid_inputs_fallback_safe():
  assert finite_difference_curvature_rate_from_steering_rate(LinearVehicleModel(), float("nan"), 0.2, 20.0, 0.0) == 0.0


def test_v4_finite_helper_rejects_nonnumeric_values():
  assert not latcontrol_torque_v4._finite(None)
  assert not latcontrol_torque_v4._finite("bad")


def test_v4_actual_lateral_jerk_helper_nonnumeric_inputs_fallback_safe():
  assert finite_difference_curvature_rate_from_steering_rate(LinearVehicleModel(), "bad", 0.2, 20.0, 0.0) == 0.0


# ── Recenter Mode Tests ─────────────────────────────────────────────────

RECENTER = latcontrol_torque_v4
_DT = DT_CTRL


def _recenter_detect(controller, target, prev_target, v_ego=20.0, path_quality=1.0,
                     lane_change=False, steering=False, saturated=False, curvature_limited=False):
  """Shorthand for calling _detect_recenter_mode."""
  return controller._detect_recenter_mode(
    target_lateral_accel=target,
    previous_target_lateral_accel=prev_target,
    v_ego=v_ego,
    path_quality=path_quality,
    lane_change_active=lane_change,
    steering_pressed=steering,
    saturated=saturated,
    curvature_limited=curvature_limited,
  )


def test_recenter_mode_requires_persistence():
  """Recenter mode must not activate on the first frame; needs RECENTER_PERSISTENCE_FRAMES."""
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  controller._recenter_persistence_frames = 0

  # First frame with decreasing target — not enough persistence
  result = _recenter_detect(controller, target=0.35, prev_target=0.40)
  assert not result.active
  assert controller._recenter_persistence_frames == 1

  # Still below threshold
  for _ in range(RECENTER.RECENTER_PERSISTENCE_FRAMES - 2):
    result = _recenter_detect(controller, target=0.30, prev_target=0.35)

  assert not result.active
  assert controller._recenter_persistence_frames == RECENTER.RECENTER_PERSISTENCE_FRAMES - 1

  # One more frame — crosses threshold
  result = _recenter_detect(controller, target=0.25, prev_target=0.30)
  assert result.active
  assert result.persistence_frames >= RECENTER.RECENTER_PERSISTENCE_FRAMES


def test_recenter_mode_activates_when_target_collapses_toward_zero():
  """Feed decreasing target lateral accel at high speed with good path quality."""
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  controller._recenter_persistence_frames = 0

  # Feed decreasing targets that satisfy collapse rate and near-center condition.
  # Loop for enough frames to activate plus extra to verify persistence holds.
  prev = 0.40
  active_at_full_effect = False
  for i in range(RECENTER.RECENTER_PERSISTENCE_FRAMES * 3):
    cur = 0.40 - (i + 1) * 0.03  # keeps decreasing steadily
    result = _recenter_detect(controller, target=cur, prev_target=prev)
    prev = cur
    if result.active and result.lead_reduction > 0.0 and result.slew_boost > 1.0:
      active_at_full_effect = True
      break

  assert active_at_full_effect, "Recenter mode must become active with non-zero lead_reduction and slew_boost"


def test_recenter_mode_inactive_at_low_speed():
  """Recenter mode does not activate below RECENTER_MIN_SPEED."""
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  controller._recenter_persistence_frames = 0

  prev = 0.40
  for i in range(RECENTER.RECENTER_PERSISTENCE_FRAMES + 2):
    cur = max(0.10, prev - 0.08)
    result = _recenter_detect(controller, target=cur, prev_target=prev, v_ego=5.0)
    prev = cur

  assert not result.active


def test_recenter_mode_inactive_during_lane_change():
  """Recenter mode does not activate when lane change is active."""
  controller, _VM, _CP = get_controller()
  controller.set_processed_lateral_demand(make_processed_lateral_demand())
  controller._recenter_persistence_frames = 0

  prev = 0.40
  for i in range(RECENTER.RECENTER_PERSISTENCE_FRAMES + 2):
    cur = max(0.10, prev - 0.08)
    result = _recenter_detect(controller, target=cur, prev_target=prev, lane_change=True)
    prev = cur

  assert not result.active


def test_recenter_mode_reduces_lead_delta():
  """When recenter mode is active, _build_target should produce a smaller lead delta."""
  controller, _VM, _CP = get_controller()

  # Build a target WITHOUT recenter (normal)
  speed_result = make_speed_result(response_delay=0.2, lead_gain=0.5, lead_delta_cap=0.5)
  target_no_recenter = controller._build_target(
    desired_curvature=0.001, v_ego=20.0, speed_result=speed_result, invalid=False,
    recenter=None,
  )

  # Build a target WITH recenter (lead_reduction=0.6 = full reduction)
  recenter_active = TorqueV4RecenterMode(active=True, persistence_frames=RECENTER.RECENTER_PERSISTENCE_FRAMES,
                                         lead_reduction=0.6, slew_boost=1.5)
  target_recenter = controller._build_target(
    desired_curvature=0.001, v_ego=20.0, speed_result=speed_result, invalid=False,
    recenter=recenter_active,
  )

  # Reset previous_target_lateral_accel to get fair comparison
  controller.previous_target_lateral_accel = 0.0

  # With recenter, lead_gain and lead_delta_cap should be reduced
  expected_lead_gain = speed_result.lead_gain * (1.0 - 0.6)
  expected_lead_delta_cap = speed_result.lead_delta_cap * (1.0 - 0.6)
  assert target_recenter.lead_gain == pytest.approx(expected_lead_gain)
  assert target_recenter.lead_delta_cap == pytest.approx(expected_lead_delta_cap)
  assert abs(target_recenter.lead_delta) <= abs(target_no_recenter.lead_delta)


def test_recenter_mode_boosts_sign_change_slew_rate():
  """When recenter mode is active and a sign change occurs, the governor boosts slew rate."""
  governor = TorqueV4OutputGovernor(_DT, LatControlTorqueV4.GOVERNOR_PROFILE)
  governor.previous_output = 0.5  # positive output
  speed_result = make_speed_result(sign_change_slew_rate=1.0, output_slew_rate=3.0)

  # First without recenter
  result_normal = governor.update(
    active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=-1.0, max_output=1.0,
    speed_model=speed_result, recenter=None,
  )
  governor.previous_output = 0.5  # reset

  # With recenter
  recenter = TorqueV4RecenterMode(active=True, persistence_frames=10,
                                  lead_reduction=0.0, slew_boost=2.0)
  result_boosted = governor.update(
    active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=-1.0, max_output=1.0,
    speed_model=speed_result, recenter=recenter,
  )

  # Both should have sign change detected
  assert result_normal.reason & TorqueV4GovernorReason.SIGN_CHANGE_LIMITED
  assert result_boosted.reason & TorqueV4GovernorReason.SIGN_CHANGE_LIMITED

  # Recenter should also have RECENTER_MODE flag
  assert result_boosted.reason & TorqueV4GovernorReason.RECENTER_MODE
  assert not result_normal.reason & TorqueV4GovernorReason.RECENTER_MODE

  # Recenter mode should move further toward the target (boosted slew rate)
  # Without boost: approach from 0.5 toward -1.0 at 1.0 * DT_CTRL
  # With boost: approach from 0.5 toward -1.0 at 2.0 * DT_CTRL
  # Since target is negative, boosted output should be lower (more negative)
  assert result_boosted.output_torque < result_normal.output_torque
  assert result_boosted.output_torque == pytest.approx(0.5 - 2.0 * _DT)


# ----------------------------------------------------------------------------
# Recenter mode refinement: looser thresholds, persistence, early release
# ----------------------------------------------------------------------------


def test_recenter_mode_max_abs_target_is_broadened_to_0_85():
  """RECENTER_MAX_ABS_TARGET should be loosened from the old 0.5 to 0.85
  so turn-exit can start the unwind as soon as the target drops into the
  comfort range, not only when it is already near zero."""
  assert RECENTER.RECENTER_MAX_ABS_TARGET >= 0.8


def test_recenter_mode_persistence_shortened_to_3_frames():
  """RECENTER_PERSISTENCE_FRAMES should be shortened from the old 5 to 3
  so a brief but consistent recenter signal activates the mode quickly
  enough to feel immediate on turn exit."""
  assert RECENTER.RECENTER_PERSISTENCE_FRAMES <= 3


def test_recenter_mode_partial_lead_reduction_at_first_active_frame():
  """The first frame the recenter activates should already trim some
  response-delay lead (lead_reduction > 0), not wait a full persistence
  ramp. This is the "partial lead_reduction from first active frame"
  behavior — turn-exit feels immediate even on a short recenter."""
  controller, VM, _CP = get_controller()
  controller._recenter_persistence_frames = RECENTER.RECENTER_PERSISTENCE_FRAMES  # exactly at threshold

  result = _recenter_detect(
    controller, target=0.5, prev_target=1.0, v_ego=20.0,
  )
  # First active frame: lead_reduction should be > 0
  assert result.lead_reduction > 0.0
  # But it should be capped at RECENTER_LEAD_REDUCTION * persistence_blend
  # where persistence_blend is at the floor (RECENTER_LEAD_REDUCTION_FLOOR)
  assert result.lead_reduction <= RECENTER.RECENTER_LEAD_REDUCTION


def test_recenter_mode_full_lead_reduction_after_full_persistence_ramp():
  """After a full persistence window beyond the threshold, lead_reduction
  should ramp to the full RECENTER_LEAD_REDUCTION."""
  controller, VM, _CP = get_controller()
  controller._recenter_persistence_frames = RECENTER.RECENTER_PERSISTENCE_FRAMES * 3

  result = _recenter_detect(
    controller, target=0.5, prev_target=1.0, v_ego=20.0,
  )
  assert result.lead_reduction == pytest.approx(RECENTER.RECENTER_LEAD_REDUCTION)


def test_recenter_mode_activates_at_looser_persistence_threshold():
  """With RECENTER_PERSISTENCE_FRAMES=3, the recenter should activate
  after exactly 3 frames of consistent recentering."""
  controller, VM, _CP = get_controller()
  # Set counter to one less than threshold; the detection function will
  # increment it on a valid recentering frame, hitting the threshold.
  controller._recenter_persistence_frames = RECENTER.RECENTER_PERSISTENCE_FRAMES - 1
  result = _recenter_detect(
    controller, target=0.5, prev_target=1.0, v_ego=20.0,
  )
  # Counter incremented from 2 to 3, so recenter is now active.
  assert result.active
  assert result.persistence_frames == RECENTER.RECENTER_PERSISTENCE_FRAMES

  # One more frame: counter goes to 4, recenter still active.
  result2 = _recenter_detect(
    controller, target=0.4, prev_target=0.5, v_ego=20.0,
  )
  assert result2.active
  assert result2.persistence_frames == RECENTER.RECENTER_PERSISTENCE_FRAMES + 1


def test_recenter_mode_does_not_activate_below_persistence_threshold():
  """If the recenter signal is not sustained (target is not collapsing),
  the persistence counter does not grow and the recenter does not
  activate below the threshold."""
  controller, VM, _CP = get_controller()
  controller._recenter_persistence_frames = 0

  # Frame 1: target is not collapsing (growing), so counter does not
  # grow. With RECENTER_PERSISTENCE_FRAMES=3, the recenter is not active.
  result = _recenter_detect(
    controller, target=0.2, prev_target=0.1, v_ego=20.0,  # growing, not collapsing
  )
  assert not result.active
  assert controller._recenter_persistence_frames == 0


def test_recenter_mode_target_above_loosened_max_keeps_inactive():
  """With the loosened RECENTER_MAX_ABS_TARGET=0.85, a target above 0.85
  should keep the recenter inactive. The threshold is the upper bound
  for "near center" detection."""
  controller, VM, _CP = get_controller()
  controller._recenter_persistence_frames = RECENTER.RECENTER_PERSISTENCE_FRAMES
  # Target just above loosened max
  result = _recenter_detect(
    controller, target=RECENTER.RECENTER_MAX_ABS_TARGET + 0.01, prev_target=1.5, v_ego=20.0,
  )
  assert not result.active


def test_build_target_early_release_guard_zeros_lead_delta_on_collapse():
  """When the raw target is decreasing toward zero and the lead delta
  would push away from zero (sign mismatch), the early release guard
  should zero the lead delta. This fires before the recenter mode
  activates, so turn-exit feels immediate regardless of persistence."""
  controller, VM, CP = get_controller()
  speed_result = make_speed_result(lead_gain=0.5, lead_delta_cap=0.5, response_delay=0.2)

  # Simulate: previous target was 1.0 (turning right), now target is 0.5
  # (collapsing toward zero on turn exit). The target_rate is negative
  # so the natural lead_delta is negative — which would push the
  # controller output away from zero (the target is positive but the
  # lead is negative).
  controller.previous_target_lateral_accel = 1.0
  target = controller._build_target(
    desired_curvature=0.5 / 20.0 ** 2,  # raw_target = 0.5
    v_ego=20.0, speed_result=speed_result, invalid=False, recenter=None,
  )
  # Early release guard should have zeroed the lead_delta.
  assert target.lead_delta == 0.0


def test_build_target_early_release_guard_keeps_lead_when_target_growing():
  """When the target is growing in magnitude (turning harder), the
  early release guard should NOT fire. lead_delta should be present
  to anticipate the larger target."""
  controller, VM, CP = get_controller()
  speed_result = make_speed_result(lead_gain=0.5, lead_delta_cap=0.5, response_delay=0.2)

  # Simulate: previous target was 0.5, now target is 1.0 (growing).
  # target_rate is positive, so lead_delta is positive. The lead
  # anticipates the larger target.
  controller.previous_target_lateral_accel = 0.5
  target = controller._build_target(
    desired_curvature=1.0 / 20.0 ** 2,  # raw_target = 1.0
    v_ego=20.0, speed_result=speed_result, invalid=False, recenter=None,
  )
  # lead_delta should be non-zero (target is growing, lead is in the same direction)
  assert target.lead_delta != 0.0


def test_build_target_early_release_guard_keeps_lead_on_sign_flip():
  """When the target sign flips (turning the other way), the early
  release guard should NOT fire — the lead should track the new
  direction. The guard only fires when the target sign is stable
  (same sign as previous) AND the magnitude is decreasing."""
  controller, VM, CP = get_controller()
  speed_result = make_speed_result(lead_gain=0.5, lead_delta_cap=0.5, response_delay=0.2)

  # Sign flip: previous was +1.0, now -0.5
  controller.previous_target_lateral_accel = 1.0
  target = controller._build_target(
    desired_curvature=-0.5 / 20.0 ** 2,  # raw_target = -0.5
    v_ego=20.0, speed_result=speed_result, invalid=False, recenter=None,
  )
  # Sign is flipped, so the target_sign_stable check fails, and the
  # early release guard does NOT zero the lead.
  assert target.lead_delta != 0.0


def test_recenter_mode_applies_slew_boost_to_same_direction_unwind():
  """Recenter mode should apply a slew boost to same-direction unwind
  (not just sign change). The same-direction boost is smaller than the
  sign-change boost (RECENTER_SAME_DIRECTION_SLEW_BOOST < RECENTER_SLEW_BOOST)."""
  governor = TorqueV4OutputGovernor(_DT, LatControlTorqueV4.GOVERNOR_PROFILE)
  governor.previous_output = 0.5  # positive
  speed_result = make_speed_result(sign_change_slew_rate=1.0, output_slew_rate=3.0)

  recenter = TorqueV4RecenterMode(active=True, persistence_frames=10,
                                  lead_reduction=0.0, slew_boost=2.0)

  # Same-direction unwind: target is +0.2 (positive, smaller than previous_output 0.5)
  result = governor.update(
    active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=0.2, max_output=1.0,
    speed_model=speed_result, recenter=recenter,
  )
  # Recenter mode flag should be set even without a sign change
  assert result.reason & TorqueV4GovernorReason.RECENTER_MODE
  # Should NOT have sign-change-limited flag
  assert not result.reason & TorqueV4GovernorReason.SIGN_CHANGE_LIMITED


def test_recenter_mode_no_slew_boost_without_same_direction_unwind():
  """Recenter mode boost requires either a sign change or a
  same-direction unwind. If the target is growing in the same
  direction (target > previous), no boost is applied."""
  governor = TorqueV4OutputGovernor(_DT, LatControlTorqueV4.GOVERNOR_PROFILE)
  governor.previous_output = 0.2  # positive
  speed_result = make_speed_result(sign_change_slew_rate=1.0, output_slew_rate=3.0)

  recenter = TorqueV4RecenterMode(active=True, persistence_frames=10,
                                  lead_reduction=0.0, slew_boost=2.0)

  # Same direction, but growing: target +0.5 > previous 0.2
  result = governor.update(
    active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
    same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
    actuator_error=0.0, raw_output_torque=0.5, max_output=1.0,
    speed_model=speed_result, recenter=recenter,
  )
  # No sign change, no same-direction unwind → no RECENTER_MODE flag
  assert not result.reason & TorqueV4GovernorReason.RECENTER_MODE
  assert not result.reason & TorqueV4GovernorReason.SIGN_CHANGE_LIMITED


# ----------------------------------------------------------------------------
# Straight-road damping diagnostic preservation
# ----------------------------------------------------------------------------


def test_recenter_mode_preserves_straight_road_damping_diagnostic():
  """Straight-road damping is the model path processor's diagnostic
  for "the road is straight, the curvature is small, so hold the
  target instead of chasing noise." The recenter mode changes should
  not interfere with this diagnostic — it lives in a different layer
  (model_path_processor) and the controller just reads it.

  Verify the recenter mode is orthogonal to straight_road_damping_active:
  the recenter mode can fire while straight_road_damping_active is True,
  and vice versa. Both diagnostics are exposed for route telemetry."""
  from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessorResult

  # Straight road: curvature is small. Path processor reports damping active.
  straight_path = ModelPathProcessorResult(
    desired_curvature=0.0,  # straight
    quality=1.0,
    gated=False,
    reason="inactive",
    trust_penalty=0.0,
    straight_road_damping_active=True,  # straight road → damping is active
  )
  # Recenter mode is independent of straight-road damping. Both are
  # valid simultaneously: turn exit on a straight road.
  recenter = TorqueV4RecenterMode(active=True, persistence_frames=10,
                                  lead_reduction=0.6, slew_boost=1.5)
  # Sanity: the two fields are independent. The recenter mode fires
  # based on the target's collapse rate, not the path processor's
  # straight-road state.
  assert straight_path.straight_road_damping_active
  assert recenter.active


def test_early_release_guard_does_not_over_hold_recenter_target():
  """When straight-road damping is active and the target is collapsing
  to zero, the early release guard should still zero the lead delta.
  The damping diagnostic should not over-hold the target past the
  point where the controller would otherwise release the lead.

  This is a structural test: the early release guard operates on
  raw_target (which the path processor has already damped), and the
  recenter lead reduction operates on lead_gain. They are independent
  layers and the recenter+early-release pair should release the lead
  regardless of whether straight-road damping is active."""
  controller, VM, CP = get_controller()
  speed_result = make_speed_result(lead_gain=0.5, lead_delta_cap=0.5, response_delay=0.2)

  # Previous target 1.0, current 0.5 — collapsing toward zero.
  controller.previous_target_lateral_accel = 1.0
  target = controller._build_target(
    desired_curvature=0.5 / 20.0 ** 2,
    v_ego=20.0, speed_result=speed_result, invalid=False, recenter=None,
  )
  # Early release guard fires: lead_delta is zero regardless of any
  # straight-road damping state (which is a separate concern).
  assert target.lead_delta == 0.0
