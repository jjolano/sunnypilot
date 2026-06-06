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
