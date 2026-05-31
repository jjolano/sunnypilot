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
TorqueV4GovernorReason = latcontrol_torque_v4.TorqueV4GovernorReason
TorqueV4LearnerRejectReason = latcontrol_torque_v4.TorqueV4LearnerRejectReason
TorqueV4Observation = latcontrol_torque_v4.TorqueV4Observation
TorqueV4OutputGovernor = latcontrol_torque_v4.TorqueV4OutputGovernor
TorqueV4SessionAdaptation = latcontrol_torque_v4.TorqueV4SessionAdaptation
TorqueV4SpeedModel = latcontrol_torque_v4.TorqueV4SpeedModel
TorqueV4SpeedModelResult = latcontrol_torque_v4.TorqueV4SpeedModelResult
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
    "target_lateral_accel": 0.5,
    "target_lateral_accel_rate": 0.0,
    "actual_lateral_accel": 0.4,
    "actual_lateral_jerk": 0.1,
    "measurement_rate": 0.0,
    "finite": True,
  }
  values.update(overrides)
  return TorqueV4Observation(**values)


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
  assert hasattr(controller, "reset")
  assert not hasattr(controller, "extension")
  assert not hasattr(controller, "update_model_v2")
  assert not hasattr(controller, "model_v2")
  assert not hasattr(controller, "model_valid")


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


def test_v4_inactive_resets_governor_and_adaptation_state():
  controller, VM, _CP = get_controller()
  CS = make_car_state(v_ego=20.0)
  controller.session_adaptation.response_scale = 1.1
  controller.session_adaptation.trim_lateral_accel = 0.1
  update(controller, VM, CS, 0.002)

  update(controller, VM, CS, 0.002, active=False)

  assert controller.governor.previous_output == 0.0
  assert controller.session_adaptation.response_scale == pytest.approx(1.0)
  assert controller.session_adaptation.trim_lateral_accel == pytest.approx(0.0)


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


def test_v4_governor_driver_override_releases_with_bounded_decay():
  governor = TorqueV4OutputGovernor(DT_CTRL)
  speed_result = make_speed_result(output_slew_rate=4.0)
  active = governor.update(active=True, v_ego=20.0, steering_pressed=False, steering_rate_deg=0.0,
                           same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                           actuator_error=0.0, raw_output_torque=1.0, max_output=1.0, speed_model=speed_result)

  override = governor.update(active=True, v_ego=20.0, steering_pressed=True, steering_rate_deg=0.0,
                             same_direction_limit=False, steer_limit_unwind=False, actuator_mismatch=False,
                             actuator_error=0.0, raw_output_torque=1.0, max_output=1.0, speed_model=speed_result)

  assert override.reason & TorqueV4GovernorReason.DRIVER_OVERRIDE
  assert abs(override.output_torque) <= abs(active.output_torque)


def test_v4_governor_same_direction_limit_and_high_rate_cap_output():
  governor = TorqueV4OutputGovernor(DT_CTRL)
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

  result = TorqueV4SpeedModel().update(5.0, controller.torque_params, params, controller.session_adaptation)

  assert result.speed_aware_confidence == 0.0
  assert result.speed_aware_factor == pytest.approx(factor)
  assert result.response_scale == pytest.approx(1.0)


def test_v4_speed_model_missing_bucket_falls_back_to_global():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"20_30": (factor * 1.2, 0.12, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(15.0, controller.torque_params, params, controller.session_adaptation)

  assert result.speed_aware_confidence == 0.0
  assert result.speed_aware_factor == pytest.approx(factor)
  assert result.response_scale == pytest.approx(1.0)


def test_v4_speed_model_valid_medium_bucket_uses_shrinkage():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"20_30": (factor * 1.2, 0.12, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(25.0, controller.torque_params, params, controller.session_adaptation)

  assert 0.0 < result.speed_aware_confidence < 1.0
  assert factor < result.speed_aware_factor < factor * 1.2
  assert latcontrol_torque_v4.RESPONSE_SCALE_MIN <= result.response_scale <= latcontrol_torque_v4.RESPONSE_SCALE_MAX


def test_v4_speed_model_valid_high_bucket_uses_shrinkage():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {"30_40": (factor * 0.85, -0.08, float(controller.torque_params.friction))}

  result = TorqueV4SpeedModel().update(35.0, controller.torque_params, params, controller.session_adaptation)

  assert 0.0 < result.speed_aware_confidence < 1.0
  assert factor * 0.85 < result.speed_aware_factor < factor
  assert latcontrol_torque_v4.TRIM_LAT_ACCEL_MIN <= result.trim_lateral_accel <= latcontrol_torque_v4.TRIM_LAT_ACCEL_MAX


def test_v4_speed_model_transition_across_valid_buckets_is_bounded():
  controller, _VM, _CP = get_controller()
  factor = float(controller.torque_params.latAccelFactor)
  params = {
    "10_20": (factor * 1.10, 0.0, float(controller.torque_params.friction)),
    "20_30": (factor * 1.12, 0.0, float(controller.torque_params.friction)),
  }

  low = TorqueV4SpeedModel().update(19.9, controller.torque_params, params, controller.session_adaptation)
  high = TorqueV4SpeedModel().update(20.1, controller.torque_params, params, controller.session_adaptation)

  assert abs(high.response_scale - low.response_scale) < 0.01


@pytest.mark.parametrize("overrides,expected", [
  ({"steering_pressed": True}, TorqueV4LearnerRejectReason.STEERING_PRESSED),
  ({"saturated": True}, TorqueV4LearnerRejectReason.SATURATED),
  ({"steer_limited_by_safety": True}, TorqueV4LearnerRejectReason.STEER_LIMITED),
  ({"curvature_limited": True}, TorqueV4LearnerRejectReason.CURVATURE_LIMITED),
  ({"actual_lateral_jerk": 20.0}, TorqueV4LearnerRejectReason.HIGH_JERK),
  ({"actual_lateral_accel": -0.4}, TorqueV4LearnerRejectReason.SIGN_CONFLICT),
  ({"target_lateral_accel": 0.01}, TorqueV4LearnerRejectReason.LOW_DEMAND),
  ({"finite": False}, TorqueV4LearnerRejectReason.NON_FINITE),
])
def test_v4_session_learner_rejects_bad_samples(overrides, expected):
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

  result = learner.update(make_observation(target_lateral_accel=0.5, actual_lateral_accel=0.4), TorqueV4GovernorReason.NONE)

  assert result.sample_accepted
  assert result.reject_reason == TorqueV4LearnerRejectReason.NONE
  assert latcontrol_torque_v4.RESPONSE_SCALE_MIN <= learner.response_scale <= latcontrol_torque_v4.RESPONSE_SCALE_MAX
  assert latcontrol_torque_v4.TRIM_LAT_ACCEL_MIN <= learner.trim_lateral_accel <= latcontrol_torque_v4.TRIM_LAT_ACCEL_MAX
  assert learner.response_scale > 1.0
  assert learner.trim_lateral_accel > 0.0


class LinearVehicleModel:
  def __init__(self, gain=0.02):
    self.gain = gain

  def calc_curvature(self, steering_angle_rad, _v_ego, _roll):
    return self.gain * steering_angle_rad


def test_v4_actual_lateral_jerk_helper_zero_rate_is_zero():
  assert finite_difference_curvature_rate_from_steering_rate(LinearVehicleModel(), 0.1, 0.0, 20.0, 0.0) == pytest.approx(0.0)


def test_v4_actual_lateral_jerk_helper_sign_follows_steering_rate():
  vm = LinearVehicleModel()

  positive = finite_difference_curvature_rate_from_steering_rate(vm, 0.1, 0.2, 20.0, 0.0)
  negative = finite_difference_curvature_rate_from_steering_rate(vm, 0.1, -0.2, 20.0, 0.0)

  assert positive > 0.0
  assert negative < 0.0


def test_v4_actual_lateral_jerk_helper_is_finite_and_close_to_linear_approximation():
  vm = LinearVehicleModel(gain=0.03)
  rate = 0.2
  v_ego = 15.0

  result = finite_difference_curvature_rate_from_steering_rate(vm, 0.05, rate, v_ego, 0.0)

  assert math.isfinite(result)
  assert result == pytest.approx(vm.calc_curvature(rate, v_ego, 0.0) * v_ego ** 2)


def test_v4_actual_lateral_jerk_helper_invalid_inputs_fallback_safe():
  assert finite_difference_curvature_rate_from_steering_rate(LinearVehicleModel(), float("nan"), 0.2, 20.0, 0.0) == 0.0
