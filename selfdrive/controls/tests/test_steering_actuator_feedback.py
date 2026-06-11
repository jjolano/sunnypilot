import pytest
import sys
import types
from types import SimpleNamespace

msgq = types.ModuleType("msgq")
msgq.fake_event_handle = object()
msgq.drain_sock_raw = lambda *args, **kwargs: []
msgq.MultiplePublishersError = RuntimeError
msgq.IpcError = RuntimeError
msgq.Context = object
msgq.Poller = object
msgq.SubSocket = object
msgq.PubSocket = object
msgq.SocketEventHandle = object
msgq.toggle_fake_events = lambda *args, **kwargs: None
msgq.set_fake_prefix = lambda *args, **kwargs: None
msgq.get_fake_prefix = lambda *args, **kwargs: ""
msgq.delete_fake_prefix = lambda *args, **kwargs: None
msgq.wait_for_one_event = lambda *args, **kwargs: None
msgq.pub_sock = lambda *args, **kwargs: None
msgq.sub_sock = lambda *args, **kwargs: None
msgq.context = None
sys.modules.setdefault("msgq", msgq)

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from cereal import car
from openpilot.selfdrive.controls.controlsd import compute_steering_actuator_feedback
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import (
  SteeringActuatorFeedback,
  SteeringActuatorRequest,
  SteeringLimitReason,
  build_steering_actuator_feedback,
  classify_steering_limit_context,
  classify_steering_limit_direction,
)


def make_actuators(**overrides):
  actuators = car.CarControl.Actuators.new_message()
  actuators.torque = 0.0
  actuators.steeringAngleDeg = 0.0
  actuators.curvature = 0.0
  for key, value in overrides.items():
    setattr(actuators, key, value)
  return actuators


def test_no_previous_request_is_invalid_and_not_limited():
  result = build_steering_actuator_feedback(None, make_actuators(torque=0.4), car.CarParams.SteerControlType.torque)

  assert result == SteeringActuatorFeedback.invalid()


def test_torque_feedback_reports_signed_same_direction_limit():
  requested = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7, steeringAngleDeg=1.0, curvature=0.01))

  result = build_steering_actuator_feedback(requested, make_actuators(torque=0.45), car.CarParams.SteerControlType.torque,
                                           current_command=0.6)

  assert result.valid
  assert result.limited
  assert result.reason & SteeringLimitReason.ACTUATOR_MISMATCH
  assert result.requested == pytest.approx(0.7)
  assert result.applied == pytest.approx(0.45)
  assert result.error == pytest.approx(0.25)
  assert result.same_direction_limited
  assert not result.unwind_allowed


def test_torque_feedback_allows_clear_unwind_from_positive_limit():
  requested = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7))

  result = build_steering_actuator_feedback(requested, make_actuators(torque=0.45), car.CarParams.SteerControlType.torque,
                                           current_command=-0.2)

  assert result.valid
  assert result.limited
  assert not result.same_direction_limited
  assert result.unwind_allowed


def test_angle_feedback_uses_angle_threshold():
  requested = SteeringActuatorRequest.from_actuators(make_actuators(steeringAngleDeg=8.0))

  result = build_steering_actuator_feedback(requested, make_actuators(steeringAngleDeg=4.0), car.CarParams.SteerControlType.angle,
                                           current_command=5.0)

  assert result.valid
  assert result.limited
  assert result.requested == pytest.approx(8.0)
  assert result.applied == pytest.approx(4.0)
  assert result.error == pytest.approx(4.0)
  assert result.same_direction_limited


def test_small_torque_difference_is_not_limited():
  requested = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7))

  result = build_steering_actuator_feedback(requested, make_actuators(torque=0.695), car.CarParams.SteerControlType.torque,
                                           current_command=0.6)

  assert result.valid
  assert not result.limited
  assert result.reason == SteeringLimitReason.NONE
  assert not result.same_direction_limited
  assert not result.unwind_allowed


def test_feedback_without_current_command_does_not_classify_direction():
  requested = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7))

  result = build_steering_actuator_feedback(requested, make_actuators(torque=0.45), car.CarParams.SteerControlType.torque)

  assert result.valid
  assert result.limited
  assert not result.same_direction_limited
  assert not result.unwind_allowed


def test_limit_direction_can_be_classified_for_current_command():
  feedback = SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, True, True)

  same_direction, unwind = classify_steering_limit_direction(feedback, -0.2)

  assert not same_direction
  assert unwind


def test_limit_context_classifies_direction_with_current_command():
  feedback = SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, False, False)

  same_direction = classify_steering_limit_context(feedback, 0.2)
  unwind = classify_steering_limit_context(feedback, -0.2)

  assert same_direction.same_direction_limited
  assert not same_direction.unwind_allowed
  assert not unwind.same_direction_limited
  assert unwind.unwind_allowed


class DummyLatControl(LatControl):
  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    raise NotImplementedError


def test_latcontrol_stores_steering_actuator_feedback():
  controller = DummyLatControl(SimpleNamespace(steerLimitTimer=1.0), None, None, 0.01)
  feedback = SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, True, False)

  controller.set_steering_actuator_feedback(feedback)

  assert controller.steering_actuator_feedback == feedback


def test_controlsd_feedback_uses_previous_request_against_latest_output():
  previous_request = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7))

  result = compute_steering_actuator_feedback(previous_request, make_actuators(torque=0.45), car.CarParams.SteerControlType.torque,
                                             lat_active=True)

  assert result.valid
  assert result.limited
  assert result.requested == pytest.approx(0.7)
  assert result.applied == pytest.approx(0.45)
  assert not result.same_direction_limited
  assert not result.unwind_allowed


def test_controlsd_feedback_clears_when_lateral_inactive():
  previous_request = SteeringActuatorRequest.from_actuators(make_actuators(torque=0.7))

  result = compute_steering_actuator_feedback(previous_request, make_actuators(torque=0.45), car.CarParams.SteerControlType.torque,
                                             lat_active=False)

  assert result == SteeringActuatorFeedback.invalid(SteeringLimitReason.INACTIVE)
