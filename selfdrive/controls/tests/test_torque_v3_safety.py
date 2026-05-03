from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import ConservativeOutputShapingReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_safety import TorqueV3SafetyEnvelope, TorqueV3SafetyInputs


def make_inputs(**overrides):
  values = dict(
    active=True,
    v_ego=20.0,
    steering_pressed=False,
    steer_limited_by_safety=False,
    release_active=False,
    max_output=1.0,
    unshaped_output=0.8,
    desired_lateral_accel=1.0,
    actual_lateral_accel=0.8,
    desired_lateral_jerk=0.1,
    actual_lateral_jerk=0.1,
    lookahead_lateral_jerk=0.1,
    same_sign_unwind_release=False,
    authority_scale=0.5,
  )
  values.update(overrides)
  return TorqueV3SafetyInputs(**values)


def test_authority_scale_caps_unshaped_output():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)

  result = envelope.update(make_inputs(unshaped_output=0.8, max_output=1.0, authority_scale=0.5))

  assert result.output_torque == 0.5
  assert result.authority_limited
  assert result.authority_cap == 0.5


def test_authority_scale_does_not_distort_low_command():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)
  result = envelope.update(make_inputs(unshaped_output=0.2, max_output=1.0, authority_scale=0.5))

  assert result.output_torque == 0.2
  assert not result.authority_limited
  assert result.authority_cap == 0.5


def test_full_authority_does_not_cap_clean_output():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)
  result = envelope.update(make_inputs(unshaped_output=0.8, authority_scale=1.0))

  assert result.output_torque == 0.8
  assert not result.authority_limited


def test_v2_shaping_still_applies_after_authority_cap():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)
  result = envelope.update(make_inputs(unshaped_output=0.8, max_output=1.0, authority_scale=0.5, steering_pressed=True, release_active=True))

  assert result.shaping_result.active
  assert result.shaping_result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert result.authority_limited
  assert result.authority_cap == 0.5
  assert result.shaping_result.unshaped_output == 0.5
  assert result.output_torque == 0.4


def test_stale_actuator_reversal_shaping_applies_after_authority_cap():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)

  result = envelope.update(make_inputs(v_ego=5.0, steer_limited_by_safety=True, steer_limit_same_direction=True,
                                       steering_rate_deg=40.0, unshaped_output=0.8, authority_scale=0.5,
                                       desired_lateral_accel=0.8, actual_lateral_accel=0.72,
                                       steer_limit_requested_output=0.8, steer_limit_applied_output=-0.25))

  assert result.authority_limited
  assert result.authority_cap == 0.5
  assert result.shaping_result.reason & ConservativeOutputShapingReason.STALE_ACTUATOR_REVERSAL
  assert result.shaping_result.output_cap == 0.35
  assert result.output_torque == 0.175
