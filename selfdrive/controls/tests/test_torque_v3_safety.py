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

  result = envelope.update(make_inputs(unshaped_output=0.8, authority_scale=0.5))

  assert result.output_torque == 0.4
  assert result.authority_limited


def test_full_authority_does_not_cap_clean_output():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)
  result = envelope.update(make_inputs(unshaped_output=0.8, authority_scale=1.0))

  assert result.output_torque == 0.8
  assert not result.authority_limited


def test_v2_shaping_still_applies_after_authority_cap():
  envelope = TorqueV3SafetyEnvelope(dt=0.01)
  result = envelope.update(make_inputs(unshaped_output=0.8, authority_scale=1.0, steering_pressed=True, release_active=True))

  assert result.shaping_result.active
  assert result.shaping_result.reason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert abs(result.output_torque) <= 0.64
