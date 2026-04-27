from openpilot.sunnypilot.selfdrive.controls.lib.torque_authority_envelope import BUMP_FREEZE_TIME, EnvelopeInputs, TorqueAuthorityEnvelope


def make_inputs(**overrides):
  values = {
    "active": True,
    "v_ego": 30.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "saturated": False,
    "nominal_torque": 0.05,
    "desired_lateral_accel": 0.6,
    "actual_lateral_accel": 0.45,
    "desired_lateral_jerk": 0.4,
    "actual_lateral_jerk": 0.2,
    "lookahead_lateral_jerk": 0.4,
    "tracking_torque_error": 0.0,
    "lane_change_active": False,
  }
  values.update(overrides)
  return EnvelopeInputs(**values)


def prime_hold(envelope: TorqueAuthorityEnvelope, **overrides):
  result = None
  for _ in range(80):
    result = envelope.update(make_inputs(**overrides))
  return result


def test_ramp_hold_and_taper():
  envelope = TorqueAuthorityEnvelope(0.01)
  result = prime_hold(envelope)

  assert result.phase == "HOLD"
  assert result.phase_id == 2
  assert result.phase_gain == 1.0
  assert result.output_torque > 0.0
  assert not result.release_active
  assert result.nominal_torque > 0.0
  assert result.response_deficit == 0.0

  taper_seen = False
  for _ in range(40):
    result = envelope.update(
      make_inputs(
        nominal_torque=0.0, desired_lateral_accel=0.0, actual_lateral_accel=0.0, desired_lateral_jerk=0.0, actual_lateral_jerk=0.0, lookahead_lateral_jerk=0.0
      )
    )
    taper_seen |= result.phase == "TAPER_OUT"

  assert taper_seen
  assert result.phase == "IDLE"
  assert result.phase_id == 0
  assert result.phase_gain == 0.0
  assert not result.release_active


def test_disturbance_bias_builds_and_decays():
  envelope = TorqueAuthorityEnvelope(0.01)
  prime_hold(envelope, desired_lateral_jerk=0.0, actual_lateral_jerk=0.0, lookahead_lateral_jerk=0.0)

  for _ in range(120):
    result = envelope.update(
      make_inputs(desired_lateral_jerk=0.0, actual_lateral_jerk=0.0, lookahead_lateral_jerk=0.0, actual_lateral_accel=0.25, tracking_torque_error=0.03)
    )

  assert result.disturbance_bias > 0.0
  assert result.bias_torque > 0.0
  biased = result.disturbance_bias

  for _ in range(120):
    result = envelope.update(
      make_inputs(
        active=False,
        nominal_torque=0.0,
        desired_lateral_accel=0.0,
        actual_lateral_accel=0.0,
        desired_lateral_jerk=0.0,
        actual_lateral_jerk=0.0,
        lookahead_lateral_jerk=0.0,
        tracking_torque_error=0.0,
      )
    )

  assert result.disturbance_bias < biased


def test_output_torque_is_clamped_to_max_output():
  envelope = TorqueAuthorityEnvelope(0.01)
  result = None
  for _ in range(150):
    result = envelope.update(
      make_inputs(
        max_output=1.0,
        nominal_torque=0.98,
        desired_lateral_jerk=0.0,
        actual_lateral_jerk=0.0,
        lookahead_lateral_jerk=0.0,
        actual_lateral_accel=0.2,
        tracking_torque_error=0.06,
      )
    )

  assert result.disturbance_bias > 0.0
  assert result.output_torque <= 1.0


def test_bump_freezes_learning():
  envelope = TorqueAuthorityEnvelope(0.01)
  prime_hold(envelope)

  bucket = next(iter(envelope.buckets.values()))
  authority_floor = bucket.authority_floor
  for _ in range(int(BUMP_FREEZE_TIME / envelope.dt / 2)):
    result = envelope.update(make_inputs(actual_lateral_jerk=3.0, lookahead_lateral_jerk=0.0, desired_lateral_jerk=0.0, tracking_torque_error=0.06))

  assert result.learning_frozen
  assert next(iter(envelope.buckets.values())).authority_floor == authority_floor


def test_override_low_demand_releases_envelope_quickly():
  envelope = TorqueAuthorityEnvelope(0.01)
  prime_hold(envelope)

  result = envelope.update(
    make_inputs(
      steering_pressed=True,
      nominal_torque=0.005,
      desired_lateral_accel=0.05,
      actual_lateral_accel=0.02,
      desired_lateral_jerk=0.05,
      actual_lateral_jerk=0.05,
      lookahead_lateral_jerk=0.05,
    )
  )

  assert result.phase == "TAPER_OUT"
  assert result.phase_gain <= 0.25
  assert result.authority_floor == 0.0
  assert result.release_active
  assert abs(result.assist_torque) < 1e-9
  assert abs(result.output_torque) < 0.02


def test_override_sign_conflict_releases_envelope():
  envelope = TorqueAuthorityEnvelope(0.01)
  prime_hold(envelope)

  result = envelope.update(
    make_inputs(
      steering_pressed=True,
      nominal_torque=0.08,
      desired_lateral_accel=0.25,
      actual_lateral_accel=-0.25,
      desired_lateral_jerk=0.1,
      actual_lateral_jerk=-0.2,
      lookahead_lateral_jerk=0.05,
    )
  )

  assert result.phase == "TAPER_OUT"
  assert result.phase_gain <= 0.25
  assert result.authority_floor == 0.0
  assert result.release_active
  assert abs(result.assist_torque) < 1e-9
