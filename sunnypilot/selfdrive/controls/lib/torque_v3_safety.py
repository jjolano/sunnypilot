from dataclasses import dataclass

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import (
  ConservativeOutputShaperInputs,
  ConservativeOutputShaperResult,
  TorqueConservativeOutputShaper,
)


@dataclass
class TorqueV3SafetyInputs:
  active: bool
  v_ego: float
  steering_pressed: bool
  steer_limited_by_safety: bool
  release_active: bool
  max_output: float
  unshaped_output: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  actual_lateral_jerk: float
  lookahead_lateral_jerk: float
  same_sign_unwind_release: bool
  authority_scale: float
  steering_rate_deg: float = 0.0
  steer_limit_same_direction: bool = True
  steer_limit_unwind: bool = False


@dataclass
class TorqueV3SafetyResult:
  output_torque: float
  authority_limited: bool
  authority_cap: float
  shaping_result: ConservativeOutputShaperResult


class TorqueV3SafetyEnvelope:
  def __init__(self, dt: float):
    self.output_shaper = TorqueConservativeOutputShaper(dt)

  def update(self, inputs: TorqueV3SafetyInputs) -> TorqueV3SafetyResult:
    authority_cap = max(0.0, min(float(inputs.authority_scale), 1.0)) * max(inputs.max_output, 0.0)
    authority_limited_output = float(np.clip(inputs.unshaped_output, -authority_cap, authority_cap))
    authority_limited = abs(authority_limited_output) < abs(inputs.unshaped_output) - 1e-6
    shaping_result = self.output_shaper.update(
      ConservativeOutputShaperInputs(
        active=inputs.active,
        v_ego=inputs.v_ego,
        steering_pressed=inputs.steering_pressed,
        steer_limited_by_safety=inputs.steer_limited_by_safety,
        release_active=inputs.release_active,
        max_output=inputs.max_output,
        unshaped_output=authority_limited_output,
        desired_lateral_accel=inputs.desired_lateral_accel,
        actual_lateral_accel=inputs.actual_lateral_accel,
        desired_lateral_jerk=inputs.desired_lateral_jerk,
        actual_lateral_jerk=inputs.actual_lateral_jerk,
        lookahead_lateral_jerk=inputs.lookahead_lateral_jerk,
        same_sign_unwind_release=inputs.same_sign_unwind_release,
        steering_rate_deg=inputs.steering_rate_deg,
        steer_limit_same_direction=inputs.steer_limit_same_direction,
        steer_limit_unwind=inputs.steer_limit_unwind,
      )
    )
    return TorqueV3SafetyResult(
      output_torque=round(shaping_result.output_torque, 12),
      authority_limited=authority_limited,
      authority_cap=authority_cap,
      shaping_result=shaping_result,
    )
