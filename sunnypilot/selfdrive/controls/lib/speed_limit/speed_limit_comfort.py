from cereal import custom


LongitudinalPlanSourceSP = custom.LongitudinalPlanSP.LongitudinalPlanSource

SPEED_LIMIT_GENTLE_ACCEL_MAX = 0.4  # m/s^2


def apply_speed_limit_comfort_accel(v_ego: float, v_target: float, accel_coast: float, a_target: float) -> float:
  if v_target <= 0.:
    return a_target

  if v_ego > v_target:
    return max(a_target, accel_coast)

  if v_target > v_ego:
    return min(a_target, SPEED_LIMIT_GENTLE_ACCEL_MAX)

  return a_target


def should_apply_speed_limit_comfort_accel(reset_state: bool, force_slow_decel: bool, e2e_active: bool,
                                           has_lead: bool, should_stop: bool,
                                           source: custom.LongitudinalPlanSP.LongitudinalPlanSource) -> bool:
  return bool(
    not reset_state
    and not force_slow_decel
    and not e2e_active
    and not has_lead
    and not should_stop
    and source == LongitudinalPlanSourceSP.speedLimitAssist
  )
