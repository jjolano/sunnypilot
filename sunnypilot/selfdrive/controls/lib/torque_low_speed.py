LOW_SPEED_PID_GAIN_FLOOR = 3.0


def low_speed_pid_gain_speed(v_ego: float, unwind_gain_floor: float | None = None) -> float:
  gain_floor = LOW_SPEED_PID_GAIN_FLOOR if unwind_gain_floor is None else unwind_gain_floor
  return max(float(v_ego), gain_floor)
