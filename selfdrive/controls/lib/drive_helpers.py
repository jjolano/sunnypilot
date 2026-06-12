import numpy as np
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.lateral_accel import roll_lateral_accel

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MAX_VEL_ERR = 5.0  # m/s
MIN_STABLE_DELAY = 0.3

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2
MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL = 5.0  # m/s^2
LATERAL_ACCEL_DRIVER_GAS_DECAY_SECONDS = 1.25


def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val


def update_lateral_accel_limit(current_limit, manual_gas_override, lat_active, brake_pressed, steering_pressed,
                               dt=DT_CTRL, default_lateral_accel_limited=False):
  if not lat_active or brake_pressed or steering_pressed or not np.isfinite(current_limit):
    return MAX_LATERAL_ACCEL_NO_ROLL
  if manual_gas_override or default_lateral_accel_limited:
    return MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL

  decay_rate = (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / LATERAL_ACCEL_DRIVER_GAS_DECAY_SECONDS
  return float(np.clip(current_limit - decay_rate * max(dt, 0.0),
                       MAX_LATERAL_ACCEL_NO_ROLL,
                       MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL))


def should_latch_lateral_accel_burst(default_lateral_accel_limited, lat_active, brake_pressed, steering_pressed,
                                     manual_gas_override=False):
  return bool(default_lateral_accel_limited and lat_active and not brake_pressed and not steering_pressed and not manual_gas_override)


def _roll_compensation(roll, accurate_lateral_accel=False):
  return roll_lateral_accel(roll) if accurate_lateral_accel else roll * ACCELERATION_DUE_TO_GRAVITY


def is_default_lateral_accel_limited(v_ego, curvature, roll, accurate_lateral_accel=False):
  v_ego = max(v_ego, MIN_SPEED)
  roll_compensation = _roll_compensation(roll, accurate_lateral_accel)
  max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  min_lat_accel = -MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  lat_accel = curvature * v_ego ** 2
  return lat_accel < min_lat_accel or lat_accel > max_lat_accel


def clip_curvature(v_ego, prev_curvature, new_curvature, roll, lateral_accel_limit=MAX_LATERAL_ACCEL_NO_ROLL,
                   accurate_lateral_accel=False) -> tuple[float, bool, bool]:
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)
  default_lateral_accel_limited = is_default_lateral_accel_limited(v_ego, new_curvature, roll, accurate_lateral_accel)

  if not np.isfinite(lateral_accel_limit):
    lateral_accel_limit = MAX_LATERAL_ACCEL_NO_ROLL
  lateral_accel_limit = float(np.clip(lateral_accel_limit, 0.0, MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL))

  roll_compensation = _roll_compensation(roll, accurate_lateral_accel)
  max_lat_accel = lateral_accel_limit + roll_compensation
  min_lat_accel = -lateral_accel_limit + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv, bool(default_lateral_accel_limited)


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL, vEgoStopping=0.3):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    v_now = 0.0
    v_target = 0.0
    a_target = 0.0
  should_stop = (v_now < vEgoStopping and a_target < 0.1)
  return a_target, should_stop

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
