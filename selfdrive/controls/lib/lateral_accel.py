import math

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY


def roll_lateral_accel(roll: float) -> float:
  return math.sin(float(roll)) * ACCELERATION_DUE_TO_GRAVITY


def lateral_accel_from_curvature(v_ego: float, curvature: float, roll: float = 0.0) -> float:
  return float(curvature) * float(v_ego) ** 2 - roll_lateral_accel(roll)


def lateral_accel_from_steering_angle(v_ego: float, steering_angle_rad: float, vehicle_model, roll: float = 0.0) -> float:
  curvature = vehicle_model.calc_curvature(float(steering_angle_rad), float(v_ego), float(roll))
  return lateral_accel_from_curvature(v_ego, curvature, roll)
