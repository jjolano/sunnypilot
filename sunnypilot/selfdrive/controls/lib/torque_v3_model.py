import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Protocol

import numpy as np


MIN_LAT_ACCEL_FACTOR = 0.2
MAX_LAT_ACCEL_FACTOR = 8.0
MAX_LAT_ACCEL_OFFSET = 1.5
MAX_FRICTION = 1.0
MAX_RESIDUAL_TORQUE = 0.15
SYNTHETIC_LAT_ACCEL_FACTOR = 2.5
SYNTHETIC_LAT_ACCEL_OFFSET = 0.0
SYNTHETIC_FRICTION = 0.1


class TorqueModelMode(IntEnum):
  fallback = 0
  native = 1
  synthetic = 2
  learned = 3


class TorqueParamsLike(Protocol):
  latAccelFactor: float
  latAccelOffset: float
  friction: float


@dataclass
class TorqueModelParams:
  lat_accel_factor: float
  lat_accel_offset: float
  friction: float

  @property
  def latAccelFactor(self) -> float:
    return self.lat_accel_factor

  @latAccelFactor.setter
  def latAccelFactor(self, value: float) -> None:
    self.lat_accel_factor = value

  @property
  def latAccelOffset(self) -> float:
    return self.lat_accel_offset

  @latAccelOffset.setter
  def latAccelOffset(self, value: float) -> None:
    self.lat_accel_offset = value


TorqueFromLateralAccel = Callable[[float, TorqueParamsLike], float]
LateralAccelFromTorque = Callable[[float, TorqueParamsLike], float]


def _finite(value: float) -> bool:
  return math.isfinite(float(value))


def _valid_params(params: TorqueModelParams) -> bool:
  return (
    _finite(params.lat_accel_factor)
    and MIN_LAT_ACCEL_FACTOR <= params.lat_accel_factor <= MAX_LAT_ACCEL_FACTOR
    and _finite(params.lat_accel_offset)
    and abs(params.lat_accel_offset) <= MAX_LAT_ACCEL_OFFSET
    and _finite(params.friction)
    and 0.0 <= params.friction <= MAX_FRICTION
  )


class TorqueModelAdapter:
  def __init__(self, mode: TorqueModelMode, params: TorqueParamsLike, torque_from_lateral_accel: TorqueFromLateralAccel,
               lateral_accel_from_torque: LateralAccelFromTorque):
    self.mode = mode
    self.params = params
    self._torque_from_lateral_accel = torque_from_lateral_accel
    self._lateral_accel_from_torque = lateral_accel_from_torque
    self.confidence = 0.0
    self.residual_torque = 0.0

  @classmethod
  def synthetic(cls) -> "TorqueModelAdapter":
    params = TorqueModelParams(SYNTHETIC_LAT_ACCEL_FACTOR, SYNTHETIC_LAT_ACCEL_OFFSET, SYNTHETIC_FRICTION)
    return cls(TorqueModelMode.synthetic, params, cls._linear_torque_from_lateral_accel, cls._linear_lateral_accel_from_torque)

  @classmethod
  def native(cls, torque_params: TorqueParamsLike, torque_from_lateral_accel: TorqueFromLateralAccel,
             lateral_accel_from_torque: LateralAccelFromTorque) -> "TorqueModelAdapter":
    adapter = cls(TorqueModelMode.native, torque_params, torque_from_lateral_accel, lateral_accel_from_torque)
    adapter.confidence = 0.8
    return adapter

  @staticmethod
  def _linear_torque_from_lateral_accel(lateral_accel: float, params: TorqueParamsLike) -> float:
    return (lateral_accel - float(params.latAccelOffset)) / max(float(params.latAccelFactor), MIN_LAT_ACCEL_FACTOR)

  @staticmethod
  def _linear_lateral_accel_from_torque(torque: float, params: TorqueParamsLike) -> float:
    return torque * max(float(params.latAccelFactor), MIN_LAT_ACCEL_FACTOR) + float(params.latAccelOffset)

  def torque_from_lateral_accel(self, lateral_accel: float) -> float:
    if not _finite(lateral_accel):
      return 0.0
    torque = self._torque_from_lateral_accel(float(lateral_accel), self.params) + self.residual_torque
    return float(np.clip(torque, -1.0, 1.0)) if _finite(torque) else 0.0

  def lateral_accel_from_torque(self, torque: float) -> float:
    if not _finite(torque):
      return 0.0
    adjusted_torque = float(np.clip(torque, -1.0, 1.0)) - self.residual_torque
    lat_accel = self._lateral_accel_from_torque(adjusted_torque, self.params)
    return float(lat_accel) if _finite(lat_accel) else 0.0

  def update_learned_params(self, params: TorqueModelParams, confidence: float) -> bool:
    if not _valid_params(params) or not _finite(confidence):
      return False
    self.params = params
    self.mode = TorqueModelMode.learned
    self.confidence = float(np.clip(confidence, 0.0, 1.0))
    self._torque_from_lateral_accel = self._linear_torque_from_lateral_accel
    self._lateral_accel_from_torque = self._linear_lateral_accel_from_torque
    return True

  def set_residual(self, residual_torque: float) -> None:
    self.residual_torque = float(np.clip(residual_torque if _finite(residual_torque) else 0.0, -MAX_RESIDUAL_TORQUE, MAX_RESIDUAL_TORQUE))
