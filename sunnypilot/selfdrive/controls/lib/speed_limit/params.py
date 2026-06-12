"""Speed-limit parameter helpers."""

import math
from typing import Any

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD


def should_refresh_params(frame: int, period_s: float = PARAMS_UPDATE_PERIOD, dt: float = DT_MDL) -> bool:
  return frame % int(period_s / dt) == 0


def finite_float_param(value: Any, default: float = 0.0) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return default

  return parsed if math.isfinite(parsed) else default
