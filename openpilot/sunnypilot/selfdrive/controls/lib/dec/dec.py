"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
# Version = 2025-6-30

# Compatibility facade: the canonical DEC implementation now lives in
# openpilot.sunnypilot.custom.longitudinal.dec_controller. Keep this import path
# intact for existing callers and tests.
from openpilot.sunnypilot.custom.longitudinal.dec_controller import (
  DynamicExperimentalController,
  ModeTransitionManager,
  ModeType,
  SET_MODE_TIMEOUT,
  SmoothKalmanFilter,
  TRAJECTORY_SIZE,
)

__all__ = [
  "DynamicExperimentalController",
  "ModeTransitionManager",
  "ModeType",
  "SET_MODE_TIMEOUT",
  "SmoothKalmanFilter",
  "TRAJECTORY_SIZE",
]
