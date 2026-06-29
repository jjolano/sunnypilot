"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
# Phase-2 compatibility facade. The implementation now lives in
# ``sunnypilot.custom.lateral.nnlc_response``.
from openpilot.sunnypilot.custom.lateral.nnlc_response import (  # noqa: F401
  LOW_SPEED_X,
  LOW_SPEED_Y,
  NeuralNetworkLateralControl,
  _is_real_model_path,
  roll_pitch_adjust,
)
