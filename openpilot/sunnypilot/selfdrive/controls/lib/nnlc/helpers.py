"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
# Phase-2 compatibility facade. The implementation now lives in
# ``sunnypilot.custom.lateral.nnlc_helpers``.
from openpilot.sunnypilot.custom.lateral.nnlc_helpers import (  # noqa: F401
  MOCK_MODEL_PATH,
  TORQUE_NN_MODEL_PATH,
  TORQUE_NN_MODEL_SUBSTITUTE_PATH,
  get_nn_model_path,
  similarity,
)
