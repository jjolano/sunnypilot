from __future__ import annotations

# Compatibility re-export: UnderresponseSentinel now lives in the lateral parameter
# orchestrator.  The original implementation is preserved in
# ``underresponse_sentinel_v1.py`` for comparison and parity tests.
from openpilot.sunnypilot.custom.lateral.parameter_orchestrator import (  # noqa: F401
  BLOCK_ACTUAL_OPPOSING,
  BLOCK_CURVATURE_LIMITED,
  BLOCK_DESIRED_NOT_PERSISTENT,
  BLOCK_DESIRED_TOO_SMALL,
  BLOCK_ERROR_TOO_SMALL,
  BLOCK_FAST_CLOSING,
  BLOCK_INACTIVE,
  BLOCK_INVALID_INPUT,
  BLOCK_LOW_SPEED,
  BLOCK_NAMES,
  BLOCK_ROLL_TOO_HIGH,
  BLOCK_ROLL_UNSTABLE,
  BLOCK_SIGN_MISMATCH,
  BLOCK_STEER_LIMITED,
  BLOCK_STEERING_PRESSED,
  BLOCK_TORQUE_SATURATED,
  UnderresponseDebug,
  UnderresponseSentinel,
  write_underresponse_debug,
)
