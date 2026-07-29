"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.custom.longitudinal.curve_evidence.vision_controller import (
  SmartCruiseControlVision,
  VisionState,
  ACTIVE_STATES,
  ENABLED_STATES,
  _ENTERING_PRED_LAT_ACC_TH,
  _ABORT_ENTERING_PRED_LAT_ACC_TH,
  _TURNING_LAT_ACC_TH,
  _CURRENT_LAT_ACC_BLEED_TH,
  _LEAVING_LAT_ACC_TH,
  _FINISH_LAT_ACC_TH,
  _A_LAT_REG_MAX,
  _NO_OVERSHOOT_TIME_HORIZON,
  _ENTERING_SMOOTH_DECEL_V,
  _ENTERING_SMOOTH_DECEL_BP,
  _PRE_ENTRY_PRED_LAT_ACC_TH,
  _PRE_ENTRY_MIN_FRAMES,
  _PRE_ENTRY_GENTLE_DECEL,
  _TURNING_ACC_V,
  _TURNING_ACC_BP,
  _LEAVING_ACC,
  _EPS,
)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

__all__ = [
  "SmartCruiseControlVision",
  "VisionState",
  "ACTIVE_STATES",
  "ENABLED_STATES",
  "_ENTERING_PRED_LAT_ACC_TH",
  "_ABORT_ENTERING_PRED_LAT_ACC_TH",
  "_TURNING_LAT_ACC_TH",
  "_CURRENT_LAT_ACC_BLEED_TH",
  "_LEAVING_LAT_ACC_TH",
  "_FINISH_LAT_ACC_TH",
  "_A_LAT_REG_MAX",
  "_NO_OVERSHOOT_TIME_HORIZON",
  "_ENTERING_SMOOTH_DECEL_V",
  "_ENTERING_SMOOTH_DECEL_BP",
  "_PRE_ENTRY_PRED_LAT_ACC_TH",
  "_PRE_ENTRY_MIN_FRAMES",
  "_PRE_ENTRY_GENTLE_DECEL",
  "_TURNING_ACC_V",
  "_TURNING_ACC_BP",
  "_LEAVING_ACC",
  "_EPS",
  "MIN_V",
]
