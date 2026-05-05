import numpy as np

from openpilot.common.params import UnknownKeyName
from openpilot.sunnypilot.selfdrive.controls.lib.long_learned_mass_drag import CDRAG_MAX, CDRAG_MIN, KF_MAX, KF_MIN

MAX_DRAG_COMPENSATION = 0.5


class LongControlExt:
  def __init__(self, longcontrol, CP, CP_SP):
    from openpilot.common.params import Params
    self.CP = CP
    self.CP_SP = CP_SP
    self._params = Params()
    try:
      self.mass_drag_enabled = self._params.get_bool("LongLearnedMassDragToggle")
    except UnknownKeyName:
      self.mass_drag_enabled = False
    try:
      self.mass_drag_apply_enabled = self._params.get_bool("LongLearnedMassDragApplyToggle")
    except UnknownKeyName:
      self.mass_drag_apply_enabled = False
    self.k_force = 1.0
    self.c_drag = 0.0

  def adjust_output(self, output_accel, CS, a_target):
    if not self.mass_drag_apply_enabled:
      return output_accel

    try:
      self.k_force = float(self._params.get("LongLearnedKForce", return_default=True))
      self.c_drag = float(self._params.get("LongLearnedCDrag", return_default=True))
    except (TypeError, ValueError, UnknownKeyName):
      return output_accel

    if KF_MIN <= self.k_force <= KF_MAX and CDRAG_MIN <= self.c_drag <= CDRAG_MAX and np.isfinite(self.k_force) and np.isfinite(self.c_drag):
      drag_term = np.clip(self.c_drag * CS.vEgo ** 2, 0.0, MAX_DRAG_COMPENSATION)
      output_accel = (output_accel + drag_term) / self.k_force

    return float(output_accel)
