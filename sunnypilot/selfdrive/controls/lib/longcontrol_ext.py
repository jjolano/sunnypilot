import numpy as np

from cereal import car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

LongCtrlState = car.CarControl.Actuators.LongControlState

ACCEL_BUCKET_BP = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
MIN_BUCKET_SAMPLES = 10
MAX_OFFSET = 0.5
FILTER_DECAY = 100


class ResponseCurveLearner:
  def __init__(self):
    self.buckets = {i: [] for i in range(len(ACCEL_BUCKET_BP))}
    self.cmd_buckets = {i: [] for i in range(len(ACCEL_BUCKET_BP))}
    self.filtered_offsets = {i: FirstOrderFilter(0.0, FILTER_DECAY, DT_CTRL, initialized=False) for i in range(len(ACCEL_BUCKET_BP))}
    self._valid = {i: False for i in range(len(ACCEL_BUCKET_BP))}

  def _bucket_idx(self, a_cmd):
    for i in range(len(ACCEL_BUCKET_BP) - 1):
      if ACCEL_BUCKET_BP[i] <= a_cmd < ACCEL_BUCKET_BP[i + 1]:
        return i
    return len(ACCEL_BUCKET_BP) - 1

  def update(self, a_cmd, a_ego):
    idx = self._bucket_idx(a_cmd)
    offset = a_ego - a_cmd
    if not np.isfinite(offset):
      return
    self.buckets[idx].append(offset)
    self.cmd_buckets[idx].append(a_cmd)
    if len(self.buckets[idx]) > 100:
      self.buckets[idx].pop(0)
      self.cmd_buckets[idx].pop(0)

    if len(self.buckets[idx]) >= MIN_BUCKET_SAMPLES:
      mean_offset = float(np.mean(self.buckets[idx]))
      mean_offset = np.clip(mean_offset, -MAX_OFFSET, MAX_OFFSET)
      self.filtered_offsets[idx].update(mean_offset)
      self._valid[idx] = True

  def is_bucket_valid(self, idx):
    return self._valid.get(idx, False)

  def lookup_offset(self, a_cmd):
    idx = self._bucket_idx(a_cmd)
    if self._valid[idx]:
      return float(self.filtered_offsets[idx].x)

    # Find nearest valid buckets for interpolation
    valid_indices = [i for i in range(len(ACCEL_BUCKET_BP)) if self._valid[i]]
    if not valid_indices:
      return 0.0
    if len(valid_indices) == 1:
      return float(self.filtered_offsets[valid_indices[0]].x)

    # Interpolate based on bucket mean a_cmd points
    centers = []
    offsets = []
    for i in valid_indices:
      centers.append(float(np.mean(self.cmd_buckets[i])))
      offsets.append(float(self.filtered_offsets[i].x))

    return float(np.interp(a_cmd, centers, offsets))

  def serialize(self):
    data = {}
    for i in range(len(ACCEL_BUCKET_BP)):
      if self._valid[i]:
        data[i] = float(self.filtered_offsets[i].x)
    return str(data)

  def deserialize(self, s):
    if not s:
      return
    try:
      import ast
      data = ast.literal_eval(s)
      for i, val in data.items():
        i = int(i)
        if 0 <= i < len(ACCEL_BUCKET_BP):
          self.filtered_offsets[i].x = float(val)
          self._valid[i] = True
    except Exception:
      pass


class LongControlExt:
  def __init__(self, longcontrol, CP, CP_SP):
    from openpilot.common.params import Params
    self.CP = CP
    self.CP_SP = CP_SP
    self._params = Params()
    self.mass_drag_enabled = self._params.get_bool("LongLearnedMassDragToggle")
    self.k_force = 1.0
    self.c_drag = 0.0
    self.response_curve_enabled = self._params.get_bool("LongLearnedResponseCurveToggle")
    self.response_learner = ResponseCurveLearner()
    # Restore cached offsets
    cached = self._params.get("LongLearnedResponseOffsets")
    if cached:
      self.response_learner.deserialize(cached)
    self._response_learn_count = 0

  def adjust_output(self, output_accel, CS, a_target):
    if not self.mass_drag_enabled:
      return output_accel

    self.k_force = float(self._params.get("LongLearnedKForce", return_default=True))
    self.c_drag = float(self._params.get("LongLearnedCDrag", return_default=True))

    if 0.5 <= self.k_force <= 2.0 and self.k_force != 1.0:
      drag_term = self.c_drag * CS.vEgo ** 2
      output_accel = (output_accel + drag_term) / self.k_force

    return float(output_accel)

  def get_response_offset(self, a_target):
    if not self.response_curve_enabled:
      return 0.0
    return self.response_learner.lookup_offset(a_target)

  def learn_response(self, a_target, a_ego, long_control_state, saturated):
    if not self.response_curve_enabled:
      return
    if long_control_state != LongCtrlState.pid:
      return
    if saturated:
      return
    if abs(a_ego - a_target) >= 1.0:
      return
    self.response_learner.update(a_target, a_ego)
    self._response_learn_count += 1
    if self._response_learn_count % 500 == 0:
      self._params.put_nonblocking("LongLearnedResponseOffsets", self.response_learner.serialize())
