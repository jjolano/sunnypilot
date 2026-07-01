"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Global-shutter roll/pitch electronic image stabilization core for modeld_v2.

Integrates raw gyroscope samples into a causal actual/virtual orientation pair,
computes a high-pass roll/pitch correction, and exposes it as a rotation matrix.
Row-wise rolling shutter is intentionally not implemented.
"""
import math

import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.model import calib_from_medmodel, calib_from_sbigmodel
from openpilot.common.transformations.orientation import quat_from_rot, rot_from_quat

CAMERA_STABILIZATION_PARAM = "CameraStabilizationMode"
_CAMERA_STABILIZATION_MODES = ("off", "shadow", "apply")
_GYRO_FIELDS = ("gyro", "gyroUncalibrated")

_MAX_CORRECTION_DEG = 0.25
_TAU_S = 0.4
_MAX_BUFFER_AGE_NS = 2_000_000_000
_MAX_GYRO_GAP_NS = 200_000_000
_MAX_EXTRAPOLATE_NS = 20_000_000


def sanitize_camera_stabilization_mode(value):
  """Return a canonical stabilization mode; unknown/missing values fail-closed to off."""
  if isinstance(value, bytes):
    value = value.decode("utf-8", errors="ignore")
  text = str(value).strip().lower() if value is not None else "off"
  return text if text in _CAMERA_STABILIZATION_MODES else "off"


def camera_stabilization_blocks_camera_odometry(value, calibration_ready=True):
  """Apply mode can make camera odometry virtual-camera motion."""
  return sanitize_camera_stabilization_mode(value) == "apply" and calibration_ready


def skew(v):
  """3x3 skew-symmetric matrix from a length-3 vector."""
  v = np.asarray(v, dtype=np.float64)
  return np.array([
    [0.0, -v[2], v[1]],
    [v[2], 0.0, -v[0]],
    [-v[1], v[0], 0.0],
  ], dtype=np.float64)


def rot_from_rotvec(rotvec):
  """Rotation matrix from an angle-axis (Rodrigues) vector."""
  rotvec = np.asarray(rotvec, dtype=np.float64)
  angle = float(np.linalg.norm(rotvec))
  if angle < 1e-8:
    return np.eye(3, dtype=np.float64)
  axis = rotvec / angle
  K = skew(axis)
  return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotvec_from_rot(R):
  """Angle-axis vector representing a rotation matrix."""
  R = np.asarray(R, dtype=np.float64)
  trace = float(np.trace(R))
  cos_angle = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
  angle = math.acos(cos_angle)
  if angle < 1e-6:
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
  denom = 2.0 * math.sin(angle)
  if abs(denom) < 1e-10:
    return np.zeros(3, dtype=np.float64)
  return angle / denom * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)


def _nearest_rot(R):
  """Re-project a matrix onto SO(3) to counter integration drift."""
  u, _, vt = np.linalg.svd(R)
  result = u @ vt
  if float(np.linalg.det(result)) < 0.0:
    u[:, -1] *= -1.0
    result = u @ vt
  return result


def _slerp_quat(q0, q1, t):
  """Spherical linear interpolation between two unit quaternions."""
  q0 = np.asarray(q0, dtype=np.float64)
  q1 = np.asarray(q1, dtype=np.float64)
  dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  if dot > 0.9995:
    q = q0 + t * (q1 - q0)
    n = np.linalg.norm(q)
    return q if n < 1e-10 else q / n
  theta_0 = math.acos(dot)
  theta = theta_0 * t
  q2 = q1 - dot * q0
  q2 = q2 / np.linalg.norm(q2)
  return math.cos(theta) * q0 + math.sin(theta) * q2


def warp_matrix_from_device_from_calib_rot(device_from_calib_rot, intrinsics, bigmodel_frame=False):
  """Build a model warp matrix from a device-from-calib rotation matrix.

  Mirrors common.transformations.model.get_warp_matrix but accepts a rotation
  matrix instead of Euler angles.
  """
  calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
  camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib_rot
  return camera_from_calib @ calib_from_model


class _State:
  __slots__ = ("t_ns", "actual_R", "virtual_R")

  def __init__(self, t_ns, actual_R, virtual_R):
    self.t_ns = int(t_ns)
    self.actual_R = actual_R.astype(np.float64, copy=True)
    self.virtual_R = virtual_R.astype(np.float64, copy=True)


class CameraStabilizer:
  def __init__(self):
    self.last_mode = "off"
    self.last_reason: str | None = None
    self.last_frame_time_s = 0.0
    self.last_dt_s = 0.0
    self.last_correction = np.zeros(3, dtype=np.float64)
    self.last_clipped = np.zeros(3, dtype=bool)

    self._actual_R = np.eye(3, dtype=np.float64)
    self._virtual_R = np.eye(3, dtype=np.float64)
    self._buffer: list[_State] = []
    self._last_gyro_t_ns: int | None = None
    self._prev_w = np.zeros(3, dtype=np.float64)
    self._correction_R = np.eye(3, dtype=np.float64)
    self._valid = False

  @staticmethod
  def _read_gyro_vec(gyro_msg):
    """Return a finite 3-axis gyro vector in rad/s, or None if invalid."""
    if gyro_msg is None:
      return None
    for field in _GYRO_FIELDS:
      try:
        event = getattr(gyro_msg, field, None)
      except Exception:
        continue
      if event is None:
        continue
      v = getattr(event, "v", None)
      if v is None or len(v) != 3:
        continue
      try:
        vec = np.asarray(v, dtype=np.float64)
      except (TypeError, ValueError):
        continue
      if np.all(np.isfinite(vec)):
        return vec
    return None

  def _reset_state(self, t_ns, w):
    """Reset orientations and start a fresh track at the given sample."""
    self._actual_R = np.eye(3, dtype=np.float64)
    self._virtual_R = np.eye(3, dtype=np.float64)
    self._buffer = [_State(t_ns, self._actual_R, self._virtual_R)]
    self._last_gyro_t_ns = int(t_ns)
    self._prev_w = np.asarray(w, dtype=np.float64)

  def _clear_gyro_state(self):
    self._actual_R = np.eye(3, dtype=np.float64)
    self._virtual_R = np.eye(3, dtype=np.float64)
    self._buffer.clear()
    self._last_gyro_t_ns = None
    self._prev_w = np.zeros(3, dtype=np.float64)

  def _trim_buffer(self, t_ns):
    cutoff = int(t_ns) - _MAX_BUFFER_AGE_NS
    while self._buffer and self._buffer[0].t_ns < cutoff:
      self._buffer.pop(0)

  def _append_state(self, t_ns):
    self._buffer.append(_State(t_ns, self._actual_R, self._virtual_R))
    self._trim_buffer(t_ns)

  def ingest_gyro(self, gyro_msg, gyro_valid: bool, gyro_log_mono_time_ns: int):
    """Drain a raw gyro event into the timestamped orientation buffer."""
    try:
      sample_ns = int(getattr(gyro_msg, "timestamp", 0) or gyro_log_mono_time_ns)
    except (TypeError, ValueError, OverflowError):
      self._valid = False
      self.last_reason = "gyro_timestamp_invalid"
      self._clear_gyro_state()
      return

    if not gyro_valid:
      self._valid = False
      self.last_reason = "gyro_invalid"
      self._clear_gyro_state()
      return

    w = self._read_gyro_vec(gyro_msg)
    if w is None:
      self._valid = False
      self.last_reason = "gyro_vector_invalid"
      self._clear_gyro_state()
      return

    if self._last_gyro_t_ns is None or sample_ns <= self._last_gyro_t_ns or (sample_ns - self._last_gyro_t_ns) > _MAX_GYRO_GAP_NS:
      self._reset_state(sample_ns, w)
      return

    dt = (sample_ns - self._last_gyro_t_ns) * 1e-9
    midpoint = 0.5 * (self._prev_w + w)
    self._actual_R = _nearest_rot(self._actual_R @ rot_from_rotvec(midpoint * dt))

    error_vec = rotvec_from_rot(self._virtual_R.T @ self._actual_R)
    alpha = 1.0 - math.exp(-dt / _TAU_S)
    self._virtual_R = _nearest_rot(self._virtual_R @ rot_from_rotvec(alpha * error_vec))

    self._last_gyro_t_ns = sample_ns
    self._prev_w = w
    self._append_state(sample_ns)

  def _orientation_at(self, t_ns: int):
    """Return (actual_R, virtual_R) for an exact or interpolated time, or None."""
    if not self._buffer:
      return None
    if t_ns >= self._buffer[-1].t_ns:
      dt_ns = t_ns - self._buffer[-1].t_ns
      return (self._buffer[-1].actual_R, self._buffer[-1].virtual_R) if dt_ns <= _MAX_EXTRAPOLATE_NS else None
    if t_ns <= self._buffer[0].t_ns:
      dt_ns = self._buffer[0].t_ns - t_ns
      return (self._buffer[0].actual_R, self._buffer[0].virtual_R) if dt_ns <= _MAX_EXTRAPOLATE_NS else None

    # binary search bracket
    lo, hi = 0, len(self._buffer) - 1
    while hi - lo > 1:
      mid = (lo + hi) // 2
      if self._buffer[mid].t_ns <= t_ns:
        lo = mid
      else:
        hi = mid
    s0 = self._buffer[lo]
    s1 = self._buffer[hi]
    span_ns = s1.t_ns - s0.t_ns
    if span_ns <= 0:
      return s0.actual_R, s0.virtual_R
    t = (t_ns - s0.t_ns) / span_ns

    q_actual_0 = quat_from_rot(s0.actual_R)
    q_actual_1 = quat_from_rot(s1.actual_R)
    q_virtual_0 = quat_from_rot(s0.virtual_R)
    q_virtual_1 = quat_from_rot(s1.virtual_R)
    actual_R = rot_from_quat(_slerp_quat(q_actual_0, q_actual_1, t))
    virtual_R = rot_from_quat(_slerp_quat(q_virtual_0, q_virtual_1, t))
    return actual_R, virtual_R

  def _invalidate(self, reason: str):
    self._valid = False
    self.last_reason = reason
    self.last_dt_s = 0.0
    self.last_frame_time_s = 0.0
    self.last_correction = np.zeros(3, dtype=np.float64)
    self.last_clipped = np.zeros(3, dtype=bool)
    self._correction_R = np.eye(3, dtype=np.float64)

  def update(self, mode_or_current, frame_timestamp_sof_ns: int, frame_timestamp_eof_ns: int):
    """Compute the roll/pitch correction for a frame centered at the mid-exposure time."""
    mode = sanitize_camera_stabilization_mode(mode_or_current)
    self.last_mode = mode
    if mode == "off":
      self._invalidate("mode_off")
      return

    self._invalidate("unknown_state")

    if not self._buffer:
      self._invalidate("no_gyro_data")
      return

    try:
      frame_t_ns = int((int(frame_timestamp_sof_ns) + int(frame_timestamp_eof_ns)) // 2)
    except (TypeError, ValueError, OverflowError):
      self._invalidate("frame_timestamp_invalid")
      return

    self.last_frame_time_s = frame_t_ns * 1e-9
    orientations = self._orientation_at(frame_t_ns)
    if orientations is None:
      self._invalidate("gyro_unavailable_or_stale")
      return

    actual_R, virtual_R = orientations
    # actual_R/virtual_R are world-from-device rotations integrated from body-rate gyro.
    # Model warping needs device-from-calib, so map the virtual calibration frame into
    # the actual captured device frame.
    correction_R = actual_R.T @ virtual_R
    correction_vec = rotvec_from_rot(correction_R)

    max_rad = math.radians(_MAX_CORRECTION_DEG)
    clipped = np.abs(correction_vec[:2]) > max_rad
    correction_vec[:2] = np.clip(correction_vec[:2], -max_rad, max_rad)
    correction_vec[2] = 0.0

    self.last_correction = correction_vec
    self.last_clipped = np.zeros(3, dtype=bool)
    self.last_clipped[:2] = clipped
    self._correction_R = rot_from_rotvec(correction_vec)
    self._valid = True
    self.last_reason = "ok"

  @property
  def correction_valid(self):
    return self._valid

  def correction_for_model_rot(self):
    """Return the correction rotation matrix to left-multiply device_from_calib."""
    if self.last_mode == "apply" and self._valid:
      return np.copy(self._correction_R)
    return np.eye(3, dtype=np.float64)

  def correction_for_model(self):
    """Return the roll/pitch correction vector (yaw zeroed) for diagnostics/tests."""
    if self.last_mode == "apply" and self._valid:
      return np.copy(self.last_correction)
    return np.zeros(3, dtype=np.float64)
