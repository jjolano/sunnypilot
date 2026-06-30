"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Focused helper tests for the global-shutter gyro EIS core. No process replay.
"""
import numpy as np
import pytest

import cereal.messaging as messaging
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, get_warp_matrix
from openpilot.sunnypilot.modeld_v2.camera_stabilization import (
  CameraStabilizer,
  CAMERA_STABILIZATION_PARAM,
  rot_from_rotvec,
  rotvec_from_rot,
  sanitize_camera_stabilization_mode,
  skew,
  warp_matrix_from_device_from_calib_rot,
)


class MockStruct:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


def make_gyro_mock(vec, field="gyro", timestamp=None):
  kwargs = {field: MockStruct(v=vec)}
  if timestamp is not None:
    kwargs["timestamp"] = timestamp
  return MockStruct(**kwargs)


def feed_constant_gyro(stabilizer, rate_vec, t_start_ns, t_end_ns, step_ns=10_000_000):
  """Ingest a constant-rate gyro signal over [t_start_ns, t_end_ns]."""
  t = int(t_start_ns)
  end = int(t_end_ns)
  rate = np.asarray(rate_vec, dtype=np.float64)
  while t <= end:
    msg = make_gyro_mock(rate.tolist(), timestamp=t)
    stabilizer.ingest_gyro(msg, True, t)
    t += int(step_ns)


class TestSanitizeCameraStabilizationMode:
  def test_camera_stabilization_param_constant(self):
    assert CAMERA_STABILIZATION_PARAM == "CameraStabilizationMode"

  def test_sanitizer_defaults_to_off(self):
    assert sanitize_camera_stabilization_mode(None) == "off"
    assert sanitize_camera_stabilization_mode("") == "off"
    assert sanitize_camera_stabilization_mode("bad") == "off"
    assert sanitize_camera_stabilization_mode(b"bad") == "off"

  def test_sanitizer_accepts_canonical_and_cased_values(self):
    assert sanitize_camera_stabilization_mode("off") == "off"
    assert sanitize_camera_stabilization_mode("OFF") == "off"
    assert sanitize_camera_stabilization_mode(b"off") == "off"
    assert sanitize_camera_stabilization_mode("shadow") == "shadow"
    assert sanitize_camera_stabilization_mode("SHADOW") == "shadow"
    assert sanitize_camera_stabilization_mode("apply") == "apply"
    assert sanitize_camera_stabilization_mode(" Apply ") == "apply"


class TestRodriguesHelpers:
  def test_skew_matrix(self):
    v = np.array([1.0, 2.0, 3.0])
    K = skew(v)
    np.testing.assert_array_equal(K.T, -K)
    np.testing.assert_array_equal(K @ v, np.zeros(3))

  def test_rotvec_roundtrip(self):
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
      axis = np.array(axis, dtype=np.float64)
      axis = axis / np.linalg.norm(axis)
      for angle in (0.01, 0.5, np.pi - 0.01):
        v = axis * angle
        R = rot_from_rotvec(v)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)
        v2 = rotvec_from_rot(R)
        R2 = rot_from_rotvec(v2)
        np.testing.assert_allclose(R, R2, atol=1e-8)


class TestIntegration:
  def test_constant_roll_rate_integrates_orientation(self):
    stabilizer = CameraStabilizer()
    feed_constant_gyro(stabilizer, [0.5, 0.0, 0.0], 0, 500_000_000)
    assert len(stabilizer._buffer) >= 10
    actual_vec = rotvec_from_rot(stabilizer._buffer[-1].actual_R)
    assert actual_vec[0] > 0.2
    assert abs(actual_vec[1]) < 0.05
    assert abs(actual_vec[2]) < 0.05

  def test_virtual_lags_actual_under_constant_rate(self):
    stabilizer = CameraStabilizer()
    feed_constant_gyro(stabilizer, [0.05, 0.0, 0.0], 0, 500_000_000)
    state = stabilizer._buffer[-1]
    actual_vec = rotvec_from_rot(state.actual_R)
    virtual_vec = rotvec_from_rot(state.virtual_R)
    # virtual orientation should be slightly behind actual because of the low-pass filter
    assert virtual_vec[0] < actual_vec[0]

  def test_ingest_rejects_invalid_and_resets(self):
    stabilizer = CameraStabilizer()
    feed_constant_gyro(stabilizer, [0.5, 0.0, 0.0], 0, 100_000_000)
    assert len(stabilizer._buffer) >= 5

    stabilizer.ingest_gyro(make_gyro_mock([0.1, 0.0, 0.0], timestamp=50_000_000), True, 50_000_000)
    assert len(stabilizer._buffer) == 1

    stabilizer.ingest_gyro(make_gyro_mock([0.1, 0.0, 0.0], timestamp=400_000_000), True, 400_000_000)
    assert len(stabilizer._buffer) == 1

  def test_buffer_trims_old_states(self):
    stabilizer = CameraStabilizer()
    feed_constant_gyro(stabilizer, [0.1, 0.0, 0.0], 0, 2_500_000_000, step_ns=100_000_000)
    span_ns = stabilizer._buffer[-1].t_ns - stabilizer._buffer[0].t_ns
    assert span_ns <= 2_100_000_000


class TestCorrection:
  def setup_method(self):
    self.stabilizer = CameraStabilizer()

  def _warmup(self, rate=0.5, duration_ns=500_000_000):
    feed_constant_gyro(self.stabilizer, [rate, 0.0, 0.0], 0, duration_ns)

  def test_off_fails_closed(self):
    self._warmup()
    self.stabilizer.update("off", 200_000_000, 300_000_000)
    assert not self.stabilizer.correction_valid
    assert self.stabilizer.last_reason == "mode_off"
    np.testing.assert_allclose(self.stabilizer.correction_for_model_rot(), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(self.stabilizer.correction_for_model(), np.zeros(3), atol=1e-10)

  def test_shadow_valid_but_no_model_correction(self):
    self._warmup()
    self.stabilizer.update("shadow", 200_000_000, 300_000_000)
    assert self.stabilizer.correction_valid
    assert self.stabilizer.last_reason == "ok"
    np.testing.assert_allclose(self.stabilizer.correction_for_model_rot(), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(self.stabilizer.correction_for_model(), np.zeros(3), atol=1e-10)

  def test_apply_returns_non_identity_correction(self):
    self._warmup()
    self.stabilizer.update("apply", 200_000_000, 300_000_000)
    assert self.stabilizer.correction_valid
    R = self.stabilizer.correction_for_model_rot()
    assert np.trace(R) < 3.0 - 1e-5
    v = self.stabilizer.correction_for_model()
    assert np.linalg.norm(v[:2]) > 1e-4
    assert v[0] < 0.0
    assert abs(v[2]) < 1e-8

  def test_correction_yaw_is_zeroed(self):
    # constant rate with a yaw component; yaw must be removed from the correction
    feed_constant_gyro(self.stabilizer, [0.3, 0.0, 0.4], 0, 500_000_000)
    self.stabilizer.update("apply", 200_000_000, 300_000_000)
    assert self.stabilizer.correction_valid
    v = self.stabilizer.correction_for_model()
    assert abs(v[2]) < 1e-6

  def test_no_data_fails_closed(self):
    self.stabilizer.update("apply", 100_000_000, 200_000_000)
    assert not self.stabilizer.correction_valid
    assert self.stabilizer.last_reason == "no_gyro_data"

  def test_invalid_gyro_clears_buffer_and_fails_closed(self):
    self._warmup()
    assert self.stabilizer._buffer
    self.stabilizer.ingest_gyro(make_gyro_mock([0.1, 0.0, 0.0], timestamp=600_000_000), False, 600_000_000)
    assert not self.stabilizer._buffer
    self.stabilizer.update("apply", 200_000_000, 300_000_000)
    assert not self.stabilizer.correction_valid
    assert self.stabilizer.last_reason == "no_gyro_data"

  def test_stale_data_fails_closed(self):
    feed_constant_gyro(self.stabilizer, [0.5, 0.0, 0.0], 0, 100_000_000)
    self.stabilizer.update("apply", 400_000_000, 500_000_000)
    assert not self.stabilizer.correction_valid
    assert "stale" in self.stabilizer.last_reason

  def test_clamp_roll_pitch(self):
    # high rate causes large actual-vs-virtual lag; roll component is clamped to the small-angle limit.
    feed_constant_gyro(self.stabilizer, [2.0, 0.0, 0.0], 0, 800_000_000)
    self.stabilizer.update("apply", 300_000_000, 400_000_000)
    assert self.stabilizer.correction_valid
    v = self.stabilizer.correction_for_model()
    max_rad = np.deg2rad(0.25)
    assert abs(v[0]) <= max_rad + 1e-12
    assert abs(v[1]) <= max_rad + 1e-12
    assert abs(v[0]) > max_rad * 0.5
    assert abs(v[2]) < 1e-8
    assert self.stabilizer.last_clipped[0]
    assert not self.stabilizer.last_clipped[2]

  def test_correction_for_model_rot_returns_copy(self):
    self._warmup()
    self.stabilizer.update("apply", 200_000_000, 300_000_000)
    a = self.stabilizer.correction_for_model_rot()
    b = self.stabilizer.correction_for_model_rot()
    assert a is not b
    np.testing.assert_array_equal(a, b)


class TestWarpMatrix:
  def test_warp_from_identity_matches_get_warp_matrix_zeros(self):
    intrinsics = DEVICE_CAMERAS[("tici", "unknown")].fcam.intrinsics
    W_default = get_warp_matrix(np.zeros(3, dtype=np.float32), intrinsics, False)
    W_new = warp_matrix_from_device_from_calib_rot(np.eye(3), intrinsics, False)
    np.testing.assert_allclose(W_default, W_new, atol=1e-6)

  def _project_center(self, warp):
    pt = np.array([MEDMODEL_INPUT_SIZE[0] / 2, MEDMODEL_INPUT_SIZE[1] / 2, 1.0])
    p = warp @ pt
    return p[:2] / p[2]

  def test_positive_roll_pitch_correction_projection_direction(self):
    intrinsics = DEVICE_CAMERAS[("tici", "unknown")].fcam.intrinsics
    base_R = np.eye(3, dtype=np.float64)
    base_warp = warp_matrix_from_device_from_calib_rot(base_R, intrinsics, False)
    base = self._project_center(base_warp)

    roll_R = rot_from_rotvec([np.deg2rad(0.1), 0.0, 0.0]) @ base_R
    roll_warp = warp_matrix_from_device_from_calib_rot(roll_R, intrinsics, False)
    roll_center = self._project_center(roll_warp)

    pitch_R = rot_from_rotvec([0.0, np.deg2rad(0.1), 0.0]) @ base_R
    pitch_warp = warp_matrix_from_device_from_calib_rot(pitch_R, intrinsics, False)
    pitch_center = self._project_center(pitch_warp)

    assert roll_center[0] < base[0] - 1e-3
    assert abs(roll_center[1] - base[1]) < 0.01
    assert pitch_center[1] < base[1] - 1e-3
    assert abs(pitch_center[0] - base[0]) < 0.01


class TestModelDataV2SPTelemetry:
  def test_camera_stabilization_fields_roundtrip(self):
    stabilizer = CameraStabilizer()
    feed_constant_gyro(stabilizer, [0.05, 0.0, 0.0], 0, 500_000_000)
    stabilizer.update("apply", 200_000_000, 300_000_000)

    msg = messaging.new_message("modelDataV2SP")
    sp = msg.modelDataV2SP
    sp.cameraStabilizationMode = stabilizer.last_mode
    sp.cameraStabilizationApplied = True
    sp.cameraStabilizationMainReason = stabilizer.last_reason
    sp.cameraStabilizationExtraReason = "ok"
    sp.cameraStabilizationMainCorrectionRoll = float(stabilizer.last_correction[0])
    sp.cameraStabilizationMainCorrectionPitch = float(stabilizer.last_correction[1])
    sp.cameraStabilizationExtraCorrectionRoll = 0.0
    sp.cameraStabilizationExtraCorrectionPitch = 0.0
    sp.cameraStabilizationMainClipped = bool(stabilizer.last_clipped[0] or stabilizer.last_clipped[1])
    sp.cameraStabilizationExtraClipped = False

    with messaging.log.Event.from_bytes(msg.to_bytes()) as decoded:
      out = decoded.modelDataV2SP
      assert out.cameraStabilizationMode == "apply"
      assert out.cameraStabilizationApplied
      assert out.cameraStabilizationMainReason == "ok"
      assert out.cameraStabilizationExtraReason == "ok"
      assert abs(out.cameraStabilizationMainCorrectionRoll) > 1e-4
      assert abs(out.cameraStabilizationMainCorrectionPitch) < 1e-8
      assert out.cameraStabilizationMainClipped == bool(stabilizer.last_clipped[0] or stabilizer.last_clipped[1])
      assert not out.cameraStabilizationExtraClipped
