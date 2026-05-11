from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_authority import (
  AUTHORITY_RECOVERY_CLEAN_FRAMES,
  AUTHORITY_SCALE_RECOVERY_STEP,
  AuthorityBand,
  AuthorityManager,
)
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_estimator import EstimatorRejectReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelMode


def test_synthetic_starts_limited():
  manager = AuthorityManager()

  state = manager.update(TorqueModelMode.synthetic, confidence=0.0, positive_coverage=0.0, negative_coverage=0.0,
                         reject_reason=EstimatorRejectReason.NONE)

  assert state.band == AuthorityBand.limited
  assert state.scale == 0.45


def test_synthetic_reaches_full_with_bidirectional_convergence():
  manager = AuthorityManager()

  state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                         reject_reason=EstimatorRejectReason.NONE)

  assert state.band == AuthorityBand.full
  assert state.scale == 1.0


def test_native_starts_near_full_but_not_full():
  manager = AuthorityManager()

  state = manager.update(TorqueModelMode.native, confidence=0.8, positive_coverage=0.0, negative_coverage=0.0,
                         reject_reason=EstimatorRejectReason.NONE)

  assert state.band == AuthorityBand.near_full
  assert state.scale == 0.85


def test_clipping_rejects_do_not_demote_native_authority():
  manager = AuthorityManager()

  state = manager.update(TorqueModelMode.native, confidence=0.8, positive_coverage=0.0, negative_coverage=0.0,
                         reject_reason=EstimatorRejectReason.STEER_LIMITED | EstimatorRejectReason.SATURATED)

  assert state.band == AuthorityBand.near_full
  assert state.scale == 0.85
  assert not state.fallback_active


def test_sign_conflict_during_steer_limited_saturation_does_not_force_fallback():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)

  state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                         reject_reason=EstimatorRejectReason.SIGN_CONFLICT | EstimatorRejectReason.STEER_LIMITED |
                                       EstimatorRejectReason.SATURATED)

  assert state.band == AuthorityBand.full
  assert state.scale == 1.0
  assert not state.fallback_active


def test_sign_conflict_with_steer_limit_only_still_faults():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)

  state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                         reject_reason=EstimatorRejectReason.SIGN_CONFLICT | EstimatorRejectReason.STEER_LIMITED)

  assert state.band == AuthorityBand.limited
  assert state.scale == 0.45
  assert state.fallback_active


def test_fault_demotes_authority_immediately():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)

  state = manager.update(TorqueModelMode.learned, confidence=0.7, positive_coverage=0.7, negative_coverage=0.7,
                         reject_reason=EstimatorRejectReason.SIGN_CONFLICT)

  assert state.band == AuthorityBand.limited
  assert state.scale == 0.45


def test_unclipped_residual_and_stale_faults_demote_authority_immediately():
  for reject_reason in (EstimatorRejectReason.RESIDUAL_SPIKE, EstimatorRejectReason.STALE_MODEL):
    manager = AuthorityManager()
    _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                       reject_reason=EstimatorRejectReason.NONE)

    state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                           reject_reason=reject_reason)

    assert state.band == AuthorityBand.limited
    assert state.scale == 0.45
    assert state.fallback_active


def test_residual_spike_during_clipping_does_not_force_fallback():
  manager = AuthorityManager()

  state = manager.update(TorqueModelMode.native, confidence=0.8, positive_coverage=0.0, negative_coverage=0.0,
                         reject_reason=EstimatorRejectReason.RESIDUAL_SPIKE | EstimatorRejectReason.STEER_LIMITED)

  assert state.band == AuthorityBand.near_full
  assert state.scale == 0.85
  assert not state.fallback_active


def test_fault_recovery_requires_consecutive_clean_frames():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)

  fault_state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                               reject_reason=EstimatorRejectReason.SIGN_CONFLICT)
  clean_state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                               reject_reason=EstimatorRejectReason.NONE)

  assert fault_state.band == AuthorityBand.limited
  assert fault_state.scale == 0.45
  assert fault_state.fallback_active
  assert clean_state.band == AuthorityBand.limited
  assert clean_state.scale == 0.45
  assert clean_state.fallback_active


def test_alternating_fault_and_clean_frames_do_not_chatter_to_full_authority():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)

  for _ in range(AUTHORITY_RECOVERY_CLEAN_FRAMES * 3):
    fault_state = manager.update(TorqueModelMode.learned, confidence=0.0, positive_coverage=0.7, negative_coverage=0.7,
                                 reject_reason=EstimatorRejectReason.RESIDUAL_SPIKE)
    clean_state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                                 reject_reason=EstimatorRejectReason.NONE)

    assert fault_state.band == AuthorityBand.limited
    assert fault_state.scale == 0.45
    assert fault_state.fallback_active
    assert clean_state.band == AuthorityBand.limited
    assert clean_state.scale == 0.45
    assert clean_state.fallback_active


def test_fault_recovery_ramps_upward_after_clean_frame_gate():
  manager = AuthorityManager()
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.NONE)
  _ = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                     reject_reason=EstimatorRejectReason.STALE_MODEL)

  scales: list[float] = []
  state = manager.current_state()
  for _ in range(AUTHORITY_RECOVERY_CLEAN_FRAMES + 40):
    state = manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                           reject_reason=EstimatorRejectReason.NONE)
    scales.append(state.scale)

  assert scales[0] == 0.45
  assert scales[AUTHORITY_RECOVERY_CLEAN_FRAMES - 1] == 0.45
  assert scales[AUTHORITY_RECOVERY_CLEAN_FRAMES] == 0.45 + AUTHORITY_SCALE_RECOVERY_STEP
  assert all(next_scale >= scale for scale, next_scale in zip(scales, scales[1:]))
  assert all(
    next_scale - scale <= AUTHORITY_SCALE_RECOVERY_STEP + 1e-9
    for scale, next_scale in zip(scales[AUTHORITY_RECOVERY_CLEAN_FRAMES:], scales[AUTHORITY_RECOVERY_CLEAN_FRAMES + 1:])
  )
  assert scales[-1] == 1.0
  assert state.band == AuthorityBand.full
  assert not state.fallback_active
