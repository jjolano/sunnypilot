from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_authority import AuthorityBand, AuthorityManager
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


def test_fault_demotes_authority_immediately():
  manager = AuthorityManager()
  manager.update(TorqueModelMode.learned, confidence=0.96, positive_coverage=0.7, negative_coverage=0.7,
                 reject_reason=EstimatorRejectReason.NONE)

  state = manager.update(TorqueModelMode.learned, confidence=0.7, positive_coverage=0.7, negative_coverage=0.7,
                         reject_reason=EstimatorRejectReason.SIGN_CONFLICT)

  assert state.band == AuthorityBand.limited
  assert state.scale == 0.45
