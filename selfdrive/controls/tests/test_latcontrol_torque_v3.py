from cereal import log


def test_v3_adaptive_torque_schema_fields_exist():
  torque_log = log.ControlsState.LateralTorqueState.new_message()
  adaptive_log = torque_log.init('adaptiveTorqueState')

  adaptive_log.modelMode = 2
  adaptive_log.modelConfidence = 0.5
  adaptive_log.authorityBand = 1
  adaptive_log.authorityScale = 0.65
  adaptive_log.fallbackActive = False
  adaptive_log.learnedLatAccelFactor = 2.5
  adaptive_log.learnedFriction = 0.1
  adaptive_log.learnedLatAccelOffset = 0.0
  adaptive_log.learnedResponseDelay = 0.2
  adaptive_log.residualError = 0.05
  adaptive_log.sampleAccepted = True
  adaptive_log.sampleRejectReason = 0

  assert adaptive_log.modelMode == 2
  assert adaptive_log.authorityBand == 1
  assert adaptive_log.sampleAccepted
