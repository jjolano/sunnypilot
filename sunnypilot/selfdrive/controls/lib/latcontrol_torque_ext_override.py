"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.custom.lateral.parameter_orchestrator import ParameterOrchestrator


class LatControlTorqueExtOverride(ParameterOrchestrator):
  """Compatibility facade for the legacy torque-parameter override policy.

  The implementation has moved to ``sunnypilot.custom.lateral.parameter_orchestrator``.
  This class preserves the original constructor signature and public surface so that
  existing callers (including ``LatControlTorqueExt``) continue to work without changes
  during Phase 1.
  """

  def __init__(self, CP):
    if not hasattr(self, '_torque_parameter_override_policy_initialized'):
      ParameterOrchestrator.__init__(self, CP=CP, init_evidence=False)

  # The original public methods are supplied by ParameterOrchestrator:
  #   set_torque_override_refresh_allowed, update_override_torque_params
