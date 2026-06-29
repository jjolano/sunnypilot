"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.custom.lateral.parameter_orchestrator import (
  LAT_PLAN_MIN_IDX,  # noqa: F401
  LATERAL_LAG_MOD,  # noqa: F401
  KP,  # noqa: F401
  KI,  # noqa: F401
  ParameterOrchestrator,
  get_lookahead_value,  # noqa: F401
  get_predicted_lateral_jerk,  # noqa: F401
  sign,  # noqa: F401
)


class LatControlTorqueExtBase(ParameterOrchestrator):
  """Compatibility facade for the legacy torque extension base.

  The implementation has moved to ``sunnypilot.custom.lateral.parameter_orchestrator``.
  This class preserves the original constructor signature and public surface so that
  ``LatControlTorqueExt`` and ``NeuralNetworkLateralControl`` continue to work without
  changes during Phase 1.
  """

  def __init__(self, lac_torque, CP, CP_SP, CI):
    if not hasattr(self, '_torque_model_evidence_initialized'):
      ParameterOrchestrator.__init__(
        self,
        lac_torque=lac_torque,
        CP=CP,
        CP_SP=CP_SP,
        CI=CI,
        init_override=False,
      )

  # The original public methods are supplied by ParameterOrchestrator:
  #   update_model_v2, update_lateral_lag, update_friction_input, update_calculations
