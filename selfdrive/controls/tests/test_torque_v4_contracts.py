import sys
import types

params_pyx = types.ModuleType("openpilot.common.params_pyx")


class FakeParams:
  def get_bool(self, _key: str) -> bool:
    return False

  def remove(self, _key: str) -> None:
    pass

  def get(self, _key: str, *_args, **_kwargs):
    return None


params_pyx.Params = FakeParams
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from openpilot.selfdrive.controls.lib.lateral_demand import DEMAND_SOURCE_LATERAL_MANEUVER
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_v4


def make_observation(**overrides):
  values = {
    "active": True,
    "v_ego": 20.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "saturated": False,
    "raw_target_lateral_accel": 0.5,
    "delay_lead_lateral_accel": 0.5,
    "target_lateral_accel_rate": 0.0,
    "actual_lateral_accel": 0.4,
    "actual_lateral_jerk": 0.1,
    "measurement_rate": 0.0,
    "finite": True,
  }
  values.update(overrides)
  return latcontrol_torque_v4.TorqueV4Observation(**values)


def test_v41_is_v4_shared_core_with_governor_profile_variant():
  assert latcontrol_torque_v4.LatControlTorqueV41.update is latcontrol_torque_v4.LatControlTorqueV4.update
  assert latcontrol_torque_v4.LatControlTorqueV41._build_target is latcontrol_torque_v4.LatControlTorqueV4._build_target
  assert latcontrol_torque_v4.LatControlTorqueV41._effective_torque_params is latcontrol_torque_v4.LatControlTorqueV4._effective_torque_params
  assert latcontrol_torque_v4.LatControlTorqueV41.VERSION == 41
  assert latcontrol_torque_v4.LatControlTorqueV41.GOVERNOR_PROFILE != latcontrol_torque_v4.LatControlTorqueV4.GOVERNOR_PROFILE


def test_v41_inherits_processed_demand_learning_gates():
  learner = latcontrol_torque_v4.TorqueV4SessionAdaptation(0.2)

  rejected = learner.update(
    make_observation(demand_source=DEMAND_SOURCE_LATERAL_MANEUVER),
    latcontrol_torque_v4.TorqueV4GovernorReason.NONE,
  )

  assert not rejected.sample_accepted
  assert rejected.reject_reason & latcontrol_torque_v4.TorqueV4LearnerRejectReason.NON_MODEL_DEMAND


def test_processed_demand_learning_gates_cover_quality_reason_shaping_and_limits():
  cases = [
    ({"path_quality": 0.5}, latcontrol_torque_v4.TorqueV4LearnerRejectReason.LOW_PATH_QUALITY),
    ({"path_reason": "path_disagreement"}, latcontrol_torque_v4.TorqueV4LearnerRejectReason.PATH_REASON),
    ({"lane_change_shaping_active": True}, latcontrol_torque_v4.TorqueV4LearnerRejectReason.LANE_CHANGE_SHAPING),
    ({"lane_change_blend": 0.2}, latcontrol_torque_v4.TorqueV4LearnerRejectReason.LANE_CHANGE_SHAPING),
    ({"curvature_limited": True}, latcontrol_torque_v4.TorqueV4LearnerRejectReason.CURVATURE_LIMITED),
  ]

  for overrides, reason in cases:
    result = latcontrol_torque_v4.TorqueV4SessionAdaptation(0.2).update(
      make_observation(**overrides),
      latcontrol_torque_v4.TorqueV4GovernorReason.NONE,
    )

    assert not result.sample_accepted
    assert result.reject_reason & reason
