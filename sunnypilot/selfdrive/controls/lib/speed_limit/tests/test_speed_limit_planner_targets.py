from types import SimpleNamespace

from cereal import custom
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class FakeResolver:
  speed_limit = 30.0
  speed_limit_final_last = 30.0
  speed_limit_valid = True
  speed_limit_last_valid = False
  distance = 0.0

  def update(self, v_ego, sm):
    pass


class FakeSpeedLimitAssist:
  is_active = True
  output_v_target = 30.0
  output_a_target = -0.4

  def update(self, *args):
    pass


class FakeSmartCruiseControl:
  def __init__(self):
    self.vision = SimpleNamespace(output_v_target=255.0, output_a_target=0.0)
    self.map = SimpleNamespace(output_v_target=255.0, output_a_target=0.0)

  def update(self, sm, long_enabled, long_override, v_ego, a_ego, v_cruise):
    pass


class FakeSubMaster(dict):
  def __getitem__(self, key):
    return super().__getitem__(key)


def test_speed_limit_auto_can_raise_planner_target_above_manual_cruise():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.events_sp = SimpleNamespace()

  sm = FakeSubMaster({
    'carState': SimpleNamespace(vCruiseCluster=20.0),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  })

  v_cruise = 20.0 * CV.KPH_TO_MS
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=0.0, v_cruise=v_cruise)

  assert v_target == FakeSpeedLimitAssist.output_v_target
  assert a_target == FakeSpeedLimitAssist.output_a_target
  assert planner.source == LongitudinalPlanSource.speedLimitAssist
