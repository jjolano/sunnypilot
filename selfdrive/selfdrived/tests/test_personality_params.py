import sys
import types

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionIpcClient = object
visionipc.VisionStreamType = object
sys.modules.setdefault("msgq.visionipc", visionipc)

from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class FakeParams:
  def __init__(self, personality):
    self.personality = personality

  def get(self, key, *args, **kwargs):
    assert key == "LongitudinalPersonality"
    return self.personality


def make_selfdrived(personality=1, param_personality=2, hold_until=0.0):
  selfdrived = SelfdriveD.__new__(SelfdriveD)
  selfdrived.params = FakeParams(param_personality)
  selfdrived.personality = personality
  selfdrived._personality_param_hold_until = hold_until
  return selfdrived


def test_personality_param_read_skips_stale_value_during_local_holdoff():
  selfdrived = make_selfdrived(personality=1, param_personality=2, hold_until=10.0)

  selfdrived.update_personality_from_params(now=9.9)

  assert selfdrived.personality == 1


def test_personality_param_read_applies_after_local_holdoff():
  selfdrived = make_selfdrived(personality=1, param_personality=2, hold_until=10.0)

  selfdrived.update_personality_from_params(now=10.0)

  assert selfdrived.personality == 2
