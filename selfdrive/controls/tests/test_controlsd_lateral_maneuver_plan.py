import math
import sys
import types

import pytest

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

msgq = types.ModuleType("msgq")
msgq.fake_event_handle = object()
msgq.drain_sock_raw = lambda *args, **kwargs: []
msgq.MultiplePublishersError = RuntimeError
msgq.IpcError = RuntimeError
msgq.Context = object
msgq.Poller = object
msgq.SubSocket = object
msgq.PubSocket = object
msgq.SocketEventHandle = object
msgq.toggle_fake_events = lambda *args, **kwargs: None
msgq.set_fake_prefix = lambda *args, **kwargs: None
msgq.get_fake_prefix = lambda *args, **kwargs: ""
msgq.delete_fake_prefix = lambda *args, **kwargs: None
msgq.wait_for_one_event = lambda *args, **kwargs: None
msgq.pub_sock = lambda *args, **kwargs: None
msgq.sub_sock = lambda *args, **kwargs: None
msgq.context = None
sys.modules.setdefault("msgq", msgq)

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from openpilot.selfdrive.controls.controlsd import Controls


class FakeLateralManeuverPlan:
  def __init__(self, desired_curvature):
    self.desiredCurvature = desired_curvature


class FakeSubMaster(dict):
  def __init__(self, *, desired_curvature, checks_ok):
    super().__init__(lateralManeuverPlan=FakeLateralManeuverPlan(desired_curvature))
    self._checks_ok = checks_ok

  def all_checks(self, services):
    assert services == ['lateralManeuverPlan']
    return self._checks_ok


def make_controls(*, desired_curvature=0.001, checks_ok=True):
  controls = Controls.__new__(Controls)
  controls.sm = FakeSubMaster(desired_curvature=desired_curvature, checks_ok=checks_ok)
  return controls


def test_lateral_maneuver_curvature_uses_fresh_finite_plan():
  controls = make_controls(desired_curvature=0.0015, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(True) == pytest.approx(0.0015)


def test_lateral_maneuver_curvature_ignores_stale_plan():
  controls = make_controls(desired_curvature=0.0015, checks_ok=False)

  assert controls.get_lateral_maneuver_curvature(True) is None


@pytest.mark.parametrize("desired_curvature", [math.nan, math.inf, -math.inf])
def test_lateral_maneuver_curvature_ignores_nonfinite_plan(desired_curvature):
  controls = make_controls(desired_curvature=desired_curvature, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(True) is None


def test_lateral_maneuver_curvature_ignores_plan_when_lateral_inactive():
  controls = make_controls(desired_curvature=0.0015, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(False) is None
