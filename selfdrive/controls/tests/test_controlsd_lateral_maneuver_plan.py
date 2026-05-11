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

from cereal import log
import cereal.messaging as messaging
from openpilot.selfdrive.controls.controlsd import Controls, fill_model_path_state, model_path_reason_to_capnp
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessorResult


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


def test_model_path_reason_mapping_uses_controls_state_schema_enum():
  assert model_path_reason_to_capnp("path_disagreement") == log.ControlsState.ModelPathState.Reason.pathDisagreement
  assert model_path_reason_to_capnp("unexpected") == log.ControlsState.ModelPathState.Reason.unknown


def test_fill_model_path_state_publishes_processed_path_debug_values():
  msg = messaging.new_message('controlsState')
  result = ModelPathProcessorResult(
    0.0015,
    0.65,
    True,
    "path_disagreement",
    1,
    smoothing_tau_s=0.11,
    damping_alpha=0.18,
    trust_penalty=0.4,
    spatial_smoothed_curvature=0.00155,
    lane_change_fade=0.75,
  )

  fill_model_path_state(msg.controlsState.modelPathState, result, 0.002)

  state = msg.controlsState.modelPathState
  assert state.active
  assert state.gated
  assert state.quality == pytest.approx(0.65)
  assert state.reason == log.ControlsState.ModelPathState.Reason.pathDisagreement
  assert state.rawDesiredCurvature == pytest.approx(0.002)
  assert state.processedDesiredCurvature == pytest.approx(0.0015)
  assert state.holdFramesRemaining == 1
  assert state.smoothingTauS == pytest.approx(0.11)
  assert state.dampingAlpha == pytest.approx(0.18)
  assert state.trustPenalty == pytest.approx(0.4)
  assert state.spatialSmoothedCurvature == pytest.approx(0.00155)
  assert state.laneChangeFade == pytest.approx(0.75)
