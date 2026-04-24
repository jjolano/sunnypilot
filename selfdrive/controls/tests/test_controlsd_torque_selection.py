import sys
import types

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as LatControlTorqueV1
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque as LatControlTorqueV2
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v3 import LatControlTorque as LatControlTorqueV3
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v4 import LatControlTorque as LatControlTorqueV4

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

from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


class FakeParams:
  def __init__(self, enforce: bool, tune=None):
    self.enforce = enforce
    self.tune = tune

  def get_bool(self, key: str) -> bool:
    return self.enforce if key == "EnforceTorqueControl" else False

  def get(self, key: str, *args, **kwargs):
    return self.tune if key == "TorqueControlTune" else None


def get_test_context():
  car_name = TOYOTA.TOYOTA_RAV4
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CP_SP.neuralNetworkLateralControl.model.path = MOCK_MODEL_PATH
  CI = CarInterface(CP, CP_SP)
  return CP, convert_to_capnp(CP_SP), CI


def make_controls_ext(CP, CP_SP, params):
  controls_ext = ControlsExt.__new__(ControlsExt)
  controls_ext.CP = CP.as_reader()
  controls_ext.CP_SP = CP_SP.as_reader()
  controls_ext.params = params
  return controls_ext


def test_normalize_torque_tune_version():
  assert ControlsExt.normalize_torque_tune_version(None) is None
  assert ControlsExt.normalize_torque_tune_version(b"2.0") == 2.0
  assert ControlsExt.normalize_torque_tune_version("1.0") == 1.0
  assert ControlsExt.normalize_torque_tune_version("bad") is None


def test_torque_controller_selection_variants():
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(False))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV0)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 0.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV0)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, b"2.0"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV2)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 3.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV3)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 4.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV4)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 1.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, None))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac
