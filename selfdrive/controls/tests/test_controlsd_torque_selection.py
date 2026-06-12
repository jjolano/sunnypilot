import sys
import types

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from cereal import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as LatControlTorqueV1
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque as LatControlTorqueV2, LatControlTorqueV21
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v3 import LatControlTorqueV3
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v4 import LatControlTorqueV4, LatControlTorqueV41, LatControlTorqueV5
from openpilot.sunnypilot.selfdrive.controls.lib.torque_versions import (
  DEFAULT_TORQUE_TUNE_VERSION,
  TorqueControllerDefinition,
  TorqueControllerRegistry,
  resolve_torque_tune_version,
)
from openpilot.selfdrive.controls.lib.controls_profile import resolve_controls_profile

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

import openpilot.sunnypilot.selfdrive.controls.controlsd_ext as controlsd_ext
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


class FakeParams:
  def __init__(self, enforce: bool, tune=None, default_tune=4.1, speed_aware_params=None):
    self.enforce = enforce
    self.tune = tune
    self.default_tune = default_tune
    self.speed_aware_params = speed_aware_params
    self.writes = {}

  def get_bool(self, key: str) -> bool:
    return self.enforce if key == "EnforceTorqueControl" else False

  def get(self, key: str, *args, **kwargs):
    if key == "LiveTorqueSpeedAdaptiveParams":
      return self.speed_aware_params
    if key != "TorqueControlTune":
      return None
    if self.tune is None and kwargs.get("return_default", False):
      return self.default_tune
    return self.tune

  def put(self, key: str, value) -> None:
    self.writes[key] = value


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


def make_pid_origin_controller():
  CP, CP_SP, CI = get_test_context()
  CP.lateralTuning.init('pid')
  CP.lateralTuning.pid.kpBP = [0.0]
  CP.lateralTuning.pid.kpV = [0.1]
  CP.lateralTuning.pid.kiBP = [0.0]
  CP.lateralTuning.pid.kiV = [0.01]
  CP.lateralTuning.pid.kf = 0.00006
  lac = LatControlPID(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return CP, CP_SP, CI, lac


def test_normalize_torque_tune_version():
  assert ControlsExt.normalize_torque_tune_version(None) is None
  assert ControlsExt.normalize_torque_tune_version(b"2.0") == 2.0
  assert ControlsExt.normalize_torque_tune_version("1.0") == 1.0
  assert ControlsExt.normalize_torque_tune_version("bad") is None


def test_torque_tune_resolution_keeps_reactivated_v4_numeric():
  resolution = resolve_torque_tune_version("4.0")

  assert DEFAULT_TORQUE_TUNE_VERSION == 4.1
  assert resolution.requested_version == 4.0
  assert resolution.resolved_version == 4.0
  assert resolution.persist_value is None
  assert resolve_torque_tune_version("4.1").resolved_version == 4.1
  assert resolve_torque_tune_version(b"2.1").resolved_version == 2.1
  assert resolve_torque_tune_version(b"3.0").resolved_version == 3.0
  assert resolve_torque_tune_version("bad").resolved_version == 4.1


def test_torque_controller_registry_resolves_factories():
  class ControllerA:
    pass

  registry = TorqueControllerRegistry((TorqueControllerDefinition(2.0, ControllerA),))

  assert registry.factory_for(2.0) is ControllerA
  assert registry.factory_for(3.0) is None


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
  assert hasattr(selected, "output_shaper")

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, b"2.1"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV21)
  assert selected.USE_REFINED_OUTPUT_GOVERNOR

  params = FakeParams(True, 3.0)
  controls_ext = make_controls_ext(CP, CP_SP, params)
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV3)
  assert "TorqueControlTune" not in params.writes

  params = FakeParams(True, 4.0)
  controls_ext = make_controls_ext(CP, CP_SP, params)
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV4)
  assert not hasattr(selected, "extension")
  assert "TorqueControlTune" not in params.writes

  params = FakeParams(True, 4.1)
  controls_ext = make_controls_ext(CP, CP_SP, params)
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)
  assert not hasattr(selected, "extension")
  assert "TorqueControlTune" not in params.writes

  # 5.0 is the first torque version with active profile-aware
  # command shaping. There is no hidden selector; selecting 5.0
  # directly selects LatControlTorqueV5.
  params = FakeParams(True, 5.0)
  controls_ext = make_controls_ext(CP, CP_SP, params)
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV5)
  assert selected.VERSION == 50
  assert "TorqueControlTune" not in params.writes


def test_torque_control_tune_5_instantiates_latcontrol_torque_v5():
  """TorqueControlTune=5.0 must select LatControlTorqueV5, the
  first torque version with active profile-aware command
  shaping (preview lead + turn-exit source-of-truth)."""
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 5.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV5)
  assert selected.VERSION == 50


def test_torque_control_tune_41_instantiates_latcontrol_torque_v41():
  """TorqueControlTune=4.1 must select LatControlTorqueV41, the
  current stable default. The 4.1 path is the v4.1 controller
  (no v5 active delta)."""
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 4.1))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)
  assert selected.VERSION == 41


def test_torque_control_tune_unknown_falls_back_safely():
  """An unknown TorqueControlTune value must fall back safely
  (to 4.1, the stable default) and must NOT select
  5.0. The fallback path is exercised by users who set a
  deprecated or future value."""
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, None))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert not isinstance(selected, LatControlTorqueV5)
  assert isinstance(selected, LatControlTorqueV41)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 1.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, None))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)


def test_unknown_torque_tune_falls_back_to_41():
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, "not-a-tune"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)


def test_pid_origin_non_angle_controller_keeps_original_lac_for_v3():
  CP, CP_SP, CI, lac = make_pid_origin_controller()

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 3.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_pid_origin_non_angle_controller_keeps_original_lac_for_v4():
  CP, CP_SP, CI, lac = make_pid_origin_controller()

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, "4.0"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_pid_origin_non_angle_controller_keeps_original_lac_for_v41():
  CP, CP_SP, CI, lac = make_pid_origin_controller()

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, "4.1"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_angle_controller_keeps_original_lac_for_v41():
  CP, CP_SP, CI, lac = make_pid_origin_controller()
  CP.steerControlType = car.CarParams.SteerControlType.angle

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, "4.1"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_angle_controller_keeps_original_lac_for_v4():
  CP, CP_SP, CI, lac = make_pid_origin_controller()
  CP.steerControlType = car.CarParams.SteerControlType.angle

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, "4.0"))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_pid_origin_non_angle_controller_keeps_original_lac_without_enforce():
  CP, CP_SP, CI, lac = make_pid_origin_controller()

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(False, 3.0))
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert selected is lac


def test_pid_origin_non_angle_controller_keeps_original_lac_for_non_v3_tunes():
  CP, CP_SP, CI, lac = make_pid_origin_controller()

  for tune in (2.0, 0.0):
    controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, tune))
    selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
    assert selected is lac


def test_controls_profile_experimental_instantiates_latcontrol_torque_v5():
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 4.1))
  controls_ext.controls_profile_resolution = resolve_controls_profile("custom-experimental")
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV5)


def test_controls_profile_custom_2_instantiates_latcontrol_torque_v41():
  CP, CP_SP, CI = get_test_context()
  lac = LatControlTorqueV1(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)

  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, 5.0))
  controls_ext.controls_profile_resolution = resolve_controls_profile("custom-2.0")
  selected = controls_ext.initialize_lateral_control(lac, CI, DT_CTRL)
  assert isinstance(selected, LatControlTorqueV41)


def test_update_lateral_controller_inputs_refreshes_extension_limits_after_live_torque_params():
  class FakeTorqueParams:
    useParams = True
    latAccelFactorFiltered = 1.0
    latAccelOffsetFiltered = 2.0
    frictionCoefficientFiltered = 3.0

  class FakeSubMaster(dict):
    def all_checks(self, services):
      return services == ['liveTorqueParameters']

  class FakeExtension:
    def __init__(self):
      self.updated_limits = False

    def update_limits(self):
      self.updated_limits = True

  class FakeController:
    def __init__(self):
      self.extension = FakeExtension()
      self.live_torque_params = None

    def update_live_torque_params(self, *params):
      self.live_torque_params = params

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  controls.sm = FakeSubMaster(liveTorqueParameters=FakeTorqueParams(), modelV2=object())
  controls.lat_delay = 0.2

  controls.update_lateral_controller_inputs()

  assert controls.LaC.live_torque_params == (1.0, 2.0, 3.0)
  assert controls.LaC.extension.updated_limits is True


def test_update_lateral_controller_inputs_updates_model_only_when_fresh():
  class FakeTorqueParams:
    useParams = False

  class FakeSubMaster(dict):
    def __init__(self):
      super().__init__(liveTorqueParameters=FakeTorqueParams(), modelV2=object())
      self.updated = {'modelV2': False}

    def all_checks(self, _services):
      return False

  class FakeController:
    def __init__(self):
      self.model_updates = 0

    def update_model_v2(self, _model_v2):
      self.model_updates += 1

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  controls.sm = FakeSubMaster()
  controls.lat_delay = 0.2

  controls.update_lateral_controller_inputs()
  assert controls.LaC.model_updates == 0

  controls.sm.updated['modelV2'] = True
  controls.update_lateral_controller_inputs()
  assert controls.LaC.model_updates == 1


def test_update_lateral_controller_inputs_normalizes_cached_lat_delay():
  class FakeTorqueParams:
    useParams = False

  class FakeSubMaster(dict):
    def __init__(self):
      super().__init__(liveTorqueParameters=FakeTorqueParams(), modelV2=object())
      self.updated = {'modelV2': False}

    def all_checks(self, _services):
      return False

  class FakeController:
    def __init__(self):
      self.lat_delay = None

    def update_lateral_lag(self, lat_delay):
      self.lat_delay = lat_delay

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  controls.sm = FakeSubMaster()
  controls.lat_delay = b"0.2"

  controls.update_lateral_controller_inputs()

  assert controls.lat_delay == 0.2
  assert controls.LaC.lat_delay == 0.2


def test_get_params_sp_updates_lat_delay_for_selected_torque_controller(monkeypatch):
  class FakeBlinkerPauseLateral:
    def get_params(self):
      pass

  class FakeLiveDelay:
    lateralDelay = 0.42

  class FakeTorqueController:
    CONTROL_STATE = "torque"

  CP, CP_SP, _CI = get_test_context()
  CP.lateralTuning.init('pid')
  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(False))
  controls_ext._param_update_time = 0.0
  controls_ext.blinker_pause_lateral = FakeBlinkerPauseLateral()
  controls_ext.LaC = FakeTorqueController()
  monkeypatch.setattr(controlsd_ext, "get_lat_delay", lambda _params, lateral_delay: lateral_delay + 0.1)

  controls_ext.get_params_sp({"liveDelay": FakeLiveDelay()})

  assert controls_ext.lat_delay == 0.52


def test_get_params_sp_skips_speed_aware_params_without_extension(monkeypatch):
  class FakeBlinkerPauseLateral:
    def get_params(self):
      pass

  class FakeLiveDelay:
    lateralDelay = 0.42

  class FakeTorqueController:
    CONTROL_STATE = "torque"

  CP, CP_SP, _CI = get_test_context()
  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, speed_aware_params="speed-aware-payload"))
  controls_ext._param_update_time = 0.0
  controls_ext.blinker_pause_lateral = FakeBlinkerPauseLateral()
  controls_ext.LaC = FakeTorqueController()
  monkeypatch.setattr(controlsd_ext, "get_lat_delay", lambda _params, lateral_delay: lateral_delay + 0.1)

  controls_ext.get_params_sp({"liveDelay": FakeLiveDelay()})

  assert controls_ext.lat_delay == 0.52


def test_get_params_sp_updates_speed_aware_params_when_extension_exists(monkeypatch):
  class FakeBlinkerPauseLateral:
    def get_params(self):
      pass

  class FakeLiveDelay:
    lateralDelay = 0.42

  class FakeExtension:
    def __init__(self):
      self.speed_aware_params = None

    def update_speed_aware_params(self, speed_aware_params):
      self.speed_aware_params = speed_aware_params

  class FakeTorqueController:
    CONTROL_STATE = "torque"

    def __init__(self):
      self.extension = FakeExtension()

  CP, CP_SP, _CI = get_test_context()
  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, speed_aware_params="speed-aware-payload"))
  controls_ext._param_update_time = 0.0
  controls_ext.blinker_pause_lateral = FakeBlinkerPauseLateral()
  controls_ext.LaC = FakeTorqueController()
  monkeypatch.setattr(controlsd_ext, "get_lat_delay", lambda _params, lateral_delay: lateral_delay + 0.1)

  controls_ext.get_params_sp({"liveDelay": FakeLiveDelay()})

  assert controls_ext.lat_delay == 0.52
  assert controls_ext.LaC.extension.speed_aware_params == "speed-aware-payload"


def test_get_params_sp_prefers_direct_speed_aware_params_hook(monkeypatch):
  class FakeBlinkerPauseLateral:
    def get_params(self):
      pass

  class FakeLiveDelay:
    lateralDelay = 0.42

  class FakeExtension:
    def __init__(self):
      self.speed_aware_params = None

    def update_speed_aware_params(self, speed_aware_params):
      self.speed_aware_params = speed_aware_params

  class FakeTorqueController:
    CONTROL_STATE = "torque"

    def __init__(self):
      self.speed_aware_params = None
      self.extension = FakeExtension()

    def update_speed_aware_params(self, speed_aware_params):
      self.speed_aware_params = speed_aware_params

  CP, CP_SP, _CI = get_test_context()
  controls_ext = make_controls_ext(CP, CP_SP, FakeParams(True, speed_aware_params="speed-aware-payload"))
  controls_ext._param_update_time = 0.0
  controls_ext.blinker_pause_lateral = FakeBlinkerPauseLateral()
  controls_ext.LaC = FakeTorqueController()
  monkeypatch.setattr(controlsd_ext, "get_lat_delay", lambda _params, lateral_delay: lateral_delay + 0.1)

  controls_ext.get_params_sp({"liveDelay": FakeLiveDelay()})

  assert controls_ext.lat_delay == 0.52
  assert controls_ext.LaC.speed_aware_params == "speed-aware-payload"
  assert controls_ext.LaC.extension.speed_aware_params is None
