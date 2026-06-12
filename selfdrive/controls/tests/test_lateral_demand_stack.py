import sys
import types

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from openpilot.selfdrive.controls.controlsd import build_lateral_demand_stack_from_resolution
from openpilot.selfdrive.controls.lib.lateral_demand_stacks import (
  CUSTOM_EXPERIMENTAL,
  CUSTOM_V2,
  SUNNYPILOT_CURRENT,
  resolve_lateral_demand_stack,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_experimental import CustomExperimentalLateralDemandStack
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.sunnypilot_current import SunnypilotCurrentLateralDemandStack


def test_lateral_demand_stack_custom_v2_instantiates_custom_v2_stack():
  stack = build_lateral_demand_stack_from_resolution(resolve_lateral_demand_stack(CUSTOM_V2), dt=0.05)
  assert isinstance(stack, CustomV2LateralDemandStack)


def test_lateral_demand_stack_sunnypilot_current_instantiates_current_stack():
  stack = build_lateral_demand_stack_from_resolution(resolve_lateral_demand_stack(SUNNYPILOT_CURRENT), dt=0.05)
  assert isinstance(stack, SunnypilotCurrentLateralDemandStack)


def test_lateral_demand_stack_custom_experimental_instantiates_experimental_stack():
  stack = build_lateral_demand_stack_from_resolution(resolve_lateral_demand_stack(CUSTOM_EXPERIMENTAL), dt=0.05)
  assert isinstance(stack, CustomExperimentalLateralDemandStack)


def test_unknown_lateral_stack_falls_back_safely():
  resolution = resolve_lateral_demand_stack("not-a-stack")
  stack = build_lateral_demand_stack_from_resolution(resolution, dt=0.05)
  assert resolution.resolved_stack == CUSTOM_V2
  assert isinstance(stack, CustomV2LateralDemandStack)
