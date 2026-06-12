"""Tests for controlsd same-frame lateral demand profile forwarding.

Covers the contract that controlsd's state_control:
1. Calls the lateral demand stack with a per-frame
   LateralDemandStackInputs bundle.
2. Stores stack_output on self.lateral_demand_stack_output.
3. Forwards stack_output.profile to LaC (or LaC.extension) BEFORE
   self.LaC.update runs.
4. Does NOT call the legacy update_lateral_demand_profile wrapper
   from state_control (no double-update).
5. Sets self.desired_curvature from the same-frame demand.
6. LatControlTorqueV5 sees a same-frame TURN_IN profile for
   preview gating.
"""
import inspect
import sys
import types

import pytest

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

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
  LateralDemandStackOutput,
)


def _make_demand(processed_curvature: float = 0.001) -> ProcessedLateralDemand:
  return ProcessedLateralDemand(
    raw_curvature=processed_curvature,
    processed_curvature=processed_curvature,
    measured_curvature=0.0,
    curvature_limited=False,
    path_quality=1.0,
    path_reason="ok",
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    lateral_accel_limit=2.5,
    demand_source="model_path",
  )


# ---------------------------------------------------------------------------
# Static source checks on state_control
# ---------------------------------------------------------------------------


def test_controlsd_forwards_stack_profile_before_lateral_update():
  """state_control must call self.lateral_demand_stack.update(...) and
  forward stack_output.profile to the LaC BEFORE
  self.LaC.update(...). Static source check: parse the method body
  and verify the call ordering."""
  from openpilot.selfdrive.controls import controlsd

  source = inspect.getsource(controlsd.Controls.state_control)
  stack_update_idx = source.find("self.lateral_demand_stack.update(")
  push_profile_idx = source.find("self.push_lateral_demand_stack_output(")
  if push_profile_idx == -1:
    push_profile_idx = source.find("set_lateral_demand_profile(profile)")
  lac_update_idx = source.find("self.LaC.update(")
  assert stack_update_idx != -1, "state_control does not call self.lateral_demand_stack.update"
  assert push_profile_idx != -1, "state_control does not push the profile before LaC.update"
  assert lac_update_idx != -1, "state_control does not call self.LaC.update"
  assert stack_update_idx < push_profile_idx < lac_update_idx, (
    "lateral demand stack update must run, then profile must be "
    "pushed, then LaC.update must run; got "
    f"stack_update at {stack_update_idx}, push at {push_profile_idx}, "
    f"LaC.update at {lac_update_idx}"
  )


def test_controlsd_does_not_double_update_lateral_demand_profile_builder():
  """state_control must not call self.lateral_demand_profile_builder.update
  directly. All builds must go through the lateral demand stack
  (self.lateral_demand_stack.update) so the build and push are atomic.
  The legacy update_lateral_demand_profile wrapper is retained for
  unit-test backward compat but must not be called from state_control."""
  from openpilot.selfdrive.controls import controlsd

  source = inspect.getsource(controlsd.Controls.state_control)
  builder_call_count = source.count("self.lateral_demand_profile_builder.update(")
  legacy_call_count = source.count("self.update_lateral_demand_profile(")
  assert builder_call_count == 0, (
    "state_control must not call lateral_demand_profile_builder.update directly; "
    f"found {builder_call_count} direct call(s)."
  )
  assert legacy_call_count == 0, (
    "state_control must not call the legacy update_lateral_demand_profile wrapper; "
    f"found {legacy_call_count} call(s). The lateral demand stack is the single builder."
  )


def test_state_control_does_not_call_update_lateral_demand_profile():
  from openpilot.selfdrive.controls import controlsd

  source = inspect.getsource(controlsd.Controls.state_control)
  assert "self.update_lateral_demand_profile(" not in source


def test_controlsd_state_control_records_stack_output_and_desired_curvature():
  """state_control must store stack_output on
  self.lateral_demand_stack_output and set self.desired_curvature
  from the same-frame demand so telemetry and post-frame consumers
  can read the contract surface."""
  from openpilot.selfdrive.controls import controlsd

  source = inspect.getsource(controlsd.Controls.state_control)
  assert "self.lateral_demand_stack_output = " in source
  assert "self.desired_curvature = " in source
  assert "self.processed_lateral_demand = " in source


def test_controlsd_does_not_construct_default_stack_before_profile_resolution():
  from openpilot.selfdrive.controls import controlsd

  source = inspect.getsource(controlsd.Controls.__init__)
  first_build_idx = source.find("build_lateral_demand_stack_from_resolution(")
  profile_resolution_idx = source.find("resolve_controls_profile_from_params(self.params)")
  redundant_default_idx = source.find("CustomV2LateralDemandStack(dt=DT_CTRL)")

  assert first_build_idx != -1
  assert profile_resolution_idx != -1
  assert redundant_default_idx == -1
  assert profile_resolution_idx < first_build_idx


def test_controlsd_build_lateral_demand_stack_inputs_method_exists():
  """controlsd must expose build_lateral_demand_stack_inputs as the
  per-frame bundle the lateral demand stack consumes. The method
  is the contract surface for state_control's stack update call."""
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
    LateralDemandStackInputs,
  )
  assert hasattr(Controls, "build_lateral_demand_stack_inputs")
  sig = inspect.signature(Controls.build_lateral_demand_stack_inputs)
  params = list(sig.parameters.keys())
  # First four args (after self) must be CC, CS, model_v2, and live_params.
  assert params[:5] == ["self", "CC", "CS", "model_v2", "live_params"]
  return_annotation = sig.return_annotation
  assert return_annotation is LateralDemandStackInputs or return_annotation is inspect.Signature.empty


# ---------------------------------------------------------------------------
# Push surface
# ---------------------------------------------------------------------------


def test_controlsd_pushes_stack_output_profile_to_lac_directly():
  """push_lateral_demand_stack_output must call
  set_lateral_demand_profile on the LaC. This is the same-frame
  forwarding path used by state_control."""
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile

  class FakeController:
    def __init__(self):
      self.profile = None

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  class FakeStackOutput:
    def __init__(self, profile):
      self.profile = profile
      self.legacy = None
      self.requested_stack = "custom-2.0"
      self.resolved_stack = "custom-2.0"
      self.fallback_reason = ""
      self.version = "2.0"

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  profile = LateralDemandProfile(
    raw_curvature=0.001, processed_curvature=0.001, curvature_limited=False,
    path_quality=0.9, path_reason="ok", lane_change_shaping_active=False,
    lane_change_blend=0.0, demand_source="model_path", mode="turn_in",
    mode_confidence=0.9,
  )
  controls.push_lateral_demand_stack_output(FakeStackOutput(profile))
  assert controls.LaC.profile is profile


def test_controlsd_push_lateral_demand_stack_output_uses_extension_hook():
  """When LaC itself has no set_lateral_demand_profile, the helper
  must fall back to LaC.extension.set_lateral_demand_profile. This
  is the v4.1 → v5 transition path (LaC shim lacks the hook; the
  extension has it)."""
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile

  class FakeExtension:
    def __init__(self):
      self.profile = None

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  class FakeController:
    def __init__(self):
      self.extension = FakeExtension()

  class FakeStackOutput:
    def __init__(self, profile):
      self.profile = profile
      self.legacy = None
      self.requested_stack = "custom-2.0"
      self.resolved_stack = "custom-2.0"
      self.fallback_reason = ""
      self.version = "2.0"

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  profile = LateralDemandProfile(
    raw_curvature=0.001, processed_curvature=0.001, curvature_limited=False,
    path_quality=0.9, path_reason="ok", lane_change_shaping_active=False,
    lane_change_blend=0.0, demand_source="model_path", mode="turn_in",
    mode_confidence=0.9,
  )
  controls.push_lateral_demand_stack_output(FakeStackOutput(profile))
  assert controls.LaC.extension.profile is profile


def test_controlsd_pushes_none_profile_to_clear_stale_lac_profile():
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile

  class FakeController:
    def __init__(self):
      self.profile = LateralDemandProfile(
        raw_curvature=0.001, processed_curvature=0.001, curvature_limited=False,
        path_quality=0.9, path_reason="ok", lane_change_shaping_active=False,
        lane_change_blend=0.0, demand_source="model_path", mode="turn_in",
        mode_confidence=0.9,
      )

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  class FakeStackOutput:
    profile = None

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  controls.push_lateral_demand_stack_output(FakeStackOutput())
  assert controls.LaC.profile is None


def test_controlsd_pushes_none_profile_to_extension_hook():
  from openpilot.selfdrive.controls.controlsd import Controls

  class FakeExtension:
    def __init__(self):
      self.profile = object()

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  class FakeController:
    def __init__(self):
      self.extension = FakeExtension()

  class FakeStackOutput:
    profile = None

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  controls.push_lateral_demand_stack_output(FakeStackOutput())
  assert controls.LaC.extension.profile is None


def test_controlsd_push_lateral_demand_stack_output_docstring_does_not_claim_legacy_forwarding():
  from openpilot.selfdrive.controls.controlsd import Controls

  docstring = Controls.push_lateral_demand_stack_output.__doc__ or ""
  assert "legacy" not in docstring.lower()
  assert "profile=None" in docstring


# ---------------------------------------------------------------------------
# V5 same-frame preview gate
# ---------------------------------------------------------------------------


def test_v5_uses_same_frame_turn_in_profile_for_preview_gate():
  """LatControlTorqueV5 must activate the preview boost on a clean
  TURN_IN frame using a profile built and pushed in the same frame.
  Uses the real LateralDemandProfileBuilder on a fresh
  ProcessedLateralDemand to mirror controlsd's same-frame
  build+push flow."""
  from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
    LateralDemandProfileBuilder, LateralMode,
  )
  from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v4 import (
    LatControlTorqueV5, TorqueV5Target,
  )
  from openpilot.selfdrive.controls.lib.lateral_turn_exit_controller import (
    TurnExitDecision,
  )

  sys.modules.setdefault("msgq", types.ModuleType("msgq"))
  from cereal import car
  from opendbc.car.car_helpers import interfaces
  from opendbc.car.toyota.values import CAR as TOYOTA
  from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
  from openpilot.common.realtime import DT_CTRL
  from openpilot.selfdrive.car.helpers import convert_to_capnp
  from openpilot.selfdrive.controls.lib.lateral_demand import (
    DEMAND_SOURCE_MODEL_PATH, ProcessedLateralDemand,
  )

  car_name = TOYOTA.TOYOTA_RAV4
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CP_SP.neuralNetworkLateralControl.model.path = MOCK_MODEL_PATH
  CI = CarInterface(CP, CP_SP)

  def make_demand(processed_curvature, path_quality=1.0, path_reason="ok"):
    return ProcessedLateralDemand(
      raw_curvature=processed_curvature,
      processed_curvature=processed_curvature,
      measured_curvature=0.0,
      curvature_limited=False,
      path_quality=path_quality,
      path_reason=path_reason,
      lane_change_shaping_active=False,
      lane_change_blend=0.0,
      lateral_accel_limit=2.5,
      demand_source=DEMAND_SOURCE_MODEL_PATH,
    )

  def make_speed_result(**overrides):
    values = {
      "response_scale": 1.0,
      "trim_lateral_accel": 0.0,
      "response_delay": 0.2,
      "lead_gain": 0.5,
      "lead_delta_cap": 0.5,
      "feedback_gain": 0.2,
      "output_slew_rate": 1.0,
      "sign_change_slew_rate": 1.5,
      "friction": 0.0,
      "total_accel": 0.0,
      "rate_filter": 0.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)

  v5 = LatControlTorqueV5(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)

  def _decision(**_kwargs):
    return TurnExitDecision(
      mode="turn_in", persistence_frames=0,
      lead_gain_multiplier=1.0, lead_delta_cap_multiplier=1.0,
      slew_boost=1.0, same_direction_slew_boost=1.0,
      early_release_lead_zero=False, preview_boost=0.05,
      confidence=0.9,
    )
  v5.turn_exit_controller.update = _decision

  builder = LateralDemandProfileBuilder(dt=DT_CTRL)
  v_ego = 20.0
  demand0 = make_demand(0.0, path_quality=0.9)
  profile0 = builder.update(demand0, v_ego)
  assert profile0.mode != LateralMode.TURN_IN.value
  v5.set_lateral_demand_profile(profile0)
  v5.set_processed_lateral_demand(demand0)

  processed_curvature_turn_in = 0.0010
  demand1 = make_demand(processed_curvature_turn_in, path_quality=0.9)
  profile1 = builder.update(demand1, v_ego)
  assert profile1.mode == LateralMode.TURN_IN.value
  v5.set_lateral_demand_profile(profile1)
  v5.set_processed_lateral_demand(demand1)

  speed_result = make_speed_result(lead_gain=0.5, lead_delta_cap=0.5, response_delay=0.2)
  v5.previous_target_lateral_accel = 0.0
  target = v5._build_target(processed_curvature_turn_in, v_ego, speed_result, False, curvature_limited=False)

  assert isinstance(target, TorqueV5Target)
  assert target.preview_boost_applied > 0.0
  assert target.v5_active is True
  assert target.v5_reason == "preview_boost_applied"
