from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
  LateralDemandStackInputs,
  LateralDemandStackOutput,
  validate_lateral_demand_stack_output,
)
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfile,
  LateralDemandProfileBuilder,
)
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import LaneChangePathShaper, LaneChangePathShaperResult
from openpilot.selfdrive.controls.lib.lane_centering_assist import LaneCenteringAssistTracker
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessor, ModelPathProcessorResult


class CustomExperimentalLateralDemandStack:
  NAME = "custom-experimental"
  VERSION = "experimental"

  def __init__(self, dt: float) -> None:
    self.dt = dt
    self._lane_change_path_shaper = LaneChangePathShaper(dt)
    self._lane_centering_assist_tracker = LaneCenteringAssistTracker()
    self._model_path_processor = ModelPathProcessor()
    self._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=dt)
    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0
    self._smoothed_model_path_curvature = False
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False
    self._last_legacy_demand: ProcessedLateralDemand | None = None
    self._last_profile: LateralDemandProfile | None = None
    self._stage: str = "v2_baseline"

  @property
  def model_path_result(self) -> ModelPathProcessorResult:
    return self._model_path_result

  @property
  def model_path_raw_desired_curvature(self) -> float:
    return self._model_path_raw_desired_curvature

  @property
  def smoothed_model_path_curvature(self) -> bool:
    return self._smoothed_model_path_curvature

  @property
  def lateral_accel_limit_no_roll(self) -> float:
    return self._lateral_accel_limit_no_roll

  @property
  def default_lateral_accel_limited(self) -> bool:
    return self._default_lateral_accel_limited

  @property
  def previous_desired_curvature(self) -> float:
    return self._previous_desired_curvature

  @property
  def last_legacy_demand(self) -> ProcessedLateralDemand | None:
    return self._last_legacy_demand

  @property
  def last_profile(self) -> LateralDemandProfile | None:
    return self._last_profile

  @property
  def stage(self) -> str:
    return self._stage

  def reset(self) -> None:
    self._lane_change_path_shaper.reset()
    self._lane_centering_assist_tracker.reset()
    self._model_path_processor.reset()
    self._model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self._model_path_raw_desired_curvature = 0.0
    self._smoothed_model_path_curvature = False
    self._previous_desired_curvature = 0.0
    self._lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self._default_lateral_accel_limited = False
    self._last_legacy_demand = None
    self._last_profile = None

  def update(self, inputs: LateralDemandStackInputs) -> LateralDemandStackOutput:
    from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
    delegate = CustomV2LateralDemandStack(self.dt)
    delegate._lane_change_path_shaper = self._lane_change_path_shaper
    delegate._lane_centering_assist_tracker = self._lane_centering_assist_tracker
    delegate._model_path_processor = self._model_path_processor
    delegate._lateral_demand_profile_builder = self._lateral_demand_profile_builder
    delegate._model_path_result = self._model_path_result
    delegate._model_path_raw_desired_curvature = self._model_path_raw_desired_curvature
    delegate._smoothed_model_path_curvature = self._smoothed_model_path_curvature
    delegate._previous_desired_curvature = self._previous_desired_curvature
    delegate._lateral_accel_limit_no_roll = self._lateral_accel_limit_no_roll
    delegate._default_lateral_accel_limited = self._default_lateral_accel_limited
    output = delegate.update(inputs)
    self._lane_change_path_shaper = delegate._lane_change_path_shaper
    self._lane_centering_assist_tracker = delegate._lane_centering_assist_tracker
    self._model_path_processor = delegate._model_path_processor
    self._lateral_demand_profile_builder = delegate._lateral_demand_profile_builder
    self._model_path_result = delegate._model_path_result
    self._model_path_raw_desired_curvature = delegate._model_path_raw_desired_curvature
    self._smoothed_model_path_curvature = delegate._smoothed_model_path_curvature
    self._previous_desired_curvature = delegate._previous_desired_curvature
    self._lateral_accel_limit_no_roll = delegate._lateral_accel_limit_no_roll
    self._default_lateral_accel_limited = delegate._default_lateral_accel_limited
    self._last_legacy_demand = delegate._last_legacy_demand
    self._last_profile = delegate._last_profile
    self._stage = "v2_baseline"
    staged = LateralDemandStackOutput(
      requested_stack=self.NAME,
      resolved_stack=self.NAME,
      fallback_reason="",
      version=self.VERSION,
      legacy=output.legacy,
      profile=output.profile,
      debug={**output.debug, "experimental_stage": self._stage},
    )
    validation = validate_lateral_demand_stack_output(staged)
    if not validation.valid:
      raise RuntimeError(f"custom-experimental stack produced invalid output: {validation.reason}")
    return staged
