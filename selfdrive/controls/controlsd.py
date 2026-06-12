#!/usr/bin/env python3
import math
from numbers import Number
from typing import cast

from cereal import car, custom, log
import cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  clip_curvature_with_result,
  should_latch_lateral_accel_burst,
  update_lateral_accel_limit,
)
from openpilot.selfdrive.controls.lib.lane_centering_assist import (
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
  inactive_lane_centering_assist_result,
)
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import LaneChangePathShaper, LaneChangePathShaperInputs
from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfileBuilder
from openpilot.selfdrive.controls.lib.lateral_demand_stacks import (
  CUSTOM_EXPERIMENTAL as LATERAL_STACK_CUSTOM_EXPERIMENTAL,
  CUSTOM_V2 as LATERAL_STACK_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LATERAL_STACK_SUNNYPILOT_CURRENT,
  LateralDemandStackInputs,
  LateralDemandStackOutput,
  LateralDemandStackResolution,
  resolve_lateral_demand_stack as resolve_lateral_demand_stack_selection,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_experimental import CustomExperimentalLateralDemandStack
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.sunnypilot_current import SunnypilotCurrentLateralDemandStack
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessor, ModelPathProcessorInputs, ModelPathProcessorResult
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.selfdrive.controls.lib.controls_profile import (
  ControlsProfileId,
  ControlsProfileParamResolution,
  ControlsProfileResolution,
  _param_has_value,
  resolve_controls_profile_from_params,
)
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import (
  SteeringActuatorFeedback,
  SteeringActuatorRequest,
  build_steering_actuator_feedback,
)

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = custom.ModelDataV2SP.TurnDirection
StackId = custom.LongitudinalPlanSP.Stack.StackId

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())
TOYOTA_EPS_HIGH_RATE_DEG = 100.0
TOYOTA_EPS_HIGH_RATE_FRAMES = 15
TOYOTA_EPS_HIGH_RATE_CUT_FRAMES = 2
TOYOTA_EPS_HIGH_RATE_FINGERPRINTS = frozenset(platform.value for platform in TOYOTA)
MODEL_PATH_REASON_TO_CAPNP = {
  "ok": log.ControlsState.ModelPathState.Reason.ok,
  "inactive": log.ControlsState.ModelPathState.Reason.inactive,
  "nonfinite_curvature": log.ControlsState.ModelPathState.Reason.nonfiniteCurvature,
  "invalid_path": log.ControlsState.ModelPathState.Reason.invalidPath,
  "turn_opposite_curvature": log.ControlsState.ModelPathState.Reason.turnOppositeCurvature,
  "high_path_std": log.ControlsState.ModelPathState.Reason.highPathStd,
  "low_lane_confidence": log.ControlsState.ModelPathState.Reason.lowLaneConfidence,
  "frame_drop": log.ControlsState.ModelPathState.Reason.frameDrop,
  "path_disagreement": log.ControlsState.ModelPathState.Reason.pathDisagreement,
  "curvature_jump": log.ControlsState.ModelPathState.Reason.curvatureJump,
  "lateral_maneuver": log.ControlsState.ModelPathState.Reason.lateralManeuver,
}


def model_path_reason_to_capnp(reason: str):
  return MODEL_PATH_REASON_TO_CAPNP.get(reason, log.ControlsState.ModelPathState.Reason.unknown)


def _enum_int(value) -> int:
  raw = getattr(value, "raw", value)
  return int(raw)


def build_lateral_demand_stack_from_resolution(resolution: LateralDemandStackResolution, dt: float):
  if resolution.resolved_stack == LATERAL_STACK_SUNNYPILOT_CURRENT:
    return SunnypilotCurrentLateralDemandStack(dt=dt)
  if resolution.resolved_stack == LATERAL_STACK_CUSTOM_EXPERIMENTAL:
    return CustomExperimentalLateralDemandStack(dt=dt)
  # custom-recommended currently resolves through the manifest to
  # a concrete implemented stack (custom-2.0 by default). Unknown or
  # unavailable stacks also resolve to the manifest default, which is
  # custom-2.0 in this fork.
  return CustomV2LateralDemandStack(dt=dt)


def fill_model_path_state(model_path_state, model_path_result: ModelPathProcessorResult, raw_desired_curvature: float) -> None:
  model_path_state.active = model_path_result.reason != "inactive"
  model_path_state.gated = bool(model_path_result.gated)
  model_path_state.quality = float(model_path_result.quality)
  model_path_state.reason = model_path_reason_to_capnp(model_path_result.reason)
  raw_desired_curvature = float(raw_desired_curvature)
  processed_desired_curvature = float(model_path_result.desired_curvature)
  model_path_state.rawDesiredCurvature = raw_desired_curvature if math.isfinite(raw_desired_curvature) else 0.0
  model_path_state.processedDesiredCurvature = processed_desired_curvature if math.isfinite(processed_desired_curvature) else 0.0
  model_path_state.holdFramesRemaining = max(0, min(255, int(model_path_result.hold_frames_remaining)))
  model_path_state.smoothingTauS = float(model_path_result.smoothing_tau_s) if math.isfinite(model_path_result.smoothing_tau_s) else 0.0
  model_path_state.dampingAlpha = float(model_path_result.damping_alpha) if math.isfinite(model_path_result.damping_alpha) else 0.0
  model_path_state.trustPenalty = float(model_path_result.trust_penalty) if math.isfinite(model_path_result.trust_penalty) else 0.0
  model_path_state.spatialSmoothedCurvature = (
    float(model_path_result.spatial_smoothed_curvature) if math.isfinite(model_path_result.spatial_smoothed_curvature) else 0.0
  )
  model_path_state.laneChangeFade = float(model_path_result.lane_change_fade) if math.isfinite(model_path_result.lane_change_fade) else 0.0


def compute_steering_actuator_feedback(previous_request, actuators_output, steer_control_type, lat_active=True):
  return build_steering_actuator_feedback(previous_request, actuators_output, steer_control_type,
                                          lat_active=lat_active)


def apply_toyota_eps_high_rate_guard(CP, CC, CS, high_rate_frames, cut_frames):
  toyota_torque_control = CP.carFingerprint in TOYOTA_EPS_HIGH_RATE_FINGERPRINTS and \
                          CP.steerControlType == car.CarParams.SteerControlType.torque
  if not toyota_torque_control or not CC.latActive or CS.steerFaultTemporary or CS.steerFaultPermanent:
    return 0, 0

  if cut_frames > 0:
    CC.latActive = False
    CC.actuators.torque = 0.0
    return 0, cut_frames - 1

  if abs(CS.steeringRateDeg) < TOYOTA_EPS_HIGH_RATE_DEG:
    return 0, 0

  high_rate_frames += 1
  if high_rate_frames < TOYOTA_EPS_HIGH_RATE_FRAMES:
    return high_rate_frames, 0

  CC.latActive = False
  CC.actuators.torque = 0.0
  return 0, TOYOTA_EPS_HIGH_RATE_CUT_FRAMES - 1


def get_traction_risk(car_state_sp) -> float:
  try:
    traction_risk = float(getattr(car_state_sp, "tractionRisk", 0.0))
  except (TypeError, ValueError, AttributeError):
    return 0.0
  return float(min(1.0, max(0.0, traction_risk))) if math.isfinite(traction_risk) else 0.0


class Controls(ControlsExt):
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    # Initialize sunnypilot controlsd extension and base model state
    ControlsExt.__init__(self, self.CP, self.params)

    self.CI = interfaces[self.CP.carFingerprint](self.CP, self.CP_SP)

    self.sm = messaging.SubMaster(['liveDelay', 'liveParameters', 'liveTorqueParameters', 'modelV2', 'modelDataV2SP', 'selfdriveState',
                                    'liveCalibration', 'livePose', 'longitudinalPlan', 'longitudinalPlanSP', 'lateralManeuverPlan', 'carState', 'carOutput',
                                    'carStateSP', 'driverMonitoringState', 'onroadEvents', 'driverAssistance', 'liveDelay'] + self.sm_services_ext,
                                   poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'] + self.pm_services_ext)

    self.steer_limited_by_safety = False
    self.toyota_eps_high_rate_frames = 0
    self.toyota_eps_cut_frames = 0
    self.steering_actuator_feedback = SteeringActuatorFeedback.invalid()
    self._previous_steering_actuator_request: SteeringActuatorRequest | None = None
    self.curvature = 0.0
    self.desired_curvature = 0.0
    self.lateral_demand_stack_output: LateralDemandStackOutput | None = None
    self.lateral_demand_stack_resolution: LateralDemandStackResolution | None = None
    self.controls_profile_param_resolution: ControlsProfileParamResolution | None = None
    self.controls_profile_resolution: ControlsProfileResolution | None = None
    self.resolved_longitudinal_stack = ""
    self._last_logged_stack: tuple[str, str, bool, str] | None = None
    self.processed_lateral_demand = ProcessedLateralDemand(
      0.0,
      0.0,
      0.0,
      True,
      0.0,
      "inactive",
      False,
      0.0,
      MAX_LATERAL_ACCEL_NO_ROLL,
    )

    # ControlsProfile is a user-facing alias for longitudinal stack,
    # lateral demand stack, and torque controller tune. Missing profile
    # params are migration-safe: existing explicit TorqueControlTune /
    # LateralDemandStack / LongitudinalStack values are preserved, while
    # absent torque safely defaults to 4.1 (never 5.0). Registry/default
    # return values are not treated as explicit user choices.
    controls_profile_explicit = _param_has_value(self.params, "ControlsProfile")
    self.controls_profile_param_resolution = resolve_controls_profile_from_params(self.params)
    self.controls_profile_resolution = self.controls_profile_param_resolution.controls_profile_resolution
    self.controls_profile_id: ControlsProfileId = self.controls_profile_resolution.resolved_profile
    self.resolved_longitudinal_stack = self.controls_profile_resolution.longitudinal_stack
    if controls_profile_explicit:
      self.params.put("LongitudinalStack", self.resolved_longitudinal_stack)

    # Rich lateral demand stack. The manifest resolver owns
    # availability, fallback metadata, and custom-recommended
    # resolution. The selected concrete stack is then built from
    # the resolved stack id.
    self.lateral_demand_stack_resolution = resolve_lateral_demand_stack_selection(
      self.controls_profile_resolution.lateral_demand_stack,
      CP=self.CP,
      CP_SP=self.CP_SP,
    )
    self.lateral_demand_stack = build_lateral_demand_stack_from_resolution(
      self.lateral_demand_stack_resolution,
      DT_CTRL,
    )

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP, self.CP_SP)
    self.VM = VehicleModel(self.CP)
    self.LaC: LatControl
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CP_SP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CP_SP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'torque':
      self.LaC = LatControlTorque(self.CP, self.CP_SP, self.CI, DT_CTRL)

    self.LaC = ControlsExt.initialize_lateral_control(self, self.LaC, self.CI, DT_CTRL)

  def update(self):
    self.sm.update(15)
    if self.sm.updated["liveCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated["livePose"]:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

  def get_lateral_maneuver_curvature(self, lat_active: bool) -> float | None:
    if not lat_active or not self.sm.all_checks(['lateralManeuverPlan']):
      return None

    desired_curvature = self.sm['lateralManeuverPlan'].desiredCurvature
    if not math.isfinite(desired_curvature):
      cloudlog.error(f"lateralManeuverPlan.desiredCurvature not finite {desired_curvature}")
      return None
    return float(desired_curvature)

  def build_lateral_demand_stack_inputs(self, CC, CS, model_v2, live_params) -> LateralDemandStackInputs:
    raw_curvature = float(model_v2.action.desiredCurvature)
    turn_direction = 0
    if (
      model_v2.meta.laneChangeState == LaneChangeState.off
      and self.sm.valid['modelDataV2SP']
    ):
      turn_direction = _enum_int(self.sm['modelDataV2SP'].laneTurnDirection)
    return LateralDemandStackInputs(
      lat_active=CC.latActive,
      v_ego=CS.vEgo,
      desired_curvature=raw_curvature,
      measured_curvature=self.curvature,
      model_v2=model_v2,
      live_params=live_params,
      curvature_limited=False,
      accurate_lateral_accel=self.params.get_bool("AccurateLateralAccel"),
      manual_gas_lateral_accel_override=CS.gasPressed and not CC.longActive,
      lateral_maneuver_curvature=self.get_lateral_maneuver_curvature(CC.latActive),
      roll=live_params.roll,
      lateral_accel_limit_no_roll=MAX_LATERAL_ACCEL_NO_ROLL,
      default_lateral_accel_limited=False,
      lane_change_state=_enum_int(model_v2.meta.laneChangeState),
      lane_change_direction=_enum_int(model_v2.meta.laneChangeDirection),
      turn_direction=turn_direction,
      model_data_v2_sp_valid=bool(self.sm.valid['modelDataV2SP']),
      lane_centering_assist_enabled=self.params.get_bool("LaneCenteringAssistEnabled"),
      gas_pressed=CS.gasPressed,
      brake_pressed=CS.brakePressed,
      steering_pressed=CS.steeringPressed,
      left_blinker=CS.leftBlinker,
      right_blinker=CS.rightBlinker,
      left_lane_y0=(
        model_v2.laneLines[1].y[0]
        if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[1].y)
        else None
      ),
      right_lane_y0=(
        model_v2.laneLines[2].y[0]
        if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y)
        else None
      ),
      frame_drop_perc=model_v2.frameDropPerc,
      smoothed_model_path_curvature=False,
      position_x=tuple(model_v2.position.x),
      position_y=tuple(model_v2.position.y),
      position_y_std=tuple(model_v2.position.yStd),
      orientation_z=tuple(model_v2.orientation.z),
      orientation_rate_z=tuple(model_v2.orientationRate.z),
      lane_line_probs=tuple(model_v2.laneLineProbs),
      lane_line_stds=tuple(model_v2.laneLineStds),
      sm_valid_model_v2=bool(self.sm.valid['modelV2']),
      sm_valid_model_data_v2=bool(self.sm.valid['modelDataV2SP']),
      sm_valid_live_parameters=bool(self.sm.valid['liveParameters']),
      sm_valid_lateral_maneuver_plan=bool(self.sm.valid['lateralManeuverPlan']),
    )

  def update_lateral_controller_demand(self, demand: ProcessedLateralDemand) -> None:
    set_processed_lateral_demand = getattr(self.LaC, "set_processed_lateral_demand", None)
    if set_processed_lateral_demand is None and hasattr(self.LaC, "extension"):
      set_processed_lateral_demand = getattr(self.LaC.extension, "set_processed_lateral_demand", None)
    if set_processed_lateral_demand is not None:
      set_processed_lateral_demand(demand)

  def update_lateral_demand_profile(self, demand: ProcessedLateralDemand, v_ego: float, *,
                                    curvature_limited: bool = False, saturated: bool = False,
                                    steer_limited_by_safety: bool = False, steering_pressed: bool = False) -> None:
    # Deprecated compatibility wrapper. state_control forwards
    # stack_output.profile instead; stacks without the legacy private
    # builder intentionally no-op here.
    builder = getattr(self.lateral_demand_stack, "_lateral_demand_profile_builder", None)
    if builder is None:
      return
    profile = builder.update(
      demand,
      v_ego,
      curvature_limited=curvature_limited,
      saturated=saturated,
      steer_limited_by_safety=steer_limited_by_safety,
      steering_pressed=steering_pressed,
    )
    set_lateral_demand_profile = getattr(self.LaC, "set_lateral_demand_profile", None)
    if set_lateral_demand_profile is None and hasattr(self.LaC, "extension"):
      set_lateral_demand_profile = getattr(self.LaC.extension, "set_lateral_demand_profile", None)
    if set_lateral_demand_profile is not None:
      set_lateral_demand_profile(profile)

  def push_lateral_demand_stack_output(self, stack_output, *, steering_pressed: bool = False) -> None:
    """Forward the same-frame lateral demand profile to the lateral controller.

    Passing profile=None is intentional: it clears stale profile state when
    the selected lateral demand stack does not provide a profile.
    """
    profile = getattr(stack_output, "profile", None)
    set_lateral_demand_profile = getattr(self.LaC, "set_lateral_demand_profile", None)
    if set_lateral_demand_profile is None and hasattr(self.LaC, "extension"):
      set_lateral_demand_profile = getattr(self.LaC.extension, "set_lateral_demand_profile", None)
    if set_lateral_demand_profile is not None:
      set_lateral_demand_profile(profile)

  def _auto_couple_torque_for_stack(self, stack):
    """Map a lateral demand stack to a TorqueControlTune value
    for first-run auto-couple. custom-experimental → 5.0,
    custom-2.0/custom-recommended → 2.1. sunnypilot-current
    and unknown stacks do not auto-couple torque."""
    stack_id = getattr(stack, "stack_id", getattr(stack, "NAME", None))
    stack_value = getattr(stack_id, "value", stack_id)
    if stack_value == "custom-experimental":
      return 5.0
    if stack_value in ("custom-2.0", "custom-recommended"):
      return 2.1
    return None

  def state_control(self):
    CS = self.sm['carState']

    # Update VehicleModel
    lp = self.sm['liveParameters']
    x = max(lp.stiffnessFactor, 0.1)
    sr = max(lp.steerRatio, 0.1)
    self.VM.update_params(x, sr)

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)

    self.update_lateral_controller_inputs()

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill

    # Get which state to use for active lateral control
    _lat_active = self.get_lat_active(self.sm)

    CC.latActive = _lat_active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and \
                    (self.CP.openpilotLongitudinalControl or not self.CP_SP.pcmCruiseSpeed)

    actuators = CC.actuators
    actuators.longControlState = self.LoC.long_control_state

    # Enable blinkers while lane changing
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # accel PID loop
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, self.CP_SP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    custom_longitudinal_stack = self.sm.valid['longitudinalPlanSP'] and self.sm['longitudinalPlanSP'].stack.actuatedStack == StackId.customV2
    actuators.accel = float(self.LoC.update(
      CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits, long_plan.hasLead,
      custom_longitudinal_stack=custom_longitudinal_stack,
      traction_risk=get_traction_risk(self.sm['carStateSP']),
    ))

    # Steering PID loop and lateral MPC
    # Reset desired curvature to current to avoid violating the limits on engage
    stack_inputs = self.build_lateral_demand_stack_inputs(CC, CS, model_v2, lp)
    stack_output = self.lateral_demand_stack.update(stack_inputs)
    self.lateral_demand_stack_output = stack_output
    processed_lateral_demand = cast(ProcessedLateralDemand, stack_output.legacy)
    self.processed_lateral_demand = processed_lateral_demand
    self.desired_curvature = processed_lateral_demand.processed_curvature
    self.update_lateral_controller_demand(processed_lateral_demand)
    curvature_limited = processed_lateral_demand.curvature_limited
    lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS

    actuators.curvature = processed_lateral_demand.processed_curvature
    self.update_steering_actuator_feedback(CC.latActive, actuators)
    self.LaC.set_steering_actuator_feedback(self.steering_actuator_feedback)
    # Push the same-frame stack output to the controller BEFORE
    # LaC.update so v5 profile-aware preview gating, turn-exit
    # source-of-truth, and demand-mode telemetry see current-frame
    # mode/rate. The lateral demand stack is the single builder;
    # do not run a second legacy profile update here.
    self.push_lateral_demand_stack_output(stack_output, steering_pressed=CS.steeringPressed)
    steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                       self.steer_limited_by_safety, processed_lateral_demand.processed_curvature,
                                                       self.calibrated_pose, curvature_limited, lat_delay)
    actuators.torque = float(steer)
    actuators.steeringAngleDeg = float(steeringAngleDeg)
    self.toyota_eps_high_rate_frames, self.toyota_eps_cut_frames = apply_toyota_eps_high_rate_guard(
      self.CP, CC, CS, self.toyota_eps_high_rate_frames, self.toyota_eps_cut_frames
    )
    self._previous_steering_actuator_request = SteeringActuatorRequest.from_actuators(actuators)
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    return CC, lac_log

  def update_steering_actuator_feedback(self, lat_active, actuators):
    if not lat_active or not self.sm.valid['carOutput']:
      self.steering_actuator_feedback = SteeringActuatorFeedback.invalid()
    else:
      self.steering_actuator_feedback = compute_steering_actuator_feedback(
        self._previous_steering_actuator_request,
        self.sm['carOutput'].actuatorsOutput,
        self.CP.steerControlType,
        lat_active=lat_active,
      )
    self.steer_limited_by_safety = self.steering_actuator_feedback.limited

  def update_lateral_controller_inputs(self):
    update_live_torque_params = getattr(self.LaC, "update_live_torque_params", None)
    if update_live_torque_params is not None:
      torque_params = self.sm['liveTorqueParameters']
      if self.sm.all_checks(['liveTorqueParameters']) and torque_params.useParams:
        update_live_torque_params(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                  torque_params.frictionCoefficientFiltered)
        if hasattr(self.LaC, "extension"):
          update_limits = getattr(self.LaC.extension, "update_limits", None)
          if update_limits is not None:
            update_limits()

    update_model_v2 = getattr(self.LaC, "update_model_v2", None)
    if update_model_v2 is None and hasattr(self.LaC, "extension"):
      update_model_v2 = getattr(self.LaC.extension, "update_model_v2", None)
    if update_model_v2 is not None and self.sm.updated['modelV2']:
      update_model_v2(self.sm['modelV2'])

    update_lateral_lag = getattr(self.LaC, "update_lateral_lag", None)
    if update_lateral_lag is None and hasattr(self.LaC, "extension"):
      update_lateral_lag = getattr(self.LaC.extension, "update_lateral_lag", None)
    if update_lateral_lag is not None:
      try:
        lat_delay = float(self.lat_delay)
      except (TypeError, ValueError):
        lat_delay = 0.2
      if not math.isfinite(lat_delay):
        lat_delay = 0.2
      self.lat_delay = lat_delay
      update_lateral_lag(lat_delay)

  def _log_lateral_demand_stack_telemetry(self) -> None:
    if self.lateral_demand_stack_output is None or self.lateral_demand_stack_resolution is None:
      return
    output = self.lateral_demand_stack_output
    resolution = self.lateral_demand_stack_resolution
    fallback_active = bool(resolution.fallback_reason) or resolution.resolved_stack != resolution.requested_stack
    cache_key = (output.resolved_stack, output.version, fallback_active, resolution.fallback_reason)
    if self._last_logged_stack == cache_key:
      return
    cloudlog.info(
      "lateral_demand_stack resolved=%s requested=%s version=%s fallback=%s reason=%s",
      output.resolved_stack,
      output.requested_stack,
      output.version,
      fallback_active,
      resolution.fallback_reason,
    )
    self._last_logged_stack = cache_key

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and (self.CP.openpilotLongitudinalControl or not self.CP_SP.pcmCruiseSpeed)
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise)
    CC.cruiseControl.resume = CC.enabled and CS.cruiseState.standstill and not self.sm['longitudinalPlan'].shouldStop

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool((self.sm['driverMonitoringState'].alertLevel == log.DriverMonitoringState.AlertLevel.three) or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    fill_model_path_state(cs.modelPathState, self.lateral_demand_stack.model_path_result, self.lateral_demand_stack.model_path_raw_desired_curvature)

    lat_control_state = getattr(self.LaC, 'CONTROL_STATE', self.CP.lateralTuning.which())
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif lat_control_state == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_control_state == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)
    self._log_lateral_demand_stack_telemetry()

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      self.update()
      CC, lac_log = self.state_control()
      self.publish(CC, lac_log)
      self.get_params_sp(self.sm)
      self.run_ext(self.sm, self.pm)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
