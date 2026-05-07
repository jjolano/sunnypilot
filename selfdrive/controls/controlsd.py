#!/usr/bin/env python3
import math
from numbers import Number

from cereal import car, custom, log
import cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, DT_MDL, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  should_latch_lateral_accel_burst,
  update_lateral_accel_limit,
)
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import LaneChangePathShaper, LaneChangePathShaperInputs
from openpilot.selfdrive.controls.lib.model_path_processor import (
  ModelPathProcessor,
  ModelPathProcessorInputs,
  ModelPathProcessorResult,
  PATH_CURVATURE_ACTION_T,
)
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import (
  SteeringActuatorFeedback,
  SteeringActuatorRequest,
  build_steering_actuator_feedback,
)

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = custom.ModelDataV2SP.TurnDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())
TOYOTA_EPS_HIGH_RATE_DEG = 100.0
TOYOTA_EPS_HIGH_RATE_FRAMES = 15
TOYOTA_EPS_HIGH_RATE_CUT_FRAMES = 2
TOYOTA_EPS_HIGH_RATE_FINGERPRINTS = frozenset(platform.value for platform in TOYOTA)
MAP_CURVATURE_DISTANCE_WINDOW = 5.0
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
  "map_curvature_fallback": log.ControlsState.ModelPathState.Reason.mapCurvatureFallback,
}


def model_path_reason_to_capnp(reason: str):
  return MODEL_PATH_REASON_TO_CAPNP.get(reason, log.ControlsState.ModelPathState.Reason.unknown)


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
  model_path_state.mapCurvatureUsed = bool(model_path_result.map_curvature_used)


def map_curvature_at_model_horizon(live_map_data, v_ego: float, enabled: bool, action_t: float = PATH_CURVATURE_ACTION_T) -> float | None:
  if not enabled or not getattr(live_map_data, "roadCurvatureValid", False):
    return None

  try:
    distances = tuple(float(d) for d in live_map_data.roadCurvatureDistances)
    curvatures = tuple(float(c) for c in live_map_data.roadCurvatures)
    action_t = float(action_t)
  except (AttributeError, TypeError, ValueError):
    return None

  if not distances or len(distances) != len(curvatures):
    return None
  v_ego = float(v_ego)
  if not math.isfinite(v_ego):
    return None
  if not math.isfinite(action_t) or action_t < 0.0:
    return None
  if not all(math.isfinite(d) and d >= 0.0 for d in distances) or not all(math.isfinite(c) for c in curvatures):
    return None
  if any(next_distance < distance for distance, next_distance in zip(distances, distances[1:])):
    return None

  target_distance = max(0.0, v_ego) * action_t
  if target_distance > distances[-1] + MAP_CURVATURE_DISTANCE_WINDOW:
    return None
  if target_distance <= distances[0]:
    return curvatures[0]

  for i in range(1, len(distances)):
    if target_distance <= distances[i]:
      span = distances[i] - distances[i - 1]
      if span <= 0.0:
        return curvatures[i]
      alpha = (target_distance - distances[i - 1]) / span
      return curvatures[i - 1] + alpha * (curvatures[i] - curvatures[i - 1])

  return curvatures[-1]


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
                                    'liveCalibration', 'livePose', 'longitudinalPlan', 'lateralManeuverPlan', 'carState', 'carOutput',
                                    'driverMonitoringState', 'onroadEvents', 'driverAssistance', 'liveDelay', 'liveMapDataSP'] + self.sm_services_ext,
                                   poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'] + self.pm_services_ext)

    self.steer_limited_by_safety = False
    self.toyota_eps_high_rate_frames = 0
    self.toyota_eps_cut_frames = 0
    self.steering_actuator_feedback = SteeringActuatorFeedback.invalid()
    self._previous_steering_actuator_request: SteeringActuatorRequest | None = None
    self.curvature = 0.0
    self.desired_curvature = 0.0
    self.lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
    self.default_lateral_accel_limited = False
    self.lane_change_path_shaper = LaneChangePathShaper(DT_CTRL)
    self.model_path_processor = ModelPathProcessor()
    self.model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
    self.model_path_raw_desired_curvature = 0.0

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
    actuators.accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits, long_plan.hasLead))

    # Steering PID loop and lateral MPC
    # Reset desired curvature to current to avoid violating the limits on engage
    lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    lateral_maneuver_curvature = self.get_lateral_maneuver_curvature(CC.latActive)
    model_path_raw_curvature = float(model_v2.action.desiredCurvature)
    if lateral_maneuver_curvature is not None:
      self.lane_change_path_shaper.reset()
      self.model_path_processor.reset()
      new_desired_curvature = lateral_maneuver_curvature
      self.model_path_result = ModelPathProcessorResult(lateral_maneuver_curvature, 0.0, True, "lateral_maneuver")
      self.model_path_raw_desired_curvature = model_path_raw_curvature
    else:
      turn_curvature_sign = 0
      if model_v2.meta.laneChangeState == LaneChangeState.off and self.sm.valid['modelDataV2SP']:
        turn_direction = self.sm['modelDataV2SP'].laneTurnDirection
        if turn_direction == TurnDirection.turnRight:
          turn_curvature_sign = 1
        elif turn_direction == TurnDirection.turnLeft:
          turn_curvature_sign = -1

      map_curvature = map_curvature_at_model_horizon(
        self.sm['liveMapDataSP'],
        CS.vEgo,
        self.lateral_map_curvature_fallback_enabled and self.sm.all_checks(['liveMapDataSP']) and model_v2.meta.laneChangeState == LaneChangeState.off,
        action_t=lat_delay + DT_MDL * 1.5,
      )
      path_result = self.model_path_processor.update(
        ModelPathProcessorInputs(
          lat_active=CC.latActive,
          v_ego=CS.vEgo,
          desired_curvature=model_v2.action.desiredCurvature,
          measured_curvature=self.curvature,
          previous_desired_curvature=self.desired_curvature,
          position_x=tuple(model_v2.position.x),
          position_y=tuple(model_v2.position.y),
          position_y_std=tuple(model_v2.position.yStd),
          orientation_z=tuple(model_v2.orientation.z),
          orientation_rate_z=tuple(model_v2.orientationRate.z),
          lane_line_probs=tuple(model_v2.laneLineProbs),
          turn_curvature_sign=turn_curvature_sign,
          frame_drop_perc=model_v2.frameDropPerc,
          map_curvature_enabled=map_curvature is not None,
          map_curvature=map_curvature,
        )
      )
      self.model_path_result = path_result
      self.model_path_raw_desired_curvature = model_path_raw_curvature
      model_desired_curvature = path_result.desired_curvature if CC.latActive else self.curvature
      left_lane_y0 = model_v2.laneLines[1].y[0] if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[1].y) else None
      right_lane_y0 = model_v2.laneLines[2].y[0] if len(model_v2.laneLines) > 2 and len(model_v2.laneLines[2].y) else None
      lane_change_result = self.lane_change_path_shaper.update(
        LaneChangePathShaperInputs(
          lat_active=CC.latActive,
          v_ego=CS.vEgo,
          left_blinker=CS.leftBlinker,
          right_blinker=CS.rightBlinker,
          steering_pressed=CS.steeringPressed,
          lane_change_state=model_v2.meta.laneChangeState,
          lane_change_direction=model_v2.meta.laneChangeDirection,
          model_curvature=model_desired_curvature,
          prev_desired_curvature=self.desired_curvature if CC.latActive else self.curvature,
          lane_line_probs=tuple(model_v2.laneLineProbs),
          left_lane_y0=left_lane_y0,
          right_lane_y0=right_lane_y0,
        )
      )
      new_desired_curvature = lane_change_result.desired_curvature if CC.latActive else self.curvature
    manual_gas_lateral_accel_override = CS.gasPressed and not CC.longActive
    self.lateral_accel_limit_no_roll = update_lateral_accel_limit(
      self.lateral_accel_limit_no_roll,
      manual_gas_lateral_accel_override,
      CC.latActive,
      CS.brakePressed,
      CS.steeringPressed,
      default_lateral_accel_limited=self.default_lateral_accel_limited,
    )
    self.desired_curvature, curvature_limited, default_lateral_accel_limited = clip_curvature(
      CS.vEgo,
      self.desired_curvature,
      new_desired_curvature,
      lp.roll,
      self.lateral_accel_limit_no_roll,
    )
    self.default_lateral_accel_limited = should_latch_lateral_accel_burst(
      default_lateral_accel_limited,
      CC.latActive,
      CS.brakePressed,
      CS.steeringPressed,
      manual_gas_lateral_accel_override,
    )
    actuators.curvature = self.desired_curvature
    self.update_steering_actuator_feedback(CC.latActive, actuators)
    self.LaC.set_steering_actuator_feedback(self.steering_actuator_feedback)
    steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                       self.steer_limited_by_safety, self.desired_curvature,
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
      update_lateral_lag(self.lat_delay)

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
    cs.forceDecel = bool((self.sm['driverMonitoringState'].awarenessStatus < 0.) or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    fill_model_path_state(cs.modelPathState, self.model_path_result, self.model_path_raw_desired_curvature)

    lat_control_state = getattr(self.LaC, 'CONTROL_STATE', self.CP.lateralTuning.which())
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif lat_control_state == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_control_state == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)

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
