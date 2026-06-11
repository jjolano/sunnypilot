#!/usr/bin/env python3
from dataclasses import replace
import math
import numpy as np

from cereal import car, custom, log
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  LongitudinalArbiter,
  LongitudinalDecisionTelemetry,
  apply_longitudinal_decision_output_with_telemetry,
  build_core_longitudinal_candidates,
  get_active_lead_confidence,
  resolve_longitudinal_decision,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.adapters import planner_state_to_stack_output
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomV2Scene, ONE_PEDAL_MODE_OFF, ONE_PEDAL_MODES
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
  PLANNER_SEED_CAP,
  PLANNER_SEED_FLOOR,
  PLANNER_SEED_MPC_REASON,
  PlannerSeedCandidate,
  planner_seed_intent_for_reason,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import is_custom_stack
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, SunnypilotLongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE,
  LEAD_CRAWL_ACCEL_LIMIT,
  get_T_FOLLOW,
  get_lead_accel_recovery_a_min,
  get_lead_approach_gaps,
  get_lead_crawl_accel_max,
  get_lead_stop_presentation_distance,
  get_lead_stop_runway_required_decel,
  get_moving_lead_stop_approach_comfort_target,
)
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.controls.lib.lateral_accel import lateral_accel_from_steering_angle
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.vehicle_math import stopping_decel

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ButtonType = car.CarState.ButtonEvent.Type
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
ONE_PEDAL_LONGITUDINAL_MODE_PARAM = "OnePedalLongitudinalMode"
ONE_PEDAL_CRUISE_HOLD_BUTTON_TYPES = frozenset((
  ButtonType.accelCruise,
  ButtonType.decelCruise,
  ButtonType.resumeCruise,
  ButtonType.setCruise,
))
DECISION_ACCEL_COMFORT_MIN_V_EGO = 1.0
CREEP_TO_STOP_GAP_START_EXCESS = 1.2
CREEP_TO_STOP_GAP_FOLLOW_EXCESS = 1.0
CREEP_TO_STOP_GAP_ARM_EXCESS = CREEP_TO_STOP_GAP_FOLLOW_EXCESS + 0.05
CREEP_TO_STOP_GAP_STOP_EXCESS = 0.05
CREEP_TO_STOP_GAP_MAX_V_EGO_ARM = 0.3
CREEP_TO_STOP_GAP_MAX_V_EGO = 1.0
# Treat pullaway creep as a near stopped-gap behavior; farther leads return to normal MPC handling.
CREEP_TO_STOP_GAP_MAX_EXCESS = 4.0
CREEP_TO_STOP_GAP_MIN_LEAD_SPEED = -0.3
CREEP_TO_STOP_GAP_MIN_MODEL_PROB = 0.5
CREEP_TO_STOP_GAP_SPEED_MAX = 0.75
CREEP_TO_STOP_GAP_SPEED_BP = [
  CREEP_TO_STOP_GAP_STOP_EXCESS,
  CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
  CREEP_TO_STOP_GAP_START_EXCESS,
  CREEP_TO_STOP_GAP_MAX_EXCESS,
]
CREEP_TO_STOP_GAP_SPEED_V = [0.0, 0.16, 0.30, CREEP_TO_STOP_GAP_SPEED_MAX]
CREEP_TO_STOP_GAP_ACCEL_GAIN = 1.0
CREEP_TO_STOP_GAP_ACCEL_MIN = -0.25
CREEP_TO_STOP_GAP_ACCEL_MAX = 0.18
CREEP_TO_STOP_GAP_HOLD_BUFFER = 0.5
CREEP_TO_STOP_GAP_HOLD_DECEL_CAP = 1.8
CREEP_TO_STOP_GAP_HOLD_EXCESS = 0.35
CREEP_TO_STOP_GAP_REHOLD_EXCESS = 0.2
CREEP_TO_STOP_GAP_HOLD_RELEASE_EXCESS = CREEP_TO_STOP_GAP_FOLLOW_EXCESS
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_SPEED = 0.05
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_ACCEL = 0.15
CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED = 0.15
CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX = 1.2
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX = 0.55
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN = 0.30
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN = 0.70
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX = 0.75
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MAX = 1.20
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS = CREEP_TO_STOP_GAP_STOP_EXCESS
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MAX_EXCESS = 8.0
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MIN_LEAD_ACCEL = 0.6
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP = 7.5 * DT_MDL
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO = 3.0
CREEP_TO_STOP_GAP_PREDICT_T = 0.8
CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED = 0.35
CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL = 0.15
CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING = 0.2
CREEP_TO_STOP_GAP_PREDICT_ARM_EXCESS = 0.35
CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB = 0.75
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_DIST_ERROR = 1.5
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_ERROR = 1.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD = 1.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_Y_ERROR = 0.75
CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_ERROR = 1.25
CREEP_TO_STOP_GAP_MODEL_LEAD_HORIZON = 2.0
CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET = 1.52
STOPPED_LEAD_STOP_GAP_GUARD_MAX_V_EGO = 25.0
STOPPED_LEAD_STOP_GAP_GUARD_MAX_LEAD_SPEED = 0.15
STOPPED_LEAD_STOP_GAP_GUARD_MAX_LEAD_ACCEL = 0.05
STOPPED_LEAD_STOP_GAP_GUARD_EXCESS = 100.0
STOPPED_LEAD_STOP_GAP_GUARD_TARGET_BUFFER = -0.75
STOPPED_LEAD_STOP_GAP_GUARD_MIN_REQUIRED_DECEL = 0.1
STOPPED_LEAD_STOP_GAP_GUARD_DECEL_CAP_BP = [4.0, 8.0]
STOPPED_LEAD_STOP_GAP_GUARD_DECEL_CAP_V = [1.1, 2.0]
MOVING_LEAD_STOP_GAP_GUARD_MIN_V_EGO = 6.0
MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD = 0.5
MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL = 0.25
MOVING_LEAD_STOP_GAP_GUARD_MIN_TARGET_DECEL = 0.4
MOVING_LEAD_STOP_GAP_GUARD_MAX_Y_REL = 1.5
MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP = 1.95
MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL = 2.0
LEAD_STOP_APPROACH_DECEL_SLEW_MIN_V_EGO = 3.0
LEAD_STOP_APPROACH_DECEL_SLEW_MIN_LEAD_DECEL = 0.6
LEAD_STOP_APPROACH_DECEL_SLEW_STOPPED_LEAD_V = 0.2
LEAD_STOP_APPROACH_DECEL_SLEW_MIN_GAP_EXCESS = 10.0
LEAD_STOP_APPROACH_DECEL_SLEW_MAX_JERK = 7.5
LEAD_LOSS_E2E_GUARD_TIME = 3.0
LEAD_LOSS_E2E_GUARD_ACCEL_FLOOR = -0.45
LEAD_LOSS_E2E_GUARD_MIN_D_REL = 45.0
LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB = 0.8
STOPPED_LEAD_GAP_FILL_ARM_TIME = 8.0
STOPPED_LEAD_GAP_FILL_ARM_MAX_V_EGO = 0.3
STOPPED_LEAD_GAP_FILL_ARM_MAX_GAP_EXCESS = 0.6
STOPPED_LEAD_GAP_FILL_ARM_MAX_LEAD_SPEED = 0.25
# Start stopped-lead gap fill once near pullaway creep stops chasing the lead.
STOPPED_LEAD_GAP_FILL_MIN_EXCESS = CREEP_TO_STOP_GAP_MAX_EXCESS
STOPPED_LEAD_GAP_FILL_MAX_EXCESS = 35.0
STOPPED_LEAD_GAP_FILL_MAX_V_EGO = 2.5
STOPPED_LEAD_GAP_FILL_MAX_LEAD_SPEED = 1.0
STOPPED_LEAD_GAP_FILL_MIN_MODEL_PROB = 0.75
STOPPED_LEAD_GAP_FILL_MIN_LEAD_ACCEL = -0.05
STOPPED_LEAD_GAP_FILL_SPEED_MAX = 1.5
STOPPED_LEAD_GAP_FILL_SPEED_BP = [STOPPED_LEAD_GAP_FILL_MIN_EXCESS, 20.0, STOPPED_LEAD_GAP_FILL_MAX_EXCESS]
STOPPED_LEAD_GAP_FILL_SPEED_V = [CREEP_TO_STOP_GAP_SPEED_MAX, 1.2, STOPPED_LEAD_GAP_FILL_SPEED_MAX]
STOPPED_LEAD_GAP_FILL_ACCEL_GAIN = 0.6
STOPPED_LEAD_GAP_FILL_ACCEL_MAX = 0.35
STOPPED_LEAD_GAP_FILL_ACCEL_MIN = -0.25
ENGAGE_STOP_BOOTSTRAP_TIME = 0.75
ENGAGE_STOP_BOOTSTRAP_MIN_SPEED = 5.0
ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL = -1.0
E2E_STOP_APPROACH_MIN_V_EGO = 3.0
E2E_STOP_APPROACH_MAX_MODEL_ACCEL = 0.2
E2E_STOP_APPROACH_MIN_ENDPOINT = 5.0
E2E_STOP_APPROACH_CRAWL_RESERVE = 2.0
E2E_STOP_APPROACH_PROTECTION_MIN_V_EGO = 2.0
E2E_STOP_APPROACH_EXPECTED_DIST_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0]
E2E_STOP_APPROACH_EXPECTED_DIST_V = [8.0, 18.0, 30.0, 43.0, 58.0, 74.0, 85.0, 96.0]
E2E_STOP_APPROACH_SHORTAGE_BP = [0.05, 0.5]
E2E_STOP_APPROACH_DECEL_BP = [0.18, 1.35]
E2E_STOP_APPROACH_MAX_DECEL_SHORTAGE = 0.15
E2E_STOP_APPROACH_REQUIRED_DECEL_SHORTAGE_BP = [0.12, 0.3]
E2E_STOP_APPROACH_REQUIRED_DECEL_BLEND = 0.65
E2E_STOP_APPROACH_DECEL_MAX = 1.5
CRUISE_COAST_FLAT_OVERSPEED = 0.45  # ~1 mph
CRUISE_COAST_DOWNHILL_OVERSPEED = 1.35  # ~3 mph
CRUISE_COAST_DOWNHILL_ACCEL = 0.25
CRUISE_COAST_RECOVERY_OVERSPEED = 0.9  # ~2 mph from coast back to normal decel
E2E_STOP_APPROACH_MODEL_STOP_ENDPOINT_MARGIN = 5.0
E2E_STOP_APPROACH_CLOSE_ENDPOINT_DECEL = -1.0
E2E_RUNWAY_COMFORT_MIN_V_EGO = 3.0
E2E_RUNWAY_COMFORT_MIN_ENDPOINT = 1.0
E2E_RUNWAY_COMFORT_COAST_MARGIN = 0.02
E2E_RUNWAY_COMFORT_LIGHT_DECEL = 0.30
E2E_RUNWAY_COMFORT_DECEL_BLEND_BP = [1.2, 2.0]
E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP = [0.5, 1.0]
E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE = 0.35
E2E_RUNWAY_POSITIVE_CAP_REF_ACCEL = 0.45
E2E_RUNWAY_POSITIVE_CAP_PREVIEW_T = 6.0
E2E_RUNWAY_POSITIVE_CAP_MAX_ENDPOINT_V = 1.0
E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX = 0.08
E2E_CLOSE_STOP_MAX_DIST = 1.0
E2E_CLOSE_STOP_RELEASE_DIST = 1.0
E2E_CLOSE_STOP_SHOULD_STOP_DIST = 0.4
E2E_CLOSE_STOP_MIN_ROLLING_V = 0.25
E2E_CLOSE_STOP_SHOULD_STOP_MAX_V = 1.0
E2E_CLOSE_STOP_DECEL_BUFFER = 0.25
E2E_CLOSE_STOP_DECEL_MAX = 0.8
ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED = 1.0
STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_D_REL_DELTA = 3.0
STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_V_LEAD_DELTA = 1.0

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20.0, 40.0]


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def has_valid_radar_lead(radar_state):
  return radar_state.leadOne.status or radar_state.leadTwo.status


def has_model_stop_context(model_msg):
  if model_msg.action.shouldStop:
    return True

  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  return any(x > 0.0 and v <= ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED for x, v in zip(positions, velocities, strict=False))


def should_enable_longitudinal_decision_layer(stack_resolution) -> bool:
  return stack_resolution is None or is_custom_stack(getattr(stack_resolution, "resolved_stack", ""))


def get_one_pedal_longitudinal_mode(params) -> int:
  try:
    mode = int(params.get(ONE_PEDAL_LONGITUDINAL_MODE_PARAM, return_default=True))
  except (TypeError, ValueError, UnknownKeyName):
    return ONE_PEDAL_MODE_OFF
  return mode if mode in ONE_PEDAL_MODES else ONE_PEDAL_MODE_OFF


def one_pedal_cruise_hold_requested(button_events) -> bool:
  return any(getattr(event, "type", None) in ONE_PEDAL_CRUISE_HOLD_BUTTON_TYPES for event in button_events)


def update_one_pedal_cruise_hold(active: bool, button_events, gas_pressed: bool, brake_pressed: bool, enabled: bool) -> bool:
  if not enabled or gas_pressed or brake_pressed:
    return False
  return bool(active or one_pedal_cruise_hold_requested(button_events))


def get_custom_v2_curve_scene_target(*controllers):
  active_targets = []
  for controller in controllers:
    if not bool(getattr(controller, "is_active", False)):
      continue
    try:
      target = float(getattr(controller, "output_a_target", 0.0))
    except (TypeError, ValueError):
      continue
    if math.isfinite(target):
      active_targets.append(target)
  return bool(active_targets), float(min(active_targets, default=0.0))


def get_model_stop_distance(model_msg):
  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  for idx, (x, v) in enumerate(zip(positions, velocities, strict=False)):
    if idx > 0 and x >= 0.0 and v <= ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED:
      return float(x)
  return None


def get_e2e_confirmed_model_stop_distance(model_msg):
  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  if len(positions) < 3 or len(velocities) != len(positions):
    return None

  endpoint_x = float(positions[-1])
  for idx, (x, v) in enumerate(zip(positions, velocities, strict=False)):
    if idx == 0 or idx == len(positions) - 1:
      continue
    if x >= 0.0 and v <= ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED and endpoint_x - x >= E2E_STOP_APPROACH_MODEL_STOP_ENDPOINT_MARGIN:
      return float(x)
  return None


def should_run_engage_stop_bootstrap(timer, v_ego, radar_state, model_msg):
  if timer <= 0.0 or v_ego < ENGAGE_STOP_BOOTSTRAP_MIN_SPEED or has_valid_radar_lead(radar_state):
    return False

  return bool(
    model_msg.action.shouldStop or
    (model_msg.action.desiredAcceleration <= ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL and has_model_stop_context(model_msg))
  )


def get_e2e_stop_approach_accel(v_ego, model_msg, radar_state, e2e_active, force_slow_decel=False,
                                brake_pressed=False, gas_pressed=False, model_stop_protection_active=False):
  protection_active = e2e_active or model_stop_protection_active
  min_v_ego = E2E_STOP_APPROACH_PROTECTION_MIN_V_EGO if model_stop_protection_active else E2E_STOP_APPROACH_MIN_V_EGO
  blocked = not protection_active or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or v_ego < min_v_ego or has_valid_radar_lead(radar_state)
  blocked = blocked or model_msg.action.shouldStop or model_msg.action.desiredAcceleration > E2E_STOP_APPROACH_MAX_MODEL_ACCEL
  blocked = blocked or len(model_msg.position.x) == 0
  if blocked:
    return 0.0
  endpoint_x = float(model_msg.position.x[-1])
  if not np.isfinite(endpoint_x) or endpoint_x <= 0.0:
    return 0.0

  stop_distance = get_e2e_confirmed_model_stop_distance(model_msg)
  close_endpoint_stop = (
    endpoint_x <= E2E_STOP_APPROACH_MIN_ENDPOINT + E2E_STOP_APPROACH_CRAWL_RESERVE and
    model_msg.action.desiredAcceleration <= E2E_STOP_APPROACH_CLOSE_ENDPOINT_DECEL
  )
  if stop_distance is None and not close_endpoint_stop:
    return 0.0

  approach_distance = endpoint_x
  if stop_distance is not None and np.isfinite(stop_distance) and stop_distance > 0.0:
    approach_distance = min(endpoint_x, max(E2E_STOP_APPROACH_MIN_ENDPOINT, stop_distance - E2E_STOP_APPROACH_CRAWL_RESERVE))

  expected_distance = float(np.interp(v_ego * CV.MS_TO_KPH, E2E_STOP_APPROACH_EXPECTED_DIST_BP, E2E_STOP_APPROACH_EXPECTED_DIST_V))
  max_decel_distance = v_ego**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX * (1.0 - E2E_STOP_APPROACH_MAX_DECEL_SHORTAGE))
  expected_distance = max(expected_distance, max_decel_distance)
  if expected_distance <= 0.0:
    return 0.0

  shortage = max(0.0, expected_distance - approach_distance) / expected_distance
  if shortage <= E2E_STOP_APPROACH_SHORTAGE_BP[0]:
    return 0.0

  shortage_decel = float(np.interp(shortage, E2E_STOP_APPROACH_SHORTAGE_BP, E2E_STOP_APPROACH_DECEL_BP))
  required_decel_blend = float(np.interp(shortage, E2E_STOP_APPROACH_REQUIRED_DECEL_SHORTAGE_BP, [0.0, 1.0]))
  required_decel = required_decel_blend * E2E_STOP_APPROACH_REQUIRED_DECEL_BLEND * -stopping_decel(
    v_ego, endpoint_x, min_distance=E2E_STOP_APPROACH_MIN_ENDPOINT,
  )
  target_decel = min(max(shortage_decel, required_decel), E2E_STOP_APPROACH_DECEL_MAX)
  return -target_decel


def get_cruise_coast_overspeed_leeway(accel_coast):
  return float(np.interp(accel_coast, [0.0, CRUISE_COAST_DOWNHILL_ACCEL],
                         [CRUISE_COAST_FLAT_OVERSPEED, CRUISE_COAST_DOWNHILL_OVERSPEED]))


def apply_cruise_coast_overspeed(v_ego, v_cruise, accel_coast, a_target):
  overspeed = v_ego - v_cruise
  if overspeed <= 0.0:
    return a_target

  leeway = get_cruise_coast_overspeed_leeway(accel_coast)
  recovery_blend = float(np.clip((overspeed - leeway) / CRUISE_COAST_RECOVERY_OVERSPEED, 0.0, 1.0))
  coast_target = (1.0 - recovery_blend) * accel_coast + recovery_blend * a_target
  return min(0.0, max(a_target, coast_target))


def build_planner_seed_accel_candidate(planner, name, a_target, has_lead, reason, accel_limits, should_stop=None,
                                        selection=PLANNER_SEED_CAP, force=False, group=""):
  candidate_a_target = float(np.clip(a_target, accel_limits[0], accel_limits[1]))
  baseline_should_stop = bool(getattr(planner, "output_should_stop", False))
  candidate_should_stop = bool(baseline_should_stop if should_stop is None else should_stop)
  if force:
    pass
  elif selection == PLANNER_SEED_FLOOR:
    if candidate_a_target <= planner.output_a_target and not (baseline_should_stop and not candidate_should_stop):
      return None
  elif candidate_a_target >= planner.output_a_target and not (candidate_should_stop and not baseline_should_stop):
    return None
  base_output = getattr(planner, "planner_seed_candidate_base_output", None)
  if base_output is None:
    base_output = planner_state_to_stack_output(planner, has_lead)
  seed_intent = planner_seed_intent_for_reason(reason, has_lead, candidate_should_stop, base_output.source)
  output = replace(
    base_output,
    a_target=candidate_a_target,
    should_stop=candidate_should_stop,
    debug={"planner_seed_candidate_reason": reason},
    seed_intent=seed_intent,
    seed_reason=reason,
  )
  return PlannerSeedCandidate(name, output, selection=selection, group=group, intent=seed_intent, reason=reason)


def build_planner_seed_mpc_candidate(planner, mpc, a_target, should_stop, has_lead, accel_limits, speeds, accels, jerks, fcw):
  candidate_a_target = float(np.clip(a_target, accel_limits[0], accel_limits[1]))
  baseline_should_stop = bool(getattr(planner, "output_should_stop", False))
  candidate_should_stop = bool(should_stop)
  if np.isclose(candidate_a_target, planner.output_a_target) and candidate_should_stop == baseline_should_stop:
    return None

  if candidate_a_target > planner.output_a_target and not candidate_should_stop:
    if mpc.source in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1):
      return None
    selection = PLANNER_SEED_FLOOR
  else:
    selection = PLANNER_SEED_CAP
  seed_intent = planner_seed_intent_for_reason(PLANNER_SEED_MPC_REASON, has_lead, candidate_should_stop, mpc.source)
  output = LongitudinalStackOutput(
    a_target=candidate_a_target,
    should_stop=candidate_should_stop,
    has_lead=bool(has_lead),
    source=mpc.source,
    allow_throttle=bool(getattr(planner, "allow_throttle", True)),
    allow_brake=True,
    speeds=tuple(float(v) for v in speeds),
    accels=tuple(float(a) for a in accels),
    jerks=tuple(float(j) for j in jerks),
    fcw=bool(fcw),
    debug={"planner_seed_candidate_reason": PLANNER_SEED_MPC_REASON},
    seed_intent=seed_intent,
    seed_reason=PLANNER_SEED_MPC_REASON,
  )
  return PlannerSeedCandidate(PLANNER_SEED_MPC_REASON, output, selection=selection, group=PLANNER_SEED_MPC_REASON,
                              intent=seed_intent, reason=PLANNER_SEED_MPC_REASON)


def should_apply_cruise_coast_overspeed(reset_state, force_slow_decel, e2e_active, _has_lead, should_stop, source):
  # A radar lead can be present while the selected longitudinal source is still plain cruise.
  # Keep the overspeed comfort cap active in that case so lead flicker/reacquisition cannot
  # reintroduce positive cruise accel above the set speed.
  return bool(
    not reset_state
    and not force_slow_decel
    and not e2e_active
    and not should_stop
    and source == custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise
  )


def get_e2e_close_stop_settle(v_ego, raw_e2e_accel, model_msg, radar_state, e2e_active, active=False,
                              force_slow_decel=False, brake_pressed=False, gas_pressed=False, reset_state=False):
  blocked = not e2e_active or reset_state or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or has_valid_radar_lead(radar_state) or model_msg.action.desiredAcceleration > 0.0
  if blocked:
    return raw_e2e_accel, False, False

  stop_distance = get_model_stop_distance(model_msg)
  if stop_distance is None or not np.isfinite(stop_distance) or stop_distance < 0.0:
    return raw_e2e_accel, False, False

  if stop_distance > (E2E_CLOSE_STOP_RELEASE_DIST if active else E2E_CLOSE_STOP_MAX_DIST):
    return raw_e2e_accel, False, False

  should_stop = bool(model_msg.action.shouldStop or (
    stop_distance <= E2E_CLOSE_STOP_SHOULD_STOP_DIST and v_ego <= E2E_CLOSE_STOP_SHOULD_STOP_MAX_V
  ))
  if v_ego < E2E_CLOSE_STOP_MIN_ROLLING_V:
    return raw_e2e_accel, should_stop, should_stop

  required_decel = -stopping_decel(
    v_ego, stop_distance + E2E_CLOSE_STOP_DECEL_BUFFER, min_distance=E2E_CLOSE_STOP_DECEL_BUFFER,
  )
  target_decel = min(required_decel, E2E_CLOSE_STOP_DECEL_MAX)
  return min(raw_e2e_accel, -target_decel), should_stop, True


def get_e2e_runway_comfort_accel(v_ego, raw_e2e_accel, coast_accel, model_msg, e2e_active, prev_output_a_target,
                                 reset_state=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
                                 engage_stop_bootstrap_active=False, has_radar_lead=False, dt=DT_MDL):
  blocked = not e2e_active or reset_state or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or engage_stop_bootstrap_active or has_radar_lead
  blocked = blocked or model_msg.action.shouldStop or v_ego < E2E_RUNWAY_COMFORT_MIN_V_EGO
  blocked = blocked or raw_e2e_accel >= coast_accel or len(model_msg.position.x) == 0
  if blocked:
    return raw_e2e_accel

  endpoint_x = float(model_msg.position.x[-1])
  if not np.isfinite(endpoint_x) or endpoint_x <= E2E_RUNWAY_COMFORT_MIN_ENDPOINT:
    return raw_e2e_accel

  expected_distance = float(np.interp(v_ego * CV.MS_TO_KPH, E2E_STOP_APPROACH_EXPECTED_DIST_BP, E2E_STOP_APPROACH_EXPECTED_DIST_V))
  model_expected_distance = expected_distance
  max_decel_distance = v_ego**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX * (1.0 - E2E_STOP_APPROACH_MAX_DECEL_SHORTAGE))
  expected_distance = max(expected_distance, max_decel_distance)
  if expected_distance <= 0.0:
    return raw_e2e_accel

  required_decel = v_ego**2 / (2.0 * endpoint_x)
  runway_ratio = endpoint_x / expected_distance
  urgency_blend = float(np.interp(required_decel, E2E_RUNWAY_COMFORT_DECEL_BLEND_BP, [0.0, 1.0]))
  has_model_expected_runway = endpoint_x >= model_expected_distance
  if has_model_expected_runway:
    urgency_blend = 0.0
  runway_blend = float(np.interp(runway_ratio, E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP, [1.0, 0.0]))
  if has_model_expected_runway:
    runway_blend = 0.0
  blend = max(urgency_blend, runway_blend)
  if blend >= 1.0:
    return raw_e2e_accel

  light_decel_cap = min(coast_accel - E2E_RUNWAY_COMFORT_COAST_MARGIN, -E2E_RUNWAY_COMFORT_LIGHT_DECEL)
  comfort_cap = (1.0 - blend) * light_decel_cap + blend * raw_e2e_accel
  governed_accel = max(raw_e2e_accel, comfort_cap)

  max_negative_step = E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE * max(dt, 0.0)
  if np.isfinite(prev_output_a_target):
    governed_accel = max(governed_accel, prev_output_a_target - max_negative_step)
  return governed_accel


def get_e2e_runway_positive_accel_cap(v_ego, model_msg, e2e_active, reset_state=False, force_slow_decel=False,
                                      brake_pressed=False, gas_pressed=False, engage_stop_bootstrap_active=False,
                                      has_radar_lead=False, model_stop_protection_active=False):
  blocked = not (e2e_active or model_stop_protection_active) or reset_state or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or engage_stop_bootstrap_active or has_radar_lead or v_ego >= E2E_RUNWAY_COMFORT_MIN_V_EGO
  blocked = blocked or len(model_msg.position.x) == 0 or len(model_msg.velocity.x) == 0
  if blocked:
    return ACCEL_MAX

  endpoint_x = float(model_msg.position.x[-1])
  endpoint_v = float(model_msg.velocity.x[-1])
  if not np.isfinite(endpoint_x) or not np.isfinite(endpoint_v) or endpoint_x < 0.0:
    return ACCEL_MAX
  if endpoint_x <= E2E_RUNWAY_COMFORT_MIN_ENDPOINT:
    return E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX if endpoint_v <= E2E_RUNWAY_POSITIVE_CAP_MAX_ENDPOINT_V else ACCEL_MAX
  if not model_msg.action.shouldStop and endpoint_v > E2E_RUNWAY_POSITIVE_CAP_MAX_ENDPOINT_V:
    return ACCEL_MAX

  preview_distance = v_ego * E2E_RUNWAY_POSITIVE_CAP_PREVIEW_T + 0.5 * E2E_RUNWAY_POSITIVE_CAP_REF_ACCEL * E2E_RUNWAY_POSITIVE_CAP_PREVIEW_T**2
  preview_speed = v_ego + E2E_RUNWAY_POSITIVE_CAP_REF_ACCEL * E2E_RUNWAY_POSITIVE_CAP_PREVIEW_T
  no_cap_runway = preview_distance + preview_speed**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX)
  if no_cap_runway <= 0.0:
    return ACCEL_MAX

  usable_runway = max(0.0, endpoint_x - E2E_RUNWAY_COMFORT_MIN_ENDPOINT)
  if usable_runway >= no_cap_runway:
    return ACCEL_MAX

  endpoint_stop_speed = math.sqrt(2.0 * E2E_STOP_APPROACH_DECEL_MAX * usable_runway)
  if endpoint_v > endpoint_stop_speed:
    return ACCEL_MAX

  runway_ratio = usable_runway / no_cap_runway
  return min(ACCEL_MAX, E2E_STOP_APPROACH_DECEL_MAX * runway_ratio)


def has_confirmed_radar_lead(radar_state):
  return any(
    getattr(lead, "status", False) and float(getattr(lead, "modelProb", 0.0)) >= LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB
    for lead in (radar_state.leadOne, radar_state.leadTwo)
  )


def is_lane_change_active(model_msg):
  return model_msg.meta.laneChangeState != log.LaneChangeState.off


def update_lead_loss_e2e_guard_timer(timer, dt, previous_lead_status, previous_d_rel, previous_model_prob,
                                     current_has_lead, lane_change_active, reset_state=False, force_slow_decel=False,
                                     brake_pressed=False, gas_pressed=False):
  blocked = reset_state or force_slow_decel or brake_pressed or gas_pressed or current_has_lead
  if blocked:
    return 0.0

  lost_far_confirmed_lead = (
    previous_lead_status and
    previous_d_rel >= LEAD_LOSS_E2E_GUARD_MIN_D_REL and
    previous_model_prob >= LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB
  )
  if lost_far_confirmed_lead and lane_change_active:
    return LEAD_LOSS_E2E_GUARD_TIME

  return max(0.0, timer - dt)


def get_lead_loss_e2e_guard_lead(radar_state):
  return max(
    (lead for lead in (radar_state.leadOne, radar_state.leadTwo)
     if getattr(lead, "status", False) and float(getattr(lead, "dRel", 0.0)) >= LEAD_LOSS_E2E_GUARD_MIN_D_REL and
     float(getattr(lead, "modelProb", 0.0)) >= LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB),
    key=lambda lead: float(lead.dRel),
    default=None,
  )


def apply_lead_loss_e2e_guard_accel(e2e_accel, e2e_should_stop, timer, has_lead):
  if timer <= 0.0 or has_lead or e2e_should_stop:
    return e2e_accel
  return max(e2e_accel, LEAD_LOSS_E2E_GUARD_ACCEL_FLOOR)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP, control_calculation_hardening=False,
                         vehicle_model=None, roll=0.0, accurate_lateral_accel=False):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  if accurate_lateral_accel and vehicle_model is not None:
    a_y = lateral_accel_from_steering_angle(v_ego, angle_steers * CV.DEG_TO_RAD, vehicle_model, roll)
  elif control_calculation_hardening:
    a_y = v_ego**2 * VehicleModel(CP).calc_curvature(angle_steers * CV.DEG_TO_RAD, v_ego, 0.0)
  else:
    a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.0))

  return [a_target[0], min(a_target[1], a_x_allowed)]


def get_predicted_lead_pullaway(v_lead, a_lead, a_lead_tau, horizon=CREEP_TO_STOP_GAP_PREDICT_T):
  steps = 4
  ts = np.linspace(horizon / steps, horizon, steps)
  dt = horizon / steps
  a_traj = a_lead * np.exp(-max(a_lead_tau, 0.0) * (ts**2) / 2.0)
  v_traj = np.clip(v_lead + np.cumsum(dt * a_traj), 0.0, 1e8)
  return float(v_traj[-1]), float(np.sum(dt * v_traj))


def has_predicted_lead_pullaway(gap_excess, predicted_v_lead, predicted_gap_opening):
  return (
    predicted_v_lead >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED and
    predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING and
    gap_excess + predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_ARM_EXCESS
  )


def get_creep_to_stop_gap_pullaway_accel_min(pullaway_excess):
  pullaway_excess = max(0.0, pullaway_excess)
  return float(np.interp(
    pullaway_excess,
    [0.0, CREEP_TO_STOP_GAP_FOLLOW_EXCESS, CREEP_TO_STOP_GAP_START_EXCESS],
    [0.0, CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN, CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX],
  ))


def get_creep_pullaway_launch_accel_max(lead_gap_excess, predicted_gap_opening):
  lead_gap_excess = max(0.0, lead_gap_excess)
  predicted_gap_opening = max(0.0, predicted_gap_opening)
  runway_blend = np.interp(
    lead_gap_excess + predicted_gap_opening,
    [CREEP_TO_STOP_GAP_PREDICT_ARM_EXCESS, CREEP_TO_STOP_GAP_START_EXCESS],
    [0.0, 1.0],
  )
  opening_blend = np.interp(
    predicted_gap_opening,
    [CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING, CREEP_TO_STOP_GAP_START_EXCESS],
    [0.0, 1.0],
  )
  blend = min(runway_blend, opening_blend)
  return float(CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX + blend * (
    CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MAX - CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX
  ))


def get_model_lead_pullaway(model_msg, radar_lead, v_ego, horizon=CREEP_TO_STOP_GAP_MODEL_LEAD_HORIZON):
  if v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO_ARM or not getattr(radar_lead, "status", False):
    return 0.0, 0.0

  leads_v3 = getattr(model_msg, "leadsV3", [])
  if len(leads_v3) == 0:
    return 0.0, 0.0

  lead_msg = leads_v3[0]
  if float(getattr(lead_msg, "prob", 0.0)) < CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB:
    return 0.0, 0.0
  if float(getattr(radar_lead, "modelProb", getattr(lead_msg, "prob", 0.0))) < CREEP_TO_STOP_GAP_MIN_MODEL_PROB:
    return 0.0, 0.0

  d_rel = float(getattr(radar_lead, "dRel", 0.0))
  if not np.isfinite(d_rel):
    return 0.0, 0.0
  radar_y_rel = float(getattr(radar_lead, "yRel", 0.0))
  if not np.isfinite(radar_y_rel):
    return 0.0, 0.0
  radar_v_lead = float(getattr(radar_lead, "vLeadK", getattr(radar_lead, "vLead", 0.0)))
  if not np.isfinite(radar_v_lead):
    return 0.0, 0.0

  ts = np.asarray(getattr(lead_msg, "t", []), dtype=float)
  xs = np.asarray(getattr(lead_msg, "x", []), dtype=float)
  ys = np.asarray(getattr(lead_msg, "y", []), dtype=float)
  vs = np.asarray(getattr(lead_msg, "v", []), dtype=float)
  x_stds = np.asarray(getattr(lead_msg, "xStd", []), dtype=float)
  y_stds = np.asarray(getattr(lead_msg, "yStd", []), dtype=float)
  v_stds = np.asarray(getattr(lead_msg, "vStd", []), dtype=float)
  if any(values.ndim != 1 for values in (ts, xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if ts.size == 0 or any(values.size != ts.size for values in (xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if any(not np.all(np.isfinite(values)) for values in (ts, xs, ys, vs, x_stds, y_stds, v_stds)):
    return 0.0, 0.0
  if ts[0] > 0.05 or ts[-1] < horizon or np.any(np.diff(ts) <= 0.0):
    return 0.0, 0.0

  horizon_mask = ts <= horizon
  x_std = max(float(np.max(x_stds[horizon_mask])), float(np.interp(horizon, ts, x_stds)))
  y_std = max(float(np.max(y_stds[horizon_mask])), float(np.interp(horizon, ts, y_stds)))
  v_std = max(float(np.max(v_stds[horizon_mask])), float(np.interp(horizon, ts, v_stds)))
  if x_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD or y_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD:
    return 0.0, 0.0
  if v_std > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD:
    return 0.0, 0.0

  model_d_rel_now = float(xs[0] - CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET)
  if abs(model_d_rel_now - d_rel) > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_DIST_ERROR:
    return 0.0, 0.0
  model_y_rel_now = float(-ys[0])
  max_y_error = min(CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_ERROR,
                    max(CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_Y_ERROR, 2.0 * float(y_std)))
  if abs(model_y_rel_now - radar_y_rel) > max_y_error:
    return 0.0, 0.0
  if abs(float(vs[0]) - radar_v_lead) > CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_ERROR:
    return 0.0, 0.0

  model_d_rel_horizon = float(np.interp(horizon, ts, xs) - CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET)
  predicted_v_lead = float(np.interp(horizon, ts, vs))
  predicted_gap_opening = max(0.0, model_d_rel_horizon - d_rel)
  return predicted_v_lead, predicted_gap_opening


def get_lead_stop_approach_slewed_accel(v_ego, d_rel, v_lead, a_lead, prev_a_target, a_target, dt):
  stopped_lead_with_runway = (
    v_lead <= LEAD_STOP_APPROACH_DECEL_SLEW_STOPPED_LEAD_V and
    d_rel > STOP_DISTANCE + LEAD_STOP_APPROACH_DECEL_SLEW_MIN_GAP_EXCESS
  )
  hard_braking_lead = a_lead <= -LEAD_STOP_APPROACH_DECEL_SLEW_MIN_LEAD_DECEL
  if (
    v_ego < LEAD_STOP_APPROACH_DECEL_SLEW_MIN_V_EGO or
    v_lead >= v_ego or
    not (hard_braking_lead or stopped_lead_with_runway)
  ):
    return a_target

  max_delta = LEAD_STOP_APPROACH_DECEL_SLEW_MAX_JERK * dt
  return float(np.clip(a_target, prev_a_target - max_delta, prev_a_target + max_delta))


def should_defer_e2e_to_stopped_lead_mpc(v_ego, lead, mpc_source, reset_state=False, force_slow_decel=False,
                                         brake_pressed=False, gas_pressed=False):
  if reset_state or force_slow_decel or brake_pressed or gas_pressed:
    return False
  if mpc_source not in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1):
    return False
  if not getattr(lead, "status", False) or v_ego <= CREEP_TO_STOP_GAP_MAX_V_EGO:
    return False

  d_rel = float(getattr(lead, "dRel", 0.0))
  v_lead = float(getattr(lead, "vLeadK", 0.0))
  a_lead = float(getattr(lead, "aLeadK", 0.0))
  model_prob = float(getattr(lead, "modelProb", 0.0))
  if model_prob < STOPPED_LEAD_GAP_FILL_MIN_MODEL_PROB or v_lead > STOPPED_LEAD_GAP_FILL_MAX_LEAD_SPEED:
    return False

  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  return d_rel > stop_target + CREEP_TO_STOP_GAP_HOLD_EXCESS


def get_mpc_source_lead(radar_state, mpc_source):
  if mpc_source == LongitudinalPlanSource.lead0:
    return radar_state.leadOne
  if mpc_source == LongitudinalPlanSource.lead1:
    return radar_state.leadTwo
  return None


def creep_to_stop_gap_blocked(v_ego, d_rel, v_lead, model_prob, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                              a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  blocked = brake_pressed or gas_pressed or force_slow_decel or model_prob < CREEP_TO_STOP_GAP_MIN_MODEL_PROB
  blocked = blocked or v_lead < CREEP_TO_STOP_GAP_MIN_LEAD_SPEED or v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO
  return blocked or gap_excess <= 0.0 or gap_excess > CREEP_TO_STOP_GAP_MAX_EXCESS


def get_creep_to_stop_gap_accel(v_ego, d_rel, v_lead, model_prob, active, brake_pressed=False, gas_pressed=False,
                                force_slow_decel=False, a_lead=0.0, a_lead_tau=0.0,
                                model_predicted_v_lead=0.0, model_predicted_gap_opening=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  if creep_to_stop_gap_blocked(v_ego, d_rel, v_lead, model_prob, brake_pressed, gas_pressed, force_slow_decel, a_lead):
    return False, 0.0

  radar_predicted_v_lead, radar_predicted_gap_opening = get_predicted_lead_pullaway(v_lead, a_lead, a_lead_tau)
  lead_pullaway = v_lead >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
  radar_predicted_pullaway = a_lead >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and has_predicted_lead_pullaway(
    gap_excess, radar_predicted_v_lead, radar_predicted_gap_opening
  )
  model_predicted_pullaway = has_predicted_lead_pullaway(gap_excess, model_predicted_v_lead, model_predicted_gap_opening)
  predicted_pullaway = radar_predicted_pullaway or model_predicted_pullaway
  pullaway_gap_excess = gap_excess + max(
    radar_predicted_gap_opening if radar_predicted_pullaway else 0.0,
    model_predicted_gap_opening if model_predicted_pullaway else 0.0,
  )
  should_arm = gap_excess >= CREEP_TO_STOP_GAP_ARM_EXCESS and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM
  if lead_pullaway or predicted_pullaway:
    should_arm = should_arm or (
      pullaway_gap_excess >= CREEP_TO_STOP_GAP_PREDICT_ARM_EXCESS and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM
    )
  if not active and not should_arm:
    return False, 0.0

  target_speed = float(np.interp(pullaway_gap_excess, CREEP_TO_STOP_GAP_SPEED_BP, CREEP_TO_STOP_GAP_SPEED_V))
  accel_max = CREEP_TO_STOP_GAP_ACCEL_MAX
  if lead_pullaway or predicted_pullaway:
    target_speed = min(target_speed, CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX)
    accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
  accel = np.clip((target_speed - v_ego) * CREEP_TO_STOP_GAP_ACCEL_GAIN, CREEP_TO_STOP_GAP_ACCEL_MIN, accel_max)
  if (lead_pullaway or predicted_pullaway) and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM and accel > 0.0:
    accel = max(accel, min(get_creep_to_stop_gap_pullaway_accel_min(pullaway_gap_excess), accel_max))
  return True, float(accel)


def should_release_creep_stop_hold(release_active, v_ego, d_rel, v_lead, a_lead, predicted_pullaway=False, model_prob=1.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  if v_ego >= CREEP_TO_STOP_GAP_MAX_V_EGO or d_rel <= stop_target + CREEP_TO_STOP_GAP_REHOLD_EXCESS:
    return False
  if should_hold_stopped_lead_micro_creep(v_ego, d_rel, v_lead, a_lead, predicted_pullaway):
    return False
  if release_active:
    return True
  return (
    d_rel >= stop_target + CREEP_TO_STOP_GAP_HOLD_RELEASE_EXCESS and
    (predicted_pullaway or v_lead >= CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_SPEED or a_lead >= CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_ACCEL)
  )


def should_hold_stopped_lead_micro_creep(v_ego, d_rel, v_lead, a_lead, predicted_pullaway=False):
  return (
    not predicted_pullaway and
    v_ego < CREEP_TO_STOP_GAP_STOP_EXCESS and
    v_lead < CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED and
    a_lead <= 0.05 and
    d_rel <= STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS
  )


def should_hold_creep_to_stop_gap(v_ego, d_rel, v_lead, a_lead, predicted_pullaway=False, release_active=False, model_prob=1.0):
  if should_release_creep_stop_hold(release_active, v_ego, d_rel, v_lead, a_lead, predicted_pullaway, model_prob):
    return False
  if should_hold_stopped_lead_micro_creep(v_ego, d_rel, v_lead, a_lead, predicted_pullaway):
    return True
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  return (
    not predicted_pullaway and
    v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO and
    v_lead < CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED and
    a_lead <= 0.05 and
    d_rel <= stop_target + CREEP_TO_STOP_GAP_HOLD_EXCESS
  )


def get_creep_to_stop_gap_hold_accel(v_ego, d_rel):
  if v_ego <= 0.0:
    return 0.0
  available_gap = max(d_rel - CREEP_TO_STOP_GAP_HOLD_BUFFER, 0.1)
  required_decel = (v_ego**2) / (2.0 * available_gap)
  return -min(CREEP_TO_STOP_GAP_HOLD_DECEL_CAP, required_decel)


def get_stopped_lead_stop_gap_guard_accel(v_ego, d_rel, v_lead, a_lead, model_prob):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  if (
    model_prob < CREEP_TO_STOP_GAP_MIN_MODEL_PROB or
    v_ego <= 0.0 or v_ego >= STOPPED_LEAD_STOP_GAP_GUARD_MAX_V_EGO or
    abs(v_lead) > STOPPED_LEAD_STOP_GAP_GUARD_MAX_LEAD_SPEED or
    a_lead > STOPPED_LEAD_STOP_GAP_GUARD_MAX_LEAD_ACCEL or
    d_rel > stop_target + STOPPED_LEAD_STOP_GAP_GUARD_EXCESS
  ):
    return None

  available_gap = max(d_rel - stop_target + STOPPED_LEAD_STOP_GAP_GUARD_TARGET_BUFFER, 0.1)
  required_decel = (v_ego**2) / (2.0 * available_gap)
  if required_decel < STOPPED_LEAD_STOP_GAP_GUARD_MIN_REQUIRED_DECEL:
    return None
  decel_cap = float(np.interp(v_ego, STOPPED_LEAD_STOP_GAP_GUARD_DECEL_CAP_BP, STOPPED_LEAD_STOP_GAP_GUARD_DECEL_CAP_V))
  return -min(decel_cap, max(-CREEP_TO_STOP_GAP_ACCEL_MIN, required_decel))


def get_moving_lead_stop_gap_guard_accel(v_ego, d_rel, v_lead, a_lead, y_rel, t_follow):
  if (
    v_ego < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_EGO or
    v_lead < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD or
    v_lead >= v_ego or
    a_lead > -MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL or
    abs(y_rel) > MOVING_LEAD_STOP_GAP_GUARD_MAX_Y_REL
  ):
    return None

  _desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  if d_rel > caution_gap:
    return None

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)
  target = float(target)
  if float(cost) <= 0.0 or target > -MOVING_LEAD_STOP_GAP_GUARD_MIN_TARGET_DECEL:
    return None
  closing_speed = max(v_ego - v_lead, 0.0)
  required_decel = float(get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead))
  decel_cap = -ACCEL_MIN if a_lead <= -MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL else MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP
  target = min(target, -min(decel_cap, required_decel))
  return target


def should_arm_stopped_lead_gap_fill(v_ego, d_rel, v_lead, model_prob, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                                     a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  return (
    not brake_pressed and not gas_pressed and not force_slow_decel and
    model_prob >= CREEP_TO_STOP_GAP_MIN_MODEL_PROB and
    v_ego < STOPPED_LEAD_GAP_FILL_ARM_MAX_V_EGO and
    abs(v_lead) <= STOPPED_LEAD_GAP_FILL_ARM_MAX_LEAD_SPEED and
    a_lead >= STOPPED_LEAD_GAP_FILL_MIN_LEAD_ACCEL and
    0.0 <= gap_excess <= STOPPED_LEAD_GAP_FILL_ARM_MAX_GAP_EXCESS
  )


def get_stopped_lead_gap_fill_accel(v_ego, d_rel, v_lead, model_prob, armed, brake_pressed=False, gas_pressed=False, force_slow_decel=False,
                                    a_lead=0.0):
  stop_target = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, model_prob)
  gap_excess = d_rel - stop_target
  blocked = (
    not armed or brake_pressed or gas_pressed or force_slow_decel or
    model_prob < STOPPED_LEAD_GAP_FILL_MIN_MODEL_PROB or
    v_ego >= STOPPED_LEAD_GAP_FILL_MAX_V_EGO or
    a_lead < STOPPED_LEAD_GAP_FILL_MIN_LEAD_ACCEL or
    v_lead < CREEP_TO_STOP_GAP_MIN_LEAD_SPEED or
    v_lead > STOPPED_LEAD_GAP_FILL_MAX_LEAD_SPEED or
    gap_excess <= STOPPED_LEAD_GAP_FILL_MIN_EXCESS or
    gap_excess > STOPPED_LEAD_GAP_FILL_MAX_EXCESS
  )
  if blocked:
    return False, 0.0

  target_speed = float(np.interp(gap_excess, STOPPED_LEAD_GAP_FILL_SPEED_BP, STOPPED_LEAD_GAP_FILL_SPEED_V))
  accel = np.clip((target_speed - v_ego) * STOPPED_LEAD_GAP_FILL_ACCEL_GAIN,
                  STOPPED_LEAD_GAP_FILL_ACCEL_MIN, STOPPED_LEAD_GAP_FILL_ACCEL_MAX)
  return True, float(accel)


def stopped_lead_gap_fill_lead_continuous(track_id, prev_track_id, d_rel, prev_d_rel, v_lead, prev_v_lead):
  if prev_track_id == -2:
    return False
  if track_id >= 0 and prev_track_id >= 0 and track_id != prev_track_id:
    return False
  return bool(
    abs(d_rel - prev_d_rel) <= STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_D_REL_DELTA and
    abs(v_lead - prev_v_lead) <= STOPPED_LEAD_GAP_FILL_CONTINUITY_MAX_V_LEAD_DELTA
  )


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = SunnypilotLongitudinalMpc(dt=dt)
    self.planner_seed_mpc = LongitudinalMpc(dt=dt)
    self.params = Params()
    self.VM = VehicleModel(CP)
    self.longitudinal_arbiter = LongitudinalArbiter()
    self.longitudinal_decision = None
    self.longitudinal_decision_candidates = []
    self.longitudinal_decision_telemetry: LongitudinalDecisionTelemetry | None = None
    self.planner_seed_candidates = []
    self.one_pedal_mode = get_one_pedal_longitudinal_mode(self.params)
    self.one_pedal_cruise_hold_active = False
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.prev_reset_state = True
    self.engage_stop_bootstrap_timer = 0.0
    self.e2e_close_stop_settle_active = False
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.creep_to_stop_gap_active = False
    self.creep_stop_hold_released = False
    self.stopped_lead_gap_fill_timer = 0.0
    self.lead_loss_e2e_guard_timer = 0.0
    self.previous_lead_loss_status = False
    self.previous_lead_loss_d_rel = 0.0
    self.previous_lead_loss_model_prob = 0.0
    self.stopped_lead_gap_fill_track_id = -2
    self.stopped_lead_gap_fill_d_rel = 0.0
    self.stopped_lead_gap_fill_v_lead = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    try:
      self.control_calculation_hardening = Params().get_bool("ControlCalculationHardening")
    except UnknownKeyName:
      self.control_calculation_hardening = False

  @staticmethod
  def parse_model(model_msg):
    if (
      len(model_msg.position.x) == ModelConstants.IDX_N
      and len(model_msg.velocity.x) == ModelConstants.IDX_N
      and len(model_msg.acceleration.x) == ModelConstants.IDX_N
    ):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
      speed_limit_coast_accel = accel_coast
      cruise_coast_accel = accel_coast
    else:
      accel_coast = ACCEL_MAX
      speed_limit_coast_accel = 0.0
      cruise_coast_accel = 0.0

    v_ego = sm['carState'].vEgo
    prev_output_a_target = self.output_a_target
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized
    self.one_pedal_cruise_hold_active = update_one_pedal_cruise_hold(
      self.one_pedal_cruise_hold_active and self.one_pedal_mode != ONE_PEDAL_MODE_OFF,
      sm['carState'].buttonEvents,
      sm['carState'].gasPressed,
      sm['carState'].brakePressed,
      not reset_state and sm['selfdriveState'].enabled,
    )

    if reset_state:
      self.engage_stop_bootstrap_timer = 0.0
    elif self.prev_reset_state:
      self.engage_stop_bootstrap_timer = ENGAGE_STOP_BOOTSTRAP_TIME
    else:
      self.engage_stop_bootstrap_timer = max(0.0, self.engage_stop_bootstrap_timer - self.dt)
    self.prev_reset_state = reset_state

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    live_params = sm['liveParameters']
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - live_params.angleOffsetDeg
    accurate_lateral_accel = self.params.get_bool("AccurateLateralAccel")
    if accurate_lateral_accel:
      self.VM.update_params(max(live_params.stiffnessFactor, 0.1), max(live_params.steerRatio, 0.1))
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP,
                                      control_calculation_hardening=self.control_calculation_hardening,
                                      vehicle_model=self.VM, roll=live_params.roll,
                                      accurate_lateral_accel=accurate_lateral_accel)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED * 2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired,
                                                                    v_cruise, coast_accel=speed_limit_coast_accel)

    if force_slow_decel:
      v_cruise = 0.0

    lead_one = sm['radarState'].leadOne
    has_radar_lead = has_valid_radar_lead(sm['radarState'])
    lead_loss_guard_lead = get_lead_loss_e2e_guard_lead(sm['radarState'])
    custom_engage_stop_bootstrap_active = should_run_engage_stop_bootstrap(self.engage_stop_bootstrap_timer, v_ego, sm['radarState'], sm['modelV2'])

    has_confirmed_lead = has_confirmed_radar_lead(sm['radarState'])
    self.lead_loss_e2e_guard_timer = update_lead_loss_e2e_guard_timer(
      self.lead_loss_e2e_guard_timer, self.dt,
      self.previous_lead_loss_status, self.previous_lead_loss_d_rel, self.previous_lead_loss_model_prob,
      has_confirmed_lead, is_lane_change_active(sm['modelV2']),
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
    )

    mpc_v_desired = self.v_desired_filter.x
    mpc_a_desired = self.a_desired
    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(mpc_v_desired, mpc_a_desired)
    self.mpc.update(
      sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
      block_short_gap_pullaway_response=sm['carState'].brakePressed or sm['carState'].gasPressed or force_slow_decel or reset_state,
      model_msg=sm['modelV2'],
    )
    planner_seed_mpc_v_desired_trajectory = None
    planner_seed_mpc_a_desired_trajectory = None
    planner_seed_mpc_j_desired_trajectory = None
    planner_seed_mpc = getattr(self, "planner_seed_mpc", None)
    stack_resolution = getattr(self, "longitudinal_stack_resolution", None)
    run_planner_seed_mpc = (
      bool(getattr(sm['selfdriveState'], "enabled", True)) and
      (has_radar_lead or force_slow_decel) and
      is_custom_stack(getattr(stack_resolution, "resolved_stack", ""))
    )
    if planner_seed_mpc is not None and run_planner_seed_mpc:
      planner_seed_mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
      planner_seed_mpc.set_cur_state(mpc_v_desired, mpc_a_desired)
      planner_seed_mpc.update(
        sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
        block_short_gap_pullaway_response=sm['carState'].brakePressed or sm['carState'].gasPressed or force_slow_decel or reset_state,
        model_msg=sm['modelV2'],
      )
      planner_seed_mpc_v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner_seed_mpc.v_solution)
      planner_seed_mpc_a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner_seed_mpc.a_solution)
      planner_seed_mpc_j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], planner_seed_mpc.j_solution)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    state_a_desired_trajectory = (
      planner_seed_mpc_a_desired_trajectory if planner_seed_mpc_a_desired_trajectory is not None else self.a_desired_trajectory
    )
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, state_a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(
      self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX, action_t=action_t, vEgoStopping=self.CP.vEgoStopping
    )
    planner_seed_mpc_a_target = None
    planner_seed_mpc_should_stop = False
    planner_seed_mpc_fcw = False
    if planner_seed_mpc_v_desired_trajectory is not None:
      planner_seed_mpc_a_target, planner_seed_mpc_should_stop = get_accel_from_plan(
        planner_seed_mpc_v_desired_trajectory, planner_seed_mpc_a_desired_trajectory, CONTROL_N_T_IDX,
        action_t=action_t, vEgoStopping=self.CP.vEgoStopping,
      )
      planner_seed_mpc_fcw = planner_seed_mpc.crash_cnt > 2 and not sm['carState'].standstill
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
    e2e_active = self.is_e2e(sm)
    custom_e2e_runway_comfort_a_target = get_e2e_runway_comfort_accel(
      v_ego, output_a_target_e2e, accel_coast, sm['modelV2'], e2e_active, prev_output_a_target,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      engage_stop_bootstrap_active=custom_engage_stop_bootstrap_active,
      has_radar_lead=has_radar_lead,
      dt=self.dt,
    )
    custom_e2e_close_stop_a_target, custom_close_stop_should_stop, self.e2e_close_stop_settle_active = get_e2e_close_stop_settle(
      v_ego, output_a_target_e2e, sm['modelV2'], sm['radarState'], e2e_active,
      active=self.e2e_close_stop_settle_active,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
    )
    mpc_source_lead = get_mpc_source_lead(sm['radarState'], self.mpc.source)
    defer_e2e_to_stopped_lead_mpc = e2e_active and should_defer_e2e_to_stopped_lead_mpc(
      v_ego, mpc_source_lead, self.mpc.source,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
    )
    if e2e_active and not defer_e2e_to_stopped_lead_mpc:
      custom_e2e_runway_comfort_output_a_target = min(custom_e2e_runway_comfort_a_target, output_a_target_mpc)
      lead_loss_guarded_e2e_a_target = apply_lead_loss_e2e_guard_accel(
        output_a_target_e2e, output_should_stop_e2e, self.lead_loss_e2e_guard_timer, has_confirmed_lead,
      )
      custom_lead_loss_e2e_guard_a_target = min(lead_loss_guarded_e2e_a_target, output_a_target_mpc)
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      custom_e2e_runway_comfort_output_a_target = None
      custom_lead_loss_e2e_guard_a_target = None
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    model_stop_protection_active = sm['selfdriveState'].experimentalMode and self.dec.active() and not e2e_active
    e2e_runway_positive_accel_cap = get_e2e_runway_positive_accel_cap(
      v_ego, sm['modelV2'], e2e_active,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      engage_stop_bootstrap_active=custom_engage_stop_bootstrap_active,
      has_radar_lead=has_radar_lead,
      model_stop_protection_active=model_stop_protection_active,
    )
    custom_e2e_stop_approach_a_target = get_e2e_stop_approach_accel(
      v_ego, sm['modelV2'], sm['radarState'], e2e_active,
      force_slow_decel=force_slow_decel or reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      model_stop_protection_active=model_stop_protection_active,
    )
    e2e_stop_approach_a_target = 0.0

    lead_track_id = int(getattr(lead_one, "radarTrackId", -2)) if lead_one.status else -2
    lead_d_rel = float(lead_one.dRel) if lead_one.status else 0.0
    lead_v_lead = float(lead_one.vLeadK) if lead_one.status else 0.0
    if lead_one.status and should_arm_stopped_lead_gap_fill(
      v_ego, lead_d_rel, lead_v_lead, float(lead_one.modelProb),
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
    ):
      self.stopped_lead_gap_fill_timer = STOPPED_LEAD_GAP_FILL_ARM_TIME
      self.stopped_lead_gap_fill_track_id = lead_track_id
      self.stopped_lead_gap_fill_d_rel = lead_d_rel
      self.stopped_lead_gap_fill_v_lead = lead_v_lead
    elif not (lead_one.status and stopped_lead_gap_fill_lead_continuous(
      lead_track_id, self.stopped_lead_gap_fill_track_id, lead_d_rel, self.stopped_lead_gap_fill_d_rel,
      lead_v_lead, self.stopped_lead_gap_fill_v_lead,
    )):
      self.stopped_lead_gap_fill_timer = 0.0
      self.stopped_lead_gap_fill_track_id = -2
      self.stopped_lead_gap_fill_d_rel = 0.0
      self.stopped_lead_gap_fill_v_lead = 0.0
    else:
      self.stopped_lead_gap_fill_timer = max(0.0, self.stopped_lead_gap_fill_timer - self.dt)
      self.stopped_lead_gap_fill_d_rel = lead_d_rel
      self.stopped_lead_gap_fill_v_lead = lead_v_lead

    model_predicted_v_lead, model_predicted_gap_opening = (
      get_model_lead_pullaway(sm['modelV2'], lead_one, v_ego) if lead_one.status else (0.0, 0.0)
    )
    lead_gap_excess = float(lead_one.dRel) - get_lead_stop_presentation_distance(
      v_ego, float(lead_one.vLeadK), float(lead_one.aLeadK), float(lead_one.modelProb)
    ) if lead_one.status else 0.0
    lead_v_rel = float(getattr(lead_one, "vRel", float(lead_one.vLeadK) - v_ego)) if lead_one.status else 0.0
    radar_predicted_v_lead, radar_predicted_gap_opening = (
      get_predicted_lead_pullaway(float(lead_one.vLeadK), float(lead_one.aLeadK), float(lead_one.aLeadTau)) if lead_one.status else (0.0, 0.0)
    )
    radar_predicted_pullaway = lead_one.status and float(lead_one.aLeadK) >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and has_predicted_lead_pullaway(
      lead_gap_excess, radar_predicted_v_lead, radar_predicted_gap_opening
    )
    model_predicted_pullaway = lead_one.status and not creep_to_stop_gap_blocked(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      sm['carState'].brakePressed, sm['carState'].gasPressed, force_slow_decel or reset_state,
      float(lead_one.aLeadK),
    ) and has_predicted_lead_pullaway(
      lead_gap_excess, model_predicted_v_lead, model_predicted_gap_opening,
    )
    creep_pullaway_release = lead_one.status and (
      float(lead_one.vLeadK) >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED or radar_predicted_pullaway or model_predicted_pullaway
    )
    confirmed_creep_pullaway_launch = lead_one.status and creep_pullaway_release and (radar_predicted_pullaway or model_predicted_pullaway)
    confirmed_creep_pullaway_stop_release = confirmed_creep_pullaway_launch and not (
      e2e_active and output_should_stop_e2e and output_a_target_e2e < 0.0
    )
    lead_pullaway_predicted_gap_opening = max(
      radar_predicted_gap_opening if radar_predicted_pullaway else 0.0,
      model_predicted_gap_opening if model_predicted_pullaway else 0.0,
    )
    lead_pullaway_runway_excess = lead_gap_excess + lead_pullaway_predicted_gap_opening
    allow_creep_pullaway_release = creep_pullaway_release and (
      not (e2e_active and output_should_stop_e2e) or confirmed_creep_pullaway_stop_release
    )
    prev_creep_to_stop_gap_active = self.creep_to_stop_gap_active
    self.creep_to_stop_gap_active, creep_a_target = get_creep_to_stop_gap_accel(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      self.creep_to_stop_gap_active and not reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
      a_lead_tau=float(lead_one.aLeadTau),
      model_predicted_v_lead=model_predicted_v_lead,
      model_predicted_gap_opening=model_predicted_gap_opening,
    ) if lead_one.status else (False, 0.0)
    custom_creep_to_stop_gap_a_target = None
    custom_creep_to_stop_gap_should_stop = None
    custom_creep_to_stop_gap_selection = PLANNER_SEED_CAP
    custom_creep_to_stop_gap_accel_max = None
    if self.creep_to_stop_gap_active:
      if creep_a_target >= 0.0:
        if not self.output_should_stop or allow_creep_pullaway_release or not (e2e_active and output_should_stop_e2e):
          custom_creep_to_stop_gap_a_target = creep_a_target
          custom_creep_to_stop_gap_should_stop = self.output_should_stop and not allow_creep_pullaway_release
          custom_creep_to_stop_gap_selection = PLANNER_SEED_FLOOR
      else:
        custom_creep_to_stop_gap_a_target = creep_a_target
        custom_creep_to_stop_gap_should_stop = self.output_should_stop or (
          not allow_creep_pullaway_release and v_ego < self.CP.vEgoStopping
        )
      creep_accel_max = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX if creep_a_target > CREEP_TO_STOP_GAP_ACCEL_MAX else CREEP_TO_STOP_GAP_ACCEL_MAX
      if custom_creep_to_stop_gap_a_target is not None and not (
        creep_pullaway_release and lead_gap_excess >= CREEP_TO_STOP_GAP_START_EXCESS
      ):
        custom_creep_to_stop_gap_accel_max = creep_accel_max
    limit_creep_pullaway_accel_step = creep_pullaway_release and (prev_creep_to_stop_gap_active or self.creep_to_stop_gap_active)

    gap_fill_active, gap_fill_a_target = get_stopped_lead_gap_fill_accel(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.modelProb),
      lead_one.status and self.stopped_lead_gap_fill_timer > 0.0,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=float(lead_one.aLeadK),
    ) if lead_one.status else (False, 0.0)
    custom_gap_fill_a_target = None
    custom_gap_fill_should_stop = None
    custom_gap_fill_selection = PLANNER_SEED_CAP
    custom_gap_fill_accel_max = None
    if gap_fill_active:
      if gap_fill_a_target >= 0.0:
        if not self.output_should_stop:
          custom_gap_fill_a_target = gap_fill_a_target
          custom_gap_fill_selection = PLANNER_SEED_FLOOR
          custom_gap_fill_accel_max = STOPPED_LEAD_GAP_FILL_ACCEL_MAX
      else:
        custom_gap_fill_a_target = gap_fill_a_target
        custom_gap_fill_should_stop = self.output_should_stop or v_ego < self.CP.vEgoStopping

    custom_lead_accel_recovery_a_target = None
    if lead_one.status and not self.output_should_stop and not reset_state and self.mpc.source != LongitudinalPlanSource.e2e and \
       self.source == custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise:
      recovery_a_min = get_lead_accel_recovery_a_min(
        v_ego, float(lead_one.vLeadK), float(lead_one.dRel), float(lead_one.aLeadK), get_T_FOLLOW(sm['selfdriveState'].personality)
      )
      if v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO:
        recovery_a_min = min(recovery_a_min, CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX)
      custom_lead_accel_recovery_a_target = recovery_a_min

    if lead_one.status and not sm['carState'].brakePressed and not sm['carState'].gasPressed and not force_slow_decel and not reset_state:
      self.creep_stop_hold_released = should_release_creep_stop_hold(
        self.creep_stop_hold_released, v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK),
        model_predicted_pullaway, float(lead_one.modelProb),
      )
    else:
      self.creep_stop_hold_released = False
    custom_creep_hold_a_target = None
    if lead_one.status and not (self.creep_to_stop_gap_active and creep_a_target > 0.0) and should_hold_creep_to_stop_gap(
      v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK), model_predicted_pullaway,
      self.creep_stop_hold_released, float(lead_one.modelProb),
    ):
      custom_creep_hold_a_target = min(CREEP_TO_STOP_GAP_ACCEL_MIN, get_creep_to_stop_gap_hold_accel(v_ego, float(lead_one.dRel)))

    custom_stopped_stop_gap_guard_a_target = None
    custom_moving_stop_guard_a_target = None
    if lead_one.status:
      if not defer_e2e_to_stopped_lead_mpc and self.mpc.source != LongitudinalPlanSource.e2e:
        stop_gap_guard_a_target = get_stopped_lead_stop_gap_guard_accel(
          v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK), float(lead_one.modelProb),
        )
        if stop_gap_guard_a_target is not None:
          custom_stopped_stop_gap_guard_a_target = stop_gap_guard_a_target

      moving_stop_guard_a_target = get_moving_lead_stop_gap_guard_accel(
        v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK), float(lead_one.yRel),
        get_T_FOLLOW(sm['selfdriveState'].personality),
      )
      if moving_stop_guard_a_target is not None:
        custom_moving_stop_guard_a_target = moving_stop_guard_a_target

    custom_lead_stop_approach_slewed_a_target = None
    custom_lead_stop_approach_base_a_target = None
    if lead_one.status and not reset_state and not sm['carState'].brakePressed and not sm['carState'].gasPressed:
      lead_stop_approach_base_a_target = output_a_target
      if planner_seed_mpc_a_target is not None:
        planner_seed_mpc_lead_floor_blocked = (
          planner_seed_mpc_a_target > output_a_target and
          planner_seed_mpc.source in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)
        )
        if not planner_seed_mpc_lead_floor_blocked:
          lead_stop_approach_base_a_target = planner_seed_mpc_a_target
      for lead_stop_pre_slew_a_target in (custom_stopped_stop_gap_guard_a_target, custom_moving_stop_guard_a_target):
        if lead_stop_pre_slew_a_target is not None:
          lead_stop_approach_base_a_target = min(lead_stop_approach_base_a_target, lead_stop_pre_slew_a_target)
      custom_lead_stop_approach_base_a_target = lead_stop_approach_base_a_target
      custom_lead_stop_approach_slewed_a_target = get_lead_stop_approach_slewed_accel(
        v_ego, float(lead_one.dRel), float(lead_one.vLeadK), float(lead_one.aLeadK),
        prev_output_a_target, lead_stop_approach_base_a_target, self.dt,
      )

    self.previous_lead_loss_status = lead_loss_guard_lead is not None
    self.previous_lead_loss_d_rel = float(lead_loss_guard_lead.dRel) if lead_loss_guard_lead is not None else 0.0
    self.previous_lead_loss_model_prob = float(lead_loss_guard_lead.modelProb) if lead_loss_guard_lead is not None else 0.0

    continuing_creep_pullaway_launch = lead_one.status and prev_output_a_target >= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN and \
      v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO and \
      CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS <= lead_gap_excess <= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MAX_EXCESS and \
      float(lead_one.aLeadK) >= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MIN_LEAD_ACCEL and \
      confirmed_creep_pullaway_stop_release
    creep_pullaway_launch = lead_one.status and (not self.output_should_stop or confirmed_creep_pullaway_stop_release) and \
      not sm['carState'].brakePressed and not sm['carState'].gasPressed and \
      not force_slow_decel and not reset_state and \
      (v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM or continuing_creep_pullaway_launch) and \
      float(lead_one.modelProb) >= CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB and \
      ((CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS <= lead_gap_excess <= CREEP_TO_STOP_GAP_MAX_EXCESS and
        self.creep_to_stop_gap_active and creep_a_target > 0.0) or continuing_creep_pullaway_launch) and \
      confirmed_creep_pullaway_stop_release
    custom_creep_pullaway_launch_floor = None
    custom_creep_pullaway_launch_cap = None
    lead_pullaway_crawl_cap_released = lead_one.status and creep_pullaway_release and \
      lead_pullaway_runway_excess >= CREEP_TO_STOP_GAP_START_EXCESS and lead_v_rel > 0.0 and float(lead_one.aLeadK) >= 0.0
    if creep_pullaway_launch:
      custom_creep_to_stop_gap_accel_max = None
      launch_accel_max = get_creep_pullaway_launch_accel_max(lead_gap_excess, lead_pullaway_predicted_gap_opening)
      if not lead_pullaway_crawl_cap_released:
        crawl_accel_max = get_lead_crawl_accel_max(
          float(lead_one.dRel), v_ego, float(lead_one.vLeadK), float(lead_one.aLeadK), get_T_FOLLOW(sm['selfdriveState'].personality),
        )
        launch_accel_max = min(launch_accel_max, float(crawl_accel_max))
      custom_creep_pullaway_launch_floor = CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN
      custom_creep_pullaway_launch_cap = launch_accel_max

    has_lead = sm['radarState'].leadOne.status or sm['radarState'].leadTwo.status
    cruise_coast_applied = False
    cruise_coast_a_target = output_a_target

    legacy_a_target = float(output_a_target)
    legacy_should_stop = bool(self.output_should_stop)
    lead_confidence = get_active_lead_confidence(sm['radarState'].leadOne, sm['radarState'].leadTwo)
    self.longitudinal_decision_candidates = list(getattr(self, "decision_candidates_sp", [])) + build_core_longitudinal_candidates(
      has_lead=has_lead,
      lead_confidence=lead_confidence,
      v_cruise=v_cruise,
      a_cruise=self.a_desired,
      output_a_target_mpc=output_a_target_mpc,
      output_should_stop_mpc=output_should_stop_mpc,
      e2e_active=e2e_active,
      output_a_target_e2e=output_a_target_e2e,
      output_should_stop_e2e=output_should_stop_e2e,
      e2e_stop_approach_a_target=e2e_stop_approach_a_target,
      cruise_coast_applied=cruise_coast_applied,
      cruise_coast_a_target=cruise_coast_a_target,
    )
    source_stability_v_ego = None if (
      reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed
    ) else v_ego
    self.longitudinal_decision = resolve_longitudinal_decision(
      enabled=should_enable_longitudinal_decision_layer(stack_resolution),
      candidates=self.longitudinal_decision_candidates,
      fallback_v_target=v_cruise,
      fallback_a_target=legacy_a_target,
      fallback_should_stop=legacy_should_stop,
      accel_limits=(accel_clip[0], accel_clip[1]),
      arbiter=self.longitudinal_arbiter,
      v_ego=source_stability_v_ego,
    )
    self.longitudinal_decision_telemetry = None
    if self.longitudinal_decision.enabled:
      decision_accel_comfort_active = not (
        reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed or
        v_ego < DECISION_ACCEL_COMFORT_MIN_V_EGO or
        sm['controlsState'].longControlState == LongCtrlState.starting or
        limit_creep_pullaway_accel_step
      )
      self.longitudinal_decision_telemetry = apply_longitudinal_decision_output_with_telemetry(
        self.longitudinal_decision, legacy_a_target, legacy_should_stop,
        prev_a_target=prev_output_a_target,
        personality=sm['selfdriveState'].personality,
        dt=self.dt,
        comfort_active=decision_accel_comfort_active,
      )
      output_a_target = self.longitudinal_decision_telemetry.applied_a_target
      self.output_should_stop = self.longitudinal_decision_telemetry.applied_should_stop

    lead_loss_snapshot_lead = lead_loss_guard_lead
    self.previous_lead_loss_status = lead_loss_snapshot_lead is not None
    self.previous_lead_loss_d_rel = float(lead_loss_snapshot_lead.dRel) if lead_loss_snapshot_lead is not None else 0.0
    self.previous_lead_loss_model_prob = float(lead_loss_snapshot_lead.modelProb) if lead_loss_snapshot_lead is not None else 0.0

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    if output_a_target < 0.0:
      accel_clip[0] = min(accel_clip[0], output_a_target)
    if self.output_should_stop or self.mpc.source == LongitudinalPlanSource.e2e:
      accel_clip[0] = ACCEL_MIN
    low_speed_pullaway_accel_step = lead_one.status and not sm['carState'].brakePressed and not sm['carState'].gasPressed and \
      not force_slow_decel and not reset_state and v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO and \
      (prev_output_a_target > 0.0 or output_a_target > 0.0)
    custom_pullaway_accel_step_floor = None
    custom_pullaway_accel_step_cap = None
    if limit_creep_pullaway_accel_step or low_speed_pullaway_accel_step:
      if output_a_target > -CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP and not self.output_should_stop:
        custom_pullaway_accel_step_floor = prev_output_a_target - CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP
      custom_pullaway_accel_step_cap = prev_output_a_target + CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP
    custom_lead_crawl_accel_cap = None
    if lead_one.status and not creep_pullaway_launch and not lead_pullaway_crawl_cap_released and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO:
      custom_lead_crawl_accel_cap = LEAD_CRAWL_ACCEL_LIMIT
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.planner_seed_candidates = []
    if planner_seed_mpc_a_target is not None:
      planner_seed_mpc_candidate = build_planner_seed_mpc_candidate(
        self, planner_seed_mpc, planner_seed_mpc_a_target, planner_seed_mpc_should_stop, has_lead, accel_clip,
        planner_seed_mpc_v_desired_trajectory, planner_seed_mpc_a_desired_trajectory, planner_seed_mpc_j_desired_trajectory,
        planner_seed_mpc_fcw,
      )
      self.planner_seed_candidate_base_output = LongitudinalStackOutput(
        a_target=float(np.clip(planner_seed_mpc_a_target, accel_clip[0], accel_clip[1])),
        should_stop=bool(planner_seed_mpc_should_stop),
        has_lead=bool(has_lead),
        source=planner_seed_mpc.source,
        allow_throttle=bool(self.allow_throttle),
        allow_brake=True,
        speeds=tuple(float(v) for v in planner_seed_mpc_v_desired_trajectory),
        accels=tuple(float(a) for a in planner_seed_mpc_a_desired_trajectory),
        jerks=tuple(float(j) for j in planner_seed_mpc_j_desired_trajectory),
        fcw=bool(planner_seed_mpc_fcw),
        debug={"planner_seed_candidate_base": "planner_seed_mpc"},
      )
      if planner_seed_mpc_candidate is not None:
        self.planner_seed_candidates.append(planner_seed_mpc_candidate)
    else:
      self.planner_seed_candidate_base_output = None
    if custom_engage_stop_bootstrap_active:
      engage_bootstrap_candidate = build_planner_seed_accel_candidate(
        self, "engage_stop_bootstrap", output_a_target_e2e, has_lead,
        "engage_model_stop_bootstrap", accel_clip, should_stop=output_should_stop_e2e,
      )
      if engage_bootstrap_candidate is not None:
        self.planner_seed_candidates.append(engage_bootstrap_candidate)
    if custom_stopped_stop_gap_guard_a_target is not None:
      stopped_stop_gap_guard_candidate = build_planner_seed_accel_candidate(
        self, "stopped_lead_stop_gap_guard", custom_stopped_stop_gap_guard_a_target, has_lead,
        "stopped_lead_stop_gap_guard", accel_clip, should_stop=True,
      )
      if stopped_stop_gap_guard_candidate is not None:
        self.planner_seed_candidates.append(stopped_stop_gap_guard_candidate)
    if custom_creep_to_stop_gap_a_target is not None:
      creep_to_stop_gap_candidate = build_planner_seed_accel_candidate(
        self, "creep_to_stop_gap", custom_creep_to_stop_gap_a_target, has_lead,
        "creep_to_stop_gap", accel_clip, should_stop=custom_creep_to_stop_gap_should_stop,
        selection=custom_creep_to_stop_gap_selection, group="creep_to_stop_gap",
      )
      if creep_to_stop_gap_candidate is not None:
        self.planner_seed_candidates.append(creep_to_stop_gap_candidate)
    if custom_creep_to_stop_gap_accel_max is not None:
      creep_to_stop_gap_cap_candidate = build_planner_seed_accel_candidate(
        self, "creep_to_stop_gap_accel_cap", custom_creep_to_stop_gap_accel_max, has_lead,
        "creep_to_stop_gap_accel_cap", accel_clip, should_stop=custom_creep_to_stop_gap_should_stop,
        force=True, group="creep_to_stop_gap",
      )
      if creep_to_stop_gap_cap_candidate is not None:
        self.planner_seed_candidates.append(creep_to_stop_gap_cap_candidate)
    if custom_gap_fill_a_target is not None:
      gap_fill_candidate = build_planner_seed_accel_candidate(
        self, "stopped_lead_gap_fill", custom_gap_fill_a_target, has_lead,
        "stopped_lead_gap_fill", accel_clip, should_stop=custom_gap_fill_should_stop,
        selection=custom_gap_fill_selection, group="stopped_lead_gap_fill",
      )
      if gap_fill_candidate is not None:
        self.planner_seed_candidates.append(gap_fill_candidate)
    if custom_gap_fill_accel_max is not None:
      gap_fill_cap_candidate = build_planner_seed_accel_candidate(
        self, "stopped_lead_gap_fill_accel_cap", custom_gap_fill_accel_max, has_lead,
        "stopped_lead_gap_fill_accel_cap", accel_clip, should_stop=custom_gap_fill_should_stop,
        force=True, group="stopped_lead_gap_fill",
      )
      if gap_fill_cap_candidate is not None:
        self.planner_seed_candidates.append(gap_fill_cap_candidate)
    if custom_creep_pullaway_launch_floor is not None:
      creep_pullaway_launch_candidate = build_planner_seed_accel_candidate(
        self, "creep_pullaway_launch", custom_creep_pullaway_launch_floor, has_lead,
        "creep_pullaway_launch", accel_clip, should_stop=False, selection=PLANNER_SEED_FLOOR,
        group="creep_pullaway_launch",
      )
      if creep_pullaway_launch_candidate is not None:
        self.planner_seed_candidates.append(creep_pullaway_launch_candidate)
    if custom_creep_pullaway_launch_cap is not None:
      creep_pullaway_launch_cap_candidate = build_planner_seed_accel_candidate(
        self, "creep_pullaway_launch_accel_cap", custom_creep_pullaway_launch_cap, has_lead,
        "creep_pullaway_launch_accel_cap", accel_clip, should_stop=False, force=True,
        group="creep_pullaway_launch",
      )
      if creep_pullaway_launch_cap_candidate is not None:
        self.planner_seed_candidates.append(creep_pullaway_launch_cap_candidate)
    if custom_pullaway_accel_step_floor is not None:
      pullaway_accel_step_floor_candidate = build_planner_seed_accel_candidate(
        self, "low_speed_pullaway_accel_step_floor", custom_pullaway_accel_step_floor, has_lead,
        "low_speed_pullaway_accel_step_floor", accel_clip, selection=PLANNER_SEED_FLOOR,
        force=True, group="low_speed_pullaway_accel_step",
      )
      if pullaway_accel_step_floor_candidate is not None:
        self.planner_seed_candidates.append(pullaway_accel_step_floor_candidate)
    if custom_pullaway_accel_step_cap is not None:
      pullaway_accel_step_cap_candidate = build_planner_seed_accel_candidate(
        self, "low_speed_pullaway_accel_step_cap", custom_pullaway_accel_step_cap, has_lead,
        "low_speed_pullaway_accel_step_cap", accel_clip, force=True, group="low_speed_pullaway_accel_step",
      )
      if pullaway_accel_step_cap_candidate is not None:
        self.planner_seed_candidates.append(pullaway_accel_step_cap_candidate)
    if custom_lead_crawl_accel_cap is not None:
      lead_crawl_accel_cap_candidate = build_planner_seed_accel_candidate(
        self, "lead_crawl_accel_cap", custom_lead_crawl_accel_cap, has_lead,
        "lead_crawl_accel_cap", accel_clip, force=True,
      )
      if lead_crawl_accel_cap_candidate is not None:
        self.planner_seed_candidates.append(lead_crawl_accel_cap_candidate)
    if custom_creep_hold_a_target is not None:
      creep_hold_candidate = build_planner_seed_accel_candidate(
        self, "stopped_lead_creep_hold", custom_creep_hold_a_target, has_lead,
        "stopped_lead_creep_hold", accel_clip, should_stop=True,
      )
      if creep_hold_candidate is not None:
        self.planner_seed_candidates.append(creep_hold_candidate)
    if custom_moving_stop_guard_a_target is not None:
      moving_stop_guard_candidate = build_planner_seed_accel_candidate(
        self, "moving_lead_stop_gap_guard", custom_moving_stop_guard_a_target, has_lead,
        "moving_lead_stop_gap_guard", accel_clip,
      )
      if moving_stop_guard_candidate is not None:
        self.planner_seed_candidates.append(moving_stop_guard_candidate)
    if custom_lead_accel_recovery_a_target is not None:
      lead_accel_recovery_candidate = build_planner_seed_accel_candidate(
        self, "lead_accel_recovery", custom_lead_accel_recovery_a_target, has_lead,
        "lead_accel_recovery", accel_clip, selection=PLANNER_SEED_FLOOR,
      )
      if lead_accel_recovery_candidate is not None:
        self.planner_seed_candidates.append(lead_accel_recovery_candidate)
    if custom_lead_stop_approach_slewed_a_target is not None and not np.isclose(
      custom_lead_stop_approach_slewed_a_target, custom_lead_stop_approach_base_a_target,
    ):
      lead_stop_approach_slew_selection = (
        PLANNER_SEED_FLOOR if custom_lead_stop_approach_slewed_a_target > custom_lead_stop_approach_base_a_target else PLANNER_SEED_CAP
      )
      lead_stop_approach_slew_candidate = build_planner_seed_accel_candidate(
        self, "lead_stop_approach_slew", custom_lead_stop_approach_slewed_a_target, has_lead,
        "lead_stop_approach_slew", accel_clip,
        selection=lead_stop_approach_slew_selection,
        force=True, group="lead_stop_approach_slew" if lead_stop_approach_slew_selection == PLANNER_SEED_FLOOR else "",
      )
      if lead_stop_approach_slew_candidate is not None:
        self.planner_seed_candidates.append(lead_stop_approach_slew_candidate)
    if self.e2e_close_stop_settle_active:
      e2e_close_stop_candidate = build_planner_seed_accel_candidate(
        self, "e2e_close_stop_settle", custom_e2e_close_stop_a_target, has_lead,
        "no_lead_close_stop_settle", accel_clip, should_stop=custom_close_stop_should_stop,
      )
      if e2e_close_stop_candidate is not None:
        self.planner_seed_candidates.append(e2e_close_stop_candidate)
    if custom_e2e_runway_comfort_output_a_target is not None:
      e2e_runway_comfort_candidate = build_planner_seed_accel_candidate(
        self, "e2e_runway_comfort", custom_e2e_runway_comfort_output_a_target, has_lead,
        "no_lead_model_runway_comfort", accel_clip, selection=PLANNER_SEED_FLOOR,
      )
      if e2e_runway_comfort_candidate is not None:
        self.planner_seed_candidates.append(e2e_runway_comfort_candidate)
    if custom_lead_loss_e2e_guard_a_target is not None:
      lead_loss_e2e_guard_candidate = build_planner_seed_accel_candidate(
        self, "lead_loss_e2e_guard", custom_lead_loss_e2e_guard_a_target, has_lead,
        "lead_loss_e2e_guard", accel_clip, selection=PLANNER_SEED_FLOOR,
      )
      if lead_loss_e2e_guard_candidate is not None:
        self.planner_seed_candidates.append(lead_loss_e2e_guard_candidate)
    if custom_e2e_stop_approach_a_target < 0.0:
      e2e_stop_approach_candidate = build_planner_seed_accel_candidate(
        self, "e2e_stop_approach", custom_e2e_stop_approach_a_target, has_lead, "no_lead_model_stop_approach", accel_clip,
      )
      if e2e_stop_approach_candidate is not None:
        self.planner_seed_candidates.append(e2e_stop_approach_candidate)
    e2e_runway_positive_cap_candidate = build_planner_seed_accel_candidate(
      self, "e2e_runway_positive_cap", e2e_runway_positive_accel_cap, has_lead,
      "low_speed_model_runway_positive_cap", accel_clip,
    )
    if e2e_runway_positive_cap_candidate is not None:
      self.planner_seed_candidates.append(e2e_runway_positive_cap_candidate)
    if should_apply_cruise_coast_overspeed(
      reset_state, force_slow_decel, e2e_active, has_lead, self.output_should_stop, self.source
    ):
      cruise_coast_candidate = build_planner_seed_accel_candidate(
        self, "cruise_coast", apply_cruise_coast_overspeed(v_ego, v_cruise, cruise_coast_accel, self.output_a_target),
        has_lead, "plain_cruise_overspeed_coast", accel_clip, selection=PLANNER_SEED_FLOOR,
      )
      if cruise_coast_candidate is not None:
        self.planner_seed_candidates.append(cruise_coast_candidate)
    active_scc = getattr(self, "active_scc", None) or getattr(self, "scc", None)
    active_scc_vision = getattr(active_scc, "vision", None)
    active_scc_map = getattr(active_scc, "map", None)
    active_sla = getattr(self, "active_sla", None) or getattr(self, "sla", None)
    osm_traffic_control_prior = getattr(self, "osm_traffic_control_prior", None)
    custom_v2_curve_active, custom_v2_curve_a_target = get_custom_v2_curve_scene_target(active_scc_vision, active_scc_map)
    self.custom_v2_scene = CustomV2Scene(
      v_ego=float(v_ego),
      v_cruise=float(v_cruise),
      a_ego=float(sm['carState'].aEgo),
      accel_coast=float(cruise_coast_accel),
      force_slow_decel=bool(force_slow_decel),
      brake_pressed=bool(sm['carState'].brakePressed),
      gas_pressed=bool(sm['carState'].gasPressed),
      has_lead=bool(has_lead),
      lead_v=float(lead_one.vLeadK) if lead_one.status else 0.0,
      lead_v_rel=lead_v_rel,
      lead_gap_excess=float(lead_gap_excess),
      lead_opening_prediction=bool(radar_predicted_pullaway or model_predicted_pullaway),
      lead_confirmed_pullaway=bool(creep_pullaway_release),
      stop_threat=bool(custom_e2e_stop_approach_a_target < 0.0 or self.e2e_close_stop_settle_active or output_should_stop_e2e),
      independent_stop_threat=bool(not lead_one.status and (custom_e2e_stop_approach_a_target < 0.0 or output_should_stop_e2e)),
      model_should_stop=bool(output_should_stop_e2e),
      model_stop_distance=get_model_stop_distance(sm['modelV2']),
      model_desired_accel=float(output_a_target_e2e),
      speed_limit_active=bool(getattr(active_sla, "is_active", False)),
      speed_limit_v_target=float(getattr(active_sla, "output_v_target", 0.0)),
      speed_limit_a_target=float(getattr(active_sla, "output_a_target", 0.0)),
      curve_active=custom_v2_curve_active,
      curve_a_target=custom_v2_curve_a_target,
      map_caution_active=bool(getattr(osm_traffic_control_prior, "active", False)),
      map_caution_confirmed=bool(getattr(osm_traffic_control_prior, "active", False)),
      map_caution_a_target=float(getattr(osm_traffic_control_prior, "output_a_target", 0.0)),
      one_pedal_mode=int(getattr(self, "one_pedal_mode", ONE_PEDAL_MODE_OFF)),
      one_pedal_cruise_hold=bool(getattr(self, "one_pedal_cruise_hold_active", False)),
    )
    self.prev_accel_clip = accel_clip
    self.apply_longitudinal_stack_selection(sm, has_lead, tuple(accel_clip))

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime - sm.logMonoTime['modelV2']) / 1e9
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = has_valid_radar_lead(sm['radarState'])
    longitudinalPlan.longitudinalPlanSource = getattr(self, "longitudinal_plan_source", self.mpc.source)
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
