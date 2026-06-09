#!/usr/bin/env python3
from dataclasses import dataclass, field, replace
from enum import Enum
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
from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolver,
  ResolvedLongitudinalImplementation,
  SccModeEvidence,
)
from openpilot.selfdrive.controls.lib.longitudinal_profile import jerk_limited_braking_profile
from openpilot.selfdrive.controls.lib.scc_evidence import SccEvidenceSelector, SccEvidenceTier
from openpilot.selfdrive.controls.lib.lead_confidence import (
  LEAD_CONFIDENCE_TRACK_UNKNOWN,
  LeadConfidenceState,
  LEAD_FLICKER_CLOSE_COUNT_THRESHOLD,
  LEAD_FLICKER_CLOSE_D_REL,
  LEAD_FLICKER_CLOSE_GUARD_TIME,
  LEAD_FLICKER_CLOSE_V_LEAD,
  LEAD_FLICKER_COUNT_THRESHOLD,
  LEAD_FLICKER_GUARD_TIME,
  LEAD_FLICKER_WINDOW,
)
from openpilot.selfdrive.controls.lib.lead_context import (
  LEAD_CONTEXT_RISK_REQUIRED_DECEL,
  LEAD_CONTEXT_RISK_TTC,
  LeadContextTracker,
  PrimaryLeadContext,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.adapters import planner_state_to_stack_output
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  CustomV2Scene,
  LEAD_LATERAL_PROGRESS_BLOCK_Y,
  ONE_PEDAL_MODE_OFF,
  ONE_PEDAL_MODES,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
  PLANNER_SEED_CAP,
  PLANNER_SEED_FLOOR,
  PLANNER_SEED_MPC_REASON,
  PlannerSeedCandidate,
  planner_seed_intent_for_reason,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2, is_custom_stack
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, SunnypilotLongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE,
  LEAD_CRAWL_ACCEL_LIMIT,
  get_T_FOLLOW,
  get_desired_follow_distance,
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

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL,
  LongitudinalPlannerSP,
  should_block_lead_speedup,
)

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ButtonType = car.CarState.ButtonEvent.Type
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
ONE_PEDAL_LONGITUDINAL_MODE_PARAM = "OnePedalLongitudinalMode"
FAST_LEAD_MOTION_EVIDENCE_PARAM = "FastLeadMotionEvidenceEnabled"
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
CREEP_TO_STOP_GAP_RESERVE_CREEP_MAX_V_EGO = 0.08
CREEP_TO_STOP_GAP_RESERVE_CREEP_ACCEL_FLOOR = 0.0
CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED = 0.15
FAST_LEAD_MOTION_OPENING_DEADBAND = CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
CREEP_TO_STOP_GAP_PULLAWAY_SPEED_MAX = 1.2
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX = 0.55
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN = 0.30
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN = 0.70
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX = 0.75
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MAX = 1.20
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS = CREEP_TO_STOP_GAP_STOP_EXCESS
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MAX_EXCESS = 8.0
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MIN_LEAD_ACCEL = 0.6
CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_OPENING_MAX_V_EGO = CREEP_TO_STOP_GAP_MAX_V_EGO
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP = 7.5 * DT_MDL
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO = 3.0
CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_HANDOFF_TIME = 0.5
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
STOPPED_LEAD_STOP_GAP_GUARD_REBOUND_JERK = 7.5
STOPPED_LEAD_STOP_GAP_GUARD_LANE_CHANGE_MAX_TTC = 4.0
STOPPED_LEAD_MOVING_REBOUND_HOLD_TIME = 10.0
MOVING_LEAD_STOP_GAP_GUARD_MIN_V_EGO = 6.0
MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD = 0.5
MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL = 0.25
MOVING_LEAD_STOP_GAP_GUARD_MIN_TARGET_DECEL = 0.4
MOVING_LEAD_STOP_GAP_GUARD_MAX_Y_REL = 1.5
MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP = 1.95
MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL = 2.0
MOVING_LEAD_STOP_GAP_GUARD_PREDICT_T = 0.8
MOVING_LEAD_STOP_GAP_GUARD_ALLOWED_CLOSING = 1.2
MOVING_LEAD_STOP_GAP_GUARD_CLOSING_DECEL_CAP = 1.2
MOVING_LEAD_STOP_GAP_GUARD_URGENT_CLOSING = 3.0
MOVING_LEAD_STOP_GAP_GUARD_URGENT_REQUIRED_DECEL = 3.0
MOVING_LEAD_STOP_GAP_GUARD_PRE_DANGER_URGENT_MARGIN = 2.0
MOVING_LEAD_STOP_GAP_GUARD_PRE_DANGER_URGENT_TTC = 2.0
MOVING_LEAD_SLOWER_APPROACH_MIN_CLOSING = 0.3
MOVING_LEAD_SLOWER_APPROACH_MIN_GAP_DEFICIT = 0.25
MOVING_LEAD_SLOWER_APPROACH_MIN_TARGET_DECEL = 0.12
MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP = 0.45
MOVING_LEAD_SLOWER_APPROACH_CLOSING_GAIN = 0.25
MOVING_LEAD_SLOWER_APPROACH_GAP_GAIN = 0.03
MOVING_LEAD_SLOWER_APPROACH_LEAD_DECEL_GAIN = 0.05
ROUTINE_LEAD_APPROACH_PREVIEW_T = 2.0
ROUTINE_LEAD_RESPONSE_TIME = 0.35
ROUTINE_LEAD_APPROACH_MIN_CLOSING = 0.25
ROUTINE_LEAD_APPROACH_MIN_BLEND = 0.05
ROUTINE_LEAD_APPROACH_DANGER_GAP_MARGIN = 1.0
ROUTINE_LEAD_APPROACH_COAST_BLEND = 0.18
ROUTINE_LEAD_APPROACH_SOFT_BLEND = 0.45
ROUTINE_LEAD_APPROACH_SOFT_DECEL_CAP = 0.25
ROUTINE_LEAD_APPROACH_DECEL_MIN = MOVING_LEAD_SLOWER_APPROACH_MIN_TARGET_DECEL
ROUTINE_LEAD_APPROACH_DECEL_CAP = MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP
ROUTINE_LEAD_APPROACH_FIRM_DECEL_CAP = 1.5
ROUTINE_LEAD_APPROACH_NEGATIVE_JERK = 0.45
ROUTINE_LEAD_APPROACH_RELEASE_JERK = 0.25
ROUTINE_LEAD_FAR_COAST_TTC = 7.0  # seconds - time to caution gap threshold for far coast
ROUTINE_LEAD_FAR_COAST_MIN_CLOSING = 0.25  # m/s - minimum closing speed for far coast
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
SCC_NEAR_ENDPOINT_MODEL_STOP_MAX_DISTANCE = 12.0
SCC_NEAR_ENDPOINT_MODEL_STOP_MAX_V = 0.5
SCC_NEAR_ENDPOINT_MODEL_STOP_ACCEL = -1.0
SCC_EARLY_MODEL_STOP_ACCEL = -1.0
SCC_EARLY_MODEL_STOP_MIN_INITIAL_V = 8.0
SCC_EARLY_MODEL_STOP_MAX_MID_V = 4.0
SCC_EARLY_MODEL_STOP_MIN_SPEED_DROP = 6.0
SCC_EARLY_MODEL_STOP_ENDPOINT_MARGIN = 5.0
SCC_EARLY_MODEL_STOP_MIN_REQUIRED_DECEL = 1.0
SCC_EARLY_MODEL_STOP_EXPECTED_DISTANCE_SCALE = 1.0
SCC_MODEL_SLOWDOWN_ACCEL = -0.8
SCC_MODEL_SLOWDOWN_MIN_SPEED_DROP = 4.0
SCC_MODEL_SLOWDOWN_MIN_ENDPOINT_DROP = 3.0
E2E_RUNWAY_COMFORT_MIN_V_EGO = 3.0
E2E_RUNWAY_COMFORT_MIN_ENDPOINT = 1.0
E2E_RUNWAY_COMFORT_COAST_MARGIN = 0.02
E2E_RUNWAY_COMFORT_LIGHT_DECEL = 0.30
E2E_RUNWAY_COMFORT_TRACTION_LIGHT_DECEL = 0.25
E2E_RUNWAY_COMFORT_DECEL_BLEND_BP = [1.2, 2.0]
E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP = [0.5, 1.0]
E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE = 0.35
TRACTION_RISK_RUNWAY_SCALE_MAX = 1.25
TRACTION_RISK_NEGATIVE_RAMP_MIN_SCALE = 0.75
TRACTION_RISK_LEAD_STOP_SLEW_MIN_SCALE = 0.45
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
LEAD_FLICKER_SPEEDUP_CAP_REASON = "lead_flicker_speedup_cap"
LEAD_FLICKER_SPEEDUP_CAP_A_TARGET_MAX = 0.0
LEAD_FLICKER_FIRST_LOSS_HOLD_TIME = 0.5
LEAD_FLICKER_FAR_CLOSING_SPEED_MIN = 1.0
LEAD_FLICKER_FAR_REQUIRED_DECEL_MIN = 0.25
LEAD_LATERAL_PROGRESS_HOLD_TIME = 2.0
ROUTINE_LEAD_APPROACH_SEED_REASON = "routine_slower_lead_approach"
LEAD_PULLAWAY_PULSE_REASON = "confirmed_lead_pullaway_pulse"
LEAD_PULLAWAY_PULSE_CAP_REASON = "lead_pullaway_pulse_accel_cap"
EXCESS_GAP_CLOSURE_REASON = "excess_gap_closure"
EXCESS_GAP_CLOSURE_CAP_REASON = "excess_gap_closure_accel_cap"
LEAD_PULLAWAY_PULSE_DURATION = 0.8
LEAD_PULLAWAY_PULSE_COOLDOWN = 2.0
LEAD_PULLAWAY_MAX_V_EGO = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO
LEAD_PULLAWAY_PULSE_A_FLOOR = CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN
LEAD_PULLAWAY_PULSE_ACCEL_CAP = CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MAX
LEAD_PULLAWAY_PULSE_JERK_UP = 2.0
LEAD_PULLAWAY_PULSE_JERK_DOWN = 5.0
LEAD_PULLAWAY_RUNWAY_T = 2.0
LEAD_PULLAWAY_RUNWAY_MARGIN = 0.5
LEAD_PULLAWAY_RUNWAY_CREATION_MIN = 0.10
LEAD_PULLAWAY_EARLY_MIN_SAFE_ACCEL = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN
LEAD_PULLAWAY_EARLY_MIN_LEAD_ACCEL = 2.0 * CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL
LEAD_PULLAWAY_CRAWL_CAP_JERK_BUFFER = 0.03
LEAD_PULLAWAY_RUNWAY_CAP_RELEASE_JERK = 7.5
LEAD_PULLAWAY_COOLDOWN_ACCEL_CAP = 0.84
LEAD_PULLAWAY_ACCEL_TREND_DECREASE = -0.05
LEAD_PULLAWAY_DECEL_ABORT = -0.05
EXCESS_GAP_CLOSURE_START_EXCESS = CREEP_TO_STOP_GAP_MAX_EXCESS
EXCESS_GAP_CLOSURE_FULL_EXCESS = 8.0
EXCESS_GAP_CLOSURE_ACCEL_BASE = CREEP_TO_STOP_GAP_ACCEL_MAX
EXCESS_GAP_CLOSURE_ACCEL_CAP = STOPPED_LEAD_GAP_FILL_ACCEL_MAX
EXCESS_GAP_CLOSURE_JERK_UP = 0.8
EXCESS_GAP_CLOSURE_JERK_DOWN = 4.0
STOP_RELEASE_GUARD_HOLD_TIME = 1.5
STOP_RELEASE_GUARD_MAX_V_EGO = 0.3
STOP_RELEASE_GUARD_WAITING_REASON = "waiting_for_stop_clear"
STOP_RELEASE_GUARD_LEAD_RELEASE_REASON = "lead_confirmed_release"
STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON = "lead_confirmed_capped_release"
STOP_RELEASE_GUARD_DRIVER_REASON = "driver_override"
STOP_RELEASE_GUARD_FORCE_BLOCK_REASON = "driver_or_force_blocked"

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20.0, 40.0]
CURVE_LOAD_COMFORT_MIN_V_EGO = 8.0
CURVE_LOAD_COMFORT_TAPER_START = 0.55
CURVE_LOAD_COMFORT_TAPER_FULL = 0.90


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def has_valid_radar_lead(radar_state):
  return radar_state.leadOne.status or radar_state.leadTwo.status


def empty_primary_lead_context():
  return PrimaryLeadContext(
    physical_idx=None,
    behavior_idx=None,
    physical=None,
    behavior=None,
    alternate_threat_active=False,
    shadow_active=False,
    reason="no_lead",
    states=(),
    lead_progress_allowed=False,
    lead_release_blocked_reason="",
  )


def get_mpc_lead_confidence_states(mpc):
  states = getattr(mpc, "lead_confidence_states", None)
  if states is not None and len(states) >= 2:
    return (states[0], states[1])
  # Unit-test fixtures may bypass LongitudinalMpc construction. In production,
  # real MPC instances always expose tracker-backed states, so this fallback
  # preserves legacy fixture behavior without weakening runtime new-lead guards.
  fallback = LeadConfidenceState(status=True, stable=True, speed_trusted=True, age=1.0, accel_blend=1.0)
  return (fallback, fallback)


def _finite_float(value, default=0.0):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return default
  return value if math.isfinite(value) else default


@dataclass(frozen=True)
class FastLeadMotionEvidence:
  v_lead: float = 0.0
  v_rel: float = 0.0

  def opening(self, deadband=FAST_LEAD_MOTION_OPENING_DEADBAND) -> bool:
    return self.v_rel >= deadband

  def moving(self, threshold=CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED) -> bool:
    return self.v_lead >= threshold


@dataclass(frozen=True)
class LongitudinalComfortBudget:
  far_coast_ttc: float = ROUTINE_LEAD_FAR_COAST_TTC
  soft_decel_cap: float = ROUTINE_LEAD_APPROACH_SOFT_DECEL_CAP
  routine_decel_cap: float = ROUTINE_LEAD_APPROACH_DECEL_CAP
  firm_routine_decel_cap: float = ROUTINE_LEAD_APPROACH_FIRM_DECEL_CAP
  routine_negative_jerk: float = ROUTINE_LEAD_APPROACH_NEGATIVE_JERK
  routine_release_jerk: float = ROUTINE_LEAD_APPROACH_RELEASE_JERK
  response_time: float = ROUTINE_LEAD_RESPONSE_TIME


COMFORT_BUDGET_BY_PERSONALITY = {
  0: LongitudinalComfortBudget(  # relaxed
    far_coast_ttc=9.0,
    soft_decel_cap=0.20,
    routine_decel_cap=0.35,
    firm_routine_decel_cap=1.2,
    routine_negative_jerk=0.35,
    routine_release_jerk=0.20,
    response_time=0.40,
  ),
  1: LongitudinalComfortBudget(  # standard
    far_coast_ttc=7.0,
    soft_decel_cap=0.25,
    routine_decel_cap=0.45,
    firm_routine_decel_cap=1.5,
    routine_negative_jerk=0.45,
    routine_release_jerk=0.25,
    response_time=0.35,
  ),
  2: LongitudinalComfortBudget(  # aggressive
    far_coast_ttc=5.5,
    soft_decel_cap=0.30,
    routine_decel_cap=0.55,
    firm_routine_decel_cap=1.8,
    routine_negative_jerk=0.55,
    routine_release_jerk=0.30,
    response_time=0.30,
  ),
}


def get_comfort_budget(personality: int = 1) -> LongitudinalComfortBudget:
  return COMFORT_BUDGET_BY_PERSONALITY.get(personality, COMFORT_BUDGET_BY_PERSONALITY[1])


@dataclass(frozen=True)
class RoutineLeadApproach:
  active: bool = False
  far_coast_active: bool = False
  urgent: bool = False
  raw_a_target: float = 0.0
  ramped_a_target: float = 0.0
  required_decel: float = 0.0
  allowed_closing_speed: float = 0.0
  closing_excess: float = 0.0
  compression_blend: float = 0.0
  compression_budget: float = 0.0
  comfort_budget: float = 0.0
  firm_routine_decel_cap: float = ROUTINE_LEAD_APPROACH_FIRM_DECEL_CAP
  projected_compression_budget: float = 0.0
  projected_comfort_budget: float = 0.0
  valid_lead_approach: bool = False
  predicted_gap: float = 0.0
  projected_closing_speed: float = 0.0
  reason: str = "inactive"
  debug: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StopReleaseGuardState:
  active: bool = False
  reason: str = "clear"
  recent_stop_timer: float = 0.0
  lead_confirmed_release: bool = False
  applied: bool = False
  release_accel_cap: float = 0.0


@dataclass
class StopReleaseGuardTracker:
  _recent_stop_timer: float = 0.0

  def reset(self) -> None:
    self._recent_stop_timer = 0.0

  def update(self, *, v_ego, standstill, stop_evidence_active, lead_confirmed_release,
             reset_state=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
             dt=DT_MDL) -> StopReleaseGuardState:
    dt = max(0.0, _finite_float(dt))
    if reset_state:
      self.reset()
      return StopReleaseGuardState(reason="reset")

    if stop_evidence_active:
      self._recent_stop_timer = STOP_RELEASE_GUARD_HOLD_TIME
    else:
      self._recent_stop_timer = max(0.0, self._recent_stop_timer - dt)

    near_stopped = bool(standstill or _finite_float(v_ego) <= STOP_RELEASE_GUARD_MAX_V_EGO)
    if not near_stopped or self._recent_stop_timer <= 0.0:
      return StopReleaseGuardState(reason="clear", recent_stop_timer=self._recent_stop_timer)

    if bool(gas_pressed):
      return StopReleaseGuardState(
        reason=STOP_RELEASE_GUARD_DRIVER_REASON,
        recent_stop_timer=self._recent_stop_timer,
      )
    if bool(brake_pressed or force_slow_decel):
      return StopReleaseGuardState(
        active=True,
        reason=STOP_RELEASE_GUARD_FORCE_BLOCK_REASON,
        recent_stop_timer=self._recent_stop_timer,
      )
    if bool(lead_confirmed_release):
      return StopReleaseGuardState(
        active=True,
        reason=STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON,
        recent_stop_timer=self._recent_stop_timer,
        lead_confirmed_release=True,
        release_accel_cap=LEAD_PULLAWAY_PULSE_A_FLOOR,
      )

    return StopReleaseGuardState(
      active=True,
      reason=STOP_RELEASE_GUARD_WAITING_REASON,
      recent_stop_timer=self._recent_stop_timer,
    )


def apply_stop_release_guard_accel(a_target, guard: StopReleaseGuardState) -> tuple[float, StopReleaseGuardState]:
  a_target = _finite_float(a_target)
  if guard.active and a_target > 0.0:
    cap = _finite_float(guard.release_accel_cap)
    if cap > 0.0:
      clamped = min(a_target, cap)
      return clamped, replace(guard, applied=clamped < a_target)
    return 0.0, replace(guard, applied=True)
  return a_target, replace(guard, applied=False)


def lead_confirmed_stop_release(primary_lead_context: PrimaryLeadContext, behavior_lead, *, lead_opening=False,
                                lead_moving=False, lead_accel=0.0, predicted_gap_opening=0.0,
                                independent_stop_threat=False, brake_pressed=False, gas_pressed=False,
                                force_slow_decel=False) -> bool:
  if bool(independent_stop_threat or brake_pressed or gas_pressed or force_slow_decel):
    return False
  if not bool(getattr(primary_lead_context, "lead_progress_allowed", False)):
    return False
  if bool(getattr(primary_lead_context, "alternate_threat_active", False) or getattr(primary_lead_context, "shadow_active", False)):
    return False

  behavior_state = getattr(primary_lead_context, "behavior", None)
  if behavior_lead is None or behavior_state is None:
    return False
  if bool(getattr(behavior_state, "shadow", False) or getattr(behavior_state, "flicker_guard_timer", 0.0) > 0.0):
    return False
  if bool(getattr(behavior_state, "new_lead", False) or not getattr(behavior_state, "stable", False)):
    return False

  progress_model = getattr(behavior_state, "progress_model", None)
  if progress_model is None:
    return False
  if not bool(getattr(progress_model, "confidence_stability_sufficient", False)):
    return False
  if not bool(getattr(progress_model, "alternate_threat_absent", True) and getattr(progress_model, "shadow_absent", True)):
    return False
  if str(getattr(behavior_state, "reason", "")) == "stable_stopped_gap_creep_authorized_lead":
    return True
  if not bool(getattr(progress_model, "allowed", False)):
    return False
  if not bool(getattr(progress_model, "stop_threat_absent", False)):
    return False

  opening_evidence = bool(
    lead_opening or
    lead_moving or
    _finite_float(getattr(progress_model, "opening_speed", 0.0)) >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED or
    _finite_float(lead_accel) >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL or
    _finite_float(predicted_gap_opening) >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING or
    bool(getattr(progress_model, "predicted_gap_opening", False))
  )
  return bool(opening_evidence)


def get_fast_lead_motion_evidence(lead, v_ego) -> FastLeadMotionEvidence:
  stable_v_lead = _finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))
  v_rel = _finite_float(getattr(lead, "vRel", stable_v_lead - _finite_float(v_ego)), stable_v_lead - _finite_float(v_ego))
  v_lead = _finite_float(getattr(lead, "vLead", math.nan), math.nan)
  if not math.isfinite(v_lead):
    v_lead = _finite_float(v_ego) + v_rel
  return FastLeadMotionEvidence(v_lead=v_lead, v_rel=v_rel)


def fast_lead_motion_evidence_enabled(stack_resolution, param_enabled) -> bool:
  return bool(param_enabled and getattr(stack_resolution, "resolved_stack", "") == CUSTOM_V2)


def get_planner_lead_motion_values(lead, v_ego, use_fast_evidence) -> tuple[float, float, FastLeadMotionEvidence]:
  fast_evidence = get_fast_lead_motion_evidence(lead, v_ego)
  # Raw fast evidence is returned separately; planner/control values stay stable regardless of the setting.
  stable_v_lead = get_lead_v_lead(lead)
  stable_v_rel = stable_v_lead - _finite_float(v_ego)
  return stable_v_lead, stable_v_rel, fast_evidence


def get_lead_flicker_required_decel(d_rel, v_rel):
  closing_speed = max(0.0, -_finite_float(v_rel))
  return closing_speed**2 / (2.0 * max(_finite_float(d_rel) - STOP_DISTANCE, 0.1))


def should_cap_lead_flicker_speedup(v_ego, lead_status, d_rel, v_rel, v_lead, y_rel):
  if not lead_status or abs(_finite_float(y_rel)) >= LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL:
    return False

  d_rel = _finite_float(d_rel)
  v_rel = _finite_float(v_rel)
  v_lead = _finite_float(v_lead)
  if should_block_lead_speedup(v_ego, True, d_rel, v_rel, _finite_float(y_rel), False, False):
    return True

  closing_speed = max(0.0, -v_rel, _finite_float(v_ego) - v_lead)
  required_decel = get_lead_flicker_required_decel(d_rel, -closing_speed)
  return closing_speed >= LEAD_FLICKER_FAR_CLOSING_SPEED_MIN and required_decel >= LEAD_FLICKER_FAR_REQUIRED_DECEL_MIN


@dataclass(frozen=True)
class LeadFlickerSafetyCapState:
  active: bool = False
  timer: float = 0.0
  risky_lead: bool = False


@dataclass
class LeadFlickerSafetyCapTracker:
  _prev_status: bool = False
  _transitions: list[float] = field(default_factory=list)
  _timer: float = 0.0
  _last_d_rel: float = 0.0
  _last_v_rel: float = 0.0
  _last_v_lead: float = 0.0
  _last_y_rel: float = 0.0
  _last_risky_lead: bool = False

  def update(self, lead, v_ego, dt, reset_state=False, force_slow_decel=False, gas_pressed=False, brake_pressed=False):
    dt = max(_finite_float(dt), 0.0)
    if reset_state:
      self._prev_status = False
      self._transitions.clear()
      self._timer = 0.0
      self._last_risky_lead = False
      return LeadFlickerSafetyCapState()

    self._timer = max(0.0, self._timer - dt)
    status = bool(getattr(lead, "status", False)) if lead is not None else False
    d_rel, v_rel, v_lead, y_rel = self._lead_values(lead) if status else (
      self._last_d_rel,
      self._last_v_rel,
      self._last_v_lead,
      self._last_y_rel,
    )
    risky_lead = should_cap_lead_flicker_speedup(v_ego, status, d_rel, v_rel, v_lead, y_rel)
    if status:
      self._last_d_rel = d_rel
      self._last_v_rel = v_rel
      self._last_v_lead = v_lead
      self._last_y_rel = y_rel
      self._last_risky_lead = risky_lead

    close_stop_go_context = self._close_stop_go_context(d_rel, v_lead, y_rel)
    risk_context = risky_lead or self._last_risky_lead or close_stop_go_context
    if status != self._prev_status and risk_context:
      self._transitions.append(0.0)
      if not status and (self._last_risky_lead or close_stop_go_context):
        self._timer = max(self._timer, LEAD_FLICKER_FIRST_LOSS_HOLD_TIME)
    self._prev_status = status

    self._transitions = [t + dt for t in self._transitions if t + dt <= LEAD_FLICKER_WINDOW]
    if risk_context and len(self._transitions) >= LEAD_FLICKER_COUNT_THRESHOLD:
      self._timer = max(self._timer, LEAD_FLICKER_GUARD_TIME)
    if close_stop_go_context and len(self._transitions) >= LEAD_FLICKER_CLOSE_COUNT_THRESHOLD:
      self._timer = max(self._timer, LEAD_FLICKER_CLOSE_GUARD_TIME)

    blocked = bool(force_slow_decel or gas_pressed or brake_pressed)
    return LeadFlickerSafetyCapState(
      active=bool(not blocked and (risky_lead or self._timer > 0.0)),
      timer=self._timer,
      risky_lead=bool(risky_lead),
    )

  @staticmethod
  def _lead_values(lead):
    d_rel = _finite_float(getattr(lead, "dRel", 0.0))
    v_lead = _finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))
    v_rel = _finite_float(getattr(lead, "vRel", v_lead))
    y_rel = _finite_float(getattr(lead, "yRel", 0.0))
    return d_rel, v_rel, v_lead, y_rel

  @staticmethod
  def _close_stop_go_context(d_rel, v_lead, y_rel=0.0):
    return (
      abs(y_rel) < LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL and
      0.0 < d_rel <= LEAD_FLICKER_CLOSE_D_REL and
      0.0 <= v_lead <= LEAD_FLICKER_CLOSE_V_LEAD
    )


class LeadPullawayPhase(Enum):
  HOLD = "hold"
  ARMED = "armed"
  PULSE = "pulse"
  GAP_CLOSURE = "gap_closure"
  NORMAL = "normal"


@dataclass(frozen=True)
class LeadPullawayIntent:
  phase: LeadPullawayPhase = LeadPullawayPhase.HOLD
  active: bool = False
  a_floor: float = 0.0
  reason: str = "no_lead_progress_authority"
  track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  pulse_timer: float = 0.0
  cooldown_timer: float = 0.0
  gap_excess: float = 0.0
  predicted_gap_opening: float = 0.0
  predicted_gap: float = 0.0
  safe_accel_cap: float = LEAD_PULLAWAY_PULSE_ACCEL_CAP
  lead_accel_trend: float = 0.0
  runway_margin: float = 0.0
  runway_margin_now: float = 0.0
  runway_margin_t: float = 0.0
  runway_creation: float = 0.0
  lead_created_runway: bool = False
  early_authority: bool = False
  early_authority_reason: str = ""
  coast_required: bool = False
  pulse_capped_by_runway: bool = False
  runway_trend: str = "stable"
  crawl_cap_released_by_runway: bool = False
  low_speed_step_cap_suppressed_by_runway: bool = False


@dataclass(frozen=True)
class LeadPullawayRunway:
  predicted_gap: float = 0.0
  safe_accel_cap: float = LEAD_PULLAWAY_PULSE_ACCEL_CAP
  lead_accel_trend: float = 0.0
  runway_margin: float = 0.0
  runway_margin_now: float = 0.0
  runway_margin_t: float = 0.0
  runway_creation: float = 0.0
  lead_created_runway: bool = False
  coast_required: bool = False
  trend: str = "stable"


def get_lead_pullaway_runway(v_ego, d_rel, v_lead, a_lead, lead_accel_trend,
                             desired_gap=STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
                             margin=LEAD_PULLAWAY_RUNWAY_MARGIN,
                             horizon=LEAD_PULLAWAY_RUNWAY_T) -> LeadPullawayRunway:
  v_ego = max(0.0, _finite_float(v_ego))
  d_rel = max(0.0, _finite_float(d_rel))
  v_lead = max(0.0, _finite_float(v_lead))
  a_lead = _finite_float(a_lead)
  lead_accel_trend = _finite_float(lead_accel_trend)
  desired_gap = max(0.0, _finite_float(desired_gap))
  margin = max(0.0, _finite_float(margin))
  horizon = max(0.1, _finite_float(horizon, LEAD_PULLAWAY_RUNWAY_T))

  if a_lead <= LEAD_PULLAWAY_DECEL_ABORT:
    trend = "decelerating"
    a_lead_pred = min(a_lead, 0.0)
  elif lead_accel_trend < LEAD_PULLAWAY_ACCEL_TREND_DECREASE:
    trend = "decreasing"
    a_lead_pred = max(0.0, a_lead + lead_accel_trend * horizon)
  elif a_lead > CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL:
    trend = "increasing"
    a_lead_pred = a_lead
  else:
    trend = "stable"
    a_lead_pred = max(0.0, a_lead)

  runway_margin_now = d_rel - desired_gap - margin
  runway_without_ego_accel = runway_margin_now + (v_lead - v_ego) * horizon
  a_cap = a_lead_pred + 2.0 * runway_without_ego_accel / horizon**2
  safe_accel_cap = float(np.clip(a_cap, 0.0, LEAD_PULLAWAY_PULSE_ACCEL_CAP))
  if trend == "decelerating":
    safe_accel_cap = 0.0
  predicted_gap = max(0.0, d_rel + (v_lead - v_ego) * horizon + 0.5 * (a_lead_pred - safe_accel_cap) * horizon**2)
  runway_margin_t = predicted_gap - desired_gap - margin
  runway_creation = runway_margin_t - runway_margin_now
  coast_required = bool(a_lead <= LEAD_PULLAWAY_DECEL_ABORT or safe_accel_cap <= 1e-3)
  lead_created_runway = bool(
    runway_creation >= LEAD_PULLAWAY_RUNWAY_CREATION_MIN and
    safe_accel_cap >= LEAD_PULLAWAY_EARLY_MIN_SAFE_ACCEL and
    not coast_required
  )
  return LeadPullawayRunway(
    predicted_gap=predicted_gap,
    safe_accel_cap=safe_accel_cap,
    lead_accel_trend=lead_accel_trend,
    runway_margin=runway_margin_t,
    runway_margin_now=runway_margin_now,
    runway_margin_t=runway_margin_t,
    runway_creation=runway_creation,
    lead_created_runway=lead_created_runway,
    coast_required=coast_required,
    trend=trend,
  )


def approach_accel_with_jerk_limit(prev_a, target_a, dt, jerk_up, jerk_down):
  prev_a = _finite_float(prev_a)
  target_a = _finite_float(target_a)
  dt = max(0.0, _finite_float(dt))
  jerk_up = max(0.0, _finite_float(jerk_up))
  jerk_down = max(0.0, _finite_float(jerk_down))
  if target_a > prev_a:
    return min(target_a, prev_a + jerk_up * dt)
  return max(target_a, prev_a - jerk_down * dt)


@dataclass
class LeadPullawayIntentTracker:
  _phase: LeadPullawayPhase = LeadPullawayPhase.HOLD
  _track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  _pulse_used_track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  _pulse_timer: float = 0.0
  _cooldown_timer: float = 0.0
  _last_a_floor: float = 0.0
  _last_lead_accel: float | None = None
  _last_lead_accel_track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  _early_authority: bool = False
  _early_authority_reason: str = ""

  def reset(self):
    self._phase = LeadPullawayPhase.HOLD
    self._track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._pulse_used_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._pulse_timer = 0.0
    self._cooldown_timer = 0.0
    self._last_a_floor = 0.0
    self._last_lead_accel = None
    self._last_lead_accel_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._early_authority = False
    self._early_authority_reason = ""

  def clamp_active_floor(self, a_floor) -> None:
    self._last_a_floor = min(
      max(0.0, _finite_float(self._last_a_floor)),
      max(0.0, _finite_float(a_floor)),
    )

  def update(
    self,
    *,
    v_ego,
    behavior_lead,
    primary_lead_context,
    lead_gap_excess,
    predicted_gap_opening,
    lead_opening,
    lead_moving,
    lead_accel,
    independent_stop_threat,
    alternate_lead_threat_active,
    brake_pressed,
    gas_pressed,
    force_slow_decel,
    reset_state,
    dt,
  ) -> LeadPullawayIntent:
    dt = max(0.0, _finite_float(dt))
    lead_gap_excess = max(0.0, _finite_float(lead_gap_excess))
    predicted_gap_opening = max(0.0, _finite_float(predicted_gap_opening))
    v_ego = max(0.0, _finite_float(v_ego))
    lead_accel = _finite_float(lead_accel)
    self._cooldown_timer = max(0.0, self._cooldown_timer - dt)

    if reset_state:
      self.reset()
      return self._intent(LeadPullawayPhase.HOLD, False, 0.0, "driver_or_force_blocked", lead_gap_excess, predicted_gap_opening)

    behavior_state = getattr(primary_lead_context, "behavior", None)
    physical_state = getattr(primary_lead_context, "physical", None)
    lead_state = behavior_state or physical_state
    track_id = self._lead_track_id(behavior_lead, lead_state)
    if self._fresh_stable_track(track_id):
      self._phase = LeadPullawayPhase.HOLD
      self._pulse_timer = 0.0
      self._last_a_floor = 0.0
      self._pulse_used_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
      self._cooldown_timer = 0.0
      self._last_lead_accel = None
      self._last_lead_accel_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._track_id = track_id
    lead_accel_trend = self._lead_accel_trend(track_id, lead_accel, dt)
    runway = self._runway(behavior_lead, lead_state, v_ego, lead_accel, lead_accel_trend)
    early_pullaway_authority, early_authority_reason = self._early_pullaway_authority(
      lead_state=lead_state,
      primary_lead_context=primary_lead_context,
      runway=runway,
      lead_opening=lead_opening,
      lead_moving=lead_moving,
      lead_accel=lead_accel,
      independent_stop_threat=independent_stop_threat,
      alternate_lead_threat_active=alternate_lead_threat_active,
      brake_pressed=brake_pressed,
      gas_pressed=gas_pressed,
      force_slow_decel=force_slow_decel,
    )
    self._early_authority = bool(early_pullaway_authority)
    self._early_authority_reason = str(early_authority_reason)

    blocker = self._blocking_reason(
      behavior_lead=behavior_lead,
      behavior_state=lead_state,
      primary_lead_context=primary_lead_context,
      lead_opening=lead_opening,
      lead_moving=lead_moving,
      lead_accel=lead_accel,
      predicted_gap_opening=predicted_gap_opening,
      independent_stop_threat=independent_stop_threat,
      alternate_lead_threat_active=alternate_lead_threat_active,
      brake_pressed=brake_pressed,
      gas_pressed=gas_pressed,
      force_slow_decel=force_slow_decel,
      early_pullaway_authority=early_pullaway_authority,
    )
    if blocker:
      if blocker == "lead_stopped_again":
        self._pulse_used_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
        self._cooldown_timer = 0.0
      self._phase = LeadPullawayPhase.HOLD
      self._pulse_timer = 0.0
      self._last_a_floor = 0.0
      return self._intent(LeadPullawayPhase.HOLD, False, 0.0, blocker, lead_gap_excess, predicted_gap_opening, runway)

    if v_ego >= LEAD_PULLAWAY_MAX_V_EGO:
      self._phase = LeadPullawayPhase.NORMAL
      self._pulse_timer = 0.0
      self._last_a_floor = 0.0
      return self._intent(LeadPullawayPhase.NORMAL, False, 0.0, "outside_low_speed_launch", lead_gap_excess, predicted_gap_opening, runway)

    progress_model = getattr(lead_state, "progress_model", None)
    normal_progress_authority = bool(
      getattr(primary_lead_context, "lead_progress_allowed", False) and
      progress_model is not None and getattr(progress_model, "allowed", False)
    )
    opening_evidence = bool(
      lead_opening or
      getattr(progress_model, "opening_speed", 0.0) >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED or
      predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING or
      (lead_accel >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and getattr(progress_model, "predicted_gap_opening", False)) or
      early_pullaway_authority
    )
    moving_evidence = bool(
      lead_moving or getattr(progress_model, "lead_moving", False) or
      (early_pullaway_authority and lead_accel >= LEAD_PULLAWAY_EARLY_MIN_LEAD_ACCEL)
    )
    runway_allows_progress = bool(not runway.coast_required and runway.safe_accel_cap > 1e-3)
    pulse_evidence = bool(opening_evidence and moving_evidence)
    gap_closure_allowed = bool(
      normal_progress_authority and
      runway_allows_progress and
      moving_evidence and
      lead_gap_excess >= EXCESS_GAP_CLOSURE_START_EXCESS and
      not self._lead_closing(progress_model, lead_opening)
    )

    if self._phase == LeadPullawayPhase.PULSE:
      self._pulse_timer = max(0.0, self._pulse_timer - dt)
      if self._pulse_timer > 0.0 and v_ego < LEAD_PULLAWAY_MAX_V_EGO and pulse_evidence and runway_allows_progress:
        return self._active_intent(
          LeadPullawayPhase.PULSE, LEAD_PULLAWAY_PULSE_A_FLOOR, LEAD_PULLAWAY_PULSE_REASON,
          lead_gap_excess, predicted_gap_opening, LEAD_PULLAWAY_PULSE_JERK_UP, LEAD_PULLAWAY_PULSE_JERK_DOWN, dt, runway,
        )
      self._phase = LeadPullawayPhase.GAP_CLOSURE if gap_closure_allowed else LeadPullawayPhase.NORMAL

    if self._phase == LeadPullawayPhase.GAP_CLOSURE:
      if gap_closure_allowed:
        return self._active_intent(
          LeadPullawayPhase.GAP_CLOSURE, self._gap_closure_floor(lead_gap_excess), EXCESS_GAP_CLOSURE_REASON,
          lead_gap_excess, predicted_gap_opening, EXCESS_GAP_CLOSURE_JERK_UP, EXCESS_GAP_CLOSURE_JERK_DOWN, dt, runway,
        )
      self._phase = LeadPullawayPhase.NORMAL

    if gap_closure_allowed and (self._pulse_used_track_id == track_id or self._cooldown_timer > 0.0):
      self._phase = LeadPullawayPhase.GAP_CLOSURE
      return self._active_intent(
        LeadPullawayPhase.GAP_CLOSURE, self._gap_closure_floor(lead_gap_excess), EXCESS_GAP_CLOSURE_REASON,
        lead_gap_excess, predicted_gap_opening, EXCESS_GAP_CLOSURE_JERK_UP, EXCESS_GAP_CLOSURE_JERK_DOWN, dt, runway,
      )

    if not pulse_evidence or not runway_allows_progress:
      self._phase = LeadPullawayPhase.NORMAL if gap_closure_allowed else LeadPullawayPhase.HOLD
      reason = EXCESS_GAP_CLOSURE_REASON if gap_closure_allowed else "lead_pullaway_runway_coast" if runway.coast_required else "lead_not_opening"
      return self._intent(self._phase, False, 0.0, reason, lead_gap_excess, predicted_gap_opening, runway)

    if self._pulse_used_track_id == track_id or self._cooldown_timer > 0.0:
      self._phase = LeadPullawayPhase.NORMAL
      return self._intent(LeadPullawayPhase.NORMAL, False, 0.0, "pullaway_cooldown", lead_gap_excess, predicted_gap_opening, runway)

    if self._phase == LeadPullawayPhase.ARMED:
      self._phase = LeadPullawayPhase.PULSE
      self._pulse_timer = LEAD_PULLAWAY_PULSE_DURATION
      self._cooldown_timer = LEAD_PULLAWAY_PULSE_COOLDOWN
      self._pulse_used_track_id = track_id
      return self._active_intent(
        LeadPullawayPhase.PULSE, LEAD_PULLAWAY_PULSE_A_FLOOR, LEAD_PULLAWAY_PULSE_REASON,
        lead_gap_excess, predicted_gap_opening, LEAD_PULLAWAY_PULSE_JERK_UP, LEAD_PULLAWAY_PULSE_JERK_DOWN, dt, runway,
      )

    self._phase = LeadPullawayPhase.ARMED
    self._last_a_floor = 0.0
    return self._intent(LeadPullawayPhase.ARMED, False, 0.0, "lead_pullaway_armed", lead_gap_excess, predicted_gap_opening, runway)

  def _active_intent(self, phase, target_floor, reason, gap_excess, predicted_gap_opening, jerk_up, jerk_down, dt,
                     runway: LeadPullawayRunway | None = None):
    runway = runway or LeadPullawayRunway()
    capped_target_floor = min(max(0.0, _finite_float(target_floor)), runway.safe_accel_cap)
    a_floor = approach_accel_with_jerk_limit(self._last_a_floor, capped_target_floor, dt, jerk_up, jerk_down)
    a_floor = min(a_floor, capped_target_floor)
    self._last_a_floor = a_floor
    return self._intent(
      phase, True, a_floor, reason, gap_excess, predicted_gap_opening, runway,
      pulse_capped_by_runway=capped_target_floor + 1e-3 < _finite_float(target_floor),
    )

  def _intent(self, phase, active, a_floor, reason, gap_excess, predicted_gap_opening,
              runway: LeadPullawayRunway | None = None, pulse_capped_by_runway=False):
    runway = runway or LeadPullawayRunway()
    return LeadPullawayIntent(
      phase=phase,
      active=bool(active),
      a_floor=max(0.0, _finite_float(a_floor)),
      reason=str(reason),
      track_id=int(self._track_id),
      pulse_timer=max(0.0, float(self._pulse_timer)),
      cooldown_timer=max(0.0, float(self._cooldown_timer)),
      gap_excess=max(0.0, _finite_float(gap_excess)),
      predicted_gap_opening=max(0.0, _finite_float(predicted_gap_opening)),
      predicted_gap=max(0.0, _finite_float(runway.predicted_gap)),
      safe_accel_cap=max(0.0, _finite_float(runway.safe_accel_cap)),
      lead_accel_trend=_finite_float(runway.lead_accel_trend),
      runway_margin=_finite_float(runway.runway_margin),
      runway_margin_now=_finite_float(runway.runway_margin_now),
      runway_margin_t=_finite_float(runway.runway_margin_t),
      runway_creation=_finite_float(runway.runway_creation),
      lead_created_runway=bool(runway.lead_created_runway),
      early_authority=bool(self._early_authority),
      early_authority_reason=str(self._early_authority_reason),
      coast_required=bool(runway.coast_required),
      pulse_capped_by_runway=bool(pulse_capped_by_runway),
      runway_trend=str(runway.trend),
    )

  def _lead_accel_trend(self, track_id, lead_accel, dt):
    lead_accel = _finite_float(lead_accel)
    if track_id == LEAD_CONFIDENCE_TRACK_UNKNOWN or self._last_lead_accel_track_id != track_id or self._last_lead_accel is None:
      trend = 0.0
    else:
      trend = (lead_accel - self._last_lead_accel) / max(dt, DT_MDL)
    self._last_lead_accel = lead_accel
    self._last_lead_accel_track_id = track_id
    return trend

  @staticmethod
  def _runway(behavior_lead, behavior_state, v_ego, lead_accel, lead_accel_trend):
    if behavior_state is not None:
      d_rel = getattr(behavior_state, "d_rel", 0.0)
      v_lead = getattr(behavior_state, "v_lead", 0.0)
    elif behavior_lead is not None:
      d_rel = getattr(behavior_lead, "dRel", 0.0)
      v_lead = getattr(behavior_lead, "vLeadK", getattr(behavior_lead, "vLead", 0.0))
    else:
      d_rel = 0.0
      v_lead = 0.0
    return get_lead_pullaway_runway(v_ego, d_rel, v_lead, lead_accel, lead_accel_trend)

  def _fresh_stable_track(self, track_id):
    return bool(
      track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN and
      self._track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN and
      track_id != self._track_id
    )

  @staticmethod
  def _lead_track_id(behavior_lead, behavior_state):
    if behavior_state is not None:
      try:
        return int(getattr(behavior_state, "track_id", LEAD_CONFIDENCE_TRACK_UNKNOWN))
      except (TypeError, ValueError):
        return LEAD_CONFIDENCE_TRACK_UNKNOWN
    try:
      return int(getattr(behavior_lead, "radarTrackId", LEAD_CONFIDENCE_TRACK_UNKNOWN))
    except (TypeError, ValueError):
      return LEAD_CONFIDENCE_TRACK_UNKNOWN

  @staticmethod
  def _lead_closing(progress_model, lead_opening):
    if progress_model is None:
      return True
    return bool(not lead_opening and getattr(progress_model, "opening_speed", 0.0) <= 0.05)

  @staticmethod
  def _early_pullaway_authority(*, lead_state, primary_lead_context, runway, lead_opening, lead_moving,
                                lead_accel, independent_stop_threat, alternate_lead_threat_active,
                                brake_pressed, gas_pressed, force_slow_decel):
    if lead_state is None:
      return False, "no_lead_state"
    if bool(getattr(primary_lead_context, "alternate_threat_active", False) or alternate_lead_threat_active):
      return False, "alternate_lead_threat"
    if bool(independent_stop_threat):
      return False, "independent_stop_threat"
    if bool(force_slow_decel or brake_pressed or gas_pressed):
      return False, "driver_or_force_blocked"
    if bool(getattr(primary_lead_context, "shadow_active", False) or getattr(lead_state, "shadow", False)):
      return False, "shadow_or_flicker"
    if bool(getattr(lead_state, "flicker_guard_timer", 0.0) > 0.0):
      return False, "shadow_or_flicker"
    if bool(getattr(lead_state, "new_lead", False) or not getattr(lead_state, "stable", False)):
      return False, "lead_confidence_low"
    try:
      track_id = int(getattr(lead_state, "track_id", LEAD_CONFIDENCE_TRACK_UNKNOWN))
    except (TypeError, ValueError):
      track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    if track_id == LEAD_CONFIDENCE_TRACK_UNKNOWN:
      return False, "lead_confidence_low"

    blocked_reason = str(getattr(primary_lead_context, "lead_release_blocked_reason", ""))
    if blocked_reason == "primary_physical_lead_suppressive":
      return False, blocked_reason

    progress_model = getattr(lead_state, "progress_model", None)
    if progress_model is None:
      return False, "no_lead_progress_model"
    if not bool(getattr(progress_model, "confidence_stability_sufficient", False)):
      return False, "lead_confidence_low"
    if not bool(getattr(progress_model, "alternate_threat_absent", True)):
      return False, "alternate_lead_threat"
    if not bool(getattr(progress_model, "shadow_absent", True)):
      return False, "shadow_or_flicker"

    risk_model = getattr(lead_state, "risk_model", None)
    closing_speed = _finite_float(getattr(risk_model, "closing_speed", 0.0)) if risk_model is not None else max(0.0, -_finite_float(getattr(lead_state, "v_rel", 0.0)))
    required_decel = _finite_float(getattr(risk_model, "required_decel", 0.0)) if risk_model is not None else _finite_float(getattr(lead_state, "required_decel", 0.0))
    ttc = _finite_float(getattr(risk_model, "ttc", math.inf), math.inf) if risk_model is not None else _finite_float(getattr(lead_state, "ttc", math.inf), math.inf)
    if bool(
      (closing_speed > 0.05 and not lead_opening) or
      required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or
      ttc <= LEAD_CONTEXT_RISK_TTC
    ):
      return False, "close_closing_lead"

    lead_accel = _finite_float(lead_accel)
    opening_speed = _finite_float(getattr(progress_model, "opening_speed", 0.0))
    motion_evidence = bool(
      lead_opening or lead_moving or
      opening_speed >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED or
      getattr(progress_model, "lead_moving", False) or
      lead_accel >= LEAD_PULLAWAY_EARLY_MIN_LEAD_ACCEL
    )
    if not motion_evidence:
      return False, "lead_not_opening"
    if bool(runway.coast_required):
      return False, "lead_pullaway_runway_coast"
    if _finite_float(runway.safe_accel_cap) < LEAD_PULLAWAY_EARLY_MIN_SAFE_ACCEL:
      return False, "lead_pullaway_runway_cap_low"
    if not bool(runway.lead_created_runway):
      return False, "lead_created_runway_absent"
    return True, "lead_created_runway"

  @staticmethod
  def _gap_closure_floor(gap_excess):
    strength = float(np.clip(
      (max(0.0, _finite_float(gap_excess)) - EXCESS_GAP_CLOSURE_START_EXCESS) / EXCESS_GAP_CLOSURE_FULL_EXCESS,
      0.0,
      1.0,
    ))
    return EXCESS_GAP_CLOSURE_ACCEL_BASE + strength * (EXCESS_GAP_CLOSURE_ACCEL_CAP - EXCESS_GAP_CLOSURE_ACCEL_BASE)

  def _blocking_reason(self, *, behavior_lead, behavior_state, primary_lead_context, lead_opening, lead_moving,
                        lead_accel, predicted_gap_opening, independent_stop_threat, alternate_lead_threat_active,
                        brake_pressed, gas_pressed, force_slow_decel, early_pullaway_authority=False):
    if behavior_state is None:
      physical_state = getattr(primary_lead_context, "physical", None)
      physical_track_id = int(getattr(physical_state, "track_id", LEAD_CONFIDENCE_TRACK_UNKNOWN)) if physical_state is not None else LEAD_CONFIDENCE_TRACK_UNKNOWN
      physical_stopped = bool(getattr(getattr(physical_state, "risk_model", None), "stopped_or_crawling", False))
      if self._phase != LeadPullawayPhase.HOLD and physical_track_id == self._pulse_used_track_id and physical_stopped:
        return "lead_stopped_again"
      return "no_lead_progress_authority"
    if self._track_id == LEAD_CONFIDENCE_TRACK_UNKNOWN:
      return "lead_confidence_low"
    if bool(getattr(primary_lead_context, "alternate_threat_active", False) or alternate_lead_threat_active):
      return "alternate_lead_threat"
    if bool(independent_stop_threat):
      return "independent_stop_threat"
    if bool(force_slow_decel or brake_pressed or gas_pressed):
      return "driver_or_force_blocked"
    if bool(getattr(primary_lead_context, "shadow_active", False) or getattr(behavior_state, "shadow", False)):
      return "shadow_or_flicker"
    if bool(getattr(behavior_state, "flicker_guard_timer", 0.0) > 0.0):
      return "shadow_or_flicker"
    if bool(getattr(behavior_state, "new_lead", False) or not getattr(behavior_state, "stable", False)):
      return "lead_confidence_low"
    state_stopped_again = bool(
      self._phase in (LeadPullawayPhase.ARMED, LeadPullawayPhase.PULSE, LeadPullawayPhase.GAP_CLOSURE) and
      self._track_id == self._pulse_used_track_id and
      getattr(getattr(behavior_state, "risk_model", None), "stopped_or_crawling", False) and
      not lead_opening and not lead_moving and
      _finite_float(lead_accel) <= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and
      _finite_float(predicted_gap_opening) < CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING
    )
    if state_stopped_again:
      return "lead_stopped_again"
    if not bool(getattr(primary_lead_context, "lead_progress_allowed", False)):
      blocked_reason = str(getattr(primary_lead_context, "lead_release_blocked_reason", ""))
      if not bool(early_pullaway_authority) or blocked_reason == "primary_physical_lead_suppressive":
        return blocked_reason or "no_lead_progress_authority"

    progress_model = getattr(behavior_state, "progress_model", None)
    if progress_model is None:
      return "no_lead_progress_authority"
    if not getattr(progress_model, "allowed", False) and not bool(early_pullaway_authority):
      return "no_lead_progress_authority"
    if not bool(getattr(progress_model, "confidence_stability_sufficient", False)):
      return "lead_confidence_low"
    if not bool(getattr(progress_model, "stop_threat_absent", False)) and not bool(early_pullaway_authority):
      return "lead_stopped_again"
    lead_stopped_again = bool(
      self._phase in (LeadPullawayPhase.ARMED, LeadPullawayPhase.PULSE, LeadPullawayPhase.GAP_CLOSURE) and
      not lead_opening and not lead_moving and _finite_float(lead_accel) <= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and
      _finite_float(predicted_gap_opening) < CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING
    )
    if lead_stopped_again:
      return "lead_stopped_again"
    return ""


def has_model_stop_context(model_msg):
  if model_msg.action.shouldStop:
    return True

  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  return any(x > 0.0 and v <= ENGAGE_STOP_BOOTSTRAP_MODEL_STOP_SPEED for x, v in zip(positions, velocities, strict=False))


def build_scc_mode_evidence(has_confirmed_lead: bool, model_msg, scc, sla, osm_traffic_control_prior,
                            speed_limit_handoff_active: bool = False, lead_distance: float | None = None,
                            lead_path_y_rel: float = 0.0, lead_idx: int | None = None,
                            v_ego: float = 0.0) -> SccModeEvidence:
  model_stop_distance = get_e2e_confirmed_model_stop_distance(model_msg)
  model_stop = bool(
    model_msg.action.shouldStop or
    model_stop_distance is not None or
    has_scc_near_endpoint_model_stop(model_msg) or
    has_scc_early_model_stop(model_msg)
  )
  if model_stop and model_stop_distance is None:
    positions = _finite_model_array(getattr(model_msg.position, "x", []))
    if positions is not None and len(positions) > 0:
      model_stop_distance = float(max(0.0, positions[-1]))
  stop_profile = jerk_limited_braking_profile(v_ego, 0.0, model_stop_distance) if model_stop_distance is not None else None
  urgent_stop = bool(stop_profile is not None and stop_profile.urgent)
  model_slowdown = bool(not model_stop and not urgent_stop and has_scc_model_slowdown(model_msg))
  confirmed_geometry = bool(has_confirmed_lead and lead_distance is not None)
  return SccModeEvidence(
    confirmed_lead=confirmed_geometry,
    model_stop=model_stop,
    model_slowdown=model_slowdown,
    urgent_stop=urgent_stop,
    independent_of_lead=bool(model_stop and not confirmed_geometry),
    model_stop_distance=model_stop_distance,
    lead_distance=lead_distance,
    lead_path_y_rel=lead_path_y_rel,
    lead_idx=lead_idx,
    v_ego=v_ego,
    urgency=None if stop_profile is None else min(1.0, max(0.0, abs(stop_profile.required_accel) / 3.5)),
    curve_control=bool(getattr(getattr(scc, "vision", None), "is_active", False)),
    map_control=bool(getattr(getattr(scc, "map", None), "is_active", False)),
    speed_limit_control=bool(getattr(sla, "is_active", False) or speed_limit_handoff_active),
    traffic_control=bool(getattr(osm_traffic_control_prior, "active", False)),
  )


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


def has_scc_near_endpoint_model_stop(model_msg):
  desired_accel = float(model_msg.action.desiredAcceleration)
  if not np.isfinite(desired_accel) or desired_accel > SCC_NEAR_ENDPOINT_MODEL_STOP_ACCEL:
    return False

  positions = list(getattr(model_msg.position, "x", []))
  velocities = list(getattr(model_msg.velocity, "x", []))
  if len(positions) == 0 or len(velocities) != len(positions):
    return False

  endpoint_x = float(positions[-1])
  endpoint_v = float(velocities[-1])
  return bool(
    np.isfinite(endpoint_x) and np.isfinite(endpoint_v) and
    0.0 < endpoint_x <= SCC_NEAR_ENDPOINT_MODEL_STOP_MAX_DISTANCE and
    endpoint_v <= SCC_NEAR_ENDPOINT_MODEL_STOP_MAX_V
  )


def has_scc_early_model_stop(model_msg):
  desired_accel = float(model_msg.action.desiredAcceleration)
  if not np.isfinite(desired_accel) or desired_accel > SCC_EARLY_MODEL_STOP_ACCEL:
    return False

  positions = _finite_model_array(getattr(model_msg.position, "x", []))
  velocities = _finite_model_array(getattr(model_msg.velocity, "x", []))
  if positions is None or velocities is None or len(positions) < 3 or len(positions) != len(velocities):
    return False

  initial_v = float(velocities[0])
  endpoint_x = float(positions[-1])
  endpoint_v = max(float(velocities[-1]), 0.0)
  if initial_v < SCC_EARLY_MODEL_STOP_MIN_INITIAL_V or endpoint_x <= 0.0:
    return False
  if initial_v - float(np.min(velocities)) < SCC_EARLY_MODEL_STOP_MIN_SPEED_DROP:
    return False

  expected_distance = float(np.interp(
    initial_v * CV.MS_TO_KPH,
    E2E_STOP_APPROACH_EXPECTED_DIST_BP,
    E2E_STOP_APPROACH_EXPECTED_DIST_V,
  ))
  if endpoint_x > expected_distance * SCC_EARLY_MODEL_STOP_EXPECTED_DISTANCE_SCALE:
    return False

  required_decel = (initial_v**2 - endpoint_v**2) / (2.0 * endpoint_x)
  if required_decel < SCC_EARLY_MODEL_STOP_MIN_REQUIRED_DECEL:
    return False

  middle_positions = positions[1:-1]
  middle_velocities = velocities[1:-1]
  return bool(np.any(
    (middle_positions >= 0.0) &
    (endpoint_x - middle_positions >= SCC_EARLY_MODEL_STOP_ENDPOINT_MARGIN) &
    (middle_velocities <= SCC_EARLY_MODEL_STOP_MAX_MID_V)
  ))


def has_scc_model_slowdown(model_msg) -> bool:
  try:
    desired_accel = float(model_msg.action.desiredAcceleration)
  except (TypeError, ValueError, AttributeError):
    desired_accel = 0.0
  positions = _finite_model_array(getattr(model_msg.position, "x", []))
  velocities = _finite_model_array(getattr(model_msg.velocity, "x", []))
  if positions is None or velocities is None or len(positions) < 2 or len(positions) != len(velocities):
    return False

  initial_v = max(0.0, float(velocities[0]))
  endpoint_v = max(0.0, float(velocities[-1]))
  min_v = max(0.0, float(np.min(velocities)))
  speed_drop = initial_v - min_v
  endpoint_drop = initial_v - endpoint_v
  return bool(
    desired_accel <= SCC_MODEL_SLOWDOWN_ACCEL and
    (
      speed_drop >= SCC_MODEL_SLOWDOWN_MIN_SPEED_DROP or
      endpoint_drop >= SCC_MODEL_SLOWDOWN_MIN_ENDPOINT_DROP
    )
  )


def _valid_scc_lead_geometry_values(lead) -> tuple[float | None, float, int | None]:
  if lead is None or not bool(getattr(lead, "status", False)):
    return None, 0.0, None
  try:
    d_rel = float(getattr(lead, "dRel", 0.0))
  except (TypeError, ValueError):
    return None, 0.0, None
  if not math.isfinite(d_rel) or d_rel < 0.0:
    return None, 0.0, None
  try:
    y_rel = float(getattr(lead, "yRel", 0.0))
  except (TypeError, ValueError):
    y_rel = 0.0
  try:
    track_id = int(getattr(lead, "radarTrackId", -1))
  except (TypeError, ValueError):
    track_id = -1
  return d_rel, y_rel if math.isfinite(y_rel) else 0.0, track_id if track_id >= 0 else None


def scc_lead_geometry_from_context(context: PrimaryLeadContext | None, radar_state) -> tuple[float | None, float, int | None]:
  leads = (getattr(radar_state, "leadOne", None), getattr(radar_state, "leadTwo", None))
  shadow_indices = _scc_shadow_lead_indices(context)
  physical = getattr(context, "physical", None)
  if physical is not None and not bool(getattr(physical, "shadow", False)) and bool(getattr(physical, "status", False)):
    try:
      lead_idx = int(getattr(physical, "lead_idx"))
      path_y_rel = float(getattr(physical, "path_y_rel", getattr(physical, "y_rel", 0.0)))
    except (TypeError, ValueError):
      path_y_rel = 0.0
      lead_idx = -1
    raw_lead = leads[lead_idx] if 0 <= lead_idx < len(leads) else None
    d_rel, raw_y_rel, _track_id = _valid_scc_lead_geometry_values(raw_lead)
    if d_rel is not None:
      return d_rel, path_y_rel if math.isfinite(path_y_rel) else raw_y_rel, lead_idx

  confirmed = []
  for idx, lead in enumerate(leads):
    if idx in shadow_indices:
      continue
    if not bool(getattr(lead, "status", False)):
      continue
    if _finite_float(getattr(lead, "modelProb", 0.0)) < LEAD_LOSS_E2E_GUARD_MIN_MODEL_PROB:
      continue
    d_rel, y_rel, _track_id = _valid_scc_lead_geometry_values(lead)
    if d_rel is not None:
      confirmed.append((idx, d_rel, y_rel))
  if not confirmed:
    return None, 0.0, None
  idx, d_rel, y_rel = min(confirmed, key=lambda item: item[1])
  return d_rel, y_rel, idx


def _scc_shadow_lead_indices(context: PrimaryLeadContext | None) -> set[int]:
  indices: set[int] = set()
  for state in (*tuple(getattr(context, "states", ())), getattr(context, "physical", None), getattr(context, "behavior", None)):
    if state is None or not bool(getattr(state, "shadow", False)):
      continue
    try:
      lead_idx = int(getattr(state, "lead_idx"))
    except (TypeError, ValueError):
      continue
    if lead_idx >= 0:
      indices.add(lead_idx)
  return indices


def _finite_model_array(values):
  try:
    array = np.asarray(list(values), dtype=float)
  except (TypeError, ValueError):
    return None
  if len(array) == 0 or not np.all(np.isfinite(array)):
    return None
  return array


def should_run_engage_stop_bootstrap(timer, v_ego, radar_state, model_msg):
  if timer <= 0.0 or v_ego < ENGAGE_STOP_BOOTSTRAP_MIN_SPEED or has_valid_radar_lead(radar_state):
    return False

  return bool(
    model_msg.action.shouldStop or
    (model_msg.action.desiredAcceleration <= ENGAGE_STOP_BOOTSTRAP_MODEL_ACCEL and has_model_stop_context(model_msg))
  )


def clip_traction_risk(traction_risk: float) -> float:
  try:
    traction_risk = float(traction_risk)
  except (TypeError, ValueError):
    traction_risk = 0.0
  return float(np.clip(traction_risk if np.isfinite(traction_risk) else 0.0, 0.0, 1.0))


def get_traction_risk(car_state_sp) -> float:
  try:
    traction_risk = float(getattr(car_state_sp, "tractionRisk", 0.0))
  except (TypeError, ValueError, AttributeError):
    return 0.0
  return clip_traction_risk(traction_risk)


def get_traction_runway_scale(traction_risk: float) -> float:
  return 1.0 + (TRACTION_RISK_RUNWAY_SCALE_MAX - 1.0) * clip_traction_risk(traction_risk)


def get_traction_light_decel(traction_risk: float) -> float:
  return float(np.interp(clip_traction_risk(traction_risk), [0.0, 1.0],
                         [E2E_RUNWAY_COMFORT_LIGHT_DECEL, E2E_RUNWAY_COMFORT_TRACTION_LIGHT_DECEL]))


def get_traction_negative_ramp_rate(base_rate: float, traction_risk: float) -> float:
  scale = 1.0 - (1.0 - TRACTION_RISK_NEGATIVE_RAMP_MIN_SCALE) * clip_traction_risk(traction_risk)
  return base_rate * scale


def get_e2e_stop_approach_accel(v_ego, model_msg, radar_state, e2e_active, force_slow_decel=False,
                                brake_pressed=False, gas_pressed=False, model_stop_protection_active=False,
                                traction_risk=0.0):
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
  expected_distance *= get_traction_runway_scale(traction_risk)
  max_decel_distance = v_ego**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX * (1.0 - E2E_STOP_APPROACH_MAX_DECEL_SHORTAGE))
  expected_distance = max(expected_distance, max_decel_distance)
  if expected_distance <= 0.0:
    return 0.0

  shortage = max(0.0, expected_distance - approach_distance) / expected_distance
  if shortage <= E2E_STOP_APPROACH_SHORTAGE_BP[0]:
    return 0.0

  shortage_decel = float(np.interp(shortage, E2E_STOP_APPROACH_SHORTAGE_BP, E2E_STOP_APPROACH_DECEL_BP))
  required_decel_blend = float(np.interp(shortage, E2E_STOP_APPROACH_REQUIRED_DECEL_SHORTAGE_BP, [0.0, 1.0]))
  required_decel = required_decel_blend * E2E_STOP_APPROACH_REQUIRED_DECEL_BLEND * v_ego**2 / (2.0 * max(endpoint_x, E2E_STOP_APPROACH_MIN_ENDPOINT))
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
                                        selection=PLANNER_SEED_CAP, force=False, group="", debug=None):
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
  seed_debug = {"planner_seed_candidate_reason": reason, "planner_seed_scalar": True}
  if debug is not None:
    seed_debug.update(debug)
  output = replace(
    base_output,
    a_target=candidate_a_target,
    should_stop=candidate_should_stop,
    debug=seed_debug,
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


def _planner_seed_candidates(*candidates) -> tuple[PlannerSeedCandidate, ...]:
  return tuple(candidate for candidate in candidates if candidate is not None)


def build_no_lead_stop_seed_candidates(planner, has_lead, accel_limits, *, engage_bootstrap_active=False,
                                       engage_bootstrap_a_target=0.0, engage_bootstrap_should_stop=False,
                                       e2e_close_stop_active=False, e2e_close_stop_a_target=0.0,
                                       e2e_close_stop_should_stop=False,
                                       e2e_runway_comfort_a_target=None,
                                       e2e_stop_approach_a_target=0.0,
                                       e2e_runway_positive_accel_cap=None) -> tuple[PlannerSeedCandidate, ...]:
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "engage_stop_bootstrap", engage_bootstrap_a_target, has_lead,
      "engage_model_stop_bootstrap", accel_limits, should_stop=engage_bootstrap_should_stop,
    ) if engage_bootstrap_active else None,
    build_planner_seed_accel_candidate(
      planner, "e2e_close_stop_settle", e2e_close_stop_a_target, has_lead,
      "no_lead_close_stop_settle", accel_limits, should_stop=e2e_close_stop_should_stop,
    ) if e2e_close_stop_active else None,
    build_planner_seed_accel_candidate(
      planner, "e2e_runway_comfort", e2e_runway_comfort_a_target, has_lead,
      "no_lead_model_runway_comfort", accel_limits, selection=PLANNER_SEED_FLOOR,
    ) if e2e_runway_comfort_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "e2e_stop_approach", e2e_stop_approach_a_target, has_lead,
      "no_lead_model_stop_approach", accel_limits,
    ) if e2e_stop_approach_a_target < 0.0 else None,
    build_planner_seed_accel_candidate(
      planner, "e2e_runway_positive_cap", e2e_runway_positive_accel_cap, has_lead,
      "low_speed_model_runway_positive_cap", accel_limits,
    ) if e2e_runway_positive_accel_cap is not None else None,
  )


def build_stopped_lead_seed_candidates(planner, has_lead, accel_limits, *, stopped_stop_gap_guard_a_target=None,
                                       stopped_stop_gap_guard_group="", creep_to_stop_gap_a_target=None,
                                       creep_to_stop_gap_should_stop=None, creep_to_stop_gap_selection=PLANNER_SEED_CAP,
                                       creep_to_stop_gap_accel_max=None, gap_fill_a_target=None,
                                       gap_fill_should_stop=None, gap_fill_selection=PLANNER_SEED_CAP,
                                       gap_fill_accel_max=None, lead_crawl_accel_cap=None,
                                       creep_hold_a_target=None) -> tuple[PlannerSeedCandidate, ...]:
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "stopped_lead_stop_gap_guard", stopped_stop_gap_guard_a_target, has_lead,
      "stopped_lead_stop_gap_guard", accel_limits, should_stop=True, group=stopped_stop_gap_guard_group,
    ) if stopped_stop_gap_guard_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "creep_to_stop_gap", creep_to_stop_gap_a_target, has_lead,
      "creep_to_stop_gap", accel_limits, should_stop=creep_to_stop_gap_should_stop,
      selection=creep_to_stop_gap_selection, group="creep_to_stop_gap",
    ) if creep_to_stop_gap_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "creep_to_stop_gap_accel_cap", creep_to_stop_gap_accel_max, has_lead,
      "creep_to_stop_gap_accel_cap", accel_limits, should_stop=creep_to_stop_gap_should_stop,
      force=True, group="creep_to_stop_gap",
    ) if creep_to_stop_gap_accel_max is not None else None,
    build_planner_seed_accel_candidate(
      planner, "stopped_lead_gap_fill", gap_fill_a_target, has_lead,
      "stopped_lead_gap_fill", accel_limits, should_stop=gap_fill_should_stop,
      selection=gap_fill_selection, group="stopped_lead_gap_fill",
    ) if gap_fill_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "stopped_lead_gap_fill_accel_cap", gap_fill_accel_max, has_lead,
      "stopped_lead_gap_fill_accel_cap", accel_limits, should_stop=gap_fill_should_stop,
      force=True, group="stopped_lead_gap_fill",
    ) if gap_fill_accel_max is not None else None,
    build_planner_seed_accel_candidate(
      planner, "lead_crawl_accel_cap", lead_crawl_accel_cap, has_lead,
      "lead_crawl_accel_cap", accel_limits, force=True,
    ) if lead_crawl_accel_cap is not None else None,
    build_planner_seed_accel_candidate(
      planner, "stopped_lead_creep_hold", creep_hold_a_target, has_lead,
      "stopped_lead_creep_hold", accel_limits, should_stop=True,
    ) if creep_hold_a_target is not None else None,
  )


def build_lead_pullaway_seed_candidates(planner, has_lead, accel_limits, *, creep_pullaway_launch_floor=None,
                                        creep_pullaway_launch_cap=None, pullaway_accel_step_floor=None,
                                        pullaway_accel_step_cap=None,
                                        pullaway_step_cap_suppressed=False) -> tuple[PlannerSeedCandidate, ...]:
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "creep_pullaway_launch", creep_pullaway_launch_floor, has_lead,
      "creep_pullaway_launch", accel_limits, should_stop=False, selection=PLANNER_SEED_FLOOR,
      group="creep_pullaway_launch",
    ) if creep_pullaway_launch_floor is not None else None,
    build_planner_seed_accel_candidate(
      planner, "creep_pullaway_launch_accel_cap", creep_pullaway_launch_cap, has_lead,
      "creep_pullaway_launch_accel_cap", accel_limits, should_stop=False, force=True,
      group="creep_pullaway_launch",
    ) if creep_pullaway_launch_cap is not None else None,
    build_planner_seed_accel_candidate(
      planner, "low_speed_pullaway_accel_step_floor", pullaway_accel_step_floor, has_lead,
      "low_speed_pullaway_accel_step_floor", accel_limits, selection=PLANNER_SEED_FLOOR,
      should_stop=False, force=True, group="low_speed_pullaway_accel_step",
    ) if pullaway_accel_step_floor is not None else None,
    build_planner_seed_accel_candidate(
      planner, "low_speed_pullaway_accel_step_cap", pullaway_accel_step_cap, has_lead,
      "low_speed_pullaway_accel_step_cap", accel_limits, should_stop=False, force=True,
      group="low_speed_pullaway_accel_step",
    ) if pullaway_accel_step_cap is not None and not pullaway_step_cap_suppressed else None,
  )


def lead_pullaway_intent_debug(intent: LeadPullawayIntent) -> dict[str, object]:
  rejected_reason = "" if intent.active else str(intent.reason)
  pulse_cap = get_lead_pullaway_runway_output_cap(intent)
  return {
    "lead_pullaway_phase": intent.phase.value,
    "lead_pullaway_reason": str(intent.reason),
    "lead_pullaway_track_id": int(intent.track_id),
    "lead_pullaway_pulse_timer": float(intent.pulse_timer),
    "lead_pullaway_cooldown_timer": float(intent.cooldown_timer),
    "lead_pullaway_gap_excess": float(intent.gap_excess),
    "lead_pullaway_predicted_gap_opening": float(intent.predicted_gap_opening),
    "lead_pullaway_a_floor": float(intent.a_floor),
    "lead_pullaway_predicted_gap": float(intent.predicted_gap),
    "lead_pullaway_safe_accel_cap": float(intent.safe_accel_cap),
    "lead_pullaway_lead_accel_trend": float(intent.lead_accel_trend),
    "lead_pullaway_runway_margin": float(intent.runway_margin),
    "lead_pullaway_runway_margin_now": float(intent.runway_margin_now),
    "lead_pullaway_runway_margin_t": float(intent.runway_margin_t),
    "lead_pullaway_runway_creation": float(intent.runway_creation),
    "lead_pullaway_lead_created_runway": bool(intent.lead_created_runway),
    "lead_pullaway_early_authority": bool(intent.early_authority),
    "lead_pullaway_early_authority_reason": str(intent.early_authority_reason),
    "lead_pullaway_pulse_floor": float(intent.a_floor),
    "lead_pullaway_pulse_cap": 0.0 if pulse_cap is None else float(pulse_cap),
    "lead_pullaway_coast_required": bool(intent.coast_required),
    "lead_pullaway_pulse_capped_by_runway": bool(intent.pulse_capped_by_runway),
    "lead_pullaway_crawl_cap_released_by_runway": bool(intent.crawl_cap_released_by_runway),
    "lead_pullaway_low_speed_step_cap_suppressed_by_runway": bool(intent.low_speed_step_cap_suppressed_by_runway),
    "lead_pullaway_runway_trend": str(intent.runway_trend),
    "lead_pullaway_rejected_reason": rejected_reason,
    "lead_pullaway_selected_or_rejected_reason": str(intent.reason),
  }


def get_lead_pullaway_runway_output_cap(intent: LeadPullawayIntent | None) -> float | None:
  if intent is None:
    return None
  safe_cap = max(0.0, _finite_float(intent.safe_accel_cap))
  if intent.phase == LeadPullawayPhase.NORMAL and str(intent.reason) == "pullaway_cooldown":
    if bool(intent.coast_required):
      return 0.0
    return min(LEAD_PULLAWAY_COOLDOWN_ACCEL_CAP, safe_cap)
  if not bool(intent.active):
    return None
  if bool(intent.coast_required):
    return 0.0
  if intent.phase == LeadPullawayPhase.PULSE:
    return min(LEAD_PULLAWAY_PULSE_ACCEL_CAP, safe_cap)
  if intent.phase == LeadPullawayPhase.GAP_CLOSURE:
    return min(EXCESS_GAP_CLOSURE_ACCEL_CAP, safe_cap)
  return None


def apply_lead_pullaway_runway_output_cap(a_target, intent: LeadPullawayIntent | None) -> float:
  cap = get_lead_pullaway_runway_output_cap(intent)
  a_target = _finite_float(a_target)
  return a_target if cap is None else min(a_target, cap)


def apply_lead_pullaway_final_output_shaping(a_target, intent: LeadPullawayIntent | None, prev_a_target, dt,
                                             selected_reason="") -> float:
  output = _finite_float(a_target)
  lead_pullaway_gap_closure_selected = bool(str(selected_reason) == EXCESS_GAP_CLOSURE_REASON)
  if (
    intent is not None and
    lead_pullaway_gap_closure_selected and
    getattr(intent, "phase", LeadPullawayPhase.HOLD) == LeadPullawayPhase.GAP_CLOSURE and
    not bool(getattr(intent, "coast_required", False)) and
    output < _finite_float(prev_a_target)
  ):
    output = max(
      output,
      _finite_float(prev_a_target) - LEAD_PULLAWAY_RUNWAY_CAP_RELEASE_JERK * max(0.0, _finite_float(dt)),
    )
  return apply_lead_pullaway_runway_output_cap(output, intent)


def build_lead_pullaway_intent_seed_candidates(planner, has_lead, accel_limits, intent: LeadPullawayIntent) -> tuple[PlannerSeedCandidate, ...]:
  if not intent.active:
    return ()

  debug = lead_pullaway_intent_debug(intent)
  if intent.phase == LeadPullawayPhase.PULSE:
    pulse_cap = min(LEAD_PULLAWAY_PULSE_ACCEL_CAP, max(0.0, _finite_float(intent.safe_accel_cap)))
    return _planner_seed_candidates(
      build_planner_seed_accel_candidate(
        planner, "lead_pullaway_pulse", intent.a_floor, has_lead,
        LEAD_PULLAWAY_PULSE_REASON, accel_limits, should_stop=False, selection=PLANNER_SEED_FLOOR,
        group="lead_pullaway_pulse", debug=debug,
      ),
      build_planner_seed_accel_candidate(
        planner, "lead_pullaway_pulse_accel_cap", pulse_cap, has_lead,
        LEAD_PULLAWAY_PULSE_CAP_REASON, accel_limits, should_stop=False, force=True,
        group="lead_pullaway_pulse", debug=debug,
      ),
    )

  if intent.phase == LeadPullawayPhase.GAP_CLOSURE:
    gap_closure_cap = min(EXCESS_GAP_CLOSURE_ACCEL_CAP, max(0.0, _finite_float(intent.safe_accel_cap)))
    return _planner_seed_candidates(
      build_planner_seed_accel_candidate(
        planner, "excess_gap_closure", intent.a_floor, has_lead,
        EXCESS_GAP_CLOSURE_REASON, accel_limits, should_stop=False, selection=PLANNER_SEED_FLOOR,
        group="excess_gap_closure", debug=debug,
      ),
      build_planner_seed_accel_candidate(
        planner, "excess_gap_closure_accel_cap", gap_closure_cap, has_lead,
        EXCESS_GAP_CLOSURE_CAP_REASON, accel_limits, should_stop=False, force=True,
        group="excess_gap_closure", debug=debug,
      ),
    )

  return ()


def build_moving_lead_seed_candidates(planner, has_lead, accel_limits, *, moving_stop_guard_a_target=None,
                                      moving_stop_guard_debug=None, lead_accel_recovery_a_target=None, lead_stop_approach_slewed_a_target=None,
                                      lead_stop_approach_base_a_target=None,
                                      routine_lead_approach_a_target=None, routine_lead_approach_debug=None) -> tuple[PlannerSeedCandidate, ...]:
  lead_stop_approach_slew_selection = PLANNER_SEED_CAP
  if lead_stop_approach_slewed_a_target is not None and lead_stop_approach_base_a_target is not None:
    lead_stop_approach_slew_selection = (
      PLANNER_SEED_FLOOR if lead_stop_approach_slewed_a_target > lead_stop_approach_base_a_target else PLANNER_SEED_CAP
    )
  moving_stop_guard_group = "lead_stop_approach_slew" if lead_stop_approach_slew_selection == PLANNER_SEED_FLOOR else ""
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "moving_lead_stop_gap_guard", moving_stop_guard_a_target, has_lead,
      "moving_lead_stop_gap_guard", accel_limits, group=moving_stop_guard_group,
      debug=moving_stop_guard_debug,
    ) if moving_stop_guard_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "routine_lead_approach", routine_lead_approach_a_target, has_lead,
      ROUTINE_LEAD_APPROACH_SEED_REASON, accel_limits, selection=PLANNER_SEED_FLOOR,
      debug=routine_lead_approach_debug,
    ) if routine_lead_approach_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "lead_accel_recovery", lead_accel_recovery_a_target, has_lead,
      "lead_accel_recovery", accel_limits, selection=PLANNER_SEED_FLOOR,
    ) if lead_accel_recovery_a_target is not None else None,
    build_planner_seed_accel_candidate(
      planner, "lead_stop_approach_slew", lead_stop_approach_slewed_a_target, has_lead,
      "lead_stop_approach_slew", accel_limits, selection=lead_stop_approach_slew_selection,
      force=True, group="lead_stop_approach_slew" if lead_stop_approach_slew_selection == PLANNER_SEED_FLOOR else "",
    ) if lead_stop_approach_slewed_a_target is not None and lead_stop_approach_base_a_target is not None and not np.isclose(
      lead_stop_approach_slewed_a_target, lead_stop_approach_base_a_target,
    ) else None,
  )


def build_lead_loss_seed_candidates(planner, has_lead, accel_limits, *, lead_loss_e2e_guard_a_target=None) -> tuple[PlannerSeedCandidate, ...]:
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "lead_loss_e2e_guard", lead_loss_e2e_guard_a_target, has_lead,
      "lead_loss_e2e_guard", accel_limits, selection=PLANNER_SEED_FLOOR,
    ) if lead_loss_e2e_guard_a_target is not None else None,
  )


def build_cruise_coast_seed_candidates(planner, has_lead, accel_limits, *, active=False, a_target=0.0) -> tuple[PlannerSeedCandidate, ...]:
  return _planner_seed_candidates(
    build_planner_seed_accel_candidate(
      planner, "cruise_coast", a_target, has_lead, "plain_cruise_overspeed_coast", accel_limits,
      selection=PLANNER_SEED_FLOOR,
    ) if active else None,
  )


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

  required_decel = v_ego**2 / (2.0 * max(stop_distance + E2E_CLOSE_STOP_DECEL_BUFFER, E2E_CLOSE_STOP_DECEL_BUFFER))
  target_decel = min(required_decel, E2E_CLOSE_STOP_DECEL_MAX)
  return min(raw_e2e_accel, -target_decel), should_stop, True


def get_e2e_runway_comfort_accel(v_ego, raw_e2e_accel, coast_accel, model_msg, e2e_active, prev_output_a_target,
                                  reset_state=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
                                  engage_stop_bootstrap_active=False, has_radar_lead=False, dt=DT_MDL, traction_risk=0.0):
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
  expected_distance *= get_traction_runway_scale(traction_risk)
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

  light_decel_cap = min(coast_accel - E2E_RUNWAY_COMFORT_COAST_MARGIN, -get_traction_light_decel(traction_risk))
  comfort_cap = (1.0 - blend) * light_decel_cap + blend * raw_e2e_accel
  governed_accel = max(raw_e2e_accel, comfort_cap)

  max_negative_step = get_traction_negative_ramp_rate(E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE, traction_risk) * max(dt, 0.0)
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


def get_turn_lateral_accel(v_ego, angle_steers, CP, control_calculation_hardening=False,
                           vehicle_model=None, roll=0.0, accurate_lateral_accel=False):
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  if accurate_lateral_accel and vehicle_model is not None:
    return lateral_accel_from_steering_angle(v_ego, angle_steers * CV.DEG_TO_RAD, vehicle_model, roll)
  if control_calculation_hardening:
    return v_ego**2 * VehicleModel(CP).calc_curvature(angle_steers * CV.DEG_TO_RAD, v_ego, 0.0)
  return v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP, control_calculation_hardening=False,
                         vehicle_model=None, roll=0.0, accurate_lateral_accel=False):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = get_turn_lateral_accel(v_ego, angle_steers, CP, control_calculation_hardening,
                               vehicle_model, roll, accurate_lateral_accel)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.0))

  return [a_target[0], min(a_target[1], a_x_allowed)]


def apply_curve_load_comfort_accel_limit(v_ego, angle_steers, a_target, CP, control_calculation_hardening=False,
                                         vehicle_model=None, roll=0.0, accurate_lateral_accel=False,
                                         urgent_bypass=False):
  if urgent_bypass or v_ego < CURVE_LOAD_COMFORT_MIN_V_EGO or a_target[1] <= 0.0:
    return list(a_target)

  a_total_max = float(np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V))
  if a_total_max <= 0.0:
    return list(a_target)
  a_y = abs(get_turn_lateral_accel(v_ego, angle_steers, CP, control_calculation_hardening,
                                   vehicle_model, roll, accurate_lateral_accel))
  lateral_load_ratio = a_y / a_total_max
  if lateral_load_ratio <= CURVE_LOAD_COMFORT_TAPER_START:
    return list(a_target)

  positive_accel_scale = float(np.interp(
    lateral_load_ratio,
    [CURVE_LOAD_COMFORT_TAPER_START, CURVE_LOAD_COMFORT_TAPER_FULL],
    [1.0, 0.0],
  ))
  comfort_upper = max(0.0, float(a_target[1]) * positive_accel_scale)
  return [a_target[0], min(a_target[1], comfort_upper)]


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


def get_model_lead_pullaway(model_msg, radar_lead, v_ego, horizon=CREEP_TO_STOP_GAP_MODEL_LEAD_HORIZON, radar_v_lead_override=None):
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
  radar_v_lead = (
    float(radar_v_lead_override) if radar_v_lead_override is not None
    else float(getattr(radar_lead, "vLeadK", getattr(radar_lead, "vLead", 0.0)))
  )
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


def get_lead_stop_approach_slewed_accel(v_ego, d_rel, v_lead, a_lead, prev_a_target, a_target, dt, traction_risk=0.0):
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

  jerk_scale = 1.0 if hard_braking_lead else (1.0 - (1.0 - TRACTION_RISK_LEAD_STOP_SLEW_MIN_SCALE) * clip_traction_risk(traction_risk))
  max_delta = LEAD_STOP_APPROACH_DECEL_SLEW_MAX_JERK * jerk_scale * dt
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


def get_lead_d_rel(lead):
  return _finite_float(getattr(lead, "dRel", 0.0))


def get_lead_v_lead(lead):
  return _finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))


def get_lead_v_rel(lead, v_ego):
  v_lead = get_lead_v_lead(lead)
  return _finite_float(getattr(lead, "vRel", v_lead - v_ego), v_lead - v_ego)


def get_lead_a_lead(lead):
  return _finite_float(getattr(lead, "aLeadK", 0.0))


def get_lead_a_tau(lead):
  return _finite_float(getattr(lead, "aLeadTau", 0.0))


def get_lead_model_prob(lead):
  return _finite_float(getattr(lead, "modelProb", 0.0))


def get_lead_y_rel(lead):
  return _finite_float(getattr(lead, "yRel", 0.0))


def lead_laterally_exited(lead) -> bool:
  return bool(
    getattr(lead, "status", False) and
    abs(_finite_float(getattr(lead, "yRel", 0.0))) >= LEAD_SPEEDUP_GUARD_LATERAL_EXIT_Y_REL
  )


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


def should_allow_stopped_lead_stop_gap_guard(v_ego, d_rel, v_lead, lane_change_active=False):
  if not lane_change_active:
    return True
  closing_speed = max(_finite_float(v_ego) - _finite_float(v_lead), 0.0)
  if closing_speed <= 0.1:
    return True
  ttc = max(_finite_float(d_rel), 0.0) / closing_speed
  return ttc <= STOPPED_LEAD_STOP_GAP_GUARD_LANE_CHANGE_MAX_TTC


def get_moving_lead_stop_gap_guard_gradual_accel(v_ego, d_rel, v_lead, a_lead, t_follow):
  _desired_gap, caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  predicted_v_lead = max(0.0, v_lead + min(a_lead, 0.0) * MOVING_LEAD_STOP_GAP_GUARD_PREDICT_T)
  predicted_closing_speed = max(v_ego - predicted_v_lead, 0.0)
  allowed_closing_speed = float(np.interp(
    d_rel, [danger_gap, caution_gap], [0.0, MOVING_LEAD_STOP_GAP_GUARD_ALLOWED_CLOSING]
  ))
  closing_excess = max(predicted_closing_speed - allowed_closing_speed, 0.0)
  if closing_excess <= 0.0:
    return 0.0
  return -min(MOVING_LEAD_STOP_GAP_GUARD_CLOSING_DECEL_CAP, closing_excess / MOVING_LEAD_STOP_GAP_GUARD_PREDICT_T)


def get_slower_lead_approach_accel(v_ego, d_rel, v_lead, a_lead, t_follow, desired_gap=None, caution_gap=None):
  if desired_gap is None or caution_gap is None:
    desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  closing_speed = max(_finite_float(v_ego) - _finite_float(v_lead), 0.0)
  gap_deficit = _finite_float(desired_gap) - _finite_float(d_rel)
  if (
    _finite_float(d_rel) <= _finite_float(caution_gap) or
    gap_deficit < MOVING_LEAD_SLOWER_APPROACH_MIN_GAP_DEFICIT or
    closing_speed < MOVING_LEAD_SLOWER_APPROACH_MIN_CLOSING
  ):
    return None

  lead_decel_excess = max(0.0, -_finite_float(a_lead) - MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL)
  decel = (
    MOVING_LEAD_SLOWER_APPROACH_MIN_TARGET_DECEL +
    (closing_speed - MOVING_LEAD_SLOWER_APPROACH_MIN_CLOSING) * MOVING_LEAD_SLOWER_APPROACH_CLOSING_GAIN +
    gap_deficit * MOVING_LEAD_SLOWER_APPROACH_GAP_GAIN +
    lead_decel_excess * MOVING_LEAD_SLOWER_APPROACH_LEAD_DECEL_GAIN
  )
  decel = float(np.clip(
    decel,
    MOVING_LEAD_SLOWER_APPROACH_MIN_TARGET_DECEL,
    MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP,
  ))
  return -decel


def _routine_lead_approach_debug(**overrides):
  debug = {
    "routine_lead_approach_active": False,
    "routine_lead_approach_reason": "inactive",
    "routine_lead_approach_urgent": False,
    "routine_lead_phase": "inactive",
    "routine_lead_coast_first_active": False,
    "routine_lead_anticipatory_active": False,
    "routine_lead_response_time": ROUTINE_LEAD_RESPONSE_TIME,
    "routine_lead_gap_lost_to_response": 0.0,
    "routine_lead_effective_d_rel": 0.0,
    "routine_lead_projected_gap_raw": 0.0,
    "routine_lead_projected_gap_response_compensated": 0.0,
    "routine_lead_gap_after_coast": 0.0,
    "routine_lead_preview_t": ROUTINE_LEAD_APPROACH_PREVIEW_T,
    "routine_lead_projected_gap": 0.0,
    "routine_lead_projected_closing": 0.0,
    "routine_lead_desired_gap": 0.0,
    "routine_lead_caution_gap": 0.0,
    "routine_lead_danger_gap": 0.0,
    "routine_lead_required_decel": 0.0,
    "routine_lead_required_decel_after_coast": 0.0,
    "routine_lead_allowed_closing": 0.0,
    "routine_lead_closing_excess": 0.0,
    "routine_lead_compression_blend": 0.0,
    "routine_lead_compression_budget": 0.0,
    "routine_lead_comfort_budget": 0.0,
    "routine_lead_projected_compression_budget": 0.0,
    "routine_lead_projected_comfort_budget": 0.0,
    "routine_lead_valid_approach": False,
    "routine_lead_raw_a_target": 0.0,
    "routine_lead_ramped_a_target": 0.0,
    "routine_lead_a_ego": 0.0,
    "routine_lead_decel_shortfall": 0.0,
    "routine_lead_jerk_limited": False,
    "routine_lead_release_limited": False,
    "routine_lead_urgent_bypass": False,
    "routine_lead_distance_to_caution": 0.0,
    "routine_lead_distance_to_danger": 0.0,
    "routine_lead_time_to_caution": 0.0,
    "routine_lead_far_coast_active": False,
  }
  debug.update(overrides)
  return debug


def get_routine_lead_approach_accel(*, v_ego, d_rel, v_lead, a_lead, y_rel, t_follow,
                                    prev_a_target=None, dt=DT_MDL, a_ego=None,
                                    budget=None) -> RoutineLeadApproach:
  if budget is None:
    budget = get_comfort_budget(1)  # standard
  v_ego = _finite_float(v_ego)
  d_rel = _finite_float(d_rel)
  v_lead = _finite_float(v_lead)
  a_lead = _finite_float(a_lead)
  y_rel = _finite_float(y_rel)
  t_follow = _finite_float(t_follow)
  a_ego = _finite_float(a_ego)
  desired_gap, caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  desired_gap = float(desired_gap)
  caution_gap = float(caution_gap)
  danger_gap = float(danger_gap)
  closing_speed = max(v_ego - v_lead, 0.0)
  relative_accel = max(-a_lead, 0.0)
  preview_t = ROUTINE_LEAD_APPROACH_PREVIEW_T
  response_t = budget.response_time
  response_gap_loss = closing_speed * response_t + 0.5 * relative_accel * response_t**2
  effective_d_rel = max(0.0, d_rel - response_gap_loss)
  delayed_closing_speed = closing_speed + relative_accel * response_t
  projected_closing_speed = max(delayed_closing_speed + relative_accel * preview_t, 0.0)
  predicted_gap_raw = max(0.0, d_rel - closing_speed * preview_t - 0.5 * relative_accel * preview_t**2)
  predicted_gap = max(0.0, effective_d_rel - delayed_closing_speed * preview_t - 0.5 * relative_accel * preview_t**2)
  required_decel = float(get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead))
  distance_to_caution = d_rel - caution_gap
  distance_to_danger = d_rel - danger_gap
  time_to_caution = (distance_to_caution / max(closing_speed, 1e-3)) if closing_speed > ROUTINE_LEAD_FAR_COAST_MIN_CLOSING else float('inf')
  compression_budget = d_rel - danger_gap
  comfort_budget = d_rel - caution_gap
  projected_compression_budget = predicted_gap - danger_gap
  projected_comfort_budget = predicted_gap - caution_gap
  routine_floor_gap = danger_gap + ROUTINE_LEAD_APPROACH_DANGER_GAP_MARGIN
  gap_after_coast = predicted_gap
  required_decel_after_coast = projected_closing_speed**2 / (2.0 * max(gap_after_coast - routine_floor_gap, 0.1))
  base_debug = _routine_lead_approach_debug(
    routine_lead_projected_gap=predicted_gap,
    routine_lead_response_time=response_t,
    routine_lead_gap_lost_to_response=response_gap_loss,
    routine_lead_effective_d_rel=effective_d_rel,
    routine_lead_projected_gap_raw=predicted_gap_raw,
    routine_lead_projected_gap_response_compensated=predicted_gap,
    routine_lead_gap_after_coast=gap_after_coast,
    routine_lead_projected_closing=projected_closing_speed,
    routine_lead_desired_gap=desired_gap,
    routine_lead_caution_gap=caution_gap,
    routine_lead_danger_gap=danger_gap,
    routine_lead_required_decel=required_decel,
    routine_lead_required_decel_after_coast=required_decel_after_coast,
    routine_lead_a_ego=a_ego,
    routine_lead_distance_to_caution=distance_to_caution,
    routine_lead_distance_to_danger=distance_to_danger,
    routine_lead_time_to_caution=time_to_caution,
    routine_lead_compression_budget=compression_budget,
    routine_lead_comfort_budget=comfort_budget,
    routine_lead_projected_compression_budget=projected_compression_budget,
    routine_lead_projected_comfort_budget=projected_comfort_budget,
  )

  invalid = (
    v_ego < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_EGO or
    v_lead < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD or
    v_lead >= v_ego or
    closing_speed < ROUTINE_LEAD_APPROACH_MIN_CLOSING or
    abs(y_rel) > MOVING_LEAD_STOP_GAP_GUARD_MAX_Y_REL
  )
  urgent = (
    d_rel <= danger_gap or
    closing_speed >= MOVING_LEAD_STOP_GAP_GUARD_URGENT_CLOSING or
    a_lead <= -MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL
  )
  if invalid:
    return RoutineLeadApproach(urgent=urgent, required_decel=required_decel, predicted_gap=predicted_gap,
                               projected_closing_speed=projected_closing_speed, reason="invalid",
                               compression_budget=compression_budget, comfort_budget=comfort_budget,
                               projected_compression_budget=projected_compression_budget,
                               projected_comfort_budget=projected_comfort_budget, far_coast_active=False,
                               firm_routine_decel_cap=budget.firm_routine_decel_cap,
                               debug={**base_debug, "routine_lead_approach_reason": "invalid",
                                      "routine_lead_approach_urgent": urgent})

  # Far-lead coast: remove positive accel early for stable valid slower leads
  # that are still well above caution gap but closing with finite TTC.
  if not invalid and not urgent and d_rel > caution_gap and closing_speed >= ROUTINE_LEAD_FAR_COAST_MIN_CLOSING and time_to_caution <= budget.far_coast_ttc and time_to_caution > 0.0:
    far_coast_active = True
  else:
    far_coast_active = False

  routine_start_gap = caution_gap + max(closing_speed, projected_closing_speed) * preview_t
  risk_gap = min(effective_d_rel, predicted_gap)
  blend_span = max(routine_start_gap - routine_floor_gap, 0.1)
  blend_x = float(np.clip((routine_start_gap - risk_gap) / blend_span, 0.0, 1.0))
  compression_blend = blend_x * blend_x * (3.0 - 2.0 * blend_x)
  allowed_closing_speed = float(np.interp(
    risk_gap, [routine_floor_gap, routine_start_gap], [0.0, MOVING_LEAD_STOP_GAP_GUARD_ALLOWED_CLOSING]
  ))
  closing_excess = max(projected_closing_speed - allowed_closing_speed, 0.0)
  anticipatory_active = bool(d_rel > caution_gap and predicted_gap <= caution_gap)
  _routine_approach_active = bool(
    compression_blend >= ROUTINE_LEAD_APPROACH_MIN_BLEND and
    closing_excess > 0.0
  )
  active = bool(_routine_approach_active or far_coast_active)
  if not active and not far_coast_active:
    reason = "below_threshold"
    debug = {
      **base_debug,
      "routine_lead_approach_reason": reason,
      "routine_lead_approach_urgent": urgent,
      "routine_lead_anticipatory_active": anticipatory_active,
      "routine_lead_allowed_closing": allowed_closing_speed,
      "routine_lead_closing_excess": closing_excess,
      "routine_lead_compression_blend": compression_blend,
      "routine_lead_urgent_bypass": False,
      "routine_lead_far_coast_active": far_coast_active,
    }
    return RoutineLeadApproach(active=False, far_coast_active=far_coast_active, urgent=urgent, required_decel=required_decel,
                               allowed_closing_speed=allowed_closing_speed, closing_excess=closing_excess,
                               compression_blend=compression_blend, compression_budget=compression_budget,
                               comfort_budget=comfort_budget, projected_compression_budget=projected_compression_budget,
                               projected_comfort_budget=projected_comfort_budget,
                               predicted_gap=predicted_gap,
                               projected_closing_speed=projected_closing_speed,
                               firm_routine_decel_cap=budget.firm_routine_decel_cap,
                               reason=reason, debug=debug)

  lead_decel_excess = max(0.0, -a_lead - MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL)
  actual_decel = max(0.0, -a_ego)
  decel = (
    ROUTINE_LEAD_APPROACH_DECEL_MIN +
    closing_excess * MOVING_LEAD_SLOWER_APPROACH_CLOSING_GAIN +
    compression_blend * (budget.routine_decel_cap - ROUTINE_LEAD_APPROACH_DECEL_MIN) * 0.5 +
    lead_decel_excess * MOVING_LEAD_SLOWER_APPROACH_LEAD_DECEL_GAIN
  )
  decel = float(np.clip(decel, ROUTINE_LEAD_APPROACH_DECEL_MIN, budget.routine_decel_cap))
  if far_coast_active and not _routine_approach_active:
    phase = "far_lead_coast"
    raw_a_target = 0.0
  elif compression_blend < ROUTINE_LEAD_APPROACH_COAST_BLEND and not urgent:
    phase = "free_coast"
    raw_a_target = 0.0
  elif compression_blend < ROUTINE_LEAD_APPROACH_SOFT_BLEND and not urgent:
    phase = "soft_decel"
    soft_x = float(np.clip(
      (compression_blend - ROUTINE_LEAD_APPROACH_COAST_BLEND) /
      max(ROUTINE_LEAD_APPROACH_SOFT_BLEND - ROUTINE_LEAD_APPROACH_COAST_BLEND, 1e-3),
      0.0, 1.0,
    ))
    soft_cap = ROUTINE_LEAD_APPROACH_DECEL_MIN + soft_x * (budget.soft_decel_cap - ROUTINE_LEAD_APPROACH_DECEL_MIN)
    decel = min(decel, soft_cap)
    raw_a_target = -decel
  else:
    phase = "routine_decel"
    raw_a_target = -decel
  requested_decel = max(0.0, -raw_a_target)
  if prev_a_target is None:
    ramped_a_target = raw_a_target
  else:
    ramped_a_target = approach_accel_with_jerk_limit(
      prev_a_target, raw_a_target, dt,
      jerk_up=budget.routine_release_jerk,
      jerk_down=budget.routine_negative_jerk,
    )
  jerk_limited = bool(ramped_a_target > raw_a_target + 1e-6)
  release_limited = bool(ramped_a_target < raw_a_target - 1e-6)
  debug = {
    **base_debug,
    "routine_lead_approach_active": True,
    "routine_lead_approach_reason": "routine_slower_lead_approach",
    "routine_lead_approach_urgent": urgent,
    "routine_lead_phase": phase,
    "routine_lead_coast_first_active": phase == "free_coast",
    "routine_lead_anticipatory_active": anticipatory_active,
    "routine_lead_allowed_closing": allowed_closing_speed,
    "routine_lead_closing_excess": closing_excess,
    "routine_lead_compression_blend": compression_blend,
    "routine_lead_raw_a_target": raw_a_target,
    "routine_lead_ramped_a_target": ramped_a_target,
    "routine_lead_decel_shortfall": max(0.0, requested_decel - actual_decel),
    "routine_lead_jerk_limited": jerk_limited,
    "routine_lead_release_limited": release_limited,
    "routine_lead_far_coast_active": far_coast_active,
    "routine_lead_time_to_caution": time_to_caution,
  }
  return RoutineLeadApproach(active=(active or far_coast_active), far_coast_active=far_coast_active, urgent=urgent,
                             raw_a_target=raw_a_target, ramped_a_target=ramped_a_target,
                             required_decel=required_decel, allowed_closing_speed=allowed_closing_speed,
                             closing_excess=closing_excess, compression_blend=compression_blend,
                             compression_budget=compression_budget, comfort_budget=comfort_budget,
                             projected_compression_budget=projected_compression_budget,
                             projected_comfort_budget=projected_comfort_budget,
                             predicted_gap=predicted_gap, projected_closing_speed=projected_closing_speed,
                             firm_routine_decel_cap=budget.firm_routine_decel_cap,
                             reason="routine_slower_lead_approach", debug=debug)


def is_valid_routine_lead_approach(*, primary_lead_context, brake_pressed=False, gas_pressed=False,
                                    force_slow_decel=False, independent_stop_threat=False,
                                    alternate_lead_threat_active=False) -> bool:
  """Check whether a routine lead approach is valid for comfort shaping.

  A valid routine lead approach requires:
  - lead exists and is path-relevant
  - lead is stable, not new, not flicker, not shadow
  - no driver brake/gas override
  - no force_slow.
  - no independent stop threat.
  - no alternate closer threat.
  - lead progress is allowed (not suppressive-only).
  """
  if primary_lead_context is None:
    return False
  if bool(brake_pressed or gas_pressed or force_slow_decel):
    return False
  if bool(independent_stop_threat):
    return False
  if bool(alternate_lead_threat_active):
    return False
  if bool(getattr(primary_lead_context, "alternate_threat_active", False)):
    return False
  if bool(getattr(primary_lead_context, "shadow_active", False)):
    return False
  behavior = getattr(primary_lead_context, "behavior", None)
  if behavior is None:
    return False
  if bool(getattr(behavior, "shadow", False)):
    return False
  if bool(getattr(behavior, "flicker_guard_timer", 0.0) > 0.0):
    return False
  if bool(getattr(behavior, "new_lead", False)):
    return False
  if not bool(getattr(behavior, "stable", False)):
    return False
  if not bool(getattr(primary_lead_context, "lead_progress_allowed", False)):
    blocked_reason = str(getattr(primary_lead_context, "lead_release_blocked_reason", ""))
    if blocked_reason == "primary_physical_lead_suppressive":
      return False
  return True


def get_moving_lead_stop_gap_guard_accel(v_ego, d_rel, v_lead, a_lead, y_rel, t_follow, a_ego=None,
                                         prev_a_target=None, dt=DT_MDL, return_debug=False,
                                         budget=None):
  routine = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=a_lead, y_rel=y_rel, t_follow=t_follow,
    prev_a_target=prev_a_target, dt=dt, a_ego=a_ego, budget=budget,
  )
  debug = dict(routine.debug)

  def _result(a_target):
    return (a_target, debug) if return_debug else a_target

  def _selected_result(a_target, *, existing_target=None, existing_reason="none", safety_relevant=False):
    routine_can_own = bool(routine.active and existing_target is not None and not safety_relevant)
    debug.update({
      "routine_lead_existing_target": 0.0 if existing_target is None else float(existing_target),
      "routine_lead_existing_target_reason": str(existing_reason),
      "routine_lead_existing_target_safety_relevant": bool(safety_relevant and existing_target is not None),
      "routine_lead_can_own_nonurgent_shape": routine_can_own,
      "routine_lead_should_defer_to_existing_target": bool(routine.active and existing_target is not None and safety_relevant),
      "routine_lead_selected_target": 0.0 if a_target is None else float(a_target),
    })
    return _result(a_target)

  if (
    v_ego < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_EGO or
    v_lead < MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD or
    v_lead >= v_ego or
    abs(y_rel) > MOVING_LEAD_STOP_GAP_GUARD_MAX_Y_REL
  ):
    return _selected_result(None)

  _desired_gap, caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow, a_ego=a_ego)
  target = float(target)
  closing_speed = max(v_ego - v_lead, 0.0)
  required_decel = float(get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead))
  # Crossing the caution gap while still comfortably outside danger should stay
  # on the pre-danger closing ramp. Reserve urgent bypass for true danger-gap,
  # short-TTC, limited-runway, or hard lead-braking cases.
  danger_margin = d_rel - danger_gap
  danger_ttc = danger_margin / max(closing_speed, 1e-3) if closing_speed > 0.0 else float("inf")
  hard_lead_braking = bool(a_lead <= -MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL)
  near_danger_gap = bool(danger_margin <= MOVING_LEAD_STOP_GAP_GUARD_PRE_DANGER_URGENT_MARGIN)
  runway_urgent = bool(required_decel >= MOVING_LEAD_STOP_GAP_GUARD_URGENT_REQUIRED_DECEL)
  hard_lead_urgent = bool(
    hard_lead_braking and (
      d_rel <= caution_gap or
      required_decel >= MOVING_LEAD_STOP_GAP_GUARD_URGENT_REQUIRED_DECEL * 0.5
    )
  )
  closing_urgent = bool(
    closing_speed >= MOVING_LEAD_STOP_GAP_GUARD_URGENT_CLOSING and
    (near_danger_gap or danger_ttc <= MOVING_LEAD_STOP_GAP_GUARD_PRE_DANGER_URGENT_TTC or runway_urgent)
  )
  moving_guard_urgent = bool(
    d_rel <= danger_gap or
    runway_urgent or
    hard_lead_urgent or
    closing_urgent
  )
  existing_target_safety_relevant = bool(
    d_rel <= danger_gap or
    danger_ttc <= MOVING_LEAD_STOP_GAP_GUARD_PRE_DANGER_URGENT_TTC or
    runway_urgent or
    hard_lead_urgent or
    closing_urgent or
    hard_lead_braking
  )
  debug.update({
    "moving_lead_stop_gap_guard_target": target,
    "moving_lead_stop_gap_guard_cost": float(cost),
    "moving_lead_stop_gap_guard_closing_speed": closing_speed,
    "moving_lead_stop_gap_guard_required_decel": required_decel,
    "moving_lead_stop_gap_guard_danger_margin": danger_margin,
    "moving_lead_stop_gap_guard_danger_ttc": danger_ttc,
    "moving_lead_stop_gap_guard_near_danger": near_danger_gap,
    "moving_lead_stop_gap_guard_runway_urgent": runway_urgent,
    "moving_lead_stop_gap_guard_closing_urgent": closing_urgent,
    "moving_lead_stop_gap_guard_hard_lead_urgent": hard_lead_urgent,
    "moving_lead_stop_gap_guard_urgent": moving_guard_urgent,
    "routine_lead_existing_target": 0.0,
    "routine_lead_existing_target_reason": "none",
    "routine_lead_existing_target_safety_relevant": False,
    "routine_lead_can_own_nonurgent_shape": False,
    "routine_lead_should_defer_to_existing_target": False,
    "routine_lead_selected_target": 0.0,
    "routine_lead_existing_target_fcw_or_force_slow_unavailable": False,
  })
  far_hard_braking_limited_runway = bool(
    d_rel > caution_gap and
    hard_lead_braking and
    required_decel >= MOVING_LEAD_STOP_GAP_GUARD_URGENT_REQUIRED_DECEL * 0.5
  )
  if a_lead > -MOVING_LEAD_STOP_GAP_GUARD_MIN_LEAD_DECEL:
    if routine.active:
      return _selected_result(routine.ramped_a_target)
    return _selected_result(None)
  if d_rel > caution_gap and not far_hard_braking_limited_runway and not moving_guard_urgent:
    slower_lead_a_target = get_slower_lead_approach_accel(
      v_ego, d_rel, v_lead, a_lead, t_follow, desired_gap=_desired_gap, caution_gap=caution_gap,
    )
    if routine.active and slower_lead_a_target is not None:
      slower_target_safety_relevant = bool(hard_lead_braking)
      return _selected_result(
        min(slower_lead_a_target, routine.ramped_a_target) if slower_target_safety_relevant else routine.ramped_a_target,
        existing_target=slower_lead_a_target,
        existing_reason="slower_lead_approach",
        safety_relevant=slower_target_safety_relevant,
      )
    if routine.active:
      return _selected_result(routine.ramped_a_target)
    return _selected_result(slower_lead_a_target, existing_target=slower_lead_a_target, existing_reason="slower_lead_approach")
  if float(cost) <= 0.0 or target > -MOVING_LEAD_STOP_GAP_GUARD_MIN_TARGET_DECEL:
    if routine.active:
      return _selected_result(routine.ramped_a_target)
    return _selected_result(None)
  if far_hard_braking_limited_runway:
    debug["routine_lead_urgent_bypass"] = True
    debug["routine_lead_phase"] = "urgent_bypass"
    return _selected_result(target, existing_target=target, existing_reason="far_hard_braking_limited_runway", safety_relevant=True)
  decel_cap = -ACCEL_MIN if a_lead <= -MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL else MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP
  if moving_guard_urgent:
    debug["routine_lead_approach_urgent"] = True
    debug["routine_lead_urgent_bypass"] = True
    debug["routine_lead_phase"] = "urgent_bypass"
    target = min(target, -min(decel_cap, required_decel))
    return _selected_result(target, existing_target=target, existing_reason="moving_lead_stop_gap_guard", safety_relevant=True)
  else:
    target = max(target, get_moving_lead_stop_gap_guard_gradual_accel(v_ego, d_rel, v_lead, a_lead, t_follow))
    if target > -MOVING_LEAD_STOP_GAP_GUARD_MIN_TARGET_DECEL:
      if routine.active:
        return _selected_result(routine.ramped_a_target)
      return _selected_result(None)
    if routine.active:
      return _selected_result(
        min(target, routine.ramped_a_target) if existing_target_safety_relevant else routine.ramped_a_target,
        existing_target=target,
        existing_reason="moving_lead_stop_gap_guard",
        safety_relevant=existing_target_safety_relevant,
      )
  return _selected_result(target, existing_target=target, existing_reason="moving_lead_stop_gap_guard", safety_relevant=existing_target_safety_relevant)


def should_reserve_creep_to_stop_gap(primary_behavior_progress_allowed, output_should_stop, v_ego, d_rel, v_lead,
                                     brake_pressed=False, gas_pressed=False, force_slow_decel=False, reset_state=False):
  return bool(
    primary_behavior_progress_allowed and
    not output_should_stop and
    not brake_pressed and not gas_pressed and
    not force_slow_decel and not reset_state and
    v_ego < CREEP_TO_STOP_GAP_RESERVE_CREEP_MAX_V_EGO and
    v_lead < CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED and
    STOP_DISTANCE + CREEP_TO_STOP_GAP_HOLD_BUFFER < d_rel <= STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS
  )


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
    try:
      self.fast_lead_motion_evidence_param_enabled = self.params.get_bool(FAST_LEAD_MOTION_EVIDENCE_PARAM)
    except UnknownKeyName:
      self.fast_lead_motion_evidence_param_enabled = False
    self.one_pedal_cruise_hold_active = False
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.prev_reset_state = True
    # Stop/go Intent is represented by these planner-owned latches and timers.
    # They may hold or restrict stop/creep/gap-fill behavior, but positive
    # progress still requires stable lead/runway evidence and must yield to
    # reset, force-slow, brake/gas, stop-threat, and physical lead safety paths.
    self.engage_stop_bootstrap_timer = 0.0
    self.e2e_close_stop_settle_active = False
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.creep_to_stop_gap_active = False
    self.pullaway_accel_step_handoff_timer = 0.0
    self.creep_stop_hold_released = False
    self.stopped_lead_gap_fill_timer = 0.0
    self.lead_loss_e2e_guard_timer = 0.0
    self.previous_lead_loss_status = False
    self.previous_lead_loss_d_rel = 0.0
    self.previous_lead_loss_model_prob = 0.0
    self.stopped_lead_gap_fill_track_id = -2
    self.stopped_lead_gap_fill_d_rel = 0.0
    self.stopped_lead_gap_fill_v_lead = 0.0
    self.lead_flicker_safety_cap_trackers = [LeadFlickerSafetyCapTracker(), LeadFlickerSafetyCapTracker()]
    self.lead_pullaway_intent_tracker = LeadPullawayIntentTracker()
    self.lead_pullaway_intent = LeadPullawayIntent()
    self.stop_release_guard_tracker = StopReleaseGuardTracker()
    self.stop_release_guard_state = StopReleaseGuardState()
    self.primary_lead_context_tracker = LeadContextTracker()
    self.primary_lead_context = empty_primary_lead_context()
    self.scc_evidence_selector = SccEvidenceSelector()

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
    mode_resolution = getattr(self, "longitudinal_mode_resolution", None)
    set_speed_advisory_mode = bool(
      mode_resolution is not None and mode_resolution.actuation_type == LongitudinalActuationType.SET_SPEED_ADVISORY
    )
    acc_mode_requested = bool(
      mode_resolution is not None and (mode_resolution.requested_mode == LongitudinalMode.ACC or set_speed_advisory_mode)
    )
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
      speed_limit_coast_accel = accel_coast
      cruise_coast_accel = accel_coast
    else:
      accel_coast = ACCEL_MAX
      speed_limit_coast_accel = 0.0
      cruise_coast_accel = 0.0

    v_ego = sm['carState'].vEgo
    try:
      car_state_sp = sm['carStateSP']
    except (KeyError, TypeError):
      car_state_sp = None
    traction_risk = get_traction_risk(car_state_sp)
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
    if acc_mode_requested:
      self.allow_throttle = True
    else:
      _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
      # Don't clip at low speeds since throttle_prob doesn't account for creep
      self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED * 2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # MPC safety keeps both real lead columns. Primary physical lead only gates suppressive/safety behavior,
    # primary behavior lead gates positive lead progress, and shadow/unstable leads may suppress but cannot
    # authorize progress. False-positive release needs both path-exit and low-risk evidence.
    lead_context_tracker = getattr(self, "primary_lead_context_tracker", None)
    if lead_context_tracker is None:
      lead_context_tracker = LeadContextTracker()
      self.primary_lead_context_tracker = lead_context_tracker
    lead_tuple = (sm['radarState'].leadOne, sm['radarState'].leadTwo)
    model_msg_for_lead_context = None if acc_mode_requested else sm['modelV2']
    self.primary_lead_context = lead_context_tracker.update(
      lead_tuple,
      get_mpc_lead_confidence_states(getattr(self, "planner_seed_mpc", self.mpc)),
      v_ego,
      self.dt,
      model_msg=model_msg_for_lead_context,
      reset_state=reset_state,
    )

    has_radar_lead = has_valid_radar_lead(sm['radarState'])
    has_confirmed_lead = has_confirmed_radar_lead(sm['radarState'])
    scc_lead_distance, scc_lead_path_y_rel, scc_lead_idx = scc_lead_geometry_from_context(
      self.primary_lead_context, sm['radarState'],
    )
    scc_evidence_selector = getattr(self, "scc_evidence_selector", None)
    if scc_evidence_selector is None:
      scc_evidence_selector = SccEvidenceSelector()
      self.scc_evidence_selector = scc_evidence_selector
    scc_selector_reset = bool(reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed)
    if mode_resolution is not None and mode_resolution.requested_mode == LongitudinalMode.SCC and not set_speed_advisory_mode:
      raw_scc_evidence = build_scc_mode_evidence(
        has_confirmed_lead, sm['modelV2'], self.scc, self.sla, self.osm_traffic_control_prior,
        speed_limit_handoff_active=bool(getattr(self, "_speed_limit_handoff_active", False)),
        lead_distance=scc_lead_distance,
        lead_path_y_rel=scc_lead_path_y_rel,
        lead_idx=scc_lead_idx,
        v_ego=v_ego,
      )
      scc_evidence = scc_evidence_selector.update(raw_scc_evidence.classify(), self.dt, reset=scc_selector_reset)
      self.longitudinal_mode_resolution = LongitudinalModeResolver.resolve(self.params, self.CP, scc_evidence=scc_evidence)
      mode_resolution = self.longitudinal_mode_resolution
    else:
      scc_evidence_selector.reset()

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist. SCC target providers are gated by
    # the pre-target SCC evidence resolution so SCC_E2E cycles cannot select stale SCC curve/map actuation first.
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired,
                                                                    v_cruise, coast_accel=speed_limit_coast_accel)

    if force_slow_decel:
      v_cruise = 0.0

    if mode_resolution is not None and mode_resolution.requested_mode == LongitudinalMode.SCC and not set_speed_advisory_mode:
      raw_scc_evidence = build_scc_mode_evidence(
        has_confirmed_lead, sm['modelV2'], self.scc, self.sla, self.osm_traffic_control_prior,
        speed_limit_handoff_active=bool(getattr(self, "_speed_limit_handoff_active", False)),
        lead_distance=scc_lead_distance,
        lead_path_y_rel=scc_lead_path_y_rel,
        lead_idx=scc_lead_idx,
        v_ego=v_ego,
      )
      scc_evidence = scc_evidence_selector.select(raw_scc_evidence.classify())
      self.longitudinal_mode_resolution = LongitudinalModeResolver.resolve(self.params, self.CP, scc_evidence=scc_evidence)
      mode_resolution = self.longitudinal_mode_resolution
      self._update_e2e_alerts_for_mode(sm)
    e2e_active = self.is_e2e(sm)
    direct_actuation_mode = bool(mode_resolution is not None and mode_resolution.actuation_type == LongitudinalActuationType.DIRECT)
    e2e_or_scc_direct = bool(
      mode_resolution is not None and mode_resolution.requested_mode in (LongitudinalMode.E2E, LongitudinalMode.SCC) and direct_actuation_mode
    )
    scc_curve_scene_allowed = bool(
      mode_resolution is not None and
      mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC and
      direct_actuation_mode and
      mode_resolution.scc_evidence.tier not in (SccEvidenceTier.STOP, SccEvidenceTier.URGENT_STOP)
    )
    lead_loss_guard_lead = get_lead_loss_e2e_guard_lead(sm['radarState'])
    custom_engage_stop_bootstrap_active = e2e_active and should_run_engage_stop_bootstrap(
      self.engage_stop_bootstrap_timer, v_ego, sm['radarState'], sm['modelV2']
    )

    self.lead_loss_e2e_guard_timer = update_lead_loss_e2e_guard_timer(
      self.lead_loss_e2e_guard_timer, self.dt,
      self.previous_lead_loss_status, self.previous_lead_loss_d_rel, self.previous_lead_loss_model_prob,
      has_confirmed_lead, is_lane_change_active(sm['modelV2']) if e2e_active else False,
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
      model_msg=None if acc_mode_requested else sm['modelV2'],
    )
    planner_seed_mpc_v_desired_trajectory = None
    planner_seed_mpc_a_desired_trajectory = None
    planner_seed_mpc_j_desired_trajectory = None
    planner_seed_mpc = getattr(self, "planner_seed_mpc", None)
    stack_resolution = getattr(self, "longitudinal_stack_resolution", None)
    custom_stack_active = is_custom_stack(getattr(stack_resolution, "resolved_stack", ""))
    use_fast_lead_motion_evidence = fast_lead_motion_evidence_enabled(
      stack_resolution, getattr(self, "fast_lead_motion_evidence_param_enabled", False)
    )
    run_planner_seed_mpc = (
      bool(getattr(sm['selfdriveState'], "enabled", True)) and
      (has_radar_lead or force_slow_decel) and
      custom_stack_active
    )
    if planner_seed_mpc is not None and run_planner_seed_mpc:
      planner_seed_mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
      planner_seed_mpc.set_cur_state(mpc_v_desired, mpc_a_desired)
      planner_seed_mpc.update(
        sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
        block_short_gap_pullaway_response=sm['carState'].brakePressed or sm['carState'].gasPressed or force_slow_decel or reset_state,
        model_msg=None if acc_mode_requested else sm['modelV2'], lead_context=self.primary_lead_context,
      )
      planner_seed_mpc_v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner_seed_mpc.v_solution)
      planner_seed_mpc_a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner_seed_mpc.a_solution)
      planner_seed_mpc_j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], planner_seed_mpc.j_solution)

    context_mpc = planner_seed_mpc if planner_seed_mpc is not None and run_planner_seed_mpc else self.mpc
    self.primary_lead_context = lead_context_tracker.update(
      lead_tuple,
      get_mpc_lead_confidence_states(context_mpc),
      v_ego,
      0.0,
      model_msg=model_msg_for_lead_context,
      dominant_idx=getattr(context_mpc, "dominant_obstacle_idx", None),
      lead_dominant_idx=getattr(context_mpc, "lead_dominant_obstacle_idx", None),
      reset_state=reset_state,
    )
    primary_lead_context = self.primary_lead_context
    primary_physical_lead = primary_lead_context.physical_lead_data(lead_tuple)
    primary_behavior_lead = primary_lead_context.behavior_lead_data(lead_tuple)
    primary_physical_state = primary_lead_context.physical
    primary_behavior_state = primary_lead_context.behavior

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
    active_mpc_lead_lateral_exit = lead_laterally_exited(get_mpc_source_lead(sm['radarState'], self.mpc.source))
    planner_seed_mpc_lead_lateral_exit = bool(
      planner_seed_mpc is not None and lead_laterally_exited(get_mpc_source_lead(sm['radarState'], planner_seed_mpc.source))
    )
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration if e2e_active else 0.0
    output_should_stop_e2e = sm['modelV2'].action.shouldStop if e2e_active else False
    if e2e_active:
      custom_e2e_runway_comfort_a_target = get_e2e_runway_comfort_accel(
        v_ego, output_a_target_e2e, accel_coast, sm['modelV2'], e2e_active, prev_output_a_target,
        reset_state=reset_state,
        force_slow_decel=force_slow_decel,
        brake_pressed=sm['carState'].brakePressed,
        gas_pressed=sm['carState'].gasPressed,
        engage_stop_bootstrap_active=custom_engage_stop_bootstrap_active,
        has_radar_lead=has_radar_lead,
        dt=self.dt,
        traction_risk=traction_risk,
      )
      custom_e2e_close_stop_a_target, custom_close_stop_should_stop, self.e2e_close_stop_settle_active = get_e2e_close_stop_settle(
        v_ego, output_a_target_e2e, sm['modelV2'], sm['radarState'], e2e_active,
        active=self.e2e_close_stop_settle_active,
        reset_state=reset_state,
        force_slow_decel=force_slow_decel,
        brake_pressed=sm['carState'].brakePressed,
        gas_pressed=sm['carState'].gasPressed,
      )
    else:
      custom_e2e_runway_comfort_a_target = output_a_target_e2e
      custom_e2e_close_stop_a_target = output_a_target_e2e
      custom_close_stop_should_stop = False
      self.e2e_close_stop_settle_active = False
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

    model_stop_protection_active = False
    if e2e_active:
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
        traction_risk=traction_risk,
      )
    else:
      e2e_runway_positive_accel_cap = ACCEL_MAX
      custom_e2e_stop_approach_a_target = 0.0
    e2e_stop_approach_a_target = custom_e2e_stop_approach_a_target

    primary_behavior_progress_allowed = bool(
      primary_behavior_lead is not None and primary_lead_context.lead_progress_allowed and
      not primary_lead_context.alternate_threat_active
    )
    behavior_lead_track_id = int(getattr(primary_behavior_lead, "radarTrackId", -2)) if primary_behavior_lead is not None else -2
    behavior_lead_d_rel = get_lead_d_rel(primary_behavior_lead) if primary_behavior_lead is not None else 0.0
    behavior_lead_v_lead, behavior_lead_v_rel, behavior_fast_motion = (
      get_planner_lead_motion_values(primary_behavior_lead, v_ego, use_fast_lead_motion_evidence)
      if primary_behavior_lead is not None else (0.0, 0.0, FastLeadMotionEvidence())
    )
    behavior_lead_opening = behavior_fast_motion.opening() if use_fast_lead_motion_evidence else behavior_lead_v_rel > 0.0
    behavior_lead_moving = behavior_fast_motion.moving() if use_fast_lead_motion_evidence else behavior_lead_v_lead >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
    behavior_lead_a = get_lead_a_lead(primary_behavior_lead) if primary_behavior_lead is not None else 0.0
    behavior_lead_tau = get_lead_a_tau(primary_behavior_lead) if primary_behavior_lead is not None else 0.0
    behavior_lead_model_prob = get_lead_model_prob(primary_behavior_lead) if primary_behavior_lead is not None else 0.0
    physical_lead_d_rel = get_lead_d_rel(primary_physical_lead) if primary_physical_lead is not None else 0.0
    physical_lead_v_lead, physical_lead_v_rel, _physical_fast_motion = (
      get_planner_lead_motion_values(primary_physical_lead, v_ego, use_fast_lead_motion_evidence)
      if primary_physical_lead is not None else (0.0, 0.0, FastLeadMotionEvidence())
    )
    physical_lead_opening = _physical_fast_motion.opening() if use_fast_lead_motion_evidence else physical_lead_v_rel > 0.0
    physical_lead_moving = _physical_fast_motion.moving() if use_fast_lead_motion_evidence else physical_lead_v_lead >= CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED
    physical_lead_a = get_lead_a_lead(primary_physical_lead) if primary_physical_lead is not None else 0.0
    physical_lead_model_prob = get_lead_model_prob(primary_physical_lead) if primary_physical_lead is not None else 0.0
    physical_lead_y_rel = get_lead_y_rel(primary_physical_lead) if primary_physical_lead is not None else 0.0
    if primary_behavior_progress_allowed and should_arm_stopped_lead_gap_fill(
      v_ego, behavior_lead_d_rel, behavior_lead_v_lead, behavior_lead_model_prob,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=behavior_lead_a,
    ):
      self.stopped_lead_gap_fill_timer = STOPPED_LEAD_GAP_FILL_ARM_TIME
      self.stopped_lead_gap_fill_track_id = behavior_lead_track_id
      self.stopped_lead_gap_fill_d_rel = behavior_lead_d_rel
      self.stopped_lead_gap_fill_v_lead = behavior_lead_v_lead
    elif not (primary_behavior_progress_allowed and stopped_lead_gap_fill_lead_continuous(
      behavior_lead_track_id, self.stopped_lead_gap_fill_track_id, behavior_lead_d_rel, self.stopped_lead_gap_fill_d_rel,
      behavior_lead_v_lead, self.stopped_lead_gap_fill_v_lead,
    )):
      self.stopped_lead_gap_fill_timer = 0.0
      self.stopped_lead_gap_fill_track_id = -2
      self.stopped_lead_gap_fill_d_rel = 0.0
      self.stopped_lead_gap_fill_v_lead = 0.0
    else:
      self.stopped_lead_gap_fill_timer = max(0.0, self.stopped_lead_gap_fill_timer - self.dt)
      self.stopped_lead_gap_fill_d_rel = behavior_lead_d_rel
      self.stopped_lead_gap_fill_v_lead = behavior_lead_v_lead

    model_predicted_v_lead, model_predicted_gap_opening = (
      get_model_lead_pullaway(sm['modelV2'], primary_behavior_lead, v_ego)
      if primary_behavior_progress_allowed and not acc_mode_requested else (0.0, 0.0)
    )
    lead_gap_excess = behavior_lead_d_rel - get_lead_stop_presentation_distance(
      v_ego, behavior_lead_v_lead, behavior_lead_a, behavior_lead_model_prob
    ) if primary_behavior_lead is not None else 0.0
    lead_follow_gap_excess = behavior_lead_d_rel - get_desired_follow_distance(
      v_ego, behavior_lead_v_lead, get_T_FOLLOW(sm['selfdriveState'].personality)
    ) if primary_behavior_lead is not None else 0.0
    radar_predicted_v_lead, radar_predicted_gap_opening = (
      get_predicted_lead_pullaway(behavior_lead_v_lead, behavior_lead_a, behavior_lead_tau)
      if primary_behavior_progress_allowed else (0.0, 0.0)
    )
    radar_predicted_pullaway = primary_behavior_progress_allowed and behavior_lead_a >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_ACCEL and has_predicted_lead_pullaway(
      lead_gap_excess, radar_predicted_v_lead, radar_predicted_gap_opening
    )
    model_predicted_pullaway = primary_behavior_progress_allowed and not creep_to_stop_gap_blocked(
      v_ego, behavior_lead_d_rel, behavior_lead_v_lead, behavior_lead_model_prob,
      sm['carState'].brakePressed, sm['carState'].gasPressed, force_slow_decel or reset_state,
      behavior_lead_a,
    ) and has_predicted_lead_pullaway(
      lead_gap_excess, model_predicted_v_lead, model_predicted_gap_opening,
    )
    creep_pullaway_release = primary_behavior_progress_allowed and (
      behavior_lead_moving or radar_predicted_pullaway or model_predicted_pullaway
    )
    confirmed_creep_pullaway_launch = primary_behavior_progress_allowed and creep_pullaway_release and (radar_predicted_pullaway or model_predicted_pullaway)
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
    scc_independent_stop_threat = bool(
      mode_resolution is not None and
      mode_resolution.requested_mode == LongitudinalMode.SCC and
      getattr(mode_resolution.scc_evidence, "independent_of_lead", False) and
      mode_resolution.scc_evidence.tier in (SccEvidenceTier.STOP, SccEvidenceTier.URGENT_STOP)
    )
    lead_pullaway_independent_stop_threat = bool(
      scc_independent_stop_threat or (
        e2e_active and not defer_e2e_to_stopped_lead_mpc and
        not primary_behavior_progress_allowed and
        (output_should_stop_e2e or custom_e2e_stop_approach_a_target < 0.0)
      )
    )
    # Excess Gap Closure should recover only gap beyond the active follow target,
    # while preserving the stop-target excess safety cushion used by launch/creep.
    lead_pullaway_gap_excess = max(0.0, min(lead_gap_excess, lead_follow_gap_excess))
    lead_pullaway_tracker = getattr(self, "lead_pullaway_intent_tracker", None)
    if lead_pullaway_tracker is None:
      lead_pullaway_tracker = LeadPullawayIntentTracker()
      self.lead_pullaway_intent_tracker = lead_pullaway_tracker
    lead_pullaway_opening = behavior_lead_opening if primary_behavior_lead is not None else physical_lead_opening
    lead_pullaway_moving = behavior_lead_moving if primary_behavior_lead is not None else physical_lead_moving
    lead_pullaway_accel = behavior_lead_a if primary_behavior_lead is not None else physical_lead_a
    self.lead_pullaway_intent = lead_pullaway_tracker.update(
      v_ego=v_ego,
      behavior_lead=primary_behavior_lead,
      primary_lead_context=primary_lead_context,
      lead_gap_excess=lead_pullaway_gap_excess,
      predicted_gap_opening=lead_pullaway_predicted_gap_opening,
      lead_opening=lead_pullaway_opening,
      lead_moving=lead_pullaway_moving,
      lead_accel=lead_pullaway_accel,
      independent_stop_threat=lead_pullaway_independent_stop_threat,
      alternate_lead_threat_active=primary_lead_context.alternate_threat_active,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel,
      reset_state=reset_state or not custom_stack_active,
      dt=self.dt,
    )
    stop_release_lead_confirmed = bool(
      getattr(self.lead_pullaway_intent, "active", False) or
      lead_confirmed_stop_release(
        primary_lead_context,
        primary_behavior_lead,
        lead_opening=behavior_lead_opening,
        lead_moving=behavior_lead_moving,
        lead_accel=behavior_lead_a,
        predicted_gap_opening=lead_pullaway_predicted_gap_opening,
        independent_stop_threat=lead_pullaway_independent_stop_threat,
        brake_pressed=sm['carState'].brakePressed,
        gas_pressed=sm['carState'].gasPressed,
        force_slow_decel=force_slow_decel,
      )
    )
    scc_stop_evidence_active = bool(
      mode_resolution is not None and
      getattr(mode_resolution.scc_evidence, "tier", SccEvidenceTier.NONE) in (SccEvidenceTier.STOP, SccEvidenceTier.URGENT_STOP)
    )
    stop_release_stop_evidence_active = bool(
      self.output_should_stop or output_should_stop_mpc or output_should_stop_e2e or custom_close_stop_should_stop or
      custom_e2e_stop_approach_a_target < 0.0 or scc_stop_evidence_active or
      (not acc_mode_requested and has_model_stop_context(sm['modelV2']))
    )
    stop_release_guard_tracker = getattr(self, "stop_release_guard_tracker", None)
    if stop_release_guard_tracker is None:
      stop_release_guard_tracker = StopReleaseGuardTracker()
      self.stop_release_guard_tracker = stop_release_guard_tracker
    self.stop_release_guard_state = stop_release_guard_tracker.update(
      v_ego=v_ego,
      standstill=sm['carState'].standstill,
      stop_evidence_active=stop_release_stop_evidence_active,
      lead_confirmed_release=stop_release_lead_confirmed,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      dt=self.dt,
    )
    prev_creep_to_stop_gap_active = self.creep_to_stop_gap_active
    self.creep_to_stop_gap_active, creep_a_target = get_creep_to_stop_gap_accel(
      v_ego, behavior_lead_d_rel, behavior_lead_v_lead, behavior_lead_model_prob,
      self.creep_to_stop_gap_active and not reset_state,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=behavior_lead_a,
      a_lead_tau=behavior_lead_tau,
      model_predicted_v_lead=model_predicted_v_lead,
      model_predicted_gap_opening=model_predicted_gap_opening,
    ) if primary_behavior_progress_allowed else (False, 0.0)
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
      v_ego, behavior_lead_d_rel, behavior_lead_v_lead, behavior_lead_model_prob,
      primary_behavior_progress_allowed and self.stopped_lead_gap_fill_timer > 0.0,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      force_slow_decel=force_slow_decel or reset_state,
      a_lead=behavior_lead_a,
    ) if primary_behavior_progress_allowed else (False, 0.0)
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
    if primary_behavior_progress_allowed and not self.output_should_stop and not reset_state and self.mpc.source != LongitudinalPlanSource.e2e and \
       self.source == custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise:
      recovery_a_min = get_lead_accel_recovery_a_min(
        v_ego, behavior_lead_v_lead, behavior_lead_d_rel, behavior_lead_a,
        get_T_FOLLOW(sm['selfdriveState'].personality)
      )
      if v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO:
        recovery_a_min = min(recovery_a_min, CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_BASE_MAX)
      custom_lead_accel_recovery_a_target = recovery_a_min

    if primary_physical_lead is not None and not sm['carState'].brakePressed and not sm['carState'].gasPressed and not force_slow_decel and not reset_state:
      self.creep_stop_hold_released = should_release_creep_stop_hold(
        self.creep_stop_hold_released, v_ego, physical_lead_d_rel, physical_lead_v_lead,
        physical_lead_a, model_predicted_pullaway, physical_lead_model_prob,
      )
    else:
      self.creep_stop_hold_released = False
    custom_creep_hold_a_target = None
    if primary_physical_lead is not None and not (self.creep_to_stop_gap_active and creep_a_target > 0.0) and should_hold_creep_to_stop_gap(
      v_ego, physical_lead_d_rel, physical_lead_v_lead, physical_lead_a, model_predicted_pullaway,
      self.creep_stop_hold_released, physical_lead_model_prob,
    ):
      custom_creep_hold_a_target = min(CREEP_TO_STOP_GAP_ACCEL_MIN, get_creep_to_stop_gap_hold_accel(v_ego, physical_lead_d_rel))

    custom_stopped_stop_gap_guard_a_target = None
    stopped_stop_gap_guard_group = ""
    custom_moving_stop_guard_a_target = None
    custom_moving_stop_guard_debug = None
    custom_routine_lead_approach_a_target = None
    custom_routine_lead_approach_debug = None
    stopped_lead_moving_rebound_timer = max(0.0, float(getattr(self, "stopped_lead_moving_rebound_timer", 0.0)) - self.dt)
    if primary_physical_lead is not None and physical_lead_v_lead > MOVING_LEAD_STOP_GAP_GUARD_MIN_V_LEAD:
      stopped_lead_moving_rebound_timer = STOPPED_LEAD_MOVING_REBOUND_HOLD_TIME
    elif primary_physical_lead is None or reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed:
      stopped_lead_moving_rebound_timer = 0.0
    self.stopped_lead_moving_rebound_timer = stopped_lead_moving_rebound_timer
    if primary_physical_lead is not None:
      if not defer_e2e_to_stopped_lead_mpc and self.mpc.source != LongitudinalPlanSource.e2e:
        stop_gap_guard_a_target = get_stopped_lead_stop_gap_guard_accel(
          v_ego, physical_lead_d_rel, physical_lead_v_lead, physical_lead_a, physical_lead_model_prob,
        )
        if not should_allow_stopped_lead_stop_gap_guard(
          v_ego, physical_lead_d_rel, physical_lead_v_lead, is_lane_change_active(sm['modelV2']) if e2e_active else False,
        ):
          stop_gap_guard_a_target = None
        if stop_gap_guard_a_target is not None:
          if (
            stopped_lead_moving_rebound_timer > 0.0 and
            stop_gap_guard_a_target > prev_output_a_target and
            physical_lead_d_rel > STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS
          ):
            stop_gap_guard_a_target = min(
              stop_gap_guard_a_target,
              prev_output_a_target + STOPPED_LEAD_STOP_GAP_GUARD_REBOUND_JERK * self.dt,
            )
          custom_stopped_stop_gap_guard_a_target = stop_gap_guard_a_target
          stopped_stop_gap_guard_group = "lead_stop_approach_slew" if stopped_lead_moving_rebound_timer > 0.0 else ""

      moving_stop_guard_a_target, moving_stop_guard_debug = get_moving_lead_stop_gap_guard_accel(
        v_ego, physical_lead_d_rel, physical_lead_v_lead, physical_lead_a, physical_lead_y_rel,
        get_T_FOLLOW(sm['selfdriveState'].personality), a_ego=sm['carState'].aEgo,
        prev_a_target=prev_output_a_target, dt=self.dt, return_debug=True,
        budget=get_comfort_budget(sm['selfdriveState'].personality),
      )
      if moving_stop_guard_a_target is not None:
        custom_moving_stop_guard_a_target = moving_stop_guard_a_target
        custom_moving_stop_guard_debug = moving_stop_guard_debug
      # Emit routine lead approach floor seed when routine is active, valid, and non-urgent.
      # This allows routine comfort shaping to relax non-urgent baseline targets at seed level.
      if moving_stop_guard_debug is not None:
        routine_active = bool(moving_stop_guard_debug.get("routine_lead_approach_active", False))
        routine_urgent = bool(moving_stop_guard_debug.get("routine_lead_approach_urgent", False))
        routine_ramped = float(moving_stop_guard_debug.get("routine_lead_ramped_a_target", 0.0))
        routine_can_own = bool(moving_stop_guard_debug.get("routine_lead_can_own_nonurgent_shape", False))
        routine_safety_relevant = bool(moving_stop_guard_debug.get("routine_lead_existing_target_safety_relevant", False))
        valid_approach = is_valid_routine_lead_approach(
          primary_lead_context=primary_lead_context,
          brake_pressed=sm['carState'].brakePressed,
          gas_pressed=sm['carState'].gasPressed,
          force_slow_decel=force_slow_decel,
          independent_stop_threat=lead_pullaway_independent_stop_threat,
          alternate_lead_threat_active=bool(getattr(primary_lead_context, "alternate_threat_active", False)),
        )
        if routine_active and not routine_urgent and routine_can_own and not routine_safety_relevant and valid_approach:
          custom_routine_lead_approach_a_target = routine_ramped
          custom_routine_lead_approach_debug = {k: v for k, v in moving_stop_guard_debug.items() if k.startswith("routine_lead_")}

    custom_lead_stop_approach_slewed_a_target = None
    custom_lead_stop_approach_base_a_target = None
    if primary_physical_lead is not None and not active_mpc_lead_lateral_exit and not reset_state and not sm['carState'].brakePressed and not sm['carState'].gasPressed:
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
        v_ego, physical_lead_d_rel, physical_lead_v_lead, physical_lead_a,
        prev_output_a_target, lead_stop_approach_base_a_target, self.dt, traction_risk=traction_risk,
      )

    self.previous_lead_loss_status = lead_loss_guard_lead is not None
    self.previous_lead_loss_d_rel = float(lead_loss_guard_lead.dRel) if lead_loss_guard_lead is not None else 0.0
    self.previous_lead_loss_model_prob = float(lead_loss_guard_lead.modelProb) if lead_loss_guard_lead is not None else 0.0

    continuing_creep_pullaway_launch = primary_behavior_progress_allowed and prev_output_a_target >= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN and \
      v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO and \
      CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS <= lead_gap_excess <= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MAX_EXCESS and \
      behavior_lead_a >= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MIN_LEAD_ACCEL and \
      confirmed_creep_pullaway_stop_release
    strong_opening_creep_pullaway_launch = primary_behavior_progress_allowed and \
      v_ego < CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_OPENING_MAX_V_EGO and \
      CREEP_TO_STOP_GAP_START_EXCESS <= lead_gap_excess <= CREEP_TO_STOP_GAP_MAX_EXCESS and \
      behavior_lead_opening and behavior_lead_a >= CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_CONTINUE_MIN_LEAD_ACCEL and \
      self.creep_to_stop_gap_active and creep_a_target > 0.0 and \
      confirmed_creep_pullaway_stop_release
    creep_pullaway_launch = primary_behavior_progress_allowed and (not self.output_should_stop or confirmed_creep_pullaway_stop_release) and \
      not sm['carState'].brakePressed and not sm['carState'].gasPressed and \
      not force_slow_decel and not reset_state and \
      (v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM or continuing_creep_pullaway_launch or strong_opening_creep_pullaway_launch) and \
      behavior_lead_model_prob >= CREEP_TO_STOP_GAP_MODEL_LEAD_MIN_PROB and \
      ((CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_MIN_EXCESS <= lead_gap_excess <= CREEP_TO_STOP_GAP_MAX_EXCESS and
        self.creep_to_stop_gap_active and creep_a_target > 0.0) or continuing_creep_pullaway_launch or strong_opening_creep_pullaway_launch) and \
      confirmed_creep_pullaway_stop_release
    custom_creep_pullaway_launch_floor = None
    custom_creep_pullaway_launch_cap = None
    lead_pullaway_crawl_cap_released = primary_behavior_progress_allowed and creep_pullaway_release and \
      lead_pullaway_runway_excess >= CREEP_TO_STOP_GAP_START_EXCESS and behavior_lead_opening and behavior_lead_a >= 0.0
    lead_pullaway_crawl_cap_released_by_runway = bool(
      getattr(self.lead_pullaway_intent, "early_authority", False) and
      getattr(self.lead_pullaway_intent, "phase", LeadPullawayPhase.HOLD) not in (LeadPullawayPhase.PULSE, LeadPullawayPhase.GAP_CLOSURE) and
      _finite_float(getattr(self.lead_pullaway_intent, "cooldown_timer", 0.0)) <= 0.0 and
      _finite_float(getattr(self.lead_pullaway_intent, "safe_accel_cap", 0.0)) >= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX and
      _finite_float(getattr(self.lead_pullaway_intent, "runway_margin_now", 0.0)) >= 0.0 and
      _finite_float(getattr(self.lead_pullaway_intent, "runway_margin", 0.0)) >= 0.0
    )
    lead_pullaway_crawl_cap_released = bool(lead_pullaway_crawl_cap_released or lead_pullaway_crawl_cap_released_by_runway)
    if creep_pullaway_launch:
      custom_creep_to_stop_gap_accel_max = None
      launch_accel_max = get_creep_pullaway_launch_accel_max(lead_gap_excess, lead_pullaway_predicted_gap_opening)
      if not lead_pullaway_crawl_cap_released:
        crawl_accel_max = get_lead_crawl_accel_max(
          behavior_lead_d_rel, v_ego, behavior_lead_v_lead, behavior_lead_a,
          get_T_FOLLOW(sm['selfdriveState'].personality),
        )
        crawl_accel_max = min(crawl_accel_max, max(0.0, LEAD_CRAWL_ACCEL_LIMIT - LEAD_PULLAWAY_CRAWL_CAP_JERK_BUFFER))
        launch_accel_max = min(launch_accel_max, float(crawl_accel_max))
      custom_creep_pullaway_launch_floor = CREEP_TO_STOP_GAP_PULLAWAY_LAUNCH_ACCEL_MIN
      if bool(getattr(self.lead_pullaway_intent, "active", False)):
        custom_creep_pullaway_launch_floor = min(
          custom_creep_pullaway_launch_floor,
          max(0.0, _finite_float(getattr(self.lead_pullaway_intent, "a_floor", 0.0))),
        )
      custom_creep_pullaway_launch_cap = launch_accel_max

    has_lead = sm['radarState'].leadOne.status or sm['radarState'].leadTwo.status
    cruise_coast_applied = False
    cruise_coast_a_target = output_a_target
    curve_load_comfort_bypass = bool(
      reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed or
      self.output_should_stop or self.fcw
    )
    accel_clip = apply_curve_load_comfort_accel_limit(
      v_ego, steer_angle_without_offset, accel_clip, self.CP,
      control_calculation_hardening=self.control_calculation_hardening,
      vehicle_model=self.VM, roll=live_params.roll,
      accurate_lateral_accel=accurate_lateral_accel,
      urgent_bypass=curve_load_comfort_bypass,
    )

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
      lead_mpc_allowed=not active_mpc_lead_lateral_exit,
    )
    source_stability_v_ego = None if (
      reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed
    ) else v_ego
    decision_layer_applies_before_stack = should_enable_longitudinal_decision_layer(stack_resolution) and \
      getattr(stack_resolution, "resolved_stack", "") != CUSTOM_V2
    custom_v2_policy_resolves_decision = (
      getattr(stack_resolution, "resolved_stack", "") == CUSTOM_V2 and
      bool(sm['selfdriveState'].enabled) and
      not bool(getattr(self, "custom_v2_fault_latched", False))
    )
    self.longitudinal_decision = resolve_longitudinal_decision(
      enabled=decision_layer_applies_before_stack,
      candidates=self.longitudinal_decision_candidates,
      fallback_v_target=v_cruise,
      fallback_a_target=legacy_a_target,
      fallback_should_stop=legacy_should_stop,
      accel_limits=(accel_clip[0], accel_clip[1]),
      arbiter=self.longitudinal_arbiter,
      v_ego=source_stability_v_ego,
      reset_when_disabled=not custom_v2_policy_resolves_decision,
    )
    self.longitudinal_decision_telemetry = None
    if decision_layer_applies_before_stack and self.longitudinal_decision.enabled:
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
    low_speed_pullaway_accel_step = primary_behavior_progress_allowed and not sm['carState'].brakePressed and not sm['carState'].gasPressed and \
      not force_slow_decel and not reset_state and v_ego < CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_MAX_V_EGO and \
      (prev_output_a_target > 0.0 or output_a_target > 0.0)
    prev_pullaway_accel_step_handoff_timer = max(0.0, float(getattr(self, "pullaway_accel_step_handoff_timer", 0.0)) - self.dt)
    pullaway_accel_step_active = bool(limit_creep_pullaway_accel_step or low_speed_pullaway_accel_step)
    pullaway_accel_step_handoff_active = bool(
      not pullaway_accel_step_active and
      prev_pullaway_accel_step_handoff_timer > 0.0 and
      primary_behavior_progress_allowed and
      not sm['carState'].brakePressed and not sm['carState'].gasPressed and
      not force_slow_decel and not reset_state and
      not self.output_should_stop and
      prev_output_a_target > 0.0
    )
    if pullaway_accel_step_active:
      self.pullaway_accel_step_handoff_timer = CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP_HANDOFF_TIME
    elif pullaway_accel_step_handoff_active:
      self.pullaway_accel_step_handoff_timer = prev_pullaway_accel_step_handoff_timer
    else:
      self.pullaway_accel_step_handoff_timer = 0.0
    custom_pullaway_accel_step_floor = None
    custom_pullaway_accel_step_cap = None
    custom_pullaway_accel_step_seed_cap = None
    if pullaway_accel_step_active or pullaway_accel_step_handoff_active:
      if pullaway_accel_step_active and output_a_target > -CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP and not self.output_should_stop:
        custom_pullaway_accel_step_floor = prev_output_a_target - CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP
      custom_pullaway_accel_step_cap = prev_output_a_target + CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_STEP
      if pullaway_accel_step_active:
        custom_pullaway_accel_step_seed_cap = custom_pullaway_accel_step_cap
    if custom_pullaway_accel_step_seed_cap is not None:
      if custom_creep_pullaway_launch_floor is not None:
        custom_creep_pullaway_launch_floor = min(custom_creep_pullaway_launch_floor, custom_pullaway_accel_step_seed_cap)
      if bool(getattr(self.lead_pullaway_intent, "active", False)):
        clamped_lead_pullaway_floor = min(
          max(0.0, _finite_float(getattr(self.lead_pullaway_intent, "a_floor", 0.0))),
          max(0.0, _finite_float(custom_pullaway_accel_step_seed_cap)),
        )
        self.lead_pullaway_intent = replace(
          self.lead_pullaway_intent,
          a_floor=clamped_lead_pullaway_floor,
        )
        lead_pullaway_tracker = getattr(self, "lead_pullaway_intent_tracker", None)
        if lead_pullaway_tracker is not None:
          lead_pullaway_tracker.clamp_active_floor(clamped_lead_pullaway_floor)
    custom_lead_crawl_accel_cap = None
    if primary_physical_lead is not None and not creep_pullaway_launch and not lead_pullaway_crawl_cap_released and v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO:
      custom_lead_crawl_accel_cap = LEAD_CRAWL_ACCEL_LIMIT
    lead_flicker_trackers = getattr(self, "lead_flicker_safety_cap_trackers", None)
    if lead_flicker_trackers is None:
      lead_flicker_trackers = [LeadFlickerSafetyCapTracker(), LeadFlickerSafetyCapTracker()]
      self.lead_flicker_safety_cap_trackers = lead_flicker_trackers
    lead_flicker_safety_cap_states = [
      tracker.update(
        lead, v_ego, self.dt,
        reset_state=reset_state,
        force_slow_decel=force_slow_decel,
        gas_pressed=sm['carState'].gasPressed,
        brake_pressed=sm['carState'].brakePressed,
      )
      for tracker, lead in zip(lead_flicker_trackers, (sm['radarState'].leadOne, sm['radarState'].leadTwo), strict=False)
    ]
    shadow_suppress_accel_active = any(
      bool(getattr(state, "shadow", False)) and (
        float(getattr(state, "risk_score", 0.0)) >= 0.35 or float(getattr(state, "required_decel", 0.0)) >= LEAD_FLICKER_FAR_REQUIRED_DECEL_MIN
      )
      for state in getattr(primary_lead_context, "states", ())
    )
    lead_flicker_safety_cap_active = any(state.active for state in lead_flicker_safety_cap_states) or shadow_suppress_accel_active
    lead_lateral_progress_block_timer = max(0.0, float(getattr(self, "lead_lateral_progress_block_timer", 0.0)) - self.dt)
    lateral_progress_state = primary_behavior_state or primary_physical_state
    if lateral_progress_state is not None and abs(float(lateral_progress_state.path_y_rel)) >= LEAD_LATERAL_PROGRESS_BLOCK_Y:
      lead_lateral_progress_block_timer = LEAD_LATERAL_PROGRESS_HOLD_TIME
    elif lateral_progress_state is None or reset_state or force_slow_decel or sm['carState'].brakePressed or sm['carState'].gasPressed:
      lead_lateral_progress_block_timer = 0.0
    self.lead_lateral_progress_block_timer = lead_lateral_progress_block_timer
    lead_lateral_progress_blocked = lead_lateral_progress_block_timer > 0.0
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.planner_seed_candidates = []
    if lead_flicker_safety_cap_active:
      lead_flicker_speedup_cap_candidate = build_planner_seed_accel_candidate(
        self, "lead_flicker_speedup_cap", LEAD_FLICKER_SPEEDUP_CAP_A_TARGET_MAX, has_lead,
        LEAD_FLICKER_SPEEDUP_CAP_REASON, accel_clip, should_stop=False, force=True,
      )
      if lead_flicker_speedup_cap_candidate is not None:
        self.planner_seed_candidates.append(lead_flicker_speedup_cap_candidate)
    planner_seed_mpc_candidate_allowed = not (
      planner_seed_mpc_lead_lateral_exit and planner_seed_mpc_a_target is not None and planner_seed_mpc_a_target < 0.0
    )
    if planner_seed_mpc_a_target is not None and planner_seed_mpc_candidate_allowed:
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
    self.planner_seed_candidates.extend(build_no_lead_stop_seed_candidates(
      self, has_lead, accel_clip,
      engage_bootstrap_active=custom_engage_stop_bootstrap_active,
      engage_bootstrap_a_target=output_a_target_e2e,
      engage_bootstrap_should_stop=output_should_stop_e2e,
    ))
    creep_candidate_blocked_by_pullaway_floor = (
      custom_pullaway_accel_step_floor is not None and
      custom_creep_to_stop_gap_a_target is not None and
      custom_creep_to_stop_gap_a_target < custom_pullaway_accel_step_floor and
      not bool(custom_creep_to_stop_gap_should_stop)
    )
    self.planner_seed_candidates.extend(build_stopped_lead_seed_candidates(
      self, has_lead, accel_clip,
      stopped_stop_gap_guard_a_target=custom_stopped_stop_gap_guard_a_target,
      stopped_stop_gap_guard_group=stopped_stop_gap_guard_group,
      creep_to_stop_gap_a_target=None if creep_candidate_blocked_by_pullaway_floor else custom_creep_to_stop_gap_a_target,
      creep_to_stop_gap_should_stop=custom_creep_to_stop_gap_should_stop,
      creep_to_stop_gap_selection=custom_creep_to_stop_gap_selection,
      creep_to_stop_gap_accel_max=custom_creep_to_stop_gap_accel_max,
      gap_fill_a_target=custom_gap_fill_a_target,
      gap_fill_should_stop=custom_gap_fill_should_stop,
      gap_fill_selection=custom_gap_fill_selection,
      gap_fill_accel_max=custom_gap_fill_accel_max,
    ))
    pullaway_step_cap_suppressed = bool(
      custom_creep_pullaway_launch_floor is not None and (legacy_should_stop or strong_opening_creep_pullaway_launch)
    )
    low_speed_step_cap_suppressed_by_runway = bool(
      custom_pullaway_accel_step_seed_cap is not None and
      getattr(self.lead_pullaway_intent, "active", False) and
      getattr(self.lead_pullaway_intent, "early_authority", False) and
      _finite_float(getattr(self.lead_pullaway_intent, "a_floor", 0.0)) <= _finite_float(custom_pullaway_accel_step_seed_cap) + 1e-3 and
      _finite_float(getattr(self.lead_pullaway_intent, "safe_accel_cap", 0.0)) >= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX and
      _finite_float(getattr(self.lead_pullaway_intent, "runway_margin", 0.0)) >= 0.0
    )
    pullaway_step_cap_suppressed = bool(pullaway_step_cap_suppressed or low_speed_step_cap_suppressed_by_runway)
    self.lead_pullaway_intent = replace(
      getattr(self, "lead_pullaway_intent", LeadPullawayIntent()),
      crawl_cap_released_by_runway=lead_pullaway_crawl_cap_released_by_runway,
      low_speed_step_cap_suppressed_by_runway=low_speed_step_cap_suppressed_by_runway,
    )
    self.planner_seed_candidates.extend(build_lead_pullaway_seed_candidates(
      self, has_lead, accel_clip,
      creep_pullaway_launch_floor=custom_creep_pullaway_launch_floor,
      creep_pullaway_launch_cap=custom_creep_pullaway_launch_cap,
      pullaway_accel_step_floor=custom_pullaway_accel_step_floor,
      pullaway_accel_step_cap=custom_pullaway_accel_step_seed_cap,
      pullaway_step_cap_suppressed=pullaway_step_cap_suppressed,
    ))
    self.planner_seed_candidates.extend(build_lead_pullaway_intent_seed_candidates(
      self, has_lead, accel_clip, getattr(self, "lead_pullaway_intent", LeadPullawayIntent()),
    ))
    self.planner_seed_candidates.extend(build_stopped_lead_seed_candidates(
      self, has_lead, accel_clip,
      lead_crawl_accel_cap=custom_lead_crawl_accel_cap,
      creep_hold_a_target=custom_creep_hold_a_target,
    ))
    self.planner_seed_candidates.extend(build_moving_lead_seed_candidates(
      self, has_lead, accel_clip,
      moving_stop_guard_a_target=custom_moving_stop_guard_a_target,
      moving_stop_guard_debug=custom_moving_stop_guard_debug,
      lead_accel_recovery_a_target=custom_lead_accel_recovery_a_target,
      lead_stop_approach_slewed_a_target=custom_lead_stop_approach_slewed_a_target,
      lead_stop_approach_base_a_target=custom_lead_stop_approach_base_a_target,
      routine_lead_approach_a_target=custom_routine_lead_approach_a_target,
      routine_lead_approach_debug=custom_routine_lead_approach_debug,
    ))
    self.planner_seed_candidates.extend(build_no_lead_stop_seed_candidates(
      self, has_lead, accel_clip,
      e2e_close_stop_active=self.e2e_close_stop_settle_active,
      e2e_close_stop_a_target=custom_e2e_close_stop_a_target,
      e2e_close_stop_should_stop=custom_close_stop_should_stop,
      e2e_runway_comfort_a_target=custom_e2e_runway_comfort_output_a_target,
    ))
    self.planner_seed_candidates.extend(build_lead_loss_seed_candidates(
      self, has_lead, accel_clip, lead_loss_e2e_guard_a_target=custom_lead_loss_e2e_guard_a_target,
    ))
    self.planner_seed_candidates.extend(build_no_lead_stop_seed_candidates(
      self, has_lead, accel_clip,
      e2e_stop_approach_a_target=custom_e2e_stop_approach_a_target,
      e2e_runway_positive_accel_cap=e2e_runway_positive_accel_cap,
    ))
    self.planner_seed_candidates.extend(build_cruise_coast_seed_candidates(
      self, has_lead, accel_clip,
      active=should_apply_cruise_coast_overspeed(
        reset_state, force_slow_decel, e2e_active, has_lead, self.output_should_stop, self.source
      ),
      a_target=apply_cruise_coast_overspeed(v_ego, v_cruise, cruise_coast_accel, self.output_a_target),
    ))
    active_scc = getattr(self, "active_scc", None) or getattr(self, "scc", None)
    active_scc_vision = getattr(active_scc, "vision", None)
    active_scc_map = getattr(active_scc, "map", None)
    active_sla = getattr(self, "active_sla", None) or getattr(self, "sla", None)
    osm_traffic_control_prior = getattr(self, "osm_traffic_control_prior", None)
    if acc_mode_requested:
      custom_v2_curve_active, custom_v2_curve_a_target = False, 0.0
      speed_limit_active_for_scene = False
      speed_limit_v_target_for_scene = 0.0
      speed_limit_a_target_for_scene = 0.0
      map_caution_active_for_scene = False
      map_caution_a_target_for_scene = 0.0
    else:
      custom_v2_curve_active, custom_v2_curve_a_target = (
        get_custom_v2_curve_scene_target(active_scc_vision, active_scc_map) if scc_curve_scene_allowed else (False, 0.0)
      )
      speed_limit_active_for_scene = bool(getattr(active_sla, "is_active", False))
      speed_limit_v_target_for_scene = float(getattr(active_sla, "output_v_target", 0.0))
      speed_limit_a_target_for_scene = float(getattr(active_sla, "output_a_target", 0.0))
      map_caution_active_for_scene = bool(getattr(osm_traffic_control_prior, "active", False))
      map_caution_a_target_for_scene = float(getattr(osm_traffic_control_prior, "output_a_target", 0.0))
    scene_lead_state = primary_behavior_state if primary_behavior_state is not None else primary_physical_state
    scene_has_lead = bool(primary_lead_context.has_physical_lead)
    if primary_behavior_state is not None and primary_behavior_lead is not None:
      scene_lead_v = behavior_lead_v_lead
      scene_lead_v_rel = behavior_lead_v_rel
    elif primary_physical_state is not None and primary_physical_lead is not None:
      scene_lead_v = physical_lead_v_lead
      scene_lead_v_rel = physical_lead_v_rel
    else:
      scene_lead_v = 0.0
      scene_lead_v_rel = 0.0
    scene_lead_y_rel = float(scene_lead_state.path_y_rel) if scene_lead_state is not None else 0.0
    scene_lead_gap_excess = float(lead_gap_excess) if primary_behavior_lead is not None else 0.0
    scene_lead_follow_gap_excess = float(lead_follow_gap_excess) if primary_behavior_lead is not None else 0.0
    scene_lead_opening_prediction = bool(primary_behavior_progress_allowed and (radar_predicted_pullaway or model_predicted_pullaway))
    scene_lead_confirmed_pullaway = bool(
      primary_behavior_progress_allowed and creep_pullaway_release and
      (behavior_lead_opening or v_ego < CREEP_TO_STOP_GAP_MAX_V_EGO_ARM)
    )
    lead_pullaway_debug = lead_pullaway_intent_debug(getattr(self, "lead_pullaway_intent", LeadPullawayIntent()))
    self.custom_v2_scene = CustomV2Scene(
      v_ego=float(v_ego),
      v_cruise=float(v_cruise),
      personality=getattr(sm['selfdriveState'], "personality", log.LongitudinalPersonality.standard),
      a_ego=float(sm['carState'].aEgo),
      accel_coast=float(cruise_coast_accel),
      force_slow_decel=bool(force_slow_decel),
      brake_pressed=bool(sm['carState'].brakePressed),
      gas_pressed=bool(sm['carState'].gasPressed),
      has_lead=scene_has_lead,
      lead_v=scene_lead_v,
      lead_v_rel=scene_lead_v_rel,
      lead_y_rel=scene_lead_y_rel,
      lead_gap_excess=scene_lead_gap_excess,
      lead_follow_gap_excess=scene_lead_follow_gap_excess,
      lead_lateral_progress_blocked=bool(lead_lateral_progress_blocked),
      lead_progress_allowed=bool(primary_behavior_progress_allowed),
      lead_opening_prediction=scene_lead_opening_prediction,
      lead_confirmed_pullaway=scene_lead_confirmed_pullaway,
      fast_lead_motion_opening=bool(behavior_lead_opening),
      fast_lead_motion_moving=bool(behavior_lead_moving),
      primary_physical_lead_idx=-1 if primary_lead_context.physical_idx is None else int(primary_lead_context.physical_idx),
      primary_behavior_lead_idx=-1 if primary_lead_context.behavior_idx is None else int(primary_lead_context.behavior_idx),
      primary_lead_reason=str(primary_lead_context.reason),
      primary_lead_authority="" if scene_lead_state is None else str(scene_lead_state.authority),
      alternate_lead_threat_active=bool(primary_lead_context.alternate_threat_active),
      shadow_lead_active=bool(primary_lead_context.shadow_active),
      lead_release_blocked_reason=str(primary_lead_context.lead_release_blocked_reason),
      fast_lead_motion_evidence_enabled=bool(use_fast_lead_motion_evidence),
      stop_threat=bool(e2e_active and (custom_e2e_stop_approach_a_target < 0.0 or self.e2e_close_stop_settle_active or output_should_stop_e2e)),
      independent_stop_threat=bool(e2e_active and not scene_has_lead and (custom_e2e_stop_approach_a_target < 0.0 or output_should_stop_e2e)),
      model_should_stop=bool(output_should_stop_e2e),
      model_stop_distance=get_model_stop_distance(sm['modelV2']) if e2e_active else None,
      model_desired_accel=float(output_a_target_e2e),
      speed_limit_active=speed_limit_active_for_scene,
      speed_limit_v_target=speed_limit_v_target_for_scene,
      speed_limit_a_target=speed_limit_a_target_for_scene,
      curve_active=custom_v2_curve_active,
      curve_a_target=custom_v2_curve_a_target,
      map_caution_active=map_caution_active_for_scene,
      map_caution_confirmed=map_caution_active_for_scene,
      map_caution_a_target=map_caution_a_target_for_scene,
      one_pedal_mode=int(getattr(self, "one_pedal_mode", ONE_PEDAL_MODE_OFF)),
      one_pedal_cruise_hold=bool(getattr(self, "one_pedal_cruise_hold_active", False)),
      allow_speed_limit_advisory=e2e_or_scc_direct,
      allow_curve_advisory=scc_curve_scene_allowed,
      allow_map_caution_advisory=e2e_or_scc_direct,
      allow_no_lead_progress=bool(mode_resolution.e2e_like) if mode_resolution is not None else False,
      allow_lead_progress=direct_actuation_mode,
      lead_pullaway_phase=str(lead_pullaway_debug["lead_pullaway_phase"]),
      lead_pullaway_reason=str(lead_pullaway_debug["lead_pullaway_reason"]),
      lead_pullaway_track_id=int(lead_pullaway_debug["lead_pullaway_track_id"]),
      lead_pullaway_pulse_timer=float(lead_pullaway_debug["lead_pullaway_pulse_timer"]),
      lead_pullaway_cooldown_timer=float(lead_pullaway_debug["lead_pullaway_cooldown_timer"]),
      lead_pullaway_gap_excess=float(lead_pullaway_debug["lead_pullaway_gap_excess"]),
      lead_pullaway_predicted_gap_opening=float(lead_pullaway_debug["lead_pullaway_predicted_gap_opening"]),
      lead_pullaway_a_floor=float(lead_pullaway_debug["lead_pullaway_a_floor"]),
      lead_pullaway_rejected_reason=str(lead_pullaway_debug["lead_pullaway_rejected_reason"]),
      lead_pullaway_predicted_gap=float(lead_pullaway_debug["lead_pullaway_predicted_gap"]),
      lead_pullaway_safe_accel_cap=float(lead_pullaway_debug["lead_pullaway_safe_accel_cap"]),
      lead_pullaway_lead_accel_trend=float(lead_pullaway_debug["lead_pullaway_lead_accel_trend"]),
      lead_pullaway_runway_margin=float(lead_pullaway_debug["lead_pullaway_runway_margin"]),
      lead_pullaway_runway_margin_now=float(lead_pullaway_debug["lead_pullaway_runway_margin_now"]),
      lead_pullaway_runway_margin_t=float(lead_pullaway_debug["lead_pullaway_runway_margin_t"]),
      lead_pullaway_runway_creation=float(lead_pullaway_debug["lead_pullaway_runway_creation"]),
      lead_pullaway_lead_created_runway=bool(lead_pullaway_debug["lead_pullaway_lead_created_runway"]),
      lead_pullaway_early_authority=bool(lead_pullaway_debug["lead_pullaway_early_authority"]),
      lead_pullaway_early_authority_reason=str(lead_pullaway_debug["lead_pullaway_early_authority_reason"]),
      lead_pullaway_pulse_floor=float(lead_pullaway_debug["lead_pullaway_pulse_floor"]),
      lead_pullaway_pulse_cap=float(lead_pullaway_debug["lead_pullaway_pulse_cap"]),
      lead_pullaway_coast_required=bool(lead_pullaway_debug["lead_pullaway_coast_required"]),
      lead_pullaway_pulse_capped_by_runway=bool(lead_pullaway_debug["lead_pullaway_pulse_capped_by_runway"]),
      lead_pullaway_crawl_cap_released_by_runway=bool(lead_pullaway_debug["lead_pullaway_crawl_cap_released_by_runway"]),
      lead_pullaway_low_speed_step_cap_suppressed_by_runway=bool(lead_pullaway_debug["lead_pullaway_low_speed_step_cap_suppressed_by_runway"]),
      lead_pullaway_runway_trend=str(lead_pullaway_debug["lead_pullaway_runway_trend"]),
      lead_pullaway_selected_or_rejected_reason=str(lead_pullaway_debug["lead_pullaway_selected_or_rejected_reason"]),
    )
    self.prev_accel_clip = accel_clip
    self.apply_longitudinal_stack_selection(sm, has_lead, tuple(accel_clip))
    if is_custom_stack(getattr(stack_resolution, "resolved_stack", "")):
      if custom_pullaway_accel_step_floor is not None and not self.output_should_stop:
        self.output_a_target = max(self.output_a_target, custom_pullaway_accel_step_floor)
      if custom_pullaway_accel_step_cap is not None and not pullaway_step_cap_suppressed:
        self.output_a_target = min(self.output_a_target, custom_pullaway_accel_step_cap)
      reserve_creep_to_stop_gap = should_reserve_creep_to_stop_gap(
        primary_behavior_progress_allowed, self.output_should_stop, v_ego, behavior_lead_d_rel, behavior_lead_v_lead,
        brake_pressed=sm['carState'].brakePressed, gas_pressed=sm['carState'].gasPressed,
        force_slow_decel=force_slow_decel, reset_state=reset_state,
      )
      if reserve_creep_to_stop_gap:
        self.output_a_target = max(self.output_a_target, CREEP_TO_STOP_GAP_RESERVE_CREEP_ACCEL_FLOOR)
        self.output_should_stop = False
      if getattr(stack_resolution, "resolved_stack", "") == CUSTOM_V2:
        self.output_a_target, self.stop_release_guard_state = apply_stop_release_guard_accel(
          self.output_a_target,
          getattr(self, "stop_release_guard_state", StopReleaseGuardState()),
        )
      self.output_a_target = apply_lead_pullaway_final_output_shaping(
        self.output_a_target,
        getattr(self, "lead_pullaway_intent", None),
        prev_output_a_target,
        self.dt,
        selected_reason=getattr(self, "longitudinal_stack_selected_reason", ""),
      )

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
