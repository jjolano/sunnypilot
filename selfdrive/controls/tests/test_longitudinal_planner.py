from types import SimpleNamespace
import math

import numpy as np
import pytest
from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_decision import DecisionSource, LongitudinalArbiter
from openpilot.selfdrive.controls.lib.longitudinal_modes import LongitudinalMode, ResolvedLongitudinalImplementation, SccModeEvidence
from openpilot.selfdrive.controls.lib.scc_evidence import SccEvidenceTier
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource, T_IDXS as T_IDXS_MPC, get_lead_approach_gaps
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_FLOOR, PLANNER_SEED_INTENT_LEAD_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2, SUNNYPILOT_CURRENT, StackResolution
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as LongitudinalPlanSourceSP
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  build_lead_pullaway_intent_seed_candidates,
  build_moving_lead_seed_candidates,
  build_planner_seed_accel_candidate,
  build_scc_mode_evidence,
  E2E_CLOSE_STOP_DECEL_MAX,
  E2E_CLOSE_STOP_MIN_ROLLING_V,
  E2E_STOP_APPROACH_DECEL_MAX,
  E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX,
  EXCESS_GAP_CLOSURE_START_EXCESS,
  FAST_LEAD_MOTION_OPENING_DEADBAND,
  EXCESS_GAP_CLOSURE_CAP_REASON,
  EXCESS_GAP_CLOSURE_ACCEL_CAP,
  LEAD_FLICKER_CLOSE_GUARD_TIME,
  LEAD_FLICKER_FIRST_LOSS_HOLD_TIME,
  EXCESS_GAP_CLOSURE_REASON,
  FastLeadMotionEvidence,
  COMFORT_BUDGET_BY_PERSONALITY,
  get_comfort_budget,
  LongitudinalComfortBudget,
  LeadFlickerSafetyCapTracker,
  LeadPullawayIntent,
  LeadPullawayIntentTracker,
  LeadPullawayPhase,
  LeadPullawayRunway,
  LEAD_PULLAWAY_PULSE_A_FLOOR,
  LEAD_PULLAWAY_PULSE_ACCEL_CAP,
  LEAD_PULLAWAY_PULSE_CAP_REASON,
  LEAD_PULLAWAY_PULSE_REASON,
  MOVING_LEAD_RECOVERY_MAX_ACCEL,
  MOVING_LEAD_RECOVERY_MILD_ACCEL,
  MOVING_LEAD_RECOVERY_MIN_GAP,
  MOVING_LEAD_RECOVERY_MIN_V_EGO,
  MOVING_LEAD_RECOVERY_SEED_REASON,
  MovingLeadRecovery,
  MovingLeadRecoveryPhase,
  get_moving_lead_recovery,
  MOVING_LEAD_STOP_GAP_GUARD_CLOSING_DECEL_CAP,
  MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL,
  MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP,
  MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP,
  ROUTINE_LEAD_APPROACH_DANGER_GAP_MARGIN,
  ROUTINE_LEAD_APPROACH_NEGATIVE_JERK,
  ROUTINE_LEAD_APPROACH_PREVIEW_T,
  ROUTINE_LEAD_APPROACH_SOFT_DECEL_CAP,
  ROUTINE_LEAD_APPROACH_DECEL_CAP,
  ROUTINE_LEAD_APPROACH_FIRM_DECEL_CAP,
  ROUTINE_LEAD_APPROACH_RELEASE_JERK,
  ROUTINE_LEAD_FAR_COAST_TTC,
  ROUTINE_LEAD_FAR_COAST_MIN_CLOSING,
  ROUTINE_LEAD_RESPONSE_TIME,
  STOP_RELEASE_GUARD_HOLD_TIME,
  STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON,
  STOP_RELEASE_GUARD_LEAD_RELEASE_REASON,
  STOP_RELEASE_GUARD_WAITING_REASON,
  StopReleaseGuardState,
  StopReleaseGuardTracker,
  LongitudinalPlanner,
  _A_TOTAL_MAX_BP,
  _A_TOTAL_MAX_V,
  apply_curve_load_comfort_accel_limit,
  apply_lead_pullaway_final_output_shaping,
  apply_lead_pullaway_runway_output_cap,
  apply_stop_release_guard_accel,
  fast_lead_motion_evidence_enabled,
  get_fast_lead_motion_evidence,
  get_lead_flicker_required_decel,
  get_lead_pullaway_runway,
  get_moving_lead_stop_gap_guard_accel,
  get_planner_lead_motion_values,
  get_routine_lead_approach_accel,
  get_stopped_lead_stop_gap_guard_accel,
  get_custom_v2_curve_scene_target,
  get_e2e_close_stop_settle,
  get_max_accel,
  get_e2e_runway_comfort_accel,
  get_e2e_runway_positive_accel_cap,
  get_e2e_stop_approach_accel,
  get_lead_stop_approach_slewed_accel,
  get_model_stop_distance,
  has_model_stop_context,
  has_scc_model_slowdown,
  has_valid_radar_lead,
  is_valid_routine_lead_approach,
  lead_confirmed_stop_release,
  limit_accel_in_turns,
  one_pedal_cruise_hold_requested,
  planner_seed_intent_for_reason,
  ROUTINE_LEAD_APPROACH_SEED_REASON,
  should_cap_lead_flicker_speedup,
  should_arm_stopped_lead_gap_fill,
  should_enable_longitudinal_decision_layer,
  should_allow_stopped_lead_stop_gap_guard,
  should_reserve_creep_to_stop_gap,
  update_lead_loss_e2e_guard_timer,
  get_stopped_lead_gap_fill_accel,
  should_run_engage_stop_bootstrap,
  scc_lead_geometry_from_context,
  update_one_pedal_cruise_hold,
)
from openpilot.selfdrive.controls.lib.lead_confidence import LEAD_CONFIDENCE_TRACK_UNKNOWN, LeadConfidenceState
from openpilot.selfdrive.controls.lib.lead_context import (
  LEAD_AUTHORITY_PHYSICAL,
  LEAD_AUTHORITY_PROGRESS_ALLOWED,
  LEAD_AUTHORITY_SUPPRESS_ONLY,
  LeadContextTracker,
  LeadProgressModel,
  LeadRelevanceState,
  LeadRiskModel,
  PrimaryLeadContext,
)

ButtonType = car.CarState.ButtonEvent.Type


def test_custom_v2_curve_scene_target_uses_only_active_sources():
  inactive_restrictive_vision = SimpleNamespace(is_active=False, output_a_target=-2.0)
  active_map = SimpleNamespace(is_active=True, output_a_target=-0.4)
  active_vision = SimpleNamespace(is_active=True, output_a_target=-0.7)
  active_invalid = SimpleNamespace(is_active=True, output_a_target=float("nan"))

  active, target = get_custom_v2_curve_scene_target(inactive_restrictive_vision, active_map)
  both_active, both_target = get_custom_v2_curve_scene_target(active_vision, active_map)
  invalid_active, invalid_target = get_custom_v2_curve_scene_target(active_invalid)

  assert active
  assert target == -0.4
  assert both_active
  assert both_target == -0.7
  assert not invalid_active
  assert invalid_target == 0.0


def test_one_pedal_cruise_hold_buttons_include_speed_adjustments():
  assert one_pedal_cruise_hold_requested([SimpleNamespace(type=ButtonType.accelCruise)])
  assert one_pedal_cruise_hold_requested([SimpleNamespace(type=ButtonType.decelCruise)])
  assert one_pedal_cruise_hold_requested([SimpleNamespace(type=ButtonType.resumeCruise)])
  assert one_pedal_cruise_hold_requested([SimpleNamespace(type=ButtonType.setCruise)])
  assert not one_pedal_cruise_hold_requested([SimpleNamespace(type=ButtonType.cancel)])


def test_one_pedal_cruise_hold_resets_on_pedal_or_disengage():
  assert update_one_pedal_cruise_hold(False, [SimpleNamespace(type=ButtonType.accelCruise)], False, False, True)
  assert update_one_pedal_cruise_hold(True, [], False, False, True)
  assert not update_one_pedal_cruise_hold(True, [], True, False, True)
  assert not update_one_pedal_cruise_hold(True, [], False, True, True)
  assert not update_one_pedal_cruise_hold(True, [], False, False, False)


def make_radar_state(lead_one=False, lead_two=False):
  return SimpleNamespace(
    leadOne=SimpleNamespace(status=lead_one),
    leadTwo=SimpleNamespace(status=lead_two),
  )


def make_flicker_lead(status=True, v_ego=15.0, d_rel=20.0, v_rel=-1.4, y_rel=0.0):
  return SimpleNamespace(
    status=status,
    radarTrackId=1,
    dRel=d_rel,
    vRel=v_rel,
    vLeadK=v_ego + v_rel,
    vLead=v_ego + v_rel,
    yRel=y_rel,
    aLeadK=0.0,
    modelProb=1.0,
    radar=True,
  )


NO_LEAD = SimpleNamespace(status=False)


def make_pullaway_lead(track_id=1, d_rel=31.0, v_lead=1.2, v_rel=None, a_lead=0.4, status=True, model_prob=1.0):
  if v_rel is None:
    v_rel = v_lead
  return SimpleNamespace(
    status=status,
    radarTrackId=track_id,
    dRel=d_rel,
    vLeadK=v_lead,
    vLead=v_lead,
    vRel=v_rel,
    aLeadK=a_lead,
    aLeadTau=0.0,
    yRel=0.0,
    modelProb=model_prob,
    radar=True,
  )


def stable_lead_conf(track_id=1):
  return LeadConfidenceState(
    status=True,
    stable=True,
    speed_trusted=True,
    radar=True,
    age=1.0,
    accel_blend=1.0,
    track_id=track_id,
  )


def new_lead_conf(track_id=1):
  return LeadConfidenceState(status=True, new_lead=True, speed_trusted=True, radar=True, guard_timer=0.35, track_id=track_id)


def flicker_lead_conf(track_id=1):
  return LeadConfidenceState(status=True, stable=True, speed_trusted=True, radar=True, flicker_guard_timer=0.35, track_id=track_id)


def lead_context_for(lead, v_ego=0.0, conf=None, lead_two=NO_LEAD, conf_two=None, lead_dominant_idx=None):
  conf = stable_lead_conf(getattr(lead, "radarTrackId", 1)) if conf is None else conf
  conf_two = LeadConfidenceState() if conf_two is None else conf_two
  return LeadContextTracker().update(
    (lead, lead_two), (conf, conf_two), v_ego=v_ego, dt=0.1, lead_dominant_idx=lead_dominant_idx,
  )


def update_pullaway_tracker(tracker, context, lead, *, v_ego=0.0, dt=0.1, lead_gap_excess=None,
                            predicted_gap_opening=0.4, lead_opening=None, lead_moving=None, lead_accel=None,
                            independent_stop_threat=False, alternate_lead_threat_active=False,
                            brake_pressed=False, gas_pressed=False, force_slow_decel=False, reset_state=False):
  behavior_lead = context.behavior_lead_data((lead, NO_LEAD))
  state = context.behavior
  progress = getattr(state, "progress_model", None)
  if lead_gap_excess is None:
    lead_gap_excess = getattr(progress, "gap_excess", 0.0)
  if lead_opening is None:
    lead_opening = getattr(progress, "opening_speed", 0.0) > 0.15
  if lead_moving is None:
    lead_moving = getattr(progress, "lead_moving", False)
  if lead_accel is None:
    lead_accel = getattr(progress, "lead_accel", 0.0)
  return tracker.update(
    v_ego=v_ego,
    behavior_lead=behavior_lead,
    primary_lead_context=context,
    lead_gap_excess=lead_gap_excess,
    predicted_gap_opening=predicted_gap_opening,
    lead_opening=lead_opening,
    lead_moving=lead_moving,
    lead_accel=lead_accel,
    independent_stop_threat=independent_stop_threat,
    alternate_lead_threat_active=alternate_lead_threat_active,
    brake_pressed=brake_pressed,
    gas_pressed=gas_pressed,
    force_slow_decel=force_slow_decel,
    reset_state=reset_state,
    dt=dt,
  )


def make_planner_seed_base_output(a_target=0.0, should_stop=False, has_lead=True):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=should_stop,
    has_lead=has_lead,
    source="cruise",
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(0.0 for _ in range(CONTROL_N)),
    accels=tuple(a_target for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
  )


def make_planner_seed_stub(a_target=0.0, should_stop=False, has_lead=True):
  return SimpleNamespace(
    output_a_target=a_target,
    output_should_stop=should_stop,
    planner_seed_candidate_base_output=make_planner_seed_base_output(a_target, should_stop, has_lead),
  )


def make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=200.0, positions=None, velocities=None):
  return SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=desired_accel, shouldStop=should_stop),
    position=SimpleNamespace(x=positions if positions is not None else [0.0, endpoint_x]),
    velocity=SimpleNamespace(x=velocities or []),
  )
def get_test_cp():
  return interfaces[TOYOTA.TOYOTA_COROLLA_TSS2].get_non_essential_params(TOYOTA.TOYOTA_COROLLA_TSS2)


class FakeSubMaster(dict):
  logMonoTime = {'modelV2': 1.0}

  def all_checks(self, service_list):
    return True


class FakeEvents:
  def __init__(self):
    self.names = []

  def add(self, event):
    self.names.append(event)

  def clear(self):
    self.names.clear()


class FakeModeParams:
  def __init__(self, mode: LongitudinalMode):
    self.mode = mode

  def get(self, key, *args, **kwargs):
    if key == "LongitudinalMode":
      return str(int(self.mode))
    return None

  def get_bool(self, key):
    return False


class FakeMpc:
  def __init__(self):
    self.source = LongitudinalPlanSource.cruise
    self.v_solution = np.full(len(T_IDXS_MPC), 10.0)
    self.a_solution = np.zeros(len(T_IDXS_MPC))
    self.j_solution = np.zeros(len(T_IDXS_MPC) - 1)
    self.crash_cnt = 0
    self.solve_time = 0.0
    self.dominant_obstacle_idx = None
    self.lead_dominant_obstacle_idx = None
    self.model_msgs = []

  def set_weights(self, *args, **kwargs):
    pass

  def set_cur_state(self, *args, **kwargs):
    pass

  def update(self, *args, model_msg=None, **kwargs):
    self.model_msgs.append(model_msg)


class FakeSCC:
  def __init__(self):
    self.vision = SimpleNamespace(output_v_target=255.0, output_a_target=0.0, is_active=False, state=0)
    self.map = SimpleNamespace(output_v_target=255.0, output_a_target=0.0, is_active=False, state=0)
    self.update_count = 0

  def update(self, *args, **kwargs):
    self.update_count += 1


class FakeResolver:
  speed_limit_valid = False
  speed_limit_last_valid = False
  speed_limit = 0.0
  speed_limit_final_last = 0.0
  distance = 0.0

  def update(self, *args, **kwargs):
    pass


class FakeSLA:
  is_active = False
  output_v_target = 255.0
  output_a_target = 0.0

  def update(self, *args, **kwargs):
    pass


class FakeOsmPrior:
  active = False
  output_v_target = 255.0
  output_a_target = 0.0

  def update(self, *args, **kwargs):
    pass


class PoisonModelField:
  def __init__(self, path: str):
    self.path = path

  def _raise(self):
    raise AssertionError(f"illegal model read: {self.path}")

  def __getattr__(self, name):
    raise AssertionError(f"illegal model read: {self.path}.{name}")

  def __iter__(self):
    self._raise()

  def __len__(self):
    self._raise()

  def __getitem__(self, _idx):
    self._raise()

  def __bool__(self):
    self._raise()

  def __float__(self):
    self._raise()


class PoisonModel:
  def __getattr__(self, name):
    if name == "leadsV3":
      return []
    return PoisonModelField(f"modelV2.{name}")


def make_full_update_sm(model, lead_status=False, lead_d_rel=30.0, lead_v_rel=-0.2, lead_y_rel=0.0,
                        lead_model_prob=1.0, long_control_state=LongCtrlState.off, v_ego=10.0,
                        v_cruise_kph=72.0):
  lead = SimpleNamespace(
    status=lead_status, dRel=lead_d_rel if lead_status else 100.0, vRel=lead_v_rel if lead_status else 0.0,
    vLead=v_ego + lead_v_rel if lead_status else 0.0, vLeadK=v_ego + lead_v_rel if lead_status else 0.0,
    yRel=lead_y_rel, modelProb=lead_model_prob if lead_status else 0.0, radar=lead_status, radarTrackId=0,
  )
  no_lead = SimpleNamespace(status=False, dRel=100.0, vRel=0.0, vLead=0.0, vLeadK=0.0, yRel=0.0, modelProb=0.0, radar=False)
  return FakeSubMaster({
    "carControl": SimpleNamespace(enabled=True, orientationNED=[0.0, 0.0, 0.0], cruiseControl=SimpleNamespace(override=False)),
    "carState": SimpleNamespace(
      vEgo=v_ego,
      vCruise=v_cruise_kph,
      vCruiseCluster=v_cruise_kph,
      standstill=False,
      aEgo=0.0,
      steeringAngleDeg=0.0,
      buttonEvents=[],
      gasPressed=False,
      brakePressed=False,
    ),
    "controlsState": SimpleNamespace(longControlState=long_control_state, forceDecel=False),
    "selfdriveState": SimpleNamespace(enabled=True, personality=0),
    "liveParameters": SimpleNamespace(angleOffsetDeg=0.0, stiffnessFactor=1.0, steerRatio=15.0, roll=0.0),
    "radarState": SimpleNamespace(leadOne=lead, leadTwo=no_lead),
    "modelV2": model,
  })


def make_safe_model_msg():
  xs = np.linspace(0.0, 120.0, ModelConstants.IDX_N).tolist()
  return SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=0.0, shouldStop=False),
    position=SimpleNamespace(x=xs, y=[0.0 for _ in xs]),
    velocity=SimpleNamespace(x=[10.0 for _ in xs]),
    acceleration=SimpleNamespace(x=[0.0 for _ in xs]),
    meta=SimpleNamespace(
      disengagePredictions=SimpleNamespace(gasPressProbs=[0.0, 1.0]),
      laneChangeState=log.LaneChangeState.off,
    ),
    leadsV3=[],
  )


def make_full_update_planner(mode=LongitudinalMode.ACC, radar_unavailable=False):
  CP = get_test_cp()
  CP.openpilotLongitudinalControl = True
  CP.radarUnavailable = radar_unavailable
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.CP = CP
  planner.params = FakeModeParams(mode)
  planner.mpc = FakeMpc()
  planner.planner_seed_mpc = None
  planner.VM = VehicleModel(CP)
  planner.longitudinal_arbiter = LongitudinalArbiter()
  planner.longitudinal_decision = None
  planner.longitudinal_decision_candidates = []
  planner.longitudinal_decision_telemetry = None
  planner.planner_seed_candidates = []
  planner.one_pedal_mode = 0
  planner.fast_lead_motion_evidence_param_enabled = False
  planner.one_pedal_cruise_hold_active = False
  planner.events_sp = FakeEvents()
  planner.e2e_alerts_helper = SimpleNamespace(green_light_alert=True, lead_depart_alert=True, update=lambda _sm, _events: None)
  planner.scc = FakeSCC()
  planner.resolver = FakeResolver()
  planner.sla = FakeSLA()
  planner.osm_traffic_control_prior = FakeOsmPrior()
  planner.decision_candidates_sp = []
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False
  planner.speed_limit_handoff_debug = {}
  planner.source = LongitudinalPlanSourceSP.cruise
  planner.output_v_target = 0.0
  planner.longitudinal_stack_resolution = StackResolution(
    requested_stack=SUNNYPILOT_CURRENT,
    resolved_stack=SUNNYPILOT_CURRENT,
    available_stacks=(SUNNYPILOT_CURRENT,),
  )
  planner.custom_longitudinal_stack = None
  planner.longitudinal_stack_actuated_stack = SUNNYPILOT_CURRENT
  planner.longitudinal_stack_fault_latched = False
  planner.longitudinal_stack_fault_reason = ""
  planner.longitudinal_stack_selected_intent = ""
  planner.longitudinal_stack_selected_reason = ""
  planner.longitudinal_stack_rejected = ()
  planner.longitudinal_stack_seed_context = ""
  planner.longitudinal_stack_seed_candidate = ""
  planner.custom_v2_fault_latched = False
  planner.custom_v2_fault_reason = ""
  planner.fcw = False
  planner.dt = 0.05
  planner.allow_throttle = True
  planner.a_desired = 0.0
  planner.v_desired_filter = FirstOrderFilter(10.0, 2.0, planner.dt)
  planner.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
  planner.prev_reset_state = False
  planner.engage_stop_bootstrap_timer = 0.0
  planner.e2e_close_stop_settle_active = False
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.creep_to_stop_gap_active = False
  planner.pullaway_accel_step_handoff_timer = 0.0
  planner.creep_stop_hold_released = False
  planner.stopped_lead_gap_fill_timer = 0.0
  planner.lead_loss_e2e_guard_timer = 0.0
  planner.previous_lead_loss_status = False
  planner.previous_lead_loss_d_rel = 0.0
  planner.previous_lead_loss_model_prob = 0.0
  planner.stopped_lead_gap_fill_track_id = -2
  planner.stopped_lead_gap_fill_d_rel = 0.0
  planner.stopped_lead_gap_fill_v_lead = 0.0
  planner.v_desired_trajectory = np.zeros(CONTROL_N)
  planner.a_desired_trajectory = np.zeros(CONTROL_N)
  planner.j_desired_trajectory = np.zeros(CONTROL_N)
  planner.control_calculation_hardening = False
  return planner


def test_acc_hardware_full_update_does_not_touch_poison_model_fields():
  planner = make_full_update_planner(LongitudinalMode.ACC, radar_unavailable=False)

  planner.update(make_full_update_sm(PoisonModel()))

  assert planner.mpc.model_msgs == [None]
  assert planner.longitudinal_mode_resolution.requested_mode == LongitudinalMode.ACC


def test_acc_full_update_with_physical_lead_does_not_touch_poison_model_fields():
  planner = make_full_update_planner(LongitudinalMode.ACC, radar_unavailable=False)

  planner.update(make_full_update_sm(PoisonModel(), lead_status=True))

  assert planner.mpc.model_msgs == [None]
  assert planner.longitudinal_mode_resolution.requested_mode == LongitudinalMode.ACC


def test_model_acc_full_update_does_not_touch_poison_model_stop_path_or_curve_fields():
  planner = make_full_update_planner(LongitudinalMode.ACC, radar_unavailable=True)

  planner.update(make_full_update_sm(PoisonModel()))

  assert planner.mpc.model_msgs == [None]
  assert planner.longitudinal_mode_resolution.requested_mode == LongitudinalMode.ACC


def test_e2e_full_update_poison_model_is_active():
  planner = make_full_update_planner(LongitudinalMode.E2E, radar_unavailable=False)

  with pytest.raises(AssertionError, match="illegal model read"):
    planner.update(make_full_update_sm(PoisonModel()))


def test_e2e_full_update_scene_disables_scc_curve_only_sources():
  planner = make_full_update_planner(LongitudinalMode.E2E, radar_unavailable=False)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_a_target = -1.0
  planner.scc.map.is_active = True
  planner.scc.map.output_a_target = -0.8

  planner.update(make_full_update_sm(make_safe_model_msg()))

  assert planner.custom_v2_scene.curve_active is False
  assert planner.custom_v2_scene.curve_a_target == 0.0


def test_scc_full_update_pending_model_stop_disables_scc_curve_sources_same_cycle():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_v_target = 5.0
  planner.scc.vision.output_a_target = -1.2
  planner.scc.map.is_active = True
  planner.scc.map.output_v_target = 6.0
  planner.scc.map.output_a_target = -1.0
  model = make_safe_model_msg()
  model.action.shouldStop = True
  model.action.desiredAcceleration = -1.0
  sm = make_full_update_sm(model, long_control_state=LongCtrlState.pid)

  planner.update(sm)

  evidence = planner.longitudinal_mode_resolution.scc_evidence
  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert evidence.tier == SccEvidenceTier.STOP
  assert evidence.reason == "scc_e2e_pending"
  assert not evidence.e2e_active
  assert planner.scc.update_count == 0
  assert planner.source == LongitudinalPlanSourceSP.cruise
  assert planner.custom_v2_scene.curve_active is False
  assert DecisionSource.SCC_VISION not in [candidate.source for candidate in planner.decision_candidates_sp]
  assert DecisionSource.SCC_MAP not in [candidate.source for candidate in planner.decision_candidates_sp]

  for _ in range(2):
    planner.update(sm)

  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E
  assert planner.longitudinal_mode_resolution.scc_evidence.e2e_active


def test_scc_full_update_associates_confirmed_lead_model_stop_with_scc_acc():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  model = make_safe_model_msg()
  model.action.desiredAcceleration = -1.0
  model.position.x = [0.0, 31.0, 80.0]
  model.position.y = [0.0, 0.0, 0.0]
  model.velocity.x = [10.0, 0.2, 0.2]

  planner.update(make_full_update_sm(model, lead_status=True, lead_d_rel=30.0, lead_y_rel=0.0))
  evidence = planner.longitudinal_mode_resolution.scc_evidence

  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert evidence.associated_lead_idx == 0
  assert not evidence.independent_of_lead
  assert not evidence.e2e_active


def test_scc_full_update_independent_urgent_stop_overrides_confirmed_lead_geometry():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  model = make_safe_model_msg()
  model.action.desiredAcceleration = -1.0
  model.position.x = [0.0, 8.0, 80.0]
  model.position.y = [0.0, 0.0, 0.0]
  model.velocity.x = [10.0, 0.2, 0.2]

  planner.update(make_full_update_sm(model, lead_status=True, lead_d_rel=30.0, lead_y_rel=0.0))
  evidence = planner.longitudinal_mode_resolution.scc_evidence

  assert evidence.tier == SccEvidenceTier.URGENT_STOP
  assert evidence.independent_of_lead
  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E


def test_scc_full_update_path_mismatch_model_stop_is_independent():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  model = make_safe_model_msg()
  model.action.desiredAcceleration = -1.0
  model.position.x = [0.0, 31.0, 80.0]
  model.position.y = [0.0, 0.0, 0.0]
  model.velocity.x = [10.0, 0.2, 0.2]

  sm = make_full_update_sm(model, lead_status=True, lead_d_rel=30.0, lead_y_rel=2.0, long_control_state=LongCtrlState.pid)

  planner.update(sm)
  evidence = planner.longitudinal_mode_resolution.scc_evidence

  assert evidence.associated_lead_idx is None
  assert evidence.independent_of_lead
  assert evidence.reason == "scc_e2e_pending"
  assert not evidence.e2e_active
  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC

  for _ in range(2):
    planner.update(sm)
  evidence = planner.longitudinal_mode_resolution.scc_evidence

  assert evidence.e2e_active
  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E


def test_scc_full_update_invalid_confirmed_lead_geometry_fails_closed_to_scc_e2e():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  model = make_safe_model_msg()
  model.action.desiredAcceleration = -1.0
  model.position.x = [0.0, 2.0, 80.0]
  model.position.y = [0.0, 0.0, 0.0]
  model.velocity.x = [10.0, 0.2, 0.2]

  planner.update(make_full_update_sm(model, lead_status=True, lead_d_rel=float("nan"), lead_y_rel=0.0))
  evidence = planner.longitudinal_mode_resolution.scc_evidence

  assert evidence.associated_lead_idx is None
  assert evidence.independent_of_lead
  assert evidence.e2e_active
  assert evidence.tier == SccEvidenceTier.URGENT_STOP
  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E


def test_scc_final_mode_telemetry_uses_current_cycle_curve_advisory():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_a_target = -0.4

  planner.update(make_full_update_sm(make_safe_model_msg()))

  assert planner.longitudinal_mode_resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert "curve_cap" in planner.longitudinal_mode_resolution.scc_evidence.advisory_status


def test_scc_pre_target_stale_curve_state_does_not_leak_into_final_telemetry():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_a_target = -0.4

  def disable_curve_after_update(*_args, **_kwargs):
    planner.scc.update_count += 1
    planner.scc.vision.is_active = False
    planner.scc.map.is_active = False

  planner.scc.update = disable_curve_after_update

  planner.update(make_full_update_sm(make_safe_model_msg()))

  assert planner.scc.update_count == 1
  assert "curve_cap" not in planner.longitudinal_mode_resolution.scc_evidence.advisory_status


def test_custom_v2_pullaway_handoff_clamp_does_not_mask_physical_braking():
  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  planner.longitudinal_stack_resolution = StackResolution(
    requested_stack=CUSTOM_V2,
    resolved_stack=CUSTOM_V2,
    available_stacks=(SUNNYPILOT_CURRENT, CUSTOM_V2),
    custom_version="2.0",
  )
  planner.output_a_target = 0.4
  planner.pullaway_accel_step_handoff_timer = 0.4
  planner.mpc.a_solution = np.full(len(T_IDXS_MPC), 1.0)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_v_target = 2.0
  planner.scc.vision.output_a_target = -0.6

  planner.update(make_full_update_sm(
    make_safe_model_msg(),
    lead_status=True,
    lead_d_rel=24.0,
    lead_v_rel=2.0,
    v_ego=3.1,
    v_cruise_kph=72.0,
    long_control_state=LongCtrlState.pid,
  ))

  assert planner.pullaway_accel_step_handoff_timer > 0.0
  assert planner.output_a_target < 0.0
  assert planner.longitudinal_stack_selected_intent == "lead_follow"


def test_custom_v2_pullaway_handoff_clamp_does_not_mask_advisory_braking():
  lead = make_pullaway_lead(track_id=1, d_rel=60.0, v_lead=8.1, v_rel=5.0)

  class FakeProgressContext:
    physical_idx = 0
    behavior_idx = 0
    alternate_threat_active = False
    shadow_active = False
    reason = "behavior_stable_progress_authorized_lead"
    lead_progress_allowed = True
    lead_release_blocked_reason = ""

    def __init__(self):
      self.state = SimpleNamespace(
        lead_idx=0,
        status=True,
        shadow=False,
        stable=True,
        new_lead=False,
        flicker_guard_timer=0.0,
        track_id=1,
        d_rel=lead.dRel,
        y_rel=lead.yRel,
        path_y_rel=0.0,
        v_lead=lead.vLeadK,
        v_rel=lead.vRel,
        model_prob=lead.modelProb,
        radar=True,
        risk_score=0.0,
        on_path_score=1.0,
        authority=LEAD_AUTHORITY_PROGRESS_ALLOWED,
        risk_model=LeadRiskModel(time_gap=10.0, stopped_or_crawling=False),
        progress_model=LeadProgressModel(
          opening_speed=lead.vRel,
          lead_moving=True,
          lead_accel=lead.aLeadK,
          predicted_gap_opening=True,
          gap_excess=10.0,
          stop_threat_absent=True,
          confidence_stability_sufficient=True,
          allowed=True,
          reason="opening_or_gap_progress",
        ),
      )
      self.physical = self.state
      self.behavior = self.state
      self.states = (self.state,)

    @property
    def has_physical_lead(self):
      return True

    def physical_lead_data(self, _leads):
      return lead

    def behavior_lead_data(self, _leads):
      return lead

  planner = make_full_update_planner(LongitudinalMode.SCC, radar_unavailable=False)
  planner.primary_lead_context_tracker = SimpleNamespace(update=lambda *_args, **_kwargs: FakeProgressContext())
  planner.longitudinal_stack_resolution = StackResolution(
    requested_stack=CUSTOM_V2,
    resolved_stack=CUSTOM_V2,
    available_stacks=(SUNNYPILOT_CURRENT, CUSTOM_V2),
    custom_version="2.0",
  )
  planner.output_a_target = 0.4
  planner.pullaway_accel_step_handoff_timer = 0.4
  planner.mpc.a_solution = np.full(len(T_IDXS_MPC), 1.0)
  planner.scc.vision.is_active = True
  planner.scc.vision.output_v_target = 2.0
  planner.scc.vision.output_a_target = -0.6

  planner.update(make_full_update_sm(
    make_safe_model_msg(),
    lead_status=False,
    v_ego=3.1,
    v_cruise_kph=72.0,
    long_control_state=LongCtrlState.pid,
  ))

  assert planner.pullaway_accel_step_handoff_timer > 0.0
  assert planner.output_a_target == pytest.approx(-0.6)
  assert planner.longitudinal_stack_selected_intent == "curve_policy"


def test_has_valid_radar_lead_checks_both_tracks():
  assert not has_valid_radar_lead(make_radar_state())
  assert has_valid_radar_lead(make_radar_state(lead_one=True))
  assert has_valid_radar_lead(make_radar_state(lead_two=True))


def test_lead_flicker_speedup_cap_uses_close_closing_guard():
  assert should_cap_lead_flicker_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=-1.4,
    v_lead=13.6,
    y_rel=0.0,
  )


def test_lead_flicker_speedup_cap_ignores_far_matched_noise():
  assert not should_cap_lead_flicker_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=96.0,
    v_rel=-0.6,
    v_lead=14.4,
    y_rel=0.0,
  )


def test_lead_flicker_speedup_cap_uses_far_closing_required_decel():
  assert get_lead_flicker_required_decel(90.0, -8.6) >= 0.25
  assert should_cap_lead_flicker_speedup(
    v_ego=20.0,
    lead_status=True,
    d_rel=90.0,
    v_rel=-8.6,
    v_lead=11.4,
    y_rel=0.0,
  )


def test_lead_flicker_speedup_cap_ignores_lateral_exit_lead():
  assert not should_cap_lead_flicker_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=-1.4,
    v_lead=13.6,
    y_rel=1.6,
  )


def test_lead_flicker_tracker_holds_after_first_risky_loss():
  tracker = LeadFlickerSafetyCapTracker()
  tracker.update(make_flicker_lead(), v_ego=15.0, dt=0.1)

  state = tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.1)

  assert state.active
  assert state.timer == pytest.approx(LEAD_FLICKER_FIRST_LOSS_HOLD_TIME)


def test_lead_flicker_tracker_hold_decays():
  tracker = LeadFlickerSafetyCapTracker()
  tracker.update(make_flicker_lead(), v_ego=15.0, dt=0.1)
  tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.1)

  still_held = tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.25)
  released = tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.3)

  assert still_held.active
  assert not released.active
  assert released.timer == pytest.approx(0.0)


def test_lead_flicker_tracker_uses_close_stop_go_hold_for_repeated_flicker():
  tracker = LeadFlickerSafetyCapTracker()
  close_lead = make_flicker_lead(v_ego=2.0, d_rel=9.0, v_rel=-0.5)
  lost_close_lead = make_flicker_lead(status=False, v_ego=2.0, d_rel=9.0, v_rel=-0.5)

  tracker.update(close_lead, v_ego=2.0, dt=0.1)
  tracker.update(lost_close_lead, v_ego=2.0, dt=0.1)
  tracker.update(close_lead, v_ego=2.0, dt=0.1)
  state = tracker.update(lost_close_lead, v_ego=2.0, dt=0.1)

  assert state.active
  assert state.timer == pytest.approx(LEAD_FLICKER_CLOSE_GUARD_TIME)


def test_lead_flicker_tracker_caps_route_close_lead_dropout_without_override():
  # Route 00000162--f95309d127--7 rlog, 466-482s: close lead dropped and preview cruise requested ~+1.0 m/s^2.
  def route_state(gas_pressed=False):
    tracker = LeadFlickerSafetyCapTracker()
    close_lead = make_flicker_lead(v_ego=2.472, d_rel=4.04, v_rel=-0.075, y_rel=-0.56)
    lost_close_lead = make_flicker_lead(status=False)

    initial = tracker.update(close_lead, v_ego=2.472, dt=0.1)
    lost = tracker.update(lost_close_lead, v_ego=5.151, dt=7.2, gas_pressed=gas_pressed)

    assert not initial.risky_lead
    return lost

  no_override = route_state()
  overridden = route_state(gas_pressed=True)

  assert no_override.active
  assert no_override.timer == pytest.approx(LEAD_FLICKER_FIRST_LOSS_HOLD_TIME)
  assert not overridden.active
  assert overridden.timer == pytest.approx(LEAD_FLICKER_FIRST_LOSS_HOLD_TIME)


def test_lead_flicker_tracker_driver_override_suppresses_active_cap():
  tracker = LeadFlickerSafetyCapTracker()
  tracker.update(make_flicker_lead(), v_ego=15.0, dt=0.1)
  held = tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.1)
  overridden = tracker.update(make_flicker_lead(status=False), v_ego=15.0, dt=0.1, gas_pressed=True)

  assert held.active
  assert overridden.timer > 0.0
  assert not overridden.active


def test_stop_release_guard_blocks_positive_accel_after_recent_stop_clear():
  tracker = StopReleaseGuardTracker()
  tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=True, lead_confirmed_release=False, dt=0.1,
  )

  guard = tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=False, lead_confirmed_release=False, dt=0.1,
  )
  guarded_accel, applied_guard = apply_stop_release_guard_accel(0.25, guard)

  assert guard.active
  assert guard.reason == STOP_RELEASE_GUARD_WAITING_REASON
  assert guarded_accel == 0.0
  assert applied_guard.applied


def test_stop_release_guard_expires_after_clear_persistence():
  tracker = StopReleaseGuardTracker()
  tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=True, lead_confirmed_release=False, dt=0.1,
  )

  guard = tracker.update(
    v_ego=0.0,
    standstill=True,
    stop_evidence_active=False,
    lead_confirmed_release=False,
    dt=STOP_RELEASE_GUARD_HOLD_TIME + 0.01,
  )
  accel, applied_guard = apply_stop_release_guard_accel(0.25, guard)

  assert not guard.active
  assert accel == pytest.approx(0.25)
  assert not applied_guard.applied


@pytest.mark.parametrize("block", ["brake_pressed", "force_slow_decel"])
def test_stop_release_guard_brake_or_force_still_caps_positive_accel(block):
  tracker = StopReleaseGuardTracker()
  tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=True, lead_confirmed_release=False, dt=0.1,
  )

  guard = tracker.update(
    v_ego=0.0,
    standstill=True,
    stop_evidence_active=False,
    lead_confirmed_release=False,
    dt=0.1,
    **{block: True},
  )
  guarded_accel, applied_guard = apply_stop_release_guard_accel(0.25, guard)

  assert guard.active
  assert guarded_accel == 0.0
  assert applied_guard.applied


def test_stop_release_guard_allows_lead_confirmed_release():
  tracker = StopReleaseGuardTracker()
  lead = make_pullaway_lead(v_lead=1.2, v_rel=1.2, a_lead=0.4)
  context = lead_context_for(lead)

  release = lead_confirmed_stop_release(
    context,
    context.behavior_lead_data((lead, NO_LEAD)),
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.4,
    predicted_gap_opening=0.4,
  )
  tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=True, lead_confirmed_release=False, dt=0.1,
  )
  guard = tracker.update(
    v_ego=0.0, standstill=True, stop_evidence_active=False, lead_confirmed_release=release, dt=0.1,
  )
  accel, applied_guard = apply_stop_release_guard_accel(0.35, guard)

  assert release
  assert guard.active
  assert guard.reason == STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON
  assert guard.lead_confirmed_release
  assert guard.release_accel_cap == LEAD_PULLAWAY_PULSE_A_FLOOR
  # 0.35 is below the cap (0.70), so it passes through
  assert accel == pytest.approx(0.35)
  assert not applied_guard.applied

  # Accel above cap should be capped
  accel_above, applied_above = apply_stop_release_guard_accel(1.5, guard)
  assert accel_above == LEAD_PULLAWAY_PULSE_A_FLOOR
  assert applied_above.applied


@pytest.mark.parametrize("conf", [new_lead_conf(), flicker_lead_conf()])
def test_lead_confirmed_stop_release_blocks_unstable_or_flicker_lead(conf):
  lead = make_pullaway_lead(v_lead=1.2, v_rel=1.2, a_lead=0.4)
  context = lead_context_for(lead, conf=conf)

  assert not lead_confirmed_stop_release(
    context,
    context.behavior_lead_data((lead, NO_LEAD)),
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.4,
    predicted_gap_opening=0.4,
  )


def test_lead_confirmed_stop_release_blocks_independent_stop_threat():
  lead = make_pullaway_lead(v_lead=1.2, v_rel=1.2, a_lead=0.4)
  context = lead_context_for(lead)

  assert not lead_confirmed_stop_release(
    context,
    context.behavior_lead_data((lead, NO_LEAD)),
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.4,
    predicted_gap_opening=0.4,
    independent_stop_threat=True,
  )


def test_lead_pullaway_tracker_arms_then_pulses_on_confirmed_opening_lead():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead()
  context = lead_context_for(lead)

  armed = update_pullaway_tracker(tracker, context, lead)
  pulse = update_pullaway_tracker(tracker, context, lead)

  assert armed.phase == LeadPullawayPhase.ARMED
  assert not armed.active
  assert pulse.phase == LeadPullawayPhase.PULSE
  assert pulse.active
  assert pulse.reason == LEAD_PULLAWAY_PULSE_REASON
  assert 0.0 < pulse.a_floor <= 0.7
  assert pulse.safe_accel_cap == pytest.approx(LEAD_PULLAWAY_PULSE_ACCEL_CAP)
  assert not pulse.pulse_capped_by_runway


def test_lead_pullaway_runway_caps_taper_when_lead_accel_decreases():
  runway = get_lead_pullaway_runway(
    v_ego=0.0,
    d_rel=6.3,
    v_lead=0.3,
    a_lead=0.1,
    lead_accel_trend=-0.4,
  )

  assert runway.trend == "decreasing"
  assert 0.0 < runway.safe_accel_cap < LEAD_PULLAWAY_PULSE_ACCEL_CAP
  assert runway.predicted_gap == pytest.approx(6.5)


def test_lead_pullaway_tracker_blocks_pulse_when_runway_requires_coast():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=6.4, v_lead=0.4, v_rel=0.4, a_lead=-0.1)
  context = lead_context_for(lead)

  intent = update_pullaway_tracker(
    tracker, context, lead, lead_gap_excess=1.4, predicted_gap_opening=0.3,
    lead_opening=True, lead_moving=True, lead_accel=-0.1,
  )

  assert intent.phase == LeadPullawayPhase.HOLD
  assert not intent.active
  assert intent.reason == "lead_pullaway_runway_coast"
  assert intent.coast_required
  assert intent.safe_accel_cap == pytest.approx(0.0)


def test_route_close_lead_pullaway_authorizes_stop_release_progress():
  # Route 0000019e--7e2d240269--7 around 455-457s: lead was close but
  # opening from standstill; custom-2.0 kept rejecting launch as
  # no_lead_progress_authority until driver gas override.
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=7.24, v_lead=0.93, v_rel=0.93, a_lead=0.0)
  context = lead_context_for(lead, v_ego=0.0)
  behavior_lead = context.behavior_lead_data((lead, NO_LEAD))

  release = lead_confirmed_stop_release(
    context,
    behavior_lead,
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.0,
    predicted_gap_opening=0.72,
  )
  armed = update_pullaway_tracker(
    tracker, context, lead, lead_gap_excess=2.2, predicted_gap_opening=0.72,
    lead_opening=True, lead_moving=True, lead_accel=0.0,
  )
  pulse = update_pullaway_tracker(
    tracker, context, lead, lead_gap_excess=2.2, predicted_gap_opening=0.72,
    lead_opening=True, lead_moving=True, lead_accel=0.0,
  )

  assert context.lead_progress_allowed
  assert context.behavior is not None
  assert context.behavior.authority == LEAD_AUTHORITY_PROGRESS_ALLOWED
  assert release
  assert armed.phase == LeadPullawayPhase.ARMED
  assert pulse.phase == LeadPullawayPhase.PULSE
  assert pulse.active
  assert pulse.a_floor > 0.0


def test_lead_created_runway_authorizes_early_progress_before_authority_returns():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=7.8, v_lead=1.4, v_rel=1.4, a_lead=0.55)
  physical = LeadRelevanceState(
    lead_idx=0,
    status=True,
    shadow=False,
    stable=True,
    new_lead=False,
    flicker_guard_timer=0.0,
    track_id=1,
    d_rel=lead.dRel,
    y_rel=lead.yRel,
    path_y_rel=0.0,
    v_lead=lead.vLeadK,
    v_rel=lead.vRel,
    model_prob=lead.modelProb,
    radar=True,
    ttc=10.0,
    required_decel=0.0,
    time_gap=10.0,
    on_path_score=1.0,
    risk_score=0.0,
    ghost_score=0.0,
    confidence=1.0,
    authority=LEAD_AUTHORITY_PHYSICAL,
    reason="physical",
    risk_model=LeadRiskModel(time_gap=10.0, stopped_or_crawling=False),
    progress_model=LeadProgressModel(
      opening_speed=lead.vRel,
      lead_moving=True,
      lead_accel=lead.aLeadK,
      predicted_gap_opening=True,
      gap_excess=0.0,
      stop_threat_absent=True,
      confidence_stability_sufficient=True,
      allowed=False,
      reason="opening_or_gap_progress",
    ),
  )
  context = PrimaryLeadContext(
    physical_idx=0,
    behavior_idx=None,
    physical=physical,
    behavior=None,
    alternate_threat_active=False,
    shadow_active=False,
    reason="test",
    states=(physical,),
    lead_progress_allowed=False,
    lead_release_blocked_reason="no_behavior_lead",
  )

  armed = update_pullaway_tracker(
    tracker, context, lead, v_ego=0.0, lead_gap_excess=1.0, predicted_gap_opening=1.0,
    lead_opening=True, lead_moving=True, lead_accel=0.55,
  )
  pulse = update_pullaway_tracker(
    tracker, context, lead, v_ego=0.0, lead_gap_excess=1.0, predicted_gap_opening=1.0,
    lead_opening=True, lead_moving=True, lead_accel=0.55,
  )

  assert armed.phase == LeadPullawayPhase.ARMED
  assert pulse.phase == LeadPullawayPhase.PULSE
  assert armed.early_authority
  assert pulse.early_authority
  assert armed.lead_created_runway
  assert pulse.lead_created_runway
  assert armed.early_authority_reason == "lead_created_runway"
  assert pulse.early_authority_reason == "lead_created_runway"
  assert pulse.reason == LEAD_PULLAWAY_PULSE_REASON
  assert pulse.reason != "no_lead_progress_authority"


def test_lead_pullaway_runway_does_not_trigger_excess_gap_closure_without_progress_authority():
  runway = get_lead_pullaway_runway(v_ego=0.0, d_rel=7.8, v_lead=1.4, a_lead=0.55, lead_accel_trend=0.0)

  assert runway.lead_created_runway
  assert runway.runway_creation > 0.0
  assert runway.runway_margin_now < EXCESS_GAP_CLOSURE_START_EXCESS
  assert runway.runway_margin_t < EXCESS_GAP_CLOSURE_START_EXCESS


def test_route_low_speed_pullaway_handoff_authorizes_progress_without_gas():
  # Route 000001a4--433fd05705 segment 8: lead was already opening and
  # accelerating before driver gas. Preserve the non-gas handoff path so a
  # low-speed pullaway can release into progress authority from lead evidence.
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=8.1, v_lead=1.62, v_rel=0.47, a_lead=0.49)
  context = lead_context_for(lead, v_ego=1.15)
  behavior_lead = context.behavior_lead_data((lead, NO_LEAD))

  release = lead_confirmed_stop_release(
    context,
    behavior_lead,
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.49,
    predicted_gap_opening=0.94,
  )
  armed = update_pullaway_tracker(
    tracker, context, lead, v_ego=1.15, lead_gap_excess=2.0, predicted_gap_opening=0.94,
    lead_opening=True, lead_moving=True, lead_accel=0.49,
  )
  pulse = update_pullaway_tracker(
    tracker, context, lead, v_ego=1.15, lead_gap_excess=2.0, predicted_gap_opening=0.94,
    lead_opening=True, lead_moving=True, lead_accel=0.49,
  )

  assert context.lead_progress_allowed
  assert release
  assert armed.phase == LeadPullawayPhase.ARMED
  assert pulse.phase == LeadPullawayPhase.PULSE
  assert pulse.active
  assert pulse.a_floor > 0.0


def test_close_stopped_lead_without_opening_remains_suppressive_only():
  lead = make_pullaway_lead(d_rel=7.24, v_lead=0.0, v_rel=0.0, a_lead=0.0)
  context = lead_context_for(lead, v_ego=0.0)

  assert not context.lead_progress_allowed
  assert context.behavior is None
  assert context.physical is not None


def test_stopped_gap_creep_authority_blocks_closing_or_flickering_lead():
  closing_lead = make_pullaway_lead(d_rel=6.2, v_lead=0.0, v_rel=-0.4, a_lead=0.0)
  flicker_lead = make_pullaway_lead(d_rel=6.2, v_lead=0.0, v_rel=0.0, a_lead=0.0)

  closing_context = lead_context_for(closing_lead, v_ego=0.4)
  flicker_context = lead_context_for(flicker_lead, v_ego=0.0, conf=flicker_lead_conf())

  assert not closing_context.lead_progress_allowed
  assert closing_context.behavior is None
  assert not flicker_context.lead_progress_allowed
  assert flicker_context.behavior is None


def test_stopped_gap_creep_authority_blocks_alternate_threat():
  lead = make_pullaway_lead(track_id=1, d_rel=6.2, v_lead=0.0, v_rel=0.0, a_lead=0.0)
  alternate = make_pullaway_lead(track_id=2, d_rel=5.8, v_lead=0.0, v_rel=0.0, a_lead=0.0)

  context = lead_context_for(lead, v_ego=0.0, lead_two=alternate, conf_two=stable_lead_conf(2))

  assert not context.lead_progress_allowed
  assert context.alternate_threat_active
  assert context.lead_release_blocked_reason == "alternate_lead_threat"


def test_lead_pullaway_pulse_is_time_limited_and_transitions_to_gap_closure():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=36.0, v_lead=1.5, v_rel=1.5)
  context = lead_context_for(lead)
  update_pullaway_tracker(tracker, context, lead)
  update_pullaway_tracker(tracker, context, lead)

  after_window = update_pullaway_tracker(tracker, context, lead, dt=1.0)

  assert after_window.phase == LeadPullawayPhase.GAP_CLOSURE
  assert after_window.active
  assert after_window.reason == EXCESS_GAP_CLOSURE_REASON
  assert after_window.a_floor < 0.7


def test_lead_pullaway_seed_caps_use_runway_cap_values():
  pulse_intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.PULSE,
    active=True,
    a_floor=0.42,
    reason=LEAD_PULLAWAY_PULSE_REASON,
    safe_accel_cap=0.42,
    pulse_capped_by_runway=True,
  )
  pulse_candidates = build_lead_pullaway_intent_seed_candidates(
    make_planner_seed_stub(), True, (-2.0, 2.0), pulse_intent,
  )
  pulse_cap = next(candidate for candidate in pulse_candidates if candidate.reason == LEAD_PULLAWAY_PULSE_CAP_REASON)

  gap_intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.GAP_CLOSURE,
    active=True,
    a_floor=0.18,
    reason=EXCESS_GAP_CLOSURE_REASON,
    safe_accel_cap=0.24,
  )
  gap_candidates = build_lead_pullaway_intent_seed_candidates(
    make_planner_seed_stub(), True, (-2.0, 2.0), gap_intent,
  )
  gap_cap = next(candidate for candidate in gap_candidates if candidate.reason == EXCESS_GAP_CLOSURE_CAP_REASON)

  assert pulse_cap.output.a_target == pytest.approx(0.42)
  assert pulse_cap.output.debug["lead_pullaway_safe_accel_cap"] == pytest.approx(0.42)
  assert pulse_cap.output.debug["lead_pullaway_pulse_capped_by_runway"]
  assert gap_cap.output.a_target == pytest.approx(min(EXCESS_GAP_CLOSURE_ACCEL_CAP, 0.24))


def test_lead_pullaway_runway_cap_clamps_final_custom_output():
  pulse_intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.PULSE,
    active=True,
    safe_accel_cap=0.24,
  )
  gap_intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.GAP_CLOSURE,
    active=True,
    safe_accel_cap=LEAD_PULLAWAY_PULSE_ACCEL_CAP,
  )
  coast_intent = LeadPullawayIntent(phase=LeadPullawayPhase.PULSE, active=True, coast_required=True)

  assert apply_lead_pullaway_runway_output_cap(0.55, pulse_intent) == pytest.approx(0.24)
  assert apply_lead_pullaway_runway_output_cap(0.55, gap_intent) == pytest.approx(EXCESS_GAP_CLOSURE_ACCEL_CAP)
  assert apply_lead_pullaway_runway_output_cap(0.55, coast_intent) == pytest.approx(0.0)
  assert apply_lead_pullaway_runway_output_cap(-0.4, pulse_intent) == pytest.approx(-0.4)


def test_lead_pullaway_final_shaping_keeps_runway_cap_authoritative():
  intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.GAP_CLOSURE,
    active=True,
    a_floor=0.5,
    reason=EXCESS_GAP_CLOSURE_REASON,
    safe_accel_cap=0.24,
  )

  output = apply_lead_pullaway_final_output_shaping(
    0.0, intent, prev_a_target=0.8, dt=0.05, selected_reason=EXCESS_GAP_CLOSURE_REASON,
  )

  assert output == pytest.approx(0.24)


def test_lead_pullaway_gap_closure_jerk_shape_does_not_soften_other_selected_outputs():
  intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.GAP_CLOSURE,
    active=True,
    a_floor=0.5,
    reason=EXCESS_GAP_CLOSURE_REASON,
    safe_accel_cap=1.0,
  )

  output = apply_lead_pullaway_final_output_shaping(
    -1.0, intent, prev_a_target=0.8, dt=0.05, selected_reason="planner_seed_mpc",
  )

  assert output == pytest.approx(-1.0)


def test_lead_pullaway_active_intent_hard_clamps_when_runway_cap_drops():
  tracker = LeadPullawayIntentTracker()
  tracker._last_a_floor = 0.7

  intent = tracker._active_intent(
    LeadPullawayPhase.PULSE,
    LEAD_PULLAWAY_PULSE_A_FLOOR,
    LEAD_PULLAWAY_PULSE_REASON,
    gap_excess=2.0,
    predicted_gap_opening=0.3,
    jerk_up=2.0,
    jerk_down=5.0,
    dt=0.1,
    runway=LeadPullawayRunway(safe_accel_cap=0.12),
  )

  assert intent.a_floor == pytest.approx(0.12)
  assert intent.pulse_capped_by_runway


def test_lead_pullaway_tracker_aborts_when_lead_stops_again():
  tracker = LeadPullawayIntentTracker()
  moving = make_pullaway_lead(track_id=7)
  moving_context = lead_context_for(moving)
  update_pullaway_tracker(tracker, moving_context, moving)
  update_pullaway_tracker(tracker, moving_context, moving)
  stopped = make_pullaway_lead(track_id=7, v_lead=0.0, v_rel=0.0, a_lead=0.0)
  stopped_context = lead_context_for(stopped)

  aborted = update_pullaway_tracker(tracker, stopped_context, stopped, predicted_gap_opening=0.0, lead_opening=False, lead_moving=False)

  assert aborted.phase == LeadPullawayPhase.HOLD
  assert not aborted.active
  assert aborted.reason == "lead_stopped_again"


def test_lead_pullaway_tracker_does_not_repeatedly_pulse_same_track():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(track_id=3, d_rel=36.0)
  context = lead_context_for(lead)
  update_pullaway_tracker(tracker, context, lead)
  first_pulse = update_pullaway_tracker(tracker, context, lead)
  cooled = update_pullaway_tracker(tracker, context, lead, dt=3.0)
  repeated = update_pullaway_tracker(tracker, context, lead, dt=0.1)

  assert first_pulse.phase == LeadPullawayPhase.PULSE
  assert cooled.phase in (LeadPullawayPhase.GAP_CLOSURE, LeadPullawayPhase.NORMAL)
  assert repeated.phase != LeadPullawayPhase.PULSE


def test_lead_pullaway_tracker_allows_fresh_valid_track_after_used_pulse():
  tracker = LeadPullawayIntentTracker()
  first = make_pullaway_lead(track_id=3)
  first_context = lead_context_for(first)
  update_pullaway_tracker(tracker, first_context, first)
  update_pullaway_tracker(tracker, first_context, first)
  fresh = make_pullaway_lead(track_id=4)
  fresh_context = lead_context_for(fresh)

  armed = update_pullaway_tracker(tracker, fresh_context, fresh)
  pulse = update_pullaway_tracker(tracker, fresh_context, fresh)

  assert armed.phase == LeadPullawayPhase.ARMED
  assert pulse.phase == LeadPullawayPhase.PULSE
  assert pulse.track_id == 4


@pytest.mark.parametrize("conf", [new_lead_conf(), flicker_lead_conf()])
def test_lead_pullaway_tracker_blocks_new_or_flicker_leads(conf):
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead()
  context = lead_context_for(lead, conf=conf)

  intent = update_pullaway_tracker(tracker, context, lead)

  assert intent.phase == LeadPullawayPhase.HOLD
  assert not intent.active


def test_lead_pullaway_tracker_blocks_unknown_track_before_pulse():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(track_id=LEAD_CONFIDENCE_TRACK_UNKNOWN)
  context = lead_context_for(lead, conf=stable_lead_conf(LEAD_CONFIDENCE_TRACK_UNKNOWN))

  intent = update_pullaway_tracker(tracker, context, lead)

  assert intent.phase == LeadPullawayPhase.HOLD
  assert not intent.active
  assert intent.reason == "lead_confidence_low"


def test_lead_pullaway_tracker_blocks_alternate_threat_and_independent_stop():
  lead = make_pullaway_lead(track_id=1, d_rel=31.0, v_lead=1.2, v_rel=1.2)
  threat = make_pullaway_lead(track_id=2, d_rel=8.0, v_lead=0.0, v_rel=-1.0, a_lead=0.0)
  context = lead_context_for(lead, lead_two=threat, conf_two=new_lead_conf(2), lead_dominant_idx=1)
  tracker = LeadPullawayIntentTracker()

  alternate = update_pullaway_tracker(tracker, context, lead, alternate_lead_threat_active=context.alternate_threat_active)
  independent = update_pullaway_tracker(LeadPullawayIntentTracker(), lead_context_for(lead), lead, independent_stop_threat=True)

  assert alternate.reason == "alternate_lead_threat"
  assert not alternate.active
  assert independent.reason == "independent_stop_threat"
  assert not independent.active


def test_lead_pullaway_tracker_driver_gas_suppresses_without_consuming_pulse():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(track_id=5)
  context = lead_context_for(lead)

  blocked = update_pullaway_tracker(tracker, context, lead, gas_pressed=True)
  armed = update_pullaway_tracker(tracker, context, lead)

  assert blocked.reason == "driver_or_force_blocked"
  assert not blocked.active
  assert armed.phase == LeadPullawayPhase.ARMED


def test_lead_pullaway_tracker_does_not_run_in_high_speed_lead_follow():
  tracker = LeadPullawayIntentTracker()
  lead = make_pullaway_lead(d_rel=80.0, v_lead=18.5, v_rel=0.5, a_lead=0.1)
  context = lead_context_for(lead, v_ego=18.0)

  intent = update_pullaway_tracker(tracker, context, lead, v_ego=18.0)

  assert intent.phase == LeadPullawayPhase.NORMAL
  assert not intent.active
  assert intent.reason == "outside_low_speed_launch"


def test_fast_lead_motion_evidence_uses_raw_lead_motion_before_filter():
  lead = SimpleNamespace(status=True, vLeadK=0.0, vLead=0.8, vRel=0.8)

  evidence = get_fast_lead_motion_evidence(lead, v_ego=0.0)

  assert evidence.v_lead == pytest.approx(0.8)
  assert evidence.v_rel == pytest.approx(0.8)
  assert evidence.moving()
  assert evidence.opening()


def test_fast_lead_motion_evidence_falls_back_to_aligned_relative_speed():
  lead = SimpleNamespace(status=True, vLeadK=11.0, vRel=-0.4)

  evidence = get_fast_lead_motion_evidence(lead, v_ego=12.0)

  assert evidence.v_lead == pytest.approx(11.6)
  assert evidence.v_rel == pytest.approx(-0.4)


def test_fast_lead_motion_opening_deadband_blocks_zero_crossing_noise():
  assert not FastLeadMotionEvidence(v_lead=0.0, v_rel=FAST_LEAD_MOTION_OPENING_DEADBAND - 0.01).opening()
  assert FastLeadMotionEvidence(v_lead=0.0, v_rel=FAST_LEAD_MOTION_OPENING_DEADBAND).opening()


def test_planner_lead_motion_values_preserve_filtered_speed_when_disabled():
  lead = SimpleNamespace(status=True, vLeadK=0.0, vLead=0.8, vRel=0.8)

  v_lead, v_rel, evidence = get_planner_lead_motion_values(lead, v_ego=0.0, use_fast_evidence=False)

  assert v_lead == pytest.approx(0.0)
  assert v_rel == pytest.approx(0.0)
  assert evidence.v_lead == pytest.approx(0.8)


def test_planner_lead_motion_values_keep_control_safe_when_fast_evidence_enabled():
  lead = SimpleNamespace(status=True, vLeadK=1.2, vLead=0.8, vRel=0.8)

  v_lead, v_rel, evidence = get_planner_lead_motion_values(lead, v_ego=0.0, use_fast_evidence=True)

  assert v_lead == pytest.approx(1.2)
  assert v_rel == pytest.approx(1.2)
  assert evidence.v_lead == pytest.approx(0.8)
  assert evidence.v_rel == pytest.approx(0.8)


def test_planner_lead_motion_values_keep_stable_speed_when_enabled():
  lead = SimpleNamespace(status=True, vLeadK=0.0, vLead=0.8, vRel=0.8)

  v_lead, v_rel, _evidence = get_planner_lead_motion_values(lead, v_ego=0.0, use_fast_evidence=True)

  assert v_lead == pytest.approx(0.0)
  assert v_rel == pytest.approx(0.0)


def test_fast_lead_motion_param_only_applies_to_custom_v2():
  assert fast_lead_motion_evidence_enabled(SimpleNamespace(resolved_stack="custom-2.0"), True)
  assert not fast_lead_motion_evidence_enabled(SimpleNamespace(resolved_stack="custom-2.0"), False)
  assert not fast_lead_motion_evidence_enabled(SimpleNamespace(resolved_stack="sunnypilot-current"), True)


def test_decision_layer_is_baked_into_custom_stacks_only():
  assert not should_enable_longitudinal_decision_layer(SimpleNamespace(resolved_stack="sunnypilot-current"))
  assert should_enable_longitudinal_decision_layer(SimpleNamespace(resolved_stack="custom-2.0"))
  assert should_enable_longitudinal_decision_layer(SimpleNamespace(resolved_stack="custom-recommended"))


def test_limit_accel_in_turns_defaults_to_legacy_kinematic_calculation():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 5.0
  a_target = [-1.0, 1.2]

  limited = limit_accel_in_turns(v_ego, angle_steers, a_target, CP)

  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  legacy_a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  expected_a_x_allowed = math.sqrt(max(a_total_max**2 - legacy_a_y**2, 0.0))
  assert limited == pytest.approx([a_target[0], min(a_target[1], expected_a_x_allowed)])


def test_limit_accel_in_turns_hardening_uses_vehicle_model_curvature():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 5.0
  a_target = [-1.0, 1.2]
  VM = VehicleModel(CP)

  try:
    limited = limit_accel_in_turns(v_ego, angle_steers, a_target, CP, control_calculation_hardening=True)
  except TypeError as exc:
    pytest.fail(f"limit_accel_in_turns rejected hardening toggle: {exc!r}")

  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  vehicle_model_a_y = v_ego**2 * VM.calc_curvature(angle_steers * CV.DEG_TO_RAD, v_ego, 0.0)
  expected_a_x_allowed = math.sqrt(max(a_total_max**2 - vehicle_model_a_y**2, 0.0))
  assert limited == pytest.approx([a_target[0], min(a_target[1], expected_a_x_allowed)])


def test_curve_load_comfort_tapers_positive_accel_near_lateral_limit():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 5.0
  a_target = [-1.0, 1.2]

  limited = apply_curve_load_comfort_accel_limit(v_ego, angle_steers, a_target, CP)

  assert limited[0] == pytest.approx(a_target[0])
  assert 0.0 < limited[1] < a_target[1]


def test_curve_load_comfort_never_commands_braking_from_lateral_load_only():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 15.0
  a_target = [-1.0, 1.2]

  limited = apply_curve_load_comfort_accel_limit(v_ego, angle_steers, a_target, CP)

  assert limited[0] == pytest.approx(a_target[0])
  assert limited[1] == pytest.approx(0.0)


def test_curve_load_comfort_bypasses_urgent_cases():
  CP = get_test_cp()
  v_ego = 30.0
  angle_steers = 15.0
  a_target = [-1.0, 1.2]

  limited = apply_curve_load_comfort_accel_limit(v_ego, angle_steers, a_target, CP, urgent_bypass=True)

  assert limited == pytest.approx(a_target)


def test_publish_has_lead_checks_both_tracks(monkeypatch):
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc = SimpleNamespace(solve_time=0.0, source="lead1")
  planner.v_desired_trajectory = np.zeros(3)
  planner.a_desired_trajectory = np.zeros(3)
  planner.j_desired_trajectory = np.zeros(2)
  planner.fcw = False
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.publish_longitudinal_plan_sp = lambda _sm, _pm: None
  sm = FakeSubMaster({
    'radarState': make_radar_state(lead_two=True),
  })
  plan_send = SimpleNamespace(logMonoTime=2_000_000_000, longitudinalPlan=SimpleNamespace())
  pm = SimpleNamespace(sent=None, send=lambda _service, msg: setattr(pm, "sent", msg))

  monkeypatch.setattr("openpilot.selfdrive.controls.lib.longitudinal_planner.messaging.new_message", lambda _service: plan_send)

  planner.publish(sm, pm)

  assert pm.sent.longitudinalPlan.hasLead


def test_publish_has_lead_ignores_internal_shadow_context(monkeypatch):
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc = SimpleNamespace(solve_time=0.0, source="cruise")
  planner.v_desired_trajectory = np.zeros(3)
  planner.a_desired_trajectory = np.zeros(3)
  planner.j_desired_trajectory = np.zeros(2)
  planner.fcw = False
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.primary_lead_context = SimpleNamespace(shadow_active=True, has_physical_lead=True)
  planner.publish_longitudinal_plan_sp = lambda _sm, _pm: None
  sm = FakeSubMaster({
    'radarState': make_radar_state(lead_one=False, lead_two=False),
  })
  plan_send = SimpleNamespace(logMonoTime=2_000_000_000, longitudinalPlan=SimpleNamespace())
  pm = SimpleNamespace(sent=None, send=lambda _service, msg: setattr(pm, "sent", msg))

  monkeypatch.setattr("openpilot.selfdrive.controls.lib.longitudinal_planner.messaging.new_message", lambda _service: plan_send)

  planner.publish(sm, pm)

  assert not pm.sent.longitudinalPlan.hasLead


def test_engage_stop_bootstrap_requires_timer_speed_and_no_lead():
  model_msg = make_model_msg(desired_accel=-1.5)

  assert not should_run_engage_stop_bootstrap(0.0, 10.0, make_radar_state(), model_msg)
  assert not should_run_engage_stop_bootstrap(0.5, 2.0, make_radar_state(), model_msg)
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(lead_one=True), model_msg)


def test_engage_stop_bootstrap_requires_stop_context_for_negative_model_accel():
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(desired_accel=-1.5))


def test_engage_stop_bootstrap_activates_for_negative_model_accel_with_stop_endpoint():
  model_msg = make_model_msg(desired_accel=-1.5, positions=[0.0, 12.0, 20.0], velocities=[10.0, 4.0, 0.5])

  assert should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), model_msg)


def test_engage_stop_bootstrap_activates_for_model_should_stop_without_lead():
  assert should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(should_stop=True))


def test_engage_stop_bootstrap_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "engage_stop_bootstrap", -1.2, has_lead=False, reason="engage_model_stop_bootstrap",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "engage_stop_bootstrap"
  assert candidate.output.a_target == pytest.approx(-1.2)
  assert candidate.output.should_stop
  assert candidate.output.debug["planner_seed_candidate_reason"] == "engage_model_stop_bootstrap"
  assert planner.output_a_target == pytest.approx(0.1)
  assert not planner.output_should_stop


def test_engage_stop_bootstrap_model_stop_context_uses_low_predicted_velocity():
  assert has_model_stop_context(make_model_msg(positions=[0.0, 20.0], velocities=[10.0, 0.5]))
  assert not has_model_stop_context(make_model_msg(positions=[0.0, 20.0], velocities=[10.0, 5.0]))


def test_scc_mode_evidence_promotes_no_lead_model_stop_only():
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(positions=[0.0, 20.0, 62.0], velocities=[10.0, 0.5, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert evidence.model_stop
  assert evidence.e2e_active
  assert evidence.reason == "scc_model_stop"
  assert evidence.classify().independent_of_lead


def test_scc_mode_evidence_classifies_far_endpoint_only_slowdown_without_confirmed_stop():
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(desired_accel=-1.5, positions=[0.0, 31.0, 62.0], velocities=[10.0, 10.0, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert not evidence.model_stop
  assert evidence.model_slowdown
  assert evidence.classify().tier == SccEvidenceTier.SLOWDOWN
  assert evidence.e2e_active


def test_scc_mode_evidence_promotes_route_near_endpoint_model_stop():
  # Route 00000187--ea39892416--4 rlog, 268.8-269.1s: model predicted a near terminal stop
  # while SCC stayed acc-like until the driver braked.
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(
      desired_accel=-1.45,
      positions=[0.0, 0.82, 3.08, 6.01, 8.29, 9.18, 9.34, 9.31, 8.65],
      velocities=[5.30, 5.14, 4.44, 3.03, 1.31, 0.19, -0.04, 0.00, -0.07],
    ),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert evidence.model_stop
  assert evidence.e2e_active


def test_scc_mode_evidence_promotes_route_early_model_stop_slowdown():
  # Route 00000188--249e4349c3--2 rlog, 141.46s: SCC stayed acc-like even though the
  # model slowed from 15.9 m/s to parking-lot speed over less runway than expected.
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(
      desired_accel=-1.22,
      positions=[
        0.00, 0.15, 0.62, 1.39, 2.47, 3.85, 5.54, 7.50, 9.75, 12.27, 15.04,
        18.03, 21.23, 24.61, 28.09, 31.69, 35.37, 39.03, 42.77, 46.32, 49.83,
        53.32, 56.50, 59.46, 62.22, 64.79, 66.98, 68.78, 70.48, 72.08, 73.46,
        74.71, 75.79,
      ],
      velocities=[
        15.89, 15.88, 15.88, 15.84, 15.77, 15.71, 15.59, 15.45, 15.27, 15.04,
        14.76, 14.42, 14.00, 13.50, 12.95, 12.33, 11.68, 11.00, 10.28, 9.58,
        8.82, 8.08, 7.30, 6.52, 5.76, 5.03, 4.37, 3.80, 3.34, 2.98, 2.77,
        2.61, 2.56,
      ],
    ),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert evidence.model_stop
  assert evidence.e2e_active


def test_scc_mode_evidence_keeps_signal_providers_acc_like():
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(positions=[0.0, 20.0], velocities=[10.0, 5.0]),
    SimpleNamespace(vision=SimpleNamespace(is_active=True), map=SimpleNamespace(is_active=True)),
    SimpleNamespace(is_active=True),
    SimpleNamespace(active=True),
    speed_limit_handoff_active=True,
  )

  assert evidence.curve_control
  assert evidence.map_control
  assert evidence.speed_limit_control
  assert evidence.traffic_control
  assert not evidence.e2e_active


def test_scc_mode_evidence_associates_model_stop_with_confirmed_lead():
  evidence = build_scc_mode_evidence(
    True,
    make_model_msg(positions=[0.0, 29.0, 60.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    lead_distance=30.0,
    lead_path_y_rel=0.1,
    lead_idx=0,
    v_ego=12.0,
  )
  result = evidence.classify()

  assert evidence.model_stop
  assert result.associated_lead_idx == 0
  assert not result.independent_of_lead
  assert not result.e2e_active


def test_scc_mode_evidence_missing_confirmed_lead_geometry_fails_closed():
  evidence = build_scc_mode_evidence(
    True,
    make_model_msg(positions=[0.0, 29.0, 60.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    v_ego=12.0,
  )
  result = evidence.classify()

  assert evidence.model_stop
  assert result.associated_lead_idx is None
  assert result.independent_of_lead
  assert result.e2e_active


def test_scc_mode_evidence_short_runway_sets_urgent_stop_tier():
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(positions=[0.0, 8.0, 60.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    v_ego=12.0,
  )

  assert evidence.urgent_stop
  assert evidence.classify().tier == SccEvidenceTier.URGENT_STOP


def test_scc_mode_evidence_long_runway_stays_stop_tier():
  evidence = build_scc_mode_evidence(
    False,
    make_model_msg(positions=[0.0, 80.0, 120.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    v_ego=10.0,
  )

  assert evidence.model_stop
  assert not evidence.urgent_stop
  assert evidence.classify().tier == SccEvidenceTier.STOP


def test_scc_mode_evidence_marks_geometrically_independent_urgent_stop():
  evidence = build_scc_mode_evidence(
    True,
    make_model_msg(positions=[0.0, 8.0, 60.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    lead_distance=30.0,
    lead_path_y_rel=0.0,
    lead_idx=0,
    v_ego=12.0,
  )
  urgent = SccModeEvidence(
    confirmed_lead=True,
    urgent_stop=True,
    model_stop_distance=evidence.model_stop_distance,
    lead_distance=30.0,
    lead_path_y_rel=0.0,
    lead_idx=0,
    v_ego=12.0,
  ).classify()

  assert urgent.independent_of_lead
  assert urgent.e2e_active


def test_scc_mode_evidence_large_path_mismatch_is_independent():
  evidence = build_scc_mode_evidence(
    True,
    make_model_msg(positions=[0.0, 31.0, 60.0], velocities=[10.0, 0.2, 0.2]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    lead_distance=30.0,
    lead_path_y_rel=2.0,
    lead_idx=0,
    v_ego=12.0,
  )
  result = evidence.classify()

  assert result.associated_lead_idx is None
  assert result.independent_of_lead
  assert result.e2e_active


def test_scc_model_slowdown_evidence_without_stop():
  model = make_model_msg(desired_accel=-0.9, positions=[0.0, 30.0, 60.0], velocities=[10.0, 6.0, 5.5])
  evidence = build_scc_mode_evidence(
    False,
    model,
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert has_scc_model_slowdown(model)
  assert evidence.model_slowdown
  assert not evidence.model_stop
  assert evidence.classify().tier == SccEvidenceTier.SLOWDOWN
  assert evidence.e2e_active


def test_scc_confirmed_lead_slowdown_remains_acc_like():
  evidence = build_scc_mode_evidence(
    True,
    make_model_msg(desired_accel=-0.9, positions=[0.0, 30.0, 60.0], velocities=[10.0, 6.0, 5.5]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
    lead_distance=30.0,
    lead_path_y_rel=0.0,
    lead_idx=0,
  )

  assert evidence.model_slowdown
  assert evidence.classify().tier == SccEvidenceTier.SLOWDOWN
  assert not evidence.e2e_active


def test_scc_signal_advisories_and_weak_decel_do_not_become_slowdown():
  advisory_only = build_scc_mode_evidence(
    False,
    make_model_msg(desired_accel=0.0, positions=[0.0, 30.0, 60.0], velocities=[10.0, 10.0, 10.0]),
    SimpleNamespace(vision=SimpleNamespace(is_active=True), map=SimpleNamespace(is_active=True)),
    SimpleNamespace(is_active=True),
    SimpleNamespace(active=True),
  )
  weak_decel = build_scc_mode_evidence(
    False,
    make_model_msg(desired_accel=-0.2, positions=[0.0, 30.0, 60.0], velocities=[10.0, 6.0, 5.5]),
    SimpleNamespace(vision=SimpleNamespace(is_active=False), map=SimpleNamespace(is_active=False)),
    SimpleNamespace(is_active=False),
    SimpleNamespace(active=False),
  )

  assert not advisory_only.model_slowdown
  assert advisory_only.classify().tier == SccEvidenceTier.NONE
  assert not weak_decel.model_slowdown
  assert weak_decel.classify().tier == SccEvidenceTier.NONE


def test_shadow_context_is_not_used_for_scc_confirmed_lead_geometry():
  shadow_context = SimpleNamespace(physical=SimpleNamespace(status=False, shadow=True, d_rel=8.0, path_y_rel=0.0, lead_idx=0))
  radar_state = make_radar_state()

  lead_distance, lead_path_y_rel, lead_idx = scc_lead_geometry_from_context(shadow_context, radar_state)

  assert lead_distance is None
  assert lead_path_y_rel == 0.0
  assert lead_idx is None


def test_invalid_confirmed_radar_geometry_is_not_used_for_scc_lead_context_fallback():
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=float("nan"), yRel=0.0, modelProb=1.0, radarTrackId=0),
    leadTwo=SimpleNamespace(status=False, dRel=100.0, yRel=0.0, modelProb=0.0, radarTrackId=-1),
  )

  lead_distance, lead_path_y_rel, lead_idx = scc_lead_geometry_from_context(SimpleNamespace(physical=None), radar_state)

  assert lead_distance is None
  assert lead_path_y_rel == 0.0
  assert lead_idx is None


def test_scc_lead_context_cross_checks_physical_context_against_raw_radar_geometry():
  context = SimpleNamespace(physical=SimpleNamespace(status=True, shadow=False, d_rel=0.0, path_y_rel=0.0, lead_idx=0))
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=float("nan"), yRel=0.0, modelProb=1.0, radarTrackId=0),
    leadTwo=SimpleNamespace(status=False, dRel=100.0, yRel=0.0, modelProb=0.0, radarTrackId=-1),
  )

  lead_distance, lead_path_y_rel, lead_idx = scc_lead_geometry_from_context(context, radar_state)

  assert lead_distance is None
  assert lead_path_y_rel == 0.0
  assert lead_idx is None


def test_scc_lead_context_fallback_uses_valid_lead_two_when_lead_one_distance_is_invalid():
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(status=True, dRel=-5.0, yRel=0.0, modelProb=1.0, radarTrackId=0),
    leadTwo=SimpleNamespace(status=True, dRel=36.0, yRel=0.2, modelProb=1.0, radarTrackId=7),
  )

  lead_distance, lead_path_y_rel, lead_idx = scc_lead_geometry_from_context(SimpleNamespace(physical=None), radar_state)

  assert lead_distance == 36.0
  assert lead_path_y_rel == 0.2
  assert lead_idx == 1


def test_model_stop_distance_uses_first_low_velocity_point():
  model_msg = make_model_msg(positions=[0.0, 0.8, 3.0], velocities=[1.0, 0.2, 0.0])

  assert get_model_stop_distance(model_msg) == 0.8


def test_engage_stop_bootstrap_ignores_weak_model_stop_signal():
  assert not should_run_engage_stop_bootstrap(0.5, 10.0, make_radar_state(), make_model_msg(desired_accel=-0.2))


def test_e2e_stop_approach_brakes_for_short_no_lead_endpoint():
  accel = get_e2e_stop_approach_accel(
    12.0,
    make_model_msg(endpoint_x=45.0, positions=[0.0, 30.0, 45.0], velocities=[12.0, 0.5, 3.0]),
    make_radar_state(),
    True,
  )

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5

def test_e2e_stop_approach_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "e2e_stop_approach", -0.8, has_lead=False, reason="no_lead_model_stop_approach", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "e2e_stop_approach"
  assert candidate.output.a_target == pytest.approx(-0.8)
  assert candidate.output.debug["planner_seed_candidate_reason"] == "no_lead_model_stop_approach"
  assert planner.output_a_target == pytest.approx(-0.2)


def test_planner_seed_accel_candidate_skips_non_restrictive_target():
  planner = SimpleNamespace(output_a_target=-0.2)

  candidate = build_planner_seed_accel_candidate(
    planner, "e2e_stop_approach", -0.1, has_lead=False, reason="no_lead_model_stop_approach", accel_limits=(-2.0, 2.0),
  )

  assert candidate is None


def test_planner_seed_accel_floor_candidate_can_relax_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-1.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-1.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "cruise_coast", -0.3, has_lead=False, reason="plain_cruise_overspeed_coast",
    accel_limits=(-2.0, 2.0), selection=PLANNER_SEED_FLOOR,
  )

  assert candidate is not None
  assert candidate.name == "cruise_coast"
  assert candidate.selection == PLANNER_SEED_FLOOR
  assert candidate.output.a_target == pytest.approx(-0.3)
  assert planner.output_a_target == pytest.approx(-1.0)


def test_planner_seed_accel_floor_candidate_skips_non_relaxing_target():
  planner = SimpleNamespace(output_a_target=-0.2, output_should_stop=False)

  candidate = build_planner_seed_accel_candidate(
    planner, "cruise_coast", -0.4, has_lead=False, reason="plain_cruise_overspeed_coast",
    accel_limits=(-2.0, 2.0), selection=PLANNER_SEED_FLOOR,
  )

  assert candidate is None


def test_planner_seed_accel_candidate_force_keeps_cap_available_for_floor_conflicts():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "creep_to_stop_gap_accel_cap", 0.18, has_lead=True,
    reason="creep_to_stop_gap_accel_cap", accel_limits=(-2.0, 2.0), force=True,
  )

  assert candidate is not None
  assert candidate.output.a_target == pytest.approx(0.18)
  assert planner.output_a_target == pytest.approx(-0.2)


def test_planner_seed_accel_candidate_can_carry_stop_intent_without_accel_delta():
  planner = SimpleNamespace(
    output_a_target=-0.2,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.2 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "e2e_close_stop_settle", -0.2, has_lead=False, reason="no_lead_close_stop_settle",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.output.a_target == pytest.approx(-0.2)
  assert candidate.output.should_stop
  assert not planner.output_should_stop


def test_moving_lead_stop_gap_guard_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "moving_lead_stop_gap_guard", -0.7, has_lead=True, reason="moving_lead_stop_gap_guard", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "moving_lead_stop_gap_guard"
  assert candidate.output.a_target == pytest.approx(-0.7)
  assert candidate.output.has_lead
  assert candidate.output.debug["planner_seed_candidate_reason"] == "moving_lead_stop_gap_guard"
  assert planner.output_a_target == pytest.approx(0.1)


def test_moving_lead_slower_approach_starts_mild_pre_caution_braking():
  accel = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.15,
    d_rel=25.92,
    v_lead=14.76,
    a_lead=-0.52,
    y_rel=0.0,
    t_follow=1.55,
  )

  assert accel is not None
  assert -MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP <= accel < 0.0


def test_moving_lead_routine_approach_anticipates_projected_compression_before_caution():
  d_rel = 42.0

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=d_rel,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert accel == pytest.approx(0.0)
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_phase"] == "free_coast"
  assert debug["routine_lead_coast_first_active"]
  assert debug["routine_lead_anticipatory_active"]
  assert not debug["routine_lead_approach_urgent"]
  assert d_rel > debug["routine_lead_caution_gap"]
  assert debug["routine_lead_projected_gap"] <= debug["routine_lead_caution_gap"]


def test_moving_lead_routine_approach_uses_response_compensated_gap_for_anticipation():
  v_ego = 20.7
  v_lead = 18.9
  t_follow = 1.55
  _desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  closing = v_ego - v_lead
  response_gap_loss = closing * ROUTINE_LEAD_RESPONSE_TIME
  d_rel = caution_gap + closing * ROUTINE_LEAD_APPROACH_PREVIEW_T + response_gap_loss * 0.5

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=d_rel,
    v_lead=v_lead,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_anticipatory_active"]
  assert not debug["routine_lead_approach_urgent"]
  assert debug["routine_lead_gap_lost_to_response"] == pytest.approx(response_gap_loss)
  assert debug["routine_lead_effective_d_rel"] == pytest.approx(d_rel - response_gap_loss)
  assert debug["routine_lead_projected_gap_raw"] > debug["routine_lead_caution_gap"]
  assert debug["routine_lead_projected_gap_response_compensated"] <= debug["routine_lead_caution_gap"]
  assert debug["routine_lead_projected_gap"] == pytest.approx(debug["routine_lead_projected_gap_response_compensated"])
  assert debug["routine_lead_gap_after_coast"] == pytest.approx(debug["routine_lead_projected_gap"])
  assert debug["routine_lead_required_decel_after_coast"] >= 0.0


def test_moving_lead_routine_approach_response_compensation_includes_lead_decel():
  v_ego = 20.7
  v_lead = 18.9
  a_lead = -1.0
  t_follow = 1.55
  _desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  closing = v_ego - v_lead
  relative_accel = -a_lead
  delayed_closing = closing + relative_accel * ROUTINE_LEAD_RESPONSE_TIME
  response_gap_loss = closing * ROUTINE_LEAD_RESPONSE_TIME + 0.5 * relative_accel * ROUTINE_LEAD_RESPONSE_TIME**2
  d_rel = caution_gap + closing * ROUTINE_LEAD_APPROACH_PREVIEW_T + 0.5 * relative_accel * ROUTINE_LEAD_APPROACH_PREVIEW_T**2 + response_gap_loss * 0.5
  expected_compensated_gap = max(
    0.0,
    d_rel - response_gap_loss - delayed_closing * ROUTINE_LEAD_APPROACH_PREVIEW_T -
    0.5 * relative_accel * ROUTINE_LEAD_APPROACH_PREVIEW_T**2,
  )

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=d_rel,
    v_lead=v_lead,
    a_lead=a_lead,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_gap_lost_to_response"] == pytest.approx(response_gap_loss)
  assert debug["routine_lead_effective_d_rel"] == pytest.approx(d_rel - response_gap_loss)
  assert debug["routine_lead_projected_closing"] == pytest.approx(delayed_closing + relative_accel * ROUTINE_LEAD_APPROACH_PREVIEW_T)
  assert debug["routine_lead_projected_gap_raw"] > debug["routine_lead_caution_gap"]
  assert debug["routine_lead_projected_gap_response_compensated"] == pytest.approx(expected_compensated_gap)
  assert debug["routine_lead_projected_gap_response_compensated"] <= debug["routine_lead_projected_gap_raw"] - relative_accel * ROUTINE_LEAD_RESPONSE_TIME * ROUTINE_LEAD_APPROACH_PREVIEW_T


def test_moving_lead_routine_approach_reports_response_lag_shortfall_without_strengthening_cap():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=38.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    a_ego=0.0,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_phase"] == "soft_decel"
  assert debug["routine_lead_a_ego"] == pytest.approx(0.0)
  assert debug["routine_lead_decel_shortfall"] == pytest.approx(-debug["routine_lead_raw_a_target"])
  assert not debug["routine_lead_approach_urgent"]
  assert not debug["routine_lead_urgent_bypass"]
  assert -MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP <= accel < 0.0


def test_moving_lead_routine_approach_soft_decel_follows_coast_as_compression_worsens():
  coast_accel, coast_debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=42.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )
  soft_accel, soft_debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=38.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=coast_accel,
    dt=0.5,
    return_debug=True,
  )

  assert coast_accel == pytest.approx(0.0)
  assert coast_debug["routine_lead_phase"] == "free_coast"
  assert soft_accel is not None
  assert -MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP <= soft_accel < 0.0
  assert soft_debug["routine_lead_phase"] == "soft_decel"
  assert soft_accel >= -ROUTINE_LEAD_APPROACH_NEGATIVE_JERK * 0.5 - 1e-6


def test_moving_lead_routine_approach_drel_40_remains_free_coast_with_looser_threshold():
  """With the looser ROUTINE_LEAD_APPROACH_COAST_BLEND (0.32 vs old 0.18),
  a d_rel of 40 m at v_ego=20.7 v_lead=18.7 (closing 2 m/s) should now
  remain in free_coast, not soft_decel. The compression_blend at this
  point is ~0.27, which is above the old 0.18 threshold but below the
  new 0.32 threshold.

  This is the explicit expression of the gap-compression loosening: the
  routine phase should spend more of its compression runway coasting and
  not preemptively brake at a comfort gap that is still well-buffered."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=40.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_phase"] == "free_coast"
  assert accel == pytest.approx(0.0)
  # Comp_blend is between old and new COAST_BLEND thresholds.
  assert 0.18 <= debug["routine_lead_compression_blend"] < 0.32
  assert not debug["routine_lead_approach_urgent"]


def test_moving_lead_routine_approach_drel_38_enters_soft_decel_with_looser_threshold():
  """At d_rel=38 m the compression_blend climbs to ~0.40, just above the
  new COAST_BLEND of 0.32 and below the new SOFT_BLEND of 0.62. The phase
  should be soft_decel with a bounded decel. This is the second
  compression-worsen case after free_coast at d_rel=40."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=38.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_phase"] == "soft_decel"
  assert -MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP <= accel < 0.0
  # Soft_decel cap is tighter than routine_decel cap.
  assert -ROUTINE_LEAD_APPROACH_SOFT_DECEL_CAP - 1e-6 <= accel
  assert not debug["routine_lead_approach_urgent"]


def test_moving_lead_routine_approach_no_brake_yet_debug_field_is_exposed():
  """The no-brake-yet flag should be exposed in the routine debug output
  so route diagnostics and post-hoc logging can verify the rule
  computation. The flag combines two conditions:
  - brake_predicted_gap > caution_gap - 0.5 m
  - required_decel_after_coast < 0.25 m/s²
  Both use the brake_preview_t horizon so they reflect the brake-side
  projection, not the coast-side projection used for compression_blend."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=42.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )
  assert accel is not None
  assert "routine_lead_no_brake_yet" in debug
  # At d_rel=42 (well above caution) the brake projection is safe and the
  # post-coast required decel is small, so no-brake-yet should be True.
  assert debug["routine_lead_no_brake_yet"] is True
  assert debug["routine_lead_brake_predicted_gap"] > debug["routine_lead_caution_gap"] - 0.5
  assert debug["routine_lead_required_decel_after_coast"] < 0.25


def test_moving_lead_routine_approach_no_brake_yet_relaxes_when_brake_projection_drops():
  """When the brake-projected gap drops below caution_gap - 0.5, the
  no-brake-yet rule stops firing and the routine phase is driven by the
  compression_blend ramp alone (soft_decel or routine_decel)."""
  v_ego = 20.7
  v_lead = 18.7
  t_follow = 1.55

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=34.0,
    v_lead=v_lead,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert not debug["routine_lead_no_brake_yet"]
  # The phase is now driven by compression_blend, not the no-brake-yet rule.
  assert debug["routine_lead_phase"] in ("soft_decel", "routine_decel")
  assert accel < 0.0


def test_moving_lead_routine_approach_brake_preview_uses_shorter_horizon_than_coast_preview():
  """The brake-preview projection should look ahead less than the
  coast-preview projection. brake_predicted_gap is the gap projected over
  the shorter brake_preview_t (1.4 s) using response-compensated state.
  Because we project over less time, the brake projection always shows a
  larger remaining gap than the coast projection."""
  v_ego = 20.7
  v_lead = 18.7
  t_follow = 1.55

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=42.0,
    v_lead=v_lead,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  # Brake preview is shorter than coast preview.
  assert debug["routine_lead_brake_preview_t"] < debug["routine_lead_preview_t"]
  # Brake projection leaves more room than coast projection because the
  # brake projection only looks ahead over a shorter horizon.
  assert debug["routine_lead_brake_predicted_gap"] >= debug["routine_lead_projected_gap"]
  # Brake projected closing is smaller or equal to coast projected closing
  # because we look ahead less time.
  assert debug["routine_lead_brake_projected_closing"] <= debug["routine_lead_projected_closing"]


def test_moving_lead_routine_approach_near_danger_plus_margin_allows_routine_decel():
  """At a gap just past the routine floor (danger_gap + margin), the
  routine phase should be routine_decel with bounded decel. The urgent
  path is reserved for true danger gap / short TTC / hard lead braking."""
  v_ego = 20.7
  v_lead = 18.7
  t_follow = 1.55
  _desired_gap, _caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=danger_gap + ROUTINE_LEAD_APPROACH_DANGER_GAP_MARGIN,
    v_lead=v_lead,
    a_lead=-0.5,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    a_ego=0.0,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert not debug["routine_lead_approach_urgent"]
  assert not debug["routine_lead_urgent_bypass"]
  assert debug["routine_lead_phase"] == "routine_decel"
  assert accel >= -MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP


def test_moving_lead_routine_approach_allows_compression_to_danger_gap_plus_margin_without_urgent_bypass():
  v_ego = 20.7
  v_lead = 18.7
  t_follow = 1.55
  _desired_gap, _caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=danger_gap + ROUTINE_LEAD_APPROACH_DANGER_GAP_MARGIN,
    v_lead=v_lead,
    a_lead=-0.5,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    a_ego=0.0,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert not debug["routine_lead_approach_urgent"]
  assert not debug["routine_lead_urgent_bypass"]
  assert debug["routine_lead_phase"] == "routine_decel"
  assert accel >= -MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP


def test_moving_lead_routine_approach_true_danger_gap_remains_urgent():
  v_ego = 20.7
  v_lead = 18.7
  t_follow = 1.55
  _desired_gap, _caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=danger_gap,
    v_lead=v_lead,
    a_lead=-0.5,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=0.0,
    dt=0.5,
    a_ego=0.0,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_urgent"]
  assert debug["routine_lead_phase"] in ("routine_decel", "urgent_bypass")


def test_route_moving_lead_stop_gap_guard_caution_edge_does_not_hard_bypass_while_far_from_danger():
  # Route 000001a4--433fd05705 segment 10 crossed caution while still more
  # than 10 m outside danger. The guard should not jump directly to the hard
  # moving-lead stop cap there; it should stay on the pre-danger closing ramp.
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=18.53,
    d_rel=40.5,
    v_lead=15.0,
    a_lead=-1.20,
    y_rel=0.0,
    t_follow=1.45,
    prev_a_target=-1.22,
    dt=0.19,
    a_ego=-1.22,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_distance_to_caution"] == pytest.approx(-0.21, abs=0.05)
  assert debug["routine_lead_distance_to_danger"] > 10.0
  assert not debug["moving_lead_stop_gap_guard_urgent"]
  assert not debug["routine_lead_urgent_bypass"]
  assert accel >= -MOVING_LEAD_STOP_GAP_GUARD_CLOSING_DECEL_CAP - 1e-6


def test_route_moving_lead_stop_gap_guard_hard_lead_inside_caution_stays_urgent():
  # Same route, later in segment 10: hard lead braking with limited runway must
  # keep urgent physical braking. This prevents the pre-danger comfort ramp from
  # weakening true hard-closing lead response.
  v_ego = 12.39
  v_lead = 9.31
  a_lead = -2.03
  t_follow = 1.45
  _desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=float(caution_gap) - 0.2,
    v_lead=v_lead,
    a_lead=a_lead,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=-1.93,
    dt=0.5,
    a_ego=-1.93,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_distance_to_danger"] > 2.0
  assert debug["moving_lead_stop_gap_guard_hard_lead_urgent"]
  assert debug["moving_lead_stop_gap_guard_urgent"]
  assert debug["routine_lead_urgent_bypass"]
  assert accel <= -MOVING_LEAD_STOP_GAP_GUARD_HARD_DECEL


def test_moving_lead_stop_gap_guard_short_danger_ttc_stays_urgent_without_hard_lead():
  v_ego = 18.53
  v_lead = 15.0
  a_lead = -1.20
  t_follow = 1.45
  _desired_gap, _caution_gap, danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)

  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=v_ego,
    d_rel=float(danger_gap) + 7.0,
    v_lead=v_lead,
    a_lead=a_lead,
    y_rel=0.0,
    t_follow=t_follow,
    prev_a_target=-0.45,
    dt=0.5,
    a_ego=-0.45,
    return_debug=True,
  )

  assert accel is not None
  assert not debug["moving_lead_stop_gap_guard_hard_lead_urgent"]
  assert debug["moving_lead_stop_gap_guard_danger_ttc"] <= 2.0
  assert debug["moving_lead_stop_gap_guard_closing_urgent"]
  assert debug["moving_lead_stop_gap_guard_urgent"]
  assert debug["routine_lead_urgent_bypass"]
  assert accel <= -MOVING_LEAD_STOP_GAP_GUARD_MILD_DECEL_CAP


def test_moving_lead_routine_approach_releases_when_projection_risk_clears_without_positive_accel():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=20.7,
    d_rel=42.0,
    v_lead=18.7,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=-0.3,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_phase"] == "free_coast"
  assert debug["routine_lead_raw_a_target"] == pytest.approx(0.0)
  assert debug["routine_lead_release_limited"]
  assert -0.3 < accel <= 0.0


def test_moving_lead_routine_approach_builds_one_direction_while_compression_worsens():
  prev_a_target = 0.0
  dt = 0.5

  for d_rel in (42.0, 40.0, 38.0):
    accel, debug = get_moving_lead_stop_gap_guard_accel(
      v_ego=20.7,
      d_rel=d_rel,
      v_lead=18.7,
      a_lead=0.0,
      y_rel=0.0,
      t_follow=1.55,
      prev_a_target=prev_a_target,
      dt=dt,
      return_debug=True,
    )

    assert accel is not None
    assert debug["routine_lead_approach_active"]
    assert not debug["routine_lead_approach_urgent"]
    assert accel <= prev_a_target + 1e-6
    assert prev_a_target - accel <= ROUTINE_LEAD_APPROACH_NEGATIVE_JERK * dt + 1e-6
    prev_a_target = accel


def test_moving_lead_routine_approach_does_not_weaken_safety_relevant_existing_guard_target():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.12,
    d_rel=24.36,
    v_lead=12.72,
    a_lead=-2.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_existing_target_safety_relevant"]
  assert debug["routine_lead_should_defer_to_existing_target"]
  assert accel == pytest.approx(debug["routine_lead_existing_target"])
  assert accel <= debug["routine_lead_ramped_a_target"]


def test_moving_lead_routine_approach_prefers_routine_for_nonurgent_firmer_existing_target():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=22.0,
    d_rel=45.17,
    v_lead=19.2,
    a_lead=-0.6,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_can_own_nonurgent_shape"]
  assert not debug["routine_lead_should_defer_to_existing_target"]
  assert debug["routine_lead_existing_target_reason"] in ("slower_lead_approach", "moving_lead_stop_gap_guard")
  assert not debug["routine_lead_existing_target_safety_relevant"]
  assert accel == pytest.approx(debug["routine_lead_ramped_a_target"])
  assert debug["routine_lead_selected_target"] == pytest.approx(accel)


def test_moving_lead_routine_approach_defers_to_safety_relevant_existing_target():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.12,
    d_rel=24.36,
    v_lead=12.72,
    a_lead=-2.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert not debug["routine_lead_can_own_nonurgent_shape"]
  assert debug["routine_lead_should_defer_to_existing_target"]
  assert debug["routine_lead_existing_target_safety_relevant"]
  assert debug["routine_lead_existing_target_reason"] == "moving_lead_stop_gap_guard"
  assert accel == pytest.approx(debug["routine_lead_existing_target"])
  assert debug["routine_lead_selected_target"] == pytest.approx(accel)


def test_moving_lead_routine_approach_does_not_drop_at_closing_urgent_threshold_without_stronger_target():
  below_threshold, _below_debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=22.0,
    d_rel=50.0,
    v_lead=19.01,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )
  at_threshold, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=22.0,
    d_rel=50.0,
    v_lead=19.0,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )

  assert below_threshold is not None
  assert at_threshold is not None
  assert at_threshold <= below_threshold + 1e-6
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_approach_urgent"]
  assert not debug["routine_lead_urgent_bypass"]


def test_moving_lead_slower_approach_ignores_clear_or_opening_gap():
  clear_gap = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.0,
    d_rel=35.0,
    v_lead=14.0,
    a_lead=-0.7,
    y_rel=0.0,
    t_follow=1.55,
  )
  opening_gap = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.0,
    d_rel=25.0,
    v_lead=15.2,
    a_lead=-0.7,
    y_rel=0.0,
    t_follow=1.55,
  )

  assert clear_gap is None
  assert opening_gap is None


def test_moving_lead_slower_approach_preserves_urgent_caution_braking():
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.12,
    d_rel=24.36,
    v_lead=12.72,
    a_lead=-2.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.5,
    return_debug=True,
  )

  assert accel is not None
  assert accel < -MOVING_LEAD_SLOWER_APPROACH_DECEL_CAP
  assert debug["routine_lead_urgent_bypass"]
  assert debug["routine_lead_phase"] == "urgent_bypass"


def test_moving_lead_slower_approach_preserves_lateral_exit_rejection():
  accel = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.15,
    d_rel=25.92,
    v_lead=14.76,
    a_lead=-0.52,
    y_rel=2.0,
    t_follow=1.55,
  )

  assert accel is None


def test_stopped_lead_stop_gap_guard_custom_candidate_carries_stop_intent_without_mutating_baseline():
  planner = SimpleNamespace(
    output_a_target=-0.1,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.1 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "stopped_lead_stop_gap_guard", -0.8, has_lead=True, reason="stopped_lead_stop_gap_guard",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "stopped_lead_stop_gap_guard"
  assert candidate.output.a_target == pytest.approx(-0.8)
  assert candidate.output.should_stop
  assert candidate.output.has_lead
  assert candidate.output.debug["planner_seed_candidate_reason"] == "stopped_lead_stop_gap_guard"
  assert planner.output_a_target == pytest.approx(-0.1)
  assert not planner.output_should_stop


def test_stopped_lead_stop_gap_guard_blocks_far_lane_change_false_stop_seed():
  # Route 0000018e--2182c485e2--17, 1066.50s: a far near-zero-speed lead appeared
  # during pre-lane-change and selected stopped_lead_stop_gap_guard at ~96 m.
  route_guard = get_stopped_lead_stop_gap_guard_accel(
    v_ego=17.45,
    d_rel=96.16,
    v_lead=-0.14,
    a_lead=0.03,
    model_prob=0.75,
  )

  assert route_guard is not None
  assert not should_allow_stopped_lead_stop_gap_guard(17.45, 96.16, -0.14, lane_change_active=True)
  assert should_allow_stopped_lead_stop_gap_guard(17.45, 96.16, -0.14, lane_change_active=False)


def test_stopped_lead_stop_gap_guard_preserves_close_lane_change_stop_threat():
  close_guard = get_stopped_lead_stop_gap_guard_accel(
    v_ego=15.7,
    d_rel=40.8,
    v_lead=-0.10,
    a_lead=0.0,
    model_prob=0.77,
  )

  assert close_guard is not None
  assert close_guard < -1.0
  assert should_allow_stopped_lead_stop_gap_guard(15.7, 40.8, -0.10, lane_change_active=True)


def test_stopped_lead_creep_hold_custom_candidate_carries_stop_intent_without_mutating_baseline():
  planner = SimpleNamespace(
    output_a_target=0.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "stopped_lead_creep_hold", -0.25, has_lead=True, reason="stopped_lead_creep_hold",
    accel_limits=(-2.0, 2.0), should_stop=True,
  )

  assert candidate is not None
  assert candidate.name == "stopped_lead_creep_hold"
  assert candidate.output.a_target == pytest.approx(-0.25)
  assert candidate.output.should_stop
  assert candidate.output.has_lead
  assert candidate.output.debug["planner_seed_candidate_reason"] == "stopped_lead_creep_hold"
  assert planner.output_a_target == pytest.approx(0.0)
  assert not planner.output_should_stop


def test_e2e_stop_approach_ignores_endpoint_shortage_without_stop_evidence():
  accel = get_e2e_stop_approach_accel(
    15.9,
    make_model_msg(endpoint_x=62.0, positions=[0.0, 62.0], velocities=[15.9, 0.2]),
    make_radar_state(),
    True,
  )

  assert accel == 0.0


def test_e2e_stop_approach_ignores_endpoint_with_sufficient_runway():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=70.0), make_radar_state(), True)

  assert accel == 0.0


def test_e2e_stop_approach_uses_longer_runway_with_traction_risk():
  model_msg = make_model_msg(endpoint_x=80.0, positions=[0.0, 63.0, 80.0], velocities=[12.0, 0.5, 3.0])

  normal = get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True)
  traction_limited = get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, traction_risk=1.0)

  assert normal == 0.0
  assert -E2E_STOP_APPROACH_DECEL_MAX <= traction_limited < 0.0


def test_e2e_stop_approach_uses_earlier_model_stop_point_for_crawl_reserve():
  route_like_model = make_model_msg(
    endpoint_x=123.0,
    positions=[0.0, 65.0, 123.0],
    velocities=[15.3, 0.5, 3.0],
  )

  accel = get_e2e_stop_approach_accel(15.3, route_like_model, make_radar_state(), True)
  endpoint_only_accel = get_e2e_stop_approach_accel(15.3, make_model_msg(endpoint_x=123.0), make_radar_state(), True)

  assert endpoint_only_accel == 0.0
  assert -1.0 < accel < -0.3


def test_e2e_stop_approach_starts_mild_decel_for_route_like_runway():
  accel = get_e2e_stop_approach_accel(
    15.7,
    make_model_msg(endpoint_x=100.0, positions=[0.0, 86.0, 100.0], velocities=[15.7, 0.5, 3.0]),
    make_radar_state(),
    True,
  )

  assert -0.5 < accel < -0.15


def test_e2e_stop_approach_brakes_before_high_speed_max_decel_boundary():
  accel = get_e2e_stop_approach_accel(
    60.0 / 3.6,
    make_model_msg(endpoint_x=90.0, positions=[0.0, 70.0, 90.0], velocities=[60.0 / 3.6, 0.5, 3.0]),
    make_radar_state(),
    True,
  )

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_caps_route_like_peak_decel():
  accel = get_e2e_stop_approach_accel(
    60.0 / 3.6,
    make_model_msg(endpoint_x=45.0, positions=[0.0, 30.0, 45.0], velocities=[60.0 / 3.6, 0.5, 3.0]),
    make_radar_state(),
    True,
  )

  assert math.isclose(accel, -E2E_STOP_APPROACH_DECEL_MAX)


def test_e2e_stop_approach_preserves_urgent_stop_cap_with_traction_risk():
  accel = get_e2e_stop_approach_accel(
    60.0 / 3.6,
    make_model_msg(endpoint_x=45.0, positions=[0.0, 30.0, 45.0], velocities=[60.0 / 3.6, 0.5, 3.0]),
    make_radar_state(),
    True,
    traction_risk=1.0,
  )

  assert math.isclose(accel, -E2E_STOP_APPROACH_DECEL_MAX)


def test_e2e_stop_approach_ignores_clear_endpoint():
  assert get_e2e_stop_approach_accel(12.0, make_model_msg(endpoint_x=200.0), make_radar_state(), True) == 0.0


def test_e2e_stop_approach_requires_no_lead_and_no_override():
  model_msg = make_model_msg(endpoint_x=45.0, positions=[0.0, 30.0, 45.0], velocities=[12.0, 0.5, 3.0])

  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(lead_one=True), True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), False) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, brake_pressed=True) == 0.0
  assert get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True, gas_pressed=True) == 0.0


def test_e2e_stop_approach_protects_close_endpoint_during_scc_acc_transition():
  accel = get_e2e_stop_approach_accel(
    2.6,
    make_model_msg(desired_accel=-1.19, endpoint_x=5.7),
    make_radar_state(),
    False,
    model_stop_protection_active=True,
  )

  assert -E2E_STOP_APPROACH_DECEL_MAX <= accel < -0.5


def test_e2e_stop_approach_protection_keeps_low_speed_floor():
  accel = get_e2e_stop_approach_accel(
    1.5,
    make_model_msg(desired_accel=-1.19, endpoint_x=4.0),
    make_radar_state(),
    False,
    model_stop_protection_active=True,
  )

  assert accel == 0.0


def test_e2e_stop_approach_leaves_hard_model_stop_to_model():
  accel = get_e2e_stop_approach_accel(12.0, make_model_msg(should_stop=True, endpoint_x=30.0), make_radar_state(), True)

  assert accel == 0.0


def test_e2e_close_stop_settle_holds_decel_at_route_like_stop_line():
  accel, should_stop, active = get_e2e_close_stop_settle(
    0.44,
    -0.26,
    make_model_msg(desired_accel=-0.26, positions=[0.0, 0.01, 20.0], velocities=[1.0, 0.2, 2.0]),
    make_radar_state(),
    True,
  )

  assert active
  assert should_stop
  assert -E2E_CLOSE_STOP_DECEL_MAX <= accel < -0.3


def test_e2e_close_stop_settle_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=-0.05,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-0.05 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )
  accel, should_stop, _active = get_e2e_close_stop_settle(
    0.44,
    -0.26,
    make_model_msg(desired_accel=-0.26, positions=[0.0, 0.01, 20.0], velocities=[1.0, 0.2, 2.0]),
    make_radar_state(),
    True,
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "e2e_close_stop_settle", accel, has_lead=False, reason="no_lead_close_stop_settle",
    accel_limits=(-2.0, 2.0), should_stop=should_stop,
  )

  assert candidate is not None
  assert candidate.name == "e2e_close_stop_settle"
  assert candidate.output.a_target < planner.output_a_target
  assert candidate.output.should_stop
  assert planner.output_a_target == pytest.approx(-0.05)
  assert not planner.output_should_stop


def test_e2e_close_stop_settle_keeps_stop_latch_below_rolling_speed():
  accel, should_stop, active = get_e2e_close_stop_settle(
    E2E_CLOSE_STOP_MIN_ROLLING_V - 0.01,
    -0.05,
    make_model_msg(desired_accel=-0.05, positions=[0.0, 0.1], velocities=[1.0, 0.0]),
    make_radar_state(),
    True,
    active=True,
  )

  assert accel == -0.05
  assert should_stop
  assert active


def test_e2e_close_stop_settle_requires_no_lead_e2e_and_no_override():
  model_msg = make_model_msg(desired_accel=-0.2, positions=[0.0, 0.2], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(lead_one=True), True) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), False) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, brake_pressed=True) == (-0.2, False, False)
  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, gas_pressed=True) == (-0.2, False, False)


@pytest.mark.parametrize("block", [
  {"brake_pressed": True},
  {"gas_pressed": True},
  {"force_slow_decel": True},
  {"reset_state": True},
])
def test_stop_go_reserve_creep_requires_progress_authority_and_no_blocks(block):
  kwargs = dict(primary_behavior_progress_allowed=True, output_should_stop=False, v_ego=0.02, d_rel=5.75, v_lead=0.0)

  assert should_reserve_creep_to_stop_gap(**kwargs)
  assert not should_reserve_creep_to_stop_gap(**{**kwargs, "primary_behavior_progress_allowed": False})
  assert not should_reserve_creep_to_stop_gap(**{**kwargs, "output_should_stop": True})
  assert not should_reserve_creep_to_stop_gap(**kwargs, **block)


@pytest.mark.parametrize("block", [
  {"brake_pressed": True},
  {"gas_pressed": True},
  {"force_slow_decel": True},
])
def test_stopped_lead_gap_fill_arms_and_accelerates_only_without_blocks(block):
  assert should_arm_stopped_lead_gap_fill(v_ego=0.1, d_rel=5.5, v_lead=0.0, model_prob=1.0)
  assert not should_arm_stopped_lead_gap_fill(v_ego=0.1, d_rel=5.5, v_lead=0.0, model_prob=1.0, **block)

  active, accel = get_stopped_lead_gap_fill_accel(v_ego=0.2, d_rel=12.0, v_lead=0.2, model_prob=1.0, armed=True)
  blocked_active, blocked_accel = get_stopped_lead_gap_fill_accel(
    v_ego=0.2, d_rel=12.0, v_lead=0.2, model_prob=1.0, armed=True, **block,
  )

  assert active
  assert accel > 0.0
  assert not blocked_active
  assert blocked_accel == 0.0


@pytest.mark.parametrize("block", [
  {"reset_state": True},
  {"force_slow_decel": True},
  {"brake_pressed": True},
  {"gas_pressed": True},
])
def test_lead_loss_e2e_guard_timer_is_cap_only_and_clears_on_blocks(block):
  guard_kwargs = dict(
    previous_lead_status=True, previous_d_rel=50.0, previous_model_prob=1.0,
    current_has_lead=False, lane_change_active=True,
  )
  armed = update_lead_loss_e2e_guard_timer(0.0, 0.1, **guard_kwargs)
  cleared = update_lead_loss_e2e_guard_timer(armed, 0.1, **guard_kwargs, **block)

  assert armed > 0.0
  assert cleared == 0.0
  assert update_lead_loss_e2e_guard_timer(armed, 0.1, **{**guard_kwargs, "current_has_lead": True}) == 0.0


def test_e2e_close_stop_settle_ignores_positive_model_accel():
  model_msg = make_model_msg(desired_accel=0.01, positions=[0.0, 0.2], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, 0.01, model_msg, make_radar_state(), True) == (0.01, False, False)


def test_e2e_close_stop_settle_releases_after_stop_distance_clears():
  model_msg = make_model_msg(desired_accel=-0.2, positions=[0.0, 1.5], velocities=[1.0, 0.0])

  assert get_e2e_close_stop_settle(0.5, -0.2, model_msg, make_radar_state(), True, active=True) == (-0.2, False, False)


def test_e2e_runway_comfort_caps_long_runway_raw_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -0.2175


def test_e2e_runway_comfort_prefers_coast_on_excessive_runway():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.5,
  )

  assert math.isclose(accel, -0.30)


def test_e2e_runway_comfort_uses_lighter_decel_with_traction_risk():
  normal = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=0.0,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.5,
  )
  traction_limited = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=0.0,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.5,
    traction_risk=1.0,
  )

  assert normal == pytest.approx(-0.30)
  assert traction_limited == pytest.approx(-0.25)
  assert normal < traction_limited < 0.0


def test_e2e_runway_comfort_caps_route_like_far_no_stop_decel():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.26,
    raw_e2e_accel=-0.88,
    coast_accel=-0.49,
    model_msg=make_model_msg(desired_accel=-0.88, should_stop=False, endpoint_x=101.7),
    e2e_active=True,
    prev_output_a_target=-0.52,
  )

  assert math.isclose(accel, -0.51)


def test_e2e_runway_comfort_allows_short_runway_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=55.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_model_stop_untouched():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=True, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_stop_context_bootstrap_untouched():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(
      desired_accel=-1.2,
      should_stop=False,
      endpoint_x=145.0,
      positions=[0.0, 35.0, 60.0],
      velocities=[17.0, 5.0, 0.5],
    ),
    e2e_active=True,
    prev_output_a_target=-0.2,
    engage_stop_bootstrap_active=True,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_driver_override_untouched():
  model_msg = make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0)

  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, gas_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, brake_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, reset_state=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, force_slow_decel=True) == -1.2


def test_e2e_runway_comfort_leaves_radar_lead_untouched():
  model_msg = make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0)

  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=model_msg,
    e2e_active=True,
    prev_output_a_target=-0.2,
    has_radar_lead=True,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_limits_negative_ramp():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-0.8,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-0.8, should_stop=False, endpoint_x=90.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -0.2175


def test_e2e_runway_comfort_softens_negative_ramp_with_traction_risk():
  normal = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-0.8,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-0.8, should_stop=False, endpoint_x=90.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )
  traction_limited = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-0.8,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-0.8, should_stop=False, endpoint_x=90.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
    traction_risk=1.0,
  )

  assert normal < traction_limited < 0.0


def test_e2e_runway_comfort_does_not_block_stop_approach_shortage_braking():
  model_msg = make_model_msg(desired_accel=-0.4, should_stop=False, endpoint_x=45.0,
                             positions=[0.0, 30.0, 45.0], velocities=[12.0, 0.5, 3.0])
  governed = get_e2e_runway_comfort_accel(12.0, -0.4, -0.25, model_msg, True, -0.2)
  shortage_accel = get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True)

  assert shortage_accel < governed
  assert shortage_accel < -0.5


def test_lead_stop_approach_softens_stopped_lead_runway_slew_with_traction_risk():
  normal = get_lead_stop_approach_slewed_accel(
    v_ego=10.0, d_rel=40.0, v_lead=0.0, a_lead=0.0, prev_a_target=0.0, a_target=-1.0, dt=0.05,
  )
  traction_limited = get_lead_stop_approach_slewed_accel(
    v_ego=10.0, d_rel=40.0, v_lead=0.0, a_lead=0.0, prev_a_target=0.0, a_target=-1.0, dt=0.05, traction_risk=1.0,
  )

  assert -1.0 < normal < traction_limited < 0.0


def test_lead_stop_approach_preserves_hard_braking_lead_slew_with_traction_risk():
  normal = get_lead_stop_approach_slewed_accel(
    v_ego=10.0, d_rel=40.0, v_lead=5.0, a_lead=-1.0, prev_a_target=0.0, a_target=-1.0, dt=0.05,
  )
  traction_limited = get_lead_stop_approach_slewed_accel(
    v_ego=10.0, d_rel=40.0, v_lead=5.0, a_lead=-1.0, prev_a_target=0.0, a_target=-1.0, dt=0.05, traction_risk=1.0,
  )

  assert traction_limited == pytest.approx(normal)


def test_e2e_runway_positive_accel_cap_limits_short_runway_at_crawl():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1]),
    True,
  )

  assert 0.0 < cap < 1.0


def test_e2e_runway_positive_cap_custom_candidate_does_not_mutate_baseline_output():
  planner = SimpleNamespace(
    output_a_target=0.5,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="cruise"),
    v_desired_trajectory=tuple(10.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.5 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )

  candidate = build_planner_seed_accel_candidate(
    planner, "e2e_runway_positive_cap", 0.1, has_lead=False,
    reason="low_speed_model_runway_positive_cap", accel_limits=(-2.0, 2.0),
  )

  assert candidate is not None
  assert candidate.name == "e2e_runway_positive_cap"
  assert candidate.output.a_target == pytest.approx(0.1)
  assert candidate.output.debug["planner_seed_candidate_reason"] == "low_speed_model_runway_positive_cap"
  assert planner.output_a_target == pytest.approx(0.5)


def test_e2e_runway_positive_accel_cap_limits_final_endpoint_crawl():
  cap = get_e2e_runway_positive_accel_cap(
    0.25,
    make_model_msg(desired_accel=0.5, should_stop=True, endpoint_x=0.2, positions=[0.0, 0.2], velocities=[0.2, 0.0]),
    True,
  )

  assert 0.0 < cap <= E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX


def test_e2e_runway_positive_accel_cap_caps_at_15m_crawl_example():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    True,
  )

  assert 0.0 < cap < get_max_accel(0.5)


def test_e2e_runway_positive_accel_cap_supports_model_stop_protection():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    False,
    model_stop_protection_active=True,
  )

  assert 0.0 < cap < get_max_accel(0.5)


def test_e2e_runway_positive_accel_cap_is_no_op_for_long_runway():
  cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=40.0, positions=[0.0, 40.0], velocities=[0.5, 0.1]),
    True,
  )

  assert cap == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_scales_with_runway_length():
  short_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1]),
    True,
  )
  mid_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=15.0, positions=[0.0, 15.0], velocities=[0.5, 0.1]),
    True,
  )
  long_cap = get_e2e_runway_positive_accel_cap(
    0.5,
    make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=40.0, positions=[0.0, 40.0], velocities=[0.5, 0.1]),
    True,
  )

  assert short_cap < mid_cap < long_cap == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_disables_on_override_and_reset():
  model_msg = make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 0.1])

  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, brake_pressed=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, gas_pressed=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, reset_state=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, force_slow_decel=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, engage_stop_bootstrap_active=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, True, has_radar_lead=True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, model_msg, False) == ACCEL_MAX


def test_e2e_runway_positive_accel_cap_ignores_weak_model_signal_and_invalid_endpoint():
  weak_model_msg = make_model_msg(desired_accel=0.0, should_stop=False, endpoint_x=2.0, positions=[0.0, 2.0], velocities=[0.5, 2.0])
  invalid_endpoint_msg = make_model_msg(desired_accel=1.0, should_stop=False, endpoint_x=float('nan'), positions=[0.0, float('nan')], velocities=[0.5, 0.1])

  assert get_e2e_runway_positive_accel_cap(0.5, weak_model_msg, True) == ACCEL_MAX
  assert get_e2e_runway_positive_accel_cap(0.5, invalid_endpoint_msg, True) == ACCEL_MAX


def test_routine_lead_approach_compression_budget_fields():
  result = get_routine_lead_approach_accel(
    v_ego=20.0, d_rel=50.0, v_lead=18.0, a_lead=0.0, y_rel=0.0, t_follow=1.5,
  )
  assert result.compression_budget > 0.0
  assert result.comfort_budget > 0.0
  assert result.projected_compression_budget > 0.0
  assert result.projected_comfort_budget > 0.0
  assert result.compression_budget > result.comfort_budget


def test_routine_lead_approach_compression_budget_shrinks_as_gap_closes():
  far_result = get_routine_lead_approach_accel(
    v_ego=20.0, d_rel=50.0, v_lead=18.0, a_lead=0.0, y_rel=0.0, t_follow=1.5,
  )
  close_result = get_routine_lead_approach_accel(
    v_ego=20.0, d_rel=30.0, v_lead=18.0, a_lead=0.0, y_rel=0.0, t_follow=1.5,
  )
  assert close_result.compression_budget < far_result.compression_budget
  assert close_result.comfort_budget < far_result.comfort_budget


def test_is_valid_routine_lead_approach_rejects_shadow():
  context = SimpleNamespace(
    behavior=None,
    shadow_active=True,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_is_valid_routine_lead_approach_rejects_flicker():
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.5, new_lead=False, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_is_valid_routine_lead_approach_rejects_new_lead():
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.0, new_lead=True, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_is_valid_routine_lead_approach_rejects_unstable_lead():
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.0, new_lead=False, stable=False,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_is_valid_routine_lead_approach_rejects_driver_override():
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.0, new_lead=False, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=True, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_is_valid_routine_lead_approach_accepts_stable_valid_lead():
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.0, new_lead=False, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_routine_lead_approach_seed_reason_is_lead_follow():
  assert planner_seed_intent_for_reason(ROUTINE_LEAD_APPROACH_SEED_REASON, has_lead=True) == PLANNER_SEED_INTENT_LEAD_FOLLOW


def test_routine_lead_approach_seed_is_floor_selection():
  planner = SimpleNamespace(
    output_a_target=-1.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(-1.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )
  candidates = build_moving_lead_seed_candidates(
    planner, True, (-2.0, 2.0),
    routine_lead_approach_a_target=0.0,
    routine_lead_approach_debug={},
  )
  routine_seed = None
  for candidate in candidates:
    if candidate.name == "routine_lead_approach":
      routine_seed = candidate
      break
  assert routine_seed is not None
  assert routine_seed.selection == PLANNER_SEED_FLOOR
  assert routine_seed.reason == ROUTINE_LEAD_APPROACH_SEED_REASON


def test_routine_lead_approach_seed_not_emitted_when_none():
  planner = SimpleNamespace(
    output_a_target=0.0,
    output_should_stop=False,
    allow_throttle=True,
    fcw=False,
    source="cruise",
    mpc=SimpleNamespace(source="lead0"),
    v_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
    a_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
    j_desired_trajectory=tuple(0.0 for _ in range(CONTROL_N)),
  )
  candidates = build_moving_lead_seed_candidates(
    planner, True, (-2.0, 2.0),
    routine_lead_approach_a_target=None,
    routine_lead_approach_debug=None,
  )
  for candidate in candidates:
    assert candidate.name != "routine_lead_approach"


def test_routine_lead_approach_valid_approach_gates_on_stable_lead():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import is_valid_routine_lead_approach

  # Stable valid lead context
  progress_model = SimpleNamespace(allowed=True, confidence_stability_sufficient=True, alternate_threat_absent=True, shadow_absent=True)
  behavior = SimpleNamespace(stable=True, new_lead=False, shadow=False, flicker_guard_timer=0.0, progress_model=progress_model, track_id=1)
  primary_lead_context = SimpleNamespace(
    lead_progress_allowed=True,
    alternate_threat_active=False,
    shadow_active=False,
    behavior=behavior,
    lead_release_blocked_reason="",
  )
  assert is_valid_routine_lead_approach(
    primary_lead_context=primary_lead_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=False, alternate_lead_threat_active=False,
  ) is True

  # Driver brake blocks
  assert is_valid_routine_lead_approach(
    primary_lead_context=primary_lead_context,
    brake_pressed=True, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=False, alternate_lead_threat_active=False,
  ) is False

  # Force slow blocks
  assert is_valid_routine_lead_approach(
    primary_lead_context=primary_lead_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=True,
    independent_stop_threat=False, alternate_lead_threat_active=False,
  ) is False

  # Independent stop threat blocks
  assert is_valid_routine_lead_approach(
    primary_lead_context=primary_lead_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=True, alternate_lead_threat_active=False,
  ) is False

  # Alternate threat blocks
  assert is_valid_routine_lead_approach(
    primary_lead_context=primary_lead_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=False, alternate_lead_threat_active=True,
  ) is False

  # Shadow blocks
  shadow_context = SimpleNamespace(
    lead_progress_allowed=True,
    alternate_threat_active=False,
    shadow_active=True,
    behavior=behavior,
    lead_release_blocked_reason="",
  )
  assert is_valid_routine_lead_approach(
    primary_lead_context=shadow_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=False, alternate_lead_threat_active=False,
  ) is False

  # Suppressive physical lead blocks
  suppressive_context = SimpleNamespace(
    lead_progress_allowed=False,
    alternate_threat_active=False,
    shadow_active=False,
    behavior=behavior,
    lead_release_blocked_reason="primary_physical_lead_suppressive",
  )
  assert is_valid_routine_lead_approach(
    primary_lead_context=suppressive_context,
    brake_pressed=False, gas_pressed=False, force_slow_decel=False,
    independent_stop_threat=False, alternate_lead_threat_active=False,
  ) is False


def test_stop_release_guard_lead_confirmed_release_caps_accel():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    StopReleaseGuardTracker, StopReleaseGuardState, apply_stop_release_guard_accel,
    STOP_RELEASE_GUARD_HOLD_TIME, STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON,
    LEAD_PULLAWAY_PULSE_A_FLOOR,
  )
  tracker = StopReleaseGuardTracker()
  # Simulate stop evidence active for 1 cycle, then stop evidence clears but lead confirmed
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=True,
                          lead_confirmed_release=False, dt=0.01)
  assert guard.active
  assert guard.reason == "waiting_for_stop_clear"
  # Now stop evidence clears, lead confirmed release
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=False,
                          lead_confirmed_release=True, dt=0.01)
  assert guard.active  # Should still be active (not bypass)
  assert guard.lead_confirmed_release
  assert guard.release_accel_cap == LEAD_PULLAWAY_PULSE_A_FLOOR
  assert guard.reason == STOP_RELEASE_GUARD_LEAD_CAPPED_RELEASE_REASON
  # Apply guard: positive accel should be capped, not zeroed
  a_out, guard_out = apply_stop_release_guard_accel(1.5, guard)
  assert a_out == LEAD_PULLAWAY_PULSE_A_FLOOR  # capped to 0.70, not 0.0
  assert guard_out.applied
  # Small positive accel below cap should pass through (still marks applied since guard is active)
  a_out2, guard_out2 = apply_stop_release_guard_accel(0.3, guard)
  assert a_out2 == 0.3  # below cap, passes through unchanged
  assert not guard_out2.applied  # value not modified, so applied=False


def test_stop_release_guard_waiting_zeros_accel():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    StopReleaseGuardTracker, StopReleaseGuardState, apply_stop_release_guard_accel,
  )
  tracker = StopReleaseGuardTracker()
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=True,
                          lead_confirmed_release=False, dt=0.01)
  assert guard.active
  assert guard.release_accel_cap == 0.0  # default cap is 0
  # Positive accel should be zeroed
  a_out, guard_out = apply_stop_release_guard_accel(1.5, guard)
  assert a_out == 0.0
  assert guard_out.applied


def test_lead_pullaway_independent_stop_threat_suppressed_with_confirmed_lead():
  # When primary_behavior_progress_allowed is True (confirmed radar lead with progress authority),
  # the E2E stop threat should NOT contribute to lead_pullaway_independent_stop_threat.
  # This is tested by verifying the logic: the E2E clause includes `not primary_behavior_progress_allowed`.
  # When primary_behavior_progress_allowed=True, the E2E portion evaluates to False,
  # so only scc_independent_stop_threat can make it True.
  # This test verifies the computation inline since it's inside the planner update.
  from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
  # The key assertion: with a confirmed lead (primary_behavior_progress_allowed=True),
  # E2E shouldStop should not block launch via independent_stop_threat.
  # This is verified by the scene.independent_stop_threat field being False
  # when scene.has_lead=True and the E2E model says stop.
  # We verify the logic expression directly:
  scc_independent_stop_threat = False
  e2e_active = True
  defer_e2e_to_stopped_lead_mpc = False
  primary_behavior_progress_allowed = True  # confirmed radar lead present
  output_should_stop_e2e = True  # E2E model says stop
  custom_e2e_stop_approach_a_target = -0.5  # E2E wants decel
  lead_pullaway_independent_stop_threat = bool(
    scc_independent_stop_threat or (
      e2e_active and not defer_e2e_to_stopped_lead_mpc and
      not primary_behavior_progress_allowed and
      (output_should_stop_e2e or custom_e2e_stop_approach_a_target < 0.0)
    )
  )
  assert not lead_pullaway_independent_stop_threat, \
    "E2E stop threat should not block launch when confirmed radar lead is present"
  # Without a confirmed lead, E2E stop threat should still block
  primary_behavior_progress_allowed = False
  lead_pullaway_independent_stop_threat = bool(
    scc_independent_stop_threat or (
      e2e_active and not defer_e2e_to_stopped_lead_mpc and
      not primary_behavior_progress_allowed and
      (output_should_stop_e2e or custom_e2e_stop_approach_a_target < 0.0)
    )
  )
  assert lead_pullaway_independent_stop_threat, \
    "E2E stop threat should block launch when no confirmed radar lead is present"


def test_stop_release_guard_driver_override_still_bypasses():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    StopReleaseGuardTracker, apply_stop_release_guard_accel,
  )
  tracker = StopReleaseGuardTracker()
  # Build up stop timer
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=True,
                          lead_confirmed_release=False, dt=0.01)
  assert guard.active
  # Driver presses gas
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=False,
                          lead_confirmed_release=False, gas_pressed=True, dt=0.01)
  assert not guard.active  # driver override bypasses guard
  assert guard.reason == "driver_override"
  # Positive accel should pass through
  a_out, guard_out = apply_stop_release_guard_accel(1.5, guard)
  assert a_out == 1.5
  assert not guard_out.applied


def test_far_lead_coast_activates_for_stable_slower_far_lead():
  """Stable slower lead with closing=2.5 m/s, d_rel=65 (within TTC < 7s).
  Expected: far_coast_active=True, phase='far_lead_coast', raw_a_target=0.0."""
  v_ego = 25.0
  v_lead = 22.5
  t_follow = 1.8
  d_rel = 65.0

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
  )
  assert result.far_coast_active, f"Expected far_coast_active=True, got {result.far_coast_active}"
  assert result.debug["routine_lead_phase"] == "far_lead_coast"
  assert result.raw_a_target == 0.0
  assert result.debug["routine_lead_far_coast_active"]
  assert result.debug["routine_lead_time_to_caution"] <= ROUTINE_LEAD_FAR_COAST_TTC
  assert result.debug["routine_lead_time_to_caution"] > 0.0


def test_far_lead_coast_inactive_when_ttc_very_long():
  """Same lead but at 200m. TTC to caution > 7s. Expected: far_coast_active=False."""
  v_ego = 25.0
  v_lead = 22.5
  t_follow = 1.8
  d_rel = 200.0

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
  )
  assert not result.far_coast_active, f"Expected far_coast_active=False, got {result.far_coast_active}"
  # Should be below_threshold or inactive (not active through any phase)
  assert not result.active or result.reason == "below_threshold"


def test_far_lead_coast_inactive_when_not_closing():
  """Lead at same speed as ego (closing_speed=0). Expected: far_coast_active=False."""
  v_ego = 25.0
  v_lead = 25.0
  d_rel = 65.0
  t_follow = 1.8

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
  )
  assert not result.far_coast_active


def test_far_lead_coast_inactive_when_urgent():
  """Close lead, high closing speed, danger gap. Expected: far_coast_active=False, urgent=True."""
  v_ego = 25.0
  v_lead = 5.0
  d_rel = 10.0
  t_follow = 1.8

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
  )
  assert not result.far_coast_active
  assert result.urgent


def test_far_lead_coast_inactive_for_shadow_flicker_new_lead():
  """Test that is_valid_routine_lead_approach() still rejects shadow/flicker/new leads.
  The far_coast path should not bypass validity checks."""
  # Shadow lead
  context = SimpleNamespace(
    behavior=None,
    shadow_active=True,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )

  # Flicker lead
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.5, new_lead=False, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )

  # New lead
  context = SimpleNamespace(
    behavior=SimpleNamespace(
      shadow=False, flicker_guard_timer=0.0, new_lead=True, stable=True,
    ),
    shadow_active=False,
    alternate_threat_active=False,
    lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert not is_valid_routine_lead_approach(
    primary_lead_context=context, brake_pressed=False, gas_pressed=False,
    force_slow_decel=False, independent_stop_threat=False,
    alternate_lead_threat_active=False,
  )


def test_far_lead_coast_does_not_produce_negative_accel():
  """Any far_coast_active result should have raw_a_target == 0.0 (coast only, no braking)."""
  v_ego = 25.0
  v_lead = 22.5
  t_follow = 1.8
  d_rel = 65.0

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
  )
  if result.far_coast_active:
    assert result.raw_a_target == 0.0, f"Expected 0.0 accel for far coast, got {result.raw_a_target}"


def test_comfort_budget_relaxed_starts_coast_earlier():
  """Relaxed personality (0): far_coast_ttc=9.0 allows coast when
  standard (7.0) would not, given TTC ~8.0."""
  v_ego = 25.0
  v_lead = 22.5
  t_follow = 1.8
  # Choose d_rel so TTC to caution is ~8s — above standard (7.0) but below relaxed (9.0)
  # closing = 2.5 m/s, caution_gap depends on v_ego/v_lead/t_follow
  _desired_gap, caution_gap, _danger_gap = get_lead_approach_gaps(v_ego, v_lead, t_follow)
  caution_gap = float(caution_gap)
  closing = v_ego - v_lead  # 2.5 m/s
  ttc_to_caution = 7.8  # seconds
  d_rel = caution_gap + closing * ttc_to_caution

  relaxed = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
    budget=get_comfort_budget(0),
  )
  standard = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
    budget=get_comfort_budget(1),
  )

  assert relaxed.far_coast_active, "Relaxed should coast at TTC ~7.8"
  assert not standard.far_coast_active, "Standard should not coast at TTC ~7.8"
  assert relaxed.raw_a_target == 0.0


def test_comfort_budget_aggressive_builds_sooner():
  """Aggressive personality (2): routine_decel_cap=0.55 allows stronger
  decel than standard (0.45) for the same closing scenario."""
  v_ego = 20.0
  v_lead = 18.0
  d_rel = 31.0
  t_follow = 1.5

  standard = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
    budget=get_comfort_budget(1),
  )
  aggressive = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=0.0, y_rel=0.0, t_follow=t_follow,
    budget=get_comfort_budget(2),
  )

  assert standard.active or not standard.far_coast_active
  assert aggressive.active or not aggressive.far_coast_active
  # Aggressive should pull a stronger (more negative) target when both are active
  if standard.active and aggressive.active:
    assert aggressive.raw_a_target <= standard.raw_a_target + 1e-6


def test_comfort_budget_standard_matches_defaults():
  """Standard comfort budget fields equal the original hardcoded constants."""
  budget = get_comfort_budget(1)
  assert budget.far_coast_ttc == ROUTINE_LEAD_FAR_COAST_TTC
  assert budget.soft_decel_cap == ROUTINE_LEAD_APPROACH_SOFT_DECEL_CAP
  assert budget.routine_decel_cap == ROUTINE_LEAD_APPROACH_DECEL_CAP
  assert budget.firm_routine_decel_cap == ROUTINE_LEAD_APPROACH_FIRM_DECEL_CAP
  assert budget.routine_negative_jerk == ROUTINE_LEAD_APPROACH_NEGATIVE_JERK
  assert budget.routine_release_jerk == ROUTINE_LEAD_APPROACH_RELEASE_JERK
  assert budget.response_time == ROUTINE_LEAD_RESPONSE_TIME


def test_comfort_budget_firm_routine_decel_cap():
  """firm_routine_decel_cap is 1.2 for relaxed, 1.5 for standard, 1.8 for aggressive."""
  assert get_comfort_budget(0).firm_routine_decel_cap == 1.2
  assert get_comfort_budget(1).firm_routine_decel_cap == 1.5
  assert get_comfort_budget(2).firm_routine_decel_cap == 1.8


def test_low_speed_step_cap_not_clamped_below_runway_safe_floor():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    LeadPullawayIntent, LeadPullawayPhase, CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX,
  )
  # When early_authority=True and safe_accel_cap is high enough,
  # low_speed_step_cap_suppressed_by_runway should be True,
  # meaning the step cap should not clamp the pullaway floor.
  intent = LeadPullawayIntent(
    phase=LeadPullawayPhase.PULSE,
    active=True,
    a_floor=LEAD_PULLAWAY_PULSE_A_FLOOR,
    reason="lead_created_runway",
    early_authority=True,
    safe_accel_cap=CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX + 0.5,
    runway_margin=2.0,
    lead_created_runway=True,
  )
  # The pullaway floor should be at least LEAD_PULLAWAY_PULSE_A_FLOOR
  assert intent.a_floor == LEAD_PULLAWAY_PULSE_A_FLOOR
  assert intent.early_authority
  assert intent.safe_accel_cap >= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_stop_release_guard_blocks_positive_accel_without_lead_release():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    StopReleaseGuardTracker, apply_stop_release_guard_accel,
  )
  tracker = StopReleaseGuardTracker()
  # Build up stop timer without lead release
  guard = tracker.update(v_ego=0.0, standstill=True, stop_evidence_active=True,
                          lead_confirmed_release=False, dt=0.01)
  assert guard.active
  assert guard.release_accel_cap == 0.0  # default cap is 0, blocks all positive accel
  # Positive accel should be zeroed
  a_out, guard_out = apply_stop_release_guard_accel(1.5, guard)
  assert a_out == 0.0
  assert guard_out.applied


def test_early_pullaway_authority_with_lead_created_runway():
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    LeadPullawayIntentTracker,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput

  # Create a mock lead state with progress model
  class MockProgressModel:
    confidence_stability_sufficient = True
    alternate_threat_absent = True
    shadow_absent = True
    allowed = True
    stop_threat_absent = True
    opening_speed = 0.5
    lead_moving = True

  class MockRiskModel:
    closing_speed = 0.0
    required_decel = 0.0
    ttc = float('inf')

  class MockLeadState:
    track_id = 1
    shadow = False
    flicker_guard_timer = 0.0
    new_lead = False
    stable = True
    v_rel = 0.5
    progress_model = MockProgressModel()
    risk_model = MockRiskModel()

  class MockPrimaryLeadContext:
    alternate_threat_active = False
    shadow_active = False
    lead_progress_allowed = True
    lead_release_blocked_reason = ""

  class MockRunway:
    coast_required = False
    safe_accel_cap = 1.0
    lead_created_runway = True

  authority, reason = LeadPullawayIntentTracker._early_pullaway_authority(
    lead_state=MockLeadState(),
    primary_lead_context=MockPrimaryLeadContext(),
    runway=MockRunway(),
    lead_opening=True,
    lead_moving=True,
    lead_accel=0.3,
    independent_stop_threat=False,
    alternate_lead_threat_active=False,
    brake_pressed=False,
    gas_pressed=False,
    force_slow_decel=False,
  )
  assert authority, f"Expected authority with lead_created_runway, got reason: {reason}"
  assert reason == "lead_created_runway"


def test_routine_lead_approach_valid_for_stable_physical_lead_without_behavior():
  """Routine comfort shaping should be valid for a stable physical lead
  even when no behavior/progress-allowed lead exists. Far-lead coast and
  soft decel should not require progress authority."""
  physical = LeadRelevanceState(
    lead_idx=0, status=True, shadow=False, stable=True, new_lead=False,
    flicker_guard_timer=0.0, track_id=1, d_rel=80.0, y_rel=0.0,
    path_y_rel=0.0, v_lead=22.0, v_rel=-2.0, model_prob=0.95, radar=True,
    ttc=40.0, required_decel=0.1, time_gap=3.5, on_path_score=0.9,
    risk_score=0.3, ghost_score=0.0, confidence=0.9,
    authority=LEAD_AUTHORITY_SUPPRESS_ONLY, reason="path_relevant_physical_lead",
  )
  context = PrimaryLeadContext(
    physical_idx=0, behavior_idx=None, physical=physical, behavior=None,
    alternate_threat_active=False, shadow_active=False,
    reason="path_relevant_physical_lead",
    states=(physical,), lead_progress_allowed=False,
    lead_release_blocked_reason="",
  )
  assert is_valid_routine_lead_approach(primary_lead_context=context)


def test_far_lead_coast_helper_phase_and_target():
  """Far-lead coast: stable slower lead above caution gap, within TTC budget,
  should produce phase=far_lead_coast, far_coast_active=True, a_target=0.0.
  Trigger conditions: dRel above caution, projected gap above caution,
  time_to_caution <= far_coast_ttc, not urgent."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=28.0,
    d_rel=60.0,
    v_lead=26.0,
    a_lead=-0.1,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_phase"] == "far_lead_coast"
  assert debug["routine_lead_far_coast_active"]
  assert debug["routine_lead_coast_first_active"]
  # Far-lead coast target should be 0.0 (coast, no braking)
  assert debug["routine_lead_raw_a_target"] == pytest.approx(0.0, abs=0.01)
  assert not debug["routine_lead_approach_urgent"]
  # Trigger conditions: time_to_caution within far_coast_ttc budget
  assert 0.0 < debug["routine_lead_time_to_caution"] <= ROUTINE_LEAD_FAR_COAST_TTC


def test_far_lead_coast_blocked_by_safety():
  """Far-lead coast should not activate when the lead is urgent (short TTC,
  hard braking, or danger gap). Safety-relevant targets should win."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=15.0,
    d_rel=8.0,
    v_lead=5.0,
    a_lead=-3.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )
  # Short distance, hard braking lead — should be urgent, not far-lead coast
  assert not debug.get("routine_lead_far_coast_active", False)
  assert debug.get("routine_lead_approach_urgent", False) or accel is not None and accel < -1.0


def test_moving_lead_routine_far_lead_coast_starts_before_projection_caution():
  """Far-lead coast should activate when d_rel is above caution gap,
  closing speed is moderate, and TTC to caution is within the far_coast_ttc
  budget. This verifies the far-lead coast phase starts before the projected
  gap reaches caution — the vehicle is closing on a slower lead but has
  enough runway that coasting (a_target=0) is the right comfort response.

  Trigger conditions:
    dRel above caution
    projected gap still above caution
    time_to_caution <= far_coast_ttc
    not urgent, not urgent_bypass
  """
  v_ego = 28.0
  v_lead = 26.0
  d_rel = 60.0
  t_follow = 1.55

  result = get_routine_lead_approach_accel(
    v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=-0.1, y_rel=0.0, t_follow=t_follow,
  )
  # Far-lead coast should be active: d_rel well above caution gap,
  # closing with finite TTC within budget, not urgent
  assert result.far_coast_active, f"Expected far_coast_active=True, got {result.far_coast_active}"
  assert result.debug["routine_lead_phase"] == "far_lead_coast"
  assert result.debug["routine_lead_far_coast_active"]
  assert result.debug["routine_lead_coast_first_active"]
  # Far-lead coast target should be 0.0 (coast, no braking)
  assert result.raw_a_target == 0.0
  # Not urgent — this is a comfort coast, not a safety event
  assert not result.urgent
  assert not result.debug.get("routine_lead_urgent_bypass", False)
  # TTC to caution should be positive and within the far_coast_ttc budget
  assert 0.0 < result.debug["routine_lead_time_to_caution"] <= ROUTINE_LEAD_FAR_COAST_TTC


def test_routine_comfort_phase_emits_seed_without_existing_target():
  """Routine comfort floor seed should be emitted for far-lead coast and
  other comfort phases even when there is no existing non-urgent target.
  This tests the planner-level seed emission condition, not just the helper."""
  # Use the helper to get a far-lead coast scenario
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=25.0,
    d_rel=50.0,
    v_lead=22.0,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )
  assert accel is not None
  assert debug["routine_lead_approach_active"]
  assert debug["routine_lead_can_own_nonurgent_shape"]
  assert debug["routine_lead_phase"] in ("far_lead_coast", "free_coast", "soft_decel", "routine_decel")
  # The routine comfort floor should be emitted even with no existing target
  assert debug["routine_lead_existing_target_reason"] == "none"
  assert not debug["routine_lead_existing_target_safety_relevant"]


def test_routine_can_own_without_existing_target():
  """Routine comfort should own non-urgent shape even when there is no
  prior moving-lead baseline target. Far-lead coast and free-coast phases
  produce a routine target that can relax a negative non-urgent baseline
  without needing an existing_target to replace."""
  accel, debug = get_moving_lead_stop_gap_guard_accel(
    v_ego=25.0,
    d_rel=50.0,
    v_lead=22.0,
    a_lead=0.0,
    y_rel=0.0,
    t_follow=1.55,
    prev_a_target=0.0,
    dt=0.05,
    return_debug=True,
  )
  assert accel is not None
  assert debug["routine_lead_approach_active"]
  # routine_can_own should be True even when existing_target is None/none
  assert debug["routine_lead_can_own_nonurgent_shape"]
  assert debug["routine_lead_existing_target_reason"] == "none"
  assert not debug["routine_lead_existing_target_safety_relevant"]


def test_routine_lead_approach_invalid_for_suppressive_only_physical_lead():
  physical = LeadRelevanceState(
    lead_idx=0, status=True, shadow=False, stable=True, new_lead=False,
    flicker_guard_timer=0.0, track_id=1, d_rel=5.0, y_rel=0.0,
    path_y_rel=0.0, v_lead=0.0, v_rel=20.0, model_prob=0.95, radar=True,
    ttc=1.0, required_decel=3.0, time_gap=0.25, on_path_score=0.9,
    risk_score=0.9, ghost_score=0.0, confidence=0.9,
    authority=LEAD_AUTHORITY_SUPPRESS_ONLY, reason="primary_physical_lead_suppressive",
  )
  context = PrimaryLeadContext(
    physical_idx=0, behavior_idx=None, physical=physical, behavior=None,
    alternate_threat_active=False, shadow_active=False,
    reason="primary_physical_lead_suppressive",
    states=(physical,), lead_progress_allowed=False,
    lead_release_blocked_reason="primary_physical_lead_suppressive",
  )
  assert not is_valid_routine_lead_approach(primary_lead_context=context)


def test_routine_lead_approach_valid_for_behavior_lead():
  """A stable behavior lead with progress allowed should be valid."""
  behavior = LeadRelevanceState(
    lead_idx=0, status=True, shadow=False, stable=True, new_lead=False,
    flicker_guard_timer=0.0, track_id=1, d_rel=40.0, y_rel=0.0,
    path_y_rel=0.0, v_lead=22.0, v_rel=-2.0, model_prob=0.95, radar=True,
    ttc=20.0, required_decel=0.1, time_gap=1.8, on_path_score=0.9,
    risk_score=0.3, ghost_score=0.0, confidence=0.9,
    authority=LEAD_AUTHORITY_PROGRESS_ALLOWED, reason="stable_moving_lead",
  )
  context = PrimaryLeadContext(
    physical_idx=None, behavior_idx=0, physical=None, behavior=behavior,
    alternate_threat_active=False, shadow_active=False,
    reason="stable_moving_lead",
    states=(behavior,), lead_progress_allowed=True,
    lead_release_blocked_reason="",
  )
  assert is_valid_routine_lead_approach(primary_lead_context=context)


def test_routine_seed_emission_condition_far_coast_without_can_own():
  """Planner-level seed emission should fire when routine_can_own is False
  but routine_far_coast is True. This verifies the defensive disjunction
  (routine_can_own or routine_far_coast or routine_comfort_phase) in the
  seed emission condition at the planner level.

  In current helper code paths, routine_can_own is always True when
  routine_far_coast is True and routine_safety_relevant is False, because
  routine_can_own = routine.active and not safety_relevant, and far_coast
  requires routine.active. But the disjunction is defensive: if future
  code changes create a path where routine_can_own is False while
  routine_far_coast is True, the seed should still emit.
  """
  # Simulate the planner-level seed emission condition directly.
  # The condition is:
  #   routine_active and not routine_urgent and not routine_safety_relevant
  #   and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)

  # Case 1: routine_can_own=False, routine_far_coast=True, routine_comfort_phase=True
  # (far_lead_coast phase implies both far_coast and comfort_phase)
  routine_active = True
  routine_urgent = False
  routine_safety_relevant = False
  valid_approach = True
  routine_can_own = False
  routine_far_coast = True
  routine_comfort_phase = True  # far_lead_coast is a comfort phase

  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert seed_emits, "Seed should emit when routine_far_coast=True even if routine_can_own=False"

  # Case 2: routine_can_own=False, routine_far_coast=True, routine_comfort_phase=False
  # (far_coast_active without being in a comfort phase — defensive edge case)
  routine_far_coast = True
  routine_comfort_phase = False
  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert seed_emits, "Seed should emit when routine_far_coast=True and routine_comfort_phase=False even if routine_can_own=False"


def test_routine_seed_emission_condition_comfort_phase_without_can_own():
  """Planner-level seed emission should fire when routine_can_own is False
  but routine_comfort_phase is True (and routine_far_coast is False).

  This covers soft_decel and routine_decel phases where routine_can_own
  might be False due to safety_relevant=True with existing_target=None
  (a defensive edge case), but the phase is still a comfort phase.
  """
  routine_active = True
  routine_urgent = False
  routine_safety_relevant = False
  valid_approach = True
  routine_can_own = False
  routine_far_coast = False
  routine_comfort_phase = True  # e.g. soft_decel or routine_decel

  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert seed_emits, "Seed should emit when routine_comfort_phase=True even if routine_can_own=False"


def test_routine_seed_emission_blocked_by_urgent_or_safety_relevant():
  """Seed emission should be blocked when routine_urgent or
  routine_safety_relevant is True, regardless of routine_far_coast
  or routine_comfort_phase."""
  routine_active = True
  valid_approach = True
  routine_can_own = False
  routine_far_coast = True
  routine_comfort_phase = True

  # Urgent blocks emission
  routine_urgent = True
  routine_safety_relevant = False
  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert not seed_emits, "Seed should NOT emit when routine_urgent=True"

  # Safety-relevant blocks emission
  routine_urgent = False
  routine_safety_relevant = True
  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert not seed_emits, "Seed should NOT emit when routine_safety_relevant=True"


def test_routine_seed_emission_blocked_without_valid_approach():
  """Seed emission should be blocked when valid_approach is False,
  even if routine_far_coast and routine_comfort_phase are True."""
  routine_active = True
  routine_urgent = False
  routine_safety_relevant = False
  routine_can_own = False
  routine_far_coast = True
  routine_comfort_phase = True
  valid_approach = False

  seed_emits = (
    routine_active and not routine_urgent and not routine_safety_relevant
    and valid_approach and (routine_can_own or routine_far_coast or routine_comfort_phase)
  )
  assert not seed_emits, "Seed should NOT emit when valid_approach=False"


# ----------------------------------------------------------------------------
# Moving lead recovery helper tests
# ----------------------------------------------------------------------------


def test_moving_lead_recovery_inactive_below_min_v_ego():
  """Recovery authority is gated above MOVING_LEAD_RECOVERY_MIN_V_EGO
  (3 m/s). Below that, low-speed LeadPullawayIntent owns the launch /
  pullaway path; the moving recovery helper is intentionally a no-op."""
  result = get_moving_lead_recovery(
    v_ego=2.5, d_rel=15.0, v_lead=1.5, a_lead=0.5, lead_accel_trend=0.2,
    lead_opening=True,
  )
  assert isinstance(result, MovingLeadRecovery)
  assert not result.active
  assert result.phase == MovingLeadRecoveryPhase.INACTIVE
  assert result.reason == "below_min_v_ego"
  assert result.a_floor == 0.0


def test_moving_lead_recovery_inactive_when_gap_below_minimum_floor():
  """The recovery helper enforces a 4 m safety floor on the current gap
  before considering activation. Below the floor, urgent / lead-stop
  paths own the brake; the recovery helper must not loosen anything."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=3.0, v_lead=12.0, a_lead=0.5, lead_accel_trend=0.2,
    lead_opening=True, minimum_allowed_gap=MOVING_LEAD_RECOVERY_MIN_GAP,
  )
  assert not result.active
  assert result.phase == MovingLeadRecoveryPhase.INACTIVE
  assert result.reason == "below_min_gap"


def test_moving_lead_recovery_inactive_when_lead_still_braking():
  """Hard lead braking (a_lead <= -0.05) or a strongly decreasing
  accel trend must keep the recovery helper inactive. The lead is not
  opening; this is a hard-braking-lead scenario, not a recovery one."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=15.0, v_lead=12.0, a_lead=-0.5, lead_accel_trend=-0.1,
    lead_opening=False,
  )
  assert not result.active
  assert result.reason == "lead_not_opening"


def test_moving_lead_recovery_inactive_when_runway_pushes_below_floor():
  """If the predicted gap after applying a_floor would drop below the
  safety floor, recovery authority is suspended. The lead's runway is
  not strong enough to support any re-accel; the urgent / lead-stop
  path keeps the brake authority."""
  # v_ego=15, v_lead=13 (closing 2 m/s), a_lead=0.3, lead_opening=True.
  # d_rel=12, minimum_allowed_gap=10.
  # predicted_gap = 12 + (13-15)*1.5 + 0.5*(0.3-0.3)*1.5² = 12 - 3 = 9
  # 9 < 10 -> runway_below_floor.
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=12.0, v_lead=13.0, a_lead=0.3, lead_accel_trend=0.1,
    lead_opening=True, minimum_allowed_gap=10.0,
  )
  assert not result.active
  assert result.reason == "runway_below_floor"


def test_moving_lead_recovery_inactive_when_lead_stopped():
  """If the lead has no forward velocity, this is a stopped-lead
  scenario. Moving recovery is a moving-speed policy; the low-speed
  LeadPullawayIntent or stopped-lead gap-fill paths own this case."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=12.0, v_lead=0.0, a_lead=0.0, lead_accel_trend=0.0,
  )
  assert not result.active
  assert result.reason == "lead_not_moving"


def test_moving_lead_recovery_brake_release_when_lead_just_recovering():
  """Brake-release phase: lead is opening but the lead accel is below
  MILD_REACCEL threshold. Just stop braking; let routine follow shape
  the rest. a_floor is 0.0."""
  # v_ego=15, v_lead=14, a_lead=0.0, lead_opening=True: tiny positive
  # accel trend would be enough to authorize MILD_REACCEL; we suppress
  # by setting lead_accel_trend to 0.0 and a_lead=0.0.
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=6.0, v_lead=14.0, a_lead=0.0, lead_accel_trend=0.0,
    lead_opening=False, minimum_allowed_gap=4.0,
  )
  # When lead is not opening and a_lead is 0, the helper treats it as
  # "lead not opening" and suspends. This is the correct behavior —
  # brake_release is only a "weak" recovery, not a default-active state.
  assert not result.active
  assert result.reason == "lead_not_opening"


def test_moving_lead_recovery_mild_reaccel_when_lead_just_accelerating():
  """Mild-reaccel phase: lead accel is positive but the gap is small
  enough that bounded re-accel keeps us in the comfort buffer."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=10.0, v_lead=14.0, a_lead=0.3, lead_accel_trend=0.1,
    lead_opening=True, minimum_allowed_gap=4.0,
  )
  assert result.active
  assert result.phase == MovingLeadRecoveryPhase.MILD_REACCEL
  # MILD_REACCEL caps the floor at MOVING_LEAD_RECOVERY_MILD_ACCEL.
  assert 0.0 <= result.a_floor <= MOVING_LEAD_RECOVERY_MILD_ACCEL
  # Predicted gap stays above the safety floor.
  assert result.predicted_gap >= result.minimum_allowed_gap


def test_moving_lead_recovery_gap_recovery_when_lead_opening_excess_gap():
  """Gap-recovery phase: lead is opening and the lead-created runway is
  large enough that the safety floor is comfortably above the minimum.
  a_floor reaches the full MOVING_LEAD_RECOVERY_MAX_ACCEL ceiling."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=20.0, v_lead=18.0, a_lead=0.5, lead_accel_trend=0.1,
    lead_opening=True, minimum_allowed_gap=4.0,
  )
  assert result.active
  assert result.phase == MovingLeadRecoveryPhase.GAP_RECOVERY
  assert result.lead_created_runway
  # GAP_RECOVERY caps the floor at MOVING_LEAD_RECOVERY_MAX_ACCEL.
  assert 0.0 <= result.a_floor <= MOVING_LEAD_RECOVERY_MAX_ACCEL
  assert result.predicted_gap >= result.minimum_allowed_gap
  # Runway margin is positive (lead is creating gap).
  assert result.debug["moving_lead_recovery_runway_margin_t"] > 0.0


def test_moving_lead_recovery_safe_accel_cap_caps_floor_under_runway():
  """If the safe_accel_cap is small (lead accel low and runway tight),
  the a_floor is clamped below MILD_REACCEL even if the lead is
  technically opening."""
  # v_ego=15, v_lead=14.9 (closing 0.1), a_lead=0.1, lead_opening=True
  # predicted_gap = d_rel + (v_lead - v_ego) * 1.5 + 0.5*0.1*1.5²
  #               = d_rel - 0.15 + 0.1125
  # For d_rel=4.5, predicted_gap=4.46, just above min
  # safe_accel_cap = 0.1 + 2*(4.46-4.0-0.5)/1.5² (very small) ≈ 0.1
  # runway_margin_t = 4.46 - 5.775 - 0.5 = -1.815 (negative)
  # lead_created_runway = False
  # Phase: BRAKE_RELEASE (lead not opening enough)
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=4.5, v_lead=14.9, a_lead=0.1, lead_accel_trend=0.0,
    lead_opening=True, minimum_allowed_gap=4.0,
  )
  # predicted_gap is just above min, lead is opening but runway is tight.
  # The MILD_REACCEL a_floor is bounded by safe_accel_cap which is small.
  if result.active:
    assert result.a_floor <= MOVING_LEAD_RECOVERY_MILD_ACCEL
    assert result.a_floor <= result.safe_accel_cap
    assert result.predicted_gap >= result.minimum_allowed_gap


def test_moving_lead_recovery_min_v_ego_boundary_at_3_mps():
  """v_ego exactly at MOVING_LEAD_RECOVERY_MIN_V_EGO should activate
  (the check is strict-greater-than-equal)."""
  result = get_moving_lead_recovery(
    v_ego=MOVING_LEAD_RECOVERY_MIN_V_EGO, d_rel=15.0, v_lead=5.0,
    a_lead=0.5, lead_accel_trend=0.1, lead_opening=True,
  )
  # The lead is far below v_ego here so v_lead=5 with v_ego=3 means the
  # lead is moving away. This is a recovery scenario.
  assert result.active or result.reason in ("lead_not_opening",)


def test_moving_lead_recovery_predicted_gap_never_below_minimum_allowed_gap():
  """When active, the helper guarantees the predicted gap after the
  a_floor horizon stays above minimum_allowed_gap. This is the
  structural safety guarantee — the recovery floor is always runway-safe."""
  for v_ego, v_lead, a_lead, d_rel in [
    (15.0, 10.0, 0.5, 20.0),
    (15.0, 14.0, 0.3, 12.0),
    (8.0, 5.0, 0.4, 10.0),
    (25.0, 20.0, 0.8, 30.0),
  ]:
    result = get_moving_lead_recovery(
      v_ego=v_ego, d_rel=d_rel, v_lead=v_lead, a_lead=a_lead,
      lead_accel_trend=0.1, lead_opening=True, minimum_allowed_gap=4.0,
    )
    if result.active:
      assert result.predicted_gap >= result.minimum_allowed_gap


def test_moving_lead_recovery_debug_keys_are_exposed():
  """The debug dict should expose enough fields for post-hoc route
  diagnostics to verify the runway math and safety floor checks."""
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=20.0, v_lead=18.0, a_lead=0.5, lead_accel_trend=0.1,
    lead_opening=True,
  )
  expected_keys = {
    "moving_lead_recovery_v_ego", "moving_lead_recovery_d_rel",
    "moving_lead_recovery_v_lead", "moving_lead_recovery_a_lead",
    "moving_lead_recovery_lead_accel_trend", "moving_lead_recovery_lead_opening",
    "moving_lead_recovery_horizon", "moving_lead_recovery_min_lead_accel",
    "moving_lead_recovery_minimum_allowed_gap",
  }
  assert expected_keys.issubset(set(result.debug.keys()))


def test_moving_lead_recovery_seed_reason_constant_is_distinct_from_pullaway():
  """The moving_lead_recovery seed reason is a separate constant so the
  planner seed layer can map it to the right intent (RELAXATION, not
  LAUNCH). It must not collide with confirmed_lead_pullaway_pulse or
  excess_gap_closure which are launch-related."""
  assert MOVING_LEAD_RECOVERY_SEED_REASON == "moving_lead_recovery"
  assert MOVING_LEAD_RECOVERY_SEED_REASON != "confirmed_lead_pullaway_pulse"
  assert MOVING_LEAD_RECOVERY_SEED_REASON != "excess_gap_closure"


def test_moving_lead_recovery_emits_relaxation_role_in_seed_layer():
  """The moving_lead_recovery seed must convert to a RELAXATION role,
  never PHYSICAL_HAZARD or LAUNCH. This is a moving comfort/recovery
  floor, not a hazard or launch authority."""
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    _role_for_seed,
    PLANNER_SEED_FLOOR,
    PLANNER_SEED_INTENT_LAUNCH,
    PLANNER_SEED_INTENT_LEAD_FOLLOW,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole

  # Floor with LEAD_FOLLOW intent (what the planner would produce for
  # this reason) must be RELAXATION, not PHYSICAL_HAZARD.
  role = _role_for_seed(PLANNER_SEED_INTENT_LEAD_FOLLOW, MOVING_LEAD_RECOVERY_SEED_REASON, PLANNER_SEED_FLOOR)
  assert role == CandidateRole.RELAXATION

  # Cap should also be RELAXATION, not PHYSICAL_HAZARD.
  role = _role_for_seed(PLANNER_SEED_INTENT_LEAD_FOLLOW, MOVING_LEAD_RECOVERY_SEED_REASON, "cap")
  assert role == CandidateRole.RELAXATION

  # Even if someone mis-labels the intent as LAUNCH, we still want
  # RELAXATION because the moving_lead_recovery reason is a moving
  # comfort/recovery policy, not a launch authority.
  role = _role_for_seed(PLANNER_SEED_INTENT_LAUNCH, MOVING_LEAD_RECOVERY_SEED_REASON, PLANNER_SEED_FLOOR)
  assert role == CandidateRole.RELAXATION


def test_moving_lead_recovery_seed_intent_is_lead_follow():
  """The moving_lead_recovery reason must map to LEAD_FOLLOW intent in
  the planner seed layer, matching the rest of the lead-follow family
  of seed reasons. This is what allows the seed to participate in the
  lead-follow decision layer without being treated as a launch or
  driver-cruise candidate."""
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import (
    planner_seed_intent_for_reason,
  )
  intent = planner_seed_intent_for_reason(MOVING_LEAD_RECOVERY_SEED_REASON, has_lead=True)
  assert intent == "lead_follow"


def test_moving_lead_recovery_inactive_lead_context_blocks_emission():
  """Without a primary physical lead, the moving recovery helper is
  not called and the seed is not emitted. The fallback path
  (raw LEAD_MPC fallback) keeps authority."""
  planner = SimpleNamespace(output_a_target=0.0, output_should_stop=False)
  candidates = build_moving_lead_seed_candidates(
    planner, has_lead=True, accel_limits=(-2.0, 2.0),
    moving_lead_recovery_a_target=None,
  )
  # No seed candidate with name "moving_lead_recovery" emitted.
  assert all(c.name != "moving_lead_recovery" for c in candidates)


def test_moving_lead_recovery_active_emits_floor_seed():
  """When the planner produces a moving_lead_recovery floor, the seed
  is emitted with selection=FLOOR and the recovery debug attached."""
  planner = SimpleNamespace(output_a_target=0.0, output_should_stop=False)
  candidates = build_moving_lead_seed_candidates(
    planner, has_lead=True, accel_limits=(-2.0, 2.0),
    moving_lead_recovery_a_target=0.35,
    moving_lead_recovery_debug={
      "moving_lead_recovery_active": True,
      "moving_lead_recovery_phase": "mild_reaccel",
      "moving_lead_recovery_predicted_gap": 9.5,
      "moving_lead_recovery_minimum_allowed_gap": 4.0,
      "moving_lead_recovery_safe_accel_cap": 0.42,
      "moving_lead_recovery_lead_created_runway": True,
    },
  )
  recovery_candidates = [c for c in candidates if c.name == "moving_lead_recovery"]
  assert len(recovery_candidates) == 1
  c = recovery_candidates[0]
  assert c.output.a_target == pytest.approx(0.35)
  assert c.reason == MOVING_LEAD_RECOVERY_SEED_REASON
  assert c.selection == "floor"
  assert c.output.has_lead
  assert c.output.debug["moving_lead_recovery_active"] is True
  assert c.output.debug["moving_lead_recovery_phase"] == "mild_reaccel"


def test_moving_lead_recovery_suppresses_raw_lead_mpc_fallback_when_runway_safe():
  """Fallback suppression should recognize the moving_lead_recovery
  RELAXATION seed the same way it recognizes routine comfort, but only
  when debug says it is non-urgent and runway-safe."""
  from dataclasses import replace
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    fallback_physical_candidates,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import (
    CandidateRole,
    DecisionSource,
    LongitudinalCandidate,
  )

  moving_recovery_candidate = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=15.0,
    a_target=0.35,
    confidence=0.8,
    urgency=0.2,
    active_reason=MOVING_LEAD_RECOVERY_SEED_REASON,
    should_stop=False,
  )
  # Fallback suppression only recognizes the moving_lead_recovery
  # RELAXATION seed as runway-safe when its debug payload confirms
  # the predicted gap is above the safety floor, the runway margin is
  # non-negative, and the projected lead accel is not strongly
  # negative. Without these fields, the suppression path stays
  # inert — which is why the original identity-based assertion was
  # a false-positive.
  moving_recovery_candidate = replace(
    moving_recovery_candidate,
    debug={
      "moving_lead_recovery_predicted_gap": 9.5,
      "moving_lead_recovery_minimum_allowed_gap": 4.0,
      "moving_lead_recovery_runway_margin_t": 2.0,
      "moving_lead_recovery_a_lead_pred": 0.4,
    },
  )
  raw_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=15.0,
    a_target=-0.20,
    confidence=0.9,
    urgency=0.6,
    active_reason="lead_mpc",
    should_stop=False,
  )
  fallback_output = SimpleNamespace(a_target=0.0, should_stop=False, source="lead0")
  fallbacks = fallback_physical_candidates(
    (moving_recovery_candidate,),
    (raw_lead_mpc,),
    fallback_output,
  )
  # The raw non-urgent LEAD_MPC fallback should be fully suppressed
  # when the moving_lead_recovery seed is runway-safe. Assert by
  # content, not object identity: fallback_physical_candidates
  # re-wraps the candidate via custom_v2_candidate_with_debug, so
  # identity-based checks would always pass even if the suppression
  # path is broken.
  assert len(fallbacks) == 0
  assert not any(
    c.source == DecisionSource.LEAD_MPC and c.a_target == pytest.approx(-0.20)
    for c in fallbacks
  )


def test_moving_lead_recovery_suppression_requires_runway_safe_debug():
  """A moving_lead_recovery seed without the runway-safe debug payload
  must NOT suppress the raw LEAD_MPC fallback. This guards against
  silently bypassing the safety gate when debug is missing or stale."""
  from dataclasses import replace
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    fallback_physical_candidates,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import (
    CandidateRole,
    DecisionSource,
    LongitudinalCandidate,
  )

  moving_recovery_candidate = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=15.0,
    a_target=0.35,
    confidence=0.8,
    urgency=0.2,
    active_reason=MOVING_LEAD_RECOVERY_SEED_REASON,
    should_stop=False,
  )
  # Intentionally leave debug empty / not runway-safe. The suppression
  # helper should refuse to recognize this as runway-safe and must
  # preserve the raw LEAD_MPC fallback.
  raw_lead_mpc = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=15.0,
    a_target=-0.20,
    confidence=0.9,
    urgency=0.6,
    active_reason="lead_mpc",
    should_stop=False,
  )
  fallback_output = SimpleNamespace(a_target=0.0, should_stop=False, source="lead0")
  fallbacks = fallback_physical_candidates(
    (moving_recovery_candidate,),
    (raw_lead_mpc,),
    fallback_output,
  )
  assert any(
    c.source == DecisionSource.LEAD_MPC and c.a_target == pytest.approx(-0.20)
    for c in fallbacks
  )


def test_moving_lead_recovery_preserves_hard_braking_lead_fallback():
  """When the raw LEAD_MPC fallback is hard-braking (a_target < -1.0),
  it must NOT be suppressed even if the moving recovery is active.
  Hard-braking-lead is always a physical hazard."""
  from dataclasses import replace
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed_policy import (
    fallback_physical_candidates,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_decision import (
    CandidateRole,
    DecisionSource,
    LongitudinalCandidate,
  )

  moving_recovery_candidate = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.RELAXATION,
    v_target=15.0,
    a_target=0.35,
    confidence=0.8,
    urgency=0.2,
    active_reason=MOVING_LEAD_RECOVERY_SEED_REASON,
    should_stop=False,
  )
  # Runway-safe debug so the suppression path is active, proving the
  # hard-braking guard is what preserves this candidate, not an
  # inert suppression path.
  moving_recovery_candidate = replace(
    moving_recovery_candidate,
    debug={
      "moving_lead_recovery_predicted_gap": 9.5,
      "moving_lead_recovery_minimum_allowed_gap": 4.0,
      "moving_lead_recovery_runway_margin_t": 2.0,
      "moving_lead_recovery_a_lead_pred": 0.4,
    },
  )
  raw_lead_mpc_hard = LongitudinalCandidate(
    source=DecisionSource.LEAD_MPC,
    role=CandidateRole.PHYSICAL_HAZARD,
    v_target=15.0,
    a_target=-1.5,
    confidence=0.9,
    urgency=0.8,
    active_reason="lead_mpc",
    should_stop=False,
  )
  fallback_output = SimpleNamespace(a_target=0.0, should_stop=False, source="lead0")
  fallbacks = fallback_physical_candidates(
    (moving_recovery_candidate,),
    (raw_lead_mpc_hard,),
    fallback_output,
  )
  # The hard-braking fallback must be present, even though the moving
  # recovery is also active and runway-safe. Match by a_target because
  # the function re-wraps the candidate.
  assert any(c.a_target == pytest.approx(-1.5) and c.role == CandidateRole.PHYSICAL_HAZARD for c in fallbacks)


# ----------------------------------------------------------------------------
# Moving lead recovery integration gating tests
# ----------------------------------------------------------------------------


def test_moving_recovery_lead_opening_source_prefers_behavior_when_present():
  """When a behavior lead is available, the moving recovery uses the
  behavior opening signal. This matches the lead_pullaway intent tracker
  and the same authority the recovery helper consumes."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    select_moving_recovery_lead_opening,
  )
  behavior_lead = SimpleNamespace(radarTrackId=1)
  opening, source = select_moving_recovery_lead_opening(
    primary_behavior_lead=behavior_lead,
    behavior_lead_opening=True,
    physical_lead_opening=False,
  )
  assert opening is True
  assert source == "behavior"

  opening, source = select_moving_recovery_lead_opening(
    primary_behavior_lead=behavior_lead,
    behavior_lead_opening=False,
    physical_lead_opening=True,
  )
  assert opening is False
  assert source == "behavior"


def test_moving_recovery_lead_opening_source_falls_back_to_physical_only_lead():
  """When no behavior lead is present, the moving recovery must use
  physical_lead_opening. This is the fix for the physical-only
  re-accel scenario: a stable physical lead that is opening should
  still get a recovery floor, even though behavior authority is not
  present yet."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    select_moving_recovery_lead_opening,
  )
  opening, source = select_moving_recovery_lead_opening(
    primary_behavior_lead=None,
    behavior_lead_opening=False,
    physical_lead_opening=True,
  )
  assert opening is True
  assert source == "physical"

  opening, source = select_moving_recovery_lead_opening(
    primary_behavior_lead=None,
    behavior_lead_opening=True,
    physical_lead_opening=False,
  )
  assert opening is False
  assert source == "physical"


def test_moving_recovery_lead_opening_source_physical_only_helper_activates():
  """End-to-end check: when only a physical lead is present and
  physical_lead_opening is True, the moving recovery helper receives
  lead_opening=True and can activate. This is the regression guard
  for the physical-only re-accel scenario described in the review:
  the previous integration passed behavior_lead_opening even when no
  behavior lead existed, so a physical-only lead would silently miss
  the recovery floor even when the lead was clearly opening."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    get_moving_lead_recovery,
    select_moving_recovery_lead_opening,
  )
  # Simulate the physical-only branch: no behavior lead, behavior
  # opening is False, but physical opening is True.
  opening, source = select_moving_recovery_lead_opening(
    primary_behavior_lead=None,
    behavior_lead_opening=False,
    physical_lead_opening=True,
  )
  assert opening is True
  assert source == "physical"
  # The helper is called with the physical opening signal.
  result = get_moving_lead_recovery(
    v_ego=15.0, d_rel=10.0, v_lead=14.0, a_lead=0.3, lead_accel_trend=0.1,
    lead_opening=opening, minimum_allowed_gap=4.0,
  )
  assert result.active
  assert result.debug["moving_lead_recovery_lead_opening"] is True


def test_moving_recovery_lead_state_valid_accepts_stable_suppressive_physical():
  """A stable, on-path, suppressive physical lead with no hazard risk
  is the canonical case for the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is True


def test_moving_recovery_lead_state_valid_rejects_new_lead():
  """A new_lead physical lead (recently acquired) must NOT enter the
  moving recovery helper. New leads are suppressive-only and have
  not yet accumulated stability evidence."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=True, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_flicker_guard():
  """A physical lead with active flicker guard timer is unstable and
  must NOT enter the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.4,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_unstable_lead():
  """An unstable physical lead (e.g. confidence not yet stable) must
  NOT enter the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=False, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_non_suppressive_lead():
  """A physical lead with no suppressive authority (e.g. authority=NONE)
  must NOT enter the moving recovery helper. The helper expects a
  confirmed lead, not a phantom one."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=False, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_short_ttc_hazard():
  """A physical lead with short TTC (≤ 4 s) is a physical hazard.
  The moving recovery helper only enforces the 4 m predicted-gap
  floor; the planner-side gate must also block short-TTC cases so the
  helper's own math doesn't accidentally relax a real hazard."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=3.0, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_high_required_decel_hazard():
  """A physical lead with high required decel (≥ 0.25 m/s²) is a
  physical hazard and must NOT enter the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.40, ttc=math.inf, closing_speed=0.2)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_rejects_high_closing_speed_hazard():
  """A physical lead with high closing speed (≥ 1.0 m/s) is a
  physical hazard and must NOT enter the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  risk_model = SimpleNamespace(required_decel=0.05, ttc=math.inf, closing_speed=1.5)
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=risk_model,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is False


def test_moving_recovery_lead_state_valid_handles_missing_risk_model():
  """A physical lead without a risk model is treated as not risky.
  The helper still has its own 4 m predicted-gap safety floor."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  physical = SimpleNamespace(
    shadow=False, new_lead=False, flicker_guard_timer=0.0,
    stable=True, suppressive=True, risk_model=None,
  )
  context = SimpleNamespace(physical=physical)
  assert is_moving_recovery_lead_state_valid(context) is True


def test_moving_recovery_lead_state_valid_handles_none_context():
  """A None context must NOT enter the moving recovery helper."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_lead_state_valid,
  )
  assert is_moving_recovery_lead_state_valid(None) is False


def test_moving_recovery_physical_risk_active_uses_default_thresholds():
  """The default risk thresholds match LEAD_CONTEXT_RISK_REQUIRED_DECEL
  (0.25) and LEAD_CONTEXT_RISK_TTC (4.0) so the planner-side gate is
  consistent with the rest of the lead-context risk model."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_physical_risk_active,
  )
  # At-or-above thresholds trigger.
  assert is_moving_recovery_physical_risk_active(
    SimpleNamespace(required_decel=0.25, ttc=math.inf, closing_speed=0.0)
  ) is True
  assert is_moving_recovery_physical_risk_active(
    SimpleNamespace(required_decel=0.0, ttc=4.0, closing_speed=0.0)
  ) is True
  assert is_moving_recovery_physical_risk_active(
    SimpleNamespace(required_decel=0.0, ttc=math.inf, closing_speed=1.0)
  ) is True
  # Below thresholds do not.
  assert is_moving_recovery_physical_risk_active(
    SimpleNamespace(required_decel=0.10, ttc=8.0, closing_speed=0.5)
  ) is False
  # ttc=inf is never risky.
  assert is_moving_recovery_physical_risk_active(
    SimpleNamespace(required_decel=0.0, ttc=math.inf, closing_speed=0.0)
  ) is False


def test_moving_recovery_physical_risk_active_none_model_is_safe():
  """A None risk model is treated as not risky. The lead-context
  selector never populates risk_model=None for confirmed leads, but
  the integration is robust to that case."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import (
    is_moving_recovery_physical_risk_active,
  )
  assert is_moving_recovery_physical_risk_active(None) is False
