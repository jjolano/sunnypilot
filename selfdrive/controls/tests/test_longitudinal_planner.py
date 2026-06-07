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
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource, T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_stacks.planner_seed import PLANNER_SEED_FLOOR
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2, SUNNYPILOT_CURRENT, StackResolution
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as LongitudinalPlanSourceSP
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  build_scc_mode_evidence,
  build_planner_seed_accel_candidate,
  E2E_CLOSE_STOP_DECEL_MAX,
  E2E_CLOSE_STOP_MIN_ROLLING_V,
  E2E_STOP_APPROACH_DECEL_MAX,
  E2E_RUNWAY_FINAL_CRAWL_ACCEL_MAX,
  FAST_LEAD_MOTION_OPENING_DEADBAND,
  LEAD_FLICKER_CLOSE_GUARD_TIME,
  LEAD_FLICKER_FIRST_LOSS_HOLD_TIME,
  EXCESS_GAP_CLOSURE_REASON,
  FastLeadMotionEvidence,
  LeadFlickerSafetyCapTracker,
  LeadPullawayIntentTracker,
  LeadPullawayPhase,
  LEAD_PULLAWAY_PULSE_REASON,
  LongitudinalPlanner,
  _A_TOTAL_MAX_BP,
  _A_TOTAL_MAX_V,
  fast_lead_motion_evidence_enabled,
  get_fast_lead_motion_evidence,
  get_lead_flicker_required_decel,
  get_planner_lead_motion_values,
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
  limit_accel_in_turns,
  one_pedal_cruise_hold_requested,
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
from openpilot.selfdrive.controls.lib.lead_context import LEAD_AUTHORITY_PROGRESS_ALLOWED, LeadContextTracker, LeadProgressModel, LeadRiskModel

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
