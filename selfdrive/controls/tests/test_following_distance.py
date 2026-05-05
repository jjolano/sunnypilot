import numpy as np
import pytest
import itertools
from types import SimpleNamespace
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.parameterized import parameterized_class

from cereal import custom, log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  APPROACH_BRAKE,
  APPROACH_BRAKE_MIN,
  APPROACH_ENGAGE_OFFSET_MAX,
  APPROACH_MIN_GAP_BUFFER,
  COMFORT_BRAKE,
  LEAD_DEPARTURE_RELAXATION_MAX,
  LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND,
  LEAD_ACCEL_MATCH_DECEL_CAP,
  LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND,
  LEAD_ACCEL_RECOVERY_ACCEL_MAX,
  PRE_TARGET_RUNWAY_DECEL_THRESHOLD_AGGRESSIVE,
  PRE_TARGET_RUNWAY_DECEL_THRESHOLD_RELAXED,
  PRE_TARGET_RUNWAY_DECEL_THRESHOLD_STANDARD,
  MOVING_LEAD_STOP_RESERVE_MAX,
  LEAD_STOP_GAP_EXCESS_OFFSET_MAX,
  LEAD_STOP_GAP_TAPER_MAX,
  LEAD_GAP_COMFORT_LIGHT_DECEL,
  LEAD_CRAWL_ACCEL_MAX,
  LEAD_CRAWL_ACCEL_LIMIT,
  LEAD_CRAWL_BRAKE_MAX,
  LEAD_SURGE_DAMPING_ACCEL_MAX,
  LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX,
  MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN,
  MOVING_LEAD_CLOSING_CUSHION_DECEL_MAX,
  LEAD_STOP_APPROACH_DECEL_CAP,
  LEAD_STOP_RUNWAY_BRAKE,
  LEAD_TRANSITION_PERSISTENCE,
  LEAD_TRANSITION_GUARD_FADE_TIME,
  LEAD_TRANSITION_GUARD_OUTPUT_DELAY,
  LEAD_TRANSITION_Y_REL_CONFIRM,
  LEAD_TRANSITION_Y_REL_SOFT,
  STOP_DISTANCE,
  STOP_DISTANCE_FADE_V,
  STOP_DISTANCE_MIN,
  STOPPED_LEAD_BUFFER,
  SOURCE_HYSTERESIS_MARGIN,
  apply_source_hysteresis,
  get_approach_available_runway,
  get_approach_brake,
  get_approach_engage_offset,
  get_approach_follow_distance,
  get_approach_runway_blend,
  get_desired_follow_distance,
  get_lead_accel_match_margin,
  get_lead_accel_match_target,
  get_lead_accel_match_targets,
  get_lead_accel_recovery_a_min,
  get_short_gap_pullaway_response_target,
  get_lead_gap_comfort_a_min,
  get_lead_gap_comfort_floor,
  get_lead_gap_comfort_recovery_blend,
  get_combined_accel_target,
  get_lead_departure_available_runway,
  get_lead_departure_relaxation,
  get_lead_danger_distance,
  get_lead_crawl_accel_max,
  get_lead_crawl_comfort_target,
  get_lead_surge_damping_target,
  get_lead_stop_approach_comfort_target,
  get_lead_stop_runway_preference,
  get_lead_stop_runway_required_decel,
  get_lead_stop_runway_blend,
  get_lead_stop_runway_gap,
  get_lead_stop_runway_urgency,
  get_lead_stop_presentation_distance,
  get_lead_stop_gap_excess_offset,
  get_lead_stop_gap_taper,
  get_lead_time_gap_target,
  get_moving_lead_closing_cushion_target,
  get_moving_lead_stop_approach_comfort_target,
  get_moving_lead_stop_reserve,
  get_lead_transition_accel_max,
  get_lead_transition_adjusted_accel,
  get_lead_transition_cost_obstacle,
  get_lead_transition_lateral_blend,
  get_lead_transition_obstacle_release,
  get_lead_transition_release_target,
  get_pre_target_runway_decel_threshold,
  get_safe_obstacle_distance,
  get_selected_lead_targets,
  get_stopped_lead_buffer,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
  LongitudinalMpc,
  N,
)
from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CREEP_TO_STOP_GAP_ACCEL_MAX,
  CREEP_TO_STOP_GAP_ACCEL_MIN,
  CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
  CREEP_TO_STOP_GAP_HOLD_EXCESS,
  CREEP_TO_STOP_GAP_MAX_EXCESS,
  CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX,
  CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING,
  CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED,
  CREEP_TO_STOP_GAP_START_EXCESS,
  CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD,
  STOPPED_LEAD_GAP_FILL_ACCEL_MAX,
  STOPPED_LEAD_GAP_FILL_MAX_EXCESS,
  STOPPED_LEAD_GAP_FILL_MIN_EXCESS,
  get_stopped_lead_gap_fill_accel,
  get_creep_to_stop_gap_accel,
  get_model_lead_pullaway,
  get_predicted_lead_pullaway,
  has_predicted_lead_pullaway,
  should_arm_stopped_lead_gap_fill,
  should_hold_creep_to_stop_gap,
  should_release_creep_stop_hold,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


class FakeVelocityFilter:
  def __init__(self, x):
    self.x = x

  def update(self, v_ego):
    return v_ego


class FakeMpc:
  def __init__(self):
    self.source = long_mpc.LongitudinalPlanSource.cruise
    self.crash_cnt = 0
    self.solve_time = 0.0
    self.v_solution = np.zeros(N + 1)
    self.a_solution = np.zeros(N + 1)
    self.j_solution = np.zeros(N)

  def set_weights(self, *args, **kwargs):
    pass

  def set_cur_state(self, *args):
    pass

  def update(self, *args, **kwargs):
    self.update_args = args
    self.update_kwargs = kwargs


def patch_planner_sp(monkeypatch):
  monkeypatch.setattr(longitudinal_planner.LongitudinalPlannerSP, "update", lambda _planner, _sm: None)
  monkeypatch.setattr(
    longitudinal_planner.LongitudinalPlannerSP,
    "update_targets",
    lambda _planner, _sm, _v_ego, a_ego, v_cruise, coast_accel=None: (v_cruise, a_ego),
  )


def make_planner_for_stop_preservation(v_ego=0.0, gap_fill_timer=0.0):
  planner = longitudinal_planner.LongitudinalPlanner.__new__(longitudinal_planner.LongitudinalPlanner)
  planner.CP = SimpleNamespace(
    openpilotLongitudinalControl=True,
    steerRatio=15.0,
    wheelbase=2.7,
    longitudinalActuatorDelay=0.0,
    vEgoStopping=0.5,
  )
  planner.mpc = FakeMpc()
  planner.params = SimpleNamespace(get_bool=lambda _key: False)
  planner.longitudinal_arbiter = longitudinal_planner.LongitudinalArbiter()
  planner.longitudinal_decision = None
  planner.longitudinal_decision_candidates = []
  planner.fcw = False
  planner.dt = longitudinal_planner.DT_MDL
  planner.allow_throttle = True
  planner.a_desired = 0.0
  planner.v_desired_filter = FakeVelocityFilter(v_ego)
  planner.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
  # The fixture skips __init__ and models a steady-state engaged planner.
  planner.prev_reset_state = False
  planner.engage_stop_bootstrap_timer = 0.0
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.creep_to_stop_gap_active = False
  planner.creep_stop_hold_released = False
  planner.stopped_lead_gap_fill_timer = gap_fill_timer
  planner.lead_loss_e2e_guard_timer = 0.0
  planner.previous_lead_loss_status = False
  planner.previous_lead_loss_d_rel = 0.0
  planner.previous_lead_loss_model_prob = 0.0
  planner.stopped_lead_gap_fill_track_id = -2
  planner.stopped_lead_gap_fill_d_rel = 0.0
  planner.stopped_lead_gap_fill_v_lead = 0.0
  planner.dec = SimpleNamespace(active=lambda: False)
  planner.source = custom.LongitudinalPlanSP.LongitudinalPlanSource.cruise
  return planner


def make_model_action(desired_accel=-1.0, should_stop=True):
  return SimpleNamespace(
    action=SimpleNamespace(desiredAcceleration=desired_accel, shouldStop=should_stop),
    position=SimpleNamespace(x=[]),
    velocity=SimpleNamespace(x=[]),
    acceleration=SimpleNamespace(x=[]),
    meta=SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[0.0, 1.0]), laneChangeState=log.LaneChangeState.off),
    leadsV3=[],
  )


def make_planner_sm(v_ego, lead, desired_accel=-1.0, should_stop=True, brake_pressed=False, gas_pressed=False, force_slow_decel=False):
  return {
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False), orientationNED=[0.0, 0.0, 0.0]),
    'carState': SimpleNamespace(
      vEgo=v_ego,
      vCruise=30.0,
      vCruiseCluster=30.0,
      standstill=v_ego < 0.01,
      steeringAngleDeg=0.0,
      aEgo=0.0,
      brakePressed=brake_pressed,
      gasPressed=gas_pressed,
    ),
    'controlsState': SimpleNamespace(longControlState=longitudinal_planner.LongCtrlState.pid, forceDecel=force_slow_decel),
    'selfdriveState': SimpleNamespace(enabled=True, experimentalMode=True, personality=log.LongitudinalPersonality.standard),
    'liveParameters': SimpleNamespace(angleOffsetDeg=0.0),
    'modelV2': make_model_action(desired_accel, should_stop),
    'radarState': SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False)),
  }


@pytest.mark.parametrize(
  ("brake_pressed", "gas_pressed", "force_slow_decel"),
  [(True, False, False), (False, True, False), (False, False, True)],
)
def test_planner_blocks_mpc_short_gap_response_for_driver_override_or_force_slow_decel(
  monkeypatch, brake_pressed, gas_pressed, force_slow_decel,
):
  patch_planner_sp(monkeypatch)
  planner = make_planner_for_stop_preservation(v_ego=0.6)
  lead = SimpleNamespace(
    status=True,
    dRel=0.5 * (get_lead_stop_presentation_distance(0.6, 0.35, 0.8, 1.0) + STOP_DISTANCE),
    vLeadK=0.35,
    modelProb=1.0,
    aLeadK=0.8,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(
    0.6, lead, desired_accel=-0.2, should_stop=False,
    brake_pressed=brake_pressed, gas_pressed=gas_pressed, force_slow_decel=force_slow_decel,
  ))

  assert planner.mpc.update_kwargs["block_short_gap_pullaway_response"]


def test_e2e_should_stop_survives_positive_creep_pullaway(monkeypatch):
  patch_planner_sp(monkeypatch)
  planner = make_planner_for_stop_preservation(v_ego=0.0)
  lead = SimpleNamespace(
    status=True,
    dRel=STOP_DISTANCE + 0.6,
    vLeadK=0.8,
    modelProb=1.0,
    aLeadK=0.0,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(0.0, lead, desired_accel=-1.0, should_stop=True))

  assert planner.output_should_stop
  assert planner.output_a_target < 0.0


def test_creep_pullaway_allows_validated_mpc_lead_accel(monkeypatch):
  patch_planner_sp(monkeypatch)
  monkeypatch.setattr(longitudinal_planner, "get_accel_from_plan", lambda *_args, **_kwargs: (1.0, False))
  planner = make_planner_for_stop_preservation(v_ego=0.0)
  planner.output_a_target = 0.8
  stop_target = get_lead_stop_presentation_distance(0.0, 1.2, 0.8, 1.0)
  lead = SimpleNamespace(
    status=True,
    dRel=stop_target + CREEP_TO_STOP_GAP_START_EXCESS,
    vLeadK=1.2,
    modelProb=1.0,
    aLeadK=0.8,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(0.0, lead, desired_accel=1.0, should_stop=False))

  assert planner.creep_to_stop_gap_active
  assert planner.output_a_target > CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_pullaway_keeps_cap_for_weak_cushion(monkeypatch):
  patch_planner_sp(monkeypatch)
  monkeypatch.setattr(longitudinal_planner, "get_accel_from_plan", lambda *_args, **_kwargs: (1.0, False))
  planner = make_planner_for_stop_preservation(v_ego=0.0)
  planner.creep_to_stop_gap_active = True
  planner.output_a_target = 0.8
  stop_target = get_lead_stop_presentation_distance(0.0, 0.3, 0.0, 1.0)
  lead = SimpleNamespace(
    status=True,
    dRel=stop_target + 0.1,
    vLeadK=0.3,
    modelProb=1.0,
    aLeadK=0.0,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(0.0, lead, desired_accel=1.0, should_stop=False))

  assert planner.creep_to_stop_gap_active
  assert planner.output_a_target <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_e2e_should_stop_survives_positive_gap_fill(monkeypatch):
  patch_planner_sp(monkeypatch)
  planner = make_planner_for_stop_preservation(v_ego=0.0, gap_fill_timer=1.0)
  lead = SimpleNamespace(
    status=True,
    dRel=STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0,
    vLeadK=0.0,
    modelProb=1.0,
    aLeadK=0.0,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(0.0, lead, desired_accel=-1.0, should_stop=True))

  assert planner.output_should_stop
  assert planner.output_a_target < 0.0


def test_e2e_decel_survives_lead_accel_recovery(monkeypatch):
  patch_planner_sp(monkeypatch)
  planner = make_planner_for_stop_preservation(v_ego=2.2)
  planner.mpc.v_solution = np.full(N + 1, 2.2)
  lead = SimpleNamespace(
    status=True,
    dRel=15.0,
    vLeadK=4.2,
    modelProb=1.0,
    aLeadK=0.9,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(2.2, lead, desired_accel=-0.8, should_stop=False))

  assert planner.mpc.source == long_mpc.LongitudinalPlanSource.e2e
  assert planner.output_a_target == pytest.approx(-0.8)


def test_stopped_lead_gap_fill_resets_for_discontinuous_lead(monkeypatch):
  patch_planner_sp(monkeypatch)
  planner = make_planner_for_stop_preservation(v_ego=0.0)
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  close_lead = SimpleNamespace(
    status=True,
    radarTrackId=1,
    dRel=stop_target + 0.2,
    vLeadK=0.0,
    modelProb=1.0,
    aLeadK=0.0,
    aLeadTau=0.0,
    yRel=0.0,
  )
  far_lead = SimpleNamespace(
    status=True,
    radarTrackId=2,
    dRel=stop_target + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0,
    vLeadK=0.0,
    modelProb=1.0,
    aLeadK=0.0,
    aLeadTau=0.0,
    yRel=0.0,
  )

  planner.update(make_planner_sm(0.0, close_lead, desired_accel=0.0, should_stop=False))
  assert planner.stopped_lead_gap_fill_timer > 0.0

  planner.update(make_planner_sm(0.0, far_lead, desired_accel=0.0, should_stop=False))

  assert planner.stopped_lead_gap_fill_timer == pytest.approx(0.0)
  assert planner.output_a_target <= 0.0


def stop_distance_buffer(v_ego):
  fade = (STOP_DISTANCE_FADE_V**2) / (v_ego**2 + STOP_DISTANCE_FADE_V**2)
  return STOP_DISTANCE_MIN + (STOP_DISTANCE - STOP_DISTANCE_MIN) * fade


def make_model_msg_lead(d_rel=STOP_DISTANCE + 0.35, horizon_gap=0.5, horizon_v=0.8, prob=0.9,
                        y_rel=0.0, x_std=0.5, y_std=0.2, v_std=0.5,
                        horizon_x_std=None, horizon_y_std=None, horizon_v_std=None):
  x0 = d_rel + CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET
  return SimpleNamespace(leadsV3=[SimpleNamespace(
    prob=prob,
    t=[0.0, 2.0],
    x=[x0, x0 + horizon_gap],
    y=[-y_rel, -y_rel],
    v=[0.0, horizon_v],
    xStd=[x_std, x_std if horizon_x_std is None else horizon_x_std],
    yStd=[y_std, y_std if horizon_y_std is None else horizon_y_std],
    vStd=[v_std, v_std if horizon_v_std is None else horizon_v_std],
  )])


def make_radar_lead(d_rel=STOP_DISTANCE + 0.35, status=True, model_prob=1.0, v_lead=0.0, y_rel=0.0):
  return SimpleNamespace(status=status, dRel=d_rel, modelProb=model_prob, vLeadK=v_lead, yRel=y_rel)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_safe_obstacle_distance_matches_explicit_formula(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  expected = (speed**2) / (2 * COMFORT_BRAKE) + t_follow * speed + stop_distance_buffer(speed)
  assert get_safe_obstacle_distance(speed, t_follow) == pytest.approx(expected)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_desired_follow_distance_matches_explicit_formula(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  expected = (speed**2) / (2 * COMFORT_BRAKE) + t_follow * speed + stop_distance_buffer(speed) - get_stopped_equivalence_factor(speed)
  assert get_desired_follow_distance(speed, speed, t_follow) == pytest.approx(expected)


def test_comfort_biased_follow_times():
  assert get_T_FOLLOW(log.LongitudinalPersonality.relaxed) == pytest.approx(1.85)
  assert get_T_FOLLOW(log.LongitudinalPersonality.standard) == pytest.approx(1.55)
  assert get_T_FOLLOW(log.LongitudinalPersonality.aggressive) == pytest.approx(1.30)


def test_stop_distance_buffer_fades_with_speed():
  buffer_speeds = [0.0, 5.0, 10.0, 35.0]
  buffers = [stop_distance_buffer(speed) for speed in buffer_speeds]
  assert buffers[0] == pytest.approx(STOP_DISTANCE)
  assert buffers[0] > buffers[1] > buffers[2] > buffers[3] > STOP_DISTANCE_MIN - 1e-6


def test_lead_stop_presentation_distance_adapts_between_confirmed_and_uncertain():
  presentation_distance = getattr(long_mpc, "get_lead_stop_presentation_distance", None)
  assert presentation_distance is not None

  stable_target = presentation_distance(v_ego=0.0, v_lead=0.0, a_lead=0.0, model_prob=1.0)
  weak_model_target = presentation_distance(v_ego=0.0, v_lead=0.0, a_lead=0.0, model_prob=0.5)
  moving_target = presentation_distance(v_ego=0.0, v_lead=1.5, a_lead=0.0, model_prob=1.0)
  faster_ego_target = presentation_distance(v_ego=4.0, v_lead=0.0, a_lead=0.0, model_prob=1.0)

  assert stable_target == pytest.approx(5.0)
  assert weak_model_target == pytest.approx(STOP_DISTANCE)
  assert moving_target == pytest.approx(STOP_DISTANCE)
  assert faster_ego_target == pytest.approx(STOP_DISTANCE)


def test_stopped_lead_buffer_only_applies_near_stop():
  assert get_stopped_lead_buffer(0.0, 0.0) == pytest.approx(0.0)
  assert get_stopped_lead_buffer(1.0, 0.0) == pytest.approx(STOPPED_LEAD_BUFFER)
  assert get_stopped_lead_buffer(1.0, 3.0) == pytest.approx(0.0)


def test_lead_stop_gap_taper_only_applies_for_low_speed_moving_leads():
  assert get_lead_stop_gap_taper(0.0, 0.0) == pytest.approx(0.0)
  assert 0.0 < get_lead_stop_gap_taper(0.0, 0.8) < LEAD_STOP_GAP_TAPER_MAX
  assert get_lead_stop_gap_taper(0.0, 2.0) == pytest.approx(LEAD_STOP_GAP_TAPER_MAX)
  assert get_lead_stop_gap_taper(1.5, 2.0) == pytest.approx(0.0)


def test_lead_stop_gap_excess_offset_requires_extra_runway():
  assert get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 0.9) == pytest.approx(0.0)
  assert 0.0 < get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 2.0) < LEAD_STOP_GAP_EXCESS_OFFSET_MAX
  assert get_lead_stop_gap_excess_offset(0.0, STOP_DISTANCE + 5.0) == pytest.approx(LEAD_STOP_GAP_EXCESS_OFFSET_MAX)
  assert get_lead_stop_gap_excess_offset(1.5, STOP_DISTANCE + 5.0) == pytest.approx(0.0)


def test_creep_to_stop_gap_release_waits_for_profile_start_and_tapers_to_target_gap():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS - 0.1, 0.0, 1.0, False
  )
  assert not active
  assert accel == pytest.approx(0.0)

  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.0, 1.0, False
  )
  assert active
  assert 0.0 < accel <= CREEP_TO_STOP_GAP_ACCEL_MAX

  active, follow_accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_FOLLOW_EXCESS, 0.0, 1.0, active
  )
  assert active
  assert 0.0 < follow_accel < accel

  active, soft_stop_accel = get_creep_to_stop_gap_accel(0.2, stop_target + 0.3, 0.0, 1.0, active)
  assert active
  assert CREEP_TO_STOP_GAP_ACCEL_MIN < soft_stop_accel < 0.0

  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target, 0.0, 1.0, active)
  assert not active
  assert accel == pytest.approx(0.0)


def test_creep_to_stop_gap_uses_adaptive_stopped_target_profile_start():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.0, 1.0, False
  )

  assert stop_target < STOP_DISTANCE
  assert active
  assert 0.0 < accel <= CREEP_TO_STOP_GAP_ACCEL_MAX


def test_creep_to_stop_gap_uses_stronger_accel_for_confirmed_pullaway():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.8, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.8, 1.0, False)

  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_confirmed_pullaway_can_match_no_lead_launch_cap():
  stop_target = get_lead_stop_presentation_distance(0.0, 1.2, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 1.2, 1.0, False)

  assert active
  assert CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX == pytest.approx(0.55)
  assert accel == pytest.approx(CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX)


def test_creep_to_stop_gap_uses_firmer_floor_for_initial_pullaway():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.25, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.25, 1.0, False)

  assert active
  assert 0.30 <= accel <= STOPPED_LEAD_GAP_FILL_ACCEL_MAX


def test_creep_to_stop_gap_actively_creeps_before_four_meter_excess():
  active, accel = get_creep_to_stop_gap_accel(0.1, STOP_DISTANCE + 3.5, 0.8, 1.0, True)

  assert active
  assert accel > 0.0


def test_creep_to_stop_gap_stops_chasing_after_four_meter_excess():
  active, accel = get_creep_to_stop_gap_accel(0.1, STOP_DISTANCE + 4.5, 0.8, 1.0, True)

  assert not active
  assert accel == pytest.approx(0.0)


def test_creep_to_stop_gap_uses_soft_release_inside_final_meter():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.1, 0.0, 1.0)
  active, _ = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.1, 1.0, False
  )
  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target + 0.6, 0.1, 1.0, active)

  assert active
  assert 0.0 < accel <= 0.10


def test_creep_stop_hold_release_hysteresis_blocks_crawl_chatter():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  assert should_hold_creep_to_stop_gap(0.0, stop_target + 0.2, 0.0, 0.0, release_active=False)
  assert not should_release_creep_stop_hold(False, 0.0, stop_target + 0.2, 0.0, 0.0)

  assert not should_release_creep_stop_hold(False, 0.0, stop_target + 0.5, 0.1, 0.0)
  assert not should_release_creep_stop_hold(False, 0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS, 0.1, 0.0)
  assert should_release_creep_stop_hold(False, 0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS + 0.01, 0.1, 0.0)
  assert should_hold_creep_to_stop_gap(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_FOLLOW_EXCESS, 0.0, 0.0, release_active=True)
  slow_roll_target = get_lead_stop_presentation_distance(0.2, 0.0, 0.0, 1.0)
  assert should_release_creep_stop_hold(True, 0.2, slow_roll_target + CREEP_TO_STOP_GAP_FOLLOW_EXCESS, 0.0, 0.0)
  assert not should_release_creep_stop_hold(True, 0.2, slow_roll_target + 0.15, 0.0, 0.0)


def test_creep_stop_hold_blocks_twitch_after_small_stopped_lead_creep():
  one_meter_extra_gap = STOP_DISTANCE + 1.0

  assert should_hold_creep_to_stop_gap(0.0, one_meter_extra_gap, 0.0, 0.0, release_active=True)
  assert should_hold_creep_to_stop_gap(0.0, one_meter_extra_gap, 0.1, 0.0, release_active=False)
  assert not should_hold_creep_to_stop_gap(0.0, one_meter_extra_gap, 0.8, 0.6, release_active=False)
  assert not should_hold_creep_to_stop_gap(0.0, STOP_DISTANCE + 1.01, 0.0, 0.0, release_active=True)


def test_creep_stop_hold_uses_stop_distance_for_uncertain_lead():
  assert should_hold_creep_to_stop_gap(0.0, 5.3, 0.0, 0.0, model_prob=0.5)
  assert not should_release_creep_stop_hold(False, 0.0, 5.5, 0.1, 0.0, model_prob=0.5)


def test_creep_to_stop_gap_smooths_confirmed_pullaway_step():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.8, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.8, 1.0, False)

  assert active
  assert 0.30 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_exits_stopped_gap_window_after_four_meter_excess():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 4.0, 1.2, 1.0, False)

  assert active
  assert accel == pytest.approx(CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX)

  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 4.01, 1.2, 1.0, False)

  assert not active
  assert accel == pytest.approx(0.0)


def test_creep_to_stop_gap_uses_predicted_pullaway_before_speed_threshold():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.05, 1.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.05, 1.0, False, a_lead=1.0, a_lead_tau=0.0
  )

  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_predicts_pullaway_before_full_start_excess():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.05, 1.0, 1.0)
  predicted_v_lead, predicted_gap_opening = get_predicted_lead_pullaway(0.05, 1.0, 0.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + 0.35, 0.05, 1.0, False, a_lead=1.0, a_lead_tau=0.0
  )

  assert predicted_v_lead >= 0.35
  assert predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING
  assert active
  assert accel >= longitudinal_planner.CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN


def test_creep_to_stop_gap_uses_model_lead_pullaway_prediction():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  d_rel = stop_target + CREEP_TO_STOP_GAP_START_EXCESS - 0.5
  model_v_lead, model_gap_opening = get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel), 0.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, d_rel, 0.0, 1.0, False,
    model_predicted_v_lead=model_v_lead,
    model_predicted_gap_opening=model_gap_opening,
  )

  assert has_predicted_lead_pullaway(d_rel - stop_target, model_v_lead, model_gap_opening)
  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_transitions_to_normal_following_after_pullaway_window():
  stop_target = get_lead_stop_presentation_distance(0.0, 1.2, 0.6, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 1.2, 1.0, False, a_lead=0.6
  )
  assert active
  assert accel > 0.0

  active, accel = get_creep_to_stop_gap_accel(
    1.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS + 0.5, 1.8, 1.0, active, a_lead=0.6
  )
  assert not active
  assert accel == pytest.approx(0.0)

  active, accel = get_creep_to_stop_gap_accel(
    0.2, stop_target + CREEP_TO_STOP_GAP_MAX_EXCESS + 0.1, 1.8, 1.0, True, a_lead=0.6
  )
  assert not active
  assert accel == pytest.approx(0.0)


def test_model_lead_pullaway_prediction_requires_confident_matching_lead():
  d_rel = STOP_DISTANCE + 0.35

  assert get_model_lead_pullaway(make_model_msg_lead(d_rel, prob=0.5), make_radar_lead(d_rel), 0.0) == (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel, model_prob=0.4), 0.0) == (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel + 3.0), make_radar_lead(d_rel), 0.0) == (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel, v_lead=2.0), 0.0) == (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel), 0.4) == (0.0, 0.0)


def test_model_lead_pullaway_skips_model_arrays_for_invalid_radar_distance(monkeypatch):
  calls = 0
  base_asarray = np.asarray

  def counting_asarray(*args, **kwargs):
    nonlocal calls
    calls += 1
    return base_asarray(*args, **kwargs)

  monkeypatch.setattr(longitudinal_planner.np, "asarray", counting_asarray)

  assert get_model_lead_pullaway(make_model_msg_lead(), make_radar_lead(d_rel=np.nan), 0.0) == (0.0, 0.0)
  assert calls == 0


def test_model_lead_pullaway_prediction_requires_low_uncertainty():
  d_rel = STOP_DISTANCE + 0.35

  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, x_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)
  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, y_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)
  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, v_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)
  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, horizon_x_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)
  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, horizon_y_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)
  assert get_model_lead_pullaway(
    make_model_msg_lead(d_rel, horizon_v_std=CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD + 0.1), make_radar_lead(d_rel), 0.0
  ) == (0.0, 0.0)


def test_model_lead_pullaway_prediction_requires_lateral_match():
  d_rel = STOP_DISTANCE + 0.35

  assert get_model_lead_pullaway(make_model_msg_lead(d_rel, y_rel=0.4), make_radar_lead(d_rel, y_rel=0.4), 0.0) != (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel, y_rel=1.0), make_radar_lead(d_rel, y_rel=0.0), 0.0) == (0.0, 0.0)
  assert get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel, y_rel=np.nan), 0.0) == (0.0, 0.0)


def test_predicted_pullaway_arms_at_small_gap_excess_for_strong_pullaway():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  gap_excess = 0.1
  d_rel = stop_target + gap_excess
  v_lead = 0.0
  a_lead = 0.8
  a_lead_tau = 0.0

  predicted_v_lead, predicted_gap_opening = get_predicted_lead_pullaway(v_lead, a_lead, a_lead_tau)
  assert predicted_v_lead >= CREEP_TO_STOP_GAP_PREDICT_MIN_LEAD_SPEED
  assert predicted_gap_opening >= CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING

  active, accel = get_creep_to_stop_gap_accel(
    0.0, d_rel, v_lead, 1.0, False, a_lead=a_lead, a_lead_tau=a_lead_tau,
  )

  assert active
  assert accel >= longitudinal_planner.CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MIN


def test_creep_to_stop_gap_prediction_requires_clear_gap_opening():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.1, 1.0)
  predicted_v_lead, predicted_gap_opening = get_predicted_lead_pullaway(0.0, 0.1, 0.0)
  active, _ = get_creep_to_stop_gap_accel(0.0, stop_target + 0.35, 0.0, 1.0, False, a_lead=0.1, a_lead_tau=0.0)

  assert predicted_v_lead > 0.0
  assert predicted_gap_opening < CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING
  assert not active


def test_creep_to_stop_gap_blocker_uses_adaptive_decelerating_lead_target():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, -0.6, 1.0)
  active, _ = get_creep_to_stop_gap_accel(0.0, stop_target - 0.1, 0.0, 1.0, False, a_lead=-0.6)

  assert stop_target == pytest.approx(STOP_DISTANCE)
  assert not active


def test_creep_to_stop_gap_release_requires_confirmed_safe_lead():
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 0.4, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, -0.4, 1.0, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 1.0, False, brake_pressed=True)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + 1.0, 0.0, 1.0, False, gas_pressed=True)[0]
  assert not get_creep_to_stop_gap_accel(0.4, STOP_DISTANCE + 1.0, 0.0, 1.0, False)[0]
  assert not get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_MAX_EXCESS + 0.1, 0.0, 1.0, False)[0]
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 1.0, 1.0)
  assert not get_creep_to_stop_gap_accel(0.0, stop_target, 0.0, 1.0, False, a_lead=1.0, a_lead_tau=0.0)[0]


def test_creep_to_stop_gap_hold_covers_low_speed_settle_distance():
  stop_target = get_lead_stop_presentation_distance(0.1, 0.0, 0.0, 1.0)
  assert should_hold_creep_to_stop_gap(0.1, stop_target + 0.34, 0.0, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, stop_target + CREEP_TO_STOP_GAP_HOLD_EXCESS + 0.1, 0.0, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, stop_target + 0.3, 0.3, 0.0)
  assert not should_hold_creep_to_stop_gap(0.1, stop_target + 0.3, 0.0, 0.1)
  assert not should_hold_creep_to_stop_gap(0.1, stop_target + 0.3, 0.0, 0.0, predicted_pullaway=True)


def test_stopped_lead_gap_fill_arms_from_close_stopped_lead_only():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  assert should_arm_stopped_lead_gap_fill(0.0, stop_target + 0.2, 0.0, 1.0)
  assert not should_arm_stopped_lead_gap_fill(0.4, stop_target + 0.2, 0.0, 1.0)
  assert not should_arm_stopped_lead_gap_fill(0.0, stop_target + 1.0, 0.0, 1.0)
  assert not should_arm_stopped_lead_gap_fill(0.0, stop_target + 0.2, 0.5, 1.0)
  assert not should_arm_stopped_lead_gap_fill(0.0, stop_target + 0.2, 0.0, 0.4)
  assert not should_arm_stopped_lead_gap_fill(0.0, stop_target + 0.2, 0.0, 1.0, gas_pressed=True)


def test_stopped_lead_gap_fill_arms_near_adaptive_stopped_target():
  assert should_arm_stopped_lead_gap_fill(0.0, 5.2, 0.0, 1.0)


def test_stopped_lead_gap_fill_uses_stop_distance_for_decelerating_lead():
  assert not should_arm_stopped_lead_gap_fill(0.0, 5.2, 0.0, 1.0, a_lead=-0.6)
  assert not get_stopped_lead_gap_fill_accel(0.0, 9.2, 0.0, 1.0, True, a_lead=-0.6)[0]


def test_stopped_lead_gap_fill_creeps_to_new_far_stopped_lead():
  active, accel = get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0, 0.0, 1.0, True)

  assert active
  assert 0.0 < accel <= STOPPED_LEAD_GAP_FILL_ACCEL_MAX


def test_stopped_lead_gap_fill_starts_after_near_pullaway_window():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  assert STOPPED_LEAD_GAP_FILL_MIN_EXCESS == CREEP_TO_STOP_GAP_MAX_EXCESS
  assert not get_stopped_lead_gap_fill_accel(0.0, stop_target + STOPPED_LEAD_GAP_FILL_MIN_EXCESS, 0.0, 1.0, True)[0]

  active, accel = get_stopped_lead_gap_fill_accel(0.0, stop_target + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 0.1, 0.0, 1.0, True)

  assert active
  assert 0.0 < accel <= STOPPED_LEAD_GAP_FILL_ACCEL_MAX


def test_stopped_lead_gap_fill_blocks_without_recent_close_stop_or_target_lead():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  assert not get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0, 0.0, 1.0, False)[0]
  assert not get_stopped_lead_gap_fill_accel(0.0, stop_target + CREEP_TO_STOP_GAP_MAX_EXCESS, 0.0, 1.0, True)[0]
  assert not get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MAX_EXCESS + 1.0, 0.0, 1.0, True)[0]
  assert not get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0, 1.5, 1.0, True)[0]
  assert not get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0, 0.0, 0.4, True)[0]
  assert not get_stopped_lead_gap_fill_accel(0.0, STOP_DISTANCE + STOPPED_LEAD_GAP_FILL_MIN_EXCESS + 5.0, 0.0, 1.0, True, brake_pressed=True)[0]


def test_lead_departure_relaxation_requires_gap_growth_and_pullaway():
  assert get_lead_departure_relaxation(0.0, 0.5, 1.0) == pytest.approx(0.0)
  assert get_lead_departure_relaxation(0.0, 1.5, 0.2) == pytest.approx(0.0)
  assert get_lead_departure_relaxation(0.6, 0.7, 1.0) == pytest.approx(0.0)


def test_lead_departure_runway_uses_extra_stopped_gap():
  assert get_lead_departure_available_runway(0.0, STOP_DISTANCE - 0.1, 0.0) == pytest.approx(0.0)
  assert get_lead_departure_available_runway(0.0, STOP_DISTANCE + 0.6, 0.0) == pytest.approx(0.6)


def test_lead_departure_relaxation_can_use_extra_stopped_gap():
  runway = get_lead_departure_available_runway(0.0, STOP_DISTANCE + 0.6, 0.0)

  assert get_lead_departure_relaxation(0.0, 0.5, runway) == pytest.approx(0.0)
  assert 0.0 < get_lead_departure_relaxation(0.0, 0.9, runway) < LEAD_DEPARTURE_RELAXATION_MAX


def test_lead_departure_relaxation_grows_with_confirmed_departure():
  mild_departure = get_lead_departure_relaxation(0.0, 0.9, 0.6)
  strong_departure = get_lead_departure_relaxation(0.0, 2.0, 1.0)

  assert 0.0 < mild_departure < strong_departure <= LEAD_DEPARTURE_RELAXATION_MAX


def test_lead_departure_relaxation_fades_out_as_ego_starts_creeping():
  stopped_relaxation = get_lead_departure_relaxation(0.0, 2.0, 1.0)
  creeping_relaxation = get_lead_departure_relaxation(0.8, 2.0, 1.0)

  assert 0.0 < creeping_relaxation < stopped_relaxation
  assert get_lead_departure_relaxation(1.0, 2.0, 1.0) == pytest.approx(0.0)


def test_lead_transition_release_requires_lateral_exit_and_persistence():
  assert get_lead_transition_lateral_blend(LEAD_TRANSITION_Y_REL_SOFT - 0.1) == pytest.approx(0.0)
  assert get_lead_transition_release_target(LEAD_TRANSITION_Y_REL_CONFIRM, 0.0) == pytest.approx(0.0)

  partial_release = get_lead_transition_release_target(
    0.5 * (LEAD_TRANSITION_Y_REL_SOFT + LEAD_TRANSITION_Y_REL_CONFIRM),
    0.5 * LEAD_TRANSITION_PERSISTENCE,
  )
  assert 0.0 < partial_release < 1.0
  assert get_lead_transition_release_target(LEAD_TRANSITION_Y_REL_CONFIRM, LEAD_TRANSITION_PERSISTENCE) == pytest.approx(1.0)


def test_lead_transition_releases_turning_lead_before_long_persistence():
  assert get_lead_transition_release_target(1.6, 0.2) == pytest.approx(1.0)


def test_lead_transition_preserves_release_through_lateral_track_churn():
  mpc = LongitudinalMpc(dt=0.1)
  lead = SimpleNamespace(status=True, radarTrackId=10, yRel=LEAD_TRANSITION_Y_REL_CONFIRM, dRel=20.0, vLeadK=8.0)

  mpc.update_lead_transition_state(0, lead)
  release_before_churn = mpc.update_lead_transition_state(0, lead)
  lead.radarTrackId = 11
  lead.dRel = 20.3
  lead.vLeadK = 8.2
  release_after_churn = mpc.update_lead_transition_state(0, lead)
  guard_timer_after_churn = mpc.lead_transition_guard_timers[0]

  assert release_after_churn >= release_before_churn
  assert guard_timer_after_churn > 0.0
  assert get_lead_transition_obstacle_release(release_after_churn, guard_timer_after_churn) < release_after_churn


def test_lead_transition_resets_release_on_opposite_side_track_churn():
  mpc = LongitudinalMpc(dt=0.1)
  lead = SimpleNamespace(status=True, radarTrackId=10, yRel=LEAD_TRANSITION_Y_REL_CONFIRM, dRel=20.0, vLeadK=8.0)

  mpc.update_lead_transition_state(0, lead)
  release_before_churn = mpc.update_lead_transition_state(0, lead)
  lead.radarTrackId = 11
  lead.yRel = -LEAD_TRANSITION_Y_REL_CONFIRM
  lead.dRel = 20.2
  lead.vLeadK = 8.1
  release_after_churn = mpc.update_lead_transition_state(0, lead)

  assert release_after_churn < release_before_churn


def test_lead_transition_resets_release_on_discontinuous_track_churn():
  mpc = LongitudinalMpc(dt=0.1)
  lead = SimpleNamespace(status=True, radarTrackId=10, yRel=LEAD_TRANSITION_Y_REL_CONFIRM, dRel=20.0, vLeadK=8.0)

  mpc.update_lead_transition_state(0, lead)
  release_before_churn = mpc.update_lead_transition_state(0, lead)
  lead.radarTrackId = 11
  lead.dRel = 35.0
  release_after_churn = mpc.update_lead_transition_state(0, lead)

  assert release_after_churn < release_before_churn


def test_lead_transition_fcw_counting_ignores_released_lead():
  should_count = getattr(long_mpc, "should_count_lead_transition_fcw", None)

  assert should_count is not None
  assert should_count(0.95, 0.0)
  assert not should_count(0.95, 1.0)
  assert not should_count(0.5, 0.0)


def test_lead_transition_soft_release_only_moves_toward_cruise():
  lead_obstacle = np.array([20.0, 25.0])
  cruise_obstacle = np.array([30.0, 23.0])

  released = get_lead_transition_cost_obstacle(lead_obstacle, cruise_obstacle, 0.5)

  assert released[0] == pytest.approx(25.0)
  assert released[1] == pytest.approx(25.0)


def test_lead_transition_guard_caps_near_term_positive_accel():
  assert np.all(get_lead_transition_accel_max(0.0) == ACCEL_MAX)

  guarded_accel_max = get_lead_transition_accel_max(0.55)
  assert guarded_accel_max[0] == pytest.approx(0.0)
  assert np.max(guarded_accel_max) == pytest.approx(ACCEL_MAX)


def test_lead_transition_guard_covers_output_delay_horizon():
  guarded_accel_max = get_lead_transition_accel_max(0.1)
  delayed_horizon_idx = int(np.argmax(long_mpc.T_IDXS >= LEAD_TRANSITION_GUARD_OUTPUT_DELAY))

  assert guarded_accel_max[delayed_horizon_idx] < 0.2


def test_lead_transition_guard_skips_accel_cap_generation_when_inactive(monkeypatch):
  def fail_guard_generation(_guard_timer):
    pytest.fail("inactive transition guard should not generate an accel cap")

  monkeypatch.setattr(long_mpc, "get_lead_transition_accel_max", fail_guard_generation)
  accel_max = np.array([ACCEL_MAX, ACCEL_MAX - 0.1])

  long_mpc.apply_lead_transition_accel_guard(accel_max, 0.0)

  assert accel_max.tolist() == pytest.approx([ACCEL_MAX, ACCEL_MAX - 0.1])


def test_lead_transition_obstacle_release_waits_for_guard():
  assert get_lead_transition_obstacle_release(1.0, LEAD_TRANSITION_GUARD_FADE_TIME) == pytest.approx(0.0)
  assert get_lead_transition_obstacle_release(1.0, 0.0) == pytest.approx(1.0)


def test_lead_transition_adjusted_accel_only_suppresses_decel():
  assert get_lead_transition_adjusted_accel(-2.0, 0.0) == pytest.approx(-2.0)
  assert get_lead_transition_adjusted_accel(-2.0, 0.5) == pytest.approx(-1.0)
  assert get_lead_transition_adjusted_accel(-2.0, 1.0) == pytest.approx(0.0)
  assert get_lead_transition_adjusted_accel(0.5, 1.0) == pytest.approx(0.5)

  adjusted = get_lead_transition_adjusted_accel(np.array([-2.0, 0.5]), 0.5)
  assert adjusted.tolist() == pytest.approx([-1.0, 0.5])


def test_approach_brake_stays_stock_for_small_closure():
  assert get_approach_brake(0.0) == pytest.approx(APPROACH_BRAKE)
  assert get_approach_brake(1.5) == pytest.approx(APPROACH_BRAKE)


def test_approach_brake_ramps_down_for_stronger_closure():
  moderate_closure_brake = get_approach_brake(3.0)
  strong_closure_brake = get_approach_brake(5.0)

  assert APPROACH_BRAKE_MIN <= strong_closure_brake < moderate_closure_brake < APPROACH_BRAKE
  assert strong_closure_brake == pytest.approx(APPROACH_BRAKE_MIN)


@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 35.0])
def test_approach_follow_distance_matches_steady_state_when_speeds_match(speed):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  x_lead = get_desired_follow_distance(speed, speed, t_follow)
  expected = get_lead_stop_presentation_distance(speed, speed) if speed == 0.0 else get_desired_follow_distance(speed, speed, t_follow)
  assert get_approach_follow_distance(x_lead, speed, speed, t_follow) == pytest.approx(expected)


def test_approach_follow_distance_uses_runway_before_danger_zone():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  closing_speed = v_ego - v_lead
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 25.0

  approach_gap = get_approach_follow_distance(x_lead, v_ego, v_lead, t_follow)
  expected_gap = max(
    t_follow * v_lead + stop_distance_buffer(v_lead) + (closing_speed**2) / (2 * get_approach_brake(closing_speed)),
    get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER,
  )

  assert approach_gap == pytest.approx(expected_gap)
  assert get_lead_danger_distance(v_ego, v_lead, t_follow) < approach_gap < get_desired_follow_distance(v_ego, v_lead, t_follow)


def test_approach_runway_blend_stays_off_near_legacy_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 1.0

  assert get_approach_runway_blend(x_lead, v_ego, v_lead, t_follow) == pytest.approx(0.0)


def test_approach_runway_blend_reaches_full_with_large_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  x_lead = get_desired_follow_distance(v_ego, v_lead, t_follow) + 25.0

  assert get_approach_runway_blend(x_lead, v_ego, v_lead, t_follow) == pytest.approx(1.0)


def test_approach_available_runway_uses_stop_distance_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  steady_runway = get_approach_available_runway(40.0, 20.0, 15.0, t_follow, a_lead=0.0)
  slowing_runway = get_approach_available_runway(40.0, 20.0, 15.0, t_follow, a_lead=-1.0)
  reserve = get_moving_lead_stop_reserve(20.0, 15.0, 5.0, -1.0)

  assert steady_runway == pytest.approx(max(40.0 - get_desired_follow_distance(20.0, 15.0, t_follow), 0.0))
  assert slowing_runway == pytest.approx(40.0 + get_stopped_equivalence_factor(15.0) - STOP_DISTANCE - reserve)
  assert slowing_runway > steady_runway


def test_approach_available_runway_uses_fixed_stop_gap_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 15.0
  x_lead = 35.0

  slowing_runway = get_approach_available_runway(x_lead, v_ego, v_lead, t_follow, a_lead=-1.0)
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, -1.0)

  assert slowing_runway == pytest.approx(x_lead + get_stopped_equivalence_factor(v_lead) - STOP_DISTANCE - reserve)
  assert slowing_runway != pytest.approx(x_lead + get_stopped_equivalence_factor(v_lead) - stop_distance_buffer(v_ego))


def test_moving_lead_stop_reserve_tapers_to_fixed_stop_gap():
  full_reserve = get_moving_lead_stop_reserve(v_ego=8.0, v_lead=2.0, closing_speed=2.0, a_lead=-1.0)

  assert full_reserve == pytest.approx(MOVING_LEAD_STOP_RESERVE_MAX)
  assert get_moving_lead_stop_reserve(8.0, 2.0, 0.0, -1.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(0.0, 0.0, 2.0, 0.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(8.0, 2.0, 2.0, 0.0) == pytest.approx(0.0)
  assert get_moving_lead_stop_reserve(8.0, 0.0, 2.0, 0.0) == pytest.approx(0.0)


def test_slow_moving_lead_runway_relaxation_reaches_one_meter_for_controlled_roll():
  relaxation_fn = getattr(long_mpc, "get_slow_moving_lead_runway_relaxation", None)
  assert relaxation_fn is not None

  relaxation = relaxation_fn(v_ego=3.0, v_lead=1.0, closing_speed=0.4, a_lead=-0.8)

  assert relaxation == pytest.approx(1.0)


def test_slow_moving_lead_runway_relaxation_stays_off_for_stopped_and_fast_leads():
  relaxation_fn = getattr(long_mpc, "get_slow_moving_lead_runway_relaxation", None)
  assert relaxation_fn is not None

  assert relaxation_fn(v_ego=3.0, v_lead=0.0, closing_speed=0.4, a_lead=-0.8) == pytest.approx(0.0)
  assert relaxation_fn(v_ego=12.0, v_lead=8.0, closing_speed=0.4, a_lead=-0.8) == pytest.approx(0.0)


def test_slow_moving_lead_runway_relaxation_fades_for_urgent_closure():
  relaxation_fn = getattr(long_mpc, "get_slow_moving_lead_runway_relaxation", None)
  assert relaxation_fn is not None

  assert relaxation_fn(v_ego=6.0, v_lead=1.0, closing_speed=5.0, a_lead=-0.8) == pytest.approx(0.0)


def test_slow_moving_lead_runway_relaxation_is_vectorized_and_bounded(monkeypatch):
  monkeypatch.setattr(long_mpc, "SLOW_MOVING_LEAD_RUNWAY_RELAXATION_MAX", 2.0)

  relaxations = long_mpc.get_slow_moving_lead_runway_relaxation(
    v_ego=np.array([3.0, 3.0, 6.0, 12.0]),
    v_lead=np.array([0.0, 1.0, 1.0, 8.0]),
    closing_speed=np.array([0.4, 0.4, 5.0, 0.4]),
    a_lead=np.array([-0.8, -0.8, -0.8, -0.8]),
  )

  np.testing.assert_allclose(relaxations, [0.0, 1.0, 0.0, 0.0])


def test_stop_runway_blend_normalizes_to_capped_relaxation(monkeypatch):
  monkeypatch.setattr(long_mpc, "SLOW_MOVING_LEAD_RUNWAY_RELAXATION_MAX", 2.0)

  blend = get_lead_stop_runway_blend(v_ego=1.0, v_lead=2.0, a_lead=0.0, closing_speed=0.0)

  assert blend == pytest.approx(1.0)


def test_steady_slow_moving_lead_gets_stop_runway_preference():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 3.0
  v_lead = 1.0
  x_lead = 12.0

  preference = get_lead_stop_runway_preference(x_lead, v_ego, v_lead, t_follow, a_lead=0.0)

  assert preference > 0.0


def test_low_speed_slowing_lead_runway_can_use_one_meter_relaxed_floor():
  v_ego = 0.5
  v_lead = 1.1
  a_lead = -0.8
  closing_speed = 0.0

  assert get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead) == pytest.approx(STOP_DISTANCE - 1.0)


def test_urgent_slow_moving_lead_runway_keeps_full_stop_floor_and_decel():
  x_lead = 7.0
  v_ego = 5.0
  v_lead = 1.0
  a_lead = -0.8
  closing_speed = v_ego - v_lead

  assert get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead) >= STOP_DISTANCE
  assert get_lead_stop_runway_required_decel(x_lead, v_ego, v_lead, closing_speed, a_lead) > LEAD_STOP_RUNWAY_BRAKE


def test_approach_runway_blend_uses_stop_runway_for_slowing_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=0.0) == pytest.approx(0.0)
  assert get_approach_runway_blend(40.0, 20.0, 15.0, t_follow, a_lead=-1.0) == pytest.approx(1.0)


def test_low_speed_stopped_lead_uses_stop_runway_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 2.0
  runway_gap = get_lead_stop_presentation_distance(v_ego, 0.0) + v_ego**2 / (2 * LEAD_STOP_RUNWAY_BRAKE)

  assert get_lead_stop_runway_blend(v_ego, 0.0, 0.0) == pytest.approx(1.0)
  assert get_lead_stop_runway_preference(12.0, v_ego, 0.0, t_follow, 0.0) == pytest.approx(1.0)
  assert get_lead_stop_runway_gap(v_ego, 0.0, v_ego, 0.0) == pytest.approx(runway_gap)
  assert get_approach_follow_distance(12.0, v_ego, 0.0, t_follow, a_lead=0.0) == pytest.approx(runway_gap)


def test_low_speed_slowing_lead_runway_accounts_for_lead_stop_point():
  v_ego = 3.0
  v_lead = 1.0
  a_lead = -0.8
  closing_speed = v_ego - v_lead
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, closing_speed, a_lead)
  relaxation = long_mpc.get_slow_moving_lead_runway_relaxation(v_ego, v_lead, closing_speed, a_lead)
  stop_floor = STOP_DISTANCE - relaxation
  expected_gap = stop_floor + reserve + v_ego**2 / (2 * LEAD_STOP_RUNWAY_BRAKE) - get_stopped_equivalence_factor(v_lead)

  assert get_lead_stop_runway_blend(v_ego, v_lead, a_lead) > 0.0
  assert get_lead_stop_runway_gap(v_ego, v_lead, closing_speed, a_lead) == pytest.approx(max(stop_floor, expected_gap))


def test_stop_runway_preference_fades_for_urgent_short_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  hot_preference = get_lead_stop_runway_preference(8.0, 2.0, 0.0, t_follow, 0.0)
  controlled_preference = get_lead_stop_runway_preference(8.0, 0.4, 0.0, t_follow, 0.0)
  comfortable_preference = get_lead_stop_runway_preference(12.0, 2.0, 0.0, t_follow, 0.0)

  assert get_lead_stop_runway_required_decel(8.0, 2.0, 0.0, 2.0, 0.0) > LEAD_STOP_RUNWAY_BRAKE
  assert hot_preference < controlled_preference <= comfortable_preference
  assert comfortable_preference == pytest.approx(1.0)


def test_stop_runway_urgency_blends_back_as_closing_stabilizes():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  hot_urgency = get_lead_stop_runway_urgency(8.0, 2.0, 0.0, t_follow, 0.0)
  controlled_urgency = get_lead_stop_runway_urgency(8.0, 0.4, 0.0, t_follow, 0.0)

  assert hot_urgency > controlled_urgency
  assert controlled_urgency < 1.0


def test_stop_runway_blend_stays_off_at_higher_speed():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_lead_stop_runway_blend(8.0, 0.0, 0.0) == pytest.approx(0.0)
  assert get_approach_runway_blend(12.0, 8.0, 0.0, t_follow, a_lead=0.0) == pytest.approx(0.0)


def test_crawl_comfort_stays_off_above_regular_stop_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(12.0, 4.0, 3.2, -0.4, t_follow)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_crawl_comfort_does_not_stage_closing_slowing_lead_at_ten_meters():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(STOP_DISTANCE + 4.0, 3.0, 1.0, -1.0, t_follow)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_crawl_comfort_still_brakes_near_regular_stop_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(STOP_DISTANCE + 0.6, 1.5, 0.6, -0.5, t_follow)

  assert -LEAD_CRAWL_BRAKE_MAX <= target < 0.0
  assert cost > 0.0


def test_crawl_comfort_fades_out_for_urgent_short_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(7.0, 4.0, 0.0, -1.0, t_follow)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_crawl_comfort_stays_off_for_stopped_lead_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(12.0, 2.0, 0.0, 0.0, t_follow)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_crawl_comfort_uses_gentle_accel_for_opening_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(STOP_DISTANCE + 3.5, 3.0, 4.2, 0.3, t_follow)

  assert 0.0 < target <= LEAD_CRAWL_ACCEL_MAX
  assert cost > 0.0


def test_crawl_comfort_stops_chasing_opening_lead_after_ten_meters():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_crawl_comfort_target(STOP_DISTANCE + 4.5, 0.0, 1.0, 0.2, t_follow)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_crawl_comfort_disables_outside_crawl_band():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  _, high_speed_cost = get_lead_crawl_comfort_target(12.0, 9.0, 8.0, -0.4, t_follow)
  _, far_gap_cost = get_lead_crawl_comfort_target(30.0, 3.0, 4.0, 0.3, t_follow)

  assert high_speed_cost == pytest.approx(0.0)
  assert far_gap_cost == pytest.approx(0.0)


def test_crawl_accel_limit_only_applies_to_opening_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  assert get_lead_crawl_accel_max(STOP_DISTANCE + 4.0, 3.0, 4.5, 0.3, t_follow) == pytest.approx(LEAD_CRAWL_ACCEL_LIMIT)
  assert get_lead_crawl_accel_max(12.0, 3.0, 4.5, 0.3, t_follow) == pytest.approx(LEAD_CRAWL_ACCEL_LIMIT)
  assert get_lead_crawl_accel_max(12.0, 3.0, 3.0, 0.0, t_follow) == pytest.approx(ACCEL_MAX)
  assert get_lead_crawl_accel_max(30.0, 3.0, 4.5, 0.3, t_follow) == pytest.approx(ACCEL_MAX)


def test_surge_damping_stays_off_for_steady_crawl():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_surge_damping_target(12.0, 3.0, 3.2, 0.0, t_follow, decel_memory=0.0)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_surge_damping_softens_post_decel_opening_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_surge_damping_target(12.0, 3.0, 3.7, 0.2, t_follow, decel_memory=0.8)

  assert 0.0 < target <= LEAD_SURGE_DAMPING_ACCEL_MAX
  assert target < LEAD_CRAWL_ACCEL_MAX
  assert cost > 0.0


def test_surge_damping_blocks_standstill_and_short_gaps():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  desired_gap = get_desired_follow_distance(3.0, 3.7, t_follow)

  _, standstill_cost = get_lead_surge_damping_target(12.0, 0.0, 1.5, 0.3, t_follow, decel_memory=0.8)
  _, short_gap_cost = get_lead_surge_damping_target(desired_gap - 0.2, 3.0, 3.7, 0.3, t_follow, decel_memory=0.8)

  assert standstill_cost == pytest.approx(0.0)
  assert short_gap_cost == pytest.approx(0.0)


def test_surge_damping_exits_for_clear_pullaway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_surge_damping_target(14.0, 3.0, 5.5, 0.8, t_follow, decel_memory=0.8)

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_surge_damping_decel_memory_decays_and_resets():
  mpc = LongitudinalMpc(dt=0.1)
  lead = SimpleNamespace(status=True, aLeadK=-0.8)

  assert mpc.update_lead_surge_decel_memory(0, lead) == pytest.approx(0.8)

  lead.aLeadK = 0.0
  decayed_memory = mpc.update_lead_surge_decel_memory(0, lead)

  assert 0.0 < decayed_memory < 0.8
  assert mpc.update_lead_surge_decel_memory(0, SimpleNamespace(status=False)) == pytest.approx(0.0)
  assert mpc.lead_surge_decel_memories[0] == pytest.approx(0.0)
  assert mpc.update_lead_surge_decel_memory(1, SimpleNamespace(status=True, aLeadK=-2.0)) == pytest.approx(LEAD_SURGE_DAMPING_DECEL_MEMORY_MAX)


def test_selected_lead_targets_ignore_non_dominant_lead():
  lead_0_targets = np.array([0.0, 0.0, 0.0])
  lead_1_targets = np.array([LEAD_SURGE_DAMPING_ACCEL_MAX, LEAD_SURGE_DAMPING_ACCEL_MAX, LEAD_SURGE_DAMPING_ACCEL_MAX])
  lead_0_costs = np.array([0.0, 0.0, 0.0])
  lead_1_costs = np.array([0.6, 0.6, 0.6])
  dominant_obstacle = np.array([0, 2, 0])

  targets, costs = get_selected_lead_targets(lead_0_targets, lead_1_targets, lead_0_costs, lead_1_costs, dominant_obstacle)

  assert targets.tolist() == pytest.approx([0.0, 0.0, 0.0])
  assert costs.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_combined_accel_target_uses_closing_cushion_for_dominant_lead_only():
  accel_match_targets = np.array([0.0, 0.0, 0.0])
  accel_match_costs = np.array([0.0, 0.0, 0.0])
  lead_0_closing_cushion_targets = np.array([-0.4, -0.4, -0.4])
  lead_1_closing_cushion_targets = np.array([-0.1, -0.1, -0.1])
  lead_0_closing_cushion_costs = np.array([2.0, 2.0, 2.0])
  lead_1_closing_cushion_costs = np.array([8.0, 8.0, 8.0])
  zero_targets = np.zeros(3)
  zero_costs = np.zeros(3)
  dominant_obstacle = np.array([0, 1, 2])

  targets, costs = get_combined_accel_target(
    accel_match_targets, accel_match_costs,
    lead_0_closing_cushion_targets, lead_1_closing_cushion_targets,
    lead_0_closing_cushion_costs, lead_1_closing_cushion_costs,
    dominant_obstacle,
    zero_targets, zero_costs,
    zero_targets, zero_costs,
    zero_targets, zero_costs,
    zero_targets, zero_costs,
  )

  np.testing.assert_allclose(targets, [-0.4, -0.1, 0.0], rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(costs, [2.0, 8.0, 0.0], rtol=0.0, atol=1e-12)


def test_stop_approach_comfort_targets_moderate_stopped_lead_brake():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_stop_approach_comfort_target(40.0, 10.0, 0.0, 0.0, t_follow)

  assert -LEAD_STOP_APPROACH_DECEL_CAP <= target < -0.5
  assert cost > 0.0


def test_stop_approach_comfort_stays_off_for_moving_or_low_speed_leads():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  _, moving_cost = get_lead_stop_approach_comfort_target(40.0, 10.0, 2.0, 0.0, t_follow)
  _, low_speed_cost = get_lead_stop_approach_comfort_target(12.0, 2.0, 0.0, 0.0, t_follow)

  assert moving_cost == pytest.approx(0.0)
  assert low_speed_cost == pytest.approx(0.0)


def test_moving_stop_approach_comfort_targets_confirmed_active_stop():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target_fn = getattr(long_mpc, "get_moving_lead_stop_approach_comfort_target", None)
  assert target_fn is not None

  target, cost = target_fn(45.0, 18.0, 15.0, -1.0, t_follow)

  assert -long_mpc.MOVING_LEAD_STOP_APPROACH_DECEL_CAP <= target < -0.4
  assert cost > 0.0


def test_moving_stop_approach_comfort_requires_confirmed_threat():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target_fn = getattr(long_mpc, "get_moving_lead_stop_approach_comfort_target", None)
  assert target_fn is not None

  weak_target, weak_cost = target_fn(45.0, 18.0, 15.0, -0.2, t_follow)
  opening_target, opening_cost = target_fn(45.0, 15.0, 18.0, -1.0, t_follow)
  stopped_target, stopped_cost = target_fn(45.0, 18.0, 0.5, -1.0, t_follow)

  assert weak_target == pytest.approx(0.0)
  assert weak_cost == pytest.approx(0.0)
  assert opening_target == pytest.approx(0.0)
  assert opening_cost == pytest.approx(0.0)
  assert stopped_target == pytest.approx(0.0)
  assert stopped_cost == pytest.approx(0.0)


def test_far_hard_braking_lead_uses_runway_instead_of_early_hard_brake():
  v_ego = 18.0
  v_lead = 10.0
  d_rel = 86.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel > get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_pre_target_runway_thresholds_follow_personality():
  relaxed_threshold = get_pre_target_runway_decel_threshold(get_T_FOLLOW(log.LongitudinalPersonality.relaxed))
  standard_threshold = get_pre_target_runway_decel_threshold(get_T_FOLLOW(log.LongitudinalPersonality.standard))
  aggressive_threshold = get_pre_target_runway_decel_threshold(get_T_FOLLOW(log.LongitudinalPersonality.aggressive))

  assert relaxed_threshold == pytest.approx(PRE_TARGET_RUNWAY_DECEL_THRESHOLD_RELAXED)
  assert standard_threshold == pytest.approx(PRE_TARGET_RUNWAY_DECEL_THRESHOLD_STANDARD)
  assert aggressive_threshold == pytest.approx(PRE_TARGET_RUNWAY_DECEL_THRESHOLD_AGGRESSIVE)
  assert relaxed_threshold < standard_threshold < aggressive_threshold


def test_moving_stop_approach_caps_pre_target_to_coast_without_runway_urgency():
  v_ego = 15.0
  v_lead = 12.0
  a_lead = -0.8
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = desired_gap + 40.0
  closing_speed = max(v_ego - v_lead, 0.0)

  required_decel = get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead)
  threshold = get_pre_target_runway_decel_threshold(t_follow)
  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel > desired_gap
  assert required_decel < threshold
  assert MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN <= target <= 0.0
  assert cost > 0.0


def test_moving_stop_approach_allows_pre_target_brake_when_runway_is_urgent():
  v_ego = 15.0
  v_lead = 12.0
  a_lead = -0.8
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = desired_gap + 15.0
  closing_speed = max(v_ego - v_lead, 0.0)

  required_decel = get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead)
  threshold = get_pre_target_runway_decel_threshold(t_follow)
  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel > desired_gap
  assert required_decel > threshold
  assert target < MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN
  assert cost > 0.0


def test_moving_stop_approach_anticipates_confirmed_low_closure():
  v_ego = 19.2
  v_lead = 17.0
  d_rel = 53.0
  a_lead = -1.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel > get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert target < -0.4
  assert cost > 0.0


def test_route_like_slowing_moving_lead_prefers_moderate_decel():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)
  target, cost = get_moving_lead_stop_approach_comfort_target(22.0, 11.2, 8.0, -1.68, t_follow)

  assert -1.5 < target < -0.3
  assert cost > 0.0


def test_hard_braking_moving_lead_keeps_stronger_target_when_close():
  v_ego = 18.0
  v_lead = 10.0
  d_rel = 65.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel < get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert -long_mpc.MOVING_LEAD_STOP_APPROACH_DECEL_CAP <= target < -1.0
  assert cost > 0.0


def test_hard_braking_moving_lead_keeps_stronger_target_near_danger_boundary():
  v_ego = 18.0
  v_lead = 10.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  min_gap = get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER
  d_rel = min_gap + 0.5

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert target < -1.0
  assert cost > 0.0


def test_low_speed_hard_braking_moving_lead_keeps_stronger_target_near_danger_boundary():
  v_ego = 10.0
  v_lead = 8.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  min_gap = get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER
  d_rel = min_gap + 0.5

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert target < -1.0
  assert cost > 0.0


def test_low_speed_hard_braking_moving_lead_keeps_stronger_target_with_runway_margin():
  v_ego = 10.0
  v_lead = 8.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  min_gap = get_lead_danger_distance(v_ego, v_lead, t_follow) + APPROACH_MIN_GAP_BUFFER
  d_rel = min_gap + 5.0
  closing_speed = max(v_ego - v_lead, 0.0)
  required_decel = get_lead_stop_runway_required_decel(d_rel, v_ego, v_lead, closing_speed, a_lead)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel < get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert required_decel > long_mpc.MOVING_LEAD_STOP_APPROACH_DECEL_CAP
  assert -long_mpc.MOVING_LEAD_STOP_APPROACH_DECEL_CAP <= target < -1.0
  assert cost > 0.0


def test_moving_stop_approach_keeps_stronger_target_for_fast_runway_critical_lead():
  v_ego = 18.6
  v_lead = 15.0
  d_rel = 47.7
  a_lead = -1.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel < get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert target < -1.4
  assert cost > 0.0


def test_moving_stop_approach_uses_light_decel_before_vlead_cushion_is_used():
  v_ego = 19.2
  v_lead = 17.0
  a_lead = -1.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = target_gap - 0.25 * 0.75 * t_follow * v_lead

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert -0.65 <= target < -0.2
  assert cost > 0.0


def test_moving_stop_approach_scales_decel_after_vlead_cushion_is_used():
  v_ego = 19.2
  v_lead = 17.0
  a_lead = -1.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.aggressive)
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = target_gap - 0.75 * t_follow * v_lead

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert target < -1.0
  assert cost > 0.0


def test_closer_hard_braking_lead_still_gets_comfort_brake_target():
  v_ego = 18.0
  v_lead = 10.0
  d_rel = 65.0
  a_lead = -3.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  target, cost = get_moving_lead_stop_approach_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert d_rel < get_desired_follow_distance(v_ego, v_lead, t_follow)
  assert -long_mpc.MOVING_LEAD_STOP_APPROACH_DECEL_CAP <= target < 0.0
  assert cost > 0.0


def test_lead_accel_match_tapers_positive_accel_under_time_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_lead = 12.0
  a_lead = 1.0
  target_gap = get_lead_time_gap_target(v_lead, t_follow)

  near_stop_target, near_stop_cost = get_lead_accel_match_target(v_lead, STOP_DISTANCE + 1.2, a_lead, t_follow)
  mid_gap_target, mid_gap_cost = get_lead_accel_match_target(v_lead, 0.5 * (STOP_DISTANCE + target_gap), a_lead, t_follow)
  target_gap_target, target_gap_cost = get_lead_accel_match_target(v_lead, target_gap, a_lead, t_follow)

  assert 0.0 < near_stop_target < mid_gap_target < target_gap_target <= a_lead
  assert near_stop_target == pytest.approx(a_lead * LEAD_ACCEL_MATCH_MIN_POSITIVE_BLEND, abs=0.05)
  assert 0.0 < near_stop_cost < mid_gap_cost < target_gap_cost


def test_low_speed_lead_accel_match_recovers_through_launch_cushion():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.3
  v_lead = 0.35
  a_lead = 1.0
  target_gap = get_lead_time_gap_target(v_lead, t_follow)

  near_gap_target, near_gap_cost = get_lead_accel_match_targets(
    v_lead, STOP_DISTANCE + 0.2, a_lead, t_follow, v_ego=v_ego, model_prob=1.0,
  )
  target_gap_target, target_gap_cost = get_lead_accel_match_targets(
    v_lead, target_gap, a_lead, t_follow, v_ego=v_ego, model_prob=1.0,
  )

  assert STOP_DISTANCE < target_gap < STOP_DISTANCE + 1.0
  assert 0.0 < near_gap_target < target_gap_target
  assert target_gap_target == pytest.approx(a_lead, abs=0.05)
  assert 0.0 < near_gap_cost < target_gap_cost


def test_short_gap_pullaway_response_uses_stop_distance_cushion():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  d_rel = 0.5 * (presentation_distance + STOP_DISTANCE)

  target, cost = get_short_gap_pullaway_response_target(v_ego, v_lead, d_rel, a_lead, t_follow, model_prob=1.0)

  assert presentation_distance < d_rel < STOP_DISTANCE
  assert 0.0 < target <= 0.55
  assert cost > 0.0


def test_short_gap_pullaway_response_blocks_weak_or_unsafe_cushion_use():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)

  weak_target, weak_cost = get_short_gap_pullaway_response_target(
    v_ego, v_lead, presentation_distance + 0.3, 0.1, t_follow, model_prob=1.0,
  )
  hard_closing_target, hard_closing_cost = get_short_gap_pullaway_response_target(
    v_ego=1.4, v_lead=0.35, d_rel=presentation_distance + 0.3, a_lead=0.8, t_follow=t_follow, model_prob=1.0,
  )
  floor_target, floor_cost = get_short_gap_pullaway_response_target(
    v_ego, v_lead, presentation_distance - 0.1, a_lead, t_follow, model_prob=1.0,
  )
  low_confidence_presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 0.0)
  low_confidence_target, low_confidence_cost = get_short_gap_pullaway_response_target(
    v_ego, v_lead, d_rel=low_confidence_presentation_distance - 0.1, a_lead=a_lead, t_follow=t_follow, model_prob=0.0,
  )

  assert weak_target == pytest.approx(0.0)
  assert weak_cost == pytest.approx(0.0)
  assert hard_closing_target == pytest.approx(0.0)
  assert hard_closing_cost == pytest.approx(0.0)
  assert floor_target == pytest.approx(0.0)
  assert floor_cost == pytest.approx(0.0)
  assert low_confidence_target == pytest.approx(0.0)
  assert low_confidence_cost == pytest.approx(0.0)


@pytest.mark.parametrize("blocked", [True, np.array([True, True])])
def test_short_gap_pullaway_response_blocks_driver_override_or_force_slow_decel(blocked):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  d_rel = 0.5 * (presentation_distance + STOP_DISTANCE)

  target, cost = get_short_gap_pullaway_response_target(
    v_ego, v_lead, d_rel, a_lead, t_follow, model_prob=1.0, blocked=blocked,
  )

  np.testing.assert_allclose(target, np.zeros_like(np.asarray(blocked, dtype=float)), rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(cost, np.zeros_like(np.asarray(blocked, dtype=float)), rtol=0.0, atol=1e-12)


def test_short_gap_pullaway_response_scales_with_personality():
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 1.0
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  d_rel = 0.5 * (presentation_distance + STOP_DISTANCE)

  relaxed_target, _ = get_short_gap_pullaway_response_target(
    v_ego, v_lead, d_rel, a_lead, get_T_FOLLOW(log.LongitudinalPersonality.relaxed), model_prob=1.0,
  )
  standard_target, _ = get_short_gap_pullaway_response_target(
    v_ego, v_lead, d_rel, a_lead, get_T_FOLLOW(log.LongitudinalPersonality.standard), model_prob=1.0,
  )
  aggressive_target, _ = get_short_gap_pullaway_response_target(
    v_ego, v_lead, d_rel, a_lead, get_T_FOLLOW(log.LongitudinalPersonality.aggressive), model_prob=1.0,
  )

  assert 0.0 < relaxed_target < standard_target < aggressive_target <= 0.55


def test_lead_accel_match_uses_stop_distance_cushion_for_confirmed_pullaway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  d_rel = 0.5 * (presentation_distance + STOP_DISTANCE)

  targets, costs = get_lead_accel_match_targets(np.array([v_lead]), np.array([d_rel]), np.array([a_lead]), t_follow, v_ego)

  assert presentation_distance < d_rel < STOP_DISTANCE
  assert targets[0] > 0.0
  assert costs[0] > 0.0


def test_crawl_comfort_uses_stop_distance_cushion_for_confirmed_pullaway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  presentation_distance = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  d_rel = 0.5 * (presentation_distance + STOP_DISTANCE)

  target, cost = get_lead_crawl_comfort_target(d_rel, v_ego, v_lead, a_lead, t_follow)

  assert presentation_distance < d_rel < STOP_DISTANCE
  assert target > 0.0
  assert cost > 0.0


def test_mpc_short_gap_pullaway_response_uses_lead_confidence_floor():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 0.6
  v_lead = 0.35
  a_lead = 0.8
  full_confidence_floor = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 1.0)
  low_confidence_floor = get_lead_stop_presentation_distance(v_ego, v_lead, a_lead, 0.0)
  d_rel = 0.5 * (full_confidence_floor + STOP_DISTANCE)

  full_confidence_targets, full_confidence_costs = get_lead_accel_match_targets(
    np.array([v_lead]), np.array([d_rel]), np.array([a_lead]), t_follow, v_ego, model_prob=np.array([1.0]),
  )
  low_confidence_targets, low_confidence_costs = get_lead_accel_match_targets(
    np.array([v_lead]), np.array([d_rel]), np.array([a_lead]), t_follow, v_ego, model_prob=np.array([0.0]),
  )
  low_confidence_crawl_target, low_confidence_crawl_cost = get_lead_crawl_comfort_target(
    d_rel, v_ego, v_lead, a_lead, t_follow, model_prob=0.0,
  )

  assert full_confidence_floor < d_rel < low_confidence_floor
  assert full_confidence_targets[0] > 0.0
  assert full_confidence_costs[0] > 0.0
  assert low_confidence_targets[0] == pytest.approx(0.0)
  assert low_confidence_costs[0] == pytest.approx(0.0)
  assert low_confidence_crawl_target == pytest.approx(0.0)
  assert low_confidence_crawl_cost == pytest.approx(0.0)


def test_vectorized_lead_accel_match_matches_scalar_targets():
  v_leads = np.array([0.0, 0.5, 1.5, 4.0, 8.0, 12.0])
  d_rels = np.array([STOP_DISTANCE - 0.1, STOP_DISTANCE + 0.7, STOP_DISTANCE + 2.0, 14.0, 32.0, 80.0])
  a_leads = np.array([0.0, 0.6, -0.8, -2.0, 0.9, -0.4])
  v_ego = 10.0
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)

  targets, costs = get_lead_accel_match_targets(v_leads, d_rels, a_leads, t_follow, v_ego)
  expected = [get_lead_accel_match_target(v_lead, d_rel, a_lead, t_follow, v_ego)
              for v_lead, d_rel, a_lead in zip(v_leads, d_rels, a_leads, strict=True)]

  np.testing.assert_allclose(targets, [target for target, _ in expected], rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(costs, [cost for _, cost in expected], rtol=0.0, atol=1e-12)


class FakeMpcSolver:
  def __init__(self):
    self.cost_sets = []

  def cost_set(self, *args):
    self.cost_sets.append(args)


def test_mpc_cost_weights_skip_identical_solver_writes():
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.solver = FakeMpcSolver()
  mpc._last_set_weights_key = None
  mpc._last_cost_weight_key = None
  mpc._last_accel_match_costs = None

  mpc.set_weights(True, log.LongitudinalPersonality.standard)
  first_write_count = len(mpc.solver.cost_sets)

  mpc.set_weights(True, log.LongitudinalPersonality.standard)

  assert len(mpc.solver.cost_sets) == first_write_count

  accel_costs = np.zeros(N + 1)
  mpc.set_cost_weights(mpc.cost_weights, mpc.constraint_cost_weights, accel_costs)
  dynamic_write_count = len(mpc.solver.cost_sets)

  mpc.set_cost_weights(mpc.cost_weights, mpc.constraint_cost_weights, accel_costs.copy())

  assert len(mpc.solver.cost_sets) == dynamic_write_count


@pytest.mark.parametrize("v_lead", [0.0, 1.0])
def test_lead_accel_match_waits_for_extra_gap_before_positive_match(v_lead):
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  target, cost = get_lead_accel_match_target(v_lead=v_lead, d_rel=STOP_DISTANCE + 0.6, a_lead=0.6, t_follow=t_follow, v_ego=max(0.0, v_lead - 0.2))

  assert target == pytest.approx(0.0)
  assert cost == pytest.approx(0.0)


def test_lead_accel_match_fades_far_decelerating_leads():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 25.0
  v_lead = 20.0
  a_lead = -1.0
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, a_lead)
  target_gap = get_lead_time_gap_target(v_lead, t_follow) + reserve
  far_gap = target_gap + get_lead_accel_match_margin(target_gap)

  near_target, near_cost = get_lead_accel_match_target(v_lead, target_gap, a_lead, t_follow, v_ego)
  far_target, far_cost = get_lead_accel_match_target(v_lead, far_gap, a_lead, t_follow, v_ego)

  assert near_target == pytest.approx(a_lead * LEAD_ACCEL_MATCH_DECEL_TARGET_BLEND)
  assert near_cost > 0.0
  assert far_target == pytest.approx(0.0)
  assert far_cost == pytest.approx(0.0)


def test_lead_accel_match_anticipates_steady_following_lead_brake():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  a_lead = -1.0
  target_gap = get_lead_time_gap_target(v_lead, t_follow)

  accel_target, cost = get_lead_accel_match_target(v_lead, target_gap, a_lead, t_follow, v_ego)

  assert -LEAD_ACCEL_MATCH_DECEL_CAP <= accel_target < -0.15
  assert cost > 0.0


def test_lead_accel_match_caps_soft_decel_target():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 15.0
  v_lead = 2.0
  reserve = get_moving_lead_stop_reserve(v_ego, v_lead, v_ego - v_lead, -4.0)
  target_gap = get_lead_time_gap_target(v_lead, t_follow) + reserve

  accel_target, _ = get_lead_accel_match_target(v_lead, target_gap, -4.0, t_follow, v_ego)

  assert accel_target == pytest.approx(-LEAD_ACCEL_MATCH_DECEL_CAP)


def test_moving_lead_closing_cushion_prefers_coast_before_target_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 22.0
  v_lead = 20.0
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = target_gap - 0.25 * (target_gap - get_lead_gap_comfort_floor(v_ego, v_lead, t_follow))

  target, cost = get_moving_lead_closing_cushion_target(d_rel, v_ego, v_lead, t_follow)

  assert MOVING_LEAD_CLOSING_CUSHION_ACCEL_MIN <= target <= 0.0
  assert cost > 0.0


def test_moving_lead_closing_cushion_adds_light_decel_as_cushion_is_used():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 22.0
  v_lead = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  early_gap = target_gap - 0.2 * (target_gap - comfort_floor)
  late_gap = comfort_floor + 0.2 * (target_gap - comfort_floor)

  early_target, early_cost = get_moving_lead_closing_cushion_target(early_gap, v_ego, v_lead, t_follow)
  late_target, late_cost = get_moving_lead_closing_cushion_target(late_gap, v_ego, v_lead, t_follow)

  assert -MOVING_LEAD_CLOSING_CUSHION_DECEL_MAX <= late_target < early_target <= 0.0
  assert late_cost >= early_cost > 0.0


def test_moving_lead_closing_cushion_stays_off_when_far_opening_or_urgent():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 22.0
  v_lead = 20.0
  target_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)

  far_target, far_cost = get_moving_lead_closing_cushion_target(target_gap + 8.0, v_ego, v_lead, t_follow)
  opening_target, opening_cost = get_moving_lead_closing_cushion_target(target_gap - 1.0, v_ego, v_ego + 0.5, t_follow)
  urgent_target, urgent_cost = get_moving_lead_closing_cushion_target(comfort_floor - 0.1, v_ego, v_lead, t_follow)
  stopped_target, stopped_cost = get_moving_lead_closing_cushion_target(20.0, 8.0, 0.2, t_follow)

  assert far_target == pytest.approx(0.0)
  assert far_cost == pytest.approx(0.0)
  assert opening_target == pytest.approx(0.0)
  assert opening_cost == pytest.approx(0.0)
  assert urgent_target == pytest.approx(0.0)
  assert urgent_cost == pytest.approx(0.0)
  assert stopped_target == pytest.approx(0.0)
  assert stopped_cost == pytest.approx(0.0)


def test_vectorized_moving_lead_closing_cushion_matches_scalar_targets():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 22.0
  v_leads = np.array([20.0, 20.0, 22.5, 0.3])
  d_rels = np.array([
    get_desired_follow_distance(v_ego, 20.0, t_follow) - 1.0,
    get_lead_gap_comfort_floor(v_ego, 20.0, t_follow) + 1.0,
    get_desired_follow_distance(v_ego, 22.5, t_follow) - 1.0,
    20.0,
  ])

  targets, costs = get_moving_lead_closing_cushion_target(d_rels, v_ego, v_leads, t_follow)
  expected = [get_moving_lead_closing_cushion_target(d_rel, v_ego, v_lead, t_follow)
              for d_rel, v_lead in zip(d_rels, v_leads, strict=True)]

  np.testing.assert_allclose(targets, [target for target, _ in expected], rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(costs, [cost for _, cost in expected], rtol=0.0, atol=1e-12)


def test_approach_engage_offset_stays_off_without_closure_or_runway():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_approach_engage_offset(20.0, 40.0, 20.0, t_follow) == pytest.approx(0.0)
  assert get_approach_engage_offset(20.0, 20.0, 15.0, t_follow) == pytest.approx(0.0)


def test_approach_engage_offset_grows_for_large_closing_runway_cases():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  mild_offset = get_approach_engage_offset(20.0, 70.0, 15.0, t_follow, a_lead=-0.5)
  strong_offset = get_approach_engage_offset(20.0, 80.0, 0.0, t_follow, a_lead=-1.0)

  assert 0.0 < mild_offset < strong_offset <= APPROACH_ENGAGE_OFFSET_MAX


def test_lead_gap_comfort_uses_light_brake_for_steady_under_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  comfort_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, d_rel, t_follow)

  assert -LEAD_GAP_COMFORT_LIGHT_DECEL <= comfort_a_min < 0.0


def test_lead_gap_comfort_prefers_coast_for_pullaway_lead():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.5
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  assert get_lead_gap_comfort_a_min(v_ego, v_lead, d_rel, t_follow) == pytest.approx(0.0)


def test_lead_gap_comfort_tapers_toward_coast_as_gap_recovers():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  v_lead = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_lead, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_lead, t_follow)
  early_gap = comfort_floor + 0.2
  late_gap = desired_gap - 0.2

  early_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, early_gap, t_follow)
  late_a_min = get_lead_gap_comfort_a_min(v_ego, v_lead, late_gap, t_follow)

  assert early_a_min < late_a_min < 0.0
  assert get_lead_gap_comfort_recovery_blend(late_gap, comfort_floor, desired_gap) > get_lead_gap_comfort_recovery_blend(early_gap, comfort_floor, desired_gap)


def test_lead_gap_comfort_disables_near_danger_or_real_closure():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  v_ego = 20.0
  comfort_floor = get_lead_gap_comfort_floor(v_ego, v_ego, t_follow)
  desired_gap = get_desired_follow_distance(v_ego, v_ego, t_follow)
  d_rel = 0.5 * (comfort_floor + desired_gap)

  assert get_lead_gap_comfort_a_min(v_ego, v_ego - 1.0, d_rel, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_gap_comfort_a_min(v_ego, v_ego, comfort_floor - 0.1, t_follow) == pytest.approx(ACCEL_MIN)


def test_lead_accel_recovery_allows_accel_when_lead_pulls_away():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  recovery_a_min = get_lead_accel_recovery_a_min(2.2, 4.2, 15.0, 0.9, t_follow)

  assert 0.0 < recovery_a_min <= LEAD_ACCEL_RECOVERY_ACCEL_MAX


def test_lead_accel_recovery_requires_safe_opening_gap():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  assert get_lead_accel_recovery_a_min(2.2, 4.2, 5.0, 0.9, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_accel_recovery_a_min(2.2, 2.0, 15.0, 0.9, t_follow) == pytest.approx(ACCEL_MIN)
  assert get_lead_accel_recovery_a_min(2.2, 4.2, 15.0, 0.1, t_follow) == pytest.approx(ACCEL_MIN)


def test_lead_loss_e2e_guard_arms_when_far_lead_disappears_during_lane_change():
  timer = longitudinal_planner.update_lead_loss_e2e_guard_timer(
    0.0, 0.05, True, 63.0, 0.95, False, True,
    reset_state=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
  )

  assert timer == pytest.approx(longitudinal_planner.LEAD_LOSS_E2E_GUARD_TIME)


def test_lead_loss_e2e_guard_limits_only_no_lead_non_stop_model_decel():
  guarded = longitudinal_planner.apply_lead_loss_e2e_guard_accel(-1.2, False, 1.0, False)
  stop_decel = longitudinal_planner.apply_lead_loss_e2e_guard_accel(-1.2, True, 1.0, False)
  lead_decel = longitudinal_planner.apply_lead_loss_e2e_guard_accel(-1.2, False, 1.0, True)

  assert guarded == pytest.approx(longitudinal_planner.LEAD_LOSS_E2E_GUARD_ACCEL_FLOOR)
  assert stop_decel == pytest.approx(-1.2)
  assert lead_decel == pytest.approx(-1.2)


class TestSourceHysteresis:
  def test_keeps_current_source_when_within_margin(self):
    obstacles = np.array([10.0, 11.0, 20.0])
    current_idx = 1
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 1

  def test_switches_when_new_source_better_by_margin(self):
    obstacles = np.array([10.0, 12.0, 20.0])
    current_idx = 1
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 0

  def test_switches_when_current_source_much_worse(self):
    obstacles = np.array([10.0, 25.0, 20.0])
    current_idx = 1
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 0

  def test_sticks_with_best_when_already_best(self):
    obstacles = np.array([10.0, 12.0, 20.0])
    current_idx = 0
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 0

  def test_zero_margin_behaves_like_argmin(self):
    obstacles = np.array([10.0, 11.0, 20.0])
    current_idx = 1
    result = apply_source_hysteresis(obstacles, current_idx, 0.0)
    assert result == 0

  def test_vectorized_keeps_current_per_timestep(self):
    obstacles = np.array([
      [10.0, 12.0, 20.0],
      [15.0, 10.0, 20.0],
    ])
    current = np.array([1, 1])
    result = apply_source_hysteresis(obstacles, current, SOURCE_HYSTERESIS_MARGIN)
    assert result[0] == 0
    assert result[1] == 1

  def test_vectorized_switches_only_where_margin_exceeded(self):
    obstacles = np.array([
      [10.0, 11.0, 20.0],
      [10.0, 12.0, 20.0],
    ])
    current = np.array([1, 1])
    result = apply_source_hysteresis(obstacles, current, SOURCE_HYSTERESIS_MARGIN)
    assert result[0] == 1
    assert result[1] == 0

  def test_margin_between_lead0_and_cruise(self):
    obstacles = np.array([10.0, 20.0, 11.0])
    current_idx = 2
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 2  # 11.0 - 10.0 = 1.0 < 1.2 margin, stays on cruise

  def test_stays_with_cruise_when_lead0_within_margin(self):
    obstacles = np.array([10.0, 20.0, 10.8])
    current_idx = 2
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 2

  def test_lead_transition_respects_margin(self):
    """Simulate lead flicker: lead0 appears at 10.5m while cruise is at 10.0m."""
    obstacles = np.array([10.5, 20.0, 10.0])
    current_idx = 2  # cruise
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 2  # stay on cruise, lead0 is only 0.5m better

  def test_lead_disappearance_respects_margin(self):
    """Simulate lead flicker: lead0 is current but cruise is now 0.5m better."""
    obstacles = np.array([10.0, 20.0, 10.5])
    current_idx = 0  # lead0
    result = apply_source_hysteresis(obstacles, current_idx, SOURCE_HYSTERESIS_MARGIN)
    assert result == 0  # stay on lead0, cruise is only 0.5m better


def run_following_distance_simulation(v_lead, t_end=100.0, e2e=False, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_lead),
    lead_relevancy=True,
    initial_distance_lead=100,
    speed_lead_values=[v_lead],
    breakpoints=[0.0],
    e2e=e2e,
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1, 2] - output[-1, 1]


def run_lead_closing_simulation(v_ego, v_lead, initial_distance_lead, t_end=30.0, personality=0):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=float(v_ego),
    lead_relevancy=True,
    initial_distance_lead=float(initial_distance_lead),
    speed_lead_values=[float(v_lead)],
    breakpoints=[0.0],
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output


@parameterized_class(
  ("e2e", "personality", "speed"),
  itertools.product(
    [True, False],  # e2e
    [
      log.LongitudinalPersonality.relaxed,  # personality
      log.LongitudinalPersonality.standard,
      log.LongitudinalPersonality.aggressive,
    ],
    [0, 10, 35],
  ),
)  # speed
class TestFollowingDistance:
  def test_following_distance(self):
    v_lead = float(self.speed)
    simulation_steady_state = run_following_distance_simulation(v_lead, e2e=self.e2e, personality=self.personality)
    correct_steady_state = get_desired_follow_distance(v_lead, v_lead, get_T_FOLLOW(self.personality))
    err_ratio = 0.2 if self.e2e else 0.1
    abs_err_margin = 0.5 if v_lead > 0.0 else 2.0
    assert simulation_steady_state == pytest.approx(correct_steady_state, abs=err_ratio * correct_steady_state + abs_err_margin)


def test_closing_lead_bleeds_off_speed_late_in_approach():
  t_follow = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  output = run_lead_closing_simulation(v_ego=25.0, v_lead=20.0, initial_distance_lead=90.0)

  time = output[:, 0]
  closing_speed = output[:, 3] - output[:, 4]
  late_approach = time >= (time[-1] - 5.0)

  assert np.any(late_approach)
  assert np.max(closing_speed[late_approach]) < 1.5
  assert output[-1, 6] == pytest.approx(get_desired_follow_distance(20.0, 20.0, t_follow), abs=4.0)


def test_stopped_car_approach_settles_near_stop_gap():
  output = run_lead_closing_simulation(v_ego=20.0, v_lead=0.0, initial_distance_lead=90.0, t_end=20.0)

  presentation_distance = long_mpc.get_lead_stop_presentation_distance(v_ego=0.0, v_lead=0.0, a_lead=0.0, model_prob=1.0)
  assert output[-1, 6] == pytest.approx(presentation_distance, abs=0.5)
