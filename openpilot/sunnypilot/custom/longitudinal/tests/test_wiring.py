"""Tests for the plannerd wiring adapter (opt-in custom longitudinal)."""
from __future__ import annotations

from types import SimpleNamespace
import math
import time

import pytest
from openpilot.cereal import log

from openpilot.sunnypilot.custom.longitudinal.decision import decide
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import LongitudinalScene, build_candidates
from openpilot.sunnypilot.custom.longitudinal.policy_tables import (
  Personality,
  launch_accel_max,
  stop_approach_comfort_decel,
)
from openpilot.sunnypilot.custom.longitudinal.wiring import (
  DEFAULT_ACCEL_LIMITS,
  MODEL_STOP_EARLY_MARGIN_M,
  MODEL_STALE_AGE_S,
  CustomLongitudinalAdapter,
  build_stack_inputs,
  _model_stop_distance,
  _message_age_s,
)


def lead(d_rel=30.0, v_lead=12.0, v_rel=None, status=True, radar=True, model_prob=0.9):
  ld = SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=0.0,
                       yRel=0.0, radarTrackId=3, radar=radar, modelProb=model_prob, aLeadTau=1.0)
  if v_rel is not None:
    ld.vRel = v_rel
  return ld


def fake_sm(lead_one=None, brake=False, gas=False, standstill=False, steering_angle_deg=0.0,
            steering_torque=0.0, model_should_stop=False, model_accel=0.0, pitch=0.0,
            model_x=None, model_y=None, model_v=None, model_leads=None,
            long_active=False):
  position = SimpleNamespace()
  if model_x is not None:
    position.x = model_x
  if model_y is not None:
    position.y = model_y
  return {
    'radarState': SimpleNamespace(leadOne=lead_one, leadTwo=None),
    'carState': SimpleNamespace(brakePressed=brake, gasPressed=gas, standstill=standstill,
                                steeringAngleDeg=steering_angle_deg, steeringTorque=steering_torque),
    'modelV2': SimpleNamespace(action=SimpleNamespace(shouldStop=model_should_stop,
                                                       desiredAcceleration=model_accel),
                               position=position,
                               velocity=SimpleNamespace(x=model_v),
                               leadsV3=list(model_leads or [])),
    'carControl': SimpleNamespace(orientationNED=[0.0, pitch, 0.0], longActive=long_active),
    'controlsState': SimpleNamespace(forceDecel=False),
  }


# A trajectory that decelerates to rest ~38 m ahead (velocity dips to ~0 at index 5).
STOP_TRAJ_V = [15.0, 12.0, 9.0, 6.0, 3.0, 0.2, 0.0, 0.0]
STOP_TRAJ_X = [0.0, 13.0, 24.0, 32.0, 37.0, 38.0, 38.0, 38.0]
CRUISE_TRAJ_V = [20.0] * 8
CRUISE_TRAJ_X = [20.0 * i for i in range(8)]


def fake_scc(vision_active=False, vision_a=0.0, map_active=False, map_a=0.0,
             vision_current_lat_acc=0.0, vision_max_pred_lat_acc=0.0, vision_pre_entry_active=False):
  return SimpleNamespace(
    vision=SimpleNamespace(is_active=vision_active, output_a_target=vision_a, state=0,
                           current_lat_acc=vision_current_lat_acc,
                           max_pred_lat_acc=vision_max_pred_lat_acc,
                           pre_entry_active=vision_pre_entry_active),
    map=SimpleNamespace(is_active=map_active, output_a_target=map_a, state=0,
                        target_lat=0.0, target_lon=0.0),
  )


def fake_sla(active=False, v_target=0.0, a_target=0.0):
  return SimpleNamespace(is_active=active, output_v_target=v_target, output_a_target=a_target)


class FakeParams:
  def __init__(self, **vals):
    self._v = vals
  def get_bool(self, k):
    return bool(self._v.get(k, False))
  def get(self, k):
    return self._v.get(k)
  def all_keys(self):
    return [k.encode() for k in self._v]


def test_message_age_missing_zero_nonfinite_are_stale():
  assert math.isinf(_message_age_s(SimpleNamespace(), 'modelV2'))
  assert math.isinf(_message_age_s(SimpleNamespace(recv_time={}), 'modelV2'))
  assert math.isinf(_message_age_s(SimpleNamespace(recv_time={'modelV2': 0.0}), 'modelV2'))
  assert math.isinf(_message_age_s(SimpleNamespace(recv_time={'modelV2': float('nan')}), 'modelV2'))
  assert _message_age_s(SimpleNamespace(recv_time={'modelV2': time.monotonic()}), 'modelV2') < 0.05


def test_confidence_snapshot_marks_missing_model_age_stale_and_unhealthy():
  adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))

  out = adapter.evaluate(fake_sm(), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla())

  assert out.debug["confidence_model_age_s"] == pytest.approx(-1.0)
  assert out.debug["confidence_model_stale"] is True
  assert out.debug["confidence_model_service_healthy"] is False
  assert out.debug["confidence_radar_service_healthy"] is False


class BadModelPathPosition:
  x = [0.0, 30.0, 60.0]

  @property
  def y(self):
    raise RuntimeError("bad model path")


def test_build_stack_inputs_maps_evidence():
  inp = build_stack_inputs(
    v_ego=20.0, a_ego=0.1, v_cruise=22.0, seed_a_target=0.4, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=lead(), lead_two=None,
    scc_vision_active=True, scc_vision_a_target=-0.7, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=True, sla_v_target=18.0, sla_a_target=-0.5,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(True, False),
  )
  assert inp.a_ego == pytest.approx(0.1)           # ego accel wired through
  assert inp.curve_active is True and inp.curve_a_target == pytest.approx(-0.7)
  assert inp.curve_source is EvidenceClass.CURVE_VISION   # vision-bound curve
  assert inp.speed_limit_active is True and inp.speed_limit_a_target == pytest.approx(-0.5)
  assert inp.lead_a_target == pytest.approx(0.4)   # MPC baseline carried as lead-follow accel
  assert inp.model_should_stop is False            # conservatively defaulted (harness-gated)


def test_build_stack_inputs_carries_model_stale_flag():
  inp = build_stack_inputs(
    v_ego=15.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.0, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.E2E, personality=Personality.STANDARD, sources=SourceToggles(),
    model_should_stop=True, model_desired_accel=-2.0, model_stale=True,
  )
  assert inp.model_stale is True


def test_build_stack_inputs_carries_standstill():
  inp = build_stack_inputs(
    v_ego=0.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.0, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.E2E, personality=Personality.STANDARD, sources=SourceToggles(),
    standstill=True,
  )
  assert inp.standstill is True


def test_build_stack_inputs_carries_dynamic_floor_inputs():
  inp = build_stack_inputs(
    v_ego=15.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.0, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=True, scc_vision_a_target=0.0, scc_vision_current_lat_acc=1.2,
    scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(),
    current_lat_accel=1.2, pitch=-0.05,
  )
  assert inp.current_lat_accel == pytest.approx(1.2)
  assert inp.pitch == pytest.approx(-0.05)


def test_adapter_evaluate_includes_dynamic_safety_floor_debug_without_changing_output():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode='scc'))
  out = a.evaluate(fake_sm(lead(d_rel=8.0), pitch=-0.03, long_active=True),
                   15.0, 0.0, 22.0, 0.4, fake_scc(vision_active=True, vision_current_lat_acc=1.0), fake_sla())
  debug = dict(out.debug or {})
  assert debug.get("dynamic_safety_floor_active") is True
  assert "dynamic_safety_floor_current_safe_distance" in debug
  assert "dynamic_safety_floor_proposed_safe_distance" in debug
  assert "dynamic_safety_floor_delta_safe_distance" in debug
  assert "dynamic_safety_floor_dynamic_floor_value" in debug
  assert "dynamic_safety_floor_kinematic_floor_violation" in debug
  assert "dynamic_safety_floor_comfort_brake_effective" in debug
  assert out.a_target == pytest.approx(0.4)
  assert out.should_stop is False


def test_dynamic_floor_missing_lat_accel_does_not_fault_adapter():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode='scc'))
  scc = SimpleNamespace(
    vision=SimpleNamespace(is_active=True, output_a_target=0.0, state=0,
                           max_pred_lat_acc=0.0, pre_entry_active=False),
    map=SimpleNamespace(is_active=False, output_a_target=0.0, state=0,
                        target_lat=0.0, target_lon=0.0),
  )
  out = a.evaluate(fake_sm(lead(d_rel=8.0), pitch=-0.03, long_active=True),
                   15.0, 0.0, 22.0, 0.4, scc, fake_sla())
  assert out.selected_intent != "fault"
  assert out.a_target == pytest.approx(0.4)
  assert out.debug["dynamic_safety_floor_active"] is True


def test_map_only_populates_curve_confidence_but_not_actuator_cap():
  # SCC-M is evidence-only at the wiring layer: it feeds curve-speed confidence (telemetry) but
  # must not create an actuator curve advisory cap unless SCC-Vision is also active.
  inp = build_stack_inputs(
    v_ego=20.0, a_ego=0.0, v_cruise=22.0, seed_a_target=0.4, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=True, scc_map_a_target=-0.8,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(False, True),
  )
  assert inp.curve_active is False
  assert inp.curve_a_target == pytest.approx(0.0)
  assert inp.curve_source is not EvidenceClass.CURVE_MAP
  assert inp.curve_confidence.map_active is True
  assert inp.curve_confidence.map_a_target == pytest.approx(0.0)
  scene = LongitudinalScene(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4,
                            curve_active=False, curve_a_target=0.0, curve_source=EvidenceClass.CURVE_VISION)
  cands = build_candidates(scene)
  on = decide(cands, LongitudinalMode.SCC, DEFAULT_ACCEL_LIMITS, SourceToggles(False, True))
  assert on.a_target == pytest.approx(0.4)          # map-only evidence cannot cap cruise


def test_adapter_disabled_passthrough():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=False))
  out = a.apply(fake_sm(lead()), 20.0, 0.0, 22.0, 0.42, fake_scc(), fake_sla())
  assert out == 0.42


def test_adapter_enabled_shapes_and_bounds():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc",
                                           LongitudinalPersonality="1"))
  # curve cap active -> shaped target should brake relative to the cruise seed
  out = a.apply(fake_sm(lead(d_rel=40.0)), 20.0, 0.0, 22.0, 0.5,
                fake_scc(vision_active=True, vision_a=-0.6), fake_sla())
  assert DEFAULT_ACCEL_LIMITS[0] <= out <= DEFAULT_ACCEL_LIMITS[1]


def test_adapter_fail_closed_on_bad_sm():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True))
  # sm missing radarState -> apply must return the seed unchanged, never raise
  out = a.apply({}, 20.0, 0.0, 22.0, 0.33, fake_scc(), fake_sla())
  assert out == 0.33


def test_adapter_nonfinite_seed_falls_back_to_finite_neutral_target():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True))
  out = a.apply(fake_sm(lead()), 20.0, 0.0, 22.0, float("nan"), fake_scc(), fake_sla())
  assert out == 0.0


@pytest.mark.parametrize(("value", "expected"), [("off", "off"), ("apply", "apply"), ("bad", "off")])
def test_cut_out_lead_release_mode_sanitizes_fail_closed(value, expected):
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CutOutLeadReleaseMode=value))
  assert a.cut_out_lead_release_mode == expected


def test_adapter_acc_ignores_curve():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  out = a.apply(fake_sm(), 20.0, 0.0, 22.0, 0.4, fake_scc(vision_active=True, vision_a=-1.0), fake_sla())
  assert out == pytest.approx(0.4)  # ACC excludes curve evidence -> cruise stands


def test_model_stop_from_upstream_signal_brakes_in_e2e():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  # upstream model stop -> E2E brakes; ACC ignores it
  out_e2e = a.apply(fake_sm(model_should_stop=True, model_accel=-2.0), 15.0, 0.0, 15.0, 0.0,
                    fake_scc(), fake_sla())
  acc = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  out_acc = acc.apply(fake_sm(model_should_stop=True, model_accel=-2.0), 15.0, 0.0, 15.0, 0.0,
                      fake_scc(), fake_sla())
  assert out_e2e < 0.0
  assert out_acc == pytest.approx(0.0)


def test_build_stack_inputs_carries_force_slow_decel():
  inp = build_stack_inputs(
    v_ego=10.0, a_ego=0.0, v_cruise=12.0, seed_a_target=0.2, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(),
    force_slow_decel=True,
  )
  assert inp.force_slow_decel is True


def test_adapter_exposes_lead_pullaway_release_fields_after_confidence_stabilizes():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  sm = fake_sm(lead(status=True, d_rel=6.5, v_lead=1.5))
  sm['controlsState'] = SimpleNamespace(forceDecel=False)
  out = None
  for _ in range(12):
    out = a.evaluate(sm, 0.0, 0.0, 12.0, 0.2, fake_scc(), fake_sla(), dt=0.05)
  assert out is not None
  assert out.standstill_release_allowed is True
  assert out.standstill_release_source == "lead_pullaway"
  assert out.standstill_release_a_target >= 0.15


def test_adapter_vetoes_release_for_driver_and_force_slow_blockers():
  for blocker in ("brake", "gas", "force"):
    a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
    sm = fake_sm(lead(status=True, d_rel=6.5, v_lead=1.5), brake=(blocker == "brake"), gas=(blocker == "gas"))
    sm['controlsState'] = SimpleNamespace(forceDecel=(blocker == "force"))
    out = None
    for _ in range(12):
      out = a.evaluate(sm, 0.0, 0.0, 12.0, 0.2, fake_scc(), fake_sla(), dt=0.05)
    assert out is not None
    assert out.standstill_release_allowed is False


def test_driver_gas_disagreement_lowers_stop_trust():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  before = a._stop_trust.confidence
  last = None
  for _ in range(40):
    last = a.evaluate(fake_sm(model_should_stop=True, model_accel=-2.5, gas=True),
                      15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert a._stop_trust.confidence < before  # repeated driver countermanding softens trust
  assert last is not None
  assert last.debug["confidence_stop_trust"] == pytest.approx(a._stop_trust.confidence)


def test_uncommitted_stop_depth_needs_recent_radar_corroboration():
  # Earned decel depth past the -1.5 uncommitted stop floor is radar-corroboration-gated in
  # wiring (CorroborationHold): sustained vision-only demand (the hallucination shape) stays
  # capped, while one closing echo unlocks depth and holds it across radar flicker (route 28c
  # queues lose radar lock on most frames).
  class TimestampedSm(dict):
    pass

  def run(echo_frames):
    a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
    out = 0.0
    for i in range(70):  # 3.5 s at dt=0.05: the CautionRamp earns ~-2.0 of the -2.2 demand
      lead_one = lead(d_rel=35.0, v_lead=2.0, v_rel=-10.6) if i in echo_frames else None
      sm = TimestampedSm(fake_sm(lead_one, model_accel=-2.2, model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V))
      sm.recv_time = {'modelV2': time.monotonic()}  # fresh model: exercise the non-stale path
      out = a.apply(sm, 12.6, 0.0, 17.8, 0.0, fake_scc(), fake_sla(), dt=0.05)
    return out

  vision_only = run(echo_frames=frozenset())
  corroborated = run(echo_frames=frozenset(range(20, 24)))  # brief echo ~1 s in, radar drops after
  assert vision_only == pytest.approx(-1.5, abs=0.1)   # capped at the uncommitted stop floor
  assert corroborated < -1.8                           # depth persists ~2.3 s after the last echo


def test_travel_consistent_anchor_unlocks_stop_depth_without_radar():
  # Route 2ba t=1517/1623: leadless red lights pinned at the -1.5 vision-only floor while
  # required decel passed -2.9 and the driver had to brake. A stop point whose raw distance
  # shrinks with ego travel is world-fixed (a hallucination's does not — that run stays
  # capped, see the static-trajectory vision_only case above), so it unlocks the same
  # CautionRamp-earned, -2.5-bounded depth a radar echo does.
  class TimestampedSm(dict):
    pass

  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  v_ego, dt = 12.6, 0.05
  out = 0.0
  for i in range(70):  # 3.5 s: raw rest point shrinks at 0.7x travel (consistent, stays ahead)
    rest = 38.0 - 0.7 * v_ego * dt * i
    scale = rest / 38.0
    sm = TimestampedSm(fake_sm(None, model_accel=-2.2,
                               model_x=[p * scale for p in STOP_TRAJ_X], model_v=STOP_TRAJ_V))
    sm.recv_time = {'modelV2': time.monotonic()}
    out = a.apply(sm, v_ego, 0.0, 17.8, 0.0, fake_scc(), fake_sla(), dt=dt)
  assert out < -1.8  # earned depth unlocked with no radar echo at any frame
  trace = a.evaluate(sm, v_ego, 0.0, 17.8, 0.0, fake_scc(), fake_sla(), dt=dt)
  assert trace.model_stop_corroborated is True
  assert trace.debug["confidence_anchor_travel_corroborated"] is True


def test_corroborated_anchor_keeps_kinematic_authority_through_lateral_cutout():
  # Route 2ca t=1219-1222: a world-fixed stop anchor was already travel-corroborated
  # when a far off-path radar echo disappeared. Cut-out recovery must not relax that
  # independent stop evidence, and the late approach must follow bounded kinematics
  # instead of the -2.5 caution ceiling.
  class TimestampedSm(dict):
    pass

  a = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True,
    CustomLongitudinalMode="scc",
  ))
  v_ego, dt = 15.0, 0.05
  out = None
  for i in range(80):
    rest = 105.0 - v_ego * dt * i
    echo = lead(d_rel=75.0, v_lead=0.0, v_rel=-15.0) if 62 <= i < 69 else None
    if echo is not None:
      echo.yRel = -5.7
    sm = TimestampedSm(fake_sm(
      echo, model_accel=-3.0,
      model_x=[0.0, rest], model_y=[0.0, 0.0], model_v=[v_ego, 0.0],
      long_active=True,
    ))
    sm.recv_time = {'modelV2': time.monotonic()}
    out = a.evaluate(sm, v_ego, 0.0, 20.0, 0.0, fake_scc(), fake_sla(), dt=dt)

  assert out is not None
  assert a._stop_anchor.corroborated is True
  assert out.model_stop_corroborated is True
  assert a._cut_out_caution._recovery_s > 0.0  # the competing cut-out signal really fired
  assert out.debug["confidence_cut_out_recovery_remaining_s"] > 0.0
  assert out.a_target < -3.0


def test_scc_curve_gated_by_smart_cruise_control_vision_toggle():
  scc = fake_scc(vision_active=True, vision_a=-0.7)
  on = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc",
                                            SmartCruiseControlVision=True))
  off = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc",
                                             SmartCruiseControlVision=False))
  out_on = on.apply(fake_sm(), 20.0, 0.0, 22.0, 0.5, scc, fake_sla())
  out_off = off.apply(fake_sm(), 20.0, 0.0, 22.0, 0.5, scc, fake_sla())
  assert out_off == pytest.approx(0.5)  # SCC curve source disabled -> cruise stands
  assert out_on < out_off               # SmartCruiseControlVision on -> curve cap admitted


def test_model_stop_distance_from_trajectory():
  model = SimpleNamespace(position=SimpleNamespace(x=STOP_TRAJ_X),
                          velocity=SimpleNamespace(x=STOP_TRAJ_V))
  # distance at the first horizon index where predicted speed drops to ~0 (index 5 -> 38 m)
  assert _model_stop_distance(model) == pytest.approx(38.0)


def test_model_stop_distance_none_when_cruising():
  model = SimpleNamespace(position=SimpleNamespace(x=CRUISE_TRAJ_X),
                          velocity=SimpleNamespace(x=CRUISE_TRAJ_V))
  assert _model_stop_distance(model) is None  # speed never drops -> no predicted stop


def test_model_stop_distance_none_on_bad_trajectory():
  assert _model_stop_distance(SimpleNamespace(position=None, velocity=None)) is None
  # mismatched lengths -> None, never raise
  bad = SimpleNamespace(position=SimpleNamespace(x=[0.0, 1.0]), velocity=SimpleNamespace(x=[1.0]))
  assert _model_stop_distance(bad) is None


def test_build_stack_inputs_carries_model_stop_distance():
  inp = build_stack_inputs(
    v_ego=15.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.0, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.E2E, personality=Personality.STANDARD, sources=SourceToggles(),
    model_stop_distance=38.0,
  )
  # The intake subtracts the early-stop margin so every consumer rests short of the
  # model's declared stop point.
  assert inp.model_stop_distance == pytest.approx(38.0 - MODEL_STOP_EARLY_MARGIN_M)


class TimestampedRadarSm(dict):
  def __init__(self, lead_one, *, radar_ok=True, model_age=0.0):
    super().__init__(fake_sm(lead_one, model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V, long_active=True))
    now = time.monotonic()
    self.recv_time = {'modelV2': now - model_age, 'radarState': now}
    self.valid = {'radarState': radar_ok}
    self.alive = {'radarState': radar_ok}
    self.freq_ok = {'radarState': radar_ok}


def _anchor_output(lead_one=None, *, frames=8, radar_ok=True, model_age=0.0, model_x=STOP_TRAJ_X,
                   model_v=STOP_TRAJ_V, v_ego=15.0, model_should_stop=True,
                   model_accel=-2.0, clear_frames=0):
  adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  out = None
  for frame in range(frames + clear_frames):
    clearing = frame >= frames
    sm = TimestampedRadarSm(lead_one, radar_ok=radar_ok, model_age=model_age)
    sm['modelV2'].position.x = model_x
    sm['modelV2'].velocity.x = model_v
    sm['modelV2'].action.shouldStop = model_should_stop and not clearing
    sm['modelV2'].action.desiredAcceleration = 0.0 if clearing else model_accel
    out = adapter.evaluate(sm, v_ego, 0.0, v_ego, 0.0, fake_scc(), fake_sla(), dt=0.05)
  return adapter, out


def test_confidence_snapshot_distinguishes_radar_hold_refresh_from_vision():
  radar_adapter, radar_out = _anchor_output(
    lead(d_rel=30.0, v_lead=2.0, v_rel=-10.0), frames=1,
  )
  vision_adapter, vision_out = _anchor_output(
    lead(d_rel=30.0, v_lead=2.0, v_rel=-10.0, radar=False), frames=1,
  )

  assert radar_out.debug["confidence_corroboration_hold_remaining_s"] == pytest.approx(
    radar_adapter._corroboration_hold.hold_s,
  )
  assert radar_out.debug["confidence_corroboration_hold_remaining_s"] > 0.0
  assert radar_out.debug["confidence_corroboration_refresh_source"] == "radar"
  assert vision_out.debug["confidence_corroboration_hold_remaining_s"] == pytest.approx(
    vision_adapter._corroboration_hold.hold_s,
  )
  assert vision_out.debug["confidence_corroboration_hold_remaining_s"] > 0.0
  assert vision_out.debug["confidence_corroboration_refresh_source"] == "vision"

  held_sm = TimestampedRadarSm(None)
  held_sm['modelV2'].action.shouldStop = True
  held_sm['modelV2'].action.desiredAcceleration = -2.0
  held_out = radar_adapter.evaluate(
    held_sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05,
  )
  assert 0.0 < held_out.debug["confidence_corroboration_hold_remaining_s"] < radar_out.debug[
    "confidence_corroboration_hold_remaining_s"
  ]
  assert held_out.debug["confidence_corroboration_refresh_source"] == "none"


def test_stationary_lead_correlated_anchor_uses_fresh_raw_stop_distance():
  adapter, out = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0))
  assert adapter._stop_anchor_lead_corr_frames >= 3
  assert out.debug["model_stop_distance_raw"] == pytest.approx(38.0)
  assert out.debug["model_stop_distance_used"] > 33.8
  assert out.debug["model_stop_distance_used"] <= 38.0
  assert out.debug["confidence_stationary_radar_correlation_applied"] is True
  assert out.debug["confidence_effective_caution_floor"] == pytest.approx(adapter._caution_ramp.floor)


def test_stationary_lead_correlation_needs_persistent_frames():
  for frames in (1, 2):
    adapter, out = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), frames=frames)
    assert adapter._stop_anchor_lead_corr_frames == frames
    assert out.debug["model_stop_distance_used"] <= 33.8
    assert out.debug["confidence_stationary_radar_correlation_applied"] is False


@pytest.mark.parametrize("lead_one,radar_ok,model_x,model_v", [
  (None, True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=43.0, v_lead=1.0, v_rel=1.0), True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=43.0, v_lead=0.0, v_rel=0.0, radar=False), True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=43.0, v_lead=0.0, v_rel=0.0, model_prob=0.4), True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), False, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=58.0, v_lead=0.0, v_rel=0.0), True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=float("nan"), v_lead=0.0, v_rel=0.0), True, STOP_TRAJ_X, STOP_TRAJ_V),
  (lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), True, CRUISE_TRAJ_X, CRUISE_TRAJ_V),
])
def test_noncorrelated_stops_preserve_existing_anchor_output(lead_one, radar_ok, model_x, model_v):
  adapter, out = _anchor_output(lead_one, radar_ok=radar_ok, model_x=model_x, model_v=model_v)
  assert adapter._stop_anchor_lead_corr_frames == 0
  if model_x is STOP_TRAJ_X:
    assert out.debug["model_stop_distance_used"] <= 33.8
  else:
    assert out.debug["model_stop_distance_used"] == 0.0


def test_stationary_lead_correlation_never_uses_stale_or_missing_raw_geometry():
  adapter, _ = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), frames=8)
  assert adapter._stop_anchor_lead_corr_frames >= 3
  cruise_sm = TimestampedRadarSm(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0))
  cruise_sm['modelV2'].position.x = CRUISE_TRAJ_X
  cruise_sm['modelV2'].velocity.x = CRUISE_TRAJ_V
  cruise_sm['modelV2'].action.shouldStop = True
  cruise_sm['modelV2'].action.desiredAcceleration = -2.0
  adapter.evaluate(cruise_sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor_lead_corr_frames == 0


def test_stale_model_never_admits_stationary_lead_correlation():
  adapter, out = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0),
                                model_age=MODEL_STALE_AGE_S + 0.1)
  assert adapter._stop_anchor_lead_corr_frames == 0
  assert out.debug["model_stop_distance_used"] <= 33.8


def test_degraded_frame_breaks_correlation_persistence():
  adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  sm = TimestampedRadarSm(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0))
  sm['modelV2'].action.shouldStop = True
  sm['modelV2'].action.desiredAcceleration = -2.0
  for _ in range(2):
    adapter.evaluate(sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor_lead_corr_frames == 2
  degraded = adapter.evaluate(sm, float("nan"), 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert degraded.selected_intent == "degraded_evidence"
  assert adapter._stop_anchor_lead_corr_frames == 0
  adapter.evaluate(sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor_lead_corr_frames == 1


def test_bound_clamps_to_lead_safe_target_when_raw_stop_implies_smaller_buffer():
  adapter, out = _anchor_output(lead(d_rel=41.0, v_lead=0.0, v_rel=0.0))
  assert adapter._stop_anchor_lead_corr_frames >= 3
  # floor = min(raw 38, dRel 41 - STOP_DISTANCE 5) = 36; the anchor alone sits nearer.
  assert out.debug["model_stop_distance_used"] == pytest.approx(36.0)


def test_bound_feeds_policy_distance_after_exactly_one_early_margin(monkeypatch):
  captured = {}
  import openpilot.sunnypilot.custom.longitudinal.wiring as wiring_mod
  real_build = wiring_mod.build_stack_inputs

  def spy(**kwargs):
    assert kwargs["model_stop_distance"] == pytest.approx(36.0)  # pre-margin bound
    inp = real_build(**kwargs)
    captured["model_stop_distance"] = inp.model_stop_distance
    return inp

  monkeypatch.setattr(wiring_mod, "build_stack_inputs", spy)
  _anchor_output(lead(d_rel=41.0, v_lead=0.0, v_rel=0.0))
  assert captured["model_stop_distance"] == pytest.approx(36.0 - MODEL_STOP_EARLY_MARGIN_M)


def test_correlated_relaxation_respects_blip_filter_ordering():
  # Anchor commits frame 1; correlation admits at frame 3, but consumers see nothing
  # until STOP_ANCHOR_MIN_COMMIT_S (0.25 s) has elapsed at frame 6.
  _, early = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), frames=5)
  assert early.debug["model_stop_distance_used"] == 0.0
  assert early.debug["confidence_stationary_radar_correlation_applied"] is False
  adapter, later = _anchor_output(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0), frames=6)
  assert adapter._stop_anchor_lead_corr_frames >= 3
  assert later.debug["model_stop_distance_used"] > 33.8
  assert later.debug["confidence_stationary_radar_correlation_applied"] is True


def test_stationary_lead_correlation_does_not_override_semantic_clear():
  adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  lead_one = lead(d_rel=43.0, v_lead=0.0, v_rel=0.0)
  for _ in range(8):
    sm = TimestampedRadarSm(lead_one)
    sm['modelV2'].action.shouldStop = True
    sm['modelV2'].action.desiredAcceleration = -2.0
    adapter.evaluate(sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor_lead_corr_frames >= 3

  first_clear = adapter.evaluate(TimestampedRadarSm(lead_one), 15.0, 0.0, 15.0, 0.0,
                                 fake_scc(), fake_sla(), dt=0.05)
  # The first clear frame immediately drops the relaxation (counter resets) even though
  # the anchor commitment itself is still retained by the clear-persistence gate.
  assert adapter._stop_anchor_lead_corr_frames == 0
  assert 0.0 < first_clear.debug["model_stop_distance_used"] <= 33.8

  cleared = None
  for _ in range(2):
    cleared = adapter.evaluate(TimestampedRadarSm(lead_one), 15.0, 0.0, 15.0, 0.0,
                               fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor.remaining is None
  assert cleared.debug["model_stop_distance_used"] == 0.0


def test_alternating_track_ids_do_not_break_continuous_stationary_geometry():
  adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  out = None
  adapter.evaluate(TimestampedRadarSm(lead(d_rel=43.0, v_lead=0.0, v_rel=0.0)), 15.0, 0.0, 15.0, 0.0,
                   fake_scc(), fake_sla(), dt=0.05)
  for idx in range(8):
    ld = lead(d_rel=43.0, v_lead=0.0, v_rel=0.0)
    ld.radarTrackId = 533 if idx % 2 == 0 else 556
    sm = TimestampedRadarSm(ld)
    sm['modelV2'].action.shouldStop = True
    sm['modelV2'].action.desiredAcceleration = -2.0
    out = adapter.evaluate(sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor_lead_corr_frames >= 3
  assert out.debug["model_stop_distance_used"] > 33.8


def test_distance_aware_stop_approach_brakes_in_e2e():
  # No upstream shouldStop, but the model trajectory predicts a near stop -> the distance-aware
  # stop-approach path engages and brakes in E2E (and is excluded in ACC).
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  acc = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  # a few frames: the ModelStopAnchor's blip-filter debounce must age past before the
  # distance path is admitted to consumers
  for _ in range(8):
    out_e2e = a.apply(fake_sm(model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V), 15.0, 0.0, 15.0, 0.0,
                      fake_scc(), fake_sla(), dt=0.05)
    out_acc = acc.apply(fake_sm(model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V), 15.0, 0.0, 15.0, 0.0,
                        fake_scc(), fake_sla(), dt=0.05)
  assert out_e2e < 0.0
  assert out_acc == pytest.approx(0.0)


def test_fresh_semantic_model_clear_releases_anchor_before_missing_timeout():
  class TimestampedSm(dict):
    pass

  adapter = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))

  def frame(model_should_stop, model_accel):
    sm = TimestampedSm(fake_sm(
      model_should_stop=model_should_stop, model_accel=model_accel,
      model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V, long_active=True,
    ))
    sm.recv_time = {'modelV2': time.monotonic()}
    return sm

  for _ in range(8):
    adapter.evaluate(frame(True, -2.0), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor.remaining is not None

  first_clear = adapter.evaluate(frame(False, 0.0), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor.remaining is not None
  assert first_clear.debug["model_stop_distance_used"] > 0.0

  for _ in range(2):
    cleared = adapter.evaluate(frame(False, 0.0), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor.remaining is None
  assert cleared.debug["model_stop_distance_used"] == 0.0

  for _ in range(5):
    still_clear = adapter.evaluate(frame(False, 0.0), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
    assert still_clear.debug["model_stop_distance_used"] == 0.0
  assert adapter._stop_anchor.remaining is None


def test_stale_or_absent_model_evidence_cannot_semantically_clear_anchor():
  class TimestampedSm(dict):
    pass

  adapter = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))

  def frame(model_should_stop=True, model_accel=-2.0, *, fresh: bool | None = True, action=True):
    sm = TimestampedSm(fake_sm(
      model_should_stop=model_should_stop, model_accel=model_accel,
      model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V, long_active=True,
    ))
    if action is False:
      sm['modelV2'].action = None
    if fresh is True:
      sm.recv_time = {'modelV2': time.monotonic()}
    elif fresh is False:
      sm.recv_time = {'modelV2': time.monotonic() - MODEL_STALE_AGE_S - 0.1}
    return sm

  for _ in range(8):
    adapter.evaluate(frame(), 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
  assert adapter._stop_anchor.remaining is not None

  for sm in (
    frame(False, 0.0, fresh=False),
    frame(False, 0.0, fresh=None),
    frame(False, 0.0, action=False),
    frame(False, -0.1),
  ):
    adapter.evaluate(sm, 15.0, 0.0, 15.0, 0.0, fake_scc(), fake_sla(), dt=0.05)
    assert adapter._stop_anchor.remaining is not None


def test_default_mode_is_scc_the_dec_replacement():
  a = CustomLongitudinalAdapter(params=None)
  assert a.mode is LongitudinalMode.SCC                     # default: the intelligent blend


def test_explicit_mode_from_param():
  for setting, expected in (("acc", LongitudinalMode.ACC), ("e2e", LongitudinalMode.E2E),
                            ("scc", LongitudinalMode.SCC)):
    a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode=setting))
    assert a.mode is expected


def test_enable_refreshes_live_but_mode_is_an_engagement_cycle_latch():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
                      LongitudinalPersonality="1", SmartCruiseControlVision=True, SmartCruiseControlMap=False)
  a = CustomLongitudinalAdapter(params)
  params._v.update(CustomLongitudinalEnabled=False, CustomLongitudinalMode="e2e",
                   LongitudinalPersonality="2", SmartCruiseControlVision=False, SmartCruiseControlMap=True)
  a.maybe_refresh_params()
  assert a.enabled is False
  # An onroad Param write never alters the adapter's active mode; only the value
  # selfdrived publishes (captured at the next engagement) does.
  assert a.mode is LongitudinalMode.ACC
  a.set_active_mode("e2e")
  assert a.mode is LongitudinalMode.E2E


def test_personality_and_toggles_refresh_every_params_period():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
                      LongitudinalPersonality="1", SmartCruiseControlVision=True, SmartCruiseControlMap=False)
  a = CustomLongitudinalAdapter(params)
  params._v.update(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
                   LongitudinalPersonality="2", SmartCruiseControlVision=False, SmartCruiseControlMap=True)
  # Tuning params do not change before the refresh period elapses.
  for _ in range(49):
    a.maybe_refresh_params()
  assert a.personality is Personality.STANDARD
  assert a.sources.scc_curve_vision_enabled is True
  assert a.sources.scc_curve_map_enabled is False
  # They take effect on the tick that hits the refresh period.
  a.maybe_refresh_params()
  assert a.personality is Personality.RELAXED
  assert a.sources.scc_curve_vision_enabled is False
  assert a.sources.scc_curve_map_enabled is True


PERSONALITY_POLICY_VALUES = {
  "aggressive": (1.55, -0.45),
  "standard": (1.35, -0.38),
  "relaxed": (1.10, -0.30),
}


@pytest.mark.parametrize("name, ordinal", log.LongitudinalPersonality.schema.enumerants.items())
def test_personality_param_ordinals_select_policy_tables(name, ordinal):
  expected = Personality[name.upper()]
  expected_launch, expected_stop = PERSONALITY_POLICY_VALUES[name]
  adapter = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True,
    LongitudinalPersonality=ordinal,
  ))

  assert adapter.personality is expected
  assert launch_accel_max(adapter.personality) == pytest.approx(expected_launch)
  assert stop_approach_comfort_decel(adapter.personality) == pytest.approx(expected_stop)


def test_stack_reset_on_mode_change_and_re_enable():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc")
  a = CustomLongitudinalAdapter(params)

  resets = []
  a._stack.reset = lambda: resets.append(1)

  # Active-mode change (published by selfdrived) triggers a reset.
  a.set_active_mode("acc")
  assert len(resets) == 1
  assert a.mode is LongitudinalMode.ACC
  # Re-adopting the same mode does not.
  a.set_active_mode("acc")
  assert len(resets) == 1

  # Disabling does not reset.
  params._v["CustomLongitudinalEnabled"] = False
  a.refresh_params(initial=False)
  assert len(resets) == 1
  assert a.enabled is False

  # Re-enabling clears stale state with a reset.
  params._v["CustomLongitudinalEnabled"] = True
  a.refresh_params(initial=False)
  assert len(resets) == 2
  assert a.enabled is True


def test_adapter_passes_model_path_into_shadow_debug():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  out = a.evaluate(
    fake_sm(
      lead(d_rel=60.0, v_lead=25.0, v_rel=0.0),
      model_x=[0.0, 30.0, 60.0], model_y=[0.0, 0.5, 1.0], model_v=[25.0, 25.0, 25.0],
    ),
    25.0, 0.0, 25.0, -0.3, fake_scc(), fake_sla(), dt=0.05,
  )

  assert out.enabled is True
  assert out.debug["path_shadow_model_path_available"] is True
  assert out.debug["path_shadow_fault"] is False
  assert out.debug["actual_primary_lead_path_y_rel"] == pytest.approx(0.0)
  assert out.debug["path_shadow_primary_lead_path_y_rel"] == pytest.approx(1.0)


def test_adapter_contains_path_shadow_fault_without_fail_closed():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  sm = fake_sm(
    lead(d_rel=60.0, v_lead=25.0, v_rel=0.0),
    model_x=[0.0, 30.0, 60.0], model_v=[25.0, 25.0, 25.0],
  )
  sm['modelV2'].position = BadModelPathPosition()

  out = a.evaluate(sm, 25.0, 0.0, 25.0, -0.3, fake_scc(), fake_sla(), dt=0.05)

  assert out.enabled is True
  assert out.selected_intent != "fault"
  assert out.debug["path_shadow_model_path_available"] is False
  assert out.debug["path_shadow_fault"] is True


def test_adapter_can_skip_debug_collection_when_not_needed():
  sm = fake_sm(lead(d_rel=30.0, v_lead=18.0, v_rel=-1.0), long_active=True)
  rich = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")).evaluate(
    sm, 18.0, 0.0, 18.0, 0.2, fake_scc(), fake_sla(), dt=0.05,
  )
  lazy = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")).evaluate(
    sm, 18.0, 0.0, 18.0, 0.2, fake_scc(), fake_sla(), dt=0.05, collect_debug=False,
  )
  assert lazy.a_target == pytest.approx(rich.a_target)
  assert lazy.should_stop == rich.should_stop
  assert lazy.selected_intent == rich.selected_intent
  assert lazy.reason == rich.reason
  assert lazy.debug == {}


def test_refresh_params_never_rereads_mode():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc", LongitudinalDebugTraceMode="off")
  a = CustomLongitudinalAdapter(params)
  assert a.mode is LongitudinalMode.ACC
  params._v["CustomLongitudinalMode"] = "e2e"
  a.refresh_params()
  assert a.mode is LongitudinalMode.ACC  # deferred: applies only via the next-engagement capture
  params._v["LongitudinalDebugTraceMode"] = "log"
  a.refresh_params()
  assert a.debug_trace_mode == "log"  # advisory params still refresh on the slow cadence


def test_new_shadow_modes_parse_and_refresh_on_slow_cadence():
  params = FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
    CutInBrakeAssistMode="shadow",
    StandstillReleaseConfidenceMode="gate", UphillNetDemandCapMode="shadow",
    UphillNetDemandCeiling="1.2",
  )
  a = CustomLongitudinalAdapter(params)
  assert a.cut_in_brake_assist_mode == "shadow"
  assert a.standstill_release_confidence_mode == "gate"
  assert a._uphill_grade.mode == "shadow"
  assert a._uphill_grade.ceiling == pytest.approx(1.2)

  # A full/slow-cadence refresh applies the new (invalid -> off) values.
  params._v.update(
    CutInBrakeAssistMode="bad", StandstillReleaseConfidenceMode="bad",
    UphillNetDemandCapMode="bad", UphillNetDemandCeiling="bad",
  )
  a.refresh_params()
  assert a.cut_in_brake_assist_mode == "off"
  assert a.standstill_release_confidence_mode == "off"
  assert a._uphill_grade.mode == "off"
  assert a._uphill_grade.ceiling is None
  params._v["LongitudinalDebugTraceMode"] = "bad"
  a.refresh_params()
  assert a.debug_trace_mode == "off"


def test_debug_and_shadow_modes_refresh_every_params_period():
  params = FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
    LongitudinalDebugTraceMode="off", CutInBrakeAssistMode="off",
    StandstillReleaseConfidenceMode="off",
  )
  a = CustomLongitudinalAdapter(params)
  params._v.update(
    LongitudinalDebugTraceMode="log", CutInBrakeAssistMode="shadow",
    StandstillReleaseConfidenceMode="gate",
  )
  # Advisory/shadow params do not change before the refresh period elapses.
  for _ in range(49):
    a.maybe_refresh_params()
  assert a.debug_trace_mode == "off"
  assert a.cut_in_brake_assist_mode == "off"
  assert a.standstill_release_confidence_mode == "off"
  # They take effect on the tick that hits the refresh period.
  a.maybe_refresh_params()
  assert a.debug_trace_mode == "log"
  assert a.cut_in_brake_assist_mode == "shadow"
  assert a.standstill_release_confidence_mode == "gate"


def test_new_shadow_modes_are_exactly_non_actuating():
  base_params = dict(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")
  scenario = dict(
    sm=fake_sm(lead(d_rel=22.0, v_lead=8.0, v_rel=-3.0),
               model_x=[0.0, 20.0, 40.0], model_y=[0.0, 0.0, 0.0], model_v=[15.0, 15.0, 15.0]),
    v_ego=15.0, a_ego=0.0, v_cruise=18.0, seed_a_target=-0.2,
    scc=fake_scc(vision_active=True, vision_a=-0.4, vision_max_pred_lat_acc=1.4),
    sla=fake_sla(), dt=0.05,
  )
  baseline = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(**scenario)

  for key, debug_prefix in (
    ("CutInBrakeAssistMode", "cut_in_brake_assist"),
    ("StandstillReleaseConfidenceMode", "standstill_release_confidence"),
    ("CurveTrafficAdvisorMode", "curve_traffic"),
  ):
    params = dict(base_params)
    params[key] = "shadow"
    out = CustomLongitudinalAdapter(FakeParams(**params)).evaluate(**scenario)
    assert out.a_target == pytest.approx(baseline.a_target)
    assert out.should_stop == baseline.should_stop
    assert out.selected_intent == baseline.selected_intent
    assert out.reason == baseline.reason
    assert out.standstill_release_allowed == baseline.standstill_release_allowed
    assert out.standstill_release_source == baseline.standstill_release_source
    assert out.standstill_release_a_target == pytest.approx(baseline.standstill_release_a_target)
    assert out.standstill_release_reason == baseline.standstill_release_reason
    assert out.debug[f"{debug_prefix}_mode"] == "shadow"
    assert out.debug[f"{debug_prefix}_apply_supported"] is False


def test_absent_curve_traffic_advisor_mode_does_not_block_source_refresh():
  class ParamsMissingCurveTrafficKey:
    def __init__(self, **vals):
      self._v = vals
    def get_bool(self, k):
      return bool(self._v.get(k, False))
    def get(self, k):
      if k == "CurveTrafficAdvisorMode":
        raise KeyError("unregistered param")
      return self._v.get(k)
    def all_keys(self):
      return [k.encode() for k in self._v]

  a = CustomLongitudinalAdapter(ParamsMissingCurveTrafficKey(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc",
    SmartCruiseControlVision=True, SmartCruiseControlMap=True,
  ))
  assert a.mode is LongitudinalMode.SCC
  assert a.curve_traffic_advisor_mode == "off"
  assert a.sources.scc_curve_vision_enabled is True
  assert a.sources.scc_curve_map_enabled is True


def test_malformed_steering_telemetry_is_fail_soft():
  base_params = dict(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")
  baseline_sm = fake_sm(steering_angle_deg=5.0, steering_torque=2.0)
  malformed_sm = fake_sm(steering_angle_deg="bad", steering_torque=None)
  baseline = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(
    baseline_sm, 20.0, 0.0, 22.0, 0.4, fake_scc(), fake_sla(), dt=0.05,
  )
  malformed = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(
    malformed_sm, 20.0, 0.0, 22.0, 0.4, fake_scc(), fake_sla(), dt=0.05,
  )
  assert malformed.enabled is True
  assert malformed.a_target == pytest.approx(baseline.a_target)


def test_apply_values_preserved_but_non_actuating_in_acc():
  base_params = dict(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")
  scenario = dict(
    sm=fake_sm(lead(d_rel=22.0, v_lead=8.0, v_rel=-3.0),
               model_x=[0.0, 20.0, 40.0], model_y=[0.0, 0.0, 0.0], model_v=[15.0, 15.0, 15.0]),
    v_ego=15.0, a_ego=0.0, v_cruise=18.0, seed_a_target=-0.2,
    scc=fake_scc(vision_active=True, vision_a=-0.4, vision_max_pred_lat_acc=1.4),
    sla=fake_sla(), dt=0.05,
  )
  baseline = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(**scenario)
  for key, value, debug_prefix in (
    ("CutInBrakeAssistMode", "apply", "cut_in_brake_assist"),
    ("CurveTrafficAdvisorMode", "apply_conservative", "curve_traffic"),
    ("StandstillReleaseConfidenceMode", "gate", "standstill_release_confidence"),
  ):
    params = dict(base_params)
    params[key] = value
    adapter = CustomLongitudinalAdapter(FakeParams(**params))
    adapter.research_actuation_allowed = True
    out = adapter.evaluate(**scenario)
    assert out.a_target == pytest.approx(baseline.a_target)
    assert out.should_stop == baseline.should_stop
    assert out.selected_intent == baseline.selected_intent
    assert out.reason == baseline.reason
    assert out.standstill_release_allowed == baseline.standstill_release_allowed
    assert out.debug[f"{debug_prefix}_mode"] == value
    assert out.debug[f"{debug_prefix}_apply_supported"] is True
    if key == "CurveTrafficAdvisorMode":
      assert out.debug["curve_traffic_effective_mode"] == value


def test_apply_modes_degrade_to_shadow_when_research_gate_false():
  base_params = dict(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")
  scenario = dict(
    sm=fake_sm(lead(d_rel=22.0, v_lead=8.0, v_rel=-3.0),
               model_x=[0.0, 20.0, 40.0], model_y=[0.0, 0.0, 0.0], model_v=[15.0, 15.0, 15.0]),
    v_ego=15.0, a_ego=0.0, v_cruise=18.0, seed_a_target=-0.2,
    scc=fake_scc(vision_active=True, vision_a=-0.4, vision_max_pred_lat_acc=1.4),
    sla=fake_sla(), dt=0.05,
  )
  baseline = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(**scenario)
  for key, value, debug_prefix in (
    ("CutInBrakeAssistMode", "apply", "cut_in_brake_assist"),
    ("CurveTrafficAdvisorMode", "apply_conservative", "curve_traffic"),
    ("StandstillReleaseConfidenceMode", "gate", "standstill_release_confidence"),
  ):
    params = dict(base_params)
    params[key] = value
    out = CustomLongitudinalAdapter(FakeParams(**params)).evaluate(**scenario)
    assert out.a_target == pytest.approx(baseline.a_target)
    assert out.should_stop == baseline.should_stop
    assert out.selected_intent == baseline.selected_intent
    assert out.reason == baseline.reason
    assert out.standstill_release_allowed == baseline.standstill_release_allowed
    assert out.debug[f"{debug_prefix}_mode"] == value
    assert out.debug[f"{debug_prefix}_effective_mode"] == "shadow"
    assert out.debug[f"{debug_prefix}_apply_supported"] is False
    assert out.debug[f"{debug_prefix}_eligible"] is False


def test_curve_traffic_advisor_mode_is_non_actuating_and_wired():
  base_params = dict(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc")
  # A gentle left arc modeled as an n-sample circular arc; long_active is required for activation.
  n = 20
  radius = 120.0
  thetas = [i * 7.0 / radius for i in range(n)]
  model_x = [radius * math.sin(t) for t in thetas]
  model_y = [radius * (1.0 - math.cos(t)) for t in thetas]
  class FreshFakeSM(dict):
    recv_time = {'modelV2': time.monotonic()}
  sm = FreshFakeSM(fake_sm(model_x=model_x, model_y=model_y, model_v=[15.0] * n, long_active=True))
  scenario = dict(
    sm=sm,
    v_ego=15.0, a_ego=0.0, v_cruise=18.0, seed_a_target=0.2,
    scc=fake_scc(), sla=fake_sla(), dt=0.05,
  )
  off = CustomLongitudinalAdapter(FakeParams(**base_params)).evaluate(**scenario)
  shadow = CustomLongitudinalAdapter(FakeParams(CurveTrafficAdvisorMode="shadow", **base_params)).evaluate(**scenario)
  assert shadow.a_target == pytest.approx(off.a_target)
  assert shadow.should_stop == off.should_stop
  assert shadow.selected_intent == off.selected_intent
  assert shadow.reason == off.reason
  assert shadow.standstill_release_allowed == off.standstill_release_allowed
  assert shadow.standstill_release_source == off.standstill_release_source
  assert shadow.standstill_release_a_target == pytest.approx(off.standstill_release_a_target)
  assert off.debug["curve_traffic_mode"] == "off"
  assert off.debug["curve_traffic_active"] is False
  assert shadow.debug["curve_traffic_mode"] == "shadow"
  assert shadow.debug["curve_traffic_active"] is True
  assert shadow.debug["curve_traffic_advisor_fault"] is False
  assert shadow.debug["curve_traffic_apply_supported"] is False


# --- DragEstimator wiring (learned flat-road coast decel) ---

def test_coast_accel_uses_learned_rolling_term():
  from openpilot.sunnypilot.custom.longitudinal.coast_horizon import ACCELERATION_DUE_TO_GRAVITY, DEFAULT_COAST_DECEL
  from openpilot.sunnypilot.custom.longitudinal.wiring import _coast_accel
  assert _coast_accel(0.0, -0.5) == pytest.approx(-0.5)
  assert _coast_accel(0.0) == pytest.approx(DEFAULT_COAST_DECEL)
  assert _coast_accel(0.1, -0.3) < -0.3  # uphill adds deceleration
  # predict-side grade uses the same gravity the DragEstimator removes when learning,
  # so learn->predict round-trips exactly.
  assert _coast_accel(0.05, -0.3) == pytest.approx(-0.3 - ACCELERATION_DUE_TO_GRAVITY * math.sin(0.05))


def test_drag_estimator_learns_only_on_manual_coast_frames():
  from openpilot.sunnypilot.custom.longitudinal.coast_horizon import DEFAULT_COAST_DECEL
  ad = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True))
  scc, sla = fake_scc(), fake_sla()
  assert ad._drag.coast_decel == pytest.approx(DEFAULT_COAST_DECEL)

  # Manual off-pedal coasting (disengaged): the estimate moves toward the measured decel.
  for _ in range(200):
    ad.evaluate(fake_sm(pitch=0.0, long_active=False), 20.0, -0.6, 20.0, 0.0, scc, sla)
  after_manual = ad._drag.coast_decel
  assert after_manual < DEFAULT_COAST_DECEL

  # Engaged frames count as on-throttle (system throttle/brake don't set pedal flags):
  # even hard measured decel must not move the estimate.
  for _ in range(200):
    ad.evaluate(fake_sm(pitch=0.0, long_active=True), 20.0, -1.5, 20.0, 0.0, scc, sla)
  assert ad._drag.coast_decel == pytest.approx(after_manual)

  # Pedal frames are excluded too.
  for _ in range(200):
    ad.evaluate(fake_sm(pitch=0.0, gas=True), 20.0, -1.5, 20.0, 0.0, scc, sla)
  assert ad._drag.coast_decel == pytest.approx(after_manual)


# --- map-coast tier plumbing ---

def test_build_stack_inputs_carries_map_coast_fields():
  inp = build_stack_inputs(
    v_ego=25.0, a_ego=0.0, v_cruise=25.0, seed_a_target=0.0, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD,
    sources=SourceToggles(scc_curve_map_enabled=True),
    scc_map_coast_v_target=15.0, scc_map_coast_distance=300.0, map_coast_mode="shadow",
  )
  assert inp.map_coast_mode == "shadow"
  assert inp.map_coast_v_target == pytest.approx(15.0)
  assert inp.map_coast_distance == pytest.approx(300.0)


def _map_coast_adapter(mode, research=False):
  params = FakeParams(CustomLongitudinalEnabled=True, SmartCruiseControlMap=True,
                      **({"MapCoastMode": mode} if mode is not None else {}))
  ad = CustomLongitudinalAdapter(params)
  ad.research_actuation_allowed = research
  scc = fake_scc()
  scc.map.coast_v_target = 10.0
  scc.map.coast_distance = 250.0
  return ad, scc


def test_map_coast_mode_param_fails_closed():
  assert _map_coast_adapter("shadow")[0].map_coast_mode == "shadow"
  assert _map_coast_adapter("apply")[0].map_coast_mode == "apply"
  assert _map_coast_adapter("bogus")[0].map_coast_mode == "off"
  assert _map_coast_adapter(None)[0].map_coast_mode == "off"


def test_map_coast_shadow_is_non_actuating_and_observable():
  sm = fake_sm(long_active=True)
  ad_off, scc_off = _map_coast_adapter(None)
  ad_shadow, scc_shadow = _map_coast_adapter("shadow")
  out_off = ad_off.evaluate(sm, 25.0, 0.0, 25.0, 0.0, scc_off, fake_sla())
  out_shadow = ad_shadow.evaluate(sm, 25.0, 0.0, 25.0, 0.0, scc_shadow, fake_sla())
  assert out_shadow.a_target == pytest.approx(out_off.a_target)
  assert out_shadow.debug.get("map_coast_eligible") is True
  assert out_shadow.debug.get("map_coast_cap") < 0.0
  assert out_shadow.debug.get("map_coast_applied") is False
  assert "map_coast_eligible" not in out_off.debug


def test_map_coast_apply_gated_by_research_actuation():
  sm = fake_sm(long_active=True)
  ad_ungated, scc_ungated = _map_coast_adapter("apply", research=False)
  out_ungated = ad_ungated.evaluate(sm, 25.0, 0.0, 25.0, 0.0, scc_ungated, fake_sla())
  ad_off, scc_off = _map_coast_adapter(None)
  out_off = ad_off.evaluate(sm, 25.0, 0.0, 25.0, 0.0, scc_off, fake_sla())
  assert out_ungated.a_target == pytest.approx(out_off.a_target)  # no research gate -> inert

  ad_apply, scc_apply = _map_coast_adapter("apply", research=True)
  out_apply = ad_apply.evaluate(sm, 25.0, 0.0, 25.0, 0.0, scc_apply, fake_sla())
  assert out_apply.a_target < out_off.a_target        # coast cap binds
  assert out_apply.a_target >= -0.3                   # never harder than natural coast
  assert out_apply.debug.get("map_coast_applied") is True


def test_departure_prediction_mode_is_wired_and_off_shadow_are_parity():
  kwargs = dict(
    v_ego=0.0, a_ego=0.0, v_cruise=12.0, seed_a_target=0.0,
    accel_limits=DEFAULT_ACCEL_LIMITS, lead_one=lead(d_rel=6.5, v_lead=0.8, v_rel=0.5), lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(),
    departure_prediction_mode="shadow",
  )
  assert build_stack_inputs(**kwargs).departure_prediction_mode == "shadow"

  scenario = dict(
    sm=fake_sm(lead(d_rel=6.5, v_lead=0.8, v_rel=0.5), long_active=True),
    v_ego=0.0, a_ego=0.0, v_cruise=12.0, seed_a_target=0.0,
    scc=fake_scc(), sla=fake_sla(), dt=0.05,
  )
  off = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc", DeparturePredictionMode="off",
  )).evaluate(**scenario)
  shadow = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc", DeparturePredictionMode="shadow",
  )).evaluate(**scenario)

  assert shadow.a_target == pytest.approx(off.a_target)
  assert shadow.should_stop == off.should_stop
  assert shadow.selected_intent == off.selected_intent
  assert shadow.reason == off.reason
  assert off.debug["departure_prediction_mode"] == "off"
  assert shadow.debug["departure_prediction_mode"] == "shadow"
  assert shadow.debug["departure_prediction_effective_mode"] == "shadow"
  assert shadow.debug["departure_prediction_apply_supported"] is False


def test_departure_prediction_invalid_mode_is_off_and_apply_downgrades_without_research_gate():
  scenario = dict(
    sm=fake_sm(lead(d_rel=6.5, v_lead=0.8, v_rel=0.5), long_active=True),
    v_ego=0.0, a_ego=0.0, v_cruise=12.0, seed_a_target=0.0,
    scc=fake_scc(), sla=fake_sla(), dt=0.05,
  )
  invalid = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc", DeparturePredictionMode="bogus",
  ))
  assert invalid.departure_prediction_mode == "off"

  apply = CustomLongitudinalAdapter(FakeParams(
    CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc", DeparturePredictionMode="apply",
  ))
  apply.research_actuation_allowed = False
  out = apply.evaluate(**scenario)
  assert out.debug["departure_prediction_mode"] == "apply"
  assert out.debug["departure_prediction_effective_mode"] == "shadow"
  assert out.debug["departure_prediction_apply_supported"] is False
  assert out.a_target == pytest.approx(0.0)


def test_adapter_refresh_cannot_enable_the_cp_aware_research_gate():
  params = FakeParams(
    CustomLongitudinalEnabled=True,
    CustomLongitudinalMode="scc",
    DeparturePredictionMode="apply",
    AllowLongitudinalResearchActuation=True,
  )
  adapter = CustomLongitudinalAdapter(params)

  assert adapter.research_actuation_allowed is False
  for _ in range(50):
    adapter.maybe_refresh_params()
  assert adapter.research_actuation_allowed is False

  out = adapter.evaluate(
    fake_sm(lead(d_rel=6.5, v_lead=0.8, v_rel=0.5), long_active=True),
    0.0, 0.0, 12.0, 0.0, fake_scc(), fake_sla(), dt=0.05,
  )
  assert out.debug["departure_prediction_effective_mode"] == "shadow"
  assert out.debug["departure_prediction_apply_supported"] is False
