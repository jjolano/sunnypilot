"""Tests for the plannerd wiring adapter (opt-in custom longitudinal)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.decision import decide
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy import LongitudinalScene, build_candidates
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.wiring import (
  DEFAULT_ACCEL_LIMITS,
  CustomLongitudinalAdapter,
  build_stack_inputs,
  _model_stop_distance,
)


def lead(d_rel=30.0, v_lead=12.0, v_rel=None, status=True):
  ld = SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=0.0,
                       yRel=0.0, radarTrackId=3, radar=True, modelProb=0.9, aLeadTau=1.0)
  if v_rel is not None:
    ld.vRel = v_rel
  return ld


def fake_sm(lead_one=None, brake=False, gas=False, model_should_stop=False, model_accel=0.0, pitch=0.0,
            model_x=None, model_y=None, model_v=None, model_leads=None):
  position = SimpleNamespace()
  if model_x is not None:
    position.x = model_x
  if model_y is not None:
    position.y = model_y
  return {
    'radarState': SimpleNamespace(leadOne=lead_one, leadTwo=None),
    'carState': SimpleNamespace(brakePressed=brake, gasPressed=gas),
    'modelV2': SimpleNamespace(action=SimpleNamespace(shouldStop=model_should_stop,
                                                       desiredAcceleration=model_accel),
                               position=position,
                               velocity=SimpleNamespace(x=model_v),
                               leadsV3=list(model_leads or [])),
    'carControl': SimpleNamespace(orientationNED=[0.0, pitch, 0.0]),
    'controlsState': SimpleNamespace(forceDecel=False),
  }


# A trajectory that decelerates to rest ~38 m ahead (velocity dips to ~0 at index 5).
STOP_TRAJ_V = [15.0, 12.0, 9.0, 6.0, 3.0, 0.2, 0.0, 0.0]
STOP_TRAJ_X = [0.0, 13.0, 24.0, 32.0, 37.0, 38.0, 38.0, 38.0]
CRUISE_TRAJ_V = [20.0] * 8
CRUISE_TRAJ_X = [20.0 * i for i in range(8)]


def fake_scc(vision_active=False, vision_a=0.0, map_active=False, map_a=0.0):
  return SimpleNamespace(
    vision=SimpleNamespace(is_active=vision_active, output_a_target=vision_a),
    map=SimpleNamespace(is_active=map_active, output_a_target=map_a),
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


def test_map_only_curve_tags_curve_map_and_is_admitted_in_scc():
  # #18: a map-sourced curve was always tagged CURVE_VISION, so with only the map toggle on it was
  # silently dropped under SCC. Tag by binding source so SCC admits it.
  inp = build_stack_inputs(
    v_ego=20.0, a_ego=0.0, v_cruise=22.0, seed_a_target=0.4, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=True, scc_map_a_target=-0.8,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(False, True),
  )
  assert inp.curve_source is EvidenceClass.CURVE_MAP and inp.curve_a_target == pytest.approx(-0.8)
  scene = LongitudinalScene(v_ego=20.0, v_cruise=22.0, seed_a_target=0.4,
                            curve_active=True, curve_a_target=-0.8, curve_source=EvidenceClass.CURVE_MAP)
  cands = build_candidates(scene)
  on = decide(cands, LongitudinalMode.SCC, DEFAULT_ACCEL_LIMITS, SourceToggles(False, True))
  off = decide(cands, LongitudinalMode.SCC, DEFAULT_ACCEL_LIMITS, SourceToggles(False, False))
  assert on.a_target < 0.4                          # map-curve admitted -> caps the cruise
  assert off.a_target == pytest.approx(0.4)         # map toggle off -> curve not admitted, cruise stands


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
  for _ in range(40):
    a.apply(fake_sm(model_should_stop=True, model_accel=-2.5, gas=True), 15.0, 0.0, 15.0, 0.0,
            fake_scc(), fake_sla(), dt=0.05)
  assert a._stop_trust.confidence < before  # repeated driver countermanding softens trust


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
  assert inp.model_stop_distance == pytest.approx(38.0)


def test_distance_aware_stop_approach_brakes_in_e2e():
  # No upstream shouldStop, but the model trajectory predicts a near stop -> the distance-aware
  # stop-approach path engages and brakes in E2E (and is excluded in ACC).
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="e2e"))
  out_e2e = a.apply(fake_sm(model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V), 15.0, 0.0, 15.0, 0.0,
                    fake_scc(), fake_sla())
  acc = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  out_acc = acc.apply(fake_sm(model_x=STOP_TRAJ_X, model_v=STOP_TRAJ_V), 15.0, 0.0, 15.0, 0.0,
                      fake_scc(), fake_sla())
  assert out_e2e < 0.0
  assert out_acc == pytest.approx(0.0)


def test_default_mode_is_scc_the_dec_replacement():
  a = CustomLongitudinalAdapter(params=None)
  assert a.mode is LongitudinalMode.SCC                     # default: the intelligent blend


def test_explicit_mode_from_param():
  for setting, expected in (("acc", LongitudinalMode.ACC), ("e2e", LongitudinalMode.E2E),
                            ("scc", LongitudinalMode.SCC)):
    a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode=setting))
    assert a.mode is expected


def test_mode_and_enable_refresh_live():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
                      LongitudinalPersonality="1", SmartCruiseControlVision=True, SmartCruiseControlMap=False)
  a = CustomLongitudinalAdapter(params)
  params._v.update(CustomLongitudinalEnabled=False, CustomLongitudinalMode="e2e",
                   LongitudinalPersonality="2", SmartCruiseControlVision=False, SmartCruiseControlMap=True)
  a.maybe_refresh_params()
  assert a.enabled is False
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
  assert a.personality is Personality.AGGRESSIVE
  assert a.sources.scc_curve_vision_enabled is False
  assert a.sources.scc_curve_map_enabled is True


def test_stack_reset_on_mode_change_and_re_enable():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc")
  a = CustomLongitudinalAdapter(params)

  resets = []
  a._stack.reset = lambda: resets.append(1)

  # Mode change triggers a reset.
  params._v["CustomLongitudinalMode"] = "acc"
  a.refresh_params(initial=False)
  assert len(resets) == 1
  assert a.mode is LongitudinalMode.ACC

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
  assert out.debug["path_shadow_primary_lead_path_y_rel"] == pytest.approx(-1.0)


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


def test_lead_path_clearance_modes_are_exactly_non_actuating():
  model_leads = [SimpleNamespace(
    x=[55.0, 60.0, 65.0, 70.0], y=[-0.4, -1.0, -1.8, -2.0], t=[0.0, 1.0, 2.0, 3.0],
    xStd=[0.5, 0.5, 0.5, 0.5], yStd=[0.2, 0.2, 0.2, 0.2], prob=0.9,
  )]
  outputs = {}
  for mode in ("off", "shadow", "apply"):
    adapter = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc",
                                                  LeadPathClearanceMode=mode))
    sm = fake_sm(
      lead(d_rel=55.0, v_lead=10.0, v_rel=-5.0),
      model_x=[0.0, 40.0, 80.0], model_y=[0.0, 0.0, 0.0], model_v=[15.0, 15.0, 15.0],
      model_leads=model_leads,
    )
    out = None
    for _ in range(12):
      out = adapter.evaluate(sm, 15.0, 0.0, 18.0, -0.3, fake_scc(), fake_sla(), dt=0.05)
    assert out is not None
    outputs[mode] = out

  baseline = outputs["off"]
  for mode in ("shadow", "apply"):
    out = outputs[mode]
    assert out.a_target == pytest.approx(baseline.a_target)
    assert out.should_stop == baseline.should_stop
    assert out.selected_intent == baseline.selected_intent
    assert out.reason == baseline.reason
    assert out.standstill_release_allowed == baseline.standstill_release_allowed
    assert out.standstill_release_source == baseline.standstill_release_source
    assert out.standstill_release_a_target == pytest.approx(baseline.standstill_release_a_target)
    assert out.standstill_release_reason == baseline.standstill_release_reason

  assert outputs["shadow"].debug["lead_path_clearance_mode"] == "shadow"
  assert outputs["shadow"].debug["lead_path_clearance_shadow_eligible"] is True
  assert outputs["apply"].debug["lead_path_clearance_mode"] == "apply"
  assert outputs["apply"].debug["lead_path_clearance_effective_mode"] == "shadow"
  assert outputs["apply"].debug["lead_path_clearance_apply_supported"] is False
  assert outputs["apply"].debug["lead_path_clearance_shadow_eligible"] is True


def test_adapter_refreshes_debug_trace_mode_on_mode_only():
  params = FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc", LongitudinalDebugTraceMode="off")
  a = CustomLongitudinalAdapter(params)
  assert a.debug_trace_mode == "off"
  params._v["LongitudinalDebugTraceMode"] = "log"
  a.refresh_params(mode_only=True)
  assert a.debug_trace_mode == "log"
  params._v["LongitudinalDebugTraceMode"] = "bad"
  a.refresh_params(mode_only=True)
  assert a.debug_trace_mode == "off"
