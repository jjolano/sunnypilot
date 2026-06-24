"""Tests for the shadow-only scenario context module and its stack/wiring wiring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.scenario_context import (
  MODE_OFF,
  MODE_SHADOW,
  ScenarioContextResult,
  predict_scenario_context,
)
from openpilot.sunnypilot.custom.longitudinal.stack import (
  CustomLongitudinalStack,
  LongitudinalStackInputs,
)
from openpilot.sunnypilot.custom.longitudinal.wiring import (
  DEFAULT_ACCEL_LIMITS,
  build_stack_inputs,
)

DT = 0.05
LIMITS = DEFAULT_ACCEL_LIMITS


def lead(d_rel=30.0, v_lead=12.0, v_rel=0.0, a_lead_k=0.0, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=a_lead_k, yRel=0.0, radarTrackId=3, radar=True,
                         modelProb=0.9, aLeadTau=1.0)


def inputs(**kw):
  d = dict(
    v_ego=20.0, a_ego=0.0, accel_coast=-0.3,
    standstill=False, steering_angle_deg=0.0, steering_torque=0.0,
    leads=(None, None),
    model_should_stop=False, model_stop_distance=None,
    speed_limit_active=False, curve_active=False,
    gas_pressed=False, brake_pressed=False,
  )
  d.update(kw)
  return d


def predict(**kw):
  return predict_scenario_context(**inputs(**kw))


# --- Pure classification tests ---

def test_mode_off_is_inactive():
  r = predict(mode=MODE_OFF, gas_pressed=True)
  assert r.mode == MODE_OFF
  assert r.effective_mode == MODE_OFF
  assert r.apply_supported is False
  assert r.active is False
  assert r.scenario == "mode_off"
  assert r.allowed_effect == "none"


def test_invalid_mode_sanitizes_to_off():
  r = predict_scenario_context("apply", **inputs())
  assert r.mode == MODE_OFF
  assert r.scenario == "mode_off"


def test_driver_override_takes_precedence():
  for pedal in ("gas", "brake"):
    r = predict(mode=MODE_SHADOW, **{f"{pedal}_pressed": True})
    assert r.scenario == "driver_override"
    assert r.allowed_effect == "none"
    assert r.confidence >= 0.8


def test_standstill_at_low_speed():
  r = predict(mode=MODE_SHADOW, v_ego=0.0, standstill=True)
  assert r.scenario == "standstill"
  assert r.allowed_effect == "none"


def test_approach_stop_from_model_stop_distance():
  r = predict(mode=MODE_SHADOW, v_ego=13.0, model_should_stop=True, model_stop_distance=18.0)
  assert r.scenario == "approach_stop"
  assert r.allowed_effect == "stop_commit"
  assert r.confidence >= 0.8


def test_closing_lead_detected():
  r = predict(mode=MODE_SHADOW, leads=(lead(d_rel=22.0, v_rel=-3.0), None))
  assert r.scenario == "closing_lead"
  assert r.allowed_effect == "restrict_only"


def test_lead_pullaway_detected():
  r = predict(mode=MODE_SHADOW, v_ego=2.0, leads=(lead(d_rel=20.0, v_lead=8.0, v_rel=6.0), None))
  assert r.scenario == "lead_pullaway"
  assert r.allowed_effect == "progress_with_guard"


def test_lead_follow_default():
  r = predict(mode=MODE_SHADOW, leads=(lead(d_rel=30.0, v_rel=0.0), None))
  assert r.scenario == "lead_follow"
  assert r.allowed_effect == "shadow_only"


def test_stop_and_go_close_stopped_lead():
  r = predict(mode=MODE_SHADOW, v_ego=0.5, leads=(lead(d_rel=5.0, v_lead=0.0, v_rel=0.0), None))
  assert r.scenario == "stop_and_go"
  assert r.allowed_effect == "stop_commit"


def test_curve_approach():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, curve_active=True)
  assert r.scenario == "curve_approach"
  assert r.allowed_effect == "restrict_only"


def test_speed_limit_drop():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, speed_limit_active=True)
  assert r.scenario == "speed_limit_drop"
  assert r.allowed_effect == "restrict_only"


def test_downhill_grade_and_scenario():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, accel_coast=0.5)
  assert r.road_grade == "downhill"
  assert r.scenario == "downhill_coast"
  assert r.allowed_effect == "progress_with_guard"


def test_uphill_grade_and_scenario():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, accel_coast=-1.0)
  assert r.road_grade == "uphill"
  assert r.scenario == "uphill_recovery"
  assert r.allowed_effect == "progress_with_guard"


def test_turning_from_steering_angle():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, steering_angle_deg=30.0)
  assert r.scenario == "turning"
  assert r.allowed_effect == "restrict_only"


def test_turning_from_steering_torque():
  r = predict(mode=MODE_SHADOW, v_ego=15.0, steering_torque=80.0)
  assert r.scenario == "turning"


def test_turning_outranks_grade_progress():
  # Steering input should win over a downhill/uphill grade label.
  r = predict(mode=MODE_SHADOW, v_ego=15.0, accel_coast=0.5, steering_angle_deg=30.0)
  assert r.road_grade == "downhill"
  assert r.scenario == "turning"
  assert r.allowed_effect == "restrict_only"


def test_current_effect_is_always_none_in_shadow():
  for scenario in ("driver_override", "approach_stop", "downhill_coast", "open_road"):
    kw: dict[str, object]
    if scenario == "driver_override":
      kw = {"gas_pressed": True}
    elif scenario == "approach_stop":
      kw = {"v_ego": 13.0, "model_should_stop": True, "model_stop_distance": 18.0}
    elif scenario == "downhill_coast":
      kw = {"v_ego": 15.0, "accel_coast": 0.5}
    else:
      kw = {"v_ego": 20.0}
    r = predict(mode=MODE_SHADOW, **kw)
    assert r.current_effect == "none"


def test_open_road_when_clear():
  r = predict(mode=MODE_SHADOW, v_ego=20.0)
  assert r.scenario == "open_road"
  assert r.allowed_effect == "progress_with_guard"


def test_unknown_when_low_speed_and_no_signals():
  r = predict(mode=MODE_SHADOW, v_ego=3.0)
  assert r.scenario == "unknown"


def test_debug_dict_uses_prefix():
  r = predict(mode=MODE_SHADOW, v_ego=20.0)
  d = r.debug_dict()
  assert all(k.startswith("scenario_context_") for k in d)
  assert d["scenario_context_scenario"] == "open_road"
  assert d["scenario_context_current_effect"] == "none"


# --- Stack integration tests ---

def test_scenario_context_mode_does_not_change_actuation():
  """a_target, should_stop, and decision outputs must be identical with mode off vs shadow."""
  def make_inp(scenario_mode):
    return LongitudinalStackInputs(
      v_ego=20.0, v_cruise=22.0, seed_a_target=0.4,
      accel_coast=-0.6,
      scenario_context_mode=scenario_mode,
      steering_angle_deg=5.0,
      speed_limit_active=True,
      curve_active=True,
      mode=LongitudinalMode.SCC, long_active=True,
    )
  off = CustomLongitudinalStack().update(make_inp(MODE_OFF), DT)
  shadow = CustomLongitudinalStack().update(make_inp(MODE_SHADOW), DT)

  assert shadow.a_target == pytest.approx(off.a_target)
  assert shadow.should_stop == off.should_stop
  assert shadow.decision.reason == off.decision.reason
  assert shadow.decision.selected_intent == off.decision.selected_intent
  assert shadow.standstill_release_allowed == off.standstill_release_allowed

  assert off.debug["scenario_context_scenario"] == "mode_off"
  assert off.debug["scenario_context_active"] is False
  assert shadow.debug["scenario_context_scenario"] != "mode_off"
  assert shadow.debug["scenario_context_active"] is True
  assert shadow.debug["scenario_context_fault"] is False


def test_scenario_context_fault_does_not_leak():
  class BadMode:
    def __str__(self):
      raise RuntimeError("bad mode")

  def make_inp(scenario_mode):
    return LongitudinalStackInputs(
      v_ego=15.0, v_cruise=18.0, seed_a_target=0.0,
      scenario_context_mode=scenario_mode, mode=LongitudinalMode.ACC,
    )

  baseline = CustomLongitudinalStack().update(make_inp(MODE_OFF), DT)
  broken = CustomLongitudinalStack().update(make_inp(BadMode()), DT)
  assert broken.a_target == pytest.approx(baseline.a_target)
  assert broken.debug["scenario_context_fault"] is True


# --- Wiring integration tests ---

def test_build_stack_inputs_carries_scenario_context_fields():
  inp = build_stack_inputs(
    v_ego=12.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.2,
    accel_limits=LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0,
    scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.ACC, personality=Personality.STANDARD,
    sources=None,  # type: ignore[arg-type]
    scenario_context_mode=MODE_SHADOW,
    standstill=True,
    steering_angle_deg=15.0,
    steering_torque=2.5,
  )
  assert inp.scenario_context_mode == MODE_SHADOW
  assert inp.standstill is True
  assert inp.steering_angle_deg == pytest.approx(15.0)
  assert inp.steering_torque == pytest.approx(2.5)
