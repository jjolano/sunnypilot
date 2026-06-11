import pytest

from openpilot.selfdrive.controls.lib.curve_speed_policy import (
  CurveSpeedEvidence,
  select_curve_speed_policy,
)


def evidence(source, **overrides):
  values = {
    "active": True,
    "v_target": 18.0,
    "a_target": -0.4,
    "distance_to_target": 45.0,
    "confidence": 0.8,
    "urgency": 0.5,
    "source": source,
    "reason": f"{source}_curve",
  }
  values.update(overrides)
  return CurveSpeedEvidence(**values)


def test_vision_near_field_cap_produces_active_result():
  result = select_curve_speed_policy(mode="SCC", current_speed=25.0, vision=evidence("vision", distance_to_target=30.0))

  assert result.active
  assert result.source == "vision"
  assert result.v_target == pytest.approx(18.0)


def test_model_confirmed_map_curve_produces_active_result():
  result = select_curve_speed_policy(
    mode="SCC",
    current_speed=25.0,
    map_advisory=evidence("map", v_target=12.0, a_target=-0.7, distance_to_target=120.0, model_confirmed=True, model_horizon_distance=140.0),
  )

  assert result.active
  assert result.source == "map"
  assert result.a_target == pytest.approx(-0.7)


def test_map_only_large_slowdown_without_model_confirmation_is_ignored():
  result = select_curve_speed_policy(
    mode="SCC",
    current_speed=25.0,
    map_advisory=evidence("map", v_target=12.0, a_target=-0.7, distance_to_target=80.0, model_confirmed=False, model_horizon_distance=100.0),
  )

  assert not result.active
  assert result.reason == "no_curve_evidence"


def test_more_urgent_nearer_confident_evidence_wins_when_both_exist():
  vision = evidence("vision", v_target=18.0, a_target=-0.4, distance_to_target=30.0, confidence=0.8)
  map_curve = evidence("map", v_target=15.0, a_target=-0.8, distance_to_target=80.0, confidence=0.9, model_confirmed=True)

  result = select_curve_speed_policy(mode="SCC", current_speed=25.0, vision=vision, map_advisory=map_curve)

  assert result.active
  assert result.source == "map"
  assert result.a_target == pytest.approx(-0.8)


def test_acc_and_e2e_mode_boundaries_ignore_scc_curve_components():
  vision = evidence("vision")

  acc = select_curve_speed_policy(mode="ACC", current_speed=25.0, vision=vision)
  e2e = select_curve_speed_policy(mode="E2E", current_speed=25.0, vision=vision)

  assert not acc.active
  assert not e2e.active
  assert acc.reason == "mode_boundary"
  assert e2e.reason == "mode_boundary"
