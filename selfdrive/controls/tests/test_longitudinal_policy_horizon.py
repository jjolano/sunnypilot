import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import DecisionSource
from openpilot.selfdrive.controls.lib.longitudinal_policy_horizon import (
  CurveSpeedPolicyResult,
  LongitudinalPolicyHorizon,
  select_curve_speed_policy_result,
)


def make_horizon(**overrides):
  values = {
    "source": DecisionSource.SCC_VISION,
    "active": True,
    "reason": "vision_curve",
    "v_target": 18.0,
    "a_target": -0.4,
    "distance": 40.0,
    "time": 2.0,
    "confidence": 0.9,
    "urgency": 0.5,
  }
  values.update(overrides)
  return LongitudinalPolicyHorizon(**values)


def make_curve_result(**overrides):
  values = {
    "source": DecisionSource.SCC_VISION,
    "active": True,
    "reason": "vision_curve",
    "v_target": 18.0,
    "a_target": -0.4,
    "confidence": 0.9,
    "urgency": 0.5,
    "lateral_accel_limit": 2.0,
    "horizon": make_horizon(),
  }
  values.update(overrides)
  return CurveSpeedPolicyResult(**values)


@pytest.mark.parametrize("overrides", [
  {"active": False},
  {"reason": ""},
  {"v_target": float("nan")},
  {"v_target": -1.0},
  {"a_target": float("inf")},
  {"distance": -1.0},
  {"time": float("nan")},
])
def test_policy_horizon_rejects_invalid_or_inactive_inputs(overrides):
  assert not make_horizon(**overrides).valid


def test_curve_policy_result_converts_to_advisory_candidate_with_horizon_metadata():
  result = make_curve_result(confidence=0.8, horizon=make_horizon(confidence=0.7, urgency=0.8))

  candidate = result.to_candidate()

  assert candidate is not None
  assert candidate.source == DecisionSource.SCC_VISION
  assert candidate.v_target == pytest.approx(18.0)
  assert candidate.a_target == pytest.approx(-0.4)
  assert candidate.confidence == pytest.approx(0.7)
  assert candidate.urgency == pytest.approx(0.8)
  assert candidate.debug["curve_lateral_accel_limit"] == pytest.approx(2.0)
  assert candidate.debug["curve_horizon_source"] == "scc_vision"


@pytest.mark.parametrize("overrides", [
  {"active": False},
  {"source": DecisionSource.SPEED_LIMIT},
  {"reason": ""},
  {"v_target": float("nan")},
  {"a_target": float("inf")},
  {"horizon": make_horizon(distance=-1.0)},
])
def test_curve_policy_result_rejects_invalid_inputs(overrides):
  assert make_curve_result(**overrides).to_candidate() is None


def test_curve_policy_arbitration_selects_most_restrictive_active_curve():
  vision = make_curve_result(
    source=DecisionSource.SCC_VISION, v_target=18.0, a_target=-0.4, reason="vision_curve",
  )
  map_curve = make_curve_result(
    source=DecisionSource.SCC_MAP, v_target=15.0, a_target=-0.3, reason="map_curve",
  )

  selected = select_curve_speed_policy_result((vision, map_curve), driver_v_target=25.0)
  reversed_selected = select_curve_speed_policy_result((map_curve, vision), driver_v_target=25.0)

  assert selected is map_curve
  assert reversed_selected is map_curve


def test_curve_policy_arbitration_ignores_inactive_and_low_confidence_curves():
  inactive = make_curve_result(active=False, v_target=10.0)
  weak = make_curve_result(confidence=0.3, v_target=12.0)

  selected = select_curve_speed_policy_result((inactive, weak), driver_v_target=25.0)

  assert selected is None


def test_curve_policy_arbitration_does_not_select_target_above_driver_speed():
  curve = make_curve_result(v_target=30.0, a_target=-0.4)

  selected = select_curve_speed_policy_result((curve,), driver_v_target=25.0)

  assert selected is None


def test_curve_policy_arbitration_uses_existing_source_priority_for_ties():
  vision = make_curve_result(
    source=DecisionSource.SCC_VISION, v_target=18.0, a_target=-0.4, reason="vision_curve",
  )
  map_curve = make_curve_result(
    source=DecisionSource.SCC_MAP, v_target=18.0, a_target=-0.4, reason="map_curve",
  )

  selected = select_curve_speed_policy_result((map_curve, vision), driver_v_target=25.0)

  assert selected is vision
