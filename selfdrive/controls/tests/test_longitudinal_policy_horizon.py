import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import DecisionSource
from openpilot.selfdrive.controls.lib.longitudinal_policy_horizon import (
  AdvisoryConstraint,
  CurveSpeedPolicyResult,
  LongitudinalPolicyHorizon,
  advisory_constraint_for_speed_drop,
  advisory_constraints_allowed_for_mode,
  build_longitudinal_policy_horizon,
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
  assert candidate.debug["horizon_distance"] == pytest.approx(40.0)
  assert candidate.debug["horizon_time"] == pytest.approx(2.0)
  assert candidate.debug["required_a_target"] == pytest.approx(-0.4)


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


def test_speed_drop_constraint_shorter_distance_never_weaker_decel():
  near = advisory_constraint_for_speed_drop("speed_limit", current_speed=25.0, target_speed=15.0, distance=80.0)
  far = advisory_constraint_for_speed_drop("speed_limit", current_speed=25.0, target_speed=15.0, distance=160.0)

  assert near.target_accel is not None
  assert far.target_accel is not None
  assert near.target_accel < far.target_accel < 0.0


def test_physical_hazard_suppresses_advisory_horizon_source():
  constraint = advisory_constraint_for_speed_drop("speed_limit", current_speed=25.0, target_speed=15.0, distance=80.0)

  horizon = build_longitudinal_policy_horizon((constraint,), physical_hazard_active=True, horizon_len=3)

  assert horizon.source_by_t == ("physical_hazard", "physical_hazard", "physical_hazard")
  assert horizon.confidence_by_t == (1.0, 1.0, 1.0)


def test_invalid_horizon_constraint_is_ignored():
  invalid = AdvisoryConstraint(
    source="speed_limit",
    target_speed=15.0,
    target_distance=float("nan"),
    target_accel=-0.3,
    confidence=1.0,
    urgency=0.5,
  )

  horizon = build_longitudinal_policy_horizon((invalid,), horizon_len=2, default_v_upper=30.0, default_a_max=1.0)

  assert horizon.v_upper == (30.0, 30.0)
  assert horizon.a_max == (1.0, 1.0)
  assert horizon.source_by_t == ("", "")


def test_acc_ignores_actuation_advisory_constraints():
  assert not advisory_constraints_allowed_for_mode("ACC")
  assert advisory_constraints_allowed_for_mode("SCC")
  assert advisory_constraints_allowed_for_mode("E2E")


def test_e2e_and_scc_horizon_constraints_remain_restrictive_only():
  constraint = advisory_constraint_for_speed_drop("curve", current_speed=25.0, target_speed=15.0, distance=100.0)
  informational = AdvisoryConstraint("debug", None, None, 0.5, 1.0, 0.0, authority="informational")

  horizon = build_longitudinal_policy_horizon((constraint, informational), horizon_len=2, default_a_max=1.0)

  assert horizon.a_max[0] < 0.0
  assert horizon.source_by_t == ("curve", "curve")
