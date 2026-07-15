"""Tests for the shadow-only curve-traffic advisor module and its stack/wiring integration."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.curve_traffic_advisor import (
  A_CURVE_DECEL_FLOOR,
  KAPPA_NOISE_FLOOR,
  LOW_SPEED_MIN_M_S,
  MODE_APPLY_CONSERVATIVE,
  MODE_OFF,
  MODE_SHADOW,
  CurveTrafficAdvisorInputs,
  CurveTrafficAdvisorResult,
  predict_curve_traffic_advisor,
)
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceInputs
from openpilot.sunnypilot.custom.longitudinal.stack import (
  CURVE_TRAFFIC_CORROBORATION_S,
  CustomLongitudinalStack,
  LongitudinalStackInputs,
)
from openpilot.sunnypilot.custom.longitudinal.wiring import (
  DEFAULT_ACCEL_LIMITS,
  build_stack_inputs,
)

DT = 0.05


def model_msg(xs, ys):
  return SimpleNamespace(position=SimpleNamespace(x=list(xs), y=list(ys)))


def straight_path(n=16, step=10.0):
  xs = [i * step for i in range(n)]
  return model_msg(xs, [0.0] * n)


def circular_arc_path(n=16, step=6.0, radius=100.0, sign=1.0):
  """Generate a constant-curvature arc with curvature sign*1/radius."""
  thetas = [i * step / radius for i in range(n)]
  xs = [radius * math.sin(t) for t in thetas]
  ys = [sign * radius * (1.0 - math.cos(t)) for t in thetas]
  return model_msg(xs, ys)


def s_curve_path(n=24, step=6.0, radius=100.0):
  """Two constant-curvature arcs of opposite sign separated by a short tangent."""
  half = n // 2
  first = circular_arc_path(half, step, radius, sign=1.0)
  last_x = first.position.x[-1]
  last_y = first.position.y[-1]
  heading = half * step / radius
  second = circular_arc_path(n - half, step, radius, sign=-1.0)
  xs = list(first.position.x) + [last_x + (x - second.position.x[0]) * math.cos(heading)
                                 - (y - second.position.y[0]) * math.sin(heading)
                                 for x, y in zip(second.position.x[1:], second.position.y[1:], strict=True)]
  ys = list(first.position.y) + [last_y + (x - second.position.x[0]) * math.sin(heading)
                                 + (y - second.position.y[0]) * math.cos(heading)
                                 for x, y in zip(second.position.x[1:], second.position.y[1:], strict=True)]
  return model_msg(xs, ys)


def compound_path(n=32, step=6.0):
  """Two same-sign arcs of different radii separated by a straight tangent."""
  split = n // 3
  r1 = 120.0
  first = circular_arc_path(split, step, radius=r1, sign=-1.0)
  last_x = first.position.x[-1]
  last_y = first.position.y[-1]
  # For a right-turn arc the global heading at the joint is negative.
  heading = -1.0 * split * step / r1

  # Straight tangent segment so curvature drops to near zero between bends.
  straight_steps = 4
  straight_x = [last_x + (i + 1) * step * math.cos(heading) for i in range(straight_steps)]
  straight_y = [last_y + (i + 1) * step * math.sin(heading) for i in range(straight_steps)]

  second = circular_arc_path(n - split - straight_steps, step, radius=60.0, sign=-1.0)
  sx0, sy0 = straight_x[-1], straight_y[-1]
  xs = list(first.position.x) + straight_x + [sx0 + (x - second.position.x[0]) * math.cos(heading)
                                              - (y - second.position.y[0]) * math.sin(heading)
                                              for x, y in zip(second.position.x[1:], second.position.y[1:], strict=True)]
  ys = list(first.position.y) + straight_y + [sy0 + (x - second.position.x[0]) * math.sin(heading)
                                              + (y - second.position.y[0]) * math.cos(heading)
                                              for x, y in zip(second.position.x[1:], second.position.y[1:], strict=True)]
  return model_msg(xs, ys)


def lead(d_rel=30.0, v_lead=12.0, v_rel=0.0, a_lead_k=0.0, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=a_lead_k, yRel=0.0, radarTrackId=3, radar=True,
                         modelProb=0.9, aLeadTau=1.0)


def predict(mode, *, v_ego=15.0, a_ego=0.0, model_msg=None,
            leads=(None, None), lead_shadow_active=False,
            alternate_threat_active=False, long_active=True, model_stale=False,
            brake_pressed=False, gas_pressed=False, force_slow_decel=False):
  return predict_curve_traffic_advisor(
    mode,
    CurveTrafficAdvisorInputs(
      v_ego=v_ego, a_ego=a_ego, model_msg=model_msg,
      leads=leads, lead_shadow_active=lead_shadow_active,
      alternate_threat_active=alternate_threat_active,
      long_active=long_active, model_stale=model_stale,
      brake_pressed=brake_pressed, gas_pressed=gas_pressed,
      force_slow_decel=force_slow_decel,
    ),
  )


# --- Pure helper tests ---

def test_mode_off_is_inactive():
  r = predict(MODE_OFF, model_msg=straight_path())
  assert r.mode == MODE_OFF
  assert r.effective_mode == MODE_OFF
  assert r.apply_supported is False
  assert r.active is False
  assert r.phase == "inactive"


def test_invalid_mode_sanitizes_to_off():
  r = predict("apply_aggressive", model_msg=straight_path())
  assert r.mode == MODE_OFF
  assert r.active is False


def test_apply_conservative_is_effective_apply():
  r = predict(MODE_APPLY_CONSERVATIVE, model_msg=circular_arc_path())
  assert r.mode == MODE_APPLY_CONSERVATIVE
  assert r.effective_mode == MODE_APPLY_CONSERVATIVE
  assert r.apply_supported is True


def test_invalid_path_too_few_samples():
  r = predict(MODE_SHADOW, model_msg=model_msg([0.0, 1.0], [0.0, 0.0]))
  assert r.block_reason == "invalid_path"
  assert r.active is False


def test_non_monotonic_path_rejected():
  # Enough backward steps that the "mostly increasing" check fails.
  xs = [0.0, 10.0, 5.0, 30.0, 25.0, 50.0, 45.0, 70.0]
  r = predict(MODE_SHADOW, model_msg=model_msg(xs, [0.0] * len(xs)))
  assert r.block_reason == "invalid_path"


def test_straight_path_is_below_noise_floor():
  r = predict(MODE_SHADOW, model_msg=straight_path())
  assert r.active is False
  assert r.block_reason == "below_noise_floor"
  assert r.curvature_peak < KAPPA_NOISE_FLOOR


def test_left_arc_positive_curvature_and_active():
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=80.0), v_ego=15.0)
  assert r.active is True
  assert r.eligible is True
  assert r.curvature_sign > 0.0
  assert r.curvature_peak > KAPPA_NOISE_FLOOR
  assert r.phase in ("entry", "apex", "pre_entry")
  assert r.v_curve_cap_proposed >= LOW_SPEED_MIN_M_S
  assert r.a_curve_cap_proposed <= 0.0
  assert r.a_curve_cap_proposed >= A_CURVE_DECEL_FLOOR


def test_right_arc_negative_curvature():
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=80.0, sign=-1.0), v_ego=15.0)
  assert r.curvature_sign < 0.0
  assert r.active is True


def test_low_speed_blocks():
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=80.0), v_ego=2.0)
  assert r.block_reason == "low_speed"


@pytest.mark.parametrize(
  ("flag", "reason"),
  (("model_stale", "model_stale"), ("brake_pressed", "driver_override"),
   ("gas_pressed", "driver_override"), ("force_slow_decel", "force_slow")),
)
def test_safety_gates_block_geometry(flag, reason):
  kwargs = {flag: True}
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=80.0), **kwargs)
  assert r.active is False
  assert r.block_reason == reason


def test_s_curve_detected():
  r = predict(MODE_SHADOW, model_msg=s_curve_path())
  assert r.s_curve is True
  assert r.active is True


def test_compound_curve_detected():
  r = predict(MODE_SHADOW, model_msg=compound_path())
  assert r.compound_curve is True
  assert r.active is True


def test_traffic_close_closing_lead_sets_suppress():
  r = predict(MODE_SHADOW, model_msg=straight_path(),
              leads=(lead(d_rel=12.0, v_rel=-3.0), None))
  assert r.suppress_accel is True
  assert "close_closing_lead" in r.traffic_block_reason


def test_traffic_braking_lead_sets_suppress():
  r = predict(MODE_SHADOW, model_msg=straight_path(),
              leads=(lead(d_rel=40.0, v_rel=-1.0, a_lead_k=-2.0), None))
  assert r.suppress_accel is True
  assert "braking_lead" in r.traffic_block_reason


def test_shadow_lead_suppresses():
  r = predict(MODE_SHADOW, model_msg=straight_path(), lead_shadow_active=True)
  assert r.suppress_accel is True
  assert "shadow_lead" in r.traffic_block_reason


def test_alternate_threat_suppresses():
  r = predict(MODE_SHADOW, model_msg=straight_path(), alternate_threat_active=True)
  assert r.suppress_accel is True
  assert "alternate_threat" in r.traffic_block_reason


def test_accel_cap_is_negative_only():
  v_ego = 5.0
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=50.0), v_ego=v_ego)
  assert r.v_curve_cap_proposed >= LOW_SPEED_MIN_M_S
  # If v_ego is below the cap, no decel is proposed.
  if r.v_curve_cap_proposed > v_ego:
    assert r.a_curve_cap_proposed == 0.0
  else:
    assert r.a_curve_cap_proposed <= 0.0
    assert r.a_curve_cap_proposed >= A_CURVE_DECEL_FLOOR


def test_debug_dict_uses_prefix():
  r = predict(MODE_SHADOW, model_msg=circular_arc_path(radius=100.0))
  d = r.debug_dict()
  assert all(k.startswith("curve_traffic_") for k in d)
  assert d["curve_traffic_mode"] == MODE_SHADOW


def test_result_default_has_fault_false():
  r = CurveTrafficAdvisorResult()
  assert r.fault is False


# --- Stack integration tests ---

def test_curve_traffic_advisor_mode_does_not_change_actuation():
  def make_inp(mode):
    return LongitudinalStackInputs(
      v_ego=20.0, v_cruise=22.0, seed_a_target=0.4,
      curve_traffic_advisor_mode=mode,
      model_msg=circular_arc_path(n=24, radius=100.0),
      mode=LongitudinalMode.SCC, long_active=True,
    )
  off = CustomLongitudinalStack().update(make_inp(MODE_OFF), DT)
  shadow = CustomLongitudinalStack().update(make_inp(MODE_SHADOW), DT)

  assert shadow.a_target == pytest.approx(off.a_target)
  assert shadow.should_stop == off.should_stop
  assert shadow.decision.reason == off.decision.reason
  assert shadow.decision.selected_intent == off.decision.selected_intent
  assert shadow.standstill_release_allowed == off.standstill_release_allowed

  assert off.debug["curve_traffic_mode"] == MODE_OFF
  assert off.debug["curve_traffic_active"] is False
  assert shadow.debug["curve_traffic_mode"] == MODE_SHADOW
  assert shadow.debug["curve_traffic_active"] is True
  assert shadow.debug["curve_traffic_advisor_fault"] is False


def test_curve_traffic_apply_waits_for_persistent_vision_path_agreement():
  inp = LongitudinalStackInputs(
    v_ego=20.0, v_cruise=22.0, seed_a_target=0.4,
    curve_traffic_advisor_mode=MODE_APPLY_CONSERVATIVE,
    curve_confidence=CurveSpeedConfidenceInputs(vision_active=True, vision_a_target=-0.5),
    model_msg=circular_arc_path(n=24, radius=100.0),
    mode=LongitudinalMode.SCC, long_active=True,
    sources=SourceToggles(scc_curve_vision_enabled=True),
    research_actuation_allowed=True,
  )
  stack = CustomLongitudinalStack()
  result = None
  for _ in range(round(CURVE_TRAFFIC_CORROBORATION_S / DT) - 1):
    result = stack.update(inp, DT)
  assert result is not None
  assert result.actuation.curve_traffic_advisor is not None
  assert result.actuation.curve_traffic_advisor.eligible is False
  assert "corroboration_pending" in result.actuation.curve_traffic_advisor.block_reason

  result = stack.update(inp, DT)
  assert result.actuation.curve_traffic_advisor is not None
  assert result.actuation.curve_traffic_advisor.eligible is True

  result = stack.update(
    replace(inp, curve_confidence=CurveSpeedConfidenceInputs(vision_active=False, vision_a_target=-0.5)),
    DT,
  )
  assert result.actuation.curve_traffic_advisor is not None
  assert result.actuation.curve_traffic_advisor.eligible is False


def test_curve_traffic_advisor_fault_does_not_leak():
  class BadMode:
    def __str__(self):
      raise RuntimeError("bad mode")

  def make_inp(mode):
    return LongitudinalStackInputs(
      v_ego=15.0, v_cruise=18.0, seed_a_target=0.0,
      curve_traffic_advisor_mode=mode,
      mode=LongitudinalMode.ACC,
    )

  baseline = CustomLongitudinalStack().update(make_inp(MODE_OFF), DT)
  broken = CustomLongitudinalStack().update(make_inp(BadMode()), DT)
  assert broken.a_target == pytest.approx(baseline.a_target)
  assert broken.debug["curve_traffic_advisor_fault"] is True


# --- Wiring integration tests ---

def test_build_stack_inputs_carries_curve_traffic_advisor_mode():
  inp = build_stack_inputs(
    v_ego=12.0, a_ego=0.0, v_cruise=15.0, seed_a_target=0.2,
    accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=None, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0,
    scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.ACC, personality=Personality.STANDARD,
    sources=None,  # type: ignore[arg-type]
    curve_traffic_advisor_mode=MODE_SHADOW,
  )
  assert inp.curve_traffic_advisor_mode == MODE_SHADOW
