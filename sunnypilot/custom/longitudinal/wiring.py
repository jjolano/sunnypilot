"""Plannerd wiring for the custom-2.0 longitudinal stack (opt-in).

``CustomLongitudinalAdapter`` is held by ``LongitudinalPlannerSP``; when
``CustomLongitudinalEnabled`` is set it shapes the planner's baseline ``output_a_target``
with the custom policy. Default off => stock planner behavior, so this can never change
default driving.

Evidence mapping — all from verified upstream signals (no guessing): leads from
``radarState`` LeadData; advisory caps from the planner's own SCC vision/map and
Speed-Limit-Assist outputs; the model stop from ``modelV2.action.shouldStop`` /
``desiredAcceleration`` (the same signal the base planner uses); the coast-down accel from
``get_coast_accel(pitch)``; mode/personality from params. The model stop is then trust-gated
by ``StopTrustLearner``, which learns how much to trust it from real driver disagreement
(gas during a model stop = the driver countermanding it) rather than a guessed probability.
"""
from __future__ import annotations

import math
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.model_trust import StopTrustLearner
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import CustomLongitudinalStack, LongitudinalStackInputs

PARAMS_REFRESH_PERIOD = 50  # planner ticks (~20Hz -> ~2.5s)
DEFAULT_ACCEL_LIMITS = (-4.0, 2.0)


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _coast_accel(pitch: float) -> float:
  # Inlined from longitudinal_planner.get_coast_accel (importing it would be circular).
  return math.sin(_f(pitch)) * -5.65 - 0.3


def build_stack_inputs(*, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
                       accel_limits: tuple[float, float], lead_one: Any, lead_two: Any,
                       scc_vision_active: bool, scc_vision_a_target: float,
                       scc_map_active: bool, scc_map_a_target: float,
                       sla_active: bool, sla_v_target: float, sla_a_target: float,
                       mode: LongitudinalMode, personality: Personality, sources: SourceToggles,
                       brake_pressed: bool = False, gas_pressed: bool = False,
                       model_should_stop: bool = False, model_desired_accel: float = 0.0,
                       model_stop_prob: float = 1.0, accel_coast: float = 0.0) -> LongitudinalStackInputs:
  has_lead = lead_one is not None and bool(getattr(lead_one, "status", False))
  # MPC owns lead-follow physics; carry the planner's a_target as the lead-follow accel.
  lead_a_target = float(seed_a_target) if has_lead else 0.0
  # SCC vision/map are curve-speed sources -> the curve advisory cap (most restrictive wins).
  curve_active = bool(scc_vision_active or scc_map_active)
  curve_a_target = min(
    scc_vision_a_target if scc_vision_active else 0.0,
    scc_map_a_target if scc_map_active else 0.0,
  ) if curve_active else 0.0
  return LongitudinalStackInputs(
    v_ego=v_ego, v_cruise=v_cruise, seed_a_target=seed_a_target, accel_limits=accel_limits,
    accel_coast=float(accel_coast),
    leads=(lead_one, lead_two),
    lead_a_target=lead_a_target, lead_should_stop=False,
    model_should_stop=bool(model_should_stop), model_stop_distance=None,
    model_desired_accel=float(model_desired_accel), model_stop_prob=float(model_stop_prob), stop_threat=False,
    speed_limit_active=bool(sla_active), speed_limit_v_target=float(sla_v_target), speed_limit_a_target=float(sla_a_target),
    curve_active=curve_active, curve_a_target=float(curve_a_target),
    brake_pressed=brake_pressed, gas_pressed=gas_pressed,
    mode=mode, sources=sources, personality=personality,
  )


class CustomLongitudinalAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._stack = CustomLongitudinalStack()
    self._stop_trust = StopTrustLearner()
    self._tick = 0
    self.enabled = False
    self.mode = LongitudinalMode.ACC
    self.personality = Personality.STANDARD
    self.sources = SourceToggles()
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    p = self._params
    if p is None:
      return
    try:
      self.enabled = bool(p.get_bool("CustomLongitudinalEnabled"))
      self.mode = LongitudinalMode.from_value(p.get("CustomLongitudinalMode"))
      self.personality = Personality.from_value(p.get("LongitudinalPersonality"))
      # SCC curve sources are gated by the existing upstream SCC enable toggles.
      self.sources = SourceToggles(
        scc_curve_vision_enabled=bool(p.get_bool("SmartCruiseControlVision")),
        scc_curve_map_enabled=bool(p.get_bool("SmartCruiseControlMap")),
      )
    except Exception:  # params are advisory; never fault the planner on a read error
      self.enabled = False

  def apply(self, sm: Any, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
            scc: Any, sla: Any, dt: float = 0.05) -> float:
    """Return the shaped a_target, or the unchanged seed when disabled or on any fault."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()
    if not self.enabled:
      return seed_a_target
    try:
      radar = sm['radarState']
      cs = sm['carState']
      gas_pressed = bool(getattr(cs, "gasPressed", False))

      # Upstream's verified model stop (the same signal the base planner uses), with trust
      # learned from driver disagreement: gas during a model stop = the driver countermanding it.
      action = getattr(sm['modelV2'], "action", None)
      model_should_stop = bool(getattr(action, "shouldStop", False)) if action is not None else False
      model_desired_accel = _f(getattr(action, "desiredAcceleration", 0.0)) if action is not None else 0.0
      model_stop_prob = self._stop_trust.update(model_should_stop, driver_disagrees=gas_pressed, dt=dt)

      cc = sm['carControl']
      ned = getattr(cc, "orientationNED", None)
      accel_coast = _coast_accel(ned[1]) if ned is not None and len(ned) == 3 else 0.0

      inputs = build_stack_inputs(
        v_ego=v_ego, a_ego=a_ego, v_cruise=v_cruise, seed_a_target=seed_a_target,
        accel_limits=DEFAULT_ACCEL_LIMITS,
        lead_one=getattr(radar, "leadOne", None), lead_two=getattr(radar, "leadTwo", None),
        scc_vision_active=bool(getattr(scc.vision, "is_active", False)), scc_vision_a_target=float(getattr(scc.vision, "output_a_target", 0.0)),
        scc_map_active=bool(getattr(scc.map, "is_active", False)), scc_map_a_target=float(getattr(scc.map, "output_a_target", 0.0)),
        sla_active=bool(getattr(sla, "is_active", False)), sla_v_target=float(getattr(sla, "output_v_target", 0.0)),
        sla_a_target=float(getattr(sla, "output_a_target", 0.0)),
        mode=self.mode, personality=self.personality, sources=self.sources,
        brake_pressed=bool(getattr(cs, "brakePressed", False)), gas_pressed=gas_pressed,
        model_should_stop=model_should_stop, model_desired_accel=model_desired_accel,
        model_stop_prob=model_stop_prob, accel_coast=accel_coast,
      )
      result = self._stack.update(inputs, dt)
      return float(result.a_target)
    except Exception:  # fail-closed: never let the custom stack break the planner
      return seed_a_target
