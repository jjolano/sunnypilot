"""Plannerd wiring for the custom-2.0 longitudinal stack (opt-in).

``CustomLongitudinalAdapter`` is held by ``LongitudinalPlannerSP``; when
``CustomLongitudinalEnabled`` is set it shapes the planner's baseline ``output_a_target``
with the custom policy. Default off => stock planner behavior, so this can never change
default driving.

Evidence mapping (verified cereal fields): leads from ``radarState`` LeadData, advisory caps
from the planner's own SCC vision/map and Speed-Limit-Assist outputs, mode/personality from
params. CONSERVATIVELY DEFAULTED pending engaged-log verification (so they cannot cause
spurious braking until validated): model-stop evidence (E2E traffic-light/stop detection
from modelV2) and the coast-down accel. Wiring those two is the remaining harness-gated step
— see docs/touch-points.md.
"""
from __future__ import annotations

from typing import Any

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import CustomLongitudinalStack, LongitudinalStackInputs

PARAMS_REFRESH_PERIOD = 50  # planner ticks (~20Hz -> ~2.5s)
DEFAULT_ACCEL_LIMITS = (-4.0, 2.0)


def build_stack_inputs(*, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
                       accel_limits: tuple[float, float], lead_one: Any, lead_two: Any,
                       scc_vision_active: bool, scc_vision_a_target: float,
                       scc_map_active: bool, scc_map_a_target: float,
                       sla_active: bool, sla_v_target: float, sla_a_target: float,
                       mode: LongitudinalMode, personality: Personality, sources: SourceToggles,
                       brake_pressed: bool = False, gas_pressed: bool = False) -> LongitudinalStackInputs:
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
    accel_coast=0.0,  # conservative (flat) until grade/coast estimate is wired
    leads=(lead_one, lead_two),
    lead_a_target=lead_a_target, lead_should_stop=False,
    model_should_stop=False, model_stop_distance=None, model_desired_accel=0.0, stop_threat=False,
    speed_limit_active=bool(sla_active), speed_limit_v_target=float(sla_v_target), speed_limit_a_target=float(sla_a_target),
    curve_active=curve_active, curve_a_target=float(curve_a_target),
    brake_pressed=brake_pressed, gas_pressed=gas_pressed,
    mode=mode, sources=sources, personality=personality,
  )


class CustomLongitudinalAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._stack = CustomLongitudinalStack()
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
      self.sources = SourceToggles(
        scc_curve_vision_enabled=bool(p.get_bool("SccCurveVisionEnabled")) if _has_key(p, "SccCurveVisionEnabled") else False,
        scc_curve_map_enabled=bool(p.get_bool("SccCurveMapEnabled")) if _has_key(p, "SccCurveMapEnabled") else False,
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
      inputs = build_stack_inputs(
        v_ego=v_ego, a_ego=a_ego, v_cruise=v_cruise, seed_a_target=seed_a_target,
        accel_limits=DEFAULT_ACCEL_LIMITS,
        lead_one=getattr(radar, "leadOne", None), lead_two=getattr(radar, "leadTwo", None),
        scc_vision_active=bool(getattr(scc.vision, "is_active", False)), scc_vision_a_target=float(getattr(scc.vision, "output_a_target", 0.0)),
        scc_map_active=bool(getattr(scc.map, "is_active", False)), scc_map_a_target=float(getattr(scc.map, "output_a_target", 0.0)),
        sla_active=bool(getattr(sla, "is_active", False)), sla_v_target=float(getattr(sla, "output_v_target", 0.0)),
        sla_a_target=float(getattr(sla, "output_a_target", 0.0)),
        mode=self.mode, personality=self.personality, sources=self.sources,
        brake_pressed=bool(getattr(cs, "brakePressed", False)), gas_pressed=bool(getattr(cs, "gasPressed", False)),
      )
      result = self._stack.update(inputs, dt)
      return float(result.a_target)
    except Exception:  # fail-closed: never let the custom stack break the planner
      return seed_a_target


def _has_key(params: Any, key: str) -> bool:
  try:
    return key.encode() in params.all_keys()
  except Exception:
    return False
