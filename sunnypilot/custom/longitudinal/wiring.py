"""Plannerd wiring for the custom-2.0 longitudinal stack (opt-in).

``CustomLongitudinalAdapter`` is held by ``LongitudinalPlannerSP``; when
``CustomLongitudinalEnabled`` is set it shapes the planner's baseline ``output_a_target``
with the custom policy and returns the policy-owned stop commitment. The shaper is
default-on in this fork, but fail-closed to the stock planner output on faults.

Evidence mapping — all from verified upstream signals (no guessing): leads from
``radarState`` LeadData; advisory caps from the planner's own SCC vision/map and
Speed-Limit-Assist outputs; the model stop from ``modelV2.action.shouldStop`` /
``desiredAcceleration`` plus the predicted stop distance derived from the model trajectory
(``position.x``/``velocity.x``, the same signal the base planner uses); the coast-down accel from
``get_coast_accel(pitch)``; mode/personality from params. The model stop is then trust-gated
by ``StopTrustLearner``, which learns how much to trust it from real driver disagreement
(gas during a model stop = the driver countermanding it) rather than a guessed probability.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import CurveSpeedConfidenceInputs
from openpilot.sunnypilot.custom.longitudinal.curve_traffic_advisor import (
  MODE_APPLY_CONSERVATIVE as CURVE_TRAFFIC_MODE_APPLY_CONSERVATIVE,
  MODE_OFF as CURVE_TRAFFIC_MODE_OFF,
  MODE_SHADOW as CURVE_TRAFFIC_MODE_SHADOW,
)
from openpilot.sunnypilot.custom.longitudinal.model_trust import StopTrustLearner
from openpilot.sunnypilot.custom.longitudinal.modes import EvidenceClass, LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import CustomLongitudinalStack, LongitudinalStackInputs

PARAMS_REFRESH_PERIOD = 50  # planner ticks (~20Hz -> ~2.5s)
DEFAULT_ACCEL_LIMITS = (-4.0, 2.0)
MODEL_STOP_SPEED = 0.3  # m/s; predicted speed at/below this marks the trajectory's rest point
MODEL_STALE_AGE_S = 0.20


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _param_string(params: Any, key: str) -> str | None:
  try:
    raw = params.get(key)
  except TypeError:
    raw = params.get(key, None)
  if raw is None:
    return None
  if isinstance(raw, bytes):
    raw = raw.decode(errors="ignore")
  return str(raw)


def _message_age_s(sm: Any, service: str) -> float:
  recv_time = getattr(sm, "recv_time", None)
  if not isinstance(recv_time, dict) or service not in recv_time:
    return float("inf")
  try:
    received_at = float(recv_time[service])
  except (TypeError, ValueError):
    return float("inf")
  if not math.isfinite(received_at) or received_at <= 0.0:
    return float("inf")
  return max(0.0, time.monotonic() - received_at)


def _shadow_mode(value: Any, future_values: tuple[str, ...] = ()) -> str:
  text = str(value or "").strip().lower()
  if text in ("off", "shadow"):
    return text
  if text in future_values:
    return "shadow"
  return "off"


def _curve_speed_confidence_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  return text if text in ("off", "shadow", "apply_conservative") else "off"


def _cut_in_brake_assist_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  if text in ("off", "shadow", "apply"):
    return text
  return "off"


def _curve_traffic_advisor_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  if text in (CURVE_TRAFFIC_MODE_OFF, CURVE_TRAFFIC_MODE_SHADOW, CURVE_TRAFFIC_MODE_APPLY_CONSERVATIVE):
    return text
  return CURVE_TRAFFIC_MODE_OFF


def _standstill_release_confidence_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  return text if text in ("off", "shadow", "gate") else "off"


def _debug_trace_mode(value: Any) -> str:
  text = str(value or "").strip().lower()
  return text if text in ("off", "log") else "off"


def _coast_accel(pitch: float) -> float:
  # Inlined from longitudinal_planner.get_coast_accel (importing it would be circular).
  return math.sin(_f(pitch)) * -5.65 - 0.3


def _model_stop_distance(model: Any) -> float | None:
  """Forward distance to where the model's predicted trajectory comes to rest, or None.

  Reads ``modelV2.position.x`` / ``velocity.x`` (the same trajectory the base planner parses)
  and returns the position at the first horizon index where predicted speed drops to ~0.
  Returns None when the model isn't predicting a stop within the horizon (cruise keeps speed
  up the whole horizon), so the distance-aware stop-approach path stays inert until a real
  stop is predicted. The policy still trust-gates whether to commit to the stop."""
  position = getattr(model, "position", None)
  velocity = getattr(model, "velocity", None)
  xs = getattr(position, "x", None) if position is not None else None
  vs = getattr(velocity, "x", None) if velocity is not None else None
  if not xs or not vs or len(xs) != len(vs):
    return None
  for x, v in zip(xs, vs, strict=True):
    if _f(v, default=1.0) <= MODEL_STOP_SPEED:
      d = _f(x)
      return d if d > 0.0 else None
  return None


def build_stack_inputs(*, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
                       accel_limits: tuple[float, float], lead_one: Any, lead_two: Any,
                       scc_vision_active: bool, scc_vision_a_target: float,
                       scc_map_active: bool, scc_map_a_target: float,
                       sla_active: bool, sla_v_target: float, sla_a_target: float,
                       mode: LongitudinalMode, personality: Personality, sources: SourceToggles,
                       long_active: bool = False, brake_pressed: bool = False, gas_pressed: bool = False, force_slow_decel: bool = False,
                       model_should_stop: bool = False, model_desired_accel: float = 0.0,
                       model_stop_prob: float = 1.0, model_stop_distance: float | None = None,
                       model_stale: bool = False,
                       accel_coast: float = 0.0, model_msg: Any | None = None,
                       cut_in_brake_assist_mode: str = "off",
                       curve_speed_confidence_mode: str = "off",
                       curve_traffic_advisor_mode: str = CURVE_TRAFFIC_MODE_OFF,
                       standstill_release_confidence_mode: str = "off",
                       scenario_context_mode: str = "off",
                       standstill: bool = False,
                       steering_angle_deg: float = 0.0,
                       steering_torque: float = 0.0,
                       scc_vision_state: Any = None,
                       scc_vision_current_lat_acc: float = 0.0,
                       scc_vision_max_pred_lat_acc: float = 0.0,
                       scc_vision_pre_entry_active: bool = False,
                       scc_map_state: Any = None,
                       scc_map_target_lat: float = 0.0,
                       scc_map_target_lon: float = 0.0) -> LongitudinalStackInputs:
  has_lead = lead_one is not None and bool(getattr(lead_one, "status", False))
  # Pre-MPC lead-present seed: carry the currently selected planner a_target into the custom
  # policy. Final lead-follow physics remains owned by the downstream MPC solve.
  lead_a_target = float(seed_a_target) if has_lead else 0.0
  # Actuator curve cap is built from SCC-Vision only. SCC-Map evidence is kept for telemetry and
  # curve-speed confidence, but must not directly reduce planner speed/accel until a bounded
  # apply tier is validated.
  curve_active = bool(scc_vision_active)
  v_curve = scc_vision_a_target if scc_vision_active else float("inf")
  curve_a_target = v_curve if curve_active else 0.0
  curve_source = EvidenceClass.CURVE_VISION
  return LongitudinalStackInputs(
    v_ego=v_ego, a_ego=float(a_ego), v_cruise=v_cruise, seed_a_target=seed_a_target,
    accel_limits=accel_limits, accel_coast=float(accel_coast),
    leads=(lead_one, lead_two),
    # lead_should_stop is intentionally inert: MPC owns lead-follow stop physics and the base
    # planner carries the MPC stop bit into final actuation separately. The custom stack's
    # should_stop is reserved for model-stop commitment under the active mode gate.
    lead_a_target=lead_a_target, lead_should_stop=False,
    model_should_stop=bool(model_should_stop),
    model_stop_distance=(float(model_stop_distance) if model_stop_distance is not None else None),
    # stop_threat is intentionally inert: every policy consumer (coast/launch/comfort-relax)
    # is already gated by has_lead, and the only lead-derived stop-threat signal is itself
    # lead-coupled, making it fully redundant with has_lead (zero observable effect).
    model_desired_accel=float(model_desired_accel), model_stop_prob=float(model_stop_prob),
    model_stale=bool(model_stale), stop_threat=False,
    speed_limit_active=bool(sla_active), speed_limit_v_target=float(sla_v_target), speed_limit_a_target=float(sla_a_target),
    curve_active=curve_active, curve_a_target=float(curve_a_target), curve_source=curve_source,
    long_active=bool(long_active), force_slow_decel=bool(force_slow_decel),
    brake_pressed=brake_pressed, gas_pressed=gas_pressed,
    mode=mode, sources=sources, personality=personality, model_msg=model_msg,
    cut_in_brake_assist_mode=cut_in_brake_assist_mode,
    curve_speed_confidence_mode=curve_speed_confidence_mode,
    curve_traffic_advisor_mode=curve_traffic_advisor_mode,
    standstill_release_confidence_mode=standstill_release_confidence_mode,
    scenario_context_mode=scenario_context_mode,
    standstill=bool(standstill),
    steering_angle_deg=_f(steering_angle_deg),
    steering_torque=_f(steering_torque),
    curve_confidence=CurveSpeedConfidenceInputs(
      vision_active=bool(scc_vision_active), vision_a_target=_f(scc_vision_a_target),
      vision_state=scc_vision_state, vision_current_lat_acc=_f(scc_vision_current_lat_acc),
      vision_max_pred_lat_acc=_f(scc_vision_max_pred_lat_acc),
      vision_pre_entry_active=bool(scc_vision_pre_entry_active),
      # Map may boost confidence/context, but cannot provide a braking cap in the evidence-only tier.
      map_active=bool(scc_map_active), map_a_target=0.0, map_state=scc_map_state,
      map_target_lat=_f(scc_map_target_lat), map_target_lon=_f(scc_map_target_lon),
    ),
  )


class CustomLongitudinalAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._stack = CustomLongitudinalStack()
    self._stop_trust = StopTrustLearner()
    self._tick = 0
    self.enabled = False
    self.mode = LongitudinalMode.SCC
    self.debug_trace_mode = "off"
    self.cut_in_brake_assist_mode = "off"
    self.curve_speed_confidence_mode = "off"
    self.curve_traffic_advisor_mode = CURVE_TRAFFIC_MODE_OFF
    self.standstill_release_confidence_mode = "off"
    self.scenario_context_mode = "off"
    self.personality = Personality.STANDARD
    self.sources = SourceToggles()
    if params is not None:
      self.refresh_params(initial=True)

  def refresh_params(self, initial: bool = False, mode_only: bool = False) -> None:
    p = self._params
    if p is None:
      return
    try:
      enabled = bool(p.get_bool("CustomLongitudinalEnabled"))
      # SCC is the default: the custom-2.0 intelligent ACC/E2E blend (the DEC replacement). acc/e2e
      # force OEM-like cruise or the model's stops respectively. Mode is refreshed live each cycle.
      mode = LongitudinalMode.from_value(p.get("CustomLongitudinalMode") or "scc", default=LongitudinalMode.SCC)
    except Exception:  # params are advisory; never fault the planner on a failed read
      if initial:
        self.enabled = False
      return

    prev_mode = self.mode
    was_enabled = self.enabled
    self.enabled = enabled
    self.mode = mode
    if mode is not prev_mode or (enabled and not was_enabled):
      self._stack.reset()

    if mode_only:
      return

    # Slower refresh for tuning/advisory params.
    if initial or self._tick % PARAMS_REFRESH_PERIOD == 0:
      # Telemetry-only scenario-context mode is read safely on its own so an unregistered
      # or malformed key cannot block the source-toggle refresh below.
      try:
        scenario_context_value = _param_string(p, "ScenarioContextMode")
      except Exception:
        scenario_context_value = None
      self.scenario_context_mode = _shadow_mode(scenario_context_value or "off")

      # Shadow-only advisory mode is isolated so stale/unregistered params cannot block
      # SCC source-toggle refresh or any existing longitudinal behavior.
      try:
        curve_traffic_advisor_value = _param_string(p, "CurveTrafficAdvisorMode")
      except Exception:
        curve_traffic_advisor_value = None
      self.curve_traffic_advisor_mode = _curve_traffic_advisor_mode(curve_traffic_advisor_value)

      try:
        self.personality = Personality.from_value(p.get("LongitudinalPersonality"))
        self.debug_trace_mode = _debug_trace_mode(_param_string(p, "LongitudinalDebugTraceMode"))
        self.cut_in_brake_assist_mode = _cut_in_brake_assist_mode(_param_string(p, "CutInBrakeAssistMode"))
        self.curve_speed_confidence_mode = _curve_speed_confidence_mode(_param_string(p, "CurveSpeedConfidenceMode"))
        self.standstill_release_confidence_mode = _standstill_release_confidence_mode(_param_string(p, "StandstillReleaseConfidenceMode"))
        # SCC curve sources are gated by the existing upstream SCC enable toggles.
        self.sources = SourceToggles(
          scc_curve_vision_enabled=bool(p.get_bool("SmartCruiseControlVision")),
          scc_curve_map_enabled=bool(p.get_bool("SmartCruiseControlMap")),
        )
      except Exception:
        pass

  def maybe_refresh_params(self) -> None:
    self._tick += 1
    if self._params is not None:
      self.refresh_params(initial=False)

  def evaluate(self, sm: Any, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
               scc: Any, sla: Any, dt: float = 0.05) -> CustomLongitudinalOutput:
    if not self.enabled:
      return CustomLongitudinalOutput(
        a_target=seed_a_target, should_stop=False, enabled=False, mode=self.mode,
        selected_intent="disabled", reason="disabled", debug={"seed_a_target": seed_a_target},
      )
    try:
      radar = sm['radarState']
      cs = sm.get('carState') if hasattr(sm, "get") else sm['carState']
      controls_state = sm.get('controlsState') if hasattr(sm, "get") else sm['controlsState']
      gas_pressed = bool(getattr(cs, "gasPressed", False))

      model = sm['modelV2']
      model_age_s = _message_age_s(sm, 'modelV2')
      model_stale = model_age_s > MODEL_STALE_AGE_S
      action = getattr(model, "action", None)
      model_should_stop = bool(getattr(action, "shouldStop", False)) if action is not None else False
      model_desired_accel = _f(getattr(action, "desiredAcceleration", 0.0)) if action is not None else 0.0
      model_stop_prob = self._stop_trust.update(model_should_stop, driver_disagrees=gas_pressed, dt=dt)
      model_stop_distance = _model_stop_distance(model)

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
        long_active=bool(getattr(cc, "longActive", False)),
        brake_pressed=bool(getattr(cs, "brakePressed", False)), gas_pressed=gas_pressed,
        force_slow_decel=bool(getattr(controls_state, "forceDecel", False)),
        model_should_stop=model_should_stop, model_desired_accel=model_desired_accel,
        model_stop_prob=model_stop_prob, model_stop_distance=model_stop_distance,
        model_stale=model_stale, accel_coast=accel_coast,
        model_msg=model,
        cut_in_brake_assist_mode=self.cut_in_brake_assist_mode,
        curve_speed_confidence_mode=self.curve_speed_confidence_mode,
        curve_traffic_advisor_mode=self.curve_traffic_advisor_mode,
        standstill_release_confidence_mode=self.standstill_release_confidence_mode,
        scenario_context_mode=self.scenario_context_mode,
        standstill=bool(getattr(cs, "standstill", False)),
        steering_angle_deg=_f(getattr(cs, "steeringAngleDeg", 0.0)),
        steering_torque=_f(getattr(cs, "steeringTorque", 0.0)),
        scc_vision_state=getattr(scc.vision, "state", None),
        scc_vision_current_lat_acc=_f(getattr(scc.vision, "current_lat_acc", 0.0)),
        scc_vision_max_pred_lat_acc=_f(getattr(scc.vision, "max_pred_lat_acc", 0.0)),
        scc_vision_pre_entry_active=bool(getattr(scc.vision, "pre_entry_active", False)),
        scc_map_state=getattr(scc.map, "state", None),
        scc_map_target_lat=_f(getattr(scc.map, "target_lat", 0.0)),
        scc_map_target_lon=_f(getattr(scc.map, "target_lon", 0.0)),
      )
      result = self._stack.update(inputs, dt)
      debug = dict(result.debug or {})
      return CustomLongitudinalOutput(
        a_target=float(result.a_target), should_stop=bool(result.should_stop), enabled=True, mode=self.mode,
        selected_intent=debug.get("intent"), reason=debug.get("reason"),
        standstill_release_allowed=bool(result.standstill_release_allowed),
        standstill_release_source=str(result.standstill_release_source),
        standstill_release_a_target=float(result.standstill_release_a_target),
        standstill_release_reason=str(result.standstill_release_reason),
        debug=debug,
      )
    except Exception:
      return CustomLongitudinalOutput(
        a_target=seed_a_target, should_stop=False, enabled=False, mode=self.mode,
        selected_intent="fault", reason="fault", debug={},
      )

  def apply(self, sm: Any, v_ego: float, a_ego: float, v_cruise: float, seed_a_target: float,
            scc: Any, sla: Any, dt: float = 0.05) -> float:
    """Return the shaped a_target, or the unchanged seed when disabled or on any fault."""
    self.maybe_refresh_params()
    return self.evaluate(sm, v_ego, a_ego, v_cruise, seed_a_target, scc, sla, dt).a_target


@dataclass(frozen=True)
class CustomLongitudinalOutput:
  a_target: float
  should_stop: bool
  enabled: bool
  mode: LongitudinalMode
  selected_intent: object | None
  reason: object | None
  standstill_release_allowed: bool = False
  standstill_release_source: str = ""
  standstill_release_a_target: float = 0.0
  standstill_release_reason: str = ""
  debug: dict[str, Any] = field(default_factory=dict)
