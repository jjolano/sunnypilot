#!/usr/bin/env python3
"""Stateful structural fuzzer for longitudinal trust / confidence modules.

Phase 3B target modules (no product-behavior changes, no real Params, no MPC/acados,
no route parity / comfort tuning):

  - sunnypilot.custom.longitudinal.lead_confidence
  - sunnypilot.custom.longitudinal.model_trust
"""
from __future__ import annotations

import argparse
import json
import math
import random
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_confidence import (
  LEAD_FLICKER_CLOSE_GUARD_TIME,
  LeadConfidenceState,
  LeadConfidenceTracker,
  NEW_LEAD_GUARD_TIME,
  adjust_new_lead_accel,
)
from openpilot.sunnypilot.custom.longitudinal.model_trust import (
  GENTLE_CAUTION_DECEL,
  STOP_TRUST_MAX,
  STOP_TRUST_MIN,
  StopTrustLearner,
  gate_model_stop,
)

EPS = 1e-6

DEFAULT_KINDS = (
  "lead_confidence_sequence",
  "model_trust_sequence",
)


@dataclass(frozen=True)
class StatefulFuzzerConfig:
  """Top-level fuzzer configuration."""

  seed: int = 1
  cases: int = 100
  kind: str | None = None


@dataclass(frozen=True)
class StatefulCase:
  """One generated stateful structural case."""

  kind: str
  index: int
  title: str
  params: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "index": self.index,
      "title": self.title,
      "params": _sanitize(self.params),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> StatefulCase:
    return cls(
      kind=str(data["kind"]),
      index=int(data["index"]),
      title=str(data["title"]),
      params=dict(data.get("params", {})),
    )


@dataclass(frozen=True)
class StatefulResult:
  """Result of evaluating one stateful case."""

  case: StatefulCase
  failures: list[dict[str, Any]]
  metrics: dict[str, Any]

  @property
  def valid(self) -> bool:
    return not self.failures

  @property
  def failure_count(self) -> int:
    return len(self.failures)


# ---------- helpers ----------


def _sanitize(value: Any) -> Any:
  """Make values JSON-safe; non-finite floats become null."""
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {k: _sanitize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_sanitize(v) for v in value]
  if isinstance(value, bool):
    return bool(value)
  return value


def _to_float(value: Any) -> float:
  """Convert a scalar to float; string sentinels 'nan'/'inf'/'-inf' are honored."""
  if isinstance(value, str):
    lowered = value.lower()
    if lowered == "nan":
      return float("nan")
    if lowered in ("inf", "+inf"):
      return float("inf")
    if lowered == "-inf":
      return float("-inf")
  return float(value)


def _to_float_or_passthrough(value: Any) -> Any:
  """Convert numeric-looking values via _to_float, leave booleans/payloads alone."""
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    return _to_float(value)
  return value


def _finite_default(value: Any, default: float = 0.0) -> float:
  """Mirror the modules' _finite_float helpers."""
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _simple_namespace(d: dict[str, Any] | None) -> SimpleNamespace | None:
  if d is None:
    return None
  return SimpleNamespace(**{k: _to_float_or_passthrough(v) for k, v in d.items()})


# ---------- lead-confidence helpers ----------


def _random_lead_dict(rng: random.Random, tid: int | None = None, **overrides: Any) -> dict[str, Any]:
  d = {
    "status": True,
    "dRel": rng.uniform(10.0, 60.0),
    "vLeadK": rng.uniform(0.0, 30.0),
    "yRel": rng.uniform(-2.0, 2.0),
    "radarTrackId": tid if tid is not None else rng.randint(0, 1000),
    "radar": True,
    "modelProb": rng.uniform(0.0, 1.0),
  }
  d.update(overrides)
  return d


def _dt_choice(rng: random.Random) -> float | str:
  return rng.choice((rng.uniform(0.02, 0.2), "nan", "inf", -rng.uniform(0.01, 0.1), 0.0))


# ---------- model-trust helpers ----------


def _random_finite_accel(rng: random.Random) -> float:
  return rng.uniform(-5.0, 3.0)


# ---------- scenario generators ----------


def _generate_lead_confidence_sequence_case(rng: random.Random, index: int) -> StatefulCase:
  track_id = rng.randint(0, 1000)
  frames: list[dict[str, Any]] = []
  n = rng.randint(8, 24)
  kinds = ["no_lead", "stable", "stable", "stable", "jump", "flicker", "model_only", "radarless"]
  for i in range(n):
    kind = rng.choice(kinds)
    dt = _dt_choice(rng)
    if kind == "no_lead":
      lead = None
    elif kind == "jump":
      track_id = (track_id + 1) % 1000
      lead = _random_lead_dict(rng, tid=track_id, dRel=rng.uniform(50.0, 90.0))
    elif kind == "flicker":
      lead = _random_lead_dict(rng, tid=track_id) if i % 2 == 0 else None
    elif kind == "model_only":
      lead = _random_lead_dict(rng, tid=track_id, radar=False, modelProb=rng.uniform(0.0, 1.0))
    elif kind == "radarless":
      lead = _random_lead_dict(rng, tid=track_id, radar=False)
    else:
      lead = _random_lead_dict(rng, tid=track_id)
    frames.append({"lead": lead, "dt": dt})

  probe_accel = rng.choice((rng.uniform(-5.0, 5.0), "nan", "inf", "-inf"))
  state_blend = rng.uniform(0.0, 1.0)

  return StatefulCase(
    kind="lead_confidence_sequence",
    index=index,
    title=f"lead_confidence_sequence #{index}",
    params={"frames": frames, "probe_accel": probe_accel, "state_blend": state_blend},
  )


def _generate_model_trust_sequence_case(rng: random.Random, index: int) -> StatefulCase:
  steps: list[dict[str, Any]] = []
  n = rng.randint(6, 18)
  for _ in range(n):
    if rng.choice((True, False)):
      steps.append({
        "type": "gate",
        "model_should_stop": rng.choice((True, False)),
        "model_desired_accel": _random_finite_accel(rng),
        "stop_prob": rng.uniform(0.0, 1.0),
        "has_radar_lead": rng.choice((True, False)),
        "lead_v_rel": rng.uniform(-8.0, 4.0),
        "model_stale": rng.choice((True, False)),
      })
    else:
      steps.append({
        "type": "learner",
        "initial": rng.uniform(0.0, 1.0),
        "model_should_stop": rng.choice((True, False)),
        "driver_disagrees": rng.choice((True, False)),
        "dt": rng.choice((rng.uniform(0.01, 0.2), 0.0, -rng.uniform(0.01, 0.1))),
      })

  monotonic_probs = sorted(rng.uniform(0.0, 1.0) for _ in range(5))
  radar_base_accel = -rng.uniform(1.0, 4.0)

  return StatefulCase(
    kind="model_trust_sequence",
    index=index,
    title=f"model_trust_sequence #{index}",
    params={
      "steps": steps,
      "monotonic_stop_probs": monotonic_probs,
      "radar_probe": {
        "model_should_stop": True,
        "model_desired_accel": radar_base_accel,
        "stop_prob": rng.uniform(0.3, 0.7),
        "model_stale": False,
      },
    },
  )


_GENERATORS = {
  "lead_confidence_sequence": _generate_lead_confidence_sequence_case,
  "model_trust_sequence": _generate_model_trust_sequence_case,
}


def generate_cases(config: StatefulFuzzerConfig) -> list[StatefulCase]:
  """Deterministically generate stateful structural cases."""
  rng = random.Random(config.seed)
  kinds = [config.kind] if config.kind else list(DEFAULT_KINDS)
  cases: list[StatefulCase] = []
  for idx in range(config.cases):
    kind = rng.choice(kinds)
    cases.append(_GENERATORS[kind](rng, idx))
  return cases


# ---------- evaluation ----------


def _evaluate_lead_confidence_sequence(case: StatefulCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  tracker = LeadConfidenceTracker()
  metrics = {"frames": 0, "status_frames": 0, "new_lead_frames": 0, "stable_frames": 0}

  for frame_idx, frame in enumerate(case.params["frames"]):
    lead_dict = frame.get("lead")
    lead = _simple_namespace(lead_dict)
    dt = _to_float(frame["dt"])
    try:
      state = tracker.update(lead, dt)
    except Exception as exc:
      failures.append({
        "check": "no_exception",
        "detail": f"frame {frame_idx}: update raised {type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
      })
      continue

    metrics["frames"] += 1
    if state.status:
      metrics["status_frames"] += 1
      if state.new_lead:
        metrics["new_lead_frames"] += 1
      if state.stable:
        metrics["stable_frames"] += 1

    if not isinstance(state.status, bool):
      failures.append({"check": "status_bool", "detail": f"frame {frame_idx}: status={state.status!r}"})
    if not isinstance(state.new_lead, bool):
      failures.append({"check": "new_lead_bool", "detail": f"frame {frame_idx}: new_lead={state.new_lead!r}"})
    if not isinstance(state.stable, bool):
      failures.append({"check": "stable_bool", "detail": f"frame {frame_idx}: stable={state.stable!r}"})
    if state.age < -EPS:
      failures.append({"check": "age_nonnegative", "detail": f"frame {frame_idx}: age={state.age}"})
    if not (0.0 - EPS <= state.accel_blend <= 1.0 + EPS):
      failures.append({"check": "accel_blend_range", "detail": f"frame {frame_idx}: accel_blend={state.accel_blend}"})
    if not (0.0 - EPS <= state.guard_timer <= NEW_LEAD_GUARD_TIME + EPS):
      failures.append({"check": "guard_timer_range", "detail": f"frame {frame_idx}: guard_timer={state.guard_timer}"})
    if not (0.0 - EPS <= state.flicker_guard_timer <= LEAD_FLICKER_CLOSE_GUARD_TIME + EPS):
      failures.append({"check": "flicker_guard_timer_range", "detail": f"frame {frame_idx}: flicker_guard_timer={state.flicker_guard_timer}"})

    if state.stable and not math.isclose(state.accel_blend, 1.0, abs_tol=1e-6):
      failures.append({
        "check": "stable_implies_full_blend",
        "detail": f"frame {frame_idx}: stable=True but accel_blend={state.accel_blend}",
      })

    lead_status = lead is not None and bool(getattr(lead, "status", False))
    if not lead_status and state.status:
      failures.append({"check": "no_lead_status_false", "detail": f"frame {frame_idx}: no lead but state.status=True"})

    if state.status:
      radar = bool(getattr(lead, "radar", False))
      model_prob = _finite_default(getattr(lead, "modelProb", 0.0), 0.0)
      expected_trusted = radar or model_prob >= 0.5
      if state.speed_trusted != expected_trusted:
        failures.append({
          "check": "speed_trusted",
          "detail": f"frame {frame_idx}: expected={expected_trusted} got={state.speed_trusted}",
        })

  # Probe adjust_new_lead_accel.
  probe_accel = case.params.get("probe_accel", 0.0)
  state_blend = float(case.params.get("state_blend", 0.0))
  probe_state = LeadConfidenceState(accel_blend=max(0.0, min(1.0, state_blend)))
  try:
    adjusted = adjust_new_lead_accel(_to_float(probe_accel) if isinstance(probe_accel, (int, float)) else probe_accel, probe_state)
  except Exception as exc:
    failures.append({
      "check": "adjust_no_exception",
      "detail": f"adjust_new_lead_accel raised {type(exc).__name__}: {exc}",
    })
    adjusted = None

  if adjusted is not None and math.isfinite(_finite_default(probe_accel, 0.0)):
    raw_finite = _finite_default(probe_accel, 0.0)
    if raw_finite > 0.0 and (adjusted < -EPS or adjusted > raw_finite + EPS):
      failures.append({
        "check": "positive_accel_blend",
        "detail": f"adjusted={adjusted} not in [0, {raw_finite}]",
      })
    if raw_finite <= 0.0 and not math.isclose(adjusted, raw_finite, abs_tol=1e-6):
      failures.append({
        "check": "negative_accel_unchanged",
        "detail": f"adjusted={adjusted} expected {raw_finite}",
      })

  return failures, metrics


def _evaluate_model_trust_sequence(case: StatefulCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  metrics = {"gate_steps": 0, "learner_steps": 0}

  for step_idx, step in enumerate(case.params["steps"]):
    if step["type"] == "gate":
      metrics["gate_steps"] += 1
      try:
        result = gate_model_stop(
          model_should_stop=bool(step["model_should_stop"]),
          model_desired_accel=float(step["model_desired_accel"]),
          stop_prob=float(step["stop_prob"]),
          has_radar_lead=bool(step["has_radar_lead"]),
          lead_v_rel=float(step["lead_v_rel"]),
          model_stale=bool(step["model_stale"]),
        )
      except Exception as exc:
        failures.append({
          "check": "gate_no_exception",
          "detail": f"step {step_idx}: gate_model_stop raised {type(exc).__name__}: {exc}",
          "traceback": traceback.format_exc(),
        })
        continue

      if not (0.0 - EPS <= result.trust <= 1.0 + EPS):
        failures.append({"check": "trust_range", "detail": f"step {step_idx}: trust={result.trust}"})
      if not math.isfinite(result.desired_accel):
        failures.append({"check": "finite_desired_accel", "detail": f"step {step_idx}: desired_accel={result.desired_accel}"})

      if bool(step["model_stale"]) and bool(step["model_should_stop"]):
        if result.should_stop:
          failures.append({"check": "stale_no_stop", "detail": f"step {step_idx}: stale but should_stop=True"})
        if not math.isclose(result.trust, 0.0, abs_tol=EPS):
          failures.append({"check": "stale_zero_trust", "detail": f"step {step_idx}: stale trust={result.trust}"})
        if result.reason != "model_stale":
          failures.append({"check": "stale_reason", "detail": f"step {step_idx}: reason={result.reason}"})
        if not math.isclose(result.desired_accel, GENTLE_CAUTION_DECEL, abs_tol=1e-6):
          failures.append({
            "check": "stale_gentle_caution",
            "detail": f"step {step_idx}: stale desired_accel={result.desired_accel}",
          })
      elif not bool(step["model_should_stop"]) and float(step["model_desired_accel"]) >= 0.0:
        if result.should_stop:
          failures.append({"check": "clear_no_stop", "detail": f"step {step_idx}: clear model but should_stop=True"})
        if not math.isclose(result.trust, 1.0, abs_tol=EPS):
          failures.append({"check": "clear_full_trust", "detail": f"step {step_idx}: clear trust={result.trust}"})
        if not math.isclose(result.desired_accel, float(step["model_desired_accel"]), abs_tol=1e-6):
          failures.append({
            "check": "clear_passthrough",
            "detail": f"step {step_idx}: desired_accel={result.desired_accel} expected {step['model_desired_accel']}",
          })
      elif bool(step["model_should_stop"]):
        if result.desired_accel > EPS:
          failures.append({
            "check": "caution_non_positive",
            "detail": f"step {step_idx}: caution desired_accel={result.desired_accel}",
          })
    else:
      metrics["learner_steps"] += 1
      initial = float(step["initial"])
      try:
        learner = StopTrustLearner(initial=initial)
      except Exception as exc:
        failures.append({
          "check": "learner_init_no_exception",
          "detail": f"step {step_idx}: StopTrustLearner init raised {type(exc).__name__}: {exc}",
        })
        continue
      if not (STOP_TRUST_MIN - EPS <= learner.confidence <= STOP_TRUST_MAX + EPS):
        failures.append({
          "check": "learner_initial_clip",
          "detail": f"step {step_idx}: initial={initial} confidence={learner.confidence}",
        })
      before = float(learner.confidence)
      try:
        after = learner.update(
          model_should_stop=bool(step["model_should_stop"]),
          driver_disagrees=bool(step["driver_disagrees"]),
          dt=float(step["dt"]),
        )
      except Exception as exc:
        failures.append({
          "check": "learner_update_no_exception",
          "detail": f"step {step_idx}: update raised {type(exc).__name__}: {exc}",
          "traceback": traceback.format_exc(),
        })
        continue
      if not (STOP_TRUST_MIN - EPS <= after <= STOP_TRUST_MAX + EPS):
        failures.append({"check": "learner_bounded", "detail": f"step {step_idx}: after={after}"})
      dt = float(step["dt"])
      if dt <= 0.0 and not math.isclose(after, before, abs_tol=1e-9):
        failures.append({
          "check": "learner_nonpositive_dt_frozen",
          "detail": f"step {step_idx}: dt={dt} before={before} after={after}",
        })
      if not bool(step["model_should_stop"]) and not math.isclose(after, before, abs_tol=1e-9):
        failures.append({
          "check": "learner_idle_frozen",
          "detail": f"step {step_idx}: no model stop before={before} after={after}",
        })
      if bool(step["model_should_stop"]):
        if bool(step["driver_disagrees"]) and after > before + EPS:
          failures.append({
            "check": "learner_disagree_no_increase",
            "detail": f"step {step_idx}: disagree before={before} after={after}",
          })
        if not bool(step["driver_disagrees"]) and after < before - EPS:
          failures.append({
            "check": "learner_agree_no_decrease",
            "detail": f"step {step_idx}: agree before={before} after={after}",
          })

  # Monotonic stop-probe sequence.
  monotonic_probs = case.params.get("monotonic_stop_probs", [])
  base_accel = -2.0
  prev_accel: float | None = None
  prev_stop: bool | None = None
  for prob in monotonic_probs:
    try:
      r = gate_model_stop(True, base_accel, float(prob))
    except Exception as exc:
      failures.append({
        "check": "monotonic_no_exception",
        "detail": f"gate_model_stop raised {type(exc).__name__}: {exc}",
      })
      break
    if prev_accel is not None and r.desired_accel > prev_accel + EPS:
      failures.append({
        "check": "stop_prob_monotonic_accel",
        "detail": f"desired_accel increased from {prev_accel} to {r.desired_accel} at prob={prob}",
      })
    if prev_stop is True and not r.should_stop:
      failures.append({
        "check": "stop_prob_monotonic_should_stop",
        "detail": f"should_stop went false at prob={prob}",
      })
    prev_accel = r.desired_accel
    prev_stop = r.should_stop

  # Radar corroboration probe.
  probe = case.params.get("radar_probe", {})
  if probe:
    base_kwargs = {
      "model_should_stop": bool(probe.get("model_should_stop", True)),
      "model_desired_accel": float(probe.get("model_desired_accel", -2.0)),
      "stop_prob": float(probe.get("stop_prob", 0.5)),
      "model_stale": bool(probe.get("model_stale", False)),
    }
    try:
      without = gate_model_stop(**base_kwargs, has_radar_lead=False, lead_v_rel=0.0)
      with_radar = gate_model_stop(**base_kwargs, has_radar_lead=True, lead_v_rel=-5.0)
    except Exception as exc:
      failures.append({
        "check": "radar_probe_no_exception",
        "detail": f"radar probe raised {type(exc).__name__}: {exc}",
      })
    else:
      if with_radar.trust < without.trust - EPS:
        failures.append({
          "check": "radar_corroboration_trust",
          "detail": f"trust with radar {with_radar.trust} < without {without.trust}",
        })

  return failures, metrics


_EVALUATORS = {
  "lead_confidence_sequence": _evaluate_lead_confidence_sequence,
  "model_trust_sequence": _evaluate_model_trust_sequence,
}


def evaluate_case(case: StatefulCase) -> StatefulResult:
  """Run the structural invariants for one stateful case."""
  try:
    failures, metrics = _EVALUATORS[case.kind](case)
  except Exception as exc:
    failures = [{
      "check": "evaluator_exception",
      "detail": f"evaluator raised {type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    }]
    metrics = {}

  return StatefulResult(
    case=case,
    failures=failures,
    metrics=_sanitize(metrics),
  )


# ---------- CLI / serialization ----------


def _render_case_snippet(case: StatefulCase) -> str:
  return f"# kind: {case.kind}\nStatefulCase(title={case.title!r}, params=...)"


def main() -> None:
  parser = argparse.ArgumentParser(description="Stateful structural fuzzer for longitudinal trust / confidence modules.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=DEFAULT_KINDS, help="Fuzzer kind")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")

  config = StatefulFuzzerConfig(seed=args.seed, cases=args.cases, kind=args.kind)
  cases = generate_cases(config)
  results: list[StatefulResult] = []
  for case in cases:
    result = evaluate_case(case)
    results.append(result)
    if not result.valid and args.fail_fast:
      break

  failures = [r for r in results if not r.valid]

  if args.json:
    payload = {
      "seed": args.seed,
      "cases": len(results),
      "kind": args.kind,
      "failures": [
        {
          "case": result.case.to_dict(),
          "failure_checks": [f["check"] for f in result.failures],
          "failure_details": result.failures,
          "metrics": result.metrics,
        }
        for result in failures
      ],
      "metrics": {
        "total": len(results),
        "failures": len(failures),
        "by_kind": {kind: sum(1 for r in results if r.case.kind == kind) for kind in DEFAULT_KINDS},
      },
    }
    print(json.dumps(_sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
  else:
    print(
      f"Drive Lab longitudinal stateful fuzz seed={args.seed} cases={len(results)} "
      f"kind={args.kind or 'default'} failures={len(failures)}"
    )
    for result in failures[:10]:
      print(f"\nFAILED: {result.case.title}")
      for failure in result.failures:
        print(f"  {failure['check']}: {failure['detail']}")
      print(_render_case_snippet(result.case))

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
