#!/usr/bin/env python3
"""Structural fuzzer for pure longitudinal modules.

Exercises the stateless/building-block longitudinal modules without touching Params,
MPC/acados, route logs, or comfort/tuning thresholds:

  - sunnypilot.custom.longitudinal.modes
  - sunnypilot.custom.longitudinal.decision
  - sunnypilot.custom.longitudinal.trajectory
  - sunnypilot.custom.longitudinal.acc_envelope

This is a bounded Phase 3A fuzzer: property-style checks only, deterministic,
no product-behavior changes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.sunnypilot.custom.longitudinal import acc_envelope as envelope_mod
from openpilot.sunnypilot.custom.longitudinal import decision as decision_mod
from openpilot.sunnypilot.custom.longitudinal import modes as modes_mod
from openpilot.sunnypilot.custom.longitudinal import trajectory as trajectory_mod

EPS = 1e-6

DEFAULT_KINDS = (
  "mode_matrix",
  "decision_matrix",
  "trajectory_grid",
  "acc_envelope_grid",
)

# Public-semantic expected evidence sets for mode_matrix checks.
_ACC_EVIDENCE_PUBLIC = frozenset({modes_mod.EvidenceClass.CRUISE, modes_mod.EvidenceClass.LEAD})
_E2E_EVIDENCE_PUBLIC = frozenset({
  modes_mod.EvidenceClass.CRUISE,
  modes_mod.EvidenceClass.LEAD,
  modes_mod.EvidenceClass.MODEL_STOP,
})
_SCC_BASE_EVIDENCE_PUBLIC = frozenset({
  modes_mod.EvidenceClass.CRUISE,
  modes_mod.EvidenceClass.LEAD,
  modes_mod.EvidenceClass.MODEL_STOP,
  modes_mod.EvidenceClass.SPEED_LIMIT,
})


@dataclass(frozen=True)
class ModuleFuzzerConfig:
  """Top-level fuzzer configuration."""

  seed: int = 1
  cases: int = 100
  kind: str | None = None


@dataclass(frozen=True)
class ModuleCase:
  """One generated structural case."""

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
  def from_dict(cls, data: dict[str, Any]) -> "ModuleCase":
    return cls(
      kind=str(data["kind"]),
      index=int(data["index"]),
      title=str(data["title"]),
      params=dict(data.get("params", {})),
    )


@dataclass(frozen=True)
class ModuleResult:
  """Result of evaluating one case."""

  case: ModuleCase
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
    if lowered == "inf" or lowered == "+inf":
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


def _reconstruct_source_toggles(data: dict[str, Any]) -> modes_mod.SourceToggles:
  return modes_mod.SourceToggles(
    scc_curve_vision_enabled=bool(data.get("scc_curve_vision_enabled", False)),
    scc_curve_map_enabled=bool(data.get("scc_curve_map_enabled", False)),
  )


def _candidate_to_dict(candidate: decision_mod.LongitudinalCandidate) -> dict[str, Any]:
  return {
    "a_target": candidate.a_target,
    "role": candidate.role.value,
    "source": candidate.source.name,
    "intent": candidate.intent,
    "authorized": candidate.authorized,
    "is_stop": candidate.is_stop,
  }


def _candidate_from_dict(data: dict[str, Any]) -> decision_mod.LongitudinalCandidate:
  return decision_mod.LongitudinalCandidate(
    a_target=float(data["a_target"]),
    role=decision_mod.CandidateRole(data["role"]),
    source=modes_mod.EvidenceClass[data["source"]],
    intent=str(data.get("intent", "")),
    authorized=bool(data.get("authorized", True)),
    is_stop=bool(data.get("is_stop", False)),
  )


def _reconstruct_from_value_input(item: dict[str, Any]) -> Any:
  t = item["type"]
  value = item.get("value")
  if t == "enum":
    return modes_mod.LongitudinalMode[str(value)]
  if t == "name":
    return str(value)
  if t == "bytes":
    return str(value).encode()
  if t == "int":
    return int(str(value))
  if t == "none":
    return None
  return value


# ---------- scenario generators ----------


def _from_value_vectors() -> list[dict[str, Any]]:
  """Stable coverage vectors for LongitudinalMode.from_value."""
  return [
    {"type": "enum", "value": "ACC", "expected": "ACC"},
    {"type": "enum", "value": "E2E", "expected": "E2E"},
    {"type": "enum", "value": "SCC", "expected": "SCC"},
    {"type": "value", "value": "acc", "expected": "ACC"},
    {"type": "value", "value": "e2e", "expected": "E2E"},
    {"type": "value", "value": "scc", "expected": "SCC"},
    {"type": "name", "value": "ACC", "expected": "ACC"},
    {"type": "name", "value": "E2E", "expected": "E2E"},
    {"type": "name", "value": "SCC", "expected": "SCC"},
    {"type": "bytes", "value": "acc", "expected": "ACC"},
    {"type": "int", "value": "0", "expected": "ACC"},
    {"type": "int", "value": "1", "expected": "E2E"},
    {"type": "int", "value": "2", "expected": "SCC"},
    {"type": "none", "value": None, "expected": "ACC"},
    {"type": "value", "value": "", "expected": "ACC"},
    {"type": "value", "value": "unknown", "expected": "ACC"},
  ]


def _generate_mode_matrix_case(rng: random.Random, index: int) -> ModuleCase:
  mode = rng.choice(list(modes_mod.LongitudinalMode))
  sources = modes_mod.SourceToggles(
    scc_curve_vision_enabled=rng.choice((True, False)),
    scc_curve_map_enabled=rng.choice((True, False)),
  )
  vectors = _from_value_vectors()
  # Sprinkle a few random extra vectors so different seeds still find edge values.
  for _ in range(rng.randint(0, 3)):
    raw = rng.choice(["acc", "e2e", "scc", "ACC", "FoxLauncher", "", "0", "1", "2"])
    vectors.append({"type": "value", "value": raw, "expected": "ACC" if raw not in ("e2e", "scc", "1", "2") else ("E2E" if raw in ("e2e", "1") else "SCC")})
  return ModuleCase(
    kind="mode_matrix",
    index=index,
    title=f"mode_matrix #{index} {mode.value}",
    params={
      "mode": mode.value,
      "sources": asdict(sources),
      "from_value_inputs": vectors,
    },
  )


def _generate_decision_matrix_case(rng: random.Random, index: int) -> ModuleCase:
  mode = rng.choice(list(modes_mod.LongitudinalMode))
  sources = modes_mod.SourceToggles(
    scc_curve_vision_enabled=rng.choice((True, False)),
    scc_curve_map_enabled=rng.choice((True, False)),
  )
  admitted = modes_mod.admitted_evidence(mode, sources)
  evidence_pool = list(modes_mod.EvidenceClass)

  candidates: list[decision_mod.LongitudinalCandidate] = []
  n_candidates = rng.randint(0, 6)
  for _ in range(n_candidates):
    role = rng.choice(list(decision_mod.CandidateRole))
    source = rng.choice(evidence_pool)
    a_target = rng.uniform(-6.0, 4.0)
    authorized = rng.choice((True, False))
    is_stop = role is decision_mod.CandidateRole.PHYSICAL_HAZARD and rng.choice((True, False))
    candidates.append(decision_mod.LongitudinalCandidate(
      a_target=a_target,
      role=role,
      source=source,
      intent=f"{role.value}-{source.name}-{rng.randint(0, 999)}",
      authorized=authorized,
      is_stop=is_stop,
    ))

  # Mix valid and broken accel limits.
  limit_kind = rng.choice((
    "valid", "valid", "valid", "valid",
    "swapped", "nan_min", "nan_max", "inf_min", "inf_max", "nonfinite",
  ))
  if limit_kind == "valid":
    a_min = rng.uniform(-5.0, -0.1)
    a_max = rng.uniform(0.1, 3.0)
  elif limit_kind == "swapped":
    a_min = rng.uniform(1.0, 3.0)
    a_max = rng.uniform(-5.0, -1.0)
  elif limit_kind == "nan_min":
    a_min = "nan"
    a_max = rng.uniform(0.5, 3.0)
  elif limit_kind == "nan_max":
    a_min = rng.uniform(-5.0, -0.5)
    a_max = "nan"
  elif limit_kind == "inf_min":
    a_min = "-inf"
    a_max = rng.uniform(0.5, 3.0)
  elif limit_kind == "inf_max":
    a_min = rng.uniform(-5.0, -0.5)
    a_max = "inf"
  else:
    a_min = "nan"
    a_max = "inf"

  return ModuleCase(
    kind="decision_matrix",
    index=index,
    title=f"decision_matrix #{index} {mode.value}",
    params={
      "mode": mode.value,
      "sources": asdict(sources),
      "accel_limits": [a_min, a_max],
      "candidates": [_candidate_to_dict(c) for c in candidates],
      "admitted_count": len(admitted),
    },
  )


def _generate_trajectory_grid_case(rng: random.Random, index: int) -> ModuleCase:
  a_target = rng.uniform(-4.0, 3.0)
  speeds_in = [rng.uniform(0.0, 40.0) for _ in range(CONTROL_N)]
  accels_in = [rng.uniform(-5.0, 3.0) for _ in range(CONTROL_N)]
  v_ego = rng.choice((rng.uniform(0.0, 40.0), "nan", "inf", -rng.uniform(0.0, 5.0)))
  limit_jerk = rng.choice((True, False))

  preserve = {
    "output_a_target": a_target if rng.random() < 0.4 else rng.uniform(-4.0, 3.0),
    "planner_seed_scalar": rng.choice((True, False)),
    "a_target": a_target,
  }

  return ModuleCase(
    kind="trajectory_grid",
    index=index,
    title=f"trajectory_grid #{index}",
    params={
      "v_ego": v_ego,
      "a_target": a_target,
      "limit_jerk": limit_jerk,
      "speeds_in": speeds_in,
      "accels_in": accels_in,
      "preserve": preserve,
    },
  )


def _generate_acc_envelope_grid_case(rng: random.Random, index: int) -> ModuleCase:
  scenario = rng.choice((
    "neutral", "neutral", "neutral",
    "invalid_dt", "invalid_v_ego",
    "inside_gap", "ttc_low", "high_decel",
    "stale_model", "stale_radar", "stock_long", "jerk_limit",
  ))

  previous_a = rng.uniform(-2.0, 1.0)
  dt = rng.uniform(0.05, 0.5)

  # Defaults for a neutral/no-cap case.
  kwargs: dict[str, Any] = {
    "v_ego": rng.uniform(5.0, 35.0),
    "candidate_a_target": previous_a + rng.uniform(-0.3, 0.3),
    "previous_a_target": previous_a,
    "dt": dt,
    "openpilot_longitudinal_control": True,
    "has_lead": False,
    "lead_d_rel": 100.0,
    "lead_v_rel": 0.0,
    "lead_v_lead": rng.uniform(5.0, 35.0),
    "lead_a_lead_k": 0.0,
    "lead_kinematics_valid": True,
    "model_stale": False,
    "model_progress_candidate": False,
    "radar_stale": False,
    "lead_required": False,
  }

  if scenario == "invalid_dt":
    kwargs["dt"] = rng.choice((0.0, -rng.uniform(0.01, 1.0)))
  elif scenario == "invalid_v_ego":
    kwargs["v_ego"] = "nan"
  elif scenario in ("inside_gap", "ttc_low", "high_decel", "stale_radar"):
    kwargs["has_lead"] = True
    v_ego = rng.uniform(10.0, 30.0)
    kwargs["v_ego"] = v_ego
    if scenario == "inside_gap":
      kwargs["lead_d_rel"] = rng.uniform(1.0, 5.0)
      kwargs["lead_v_rel"] = 0.0
    elif scenario == "ttc_low":
      kwargs["lead_d_rel"] = rng.uniform(15.0, 40.0)
      kwargs["lead_v_rel"] = -rng.uniform(8.0, 20.0)
    elif scenario == "high_decel":
      kwargs["lead_d_rel"] = rng.uniform(10.0, 25.0)
      kwargs["lead_v_rel"] = -rng.uniform(5.0, 12.0)
    else:
      kwargs["lead_required"] = True
      kwargs["radar_stale"] = True
  elif scenario == "stale_model":
    kwargs["model_stale"] = True
    kwargs["model_progress_candidate"] = True
  elif scenario == "stock_long":
    kwargs["openpilot_longitudinal_control"] = False
  elif scenario == "jerk_limit":
    kwargs["candidate_a_target"] = previous_a + rng.choice((
      rng.uniform(0.5, 3.0), rng.uniform(-4.0, -0.5),
    ))

  inp = envelope_mod.AccEnvelopeInputs(**kwargs)

  return ModuleCase(
    kind="acc_envelope_grid",
    index=index,
    title=f"acc_envelope_grid #{index} {scenario}",
    params={"inp": asdict(inp)},
  )


_GENERATORS = {
  "mode_matrix": _generate_mode_matrix_case,
  "decision_matrix": _generate_decision_matrix_case,
  "trajectory_grid": _generate_trajectory_grid_case,
  "acc_envelope_grid": _generate_acc_envelope_grid_case,
}


def generate_cases(config: ModuleFuzzerConfig) -> list[ModuleCase]:
  """Deterministically generate structural cases."""
  rng = random.Random(config.seed)
  kinds = [config.kind] if config.kind else list(DEFAULT_KINDS)
  cases: list[ModuleCase] = []
  for idx in range(config.cases):
    kind = rng.choice(kinds)
    cases.append(_GENERATORS[kind](rng, idx))
  return cases


# ---------- evaluation ----------


def _evaluate_mode_matrix(case: ModuleCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  mode = modes_mod.LongitudinalMode.from_value(case.params["mode"])
  sources = _reconstruct_source_toggles(case.params["sources"])
  admitted = modes_mod.admitted_evidence(mode, sources)

  for ev in modes_mod.EvidenceClass:
    expected = ev in admitted
    actual = modes_mod.is_admitted(mode, ev, sources)
    if actual != expected:
      failures.append({
        "check": "is_admitted_consistency",
        "detail": f"mode={mode.value} evidence={ev.name} expected={expected} actual={actual}",
      })

  if mode is modes_mod.LongitudinalMode.ACC:
    if admitted != _ACC_EVIDENCE_PUBLIC:
      failures.append({
        "check": "acc_evidence_exact",
        "detail": f"ACC admitted {sorted(e.name for e in admitted)}, expected {sorted(e.name for e in _ACC_EVIDENCE_PUBLIC)}",
      })
    opposite = modes_mod.SourceToggles(
      scc_curve_vision_enabled=not sources.scc_curve_vision_enabled,
      scc_curve_map_enabled=not sources.scc_curve_map_enabled,
    )
    if admitted != modes_mod.admitted_evidence(mode, opposite):
      failures.append({"check": "acc_ignores_toggles", "detail": "ACC admitted evidence changed with toggles"})

  if mode is modes_mod.LongitudinalMode.E2E:
    if admitted != _E2E_EVIDENCE_PUBLIC:
      failures.append({
        "check": "e2e_evidence_exact",
        "detail": f"E2E admitted {sorted(e.name for e in admitted)}, expected {sorted(e.name for e in _E2E_EVIDENCE_PUBLIC)}",
      })
    opposite = modes_mod.SourceToggles(
      scc_curve_vision_enabled=not sources.scc_curve_vision_enabled,
      scc_curve_map_enabled=not sources.scc_curve_map_enabled,
    )
    if admitted != modes_mod.admitted_evidence(mode, opposite):
      failures.append({"check": "e2e_ignores_toggles", "detail": "E2E admitted evidence changed with toggles"})

  if mode is modes_mod.LongitudinalMode.SCC:
    base = _SCC_BASE_EVIDENCE_PUBLIC
    if not base.issubset(admitted):
      failures.append({
        "check": "scc_base_evidence",
        "detail": f"SCC missing base evidence: {sorted(e.name for e in base - admitted)}",
      })
    cv_ok = (modes_mod.EvidenceClass.CURVE_VISION in admitted) == sources.scc_curve_vision_enabled
    cm_ok = (modes_mod.EvidenceClass.CURVE_MAP in admitted) == sources.scc_curve_map_enabled
    if not cv_ok:
      failures.append({"check": "scc_curve_vision_toggle", "detail": "CURVE_VISION admission mismatched toggle"})
    if not cm_ok:
      failures.append({"check": "scc_curve_map_toggle", "detail": "CURVE_MAP admission mismatched toggle"})

  from_value_checks = 0
  for item in case.params.get("from_value_inputs", []):
    from_value_checks += 1
    inp = _reconstruct_from_value_input(item)
    expected_name = item.get("expected")
    expected = modes_mod.LongitudinalMode[expected_name] if expected_name else modes_mod.LongitudinalMode.ACC
    actual = modes_mod.LongitudinalMode.from_value(inp)
    if actual != expected:
      failures.append({
        "check": "from_value",
        "detail": f"from_value({inp!r}) expected {expected.name} got {actual.name if actual else actual}",
      })

  metrics = {
    "mode": mode.value,
    "admitted_count": len(admitted),
    "from_value_checks": from_value_checks,
  }
  return failures, metrics


def _evaluate_decision_matrix(case: ModuleCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  mode = modes_mod.LongitudinalMode.from_value(case.params["mode"])
  sources = _reconstruct_source_toggles(case.params.get("sources", {}))
  raw_limits = case.params["accel_limits"]
  a_min = _to_float(raw_limits[0])
  a_max = _to_float(raw_limits[1])
  candidates = [_candidate_from_dict(c) for c in case.params["candidates"]]

  try:
    decision = decision_mod.decide(candidates, mode, (a_min, a_max), sources)
  except Exception as exc:
    failures.append({
      "check": "no_exception",
      "detail": f"decide raised {type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    })
    return failures, {"candidates_count": len(candidates)}

  limits_valid = math.isfinite(a_min) and math.isfinite(a_max) and a_min <= a_max
  if limits_valid:
    if not math.isfinite(decision.a_target):
      failures.append({"check": "finite_output", "detail": f"a_target={decision.a_target} with valid limits"})
    elif not (a_min - EPS <= decision.a_target <= a_max + EPS):
      failures.append({
        "check": "clamped_within_limits",
        "detail": f"a_target={decision.a_target:.4f} outside [{a_min:.4f}, {a_max:.4f}]",
      })

    admitted = modes_mod.admitted_evidence(mode, sources)
    excluded = [c for c in candidates if c.source not in admitted]
    if excluded and not any("mode_excluded" in r for r in decision.rejected):
      failures.append({
        "check": "mode_excluded_reported",
        "detail": f"{len(excluded)} excluded candidates but no mode_excluded rejection",
      })

    usable_hazards = [c for c in candidates if c.role is decision_mod.CandidateRole.PHYSICAL_HAZARD and c.source in admitted]
    if usable_hazards:
      strongest = min(c.a_target for c in usable_hazards)
      # The hazard binds unless the accel-limit floor prevents a stronger decel request.
      binding_ceiling = max(strongest, a_min)
      if decision.a_target > binding_ceiling + EPS:
        failures.append({
          "check": "physical_hazard_binds",
          "detail": f"a_target={decision.a_target:.4f} above binding ceiling {binding_ceiling:.4f} (strongest hazard {strongest:.4f})",
        })
      if any(c.is_stop for c in usable_hazards) and not decision.should_stop:
        failures.append({"check": "stop_hazard_sets_should_stop", "detail": "stop hazard present but should_stop=False"})

    authorized_progress = [
      c for c in candidates
      if c.role is decision_mod.CandidateRole.PROGRESS and c.authorized and c.source in admitted
    ]
    unauthorized_progress = [
      c for c in candidates
      if c.role is decision_mod.CandidateRole.PROGRESS and not c.authorized and c.source in admitted
    ]
    cruise = [c for c in candidates if c.role is decision_mod.CandidateRole.CRUISE and c.source in admitted]
    if unauthorized_progress:
      baseline_desire = max((c.a_target for c in cruise + authorized_progress), default=a_max)
      if decision.a_target > baseline_desire + EPS:
        failures.append({
          "check": "unauthorized_progress_no_raise",
          "detail": f"a_target={decision.a_target:.4f} raised by unauthorized progress above baseline {baseline_desire:.4f}",
        })
  else:
    if decision.reason != "invalid_accel_limits":
      failures.append({
        "check": "invalid_limits_reason",
        "detail": f"expected reason invalid_accel_limits, got {decision.reason}",
      })
    if not math.isfinite(decision.a_target):
      failures.append({"check": "finite_fault_output", "detail": f"a_target={decision.a_target} for invalid limits"})

  metrics = {
    "candidates_count": len(candidates),
    "reason": decision.reason,
    "a_target": decision.a_target,
    "should_stop": decision.should_stop,
    "rejected_count": len(decision.rejected),
  }
  return failures, metrics


def _evaluate_trajectory_grid(case: ModuleCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  p = case.params
  v_ego = _to_float(p["v_ego"])
  a_target = _to_float(p["a_target"])
  limit_jerk = bool(p["limit_jerk"])
  speeds_in = [_to_float(v) for v in p.get("speeds_in", [])]
  accels_in = [_to_float(v) for v in p.get("accels_in", [])]

  try:
    speeds, accels, jerks = trajectory_mod.synth_trajectory(speeds_in, accels_in, v_ego, a_target, limit_jerk)
  except Exception as exc:
    failures.append({
      "check": "no_exception",
      "detail": f"synth_trajectory raised {type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    })
    return failures, {"control_n": CONTROL_N}

  if not (len(speeds) == CONTROL_N and len(accels) == CONTROL_N and len(jerks) == CONTROL_N):
    failures.append({
      "check": "control_n_lengths",
      "detail": f"speeds={len(speeds)} accels={len(accels)} jerks={len(jerks)}, expected {CONTROL_N}",
    })

  if not all(math.isfinite(s) and math.isfinite(a) and math.isfinite(j) for s, a, j in zip(speeds, accels, jerks)):
    failures.append({"check": "finite_outputs", "detail": "non-finite speed/accel/jerk values"})

  if any(s < -EPS for s in speeds):
    failures.append({"check": "nonnegative_speeds", "detail": f"minimum speed={min(speeds):.4f}"})

  try:
    dts = trajectory_mod.synth_trajectory_dts()
  except Exception as exc:
    failures.append({
      "check": "dts_no_exception",
      "detail": f"synth_trajectory_dts raised {type(exc).__name__}: {exc}",
    })
    dts = ()

  if len(dts) != CONTROL_N or not all(math.isfinite(dt) and dt > 0.0 for dt in dts):
    failures.append({
      "check": "dts_positive_finite",
      "detail": f"dts length={len(dts)} values={dts}",
    })

  if limit_jerk:
    lo = trajectory_mod.NORMAL_NEGATIVE_RETREAT_JERK - EPS
    hi = trajectory_mod.POSITIVE_PROGRESS_JERK + EPS
    for idx, j in enumerate(jerks):
      if j < lo or j > hi:
        failures.append({
          "check": "jerk_limited_bounds",
          "detail": f"jerk[{idx}]={j:.4f} outside [{lo:.4f}, {hi:.4f}]",
        })

  preserve = p.get("preserve", {})
  try:
    preserved = trajectory_mod.preserve_seed_trajectory(
      _to_float(preserve.get("output_a_target", 0.0)),
      preserve.get("planner_seed_scalar", False),
      _to_float(preserve.get("a_target", 0.0)),
    )
  except Exception as exc:
    failures.append({
      "check": "preserve_no_exception",
      "detail": f"preserve_seed_trajectory raised {type(exc).__name__}: {exc}",
    })
    preserved = None

  if preserved is not None:
    planner_seed_scalar = bool(preserve.get("planner_seed_scalar", False))
    out_a = float(preserve.get("output_a_target", 0.0))
    ref_a = float(preserve.get("a_target", 0.0))
    expected: bool
    if planner_seed_scalar:
      expected = False
    else:
      expected = math.isclose(out_a, ref_a, abs_tol=trajectory_mod.A_TARGET_EPS)
    if preserved != expected:
      failures.append({
        "check": "preserve_seed_consistency",
        "detail": f"planner_seed_scalar={planner_seed_scalar} out={out_a} ref={ref_a} expected={expected} got={preserved}",
      })

  metrics = {
    "control_n": CONTROL_N,
    "limit_jerk": limit_jerk,
    "max_speed": max(speeds) if speeds and all(math.isfinite(s) for s in speeds) else None,
    "min_speed": min(speeds) if speeds and all(math.isfinite(s) for s in speeds) else None,
    "max_abs_jerk": max(abs(j) for j in jerks) if jerks and all(math.isfinite(j) for j in jerks) else None,
  }
  return failures, metrics


def _evaluate_acc_envelope_grid(case: ModuleCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  failures: list[dict[str, Any]] = []
  inp_dict = {k: _to_float_or_passthrough(v) for k, v in case.params["inp"].items()}
  inp = envelope_mod.AccEnvelopeInputs(**inp_dict)

  try:
    result = envelope_mod.evaluate_acc_envelope(inp)
  except Exception as exc:
    failures.append({
      "check": "no_exception",
      "detail": f"evaluate_acc_envelope raised {type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    })
    return failures, {}

  numeric_fields = (
    "allowed_a_target", "delta_a", "desired_gap", "usable_stopping_gap",
    "required_stopping_decel", "closing_speed_decel", "jerk_limited_a_target",
  )
  for name in numeric_fields:
    if not math.isfinite(getattr(result, name)):
      failures.append({"check": f"finite_{name}", "detail": f"{name}={getattr(result, name)}"})

  if not (math.isfinite(result.time_gap) or math.isinf(result.time_gap)):
    failures.append({"check": "time_gap_finite_or_inf", "detail": f"time_gap={result.time_gap}"})
  if not (math.isfinite(result.ttc) or math.isinf(result.ttc)):
    failures.append({"check": "ttc_finite_or_inf", "detail": f"ttc={result.ttc}"})

  finite_core = all(math.isfinite(v) for v in (
    inp.v_ego, inp.candidate_a_target, inp.previous_a_target, inp.dt,
    inp.lead_d_rel, inp.lead_v_rel, inp.lead_v_lead, inp.lead_a_lead_k,
  ))
  invalid_data = (not finite_core) or inp.dt <= 0.0
  reasons = set(result.cap_reasons)

  if invalid_data:
    if result.active:
      failures.append({"check": "invalid_data_inactive", "detail": "invalid data but result.active=True"})
    if not result.would_cap:
      failures.append({"check": "invalid_data_would_cap", "detail": "invalid data but would_cap=False"})
    if "invalid_data" not in reasons:
      failures.append({"check": "invalid_data_reason", "detail": f"reasons={result.cap_reasons}"})
    if not math.isfinite(result.allowed_a_target):
      failures.append({"check": "invalid_data_finite_allowed", "detail": f"allowed_a_target={result.allowed_a_target}"})
    return failures, {"would_cap": result.would_cap}

  neutral = (
    inp.openpilot_longitudinal_control
    and not inp.has_lead
    and not inp.model_stale
    and not inp.radar_stale
    and math.isclose(result.jerk_limited_a_target, inp.candidate_a_target, abs_tol=1e-6)
  )
  if neutral and result.would_cap:
    failures.append({
      "check": "neutral_no_cap",
      "detail": f"neutral case would_cap=True reasons={result.cap_reasons}",
    })

  if inp.has_lead and inp.lead_kinematics_valid and inp.lead_d_rel > 0.0:
    triggered_gap = result.time_gap < float(inp.time_gap_s)
    triggered_ttc = result.ttc < float(inp.ttc_min_s)
    triggered_decel = result.required_stopping_decel > float(inp.required_decel_limit)
    if triggered_gap or triggered_ttc or triggered_decel:
      if not result.would_cap:
        failures.append({"check": "lead_trigger_would_cap", "detail": "lead safety trigger but would_cap=False"})
      if result.allowed_a_target > inp.candidate_a_target + 1e-6:
        failures.append({
          "check": "lead_trigger_allowed_below_candidate",
          "detail": f"allowed={result.allowed_a_target:.4f} candidate={inp.candidate_a_target:.4f}",
        })
      if not ({"inside_time_gap", "ttc_low", "closing_decel_high"} & reasons):
        failures.append({
          "check": "lead_trigger_reason",
          "detail": f"lead trigger but reasons={result.cap_reasons}",
        })

  if inp.model_stale and inp.model_progress_candidate:
    if "model_stale_blocks_model_progress" not in reasons:
      failures.append({"check": "model_stale_reason", "detail": f"reasons={result.cap_reasons}"})

  if inp.radar_stale and inp.lead_required:
    if "radar_stale_blocks_lead_progress" not in reasons:
      failures.append({"check": "radar_stale_reason", "detail": f"reasons={result.cap_reasons}"})

  lo = inp.previous_a_target + inp.max_decel_jerk * inp.dt
  hi = inp.previous_a_target + inp.max_accel_jerk * inp.dt
  if not (lo - EPS <= result.jerk_limited_a_target <= hi + EPS):
    failures.append({
      "check": "jerk_limited_bounds",
      "detail": f"jerk_limited={result.jerk_limited_a_target:.4f} outside [{lo:.4f}, {hi:.4f}]",
    })

  metrics = {
    "would_cap": result.would_cap,
    "allowed_a_target": result.allowed_a_target,
    "candidate_a_target": inp.candidate_a_target,
    "reason_count": len(result.cap_reasons),
  }
  return failures, metrics


_EVALUATORS = {
  "mode_matrix": _evaluate_mode_matrix,
  "decision_matrix": _evaluate_decision_matrix,
  "trajectory_grid": _evaluate_trajectory_grid,
  "acc_envelope_grid": _evaluate_acc_envelope_grid,
}


def evaluate_case(case: ModuleCase) -> ModuleResult:
  """Run the structural invariants for one case."""
  try:
    failures, metrics = _EVALUATORS[case.kind](case)
  except Exception as exc:
    failures = [{
      "check": "evaluator_exception",
      "detail": f"evaluator raised {type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    }]
    metrics = {}

  return ModuleResult(
    case=case,
    failures=failures,
    metrics=_sanitize(metrics),
  )


# ---------- CLI / serialization ----------


def _render_case_snippet(case: ModuleCase) -> str:
  return f"# kind: {case.kind}\nModuleCase(title={case.title!r}, params=...)"


def main() -> None:
  parser = argparse.ArgumentParser(description="Structural fuzzer for pure longitudinal modules.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--kind", choices=DEFAULT_KINDS, help="Fuzzer kind")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
  args = parser.parse_args()

  if args.cases < 0:
    parser.error("--cases must be >= 0")

  config = ModuleFuzzerConfig(seed=args.seed, cases=args.cases, kind=args.kind)
  cases = generate_cases(config)
  results: list[ModuleResult] = []
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
      f"Drive Lab longitudinal modules fuzz seed={args.seed} cases={len(results)} "
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
