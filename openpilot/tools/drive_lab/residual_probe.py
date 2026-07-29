#!/usr/bin/env python3
"""Is the planner-vs-driver disagreement a *learnable* residual, or just noise?

The fork carries ~369 hand-tuned longitudinal constants. The standing question is whether a
learned correction could do better. Before building any of that, this answers the cheap
version: on manual driving, with the planner computing its target open-loop alongside, is
`a_ego - plan_a_target` predictable from the scene, or is it unstructured?

Method: fit a ridge regression of the residual on simple physical features and report a
*cross-validated* R^2 (grouped by route, so a fit cannot score itself on its own drive).
The CV score is the whole point — an in-sample R^2 always looks encouraging.

    R^2 <= ~0        no learnable signal. The disagreement is noise or unmodelled context;
                     a residual model is not the missing piece. Stop here, cheaply.
    R^2 modest       there IS systematic disagreement. The per-regime bias table then says
                     *where*, which is directly actionable as constant tuning — no model
                     needs to ship to collect that value.
    R^2 high         a residual model is worth designing properly.

WHAT THIS DOES NOT ESTABLISH: that applying the fitted correction would be safe or even
work. The residual is measured along the *driver's* trajectory, not the one openpilot would
have driven. Feeding it back closes a loop that was never trained closed, and the usual
behavioural-cloning compounding error applies. Treat a good score as permission to design
the real thing (closed-loop replay, safety envelope retained), never as a shipping signal.

Usage:
    python -m openpilot.tools.drive_lab.residual_probe ROUTE [ROUTE ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Only manual, moving driving is informative: standstill is dominated by creep and the
# engaged frames are what we are trying to improve, not learn from.
MIN_V_EGO = 2.0
RIDGE_LAMBDA = 1e-3
MIN_SAMPLES = 200
MIN_ROUTES_FOR_CV = 2

FEATURES = (
  "v_ego",
  "has_lead",
  "d_rel",
  "v_rel",
  "inv_time_headway",
  "required_decel",
  "model_desired_accel",
)


@dataclass(frozen=True)
class ProbeResult:
  samples: int
  routes: int
  residual_mean: float
  residual_std: float
  cv_r2: float | None
  in_sample_r2: float
  coefficients: dict[str, float]
  regime_bias: dict[str, dict[str, float]]
  verdict: str
  notes: list[str]

  def to_dict(self) -> dict[str, Any]:
    return {
      "samples": self.samples,
      "routes": self.routes,
      "residual_mean": _r(self.residual_mean),
      "residual_std": _r(self.residual_std),
      "cv_r2": None if self.cv_r2 is None else _r(self.cv_r2, 4),
      "in_sample_r2": _r(self.in_sample_r2, 4),
      "coefficients": {k: _r(v, 4) for k, v in self.coefficients.items()},
      "regime_bias": {k: {kk: _r(vv) for kk, vv in v.items()} for k, v in self.regime_bias.items()},
      "verdict": self.verdict,
      "notes": list(self.notes),
    }


def _r(v: Any, ndigits: int = 3) -> float | None:
  try:
    f = float(v)
  except (TypeError, ValueError):
    return None
  return round(f, ndigits) if math.isfinite(f) else None


def _feature_row(s: Any) -> list[float] | None:
  """Physical features only. Anything non-finite drops the sample rather than imputing:
  a fabricated feature value would show up as fabricated signal."""
  v_ego = _f(getattr(s, "v_ego", None))
  if v_ego is None or v_ego < MIN_V_EGO:
    return None
  has_lead = 1.0 if bool(getattr(s, "lead_status", False)) else 0.0
  d_rel = _f(getattr(s, "lead_d_rel", None)) if has_lead else 0.0
  v_rel = _f(getattr(s, "lead_v_rel", None)) if has_lead else 0.0
  thw = _f(getattr(s, "time_headway_s", None))
  # 1/THW is the closing-urgency scale drivers actually respond to; THW itself explodes at
  # long range and would let far leads dominate the fit.
  inv_thw = (1.0 / thw) if (thw is not None and thw > 0.05) else 0.0
  req = _f(getattr(s, "required_decel_mps2", None)) or 0.0
  model_a = _f(getattr(s, "model_desired_accel", None)) or 0.0
  row = [v_ego, has_lead, d_rel if d_rel is not None else 0.0,
         v_rel if v_rel is not None else 0.0, inv_thw, req, model_a]
  return row if all(math.isfinite(x) for x in row) else None


def _f(v: Any) -> float | None:
  try:
    f = float(v)
  except (TypeError, ValueError):
    return None
  return f if math.isfinite(f) else None


def _is_manual_moving(s: Any) -> bool:
  # long_active means openpilot was driving; those frames are the thing under test, not
  # evidence about the driver.
  return not bool(getattr(s, "long_active", False)) and not bool(getattr(s, "standstill", False))


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
  """Ridge via lstsq on the augmented system — no new dependency for a 2-line solve."""
  n_features = X.shape[1]
  X_aug = np.vstack([X, math.sqrt(lam) * np.eye(n_features)])
  y_aug = np.concatenate([y, np.zeros(n_features)])
  coef, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
  return coef


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  ss_res = float(np.sum((y_true - y_pred) ** 2))
  ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
  if ss_tot <= 1e-12:
    return 0.0
  return 1.0 - ss_res / ss_tot


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  mu = X.mean(axis=0)
  sd = X.std(axis=0)
  sd[sd < 1e-9] = 1.0
  return (X - mu) / sd, mu, sd


def _regime_bias(rows: list[tuple[Any, float]]) -> dict[str, dict[str, float]]:
  """Mean residual by regime. This is the part that pays off even when R^2 is ~0: a
  consistent nonzero bias in one regime is a tuning target, stated in m/s^2."""
  regimes: dict[str, list[float]] = {}

  def add(name: str, value: float) -> None:
    regimes.setdefault(name, []).append(value)

  for s, res in rows:
    v = _f(getattr(s, "v_ego", None)) or 0.0
    lead = bool(getattr(s, "lead_status", False))
    add("all", res)
    add("lead" if lead else "no_lead", res)
    if v < 8.0:
      add("speed_0_8", res)
    elif v < 16.0:
      add("speed_8_16", res)
    else:
      add("speed_16_plus", res)
    if lead:
      thw = _f(getattr(s, "time_headway_s", None))
      if thw is not None and thw > 0.0:
        add("thw_lt_1.5" if thw < 1.5 else "thw_ge_1.5", res)

  return {
    name: {"n": float(len(vals)), "mean": float(np.mean(vals)), "p95_abs": float(np.percentile(np.abs(vals), 95))}
    for name, vals in sorted(regimes.items()) if vals
  }


def probe(samples: list[Any]) -> ProbeResult:
  notes: list[str] = []
  usable: list[tuple[Any, list[float], float]] = []
  for s in samples:
    if not _is_manual_moving(s):
      continue
    row = _feature_row(s)
    if row is None:
      continue
    a_ego = _f(getattr(s, "a_ego", None))
    plan = _f(getattr(s, "plan_a_target", None))
    if a_ego is None or plan is None:
      continue
    usable.append((s, row, a_ego - plan))

  if len(usable) < MIN_SAMPLES:
    return ProbeResult(len(usable), 0, 0.0, 0.0, None, 0.0, {}, {},
                       "INSUFFICIENT_DATA",
                       [f"need >= {MIN_SAMPLES} manual moving samples, got {len(usable)}"])

  X = np.array([r for _, r, _ in usable], dtype=float)
  y = np.array([res for _, _, res in usable], dtype=float)
  groups = [str(getattr(s, "route_id", "") or getattr(s, "route", "")) for s, _, _ in usable]
  unique_routes = sorted(set(groups))

  Xs, _, _ = _standardize(X)
  Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # intercept
  coef = _ridge(Xs, y)
  in_sample = _r2(y, Xs @ coef)

  # Leave-one-route-out CV. Grouping by route matters: consecutive frames are heavily
  # autocorrelated, so a random split leaks and inflates the score badly.
  cv_r2: float | None = None
  if len(unique_routes) >= MIN_ROUTES_FOR_CV:
    preds = np.full_like(y, np.nan)
    g = np.array(groups)
    for route in unique_routes:
      test = g == route
      train = ~test
      if train.sum() < MIN_SAMPLES // 2 or test.sum() < 10:
        continue
      Xtr, mu, sd = _standardize(X[train])
      Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
      Xte = np.hstack([(X[test] - mu) / sd, np.ones((int(test.sum()), 1))])
      preds[test] = Xte @ _ridge(Xtr, y[train])
    ok = np.isfinite(preds)
    if ok.sum() >= 10:
      cv_r2 = _r2(y[ok], preds[ok])
    else:
      notes.append("cross-validation produced too few predictions to score")
  else:
    notes.append(f"only {len(unique_routes)} route(s); leave-one-route-out CV needs >= {MIN_ROUTES_FOR_CV}")

  if cv_r2 is None:
    verdict = "UNSCORED"
  elif cv_r2 <= 0.02:
    verdict = "NO_SIGNAL"
    notes.append("residual is not predictable from scene features — a learned residual is not the missing piece")
  elif cv_r2 < 0.15:
    verdict = "WEAK_SIGNAL"
    notes.append("systematic but small; the regime bias table is the actionable part, not a model")
  else:
    verdict = "SIGNAL"
    notes.append("worth designing properly — closed-loop replay + retained safety envelope, not a direct correction")

  return ProbeResult(
    samples=len(usable),
    routes=len(unique_routes),
    residual_mean=float(np.mean(y)),
    residual_std=float(np.std(y)),
    cv_r2=cv_r2,
    in_sample_r2=in_sample,
    coefficients=dict(zip(FEATURES + ("intercept",), [float(c) for c in coef], strict=True)),
    regime_bias=_regime_bias([(s, res) for s, _, res in usable]),
    verdict=verdict,
    notes=notes,
  )


def render(result: ProbeResult) -> str:
  lines = [
    "Residual probe: is planner-vs-driver disagreement learnable?",
    f"  samples {result.samples} over {result.routes} route(s)",
    f"  residual (a_ego - plan_a_target): mean {result.residual_mean:+.3f} std {result.residual_std:.3f} m/s^2",
    f"  R^2 in-sample {result.in_sample_r2:.4f} | cross-validated "
    f"{'n/a' if result.cv_r2 is None else f'{result.cv_r2:.4f}'}",
    f"  VERDICT: {result.verdict}",
  ]
  for note in result.notes:
    lines.append(f"    note: {note}")
  if result.regime_bias:
    lines.append("  regime bias (mean residual, m/s^2 — positive = driver accelerated more than the plan):")
    for name, stats in result.regime_bias.items():
      lines.append(f"    {name:<16} n={int(stats['n']):>7}  mean {stats['mean']:+.3f}  p95|.| {stats['p95_abs']:.3f}")
  if result.coefficients:
    lines.append("  standardized coefficients:")
    for name, c in sorted(result.coefficients.items(), key=lambda kv: -abs(kv[1])):
      lines.append(f"    {name:<20} {c:+.4f}")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("routes", nargs="+")
  parser.add_argument("--qlog", action="store_true")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  from openpilot.tools.drive_lab.compare_manual_planner_targets import extract_planner_target_samples
  from openpilot.tools.lib.logreader import ReadMode

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  samples: list[Any] = []
  for route in args.routes:
    samples.extend(extract_planner_target_samples(route, read_mode))

  result = probe(samples)
  print(json.dumps(result.to_dict(), indent=2) if args.json else render(result))


if __name__ == "__main__":
  main()
