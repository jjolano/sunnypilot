#!/usr/bin/env python3
"""Offline comparison of decoupled fallback state and causal unwind preview."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SPIKE_1_PATH = Path(__file__).parents[1] / "001-corner-safe-demand-jerk" / "experiment.py"
SPEC = importlib.util.spec_from_file_location("corner_safe_demand_jerk_spike", SPIKE_1_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError(f"cannot load {SPIKE_1_PATH}")
spike1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spike1
SPEC.loader.exec_module(spike1)

from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import ModelPathProcessor
from openpilot.sunnypilot.custom.lateral.demand.pipeline import LateralDemandPipeline
from openpilot.sunnypilot.custom.lateral.demand.preview import PreviewAssistTracker


VARIANTS = {
  "decoupled_lag08": ("decoupled", 0.08),
  "decoupled_lag20": ("decoupled", 0.20),
  "anticipatory_100ms_lag20": ("preview100", 0.20),
  "anticipatory_200ms_lag20": ("preview200", 0.20),
  "anticipatory_200ms_release_lag20": ("preview200_release", 0.20),
}
PREVIEW_HORIZONS = {
  "decoupled": 0.0,
  "preview100": 0.10,
  "preview200": 0.20,
  "preview200_release": 0.20,
}


class DecoupledCornerProcessor(spike1.CornerSafeProcessor):
  """Keep fallback truth unshaped; optionally aim the same shaper at a causal unwind preview."""

  def __init__(self, mode: str, lag_lat_accel: float) -> None:
    self.mode = mode
    self.preview_horizon = PREVIEW_HORIZONS[mode]
    self.soft_release = mode.endswith("_release")
    self._fallback_reference = 0.0
    self._preview_helper = PreviewAssistTracker(spike1.DT)
    super().__init__("post_gate", lag_lat_accel)

  def reset(self) -> None:
    super().reset()
    self._fallback_reference = 0.0
    if hasattr(self, "_preview_helper"):
      self._preview_helper.reset()

  def update(self, inputs: spike1.ModelPathProcessorInputs):
    # The model-path processor sees only its own unshaped prior result. Shaped controller
    # demand can no longer compound through low-quality fallback on the next frame.
    result = ModelPathProcessor.update(self, replace(
      inputs,
      previous_desired_curvature=self._fallback_reference,
      demand_jerk_smoothing_enabled=False,
    ))
    reference = float(result.desired_curvature)
    self._fallback_reference = reference

    eligible = self._demand_jerk_smoothing_eligible(
      inputs, reference, reference, result.quality, result.reason, None,
    )
    if not eligible and not self._can_soft_release(inputs, result):
      self._post_smoothed_curvature = None
      return result

    target = self._unwind_target(inputs, reference) if eligible else reference
    candidate, active, step, lag = self._shape_against_reference(
      self._post_smoothed_curvature, target, reference, float(inputs.v_ego),
    )
    self._post_smoothed_curvature = candidate if active or eligible else None
    return replace(
      result,
      desired_curvature=candidate,
      demand_jerk_smoothing_active=active,
      demand_jerk_smoothing_step=step,
      demand_jerk_smoothing_lag=lag,
    )

  def _can_soft_release(self, inputs: Any, result: Any) -> bool:
    return (
      self.soft_release
      and self._post_smoothed_curvature is not None
      and bool(inputs.lat_active)
      and bool(inputs.demand_jerk_smoothing_enabled)
      and bool(inputs.demand_jerk_smoothing_allowed)
      and bool(inputs.smooth_model_path_curvature)
      and not bool(inputs.lane_change_active)
      and result.reason in spike1.ALLOWED_REASONS
      and math.isfinite(float(inputs.v_ego))
      and math.isfinite(float(result.desired_curvature))
    )

  def _unwind_target(self, inputs: Any, reference: float) -> float:
    if self.preview_horizon <= 0.0:
      return reference
    v_ego = max(float(inputs.v_ego), 1.0)
    preview = self._preview_helper._preview_curvature(inputs, self.preview_horizon, v_ego)
    if preview is None:
      return reference
    reference_ay = reference * v_ego * v_ego
    preview_ay = preview * v_ego * v_ego
    same_direction = reference_ay * preview_ay >= 0.0
    predicted_unwind = abs(preview_ay) + 0.01 < abs(reference_ay)
    return float(preview) if same_direction and predicted_unwind else reference

  def _shape_against_reference(
    self,
    previous: float | None,
    target: float,
    reference: float,
    v_ego: float,
  ) -> tuple[float, bool, float, float]:
    v_ego = max(v_ego, 1.0)
    v_sq = v_ego * v_ego
    max_lat_jerk = float(np.interp(
      v_ego, spike1.DEMAND_JERK_SMOOTH_SPEED_BP, spike1.DEMAND_JERK_SMOOTH_MAX_LAT_JERK,
    ))
    max_step = max_lat_jerk * spike1.DT / v_sq
    lag_limit = self.lag_lat_accel / v_sq

    if previous is None or not math.isfinite(previous):
      return reference, False, max_step, 0.0
    previous = float(previous)
    if previous * reference < 0.0 and min(abs(previous), abs(reference)) * v_sq > spike1.SIGN_MISMATCH_LAT_ACCEL:
      return reference, False, max_step, 0.0

    delta = target - previous
    candidate = target if abs(delta) <= max_step else previous + math.copysign(max_step, delta)
    candidate = self._clamp_demand_jerk_candidate(
      previous, candidate, target, reference, lag_limit, raw_cap_active=True,
    )
    lag = abs(candidate - reference)
    return candidate, lag > 1e-9, max_step, lag


def _pipeline(variant: tuple[str, float] | None) -> LateralDemandPipeline:
  pipeline = LateralDemandPipeline(spike1.DT)
  if variant is not None:
    pipeline._model_path_processor = DecoupledCornerProcessor(*variant)
  return pipeline


def analyze(routes: list[Path]) -> dict[str, Any]:
  old_variants, old_pipeline = spike1.VARIANTS, spike1._pipeline
  spike1.VARIANTS, spike1._pipeline = VARIANTS, _pipeline
  try:
    return spike1.analyze(routes)
  finally:
    spike1.VARIANTS, spike1._pipeline = old_variants, old_pipeline


def self_check() -> None:
  proc = DecoupledCornerProcessor("preview200", 0.20)
  v_ego = 10.0
  reference = 0.004
  candidate = reference
  for _ in range(100):
    candidate, _, _, lag = proc._shape_against_reference(candidate, 0.0, reference, v_ego)
    assert lag <= 0.20 / (v_ego * v_ego) + 1e-12
  assert candidate == 0.002
  for _ in range(100):
    candidate, _, _, lag = proc._shape_against_reference(candidate, reference, reference, v_ego)
  assert candidate == reference and lag == 0.0

  reversed_out = proc._shape_against_reference(0.004, -0.004, -0.004, v_ego)
  assert reversed_out[0] == -0.004 and not reversed_out[1]
  print("self-check passed")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("routes", nargs="*", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--self-check", action="store_true")
  args = parser.parse_args()
  if args.self_check:
    self_check()
    return
  if not args.routes:
    parser.error("provide at least one cached route directory")
  missing = [str(path) for path in args.routes if not path.is_dir()]
  if missing:
    parser.error(f"missing route directories: {', '.join(missing)}")
  report = analyze(args.routes)
  if args.output:
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(spike1.render(report))


if __name__ == "__main__":
  main()
