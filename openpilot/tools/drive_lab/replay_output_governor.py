#!/usr/bin/env python3
"""Replay the torque v2.1 OutputGovernor over full rlogs.

This is a fixed-trace diagnostic.  It replays the governor inputs that are logged in
``controlsState`` and compares G0 with the logged governed output.  G1--G3 and P2 are
counterfactual governor variants; none of them changes production code or feeds a
counterfactual torque back into the trace.

Times passed to ``--start``/``--end`` are seconds from the first loaded message.  The
route is replayed from its beginning even when a reporting window is selected, so the
stateful governor has its real preceding trace.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpilot.sunnypilot.custom.lateral.output_governor import (
  GovernorReason,
  OutputGovernor,
  OutputGovernorInputs,
  SLEW_RATE_SCALE_STEP,
  _PythonHelperSet,
)
from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import (
  DEFAULT_LOG_ROOTS,
  resolve_inputs,
)
from openpilot.tools.lib.logreader import LogReader, ReadMode


DT = 0.01
DEFAULT_LAT_DELAY = 0.01
NOMINAL_SIGN_EPS = 1e-6
REASON_NAMES = {reason.value: reason.name for reason in GovernorReason}


@dataclass(frozen=True)
class GovernorFrame:
  t: float
  active: bool
  nominal_torque: float
  logged_output: float
  logged_reason: int
  logged_cap: float
  v_ego: float
  steering_rate_deg: float
  steering_pressed: bool
  desired_lateral_accel: float
  actual_lateral_accel: float
  same_direction_limit: bool
  controller_evidence_stable: bool
  path_evidence_valid: bool
  lateral_accel_error_rate: float
  lat_delay: float


@dataclass(frozen=True)
class ReplaySample:
  t: float
  active: bool
  nominal_torque: float
  output_torque: float
  logged_output: float
  reason: int
  cap: float
  logged_reason: int
  logged_cap: float


class _NoSlewHelperSet(_PythonHelperSet):
  """Production helper set with only the final slew approach replaced.

  OutputGovernor still performs its production floor, caps, clipping, and target-arrival
  blend.  Resetting the local previous output before each active update also removes the
  production sign-change unwind, which is part of the slew stage rather than a cap.
  """

  @staticmethod
  def approach(value: float, target: float, step: float) -> float:
    return target


class NoSlewGovernor(OutputGovernor):
  def __init__(self) -> None:
    super().__init__(DT, _use_cython=False)
    self._helper_set = _NoSlewHelperSet

  def update(self, inp: OutputGovernorInputs):
    # The pre-slew governor has no dependence on previous_output.  Zeroing it avoids
    # sign-change unwind while retaining the production dt and all pre-slew behavior.
    self.previous_output = 0.0
    return super().update(inp)


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
  current = obj
  for part in path.split("."):
    try:
      current = getattr(current, part)
    except Exception:
      return default
  return current


def _finite_float(value: Any, default: float = math.nan) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return default
  return result if math.isfinite(result) else default


def _latest_delay(value: Any) -> float:
  delay = _finite_float(value, DEFAULT_LAT_DELAY)
  return max(delay, DT) if delay >= 0.0 else DEFAULT_LAT_DELAY


def _union_name(obj: Any) -> str:
  try:
    which = obj.which
    return str(which()) if callable(which) else ""
  except Exception:
    return ""


def _sign(value: float) -> int:
  if value > NOMINAL_SIGN_EPS:
    return 1
  if value < -NOMINAL_SIGN_EPS:
    return -1
  return 0


def _finite_difference(value: float, t: float, previous: tuple[float, float] | None) -> tuple[float, tuple[float, float]]:
  if previous is None:
    return 0.0, (t, value)
  previous_t, previous_value = previous
  delta_t = t - previous_t
  if not math.isfinite(delta_t) or delta_t <= 0.0 or not math.isfinite(previous_value):
    return 0.0, (t, value)
  return (value - previous_value) / delta_t, (t, value)


def _sort_log_paths(paths: Iterable[str]) -> list[str]:
  """Sort explicit segment paths by numeric ``--N`` suffix, never lexically."""
  def key(path: str) -> tuple[str, int, str]:
    parent = Path(path).parent.name
    try:
      route, segment = parent.rsplit("--", 1)
      return route, int(segment), str(path)
    except ValueError:
      return parent, -1, str(path)
  return sorted(dict.fromkeys(paths), key=key)


def _make_input(frame: GovernorFrame, *, same_direction_limit: bool | None = None) -> OutputGovernorInputs:
  return OutputGovernorInputs(
    active=frame.active,
    v_ego=frame.v_ego,
    steering_rate_deg=frame.steering_rate_deg,
    nominal_torque=frame.nominal_torque,
    max_output=1.0,
    desired_lateral_accel=frame.desired_lateral_accel,
    actual_lateral_accel=frame.actual_lateral_accel,
    same_direction_limit=frame.same_direction_limit if same_direction_limit is None else same_direction_limit,
    release_active=frame.steering_pressed,
    path_evidence_valid=frame.path_evidence_valid,
    controller_evidence_stable=frame.controller_evidence_stable,
    lateral_accel_error_rate=frame.lateral_accel_error_rate,
    lat_delay=frame.lat_delay,
    # Holding torque is not logged.  None deliberately disables target-arrival blending.
    holding_torque=None,
  )


def _extract_frames(messages: Iterable[Any]) -> list[GovernorFrame]:
  ordered = sorted(messages, key=lambda msg: int(getattr(msg, "logMonoTime", 0)))
  if not ordered:
    return []
  first_time = int(getattr(ordered[0], "logMonoTime", 0))
  v_ego = math.nan
  steering_pressed = False
  steering_rate_deg = math.nan
  lat_delay = DEFAULT_LAT_DELAY
  previous_actual: tuple[float, float] | None = None
  frames: list[GovernorFrame] = []

  for message in ordered:
    t = (int(getattr(message, "logMonoTime", 0)) - first_time) / 1e9
    which = message.which()
    if which == "carState":
      car_state = message.carState
      v_ego = _finite_float(_safe_get(car_state, "vEgo"))
      steering_pressed = bool(_safe_get(car_state, "steeringPressed", False))
      steering_rate_deg = -_finite_float(_safe_get(car_state, "steeringRateDeg"))
      continue
    elif which == "liveDelay":
      lat_delay = _latest_delay(_safe_get(message.liveDelay, "lateralDelay", DEFAULT_LAT_DELAY))
      continue
    elif which != "controlsState":
      continue

    state = message.controlsState
    lateral_state = _safe_get(state, "lateralControlState")
    union_name = _union_name(lateral_state)
    # The capnp union is named ``torqueState`` in the checked-in schema; the
    # production field is commonly referred to as lateralTorqueState.
    if union_name not in ("lateralTorqueState", "torqueState"):
      continue
    torque_state = _safe_get(lateral_state, union_name)
    adaptive = _safe_get(torque_state, "adaptiveTorqueState")
    actual = _finite_float(_safe_get(torque_state, "actualLateralAccel"))
    measured_rate, previous_actual = _finite_difference(actual, t, previous_actual)
    path_invalid = bool(_safe_get(adaptive, "underResponseGuardPathEvidenceInvalid", False))
    nominal_output = _finite_float(_safe_get(adaptive, "nominalOutput"))
    frames.append(GovernorFrame(
      t=t,
      active=bool(_safe_get(torque_state, "active", False)),
      nominal_torque=-nominal_output,
      logged_output=_finite_float(_safe_get(torque_state, "output")),
      # SLEW_SCALE_APPLIED is a telemetry-only condition marker torque_v2_1 ORs into the
      # logged reason; the replayed governor never sets it, so mask it from parity.
      logged_reason=int(_safe_get(adaptive, "governorReason", 0) or 0) & ~int(GovernorReason.SLEW_SCALE_APPLIED),
      logged_cap=_finite_float(_safe_get(adaptive, "outputCap")),
      v_ego=v_ego,
      steering_rate_deg=steering_rate_deg,
      steering_pressed=steering_pressed,
      desired_lateral_accel=_finite_float(_safe_get(torque_state, "desiredLateralAccel")),
      actual_lateral_accel=actual,
      same_direction_limit=bool(_safe_get(adaptive, "steerLimitSameDirection", False)),
      controller_evidence_stable=not (
        bool(_safe_get(adaptive, "responseCoreSameSignUnwind", False)) or
        bool(_safe_get(adaptive, "responseCoreMeasurementReset", False))
      ),
      path_evidence_valid=not path_invalid,
      lateral_accel_error_rate=_finite_float(_safe_get(torque_state, "desiredLateralJerk")) - measured_rate,
      lat_delay=lat_delay,
    ))
  return frames


def _replay(frames: list[GovernorFrame]) -> dict[str, list[ReplaySample]]:
  governors: dict[str, OutputGovernor] = {
    "G0": OutputGovernor(DT),
    "G1": OutputGovernor(DT),
    "G2": NoSlewGovernor(),
    "P2": OutputGovernor(DT, slew_rate_scale=SLEW_RATE_SCALE_STEP),
  }
  samples = {name: [] for name in (*governors, "G3")}
  for frame in frames:
    production_input = _make_input(frame)
    result_g0 = governors["G0"].update(production_input)
    result_g1 = governors["G1"].update(_make_input(frame, same_direction_limit=False))
    result_g2 = governors["G2"].update(production_input)
    result_p2 = governors["P2"].update(production_input)
    results = {"G0": result_g0, "G1": result_g1, "G2": result_g2, "P2": result_p2}
    for name, result in results.items():
      samples[name].append(ReplaySample(frame.t, frame.active, frame.nominal_torque, result.output_torque,
                                        frame.logged_output, result.reason, result.cap,
                                        frame.logged_reason, frame.logged_cap))
    samples["G3"].append(ReplaySample(frame.t, frame.active, frame.nominal_torque, frame.nominal_torque,
                                      frame.logged_output, int(GovernorReason.NONE), 1.0,
                                      frame.logged_reason, frame.logged_cap))
  return samples


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  if len(values) == 1:
    return values[0]
  return float(statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1])


def _rms(values: list[float]) -> float | None:
  return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _window_samples(samples: list[ReplaySample], start: float | None, end: float | None) -> list[ReplaySample]:
  return [sample for sample in samples if (start is None or sample.t >= start) and (end is None or sample.t <= end)]


def _reversal_durations(samples: list[ReplaySample]) -> tuple[int, int, list[float]]:
  nominal_reversals = 0
  output_reversals = 0
  durations: list[float] = []
  previous_nominal = 0
  previous_output = 0
  output_signs = [_sign(sample.output_torque) for sample in samples]
  run_end = [len(samples)] * len(samples)
  for index in range(len(samples) - 2, -1, -1):
    if output_signs[index] == output_signs[index + 1]:
      run_end[index] = run_end[index + 1]
    else:
      run_end[index] = index + 1
  for index, sample in enumerate(samples):
    nominal_sign = _sign(sample.nominal_torque)
    output_sign = output_signs[index]
    if previous_nominal and nominal_sign and nominal_sign != previous_nominal:
      nominal_reversals += 1
      old_sign = previous_nominal
      durations.append((run_end[index] - index) * DT if output_sign == old_sign else 0.0)
    if previous_output and output_sign and output_sign != previous_output:
      output_reversals += 1
    if nominal_sign:
      previous_nominal = nominal_sign
    if output_sign:
      previous_output = output_sign
  return nominal_reversals, output_reversals, durations


def _g0_report(samples: list[ReplaySample]) -> dict[str, Any]:
  valid = [sample for sample in samples if math.isfinite(sample.logged_output) and math.isfinite(sample.output_torque)]
  errors = [abs(-sample.output_torque - sample.logged_output) for sample in valid]
  replay_reasons = [sample.reason for sample in valid]
  reason_matches = [sample.reason == sample.logged_reason for sample in valid]
  cap_errors = [abs(sample.cap - sample.logged_cap) for sample in valid if sample.active
                if math.isfinite(sample.cap) and math.isfinite(sample.logged_cap)]
  return {
    "sample_count": len(valid),
    "absolute_error": {
      "p50": _percentile(errors, 50),
      "p95": _percentile(errors, 95),
      "max": max(errors) if errors else None,
    },
    "reason_agreement": {
      "available": bool(reason_matches),
      "exact_count": sum(reason_matches),
      "exact_ratio": sum(reason_matches) / len(reason_matches) if reason_matches else None,
      "note": "exact bitfield comparison; target-arrival can differ because holding torque is not logged",
      "replay_reason_counts": {name: sum(1 for reason in replay_reasons if reason & value)
                               for value, name in REASON_NAMES.items()},
    },
    "cap_agreement": {
      "sample_count": len(cap_errors),
      "absolute_error_p95": _percentile(cap_errors, 95),
      "absolute_error_max": max(cap_errors) if cap_errors else None,
    },
  }


def _variant_report(samples: list[ReplaySample], start: float | None, end: float | None) -> dict[str, Any]:
  selected = _window_samples(samples, start, end)
  deltas = [sample.output_torque - sample.nominal_torque for sample in selected
            if math.isfinite(sample.output_torque) and math.isfinite(sample.nominal_torque)]
  nominal_reversals, output_reversals, durations = _reversal_durations(selected)
  return {
    "sample_count": len(selected),
    "output_minus_nominal_rms": _rms(deltas),
    "reversal_count": nominal_reversals,
    "output_reversal_count": output_reversals,
    "old_direction_torque_time_s": {
      "median": statistics.median(durations) if durations else None,
      "p95": _percentile(durations, 95),
      "samples": len(durations),
    },
  }


def _format(value: Any) -> str:
  return "n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def _render_report(source: str, identifiers: list[str], frames: list[GovernorFrame],
                   samples: dict[str, list[ReplaySample]], start: float | None, end: float | None) -> str:
  g0 = _g0_report(samples["G0"])
  lines = [
    "OutputGovernor fixed-trace replay",
    f"source: {source}",
    f"rlog segments: {len(identifiers)} (numeric --N ordering)",
    f"governor frames: {len(frames)}",
    "path evidence: inferred valid unless underResponseGuardPathEvidenceInvalid is logged; approximation only.",
    "holding torque: unavailable in rlogs; target-arrival blending is disabled rather than guessed.",
    "",
    "G0 production replay vs logged controlsState.lateralTorqueState.output",
    f"  samples={g0['sample_count']} abs error p50={_format(g0['absolute_error']['p50'])} "
    + f"p95={_format(g0['absolute_error']['p95'])} max={_format(g0['absolute_error']['max'])}",
    f"  reason exact agreement={_format(g0['reason_agreement']['exact_ratio'])} "
    + f"({g0['reason_agreement']['note']})",
    f"  cap abs error p95={_format(g0['cap_agreement']['absolute_error_p95'])} "
    + f"max={_format(g0['cap_agreement']['absolute_error_max'])}",
  ]
  if start is not None or end is not None:
    lines.append(f"  window: {_format(start)}-{_format(end)}s from first loaded message")
  for name in ("G0", "G1", "G2", "G3", "P2"):
    report = _variant_report(samples[name], start, end)
    reversal = report["old_direction_torque_time_s"]
    label = {
      "G1": "same_direction_limit=False",
      "G2": "slew bypass; production caps/arrival retained",
      "G3": "output=nominal",
      "G0": "production governor",
      "P2": f"build slew x{SLEW_RATE_SCALE_STEP} (LateralSlewScaleMode apply; sign/release baseline)",
    }[name]
    lines.extend([
      "",
      f"{name} ({label}):",
      f"  samples={report['sample_count']} output-nominal RMS={_format(report['output_minus_nominal_rms'])}",
      f"  nominal reversal count={report['reversal_count']} output reversal count={report['output_reversal_count']}",
      f"  old-direction torque-time median={_format(reversal['median'])}s "
      + f"p95={_format(reversal['p95'])}s (events={reversal['samples']})",
    ])
  lines.extend(("", "Caveat: fixed-trace replay cannot establish closed-loop damping or stability;",
                "it measures only counterfactual output on the logged demand/measurement trace."))
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay the production OutputGovernor over full rlogs.")
  parser.add_argument("inputs", nargs="+", help="Route ID/name, local route directory, or explicit rlog path")
  parser.add_argument("--log-root", action="append", default=[], help="Extra root for local short routes")
  parser.add_argument("--start", type=float, help="Report window start in seconds from first loaded message")
  parser.add_argument("--end", type=float, help="Report window end in seconds from first loaded message")
  parser.add_argument("--json", action="store_true", help="Print the report as JSON")
  args = parser.parse_args()
  if args.start is not None and args.end is not None and args.start > args.end:
    parser.error("--start must be <= --end")

  log_roots = tuple(Path(path) for path in args.log_root) + DEFAULT_LOG_ROOTS
  identifiers = _sort_log_paths(resolve_inputs(args.inputs, segment=None, read_mode=ReadMode.RLOG, log_roots=log_roots))
  frames = _extract_frames(LogReader(identifiers, default_mode=ReadMode.RLOG, sort_by_time=True))
  samples = _replay(frames)
  if args.json:
    report = {
      "source": args.inputs,
      "identifiers": identifiers,
      "frames": len(frames),
      "window": {"start": args.start, "end": args.end},
      "g0": _g0_report(samples["G0"]),
      "window_variants": {name: _variant_report(samples[name], args.start, args.end)
                          for name in ("G0", "G1", "G2", "G3", "P2")},
      "caveats": [
        "path evidence is inferred from underResponseGuardPathEvidenceInvalid",
        "holding torque is unavailable and target-arrival blending is disabled",
        "fixed-trace replay cannot establish closed-loop damping or stability",
      ],
    }
    print(json.dumps(report, indent=2))
  else:
    print(_render_report(", ".join(args.inputs), identifiers, frames, samples, args.start, args.end))


if __name__ == "__main__":
  main()
