# Manual Longitudinal Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Drive Lab tooling that measures route-derived manual longitudinal style and classifies a conservative `smooth_assertive` profile from one or more mostly manual routes.

**Architecture:** Add a focused profiler module with dataclasses and pure helpers for route inclusion, manual sample filtering, launch/stop/coast summaries, and style classification. Add a thin CLI that reads routes or log paths through LogReader, prints text or JSON, and writes output only when requested. Product behavior tuning remains out of scope for this plan and will be handled by later branch-owned plans.

**Tech Stack:** Python 3.12, pytest via `uv run`, openpilot LogReader/qlog data, existing `tools/drive_lab` package conventions.

---

## File Structure

- Create `tools/drive_lab/manual_longitudinal_profile.py`.
  - Owns `ProfileRange`, `ManualSample`, `RouteProfile`, `ManualStyleSummary`, percentile helpers, route inclusion, manual sample filtering, launch/stop/coast buckets, and `smooth_assertive` classification.
- Create `tools/drive_lab/profile_manual_longitudinal.py`.
  - CLI entrypoint that reads routes/log paths, extracts samples, filters included routes, prints summaries, and optionally writes JSON.
- Create `tools/drive_lab/tests/test_manual_longitudinal_profile.py`.
  - Deterministic synthetic-message tests for all pure profiler behavior and renderer output.

## Task 1: Profile Ranges And Classification

**Files:**
- Create: `tools/drive_lab/manual_longitudinal_profile.py`
- Create: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/drive_lab/tests/test_manual_longitudinal_profile.py` with:

```python
from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ProfileRange,
  SmoothAssertiveEnvelope,
  classify_style,
  percentile_range,
)


def test_percentile_range_uses_requested_percentiles():
  result = percentile_range([0.0, 1.0, 2.0, 3.0, 4.0], low_pct=25.0, high_pct=75.0)

  assert result == ProfileRange(low=1.0, high=3.0)


def test_classifies_smooth_assertive_profile_inside_envelope():
  style = classify_style(
    accel=ProfileRange(-0.815, 0.917),
    launch_mean=ProfileRange(0.687, 0.932),
    stop_mean=ProfileRange(-0.890, -0.409),
    coast_accel=ProfileRange(-0.336, -0.294),
    envelope=SmoothAssertiveEnvelope(),
  )

  assert style == "smooth_assertive"


def test_classifies_unknown_when_profile_is_too_aggressive():
  style = classify_style(
    accel=ProfileRange(-2.5, 2.8),
    launch_mean=ProfileRange(1.6, 2.4),
    stop_mean=ProfileRange(-2.2, -1.6),
    coast_accel=ProfileRange(-0.8, -0.6),
    envelope=SmoothAssertiveEnvelope(),
  )

  assert style == "unknown"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `manual_longitudinal_profile` symbols.

- [ ] **Step 3: Implement the minimal model**

Create `tools/drive_lab/manual_longitudinal_profile.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ProfileRange:
  low: float
  high: float

  def contains(self, value: float) -> bool:
    return self.low <= value <= self.high


@dataclass(frozen=True)
class SmoothAssertiveEnvelope:
  accel_p10: ProfileRange = ProfileRange(-1.20, -0.45)
  accel_p90: ProfileRange = ProfileRange(0.55, 1.25)
  launch_mean_p50: ProfileRange = ProfileRange(0.45, 0.90)
  launch_mean_p90: ProfileRange = ProfileRange(0.75, 1.25)
  stop_mean_p10: ProfileRange = ProfileRange(-1.25, -0.55)
  stop_mean_p50: ProfileRange = ProfileRange(-0.60, -0.20)
  coast_p50: ProfileRange = ProfileRange(-0.45, -0.18)
  coast_p90: ProfileRange = ProfileRange(-0.20, 0.05)


def clean_finite(values: Iterable[float]) -> list[float]:
  return [float(value) for value in values if isinstance(value, int | float) and isfinite(float(value))]


def percentile_range(values: Iterable[float], low_pct: float, high_pct: float) -> ProfileRange:
  clean = clean_finite(values)
  if not clean:
    return ProfileRange(0.0, 0.0)
  return ProfileRange(float(np.percentile(clean, low_pct)), float(np.percentile(clean, high_pct)))


def classify_style(accel: ProfileRange, launch_mean: ProfileRange, stop_mean: ProfileRange,
                   coast_accel: ProfileRange, envelope: SmoothAssertiveEnvelope | None = None) -> str:
  envelope = envelope or SmoothAssertiveEnvelope()
  if not envelope.accel_p10.contains(accel.low):
    return "unknown"
  if not envelope.accel_p90.contains(accel.high):
    return "unknown"
  if not envelope.launch_mean_p50.contains(launch_mean.low):
    return "unknown"
  if not envelope.launch_mean_p90.contains(launch_mean.high):
    return "unknown"
  if not envelope.stop_mean_p10.contains(stop_mean.low):
    return "unknown"
  if not envelope.stop_mean_p50.contains(stop_mean.high):
    return "unknown"
  if not envelope.coast_p50.contains(coast_accel.low):
    return "unknown"
  if not envelope.coast_p90.contains(coast_accel.high):
    return "unknown"
  return "smooth_assertive"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: add manual longitudinal profile model"
```

## Task 2: Manual Samples And Route Inclusion

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Add failing route inclusion tests**

Append to `tools/drive_lab/tests/test_manual_longitudinal_profile.py`:

```python
from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  build_route_profile,
)


def sample(t, v, a, active=False, gas=False, brake=False, lead=False, d_rel=0.0, v_rel=0.0):
  return ManualSample(
    route="route-a",
    t=t,
    v_ego=v,
    a_ego=a,
    active=active,
    gas_pressed=gas,
    brake_pressed=brake,
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
  )


def test_route_profile_includes_mostly_manual_route():
  samples = [sample(float(i), 8.0, 0.1, active=False) for i in range(20)]
  samples += [sample(20.0, 8.0, 0.1, active=True)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=10, max_active_ratio=0.25)

  assert profile.include
  assert profile.manual_moving_samples == 20
  assert profile.active_ratio == 1 / 21


def test_route_profile_excludes_routes_with_too_much_active_control():
  samples = [sample(float(i), 8.0, 0.1, active=False) for i in range(10)]
  samples += [sample(float(i + 10), 8.0, 0.1, active=True) for i in range(10)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=5, max_active_ratio=0.25)

  assert not profile.include
  assert profile.active_ratio == 0.5


def test_route_profile_ignores_stopped_samples_for_manual_moving_count():
  samples = [sample(float(i), 0.2, 0.0, active=False) for i in range(20)]
  samples += [sample(float(i + 20), 6.0, 0.1, active=False) for i in range(6)]

  profile = build_route_profile("route-a", samples, min_manual_moving_samples=10, max_active_ratio=0.25)

  assert not profile.include
  assert profile.manual_moving_samples == 6
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: FAIL with import errors for `ManualSample` and `build_route_profile`.

- [ ] **Step 3: Implement manual samples and route profile**

Add to `tools/drive_lab/manual_longitudinal_profile.py` after `SmoothAssertiveEnvelope`:

```python
@dataclass(frozen=True)
class ManualSample:
  route: str
  t: float
  v_ego: float
  a_ego: float
  active: bool
  gas_pressed: bool
  brake_pressed: bool
  lead_status: bool
  lead_d_rel: float | None = None
  lead_v_rel: float | None = None


@dataclass(frozen=True)
class RouteProfile:
  route: str
  samples: int
  manual_moving_samples: int
  active_ratio: float
  include: bool


def manual_moving_samples(samples: Iterable[ManualSample]) -> list[ManualSample]:
  return [sample for sample in samples if not sample.active and sample.v_ego > 1.0]


def build_route_profile(route: str, samples: Iterable[ManualSample], min_manual_moving_samples: int = 1200,
                        max_active_ratio: float = 0.25) -> RouteProfile:
  sample_list = list(samples)
  active_count = sum(1 for sample in sample_list if sample.active)
  active_ratio = active_count / len(sample_list) if sample_list else 1.0
  moving_count = len(manual_moving_samples(sample_list))
  include = moving_count >= min_manual_moving_samples and active_ratio <= max_active_ratio
  return RouteProfile(route, len(sample_list), moving_count, active_ratio, include)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: PASS, `6 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: classify mostly manual routes"
```

## Task 3: Launch, Stop, Coast, And Following Buckets

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Add failing bucket tests**

Append to `tools/drive_lab/tests/test_manual_longitudinal_profile.py`:

```python
from openpilot.tools.drive_lab.manual_longitudinal_profile import summarize_manual_style


def test_manual_style_summary_separates_lead_and_clear_launches():
  samples = [
    sample(0.0, 0.5, 1.4, gas=True, lead=True, d_rel=4.0),
    sample(1.0, 3.0, 0.8, gas=True, lead=True, d_rel=4.5),
    sample(2.0, 6.0, 0.4, gas=False, lead=True, d_rel=5.0),
    sample(10.0, 0.3, 1.8, gas=True, lead=False),
    sample(11.0, 4.0, 1.0, gas=True, lead=False),
    sample(12.0, 7.0, 0.2, gas=False, lead=False),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_launch_count == 1
  assert summary.clear_launch_count == 1
  assert summary.lead_launch_mean_accel.low == summary.lead_launch_mean_accel.high == 1.1
  assert summary.clear_launch_peak_accel.low == summary.clear_launch_peak_accel.high == 1.8


def test_manual_style_summary_counts_stop_approaches_and_coast():
  samples = [
    sample(0.0, 10.0, -0.6, brake=True, lead=True, d_rel=12.0, v_rel=-1.0),
    sample(1.0, 6.0, -0.4, brake=True, lead=True, d_rel=10.0, v_rel=-0.5),
    sample(2.0, 0.5, -0.1, brake=False, lead=True, d_rel=8.0, v_rel=0.0),
    sample(10.0, 14.0, -0.3, gas=False, brake=False),
    sample(11.0, 15.0, -0.4, gas=False, brake=False),
    sample(12.0, 16.0, -0.2, gas=True, brake=False),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_stop_count == 1
  assert summary.clear_stop_count == 0
  assert summary.stop_mean_accel.low == summary.stop_mean_accel.high == -0.5
  assert summary.coast_accel.low == -0.4
  assert summary.coast_accel.high == -0.3
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: FAIL with import error for `summarize_manual_style`.

- [ ] **Step 3: Implement summary buckets**

Add to `tools/drive_lab/manual_longitudinal_profile.py`:

```python
@dataclass(frozen=True)
class ManualStyleSummary:
  sample_count: int
  accel: ProfileRange
  lead_launch_count: int
  clear_launch_count: int
  lead_launch_mean_accel: ProfileRange
  clear_launch_mean_accel: ProfileRange
  lead_launch_peak_accel: ProfileRange
  clear_launch_peak_accel: ProfileRange
  lead_stop_count: int
  clear_stop_count: int
  stop_mean_accel: ProfileRange
  stop_peak_decel: ProfileRange
  coast_accel: ProfileRange
  style: str


def summarize_manual_style(samples: Iterable[ManualSample]) -> ManualStyleSummary:
  ordered = sorted(manual_moving_samples(samples), key=lambda sample: sample.t)
  launches = _pedal_episodes(ordered, pedal="gas_pressed")
  stops = _pedal_episodes(ordered, pedal="brake_pressed")
  launch_candidates = [episode for episode in launches if episode["v0"] < 1.5 and episode["v1"] > 5.0 and episode["duration"] > 1.0]
  stop_candidates = [episode for episode in stops if episode["v0"] > 5.0 and episode["v1"] < 1.0 and episode["duration"] > 1.0]
  lead_launches = [episode for episode in launch_candidates if episode["lead"]]
  clear_launches = [episode for episode in launch_candidates if not episode["lead"]]
  lead_stops = [episode for episode in stop_candidates if episode["lead"]]
  clear_stops = [episode for episode in stop_candidates if not episode["lead"]]
  coast_samples = [sample for sample in ordered if not sample.gas_pressed and not sample.brake_pressed and sample.v_ego >= 7.0]

  accel = percentile_range([sample.a_ego for sample in ordered], 10.0, 90.0)
  lead_launch_mean = percentile_range([episode["mean_accel"] for episode in lead_launches], 50.0, 90.0)
  clear_launch_mean = percentile_range([episode["mean_accel"] for episode in clear_launches], 50.0, 90.0)
  stop_mean = percentile_range([episode["mean_accel"] for episode in stop_candidates], 10.0, 50.0)
  coast = percentile_range([sample.a_ego for sample in coast_samples], 50.0, 90.0)
  style = classify_style(accel, lead_launch_mean, stop_mean, coast)

  return ManualStyleSummary(
    sample_count=len(ordered),
    accel=accel,
    lead_launch_count=len(lead_launches),
    clear_launch_count=len(clear_launches),
    lead_launch_mean_accel=lead_launch_mean,
    clear_launch_mean_accel=clear_launch_mean,
    lead_launch_peak_accel=percentile_range([episode["peak_accel"] for episode in lead_launches], 50.0, 90.0),
    clear_launch_peak_accel=percentile_range([episode["peak_accel"] for episode in clear_launches], 50.0, 90.0),
    lead_stop_count=len(lead_stops),
    clear_stop_count=len(clear_stops),
    stop_mean_accel=stop_mean,
    stop_peak_decel=percentile_range([episode["peak_decel"] for episode in stop_candidates], 10.0, 50.0),
    coast_accel=coast,
    style=style,
  )


def _pedal_episodes(samples: list[ManualSample], pedal: str) -> list[dict[str, float | bool]]:
  episodes: list[dict[str, float | bool]] = []
  current: list[ManualSample] = []
  for sample in samples:
    pressed = bool(getattr(sample, pedal))
    if pressed:
      current.append(sample)
      continue
    if current:
      episodes.append(_episode_summary(current))
      current = []
  if current:
    episodes.append(_episode_summary(current))
  return episodes


def _episode_summary(samples: list[ManualSample]) -> dict[str, float | bool]:
  accels = [sample.a_ego for sample in samples]
  return {
    "v0": samples[0].v_ego,
    "v1": samples[-1].v_ego,
    "duration": max(0.0, samples[-1].t - samples[0].t),
    "lead": bool(samples[0].lead_status),
    "mean_accel": sum(accels) / len(accels),
    "peak_accel": max(accels),
    "peak_decel": min(accels),
  }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: summarize manual longitudinal style"
```

## Task 4: Renderer And CLI

**Files:**
- Create: `tools/drive_lab/profile_manual_longitudinal.py`
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Add failing renderer test**

Append to `tools/drive_lab/tests/test_manual_longitudinal_profile.py`:

```python
from openpilot.tools.drive_lab.manual_longitudinal_profile import render_manual_style_summary


def test_render_manual_style_summary_includes_core_values():
  summary = summarize_manual_style([
    sample(0.0, 0.5, 1.2, gas=True, lead=True, d_rel=4.0),
    sample(1.0, 5.5, 0.7, gas=False, lead=True, d_rel=5.0),
    sample(10.0, 10.0, -0.4, brake=True, lead=True, d_rel=12.0, v_rel=-0.8),
    sample(11.0, 0.5, -0.2, brake=False, lead=True, d_rel=8.0, v_rel=0.0),
    sample(20.0, 12.0, -0.3, gas=False, brake=False),
    sample(21.0, 13.0, -0.2, gas=False, brake=False),
  ])

  text = render_manual_style_summary(summary)

  assert "Manual longitudinal style" in text
  assert "style:" in text
  assert "lead launches:" in text
  assert "stop approaches:" in text
  assert "coast accel:" in text
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_render_manual_style_summary_includes_core_values -q
```

Expected: FAIL with import error for `render_manual_style_summary`.

- [ ] **Step 3: Implement renderer**

Add to `tools/drive_lab/manual_longitudinal_profile.py`:

```python
def render_manual_style_summary(summary: ManualStyleSummary) -> str:
  return "\n".join([
    "Manual longitudinal style",
    f"style: {summary.style}",
    f"samples: {summary.sample_count}",
    f"accel p10-p90: {summary.accel.low:.3f} to {summary.accel.high:.3f} m/s^2",
    f"lead launches: {summary.lead_launch_count} mean {summary.lead_launch_mean_accel.low:.3f} to {summary.lead_launch_mean_accel.high:.3f} m/s^2 peak {summary.lead_launch_peak_accel.low:.3f} to {summary.lead_launch_peak_accel.high:.3f} m/s^2",
    f"clear launches: {summary.clear_launch_count} mean {summary.clear_launch_mean_accel.low:.3f} to {summary.clear_launch_mean_accel.high:.3f} m/s^2 peak {summary.clear_launch_peak_accel.low:.3f} to {summary.clear_launch_peak_accel.high:.3f} m/s^2",
    f"stop approaches: {summary.lead_stop_count + summary.clear_stop_count} mean {summary.stop_mean_accel.low:.3f} to {summary.stop_mean_accel.high:.3f} m/s^2 peak {summary.stop_peak_decel.low:.3f} to {summary.stop_peak_decel.high:.3f} m/s^2",
    f"coast accel: {summary.coast_accel.low:.3f} to {summary.coast_accel.high:.3f} m/s^2",
  ])
```

- [ ] **Step 4: Create CLI entrypoint**

Create `tools/drive_lab/profile_manual_longitudinal.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import isfinite
from typing import Any

from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  build_route_profile,
  render_manual_style_summary,
  summarize_manual_style,
)
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a manual longitudinal style profile from route logs.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, log files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--min-manual-moving", type=int, default=1200)
  parser.add_argument("--max-active-ratio", type=float, default=0.25)
  args = parser.parse_args()

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  included_samples: list[ManualSample] = []
  route_profiles = []
  for route in args.routes:
    route_samples = extract_manual_samples(route, read_mode)
    route_profile = build_route_profile(route, route_samples, args.min_manual_moving, args.max_active_ratio)
    route_profiles.append(route_profile)
    if route_profile.include:
      included_samples.extend(route_samples)

  summary = summarize_manual_style(included_samples)
  payload = {"routes": [asdict(profile) for profile in route_profiles], "summary": asdict(summary)}
  if args.output:
    with open(args.output, "w") as f:
      json.dump(payload, f, indent=2)
      f.write("\n")
  print(json.dumps(payload, indent=2) if args.json else render_manual_style_summary(summary))


def extract_manual_samples(route: str, read_mode: ReadMode) -> list[ManualSample]:
  msgs = list(LogReader(route, default_mode=read_mode, sort_by_time=True))
  base_mono_time = int(getattr(msgs[0], "logMonoTime", 0)) if msgs else 0
  active = False
  lead_status = False
  lead_d_rel = None
  lead_v_rel = None
  samples: list[ManualSample] = []
  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ == "selfdriveState":
      active = bool(safe_get(payload, "active", False))
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      lead_status = bool(safe_get(lead, "status", False))
      lead_d_rel = _finite_or_none(safe_get(lead, "dRel"))
      lead_v_rel = _finite_or_none(safe_get(lead, "vRel"))
    elif typ == "carState":
      v_ego = _finite_or_none(safe_get(payload, "vEgo"))
      a_ego = _finite_or_none(safe_get(payload, "aEgo"))
      if v_ego is None or a_ego is None:
        continue
      samples.append(ManualSample(
        route=route,
        t=msg_time_s(msg, base_mono_time),
        v_ego=v_ego,
        a_ego=a_ego,
        active=active,
        gas_pressed=bool(safe_get(payload, "gasPressed", False)),
        brake_pressed=bool(safe_get(payload, "brakePressed", False)),
        lead_status=lead_status,
        lead_d_rel=lead_d_rel,
        lead_v_rel=lead_v_rel,
      ))
  return samples


def _finite_or_none(value: Any) -> float | None:
  if isinstance(value, int | float) and isfinite(float(value)):
    return float(value)
  return None


if __name__ == "__main__":
  main()
```

- [ ] **Step 5: Run tests and compile CLI**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
uv run python -m compileall tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py
```

Expected: pytest PASS and compileall exit 0.

- [ ] **Step 6: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: add manual longitudinal profile cli"
```

## Task 5: Verify Against Existing Drive Lab Tests

**Files:**
- No source edits expected.

- [ ] **Step 1: Run all Drive Lab tests**

Run:

```bash
uv run pytest tools/drive_lab/tests -q
```

Expected: all Drive Lab tests pass.

- [ ] **Step 2: Run lint on new files**

Run:

```bash
uv run ruff check tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
```

Expected: `All checks passed!`.

- [ ] **Step 3: Run CLI help smoke check**

Run:

```bash
uv run python -m openpilot.tools.drive_lab.profile_manual_longitudinal --help
```

Expected: exit 0 and help text containing `Build a manual longitudinal style profile from route logs`.

- [ ] **Step 4: Run optional device smoke check only when a device route is available**

If this plan is being executed from the admin machine with SSH access to the device and the route `/data/media/0/realdata/000000de--d921e2f101--52/qlog.zst` exists, run this read-only command from the repository root:

```bash
ssh -o ConnectTimeout=10 "comma@100.94.10.12" "cd /data/openpilot && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -m openpilot.tools.drive_lab.profile_manual_longitudinal --qlog /data/media/0/realdata/000000de--d921e2f101--52/qlog.zst --min-manual-moving 1"
```

Expected: text summary starting with `Manual longitudinal style`. If SSH is unavailable or the route file is absent, skip this optional device smoke check and report why it was skipped.

- [ ] **Step 5: Commit any test-only fixes**

If Step 1 or Step 2 required fixes, commit them:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: verify manual longitudinal profiler"
```

If no fixes were needed, do not create an empty commit.

## Follow-Up Plans After This Plan

This plan intentionally stops after Drive Lab measurement tooling. Write separate branch-owned plans for these product behavior changes after the profiler lands and is reviewed:

- `feat/longitudinal-e2e-stop-approach`: conservative stop approach targets around -0.30 to -0.45 m/s^2 routine decel and -1.2 to -1.6 m/s^2 transient decel.
- `feat/longitudinal-launch`: lead-matched launch pulse/taper with sustained lead launch around 0.6 to 0.8 m/s^2 and bounded pulse around 1.2 to 1.5 m/s^2.
- `feat/longitudinal-cruise-coast`: harmless overspeed coast target around -0.30 m/s^2 when no hazard or advisory requires braking.
- `feat/longitudinal-decision-layer`: style candidate debug and arbitration coverage proving comfort shaping never overrides physical hazards or confident advisories.

Do not implement product behavior changes as part of this Drive Lab profiler plan.
