# Lead Crawl Manual Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Drive Lab manual profiling for low-speed confirmed-lead crawl behavior around `stop_target + 2 m`, `stop_target + 1 m`, and the final soft-stop meter.

**Architecture:** Extend the existing manual longitudinal profiler with radar lead fields, a stop-target-based gap-excess helper, dedicated lead-crawl buckets, and route-aware crawl/soft-stop episode summaries. This plan is profiling-only on `feat/drive-lab`; live `creep_to_stop_gap` tuning belongs in a later `feat/longitudinal-follow-gap` plan after these metrics exist.

**Tech Stack:** Python 3.12, dataclasses, NumPy percentiles, openpilot LogReader/qlogs, `pytest` via `uv run`, `ruff`.

---

## Scope Check

The design also defines live stopped-lead crawl behavior. That is a separate branch-owned subsystem and should not be implemented in this Drive Lab plan. This plan produces working, testable profile output only; it gives the later follow-gap behavior plan route-derived metrics and exact thresholds to validate against.

## File Structure

- Modify `tools/drive_lab/manual_longitudinal_profile.py`.
  - Add optional radar lead fields to `ManualSample`.
  - Add `lead_crawl_gap_excess()` and lead-value helpers.
  - Add `LeadCrawlBucketSummary` and `LeadCrawlEpisodeSummary` dataclasses.
  - Add crawl bucket and episode summaries to `ManualStyleSummary`.
  - Render crawl bucket and episode sections in text output.
- Modify `tools/drive_lab/profile_manual_longitudinal.py`.
  - Persist `vLeadK`, `aLeadK`, and `modelProb` from `radarState.leadOne` into `ManualSample`.
- Modify `tools/drive_lab/tests/test_manual_longitudinal_profile.py`.
  - Add deterministic crawl-gap helper tests, crawl bucket tests, route-boundary episode tests, renderer tests, and JSON-serialization coverage through `dataclasses.asdict`.

## Task 1: Radar Lead Fields And Gap Excess Helper

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Write the failing tests**

Update the test imports at the top of `tools/drive_lab/tests/test_manual_longitudinal_profile.py`:

```python
from dataclasses import asdict

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_lead_stop_presentation_distance
from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  ProfileRange,
  SmoothAssertiveEnvelope,
  build_route_profile,
  classify_style,
  lead_crawl_gap_excess,
  percentile_range,
  render_manual_style_summary,
  summarize_manual_style,
)
```

Replace the existing `sample()` helper with this version:

```python
def sample(t, v, a, active=False, gas=False, brake=False, lead=False, d_rel=0.0, v_rel=0.0,
           route="route-a", lead_v=None, lead_a=0.0, model_prob=1.0):
  return ManualSample(
    route=route,
    t=t,
    v_ego=v,
    a_ego=a,
    active=active,
    gas_pressed=gas,
    brake_pressed=brake,
    lead_status=lead,
    lead_d_rel=d_rel,
    lead_v_rel=v_rel,
    lead_v_lead=lead_v,
    lead_a_lead=lead_a,
    lead_model_prob=model_prob,
  )


def crawl_sample(t, gap_excess, v=0.3, a=0.0, lead_v=0.2, lead_a=0.0, model_prob=1.0,
                 gas=False, brake=False, route="route-a"):
  stop_target = get_lead_stop_presentation_distance(v, lead_v, lead_a, model_prob)
  return sample(
    t=t,
    v=v,
    a=a,
    gas=gas,
    brake=brake,
    lead=True,
    d_rel=stop_target + gap_excess,
    v_rel=lead_v - v,
    route=route,
    lead_v=lead_v,
    lead_a=lead_a,
    model_prob=model_prob,
  )
```

Add these tests after the route-profile tests:

```python
def test_lead_crawl_gap_excess_uses_stop_presentation_distance():
  crawl = crawl_sample(0.0, gap_excess=2.0, v=0.25, lead_v=0.15, lead_a=-0.05, model_prob=0.9)

  assert lead_crawl_gap_excess(crawl) == pytest.approx(2.0)


def test_lead_crawl_gap_excess_falls_back_to_relative_speed():
  stop_target = get_lead_stop_presentation_distance(0.3, 0.1, 0.0, 1.0)
  crawl = sample(0.0, 0.3, 0.0, lead=True, d_rel=stop_target + 1.0, v_rel=-0.2, lead_v=None)

  assert lead_crawl_gap_excess(crawl) == pytest.approx(1.0)


def test_lead_crawl_gap_excess_ignores_missing_confirmed_lead():
  no_lead = sample(0.0, 0.3, 0.0, lead=False, d_rel=10.0, v_rel=0.0)
  missing_distance = sample(1.0, 0.3, 0.0, lead=True, d_rel=None, v_rel=0.0)

  assert lead_crawl_gap_excess(no_lead) is None
  assert lead_crawl_gap_excess(missing_distance) is None
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_uses_stop_presentation_distance tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_falls_back_to_relative_speed tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_ignores_missing_confirmed_lead -q
```

Expected: FAIL because `lead_crawl_gap_excess` and the new `ManualSample` fields do not exist.

- [ ] **Step 3: Implement the minimal helper**

In `tools/drive_lab/manual_longitudinal_profile.py`, add this import below the NumPy import:

```python
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_lead_stop_presentation_distance
```

Extend `ManualSample` with these fields after `lead_v_rel`:

```python
  lead_v_lead: float | None = None
  lead_a_lead: float | None = None
  lead_model_prob: float | None = None
```

Add these helpers after `manual_moving_samples()`:

```python
def _lead_speed(sample: ManualSample) -> float | None:
  if sample.lead_v_lead is not None:
    return sample.lead_v_lead
  if sample.lead_v_rel is not None:
    return sample.v_ego + sample.lead_v_rel
  return None


def _lead_accel(sample: ManualSample) -> float:
  return sample.lead_a_lead or 0.0


def _lead_model_prob(sample: ManualSample) -> float:
  return sample.lead_model_prob if sample.lead_model_prob is not None else 1.0


def lead_crawl_gap_excess(sample: ManualSample) -> float | None:
  if not sample.lead_status or sample.lead_d_rel is None:
    return None
  v_lead = _lead_speed(sample)
  if v_lead is None:
    return None
  stop_target = get_lead_stop_presentation_distance(sample.v_ego, v_lead, _lead_accel(sample), _lead_model_prob(sample))
  return float(sample.lead_d_rel - stop_target)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_uses_stop_presentation_distance tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_falls_back_to_relative_speed tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_lead_crawl_gap_excess_ignores_missing_confirmed_lead -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: compute lead crawl gap excess"
```

## Task 2: Lead Crawl Bucket Summaries

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Write the failing bucket tests**

Add these tests after the gap-excess tests:

```python
def test_manual_style_summary_includes_lead_crawl_bins():
  samples = [
    crawl_sample(0.0, 2.5, v=0.3, a=0.10, lead_v=0.2, gas=True),
    crawl_sample(1.0, 2.1, v=0.4, a=0.05, lead_v=0.1),
    crawl_sample(2.0, 1.5, v=0.5, a=-0.10, lead_v=0.2, brake=True),
    crawl_sample(3.0, 0.5, v=0.2, a=-0.20, lead_v=0.0, brake=True),
    crawl_sample(4.0, -0.2, v=0.1, a=-0.30, lead_v=0.0, brake=True),
    crawl_sample(5.0, 3.0, v=4.0, a=0.20, lead_v=4.0, gas=True),
  ]

  summary = summarize_manual_style(samples)
  bins = {bucket.label: bucket for bucket in summary.lead_crawl_bins}

  assert bins["open_to_crawl"].sample_count == 2
  assert bins["open_to_crawl"].gas_ratio == 0.5
  assert bins["open_to_crawl"].coast_ratio == 0.5
  assert bins["crawl_to_follow"].sample_count == 1
  assert bins["soft_stop"].sample_count == 1
  assert bins["inside_stop_target"].sample_count == 1
  assert "open_to_crawl" in {bucket.label for bucket in summary.lead_crawl_bins}


def test_manual_style_summary_lead_crawl_bins_are_low_speed_only():
  samples = [
    crawl_sample(0.0, 2.5, v=0.3, a=0.10, lead_v=0.2, gas=True),
    crawl_sample(1.0, 2.5, v=4.0, a=0.10, lead_v=4.2, gas=True),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_bins[0].label == "open_to_crawl"
  assert summary.lead_crawl_bins[0].sample_count == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_includes_lead_crawl_bins tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_lead_crawl_bins_are_low_speed_only -q
```

Expected: FAIL because `ManualStyleSummary.lead_crawl_bins` does not exist.

- [ ] **Step 3: Implement crawl bucket summaries**

Add these constants after `_SPEED_BINS`:

```python
LEAD_CRAWL_MAX_SPEED = 2.5
LEAD_CRAWL_CLOSING_THRESHOLD = -0.1
_LEAD_CRAWL_BUCKETS = (
  ("open_to_crawl", 2.0, float("inf")),
  ("crawl_to_follow", 1.0, 2.0),
  ("soft_stop", 0.0, 1.0),
  ("inside_stop_target", -float("inf"), 0.0),
)
```

Add this dataclass after `FollowingBinSummary`:

```python
@dataclass(frozen=True)
class LeadCrawlBucketSummary:
  label: str
  sample_count: int
  gas_ratio: float
  brake_ratio: float
  coast_ratio: float
  gap_excess: ProfileRange
  ego_speed: ProfileRange
  lead_speed: ProfileRange
  relative_speed: ProfileRange
  accel: ProfileRange
  closing_ratio: float
  closing_speed: ProfileRange
```

Add this field to `ManualStyleSummary` after `following_bins`:

```python
  lead_crawl_bins: list[LeadCrawlBucketSummary]
```

In `summarize_manual_style()`, add this argument to the returned `ManualStyleSummary` after `following_bins`:

```python
    lead_crawl_bins=_summarize_lead_crawl_bins(ordered),
```

Add these helpers after `_summarize_following_bins()`:

```python
def _lead_crawl_sample_details(samples: list[ManualSample]) -> list[tuple[ManualSample, float, float]]:
  details: list[tuple[ManualSample, float, float]] = []
  for sample in samples:
    gap_excess = lead_crawl_gap_excess(sample)
    v_lead = _lead_speed(sample)
    if gap_excess is None or v_lead is None:
      continue
    if sample.v_ego > LEAD_CRAWL_MAX_SPEED and v_lead > LEAD_CRAWL_MAX_SPEED:
      continue
    details.append((sample, gap_excess, v_lead))
  return details


def _summarize_lead_crawl_bins(samples: list[ManualSample]) -> list[LeadCrawlBucketSummary]:
  summaries: list[LeadCrawlBucketSummary] = []
  details = _lead_crawl_sample_details(samples)
  for label, low, high in _LEAD_CRAWL_BUCKETS:
    bucket = [(sample, gap_excess, v_lead) for sample, gap_excess, v_lead in details if low <= gap_excess < high]
    if not bucket:
      continue
    closing = [(sample, gap_excess, v_lead) for sample, gap_excess, v_lead in bucket if v_lead - sample.v_ego < LEAD_CRAWL_CLOSING_THRESHOLD]
    summaries.append(LeadCrawlBucketSummary(
      label=label,
      sample_count=len(bucket),
      gas_ratio=_ratio(sum(1 for sample, _, _ in bucket if sample.gas_pressed), len(bucket)),
      brake_ratio=_ratio(sum(1 for sample, _, _ in bucket if sample.brake_pressed), len(bucket)),
      coast_ratio=_ratio(sum(1 for sample, _, _ in bucket if not sample.gas_pressed and not sample.brake_pressed), len(bucket)),
      gap_excess=percentile_range([gap_excess for _, gap_excess, _ in bucket], 10.0, 90.0),
      ego_speed=percentile_range([sample.v_ego for sample, _, _ in bucket], 10.0, 90.0),
      lead_speed=percentile_range([v_lead for _, _, v_lead in bucket], 10.0, 90.0),
      relative_speed=percentile_range([v_lead - sample.v_ego for sample, _, v_lead in bucket], 10.0, 90.0),
      accel=percentile_range([sample.a_ego for sample, _, _ in bucket], 10.0, 90.0),
      closing_ratio=_ratio(len(closing), len(bucket)),
      closing_speed=percentile_range([sample.v_ego - v_lead for sample, _, v_lead in closing], 10.0, 90.0),
    ))
  return summaries
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_includes_lead_crawl_bins tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_lead_crawl_bins_are_low_speed_only -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: summarize lead crawl buckets"
```

## Task 3: Crawl And Soft-Stop Episode Summaries

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Write the failing episode tests**

Add these tests after the crawl bucket tests:

```python
def test_manual_style_summary_extracts_lead_crawl_and_soft_stop_episodes():
  samples = [
    crawl_sample(0.0, 2.4, v=0.2, a=0.10, lead_v=0.3),
    crawl_sample(1.0, 1.6, v=0.4, a=0.05, lead_v=0.2),
    crawl_sample(2.0, 0.9, v=0.3, a=-0.10, lead_v=0.0),
    crawl_sample(3.0, 0.4, v=0.2, a=-0.20, lead_v=0.0),
    crawl_sample(4.0, 0.0, v=0.0, a=-0.10, lead_v=0.0),
  ]

  summary = summarize_manual_style(samples)
  episodes = {episode.label: episode for episode in summary.lead_crawl_episodes}

  assert episodes["crawl_to_follow"].count == 1
  assert episodes["crawl_to_follow"].start_gap_excess.low == pytest.approx(2.4)
  assert episodes["crawl_to_follow"].end_gap_excess.high == pytest.approx(0.9)
  assert episodes["soft_stop"].count == 1
  assert episodes["soft_stop"].min_gap_excess.low == pytest.approx(0.0)


def test_manual_style_summary_does_not_merge_crawl_episodes_across_routes():
  samples = [
    crawl_sample(0.0, 2.4, route="route-a"),
    crawl_sample(1.0, 1.6, route="route-b"),
    crawl_sample(2.0, 0.9, route="route-b"),
  ]

  summary = summarize_manual_style(samples)

  assert summary.lead_crawl_episodes == []
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_extracts_lead_crawl_and_soft_stop_episodes tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_does_not_merge_crawl_episodes_across_routes -q
```

Expected: FAIL because `ManualStyleSummary.lead_crawl_episodes` does not exist.

- [ ] **Step 3: Implement route-aware crawl episodes**

Add this dataclass after `LeadCrawlBucketSummary`:

```python
@dataclass(frozen=True)
class LeadCrawlEpisodeSummary:
  label: str
  count: int
  duration: ProfileRange
  start_gap_excess: ProfileRange
  end_gap_excess: ProfileRange
  min_gap_excess: ProfileRange
  mean_accel: ProfileRange
```

Add this field to `ManualStyleSummary` after `lead_crawl_bins`:

```python
  lead_crawl_episodes: list[LeadCrawlEpisodeSummary]
```

In `summarize_manual_style()`, add this argument to the returned `ManualStyleSummary` after `lead_crawl_bins`:

```python
    lead_crawl_episodes=_summarize_lead_crawl_episodes(ordered),
```

Add these helpers after `_summarize_lead_crawl_bins()`:

```python
def _summarize_lead_crawl_episodes(samples: list[ManualSample]) -> list[LeadCrawlEpisodeSummary]:
  details = sorted(_lead_crawl_sample_details(samples), key=lambda item: (item[0].route, item[0].t))
  crawl_episodes = _extract_gap_closure_episodes(details, "crawl_to_follow", start_min=2.0, end_max=1.0)
  soft_stop_episodes = _extract_gap_closure_episodes(details, "soft_stop", start_min=1.0, end_max=0.05, start_max=1.0)
  summaries = []
  for label, episodes in (("crawl_to_follow", crawl_episodes), ("soft_stop", soft_stop_episodes)):
    if not episodes:
      continue
    summaries.append(LeadCrawlEpisodeSummary(
      label=label,
      count=len(episodes),
      duration=percentile_range([episode["duration"] for episode in episodes], 50.0, 90.0),
      start_gap_excess=percentile_range([episode["start_gap_excess"] for episode in episodes], 50.0, 90.0),
      end_gap_excess=percentile_range([episode["end_gap_excess"] for episode in episodes], 50.0, 90.0),
      min_gap_excess=percentile_range([episode["min_gap_excess"] for episode in episodes], 10.0, 50.0),
      mean_accel=percentile_range([episode["mean_accel"] for episode in episodes], 50.0, 90.0),
    ))
  return summaries


def _extract_gap_closure_episodes(details: list[tuple[ManualSample, float, float]], label: str, start_min: float,
                                  end_max: float, start_max: float | None = None) -> list[dict[str, float]]:
  episodes: list[dict[str, float]] = []
  current: list[tuple[ManualSample, float, float]] = []
  current_route: str | None = None
  for detail in details:
    sample, gap_excess, _ = detail
    route_changed = current and sample.route != current_route
    if route_changed:
      current = []
    can_start = gap_excess >= start_min if start_max is None else start_min >= gap_excess >= end_max
    if not current and can_start:
      current = [detail]
      current_route = sample.route
      continue
    if not current:
      continue
    current.append(detail)
    if gap_excess <= end_max:
      episodes.append(_gap_closure_episode_summary(current))
      current = []
      current_route = None
  return episodes


def _gap_closure_episode_summary(details: list[tuple[ManualSample, float, float]]) -> dict[str, float]:
  samples = [sample for sample, _, _ in details]
  gaps = [gap_excess for _, gap_excess, _ in details]
  return {
    "duration": max(0.0, samples[-1].t - samples[0].t),
    "start_gap_excess": gaps[0],
    "end_gap_excess": gaps[-1],
    "min_gap_excess": min(gaps),
    "mean_accel": sum(sample.a_ego for sample in samples) / len(samples),
  }
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_extracts_lead_crawl_and_soft_stop_episodes tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_manual_style_summary_does_not_merge_crawl_episodes_across_routes -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: summarize lead crawl episodes"
```

## Task 4: Renderer, JSON Shape, And CLI Extraction

**Files:**
- Modify: `tools/drive_lab/manual_longitudinal_profile.py`
- Modify: `tools/drive_lab/profile_manual_longitudinal.py`
- Modify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`

- [ ] **Step 1: Write the failing renderer and JSON tests**

Add this test after `test_render_manual_style_summary_includes_core_values()`:

```python
def test_render_manual_style_summary_includes_lead_crawl_sections():
  summary = summarize_manual_style([
    crawl_sample(0.0, 2.4, v=0.2, a=0.10, lead_v=0.3, gas=True),
    crawl_sample(1.0, 1.5, v=0.4, a=0.00, lead_v=0.2),
    crawl_sample(2.0, 0.9, v=0.3, a=-0.10, lead_v=0.0, brake=True),
    crawl_sample(3.0, 0.0, v=0.0, a=-0.10, lead_v=0.0, brake=True),
  ])

  text = render_manual_style_summary(summary)
  payload = asdict(summary)

  assert "Lead crawl bins:" in text
  assert "open_to_crawl" in text
  assert "Lead crawl episodes:" in text
  assert payload["lead_crawl_bins"][0]["label"] == "open_to_crawl"
  assert payload["lead_crawl_episodes"][0]["label"] == "crawl_to_follow"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_render_manual_style_summary_includes_lead_crawl_sections -q
```

Expected: FAIL because the renderer does not print lead crawl sections.

- [ ] **Step 3: Implement renderer output**

In `render_manual_style_summary()`, append the crawl sections after the following-bin section:

```python
  lines.append("Lead crawl bins:")
  lines.extend(_render_lead_crawl_bin(crawl_bin) for crawl_bin in summary.lead_crawl_bins)
  if not summary.lead_crawl_bins:
    lines.append("  none")
  lines.append("Lead crawl episodes:")
  lines.extend(_render_lead_crawl_episode(episode) for episode in summary.lead_crawl_episodes)
  if not summary.lead_crawl_episodes:
    lines.append("  none")
```

Add these render helpers after `_render_following_bin()`:

```python
def _render_lead_crawl_bin(crawl_bin: LeadCrawlBucketSummary) -> str:
  return (
    f"  {crawl_bin.label}: samples {crawl_bin.sample_count}, gas {crawl_bin.gas_ratio:.1%}, "
    + f"brake {crawl_bin.brake_ratio:.1%}, coast {crawl_bin.coast_ratio:.1%}, "
    + f"gap excess {crawl_bin.gap_excess.low:.2f} to {crawl_bin.gap_excess.high:.2f} m, "
    + f"ego {crawl_bin.ego_speed.low:.2f} to {crawl_bin.ego_speed.high:.2f} m/s, "
    + f"lead {crawl_bin.lead_speed.low:.2f} to {crawl_bin.lead_speed.high:.2f} m/s, "
    + f"relative {crawl_bin.relative_speed.low:.2f} to {crawl_bin.relative_speed.high:.2f} m/s, "
    + f"accel {crawl_bin.accel.low:.3f} to {crawl_bin.accel.high:.3f} m/s^2, "
    + f"closing {crawl_bin.closing_ratio:.1%}"
  )


def _render_lead_crawl_episode(episode: LeadCrawlEpisodeSummary) -> str:
  return (
    f"  {episode.label}: count {episode.count}, duration {episode.duration.low:.1f} to {episode.duration.high:.1f}s, "
    + f"start gap {episode.start_gap_excess.low:.2f} to {episode.start_gap_excess.high:.2f} m, "
    + f"end gap {episode.end_gap_excess.low:.2f} to {episode.end_gap_excess.high:.2f} m, "
    + f"min gap {episode.min_gap_excess.low:.2f} to {episode.min_gap_excess.high:.2f} m, "
    + f"mean accel {episode.mean_accel.low:.3f} to {episode.mean_accel.high:.3f} m/s^2"
  )
```

- [ ] **Step 4: Persist radar lead fields in the CLI**

In `tools/drive_lab/profile_manual_longitudinal.py`, initialize the new values after `lead_v_rel = None`:

```python
  lead_v_lead = None
  lead_a_lead = None
  lead_model_prob = None
```

In the `radarState` branch, assign them after `lead_v_rel`:

```python
      lead_v_lead = _finite_or_none(safe_get(lead, "vLeadK"))
      lead_a_lead = _finite_or_none(safe_get(lead, "aLeadK"))
      lead_model_prob = _finite_or_none(safe_get(lead, "modelProb"))
```

When constructing `ManualSample`, pass the new fields after `lead_v_rel`:

```python
        lead_v_lead=lead_v_lead,
        lead_a_lead=lead_a_lead,
        lead_model_prob=lead_model_prob,
```

- [ ] **Step 5: Run renderer tests and full profiler tests**

Run:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py::test_render_manual_style_summary_includes_lead_crawl_sections -q
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
```

Expected: both commands PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: render lead crawl profile"
```

## Task 5: Final Verification And Handoff

**Files:**
- Verify: `tools/drive_lab/manual_longitudinal_profile.py`
- Verify: `tools/drive_lab/profile_manual_longitudinal.py`
- Verify: `tools/drive_lab/tests/test_manual_longitudinal_profile.py`
- Verify: `docs/superpowers/specs/2026-05-02-lead-crawl-gap-design.md`

- [ ] **Step 1: Run full Drive Lab verification**

Run:

```bash
uv run pytest tools/drive_lab/tests -q
uv run ruff check tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
uv run python -m openpilot.tools.drive_lab.profile_manual_longitudinal --help
```

Expected:

```text
pytest: all tests pass
ruff: All checks passed!
profile_manual_longitudinal --help: exits 0 and prints usage
```

- [ ] **Step 2: Request code review**

Use `superpowers:requesting-code-review` against the Drive Lab commits created by this plan. The review should focus on:

- Whether `gap_excess` matches planner stop-target semantics.
- Whether low-speed crawl samples are separated from normal following bins.
- Whether route boundaries prevent episode merging.
- Whether JSON output includes the new dataclass fields through `asdict`.
- Whether no product behavior changed on `feat/drive-lab`.

- [ ] **Step 3: Address review findings**

If review returns Critical or Important findings, fix them with TDD. For each finding:

```bash
uv run pytest tools/drive_lab/tests/test_manual_longitudinal_profile.py -q
uv run ruff check tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git add tools/drive_lab/manual_longitudinal_profile.py tools/drive_lab/profile_manual_longitudinal.py tools/drive_lab/tests/test_manual_longitudinal_profile.py
git commit -m "drive-lab: fix lead crawl profile review issue"
```

Expected: tests and ruff pass before every review-fix commit.

- [ ] **Step 4: Stop before behavior tuning**

Do not modify `selfdrive/controls/lib/longitudinal_planner.py` or `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` in this plan. After this plan is complete, write a separate `feat/longitudinal-follow-gap` implementation plan for the live `+2 m` crawl start, `+1 m` follow target, and final-meter soft stop behavior.

- [ ] **Step 5: Optional local integration after approval**

Only after the Drive Lab branch is verified and reviewed, run the repository workflow from the `custom` worktree if local integration is requested:

```bash
scripts/propagate-retained.sh --from feat/drive-lab --no-push
scripts/rebuild-custom.sh
uv run pytest tools/drive_lab/tests -q
```

Expected: propagation completes locally without pushing, custom rebuild completes, and Drive Lab tests pass on rebuilt `custom`.
