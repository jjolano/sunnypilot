# Lead Crawl Gap Design

## Summary

This design extends the manual longitudinal profile so it explicitly measures how the driver crawls behind confirmed leads and closes low-speed lead gaps. It also defines the later automation target for stopped/near-stopped lead crawl behavior: when a lead opens the gap to at least `stop_target + 2 m`, begin crawling; follow the lead around `stop_target + 1 m`; use the final `1 m` buffer to soft-stop at `stop_target` rather than chasing the lead all the way to the stop distance.

The first implementation step is Drive Lab profiling only. The later live behavior change belongs on `feat/longitudinal-follow-gap` because it changes stopped lead gap creep/release behavior behind a confirmed lead.

## User Intent

The desired behavior is not a tighter normal-speed following gap. It is a low-speed, confirmed-lead crawl style:

- If the lead crawls forward and the gap opens to at least `stop distance + 2 m`, the car should begin crawling.
- The crawl should close the gap only to about `stop distance + 1 m` while the lead keeps moving.
- The remaining `1 m` should be treated as a soft-stop runway, letting the car settle at the stop distance smoothly if the lead stops again.
- Normal-speed following distance remains unchanged.

## Definitions

Use the same stop reference as the existing low-speed lead logic:

- `stop_target`: the confidence-aware lead stop presentation distance from `get_lead_stop_presentation_distance(...)`.
- `gap_excess`: `d_rel - stop_target`.
- `crawl_start_excess`: `2.0 m`.
- `crawl_follow_excess`: `1.0 m`.
- `soft_stop_excess`: `0.0 m`.

For Drive Lab profiling, use radar lead data when available. If route samples only have relative speed, derive `v_lead = v_ego + v_rel`; otherwise persist `vLeadK`, `aLeadK`, and `modelProb` from `radarState.leadOne` so `stop_target` matches planner logic more closely.

## Drive Lab Profiling

Add a dedicated lead-crawl summary to the manual profiler. This should be observational and should not change automation behavior.

Included samples:

- Manual or inactive control samples from included mostly-manual routes.
- Confirmed `leadOne` with finite `d_rel`.
- Low-speed crawl context, roughly `v_ego <= 2.5 m/s` or `v_lead <= 2.5 m/s`.
- Finite `gap_excess` computed from the stop target.

Profile buckets:

- `open_to_crawl`: `gap_excess >= 2.0 m`.
- `crawl_to_follow`: `1.0 m <= gap_excess < 2.0 m`.
- `soft_stop`: `0.0 m <= gap_excess < 1.0 m`.
- `inside_stop_target`: `gap_excess < 0.0 m`, reported as a safety/comfort outlier rather than a target.

Metrics:

- Sample count and route coverage.
- Gas, brake, and coast ratios in each bucket.
- Ego speed, lead speed, relative speed, acceleration, and gap-excess percentiles.
- Closing ratio and closing speed percentiles.
- Crawl episodes where gap moves from at least `+2 m` toward `+1 m`.
- Soft-stop episodes where gap moves from `+1 m` toward `0 m`.
- Minimum gap-excess observed per episode.

Text and JSON output should include the crawl summary separately from normal following bins. Normal following bins remain useful for observation, but the crawl summary is the source for low-speed stopped-lead tuning.

## Behavior Target

The live behavior change should be conservative and branch-owned on `feat/longitudinal-follow-gap`.

For `creep_to_stop_gap` and related hold/release behavior:

- Do not arm ordinary stopped-lead creep until `gap_excess >= 2.0 m`, unless an already-active safe crawl is decelerating toward the stop target.
- When `gap_excess > 1.0 m`, allow gentle crawl acceleration that can close toward `gap_excess = 1.0 m` while respecting lead speed and predicted pullaway.
- Around `gap_excess = 1.0 m`, prefer matching the lead instead of continuing to close aggressively.
- Between `gap_excess = 1.0 m` and `0.0 m`, taper target speed and acceleration toward a soft stop.
- At or below `gap_excess = 0.0 m`, command no positive creep acceleration.
- Preserve driver brake/gas overrides and force-slow-decel blocking.
- Preserve existing safety limits, model confidence checks, and confirmed-lead requirements.

This should replace the current very-small stopped creep release/hold thresholds with the `+2 m` release and `+1 m` follow target shape, without changing normal moving-lead following distance.

## Branch Ownership

- `feat/drive-lab`: add crawl-profile sample fields, crawl buckets, crawl episode summaries, renderer/JSON output, and tests.
- `feat/longitudinal-follow-gap`: tune `creep_to_stop_gap`, stopped lead hold/release, and validation tests for `+2 m` start, `+1 m` follow, and final-meter soft stop.
- `custom`: rebuild only after retained branches contain the durable changes.

## Testing Strategy

Drive Lab tests:

- Unit-test `gap_excess` calculation from radar lead values.
- Unit-test crawl bucket assignment at `+2 m`, `+1 m`, `0 m`, and below-target gaps.
- Unit-test gas/brake/coast ratios and accel percentiles in crawl buckets.
- Unit-test crawl and soft-stop episode extraction across route boundaries.
- Unit-test text and JSON summaries include crawl metrics.

Follow-gap behavior tests:

- Assert stopped lead creep does not release below `stop_target + 2 m`.
- Assert creep activates when the confirmed lead opens to at least `stop_target + 2 m`.
- Assert the crawl target closes toward `stop_target + 1 m`, not to the stop target directly.
- Assert the final `1 m` produces bounded soft-stop decel and no positive acceleration at or below `stop_target`.
- Assert driver gas/brake and force-slow-decel still block crawl behavior.
- Assert normal-speed following gap calculations are unchanged.

## Non-Goals

- Do not globally tighten following distance.
- Do not use manual route gaps to change normal-speed time-gap policy.
- Do not make map/model-only stops affect lead crawl without a confirmed radar lead.
- Do not deploy behavior changes before Drive Lab metrics and branch-owned tests are in place.
