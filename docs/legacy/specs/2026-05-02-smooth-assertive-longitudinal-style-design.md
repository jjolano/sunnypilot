# Smooth Assertive Longitudinal Style Design

## Summary

This design turns recent manual driving logs into a conservative `smooth_assertive` longitudinal style profile for this fork. The goal is not to copy every manual pedal event. The goal is to extract stable comfort targets from mostly manual routes, use those targets to tune stop approach, launch, and harmless overspeed coasting, and keep existing automated following gaps unchanged.

The profile should be introduced through Drive Lab first so the same route-derived measurements can validate later behavior changes. Planner and controller changes should remain on the retained branches that own each behavior.

## Route Evidence

The initial profile used read-only analysis of recent device qlogs. Six recent route groups were scanned; five were included because they had enough mostly manual moving samples and low automated-active ratio.

Included routes:

- `000000de--d921e2f101`: 53 segments, 21,303 manual moving samples, active ratio 0.016.
- `000000dd--1fc05b7271`: 22 segments, 7,478 manual moving samples, active ratio 0.042.
- `000000dc--3aee6440c6`: 65 segments, 23,812 manual moving samples, active ratio 0.041.
- `000000db--25c42717d1`: 13 segments, 3,055 manual moving samples, active ratio 0.092.
- `000000da--57f45ab6de`: 6 segments, 1,721 manual moving samples, active ratio 0.207.

Excluded route:

- `000000d9--758a721a6b`: only 659 manual moving samples.

Aggregate sample size: 57,369 manual moving samples.

Observed aggregate values:

| Metric | Value |
|---|---:|
| Speed p10 / p50 / p90 | 1.88 / 10.76 / 20.02 m/s |
| Accel p01 / p05 / p10 | -2.22 / -1.33 / -0.82 m/s^2 |
| Accel p50 | 0.02 m/s^2 |
| Accel p90 / p95 / p99 | 0.92 / 1.21 / 1.68 m/s^2 |
| Gas episode median mean accel | 0.23 m/s^2 |
| Brake episode median mean accel | -0.31 m/s^2 |
| Lead-launch median mean accel | 0.69 m/s^2 |
| Lead-launch median peak accel | 1.72 m/s^2 |
| Clear-launch median mean accel | 0.80 m/s^2 |
| Clear-launch median peak accel | 1.94 m/s^2 |
| Stop approach median mean decel | -0.41 m/s^2 |
| Stop approach median peak decel | -1.94 m/s^2 |
| High-speed coast median accel | about -0.30 to -0.33 m/s^2 |

User validation:

- Treat the whole latest route as representative.
- Manual style label: smooth assertive.
- Do not tighten automated following gaps from this data.
- Copy human-like stop approaches.
- Copy lead-matched launch behavior.
- Prefer context-aware coasting when harmlessly above target speed.
- Use a conservative profile: median manual behavior as comfort targets, p10/p90 style percentiles as guardrails.

## Desired Behavior

The desired automated style is smooth assertive, not aggressive.

- Keep existing following gap policy unchanged.
- Preserve safety and current fallback behavior ahead of style preferences.
- Brake earlier and more gently for credible stop threats, with smooth taper near stop.
- Allow brief stronger braking when the threat is real, but avoid late routine decel spikes.
- Launch promptly when a lead pulls away, with a short controlled pulse followed by lead-matched acceleration.
- Launch slightly more assertively when no lead is present and the path is clear.
- Prefer coasting around natural drag levels when above cruise setpoint and no lead, stop, speed-limit, map, curve, or forced-slow context requires braking.
- Keep advisories confidence-gated so weak map/curve/limit information does not cause nuisance braking.

## Conservative Tuning Targets

These values are starting targets, not hard safety limits.

| Scenario | Comfort Target | Soft Limit | Notes |
|---|---:|---:|---|
| Lead-matched launch sustained accel | 0.6 to 0.8 m/s^2 | about 1.0 m/s^2 | Match lead motion after initial pulse. |
| Lead launch initial pulse | 1.2 to 1.5 m/s^2 | 1.7 to 2.0 m/s^2 | Only with confident lead pullaway and opening gap. |
| Clear/no-lead launch pulse | 1.4 to 1.7 m/s^2 | 2.0 to 2.3 m/s^2 | Slightly more assertive than lead launch. |
| Routine stop approach decel | -0.30 to -0.45 m/s^2 | about -0.9 m/s^2 | Start earlier and taper smoothly. |
| Stop approach transient decel | -1.2 to -1.6 m/s^2 | existing safety limits | Use when closing threat is real. |
| Harmless coast accel target | about -0.30 m/s^2 | about -0.45 m/s^2 | Use only when no stronger candidate requires braking. |
| Following gap target | unchanged | unchanged | Manual gaps are not copied into automation. |

## Branch Ownership

This work crosses multiple longitudinal domains. Each durable change must live on the branch that owns it.

- `feat/drive-lab`: manual route style profiling, aggregate summaries, route-style classification, and tests for analysis tooling.
- `feat/longitudinal-e2e-stop-approach`: no-lead e2e/model stop approach comfort shaping.
- `feat/longitudinal-follow-gap`: stopped lead gap creep/release only if a validated stop tuning requires confirmed-lead stopped-gap changes. Do not globally tighten following gaps.
- `feat/longitudinal-launch`: controller-side stop-to-go and lead-matched launch shaping.
- `feat/longitudinal-cruise-coast`: harmless overspeed coasting preference and downhill/set-speed leeway.
- `feat/longitudinal-decision-layer`: candidate metadata, style telemetry, and arbitration rules that coordinate style candidates.

Do not implement durable product behavior directly on `custom`.

## Drive Lab Tooling Design

Drive Lab should gain a manual longitudinal style profiler before behavior tuning begins. The profiler should read one or more routes or local log paths accepted by LogReader, filter mostly manual moving samples, and emit a compact summary suitable for comparing future routes.

Inputs:

- One or more route identifiers, segment ranges, log files, or URLs accepted by LogReader.
- Optional `--qlog` flag.
- Optional thresholds for minimum manual moving samples and maximum active ratio.
- Optional JSON output for future automation.

Outputs:

- Route inclusion table with segment count, duration, sample count, manual moving sample count, and active ratio.
- Aggregate acceleration percentiles.
- Speed-bin pedal/coast percentages and acceleration percentiles.
- Following-gap and closing-time summaries for observation only.
- Launch summary split by lead and clear/no-lead starts.
- Stop-approach summary split by lead and clear/no-lead stops.
- Coast acceleration summary by speed bin.
- Style classification, initially `smooth_assertive` when the aggregate matches the conservative profile envelope.

The tool must be read-only with respect to route logs. It may write an explicit output file only when requested.

## Behavior Integration Design

Behavior changes should consume the profile conservatively.

Stop approach:

- Use earlier mild deceleration for credible stop threats.
- Prefer routine decel around -0.30 to -0.45 m/s^2 when distance allows.
- Allow stronger transient decel only when required by closing distance or model stop shortage.
- Preserve existing safety clipping and forced slow-decel behavior.

Launch:

- Detect confident lead pullaway and opening gap.
- Apply a bounded initial pulse, then taper to sustained lead-matched acceleration.
- Use lower pulse and sustained targets with a close lead than with a clear path.
- Preserve gas/brake override behavior and existing controller safety limits.

Cruise coast:

- Prefer natural coast around -0.30 m/s^2 when harmlessly above set speed.
- Do not coast-relax braking when a lead, stop, speed-limit, curve, map, or traffic-control candidate is active.
- Keep existing downhill leeway behavior and improve it only with tests.

Decision layer:

- Keep the feature toggle default off.
- Add profile-derived debug values to candidates where useful.
- Keep physical hazards higher priority than advisories and comfort shaping.
- Let cruise coast remain comfort shaping only.
- Do not make route-derived following gaps affect automated following distance.

## Testing Strategy

Drive Lab tests:

- Unit-test manual sample filtering.
- Unit-test route inclusion thresholds.
- Unit-test percentile summaries on synthetic logs.
- Unit-test launch, stop, coast, and following buckets with deterministic fake messages.
- Unit-test `smooth_assertive` classification from a representative synthetic profile.

Behavior tests:

- Stop approach tests should assert earlier mild decel and bounded transient decel for no-lead model stops.
- Launch tests should assert lead-matched pulse/taper behavior and no surge with close or slow leads.
- Cruise coast tests should assert harmless overspeed coasting and hazard/advisory override.
- Decision-layer tests should assert comfort candidates cannot override physical hazards or confident advisories.

Validation commands should use `uv run` and targeted pytest files before any rebuild or deploy.

## Rollout

1. Implement Drive Lab profiling on `feat/drive-lab`.
2. Use the profiler to regenerate and save comparable summaries for representative routes.
3. Implement branch-owned behavior changes one branch at a time.
4. Propagate retained branches downstream after each owning branch is committed.
5. Rebuild `custom` only after retained branches contain the durable changes.
6. Validate on-device with the decision layer off first where applicable, then opt in to decision-layer behavior.

## Non-Goals

- Do not train a model from the route data.
- Do not globally tighten following distance.
- Do not replace MPC or LongControl wholesale.
- Do not make map-only traffic controls command hard stops.
- Do not add new published cereal schema fields in the first profiler pass.
- Do not make route-derived values default-on without route and on-device validation.
