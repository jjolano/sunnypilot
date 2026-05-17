# Custom Longitudinal v2 Spec

## Goal

Add selectable `custom-2.0` as a full custom longitudinal stack that prioritizes assertive progress without relaxing explicit safety caps.

## Non-Goals

- Do not change `custom-recommended` during the first selectable release.
- Do not require parity with the removed v1 stack.
- Do not let `sunnypilot-current` consume custom-only tuning or arbitration.

## Stack Selection

- `LongitudinalStack` values are `sunnypilot-current`, `custom-recommended`, and `custom-2.0`.
- Unset or unknown values resolve to `sunnypilot-current`.
- Stack selection is latched and changes require an onroad cycle.
- `AlphaLongitudinalEnabled` remains the gas/brake takeover gate.

## V2 Intents

- `safety_cap`: hard restrictions from validated safety constraints.
- `stop_approach`: comfort-first stop-threat handling for true no-lead stops.
- `lead_follow`: confirmed-lead following and lead safety behavior.
- `launch`: no-lead and planner-seeded lead-pullaway progress behavior.
- `speed_policy`: coast-biased speed-limit handling.
- `curve_policy`: existing custom SCC vision/map thresholds at initial release.
- `map_caution`: OSM/mapd hazard caps, with map-only preparation reserved for future unconfirmed map candidates.
- `comfort_relax`: small relax of advisory braking inside clear safety margins.
- `driver_cruise`: driver set-speed tracking with dynamic downhill/coast leeway.

## First-Release Tunings

- No-lead launch ceiling: `0.95 m/s^2`.
- Lead-pullaway ceiling: `1.20 m/s^2`.
- Positive progress jerk: `4.0 m/s^3`.
- Normal negative progress retreat jerk: `-5.0 m/s^3`.
- Lead motion gate: `vLead >= 0.15 m/s` or positive opening prediction.
- Launch speed caps: `3.0 m/s` no-lead and `5.0 m/s` lead pullaway.
- Excess-gap accel cap: `0.4-1.0 m/s^2` across `1-8 m` excess gap.
- Closing-speed guard: taper above `0.3 m/s`, block above `0.7 m/s` unless the gap is still very large.
- No-lead stop-clear: no `shouldStop`, no near stop point within `20 m`, and model desired accel not below `-0.5 m/s^2`.
- Lead-loss or transition occlusion guard: `0.75 s`.
- Plain-cruise overspeed leeway: dynamic by grade/coast context, bounded by `+3 to +7 mph`.

## Scene Behavior

- Lead-follow and lead-pullaway actuation is planner/MPC-seeded first; `custom-2.0` classifies and preserves planner seed behavior instead of independently creating lead acceleration.
- Planner seed telemetry maps seed reasons into v2 intents while preserving the raw seed reason as `selectedReason`.
- Classification-only planner seeds preserve their incoming speed, accel, and jerk trajectories.
- No-lead launch remains limited to clear model stop context below the no-lead launch speed cap.
- No-lead stop approach is comfort-bounded by default; `custom-2.0` may exceed the comfort bound only when `shouldStop` is true and a finite model stop distance requires harder decel, clipped to planner accel limits.
- Speed-limit reductions remain coast-biased; stronger braking must come from lead, stop, curve, or confirmed map-caution evidence.
- SCC vision/map curve policy uses only active curve sources and applies the most restrictive active curve target.
- OSM traffic-control prior `active` is treated as confirmed because that prior already requires model-distance confirmation. Map-only preparation remains future behavior until map-only candidates are exposed separately.
- Confirmed map caution is cap-only and does not set stop intent or get softened by comfort relax.
- Driver brake/gas input blocks progress floors and comfort relax, while conservative advisory caps may still apply.
- Core non-finite scene inputs fail closed; invalid optional speed, curve, or map advisory targets are ignored for that cycle.
- Normal v2-owned accel changes are jerk-limited by the first-release jerk tunings. Hard model stops, safety caps, and preserved planner lead restrictions may bypass comfort jerk limiting.

## Fail-Closed Behavior

If `custom-2.0` produces an invalid output or raises internally while enabled, latch a custom stack fault and request immediate disable. The latch resets after disengagement.

## Validation

- Selector, UI, metadata, and schema tests cover `custom-2.0` exposure.
- Unit tests cover intent names, progress-core caps, planner seed classification, trajectory preservation, stop approach gates, speed-policy coast bias, map-caution authority, dynamic cruise leeway, scene validation, jerk limiting, and fail-closed behavior.
- Promotion to `custom-recommended` requires route replay or Drive Lab validation plus a road test.
