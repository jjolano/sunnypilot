# Custom Longitudinal v2 Spec

## Goal

Add selectable `custom-2.0` as a full custom longitudinal stack that prioritizes assertive progress without relaxing explicit safety caps.

## Non-Goals

- Do not change `custom-recommended` during the first selectable release.
- Do not require `custom-1.0` parity.
- Do not let `sunnypilot-current` consume custom-only tuning or arbitration.

## Stack Selection

- `LongitudinalStack` values are `sunnypilot-current`, `custom-recommended`, `custom-1.0`, and `custom-2.0`.
- Unset or unknown values resolve to `sunnypilot-current`.
- Stack selection is latched and changes require an onroad cycle.
- `AlphaLongitudinalEnabled` remains the gas/brake takeover gate.

## V2 Intents

- `safety_cap`: hard restrictions from validated safety constraints.
- `stop_approach`: comfort-first stop-threat handling for true no-lead stops.
- `lead_follow`: confirmed-lead following and lead safety behavior.
- `launch`: no-lead and lead-pullaway progress behavior.
- `speed_policy`: coast-biased speed-limit handling.
- `curve_policy`: existing custom SCC vision/map thresholds at initial release.
- `map_caution`: OSM/mapd-only hazards apply mild prep/caps until confirmed.
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

## Fail-Closed Behavior

If `custom-2.0` produces an invalid output or raises internally while enabled, latch a custom stack fault and request immediate disable. The latch resets after disengagement.

## Validation

- Selector, UI, metadata, and schema tests cover `custom-2.0` exposure.
- Unit tests cover intent names, progress-core caps, stop release on confirmed lead pullaway, speed-policy coast bias, map-caution authority, dynamic cruise leeway, and fail-closed behavior.
- Promotion to `custom-recommended` requires route replay or Drive Lab validation plus a road test.
