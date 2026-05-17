# ADR: Custom Longitudinal v2 Architecture

## Status

Accepted for initial implementation.

## Context

The stack selector now separates upstream-equivalent `sunnypilot-current` behavior from retained custom longitudinal behavior. `custom-1.0` preserves the current candidate/fallback stack. `custom-2.0` is intended to be a full replacement custom stack with a product promise of assertive progress while keeping explicit safety caps hard.

The baseline stack must remain isolated. SCC, SLA, and OSM behavior remain user-facing feature switches, but custom stacks may arbitrate those signals differently.

## Decision

`custom-2.0` will live in `selfdrive/controls/lib/longitudinal_stacks/custom_v2.py` and own:

- normalized-scene interpretation at the stack boundary
- nine intent families: `driver_cruise`, `lead_follow`, `stop_approach`, `launch`, `speed_policy`, `curve_policy`, `map_caution`, `comfort_relax`, and `safety_cap`
- safety-first arbitration order: `safety_cap`, `stop_approach`, `lead_follow`, `launch`, advisory caps, `comfort_relax`, then `driver_cruise`
- progress-core envelopes for no-lead launch, lead pullaway, excess-gap closing, and lead-loss recovery
- fail-closed validation that requests immediate disable on invalid custom output

The first selectable release keeps `custom-recommended` unchanged and does not require v1 parity. All intent names must exist from the first release, even if some are conservative placeholders.

## Consequences

- `sunnypilot-current` cannot consume v2-only tuning or output transforms.
- `custom-2.0` starts without runtime baseline fallback output.
- `custom-1.0` and existing shadow fields remain for the stage-1 compatible rollout.
- Promotion of `custom-2.0` to `custom-recommended` requires route replay or Drive Lab validation plus a road test.
