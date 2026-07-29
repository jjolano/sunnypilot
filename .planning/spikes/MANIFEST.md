# Spike Manifest

## Idea

Test whether the existing lateral demand-jerk smoother can safely cover loaded-corner
turn-in and unwind, where routes `000002cd` and `000002ce` show model-led jerk tails.
This is an offline feasibility experiment, not a live-control proposal.

## Requirements

- Target both turn-in and unwind; steady-corner demand must remain unchanged.
- Up to 100–200 ms of bounded softness is acceptable.
- Reuse the existing demand-smoothing operation; do not add another output governor.
- Preserve sign and cap lag explicitly; any unsafe unwind retention kills the idea.
- Fallback truth must remain unshaped; controller-demand shaping cannot feed the next
  model-path fallback decision.
- Validate on cached rlogs before proposing product code, Params, or deployment changes.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | corner-safe-demand-jerk | comparison | Given clean engaged corner transitions from routes 2cd/2ce, when the existing demand slew is tested before and after low-quality fallback with 0.08 and 0.20 m/s² lag caps, then turn-in/unwind jerk tails shrink without sign error or more than 200 ms equivalent lag | PARTIAL | lateral, jerk, corner, replay |
| 002 | decoupled-anticipatory-unwind | comparison | Given Spike 001's unsafe feedback and weak unwind result, when fallback state is kept unshaped and the same jerk shaper optionally follows a causal 100-200 ms unwind preview, then the lag cap remains strict and unwind jerk falls without new excursions | PARTIAL | lateral, jerk, corner, replay, preview |
