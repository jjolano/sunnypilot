# Learned roll-compensation gain apply path

Status: accepted
Date: 2026-07-02
Relates to: `docs/adr/2026-07-02-scale-roll-compensation-gain.md`,
`sunnypilot/custom/lateral/parameter_orchestrator.py`,
`sunnypilot/custom/lateral/response_core.py`,
`sunnypilot/custom/lateral/torque_v2_1.py`.

## Context

`docs/adr/2026-07-02-scale-roll-compensation-gain.md` replaced the legacy full-gravity
roll feedforward with a fixed scale (`ROLL_COMPENSATION_GAIN = 0.55`). That change removed
the steady crown load and the resulting low-frequency lane wander on the platform where it
was measured, but it is still a single global constant. Different platforms, tire setups,
and suspension geometries can need a different scale.

Phase 3 added a shadow learner (`RollCompBuckets` in `sunnypilot/custom/lateral/roll_comp_learning.py`)
that regresses the torque controller's predicted lateral acceleration against
`−sin(roll)·g` during straight-road steady-state driving. The learner emits a profile into
`RollCompGainParams`; `RollCompGainMode` controls whether the learner runs (`shadow`) or is
off. The apply UI was intentionally deferred until a live apply path existed.

## Decision

Wire the learned gain into the live torque v2.1 response core, gated by `RollCompGainMode == apply`:

- `ResponseCore` exposes an instance attribute `roll_compensation_gain`, defaulting to
  `ROLL_COMPENSATION_GAIN`. The roll feedforward in `update()` uses this attribute.
- `TorqueParameterOverridePolicy` reads `RollCompGainMode` on its existing 300-frame poll.
  When the mode is `apply` it parses `RollCompGainParams` with `parse_roll_comp_profile`,
  validates the restore key, clamps, confidence, version, and span, and exposes the resulting
  gain as `learned_roll_gain`. Any validation failure leaves `learned_roll_gain = None`.
- `LatControlTorqueV21` copies `extension.learned_roll_gain` onto
  `response_core.roll_compensation_gain` after every override update, falling back to
  `ROLL_COMPENSATION_GAIN` when the attribute is missing or `None`.
- The Sunnylink YAML schema, generated `settings_ui.json`, and native settings dialog now
  expose `apply` alongside `off` and `shadow` for `RollCompGainMode`, with offroad-only writes
  and the existing remove-param-for-off pattern.

The learned gain is kept out of `torque_params` capture/restore entirely. It is only an
extra exposed attribute that the controller injects into the response core.

## Consequences

- Users can enable a platform-specific roll-compensation scale once the learner has collected
  enough straight-road data. Until then the fixed `ROLL_COMPENSATION_GAIN` remains the safe
  fallback.
- The apply path is user-gated and offroad-configurable. Promotion to a wider default depends
  on engaged-route replay evidence: straight-cruise tracking error, crown transition
  transients, and banked-curve behavior with the learned gain versus the fixed constant.
- Malformed, stale, or mismatched profiles fail closed to the fixed constant, so a parameter
  restore-key change or corrupted profile cannot silently change steering response.
