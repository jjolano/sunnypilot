# Scale roll-compensation gain in the torque v2.1 response core

Status: accepted
Date: 2026-07-02
Relates to: `docs/adr/2026-06-13-clean-room-torque-v2-1-architecture.md`, `sunnypilot/custom/lateral/response_core.py`.

## Context

The response core was ported unchanged-in-math from legacy `latcontrol_torque_v2.py` and, like
upstream openpilot, feeds forward the full gravity crown term `roll * g` (minus torqued's learned
constant `latAccelOffset`).

Route `00000246--6e58c526bc` (RAV4 TSS2, 2026-07-02) showed the deployed controller holding a
permanent integrator load that nearly cancels the roll feedforward on straight cruise
(`f` mean +0.216, `i` mean −0.189 in lateral-accel units at roll ≈ −0.039 rad). Regressing the
steady straight-cruise torque need against `−roll·g` over 22.7k frames across seven segments gives
slope **0.56**: the platform needs roughly half the gravity term the feedforward injects. A constant
`latAccelOffset` cannot absorb a slope error, so every crown change forces the integrator (KI 0.2)
to re-converge — producing the observed 0.15–0.2 Hz lane wander (±0.3 m) and biasing the car
~0.2 m left of center, which the driver corrected roughly 10×/min.

## Decision

Introduce `ROLL_COMPENSATION_GAIN = 0.55` in the response core and scale the roll feedforward by
it. This is a deliberate deviation from the legacy math; the parity oracle in
`test_response_core_parity.py` is updated to share the constant, and a targeted test guards the
scaling.

## Consequences

- The feedforward matches the measured platform response; the integrator no longer carries a
  steady crown load, so crown transitions stop driving slow drift-and-correct wander.
- torqued's `latAccelOffset` (learned ≈ +0.17 partly to cancel the old excess) re-learns over the
  first minutes of driving after deploy; small transient bias is expected and handled by the PI.
- The gain is platform-tuned from one car's data. If another platform is ever targeted, re-run the
  steady-need-vs-roll regression (see route 00000246 analysis) before reusing the value.
- Entering banked curves, the PI supplies the portion of bank torque the FF no longer injects;
  route validation should watch banked-curve tracking error alongside the straight-cruise metrics.
