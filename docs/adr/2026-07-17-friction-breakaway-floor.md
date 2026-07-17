# Friction breakaway floor for rack stick-slip

Status: accepted
Date: 2026-07-17

## Context

Route 000002a1 rlogs (100 Hz, four segments) showed the steering wheel moving in
discrete 0.6-1.2 deg steps while the torque command stayed smooth (command HF/LF
0.28-0.35 vs steering-rate HF/LF 0.84-1.10; dwell→jump events 9-16/min of curve
driving): rack/EPS stick-slip. In the 0.05-0.35 Hz wander band, desired lateral
accel leads actual by +0.17 s at large amplitude but +0.44 s at small amplitude —
classic deadband phase lag, uncompensated because liveDelay assumes a fixed 0.28 s.
Small corrections systematically land late, feeding the model-led wander limit cycle
(the wander itself remains model-led: desired leads actual at corr 0.98-0.99).

The response core's legacy `low_demand_friction_scale` suppresses friction
compensation quadratically at low demand (a 0.1 m/s² error gets ~11% of breakaway,
vs 50% upstream). That suppression exists for a real reason — full friction at
noise-level errors sign-chatters into dither — but it also starves persistent small
corrections, so the wheel waits for P/I to wind through the sticky zone (a
gpt-5.6-sol cross-review confirmed the integrator, not friction, currently does the
breaking out: KI=0.2 winds ~0.08 normalized per 10 s at 0.1 m/s² error, matching the
observed 4-6 s dwells). torqued's learned friction (0.126, converged, not pinned) is
a hysteresis-spread estimate and under-represents true static breakaway 2-3x.

## Decision

Add a **friction breakaway floor** (`sunnypilot/custom/lateral/friction_breakaway_floor.py`),
injected into the response core via an optional `friction_shaper` hook (hook unset =
exact legacy math; parity tests unchanged). When the tracking-error sign has persisted
150 ms above a 0.03 m/s² noise epsilon, the friction term is floored at an
error-proportional target (full at |error| ≥ 0.15 m/s²) capped at 0.7 of full
breakaway (`friction × latAccelFactor`), slew-limited at ~1.5 units/s so
engagement/release never injects a torque step. The floor only ever deepens the
friction term in the error direction, never opposes it.

Gated by `LatFrictionBreakawayMode` (off/shadow/apply, fail-closed, Settings →
Steering). Shadow computes and logs (`frictionFloorActive`/`frictionFloorDelta` on
`adaptiveTorqueState`) without actuating. A shadow-only **breakaway observer** in
`torqued_ext` records EPS torque magnitude at dwell→jump onsets per direction
(`breakawayLeftMedian`/`breakawayRightMedian`/`breakawayEvents` on
`liveTorqueParameters`) to tune the floor from data — route 2a1 shows left/right
asymmetry (~+0.20/+0.35 vs −0.15/−0.22 normalized).

Validated in `tools/drive_lab/stiction_lab.py`, a closed-loop lab running the real
`LatControlTorqueV21` against a torque-domain stick-slip plant that reproduces the
on-road signature near-quantitatively (lag 0.47/0.29/0.17 s at amp 0.15/0.3/0.6 vs
on-road 0.44/0.29/0.17-0.22). With the floor: small-correction lag 0.47→0.22 s,
tracking RMSE improves, no dither, curves untouched.

## Alternatives rejected

- **Hard floor (no ramp/slew)**: dithers in sim — post-breakout error flip re-engages
  the floor the other way (cmd HF/LF 0.21→1.94).
- **Raising KI / integrator boosts**: winds up torque behind the stuck rack, making
  the breakout snap bigger — the opposite of smooth.
- **Dither injection** (5-10 Hz torque modulation): feasible within Toyota rate
  limits but buzz/EPS-tolerance risk; reserved if the floor plateaus.
- **More centering gain / path-layer stabilization**: already proven to worsen wander
  (see StraightPathStabilization revert).

## Consequences

- Shadow first: one drive with `shadow` to verify floor firing stats and breakaway
  telemetry before flipping to `apply`.
- The floor shortens the wait for breakaway; the discrete step itself is EPS/rack
  hardware and stays. If shadow shows EPS tracking commands faithfully during dwells,
  the residual stick is fully downstream and 0.7 is the practical ceiling.
- FLOOR_FRAC 0.7 chosen with margin (0.9 stable in sim but the sim's breakaway
  conveniently matches learned friction); revisit from breakaway observer data.
