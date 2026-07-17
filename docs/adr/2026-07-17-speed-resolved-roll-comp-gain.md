# Speed-resolved roll-compensation gain

Status: accepted
Date: 2026-07-17

## Context

Roll compensation is applied ungated at every speed (`response_core.py`:
`gain × roll × g`), which is right to first order — the tire force countering crown
pull is speed-independent while rolling. But the learned gain
(`RollCompGainMode`, docs/adr/2026-07-02-learned-roll-compensation-gain.md) is fitted
exclusively from ≥15 m/s points and extrapolated downward, exactly where the roll term
is the *largest fraction* of the feedforward: path demand shrinks with v² while the
roll term stays constant. Below 5 m/s the integrator is frozen, so a low-speed gain
error cannot even be trimmed out — boosted P carries it as persistent tracking error.

A hard low-speed gate is the wrong shape: it would step the torque on every threshold
crossing of a crowned road — the same class of discontinuity repeatedly convicted as a
felt-jerk source (preview-assist flicker, SPS release snaps).

## Decision

Learn the gain per speed band and apply it as a **continuous interpolation over
v_ego** (`ROLL_COMP_SPEED_BANDS`: 5–10, 10–15, 15+ m/s; `roll_gain_at` in
`roll_comp_learning.py`):

- Collection floor drops 15 → 5 m/s (`ROLL_COMP_LEARN_MIN_V_EGO`); all quality gates
  (straightness, steer rate, |roll|, base-learner validity) unchanged. Points route
  into per-band roll-magnitude buckets.
- Per-band OLS fit with the existing validity gates (2000 points, span ≥ 0.25
  straddling zero, slope > 0, clip to [0.3, 1.0]).
- Profile payload (`RollCompGainParams`, version unchanged) gains an optional
  `bands` list; top-level fields stay as a mirror of the primary (15+) band so legacy
  payloads parse (mapped to a primary-only profile) and the device's learned gain
  survives the upgrade. Parse fails closed on any inconsistent field or band.
- Blending is band-wise; bands not refit in a cycle carry forward, so a highway-only
  drive never discards city-band evidence (and vice versa).
- Apply: the controller asks `learned_roll_gain_at(vEgo, base)` per frame. Anchors sit
  at band midpoints (top band at 20 m/s); **an unfitted primary band is pinned to the
  base gain**, so a city-learned gain never flat-extends to highway speeds, and a lone
  primary band reproduces today's constant-gain behavior exactly.
- Telemetry: `rollCompBandGains`/`rollCompBandPoints` on `liveTorqueParameters`
  (0 = band unfitted); the scalar `rollCompGain*` fields keep reporting the primary
  band. Offline gate: `roll_comp_profile.py --speed-bands` fits the same bands from
  rlogs; shadow passes when device and profiler agree within ±0.1 per fitted band.

## Caveat

Low-band `y` is reconstructed as `latAccelFactor·steer + latAccelOffset` with params
learned at ≥15 m/s, so low-band gains are *effective* gains that also absorb
torque-model speed-extrapolation error. That is correct to apply — the applied
correction rides the same extrapolated model — but the number is not a pure road-crown
constant and should not be compared across cars.

## Consequences

- No new params or settings: `RollCompGainMode` off/shadow/apply governs everything,
  fail-closed as before.
- Rollout: bands populate on city drives in shadow; flip nothing until
  `rollCompBandGains` agree with the offline profiler; if low bands converge to ≈ the
  highway gain, the extrapolation was fine all along, at zero behavior cost.
- Collection below 15 m/s still requires the base torque learner valid
  (`filtered_points.is_valid()`), which is seeded from ≥15 m/s driving or the params
  cache — a fresh device on pure city drives collects nothing until it has seen
  highway speeds once.
