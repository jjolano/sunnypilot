# Block-jackknife confidence for learned lateral gains

Status: accepted
Date: 2026-08-03
Relates to: `docs/adr/2026-07-02-learned-roll-compensation-gain.md`,
`docs/adr/2026-07-17-speed-resolved-roll-comp-gain.md`,
`sunnypilot/custom/lateral/block_jackknife.py`,
`sunnypilot/custom/lateral/roll_comp_learning.py`,
`sunnypilot/custom/lateral/direction_gain_learning.py`,
`selfdrive/locationd/torqued.py`,
`sunnypilot/selfdrive/locationd/torqued_ext.py`.

## Context

The Phase 1 lateral estimator gated learned gains on classical OLS confidence built from
per-frame point counts. That is statistically invalid for apply gating: `livePose` is a
20 Hz service, direction-gain pairs share most of their 1.35 s excursion window, roll
samples are adjacent and serially correlated, and every-fifth evidence is an explicit
decimation of a shared clock that must still advance on every livePose. IID/classical
standard errors understate the true uncertainty of serially correlated evidence, so an
OLS-relSE gate can bless a profile that is not actually stable.

## Decision

Gate all learned gain apply paths on **delete-one-block jackknife uncertainty over
deterministic, non-overlapping 60 s evidence blocks** (`EvidenceBlockClock`,
`fit_block_slope` in `block_jackknife.py`):

- **Cadence**: the evidence clock advances on every livePose opportunity (20 Hz), but
  roll/direction evidence is collected only every fifth update (`CUSTOM_EVIDENCE_DECIMATION = 5`,
  ~4 Hz evidence). The clock is session-local and time-derived; non-finite or backwards
  timestamps return no block, signal a discontinuity, and clear direction history without
  moving the clock.
- **Blocks**: each cycle is 60 s of evidence followed by a 5 s guard (`EVIDENCE_BLOCK_S`,
  `EVIDENCE_GUARD_S`). Profiles persist every 1,200 SubMaster frames — a true 60 s at the
  20 Hz poll (`TORQUED_PROFILE_PERSIST_INTERVAL_FRAMES = int(60 / DT_MDL)`) — skipping
  frame zero, and only completed blocks are eligible.
- **Fit**: per-band slope fit with centered, block-specific intercepts (per-block `Sxx`/`Sxy`
  pooled after removing each block's own mean), delete-one-block jackknife standard error,
  and a gate of `relSE <= MAX_BLOCK_REL_SE = 1/3`. A band needs at least 12 completed
  informative blocks (`MIN_EVIDENCE_BLOCKS`); a block is informative only when it clears
  the per-block point/span floors (roll: ≥ 20 points and ≥ 0.10 x-span; direction: ≥ 8
  pairs and ≥ 0.03 delta-span).
- **Roll gates**: slope within `[ROLL_GAIN_MIN, ROLL_GAIN_MAX] = [0.3, 1.0]`, ≥ 2,000
  informative points, x-span ≥ 0.25 straddling zero, positive slope, and every full/LOO
  block slope positive and in range.
- **Direction/ratio gates**: left/right slope ratio within `[RATIO_MIN, RATIO_MAX] = [0.7, 1.3]`
  for the fit and every leave-one-out value, band agreement ≤ 0.12, and per-direction plus
  ratio relSE ≤ 1/3. The ratio jackknife runs over the union of left/right block IDs from
  precomputed per-block sufficient statistics (O(points + blocks)).
- **No numeric blending**: persisted profiles are **snapshots**; a completed block set is
  replaced, never averaged with prior evidence. Repeated 60 s writes never add or average
  duplicate points. Band carry-forward is a churn guard only — an existing band is kept
  when the new fit moved more than `ROLL_BAND_REPLACEMENT_MAX_DELTA = 0.05`; the direction
  profile is replaced whole only when every band moved ≤ 0.05.
- **Fail-closed versions**: roll profiles are version 2 and direction profiles version 3;
  prior payloads (roll v1, direction v2) remain rejected. Parsers reject foreign restore
  keys, non-finite values, and any count/block/point invariant mismatch.
- **Fallbacks**: apply falls back to the fixed `ROLL_COMPENSATION_GAIN = 0.55` and identity
  direction scales on any parse or gate failure. Profile refresh while engaged/lat-active
  is deferred (`set_torque_override_refresh_allowed(not (enabled or lat_active))`), so a
  newly persisted profile never steps the steering mid-drive; it takes effect at the next
  disengage.
- **Observability**: `liveTorqueParameters` exposes block counts and jackknife uncertainty
  per profile/band (Phase 2 telemetry), and the Drive Lab `roll_comp_profile.py` profiler
  applies the same temporal-block statistical quality gates to its independent
  controls-state estimator (live and offline signals/collection gates are not identical).
  Apply deployment is not eligible until an old-gate noisy
  route is rejected and three clean independent routes all show relSE ≤ 1/3 with cross-route
  gain spread < 0.05 (Drive Lab `ROLL_COMP_VERDICT_MIN_ROUTE_COUNT = 3`,
  `ROLL_COMP_VERDICT_MAX_GAIN_SPREAD = 0.05`). **Route validation has not passed yet**; this
  gate is a precondition, not evidence.

## Superseded decision sections

Without editing their historical text, the following sections of prior ADRs are superseded:

- `2026-07-02-learned-roll-compensation-gain.md` — the **Decision** apply wiring (learned
  gain gated on the legacy parse/confidence checks) and the **Limits** v > 15 m/s
  straight/steady-state collection condition, and the **Consequences** promotion condition
  (engaged-route replay evidence).
- `2026-07-17-speed-resolved-roll-comp-gain.md` — the **Decision** per-band OLS fit with
  point-count/span validity gates and band-wise blending carry-forward, and the
  **Consequences** rollout condition (flip nothing until device and offline profiler agree
  per band).

The fixed 0.55 fallback (`2026-07-02-scale-roll-compensation-gain.md`) is not superseded;
it remains the apply fallback and the base anchor for unfitted bands.

## Consequences

- A learned profile reaches apply only when 12 completed informative blocks agree within a
  delete-one-block jackknife; correlated frames within a block are not counted as
  independent evidence. Route-wide bias is not bounded by the block gates and still
  requires the multi-route gate.
- Profiles survive parameter-restore/corruption the same way they always did: fail closed
  to the fixed constant or identity.
- The block gates add persistence latency (60 s blocks plus guard) before any learned gain
  can qualify; that is the point — nothing is applied on less than the minimum evidence.
