# Route 261 fix plan (2026-07-06)

> **Status 2026-07-06: implemented same day** — outcomes inline per PR. Design notes that
> changed during implementation: the custom stack can only shape *down* from the MPC
> (final = min(mpc, custom)), so PR 1a reduces to the crawl-cap raise (launch ramp itself is
> MPC-owned); PR 1b became a rate-limited CautionRamp floor (model_trust.py) rather than a
> policy-side ramp+hysteresis — the flapping was a symptom of the -0.4 pin; PR 2 became an
> MPC-input radarState filter (cut_out_release.py) since a policy decel cap cannot override
> MPC lead braking; PR 3 found the crawl machinery already existed but was disabled under the
> confidence gate — enabled it for stationary-creep leads (research-gated) instead of building
> new creep; PR 4's investigation showed both drags were vision-only (map never corroborated)
> with one real turn taken faster by the driver and one model-path artifact — fixed with a
> rate-limited vTarget glide for uncorroborated vision slowdowns; PR 5 concluded **no change**
> (live learner 0.531 ≈ route-246 offline 0.56; right bias collapses to 3 cm with strict
> lane-change exclusion; lateral-learning HOLD verdict stands).

Fixes for the seven complaints diagnosed from route `00000261--5439b376a7`
(see memory `route-261-longitudinal-lateral-diagnosis`; rlogs cached at
`/tmp/opencode/sunnypilot-route-logs/00000261--*`). Every fix gets validated by
replaying the exact route-261 episode windows plus the fuzz presets, then a
deploy + fresh drive re-run of the `route-drive-diagnosis` skill.

## PR 1 — launch response + model-stop authority (pure policy, low risk)

### 1a. Launch: stop damping genuine pull-aways

- **Cause**: `LEAD_CRAWL_ACCEL_MAX = 0.55` / `LEAD_CRAWL_LAUNCH_TAU = 2.5`
  (`sunnypilot/custom/longitudinal/policy_tables.py`) governed the first ~1.5 s of the
  engaged launch (t=958–968): finalA plateaued 0.36–0.55, ego moved 2.2 s after the lead.
  Driver baseline: meanA 0.9–1.2 immediately.
- **Change**: audit why the crawl branch captured a strong pull-away (memory says
  `routine_breakout` MIN_LEAD_V is dead tuning — likely the breakout never fires from
  standstill). Exit crawl damping when the lead is genuinely accelerating
  (lead opening rate ≥ ~0.5 m/s and rising, or vLead crosses 2.5 with positive aLeadK);
  keep crawl only for the true accordion case. Raise `LEAD_CRAWL_ACCEL_MAX` 0.55 → 0.8
  for what remains.
- **Accept**: replayed 958–968 window shows ego start ≤ 1.2 s after hold release and
  meanA(0–2 s) ≥ 0.9; `fuzz_longitudinal --preset udacity-acc` and `openpilot-acc` green;
  no accordion regression in the stateful fuzzer.

### 1b. Model stops: continuous decel ramp + intent hysteresis

- **Cause**: `stop_approach_accel` pins at `STOP_APPROACH_COMFORT_DECEL[STANDARD] = -0.38`
  while modelA ramps to -2.0; the -1.5 floor only blips because the intent flaps
  cruise↔stop_approach frame-to-frame (4 forced-brake stops: t≈272/326/414/1082).
- **Change** (`policy.py` + `model_trust.py`): make the stop-approach candidate track
  required kinematic decel continuously —
  `clamp(stopping_decel(v_ego, model_stop_distance), STOP_APPROACH_DECEL_MIN, comfort)` —
  instead of comfort-until-hard. Add hysteresis to the trust/intent gate: enter
  stop_approach when required decel < -0.5 (persisting ~0.3 s), exit only after
  required > -0.25 for 0.5 s.
- **Guard**: distance-based ramp stays gentle far out — must not reintroduce the
  route-25a regression (far stops braking immediately); keep `_usable_coast_decel`
  split and `STOP_LANDING_*` soften untouched. Run the 25a fixture tests.
- **Accept**: replayed windows reach ≤ -1.2 m/s² before the point where the driver braked;
  no red tests in `sunnypilot/custom/longitudinal/tests`; leadless-stops profiler on the
  next drive shows stops completing without driver brake.

### 1c. Device param flip

- Set `CurveMemoryEnabled=0` (contradicts the committed default-off decision).

## PR 2 — cut-out decel cap (small, isolated)

- **Cause**: MPC kept braking (-0.5..-0.9) for leadOne already 2–3 m off path
  (t≈1041–1045, yRel +2..+3; also -1.24 for a turning lead 47–95 m out at t≈877).
  No path-clearance declamp exists (LeadPathClearance slot retired — the deleted
  feature was *anticipation*; this is a minimal exit-declamp, keep it tiny).
- **Change**: pure function in `policy.py`'s lead candidate path: when |yRel| > ~1.6 m,
  lead moving away laterally, and TTC > 4 s, floor the *lead-driven* decel candidate at
  -0.5 as an advisory cap. Never touches hazard/safety candidates — fail side is
  "brakes like today".
- **Accept**: replayed 1041–1046 stays ≥ -0.5 once yRel > 2; 876–880 peak decel halves;
  check `ncap-acc` preset for cut-out coverage and add one scenario if absent;
  cut-in behavior unchanged (CutInBrakeAssist tests green).

## PR 3 — stop gap + creep-to-close (behavior change, shadow first)

- **Cause**: stop-hold latch (`finalizer.py`) arms at `max(stopping_distance+2, 10 m)`
  once ego ≤ ~0.5 m/s near a stopped lead → froze 5.9 m and 9.3 m gaps (driver manual
  median 1.6 m); lead crept +1 m during a hold with zero response; design targets
  (`dynamic_safety_floor._STOP_DISTANCE = 6.0`, `LEAD_CONTEXT_STOP_DISTANCE = 5.0`)
  are already 3× driver preference; no creep mechanism exists.
- **Change**:
  1. Don't latch while still rolling with gap ≫ target: require
     `lead_d_rel ≤ creep_target + 2` (or drop the 10 m arm floor to ~7 m) so the MPC
     finishes the approach before the hold freezes it.
  2. Creep-to-close: while latched, lead stationary, and latched gap > target + 0.5 m,
     emit a capped creep (≤ 0.4 m/s², v ≤ 1.4 m/s) until gap ≤ target, then re-latch.
     Reuse the existing `gap_increasing_s` tracking; creep goes through the same
     `StandstillReleaseConfidence` gate as release (mode `gate` today) — fail closed
     to today's behavior.
  3. Retune: `_STOP_DISTANCE` 6.0 → 4.5, `LEAD_CONTEXT_STOP_DISTANCE` 5.0 → 4.0,
     creep target ≈ 4.0 m radar gap. (Driver prefers 1.6; take one bounded step,
     not the full jump.)
- **Rollout**: shadow mode first (log intended creep frames), one drive, then apply —
  param + settings per the `param-settings-ui` skill checklist.
- **Accept**: engaged stop gap median ≤ 5 m on next drives; a hold that starts > 5 m
  closes to target within ~6 s; no unintended creep against a lead that never moved
  (shadow log clean); Toyota standstill resume verified on-device.

## PR 4 — SCC Vision green-light drag (investigate, then gate)

- **Cause**: both "forced gas at green" events were SCC-Vision slowdowns pinned at
  `MIN_V = 20 km/h` (`smart_cruise_control/constants_v1.py`), dragging up to -1.4 m/s²
  with no lead while modelA ≥ 0.
- **Investigate first**: extractor doesn't capture the SCC struct — add
  `smartCruiseControl` state fields to `route-drive-diagnosis/scripts/extract_route_npz.py`
  and replay 1050–1058 / 1163–1171 to see which state held vision active through the
  intersection.
- **Candidate changes** (pick after investigation): exit/suspend the vision slowdown when
  the model disagrees (modelATarget ≥ +0.2 sustained ~1 s while vision demands < -0.5);
  cap vision-driven decel at -0.8 unless the curve entry is < ~3 s away.
- **Accept**: replayed windows release within ~1 s of the model turning positive; curve
  slowdowns on genuine curves unchanged (`speed_adaptive_verdict` / SCC tests green).

## PR 5 — roll-comp gain step (lateral, separate)

- **Cause**: right-of-center bias (median 6–7 cm, right clearance p5 0.21 m, 8.4 % of
  time < 0.25 m from a line) with deployed learned gain 0.531 vs Phase-1 OLS crown
  slope ~0.7 → under-compensation.
- **Change**: inspect the RollCompGain learner bounds first (it converged at 0.53 with
  confidence 1.0 — if it's clamped, raise the clamp; if not, question the learner
  before overriding). Take one bounded step toward ~0.62–0.65, not straight to 0.7,
  per the lateral learning program's gate discipline.
- **Accept**: route-246 next-route checklist — |offset| ≤ 0.05 m, no increase in
  0.05–0.3 Hz wander band power, no new left bias.

## Explicitly not fixing

- **Wander** (0.6–0.9 m p2p at 0.05–0.15 Hz): known model-curvature limit cycle; the
  lateral stack passes it through and LCA-layer fixes were already ruled out
  (wander-loop-owner diagnosis). Re-measure after PR 5 since the bias interacts.

## Sequencing and gates

PR 1 → PR 2 are pure-policy and can land immediately after fuzz + replay gates.
PR 3 needs the shadow drive between shadow and apply. PR 4 blocks on its
investigation step. PR 5 rides the lateral program's own gates. After each deploy:
`deploy-workflow` health check, one drive, re-run `route-drive-diagnosis` and compare
the per-complaint metrics against the route-261 baselines quoted above.
