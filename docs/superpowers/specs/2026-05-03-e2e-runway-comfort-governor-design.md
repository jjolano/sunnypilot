# E2E Runway Comfort Governor Design

## Problem

Route `000000e4--15f682f800` contains a bookmarked longitudinal event where the car began braking harder than necessary during an E2E-predicted slowdown with ample runway.

In the relevant window:

- Ego speed was about `17.2 m/s` with cruise set to `62 kph`.
- No stable radar lead was initially available.
- The model endpoint remained long-range, roughly `145m` shrinking toward `86m` over several seconds.
- `modelV2.action.shouldStop` stayed `False`.
- `modelV2.action.desiredAcceleration` ramped from roughly `-0.18` to `-1.24 m/s^2`.
- The planner used the raw E2E acceleration candidate directly through `min(output_a_target_e2e, output_a_target_mpc)`.
- The existing E2E stop-approach helper did not intervene because the endpoint did not indicate a distance shortage.

The result was a direct raw-model braking ramp instead of a coast-first approach. A human driver had enough runway to coast or use only light deceleration before committing to firmer braking.

## Goals

- Prefer coast or light deceleration for long-runway E2E slowdowns when `shouldStop` is false.
- Preserve stronger braking when the model indicates a real stop, the E2E stop-approach shortage helper activates, or the lead/MPC path requires braking.
- Avoid weakening confirmed lead-follow emergency behavior.
- Keep the change self-contained on `feat/longitudinal-e2e-stop-approach`.
- Keep UI, params, FCW, speed-limit, SCC, and decision-layer behavior unchanged.

## Non-Goals

- No changes to steady-state follow distance spacing.
- No changes to lead-transition lane-exit behavior.
- No changes to speed-limit auto-cruise, SCC map/vision, or OSM planner logic.
- No broad MPC obstacle geometry refactor.
- No configurable user-facing parameter in this iteration.

## Recommended Approach

Add a runway-aware E2E comfort governor in `selfdrive/controls/lib/longitudinal_planner.py` that filters only the raw E2E model acceleration candidate before planner arbitration.

The governor should:

- Activate only when E2E is active, the planner is engaged, `modelV2.action.shouldStop` is false, and the driver is not pressing gas or brake.
- Use model endpoint distance as the no-lead runway proxy.
- Treat coast acceleration from `get_coast_accel()` as the preferred long-runway target.
- Allow light braking as runway shrinks.
- Allow the raw E2E acceleration through when runway is short enough to require it.
- Apply a mild negative ramp limit to prevent sudden raw E2E decel jumps.
- Bypass when a radar lead is present so lead/MPC arbitration remains authoritative.

The filtered E2E acceleration should then be used in the existing arbitration:

```python
output_a_target = min(output_a_target_e2e, output_a_target_mpc)
```

This keeps MPC lead braking authoritative. If MPC computes stronger braking than the governed E2E candidate, MPC still wins. If the existing shortage helper returns a stronger stop-approach target, it still wins after the raw E2E candidate is filtered.

## Alternatives Considered

### Pure E2E rate limit

Rate limiting alone would reduce snap, but it would still eventually let raw model braking dominate even when there is enough runway to coast. It improves smoothness but does not encode the desired coast-first policy.

### Lead-detection grace period

A grace period after radar lead acquisition would help only after the radar lead appears. In the bookmarked event, aggressive braking started before the lead handoff stabilized, so this would be too late to address the core behavior.

### Extend `get_e2e_stop_approach_accel`

The existing helper is shortage-oriented: it adds braking when the model endpoint is shorter than expected. The bookmarked event had no initial endpoint shortage, so the problem is the raw E2E candidate policy, not the shortage helper. Keeping the new governor separate preserves the helper's meaning.

## Data Flow

- `LongitudinalPlanner.update()` computes `accel_coast` from pitch via `get_coast_accel()`.
- The planner reads raw model acceleration from `sm['modelV2'].action.desiredAcceleration`.
- A new helper computes a governed E2E acceleration from raw acceleration, coast acceleration, model endpoint, `shouldStop`, and driver/reset gates.
- The governed E2E acceleration replaces the raw value before `min(output_a_target_e2e, output_a_target_mpc)`.
- The existing E2E stop-approach helper remains after the main arbitration and can still override with stronger braking when a real shortage exists.

## Safety Gates

- Disable the governor when E2E is inactive.
- Disable when `shouldStop` is true.
- Disable when `reset_state`, `force_slow_decel`, brake, or gas is active.
- Disable when a radar lead is present.
- Disable when engage-time stop bootstrap is active.
- Disable when no finite positive model endpoint exists.
- Fade the cap out as runway approaches the required decel zone.
- Do not cap MPC lead output or FCW/crash checks.

## Test Plan

Add focused helper-level tests in `selfdrive/controls/tests/test_longitudinal_planner.py`:

- Long-runway E2E slowdown with raw `-1.2 m/s^2`: expect a governed coast/light-brake output.
- Short-runway E2E slowdown: expect raw stronger braking to pass through.
- `shouldStop=True`: expect raw stronger braking to pass through.
- Driver gas/brake or reset/force-slow gates: expect raw acceleration to pass through.
- Radar lead present: expect raw acceleration to pass through.
- Engage-time stop bootstrap active: expect raw acceleration to pass through.
- Negative ramp limiting: expect E2E deceleration to step down gradually instead of jumping.

Run verification with:

- `uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q`
- `uv run --extra testing ruff check selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_longitudinal_planner.py`

If helper-level tests pass, re-run Drive Lab explanation on the pulled bookmarked route to confirm the t=754-758 E2E ramp is softened in planner output under replay or direct helper reproduction.
