# Lead-Following Cushion Design

## Problem

Route `000000df--bbdd748dfa` shows several engaged longitudinal cases where the car reacted later or harder than the manual driving profile suggests is desirable:

- Segment `6`, around `414-418s`: high-closing stopped or near-stopped lead, plan source `lead0`, `aTarget` near `-2.1 m/s^2`, measured `aEgo` near `-2.3 m/s^2`, with earlier lead flicker.
- Segment `32`, around `1973-1981s`: confirmed moving lead began braking hard, planner source flickered between `cruise`, `lead0`, and `lead1`, then `aTarget` reached about `-3.0 m/s^2` and measured `aEgo` reached about `-3.3 m/s^2`.
- Segment `5`, around `340-342s`: confirmed slower moving lead, `dRel` near `22m`, `vRel` near `-3.2 m/s`, and `aTarget` near `-2.4 m/s^2`.

Manual profile evidence from the same completed route supports a coast-first style for normal-speed lead closing:

- Manual coast accel at `>=7 m/s` was roughly `-0.30` to `-0.05 m/s^2` at p50-p90.
- Manual `13-20 m/s` lead-following used median time gap around `1.58s` with frequent mild closing.
- Manual low-speed lead crawl was mostly coast/brake, not eager gas.

The current code already has several partial behaviors, but they do not cover the full target:

- `get_lead_gap_comfort_a_min` provides light braking only after the gap is already below target and disables for real closing.
- `get_lead_accel_match_target` follows lead acceleration/deceleration, but only when the lead acceleration estimate is non-trivial.
- `get_moving_lead_stop_approach_comfort_target` has a cushion idea for confirmed active slowing/stopping leads, but not for steady normal-speed closing.
- `get_lead_stop_approach_comfort_target` caps stopped-lead approach decel, but high-closing stopped-lead windows can still command stronger braking through the obstacle geometry and source transitions.

## Goals

- Prefer coast or acceleration taper before braking when ego is faster than a moving lead and approaching the target follow distance.
- Allow a moving lead to recover distance naturally when the situation is non-urgent.
- Smooth confirmed slowing-lead and stopped-lead approaches so braking starts earlier and peaks lower when enough runway exists.
- Preserve decisive braking for urgent close-range or high-required-decel cases.
- Keep normal configured follow gaps unchanged; this changes approach dynamics, not steady-state target spacing.
- Keep the change self-contained on `feat/longitudinal-follow-gap`.

## Non-Goals

- No speed-limit auto-cruise, SCC map/vision, or decision-layer policy changes.
- No no-lead e2e stop-approach tuning; segment `50` belongs to `feat/longitudinal-e2e-stop-approach`.
- No change to FCW or stock AEB behavior.
- No UI or parameter changes.

## Recommended Approach

Add a lead-following cushion layer in `long_mpc.py` that is separate from target gap calculation and uses existing MPC hooks:

1. **Normal-speed moving lead closing cushion**
   - Add a helper that activates when the dominant lead is moving, ego is faster than the lead, the lead gap is approaching the target follow distance, and the gap remains above a safety/comfort floor.
   - It should output a soft acceleration target near coast or light decel, with low-to-moderate cost.
   - The target should begin as an acceleration taper/coast preference and only become light decel as the cushion is used.
   - It should turn off for urgent closing, near danger distance, stopped/near-stopped leads, and clear pullaway/opening leads.

2. **Confirmed slowing moving lead smoothing**
   - Reuse or extend the existing moving-stop approach target so that confirmed lead braking does not wait until the normal obstacle geometry demands a large correction.
   - For non-urgent cases, blend toward coast/light decel first and only scale toward the full required decel as cushion is consumed or urgency increases.
   - Keep the current hard-braking lead guard cases active so segment-like `dRel < desired_gap` or high required decel still receives a brake target.

3. **Stopped or near-stopped lead approach smoothing**
   - Adjust stopped-lead comfort targeting so route-like high-closing stopped leads use the available runway to avoid unnecessarily high peak decel where possible.
   - Preserve stronger braking when the runway-required decel or danger margin indicates an urgent stop.

These three helpers should feed the existing combined acceleration target/cost path alongside accel-match, crawl, stopped-lead stop approach, moving-stop approach, and surge damping. This keeps the change localized and avoids globally changing obstacle geometry or steady follow distance.

## Alternatives Considered

### Tune only stop-approach helpers

This is lower risk for the segment `6` stopped-lead complaint, but it would not address the user's normal-speed moving-lead requirement. It would also leave steady moving-lead closing dependent on obstacle geometry until the gap is already tight.

### Change core approach-follow-distance geometry

Changing `get_approach_follow_distance` or the cost obstacle shape could make all lead approaches less aggressive with fewer helper paths. The risk is broader regression: every lead-following scenario would see changed obstacle geometry, including urgent cut-ins and lead transitions.

### Recommended helper-layer approach

The helper-layer approach is more targeted. It can be gated by lead speed, closing speed, gap region, required decel, and urgency, while leaving steady gaps and hard safety geometry intact.

## Data Flow

- Inputs come from `lead_brake_xv_*`, `lead_*_brake_a_traj`, current `v_ego`, and `t_follow` inside `LongitudinalMpc.update`.
- Existing helper outputs are combined into `combined_accel_targets` and `combined_accel_costs`.
- The new moving-lead cushion helper should produce target/cost arrays and be selected for the dominant lead, like accel-match and surge damping.
- The stopped/slowing approach updates should keep using the same combined acceleration target path rather than changing published planner fields directly.

## Safety Gates

- Disable moving-lead coast/cushion when the gap is below the lead danger distance plus headroom.
- Disable or fade it out when required stop/runway decel is high enough to require real braking.
- Disable it for lead speeds below the stopped/near-stopped range, letting stopped-lead and crawl helpers own those cases.
- Preserve existing decel caps and tests for close hard-braking leads.
- Keep FCW and crash checks unchanged.

## Test Plan

Add focused helper-level tests in `selfdrive/controls/tests/test_following_distance.py`:

- Normal-speed moving lead, ego faster, gap approaching target: expect a coast/light-decel target and nonzero cost.
- Same scenario at a safe larger gap or opening lead: expect no target.
- Same scenario below danger/comfort floor or urgent closing: expect no coast preference, preserving hard braking.
- Segment `5`/`32`-like slowing moving lead: expect earlier moderate target rather than late max decel when runway allows.
- Segment `6`/`29`-like stopped lead: expect stopped-lead comfort target to avoid excessive peak decel when runway allows.
- Regression tests for existing hard-braking close lead behavior must continue to pass.

Run verification with:

- `uv run --extra testing pytest selfdrive/controls/tests/test_following_distance.py -q`
- `uv run --extra testing ruff check selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py selfdrive/controls/tests/test_following_distance.py`

If the fresh worktree still lacks generated modules, build the generated pieces first with the repository's normal build command before rerunning tests.

## Follow-Up Work

- Segment `50` no-lead e2e stop approach should get its own spec and implementation on `feat/longitudinal-e2e-stop-approach`.
- Speed-limit/lead-flicker arbitration should wait until lead-following and no-lead e2e improvements are verified.
