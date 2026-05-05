 # Lead-Aware Speed-Up Guard Design

## Problem

Route `000000eb--b053886a1a` showed positive longitudinal acceleration while a radar lead was present and closing. The largest sustained case was `581.50s-587.24s`, where speed rose from about `14.2` to `15.2 m/s` while `leadOne.status=True`, the minimum lead gap was `18.7 m`, and relative speed reached `-1.6 m/s`. The selected sunnypilot source was `speedLimitAssist`, which was allowed to raise the target toward the speed-limit auto-cruise target before the core MPC switched from cruise to `lead0`.

A shorter lane-change handoff case showed the same class of behavior: the planner briefly fell back to cruise during lead flicker and allowed positive acceleration before the closer lead stabilized.

## Goal

Prevent speed-limit auto-cruise and normal cruise target generation from commanding positive speed-up into a close, closing lead, while preserving normal acceleration when the lead is far, opening, or the driver is intentionally overriding with gas.

## Approach

Add a small lead-aware speed-up guard in `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`, where the SP planner builds cruise and speed-limit targets. This layer is the narrowest place that can cover both observed paths: active Speed Limit Assist and normal cruise fallback during lead handoff.

The guard activates when all of these are true:

- `radarState.leadOne.status` is true.
- Driver gas and brake are not pressed.
- Lead is closing by at least a small relative-speed threshold.
- Lead distance is within a time-gap based threshold, with a low-speed minimum distance floor.

When active, the guard caps the affected target to avoid speed-up:

- Target speed is no higher than current ego speed.
- Target acceleration is no higher than coast, so positive acceleration seeds become `<= 0.0`.

Lower advisory targets from SCC vision, SCC map, and OSM traffic-control priors remain unchanged. This keeps curve/map caution behavior independent and avoids expanding the fix into unrelated branches.

## Tests

Add focused unit coverage in `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`:

- Active Speed Limit Assist speed-up is blocked by a close, closing lead.
- Normal cruise speed-up is blocked by a close, closing lead when SLA is inactive.
- The guard does not block acceleration for a far lead, an opening lead, or driver gas override.
- Existing speed-limit target selection behavior remains unchanged when no lead guard is active.

Run the retained branch target-selection tests and the route-relevant longitudinal regression set before propagation.

## Branch Ownership

Implement on `feat/speed-limit-auto-cruise`. The changed code lives in the SP longitudinal target-selection layer already owned by the speed-limit auto-cruise branch, and the primary sustained issue was Speed Limit Assist auto-cruise speed-up. Downstream retained branches must be propagated afterward before rebuilding `custom`.
