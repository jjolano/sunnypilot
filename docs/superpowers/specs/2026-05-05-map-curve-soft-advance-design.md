# Map Curve Soft Advance Design

## Context

Latest route `000000ed--5029ca061c` showed a hot map/SCC curve entry. Around the main curve, `mapdOut.mapCurveSpeed` exposed a low curve target near `13.4 m/s` by about `763s`, but `longitudinalPlanSP.smartCruiseControl.map.vTarget` did not select a `<=13.5 m/s` cap until about `767.9s`. The lateral peak arrived at about `770.6s`, leaving only `2.7s` to shed speed.

SCC vision was active earlier, but its entering target stayed above or near cruise because model-predicted lateral acceleration was weak/late. The route therefore points to delayed map target selection/gating rather than a simple SCC-V gain problem.

## Goal

Start braking earlier for strong map curve targets when the model confirms a real curve but does not yet confirm the full map slowdown. Preserve protection against false map slowdowns from stale or inaccurate map data.

## Scope

Implement on `feat/longitudinal-osm-planner` in `sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py` and its tests.

Do not change SCC-V constants, speed-limit auto-cruise policy, or generic longitudinal planner arbitration in this design.

## Design

For large map slowdowns, keep the existing full-target confirmation path unchanged: if the model confirms the full map target, use the map target.

If full confirmation fails but the model predicts a real curve near the map target distance, compute the model-derived curve speed with `_prediction_curve_target(...)`. When that target is below ego speed and inside the map target control range, use it as an intermediate SCC-map cap instead of returning inactive.

This produces a soft advance path:

1. No model curve: keep ignoring the large map slowdown.
2. Weak-to-moderate model curve: select a bounded intermediate target, such as the model-confirmed curve speed.
3. Strong model confirmation: transition to the full map target.

The intermediate target should never be lower than the map target and should never be higher than ego speed. It should only exist while the same model/distance evidence is present, so existing release behavior remains intact when prediction drops.

## Data Flow

`SmartCruiseControlMap.update_calculations(...)` reads map target velocities/advisories and model data. `_target_range_state(...)` decides whether a target is active. The new behavior belongs in that decision path because it is where large map slowdowns are currently rejected before SCC-map can publish a target.

The published target remains `smartCruiseControl.map.vTarget`, so downstream target selection and `longitudinalPlanSP` serialization do not need new fields.

## Safety Constraints

Do not trust an unconfirmed large map slowdown blindly.

Do not select an intermediate target without finite model positions, velocities, and yaw rates covering the target distance.

Do not relax small map targets upward once the full map target is already confirmed.

Keep stale map-param invalidation behavior unchanged.

## Tests

Add map-controller tests for:

1. A large map target with no model curve still remains inactive.
2. A large map target with weak model curve evidence produces an intermediate cap instead of `V_CRUISE_UNSET`.
3. A strong model curve still selects the full map target.
4. The intermediate cap releases when model evidence disappears.
5. The same intermediate target path works for advisory targets, not only `MapTargetVelocities`.
6. A fully confirmed large raw map target releases if later model evidence no longer confirms it.

Run `uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py` in the owning retained branch worktree.
