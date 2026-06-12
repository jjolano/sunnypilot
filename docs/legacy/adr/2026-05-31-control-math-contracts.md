# Control Math Contracts

## Status

Accepted

## Context

Longitudinal, lateral, curve-speed, torque, and torque-learning code repeat the same physical formulas in several places. This makes safety-sensitive behavior harder to audit because sign conventions, unit assumptions, and sentinel handling are implicit at each call site.

This ADR documents the shared contracts introduced for future cleanup. The initial retained-baseline change only adds helpers and tests; downstream branches may adopt them where behavior is proven equivalent.

## Shared Formulas

- `smooth_speed_floor(v, floor) = sqrt(v^2 + floor^2)`. Use this when a hard `max(v, floor)` discontinuity would matter.
- `required_decel_to_target_speed(v_initial, v_target, distance, min_distance) = (v_target^2 - v_initial^2) / (2 * max(distance, min_distance))`.
- Required deceleration is signed. Slowing from a higher speed to a lower target returns a negative acceleration.
- `stopping_decel(v, distance, min_distance)` is the target-speed-zero case of required decel.
- `speed_for_lateral_accel(a_lat, curvature) = sqrt(a_lat / abs(curvature))` for valid non-negative lateral acceleration and non-zero curvature.

## Sign Conventions

- Curvature sign follows the existing controller/model convention at each call site.
- Lateral acceleration from curvature is `curvature * v^2`, with roll compensation applied explicitly by the caller or by a helper whose name says so.
- Roll compensation must distinguish the existing legacy linear approximation from exact `sin(roll) * g` paths.
- Torque sign conventions remain controller-local; shared math helpers must not encode actuator polarity.

## Safety And Comfort Boundaries

- Hard safety caps can only restrict output. Examples include lead safety, stop safety, FCW/AEB, force-slow, steering safety, curvature, lateral acceleration, and acceleration limits.
- Comfort governors and policy relaxations may shape demand, but must not collapse into or weaken safety caps.
- Generic math helpers return generic values such as `math.inf` for invalid curvature-speed calculations. Planner/UI code owns conversion to sentinels such as unset cruise speed.

## Processed Demand

Processed lateral demand means curvature or lateral acceleration after model-path validation, path shaping, and safety clipping. It is distinct from raw model curvature or raw actuator feedback.

## Horizon Time

Planner horizon code should use the actual model/control time grid when available. Fixed fallback `dt` values are acceptable only after the grid is validated as unavailable or invalid.

## Consequences

- New physical formulas should be added to the shared helper module when they are generic and dependency-light.
- Domain branches should merge the retained-baseline helper commit before using shared helpers.
- Behavior-preserving adoption should be tested at the call site before replacing repeated formulas broadly.
