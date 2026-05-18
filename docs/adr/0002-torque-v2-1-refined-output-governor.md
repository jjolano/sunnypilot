# Add Torque v2.1 as a Refined-Governor Variant of Torque v2

Status: accepted

Torque v2.1 is introduced as an opt-in native-torque tune that keeps the Torque v2 response core and adds a Refined Output Governor. The governor smooths command reversals, high steering-rate output, driver override release, and same-direction actuator limiting, while an Under-response Floor protects low-speed catch-up behavior below 9 m/s and fades that protection by 12 m/s.

## Considered Options

- Replace Torque v2 in place with the refined governor.
- Copy Torque v3's full delay-lead controller behavior into Torque v2.
- Add Torque v2.1 as a selectable Torque v2 core with only final-governor refinement.

## Consequences

- Torque v2 remains the baseline/default comparison point.
- Torque v2.1 can be A/B validated independently before any default recommendation changes.
- Torque v2.1 does not inherit Torque v3 delay-lead behavior or session response learning.
- Diagnostics must distinguish existing Torque v2 shaper reasons from final-governor reasons.
