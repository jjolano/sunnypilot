# Lateral Torque Context

This context records the project language for versioned lateral torque-control behavior.

## Language

**Torque v2**:
The established baseline lateral torque behavior used as the default comparison point.
_Avoid_: legacy torque, old torque

**Torque v2.1**:
An opt-in Torque v2 variant that preserves the Torque v2 response core while adding a refined final output governor.
_Avoid_: Torque v3, replacement v2, default torque

**Torque v3**:
An opt-in native-torque lateral behavior for sharper curve entry and exit without relaxing safety boundaries.
_Avoid_: lateral custom torque controller, universal torque controller

**Native torque car**:
A vehicle whose steering platform has a calibrated torque response suitable for torque-domain lateral control.
_Avoid_: any non-angle car, PID-origin car

**Processed curvature**:
The final controller-facing curvature after upstream path quality, maneuver decisions, lane-change shaping, and hard curvature/lateral-accel caps have been resolved.
_Avoid_: raw model path, model preview

**Delay lead**:
An anticipatory lateral-acceleration target adjustment that compensates for steering actuation lag.
_Avoid_: higher gain, curve boost

**Output governor**:
The final lateral torque guard that restricts command magnitude and command rate before platform safety limits are reached.
_Avoid_: safety model, fallback controller

**Refined Output Governor**:
The Torque v2.1 final output governor that smooths actuator command reversals and high-rate output without changing the Torque v2 response core.
_Avoid_: delay lead, response assist, safety cap

**Under-response Floor**:
A speed-shaped protection that prevents the Refined Output Governor from slowing low-speed catch-up when actual lateral acceleration is behind processed demand.
_Avoid_: torque boost, safety relaxation, authority change

## Relationships

- **Torque v3** is compared against **Torque v2** before promotion.
- **Torque v2.1** is compared against **Torque v2** before becoming a default recommendation.
- **Torque v2.1** keeps the **Torque v2** response core and adds only the **Refined Output Governor**.
- **Torque v3** applies only to a **Native torque car**.
- **Torque v3** follows **Processed curvature** rather than raw model preview.
- **Delay lead** shapes the requested lateral response.
- The **Output governor** can only restrict Torque v3 output.
- The **Refined Output Governor** is a **Tunable Lateral Conditioning Cap**, not a hard safety limit.
- The **Under-response Floor** can loosen the **Refined Output Governor** below the low-speed transition band, but it does not relax platform safety limits.
- Without EnforceTorqueControl, a **Native torque car** uses the v0 compatibility shim while non-native torque cars keep their stock controller.
- With EnforceTorqueControl, a **Native torque car** uses the selected torque version and non-native torque cars keep their stock controller.

## Example dialogue

> **Dev:** "Should Torque v3 look directly at raw model preview to enter curves earlier?"
> **Domain expert:** "No. Torque v3 follows processed curvature, then uses delay lead to account for actuator lag. Raw path interpretation stays upstream."

## Flagged ambiguities

- "lateral custom torque controller" was used for the planned controller; resolved term: **Torque v3**.
- "learning" was ambiguous between planned adaptation and current Torque v3 behavior; current Torque v3 has no learner term in this context.
- "actuator refinement" was ambiguous between Torque v3 control-law behavior and final output smoothing; resolved term: **Refined Output Governor**.
- "responsiveness" was ambiguous between subjective feel and catch-up behavior; resolved term: **Under-response Floor**.
