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
The final lateral path demand after upstream path quality and maneuver decisions have been resolved.
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
_Avoid_: torque boost, safety relaxation, response scale

**Bounded session learner**:
A drive-session-only adaptation that can make limited response-scale and trim corrections from clean evidence.
_Avoid_: persistent learner, synthetic torque model

**Response scale**:
A bounded adjustment to how strongly Torque v3 responds to lateral-acceleration error and delay lead.
_Avoid_: authority, raw torque boost

**Trim**:
A small temporary correction for steady lateral-acceleration bias.
_Avoid_: integral, learned offset

## Relationships

- **Torque v3** is compared against **Torque v2** before promotion.
- **Torque v2.1** is compared against **Torque v2** before becoming a default recommendation.
- **Torque v2.1** keeps the **Torque v2** response core and adds only the **Refined Output Governor**.
- **Torque v3** applies only to a **Native torque car**.
- **Torque v3** follows **Processed curvature** rather than raw model preview.
- **Delay lead**, **Response scale**, and **Trim** shape the requested lateral response.
- The **Output governor** can only restrict Torque v3 output.
- The **Refined Output Governor** is a **Tunable Lateral Conditioning Cap**, not a hard safety limit.
- The **Under-response Floor** can loosen the **Refined Output Governor** below the low-speed transition band, but it does not relax platform safety limits.
- The **Bounded session learner** can only influence **Response scale** and **Trim**.

## Example dialogue

> **Dev:** "Should Torque v3 look directly at raw model preview to enter curves earlier?"
> **Domain expert:** "No. Torque v3 follows processed curvature, then uses delay lead to account for actuator lag. Raw path interpretation stays upstream."

## Flagged ambiguities

- "lateral custom torque controller" was used for the planned controller; resolved term: **Torque v3**.
- "learning" was ambiguous between persistent platform learning and current-drive adaptation; resolved term: **Bounded session learner**.
- "actuator refinement" was ambiguous between Torque v3 control-law behavior and final output smoothing; resolved term: **Refined Output Governor**.
- "responsiveness" was ambiguous between subjective feel and catch-up behavior; resolved term: **Under-response Floor**.
