# Lateral Torque Context

This context records the project language for versioned lateral torque-control behavior.

## Language

**Torque v2**:
The established baseline lateral torque behavior used as the default comparison point.
_Avoid_: legacy torque, old torque

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
- **Torque v3** applies only to a **Native torque car**.
- **Torque v3** follows **Processed curvature** rather than raw model preview.
- **Delay lead**, **Response scale**, and **Trim** shape the requested lateral response.
- The **Output governor** can only restrict Torque v3 output.
- The **Bounded session learner** can only influence **Response scale** and **Trim**.

## Example dialogue

> **Dev:** "Should Torque v3 look directly at raw model preview to enter curves earlier?"
> **Domain expert:** "No. Torque v3 follows processed curvature, then uses delay lead to account for actuator lag. Raw path interpretation stays upstream."

## Flagged ambiguities

- "lateral custom torque controller" was used for the planned controller; resolved term: **Torque v3**.
- "learning" was ambiguous between persistent platform learning and current-drive adaptation; resolved term: **Bounded session learner**.
