# Lateral Control Context

This context records the project language for lateral path demand, low-speed lateral behavior, and actuation evidence.

## Language

**Low-Speed Lateral Envelope**:
The product scope for lateral behavior below the normal-speed transition band, evaluated by tier rather than as one averaged behavior.
_Avoid_: parking only, all turns, lateral tune

**Low-Speed Tier**:
A speed-bounded evaluation bucket for low-speed lateral behavior.
_Avoid_: mode, controller profile, platform class

**Hard Lateral Safety Cap**:
A lateral limit that may only restrict output and is not relaxed by low-speed behavior improvements.
_Avoid_: comfort tune, smoothing knob

**Tunable Lateral Conditioning Cap**:
A smoothing, gating, timing, or comfort-governor limit that can be tuned inside hard lateral safety caps.
_Avoid_: safety relaxation, platform limit

**Path-State Evidence**:
Telemetry that explains how raw path demand was accepted, gated, smoothed, or otherwise conditioned before actuation.
_Avoid_: controller response, torque lag

**Actuation Tracking Evidence**:
Telemetry that explains how closely lateral actuation followed the processed lateral demand.
_Avoid_: path quality, model confidence

**Processed Lateral Demand**:
The lateral demand after path quality and maneuver decisions have been resolved for controllers to follow.
_Avoid_: raw model path, actuator command

**Straight-Path Wander**:
Low-frequency side-to-side drift or weave on a broad straight-ish road while lateral control is active and the driver is not steering or signaling.
_Avoid_: fast torque twitch, lane-change behavior, curve tracking error

**Demand-Driven Straight-Path Wander**:
Straight-Path Wander where raw or processed path demand moves first and actual curvature follows that demand.
_Avoid_: torque lag, actuator oscillation

**Actuation-Driven Straight-Path Wander**:
Straight-Path Wander where Processed Lateral Demand is comparatively stable but steering or actual curvature oscillates.
_Avoid_: model path noise, path smoothing

**Lateral Performance Gate**:
A route-analysis validation report that combines Path-State Evidence, Actuation Tracking Evidence, low-speed tier metrics, and branch-ownership guidance before behavior work is opened.
_Avoid_: subjective drive score, single-metric pass/fail

**Dominant Lateral Failure Class**:
The strongest route-level lateral evidence class selected by the Lateral Performance Gate for ownership triage.
_Avoid_: root cause, user-facing mode

**Recenter Overshoot Candidate**:
A validation-only Straight-Path Wander window where lateral-offset evidence drifts to one side, correction demand returns toward center, then offset crosses or reverses beyond center with follow-up opposite correction.
_Avoid_: implemented recenter controller, lane-centering policy

## Relationships

- A **Low-Speed Lateral Envelope** is evaluated through separate **Low-Speed Tier** metrics.
- **Hard Lateral Safety Caps** win over both **Tunable Lateral Conditioning Caps** and controller output requests.
- **Tunable Lateral Conditioning Caps** may improve comfort, timing, or stability only inside **Hard Lateral Safety Caps**.
- **Path-State Evidence** classifies upstream demand conditioning before interpreting **Actuation Tracking Evidence**.
- **Actuation Tracking Evidence** measures controller and actuator response to **Processed Lateral Demand**.
- **Processed Lateral Demand** is the boundary between path processing and lateral actuation.
- **Straight-Path Wander** is evaluated at low frequency; fast reversal or twitch evidence belongs to **Actuation Tracking Evidence** unless processed demand is stable and the motion persists as broad weave.
- **Demand-Driven Straight-Path Wander** belongs to **Path-State Evidence** before any controller tuning is considered.
- **Actuation-Driven Straight-Path Wander** belongs to **Actuation Tracking Evidence** before any path smoothing is considered.
- A **Recenter Overshoot Candidate** is evidence for later investigation only; it is not permission to create a second lane-centering controller.

## Example Dialogue

> **Dev:** "Low-speed turns feel late and twitchy; should we increase torque?"
> **Domain expert:** "Check **Path-State Evidence** first. If processed demand is already gated or reversing, fix demand conditioning. If processed demand is stable but actual lateral acceleration lags or oscillates, inspect **Actuation Tracking Evidence**."

> **Dev:** "The car wandered off center and corrected back, then seemed to wander to the other side. Should we add a recenter curve?"
> **Domain expert:** "First classify whether this is **Demand-Driven Straight-Path Wander** or **Actuation-Driven Straight-Path Wander**. A **Recenter Overshoot Candidate** can justify more evidence collection, but not a second lane-centering controller."

## Flagged Ambiguities

- "All low-speed" was ambiguous between one algorithm and one product scope; resolved as **Low-Speed Lateral Envelope** with tiered metrics.
- "Caps" was ambiguous between hard safety limits and tunable comfort behavior; resolved as **Hard Lateral Safety Cap** and **Tunable Lateral Conditioning Cap**.
- "Wandering" was ambiguous between low-frequency path drift, lane-centering bias, curve weave, and fast torque twitch; resolved as **Straight-Path Wander** with separate demand-driven and actuation-driven evidence classes.
- "Recenter correction" was ambiguous between validation evidence and a behavior feature; resolved as **Recenter Overshoot Candidate** until behavior work is explicitly accepted.
