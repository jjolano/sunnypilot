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

## Relationships

- A **Low-Speed Lateral Envelope** is evaluated through separate **Low-Speed Tier** metrics.
- **Hard Lateral Safety Caps** win over both **Tunable Lateral Conditioning Caps** and controller output requests.
- **Tunable Lateral Conditioning Caps** may improve comfort, timing, or stability only inside **Hard Lateral Safety Caps**.
- **Path-State Evidence** classifies upstream demand conditioning before interpreting **Actuation Tracking Evidence**.
- **Actuation Tracking Evidence** measures controller and actuator response to **Processed Lateral Demand**.
- **Processed Lateral Demand** is the boundary between path processing and lateral actuation.

## Example Dialogue

> **Dev:** "Low-speed turns feel late and twitchy; should we increase torque?"
> **Domain expert:** "Check **Path-State Evidence** first. If processed demand is already gated or reversing, fix demand conditioning. If processed demand is stable but actual lateral acceleration lags or oscillates, inspect **Actuation Tracking Evidence**."

## Flagged Ambiguities

- "All low-speed" was ambiguous between one algorithm and one product scope; resolved as **Low-Speed Lateral Envelope** with tiered metrics.
- "Caps" was ambiguous between hard safety limits and tunable comfort behavior; resolved as **Hard Lateral Safety Cap** and **Tunable Lateral Conditioning Cap**.
