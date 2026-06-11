# Redesign Torque v3 Around Native-Torque Delay-Lead Control

Status: accepted

## Amendment: Session Learner Deferred

The native-torque, processed-curvature, and delay-lead parts of this ADR remain accepted. The bounded session response-scale and trim learner described in the original decision is not implemented in the current Torque v3 controller. Existing learner/model-authority telemetry fields are retained for schema compatibility and populated as evidence-only identity placeholders until a separate telemetry migration deliberately changes downstream consumers.

Torque v3 replaces the previous universal adaptive-controller design in place while keeping Torque v2 as the default baseline. The new design is native-torque-only, follows processed curvature, improves curve entry and exit through delay-aware lateral-acceleration control, and limits runtime learning to bounded session response-scale and trim corrections.

## Considered Options

- Keep the old universal V3 with synthetic torque support and learned model authority.
- Introduce a new V4 selector value and preserve the old V3 implementation.
- Replace V3 in place with a fresh native-torque controller while keeping V2 default.

## Consequences

- PID-origin and other non-native-torque cars keep their original lateral controller when V3 is selected.
- Existing speed-aware, manual override, and NNLC torque hooks are not stacked into fresh V3 initially.
- Route A/B validation must compare cold-start and learner-warmed V3 behavior against V2.
