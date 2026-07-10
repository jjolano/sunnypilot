# Longitudinal Mode is latched per engagement

Status: accepted
Date: 2026-07-10
Clarifies: [clean-room longitudinal architecture](2026-06-13-clean-room-longitudinal-architecture.md)

## Context

The clean-room longitudinal ADR says **Longitudinal Mode** is latched per onroad cycle, but
the product requirement is narrower: ACC, E2E, and SCC must never change while engaged;
they may change between engagements without requiring a full offroad transition. Future
maintainers need this distinction because live mode refresh is an attractive but invalid
shortcut, while a full onroad-cycle latch needlessly delays a deliberate mode change.

## Decision

`CustomLongitudinalMode` is an **Engagement-Cycle Latch**. Capture it when controls engage,
hold its evidence admission unchanged until disengagement, and use that captured value for all
planner work in the engagement. Writes while engaged are accepted and persisted, but apply only
at the next engagement. `selfdrived` owns capture and publishes the active value through the
existing `selfdriveStateSP` message; plannerd consumes that value. Card keeps using the existing
`selfdriveState.experimentalMode` value. This decision applies to mode selection only;
custom-longitudinal enablement remains a separate decision.

Custom longitudinal lead trackers consume **Lead Evidence** only from the existing lead-fusion
seam. Raw `modelV2` path geometry does not alter lead risk, progress, or candidate authority in
any mode. A future model-path use requires an explicit evidence class and mode decision.

## Consequences

- The mode module has one authoritative engagement lifecycle; parameter refresh cannot alter
  current custom authority.
- The custom message gains one active-mode field; no new process or Param is needed.
- ACC cannot gain positive progress through a raw-model-path shortcut labeled as lead evidence.
- Mode-admission and lifecycle tests cover engage, disengage, and deferred writes.
- This supersedes only the latch timing in the proposed clean-room ADR; its evidence-admission
  and MPC-ownership decisions remain unchanged.
