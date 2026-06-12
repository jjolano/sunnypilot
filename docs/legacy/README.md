# Legacy fork reference

Reference material imported from the retired `custom` branch (frozen at `8a97000e19`).
These documents describe the *old* implementation and the decisions behind it; they are
the porting source of truth for the restart plan
([2026-06-12-fork-restart-reimplementation](../plans/2026-06-12-fork-restart-reimplementation.md)),
not living documentation of this branch. New decisions get new ADRs under `docs/adr/`.

- `CONTEXT-*.md` — domain language for longitudinal planning and lateral control/torque.
  The language largely carries forward; terms tied to dropped machinery (stack selection,
  promotion gates, planner stacks) are retired.
- `adr/` — decisions preserved by the rewrite: the planner/MPC authority boundary,
  torque v2.1's refined output governor, the custom longitudinal v2 architecture and
  telemetry, control math contracts, and longitudinal modes.
- `concepts/` — behavior specs for custom longitudinal v2 and the candidate-authority
  contract; Phase 4 parity is measured against these.
- `specs/` — the implemented smooth-assertive longitudinal style (Phase 4 parity
  reference) and the four unimplemented improvement specs that form the Phase 5 backlog.
- `tuned-constants.yaml` — inventory of tuned constants extracted from the old fork's
  kept modules, with source locations. The rewrite consumes values from here instead of
  rediscovering them.
