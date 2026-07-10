# Remove the dormant Evidence Snapshot

Status: accepted
Date: 2026-07-10
Supersedes: proposed [Evidence Snapshot architecture](2026-06-18-evidence-snapshot-architecture.md)

## Context

`sunnypilot/custom/evidence/` has no runtime consumer. The proposed first lateral adapter did
not land, while lateral and longitudinal modules evolved their own extraction paths. Keeping a
zero-consumer contract is a hypothetical seam: it adds no leverage or locality, and it invites
future code to depend on an unproven shape.

## Decision

Delete the Evidence Snapshot module and its tests without replacement. Reintroduce a stateless,
in-process evidence module only when at least two runtime consumers need one shared contract;
that future decision must re-establish source-health, replay, and no-new-process constraints.

## Consequences

- No runtime behavior changes with the deletion.
- Existing consumers retain their current extraction until a real shared seam exists.
