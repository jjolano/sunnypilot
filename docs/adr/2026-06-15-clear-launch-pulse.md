# Clear Launch Pulse uses runway-confirmed Standstill Release

Status: accepted; implemented
Date: 2026-06-15
Relates to: [standstill release](2026-06-15-standstill-release.md),
[longitudinal hypermile tuning](2026-06-13-longitudinal-hypermile-tuning.md), and
`docs/legacy/CONTEXT-longitudinal.md`.

## Context

No-lead launch from standstill is a different authority path from **Lead Pullaway Pulse**. A missing lead
does not prove the runway is clear, so no-lead progress must be authorized by **Runway-confirmed** evidence,
not by raw lead absence.

## Decision

**Clear Launch Pulse** uses the same planner-owned **Standstill Release** mechanics as lead pullaway — clear
the planner stop bit and apply a small guarded positive acceleration floor — but it has separate runway
authority gates. The first implementation reuses the existing `no_lead_stop_clear` semantics: no active model
stop, model stop distance absent or far, model desired acceleration not indicating a hard brake, no
lead/shadow/flicker threat, no driver/system blocker, and cruise target requesting progress.

This ADR does not loosen clear-path thresholds. Any later reduction of clear distance or weakening of model
braking gates is a separate tuning decision that needs log/closed-loop evidence.

## Validation

Before deployment, require tests showing no-lead clear runway releases promptly, while model-stop, near stop
distance, model braking, lead/shadow/flicker threat, brake press, and force-slow cases do not release.
