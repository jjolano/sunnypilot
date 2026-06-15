# Standstill Release is planner-owned

Status: accepted; implemented
Date: 2026-06-15
Relates to: [clean-room longitudinal architecture](2026-06-13-clean-room-longitudinal-architecture.md),
[longitudinal hypermile tuning](2026-06-13-longitudinal-hypermile-tuning.md), and
`docs/legacy/CONTEXT-longitudinal.md`.

## Context

Launch hesitation behind a stopped lead is caused by a boundary mismatch: the custom policy may authorize
a **Lead Pullaway Pulse**, but the published planner stop bit can remain true because the MPC is still
near-zero/timid at the stopped-lead gap. `controlsd` then correctly withholds resume because it only
transports planner intent.

## Decision

**Standstill Release** belongs to the **Longitudinal Planner**. It may clear the stopped-lead/MPC stop
commitment only when a stable known lead has already earned **Lead-confirmed Progress**, the lead then
shows **Fast Lead Motion Evidence** using the existing pullaway thresholds, model/traffic-control stops do
not bind, driver/system blockers are absent, and **Lead MPC** is not asking for real braking. A tiny
near-zero negative acceleration deadband may be treated as timid planner output, but release must not
override actual MPC braking.

When release fires, the planner may pair `shouldStop = false` with a small guarded positive acceleration
floor so resume and starting happen promptly. The release may be pre-armed while stopped behind a stable
known lead and held briefly with immediate cancellation on brake, model stop, lost/flickering/closing lead,
or MPC braking beyond the deadband.

Implementation note: the first implementation uses direct release predicates without an explicit latch; a
planner-owned latch may be added later if route logs show resume-pulse latency or chatter.

## Rejected alternatives

- **`controlsd` resume hack** — faster to implement but duplicates planner policy and bypasses the
  documented Stop/go Intent boundary.
- **MPC stop-gap tune** — broader physics change; inappropriate for a stopped-lead release problem and
  risks changing steady lead-follow behavior.
- **LongControl standstill bypass** — platform-risky; the controller should not ignore `cruiseState.standstill`.

## Validation

Before deployment, require unit coverage for stop-bit composition and no-release cases, plus a closed-loop
launch scenario proving a stable known lead opening clears `shouldStop` quickly while stopped/noisy/new/
model-stop/braking cases remain stopped.
