---
title: Drive incident inbox
trigger_condition: Promote after the route incident finder CLI reliably identifies high-signal events across representative logs.
planted_date: 2026-07-29
---

# Drive incident inbox

Turn each drive into a ranked review queue without changing on-road behavior.

After a route, a device job should invoke the same analyzer used by `tools/drive_lab` and emit a compact incident manifest. Start with objective labels: engaged driver gas/brake/steer overrides and longitudinal comfort spikes. Add model/lead disagreement only after the initial labels are audited for usefulness.

The analyzer output must be portable so the first manual CLI workflow can later become automatic without a second detector implementation.
