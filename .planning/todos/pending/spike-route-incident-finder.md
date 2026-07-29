---
title: Spike route incident finder
date: 2026-07-29
priority: medium
---

# Spike route incident finder

Create a `tools/drive_lab` CLI that scans one selected route and emits a compact incident manifest.

Initial incident classes:

- Driver gas, brake, or steering override while engaged.
- Longitudinal acceleration or jerk comfort spikes.

Each record should include timestamp, class, severity, and the relevant signal window. Keep the manifest suitable for a later post-drive device job; do not create a separate device-only detector.

Done when a representative route can be scanned explicitly and its resulting incidents can be ranked for manual inspection.
