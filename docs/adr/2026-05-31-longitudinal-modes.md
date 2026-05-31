# ADR: Top-Level Longitudinal Modes

## Status

Accepted for staged implementation.

## Context

The legacy longitudinal UI mixed `ExperimentalMode`, `DynamicExperimentalControl`, SCC Vision, SCC Map, SLA, and OSM toggles into runtime policy. That made it possible for backup restore, Sunnylink saves, or hidden onroad toggles to resurrect behavior after a user-facing mode decision.

ACC purity also needs an input boundary: ACC may use car/radar lead-follow behavior, but must not build model-stop, model-path, map, OSM, SLA, or SCC curve actuation candidates.

## Decision

Introduce `LongitudinalMode` as the source of truth with values `ACC`, `E2E`, and `SCC`.

`LongitudinalModeResolver.resolve(...)` owns mode interpretation and returns requested mode, resolved implementation, actuation type, restriction status, and compatibility alias state. Legacy params are migration and compatibility inputs only after `LongitudinalModeMigrationVersion` is current.

`ACC` is deterministic cruise/follow. It does not read model stop/action/path policy or build map, OSM, SLA, or SCC curve candidates. `E2E` is model-primary with physical restrictions. `SCC` is the public DEC replacement and resolves through explicit SCC evidence rather than reading legacy DEC runtime state.

SCC curve source controls are mode-owned params: `SccCurveVisionEnabled` and `SccCurveMapEnabled`. They replace the legacy `SmartCruiseControlVision` and `SmartCruiseControlMap` controls in device UI and Sunnylink, and they only apply when `LongitudinalMode == SCC`.

## Consequences

- Fresh installs default to `ACC`.
- One-time migration maps legacy `ExperimentalMode + DynamicExperimentalControl` to `SCC`, legacy `ExperimentalMode` to `E2E`, and all other legacy states to `ACC`.
- Sunnylink saves and backup restores skip legacy longitudinal mode params after migration.
- `longitudinalPlanSP.dec` remains a compatibility alias; new telemetry is published under `longitudinalPlanSP.longitudinalMode`.
- Device UI and Sunnylink expose `LongitudinalMode`; legacy mode toggles remain migration/compatibility inputs only.
