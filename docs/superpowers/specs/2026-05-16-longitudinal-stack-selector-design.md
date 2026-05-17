# Longitudinal Stack Selector Design

## Summary

Introduce a versioned longitudinal stack selector so this fork can choose between true upstream openpilot behavior, immediate upstream sunnypilot behavior, and a new custom longitudinal stack. The selector separates actuator takeover permission from stack implementation, gives `custom-*` a safe shadow fallback to `sunnypilot-current`, and creates durable telemetry so route analysis and future agents can understand which stack produced each command.

This document is intentionally operational. Future implementation agents should treat it as the source of truth for the first rollout unless a later ADR/spec supersedes it.

## Locked Decisions

- Keep `AlphaLongitudinalEnabled` as the gas/brake takeover gate on alpha-long-capable cars.
- Add one flat stack-selection param, tentatively `LongitudinalStack`.
- Default/unset `LongitudinalStack` resolves to `sunnypilot-current`.
- Supported user-facing stack families are `openpilot-current`, `sunnypilot-current`, and `custom-*`.
- Expose custom versions, including `custom-recommended` and literal versions such as `custom-1.0`.
- Store `custom-recommended` as a moving alias, not as a literal resolved version.
- Resolve `custom-recommended` per platform, using a capability profile plus fingerprint overrides.
- If no custom version is recommended for the platform, `custom-recommended` resolves to `sunnypilot-current` and the UI should say so.
- Allow users to force custom versions that are available but not recommended for their platform.
- Hide or disable custom versions that are not available for the platform, except for deliberate developer/manual override paths.
- Latch stack selection until restart/onroad cycle; do not hot-switch longitudinal stack while engaged.
- `openpilot-current` is selectable only where true upstream openpilot longitudinal is valid. Factory/stock longitudinal remains `AlphaLongitudinalEnabled=false`.
- `openpilot-current` must remain hidden/disabled until its adapter is implemented; requesting it manually should resolve safely to `sunnypilot-current`.
- `openpilot-current` tracks the openpilot baseline as carried by sunnypilot upstream, not comma `master` directly.
- `openpilot-current` and `sunnypilot-current` should use adapter wrappers around upstream-current code where possible; copy code only when a clean boundary cannot be exposed.
- `custom-*` owns the whole normalized longitudinal stack: planner, target arbitration, stop/launch state, and accel-command controller.
- `custom-*` stops at the normalized openpilot control boundary and does not own brand-specific CAN controllers, SCC button emulation, safety-param behavior, or platform message generation.
- Build `custom-*` as a parallel stack, not as conditionals scattered through the current planner/controller.
- `custom-v1` may use a cleaner architecture and is not required to preserve today's deployed custom behavior exactly.
- `custom-v1` keeps the existing MPC as a lower-level tool initially.
- When `custom-*` is selected, run `sunnypilot-current` in shadow as the runtime fallback.
- If `custom-*` throws or produces invalid output, actuate `sunnypilot-current` immediately and latch fallback for the current engagement.
- Reset the fallback latch only after `selfdriveState.enabled == false`, manager restart, or offroad transition.
- Do not reset the fallback latch after gas/brake override or temporary `longActive == false` while selfdrive remains enabled.
- Raise a one-shot orange warning when custom fallback latches.
- Add structured stack telemetry to `longitudinalPlanSP`; driver alerts are not sufficient for debugging.

## Branch Ownership

Create and use retained branch `feat/longitudinal-stack-selector` for selector work. Add it to `MERGE_ORDER` after the existing retained longitudinal branches.

The branch owns:

- `LongitudinalStack` param definition and metadata.
- Stack selector manifest and platform resolution logic.
- UI selector for stack/version choice.
- Longitudinal stack interface types.
- `openpilot-current` and `sunnypilot-current` adapters.
- `custom-*` stack assembly and fallback wrapper.
- `longitudinalPlanSP.stack` telemetry schema and publisher wiring.
- One-shot fallback alert event and tests.
- Selector, fallback, and stack-contract tests.

Do not resolve cross-branch compatibility only on `custom`. If this branch changes retained branch workflow, update `.sync-config`, `AGENTS.md`, and scripts/docs together.

## Stack Options

`LongitudinalStack` should be a single flat persisted value.

Initial values:

- unset or empty: `sunnypilot-current`
- `openpilot-current`: true upstream openpilot longitudinal behavior as carried by sunnypilot upstream
- `sunnypilot-current`: immediate upstream sunnypilot longitudinal behavior
- `custom-recommended`: moving per-platform custom alias
- `custom-1.0`: first custom stack version

Avoid two-param state such as `stack=custom` plus `customVersion=1.0`; that creates invalid combinations and harder telemetry.

## Platform Resolution

Resolution inputs should include at least:

- `CP.brand`
- `CP.carFingerprint`
- `CP.openpilotLongitudinalControl`
- `CP.alphaLongitudinalAvailable`
- `CP.pcmCruise`
- `CP.radarUnavailable`
- relevant `CP_SP` flags

Use a manifest with two concepts:

- `available`: the stack can safely run on this platform/capability class.
- `recommended`: the moving alias target for this platform/capability class.

Recommended manifest behavior:

- Capability rules provide broad defaults.
- Fingerprint overrides allow known-good promotion or explicit blocklist.
- `custom-recommended` falls back to `sunnypilot-current` if unresolved.
- UI displays the resolved recommendation, for example `Recommended: Legacy for this vehicle` or `Recommended: Custom v1.0`.

## Runtime Architecture

Use parallel stacks behind a small normalized interface.

Working interface shape:

```python
@dataclass(frozen=True)
class LongitudinalStackOutput:
  a_target: float
  should_stop: bool
  has_lead: bool
  source: object
  allow_throttle: bool
  allow_brake: bool
  speeds: tuple[float, ...]
  accels: tuple[float, ...]
  jerks: tuple[float, ...]
  fcw: bool
  debug: dict[str, object]
```

Suggested flow:

1. Resolve requested stack from `LongitudinalStack`, platform capabilities, and current availability.
2. Instantiate selected primary stack at startup/offroad cycle.
3. Instantiate `sunnypilot-current` as fallback when selected stack is `custom-*`.
4. Each update, compute fallback output first or in a guarded way so a valid fallback exists.
5. Compute primary output.
6. Validate primary output against finite checks, required fields, trajectory lengths, accel envelope, stop intent consistency, and control-state invariants.
7. If validation fails for `custom-*`, latch fallback for the current engagement and publish the one-shot event.
8. Publish the actuated output through existing `longitudinalPlan` fields.
9. Publish stack metadata through `longitudinalPlanSP.stack`.

Do not make brand-specific car controllers depend on stack internals. They should continue consuming normalized `CarControl` and related messages.

## Baseline Stack Handling

`openpilot-current` and `sunnypilot-current` are tracked baselines, not pinned historical snapshots.

Normal upstream sync may update the baseline behavior. The selector still provides value because it chooses between the current true-upstream baseline, the current sunnypilot-upstream baseline, and this fork's custom stack.

Implementation guidance:

- Prefer adapters around upstream-current code so normal upstream sync updates behavior naturally.
- Keep adapter glue thin and easy to diff.
- If code must be copied to preserve a boundary, include provenance comments with upstream source file and sync assumptions.
- Add tests that prove `sunnypilot-current` bypasses this fork's custom retained longitudinal patches.

## Custom V1 Shape

`custom-v1` should be a cleaner longitudinal architecture while keeping the existing MPC solver initially.

Target internal shape:

```text
normalized inputs
  -> scene/context model
  -> candidate producers
  -> central arbiter
  -> trajectory/output envelope
  -> longitudinal controller
  -> LongitudinalStackOutput
```

Candidate producer examples:

- driver cruise intent
- lead MPC result
- model/e2e stop or slowdown
- speed-limit assist
- SCC vision curve
- SCC map curve
- OSM traffic-control prior
- cruise coast relaxation
- stop/launch state
- learned mass/drag compensation if available and validated

`custom-v1` can migrate existing retained behavior over time, but selector/fallback infrastructure should land first.

## Fallback Rules

Custom fallback is safety and debuggability infrastructure, not a tuning feature.

Fallback should latch when custom output has any of these problems:

- exception during update
- non-finite target, trajectory, or debug-critical value
- missing required output field
- invalid trajectory length
- acceleration outside hard planner/controller envelope
- impossible stop state, such as stop intent with clearly positive launch command outside validated launch state
- stale output or skipped update when a fresh command is required

Fallback behavior:

- Immediately actuate `sunnypilot-current` for that cycle.
- Latch fallback until disengagement/offroad/manager restart.
- Publish fallback telemetry every cycle while latched.
- Publish one-shot warning event only when the latch first trips.
- Do not oscillate between custom and fallback within one engagement.

One-shot warning:

- Text: `Custom Longitudinal Fallback`
- Subtext: `Using sunnypilot longitudinal`
- Event type: `ET.WARNING`
- Status: `AlertStatus.userPrompt`
- Size: `AlertSize.mid`
- Priority: `Priority.MID`
- Audible: none or one low prompt at most

## Cereal Telemetry

Add a nested struct to `custom.capnp` under `LongitudinalPlanSP`, tentatively:

```capnp
stack @9 :Stack;

struct Stack {
  requestedStack @0 :StackId;
  resolvedStack @1 :StackId;
  actuatedStack @2 :StackId;
  shadowStack @3 :StackId;
  customVersion @4 :Text;
  fallbackLatched @5 :Bool;
  fallbackReason @6 :Text;
  actuatedATarget @7 :Float32;
  shadowATarget @8 :Float32;

  enum StackId {
    unknown @0;
    openpilotCurrent @1;
    sunnypilotCurrent @2;
    customRecommended @3;
    customV1 @4;
  }
}
```

Use enums for stack identity so tests, UI, sunnylink, and Drive Lab can parse stable values. Keep `fallbackReason` as text because reasons will change more often.

Do not duplicate full shadow trajectories in `longitudinalPlanSP` initially. Publish only summary fields, especially shadow `aTarget`, until route analysis proves more is needed.

## UI Behavior

Settings should keep the current longitudinal takeover model understandable.

Recommended UI model:

- `AlphaLongitudinalEnabled` remains the alpha-long takeover toggle where applicable.
- A separate longitudinal stack/version selector appears only when longitudinal takeover is active or available.
- The selector is disabled while onroad/engaged and changing it requests an onroad cycle/restart consistent with current longitudinal toggle behavior.
- Show `sunnypilot-current` as the default baseline.
- Show `openpilot-current` only when valid for the platform.
- Show `Recommended` with resolved platform text.
- Show available but not recommended custom versions under an experimental section.

## Implementation Slices

Keep slices small enough for limited agent context windows.

1. Spec and branch workflow
- Add this spec.
- Add `feat/longitudinal-stack-selector` to `.sync-config` and `AGENTS.md` only when implementation branch is ready to enter retained workflow.

2. Param and manifest
- Add `LongitudinalStack` param.
- Add stack manifest and resolver tests.
- Add sunnylink metadata/tests if this setting syncs remotely.

3. Cereal telemetry
- Add `LongitudinalPlanSP.stack` schema.
- Publish default telemetry for current behavior.
- Add schema/publisher tests.

4. UI selector
- Add selector UI using the torque-version selector pattern where practical.
- Gate options by platform and offroad state.
- Add UI/source tests.

5. Stack interface and `sunnypilot-current`
- Introduce `LongitudinalStackOutput` and validation helpers.
- Wrap current upstream sunnypilot behavior behind the interface.
- Preserve existing actuated behavior for default/unset stack.

6. `openpilot-current`
- Add adapter for true upstream openpilot behavior where valid.
- Disable option on unsupported platforms.
- Add tests proving option visibility and selection behavior.

7. Custom wrapper and fallback
- Add `custom-v1` shell using existing MPC as lower-level tool.
- Run `sunnypilot-current` as shadow fallback.
- Implement latch, fallback reason, telemetry, and one-shot event.
- Add invalid-output and exception fallback tests.

8. Custom migration
- Move this fork's retained longitudinal behavior into `custom-v1` behind the interface.
- Ensure `sunnypilot-current` bypasses custom retained behavior.
- Add maneuver and Drive Lab route-comparison coverage before behavior tuning.

## Test Matrix

Minimum required coverage:

- Unset `LongitudinalStack` resolves to `sunnypilot-current`.
- `custom-recommended` resolves per platform.
- Unresolved `custom-recommended` falls back to `sunnypilot-current` and reports that resolution.
- Literal available custom version can be selected when not recommended.
- Unavailable stack/version is hidden or disabled in UI.
- Stack selection is latched and not hot-switched while engaged.
- `AlphaLongitudinalEnabled=false` keeps stock/factory longitudinal behavior where applicable.
- `openpilot-current` is unavailable where upstream openpilot longitudinal is not valid.
- `custom-*` invalid output falls back to `sunnypilot-current` for the same cycle.
- Fallback latch persists across gas/brake override and temporary `longActive == false`.
- Fallback latch resets after full disengagement/offroad/manager restart.
- One-shot fallback alert fires once per latch trip.
- `longitudinalPlanSP.stack` reports requested, resolved, actuated, shadow, fallback, and summary accel fields.
- Default `sunnypilot-current` path does not actuate custom retained behavior after migration.

Affected test areas likely include:

- `common/tests/test_params.py`
- `selfdrive/controls/tests/`
- `selfdrive/ui/tests/`
- `sunnypilot/sunnylink/tests/`
- `sunnypilot/selfdrive/controls/lib/tests/`
- Drive Lab route/timeline tests if stack telemetry is surfaced there

## Agent Context Rules

Future agents should not rely on chat history. Each slice should start from this document and the specific owning files.

For each implementation slice:

- Read this spec first.
- Read only the files directly owned by the slice plus adjacent tests.
- Use grep anchors: `LongitudinalStack`, `LongitudinalStackOutput`, `StackTelemetry`, `stack @9`, `custom-recommended`.
- Keep new interfaces small and named after the architecture so later agents can find them quickly.
- Add tests in the same slice before expanding behavior.
- Do not mix selector infrastructure with custom driving-behavior tuning unless the slice explicitly owns that migration.

## Non-Goals

- Do not replace the MPC solver in `custom-v1`.
- Do not change brand-specific CAN controller behavior as part of stack selection.
- Do not make `custom-recommended` the default initially.
- Do not silently select custom behavior when the param is unset.
- Do not treat warning events as sufficient telemetry.
- Do not resolve retained-branch conflicts on `custom` only.
