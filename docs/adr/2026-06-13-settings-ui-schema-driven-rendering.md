# Settings UI: schema-driven device rendering

Status: proposed
Date: 2026-06-13
Relates to: [restart plan](../plans/2026-06-12-fork-restart-reimplementation.md),
the sunnylink settings schema (`sunnypilot/sunnylink/settings_ui_src/` →
`sunnypilot/sunnylink/settings_ui.json`), and `docs/touch-points.md`.

## Context

Every setting in this fork is declared **twice**, in two paradigms:

- **Device UI** — `selfdrive/ui/sunnypilot/layouts/settings/*.py` (PyRay). Imperative.
  ~16 panels plus sub-layouts; each control hand-wired, and every dependency between
  settings re-derived by hand, every frame, in `_update_state()`.
- **Cloud/mobile schema** — `settings_ui_src/*.yaml` compiled to `settings_ui.json`,
  consumed by the sunnylink frontend (`sunnylinkd.getParamsMetadata`). Declarative, with a
  real rule engine (`offroad_only` / `capability` / `param` / `param_compare` /
  `not|any|all`), `$ref` macros, JSON-schema validation, and a test that enforces
  byte-for-byte compile output.

The two have **drifted**, bidirectionally:

- `TorqueParamsOverrideLatAccelFactor`: device steps `0.01`, schema declares `0.1`;
  `TorqueParamsOverrideFriction` min `0.01` vs `0.0` (device int×100 hand-encoding).
- `NeuralNetworkLateralControl` renders on-device but is **absent** from the schema —
  which nonetheless references it as a dependency of `EnforceTorqueControl`.
- Blinker sub-options are **hidden** on-device (`set_visible`) but **greyed** in the schema
  (`enablement`).
- `TorqueControlTune` / `LiveTorqueSpeedAdaptiveMode` render as dialog pickers on-device but
  as `multiple_button` in the schema.

sunnylink is a kept feature (owner, 2026-06-13), so the schema is an asset we already
maintain — and currently pay for twice. The schema is the better-designed artifact (rule
engine, macros, validation, byte-match test); the device re-implements it by hand. The
`_update_state()` enablement/visibility logic is the largest cost and where bugs live (e.g.
the NNLC/torque mutual-exclusion reset, offroad gating copy-pasted across panels).

## Decision

### 1. One source of truth, two renderers

The compiled `settings_ui.json` is the single source of truth. The device UI becomes a
**renderer** of it — the second consumer alongside the sunnylink frontend. Capabilities are
computed once by `generate_capabilities()` (already shared device/cloud), so device and
cloud cannot disagree on a capability value.

This is cheap because the pieces already exist on both sides: the device toolkit
(`toggle_item_sp` / `option_item_sp` / `multiple_button_item_sp` / `Scroller`) and a rule
grammar that maps 1:1 onto `ui_state`. The renderer is `build_control(item)` + a
`RuleContext`/`rules_pass` evaluator — and it **deletes** the per-panel `_update_state()`
enablement/visibility logic, replacing it with the schema's `enablement`/`visibility` arrays.

### 2. Hybrid with explicit escape hatches (100% declarative is a non-goal)

The declarative bulk (toggles, options, multi-buttons, contiguous-int enums, and their
rules) renders generically. The residue is handled by named escape hatches:

- **Named option providers** — dynamic option lists (e.g. `TorqueControlTune` from
  `latcontrol_torque_versions.json`). The schema marks the slot; the device registers the
  provider. Already mirrored cloud-side by `generate_settings_schema._inject_dynamic_options`.
- **Custom-widget registry** — dialog pickers and genuinely bespoke panels
  (vehicle/brand, network/wifi, firehose, tailscale). `widget: custom, component: <id>`.
- **`on_change` actions** — param side-effects that are not pure enablement
  (`remove`-on-default, the MADS limited-platform forcing, the NNLC/torque mutual reset).
  Modeled explicitly, or kept as the small imperative minority.

### 3. Param model: the schema is the registry

Param UI metadata (type, range, options, units) lives in the schema. The device option
**encoding** (OptionControlSP's int range / fixed-point scale) is derived from the schema's
`min`/`max`/`step` in one place (`encoding.py`), instead of each panel hand-encoding it.
This is what removes the `0.1`-vs-`0.01` class of drift at the root.

## Rollout

1. **Prototype + parity (done).** `sunnypilot/selfdrive/ui/settings_schema/`: `rules.py`,
   `schema_loader.py`, `encoding.py`, `widgets.py`. Headless tests prove the rule engine
   reproduces the hand-coded steering `_update_state()` across the full state matrix
   (offroad × torque_allowed × NNLC × brand × vehicle-bus), and prove the encoding
   reproduces the schema's numeric intent.
2. **Live A/B (done, pending on-device validation).** `steering_panel.py` renders the
   steering top level from the schema, behind the default-off `SettingsSchemaDrivenSteering`
   param; MADS/Torque sub-panels reuse the hand-coded layouts via schema-gated nav buttons.
3. **This ADR.**
4. **Close escape hatches.** Value-mapped enums (contiguous-int done — fixes
   `AutoLaneChangeTimer`); then string/float-valued selectors, the custom-widget registry,
   and `on_change` side-effects. Then schema-drive the sub-panels, then other declarative
   panels. Genuinely-custom panels (vehicle/network/software/firehose) stay behind the
   custom hatch.
5. **Make drift a CI check.** Every param key referenced by a layout exists in the schema;
   device-rendered bounds equal schema bounds — the same discipline the cloud JSON already
   enforces via its byte-match test.

## Alternatives considered

- **Codegen Python panels from the schema.** Rejected: generated UI code is hard to read and
  debug, still needs the escape hatches, and adds a codegen step to maintain. The runtime
  renderer produces zero generated code and is small given the toolkit + rule grammar
  already exist.
- **Deprecate one side.** Rejected: sunnylink (and thus the schema) is a keeper, and the
  device UI must exist. The only question is whether they share a declaration — they should.
- **Status quo / manual sync.** Rejected: the drift above is already shipping, and grows with
  every new setting.

## Consequences

- **Pro:** one declaration; `_update_state()` enablement boilerplate deleted; device/cloud
  parity guaranteed by construction; global settings search and "why is this disabled?"
  affordances become near-free (the renderer knows which rule failed); a new setting is a
  YAML edit.
- **Con:** a renderer + escape-hatch registry to build and maintain; rule evaluation each
  frame (cheap, but non-zero); custom widgets remain hand-written; migration is multi-step;
  on-device debugging gains one layer of indirection.
- **Risk:** on-device rendering and per-frame cost are unproven until step 2 is validated on
  hardware; dialog-backed and dynamic-option controls remain bespoke until their hatches land.
