# Upstream touch points

Every modified upstream file, with one line on why. Custom behavior belongs in new files;
this list is the budget (target: <10 entries at end state) and the merge-conflict map for
upstream updates.

| File | Why |
|---|---|
| `common/params_keys.h` | Tailscale param keys (12) + custom long/lat opt-in params (4) + speed-aware torque params (2) |
| `selfdrive/controls/controlsd.py` | Opt-in lateral demand pipeline hook (1 fail-closed line on the model-curvature branch). REMAINING (harness-gated): wire lane-change state/direction + lane-line y0 into `demand/wiring.build_pipeline_inputs` (currently inert) |
| `selfdrive/controls/lib/latcontrol_torque.py` | Pass `CS.vEgo` into the torque override extension hook (1 line) |
| `selfdrive/locationd/torqued.py` | Route accepted learning points through the extension hook and persist speed profiles on the 60s cache cadence |
| `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py` | Opt-in custom-2.0 longitudinal hook in `update_targets` (import + adapter + 1 fail-closed call). Model-stop now read from upstream `modelV2.action.shouldStop`/`desiredAcceleration` + coast from `get_coast_accel(pitch)` inside the adapter (no new planner touch) |
| `selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py` | Offroad speed-aware torque mode selector + dialog wiring |
| `selfdrive/ui/sunnypilot/layouts/settings/settings.py` | Mount consolidated schema-driven panels: `Driving` (steering+cruise) + `Interface` (visuals+display) + a global `Search` panel (jump-to-setting via `_navigate_to_setting`); sidebar collapses 4 entries → 2 and gains Search |
| `system/manager/process_config.py` | `manage_tailscaled` daemon process entry (2 lines) |
| `selfdrive/ui/sunnypilot/layouts/settings/developer.py` | Tailscale install/enable/login/logout settings items |
| `sunnypilot/selfdrive/controls/controlsd_ext.py` | Dispatch torque v2.1 when `TorqueControlTune == 2.1` (import + 1 `elif`); hold the opt-in `LateralDemandAdapter` |
| `sunnypilot/selfdrive/controls/lib/latcontrol_torque_versions.json` | Register `v2.1` in the native torque-selector manifest |
