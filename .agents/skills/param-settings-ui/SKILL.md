---
name: param-settings-ui
description: Ensures new repo params are added with matching Settings UI controls, encoding/schema tests, generated settings_ui.json, and runtime fallbacks. Use when adding, renaming, or changing Params keys, user-toggled modes, feature flags, shadow/apply gates, or any setting the owner may flip on-device.
---

# Param + Settings UI

## Rule

Every new user/tester-facing param must have a Settings UI path. The owner flips params
from Settings UI, not shell/Params tooling.

## Checklist

1. Add the runtime param key in `common/params_keys.h` with a safe default.
   - Shadow/debug modes default `"off"` unless intentionally collecting by default.
   - Runtime must sanitize unknown values and fail closed.
2. Add or update the Settings UI source YAML, usually:
   - `sunnypilot/sunnylink/settings_ui_src/pages/cruise.yaml`
   - or the relevant page under `sunnypilot/sunnylink/settings_ui_src/pages/`.
3. Keep Settings copy explicit about behavior.
   - If shadow-only: say `No driving changes` / `Monitor only` / `telemetry-only`.
   - If an apply option is staged but not actuating yet, state that clearly.
4. Regenerate checked-in Settings UI JSON:

   ```bash
   uv run --extra testing --extra tools python -m openpilot.sunnypilot.sunnylink.tools.compile_settings_ui
   ```

5. Update tests for the new setting:
   - `sunnypilot/sunnylink/tests/test_compile_settings_ui.py`
   - `sunnypilot/selfdrive/ui/settings_schema/tests/test_cruise_panel.py` or relevant panel test
   - `sunnypilot/selfdrive/ui/settings_schema/tests/test_driving_panel.py` if shown on Driving
   - `sunnypilot/selfdrive/ui/settings_schema/tests/test_encoding.py` for string option/index behavior
6. Add runtime/wiring tests when the setting affects planner behavior or shadow telemetry.
   - Missing/unregistered param reads must not block unrelated source-toggle refresh.
   - Shadow modes must be exactly non-actuating.

## Verification

Run affected Settings UI and runtime tests, for example:

```bash
uv run ruff check <modified python files>
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/sunnylink/tests/test_compile_settings_ui.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_cruise_panel.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_driving_panel.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_encoding.py \
  <affected runtime tests>
git diff --check
```

## Common patterns

- `off | shadow`: debug-only multiple button, labels `Off`, `Monitor only`.
- `off | shadow | apply_conservative`: allowed for staged testing, but runtime must make
  non-actuating status obvious until apply behavior exists.
- Stored legacy/future apply values should map safely in
  `sunnypilot/selfdrive/ui/settings_schema/encoding.py` when the UI does not expose them.
