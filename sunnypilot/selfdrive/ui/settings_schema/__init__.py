"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schema-driven settings UI (prototype).

Today every setting is declared twice: imperatively in the device UI
(selfdrive/ui/sunnypilot/layouts/settings/*.py) and declaratively in the
sunnylink schema (sunnypilot/sunnylink/settings_ui_src/*.yaml -> settings_ui.json).
The two have already drifted (e.g. torque slider min/max/step).

This package renders the *device* UI from the same compiled schema the
cloud/mobile frontend consumes, so a control and its enable/visible rules are
declared once. It has three pieces:

  - rules.py         pure rule evaluator (no rendering deps; the parity proof)
  - schema_loader.py load + walk the compiled settings_ui.json (no rendering deps)
  - widgets.py       build pyray ListItemSP widgets from a schema panel

See tests/test_steering_parity.py for the proof that the rule evaluator
reproduces the hand-coded _update_state() logic of the steering panel.
"""
