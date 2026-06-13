#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Headless screenshot harness for schema-driven settings panels.

Renders one settings panel offscreen (Xvfb + software GL) and exports a PNG, so
the schema-driven panels can be eyeballed without a device. Usage:

  xvfb-run -a -s "-screen 0 2200x1400x24" \\
    env LIBGL_ALWAYS_SOFTWARE=1 OFFSCREEN=1 \\
    uv run --extra testing --extra tools python -m \\
    openpilot.sunnypilot.selfdrive.ui.settings_schema.tools.screenshot_panel visuals /tmp/visuals.png
"""
import os
import sys

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("OFFSCREEN", "1")

import pyray as rl

from openpilot.system.ui.lib.application import gui_app


def load_real_fonts():
  """Replace the default fallback fonts with the .ttf/.otf assets.

  The app loads build-generated .fnt bitmap fonts that aren't present in a plain
  checkout; load the source faces directly so screenshots use real type.
  """
  from openpilot.common.basedir import BASEDIR
  from openpilot.system.ui.lib.application import FontWeight
  font_dir = os.path.join(BASEDIR, "selfdrive", "assets", "fonts")
  for fw in FontWeight:
    name = str(fw.value)
    src = name.replace(".fnt", ".otf") if "unifont" in name else name.replace(".fnt", ".ttf")
    path = os.path.join(font_dir, src)
    if os.path.exists(path):
      font = rl.load_font_ex(path, 160, None, 0)
      rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      gui_app._fonts[fw] = font


def build_panel(panel_id: str):
  # Import AFTER init_window so widget construction can load fonts.
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.nav_layout import SchemaNavLayout
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.steering_panel import SchemaSteeringLayout
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import SchemaPanelLayout

  if panel_id == "steering":
    return SchemaSteeringLayout()
  if panel_id == "cruise":
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import (
      SpeedLimitSettingsLayout,
    )
    return SchemaNavLayout("cruise", {"speed_limit_settings": SpeedLimitSettingsLayout})
  if panel_id == "driving":
    from openpilot.sunnypilot.selfdrive.ui.settings_schema.driving_panel import build_driving_layout
    return build_driving_layout()
  if panel_id == "interface":
    from openpilot.sunnypilot.selfdrive.ui.settings_schema.interface_panel import build_interface_layout
    return build_interface_layout()
  if panel_id.startswith("search:"):
    from openpilot.sunnypilot.selfdrive.ui.settings_schema.search_view import SearchLayout
    return SearchLayout(query=panel_id.split(":", 1)[1])
  return SchemaPanelLayout(panel_id)


def main():
  panel_id = sys.argv[1] if len(sys.argv) > 1 else "visuals"
  out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/{panel_id}.png"

  gui_app.init_window(f"settings:{panel_id}")
  load_real_fonts()
  panel = build_panel(panel_id)
  panel.show_event()

  # Render into a content column the width of the real settings content area
  # (full width minus the sidebar), so long titles aren't artificially clipped.
  left = 360
  rect = rl.Rectangle(left, 40, gui_app.width - left - 80, gui_app.height - 80)
  for _ in range(10):
    rl.begin_drawing()
    rl.clear_background(rl.Color(10, 10, 10, 255))
    panel.render(rect)
    rl.end_drawing()

  img = rl.load_image_from_screen()
  ok = rl.export_image(img, out)
  rl.unload_image(img)
  print(f"{'WROTE' if ok else 'FAILED'} {out}  ({int(gui_app.width)}x{int(gui_app.height)})")


if __name__ == "__main__":
  main()
