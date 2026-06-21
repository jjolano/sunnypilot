"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schema-driven vehicle settings panel with the legacy platform selector and
legend widget rendered above brand-specific compiled-schema controls.
"""
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import load_schema
from openpilot.sunnypilot.selfdrive.ui.settings_schema.widgets import SchemaPanel, live_rule_context
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import ButtonAction
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP


class VehicleSchemaPanelLayout(Widget):
  """Vehicle settings panel: platform selector + legend + brand schema panel."""

  def __init__(self):
    super().__init__()
    self._schema = load_schema()
    self._current_brand = None
    self._schema_panel = None

    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.platform_selector import PlatformSelector, LegendWidget

    self._platform_selector = PlatformSelector(self._on_platform_changed)
    self._vehicle_item = ListItemSP(
      title=self._platform_selector.text,
      action_item=ButtonAction(text=tr("SELECT")),
      callback=self._platform_selector._on_clicked,
    )
    self._vehicle_item.title_color = self._platform_selector.color
    self._legend_widget = LegendWidget(self._platform_selector)

    self._items = [self._vehicle_item, self._legend_widget]
    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  @staticmethod
  def _get_brand():
    if bundle := ui_state.params.get("CarPlatformBundle"):
      return bundle.get("brand", "")
    elif ui_state.CP is not None and ui_state.CP.carFingerprint != "MOCK":
      return ui_state.CP.brand
    return ""

  def _on_platform_changed(self):
    # Called by the platform selector when a platform is selected/removed.
    self._current_brand = None  # Force rebuild on next _update_state.

  def _build_schema_panel(self, brand):
    vehicle_settings = self._schema.get("vehicle_settings", {})
    brand_data = vehicle_settings.get(brand)
    if not brand_data or not brand_data.get("items"):
      return None

    panel = {
      "id": "vehicle",
      "sections": [{
        "id": brand,
        "title": brand_data.get("title", ""),
        "items": brand_data.get("items", []),
      }],
    }

    # Import so key-based override factories (e.g. Toyota) are registered.
    import openpilot.sunnypilot.selfdrive.ui.settings_schema.vehicle_actions  # noqa: F401

    return SchemaPanel(panel, live_rule_context, lambda: ui_state.is_metric)

  def _update_state(self):
    super()._update_state()

    # Update platform selector row from the current platform state.
    self._vehicle_item._title = self._platform_selector.text
    self._vehicle_item.title_color = self._platform_selector.color
    vehicle_text = tr("REMOVE") if ui_state.params.get("CarPlatformBundle") else tr("SELECT")
    self._vehicle_item.action_item.set_text(vehicle_text)
    self._platform_selector.refresh()

    # Check for brand change and rebuild the brand schema panel if needed.
    brand = self._get_brand()
    if brand != self._current_brand:
      self._current_brand = brand
      self._schema_panel = self._build_schema_panel(brand)

      items = [self._vehicle_item, self._legend_widget]
      if self._schema_panel is not None:
        items.append(self._schema_panel)
      self._scroller = Scroller(items, line_separator=True, spacing=0)

    # Drive per-frame state for the schema panel.
    if self._schema_panel is not None:
      self._schema_panel._update_state()

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
