"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Key-based override factories for vehicle settings toggles that require
confirmation dialogs and cross-toggle dependencies.
"""
from __future__ import annotations

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.ui.settings_schema.action_helpers import deferred_tr as _t
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog


ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")
SNG_HACK_UNAVAILABLE = tr_noop("sunnypilot Longitudinal Control must be available and enabled for your vehicle to use this feature.")

DESCRIPTIONS = {
  "enforce_stock_longitudinal": tr_noop(
    "sunnypilot will not take over control of gas and brakes. Factory Toyota longitudinal control will be used."
  ),
  "stop_and_go_hack": tr_noop(
    "sunnypilot will allow some Toyota/Lexus cars to auto resume during stop and go traffic. " +
    "This feature is only applicable to certain models that are able to use longitudinal control. This is an alpha feature. Use at your own risk."
  ),
  "tss2_smooth_longitudinal": tr_noop(
    "Use a softer Toyota TSS2 longitudinal tune for smoother traffic following and stops. " +
    "This changes gas and brake tuning only; steering is unchanged."
  ),
}


def _toyota_enforce_stock_factory(item: dict) -> ListItemSP:
  title_text = item.get("title", "")
  desc_text = item.get("description", "")

  control = toggle_item_sp(
    title=_t(title_text),
    description=desc_text,
    param=None,
    initial_state=ui_state.params.get_bool("ToyotaEnforceStockLongitudinal"),
    enabled=lambda: not ui_state.engaged,
  )

  def _on_toggle(state: bool):
    if state:
      def confirm_callback(result: DialogResult):
        if result == DialogResult.CONFIRM:
          ui_state.params.put_bool("ToyotaEnforceStockLongitudinal", True)
          if ui_state.params.get_bool("AlphaLongitudinalEnabled"):
            ui_state.params.put_bool("AlphaLongitudinalEnabled", False)
          ui_state.params.put_bool("ToyotaStopAndGoHack", False)
          ui_state.params.put_bool("OnroadCycleRequested", True)
        else:
          control.action_item.set_state(False)

      content = (f"<h1>{tr(title_text)}</h1><br>" +
                 f"<p>{tr(DESCRIPTIONS['enforce_stock_longitudinal'])}</p>")
      gui_app.push_widget(ConfirmDialog(content, tr("Enable"), rich=True, callback=confirm_callback))
    else:
      ui_state.params.put_bool("ToyotaEnforceStockLongitudinal", False)
      ui_state.params.put_bool("OnroadCycleRequested", True)

  control.callback = _on_toggle

  def _sync(action=control.action_item):
    action.set_state(ui_state.params.get_bool("ToyotaEnforceStockLongitudinal"))
    action.set_enabled(not ui_state.engaged)

    cp = ui_state.CP
    normal_desc = tr(DESCRIPTIONS["enforce_stock_longitudinal"])
    if cp is None:
      control.set_description(f"<b>{tr(ONROAD_ONLY_DESCRIPTION)}</b>\n\n{normal_desc}")
    elif not cp.openpilotLongitudinalControl:
      control.set_description(
        f"<b>{tr('sunnypilot Longitudinal Control is not available for your vehicle.')}</b>\n\n{normal_desc}"
      )
    else:
      control.set_description(normal_desc)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _toyota_stop_and_go_hack_factory(item: dict) -> ListItemSP:
  title_text = item.get("title", "")
  desc_text = item.get("description", "")

  control = toggle_item_sp(
    title=_t(title_text),
    description=desc_text,
    param=None,
    initial_state=ui_state.params.get_bool("ToyotaStopAndGoHack"),
    enabled=lambda: not ui_state.engaged,
  )

  def _on_toggle(state: bool):
    if state:
      def confirm_callback(result: DialogResult):
        if result == DialogResult.CONFIRM:
          ui_state.params.put_bool("ToyotaStopAndGoHack", True)
          ui_state.params.put_bool("OnroadCycleRequested", True)
        else:
          control.action_item.set_state(False)

      content = (f"<h1>{tr(title_text)}</h1><br>" +
                 f"<p>{tr(DESCRIPTIONS['stop_and_go_hack'])}</p>")
      gui_app.push_widget(ConfirmDialog(content, tr("Enable"), rich=True, callback=confirm_callback))
    else:
      ui_state.params.put_bool("ToyotaStopAndGoHack", False)
      ui_state.params.put_bool("OnroadCycleRequested", True)

  control.callback = _on_toggle

  def _sync(action=control.action_item):
    cp = ui_state.CP
    normal_desc = tr(DESCRIPTIONS["stop_and_go_hack"])

    if cp is None:
      action.set_enabled(False)
      control.set_description(f"<b>{tr(ONROAD_ONLY_DESCRIPTION)}</b>\n\n{normal_desc}")
      control.show_description(True)
      return

    longitudinal = cp.openpilotLongitudinalControl
    enforce_stock = ui_state.params.get_bool("ToyotaEnforceStockLongitudinal")

    if longitudinal and not enforce_stock:
      action.set_enabled(not ui_state.engaged)
      action.set_state(ui_state.params.get_bool("ToyotaStopAndGoHack"))
      control.set_description(normal_desc)
      control.show_description(False)
    else:
      action.set_enabled(False)
      action.set_state(False)
      control.set_description(f"<b>{tr(SNG_HACK_UNAVAILABLE)}</b>\n\n{normal_desc}")
      control.show_description(True)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _toyota_tss2_smooth_longitudinal_factory(item: dict) -> ListItemSP:
  title_text = item.get("title", "")
  desc_text = item.get("description", "")

  control = toggle_item_sp(
    title=_t(title_text),
    description=desc_text,
    param=None,
    initial_state=ui_state.params.get_bool("ToyotaTSS2SmoothLongitudinal"),
    enabled=lambda: not ui_state.engaged,
  )

  def _on_toggle(state: bool):
    ui_state.params.put_bool("ToyotaTSS2SmoothLongitudinal", state)
    ui_state.params.put_bool("OnroadCycleRequested", True)

  control.callback = _on_toggle

  def _sync(action=control.action_item):
    cp = ui_state.CP
    normal_desc = tr(DESCRIPTIONS["tss2_smooth_longitudinal"])

    if cp is None:
      action.set_enabled(False)
      control.set_description(f"<b>{tr(ONROAD_ONLY_DESCRIPTION)}</b>\n\n{normal_desc}")
      control.show_description(True)
      return

    longitudinal = cp.openpilotLongitudinalControl
    enforce_stock = ui_state.params.get_bool("ToyotaEnforceStockLongitudinal")

    if longitudinal and not enforce_stock:
      action.set_enabled(not ui_state.engaged)
      action.set_state(ui_state.params.get_bool("ToyotaTSS2SmoothLongitudinal"))
      control.set_description(normal_desc)
      control.show_description(False)
    else:
      action.set_enabled(False)
      action.set_state(False)
      control.set_description(f"<b>{tr(SNG_HACK_UNAVAILABLE)}</b>\n\n{normal_desc}")
      control.show_description(True)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


register_custom_widget("ToyotaEnforceStockLongitudinal", _toyota_enforce_stock_factory)
register_custom_widget("ToyotaTSS2SmoothLongitudinal", _toyota_tss2_smooth_longitudinal_factory)
register_custom_widget("ToyotaStopAndGoHack", _toyota_stop_and_go_hack_factory)
