"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Button action factories for the developer settings panel.

Each factory receives the schema item dict and returns a ListItemSP with the
appropriate callback and sync_hook, replicating the hand-coded DeveloperLayoutSP
behavior within the schema-driven renderer.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog

from openpilot.sunnypilot.selfdrive.ui.settings_schema.action_helpers import deferred_tr as _t
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget

PREBUILT_PATH = os.path.join(Paths.comma_home(), "prebuilt") if PC else "/data/openpilot/prebuilt"


def _error_log_factory(item: dict) -> ListItemSP:
  error_log_path = os.path.join(Paths.crash_log_root(), "error.log")

  def _on_click():
    text = ""
    if os.path.exists(error_log_path):
      text = f"<b>{datetime.datetime.fromtimestamp(os.path.getmtime(error_log_path)).strftime('%d-%b-%Y %H:%M:%S').upper()}</b><br><br>"
      try:
        with open(error_log_path) as file:
          text += file.read()
      except Exception:
        pass

    from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP

    def _on_closed(result, log_exists=os.path.exists(error_log_path)):
      if result == DialogResult.CONFIRM and log_exists:
        def _on_delete_confirm(del_result):
          if del_result == DialogResult.CONFIRM:
            if os.path.exists(error_log_path):
              os.remove(error_log_path)
        gui_app.push_widget(ConfirmDialog(tr("Would you like to delete this log?"), tr("Yes"), tr("No"),
                                          callback=_on_delete_confirm))

    gui_app.push_widget(HtmlModalSP(text=text, callback=_on_closed))

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("VIEW"),
    description=item.get("description", ""),
    callback=_on_click,
  )
  return control


def _tailscale_install_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("DOWNLOAD"),
    description=item.get("description", ""),
    callback=lambda: ui_state.params.put_bool("TailscaleInstallRequested", True),
  )

  def _sync(action=control.action_item):
    ts_installed_version = ui_state.params.get("TailscaleInstalledVersion") or ""
    ts_latest_version = ui_state.params.get("TailscaleLatestVersion") or ""
    ts_install_state = ui_state.params.get("TailscaleInstallState") or ""
    ts_is_installed = bool(ts_installed_version)

    if ts_install_state.startswith("error:"):
      action.set_enabled(True)
      action.set_text(tr("RETRY"))
      control.set_description(tr("Install failed: {}").format(ts_install_state[6:]))
    elif ts_install_state in ("downloading", "verifying", "extracting"):
      action.set_enabled(False)
      progress = ui_state.params.get("TailscaleInstallProgress") or ""
      label = ts_install_state.upper()
      if progress:
        label = f"{label} {progress}%"
      action.set_text(label)
    elif ts_is_installed:
      has_update = ts_latest_version and ts_latest_version != ts_installed_version
      action.set_enabled(has_update)
      action.set_text(tr("UPDATE") if has_update else ts_installed_version)
      desc = tr("Tailscale is installed. Version: {}").format(ts_installed_version)
      if has_update:
        desc += tr(" (update available: {})").format(ts_latest_version)
      control.set_description(desc)
    else:
      action.set_enabled(True)
      action.set_text(tr("DOWNLOAD"))
      control.set_description(tr("Download and manage the Tailscale VPN binary for remote SSH access."))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _tailscale_enable_factory(item: dict) -> ListItemSP:
  control = toggle_item_sp(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    param="EnableTailscale",
  )

  def _sync(action=control.action_item):
    ts_is_installed = bool(ui_state.params.get("TailscaleInstalledVersion") or "")
    ts_state = ui_state.params.get("TailscaleState") or ""
    ts_is_enabled = ui_state.params.get_bool("EnableTailscale")

    action.set_enabled(ts_is_installed)
    if ts_is_enabled:
      if ts_state.startswith("Running:"):
        ip_str = ts_state[8:]
        control.set_right_value(ip_str)
        control.set_description(tr("Connected to Tailscale network."))
      elif ts_state == "Running":
        control.set_right_value(tr("Connected"))
        control.set_description(tr("Connected to Tailscale network."))
      elif ts_state == "NeedsLogin":
        control.set_right_value(tr("Needs Login"))
        control.set_description(tr("Authentication required. Tap LOGIN below to pair."))
      else:
        control.set_right_value(ts_state or tr("Starting..."))
        control.set_description(tr("Tailscale is starting up..."))
    else:
      control.set_right_value("")
      control.set_description(tr("Start the Tailscale daemon for always-on VPN connectivity."))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _tailscale_login_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("LOGIN"),
    description=item.get("description", ""),
    callback=lambda: ui_state.params.put_bool("TailscaleLoginRequested", True),
  )

  def _sync(action=control.action_item):
    ts_is_installed = bool(ui_state.params.get("TailscaleInstalledVersion") or "")
    ts_is_enabled = ui_state.params.get_bool("EnableTailscale")
    ts_state = ui_state.params.get("TailscaleState") or ""
    visible = ts_is_installed and ts_is_enabled and ts_state == "NeedsLogin"
    control.set_visible(visible)
    if visible:
      action.set_text(tr("LOGIN"))
      action.set_enabled(True)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _tailscale_logout_factory(item: dict) -> ListItemSP:
  def _on_click():
    def _confirm_logout():
      ui_state.params.put_bool("TailscaleLogoutRequested", True)

    gui_app.push_widget(ConfirmDialog(
      tr("Are you sure you want to log out of Tailscale?\nYou will need to re-authenticate to reconnect."),
      tr("Logout"),
      callback=lambda res: _confirm_logout() if res == DialogResult.CONFIRM else None,
    ))

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("LOGOUT"),
    description=item.get("description", ""),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    ts_state = ui_state.params.get("TailscaleState") or ""
    control.set_visible(ts_state.startswith("Running"))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _quickboot_factory(item: dict) -> ListItemSP:
  """QuickBootToggle has a side effect beyond param write: file creation/removal."""
  def _on_toggle(state: bool):
    if state:
      Path(PREBUILT_PATH).touch(exist_ok=True)
    else:
      if os.path.exists(PREBUILT_PATH):
        os.remove(PREBUILT_PATH)
    ui_state.params.put_bool("QuickBootToggle", state)

  control = toggle_item_sp(
    title=_t(item.get("title", "")),
    description="",
    param="QuickBootToggle",
    callback=_on_toggle,
  )

  def _sync(action=control.action_item):
    prebuilt_file = os.path.exists(PREBUILT_PATH)
    if prebuilt_file != ui_state.params.get_bool("QuickBootToggle"):
      ui_state.params.put_bool("QuickBootToggle", prebuilt_file)
      action.set_state(prebuilt_file)

    disable_updates = ui_state.params.get_bool("DisableUpdates")
    if disable_updates:
      control.set_description(tr("When toggled on, this creates a prebuilt file to allow accelerated boot times. When toggled off, it "
                                  "removes the prebuilt file so compilation of locally edited cpp files can be made."))
    else:
      control.set_description(tr("Quickboot mode requires updates to be disabled.<br>Enable 'Disable Updates' in the Software panel first."))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


# Register factories for developer button actions.
register_custom_widget("developer_error_log", _error_log_factory)
register_custom_widget("developer_tailscale_install", _tailscale_install_factory)
register_custom_widget("developer_tailscale_login", _tailscale_login_factory)
register_custom_widget("developer_tailscale_logout", _tailscale_logout_factory)

# QuickBootToggle needs a custom factory for its file side effect.
# Registered as a custom widget keyed by the param name so the schema renderer
# can override the default toggle behavior.
register_custom_widget("QuickBootToggle", _quickboot_factory)
