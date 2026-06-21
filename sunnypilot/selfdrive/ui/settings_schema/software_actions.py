"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Button action factories for the software settings panel.

Replicates the hand-coded SoftwareLayout/SoftwareLayoutSP behavior within the
schema-driven renderer. Each factory receives the schema item dict and returns
a ListItemSP with the appropriate callback and sync_hook.
"""
from __future__ import annotations

import os
import time

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog

from openpilot.sunnypilot.selfdrive.ui.settings_schema.action_helpers import deferred_tr as _t, time_ago
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget

UPDATED_TIMEOUT = 10

STATE_TO_DISPLAY_TEXT = {
  "checking...": "checking...",
  "downloading...": "downloading...",
  "finalizing update...": "finalizing update...",
}


def _version_info_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: "",
    description="",
    enabled=False,
  )

  def _sync(action=control.action_item):
    current_desc = ui_state.params.get("UpdaterCurrentDescription") or ""
    current_release_notes = (ui_state.params.get("UpdaterCurrentReleaseNotes") or b"").decode("utf-8", "replace")
    action.set_value(current_desc)
    control.set_description(current_release_notes)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _download_factory(item: dict) -> ListItemSP:
  waiting_state = {"waiting": False, "start_ts": 0.0}

  def _on_click():
    action = control.action_item
    action.set_enabled(False)
    if action.text == tr("CHECK"):
      waiting_state["waiting"] = True
      waiting_state["start_ts"] = time.monotonic()
      os.system("pkill -SIGUSR1 -f system.updated.updated")
    else:
      waiting_state["waiting"] = True
      waiting_state["start_ts"] = time.monotonic()
      os.system("pkill -SIGHUP -f system.updated.updated")

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("CHECK"),
    description=item.get("description", ""),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    updater_state = ui_state.params.get("UpdaterState") or "idle"
    failed_count = ui_state.params.get("UpdateFailedCount") or 0
    fetch_available = ui_state.params.get_bool("UpdaterFetchAvailable")

    if updater_state != "idle":
      waiting_state["waiting"] = False
      action.set_enabled(False)
      display_text = STATE_TO_DISPLAY_TEXT.get(updater_state, updater_state)
      action.set_value(tr(display_text) if display_text in STATE_TO_DISPLAY_TEXT else display_text)
    else:
      if failed_count and int(failed_count) > 0:
        action.set_value(tr("failed to check for update"))
        action.set_text(tr("CHECK"))
      elif fetch_available:
        action.set_value(tr("update available"))
        action.set_text(tr("DOWNLOAD"))
      else:
        last_update = ui_state.params.get("LastUpdateTime")
        if last_update:
          formatted = time_ago(last_update if isinstance(last_update, str) else None)
          action.set_value(tr("up to date, last checked {}").format(formatted))
        else:
          action.set_value(tr("up to date, last checked never"))
        action.set_text(tr("CHECK"))

      if waiting_state["waiting"] and (time.monotonic() - waiting_state["start_ts"] > UPDATED_TIMEOUT):
        waiting_state["waiting"] = False

      action.set_enabled(not waiting_state["waiting"])

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _install_factory(item: dict) -> ListItemSP:
  def _on_click():
    control.action_item.set_enabled(False)
    ui_state.params.put_bool("DoReboot", True, block=True)

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("INSTALL"),
    description=item.get("description", ""),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    update_available = ui_state.params.get_bool("UpdateAvailable")
    if update_available:
      new_desc = ui_state.params.get("UpdaterNewDescription") or ""
      new_release_notes = (ui_state.params.get("UpdaterNewReleaseNotes") or b"").decode("utf-8", "replace")
      action.set_text(tr("INSTALL"))
      action.set_value(new_desc)
      control.set_description(new_release_notes)
      action.set_enabled(True)
    else:
      control.set_visible(False)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _branch_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("SELECT"),
    description=item.get("description", ""),
    callback=lambda: _on_branch_clicked(),
  )

  def _on_branch_clicked():
    from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeNode, TreeFolder

    current_git_branch = ui_state.params.get("GitBranch") or ""
    branches_str = ui_state.params.get("UpdaterAvailableBranches") or ""
    branches = [b for b in branches_str.split(",") if b]
    current_target = ui_state.params.get("UpdaterTargetBranch") or ""

    top_level_branches = [current_git_branch, "release-mici", "release-tizi", "staging", "dev", "master"]

    from openpilot.system.hardware import HARDWARE
    if HARDWARE.get_device_type() == "tici":
      top_level_branches = ["release-tici", "staging-tici"]
      branches = [b for b in branches if b.endswith("-tici")]

    top_level_nodes = [TreeNode(b, {'display_name': b}) for b in top_level_branches if b in branches]
    remaining_branches = [b for b in branches if b not in top_level_branches]
    prebuilt_nodes = [TreeNode(b, {'display_name': b}) for b in remaining_branches if b.endswith("-prebuilt")]
    non_prebuilt_nodes = [TreeNode(b, {'display_name': b}) for b in remaining_branches if not b.endswith("-prebuilt")]

    folders = [
      TreeFolder("", top_level_nodes),
      TreeFolder("Prebuilt Branches", prebuilt_nodes),
      TreeFolder("Non-Prebuilt Branches", non_prebuilt_nodes),
    ]

    def _on_branch_selected(result):
      if result == DialogResult.CONFIRM:
        selection = dialog.selection_ref
        if selection:
          ui_state.params.put("UpdaterTargetBranch", selection)
          control.action_item.set_value(selection)
          os.system("pkill -SIGUSR1 -f system.updated.updated")

    dialog = TreeOptionDialog(tr("Select a branch"), folders, current_target, "", on_exit=_on_branch_selected)
    gui_app.push_widget(dialog)

  def _sync(action=control.action_item):
    current_branch = ui_state.params.get("UpdaterTargetBranch") or ""
    action.set_value(current_branch)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _uninstall_factory(item: dict) -> ListItemSP:
  def _on_click():
    def handle_uninstall_confirmation(result: DialogResult):
      if result == DialogResult.CONFIRM:
        ui_state.params.put_bool("DoUninstall", True, block=True)

    gui_app.push_widget(ConfirmDialog(tr("Are you sure you want to uninstall?"), tr("Uninstall"),
                                      callback=handle_uninstall_confirmation))

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("UNINSTALL"),
    description=item.get("description", ""),
    callback=_on_click,
  )
  return control


def _disable_updates_factory(item: dict) -> ListItemSP:
  """DisableUpdates toggle with reboot confirmation dialog."""
  def _on_toggle(state: bool):
    def _handle_reboot(result):
      if result == DialogResult.CONFIRM:
        ui_state.params.put_bool("DisableUpdates", state)
        ui_state.params.put_bool("DoReboot", True)
      else:
        control.action_item.set_state(ui_state.params.get_bool("DisableUpdates"))

    gui_app.push_widget(ConfirmDialog(
      tr("System reboot required for changes to take effect. Reboot now?"),
      tr("Reboot"),
      callback=_handle_reboot,
    ))

  control = toggle_item_sp_with_param(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    param="DisableUpdates",
    callback=_on_toggle,
  )

  def _sync(action=control.action_item):
    show_advanced = ui_state.params.get_bool("ShowAdvancedControls")
    control.set_visible(show_advanced)
    action.set_enabled(ui_state.is_offroad())
    if ui_state.is_offroad():
      control.set_description(tr("When enabled, automatic software updates will be off.<br><b>This requires a reboot to take effect.</b>"))
    else:
      control.set_description(tr("Please enable \"Always Offroad\" mode or turn off the vehicle to adjust these toggles."))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def toggle_item_sp_with_param(title, description, param, callback):
  """Create a toggle that reads from param but uses a custom callback."""
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp
  return toggle_item_sp(title=title, description=description, param=param, callback=callback)


# Register factories for software button actions.
register_custom_widget("software_download", _download_factory)
register_custom_widget("software_install", _install_factory)
register_custom_widget("software_branch", _branch_factory)
register_custom_widget("software_uninstall", _uninstall_factory)

# Key-based overrides.
register_custom_widget("UpdaterCurrentDescription", _version_info_factory)
register_custom_widget("DisableUpdates", _disable_updates_factory)
