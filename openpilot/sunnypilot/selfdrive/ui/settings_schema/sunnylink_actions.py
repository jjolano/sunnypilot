"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Button, toggle, and dual-button action factories for the sunnylink settings panel.

Each factory receives the schema item dict and returns a ListItemSP with the
appropriate callback and sync_hook, replicating the hand-coded SunnylinkLayout
behavior within the schema-driven renderer.
"""
from __future__ import annotations

from openpilot.cereal import custom
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.sunnylink.api import UNREGISTERED_SUNNYLINK_DONGLE_ID
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp, toggle_item_sp, dual_button_item_sp
from openpilot.system.ui.sunnypilot.widgets.sunnylink_pairing_dialog import SunnylinkPairingDialog
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import ButtonStyle
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog, alert_dialog
from openpilot.system.version import sunnylink_consent_version

from openpilot.sunnypilot.selfdrive.ui.settings_schema.action_helpers import deferred_tr as _t
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget

_state = {}


def _get_sunnylink_dongle_id() -> str:
  return ui_state.params.get("SunnylinkDongleId") or tr("N/A")


def _handle_pair_btn(sponsor_pairing: bool = False):
  sunnylink_dongle_id = _get_sunnylink_dongle_id()
  if sunnylink_dongle_id == UNREGISTERED_SUNNYLINK_DONGLE_ID:
    gui_app.push_widget(alert_dialog(message=tr("sunnylink Dongle ID not found. ") +
                                                tr("This may be due to weak internet connection or sunnylink registration issue. ") +
                                                tr("Please reboot and try again.")))
  else:
    gui_app.push_widget(SunnylinkPairingDialog(sponsor_pairing))


def _sunnylink_enabled_factory(item: dict) -> ListItemSP:
  control = toggle_item_sp(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    param=None,
    callback=lambda state: None,
  )

  def _update_description(state: bool):
    if state:
      description = tr("Welcome back!! We're excited to see you've enabled sunnylink again!")
    else:
      description = "😢 " + tr("Not going to lie, it's sad to see you disabled sunnylink") + \
                    tr(", but we'll be here when you're ready to come back.")
    control.set_description(description)
    control.show_description(False)

  def _on_toggle(state: bool):
    sl_consent: bool = ui_state.params.get("CompletedSunnylinkConsentVersion") == sunnylink_consent_version
    sl_enabled: bool = ui_state.params.get_bool("SunnylinkEnabled")

    if state and not sl_consent and not sl_enabled:
      def on_consent_done():
        enabled = ui_state.params.get_bool("SunnylinkEnabled")
        _update_description(enabled)
        gui_app.pop_widget()

      from openpilot.selfdrive.ui.sunnypilot.layouts.onboarding import SunnylinkConsentPage
      sl_terms_dlg = SunnylinkConsentPage(done_callback=on_consent_done)
      gui_app.push_widget(sl_terms_dlg)
    else:
      ui_state.params.put_bool("SunnylinkEnabled", state)
      _update_description(state)

  control.callback = _on_toggle

  _state["prev_sl_enabled"] = None

  def _sync(action=control.action_item):
    action.set_state(ui_state.params.get_bool("SunnylinkEnabled"))
    action.set_enabled(not ui_state.is_onroad())
    control.set_right_value(tr("Dongle ID") + ": " + _get_sunnylink_dongle_id())
    # Only update description when enabled state changes, not every frame.
    current_enabled = ui_state.params.get_bool("SunnylinkEnabled")
    if current_enabled != _state.get("prev_sl_enabled"):
      _state["prev_sl_enabled"] = current_enabled
      _update_description(current_enabled)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _sponsor_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("SPONSOR"),
    description=item.get("description", ""),
    callback=lambda: _handle_pair_btn(False),
  )

  def _sync(action=control.action_item):
    action.set_text(tr("THANKS ♥") if ui_state.sunnylink_state.is_sponsor() else tr("SPONSOR"))
    tier_name = ui_state.sunnylink_state.get_sponsor_tier().name.capitalize() or tr("Not Sponsor")
    action.set_value(tier_name, ui_state.sunnylink_state.get_sponsor_tier_color())
    action.set_enabled(ui_state.params.get_bool("SunnylinkEnabled"))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _pair_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("Not Paired"),
    description=item.get("description", ""),
    callback=lambda: _handle_pair_btn(True),
  )

  def _sync(action=control.action_item):
    action.set_text(tr("Paired") if ui_state.sunnylink_state.is_paired() else tr("Not Paired"))
    action.set_enabled(ui_state.params.get_bool("SunnylinkEnabled"))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _backup_restore_factory(item: dict) -> ListItemSP:
  def _handle_backup():
    def _callback(result: DialogResult):
      if result == DialogResult.CONFIRM:
        _state["backup_in_progress"] = True
        ui_state.params.put_bool("BackupManager_CreateBackup", True)

    gui_app.push_widget(ConfirmDialog(
      tr("Are you sure you want to backup your current sunnypilot settings?"),
      tr("Backup"),
      callback=_callback,
    ))

  def _handle_restore():
    def _callback(result: DialogResult):
      if result == DialogResult.CONFIRM:
        _state["restore_in_progress"] = True
        ui_state.params.put("BackupManager_RestoreVersion", "latest")

    control.action_item.right_button.set_enabled(False)
    gui_app.push_widget(ConfirmDialog(
      tr("Are you sure you want to restore the last backed up sunnypilot settings?"),
      tr("Restore"),
      callback=_callback,
    ))

  control = dual_button_item_sp(
    left_text=tr("Backup Settings"),
    right_text=tr("Restore Settings"),
    left_callback=_handle_backup,
    right_callback=_handle_restore,
    description=item.get("description", ""),
  )

  control.action_item.left_button.set_button_style(ButtonStyle.NORMAL)
  control.action_item.right_button.set_button_style(ButtonStyle.PRIMARY)

  def _sync():
    try:
      backup_manager = ui_state.sm["backupManagerSP"]
    except Exception:
      backup_manager = None

    backup_btn = control.action_item.left_button
    restore_btn = control.action_item.right_button

    if backup_manager is None:
      can_enable = ui_state.params.get_bool("SunnylinkEnabled") and not ui_state.is_onroad()
      backup_btn.set_enabled(can_enable)
      backup_btn.set_text(tr("Backup Settings"))
      restore_btn.set_enabled(can_enable)
      restore_btn.set_text(tr("Restore Settings"))
      return

    backup_status = backup_manager.backupStatus
    restore_status = backup_manager.restoreStatus
    backup_progress = backup_manager.backupProgress
    restore_progress = backup_manager.restoreProgress

    if _state.get("backup_in_progress", False):
      backup_btn.set_enabled(False)
      restore_btn.set_enabled(False)

      if backup_status == custom.BackupManagerSP.Status.inProgress:
        _state["backup_in_progress"] = True
        backup_btn.set_text(tr(f"Backing up {backup_progress}%"))

      elif backup_status == custom.BackupManagerSP.Status.failed:
        _state["backup_in_progress"] = False
        backup_btn.set_enabled(not ui_state.is_onroad())
        backup_btn.set_text(tr("Backup Failed"))

      elif (backup_status == custom.BackupManagerSP.Status.completed or
            (backup_status == custom.BackupManagerSP.Status.idle and backup_progress == 100.0)):
        _state["backup_in_progress"] = False
        gui_app.push_widget(alert_dialog(tr("Settings backup completed.")))
        backup_btn.set_enabled(not ui_state.is_onroad())

    elif _state.get("restore_in_progress", False):
      backup_btn.set_enabled(False)
      restore_btn.set_enabled(False)

      if restore_status == custom.BackupManagerSP.Status.inProgress:
        _state["restore_in_progress"] = True
        restore_btn.set_text(tr(f"Restoring {restore_progress}%"))

      elif restore_status == custom.BackupManagerSP.Status.failed:
        _state["restore_in_progress"] = False
        restore_btn.set_enabled(not ui_state.is_onroad())
        restore_btn.set_text(tr("Restore Failed"))
        gui_app.push_widget(alert_dialog(tr("Unable to restore the settings, try again later.")))

      elif (restore_status == custom.BackupManagerSP.Status.completed or
            (restore_status == custom.BackupManagerSP.Status.idle and restore_progress == 100.0)):
        _state["restore_in_progress"] = False
        gui_app.push_widget(ConfirmDialog(
          tr("Settings restored. Confirm to restart the interface."),
          tr("OK"),
          cancel_text="",
          callback=lambda _: gui_app.request_close(),
        ))

    else:
      can_enable = ui_state.params.get_bool("SunnylinkEnabled") and not ui_state.is_onroad()
      backup_btn.set_enabled(can_enable)
      backup_btn.set_text(tr("Backup Settings"))
      restore_btn.set_enabled(can_enable)
      restore_btn.set_text(tr("Restore Settings"))

      # Trigger detection of a backup/restore started by another source.
      if backup_status == custom.BackupManagerSP.Status.inProgress:
        _state["backup_in_progress"] = True
      if restore_status == custom.BackupManagerSP.Status.inProgress:
        _state["restore_in_progress"] = True

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _sunnylink_uploader_factory(item: dict) -> ListItemSP:
  control = toggle_item_sp(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    param="EnableSunnylinkUploader",
  )

  def _sync(action=control.action_item):
    action.set_enabled(ui_state.params.get_bool("SunnylinkEnabled"))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


# Key-based overrides (take precedence over widget type in build_control).
register_custom_widget("SunnylinkEnabled", _sunnylink_enabled_factory)
register_custom_widget("SponsorStatus", _sponsor_factory)
register_custom_widget("PairGitHub", _pair_factory)
register_custom_widget("BackupRestore", _backup_restore_factory)
register_custom_widget("EnableSunnylinkUploader", _sunnylink_uploader_factory)
