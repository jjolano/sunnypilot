"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import datetime
import os
from pathlib import Path

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item

from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp
from openpilot.system.ui.sunnypilot.widgets.tailscale_pairing_dialog import TailscalePairingDialog

PREBUILT_PATH = os.path.join(Paths.comma_home(), "prebuilt") if PC else "/data/openpilot/prebuilt"


class DeveloperLayoutSP(DeveloperLayout):
  def __init__(self):
    super().__init__()
    self.error_log_path = os.path.join(Paths.crash_log_root(), "error.log")
    self._is_release_branch: bool = self._is_release or ui_state.params.get_bool("IsReleaseSpBranch")
    self._is_development_branch: bool = ui_state.params.get_bool("IsTestedBranch") or ui_state.params.get_bool("IsDevelopmentBranch")
    self._tailscale_dialog_open: bool = False
    self._initialize_items()

    for item in self.items:
      self._scroller.add_widget(item)

  def _initialize_items(self):
    self.show_advanced_controls = toggle_item_sp(tr("Show Advanced Controls"),
                                                 tr("Toggle visibility of advanced sunnypilot controls.<br>This only changes the visibility of the toggles; " +
                                                    "it does not change the actual enabled/disabled state."), param="ShowAdvancedControls")

    self.enable_github_runner_toggle = toggle_item_sp(tr("GitHub Runner Service"), tr("Enables or disables the GitHub runner service."),
                                                      param="EnableGithubRunner")

    self.enable_copyparty_toggle = toggle_item_sp(tr("copyparty Service"),
                                                   tr("copyparty is a local file server that lets you view and download route and crash logs from a " +
                                                      "browser that can reach the device on port 8080 while offroad. It is unauthenticated and read-only " +
                                                      "while enabled."), param="EnableCopyparty")

    self.prebuilt_toggle = toggle_item_sp(tr("Quickboot Mode"), "", param="QuickBootToggle", callback=self._on_prebuilt_toggled)

    self.error_log_btn = button_item(tr("Error Log"), tr("VIEW"), tr("View the error log for sunnypilot crashes."), callback=self._on_error_log_clicked)

    self._tailscale_install_btn = ListItemSP(
      tr("Tailscale VPN"),
      action_item=NoElideButtonAction(tr("DOWNLOAD"), enabled=True),
      description=tr("Download and manage the Tailscale VPN binary for remote SSH access."),
      callback=self._on_tailscale_install,
    )

    self._tailscale_enable_toggle = toggle_item_sp(
      tr("Enable Tailscale"),
      tr("Start the Tailscale daemon for always-on VPN connectivity."),
      param="EnableTailscale",
    )

    self._tailscale_login_btn = ListItemSP(
      tr("Tailscale Login"),
      action_item=NoElideButtonAction(tr("LOGIN"), enabled=True),
      description=tr("Authentication required. Tap LOGIN to pair this device."),
      callback=self._on_tailscale_login,
    )

    self._tailscale_logout_btn = button_item(
      tr("Tailscale Logout"),
      tr("LOGOUT"),
      tr("Log out of Tailscale. You will need to re-authenticate to reconnect."),
      callback=self._on_tailscale_logout,
    )

    self.items: list = [
      self.show_advanced_controls,
      self.enable_github_runner_toggle,
      self.enable_copyparty_toggle,
      self.prebuilt_toggle,
      self.error_log_btn,
      self._tailscale_install_btn,
      self._tailscale_enable_toggle,
      self._tailscale_login_btn,
      self._tailscale_logout_btn,
    ]

  @staticmethod
  def _on_prebuilt_toggled(state):
    if state:
      Path(PREBUILT_PATH).touch(exist_ok=True)
    else:
      os.remove(PREBUILT_PATH)
    ui_state.params.put_bool("QuickBootToggle", state)

  def _on_delete_confirm(self, result):
    if result == DialogResult.CONFIRM:
      if os.path.exists(self.error_log_path):
        os.remove(self.error_log_path)

  def _on_error_log_closed(self, result, log_exists):
    if result == DialogResult.CONFIRM and log_exists:
      dialog2 = ConfirmDialog(tr("Would you like to delete this log?"), tr("Yes"), tr("No"), rich=False, callback=self._on_delete_confirm)
      gui_app.push_widget(dialog2)

  def _on_error_log_clicked(self):
    text = ""
    if os.path.exists(self.error_log_path):
      text = f"<b>{datetime.datetime.fromtimestamp(os.path.getmtime(self.error_log_path)).strftime('%d-%b-%Y %H:%M:%S').upper()}</b><br><br>"
      try:
        with open(self.error_log_path) as file:
          text += file.read()
      except Exception:
        pass
    dialog = HtmlModalSP(text=text, callback=lambda result: self._on_error_log_closed(result, os.path.exists(self.error_log_path)))
    gui_app.push_widget(dialog)

  def _on_tailscale_install(self):
    self._tailscale_install_btn.action_item.set_enabled(False)
    self._tailscale_install_btn.action_item.set_text(tr("INSTALLING..."))
    ui_state.params.put_bool("TailscaleInstallRequested", True)

  def _on_tailscale_login(self):
    self._tailscale_login_btn.action_item.set_enabled(False)
    self._tailscale_login_btn.action_item.set_text(tr("LOGGING IN..."))
    ui_state.params.put_bool("TailscaleLoginRequested", True)

  def _on_tailscale_logout(self):
    def _confirm_logout():
      ui_state.params.put_bool("TailscaleLogoutRequested", True)

    dialog = ConfirmDialog(
      tr("Are you sure you want to log out of Tailscale?\nYou will need to re-authenticate to reconnect."),
      tr("Logout"),
      callback=lambda res: _confirm_logout() if res == DialogResult.CONFIRM else None,
    )
    gui_app.push_widget(dialog)

  def _open_tailscale_pairing_dialog(self):
    if self._tailscale_dialog_open:
      return
    self._tailscale_dialog_open = True
    dialog = TailscalePairingDialog()
    gui_app.push_widget(dialog)

  def _update_state(self):
    disable_updates = ui_state.params.get_bool("DisableUpdates")
    show_advanced = ui_state.params.get_bool("ShowAdvancedControls")

    if (prebuilt_file := os.path.exists(PREBUILT_PATH)) != ui_state.params.get_bool("QuickBootToggle"):
      ui_state.params.put_bool("QuickBootToggle", prebuilt_file)
      self.prebuilt_toggle.action_item.set_state(prebuilt_file)

    self.prebuilt_toggle.set_visible(show_advanced and not (self._is_release_branch or self._is_development_branch))
    self.prebuilt_toggle.action_item.set_enabled(disable_updates)

    if disable_updates:
      self.prebuilt_toggle.set_description(tr("When toggled on, this creates a prebuilt file to allow accelerated boot times. When toggled off, it " +
                                              "removes the prebuilt file so compilation of locally edited cpp files can be made."))
    else:
      self.prebuilt_toggle.set_description(tr("Quickboot mode requires updates to be disabled.<br>Enable 'Disable Updates' in the Software panel first."))

    self.enable_copyparty_toggle.set_visible(show_advanced)
    self.enable_github_runner_toggle.set_visible(show_advanced and not self._is_release_branch)
    self.error_log_btn.set_visible(not self._is_release_branch)

    ts_installed_version = ui_state.params.get("TailscaleInstalledVersion") or ""
    ts_latest_version = ui_state.params.get("TailscaleLatestVersion") or ""
    ts_install_state = ui_state.params.get("TailscaleInstallState") or ""
    ts_state = ui_state.params.get("TailscaleState") or ""
    ts_is_installed = bool(ts_installed_version)
    ts_is_enabled = ui_state.params.get_bool("EnableTailscale")

    self._tailscale_install_btn.set_visible(show_advanced)
    if ts_install_state.startswith("error:"):
      self._tailscale_install_btn.action_item.set_enabled(True)
      self._tailscale_install_btn.action_item.set_text(tr("RETRY"))
      self._tailscale_install_btn.set_description(tr("Install failed: {}").format(ts_install_state[6:]))
    elif ts_install_state in ("downloading", "verifying", "extracting"):
      self._tailscale_install_btn.action_item.set_enabled(False)
      progress = ui_state.params.get("TailscaleInstallProgress") or ""
      label = ts_install_state.upper()
      if progress:
        label = f"{label} {progress}%"
      self._tailscale_install_btn.action_item.set_text(label)
    elif ts_is_installed:
      has_update = ts_latest_version and ts_latest_version != ts_installed_version
      self._tailscale_install_btn.action_item.set_enabled(has_update)
      self._tailscale_install_btn.action_item.set_text(tr("UPDATE") if has_update else ts_installed_version)
      desc = tr("Tailscale is installed. Version: {}").format(ts_installed_version)
      if has_update:
        desc += tr(" (update available: {})").format(ts_latest_version)
      self._tailscale_install_btn.set_description(desc)
    else:
      self._tailscale_install_btn.action_item.set_enabled(True)
      self._tailscale_install_btn.action_item.set_text(tr("DOWNLOAD"))
      self._tailscale_install_btn.set_description(tr("Download and manage the Tailscale VPN binary for remote SSH access."))

    self._tailscale_enable_toggle.set_visible(show_advanced and ts_is_installed)
    self._tailscale_enable_toggle.action_item.set_enabled(ts_is_installed)
    if ts_is_enabled:
      if ts_state.startswith("Running:"):
        ip_str = ts_state[8:]
        self._tailscale_enable_toggle.set_right_value(ip_str)
        self._tailscale_enable_toggle.set_description(tr("Connected to Tailscale network."))
      elif ts_state == "Running":
        self._tailscale_enable_toggle.set_right_value(tr("Connected"))
        self._tailscale_enable_toggle.set_description(tr("Connected to Tailscale network."))
      elif ts_state == "NeedsLogin":
        self._tailscale_enable_toggle.set_right_value(tr("Needs Login"))
        self._tailscale_enable_toggle.set_description(tr("Authentication required. Tap LOGIN below to pair."))
      else:
        self._tailscale_enable_toggle.set_right_value(ts_state or tr("Starting..."))
        self._tailscale_enable_toggle.set_description(tr("Tailscale is starting up..."))
    else:
      self._tailscale_enable_toggle.set_right_value("")
      self._tailscale_enable_toggle.set_description(tr("Start the Tailscale daemon for always-on VPN connectivity."))

    self._tailscale_login_btn.set_visible(show_advanced and ts_is_installed and ts_is_enabled and ts_state == "NeedsLogin")
    if ts_state == "NeedsLogin":
      self._tailscale_login_btn.action_item.set_text(tr("LOGIN"))
      self._tailscale_login_btn.action_item.set_enabled(True)

    auth_url = ui_state.params.get("TailscaleAuthURL") or ""
    if auth_url and ts_state == "NeedsLogin":
      self._open_tailscale_pairing_dialog()
    elif not auth_url:
      self._tailscale_dialog_open = False

    self._tailscale_logout_btn.set_visible(show_advanced and ts_state.startswith("Running"))
