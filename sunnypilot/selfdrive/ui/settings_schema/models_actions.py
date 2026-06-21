"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Button action factories for the models settings panel.

Each factory receives the schema item dict and returns a ListItemSP with the
appropriate callback and sync_hook, replicating the hand-coded ModelsLayout
behavior within the schema-driven renderer.
"""
from __future__ import annotations

import os
import re
import time

import pyray as rl
from cereal import custom
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.sunnypilot.models.default_model import DEFAULT_MODEL
from openpilot.sunnypilot.models.runners.constants import CUSTOM_MODEL_PATH
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp, option_item_sp, toggle_item_sp
from openpilot.system.ui.sunnypilot.widgets.progress_bar import ProgressBarAction
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeNode, TreeFolder
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog, alert_dialog
from openpilot.system.ui.widgets.toggle import ON_COLOR

from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget

_state = {}


def _t(text: str):
  return lambda: tr(text)


def _show_reset_params_dialog():
  def _callback(response):
    if response == DialogResult.CONFIRM:
      ui_state.params.remove("CalibrationParams")
      ui_state.params.remove("LiveTorqueParameters")

  msg = tr("Model download has started in the background. We suggest resetting calibration. Would you like to do that now?")
  gui_app.push_widget(ConfirmDialog(msg, tr("Reset Calibration"), callback=_callback))


def _bundle_to_node(bundle):
  return TreeNode(bundle.ref, {'display_name': bundle.displayName, 'short_name': bundle.internalName})


def _get_folders(model_manager, favorites):
  bundles = model_manager.availableBundles
  folders = {}
  for bundle in bundles:
    folders.setdefault(next((ov_ride.value for ov_ride in bundle.overrides if ov_ride.key == "folder"), ""), []).append(bundle)

  folders_list = [TreeFolder("", [TreeNode("Default", {'display_name': f"{DEFAULT_MODEL} (Default)", 'short_name': "Default"})])]
  for folder, folder_bundles in sorted(folders.items(), key=lambda x: max((bundle.index for bundle in x[1]), default=-1), reverse=True):
    folder_bundles.sort(key=lambda bundle: bundle.index, reverse=True)
    name = folder + (f" - (Updated: {m.group(1)})" if folder_bundles and (m := re.search(r'\(([^)]*)\)[^(]*$', folder_bundles[0].displayName)) else "")
    folders_list.append(TreeFolder(name, [_bundle_to_node(bundle) for bundle in folder_bundles]))

  if favorites and (fav_bundles := [bundle for bundle in bundles if bundle.ref in favorites]):
    folders_list.insert(1, TreeFolder("Favorites", [_bundle_to_node(bundle) for bundle in fav_bundles]))
  return folders_list


def _current_model_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: "",
    description="",
    callback=lambda: None,
  )

  def _on_model_selected(result, model_dialog):
    if result != DialogResult.CONFIRM:
      return
    try:
      mm = ui_state.sm["modelManagerSP"]
    except Exception:
      return
    selected_ref = model_dialog.selection_ref
    if selected_ref == "Default":
      ui_state.params.remove("ModelManager_ActiveBundle")
      _show_reset_params_dialog()
    elif selected_bundle := next((bundle for bundle in mm.availableBundles if bundle.ref == selected_ref), None):
      ui_state.params.put("ModelManager_DownloadIndex", selected_bundle.index)
      if mm.activeBundle and selected_bundle.generation != mm.activeBundle.generation:
        _show_reset_params_dialog()

  def _on_click():
    try:
      mm = ui_state.sm["modelManagerSP"]
    except Exception:
      return
    if mm is None:
      return

    favs = ui_state.params.get("ModelManager_Favs")
    favorites = set(favs.split(';')) if favs else set()
    folders_list = _get_folders(mm, favorites)
    active_ref = mm.activeBundle.ref if mm.activeBundle else "Default"

    dialog = TreeOptionDialog(
      tr("Select a Model"),
      folders_list,
      active_ref,
      "ModelManager_Favs",
      get_folders_fn=lambda favs: _get_folders(mm, favs),
      on_exit=lambda result: _on_model_selected(result, dialog),
    )
    gui_app.push_widget(dialog)

  control.callback = _on_click

  def _sync(action=control.action_item):
    try:
      mm = ui_state.sm["modelManagerSP"]
    except Exception:
      mm = None

    active_name = mm.activeBundle.internalName if mm and mm.activeBundle.ref else f"{DEFAULT_MODEL} (Default)"
    action.set_value(active_name)

    action.set_enabled(ui_state.is_offroad())
    if not ui_state.is_offroad():
      control.set_description(tr("Only available when vehicle is off, or always offroad mode is on"))
    else:
      control.set_description("")

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _cancel_download_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("Cancel"),
    description=item.get("description", ""),
    callback=lambda: ui_state.params.remove("ModelManager_DownloadIndex"),
  )

  def _sync(action=control.action_item):
    download_index = ui_state.params.get("ModelManager_DownloadIndex")
    control.set_visible(download_index is not None)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _model_progress_factory(item: dict) -> ListItemSP:
  progress_action = ProgressBarAction()
  control = ListItemSP(title="", action_item=progress_action)

  def _sync():
    try:
      mm = ui_state.sm["modelManagerSP"]
    except Exception:
      control.set_visible(False)
      return

    if mm is None or (not mm.selectedBundle and not mm.activeBundle):
      control.set_visible(False)
      return

    bundle = mm.selectedBundle if (
      mm.selectedBundle and mm.selectedBundle.status == custom.ModelManagerSP.DownloadStatus.downloading
    ) else mm.activeBundle
    if not bundle:
      control.set_visible(False)
      return

    download_status = bundle.status
    status_changed = _state.get("prev_download_status") != download_status
    _state["prev_download_status"] = download_status

    if download_status != custom.ModelManagerSP.DownloadStatus.downloading:
      control.set_visible(False)
      return

    device._reset_interactive_timeout()
    control.set_visible(True)

    # Show the most relevant active download, falling back to the first model.
    selected_model = next(
      (model for model in bundle.models
       if model.artifact.downloadProgress.status == custom.ModelManagerSP.DownloadStatus.downloading),
      bundle.models[0] if bundle.models else None,
    )
    if selected_model is None:
      control.set_visible(False)
      return

    p = selected_model.artifact.downloadProgress
    text, show, color = f"pending - {bundle.displayName}", False, rl.GRAY
    if p.status == custom.ModelManagerSP.DownloadStatus.downloading:
      text, show = f"{int(p.progress)}% - {bundle.displayName}", True
    elif p.status in (custom.ModelManagerSP.DownloadStatus.downloaded, custom.ModelManagerSP.DownloadStatus.cached):
      status_text = tr("from cache" if p.status == custom.ModelManagerSP.DownloadStatus.cached else "downloaded")
      text, color = f"{bundle.displayName} - {status_text if status_changed else tr('ready')}", ON_COLOR
    elif p.status == custom.ModelManagerSP.DownloadStatus.failed:
      text, color = f"download failed - {bundle.displayName}", rl.RED

    progress_action.update(p.progress, text, show, color)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _refresh_factory(item: dict) -> ListItemSP:
  def _on_click():
    ui_state.params.put("ModelManager_LastSyncTime", 0)
    gui_app.push_widget(alert_dialog(tr("Fetching Latest Models")))

  return button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("REFRESH"),
    description=item.get("description", ""),
    callback=_on_click,
  )


def _clear_cache_factory(item: dict) -> ListItemSP:
  def _on_click():
    def _callback(response):
      if response == DialogResult.CONFIRM:
        ui_state.params.put_bool("ModelManager_ClearCache", True)

    gui_app.push_widget(ConfirmDialog(
      tr("This will delete ALL downloaded models from the cache except the currently active model. Are you sure?"),
      tr("Clear Cache"),
      callback=_callback,
    ))

  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: tr("CLEAR"),
    description=item.get("description", ""),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    current_time = time.monotonic()
    last_calc = _state.get("last_cache_calc", 0)
    if current_time - last_calc > 0.5:
      _state["last_cache_calc"] = current_time
      try:
        if os.path.exists(CUSTOM_MODEL_PATH):
          cache_size = sum(os.path.getsize(os.path.join(CUSTOM_MODEL_PATH, f)) for f in os.listdir(CUSTOM_MODEL_PATH)) / (1024 ** 2)
        else:
          cache_size = 0.0
      except Exception:
        cache_size = 0.0
      action.set_value(f"{cache_size:.2f} MB")

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _lane_turn_value_factory(item: dict) -> ListItemSP:
  initial_value = 500
  try:
    initial_value = int(float(ui_state.params.get("LaneTurnValue", return_default=True)) * 100)
  except Exception:
    pass

  def _on_value_changed(v: int):
    ui_state.params.put("LaneTurnValue", str(v))

  control = option_item_sp(
    title=_t(item.get("title", "")),
    param="LaneTurnValue",
    min_value=500,
    max_value=2000,
    description=item.get("description", ""),
    value_change_step=int(round(100 / CV.MPH_TO_KPH)) if ui_state.is_metric else 100,
    on_value_changed=_on_value_changed,
    label_callback=lambda v: f"{int(round(v / 100 * (CV.MPH_TO_KPH if ui_state.is_metric else 1)))} {'km/h' if ui_state.is_metric else 'mph'}",
    write_param=False,
    initial_value=initial_value,
  )

  def _sync(action=control.action_item):
    new_step = int(round(100 / CV.MPH_TO_KPH)) if ui_state.is_metric else 100
    if action.value_change_step != new_step:
      action.value_change_step = new_step

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _lagd_toggle_delay_factory(item: dict) -> ListItemSP:
  initial_value = 20
  try:
    initial_value = int(float(ui_state.params.get("LagdToggleDelay", return_default=True)) * 100)
  except Exception:
    pass

  def _on_value_changed(v: int):
    ui_state.params.put("LagdToggleDelay", str(v / 100))

  return option_item_sp(
    title=_t(item.get("title", "")),
    param="LagdToggleDelay",
    min_value=5,
    max_value=50,
    description=item.get("description", ""),
    value_change_step=1,
    on_value_changed=_on_value_changed,
    label_callback=lambda v: f"{v / 100:.2f}s",
    write_param=False,
    initial_value=initial_value,
  )


# Component-based registrations.
register_custom_widget("models_progress", _model_progress_factory)
register_custom_widget("models_refresh", _refresh_factory)

# Key-based overrides (take precedence over widget type in build_control).
register_custom_widget("CurrentModel", _current_model_factory)
register_custom_widget("CancelDownload", _cancel_download_factory)
register_custom_widget("ClearModelCache", _clear_cache_factory)
register_custom_widget("LaneTurnValue", _lane_turn_value_factory)
register_custom_widget("LagdToggleDelay", _lagd_toggle_delay_factory)
