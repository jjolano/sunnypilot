"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Button and progress factories for the OSM settings panel.

Replicates the hand-coded OSMLayout behavior within the schema-driven renderer.
State is collected once per frame by the DatabaseUpdate sync_hook and shared
through the module-level ``_state`` dict; background threads only touch flags
and ``_state``, never UI references.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import threading
from pathlib import Path
from platform import system as platform_system
from time import monotonic

import pyray as rl
import requests
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.common.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp
from openpilot.system.ui.sunnypilot.widgets.progress_bar import ProgressBarAction
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeFolder, TreeNode, TreeOptionDialog
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog

from openpilot.sunnypilot.selfdrive.ui.settings_schema.action_helpers import deferred_tr as _t, time_ago
from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import register_custom_widget

MAP_PATH = Path(Paths.mapd_root()) / "offline"
_mem_params = Params("/dev/shm/params") if platform_system() != "Darwin" else Params()
_state: dict = {}


def _calculate_size():
  total_size = 0
  directories_to_scan = [MAP_PATH] if MAP_PATH.exists() else []
  while directories_to_scan:
    try:
      for entry in os.scandir(directories_to_scan.pop()):
        if entry.is_file():
          total_size += entry.stat().st_size
        elif entry.is_dir():
          directories_to_scan.append(entry.path)
    except OSError:
      pass
  _state["map_size_text"] = (
    f"{total_size / 1024 ** 2:.2f} MB" if total_size < 1024 ** 3 else f"{total_size / 1024 ** 3:.2f} GB"
  )


def _update_map_size():
  threading.Thread(target=_calculate_size, daemon=True).start()


def _update_db():
  def _callback(result: DialogResult):
    if result == DialogResult.CONFIRM:
      ui_state.params.put_bool("OsmDbUpdatesCheck", True)

  gui_app.push_widget(ConfirmDialog(
    tr("This will start the download process and it might take a while to complete."),
    tr("Start Download"),
    callback=_callback,
  ))


def _fetch_country_locations():
  url = "https://raw.githubusercontent.com/pfeiferj/openpilot-mapd/main/nation_bounding_boxes.json"
  try:
    data = requests.get(url, timeout=10).json()
    locations = sorted(
      [TreeNode(ref=k, data={'display_name': v['full_name']}) for k, v in data.items()],
      key=lambda n: n.data['display_name'],
    )
  except Exception:
    locations = []
  _state["country_locations"] = locations
  _state["country_fetching"] = False
  _state["country_locations_shown"] = False


def _fetch_state_locations():
  url = "https://raw.githubusercontent.com/pfeiferj/openpilot-mapd/main/us_states_bounding_boxes.json"
  try:
    data = requests.get(url, timeout=10).json()
    locations = sorted(
      [TreeNode(ref=k, data={'display_name': v['full_name']}) for k, v in data.items()],
      key=lambda n: n.data['display_name'],
    )
  except Exception:
    locations = []
  locations.insert(0, TreeNode(ref="All", data={'display_name': tr("All states (~6.0 GB)")}))
  _state["state_locations"] = locations
  _state["state_fetching"] = False
  _state["state_locations_shown"] = False


def _handle_region_selection(region_type: str, locations: list, result: DialogResult, ref: str):
  if result != DialogResult.CONFIRM or not ref:
    if region_type == "State" and result == DialogResult.CANCEL:
      if ui_state.params.get("OsmLocationName") == "US" and not ui_state.params.get("OsmStateName"):
        for param in ("OsmLocationName", "OsmLocationTitle", "OsmLocal"):
          try:
            ui_state.params.remove(param)
          except Exception:
            pass
    return

  key = "OsmLocation" if region_type == "Country" else "OsmState"

  if region_type == "Country":
    ui_state.params.put_bool("OsmLocal", True)
    for param in ("OsmStateName", "OsmStateTitle"):
      try:
        ui_state.params.remove(param)
      except Exception:
        pass

  ui_state.params.put(f"{key}Name", ref)
  name = next((n.data['display_name'] for n in locations if n.ref == ref), ref)
  ui_state.params.put(f"{key}Title", name)

  if ref == "US" and region_type == "Country":
    _state["state_fetching"] = True
    threading.Thread(target=_fetch_state_locations, daemon=True).start()
  else:
    _update_db()


def _mapd_version_factory(item: dict) -> ListItemSP:
  control = button_item_sp(
    title=_t(item.get("title", "")),
    button_text=lambda: "",
    description=item.get("description", ""),
    enabled=False,
  )

  def _sync(action=control.action_item):
    action.set_value(ui_state.params.get("MapdVersion") or "Loading...")

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _delete_maps_factory(item: dict) -> ListItemSP:
  def _do_delete_maps():
    if MAP_PATH.exists():
      shutil.rmtree(MAP_PATH)

    for param in ("OsmDownloadedDate", "OsmLocal", "OsmLocationName", "OsmLocationTitle",
                  "OsmStateName", "OsmStateTitle"):
      try:
        ui_state.params.remove(param)
      except Exception:
        pass

    _state["deleting"] = False
    _update_map_size()

  def _on_click():
    def _callback(result: DialogResult):
      if result == DialogResult.CONFIRM:
        _state["deleting"] = True
        threading.Thread(target=_do_delete_maps, daemon=True).start()

    gui_app.push_widget(ConfirmDialog(
      tr("This will delete ALL downloaded maps\n\nAre you sure you want to delete all maps?"),
      tr("Yes, delete all maps"),
      callback=_callback,
    ))

  control = ListItemSP(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    action_item=NoElideButtonAction(tr("DELETE"), enabled=True),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    if _state.get("deleting", False):
      action.set_enabled(False)
      action.set_text(tr("DELETING..."))
      action.set_value("")
    else:
      action.set_enabled(_state.get("can_enable", True))
      action.set_text(tr("DELETE"))
      action.set_value(_state.get("map_size_text", ""))

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _osm_progress_factory(item: dict) -> ListItemSP:
  progress_action = ProgressBarAction()
  control = ListItemSP(title="", action_item=progress_action)

  def _sync():
    if _state.get("active", False):
      control.set_visible(True)
      progress_action.update(
        _state.get("progress_percent", 0.0),
        _state.get("progress_text", ""),
        _state.get("show_progress", False),
        rl.GRAY,
      )
    else:
      control.set_visible(False)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _database_update_factory(item: dict) -> ListItemSP:
  def _on_click():
    _update_db()

  control = ListItemSP(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    action_item=NoElideButtonAction(tr("CHECK"), enabled=True),
    callback=_on_click,
  )

  def _sync(action=control.action_item):
    downloading = bool(_mem_params.get("OSMDownloadLocations"))
    pending = ui_state.params.get_bool("OsmDbUpdatesCheck")
    active = downloading or pending

    _state["downloading"] = downloading
    _state["pending"] = pending
    _state["active"] = active
    _state["can_enable"] = not active

    if active:
      if downloading:
        device._reset_interactive_timeout()
        now = monotonic()
        if now - _state.get("last_map_size_update", 0) >= 1.0:
          _state["last_map_size_update"] = now
          _update_map_size()

      progress = ui_state.params.get("OSMDownloadProgress")
      if isinstance(progress, (str, bytes)):
        try:
          progress = json.loads(progress)
        except Exception:
          progress = None

      if progress is None:
        total = 0
        done = 0
      else:
        total = progress.get('total_files', 0)
        done = progress.get('downloaded_files', 0)

      failed = total > 0 and not downloading and done < total

      if failed:
        progress_text = "0% - Downloading Maps"
        update_value = tr("Error: Invalid download. Retry.")
        progress_percent = 0.0
        show_progress = False
      elif total > 0 and downloading:
        progress_percent = max(0.0, min(100.0, (done / total) * 100.0))
        perc_int = int(progress_percent)
        progress_text = f"{perc_int}% - Downloading Maps"
        update_value = f"{done}/{total} ({perc_int}%)"
        show_progress = True
      else:
        progress_percent = 0.0
        progress_text = "0% - Downloading Maps"
        update_value = tr("Downloading Maps...")
        show_progress = False

      _state["progress_text"] = progress_text
      _state["progress_percent"] = progress_percent
      _state["show_progress"] = show_progress
      _state["update_value"] = update_value

      action.set_text(tr("CHECK"))
      action.set_value(update_value)
      action.set_enabled(not downloading)
    else:
      ts = ui_state.params.get("OsmDownloadedDate")
      dt: datetime.datetime | None = None
      if ts:
        try:
          ts_f = float(ts)
          if ts_f > 0:
            dt = datetime.datetime.fromtimestamp(ts_f, tz=datetime.UTC)
        except (ValueError, TypeError):
          dt = None

      formatted = time_ago(dt.isoformat() if dt else None)
      last_checked = tr("Last checked {}").format(formatted)
      _state["last_checked"] = last_checked
      _state["update_value"] = last_checked

      action.set_text(tr("CHECK"))
      action.set_value(last_checked)
      action.set_enabled(True)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _country_select_factory(item: dict) -> ListItemSP:
  control = ListItemSP(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    action_item=NoElideButtonAction(tr("SELECT"), enabled=True),
    callback=lambda: (
      _state.update({"country_fetching": True, "country_locations": None, "country_locations_shown": True}),
      threading.Thread(target=_fetch_country_locations, daemon=True).start(),
    ),
  )

  def _sync(action=control.action_item):
    if _state.get("country_fetching", False):
      action.set_enabled(False)
      action.set_text(tr("FETCHING..."))
      action.set_value("")
    else:
      action.set_enabled(_state.get("can_enable", True))
      action.set_text(tr("SELECT"))
      action.set_value(ui_state.params.get("OsmLocationTitle") or "")

    locations = _state.get("country_locations")
    if locations is not None and not _state.get("country_locations_shown", False):
      _state["country_locations_shown"] = True
      current = ui_state.params.get("OsmLocationName") or ""
      dialog = TreeOptionDialog(
        tr("Select Country"),
        [TreeFolder(folder="", nodes=locations)],
        current_ref=current,
        search_prompt=tr("Perform a search"),
      )
      dialog.on_exit = lambda res: _handle_region_selection("Country", locations, res, dialog.selection_ref)
      gui_app.push_widget(dialog)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


def _state_select_factory(item: dict) -> ListItemSP:
  control = ListItemSP(
    title=_t(item.get("title", "")),
    description=item.get("description", ""),
    action_item=NoElideButtonAction(tr("SELECT"), enabled=True),
    callback=lambda: (
      _state.update({"state_fetching": True, "state_locations": None, "state_locations_shown": True}),
      threading.Thread(target=_fetch_state_locations, daemon=True).start(),
    ),
  )

  def _sync(action=control.action_item):
    if _state.get("state_fetching", False):
      action.set_enabled(False)
      action.set_text(tr("FETCHING..."))
      action.set_value("")
    else:
      action.set_enabled(_state.get("can_enable", True))
      action.set_text(tr("SELECT"))
      action.set_value(ui_state.params.get("OsmStateTitle") or "")

    locations = _state.get("state_locations")
    if locations is not None and not _state.get("state_locations_shown", False):
      _state["state_locations_shown"] = True
      current = ui_state.params.get("OsmStateName") or ""
      dialog = TreeOptionDialog(
        tr("Select State"),
        [TreeFolder(folder="", nodes=locations)],
        current_ref=current,
        search_prompt=tr("Perform a search"),
      )
      dialog.on_exit = lambda res: _handle_region_selection("State", locations, res, dialog.selection_ref)
      gui_app.push_widget(dialog)

  control.sync_hook = _sync  # type: ignore[attr-defined]
  return control


register_custom_widget("osm_progress", _osm_progress_factory)

# Key-based overrides
register_custom_widget("MapdVersion", _mapd_version_factory)
register_custom_widget("DeleteMaps", _delete_maps_factory)
register_custom_widget("DatabaseUpdate", _database_update_factory)
register_custom_widget("CountrySelect", _country_select_factory)
register_custom_widget("StateSelect", _state_select_factory)
