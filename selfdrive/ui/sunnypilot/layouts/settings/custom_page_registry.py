"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lazy registry for the settings shell's existing imperative widgets.
"""
from __future__ import annotations

from dataclasses import dataclass

from openpilot.selfdrive.ui.layouts.settings.firehose import FirehoseLayout
from openpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise import CruiseLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.developer import DeveloperLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.device import DeviceLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.display import DisplayLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.network import NetworkUISP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm import OSMLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.software import SoftwareLayoutSP
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering import SteeringLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.sunnylink import SunnylinkLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.trips import TripsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle import VehicleLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.visuals import VisualsLayout
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.widgets import Widget


@dataclass
class SettingsPageServices:
  wifi_manager: WifiManager


class SettingsPageRegistry:
  def __init__(self, services: SettingsPageServices):
    self._services = services
    self._pages: dict[str, Widget] = {}
    self._factories = {
      "device": lambda: DeviceLayoutSP(),
      "network": lambda: NetworkUISP(self._services.wifi_manager),
      "software": lambda: SoftwareLayoutSP(),
      "sunnylink": lambda: SunnylinkLayout(),
      "models": lambda: ModelsLayout(),
      "trips": lambda: TripsLayout(),
      "vehicle": lambda: VehicleLayout(),
      "firehose": lambda: FirehoseLayout(),
      "developer": lambda: DeveloperLayoutSP(),
      "toggles": lambda: TogglesLayout(),
      "osm": lambda: OSMLayout(),
      "steering": lambda: SteeringLayout(),
      "cruise": lambda: CruiseLayout(),
      "display": lambda: DisplayLayout(),
      "visuals": lambda: VisualsLayout(),
    }

  def get(self, page_id: str) -> Widget:
    page = self._pages.get(page_id)
    if page is None:
      factory = self._factories.get(page_id)
      if factory is None:
        raise KeyError(page_id)
      page = factory()
      self._pages[page_id] = page
    return page


_REGISTRY: SettingsPageRegistry | None = None


def get_settings_page_registry() -> SettingsPageRegistry:
  global _REGISTRY
  if _REGISTRY is None:
    wifi_manager = WifiManager()
    wifi_manager.set_active(False)
    _REGISTRY = SettingsPageRegistry(SettingsPageServices(wifi_manager=wifi_manager))
  return _REGISTRY
