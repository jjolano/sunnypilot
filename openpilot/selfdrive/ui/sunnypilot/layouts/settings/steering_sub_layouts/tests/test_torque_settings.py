"""
Focused unit tests for TorqueSettingsLayout safety gating.

These tests avoid the full UI stack by mocking the heavy UI dependencies before
importing the layout module. They verify that manual torque writes are rejected
while onroad/started/engaged regardless of UI state or toggle positions.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


class _FakeParams:
  def __init__(self):
    self._store: dict[str, object] = {}

  def put(self, key: str, value: object, block: bool = False) -> None:
    self._store[key] = value

  def get(self, key: str, return_default: bool = False):
    return self._store.get(key)

  def get_bool(self, key: str) -> bool:
    return str(self._store.get(key, b"0")).lower() in ("1", "true")

  def remove(self, key: str) -> None:
    self._store.pop(key, None)


class _FakeUIState:
  def __init__(self):
    self.params = _FakeParams()
    self.started = False
    self.engaged = False

  def is_onroad(self) -> bool:
    return self.started

  def is_offroad(self) -> bool:
    return not self.started


_fake_ui_state = _FakeUIState()


def _make_mock_item(*args, **kwargs):
  item = MagicMock()
  item.action_item = MagicMock()
  item.action_item.set_enabled = MagicMock()
  item.action_item.set_value = MagicMock()
  item.action_item.set_state = MagicMock()
  item.action_item.get_state = MagicMock(return_value=False)
  item.set_visible = MagicMock()
  item.set_title = MagicMock()
  item.set_description = MagicMock()
  return item


# Mock the expensive UI stack before importing the layout module.
sys.modules.setdefault("pyray", MagicMock())

ui_state_mod = types.ModuleType("openpilot.selfdrive.ui.ui_state")
ui_state_mod.ui_state = _fake_ui_state
sys.modules["openpilot.selfdrive.ui.ui_state"] = ui_state_mod

application_mod = types.ModuleType("openpilot.system.ui.lib.application")
application_mod.gui_app = MagicMock()
application_mod.FontWeight = MagicMock()
sys.modules["openpilot.system.ui.lib.application"] = application_mod

multilang_mod = types.ModuleType("openpilot.system.ui.lib.multilang")
multilang_mod.tr = lambda text: text
sys.modules["openpilot.system.ui.lib.multilang"] = multilang_mod

utils_mod = types.ModuleType("openpilot.system.ui.sunnypilot.lib.utils")
utils_mod.NoElideButtonAction = MagicMock()
sys.modules["openpilot.system.ui.sunnypilot.lib.utils"] = utils_mod

list_view_mod = types.ModuleType("openpilot.system.ui.sunnypilot.widgets.list_view")
list_view_mod.ListItemSP = _make_mock_item
list_view_mod.toggle_item_sp = lambda **kwargs: _make_mock_item()
list_view_mod.option_item_sp = lambda **kwargs: _make_mock_item()
list_view_mod.dual_button_item_sp = lambda *args, **kwargs: _make_mock_item()
sys.modules["openpilot.system.ui.sunnypilot.widgets.list_view"] = list_view_mod

tree_mod = types.ModuleType("openpilot.system.ui.sunnypilot.widgets.tree_dialog")
tree_mod.TreeOptionDialog = MagicMock()
tree_mod.TreeFolder = MagicMock()
tree_mod.TreeNode = MagicMock()
tree_mod.__dict__["TreeOptionDialog"] = tree_mod.TreeOptionDialog
sys.modules["openpilot.system.ui.sunnypilot.widgets.tree_dialog"] = tree_mod

widgets_mod = types.ModuleType("openpilot.system.ui.widgets")
widgets_mod.Widget = object
widgets_mod.DialogResult = object
sys.modules["openpilot.system.ui.widgets"] = widgets_mod

class _FakeNavButton:
  def __init__(self, *args, **kwargs):
    pass

  def set_click_callback(self, callback):
    pass

  def set_position(self, *args, **kwargs):
    pass

  def render(self):
    pass

  @property
  def rect(self):
    r = MagicMock()
    r.height = 50
    return r


network_mod = types.ModuleType("openpilot.system.ui.widgets.network")
network_mod.NavButton = _FakeNavButton
sys.modules["openpilot.system.ui.widgets.network"] = network_mod

scroller_mod = types.ModuleType("openpilot.system.ui.widgets.scroller_tici")
scroller_mod.Scroller = MagicMock()
sys.modules["openpilot.system.ui.widgets.scroller_tici"] = scroller_mod

sunnypilot_ui_state_mod = types.ModuleType("openpilot.selfdrive.ui.sunnypilot.ui_state")
sunnypilot_ui_state_mod.UIStateSP = object
sys.modules["openpilot.selfdrive.ui.sunnypilot.ui_state"] = sunnypilot_ui_state_mod

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts import torque_settings


@pytest.fixture(autouse=True)
def _reset_fake_state():
  _fake_ui_state.started = False
  _fake_ui_state.engaged = False
  _fake_ui_state.params._store.clear()


class TestTorqueSettingsLayout:
  def test_apply_manual_torque_pending_noop_onroad(self):
    layout = torque_settings.TorqueSettingsLayout(lambda: None)
    _fake_ui_state.params._store["TorqueParamsOverrideLatAccelFactor"] = 2.0
    _fake_ui_state.params._store["TorqueParamsOverrideFriction"] = 0.2

    _fake_ui_state.started = True
    layout._pending_lat_accel_factor = 150
    layout._pending_friction = 75
    layout._apply_manual_torque_pending()

    assert _fake_ui_state.params.get("TorqueParamsOverrideLatAccelFactor") == 2.0
    assert _fake_ui_state.params.get("TorqueParamsOverrideFriction") == 0.2

  def test_apply_manual_torque_pending_noop_engaged(self):
    layout = torque_settings.TorqueSettingsLayout(lambda: None)
    _fake_ui_state.params._store["TorqueParamsOverrideLatAccelFactor"] = 2.0
    _fake_ui_state.params._store["TorqueParamsOverrideFriction"] = 0.2

    # Engaged implies started, but test the explicit engaged guard as well.
    _fake_ui_state.engaged = True
    layout._pending_lat_accel_factor = 150
    layout._pending_friction = 75
    layout._apply_manual_torque_pending()

    assert _fake_ui_state.params.get("TorqueParamsOverrideLatAccelFactor") == 2.0
    assert _fake_ui_state.params.get("TorqueParamsOverrideFriction") == 0.2

  def test_apply_manual_torque_pending_writes_offroad(self):
    layout = torque_settings.TorqueSettingsLayout(lambda: None)
    _fake_ui_state.params._store["TorqueParamsOverrideLatAccelFactor"] = 2.0
    _fake_ui_state.params._store["TorqueParamsOverrideFriction"] = 0.2

    _fake_ui_state.started = False
    layout._pending_lat_accel_factor = 150
    layout._pending_friction = 75
    layout._apply_manual_torque_pending()

    assert _fake_ui_state.params.get("TorqueParamsOverrideLatAccelFactor") == 1.5
    assert _fake_ui_state.params.get("TorqueParamsOverrideFriction") == 0.75
