"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import os
from collections.abc import Callable
import pyray as rl

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.controls.lib.lateral_demand_stacks import (
  CUSTOM_EXPERIMENTAL,
  CUSTOM_RECOMMENDED,
  CUSTOM_V2,
  SUNNYPILOT_CURRENT,
)
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp, option_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

TORQUE_VERSIONS_PATH = os.path.join(BASEDIR, "sunnypilot", "selfdrive", "controls", "lib", "latcontrol_torque_versions.json")
TORQUE_CONTROLLER_LABELS = {
  "2.0": "2.0",
  "2.1": "2.1",
  "3.0": "3.0",
  "4.0": "4.0",
  "4.1": "4.1",
  "5.0": "5.0 Experimental",
}

LATERAL_DEMAND_STACK_LABELS = {
  SUNNYPILOT_CURRENT: "Sunnypilot Current",
  CUSTOM_RECOMMENDED: "Custom Recommended",
  CUSTOM_V2: "Custom 2.0",
  CUSTOM_EXPERIMENTAL: "Custom Experimental",
}

CONTROLS_PROFILE_LABELS = {
  "sunnypilot-current": "Sunnypilot Current",
  "custom-recommended": "Custom Recommended",
  "custom-2.0": "Custom 2.0",
  "custom-experimental": "Custom Experimental",
}

CONTROLS_PROFILE_DESCRIPTIONS = {
  "sunnypilot-current": "Matches current sunnypilot behavior where possible.",
  "custom-recommended": "Uses the recommended custom controls stack for this car with torque 2.1.",
  "custom-2.0": "Stable custom controls. Uses lateral demand custom-2.0 and torque 2.1.",
  "custom-experimental": "Experimental controls. Uses torque 5.0 and experimental lateral demand stack when available.",
}


class TorqueSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._torque_version_dialog: TreeOptionDialog | None = None
    self._lateral_demand_stack_dialog: TreeOptionDialog | None = None
    self._controls_profile_dialog: TreeOptionDialog | None = None
    self.cached_torque_versions = {}
    self._load_versions()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _load_versions(self):
    with open(TORQUE_VERSIONS_PATH) as f:
      self.cached_torque_versions = json.load(f)

  @staticmethod
  def _clear_controls_profile():
    ui_state.params.remove("ControlsProfile")

  def _initialize_items(self):
    self._torque_control_versions = ListItemSP(
      title=tr("Torque Controller"),
      description="Select the torque controller version to use.",
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_torque_version_dialog,
    )
    self._lateral_demand_stack_selector = ListItemSP(
      title=tr("Lateral Demand Stack"),
      description=tr("Selects the lateral demand stack. The SunnypilotCurrent stack preserves existing behavior. "
                     + "CustomExperimental enables v5 profile-aware preview and turn-exit source-of-truth."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_lateral_demand_stack_dialog,
    )
    self._controls_profile_selector = ListItemSP(
      title=tr("Controls Profile"),
      description=tr("User-facing driving profile alias. Auto-couples the lateral demand stack and torque "
                     + "controller. Advanced per-layer selectors can break the coupling afterwards."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_controls_profile_dialog,
    )
    self._control_calculation_hardening_toggle = toggle_item_sp(
      param="ControlCalculationHardening",
      title=lambda: tr("Control Calculation Hardening (Experimental)"),
      description=lambda: tr("Enables stricter validation for experimental control math paths. Keep this off unless " +
                             "you are explicitly testing hardened control behavior."),
    )
    self._self_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsToggle",
      title=lambda: tr("Self-Tune"),
      description=lambda: tr("Enables self-tune for Torque lateral control for platforms that do not use " +
                             "Torque lateral control by default."),
    )
    self._relaxed_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsRelaxedToggle",
      title=lambda: tr("Less Restrict Settings for Self-Tune (Beta)"),
      description=lambda: tr("Less strict settings when using Self-Tune. This allows torqued to be more " +
                             "forgiving when learning values."),
    )
    self._speed_adaptive_toggle = toggle_item_sp(
      param="LiveTorqueSpeedAdaptiveToggle",
      title=lambda: tr("Speed-Adaptive Self-Tune"),
      description=lambda: tr("Learns separate torque parameters for different speed ranges to improve steering accuracy across all speeds."),
    )
    self._speed_adaptive_apply_toggle = toggle_item_sp(
      param="LiveTorqueSpeedAdaptiveApplyToggle",
      title=lambda: tr("Apply Speed-Adaptive Self-Tune"),
      description=lambda: tr("Applies learned speed-adaptive torque parameters to lateral control. Keep disabled for passive learning."),
    )
    self._custom_tune_toggle = toggle_item_sp(
      param="CustomTorqueParams",
      title=lambda: tr("Enable Custom Tuning"),
      description=lambda: tr("Enables custom tuning for Torque lateral control. " +
                             "Modifying Lateral Acceleration Factor and Friction below will override the offline values " +
                             "indicated in the YAML files within \"opendbc/car/torque_data\". " +
                             "The values will also be used live when \"Manual Real-Time Tuning\" toggle is enabled."),
    )
    self._torque_prams_override_toggle = toggle_item_sp(
      param="TorqueParamsOverrideEnabled",
      title=lambda: tr("Manual Real-Time Tuning"),
      description=lambda: tr("Enforces the torque lateral controller to use the fixed values instead of the learned " +
                             "values from Self-Tune. Enabling this toggle overrides Self-Tune values."),
    )
    self._torque_lat_accel_factor = option_item_sp(
      title=lambda: tr("Lateral Acceleration Factor"),
      param="TorqueParamsOverrideLatAccelFactor",
      description="",
      min_value=1,
      max_value=500,
      value_change_step=1,
      label_callback=(lambda x: f"{x/100} m/s^2"),
      use_float_scaling=True
    )

    self._torque_friction = option_item_sp(
      title=lambda: tr("Friction"),
      param="TorqueParamsOverrideFriction",
      description="",
      min_value=1,
      max_value=100,
      value_change_step=1,
      label_callback=(lambda x: f"{x/100}"),
      use_float_scaling=True
    )

    items = [
      self._controls_profile_selector,
      self._torque_control_versions,
      self._lateral_demand_stack_selector,
      self._control_calculation_hardening_toggle,
      self._self_tune_toggle,
      self._relaxed_tune_toggle,
      self._speed_adaptive_toggle,
      self._speed_adaptive_apply_toggle,
      self._custom_tune_toggle,
      self._torque_prams_override_toggle,
      self._torque_lat_accel_factor,
      self._torque_friction,
    ]
    return items

  def _update_state(self):
    super()._update_state()
    if not ui_state.params.get_bool("LiveTorqueParamsToggle"):
      ui_state.params.remove("LiveTorqueParamsRelaxedToggle")
      self._relaxed_tune_toggle.action_item.set_state(False)
    self._self_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._relaxed_tune_toggle.action_item.set_enabled(ui_state.is_offroad() and self._self_tune_toggle.action_item.get_state())
    self._speed_adaptive_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._speed_adaptive_apply_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._control_calculation_hardening_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._custom_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    custom_tune_enabled = self._custom_tune_toggle.action_item.get_state()
    self._torque_prams_override_toggle.set_visible(custom_tune_enabled)
    self._torque_lat_accel_factor.set_visible(custom_tune_enabled)
    self._torque_friction.set_visible(custom_tune_enabled)

    self._torque_prams_override_toggle.action_item.set_enabled(ui_state.is_offroad())
    sliders_enabled = self._torque_prams_override_toggle.action_item.get_state() or ui_state.is_offroad()
    self._torque_lat_accel_factor.action_item.set_enabled(sliders_enabled)
    self._torque_friction.action_item.set_enabled(sliders_enabled)

    title_text = tr("Real-Time & Offline") if ui_state.params.get("TorqueParamsOverrideEnabled") else tr("Offline Only")
    self._torque_lat_accel_factor.set_title(lambda: tr("Lateral Acceleration Factor") + " (" + title_text + ")")
    self._torque_friction.set_title(lambda: tr("Friction") + " (" + title_text + ")")
    self._torque_control_versions.action_item.set_value(self._get_current_torque_version_label())

    show_advanced = ui_state.params.get_bool("ShowAdvancedControls")
    self._torque_control_versions.set_visible(show_advanced)
    self._lateral_demand_stack_selector.set_visible(show_advanced)
    self._controls_profile_selector.set_visible(True)

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    # subtract button
    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40, rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()

  def _get_current_torque_version_label(self):
    current_val_bytes = ui_state.params.get("TorqueControlTune")
    if current_val_bytes is None:
      return tr("Default")

    try:
      current_val = f"{float(current_val_bytes):.1f}"
      for version, label in TORQUE_CONTROLLER_LABELS.items():
        if math.isclose(float(version), float(current_val), rel_tol=1e-5):
          return label
    except ValueError:
      pass

    return tr("Default")

  def _show_torque_version_dialog(self):
    options_map = {}
    available_versions = {
      f"{float(info.get('version')):.1f}"
      for info in self.cached_torque_versions.values()
      if str(info.get("version", ""))
    }
    for version, label in TORQUE_CONTROLLER_LABELS.items():
      if version in available_versions:
        options_map[label] = version

    nodes = [TreeNode(tr("Default"))]
    for label in options_map:
      nodes.append(TreeNode(label))

    folders = [TreeFolder("", nodes)]

    current_label = self._get_current_torque_version_label()

    def handle_selection(result: int):
      if result == DialogResult.CONFIRM and self._torque_version_dialog:
        selected_ref = self._torque_version_dialog.selection_ref
        if selected_ref == tr("Default"):
          self._clear_controls_profile()
          ui_state.params.remove("TorqueControlTune")
        elif selected_ref in options_map:
          self._clear_controls_profile()
          ui_state.params.put("TorqueControlTune", options_map[selected_ref])
      self._torque_version_dialog = None

    self._torque_version_dialog = TreeOptionDialog(
      tr("Select Torque Controller"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._torque_version_dialog)

  def _show_lateral_demand_stack_dialog(self):
    current_bytes = ui_state.params.get("LateralDemandStack")
    current_value = current_bytes.decode("utf-8") if isinstance(current_bytes, bytes) else current_bytes
    current_label = LATERAL_DEMAND_STACK_LABELS.get(current_value or "", tr("Default"))

    def handle_selection(result: int):
      if result == DialogResult.CONFIRM and self._lateral_demand_stack_dialog:
        selected_ref = self._lateral_demand_stack_dialog.selection_ref
        if selected_ref == tr("Default"):
          self._clear_controls_profile()
          ui_state.params.remove("LateralDemandStack")
        else:
          self._clear_controls_profile()
          for value, label in LATERAL_DEMAND_STACK_LABELS.items():
            if label == selected_ref:
              ui_state.params.put("LateralDemandStack", value)
              break
      self._lateral_demand_stack_dialog = None

    nodes = [TreeNode(tr("Default"))]
    for value, label in LATERAL_DEMAND_STACK_LABELS.items():
      nodes.append(TreeNode(label))
    folders = [TreeFolder("", nodes)]

    self._lateral_demand_stack_dialog = TreeOptionDialog(
      tr("Select Lateral Demand Stack"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._lateral_demand_stack_dialog)

  def _show_controls_profile_dialog(self):
    current_bytes = ui_state.params.get("ControlsProfile")
    current_value = current_bytes.decode("utf-8") if isinstance(current_bytes, bytes) else current_bytes
    current_label = CONTROLS_PROFILE_LABELS.get(current_value or "", tr("Manual / No Profile"))

    def handle_selection(result: int):
      if result == DialogResult.CONFIRM and self._controls_profile_dialog:
        selected_ref = self._controls_profile_dialog.selection_ref
        if selected_ref == tr("Manual / No Profile"):
          ui_state.params.remove("ControlsProfile")
        else:
          for value, label in CONTROLS_PROFILE_LABELS.items():
            if label == selected_ref:
              ui_state.params.remove("TorqueControlTune")
              ui_state.params.remove("LateralDemandStack")
              ui_state.params.remove("LongitudinalStack")
              ui_state.params.put("ControlsProfile", value)
              break
      self._controls_profile_dialog = None

    nodes = [TreeNode(tr("Manual / No Profile"))]
    for value, label in CONTROLS_PROFILE_LABELS.items():
      nodes.append(TreeNode(label))
    folders = [TreeFolder("", nodes)]

    self._controls_profile_dialog = TreeOptionDialog(
      tr("Select Controls Profile"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._controls_profile_dialog)
