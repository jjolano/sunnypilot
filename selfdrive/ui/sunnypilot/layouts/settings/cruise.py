"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_RECOMMENDED,
  load_stack_manifest,
  resolve_longitudinal_stack,
)
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import SpeedLimitSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp, option_item_sp, simple_button_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  CRUISE = 0
  SLA = 1


ICBM_DESC = tr_noop("When enabled, sunnypilot will attempt to manage the built-in cruise control buttons " +
                    "by emulating button presses for limited longitudinal control.")
ICMB_UNAVAILABLE = tr_noop("Intelligent Cruise Button Management is currently unavailable on this platform.")
ICMB_UNAVAILABLE_LONG_AVAILABLE = tr_noop("Disable the sunnypilot Longitudinal Control (alpha) toggle to allow Intelligent Cruise Button Management.")
ICMB_UNAVAILABLE_LONG_UNAVAILABLE = tr_noop("sunnypilot Longitudinal Control is the default longitudinal control for this platform.")

ACC_ENABLED_DESCRIPTION = tr_noop("Enable custom Short & Long press increments for cruise speed increase/decrease.")
ACC_NOLONG_DESCRIPTION = tr_noop("This feature can only be used with sunnypilot longitudinal control enabled.")
ACC_PCMCRUISE_DISABLED_DESCRIPTION = tr_noop("This feature is not supported on this platform due to vehicle limitations.")
LONG_STACK_DESCRIPTION = tr_noop("Select which longitudinal control stack runs after sunnypilot longitudinal control is active. " +
                                 "Changing this requires an onroad cycle.")
LONG_STACK_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control to use the longitudinal stack selector.")
ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")


class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.CRUISE
    self._speed_limit_layout = SpeedLimitSettingsLayout(lambda: self._set_current_panel(PanelType.CRUISE))
    self._longitudinal_stack_dialog: TreeOptionDialog | None = None
    self._longitudinal_stack_manifest = load_stack_manifest()

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):

    self.icbm_toggle = toggle_item_sp(
      title=tr("Intelligent Cruise Button Management (ICBM) (Alpha)"),
      description="",
      param="IntelligentCruiseButtonManagement")

    self.scc_v_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Vision"),
      description=tr("Use vision path predictions to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlVision")

    self.scc_m_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Map"),
      description=tr("Use map data to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlMap")

    self.custom_acc_toggle = toggle_item_sp(
      title=tr("Custom ACC Speed Increments"),
      description="",
      param="CustomAccIncrementsEnabled",
      callback=self._on_custom_acc_toggle)

    self.custom_acc_short_increment = option_item_sp(
      title=tr("Short Press Increment"),
      param="CustomAccShortPressIncrement",
      min_value=1, max_value=10, value_change_step=1,
      inline=True)

    self.custom_acc_long_increment = option_item_sp(
      title=tr("Long Press Increment"),
      param="CustomAccLongPressIncrement",
      value_map={1: 1, 2: 5, 3: 10},
      min_value=1, max_value=3, value_change_step=1,
      inline=True)

    self.sla_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Speed Limit"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SLA)
    )

    self.dec_toggle = toggle_item_sp(
      title=tr("Enable Dynamic Experimental Control"),
      description=tr("Enable toggle to allow the model to determine when to use sunnypilot ACC or sunnypilot End to End Longitudinal."),
      param="DynamicExperimentalControl")

    self.longitudinal_decision_layer_toggle = toggle_item_sp(
      title=tr("Longitudinal Decision Layer (Experimental)"),
      description=tr("Use the new unified longitudinal arbitration layer. This is opt-in and falls back to current planner behavior if disabled or invalid."),
      param="LongitudinalDecisionLayer")

    self.longitudinal_stack_item = ListItemSP(
      title=tr("Longitudinal Stack"),
      description=tr(LONG_STACK_DESCRIPTION),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_longitudinal_stack_dialog,
    )

    items = [
      self.icbm_toggle,
      self.dec_toggle,
      self.longitudinal_decision_layer_toggle,
      self.longitudinal_stack_item,
      self.scc_v_toggle,
      self.scc_m_toggle,
      self.custom_acc_toggle,
      self.custom_acc_short_increment,
      self.custom_acc_long_increment,
      self.sla_settings_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.SLA:
      self._speed_limit_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.CRUISE)
    self._scroller.show_event()
    self.icbm_toggle.show_description(True)
    self.custom_acc_toggle.show_description(True)

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.SLA:
      self._speed_limit_layout.show_event()

  def _update_state(self):
    super()._update_state()

    if ui_state.CP is not None and ui_state.CP_SP is not None:
      has_icbm = ui_state.has_icbm
      has_long = ui_state.has_longitudinal_control

      if ui_state.CP_SP.intelligentCruiseButtonManagementAvailable and not has_long:
        self.icbm_toggle.action_item.set_enabled(ui_state.is_offroad())
        self.icbm_toggle.set_description(tr(ICBM_DESC))
      else:
        ui_state.params.remove("IntelligentCruiseButtonManagement")
        self.icbm_toggle.action_item.set_enabled(False)

        long_desc = ICMB_UNAVAILABLE
        if has_long:
          if ui_state.CP.alphaLongitudinalAvailable:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_AVAILABLE
          else:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_UNAVAILABLE

        new_desc = "<b>" + tr(long_desc) + "</b>\n\n" + tr(ICBM_DESC)
        if self.icbm_toggle.description != new_desc:
          self.icbm_toggle.set_description(new_desc)
          self.icbm_toggle.show_description(True)

      if has_long or has_icbm:
        self.custom_acc_toggle.action_item.set_enabled(((has_long and not ui_state.CP.pcmCruise) or has_icbm) and ui_state.is_offroad())
        self.dec_toggle.action_item.set_enabled(has_long)
        self.longitudinal_decision_layer_toggle.action_item.set_enabled(has_long)
        self.longitudinal_stack_item.action_item.set_enabled(has_long and ui_state.is_offroad())
        self.scc_v_toggle.action_item.set_enabled(True)
        self.scc_m_toggle.action_item.set_enabled(True)
      else:
        ui_state.params.remove("CustomAccIncrementsEnabled")
        ui_state.params.remove("DynamicExperimentalControl")
        ui_state.params.remove("LongitudinalDecisionLayer")
        ui_state.params.remove("SmartCruiseControlVision")
        ui_state.params.remove("SmartCruiseControlMap")
        self.custom_acc_toggle.action_item.set_enabled(False)
        self.dec_toggle.action_item.set_enabled(False)
        self.longitudinal_decision_layer_toggle.action_item.set_enabled(False)
        self.longitudinal_stack_item.action_item.set_enabled(False)
        self.scc_v_toggle.action_item.set_enabled(False)
        self.scc_m_toggle.action_item.set_enabled(False)
      self._update_longitudinal_stack_item(has_long)

    else:
      has_icbm = has_long = False
      self.icbm_toggle.action_item.set_enabled(False)
      self.icbm_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.longitudinal_stack_item.action_item.set_enabled(False)
      self.longitudinal_stack_item.set_description(tr(ONROAD_ONLY_DESCRIPTION))

    show_custom_acc_desc = False

    if ui_state.is_offroad():
      new_custom_acc_desc = tr(ONROAD_ONLY_DESCRIPTION)
      show_custom_acc_desc = True
    else:
      if has_long or has_icbm:
        if has_long and ui_state.CP.pcmCruise:
          new_custom_acc_desc = tr(ACC_PCMCRUISE_DISABLED_DESCRIPTION)
          show_custom_acc_desc = True
        else:
          new_custom_acc_desc = tr(ACC_ENABLED_DESCRIPTION)
      else:
        new_custom_acc_desc = tr(ACC_NOLONG_DESCRIPTION)
        show_custom_acc_desc = True
        self.custom_acc_toggle.action_item.set_state(False)

    if self.custom_acc_toggle.description != new_custom_acc_desc:
      self.custom_acc_toggle.set_description(new_custom_acc_desc)
      if show_custom_acc_desc:
        self.custom_acc_toggle.show_description(True)

    self._on_custom_acc_toggle(self.custom_acc_toggle.action_item.get_state())

  def _update_longitudinal_stack_item(self, has_long: bool):
    resolution = self._get_longitudinal_stack_resolution()
    self.longitudinal_stack_item.action_item.set_value(self._longitudinal_stack_label(resolution.requested_stack, resolution))
    self.longitudinal_stack_item.set_description(tr(LONG_STACK_DESCRIPTION if has_long else LONG_STACK_NOLONG_DESCRIPTION))

  def _get_longitudinal_stack_resolution(self):
    return resolve_longitudinal_stack(
      ui_state.params.get("LongitudinalStack", return_default=True), ui_state.CP, ui_state.CP_SP,
      manifest=self._longitudinal_stack_manifest,
    )

  def _stack_label(self, stack: str) -> str:
    info = self._longitudinal_stack_manifest.get("stacks", {}).get(stack, {})
    return str(info.get("label") or stack)

  def _longitudinal_stack_label(self, stack: str, resolution=None) -> str:
    if stack == CUSTOM_RECOMMENDED:
      resolution = resolution or self._get_longitudinal_stack_resolution()
      return tr("Recommended") + ": " + tr(self._stack_label(resolution.recommended_stack or resolution.resolved_stack))
    return tr(self._stack_label(stack))

  def _longitudinal_stack_nodes(self, resolution) -> list[TreeFolder]:
    available = set(resolution.available_stacks)
    baseline_nodes = []
    custom_nodes = []
    for stack, info in self._longitudinal_stack_manifest.get("stacks", {}).items():
      if stack not in available:
        continue
      node = TreeNode(stack, {"display_name": self._longitudinal_stack_label(stack, resolution), "short_name": stack})
      if str(info.get("family", "")).startswith("custom"):
        custom_nodes.append(node)
      else:
        baseline_nodes.append(node)

    folders = []
    if baseline_nodes:
      folders.append(TreeFolder(tr("Baselines"), baseline_nodes))
    if custom_nodes:
      folders.append(TreeFolder(tr("Custom"), custom_nodes))
    return folders

  def _show_longitudinal_stack_dialog(self):
    if not ui_state.is_offroad() or ui_state.CP is None or ui_state.CP_SP is None:
      return

    resolution = self._get_longitudinal_stack_resolution()
    current_ref = resolution.requested_stack if resolution.requested_stack in resolution.available_stacks else resolution.resolved_stack
    folders = self._longitudinal_stack_nodes(resolution)

    def handle_selection(result: int):
      if result == DialogResult.CONFIRM and self._longitudinal_stack_dialog:
        selected_ref = self._longitudinal_stack_dialog.selection_ref
        if selected_ref:
          ui_state.params.put("LongitudinalStack", selected_ref)
          ui_state.params.put_bool("OnroadCycleRequested", True)
          self._update_longitudinal_stack_item(ui_state.has_longitudinal_control)
      self._longitudinal_stack_dialog = None

    self._longitudinal_stack_dialog = TreeOptionDialog(
      tr("Select Longitudinal Stack"),
      folders,
      current_ref=current_ref,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._longitudinal_stack_dialog)

  def _on_custom_acc_toggle(self, state):
    self.custom_acc_short_increment.set_visible(state)
    self.custom_acc_long_increment.set_visible(state)
    self.custom_acc_short_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
    self.custom_acc_long_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
