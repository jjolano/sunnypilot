"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.common.params import UnknownKeyName
from openpilot.selfdrive.controls.lib.longitudinal_modes import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_V2,
  CUSTOM_RECOMMENDED,
  StackCatalog,
  load_stack_manifest,
  resolve_longitudinal_stack,
)
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import SpeedLimitSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp, option_item_sp, simple_button_item_sp, multiple_button_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
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
LONG_MODE_DESCRIPTION = tr_noop("Select the top-level longitudinal behavior: ACC for deterministic cruise/follow, E2E for model-primary driving, or SCC for smart switching.")
LONG_MODE_EXPERIMENTAL_CONFIRMATION = tr_noop("E2E and SCC use experimental/model-based longitudinal behavior. Enable only if you understand this alpha feature can make unexpected speed or stop decisions.")
LONG_MODE_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control to use longitudinal modes.")
SCC_CURVE_DESCRIPTION = tr_noop("Allow SCC mode to slow for upcoming curves from this source. These controls only apply when Longitudinal Mode is SCC.")
SCC_CURVE_NOSCC_DESCRIPTION = tr_noop("Select SCC in Longitudinal Mode to use SCC curve controls.")
SCC_CURVE_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control to use SCC curve controls.")
LONG_STACK_DESCRIPTION = tr_noop("Select which longitudinal control stack runs after sunnypilot longitudinal control is active. " +
                                 "Changing this requires an onroad cycle.")
LONG_STACK_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control to use the longitudinal stack selector.")
FAST_LEAD_MOTION_DESCRIPTION = tr_noop("Use raw lead opening and lead speed evidence in custom v2.0 stop/go and progress behavior. " +
                                       "Changing this requires an onroad cycle.")
FAST_LEAD_MOTION_CUSTOM_V2_DESCRIPTION = tr_noop("Select custom v2.0 in Longitudinal Stack to use Fast Lead Motion.")
FAST_LEAD_MOTION_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control and custom v2.0 to use Fast Lead Motion.")
ONE_PEDAL_DESCRIPTION = tr_noop("Treat the cruise speed as a ceiling in custom v2.0. Lift-off coasts unless physical lead or stop evidence requires braking. " +
                                "Changing this requires an onroad cycle.")
ONE_PEDAL_CUSTOM_V2_DESCRIPTION = tr_noop("Select custom v2.0 in Longitudinal Stack to use One Pedal Longitudinal.")
ONE_PEDAL_NOLONG_DESCRIPTION = tr_noop("Enable sunnypilot longitudinal control and custom v2.0 to use One Pedal Longitudinal.")
ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")


class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.CRUISE
    self._speed_limit_layout = SpeedLimitSettingsLayout(lambda: self._set_current_panel(PanelType.CRUISE))
    self._longitudinal_stack_dialog: TreeOptionDialog | None = None
    self._longitudinal_stack_manifest = load_stack_manifest()
    self._longitudinal_stack_catalog = StackCatalog(self._longitudinal_stack_manifest)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):

    self.icbm_toggle = toggle_item_sp(
      title=tr("Intelligent Cruise Button Management (ICBM) (Alpha)"),
      description="",
      param="IntelligentCruiseButtonManagement")

    self.longitudinal_mode_item = multiple_button_item_sp(
      title=tr("Longitudinal Mode"),
      description=tr(LONG_MODE_DESCRIPTION),
      buttons=[tr("ACC"), tr("E2E"), tr("SCC")],
      selected_index=int(ui_state.params.get("LongitudinalMode", return_default=True)),
      callback=self._on_longitudinal_mode_changed,
    )

    self.custom_acc_toggle = toggle_item_sp(
      title=tr("Custom ACC Speed Increments"),
      description="",
      param="CustomAccIncrementsEnabled",
      callback=self._on_custom_acc_toggle)

    self.scc_curve_vision_toggle = toggle_item_sp(
      title=tr("SCC Vision Curve Control"),
      description="",
      param="SccCurveVisionEnabled")

    self.scc_curve_map_toggle = toggle_item_sp(
      title=tr("SCC Map Curve Control"),
      description="",
      param="SccCurveMapEnabled")

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

    self.longitudinal_stack_item = ListItemSP(
      title=tr("Longitudinal Stack"),
      description=tr(LONG_STACK_DESCRIPTION),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_longitudinal_stack_dialog,
    )

    self.one_pedal_longitudinal_item = multiple_button_item_sp(
      title=tr("One Pedal Longitudinal"),
      description=tr(ONE_PEDAL_DESCRIPTION),
      buttons=[tr("Off"), tr("Creep"), tr("Full Stop")],
      selected_index=int(ui_state.params.get("OnePedalLongitudinalMode", return_default=True)),
      callback=self._on_one_pedal_mode_changed,
      param="OnePedalLongitudinalMode",
    )

    self.fast_lead_motion_toggle = toggle_item_sp(
      title=tr("Fast Lead Motion"),
      description=tr(FAST_LEAD_MOTION_DESCRIPTION),
      initial_state=self._get_fast_lead_motion_state(),
      callback=self._on_fast_lead_motion_changed)

    items = [
      self.icbm_toggle,
      self.longitudinal_mode_item,
      self.scc_curve_vision_toggle,
      self.scc_curve_map_toggle,
      self.longitudinal_stack_item,
      self.fast_lead_motion_toggle,
      self.one_pedal_longitudinal_item,
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
    self.longitudinal_mode_item.show_description(True)
    self.scc_curve_vision_toggle.show_description(True)
    self.scc_curve_map_toggle.show_description(True)
    self.fast_lead_motion_toggle.show_description(True)
    self.one_pedal_longitudinal_item.show_description(True)

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
        self.longitudinal_mode_item.action_item.set_enabled(has_long)
        self.longitudinal_stack_item.action_item.set_enabled(has_long and ui_state.is_offroad())
        self._update_fast_lead_motion_item(has_long)
        self._update_one_pedal_item(has_long)
      else:
        ui_state.params.remove("CustomAccIncrementsEnabled")
        ui_state.params.remove("LongitudinalMode")
        self.custom_acc_toggle.action_item.set_enabled(False)
        self.longitudinal_mode_item.action_item.set_enabled(False)
        self.scc_curve_vision_toggle.action_item.set_enabled(False)
        self.scc_curve_map_toggle.action_item.set_enabled(False)
        self.longitudinal_stack_item.action_item.set_enabled(False)
        self.fast_lead_motion_toggle.action_item.set_enabled(False)
        self.one_pedal_longitudinal_item.action_item.set_enabled(False)
      self._update_longitudinal_mode_item(has_long)
      self._update_scc_curve_items(has_long)
      self._update_longitudinal_stack_item(has_long)

    else:
      has_icbm = has_long = False
      self.icbm_toggle.action_item.set_enabled(False)
      self.icbm_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.longitudinal_mode_item.action_item.set_enabled(False)
      self.longitudinal_mode_item.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.scc_curve_vision_toggle.action_item.set_enabled(False)
      self.scc_curve_vision_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.scc_curve_map_toggle.action_item.set_enabled(False)
      self.scc_curve_map_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.longitudinal_stack_item.action_item.set_enabled(False)
      self.longitudinal_stack_item.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.fast_lead_motion_toggle.action_item.set_enabled(False)
      self.fast_lead_motion_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.one_pedal_longitudinal_item.action_item.set_enabled(False)
      self.one_pedal_longitudinal_item.set_description(tr(ONROAD_ONLY_DESCRIPTION))

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

  def _update_longitudinal_mode_item(self, has_long: bool):
    self.longitudinal_mode_item.set_description(tr(LONG_MODE_DESCRIPTION if has_long else LONG_MODE_NOLONG_DESCRIPTION))

  def _update_scc_curve_items(self, has_long: bool):
    try:
      current_mode = LongitudinalMode(int(ui_state.params.get("LongitudinalMode", return_default=True)))
    except (TypeError, ValueError):
      current_mode = LongitudinalMode.ACC
    enabled = has_long and current_mode == LongitudinalMode.SCC
    description = SCC_CURVE_DESCRIPTION if enabled else (SCC_CURVE_NOSCC_DESCRIPTION if has_long else SCC_CURVE_NOLONG_DESCRIPTION)
    self.scc_curve_vision_toggle.action_item.set_enabled(enabled)
    self.scc_curve_map_toggle.action_item.set_enabled(enabled)
    self.scc_curve_vision_toggle.set_description(tr(description))
    self.scc_curve_map_toggle.set_description(tr(description))

  def _update_longitudinal_stack_item(self, has_long: bool):
    resolution = self._get_longitudinal_stack_resolution()
    self.longitudinal_stack_item.action_item.set_value(self._longitudinal_stack_label(resolution.requested_stack, resolution))
    self.longitudinal_stack_item.set_description(tr(LONG_STACK_DESCRIPTION if has_long else LONG_STACK_NOLONG_DESCRIPTION))

  def _update_fast_lead_motion_item(self, has_long: bool):
    resolution = self._get_longitudinal_stack_resolution()
    enabled = has_long and ui_state.is_offroad() and resolution.resolved_stack == CUSTOM_V2
    self.fast_lead_motion_toggle.action_item.set_state(self._get_fast_lead_motion_state())
    self.fast_lead_motion_toggle.action_item.set_enabled(enabled)
    if not has_long:
      description = FAST_LEAD_MOTION_NOLONG_DESCRIPTION
    elif resolution.resolved_stack != CUSTOM_V2:
      description = FAST_LEAD_MOTION_CUSTOM_V2_DESCRIPTION
    else:
      description = FAST_LEAD_MOTION_DESCRIPTION
    self.fast_lead_motion_toggle.set_description(tr(description))

  def _update_one_pedal_item(self, has_long: bool):
    resolution = self._get_longitudinal_stack_resolution()
    enabled = has_long and ui_state.is_offroad() and resolution.resolved_stack == CUSTOM_V2
    self.one_pedal_longitudinal_item.action_item.set_enabled(enabled)
    if not has_long:
      description = ONE_PEDAL_NOLONG_DESCRIPTION
    elif resolution.resolved_stack != CUSTOM_V2:
      description = ONE_PEDAL_CUSTOM_V2_DESCRIPTION
    else:
      description = ONE_PEDAL_DESCRIPTION
    self.one_pedal_longitudinal_item.set_description(tr(description))

  def _get_longitudinal_stack_resolution(self):
    return resolve_longitudinal_stack(
      ui_state.params.get("LongitudinalStack", return_default=True), ui_state.CP, ui_state.CP_SP,
      manifest=self._longitudinal_stack_manifest,
    )

  def _stack_label(self, stack: str) -> str:
    return self._longitudinal_stack_catalog.stack_definition(stack).label

  def _longitudinal_stack_label(self, stack: str, resolution=None) -> str:
    if stack == CUSTOM_RECOMMENDED:
      resolution = resolution or self._get_longitudinal_stack_resolution()
      return tr("Recommended") + ": " + tr(self._stack_label(resolution.recommended_stack or resolution.resolved_stack))
    return tr(self._stack_label(stack))

  def _longitudinal_stack_nodes(self, resolution) -> list[TreeFolder]:
    available = set(resolution.available_stacks)
    baseline_nodes = []
    custom_nodes = []
    for stack in self._longitudinal_stack_catalog.stack_names:
      if stack not in available:
        continue
      definition = self._longitudinal_stack_catalog.stack_definition(stack)
      node = TreeNode(stack, {"display_name": self._longitudinal_stack_label(stack, resolution), "short_name": stack})
      if definition.family.startswith("custom"):
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
          self._update_fast_lead_motion_item(ui_state.has_longitudinal_control)
          self._update_one_pedal_item(ui_state.has_longitudinal_control)
      self._longitudinal_stack_dialog = None

    self._longitudinal_stack_dialog = TreeOptionDialog(
      tr("Select Longitudinal Stack"),
      folders,
      current_ref=current_ref,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._longitudinal_stack_dialog)

  def _on_one_pedal_mode_changed(self, _mode: int):
    ui_state.params.put_bool("OnroadCycleRequested", True)

  @staticmethod
  def _get_fast_lead_motion_state() -> bool:
    try:
      return ui_state.params.get_bool("FastLeadMotionEvidenceEnabled")
    except UnknownKeyName:
      return False

  def _on_fast_lead_motion_changed(self, state: bool):
    try:
      ui_state.params.put_bool("FastLeadMotionEvidenceEnabled", state)
    except UnknownKeyName:
      return
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def _on_longitudinal_mode_changed(self, mode: int):
    try:
      selected_mode = LongitudinalMode(mode)
    except ValueError:
      selected_mode = LongitudinalMode.ACC
    previous_mode = self._current_longitudinal_mode()
    action = self.longitudinal_mode_item.action_item

    if selected_mode != LongitudinalMode.ACC and not ui_state.params.get_bool("ExperimentalModeConfirmed"):
      action.selected_button = int(previous_mode)

      def confirm_callback(result: DialogResult):
        if result == DialogResult.CONFIRM:
          ui_state.params.put_bool("ExperimentalModeConfirmed", True)
          self._set_longitudinal_mode(selected_mode)
        else:
          action.selected_button = int(self._current_longitudinal_mode())

      gui_app.push_widget(ConfirmDialog(
        tr(LONG_MODE_EXPERIMENTAL_CONFIRMATION),
        tr("Enable"),
        rich=True,
        callback=confirm_callback,
      ))
      self._update_scc_curve_items(ui_state.has_longitudinal_control)
      return

    self._set_longitudinal_mode(selected_mode)

  @staticmethod
  def _current_longitudinal_mode() -> LongitudinalMode:
    try:
      return LongitudinalMode(int(ui_state.params.get("LongitudinalMode", return_default=True)))
    except (TypeError, ValueError):
      return LongitudinalMode.ACC

  def _set_longitudinal_mode(self, mode: LongitudinalMode):
    ui_state.params.put("LongitudinalMode", int(mode))
    self.longitudinal_mode_item.action_item.selected_button = int(mode)
    self._update_scc_curve_items(ui_state.has_longitudinal_control)

  def _on_custom_acc_toggle(self, state):
    self.custom_acc_short_increment.set_visible(state)
    self.custom_acc_long_increment.set_visible(state)
    self.custom_acc_short_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
    self.custom_acc_long_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
