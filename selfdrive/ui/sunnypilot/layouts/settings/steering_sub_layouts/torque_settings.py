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
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp, option_item_sp, dual_button_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.sunnypilot.custom.lateral.torque_safety import (
  TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MIN,
  TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MAX,
  TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_DEFAULT,
  TORQUE_OVERRIDE_FRICTION_MIN,
  TORQUE_OVERRIDE_FRICTION_MAX,
  TORQUE_OVERRIDE_FRICTION_DEFAULT,
  validate_torque_override_friction,
  validate_torque_override_lat_accel_factor,
  validate_roll_comp_gain_mode,
  validate_friction_breakaway_mode,
  validate_direction_gain_mode,
)

TORQUE_VERSIONS_PATH = os.path.join(BASEDIR, "sunnypilot", "selfdrive", "controls", "lib", "latcontrol_torque_versions.json")


class TorqueSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._torque_version_dialog: TreeOptionDialog | None = None
    self._speed_adaptive_mode_dialog: TreeOptionDialog | None = None
    self._roll_comp_gain_mode_dialog: TreeOptionDialog | None = None
    self._friction_breakaway_mode_dialog: TreeOptionDialog | None = None
    self._direction_gain_mode_dialog: TreeOptionDialog | None = None
    self.cached_torque_versions = {}
    self._load_versions()
    self._pending_lat_accel_factor = self._read_scaled_torque_value(
      "TorqueParamsOverrideLatAccelFactor", TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_DEFAULT, validate_torque_override_lat_accel_factor)
    self._pending_friction = self._read_scaled_torque_value(
      "TorqueParamsOverrideFriction", TORQUE_OVERRIDE_FRICTION_DEFAULT, validate_torque_override_friction)
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _load_versions(self):
    with open(TORQUE_VERSIONS_PATH) as f:
      self.cached_torque_versions = json.load(f)

  def _initialize_items(self):
    self._torque_control_versions = ListItemSP(
      title=tr("Torque Control Tune Version"),
      description=tr("Select the Torque Control Tune version. Changes apply to torque steering behavior; return to Default to use the standard tune."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_torque_version_dialog,
    )
    self._self_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsToggle",
      title=lambda: tr("Self-Tune"),
      description=lambda: tr("Enables self-tune for Torque lateral control for platforms that do not use " +
                             "Torque lateral control by default. Learned values can change steering response over time; " +
                             "disable to stop learning and return to fixed/offline torque tuning."),
    )
    self._relaxed_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsRelaxedToggle",
      title=lambda: tr("Less Restrict Settings for Self-Tune (Beta)"),
      description=lambda: tr("Less strict settings when using Self-Tune. This allows torqued to be more " +
                             "forgiving when learning values, but may accept noisier steering data. Disable to return to stricter learning."),
    )
    self._speed_adaptive_mode = ListItemSP(
      title=tr("Speed-Aware Curve"),
      description=tr("Choose how learned speed correction is used. Apply mode can change steering response by speed; Off returns to the base torque tune."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_speed_adaptive_mode_dialog,
    )
    self._low_speed_shadow_toggle = toggle_item_sp(
      param="LiveTorqueLowSpeedShadow",
      title=lambda: tr("Low-Speed Shadow Collection"),
      description=lambda: tr("Monitor only — collects low-speed steering data; no driving changes."),
    )
    self._roll_comp_gain_mode = ListItemSP(
      title=tr("Roll Compensation Gain"),
      description=tr("Learn how much extra torque is needed to hold lane on roads with crown or cross-slope. " +
                     "Monitor only mode collects data without changing steering; Apply learned gain can change steering " +
                     "response on crowned roads; turn Off to stop learning."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_roll_comp_gain_mode_dialog,
    )
    self._friction_breakaway_mode = ListItemSP(
      title=tr("Friction Breakaway Floor"),
      description=tr("Counteract steering-rack stick-slip by boosting friction compensation for small persistent corrections. " +
                     "Monitor only logs what the floor would add without changing steering; Apply can change steering " +
                     "response — expect slightly quicker small corrections; turn Off to disable."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_friction_breakaway_mode_dialog,
    )
    self._direction_gain_mode = ListItemSP(
      title=tr("Direction Gain Asymmetry"),
      description=tr("Learn whether steering responds more strongly per unit torque in one direction and balance it. " +
                     "Monitor only learns without changing steering; Apply scales torque per direction from the learned " +
                     "ratio; turn Off to disable."),
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_direction_gain_mode_dialog,
    )
    self._custom_tune_toggle = toggle_item_sp(
      param="CustomTorqueParams",
      title=lambda: tr("Enable Custom Tuning"),
      description=lambda: tr("Enables custom tuning for Torque lateral control. " +
                             "Modifying Lateral Acceleration Factor and Friction below will override the offline values " +
                             "indicated in the YAML files within \"opendbc/car/torque_data\". " +
                             "The values will also be used live when \"Manual Real-Time Tuning\" toggle is enabled. " +
                             "Large changes can bias steering or degrade tracking; turn this off to return to the selected tune."),
    )
    self._torque_prams_override_toggle = toggle_item_sp(
      param="TorqueParamsOverrideEnabled",
      title=lambda: tr("Manual Real-Time Tuning"),
      description=lambda: tr("Enforces the torque lateral controller to use the fixed values instead of the learned " +
                             "values from Self-Tune. Enabling this toggle applies the custom values live and overrides Self-Tune values. " +
                             "Monitor steering response and disable to return to learned/offline values."),
    )
    self._torque_lat_accel_factor = option_item_sp(
      title=lambda: tr("Lateral Acceleration Factor"),
      param="TorqueParamsOverrideLatAccelFactor",
      description=tr("Adjusts how strongly torque maps to lateral acceleration. In Real-Time mode this can affect steering " +
                     "immediately; revert by disabling Manual Real-Time Tuning or Custom Tuning."),
      min_value=int(TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MIN * 100),
      max_value=int(TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_MAX * 100),
      value_change_step=1,
      on_value_changed=self._on_pending_lat_accel_factor_changed,
      label_callback=(lambda x: f"{x/100} m/s^2"),
      use_float_scaling=True,
      write_param=False,
      initial_value=self._pending_lat_accel_factor,
    )

    self._torque_friction = option_item_sp(
      title=lambda: tr("Friction"),
      param="TorqueParamsOverrideFriction",
      description=tr("Adjusts friction compensation for torque steering. In Real-Time mode this can affect steering " +
                     "immediately; revert by disabling Manual Real-Time Tuning or Custom Tuning."),
      min_value=int(TORQUE_OVERRIDE_FRICTION_MIN * 100),
      max_value=int(TORQUE_OVERRIDE_FRICTION_MAX * 100),
      value_change_step=1,
      on_value_changed=self._on_pending_friction_changed,
      label_callback=(lambda x: f"{x/100}"),
      use_float_scaling=True,
      write_param=False,
      initial_value=self._pending_friction,
    )

    self._manual_torque_apply_revert = dual_button_item_sp(
      tr("Revert"), tr("Apply"),
      left_callback=self._revert_manual_torque_pending,
      right_callback=self._apply_manual_torque_pending,
      description=tr("Edit pending torque values above, then Apply to write both values together. Revert discards pending edits."),
      border_radius=20,
    )

    items = [
      self._torque_control_versions,
      self._self_tune_toggle,
      self._relaxed_tune_toggle,
      self._speed_adaptive_mode,
      self._low_speed_shadow_toggle,
      self._roll_comp_gain_mode,
      self._friction_breakaway_mode,
      self._direction_gain_mode,
      self._custom_tune_toggle,
      self._torque_prams_override_toggle,
      self._torque_lat_accel_factor,
      self._torque_friction,
      self._manual_torque_apply_revert,
    ]
    return items

  @staticmethod
  def _format_scaled(value: int, suffix: str = "") -> str:
    formatted = f"{value / 100:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}{suffix}"

  def _read_scaled_torque_value(self, key: str, default: float, validator: Callable[[object], float | None]) -> int:
    parsed = validator(ui_state.params.get(key, return_default=True))
    if parsed is None:
      parsed = default
    return int(round(parsed * 100))

  def _on_pending_lat_accel_factor_changed(self, value: int) -> None:
    self._pending_lat_accel_factor = value

  def _on_pending_friction_changed(self, value: int) -> None:
    self._pending_friction = value

  def _current_lat_accel_factor(self) -> int:
    return self._read_scaled_torque_value(
      "TorqueParamsOverrideLatAccelFactor", TORQUE_OVERRIDE_LAT_ACCEL_FACTOR_DEFAULT, validate_torque_override_lat_accel_factor)

  def _current_friction(self) -> int:
    return self._read_scaled_torque_value(
      "TorqueParamsOverrideFriction", TORQUE_OVERRIDE_FRICTION_DEFAULT, validate_torque_override_friction)

  def _manual_torque_pending_changed(self) -> bool:
    return self._pending_lat_accel_factor != self._current_lat_accel_factor() or self._pending_friction != self._current_friction()

  def _sync_pending_manual_torque_from_params(self) -> None:
    self._pending_lat_accel_factor = self._current_lat_accel_factor()
    self._pending_friction = self._current_friction()
    self._torque_lat_accel_factor.action_item.set_value(self._pending_lat_accel_factor)
    self._torque_friction.action_item.set_value(self._pending_friction)

  def _apply_manual_torque_pending(self) -> None:
    # Manual torque values are safety-critical and must only be written while offroad.
    if ui_state.is_onroad() or ui_state.engaged:
      return
    ui_state.params.put("TorqueParamsOverrideLatAccelFactor", self._pending_lat_accel_factor / 100.0)
    ui_state.params.put("TorqueParamsOverrideFriction", self._pending_friction / 100.0)

  def _revert_manual_torque_pending(self) -> None:
    self._sync_pending_manual_torque_from_params()

  def _update_state(self):
    super()._update_state()
    if not ui_state.params.get_bool("LiveTorqueParamsToggle"):
      ui_state.params.remove("LiveTorqueParamsRelaxedToggle")
      self._relaxed_tune_toggle.action_item.set_state(False)
    self._self_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._relaxed_tune_toggle.action_item.set_enabled(ui_state.is_offroad() and self._self_tune_toggle.action_item.get_state())
    self._speed_adaptive_mode.action_item.set_enabled(ui_state.is_offroad())
    speed_mode = ui_state.params.get("LiveTorqueSpeedAdaptiveMode") or b"off"
    try:
      speed_mode = speed_mode.decode() if isinstance(speed_mode, bytes) else str(speed_mode)
    except Exception:
      speed_mode = "off"
    self._low_speed_shadow_toggle.action_item.set_enabled(ui_state.is_offroad() and speed_mode != "off")
    self._roll_comp_gain_mode.action_item.set_enabled(ui_state.is_offroad())
    self._friction_breakaway_mode.action_item.set_enabled(ui_state.is_offroad())
    self._direction_gain_mode.action_item.set_enabled(ui_state.is_offroad())
    self._custom_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    custom_tune_enabled = self._custom_tune_toggle.action_item.get_state()
    self._torque_prams_override_toggle.set_visible(custom_tune_enabled)
    self._torque_lat_accel_factor.set_visible(custom_tune_enabled)
    self._torque_friction.set_visible(custom_tune_enabled)
    self._manual_torque_apply_revert.set_visible(custom_tune_enabled)

    self._torque_prams_override_toggle.action_item.set_enabled(ui_state.is_offroad())
    # Sliders and the Apply/Revert buttons are safety-critical and are only usable
    # offroad. Do not let a pre-enabled Manual Real-Time Tuning toggle make the
    # sliders writable while onroad.
    sliders_enabled = ui_state.is_offroad()
    self._torque_lat_accel_factor.action_item.set_enabled(sliders_enabled)
    self._torque_friction.action_item.set_enabled(sliders_enabled)
    self._manual_torque_apply_revert.action_item.set_enabled(
      ui_state.is_offroad() and custom_tune_enabled and self._manual_torque_pending_changed())

    title_text = tr("Real-Time & Offline") if ui_state.params.get("TorqueParamsOverrideEnabled") else tr("Offline Only")
    self._torque_lat_accel_factor.set_title(lambda: tr("Lateral Acceleration Factor") + " (" + title_text + ")")
    self._torque_friction.set_title(lambda: tr("Friction") + " (" + title_text + ")")
    self._torque_lat_accel_factor.set_description(
      tr("Current: ") + self._format_scaled(self._current_lat_accel_factor(), " m/s^2") + "<br>" +
      tr("Pending: ") + self._format_scaled(self._pending_lat_accel_factor, " m/s^2") + "<br>" +
      tr("Press Apply to write both pending torque values."))
    self._torque_friction.set_description(
      tr("Current: ") + self._format_scaled(self._current_friction()) + "<br>" +
      tr("Pending: ") + self._format_scaled(self._pending_friction) + "<br>" +
      tr("Press Apply to write both pending torque values."))
    self._torque_control_versions.action_item.set_value(self._get_current_torque_version_label())
    self._torque_control_versions.action_item.set_enabled(ui_state.is_offroad())
    self._speed_adaptive_mode.action_item.set_value(self._get_current_speed_mode_label())
    self._roll_comp_gain_mode.action_item.set_value(self._get_current_roll_comp_mode_label())
    self._friction_breakaway_mode.action_item.set_value(self._get_current_friction_breakaway_mode_label())
    self._direction_gain_mode.action_item.set_value(self._get_current_direction_gain_mode_label())


  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    # subtract button
    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40, rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._sync_pending_manual_torque_from_params()
    self._scroller.show_event()

  def _get_current_torque_version_label(self):
    current_val_bytes = ui_state.params.get("TorqueControlTune")
    if current_val_bytes is None:
      return tr("Default")

    try:
      current_val = float(current_val_bytes)
      for label, info in self.cached_torque_versions.items():
        if math.isclose(float(info["version"]), current_val, rel_tol=1e-5):
          return label
    except (ValueError, KeyError):
      pass

    return tr("Default")

  def _get_current_speed_mode_label(self):
    mode = ui_state.params.get("LiveTorqueSpeedAdaptiveMode") or b"off"
    try:
      mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    except Exception:
      mode = "off"
    return {"off": tr("Off"), "shadow": tr("Learn only"), "apply": tr("Apply learned curve")}.get(mode, tr("Off"))

  def _get_current_roll_comp_mode_label(self):
    mode = ui_state.params.get("RollCompGainMode") or b"off"
    try:
      mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    except Exception:
      mode = "off"
    mode = validate_roll_comp_gain_mode(mode)
    return {"off": tr("Off"), "shadow": tr("Learn only"), "apply": tr("Apply learned gain")}.get(mode, tr("Off"))

  def _get_current_direction_gain_mode_label(self):
    mode = ui_state.params.get("LatDirectionGainMode") or b"off"
    try:
      mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    except Exception:
      mode = "off"
    mode = validate_direction_gain_mode(mode)
    return {"off": tr("Off"), "shadow": tr("Monitor only"), "apply": tr("Apply")}.get(mode, tr("Off"))

  def _show_direction_gain_mode_dialog(self):
    nodes = [TreeNode(tr("Off")), TreeNode(tr("Monitor only")), TreeNode(tr("Apply"))]
    folders = [TreeFolder("", nodes)]
    current_label = self._get_current_direction_gain_mode_label()

    def handle_selection(result: int):
      if ui_state.is_onroad() or ui_state.engaged:
        self._direction_gain_mode_dialog = None
        return
      if result == DialogResult.CONFIRM and self._direction_gain_mode_dialog:
        selected = self._direction_gain_mode_dialog.selection_ref
        mapping = {tr("Off"): "off", tr("Monitor only"): "shadow", tr("Apply"): "apply"}
        mode = mapping.get(selected, "off")
        if mode == "off":
          ui_state.params.remove("LatDirectionGainMode")
        else:
          ui_state.params.put("LatDirectionGainMode", mode)
      self._direction_gain_mode_dialog = None

    # Safety gate: direction gain mode changes affect torque steering and are offroad-only.
    if ui_state.is_onroad() or ui_state.engaged:
      return

    self._direction_gain_mode_dialog = TreeOptionDialog(
      tr("Select Direction Gain Asymmetry Mode"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._direction_gain_mode_dialog)

  def _get_current_friction_breakaway_mode_label(self):
    mode = ui_state.params.get("LatFrictionBreakawayMode") or b"off"
    try:
      mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    except Exception:
      mode = "off"
    mode = validate_friction_breakaway_mode(mode)
    return {"off": tr("Off"), "shadow": tr("Monitor only"), "apply": tr("Apply")}.get(mode, tr("Off"))

  def _show_friction_breakaway_mode_dialog(self):
    nodes = [TreeNode(tr("Off")), TreeNode(tr("Monitor only")), TreeNode(tr("Apply"))]
    folders = [TreeFolder("", nodes)]
    current_label = self._get_current_friction_breakaway_mode_label()

    def handle_selection(result: int):
      if ui_state.is_onroad() or ui_state.engaged:
        self._friction_breakaway_mode_dialog = None
        return
      if result == DialogResult.CONFIRM and self._friction_breakaway_mode_dialog:
        selected = self._friction_breakaway_mode_dialog.selection_ref
        mapping = {tr("Off"): "off", tr("Monitor only"): "shadow", tr("Apply"): "apply"}
        mode = mapping.get(selected, "off")
        if mode == "off":
          ui_state.params.remove("LatFrictionBreakawayMode")
        else:
          ui_state.params.put("LatFrictionBreakawayMode", mode)
      self._friction_breakaway_mode_dialog = None

    # Safety gate: friction floor mode changes affect torque steering and are offroad-only.
    if ui_state.is_onroad() or ui_state.engaged:
      return

    self._friction_breakaway_mode_dialog = TreeOptionDialog(
      tr("Select Friction Breakaway Floor Mode"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._friction_breakaway_mode_dialog)

  def _show_speed_adaptive_mode_dialog(self):
    nodes = [TreeNode(tr("Off")), TreeNode(tr("Learn only")), TreeNode(tr("Apply learned curve"))]
    folders = [TreeFolder("", nodes)]
    current_label = self._get_current_speed_mode_label()

    def handle_selection(result: int):
      if ui_state.is_onroad() or ui_state.engaged:
        self._speed_adaptive_mode_dialog = None
        return
      if result == DialogResult.CONFIRM and self._speed_adaptive_mode_dialog:
        selected = self._speed_adaptive_mode_dialog.selection_ref
        mapping = {tr("Off"): "off", tr("Learn only"): "shadow", tr("Apply learned curve"): "apply"}
        mode = mapping.get(selected, "off")
        if mode == "off":
          ui_state.params.remove("LiveTorqueSpeedAdaptiveMode")
        else:
          ui_state.params.put("LiveTorqueSpeedAdaptiveMode", mode)
      self._speed_adaptive_mode_dialog = None

    # Safety gate: speed-adaptive mode changes affect torque steering and are offroad-only.
    if ui_state.is_onroad() or ui_state.engaged:
      return

    self._speed_adaptive_mode_dialog = TreeOptionDialog(
      tr("Select Speed-Aware Curve Mode"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._speed_adaptive_mode_dialog)

  def _show_roll_comp_gain_mode_dialog(self):
    nodes = [TreeNode(tr("Off")), TreeNode(tr("Learn only")), TreeNode(tr("Apply learned gain"))]
    folders = [TreeFolder("", nodes)]
    current_label = self._get_current_roll_comp_mode_label()

    def handle_selection(result: int):
      if ui_state.is_onroad() or ui_state.engaged:
        self._roll_comp_gain_mode_dialog = None
        return
      if result == DialogResult.CONFIRM and self._roll_comp_gain_mode_dialog:
        selected = self._roll_comp_gain_mode_dialog.selection_ref
        mapping = {tr("Off"): "off", tr("Learn only"): "shadow", tr("Apply learned gain"): "apply"}
        mode = mapping.get(selected, "off")
        if mode == "off":
          ui_state.params.remove("RollCompGainMode")
        else:
          ui_state.params.put("RollCompGainMode", mode)
      self._roll_comp_gain_mode_dialog = None

    # Safety gate: roll-comp gain mode changes affect torque steering and are offroad-only.
    if ui_state.is_onroad() or ui_state.engaged:
      return

    self._roll_comp_gain_mode_dialog = TreeOptionDialog(
      tr("Select Roll Compensation Gain Mode"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._roll_comp_gain_mode_dialog)

  def _show_torque_version_dialog(self):
    options_map = {}
    for label, info in self.cached_torque_versions.items():
      try:
        options_map[label] = float(info["version"])
      except (ValueError, KeyError):
        pass

    # Sort options by label in descending order
    sorted_labels = sorted(options_map.keys(), key=lambda k: options_map[k], reverse=True)

    nodes = [TreeNode(tr("Default"))]
    for label in sorted_labels:
      nodes.append(TreeNode(label))

    folders = [TreeFolder("", nodes)]

    current_label = self._get_current_torque_version_label()

    def handle_selection(result: int):
      if ui_state.is_onroad() or ui_state.engaged:
        self._torque_version_dialog = None
        return
      if result == DialogResult.CONFIRM and self._torque_version_dialog:
        selected_ref = self._torque_version_dialog.selection_ref
        if selected_ref == tr("Default"):
          ui_state.params.remove("TorqueControlTune")
        elif selected_ref in options_map:
          ui_state.params.put("TorqueControlTune", options_map[selected_ref])
      self._torque_version_dialog = None

    # Safety gate: version changes affect torque steering and are offroad-only.
    if ui_state.is_onroad() or ui_state.engaged:
      return

    self._torque_version_dialog = TreeOptionDialog(
      tr("Select Torque Control Tune Version"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._torque_version_dialog)
