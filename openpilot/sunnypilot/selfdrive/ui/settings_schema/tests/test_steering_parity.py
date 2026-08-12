"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity proof for the schema-driven steering panel.

Claim under test: the declarative visibility/enablement rules in the compiled
settings_ui.json, evaluated by rules.py, reproduce the hand-coded _update_state()
decisions of the device steering layouts:

  - selfdrive/ui/sunnypilot/layouts/settings/steering.py
  - selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/mads_settings.py

Each oracle below is transcribed from those files (line refs inline). The tests
sweep a state matrix and assert the rule engine agrees with the oracle. Two
tests also *document* the places the systems currently diverge — the residue
that genuinely needs escape hatches, and a value drift the unification fixes.

Pure-logic: no pyray, runs headless.
"""
from __future__ import annotations

import itertools

import pytest

from openpilot.sunnypilot.selfdrive.ui.settings_schema.rules import (
  RuleContext, RuleError, evaluate_rule, rules_pass,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import (
  find_item, get_panel, iter_items, iter_rules, load_schema,
)

SCHEMA = load_schema()
STEERING = get_panel(SCHEMA, "steering")
MODELS = get_panel(SCHEMA, "models")


class FakeParams:
  """Dict-backed stand-in for common.params.Params (get_bool / get only)."""
  def __init__(self, **values):
    self._v = dict(values)

  def get_bool(self, key: str) -> bool:
    return bool(self._v.get(key, False))

  def get(self, key: str, *args, **kwargs):
    return self._v.get(key)


def caps(brand="toyota", torque_allowed=True, steer_control_type="torque",
         enable_bsm=True, tesla_has_vehicle_bus=False, **extra):
  base = {
    "brand": brand,
    "torque_allowed": torque_allowed,
    "steer_control_type": steer_control_type,
    "enable_bsm": enable_bsm,
    "tesla_has_vehicle_bus": tesla_has_vehicle_bus,
    "has_longitudinal_control": True,
    "has_icbm": False,
  }
  base.update(extra)
  return base


def make_ctx(is_offroad=True, params=None, **cap_kwargs):
  return RuleContext(params=params or FakeParams(),
                     capabilities=caps(**cap_kwargs), is_offroad=is_offroad)


def enablement_of(key):
  item = find_item(STEERING, key)
  assert item is not None, f"{key} missing from compiled steering schema"
  return item.get("enablement")


def visibility_of(key):
  item = find_item(STEERING, key)
  assert item is not None, f"{key} missing from compiled steering schema"
  return item.get("visibility")


# --- Oracles transcribed from the device layouts -----------------------------

def oracle_mads_enabled(is_offroad):
  # steering.py:130  -> mads_toggle.set_enabled(ui_state.is_offroad())
  return is_offroad


def oracle_enforce_torque_enabled(is_offroad, torque_allowed, nnlc):
  # steering.py:143  -> is_offroad and torque_allowed and not nnlc
  return is_offroad and torque_allowed and not nnlc


def mads_limited(brand, tesla_has_vehicle_bus):
  # mads_settings.py:96-100  (_mads_limited_settings)
  if brand == "rivian":
    return True
  if brand == "tesla":
    return not tesla_has_vehicle_bus
  return False


def oracle_main_cruise_enabled(is_offroad, brand, tesla_has_vehicle_bus):
  # mads_settings.py:118/130 gated on (not limited); schema also ANDs offroad_only.
  return is_offroad and not mads_limited(brand, tesla_has_vehicle_bus)


# --- Parity tests ------------------------------------------------------------

@pytest.mark.parametrize("is_offroad", [True, False])
def test_mads_toggle_enablement(is_offroad):
  ctx = make_ctx(is_offroad=is_offroad)
  assert rules_pass(enablement_of("Mads"), ctx) == oracle_mads_enabled(is_offroad)


@pytest.mark.parametrize("is_offroad,torque_allowed,nnlc",
                         list(itertools.product([True, False], repeat=3)))
def test_enforce_torque_enablement(is_offroad, torque_allowed, nnlc):
  steer = "torque" if torque_allowed else "angle"
  ctx = make_ctx(is_offroad=is_offroad, torque_allowed=torque_allowed,
                 steer_control_type=steer,
                 params=FakeParams(NeuralNetworkLateralControl=nnlc))
  assert rules_pass(enablement_of("EnforceTorqueControl"), ctx) == \
      oracle_enforce_torque_enabled(is_offroad, torque_allowed, nnlc)


@pytest.mark.parametrize("steer_control_type", ["torque", "angle"])
def test_enforce_torque_visibility_tracks_steer_type(steer_control_type):
  # steering.yaml visibility: not(steer_control_type == angle).
  ctx = make_ctx(steer_control_type=steer_control_type)
  assert rules_pass(visibility_of("EnforceTorqueControl"), ctx) == (steer_control_type != "angle")


@pytest.mark.parametrize("is_offroad,brand,tesla_bus",
                         list(itertools.product([True, False],
                                                ["toyota", "rivian", "tesla"],
                                                [True, False])))
def test_main_cruise_enablement_matches_limited_logic(is_offroad, brand, tesla_bus):
  ctx = make_ctx(is_offroad=is_offroad, brand=brand, tesla_has_vehicle_bus=tesla_bus)
  assert rules_pass(enablement_of("MadsMainCruiseAllowed"), ctx) == \
      oracle_main_cruise_enabled(is_offroad, brand, tesla_bus)


@pytest.mark.parametrize("brand,tesla_bus",
                         list(itertools.product(["toyota", "rivian", "tesla"], [True, False])))
def test_mads_steering_mode_per_option_enablement(brand, tesla_bus):
  # mads_settings.py:127-128 -> when limited, only the DISENGAGE button is enabled.
  # Schema encodes this per-option: Remain Active(0)/Pause(1) carry mads_full_platforms;
  # Disengage(2) carries none (always enabled).
  item = find_item(STEERING, "MadsSteeringMode")
  options = item["options"]
  limited = mads_limited(brand, tesla_bus)
  ctx = make_ctx(brand=brand, tesla_has_vehicle_bus=tesla_bus)

  enabled = {opt["value"]: rules_pass(opt.get("enablement"), ctx) for opt in options}
  assert enabled[0] == (not limited)   # Remain Active
  assert enabled[1] == (not limited)   # Pause
  assert enabled[2] is True            # Disengage always selectable


@pytest.mark.parametrize("blinker_on", [True, False])
def test_blinker_subitem_enablement_tracks_parent(blinker_on):
  # steering.py:132-133 toggles VISIBILITY of these; schema encodes ENABLEMENT.
  # Same dependency (the parent param), different affordance (hide vs grey) —
  # the value asserted here is the dependency, which both agree on.
  ctx = make_ctx(params=FakeParams(BlinkerPauseLateralControl=blinker_on))
  assert rules_pass(enablement_of("BlinkerMinLateralControlSpeed"), ctx) == blinker_on
  assert rules_pass(enablement_of("BlinkerLateralReengageDelay"), ctx) == blinker_on


@pytest.mark.parametrize("enable_bsm,timer", [(True, 0), (True, 3), (False, 3), (True, -1)])
def test_bsm_delay_param_compare(enable_bsm, timer):
  # steering.yaml: enablement [capability enable_bsm, param_compare AutoLaneChangeTimer > 0]
  ctx = make_ctx(enable_bsm=enable_bsm, params=FakeParams(AutoLaneChangeTimer=timer))
  assert rules_pass(enablement_of("AutoLaneChangeBsmDelay"), ctx) == (enable_bsm and timer > 0)


@pytest.mark.parametrize("is_offroad,speed_mode", list(itertools.product([True, False], ["off", "shadow", "apply"])))
def test_low_speed_shadow_enablement_tracks_speed_aware_mode(is_offroad, speed_mode):
  # torque_settings.py: low-speed shadow toggle enabled offroad and when speed-aware mode is active.
  ctx = make_ctx(is_offroad=is_offroad, params=FakeParams(LiveTorqueSpeedAdaptiveMode=speed_mode))
  assert rules_pass(enablement_of("LiveTorqueLowSpeedShadow"), ctx) == (is_offroad and speed_mode != "off")


# --- Hygiene / guardrails ----------------------------------------------------

def test_compiled_schema_has_no_unresolved_macros():
  # The renderer must consume the compiled JSON (macros expanded), never the YAML.
  for item in iter_items(STEERING):
    for field_name in ("visibility", "enablement"):
      for rule in iter_rules(item.get(field_name) or []):
        assert "$ref" not in rule, f"{item.get('key')} has an unexpanded macro {rule}"


def test_evaluator_rejects_macro_ref():
  with pytest.raises(RuleError):
    evaluate_rule({"$ref": "#/macros/offroad"}, make_ctx())


def test_every_steering_item_has_title_and_key():
  for item in iter_items(STEERING):
    assert item.get("key"), f"item without key: {item}"
    assert item.get("title"), f"{item['key']} has no title"


# --- Divergence documentation (these assert the CURRENT drift) ---------------

def test_value_drift_between_schema_and_device_is_real():
  """The torque sliders disagree on min/step between the two systems TODAY.

  Schema (steering.yaml): LatAccelFactor min 0.1 step 0.1; Friction min 0.0.
  Device (torque_settings.py): min_value=1/100=0.01 step 1/100=0.01 for both.
  Unifying on the schema is what removes this — until then, this asserts the gap
  so the prototype can't silently claim parity it doesn't have.
  """
  lat = find_item(STEERING, "TorqueParamsOverrideLatAccelFactor")
  fric = find_item(STEERING, "TorqueParamsOverrideFriction")

  # device-effective values (torque_settings.py:86-101, use_float_scaling => /100)
  device_lat = {"min": 1 / 100, "step": 1 / 100}
  device_fric = {"min": 1 / 100, "step": 1 / 100}

  assert lat["min"] != device_lat["min"]    # 0.1  vs 0.01
  assert lat["step"] != device_lat["step"]  # 0.1  vs 0.01
  assert fric["min"] != device_fric["min"]  # 0.0  vs 0.01


def test_nnlc_present_and_mutually_exclusive_with_enforce_torque():
  """NNLC was device-only drift; the conversion added it to the schema.

  It lives in the models panel (moved there with the model-vision settings);
  its enablement mirrors steering.py:142 (offroad and torque_allowed and not
  EnforceTorqueControl), and EnforceTorqueControl reciprocally gates on NNLC.
  """
  nnlc = find_item(MODELS, "NeuralNetworkLateralControl")
  assert nnlc is not None, "NNLC must be in the schema now that it's the production panel"

  for is_offroad in (True, False):
    for torque_allowed in (True, False):
      for enforce in (True, False):
        ctx = make_ctx(is_offroad=is_offroad, torque_allowed=torque_allowed,
                       steer_control_type="torque" if torque_allowed else "angle",
                       params=FakeParams(EnforceTorqueControl=enforce))
        expected = is_offroad and torque_allowed and not enforce
        assert rules_pass(nnlc.get("enablement"), ctx) == expected

  # The mutual exclusion is symmetric: EnforceTorqueControl gates on NNLC == false.
  refs = [r for r in iter_rules(enablement_of("EnforceTorqueControl"))
          if r.get("type") == "param" and r.get("key") == "NeuralNetworkLateralControl"]
  assert refs
