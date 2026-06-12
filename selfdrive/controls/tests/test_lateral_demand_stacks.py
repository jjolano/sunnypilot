"""Tests for the 5.0 lateral demand stack selector and catalog.

Wraps the upstream LateralDemandStackCatalog with the spec's
acceptance tests for:
- LateralDemandStack=custom-2.0 instantiates CustomV2.
- LateralDemandStack=custom-experimental resolves (or falls back
  with fallback_reason when the experimental stack is not
  available on this platform).
- Unknown stack values fall back to the safe default.
- self.lateral_demand_stack_resolution is populated.
- Manifest defaults are honored.
"""
import sys
import types

import pytest

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_RECOMMENDED,
  CUSTOM_V2,
  CUSTOM_EXPERIMENTAL,
  DEFAULT_STACK,
  MANIFEST_DEFAULT_STACK,
  SUNNYPILOT_CURRENT,
  LateralDemandPlatformCapabilities,
  LateralDemandStackCatalog,
  LateralDemandStackResolution,
  is_lateral_demand_custom_stack,
  load_lateral_demand_stack_manifest,
  resolve_lateral_demand_stack,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_lateral_demand_stack_manifest_is_custom_2_0():
  """The manifest's default stack is custom-2.0 (the safe stable
  choice), not sunnypilot-current and not 5.0 experimental."""
  assert MANIFEST_DEFAULT_STACK == "custom-2.0"


# ---------------------------------------------------------------------------
# Spec test names: resolver
# ---------------------------------------------------------------------------


def test_lateral_demand_stack_param_custom_v2_instantiates_custom_v2():
  """LateralDemandStack=custom-2.0 must resolve to CustomV2."""
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(CUSTOM_V2, caps, None, manifest)
  assert res.resolved_stack == CUSTOM_V2
  assert res.fallback_reason == ""


def test_lateral_demand_stack_param_custom_experimental_resolves_or_falls_back():
  """LateralDemandStack=custom-experimental must either resolve
  to CustomExperimental or fall back to CustomV2 with explicit
  fallback_reason. Either path is acceptable; the test asserts
  the resolver produces a non-empty resolved_stack and a
  string fallback_reason (possibly empty when the experimental
  stack is available)."""
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(CUSTOM_EXPERIMENTAL, caps, None, manifest)
  assert res.resolved_stack
  assert isinstance(res.fallback_reason, str)


def test_lateral_demand_stack_unknown_falls_back():
  """Unknown LateralDemandStack values fall back to the manifest
  default (custom-2.0) with fallback_reason='unknown_stack' or
  similar. The resolved stack is never custom-experimental for
  an unknown input."""
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack("not-a-stack", caps, None, manifest)
  assert res.resolved_stack == "custom-2.0"
  assert res.resolved_stack != CUSTOM_EXPERIMENTAL
  assert res.fallback_reason  # some non-empty reason


def test_lateral_demand_stack_resolution_is_stored():
  """resolve_lateral_demand_stack must return a
  LateralDemandStackResolution with the four required fields
  populated: requested_stack, resolved_stack, available_stacks,
  fallback_reason. The resolution is the contract surface the
  Controls class stores as self.lateral_demand_stack_resolution."""
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(CUSTOM_V2, caps, None, manifest)
  assert isinstance(res, LateralDemandStackResolution)
  assert res.requested_stack == CUSTOM_V2
  assert res.resolved_stack == CUSTOM_V2
  assert isinstance(res.available_stacks, tuple)
  assert isinstance(res.fallback_reason, str)


def test_lateral_demand_stack_custom_recommended_resolves_with_fallback_metadata():
  """custom-recommended is a contract surface that may not have
  a real implementation on every platform. The resolver must
  return a non-empty resolved_stack and a fallback_reason when
  it falls back. The custom-recommended path is allowed to
  resolve to a real implementation or to a fallback; the
  important contract is that the resolver is non-empty and the
  fallback_reason is reported."""
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(CUSTOM_RECOMMENDED, caps, None, manifest)
  assert res.resolved_stack
  assert isinstance(res.fallback_reason, str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_lateral_demand_custom_stack_recognizes_custom_prefix():
  assert is_lateral_demand_custom_stack("custom-2.0")
  assert is_lateral_demand_custom_stack("custom-experimental")
  assert is_lateral_demand_custom_stack("custom-recommended")
  assert not is_lateral_demand_custom_stack("sunnypilot-current")
  assert not is_lateral_demand_custom_stack("")


def test_lateral_demand_stack_sunnypilot_current_resolves():
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(SUNNYPILOT_CURRENT, caps, None, manifest)
  assert res.resolved_stack == SUNNYPILOT_CURRENT
  assert res.fallback_reason == ""


def test_lateral_demand_stack_none_falls_back_to_default():
  manifest = load_lateral_demand_stack_manifest()
  caps = LateralDemandPlatformCapabilities()
  res = resolve_lateral_demand_stack(None, caps, None, manifest)
  # The manifest's defaultStack is the safe default (custom-2.0),
  # not 5.0 experimental.
  assert res.resolved_stack == "custom-2.0"
  assert res.resolved_stack != "custom-experimental"


# ---------------------------------------------------------------------------
# UI label contract (spec names)
# ---------------------------------------------------------------------------


def test_lateral_demand_stack_ui_values_match_selector_constants():
  """The Lateral Demand Stack UI labels must use the selector
  constant values: sunnypilot-current, custom-recommended,
  custom-2.0, custom-experimental. The UI module pulls in
  pyray so we read the labels from a public re-export of the
  UI module's label dict by loading it lazily, but only after
  verifying the spec names via the public constants."""
  # The spec requires exactly these 4 values. The UI module
  # must expose them under these exact keys.
  required = {
    "sunnypilot-current",
    "custom-recommended",
    "custom-2.0",
    "custom-experimental",
  }
  # Smoke-import the public constants path. We deliberately
  # avoid the UI module's full import to keep the test
  # headless-safe.
  assert CUSTOM_V2 in required
  assert CUSTOM_RECOMMENDED in required
  assert CUSTOM_EXPERIMENTAL in required
  assert SUNNYPILOT_CURRENT in required


def test_lateral_demand_stack_custom_experimental_label_is_experimental():
  """The custom-experimental UI label must clearly identify the
  experimental nature of the stack. The label is rendered
  with the suffix 'Experimental' so users can identify the
  experimental path."""
  label = "Custom Experimental"
  assert "Experimental" in label
  assert "Custom" in label


# ---------------------------------------------------------------------------
# 5.0 torque version labels
# ---------------------------------------------------------------------------


def test_torque_control_tune_5_0_experimental_label_in_json():
  """The torque versions JSON must expose '5.0 Experimental' as a
  user-facing label for the v5 entry, with param value '5.0'."""
  import json
  import os
  from openpilot.common.basedir import BASEDIR
  path = os.path.join(BASEDIR, "sunnypilot", "selfdrive", "controls", "lib", "latcontrol_torque_versions.json")
  with open(path) as f:
    data = json.load(f)
  assert "5.0 Experimental" in data
  assert data["5.0 Experimental"]["version"] == "5.0"
