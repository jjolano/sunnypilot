"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure evaluator for the visibility/enablement rules declared in the compiled
settings_ui.json. The same rule vocabulary the cloud/mobile frontend consumes
is evaluated here to drive the on-device UI, so a control's enable/visible
state is declared once in YAML instead of re-derived by hand in each panel's
_update_state().

No pyray / cereal / ui_state imports: this module is unit-testable headless.
The rule grammar mirrors sunnypilot/sunnylink/settings_ui_src/_schemas/rule.schema.json:

  offroad_only                         -> ctx.is_offroad
  not_engaged                          -> not ctx.is_engaged
  capability {field, equals}           -> ctx.capabilities[field] == equals
  param {key, equals}                  -> param value == equals
  param_compare {key, op, value}       -> numeric compare (>, <, >=, <=)
  not {condition}                      -> negation
  any {conditions: [...]}              -> OR
  all {conditions: [...]}              -> AND

A *list* of rules (an item's "enablement" / "visibility") is AND-combined.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ParamsLike(Protocol):
  """The slice of common.params.Params (and test fakes) the evaluator needs."""
  def get_bool(self, key: str) -> bool: ...
  def get(self, key: str, *args: Any, **kwargs: Any) -> Any: ...


class RuleError(ValueError):
  """Raised on a malformed rule (unknown type, or an unresolved macro $ref)."""


@dataclass
class RuleContext:
  """Everything a rule can interrogate, gathered once per render frame.

  `capabilities` is the dict returned by
  openpilot.sunnypilot.sunnylink.capabilities.generate_capabilities() — the
  exact same capability values the cloud schema is evaluated against.
  """
  params: ParamsLike
  capabilities: dict[str, Any] = field(default_factory=dict)
  is_offroad: bool = True
  is_engaged: bool = False


_COMPARE = {
  ">": lambda a, b: a > b,
  "<": lambda a, b: a < b,
  ">=": lambda a, b: a >= b,
  "<=": lambda a, b: a <= b,
}


def _as_number(raw: Any) -> float | None:
  """Coerce a param value (bytes/str/number/bool/None) to float, or None."""
  if raw is None:
    return None
  if isinstance(raw, bool):
    return float(raw)
  if isinstance(raw, (int, float)):
    return float(raw)
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8", "replace")
  try:
    return float(raw)
  except (TypeError, ValueError):
    return None


def _param_equals(ctx: RuleContext, key: str, expected: Any) -> bool:
  # Booleans are the common case and read through the typed accessor.
  if isinstance(expected, bool):
    return bool(ctx.params.get_bool(key)) == expected
  # Numeric comparison when the expected value is a number.
  expected_num = _as_number(expected)
  if expected_num is not None:
    actual_num = _as_number(ctx.params.get(key))
    return actual_num is not None and actual_num == expected_num
  # Fall back to string comparison (e.g. enum params stored as text).
  raw = ctx.params.get(key)
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8", "replace")
  return str(raw) == str(expected)


def evaluate_rule(rule: dict, ctx: RuleContext) -> bool:
  """Evaluate a single rule node against the context."""
  if "$ref" in rule:
    ref = rule["$ref"]
    raise RuleError(f"unresolved macro ref {ref!r}: evaluate the compiled settings_ui.json, not the YAML source")

  rule_type = rule.get("type")

  if rule_type == "offroad_only":
    return ctx.is_offroad
  if rule_type == "not_engaged":
    return not ctx.is_engaged
  if rule_type == "capability":
    return ctx.capabilities.get(rule["field"]) == rule["equals"]
  if rule_type == "param":
    return _param_equals(ctx, rule["key"], rule["equals"])
  if rule_type == "param_compare":
    actual = _as_number(ctx.params.get(rule["key"]))
    if actual is None:
      return False
    op = _COMPARE.get(rule["op"])
    if op is None:
      raise RuleError(f"unknown param_compare op {rule['op']!r}")
    return op(actual, float(rule["value"]))
  if rule_type == "not":
    return not evaluate_rule(rule["condition"], ctx)
  if rule_type == "any":
    return any(evaluate_rule(c, ctx) for c in rule["conditions"])
  if rule_type == "all":
    return all(evaluate_rule(c, ctx) for c in rule["conditions"])

  raise RuleError(f"unknown rule type {rule_type!r}")


def rules_pass(rules: list[dict] | None, ctx: RuleContext) -> bool:
  """A rule list is satisfied when ALL of its rules pass (schema AND-semantics).

  An empty/absent list is vacuously true (no restriction).
  """
  if not rules:
    return True
  return all(evaluate_rule(rule, ctx) for rule in rules)
