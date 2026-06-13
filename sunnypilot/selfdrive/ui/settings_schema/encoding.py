"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure schema -> OptionControlSP encoding (no pyray import; headless-testable).

OptionControlSP works in integers (with an optional /100 fixed-point mode), so a
float-valued numeric setting must be encoded as an int range + scale. Today each
panel hand-encodes this, and the encodings have drifted from the schema (the
torque sliders are 0.01-step on-device but 0.1-step in the schema). Deriving the
encoding from the schema's min/max/step in ONE place is what removes that class
of bug — and keeping it pure means the derivation is itself unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass

# Fixed-point denominator used by OptionControlSP.use_float_scaling.
FLOAT_SCALE = 100


@dataclass(frozen=True)
class OptionEncoding:
  """Integer encoding of a numeric schema option for OptionControlSP."""
  min_value: int
  max_value: int
  value_change_step: int
  use_float_scaling: bool

  @property
  def scale(self) -> int:
    return FLOAT_SCALE if self.use_float_scaling else 1

  def to_display(self, int_value: int) -> float:
    """Map a stored int back to the real-world value the user sees."""
    return int_value / self.scale


def encode_numeric_option(item: dict) -> OptionEncoding:
  """Derive the OptionControlSP int encoding from a numeric `option` item.

  Integer ranges pass through unscaled; any fractional bound/step uses /100
  fixed-point (step 0.1 -> int step 10, step 0.01 -> int step 1).
  """
  lo = float(item["min"])
  hi = float(item["max"])
  step = float(item.get("step", 1))

  if lo.is_integer() and hi.is_integer() and step.is_integer():
    return OptionEncoding(int(lo), int(hi), int(step), use_float_scaling=False)

  return OptionEncoding(
    min_value=round(lo * FLOAT_SCALE),
    max_value=round(hi * FLOAT_SCALE),
    value_change_step=max(1, round(step * FLOAT_SCALE)),
    use_float_scaling=True,
  )


@dataclass(frozen=True)
class ContiguousEnum:
  """A run of consecutive integer options with a label per value.

  Renders as an OptionControlSP stepper over [min_value, max_value] — exactly how
  the device hand-codes AutoLaneChangeTimer (lane_change_settings.py): a -1..5
  range with a value->label callback, no value_map needed.
  """
  min_value: int
  max_value: int
  labels_by_value: dict[int, str]


def contiguous_int_options(item: dict) -> ContiguousEnum | None:
  """Detect an enumerated option whose values are consecutive ints (any start).

  Handles the negative/non-zero-based case sequential_int_labels() rejects
  (AutoLaneChangeTimer's -1..5). Returns None for string/float/gapped values,
  which need a value-mapped or custom selector (escape hatch).
  """
  options = item.get("options")
  if not options:
    return None
  values: list[int] = []
  labels: dict[int, str] = {}
  for opt in options:
    value = opt.get("value")
    if not isinstance(value, int) or isinstance(value, bool):
      return None
    values.append(value)
    labels[value] = opt.get("label", str(value))
  for prev, nxt in zip(values, values[1:], strict=False):
    if nxt != prev + 1:
      return None
  return ContiguousEnum(min_value=values[0], max_value=values[-1], labels_by_value=labels)


def sequential_int_labels(item: dict) -> list[str] | None:
  """Return option labels iff the option values are exactly 0..n-1 ints.

  MultipleButtonActionSP writes the selected *index* to the param, so it only
  faithfully represents enums whose stored values are sequential from zero.
  Negative, string, or float values (e.g. the off/shadow/apply speed-aware
  curve, or the -1..5 lane-change timer) return None — escape-hatch cases that
  need a value-mapped or custom selector.
  """
  options = item.get("options")
  if not options:
    return None
  for i, opt in enumerate(options):
    value = opt.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or value != i:
      return None
  return [opt["label"] for opt in options]
