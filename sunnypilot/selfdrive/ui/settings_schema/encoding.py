"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure schema -> OptionControlSP encoding (no pyray import; headless-testable).

OptionControlSP works in integers (with an optional /100 fixed-point mode), so a
float-valued numeric setting must be encoded as an int range + scale. Button-row
selectors have their own safe subset: sequential integer enums can use the
widget's built-in param=index shortcut, while homogeneous string enums need an
explicit value mapping. Deriving these encodings from the schema in ONE place is
what removes drift from hand-coded panels — and keeping it pure means the
derivation is itself unit-tested.
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


@dataclass(frozen=True)
class ValueMappedOption:
  """Enumerated integer options whose stored values aren't a 0/contiguous run.

  Renders as an OptionControlSP stepper over display indices 1..N with a
  value_map to the real stored values — matching the device's hand-coded
  option_item_sp(value_map=...) controls (e.g. CustomAccLongPressIncrement's
  {1: 1, 2: 5, 3: 10}). Use this only when contiguous_int_options() returns None.
  """
  value_map: dict[int, int]        # display index (1..N) -> stored param value
  labels_by_value: dict[int, str]  # stored param value -> label (OptionControlSP passes the mapped value)

  @property
  def min_value(self) -> int:
    return 1

  @property
  def max_value(self) -> int:
    return len(self.value_map)


def value_mapped_option(item: dict) -> ValueMappedOption | None:
  """Detect an enumerated option with arbitrary (gapped/non-zero) integer values."""
  options = item.get("options")
  if not options:
    return None
  values: list[int] = []
  labels: list[str] = []
  for opt in options:
    value = opt.get("value")
    if not isinstance(value, int) or isinstance(value, bool):
      return None
    values.append(value)
    labels.append(opt.get("label", str(value)))
  value_map = {i + 1: values[i] for i in range(len(values))}
  labels_by_value = {values[i]: labels[i] for i in range(len(values))}
  return ValueMappedOption(value_map=value_map, labels_by_value=labels_by_value)


@dataclass(frozen=True)
class StringEnum:
  """Homogeneous string enum options for a mapped MultipleButtonActionSP row."""
  values: list[str]
  labels: list[str]


def homogeneous_string_options(item: dict) -> StringEnum | None:
  """Detect an enumerated option whose values are all strings.

  Mixed string/float selectors stay in the escape hatch because they often need
  dialog/custom-widget behavior, while homogeneous strings can be faithfully
  rendered by writing the selected schema value through an explicit callback.
  """
  options = item.get("options")
  if not options:
    return None
  values: list[str] = []
  labels: list[str] = []
  for opt in options:
    if not isinstance(opt, dict):
      return None
    value = opt.get("value")
    if not isinstance(value, str):
      return None
    values.append(value)
    labels.append(str(opt.get("label", value)))
  return StringEnum(values=values, labels=labels)


def string_option_index(value: object, enum: StringEnum, key: str) -> int:
  """Resolve a stored string-param value to a button index for a string enum.

  `CustomLongitudinalMode` accepts legacy numeric values for restore/migration
  compatibility. Missing/empty values keep the repo default SCC; invalid non-empty
  values fall back to ACC to match `LongitudinalMode.from_value()` fail-safe
  behavior in the planner.
  """
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  text = str(value or "").strip()
  if key == "CustomLongitudinalMode":
    if not text:
      return enum.values.index("scc") if "scc" in enum.values else 0
    text = {"0": "acc", "1": "e2e", "2": "scc"}.get(text.lower(), text.lower())
    for i, option_value in enumerate(enum.values):
      if option_value.lower() == text:
        return i
    return enum.values.index("acc") if "acc" in enum.values else 0
  return enum.values.index(text) if text in enum.values else 0


def sequential_int_labels(item: dict) -> list[str] | None:
  """Return option labels iff the option values are exactly 0..n-1 ints.

  MultipleButtonActionSP writes the selected *index* to the param, so it only
  faithfully represents enums whose stored values are sequential from zero.
  Negative, string, or float values return None. Gapped/negative integers use
  other encodings; homogeneous strings use homogeneous_string_options(); mixed
  enums remain escape-hatch cases that need a custom selector.
  """
  options = item.get("options")
  if not options:
    return None
  for i, opt in enumerate(options):
    value = opt.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or value != i:
      return None
  return [opt["label"] for opt in options]
