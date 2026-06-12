import math

import pytest

from openpilot.selfdrive.controls.lib.diagnostic_normalization import (
  NormalizedDiagnostic,
  clamp01,
  normalize_inverse_unit_interval,
  normalize_range,
  normalize_unit_interval,
)


def test_clamp01_pass_through_and_clamping():
  assert clamp01(0.25) == pytest.approx(0.25)
  assert clamp01(-0.25) == pytest.approx(0.0)
  assert clamp01(1.25) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "x"])
def test_invalid_inputs_return_invalid_diagnostic(value):
  diag = normalize_unit_interval(value, "src", "kind", "reason")

  assert diag == NormalizedDiagnostic("src", "kind", 0.0, False, "reason")


def test_inverse_mapping():
  diag = normalize_inverse_unit_interval(0.25, "src", "kind")

  assert diag.value == pytest.approx(0.75)
  assert diag.valid


def test_range_low_high_mid_and_outside():
  assert normalize_range(0.0, 0.0, 10.0, "src", "kind").value == pytest.approx(0.0)
  assert normalize_range(10.0, 0.0, 10.0, "src", "kind").value == pytest.approx(1.0)
  assert normalize_range(5.0, 0.0, 10.0, "src", "kind").value == pytest.approx(0.5)
  assert normalize_range(-5.0, 0.0, 10.0, "src", "kind").value == pytest.approx(0.0)
  assert normalize_range(15.0, 0.0, 10.0, "src", "kind").value == pytest.approx(1.0)


def test_inverted_range():
  assert normalize_range(0.0, 0.0, 10.0, "src", "kind", invert=True).value == pytest.approx(1.0)
  assert normalize_range(10.0, 0.0, 10.0, "src", "kind", invert=True).value == pytest.approx(0.0)
  assert normalize_range(5.0, 0.0, 10.0, "src", "kind", invert=True).value == pytest.approx(0.5)


@pytest.mark.parametrize("low,high", [(math.nan, 1.0), (0.0, math.inf), (2.0, 2.0)])
def test_invalid_range(low, high):
  diag = normalize_range(1.0, low, high, "src", "kind", "bad")

  assert not diag.valid
  assert diag.value == 0.0
  assert diag.reason == "bad"


def test_source_kind_reason_preserved():
  diag = normalize_range(5.0, 0.0, 10.0, "diag-source", "diag-kind", "diag-reason")

  assert diag.source == "diag-source"
  assert diag.kind == "diag-kind"
  assert diag.reason == "diag-reason"
