import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.params import finite_float_param, should_refresh_params


def test_should_refresh_params_preserves_current_cadence():
  assert should_refresh_params(0, 0.5, 0.1)
  assert should_refresh_params(5, 0.5, 0.1)
  assert not should_refresh_params(1, 0.5, 0.1)


@pytest.mark.parametrize("value", ["bad", None, float("nan"), float("inf"), float("-inf")])
def test_finite_float_param_defaults_to_zero_for_invalid_values(value):
  assert finite_float_param(value) == pytest.approx(0.0)


def test_finite_float_param_parses_numeric_values():
  assert finite_float_param("1.5") == pytest.approx(1.5)
