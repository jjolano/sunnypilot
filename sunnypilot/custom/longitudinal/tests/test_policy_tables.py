"""Tests for personality decoding and policy table selection."""
from __future__ import annotations

import pytest
from cereal import log

from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality


PERSONALITY_ENUMERANTS = log.LongitudinalPersonality.schema.enumerants


@pytest.mark.parametrize("name, ordinal", PERSONALITY_ENUMERANTS.items())
def test_from_value_uses_cereal_ordinals(name, ordinal):
  expected = Personality[name.upper()]
  assert Personality.from_value(ordinal) is expected
  assert Personality.from_value(str(ordinal)) is expected
  assert Personality.from_value(str(ordinal).encode()) is expected


@pytest.mark.parametrize("name", PERSONALITY_ENUMERANTS)
def test_from_value_preserves_named_case_and_whitespace_forms(name):
  expected = Personality[name.upper()]
  assert Personality.from_value(name) is expected
  assert Personality.from_value(name.upper()) is expected
  assert Personality.from_value(f" \t{name.upper()} \n") is expected
  assert Personality.from_value(f" {name} ".encode()) is expected


@pytest.mark.parametrize("value", (None, "", " ", 3, "3", b"3", "unknown", b"\xff"))
def test_from_value_falls_back_to_standard_for_invalid_values(value):
  assert Personality.from_value(value) is Personality.STANDARD
  for default in (Personality.AGGRESSIVE, Personality.RELAXED):
    assert Personality.from_value(value, default=default) is default
