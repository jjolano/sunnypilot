"""Exhaustive property tests for the ACC/E2E/SCC evidence-admission gate."""
from __future__ import annotations

import itertools

import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import (
  EvidenceClass,
  LongitudinalMode,
  SourceToggles,
  admitted_evidence,
  is_admitted,
)

CURVE = {EvidenceClass.CURVE_VISION, EvidenceClass.CURVE_MAP}
SCC_EXTRA = {EvidenceClass.SPEED_LIMIT}
ALL_TOGGLES = [SourceToggles(v, m) for v, m in itertools.product((False, True), repeat=2)]


def test_cruise_and_lead_admitted_in_every_mode():
  for mode in LongitudinalMode:
    for sources in ALL_TOGGLES:
      adm = admitted_evidence(mode, sources)
      assert EvidenceClass.CRUISE in adm
      assert EvidenceClass.LEAD in adm


def test_acc_is_oem_like_cruise_plus_lead_only():
  for sources in ALL_TOGGLES:  # toggles must never widen ACC
    adm = admitted_evidence(LongitudinalMode.ACC, sources)
    assert adm == frozenset({EvidenceClass.CRUISE, EvidenceClass.LEAD})
    assert EvidenceClass.MODEL_STOP not in adm
    assert not (adm & CURVE)
    assert not (adm & SCC_EXTRA)


def test_e2e_admits_model_stop_but_not_scc_sources():
  for sources in ALL_TOGGLES:  # curve toggles are SCC-owned; they must not affect E2E
    adm = admitted_evidence(LongitudinalMode.E2E, sources)
    assert EvidenceClass.MODEL_STOP in adm        # the model drives incl. traffic control
    assert not (adm & CURVE)
    assert not (adm & SCC_EXTRA)


def test_scc_blends_model_stop_and_speed_limit():
  adm = admitted_evidence(LongitudinalMode.SCC, SourceToggles())
  assert EvidenceClass.MODEL_STOP in adm
  assert SCC_EXTRA <= adm


def test_scc_curve_sources_follow_their_toggles():
  assert EvidenceClass.CURVE_VISION not in admitted_evidence(LongitudinalMode.SCC, SourceToggles(False, False))
  assert EvidenceClass.CURVE_VISION in admitted_evidence(LongitudinalMode.SCC, SourceToggles(True, False))
  assert EvidenceClass.CURVE_MAP in admitted_evidence(LongitudinalMode.SCC, SourceToggles(False, True))
  both = admitted_evidence(LongitudinalMode.SCC, SourceToggles(True, True))
  assert CURVE <= both


def test_curve_toggles_never_leak_into_acc_or_e2e():
  for mode in (LongitudinalMode.ACC, LongitudinalMode.E2E):
    assert not (admitted_evidence(mode, SourceToggles(True, True)) & CURVE)


def test_is_admitted_matches_set():
  for mode in LongitudinalMode:
    for sources in ALL_TOGGLES:
      adm = admitted_evidence(mode, sources)
      for ev in EvidenceClass:
        assert is_admitted(mode, ev, sources) == (ev in adm)


@pytest.mark.parametrize("value,expected", [
  ("acc", LongitudinalMode.ACC), ("E2E", LongitudinalMode.E2E), ("scc", LongitudinalMode.SCC),
  (b"acc", LongitudinalMode.ACC), (b"e2e", LongitudinalMode.E2E), (b"scc", LongitudinalMode.SCC),
  ("0", LongitudinalMode.ACC), ("1", LongitudinalMode.E2E), ("2", LongitudinalMode.SCC),
  ("", LongitudinalMode.ACC), (None, LongitudinalMode.ACC), ("garbage", LongitudinalMode.ACC),
  (LongitudinalMode.SCC, LongitudinalMode.SCC),
])
def test_mode_from_value(value, expected):
  assert LongitudinalMode.from_value(value) == expected
