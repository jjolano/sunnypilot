"""Focused runtime tests for DeparturePredictionMode evidence handling."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.departure_prediction import (
  MODE_APPLY,
  MODE_OFF,
  MODE_SHADOW,
  build_departure_prediction_evidence,
  effective_mode,
  sanitize_mode,
)


def _lead(*, track_id: int = 7, **overrides):
  values = dict(
    status=True,
    dRel=6.5,
    vLead=0.8,
    vLeadK=0.8,
    vRel=0.5,
    aLeadK=0.5,
    radarTrackId=track_id,
  )
  values.update(overrides)
  return SimpleNamespace(**values)


def test_mode_sanitizes_and_research_gate_is_fail_closed():
  assert sanitize_mode(b"SHADOW") == MODE_SHADOW
  assert sanitize_mode("invalid") == MODE_OFF
  assert sanitize_mode(None) == MODE_OFF
  assert effective_mode(MODE_APPLY, False) == (MODE_SHADOW, False)
  assert effective_mode(MODE_APPLY, True) == (MODE_APPLY, True)


def test_evidence_uses_the_existing_lead_prediction_and_fails_closed():
  state = SimpleNamespace(
    lead_idx=0, track_id=7, stable=True, radar=True,
    prediction=SimpleNamespace(x=(6.5, 6.7, 7.0), v=(0.8, 0.8, 0.8), a=(0.5, 0.4, 0.3), valid=True),
  )
  context = SimpleNamespace(shadow_active=False, alternate_threat_active=False, lead_progress_allowed=True)

  result = build_departure_prediction_evidence(
    mode=MODE_APPLY, research_actuation_allowed=True,
    physical_state=state, physical_lead=_lead(), lead_context=context,
  )

  assert result.eligible is True
  assert result.predicted_gap_1s == pytest.approx(7.0)
  assert result.predicted_gap_growth_1s == pytest.approx(0.5)
  assert result.predicted_gap_delta == pytest.approx(0.5)
  assert result.track_id == 7

  for bad_prediction in (
    None,
    SimpleNamespace(x=(6.5, math.nan), v=(0.8, 0.8), a=(0.5, 0.4), valid=True),
  ):
    blocked = build_departure_prediction_evidence(
      mode=MODE_SHADOW, research_actuation_allowed=False,
      physical_state=SimpleNamespace(**{**vars(state), "prediction": bad_prediction}),
      physical_lead=_lead(), lead_context=context,
    )
    assert blocked.eligible is False
    assert blocked.block_reason == "invalid_prediction"


@pytest.mark.parametrize(
  ("state", "lead", "context", "reason"),
  [
    (dict(stable=False), {}, {}, "unstable_lead"),
    (dict(radar=False), {}, {}, "unknown_radar_lead"),
    (dict(track_id=8), {}, {}, "unknown_radar_lead"),
    ({}, {"status": False}, {}, "no_physical_lead"),
    ({}, {}, {"shadow_active": True}, "shadow_threat"),
    ({}, {}, {"alternate_threat_active": True}, "alternate_threat"),
    ({}, {}, {"lead_progress_allowed": False}, "progress_not_authorized"),
    ({}, {"aLeadK": 0.1}, {}, "lead_accel_too_low"),
    ({}, {"vRel": -0.1}, {}, "lead_closing"),
    ({}, {}, {}, "insufficient_predicted_growth"),
  ],
)
def test_ineligible_evidence_reasons_are_specific(state, lead, context, reason):
  base_state = dict(
    lead_idx=0, track_id=7, stable=True, radar=True,
    prediction=SimpleNamespace(x=(6.5, 7.0), v=(0.8, 0.8), a=(0.5, 0.4), valid=True),
  )
  base_lead = dict(track_id=7)
  base_context = dict(shadow_active=False, alternate_threat_active=False, lead_progress_allowed=True)
  base_state.update(state)
  if reason == "insufficient_predicted_growth":
    base_state["prediction"] = SimpleNamespace(x=(6.5, 6.6), v=(0.8, 0.8), a=(0.5, 0.4), valid=True)
  base_lead.update(lead)
  base_context.update(context)
  result = build_departure_prediction_evidence(
    mode=MODE_SHADOW, research_actuation_allowed=False,
    physical_state=SimpleNamespace(**base_state),
    physical_lead=_lead(**base_lead),
    lead_context=SimpleNamespace(**base_context),
  )
  assert result.eligible is False
  assert result.block_reason == reason
