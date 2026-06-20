from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import (
  CurveSpeedConfidenceInputs,
  predict_curve_speed_confidence,
)
from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import predict_cut_in_brake_assist
from openpilot.sunnypilot.custom.longitudinal.standstill_release_confidence import predict_standstill_release_confidence


def state(**kwargs):
  risk = SimpleNamespace(ttc=kwargs.pop("ttc", 3.0), required_decel=kwargs.pop("required_decel", 0.5))
  defaults = dict(status=True, lead_idx=0, d_rel=20.0, path_y_rel=0.4, v_rel=-3.0,
                  confidence=0.8, risk_model=risk)
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def ctx(primary):
  return SimpleNamespace(behavior=primary, physical=None)


def test_cut_in_shadow_eligible_and_blocked_cases():
  r = predict_cut_in_brake_assist("shadow", ctx(state()), None, 15.0)
  assert r.eligible is True
  assert r.apply_supported is False
  assert r.proposed_cap < 0.0

  far = predict_cut_in_brake_assist("shadow", ctx(state(d_rel=80.0)), None, 15.0)
  assert far.eligible is False
  assert far.block_reason == "not_close"

  off = predict_cut_in_brake_assist("bad", ctx(state()), None, 15.0)
  assert off.mode == "off"
  assert off.eligible is False


def test_curve_confidence_shadow_uses_negative_active_caps_only():
  r = predict_curve_speed_confidence("shadow", CurveSpeedConfidenceInputs(
    vision_active=True, vision_a_target=-0.5, vision_max_pred_lat_acc=1.4,
  ))
  assert r.eligible is True
  assert r.apply_supported is False
  assert r.proposed_cap == pytest.approx(-0.5)
  assert r.confidence >= 0.7

  inactive = predict_curve_speed_confidence("shadow", CurveSpeedConfidenceInputs())
  assert inactive.eligible is False
  assert inactive.block_reason == "inactive"


def test_standstill_release_confidence_scores_existing_release_only():
  r = predict_standstill_release_confidence(
    mode="shadow", release_allowed=True, release_source="lead_pullaway", release_reason="lead_opening",
    release_a_target=0.25, lead_progress_allowed=True, lead_gap_excess=1.0,
    lead_shadow_active=False, alternate_threat_active=False, force_slow_decel=False,
    brake_pressed=False, gas_pressed=False, model_should_stop=False,
  )
  assert r.eligible is True
  assert r.apply_supported is False
  assert r.release_allowed is True
  assert r.release_a_target == pytest.approx(0.25)

  blocked = predict_standstill_release_confidence(
    mode="shadow", release_allowed=False, release_source="", release_reason="", release_a_target=0.0,
    lead_progress_allowed=False, lead_gap_excess=0.0, lead_shadow_active=False,
    alternate_threat_active=False, force_slow_decel=False, brake_pressed=False,
    gas_pressed=False, model_should_stop=False,
  )
  assert blocked.eligible is False
  assert blocked.block_reason == "release_not_allowed"
