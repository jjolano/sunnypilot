"""Shadow-only lead path clearance predictor tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState
from openpilot.sunnypilot.custom.longitudinal.lead_context import LeadContextTracker
from openpilot.sunnypilot.custom.longitudinal.lead_path_clearance import (
  MODE_APPLY,
  MODE_OFF,
  MODE_SHADOW,
  predict_lead_path_clearance,
)

DT = 0.05


def lead(d_rel=55.0, v_lead=10.0, v_rel=-5.0, y_rel=0.4, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
                         aLeadK=0.0, yRel=y_rel, radarTrackId=4, radar=True,
                         modelProb=0.9, aLeadTau=1.0)


def model(leads_y=(-0.4, -1.0, -1.8, -2.0), leads_t=(0.0, 1.0, 2.0, 3.0),
          leads_x=(55.0, 60.0, 65.0, 70.0), y_std=0.2, prob=0.9):
  # modelV2.leadsV3.y uses the opposite sign from radarState.yRel; the predictor
  # flips it to planner convention. Negative model y below means positive path_y_rel.
  return SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 40.0, 80.0], y=[0.0, 0.0, 0.0]),
    leadsV3=[SimpleNamespace(
      x=list(leads_x), y=list(leads_y), t=list(leads_t), prob=prob,
      xStd=[0.5] * len(leads_x), yStd=[y_std] * len(leads_x),
    )],
  )


def context(ld=None, mdl=None):
  tracker = LeadContextTracker()
  confidence = (LeadConfidenceState(status=True, stable=True, speed_trusted=True, radar=True,
                                    age=1.0, accel_blend=1.0, track_id=4), LeadConfidenceState())
  return tracker.update((lead() if ld is None else ld, None), confidence, v_ego=15.0, dt=DT, model_msg=model() if mdl is None else mdl)


def test_off_mode_is_inert_debug_only():
  r = predict_lead_path_clearance(MODE_OFF, context(), model(), v_ego=15.0)
  assert r.enabled is False
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "mode_off"


def test_exiting_lead_before_conflict_is_shadow_eligible():
  r = predict_lead_path_clearance(MODE_SHADOW, context(), model(), v_ego=15.0)
  assert r.enabled is True
  assert r.apply_supported is False
  assert r.shadow_eligible is True
  assert r.shadow_blocked_reason == ""
  assert r.path_y_rel == pytest.approx(0.4)
  assert r.lateral_velocity > 0.0
  assert r.t_clear + 0.7 < r.t_conflict


def test_apply_mode_is_downgraded_to_shadow_only():
  r = predict_lead_path_clearance(MODE_APPLY, context(), model(), v_ego=15.0)
  assert r.effective_mode == MODE_SHADOW
  assert r.apply_supported is False
  assert r.shadow_eligible is True


def test_clearance_too_late_is_blocked():
  late_model = model(leads_y=(-0.4, -0.8, -1.2, -1.8), leads_t=(0.0, 3.0, 6.0, 9.0),
                     leads_x=(45.0, 50.0, 55.0, 60.0))
  ld = lead(d_rel=45.0, v_rel=-5.0)
  r = predict_lead_path_clearance(MODE_SHADOW, context(ld, late_model), late_model, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "clears_too_late"


def test_close_ttc_blocks_even_if_model_predicts_clearance():
  ld = lead(d_rel=20.0, v_rel=-8.0)
  mdl = model(leads_x=(20.0, 25.0, 30.0, 35.0))
  r = predict_lead_path_clearance(MODE_SHADOW, context(ld, mdl), mdl, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "close_ttc"


def test_noisy_model_lateral_prediction_blocks():
  noisy = model(y_std=2.0)
  r = predict_lead_path_clearance(MODE_SHADOW, context(mdl=noisy), noisy, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "model_uncertain"


def test_missing_model_uncertainty_blocks():
  uncertain = model()
  delattr(uncertain.leadsV3[0], "xStd")
  delattr(uncertain.leadsV3[0], "yStd")
  r = predict_lead_path_clearance(MODE_SHADOW, context(mdl=uncertain), uncertain, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "model_uncertain"


def test_missing_model_lead_trajectory_blocks():
  missing = SimpleNamespace(position=SimpleNamespace(x=[0.0, 40.0, 80.0], y=[0.0, 0.0, 0.0]), leadsV3=[])
  r = predict_lead_path_clearance(MODE_SHADOW, context(mdl=missing), missing, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "no_model_lead_trajectory"


def test_model_y_sign_must_match_radarstate_convention():
  wrong_sign = model(leads_y=(0.4, 1.0, 1.8, 2.0))
  r = predict_lead_path_clearance(MODE_SHADOW, context(mdl=wrong_sign), wrong_sign, v_ego=15.0)
  assert r.shadow_eligible is False
  assert r.shadow_blocked_reason == "model_lateral_disagreement"
