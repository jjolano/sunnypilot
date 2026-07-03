from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.stack import CustomLongitudinalStack
from openpilot.sunnypilot.custom.longitudinal.curve_speed_confidence import (
  CurveSpeedConfidenceInputs,
  predict_curve_speed_confidence,
)
from openpilot.sunnypilot.custom.longitudinal.cut_in_brake_assist import (
  MODE_APPLY as CUT_IN_MODE_APPLY,
  predict_cut_in_brake_assist,
)
from openpilot.sunnypilot.custom.longitudinal.standstill_release_confidence import predict_standstill_release_confidence
from openpilot.sunnypilot.custom.longitudinal.wiring import DEFAULT_ACCEL_LIMITS, build_stack_inputs


def state(**kwargs):
  risk = SimpleNamespace(ttc=kwargs.pop("ttc", 3.0), required_decel=kwargs.pop("required_decel", 0.5))
  defaults = dict(status=True, lead_idx=0, d_rel=20.0, path_y_rel=0.4, v_rel=-3.0,
                  confidence=0.8, risk_model=risk)
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def ctx(primary):
  return SimpleNamespace(behavior=primary, physical=None)


def test_cut_in_shadow_eligible_and_blocked_cases():
  r = predict_cut_in_brake_assist("shadow", ctx(state()), None, 15.0, long_active=True)
  assert r.eligible is True
  assert r.apply_supported is False
  assert r.proposed_cap < 0.0

  far = predict_cut_in_brake_assist("shadow", ctx(state(d_rel=80.0)), None, 15.0, long_active=True)
  assert far.eligible is False
  assert far.block_reason == "not_close"

  off = predict_cut_in_brake_assist("bad", ctx(state()), None, 15.0, long_active=True)
  assert off.mode == "off"
  assert off.eligible is False


def test_cut_in_shadow_blocked_when_long_inactive():
  r = predict_cut_in_brake_assist("shadow", ctx(state()), None, 15.0, long_active=False)
  assert r.mode == "shadow"
  assert r.effective_mode == "shadow"
  assert r.apply_supported is False
  assert r.eligible is False
  assert r.block_reason == "long_inactive"


def test_cut_in_eligibility_requires_stable_or_high_confidence():
  # Absent stable attribute, confidence >= 0.6 is required.
  ok = predict_cut_in_brake_assist("shadow", ctx(state(confidence=0.6)), None, 15.0, long_active=True)
  assert ok.eligible is True

  low_conf = predict_cut_in_brake_assist("shadow", ctx(state(confidence=0.5)), None, 15.0, long_active=True)
  assert low_conf.eligible is False
  assert low_conf.block_reason == "unstable_low_confidence"

  # Explicit stable=False with low confidence is still blocked; explicit stable=True passes.
  unstable = predict_cut_in_brake_assist("shadow", ctx(state(confidence=0.5, stable=False)), None, 15.0, long_active=True)
  assert unstable.eligible is False
  assert unstable.block_reason == "unstable_low_confidence"

  stable = predict_cut_in_brake_assist("shadow", ctx(state(confidence=0.5, stable=True)), None, 15.0, long_active=True)
  assert stable.eligible is True


def test_cut_in_not_closing_blocks_small_ttc_without_ignoring_ttc_plumbing():
  # Non-closing with valid short TTC is blocked purely on closing-speed gating.
  r = predict_cut_in_brake_assist("shadow", ctx(state(v_rel=1.5, ttc=1.0)), None, 15.0, long_active=True)
  assert r.eligible is False
  assert r.block_reason == "not_closing"
  assert r.ttc == pytest.approx(1.0)


def test_cut_in_apply_reports_supported_and_preserves_mode():
  r = predict_cut_in_brake_assist(CUT_IN_MODE_APPLY, ctx(state()), None, 15.0, long_active=True)
  assert r.mode == CUT_IN_MODE_APPLY
  assert r.effective_mode == CUT_IN_MODE_APPLY
  assert r.apply_supported is True
  assert r.eligible is True
  assert r.proposed_cap < 0.0


def test_cut_in_apply_invalid_becomes_off():
  r = predict_cut_in_brake_assist("aggressive", ctx(state()), None, 15.0, long_active=True)
  assert r.mode == "off"
  assert r.apply_supported is False


def test_cut_in_apply_blocks_nonfinite_path_y_rel():
  bad = predict_cut_in_brake_assist(CUT_IN_MODE_APPLY, ctx(state(path_y_rel=float('nan'))), None, 15.0, long_active=True)
  assert bad.eligible is False
  assert bad.block_reason == "not_near_path"
  far_path = predict_cut_in_brake_assist(CUT_IN_MODE_APPLY, ctx(state(path_y_rel=2.5)), None, 15.0, long_active=True)
  assert far_path.eligible is False
  assert far_path.block_reason == "not_near_path"


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


def test_curve_confidence_apply_conservative_reports_apply_supported():
  r = predict_curve_speed_confidence("apply_conservative", CurveSpeedConfidenceInputs(
    vision_active=True, vision_a_target=-0.5, vision_max_pred_lat_acc=1.4,
  ))
  assert r.mode == "apply_conservative"
  assert r.effective_mode == "apply_conservative"
  assert r.apply_supported is True
  assert r.eligible is True
  assert r.proposed_cap == pytest.approx(-0.5)


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


def test_standstill_release_gate_reports_apply_supported():
  r = predict_standstill_release_confidence(
    mode="gate", release_allowed=True, release_source="lead_pullaway", release_reason="lead_opening",
    release_a_target=0.25, lead_progress_allowed=True, lead_gap_excess=1.0,
    lead_shadow_active=False, alternate_threat_active=False, force_slow_decel=False,
    brake_pressed=False, gas_pressed=False, model_should_stop=False,
  )
  assert r.mode == "gate"
  assert r.effective_mode == "gate"
  assert r.apply_supported is True
  assert r.eligible is True


def test_stack_can_skip_shadow_payload_when_not_collected():
  lead = SimpleNamespace(status=True, dRel=28.0, vLead=17.0, vLeadK=17.0, vRel=-1.0,
                         aLeadK=-0.4, aLeadTau=1.5, yRel=0.0, radarTrackId=1, radar=True, modelProb=0.9)
  inp = build_stack_inputs(
    v_ego=17.0, a_ego=0.0, v_cruise=18.0, seed_a_target=0.2, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=lead, lead_two=None,
    scc_vision_active=False, scc_vision_a_target=0.0, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=False, sla_v_target=0.0, sla_a_target=0.0,
    mode=LongitudinalMode.ACC, personality=Personality.STANDARD, sources=SourceToggles(),
    long_active=True,
  )

  rich = CustomLongitudinalStack().update(inp, 0.05)
  lazy = CustomLongitudinalStack().update(inp, 0.05, collect_debug=False)

  assert lazy.a_target == pytest.approx(rich.a_target)
  assert lazy.should_stop == rich.should_stop
  assert lazy.decision.selected_intent == rich.decision.selected_intent
  assert lazy.debug == {}
