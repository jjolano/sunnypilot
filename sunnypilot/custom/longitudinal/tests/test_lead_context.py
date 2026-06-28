"""Tests for the lead context risk/progress model (faithful port).

Covers the pure risk/progress functions (deterministic, no engaged data needed) plus a
tracker smoke test. The model's *policy use* is validated downstream against the corpus.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.longitudinal import lead_context as lc
from openpilot.sunnypilot.custom.longitudinal import lead_context_v1 as lc_v1
from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LeadConfidenceState


def test_lead_prediction_fields_finite():
  p = lc.lead_prediction(d_rel=30.0, v_lead=10.0, a_lead=0.5, v_ego=20.0)
  assert isinstance(p, lc.LeadTrajectoryPrediction)
  assert p.valid is True
  assert len(p.x) == len(p.v) == len(p.a) > 0
  for seq in (p.x, p.v, p.a):
    assert all(math.isfinite(float(val)) for val in seq)
  # closing lead (v_ego 20 > v_lead 10): predicted relative gap shrinks over the horizon
  assert p.x[-1] <= p.x[0]


def test_lead_prediction_decays_a_lead():
  # accelerating lead: a_lead decays over the horizon and the predicted gap is less eager than
  # constant-a_lead kinematics would give (the predicted_gap_opening over-eagerness fix)
  p = lc.lead_prediction(d_rel=30.0, v_lead=15.0, a_lead=2.0, v_ego=15.0, a_lead_tau=1.5)
  assert all(p.a[i + 1] < p.a[i] for i in range(len(p.a) - 1))
  t_end = lc.LEAD_CONTEXT_PREVIEW_T[-1]
  constant_gap = 30.0 + 0.5 * 2.0 * t_end * t_end  # v_lead == v_ego, so only the accel term
  assert p.x[-1] < constant_gap


def test_required_decel_zero_when_not_closing():
  assert lc._required_decel(d_rel=20.0, v_rel=2.0) == 0.0   # opening
  assert lc._required_decel(d_rel=20.0, v_rel=0.0) == 0.0


def test_required_decel_monotonic_in_closing_rate():
  slow = lc._required_decel(d_rel=20.0, v_rel=-3.0)
  fast = lc._required_decel(d_rel=20.0, v_rel=-8.0)
  assert fast > slow > 0.0


def test_required_decel_monotonic_in_gap():
  near = lc._required_decel(d_rel=10.0, v_rel=-5.0)
  far = lc._required_decel(d_rel=40.0, v_rel=-5.0)
  assert near > far > 0.0


def test_ttc_finite_when_closing_large_otherwise():
  assert lc._ttc(d_rel=20.0, v_rel=-4.0) == 5.0
  assert lc._ttc(d_rel=20.0, v_rel=2.0) >= 1e3  # opening -> effectively infinite


def test_time_gap_and_progress_gap():
  assert lc._time_gap(d_rel=30.0, v_ego=15.0) == 2.0
  assert lc._desired_progress_gap(v_ego=20.0) > 0.0


def test_gap_shortage_and_excess_complement():
  v_ego = 20.0
  desired = lc._desired_progress_gap(v_ego)
  # shortage when closer than desired, excess when farther
  assert lc._gap_shortage(d_rel=desired - 5.0, v_ego=v_ego) > 0.0
  assert lc._gap_excess(d_rel=desired + 5.0, v_ego=v_ego) > 0.0
  assert lc._gap_shortage(d_rel=desired + 5.0, v_ego=v_ego) == 0.0


def test_on_path_score_in_unit_interval_and_decreasing():
  scores = [lc._on_path_score(y) for y in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)]
  assert all(0.0 <= s <= 1.0 for s in scores)
  assert scores[0] == 1.0
  for a, b in zip(scores, scores[1:], strict=False):
    assert b <= a + 1e-9


def test_risk_score_bounded():
  for v_rel in (-10.0, -3.0, 0.0, 3.0):
    rd = lc._required_decel(20.0, v_rel)
    ttc = lc._ttc(20.0, v_rel)
    tg = lc._time_gap(20.0, 20.0)
    s = lc._risk_score(d_rel=20.0, v_rel=v_rel, v_lead=10.0, v_ego=20.0, required_decel=rd, ttc=ttc, time_gap=tg)
    assert 0.0 <= s <= 1.0


def lead(d_rel=30.0, v_lead=12.0, a_lead=0.0, y_rel=0.0, status=True, track_id=3, model_prob=0.9):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead,
                         yRel=y_rel, radarTrackId=track_id, radar=True, modelProb=model_prob, aLeadTau=1.0)


def model_path(xs=(0.0, 30.0, 60.0), ys=(0.0, 1.5, 2.0)):
  return SimpleNamespace(position=SimpleNamespace(x=list(xs), y=list(ys)))


def test_tracker_update_returns_primary_context():
  t = lc.LeadContextTracker()
  ctx = None
  for _ in range(10):
    ctx = t.update(
      leads=(lead(d_rel=30.0), None),
      confidence_states=(LeadConfidenceState(status=True, stable=True, accel_blend=1.0), LeadConfidenceState()),
      v_ego=20.0, dt=0.05,
    )
  assert isinstance(ctx, lc.PrimaryLeadContext)


def test_tracker_interpolates_model_path_for_path_relative_y_shadow_signal():
  raw_tracker = lc.LeadContextTracker()
  model_tracker = lc.LeadContextTracker()
  confidence = (LeadConfidenceState(status=True, stable=True, accel_blend=1.0), LeadConfidenceState())

  raw_ctx = raw_tracker.update(
    leads=(lead(d_rel=45.0, y_rel=0.0), None), confidence_states=confidence,
    v_ego=20.0, dt=0.05,
  )
  model_ctx = model_tracker.update(
    leads=(lead(d_rel=45.0, y_rel=0.0), None), confidence_states=confidence,
    v_ego=20.0, dt=0.05, model_msg=model_path(),
  )

  assert raw_ctx.states[0].path_y_rel == pytest.approx(0.0)
  assert model_ctx.states[0].path_y_rel == pytest.approx(-1.75)


def _relevance_state(authority, *, gap_excess=0.0):
  return lc.LeadRelevanceState(
    lead_idx=0, status=True, shadow=False, stable=True, new_lead=False, flicker_guard_timer=0.0,
    track_id=3, d_rel=20.0, y_rel=0.0, path_y_rel=0.0, v_lead=8.0, v_rel=-2.0, model_prob=0.9,
    radar=True, ttc=10.0, required_decel=0.0, time_gap=2.0, on_path_score=1.0, risk_score=0.0,
    ghost_score=0.0, confidence=1.0, authority=authority, reason="test",
    progress_model=lc.LeadProgressModel(gap_excess=gap_excess),
  )


def _state(idx: int, authority: str, *, track_id: int | None = None, risk: float = 0.0,
           on_path: float = 1.0, d_rel: float = 20.0, v_lead: float = 10.0,
           v_rel: float = 0.0, ttc: float = math.inf, required_decel: float = 0.0,
           time_gap: float = 2.0, path_y_rel: float = 0.0, confidence: float = 0.9,
           stable: bool = True, new_lead: bool = False, flicker_guard_timer: float = 0.0,
           model_prob: float = 0.9, radar: bool = True, reason: str = "test"):
  tid = idx + 10 if track_id is None else track_id
  return lc.LeadRelevanceState(
    lead_idx=idx, status=True, shadow=False, stable=stable, new_lead=new_lead,
    flicker_guard_timer=flicker_guard_timer, track_id=tid, d_rel=d_rel, y_rel=path_y_rel,
    path_y_rel=path_y_rel, v_lead=v_lead, v_rel=v_rel, model_prob=model_prob,
    radar=radar, ttc=ttc, required_decel=required_decel, time_gap=time_gap,
    on_path_score=on_path, risk_score=risk, ghost_score=0.0, confidence=confidence,
    authority=authority, reason=reason,
    risk_model=lc.LeadRiskModel(required_decel=required_decel, ttc=ttc, time_gap=time_gap,
                                closing_speed=max(0.0, -v_rel), on_path_score=on_path,
                                model_prob=model_prob, radar_valid=radar),
    progress_model=lc.LeadProgressModel(gap_excess=5.0 if authority == lc.LEAD_AUTHORITY_PROGRESS_ALLOWED else 0.0,
                                        allowed=authority == lc.LEAD_AUTHORITY_PROGRESS_ALLOWED,
                                        reason="test_progress"),
  )


def _primary_ctx(behavior):
  return lc.PrimaryLeadContext(
    physical_idx=None, behavior_idx=0 if behavior is not None else None,
    physical=None, behavior=behavior, alternate_threat_active=False, shadow_active=False,
    reason="test", lead_progress_allowed=behavior is not None,
  )


def test_primary_context_surfaces_lead_gap_excess():
  # The stack reads lead_gap_excess to offer the lead-pullaway progress candidate; before this
  # accessor existed the stack's getattr fell back to 0.0 and pullaway could never fire.
  ctx = _primary_ctx(_relevance_state(lc.LEAD_AUTHORITY_PROGRESS_ALLOWED, gap_excess=5.0))
  assert ctx.lead_gap_excess == 5.0


def test_primary_context_lead_gap_excess_zero_with_no_lead():
  assert _primary_ctx(None).lead_gap_excess == 0.0


def test_physical_hysteresis_keeps_previous_without_material_threat():
  previous = _state(0, lc.LEAD_AUTHORITY_PHYSICAL, track_id=10, risk=0.40)
  challenger = _state(1, lc.LEAD_AUTHORITY_PHYSICAL, track_id=11, risk=0.42)
  ctx = lc.select_primary_lead_context(
    (previous, challenger), previous_physical_idx=0, previous_physical_track_id=10,
    previous_physical_dwell_s=0.10,
  )
  assert ctx.physical_idx == 0
  assert ctx.physical_switch_reason == "hysteresis_keep_previous"
  assert ctx.physical_switched is False


def test_physical_hysteresis_switches_for_immediate_threat():
  previous = _state(0, lc.LEAD_AUTHORITY_PHYSICAL, track_id=10, risk=0.40, ttc=8.0)
  challenger = _state(1, lc.LEAD_AUTHORITY_PHYSICAL, track_id=11, risk=0.80, ttc=2.5, required_decel=0.4)
  ctx = lc.select_primary_lead_context(
    (previous, challenger), previous_physical_idx=0, previous_physical_track_id=10,
    previous_physical_dwell_s=0.10,
  )
  assert ctx.physical_idx == 1
  assert ctx.physical_switch_reason == "immediate_threat"
  assert ctx.physical_switched is True


def test_physical_hysteresis_switches_after_dwell_elapsed():
  previous = _state(0, lc.LEAD_AUTHORITY_PHYSICAL, track_id=10, risk=0.40)
  challenger = _state(1, lc.LEAD_AUTHORITY_PHYSICAL, track_id=11, risk=0.42)
  ctx = lc.select_primary_lead_context(
    (previous, challenger), previous_physical_idx=0, previous_physical_track_id=10,
    previous_physical_dwell_s=lc.LEAD_CONTEXT_SWITCH_MIN_DWELL_S,
  )
  assert ctx.physical_idx == 1
  assert ctx.physical_switch_reason == "dwell_elapsed"
  assert ctx.physical_switched is True


def test_replacement_candidate_blocks_progress_suppress_only():
  exiting_behavior = _state(
    0, lc.LEAD_AUTHORITY_PROGRESS_ALLOWED, track_id=10, risk=0.0, on_path=0.3,
    path_y_rel=1.2, v_rel=1.0, time_gap=2.5, reason="stable_progress_authorized_lead",
  )
  replacement = _state(
    1, lc.LEAD_AUTHORITY_PHYSICAL, track_id=11, risk=0.55, on_path=1.0,
    d_rel=20.0, v_rel=-4.0, ttc=5.0, required_decel=0.35, time_gap=1.5,
    confidence=0.9, stable=True, reason="close_or_closing_lead",
  )
  ctx = lc.select_primary_lead_context((exiting_behavior, replacement))
  assert ctx.replacement_candidate.active is True
  assert ctx.replacement_candidate.candidate_idx == 1
  assert ctx.alternate_threat_active is True
  assert ctx.lead_progress_allowed is False
  assert ctx.lead_release_blocked_reason == "replacement_threat"
  debug = ctx.debug_dict()
  assert debug["lead_replacement_active"] is True
  assert debug["lead_replacement_candidate_idx"] == 1


def test_shadow_tracker_benign_far_dropout_is_normal():
  trk = lc.LeadShadowTracker(0)
  stable = LeadConfidenceState(status=True, stable=True, age=1.0)
  ld = lead(d_rel=60.0, v_lead=15.0, y_rel=0.0)
  trk.update(ld, stable, v_ego=15.0, dt=0.05, path_y_rel=0.0)
  shadow = trk.update(None, LeadConfidenceState(), v_ego=15.0, dt=0.05, path_y_rel=0.0)
  assert shadow.active is True
  assert shadow.duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_NORMAL_TIME)
  assert shadow.reason == "dropout"
  assert shadow.occlusion_risk == pytest.approx(0.0)


def test_shadow_tracker_cutout_exit_is_risk_duration_and_occlusion():
  trk = lc.LeadShadowTracker(0)
  stable = LeadConfidenceState(status=True, stable=True, age=1.0)
  # Move outward across several ticks to satisfy lateral-exit evidence.
  for path_y_rel in (0.0, 0.5, 0.8, 1.1, 1.3):
    ld = lead(d_rel=20.0, v_lead=10.0, y_rel=path_y_rel, a_lead=-2.0)
    trk.update(ld, stable, v_ego=15.0, dt=0.05, path_y_rel=path_y_rel)
  shadow = trk.update(None, LeadConfidenceState(), v_ego=15.0, dt=0.05, path_y_rel=1.3)
  assert shadow.active is True
  assert shadow.duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_RISK_TIME)
  assert shadow.reason == "cutout_exit"
  assert shadow.occlusion_risk == pytest.approx(1.0)
  assert shadow.stable_at_loss is True
  assert abs(shadow.path_y_rel_at_loss) >= lc.LEAD_CONTEXT_SHADOW_CUTOUT_EXIT_Y_REL


def test_shadow_tracker_close_stop_go_dropout_is_stop_go_duration():
  trk = lc.LeadShadowTracker(0)
  stable = LeadConfidenceState(status=True, stable=True, age=1.0)
  ld = lead(d_rel=8.0, v_lead=0.0, y_rel=0.0)
  trk.update(ld, stable, v_ego=0.0, dt=0.05, path_y_rel=0.0)
  shadow = trk.update(None, LeadConfidenceState(), v_ego=0.0, dt=0.05, path_y_rel=0.0)
  assert shadow.active is True
  assert shadow.duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_STOP_GO_TIME)
  assert shadow.reason == "stop_go_dropout"


def test_cutout_shadow_state_has_suppress_only_authority():
  t = lc.LeadContextTracker()
  stable = LeadConfidenceState(status=True, stable=True, age=1.0)
  ld = lead(d_rel=22.0, v_lead=10.0, y_rel=1.3, a_lead=-1.5)
  for _ in range(10):
    ctx = t.update(
      leads=(ld, None),
      confidence_states=(stable, LeadConfidenceState()),
      v_ego=15.0, dt=0.05,
    )
  ctx = t.update(
    leads=(None, None),
    confidence_states=(LeadConfidenceState(), LeadConfidenceState()),
    v_ego=15.0, dt=0.05,
  )
  shadow_state = ctx.states[0]
  assert shadow_state.shadow is True
  assert shadow_state.status is False
  assert shadow_state.authority == lc.LEAD_AUTHORITY_SUPPRESS_ONLY
  assert shadow_state.reason == "cutout_exit"
  assert shadow_state.shadow_occlusion_risk == pytest.approx(1.0)
  assert ctx.shadow_active is True
  debug = ctx.debug_dict()
  assert debug["shadow_lead_reason"] == "cutout_exit"
  assert debug["shadow_lead_duration"] == pytest.approx(lc.LEAD_CONTEXT_SHADOW_RISK_TIME)
  assert debug["shadow_lead_occlusion_risk"] == pytest.approx(1.0)
  assert debug["shadow_lead_stable_at_loss"] is True


# ---------------------------------------------------------------------------
# V1 / V2 parity characterization tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v1_v2_public_symbols() -> set[str]:
  v1_names = {name for name in dir(lc_v1) if not name.startswith("__")}
  v2_names = {name for name in dir(lc) if not name.startswith("__")}
  return v1_names & v2_names


def test_lead_context_v1_backup_exists_and_api_matches(v1_v2_public_symbols):
  """The v1 backup is an exact rollback reference and exposes the same public API."""
  required = {
    "LeadContextTracker", "LeadShadowTracker", "select_primary_lead_context",
    "lead_prediction", "LeadRelevanceState", "PrimaryLeadContext", "LeadRiskModel",
    "LeadProgressModel", "LeadTrajectoryPrediction", "LeadReplacementCandidate",
    "LeadShadowState",
    "LEAD_AUTHORITY_NONE", "LEAD_AUTHORITY_SUPPRESS_ONLY", "LEAD_AUTHORITY_PHYSICAL",
    "LEAD_AUTHORITY_PROGRESS_ALLOWED",
  }
  assert required.issubset(v1_v2_public_symbols)
  # Top-level helpers exercised by the existing tests must remain importable.
  for helper in (
    "finite_float", "_required_decel", "_ttc", "_time_gap", "_desired_progress_gap",
    "_gap_shortage", "_gap_excess", "_on_path_score", "_risk_score", "_confidence_score",
    "_ghost_score", "_path_relative_y", "_is_close_or_closing",
  ):
    assert hasattr(lc, helper)
    assert hasattr(lc_v1, helper)


def _float_eq(a: float, b: float, *, rel: float = 1e-9, abs_tol: float = 1e-12) -> bool:
  if math.isnan(a) and math.isnan(b):
    return True
  if math.isinf(a) and math.isinf(b):
    return (a > 0) == (b > 0)
  return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def _assert_value_eq(a: Any, b: Any, path: str = "root", *, rel: float = 1e-9, abs_tol: float = 1e-12) -> None:
  if a is b:
    return
  if a is None or b is None:
    assert a is b, f"{path}: {a!r} != {b!r}"
    return
  if isinstance(a, (str, bool, int)) and type(a) is type(b):
    assert a == b, f"{path}: {a!r} != {b!r}"
    return
  if isinstance(a, float) or isinstance(b, float):
    assert _float_eq(float(a), float(b), rel=rel, abs_tol=abs_tol), f"{path}: {a!r} != {b!r}"
    return
  if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
    a_fields = {f.name for f in dataclasses.fields(a)}
    b_fields = {f.name for f in dataclasses.fields(b)}
    assert a_fields == b_fields, f"{path}: dataclass field mismatch {a_fields} vs {b_fields}"
    for field_name in sorted(a_fields):
      _assert_value_eq(
        getattr(a, field_name), getattr(b, field_name),
        f"{path}.{field_name}", rel=rel, abs_tol=abs_tol,
      )
    return
  if isinstance(a, tuple) and isinstance(b, tuple):
    assert len(a) == len(b), f"{path}: tuple length {len(a)} != {len(b)}"
    for i, (av, bv) in enumerate(zip(a, b, strict=True)):
      _assert_value_eq(av, bv, f"{path}[{i}]", rel=rel, abs_tol=abs_tol)
    return
  assert a == b, f"{path}: {a!r} != {b!r}"


def _run_v1_v2_comparison(frames: list[dict[str, Any]]) -> tuple[lc.PrimaryLeadContext, lc_v1.PrimaryLeadContext]:
  """Steps two fresh trackers through the same frames and asserts parity."""
  tracker_v1 = lc_v1.LeadContextTracker()
  tracker_v2 = lc.LeadContextTracker()
  ctx_v1 = ctx_v2 = None
  last_frame: dict[str, Any] = {}
  for frame in frames:
    last_frame = frame
    ctx_v1 = tracker_v1.update(**frame)
    ctx_v2 = tracker_v2.update(**frame)
  assert ctx_v1 is not None
  assert ctx_v2 is not None

  # Top-level context fields and debug dict.
  _assert_value_eq(ctx_v2.physical_idx, ctx_v1.physical_idx, "physical_idx", abs_tol=0.0)
  _assert_value_eq(ctx_v2.behavior_idx, ctx_v1.behavior_idx, "behavior_idx", abs_tol=0.0)
  _assert_value_eq(ctx_v2.alternate_threat_active, ctx_v1.alternate_threat_active, "alternate_threat_active")
  _assert_value_eq(ctx_v2.shadow_active, ctx_v1.shadow_active, "shadow_active")
  _assert_value_eq(ctx_v2.reason, ctx_v1.reason, "reason")
  _assert_value_eq(ctx_v2.lead_progress_allowed, ctx_v1.lead_progress_allowed, "lead_progress_allowed")
  _assert_value_eq(ctx_v2.lead_release_blocked_reason, ctx_v1.lead_release_blocked_reason, "lead_release_blocked_reason")
  _assert_value_eq(ctx_v2.replacement_candidate, ctx_v1.replacement_candidate, "replacement_candidate")
  _assert_value_eq(ctx_v2.physical_switch_reason, ctx_v1.physical_switch_reason, "physical_switch_reason")
  _assert_value_eq(ctx_v2.physical_switched, ctx_v1.physical_switched, "physical_switched")
  _assert_value_eq(ctx_v2.debug_dict(), ctx_v1.debug_dict(), "debug_dict", rel=1e-7)

  # Per-state parity.
  assert len(ctx_v2.states) == len(ctx_v1.states), "state count mismatch"
  for i, (s2, s1) in enumerate(zip(ctx_v2.states, ctx_v1.states, strict=True)):
    _assert_value_eq(s2, s1, f"state[{i}]", rel=1e-7)

  # physical_lead_data / behavior_lead_data identity behavior is unchanged.
  leads = last_frame.get("leads", (None, None))
  _assert_value_eq(ctx_v2.physical_lead_data(leads), ctx_v1.physical_lead_data(leads), "physical_lead_data")
  _assert_value_eq(ctx_v2.behavior_lead_data(leads), ctx_v1.behavior_lead_data(leads), "behavior_lead_data")

  return ctx_v2, ctx_v1


def _stable_confidence(**kwargs):
  return LeadConfidenceState(status=True, stable=True, age=1.0, **kwargs)


def _new_confidence():
  return LeadConfidenceState(status=True, stable=False, new_lead=True, guard_timer=0.25)


def _flicker_confidence():
  return LeadConfidenceState(status=True, stable=False, flicker_guard_timer=0.15)


def test_v1v2_no_lead():
  frames = [
    {"leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ]
  _run_v1_v2_comparison(frames)


def test_v1v2_far_stable_lead():
  frames = [
    {"leads": (lead(d_rel=100.0, v_lead=20.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.physical is not None
  assert ctx2.physical.authority == lc.LEAD_AUTHORITY_PROGRESS_ALLOWED


def test_v1v2_medium_far_slowing_lead():
  frames = [
    {"leads": (lead(d_rel=55.0, v_lead=18.0, a_lead=-0.8), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  _run_v1_v2_comparison(frames)


def test_v1v2_close_closing_lead():
  frames = [
    {"leads": (lead(d_rel=15.0, v_lead=5.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.physical is not None
  assert ctx2.physical.authority == lc.LEAD_AUTHORITY_PHYSICAL


def test_v1v2_stopped_lead():
  frames = [
    {"leads": (lead(d_rel=8.0, v_lead=0.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 0.0, "dt": 0.05},
  ] * 5
  _run_v1_v2_comparison(frames)


def test_v1v2_lead_pullaway():
  frames = [
    {"leads": (lead(d_rel=25.0, v_lead=22.0, a_lead=0.8), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  _run_v1_v2_comparison(frames)


def test_v1v2_new_lead_guard_blocks_progress():
  frames = [
    {"leads": (lead(d_rel=30.0, v_lead=12.0), None),
     "confidence_states": (_new_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.physical is not None
  assert ctx2.physical.new_lead is True


def test_v1v2_flicker_guard():
  frames = [
    {"leads": (lead(d_rel=30.0, v_lead=12.0), None),
     "confidence_states": (_flicker_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.physical is not None
  assert ctx2.physical.reason == "flicker_guard_suppress_only"


def test_v1v2_lead_replacement():
  # First establish a stable lead in slot 0 that is laterally exiting.
  frames = [
    {"leads": (lead(d_rel=35.0, v_lead=15.0, y_rel=0.0, track_id=10),
               lead(d_rel=40.0, v_lead=15.0, y_rel=0.0, track_id=11)),
     "confidence_states": (_stable_confidence(), _stable_confidence()),
     "v_ego": 15.0, "dt": 0.05},
  ] * 6
  # Then move slot 0 outward while slot 1 closes the gap.
  frames.extend([
    {"leads": (lead(d_rel=35.0, v_lead=15.0, y_rel=1.3, track_id=10),
               lead(d_rel=20.0, v_lead=10.0, y_rel=0.0, track_id=11)),
     "confidence_states": (_stable_confidence(), _stable_confidence()),
     "v_ego": 15.0, "dt": 0.05},
  ] * 5)
  _run_v1_v2_comparison(frames)


def test_v1v2_shadow_dropout():
  frames = [
    {"leads": (lead(d_rel=60.0, v_lead=15.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 15.0, "dt": 0.05},
  ] * 5
  # Lead drops out; shadow should become active with normal duration.
  frames.append(
    {"leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
     "v_ego": 15.0, "dt": 0.05},
  )
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.shadow_active is True
  assert ctx2.states[0].shadow is True
  assert ctx2.states[0].shadow_duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_NORMAL_TIME)


def test_v1v2_shadow_cutout_exit():
  tracker_v1 = lc_v1.LeadContextTracker()
  tracker_v2 = lc.LeadContextTracker()
  stable = _stable_confidence()
  # Move lead outward across several ticks to satisfy lateral-exit evidence.
  path_y_values = (0.0, 0.5, 0.8, 1.1, 1.3)
  ctx_v1 = ctx_v2 = None
  for path_y_rel in path_y_values:
    frame = {
      "leads": (lead(d_rel=20.0, v_lead=10.0, y_rel=path_y_rel, a_lead=-2.0), None),
      "confidence_states": (stable, LeadConfidenceState()),
      "v_ego": 15.0, "dt": 0.05, "model_msg": None,
    }
    ctx_v1 = tracker_v1.update(**frame)
    ctx_v2 = tracker_v2.update(**frame)

  # Dropout frame.
  dropout_frame = {
    "leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
    "v_ego": 15.0, "dt": 0.05, "model_msg": None,
  }
  ctx_v1 = tracker_v1.update(**dropout_frame)
  ctx_v2 = tracker_v2.update(**dropout_frame)

  _assert_value_eq(ctx_v2.states[0], ctx_v1.states[0], "cutout_state[0]")
  assert ctx_v2.states[0].shadow is True
  assert ctx_v2.states[0].reason == "cutout_exit"
  assert ctx_v2.states[0].shadow_occlusion_risk == pytest.approx(1.0)
  assert ctx_v2.states[0].shadow_duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_RISK_TIME)


def test_v1v2_shadow_model_path_relative_y():
  frames = [
    {"leads": (lead(d_rel=45.0, y_rel=0.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05, "model_msg": model_path()},
  ] * 3
  frames.append(
    {"leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05, "model_msg": model_path()},
  )
  ctx2, _ = _run_v1_v2_comparison(frames)
  assert ctx2.states[0].shadow is True
  assert ctx2.states[0].path_y_rel < -1.7
  assert ctx2.states[0].risk_model.path_y_rel == pytest.approx(ctx2.states[0].path_y_rel)


def test_v1v2_false_positive_release_uses_last_real_path_y_after_dropout():
  frames = [
    {"leads": (lead(d_rel=80.0, v_lead=20.0, y_rel=2.0, model_prob=0.4), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 3
  frames.extend([
    {"leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 30)
  frames.extend([
    {"leads": (lead(d_rel=80.0, v_lead=20.0, y_rel=1.7, model_prob=0.4), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ] * 5)
  _run_v1_v2_comparison(frames)


def test_v1v2_shadow_stop_go():
  frames = [
    {"leads": (lead(d_rel=8.0, v_lead=0.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 0.0, "dt": 0.05},
  ] * 3
  frames.append(
    {"leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
     "v_ego": 0.0, "dt": 0.05},
  )
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.states[0].shadow is True
  assert ctx2.states[0].reason == "stop_go_dropout"
  assert ctx2.states[0].shadow_duration == pytest.approx(lc.LEAD_CONTEXT_SHADOW_STOP_GO_TIME)


def test_v1v2_model_path_relative_y():
  raw_frames = [
    {"leads": (lead(d_rel=45.0, y_rel=0.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05},
  ]
  model_frames = [
    {"leads": (lead(d_rel=45.0, y_rel=0.0), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 20.0, "dt": 0.05, "model_msg": model_path()},
  ]
  # Raw path_y_rel should be ~0; model-relative should be offset by interpolated path y.
  _run_v1_v2_comparison(raw_frames)
  _run_v1_v2_comparison(model_frames)


def test_v1v2_physical_hysteresis_keeps_previous():
  # Warm up previous physical memory on idx 0.
  frames = [
    {"leads": (lead(d_rel=20.0, v_lead=10.0, track_id=10), None),
     "confidence_states": (_stable_confidence(), LeadConfidenceState()),
     "v_ego": 15.0, "dt": 0.05},
  ] * 4
  # Present a slightly stronger challenger that should be held by hysteresis.
  frames.append(
    {"leads": (lead(d_rel=20.0, v_lead=10.0, track_id=10),
               lead(d_rel=18.0, v_lead=9.5, track_id=11)),
     "confidence_states": (_stable_confidence(), _stable_confidence()),
     "v_ego": 15.0, "dt": 0.05},
  )
  ctx2, ctx1 = _run_v1_v2_comparison(frames)
  assert ctx2.physical_switch_reason == ctx1.physical_switch_reason
  assert ctx2.physical_idx == ctx1.physical_idx


def test_v1v2_dominant_hints():
  frames = [
    {"leads": (lead(d_rel=25.0, v_lead=12.0, track_id=10),
               lead(d_rel=22.0, v_lead=11.0, track_id=11)),
     "confidence_states": (_stable_confidence(), _stable_confidence()),
     "v_ego": 15.0, "dt": 0.05, "lead_dominant_idx": 1},
  ] * 5
  _run_v1_v2_comparison(frames)


def test_v1v2_reset_clears_state():
  tracker_v1 = lc_v1.LeadContextTracker()
  tracker_v2 = lc.LeadContextTracker()
  frame = {
    "leads": (lead(d_rel=30.0, v_lead=12.0), None),
    "confidence_states": (_stable_confidence(), LeadConfidenceState()),
    "v_ego": 20.0, "dt": 0.05,
  }
  ctx_v1 = tracker_v1.update(**frame)
  ctx_v2 = tracker_v2.update(**frame)
  _assert_value_eq(ctx_v2, ctx_v1, "before_reset")

  reset_frame = {
    "leads": (None, None), "confidence_states": (LeadConfidenceState(), LeadConfidenceState()),
    "v_ego": 20.0, "dt": 0.05, "reset_state": True,
  }
  ctx_v1 = tracker_v1.update(**reset_frame)
  ctx_v2 = tracker_v2.update(**reset_frame)
  _assert_value_eq(ctx_v2, ctx_v1, "after_reset")
  assert ctx_v2.physical is None
  assert ctx_v2.shadow_active is False
