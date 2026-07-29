"""Tests for the cut-out lead release MPC-input filter."""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.custom.longitudinal.cut_out_release import (
  CUT_OUT_PERSIST_S,
  CutOutLeadRelease,
)

DT = 0.05


def _rs(status=True, d_rel=35.0, y_rel=0.0, v_rel=0.0, track_id=7):
  lead = SimpleNamespace(status=status, dRel=d_rel, yRel=y_rel, vRel=v_rel, radarTrackId=track_id)
  return SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False))


def _model_path(y_at_lead=0.0):
  return SimpleNamespace(position=SimpleNamespace(x=[0.0, 35.0, 70.0], y=[0.0, y_at_lead, 2.0 * y_at_lead]))


def _run(filt, rs, n, v_ego=13.0, **gates):
  gates = {
    "long_active": True,
    "custom_long_enabled": True,
    "research_actuation_allowed": True,
    "mode": "apply",
    "model_msg": _model_path(),
    **gates,
  }
  out = rs
  for _ in range(n):
    out = filt.filtered(rs, v_ego, DT, **gates)
  return out


def test_on_path_lead_never_suppressed():
  filt = CutOutLeadRelease()
  out = _run(filt, _rs(y_rel=0.3), 100)
  assert out.leadOne.status
  assert not filt.suppressing


def test_sustained_lateral_exit_suppresses_after_persistence():
  filt = CutOutLeadRelease()
  rs = _rs(y_rel=2.6, d_rel=35.0, v_rel=-0.5)
  steps = int(CUT_OUT_PERSIST_S / DT) + 2
  out = _run(filt, rs, steps)
  assert filt.suppressing
  assert not out.leadOne.status
  # leadTwo passes through untouched
  assert out.leadTwo is rs.leadTwo


def test_brief_flicker_below_persistence_keeps_lead():
  filt = CutOutLeadRelease()
  out = _run(filt, _rs(y_rel=2.6), 4)
  assert out.leadOne.status


def test_cut_in_moving_inward_is_never_suppressed():
  filt = CutOutLeadRelease()
  # |yRel| shrinking (cut-in) resets outward tracking each frame.
  for y in [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]:
    out = filt.filtered(_rs(y_rel=y), 13.0, DT, long_active=True,
                        custom_long_enabled=True, research_actuation_allowed=True,
                        mode="apply", model_msg=_model_path())
  assert out.leadOne.status
  assert not filt.suppressing


def test_threatening_lead_is_never_suppressed():
  filt = CutOutLeadRelease()
  # Off to the side but close and closing fast -> TTC/distance guards hold the lead.
  out = _run(filt, _rs(y_rel=2.6, d_rel=12.0, v_rel=-4.0), 100)
  assert out.leadOne.status
  assert filt.block_reason == "threat"


def test_gates_off_pass_through_and_reset():
  filt = CutOutLeadRelease()
  _run(filt, _rs(y_rel=3.5), 100)
  assert filt.suppressing
  out = _run(filt, _rs(y_rel=3.5), 1, research_actuation_allowed=False)
  assert out.leadOne.status
  assert not filt.suppressing


def test_track_id_change_resets_persistence():
  filt = CutOutLeadRelease()
  _run(filt, _rs(y_rel=2.6, track_id=7), 6)
  out = _run(filt, _rs(y_rel=2.6, track_id=8), 4)
  assert out.leadOne.status


def test_vision_only_unknown_track_id_never_accumulates_suppression_state():
  filt = CutOutLeadRelease()
  out = _run(filt, _rs(y_rel=3.0, track_id=-1), 100)
  assert out.leadOne.status
  assert not filt.suppressing
  assert filt.block_reason == "track_unknown"


def test_curved_road_lead_uses_path_relative_lateral_position():
  filt = CutOutLeadRelease()
  rs = _rs(y_rel=2.6, d_rel=35.0)
  out = _run(filt, rs, 100, model_msg=_model_path(y_at_lead=-2.6))
  assert out.leadOne.status
  assert not filt.suppressing


def test_off_mode_never_suppresses_even_with_global_research_gate():
  filt = CutOutLeadRelease()
  out = _run(filt, _rs(y_rel=3.0), 100, mode="off")
  assert out.leadOne.status
  assert not filt.suppressing


def test_low_speed_stop_and_go_untouched():
  filt = CutOutLeadRelease()
  out = _run(filt, _rs(y_rel=3.0, d_rel=20.0), 100, v_ego=1.5)
  assert out.leadOne.status
