"""Tests for the cut-in override in radard."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override import (
  apply_cut_in_override,
  _is_high_risk_cut_in,
  _track_ttc,
)


class FakeTrack:
  """Minimal Track stub matching the real Track interface."""
  def __init__(self, identifier: int = 1, dRel: Any = 15.0, yRel: Any = 0.5,
               vRel: Any = -3.0, vLead: Any = 8.0, vLeadK: Any = 8.0,
               aLeadK: Any = 0.0, cnt: int = 3):
    self.identifier = identifier
    self.dRel = dRel
    self.yRel = yRel
    self.vRel = vRel
    self.vLead = vLead
    self.vLeadK = vLeadK
    self.aLeadK = aLeadK
    self.aLeadTau = SimpleNamespace(x=1.5)
    self.cnt = cnt

  def get_RadarState(self, model_prob=0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": False,
      "modelProb": float(model_prob),
      "radar": True,
      "radarTrackId": self.identifier,
    }


NO_LEAD = {"status": False}
CONFIRMED_LEAD = {"status": True, "dRel": 10.0, "vRel": 0.0, "vLead": 10.0,
                  "vLeadK": 10.0, "aLeadK": 0.0, "aLeadTau": 1.5,
                  "modelProb": 0.9, "radar": True, "radarTrackId": 7,
                  "yRel": 0.0, "fcw": False}



def _stub_params(monkeypatch, return_value=True, side_effect=None):
  class MockParams:
    def get_bool(self, _key):
      if side_effect is not None:
        raise side_effect
      return return_value
  monkeypatch.setattr(
    "openpilot.sunnypilot.custom.longitudinal.radar_cut_in.override.Params",
    MockParams,
  )


def _recording_params(monkeypatch, return_value=True):
  calls = []
  class MockParams:
    def __init__(self):
      calls.append(True)
    def get_bool(self, _key):
      return return_value
  monkeypatch.setattr(
    "openpilot.sunnypilot.custom.longitudinal.radar_cut_in.override.Params",
    MockParams,
  )
  return calls

def test_override_promotes_high_risk_cut_in(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result["status"] is True
  assert result["dRel"] == 15.0
  assert result["vRel"] == -3.0
  assert result["modelProb"] == 0.0
  assert result["radar"] is True


def test_override_does_not_replace_confirmed_lead(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(CONFIRMED_LEAD, {1: track}, v_ego=12.0)
  assert result is CONFIRMED_LEAD


def test_override_disabled_when_custom_longitudinal_off(monkeypatch):
  _stub_params(monkeypatch, return_value=False)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


def test_override_rejects_off_path_track(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=2.0, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_stationary_object(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=1.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_low_closing_speed(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-1.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_far_track(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=35.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_high_ttc(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  # dRel=30, vRel=-2.0 → TTC = 15s (too high)
  track = FakeTrack(dRel=30.0, yRel=0.5, vRel=-2.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_low_persistence(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=1)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_rejects_low_speed(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=0.5, research_actuation_allowed=True)
  assert result is NO_LEAD


def test_override_picks_most_dangerous_track(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  # Two candidates: track 1 has TTC=5s, track 2 has TTC=3s
  t1 = FakeTrack(identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  t2 = FakeTrack(identifier=2, dRel=9.0, yRel=0.3, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0, research_actuation_allowed=True)
  assert result["status"] is True
  assert result["radarTrackId"] == 2  # lower TTC


def test_override_fail_closed_on_exception(monkeypatch):
  _stub_params(monkeypatch, side_effect=Exception("params error"))
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


def test_override_true_skips_params_and_promotes(monkeypatch):
  calls = _recording_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=True,
                                 research_actuation_allowed=True)
  assert len(calls) == 0
  assert result["status"] is True
  assert result["radarTrackId"] == 1


def test_override_false_returns_original_and_skips_params(monkeypatch):
  calls = _recording_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=False)
  assert len(calls) == 0
  assert result is NO_LEAD


def test_override_empty_tracks(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  result = apply_cut_in_override(NO_LEAD, {}, v_ego=12.0)
  assert result is NO_LEAD


def test_override_fail_closed_on_malformed_track_fields(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=None, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)

  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)

  assert result is NO_LEAD


def test_is_high_risk_cut_in_gates():
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is True
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=2.0, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-1.0, vLead=8.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=1.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=1), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=35.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=None, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is False


def test_track_ttc():
  assert _track_ttc(FakeTrack(dRel=15.0, vRel=-3.0)) == 5.0
  assert _track_ttc(FakeTrack(dRel=9.0, vRel=-3.0)) == 3.0
  assert math.isinf(_track_ttc(FakeTrack(dRel=None, vRel=-3.0)))


def test_override_path_relative_rejects_ego_centerline_match(monkeypatch):
  # Track is on the ego centerline but well off the planned path; `path_y_rel` is the
  # path-relative deviation (yRel - path_y) and rejects what ego-frame alone would accept.
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=True,
                                 research_actuation_allowed=True, path_y_rel=-1.5)
  assert result is NO_LEAD


def test_override_path_relative_accepts_track_aligned_with_path(monkeypatch):
  # Track looks off the ego centerline but is actually aligned with the planned path.
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=1.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=True,
                                 research_actuation_allowed=True, path_y_rel=0.0)
  assert result["status"] is True
  assert result["radarTrackId"] == 1


def test_override_path_relative_fn_per_track(monkeypatch):
  # Callable mapper lets the override evaluate path-relative offset per track.
  _stub_params(monkeypatch, return_value=True)
  t1 = FakeTrack(identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  t2 = FakeTrack(identifier=2, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(
    NO_LEAD, {1: t1, 2: t2}, v_ego=12.0, custom_longitudinal_enabled=True,
    research_actuation_allowed=True,
    path_y_rel=lambda track: -1.5 if track.identifier == 1 else 0.0,
  )
  assert result["status"] is True
  assert result["radarTrackId"] == 2


def test_is_high_risk_cut_in_path_relative():
  # `path_y_rel` is path-relative deviation (yRel - path_y).
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert _is_high_risk_cut_in(track, 12.0, path_y_rel=None) is True
  assert _is_high_risk_cut_in(track, 12.0, path_y_rel=-1.5) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=1.5, vRel=-3.0, vLead=8.0, cnt=3),
                              12.0, path_y_rel=0.0) is True
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=1.5, vRel=-3.0, vLead=8.0, cnt=3),
                              12.0, path_y_rel=lambda track: 0.0) is True


def test_is_high_risk_cut_in_invalid_path_relative_fails_closed():
  track = FakeTrack(dRel=15.0, yRel=0.0, vRel=-3.0, vLead=8.0, cnt=3)
  bad_path: Any = "bad"
  assert _is_high_risk_cut_in(track, 12.0, path_y_rel=float("nan")) is False
  assert _is_high_risk_cut_in(track, 12.0, path_y_rel=float("inf")) is False
  assert _is_high_risk_cut_in(track, 12.0, path_y_rel=bad_path) is False


def test_override_blocked_without_research_actuation(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


def test_override_allowed_with_research_actuation(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, research_actuation_allowed=True)
  assert result["status"] is True
  assert result["radarTrackId"] == 1


def test_malformed_track_fields_skip_that_track_only(monkeypatch):
  _stub_params(monkeypatch, return_value=True)
  tracks = {1: FakeTrack(identifier=1, cnt="bogus"), 2: FakeTrack(identifier=2)}
  result = apply_cut_in_override(NO_LEAD, tracks, v_ego=12.0, research_actuation_allowed=True)
  assert result["status"] is True
  assert result["radarTrackId"] == 2


def test_raising_track_attribute_fails_closed_unchanged(monkeypatch):
  _stub_params(monkeypatch, return_value=True)

  class BrokenTrack:
    @property
    def cnt(self):
      raise RuntimeError("broken track")

  result = apply_cut_in_override(NO_LEAD, {1: BrokenTrack(), 2: FakeTrack()}, v_ego=12.0,
                                 research_actuation_allowed=True)
  assert result is NO_LEAD  # documented fail-closed: original lead dict unchanged


def test_raising_path_y_rel_callable_fails_closed_unchanged(monkeypatch):
  _stub_params(monkeypatch, return_value=True)

  def boom(_track):
    raise RuntimeError("path lookup failed")

  result = apply_cut_in_override(NO_LEAD, {1: FakeTrack()}, v_ego=12.0,
                                 research_actuation_allowed=True, path_y_rel=boom)
  assert result is NO_LEAD
