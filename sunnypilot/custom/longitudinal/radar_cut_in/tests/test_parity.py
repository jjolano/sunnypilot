"""Parity tests between canonical radar_cut_in override and v1 backup."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.sunnypilot.custom.longitudinal.radar_cut_in import override as canon
from openpilot.sunnypilot.selfdrive.controls.lib import cut_in_override_v1 as v1


NO_LEAD = {"status": False}
CONFIRMED_LEAD = {
  "status": True, "dRel": 10.0, "vRel": 0.0, "vLead": 10.0,
  "vLeadK": 10.0, "aLeadK": 0.0, "aLeadTau": 1.5,
  "modelProb": 0.9, "radar": True, "radarTrackId": 7,
  "yRel": 0.0, "fcw": False,
}


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


def test_constants_match():
  names = [
    "_MIN_CLOSING_SPEED", "_MAX_D_REL", "_MAX_TTC", "_MAX_Y_REL",
    "_MIN_V_LEAD", "_MIN_TRACK_CNT", "_MIN_V_EGO",
  ]
  for name in names:
    assert getattr(v1, name) == getattr(canon, name), name


def test_is_high_risk_cut_in_parity():
  cases = [
    (FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0, True),
    (FakeTrack(dRel=15.0, yRel=2.0, vRel=-3.0, vLead=8.0, cnt=3), 12.0, False),
    (FakeTrack(dRel=15.0, yRel=0.5, vRel=-1.0, vLead=8.0, cnt=3), 12.0, False),
    (FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=1.0, cnt=3), 12.0, False),
    (FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=1), 12.0, False),
    (FakeTrack(dRel=35.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0, False),
    (FakeTrack(dRel=None, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0, False),
    (FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 0.5, False),
  ]
  for track, v_ego, expected in cases:
    assert v1._is_high_risk_cut_in(track, v_ego) is expected
    assert canon._is_high_risk_cut_in(track, v_ego) is expected


def test_track_ttc_parity():
  cases = [
    FakeTrack(dRel=15.0, vRel=-3.0),
    FakeTrack(dRel=9.0, vRel=-3.0),
    FakeTrack(dRel=None, vRel=-3.0),
    FakeTrack(dRel=30.0, vRel=-2.0),
  ]
  for track in cases:
    v1_ttc = v1._track_ttc(track)
    canon_ttc = canon._track_ttc(track)
    if math.isinf(v1_ttc):
      assert math.isinf(canon_ttc)
    else:
      assert v1_ttc == pytest.approx(canon_ttc)


def test_no_override_confirmed_lead_identity():
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert v1.apply_cut_in_override(CONFIRMED_LEAD, {1: track}, v_ego=12.0,
                                  custom_longitudinal_enabled=True) is CONFIRMED_LEAD
  assert canon.apply_cut_in_override(CONFIRMED_LEAD, {1: track}, v_ego=12.0,
                                     custom_longitudinal_enabled=True) is CONFIRMED_LEAD


def test_no_override_custom_long_off_identity():
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert v1.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                  custom_longitudinal_enabled=False) is NO_LEAD
  assert canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                     custom_longitudinal_enabled=False) is NO_LEAD


def test_no_override_empty_tracks_identity():
  assert v1.apply_cut_in_override(NO_LEAD, {}, v_ego=12.0,
                                  custom_longitudinal_enabled=True) is NO_LEAD
  assert canon.apply_cut_in_override(NO_LEAD, {}, v_ego=12.0,
                                     custom_longitudinal_enabled=True) is NO_LEAD


def test_no_override_low_speed_identity():
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert v1.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=0.5,
                                  custom_longitudinal_enabled=True) is NO_LEAD
  assert canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=0.5,
                                     custom_longitudinal_enabled=True) is NO_LEAD


def test_no_override_malformed_track_identity():
  track = FakeTrack(dRel=None, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert v1.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                  custom_longitudinal_enabled=True) is NO_LEAD
  assert canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                     custom_longitudinal_enabled=True) is NO_LEAD


def test_no_override_params_exception_identity(monkeypatch):
  _stub_params(monkeypatch, side_effect=Exception("params error"))
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  assert canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0) is NO_LEAD


def test_override_promotes_and_results_match():
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)

  result_v1 = v1.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                       custom_longitudinal_enabled=True)
  result_canon = canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                             custom_longitudinal_enabled=True)

  for result in (result_v1, result_canon):
    assert result["status"] is True
    assert result["dRel"] == 15.0
    assert result["vRel"] == -3.0
    assert result["modelProb"] == 0.0
    assert result["radar"] is True
    assert result["radarTrackId"] == 1
  assert result_v1 == result_canon


def test_override_picks_most_dangerous_match():
  t1 = FakeTrack(identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  t2 = FakeTrack(identifier=2, dRel=9.0, yRel=0.3, vRel=-3.0, vLead=8.0, cnt=3)
  result_v1 = v1.apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0,
                                       custom_longitudinal_enabled=True)
  result_canon = canon.apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0,
                                             custom_longitudinal_enabled=True)
  assert result_v1["radarTrackId"] == 2
  assert result_canon["radarTrackId"] == 2
  assert result_v1 == result_canon


@pytest.mark.parametrize("path_y_rel,expected_promoted", [
  (-1.5, False),   # off path
  (0.0, True),     # on path
  (None, False),   # falls back to ego-frame yRel; yRel=1.5 is off path
  (lambda track: 0.0, True),
  (lambda track: -1.5, False),
])
def test_path_relative_y_parity(path_y_rel, expected_promoted):
  track = FakeTrack(dRel=15.0, yRel=1.5, vRel=-3.0, vLead=8.0, cnt=3)
  result_v1 = v1.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                       custom_longitudinal_enabled=True,
                                       path_y_rel=path_y_rel)
  result_canon = canon.apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0,
                                             custom_longitudinal_enabled=True,
                                             path_y_rel=path_y_rel)
  if expected_promoted:
    assert result_v1["status"] is True
    assert result_canon["status"] is True
    assert result_v1 == result_canon
  else:
    assert result_v1 is NO_LEAD
    assert result_canon is NO_LEAD


def test_path_relative_per_track_selected_candidate_match():
  t1 = FakeTrack(identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  t2 = FakeTrack(identifier=2, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  mapper = lambda track: -1.5 if track.identifier == 1 else 0.0  # noqa: E731
  result_v1 = v1.apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0,
                                       custom_longitudinal_enabled=True, path_y_rel=mapper)
  result_canon = canon.apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0,
                                             custom_longitudinal_enabled=True, path_y_rel=mapper)
  assert result_v1["radarTrackId"] == 2
  assert result_canon["radarTrackId"] == 2
  assert result_v1 == result_canon
