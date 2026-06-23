"""Tests for the cut-in override in radard."""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override import (
  apply_cut_in_override,
  _is_high_risk_cut_in,
  _track_ttc,
)


class FakeTrack:
  """Minimal Track stub matching the real Track interface."""
  def __init__(self, identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0,
               vLeadK=8.0, aLeadK=0.0, cnt=3):
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


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_promotes_high_risk_cut_in(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result["status"] is True
  assert result["dRel"] == 15.0
  assert result["vRel"] == -3.0
  assert result["modelProb"] == 0.0
  assert result["radar"] is True


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_does_not_replace_confirmed_lead(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(CONFIRMED_LEAD, {1: track}, v_ego=12.0)
  assert result is CONFIRMED_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_disabled_when_custom_longitudinal_off(MockParams):
  MockParams.return_value.get_bool.return_value = False
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_off_path_track(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=2.0, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_stationary_object(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=1.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_low_closing_speed(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-1.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_far_track(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=35.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_high_ttc(MockParams):
  MockParams.return_value.get_bool.return_value = True
  # dRel=30, vRel=-2.0 → TTC = 15s (too high)
  track = FakeTrack(dRel=30.0, yRel=0.5, vRel=-2.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_low_persistence(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=1)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_rejects_low_speed(MockParams):
  MockParams.return_value.get_bool.return_value = True
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=0.5)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_picks_most_dangerous_track(MockParams):
  MockParams.return_value.get_bool.return_value = True
  # Two candidates: track 1 has TTC=5s, track 2 has TTC=3s
  t1 = FakeTrack(identifier=1, dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  t2 = FakeTrack(identifier=2, dRel=9.0, yRel=0.3, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: t1, 2: t2}, v_ego=12.0)
  assert result["status"] is True
  assert result["radarTrackId"] == 2  # lower TTC


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_fail_closed_on_exception(MockParams):
  MockParams.return_value.get_bool.side_effect = Exception("params error")
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0)
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_true_skips_params_and_promotes(MockParams):
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=True)
  MockParams.assert_not_called()
  assert result["status"] is True
  assert result["radarTrackId"] == 1


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_false_returns_original_and_skips_params(MockParams):
  track = FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3)
  result = apply_cut_in_override(NO_LEAD, {1: track}, v_ego=12.0, custom_longitudinal_enabled=False)
  MockParams.assert_not_called()
  assert result is NO_LEAD


@patch("openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override.Params")
def test_override_empty_tracks(MockParams):
  MockParams.return_value.get_bool.return_value = True
  result = apply_cut_in_override(NO_LEAD, {}, v_ego=12.0)
  assert result is NO_LEAD


def test_is_high_risk_cut_in_gates():
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is True
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=2.0, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-1.0, vLead=8.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=1.0, cnt=3), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=15.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=1), 12.0) is False
  assert _is_high_risk_cut_in(FakeTrack(dRel=35.0, yRel=0.5, vRel=-3.0, vLead=8.0, cnt=3), 12.0) is False


def test_track_ttc():
  assert _track_ttc(FakeTrack(dRel=15.0, vRel=-3.0)) == 5.0
  assert _track_ttc(FakeTrack(dRel=9.0, vRel=-3.0)) == 3.0
