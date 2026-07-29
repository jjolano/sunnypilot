from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.radard import (
  _clean_lead_prob, _track_path_relative_y, get_RadarState_from_vision, get_lead,
  match_vision_to_track, KalmanParams, Track,
)


def model_lead(x=20.0, y=0.0, v=12.0, a=0.0, x_std=1.0, y_std=1.0, v_std=1.0):
  return SimpleNamespace(
    x=[x], y=[y], v=[v], a=[a],
    xStd=[x_std], yStd=[y_std], vStd=[v_std],
  )


def car_params():
  return SimpleNamespace(brand="toyota", flags=0), SimpleNamespace(flags=0)


def radar_track(identifier=1, d_rel=20.0, y_rel=0.0, v_rel=1.0, v_lead=11.0):
  params = KalmanParams(0.02)
  track = Track(identifier, v_lead, params)
  track.update(d_rel, y_rel, v_rel, v_lead)
  return track


def test_vision_only_lead_requires_finite_model_fields():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=float("nan")), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_vision_only_lead_requires_present_model_fields():
  cp, cp_sp = car_params()
  bad = model_lead()
  bad.x = []

  lead = get_lead(10.0, True, {}, bad, 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_vision_only_lead_requires_positive_distance():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=1.0), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_valid_vision_only_lead_still_publishes():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=20.0, v=11.0), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead["present"] is True
  assert lead["dRel"] > 0.0
  assert lead["vRel"] == 1.0
  assert lead["radar"] is False


def test_invalid_model_lead_does_not_match_radar_track():
  track = SimpleNamespace(dRel=20.0, yRel=0.0, vRel=1.0)

  assert match_vision_to_track(10.0, model_lead(v_std=float("nan")), {1: track}) is None


def test_get_radar_state_from_vision_rejects_invalid_model_v_ego():
  lead = get_RadarState_from_vision(model_lead(), 10.0, float("nan"), 0.9)

  assert lead == {"present": False}


def test_get_radar_state_from_vision_rejects_invalid_v_ego():
  lead = get_RadarState_from_vision(model_lead(), float("nan"), 10.0, 0.9)

  assert lead == {"present": False}


def test_get_lead_rejects_nonfinite_lead_probability():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=20.0, v=11.0), 10.0, float("inf"),
                  cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_clean_lead_probability_is_finite_and_bounded():
  assert _clean_lead_prob(None) == 0.0
  assert _clean_lead_prob(float("nan")) == 0.0
  assert _clean_lead_prob(float("inf")) == 0.0
  assert _clean_lead_prob(-0.2) == 0.0
  assert _clean_lead_prob(1.2) == 1.0
  assert _clean_lead_prob(0.7) == 0.7


def test_low_prob_model_with_matching_radar_track_is_confirmed_with_custom_long():
  cp, cp_sp = car_params()
  tracks = {1: radar_track(d_rel=22.0, y_rel=0.0, v_rel=1.0, v_lead=11.0)}

  lead = get_lead(10.0, True, tracks, model_lead(x=23.52, v=11.0), 10.0, 0.3, cp, cp_sp,
                  low_speed_override=False, custom_longitudinal_enabled=True)

  assert lead["present"] is True
  assert lead["radar"] is True
  assert lead["modelProb"] == 0.3


def test_low_prob_model_radar_confirmed_requires_custom_long_on():
  cp, cp_sp = car_params()
  tracks = {1: radar_track(d_rel=22.0, y_rel=0.0, v_rel=1.0, v_lead=11.0)}

  lead = get_lead(10.0, True, tracks, model_lead(x=23.52, v=11.0), 10.0, 0.3, cp, cp_sp,
                  low_speed_override=False, custom_longitudinal_enabled=False)

  assert lead == {"present": False}


def test_high_prob_radar_confirmed_works_with_custom_long_off():
  cp, cp_sp = car_params()
  tracks = {1: radar_track(d_rel=22.0, y_rel=0.0, v_rel=1.0, v_lead=11.0)}

  lead = get_lead(10.0, True, tracks, model_lead(x=23.52, v=11.0), 10.0, 0.6, cp, cp_sp,
                  low_speed_override=False, custom_longitudinal_enabled=False)

  assert lead["present"] is True
  assert lead["radar"] is True
  assert lead["modelProb"] == 0.6


def test_low_prob_model_without_radar_track_remains_rejected():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=23.52, v=11.0), 10.0, 0.3, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_low_prob_model_with_bad_radar_match_remains_rejected():
  cp, cp_sp = car_params()
  tracks = {1: radar_track(d_rel=80.0, y_rel=0.0, v_rel=1.0, v_lead=11.0)}

  lead = get_lead(10.0, True, tracks, model_lead(x=23.52, v=11.0), 10.0, 0.3, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_vision_only_lead_still_requires_high_probability():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=20.0, v=11.0), 10.0, 0.5, cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_nonfinite_prob_with_radar_track_still_rejected():
  cp, cp_sp = car_params()
  tracks = {1: radar_track(d_rel=22.0, y_rel=0.0, v_rel=1.0, v_lead=11.0)}

  lead = get_lead(10.0, True, tracks, model_lead(x=23.52, v=11.0), 10.0, float("nan"), cp, cp_sp, low_speed_override=False)

  assert lead == {"present": False}


def test_track_path_relative_y_on_straight_path():
  track = radar_track(d_rel=30.0, y_rel=0.0)
  model = SimpleNamespace(position=SimpleNamespace(x=[0.0, 30.0, 60.0], y=[0.0, 0.0, 0.0]))

  assert _track_path_relative_y(track, model) == pytest.approx(0.0)


def test_track_path_relative_y_on_curved_path():
  track = radar_track(d_rel=45.0, y_rel=0.5)
  model = SimpleNamespace(position=SimpleNamespace(x=[0.0, 30.0, 60.0], y=[0.0, 1.5, 2.0]))

  assert _track_path_relative_y(track, model) == pytest.approx(2.25)


def test_track_path_relative_y_returns_none_on_bad_track():
  model = SimpleNamespace(position=SimpleNamespace(x=[0.0, 30.0, 60.0], y=[0.0, 1.5, 2.0]))

  assert _track_path_relative_y(SimpleNamespace(), model) is None


def test_track_path_relative_y_falls_back_to_y_rel_when_path_invalid():
  track = radar_track(d_rel=30.0, y_rel=2.0)

  assert _track_path_relative_y(track, None) == pytest.approx(2.0)


def test_cut_in_override_rejects_ego_centerline_when_path_curves():
  from openpilot.sunnypilot.selfdrive.controls.lib.cut_in_override import apply_cut_in_override
  track = radar_track(d_rel=45.0, y_rel=0.0, v_rel=-3.0, v_lead=8.0)
  track.cnt = 3  # satisfy persistence gate
  model = SimpleNamespace(position=SimpleNamespace(x=[0.0, 30.0, 60.0], y=[0.0, 1.5, 2.0]))

  result = apply_cut_in_override(
    {"present": False}, {1: track}, v_ego=12.0,
    custom_longitudinal_enabled=True,
    path_y_rel=lambda t: _track_path_relative_y(t, model),
  )

  assert result == {"present": False}
