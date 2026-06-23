from types import SimpleNamespace

from openpilot.selfdrive.controls.radard import _clean_lead_prob, get_RadarState_from_vision, get_lead, match_vision_to_track


def model_lead(x=20.0, y=0.0, v=12.0, a=0.0, x_std=1.0, y_std=1.0, v_std=1.0):
  return SimpleNamespace(
    x=[x], y=[y], v=[v], a=[a],
    xStd=[x_std], yStd=[y_std], vStd=[v_std],
  )


def car_params():
  return SimpleNamespace(brand="toyota", flags=0), SimpleNamespace(flags=0)


def test_vision_only_lead_requires_finite_model_fields():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=float("nan")), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"status": False}


def test_vision_only_lead_requires_present_model_fields():
  cp, cp_sp = car_params()
  bad = model_lead()
  bad.x = []

  lead = get_lead(10.0, True, {}, bad, 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"status": False}


def test_vision_only_lead_requires_positive_distance():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=1.0), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead == {"status": False}


def test_valid_vision_only_lead_still_publishes():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=20.0, v=11.0), 10.0, 0.9, cp, cp_sp, low_speed_override=False)

  assert lead["status"] is True
  assert lead["dRel"] > 0.0
  assert lead["vRel"] == 1.0
  assert lead["radar"] is False


def test_invalid_model_lead_does_not_match_radar_track():
  track = SimpleNamespace(dRel=20.0, yRel=0.0, vRel=1.0)

  assert match_vision_to_track(10.0, model_lead(v_std=float("nan")), {1: track}) is None


def test_get_radar_state_from_vision_rejects_invalid_model_v_ego():
  lead = get_RadarState_from_vision(model_lead(), 10.0, float("nan"), 0.9)

  assert lead == {"status": False}


def test_get_radar_state_from_vision_rejects_invalid_v_ego():
  lead = get_RadarState_from_vision(model_lead(), float("nan"), 10.0, 0.9)

  assert lead == {"status": False}


def test_get_lead_rejects_nonfinite_lead_probability():
  cp, cp_sp = car_params()

  lead = get_lead(10.0, True, {}, model_lead(x=20.0, v=11.0), 10.0, float("inf"),
                  cp, cp_sp, low_speed_override=False)

  assert lead == {"status": False}


def test_clean_lead_probability_is_finite_and_bounded():
  assert _clean_lead_prob(None) == 0.0
  assert _clean_lead_prob(float("nan")) == 0.0
  assert _clean_lead_prob(float("inf")) == 0.0
  assert _clean_lead_prob(-0.2) == 0.0
  assert _clean_lead_prob(1.2) == 1.0
  assert _clean_lead_prob(0.7) == 0.7
