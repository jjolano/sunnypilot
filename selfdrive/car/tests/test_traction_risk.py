from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.traction_risk import (
  RISK_REASON_CLEAR_THRESHOLD,
  TractionRiskEstimator,
  TractionRiskReason,
)


def make_car_state(v_ego=10.0, steering_angle_deg=0.0, esp_active=False, wheel_speeds=None):
  if wheel_speeds is None:
    wheel_speeds = (v_ego, v_ego, v_ego, v_ego)
  return SimpleNamespace(
    vEgo=v_ego,
    steeringAngleDeg=steering_angle_deg,
    espActive=esp_active,
    wheelSpeeds=SimpleNamespace(fl=wheel_speeds[0], fr=wheel_speeds[1], rl=wheel_speeds[2], rr=wheel_speeds[3]),
  )


def test_traction_risk_rises_on_esp_active():
  estimator = TractionRiskEstimator(0.1)

  state = estimator.update(make_car_state(esp_active=True))

  assert state.raw_risk == 1.0
  assert state.risk > 0.0
  assert state.reason & TractionRiskReason.ESP_ACTIVE


def test_traction_risk_detects_straight_line_wheel_speed_spread():
  estimator = TractionRiskEstimator(0.1)

  state = estimator.update(make_car_state(wheel_speeds=(10.0, 10.1, 12.0, 10.0)))

  assert 0.0 < state.raw_risk < 1.0
  assert state.reason & TractionRiskReason.WHEEL_SPEED_SPREAD


def test_traction_risk_ignores_turning_wheel_speed_spread():
  estimator = TractionRiskEstimator(0.1)

  state = estimator.update(make_car_state(steering_angle_deg=12.0, wheel_speeds=(10.0, 10.1, 12.0, 10.0)))

  assert state.raw_risk == 0.0
  assert state.reason == TractionRiskReason.NONE


def test_traction_risk_decays_and_clears_reason():
  estimator = TractionRiskEstimator(0.1)
  active = estimator.update(make_car_state(esp_active=True))
  for _ in range(2):
    active = estimator.update(make_car_state(esp_active=True))

  state = active
  for _ in range(60):
    state = estimator.update(make_car_state())

  assert 0.0 <= state.risk < active.risk
  assert state.risk <= RISK_REASON_CLEAR_THRESHOLD
  assert state.reason == TractionRiskReason.NONE


def test_traction_risk_schema_fields_round_trip():
  from cereal import custom

  msg = custom.CarStateSP.new_message()
  msg.tractionRisk = 0.5
  msg.tractionRiskRaw = 0.75
  msg.tractionRiskReason = int(TractionRiskReason.ESP_ACTIVE)

  assert msg.tractionRisk == pytest.approx(0.5)
  assert msg.tractionRiskRaw == pytest.approx(0.75)
  assert msg.tractionRiskReason == int(TractionRiskReason.ESP_ACTIVE)
