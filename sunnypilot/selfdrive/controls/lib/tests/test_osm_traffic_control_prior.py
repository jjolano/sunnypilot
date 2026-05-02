from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.osm_traffic_control_prior import (
  OsmTrafficControlPrior,
  TRAFFIC_CONTROL_CAUTION_SPEED,
  TRAFFIC_CONTROL_MIN_ACCEL,
)


def make_model(stop_distance: float | None = None):
  if stop_distance is None:
    position_x = [0.0, 10.0, 20.0, 30.0]
    velocity_x = [15.0, 14.0, 13.0, 12.0]
    should_stop = False
  else:
    position_x = [0.0, stop_distance * 0.5, stop_distance, stop_distance + 10.0]
    velocity_x = [15.0, 5.0, 0.5, 0.3]
    should_stop = True

  return SimpleNamespace(
    position=SimpleNamespace(x=position_x),
    velocity=SimpleNamespace(x=velocity_x),
    action=SimpleNamespace(shouldStop=should_stop),
  )


def make_map_data(control_type="stop_sign", distance=30.0, ahead=True):
  return SimpleNamespace(
    trafficControlAheadValid=ahead,
    trafficControlAhead=control_type if ahead else "",
    trafficControlAheadDistance=distance if ahead else 0.0,
    trafficControlValid=not ahead,
    trafficControl=control_type if not ahead else "",
    trafficControlDistance=distance if not ahead else 0.0,
  )


def make_sm(map_data, model_data):
  return {
    "liveMapDataSP": map_data,
    "modelV2": model_data,
  }


def test_traffic_control_prior_ignores_map_without_model_stop():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data(), make_model()), True, False, 15.0, 0.0)

  assert not prior.active
  assert prior.output_v_target == V_CRUISE_UNSET


def test_traffic_control_prior_uses_model_confirmed_stop_sign_as_caution_target():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data(distance=30.0), make_model(stop_distance=32.0)), True, False, 15.0, 0.0)

  assert prior.active
  assert prior.control_type == "stop_sign"
  assert prior.output_v_target == TRAFFIC_CONTROL_CAUTION_SPEED
  assert TRAFFIC_CONTROL_MIN_ACCEL <= prior.output_a_target < 0.0


def test_traffic_control_prior_stays_active_at_caution_speed_with_stop_context():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data(distance=10.0), make_model(stop_distance=12.0)), True, False, TRAFFIC_CONTROL_CAUTION_SPEED, 0.0)

  assert prior.active
  assert prior.output_v_target == TRAFFIC_CONTROL_CAUTION_SPEED
  assert prior.output_a_target <= 0.0


def test_traffic_control_prior_ignores_misaligned_model_stop():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data(distance=30.0), make_model(stop_distance=70.0)), True, False, 15.0, 0.0)

  assert not prior.active
  assert prior.output_v_target == V_CRUISE_UNSET


def test_traffic_control_prior_ignores_high_speed_context():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data(distance=30.0), make_model(stop_distance=30.0)), True, False, 30.0, 0.0)

  assert not prior.active
  assert prior.output_v_target == V_CRUISE_UNSET


def test_traffic_control_prior_accepts_current_traffic_signal_alias():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data("traffic-signals", distance=0.0, ahead=False), make_model(stop_distance=20.0)), True, False, 15.0, 0.0)

  assert prior.active
  assert prior.control_type == "traffic_signals"
  assert prior.output_v_target == TRAFFIC_CONTROL_CAUTION_SPEED


def test_traffic_control_prior_prefers_nearer_current_control_over_far_ahead_control():
  prior = OsmTrafficControlPrior()
  map_data = SimpleNamespace(
    trafficControlAheadValid=True,
    trafficControlAhead="traffic_light",
    trafficControlAheadDistance=120.0,
    trafficControlValid=True,
    trafficControl="stop_sign",
    trafficControlDistance=30.0,
  )

  prior.update(make_sm(map_data, make_model(stop_distance=32.0)), True, False, 15.0, 0.0)

  assert prior.active
  assert prior.control_type == "stop_sign"
  assert prior.distance == pytest.approx(30.0)


def test_traffic_control_prior_ignores_unsupported_control_type():
  prior = OsmTrafficControlPrior()

  prior.update(make_sm(make_map_data("crosswalk", distance=20.0), make_model(stop_distance=20.0)), True, False, 15.0, 0.0)

  assert not prior.active
  assert prior.output_v_target == V_CRUISE_UNSET
