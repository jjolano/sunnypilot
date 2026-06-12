"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

import numpy as np
import pytest

import cereal.messaging as messaging
from cereal import custom, log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.vehicle_math import speed_for_lateral_accel
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import (
  SmartCruiseControlVision,
  SunnypilotCurrentSmartCruiseControlVision,
  _A_LAT_REG_MAX,
  _CURRENT_LAT_ACC_BLEED_TH,
  _ENTERING_PRED_LAT_ACC_TH,
  _IN_TURN_LAT_ACC_TARGET,
  _NO_OVERSHOOT_TIME_HORIZON,
  _SUNNYPILOT_CURRENT_A_LAT_REG_MAX,
  _SUNNYPILOT_CURRENT_NO_OVERSHOOT_TIME_HORIZON,
)

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState


def _th_above_f32(th: float) -> float:
  """
  Return the next representable float32 *above* `th`.
  This avoids flaky comparisons around thresholds due to float32 rounding.
  """
  th32 = np.float32(th)
  above32 = np.nextafter(th32, np.float32(np.inf), dtype=np.float32)
  return float(above32)


def _build_single_spike_filtered(n: int, base: float = 1.0) -> np.ndarray:
  """
  Create an array where max() is >= threshold but p97 is < threshold.
  This demonstrates the behavior difference vs np.amax().

  Note: We intentionally construct using float32-representable values to match
  the data path through cereal/capnp.
  """
  th = float(_ENTERING_PRED_LAT_ACC_TH)
  th32 = float(np.float32(th))

  # numpy percentile default is linear interpolation: idx=(n-1)*p/100
  idx = (n - 1) * 0.97
  w = float(idx - np.floor(idx))

  base32 = float(np.float32(base))

  # Choose spike so that p97 = base + w*(spike-base) < th
  # -> spike < base + (th-base)/w. Use a margin (0.9) and ensure spike >= th.
  if w == 0.0:
    spike = th32 + 1.0
  else:
    spike = base32 + (th32 - base32) / w * 0.9
    spike = max(spike, th32 + 0.01)

  arr = np.full(n, base32, dtype=np.float32)
  arr[-1] = np.float32(spike)
  return arr


def _constant_pred_lat_accels(value: float) -> np.ndarray:
  return np.full(len(ModelConstants.T_IDXS), np.float32(value), dtype=np.float32)


def _pred_lat_accels_crossing_at(crossing_time: float, value: float = 3.0) -> tuple[np.ndarray, float]:
  crossing_idx = next(i for i, t in enumerate(ModelConstants.T_IDXS) if t >= crossing_time)
  pred_lat_accels = np.full(len(ModelConstants.T_IDXS), np.float32(1.0), dtype=np.float32)
  pred_lat_accels[crossing_idx:] = np.float32(value)
  return pred_lat_accels, float(ModelConstants.T_IDXS[crossing_idx])


def _set_predicted_lat_accels(model, pred_lat_accels: np.ndarray) -> None:
  model.modelV2.velocity.x = [1.0 for _ in range(len(pred_lat_accels))]
  model.modelV2.orientationRate.z = [float(x) for x in pred_lat_accels]


def _set_predicted_curvature_at_model_speed(model, curvature: float, model_speed: float) -> None:
  model.modelV2.velocity.x = [float(model_speed) for _ in ModelConstants.T_IDXS]
  model.modelV2.orientationRate.z = [float(curvature * model_speed) for _ in ModelConstants.T_IDXS]


def generate_modelV2():
  model = messaging.new_message('modelV2')
  position = log.XYZTData.new_message()
  speed = 30
  position.x = [float(x) for x in (speed + 0.5) * np.array(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  curvature = 0.05
  orientation.x = [float(curvature) for _ in ModelConstants.T_IDXS]
  orientation.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.orientation = orientation
  orientationRate = log.XYZTData.new_message()
  orientationRate.z = [float(z) for z in ModelConstants.T_IDXS]
  model.modelV2.orientationRate = orientationRate
  velocity = log.XYZTData.new_message()
  velocity.x = [float(x) for x in (speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
  velocity.x[0] = float(speed)  # always start at current speed
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
  acceleration.y = [float(y) for y in np.zeros_like(ModelConstants.T_IDXS)]
  model.modelV2.acceleration = acceleration

  return model


def generate_carState():
  car_state = messaging.new_message('carState')
  speed = 30
  v_cruise = 50
  car_state.carState.vEgo = float(speed)
  car_state.carState.standstill = False
  car_state.carState.vCruise = float(v_cruise * 3.6)

  return car_state


def generate_controlsState(curvature=0.0):
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = float(curvature)

  return controls_state


def generate_liveParameters(roll=0.0):
  live_parameters = log.LiveParametersData.new_message()
  live_parameters.roll = float(roll)
  return live_parameters


class TestSmartCruiseControlVision:
  def setup_method(self):
    self.params = Params()
    self.reset_params()
    self.scc_v = SmartCruiseControlVision()

    mdl = generate_modelV2()
    cs = generate_carState()
    controls_state = generate_controlsState()
    self.sm = {
      'modelV2': mdl.modelV2,
      'carState': cs.carState,
      'controlsState': controls_state.controlsState,
      'liveParameters': generate_liveParameters(),
    }

  def reset_params(self):
    self.params.put_bool("SccCurveVisionEnabled", True, block=True)
    self.params.put_bool("AccurateLateralAccel", False, block=True)

  def test_initial_state(self):
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.0

  def test_system_disabled(self):
    self.params.put_bool("SccCurveVisionEnabled", False, block=True)
    self.scc_v.enabled = self.params.get_bool("SccCurveVisionEnabled")

    for _ in range(int(10.0 / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0.0, 0.0, 0.0)
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active

  def test_disabled(self):
    for _ in range(int(10.0 / DT_MDL)):
      self.scc_v.update(self.sm, False, False, 0.0, 0.0, 0.0)
    assert self.scc_v.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10.0 / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0.0, 0.0, 0.0)
    assert self.scc_v.state == VisionState.enabled

  def test_turn_speed_wrapper_keeps_scc_sentinel_at_boundary(self):
    assert math.isinf(speed_for_lateral_accel(_A_LAT_REG_MAX, 0.0))
    assert self.scc_v._speed_for_lateral_accel(_A_LAT_REG_MAX, 0.0) == V_CRUISE_UNSET
    assert self.scc_v._speed_for_lateral_accel(_A_LAT_REG_MAX, 0.01) == pytest.approx(
      speed_for_lateral_accel(_A_LAT_REG_MAX, 0.01)
    )

  @pytest.mark.parametrize(
    ("yaw_rates", "velocities"),
    [
      ([], []),
      ([0.0, 0.1], [20.0]),
      ([0.0, float("nan"), 0.1], [20.0, 20.0, 20.0]),
      ([0.0, 0.1, 0.2], [20.0, float("inf"), 20.0]),
    ],
  )
  def test_malformed_model_prediction_disables_predicted_turn(self, yaw_rates, velocities):
    mdl = generate_modelV2()
    mdl.modelV2.orientationRate.z = yaw_rates
    mdl.modelV2.velocity.x = velocities
    self.sm["modelV2"] = mdl.modelV2

    self.scc_v.update(self.sm, True, False, float(MIN_V + 5.0), 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, float(MIN_V + 5.0), 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert self.scc_v.max_pred_lat_acc == pytest.approx(0.0)
    assert self.scc_v.predicted_turn_time == pytest.approx(0.0)

  def test_stays_inactive_below_min_speed_even_with_high_predicted_lat_acc(self):
    pred_lat_accels = _constant_pred_lat_accels(3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V - 0.1)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_entering_state_uses_gentler_target_speed_bias(self):
    pred_lat_accels = _constant_pred_lat_accels(3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.a_target == pytest.approx(-0.7)
    assert self.scc_v.v_target - self.scc_v.output_v_target == pytest.approx(0.7 * _NO_OVERSHOOT_TIME_HORIZON)

  def test_entering_decel_reaches_conservative_target_when_turn_is_far(self):
    pred_lat_accels, turn_time = _pred_lat_accels_crossing_at(4.5, value=3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 10.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    predicted_curve = 3.0 / (v_ego**2)
    conservative_v_target = (_A_LAT_REG_MAX / predicted_curve) ** 0.5

    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.predicted_turn_time == pytest.approx(turn_time)
    assert self.scc_v.output_v_target == pytest.approx(conservative_v_target)
    assert self.scc_v.a_target == pytest.approx((conservative_v_target - v_ego) / turn_time)
    assert -0.7 < self.scc_v.a_target < 0.0

  def test_entering_decel_blends_toward_iso_target_when_turn_is_close(self):
    pred_lat_accels, turn_time = _pred_lat_accels_crossing_at(1.0, value=3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 10.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    predicted_curve = 3.0 / (v_ego**2)
    conservative_v_target = (_A_LAT_REG_MAX / predicted_curve) ** 0.5
    iso_v_target = min((_IN_TURN_LAT_ACC_TARGET / predicted_curve) ** 0.5, v_ego)
    reachable_v_target = v_ego - 0.7 * turn_time

    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.predicted_turn_time == pytest.approx(turn_time)
    assert self.scc_v.output_v_target == pytest.approx(reachable_v_target)
    assert conservative_v_target < self.scc_v.output_v_target < iso_v_target
    assert self.scc_v.a_target == pytest.approx(-0.7)

  def test_turning_state_keeps_positive_accel_in_moderate_turn(self):
    pred_lat_accels = _constant_pred_lat_accels(2.3)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    moderate_curvature = 2.3 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(moderate_curvature).controlsState

    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.a_target == pytest.approx(0.15)

  def test_turning_state_applies_current_lat_acc_bleed(self):
    pred_lat_accels = _constant_pred_lat_accels(3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    high_curvature = 3.0 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(high_curvature).controlsState

    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.a_target == pytest.approx(-0.20)

  def test_current_lat_acc_bleed_uses_measured_curve_without_prediction(self):
    pred_lat_accels = _constant_pred_lat_accels(1.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    high_current_curvature = 3.0 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(high_current_curvature).controlsState

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.max_pred_lat_acc < _ENTERING_PRED_LAT_ACC_TH
    assert self.scc_v.current_lat_acc == pytest.approx(3.0)
    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.is_active
    assert self.scc_v.a_target == pytest.approx(-0.20)
    assert self.scc_v.output_v_target < v_ego

  def test_current_lat_acc_bleed_is_inactive_below_threshold(self):
    pred_lat_accels = _constant_pred_lat_accels(1.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    below_threshold_curvature = (_CURRENT_LAT_ACC_BLEED_TH - 0.05) / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(below_threshold_curvature).controlsState

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.current_lat_acc < _CURRENT_LAT_ACC_BLEED_TH
    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_predicted_lat_accel_uses_model_speed_by_default(self):
    mdl = generate_modelV2()
    _set_predicted_curvature_at_model_speed(mdl, curvature=0.01, model_speed=10.0)
    self.sm["modelV2"] = mdl.modelV2

    self.scc_v.update(self.sm, True, False, 20.0, 0.0, 20.0)

    assert self.scc_v.max_pred_lat_acc == pytest.approx(1.0)

  def test_accurate_lateral_accel_prediction_uses_current_speed_curvature(self):
    self.params.put_bool("AccurateLateralAccel", True, block=True)
    self.scc_v = SmartCruiseControlVision()
    mdl = generate_modelV2()
    _set_predicted_curvature_at_model_speed(mdl, curvature=0.01, model_speed=10.0)
    self.sm["modelV2"] = mdl.modelV2

    self.scc_v.update(self.sm, True, False, 20.0, 0.0, 20.0)

    assert self.scc_v.max_pred_lat_acc == pytest.approx(4.0)

  def test_accurate_lateral_accel_current_turn_uses_exact_roll_compensation(self):
    self.params.put_bool("AccurateLateralAccel", True, block=True)
    self.scc_v = SmartCruiseControlVision()
    v_ego = 20.0
    roll = math.asin(2.0 / ACCELERATION_DUE_TO_GRAVITY)
    self.sm["controlsState"] = generate_controlsState(3.0 / v_ego**2).controlsState
    self.sm["liveParameters"] = generate_liveParameters(roll)

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.current_lat_acc == pytest.approx(1.0)
    assert not self.scc_v.current_lat_acc_bleed

  def test_current_lat_acc_bleed_respects_longitudinal_override(self):
    pred_lat_accels = _constant_pred_lat_accels(1.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    high_current_curvature = 3.2 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(high_current_curvature).controlsState

    self.scc_v.update(self.sm, True, True, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, True, v_ego, 0.0, v_ego)

    assert self.scc_v.current_lat_acc > _CURRENT_LAT_ACC_BLEED_TH
    assert self.scc_v.state == VisionState.overriding
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_confirmed_turn_gradually_uses_iso_lateral_accel_budget(self):
    pred_lat_accels = _constant_pred_lat_accels(2.6)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    iso_compliant_curvature = 2.6 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(iso_compliant_curvature).controlsState

    for _ in range(int(3.0 / DT_MDL)):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.output_v_target > v_ego

  def test_confirmed_turn_reaches_iso_lateral_accel_budget_after_half_second(self):
    pred_lat_accels = _constant_pred_lat_accels(2.6)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    iso_compliant_curvature = 2.6 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(iso_compliant_curvature).controlsState

    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.in_turn_lat_acc_budget < 3.0

    for _ in range(int(round(0.5 / DT_MDL))):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.in_turn_lat_acc_budget == pytest.approx(3.0)
    assert self.scc_v.output_v_target > v_ego

  def test_current_lat_acc_bleed_keeps_speed_target_below_current_speed_after_budget_ramp(self):
    pred_lat_accels = _constant_pred_lat_accels(2.8)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    bleed_curvature = (_CURRENT_LAT_ACC_BLEED_TH + 0.05) / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(bleed_curvature).controlsState

    for _ in range(int(3.0 / DT_MDL)):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.current_lat_acc_bleed
    assert self.scc_v.output_v_target < v_ego

  def test_leaving_state_does_not_accelerate_into_imminent_next_curve(self):
    pred_lat_accels = _constant_pred_lat_accels(2.2)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    turning_curvature = 2.0 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(turning_curvature).controlsState
    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    winding_gap_curvature = 1.2 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(winding_gap_curvature).controlsState
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.state == VisionState.leaving
    assert self.scc_v.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH
    assert self.scc_v.a_target <= 0.0
    assert self.scc_v.output_v_target <= v_ego

  def test_leaving_state_does_not_speed_up_for_modest_imminent_next_curve(self):
    pred_lat_accels = _constant_pred_lat_accels(1.35)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    turning_curvature = 2.0 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(turning_curvature).controlsState
    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    winding_gap_curvature = 1.2 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(winding_gap_curvature).controlsState
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.state == VisionState.leaving
    assert self.scc_v.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH
    assert self.scc_v.output_v_target <= v_ego

  @pytest.mark.parametrize(
    "case, should_enter",
    [
      ("p97_just_above_threshold", True),
      ("single_spike_filtered", False),
      ("persistent_high_values", True),
    ],
    ids=[
      "p97>threshold_enters",
      "single_spike_max_large_but_p97_below_threshold",
      "high_values_persist_trigger_entering",
    ],
  )
  def test_max_pred_lat_acc_uses_p97_and_threshold(self, case, should_enter):
    n = len(ModelConstants.T_IDXS)
    th = float(_ENTERING_PRED_LAT_ACC_TH)

    if case == "p97_just_above_threshold":
      # Use the next representable float32 above threshold to avoid float32 rounding flakiness.
      val = _th_above_f32(th)
      pred_lat_accels = np.full(n, np.float32(val), dtype=np.float32)

    elif case == "single_spike_filtered":
      pred_lat_accels = _build_single_spike_filtered(n, base=1.0)

    elif case == "persistent_high_values":
      # Make enough "high" samples so p97 is driven by the persistent trend, not a single outlier.
      high_count = max(2, int(np.ceil(n * 0.03)) + 1)
      pred_lat_accels = np.full(n, np.float32(1.0), dtype=np.float32)
      pred_lat_accels[-high_count:] = np.float32(2.0)
      pred_lat_accels[-1] = np.float32(8.0)  # keep one big outlier too

    else:
      raise AssertionError(f"Unknown case: {case}")

    # Override model predictions so:
    # predicted_lat_accels = abs(orientationRate.z) * velocity.x == pred_lat_accels
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)

    # 1st update: disabled -> enabled
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    # 2nd update: evaluate entering condition from enabled state
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    # Controller does percentile on numpy float64 arrays (values already quantized by capnp),
    # so compute expected in float64 to match behavior and avoid interpolation/rounding deltas.
    expected_p97 = float(np.percentile(pred_lat_accels.astype(np.float64), 97))

    # allow tiny numeric differences due to float conversions/interpolation
    assert np.isclose(self.scc_v.max_pred_lat_acc, expected_p97, rtol=1e-6, atol=1e-5)

    if should_enter:
      # We assert entering primarily by state (this is the actual intended behavior).
      assert self.scc_v.state == VisionState.entering
      # Optional sanity: should be >= threshold with some margin (since we used nextafter above threshold).
      assert self.scc_v.max_pred_lat_acc > th

    else:
      # Difference vs np.amax(): max can be above threshold, but p97 stays below it.
      assert float(np.max(pred_lat_accels)) >= th
      assert self.scc_v.max_pred_lat_acc < th
      assert self.scc_v.state == VisionState.enabled

  # TODO-SP: mock modelV2 data to test other states


class TestSunnypilotCurrentSmartCruiseControlVision:
  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SccCurveVisionEnabled", True, block=True)
    self.params.put_bool("AccurateLateralAccel", False, block=True)
    self.scc_v = SunnypilotCurrentSmartCruiseControlVision()

    mdl = generate_modelV2()
    cs = generate_carState()
    controls_state = generate_controlsState()
    self.sm = {
      'modelV2': mdl.modelV2,
      'carState': cs.carState,
      'controlsState': controls_state.controlsState,
      'liveParameters': generate_liveParameters(),
    }

  def test_entering_state_uses_master_decel_and_overshoot_horizon(self):
    pred_lat_accels = _constant_pred_lat_accels(3.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    predicted_curve = 3.0 / (v_ego**2)
    expected_v_target = (_SUNNYPILOT_CURRENT_A_LAT_REG_MAX / predicted_curve) ** 0.5

    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.a_target == pytest.approx(-1.0)
    assert self.scc_v.v_target == pytest.approx(expected_v_target)
    assert self.scc_v.v_target - self.scc_v.output_v_target == pytest.approx(_SUNNYPILOT_CURRENT_NO_OVERSHOOT_TIME_HORIZON)

  def test_current_curve_bleed_does_not_start_turn_without_prediction(self):
    pred_lat_accels = _constant_pred_lat_accels(1.0)
    mdl = generate_modelV2()
    _set_predicted_lat_accels(mdl, pred_lat_accels)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    high_current_curvature = 3.0 / (v_ego**2)
    self.sm["controlsState"] = generate_controlsState(high_current_curvature).controlsState

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.current_lat_acc == pytest.approx(3.0)
    assert not self.scc_v.current_lat_acc_bleed
    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_lateral_accel_uses_master_curvature_calculation_even_with_accurate_param(self):
    self.params.put_bool("AccurateLateralAccel", True, block=True)
    self.scc_v = SunnypilotCurrentSmartCruiseControlVision()
    v_ego = 20.0
    roll = math.asin(2.0 / ACCELERATION_DUE_TO_GRAVITY)
    self.sm["controlsState"] = generate_controlsState(3.0 / v_ego**2).controlsState
    self.sm["liveParameters"] = generate_liveParameters(roll)

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, v_ego)

    assert self.scc_v.current_lat_acc == pytest.approx(3.0)
