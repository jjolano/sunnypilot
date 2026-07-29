"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import (
  SmartCruiseControlVision,
  _CURRENT_LAT_ACC_BLEED_TH,
  _ENTERING_PRED_LAT_ACC_TH,
  _LEAVING_ACC,
  _PRE_ENTRY_GENTLE_DECEL,
  _PRE_ENTRY_PRED_LAT_ACC_TH,
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


def _set_modelV2_lat_acc(mdl, lat_acc_arr, v_model: float):
  """
  Populate modelV2 velocity and yaw rate so that the controller sees the
  desired predicted lateral acceleration profile (in m/s^2 at ego speed when
  v_model == v_ego).

  lat_acc = curvature * v_model**2 ; curvature = yaw_rate / v_model
  -> yaw_rate = lat_acc / v_model
  """
  n = len(ModelConstants.T_IDXS)
  lat_acc_arr = np.asarray(lat_acc_arr, dtype=np.float64)
  if lat_acc_arr.shape != (n,):
    lat_acc_arr = np.full(n, lat_acc_arr, dtype=np.float64)
  yaw_rate = lat_acc_arr / max(v_model, 0.1)
  mdl.modelV2.velocity.x = [float(v_model)] * n
  mdl.modelV2.orientationRate.z = [float(z) for z in yaw_rate]
  return mdl


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


def generate_controlsState():
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = 0.001

  return controls_state


class TestSmartCruiseControlVision:

  def setup_method(self):
    self.params = Params()
    self.reset_params()
    self.scc_v = SmartCruiseControlVision()

    mdl = generate_modelV2()
    cs = generate_carState()
    controls_state = generate_controlsState()
    self.sm = {'modelV2': mdl.modelV2, 'carState': cs.carState, 'controlsState': controls_state.controlsState}

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlVision", True, block=True)

  def test_initial_state(self):
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlVision", False, block=True)
    self.scc_v.enabled = self.params.get_bool("SmartCruiseControlVision")

    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active

  def test_disabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, False, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.enabled

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

    v_ego = float(MIN_V + 5.0)

    # Override model predictions so the ego-speed projected risk equals pred_lat_accels.
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

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

  def test_persistent_mild_curve_triggers_pre_entry(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(1.05), dtype=np.float32)

    v_ego = float(MIN_V + 5.0)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert self.scc_v.is_active
    assert self.scc_v.output_v_target != V_CRUISE_UNSET
    assert np.isclose(self.scc_v.output_a_target, _PRE_ENTRY_GENTLE_DECEL)
    assert self.scc_v.max_pred_lat_acc >= _PRE_ENTRY_PRED_LAT_ACC_TH
    assert self.scc_v.max_pred_lat_acc < _ENTERING_PRED_LAT_ACC_TH

  def test_pre_entry_does_not_act_while_overriding(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(1.05), dtype=np.float32)

    v_ego = float(MIN_V + 5.0)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(3):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.is_active

    self.scc_v.update(self.sm, True, True, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.overriding
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_pre_entry_respects_min_speed_floor(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(1.05), dtype=np.float32)

    v_ego = float(MIN_V)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(4):
      self.scc_v.update(self.sm, True, False, float(MIN_V), 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_single_mild_spike_does_not_trigger_pre_entry(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(0.95), dtype=np.float32)
    pred_lat_accels[-1] = np.float32(1.05)

    v_ego = float(MIN_V + 5.0)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_straight_low_prediction_remains_inactive(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(0.2), dtype=np.float32)

    v_ego = float(MIN_V + 5.0)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(4):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_zero_prediction_fails_closed(self):
    n = len(ModelConstants.T_IDXS)
    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [0.0 for _ in range(n)]
    mdl.modelV2.orientationRate.z = [0.0 for _ in range(n)]
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_negative_model_velocity_clamped_safe(self):
    n = len(ModelConstants.T_IDXS)
    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [-0.05 for _ in range(n)]
    mdl.modelV2.orientationRate.z = [0.1 for _ in range(n)]
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert np.isfinite(self.scc_v.v_target)
    assert self.scc_v.v_target >= 0.0

  def test_nan_ego_or_curvature_fails_closed(self):
    self.scc_v.update(self.sm, True, False, float("nan"), 0.0, 0.0)
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

    self.scc_v.update(self.sm, True, False, float(MIN_V + 5.0), 0.0, 0.0)
    self.sm["controlsState"].curvature = float("nan")
    self.scc_v.update(self.sm, True, False, float(MIN_V + 5.0), 0.0, 0.0)
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_negative_target_clamps_to_min_floor(self):
    self.scc_v.state = VisionState.entering
    self.scc_v.is_active = True
    self.scc_v.v_target = float(MIN_V + 1.0)
    self.scc_v.a_target = -10.0

    assert self.scc_v.get_v_target_from_control() == MIN_V

  def test_full_threshold_still_enters(self):
    n = len(ModelConstants.T_IDXS)
    pred_lat_accels = np.full(n, np.float32(_ENTERING_PRED_LAT_ACC_TH + 0.2), dtype=np.float32)

    v_ego = float(MIN_V + 5.0)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_lat_accels, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.is_active
    assert self.scc_v.output_a_target != _PRE_ENTRY_GENTLE_DECEL

  def test_nan_model_arrays_fail_closed(self):
    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [float("nan")] * len(ModelConstants.T_IDXS)
    mdl.modelV2.orientationRate.z = [1.0] * len(ModelConstants.T_IDXS)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    for _ in range(4):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def _patch_update_calculations(self, controller):
    called = []
    original = controller._update_calculations
    controller._update_calculations = lambda sm: called.append(True)
    return called, original

  def _restore_update_calculations(self, controller, original):
    controller._update_calculations = original

  def _seed_stale_vision_state(self):
    self.scc_v.state = VisionState.entering
    self.scc_v.max_pred_lat_acc = 2.0
    self.scc_v.current_lat_acc = 1.7
    self.scc_v.v_target = 15.0
    self.scc_v.pre_entry_frames = 5
    self.scc_v.pre_entry_active = True

  def _assert_fail_closed_state(self):
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.
    assert self.scc_v.max_pred_lat_acc == 0.
    assert self.scc_v.current_lat_acc == 0.
    assert self.scc_v.v_target == 0.
    assert self.scc_v.pre_entry_frames == 0
    assert not self.scc_v.pre_entry_active

  def test_disabled_long_skips_calculations_and_clears_stale_state(self):
    self._seed_stale_vision_state()
    called, original = self._patch_update_calculations(self.scc_v)
    try:
      self.scc_v.update(self.sm, False, False, 20.0, 0.0, 0.0)
    finally:
      self._restore_update_calculations(self.scc_v, original)

    assert not called
    self._assert_fail_closed_state()
    assert self.scc_v.frame == 0

  def test_feature_disabled_skips_calculations_and_clears_stale_state(self):
    self.scc_v.enabled = False
    self._seed_stale_vision_state()
    called, original = self._patch_update_calculations(self.scc_v)
    try:
      self.scc_v.update(self.sm, True, False, 20.0, 0.0, 0.0)
    finally:
      self._restore_update_calculations(self.scc_v, original)

    assert not called
    self._assert_fail_closed_state()
    assert self.scc_v.frame == 0

  def test_override_still_runs_calculations(self):
    self.scc_v.state = VisionState.entering
    self.scc_v.max_pred_lat_acc = 2.0
    called, original = self._patch_update_calculations(self.scc_v)
    try:
      self.scc_v.update(self.sm, True, True, 20.0, 0.0, 0.0)
    finally:
      self._restore_update_calculations(self.scc_v, original)

    assert called
    assert self.scc_v.state == VisionState.overriding
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_near_curve_brakes_harder_than_far_curve(self):
    n = len(ModelConstants.T_IDXS)
    v_ego = 30.0
    high_lat_acc = 3.0

    # Near curve: high predicted lat acc starting early in the horizon.
    near = np.zeros(n, dtype=np.float32)
    near[5:] = high_lat_acc
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, near, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.entering
    near_a = float(self.scc_v.output_a_target)
    near_v = float(self.scc_v.output_v_target)

    # Far curve: high predicted lat acc only late in the horizon.
    self.scc_v = SmartCruiseControlVision()
    self.scc_v.enabled = True
    far = np.zeros(n, dtype=np.float32)
    far[25:] = high_lat_acc
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, far, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.entering
    far_a = float(self.scc_v.output_a_target)
    far_v = float(self.scc_v.output_v_target)

    # Far curve should request less deceleration (i.e., a_target > near_a_target)
    # and should keep a higher speed target.
    assert near_a < far_a < 0.0
    assert far_v > near_v > MIN_V

  def test_binding_point_pairs_local_severity_with_local_time(self):
    n = len(ModelConstants.T_IDXS)
    v_ego = 30.0
    high_lat_acc = 3.0
    mild_lat_acc = 1.4

    # Early mild crossing followed by sharp curve late in the horizon.
    mixed = np.zeros(n, dtype=np.float32)
    mixed[5:16] = mild_lat_acc
    mixed[25:] = high_lat_acc
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, mixed, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.entering
    mixed_a = float(self.scc_v.output_a_target)

    # Near sharp only.
    self.scc_v = SmartCruiseControlVision()
    self.scc_v.enabled = True
    near = np.zeros(n, dtype=np.float32)
    near[5:] = high_lat_acc
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, near, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.entering
    near_a = float(self.scc_v.output_a_target)

    # Far sharp only.
    self.scc_v = SmartCruiseControlVision()
    self.scc_v.enabled = True
    far = np.zeros(n, dtype=np.float32)
    far[25:] = high_lat_acc
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, far, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.entering
    far_a = float(self.scc_v.output_a_target)

    # Mixed should bind to the sharp late curve, matching the far-sharp decel
    # rather than using the mild early time with the sharp severity.
    assert pytest.approx(mixed_a, abs=0.05) == far_a
    assert near_a < far_a < 0.0

  def test_current_lat_acc_bleed_only_above_high_threshold(self):
    v_ego = float(MIN_V + 10.0)
    n = len(ModelConstants.T_IDXS)

    # Current lat acc at the normal turning threshold (1.6) should *not* trigger bleed fallback.
    curvature_at_turning_th = 1.6 / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_at_turning_th)

    pred_low = np.full(n, np.float32(0.5), dtype=np.float32)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_low, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active

    # Only at the high bleed threshold should it activate turning.
    curvature_bleed = (_CURRENT_LAT_ACC_BLEED_TH + 0.2) / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_bleed)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.is_active

  def test_current_lat_acc_bleed_blocked_by_override_and_min_speed(self):
    v_ego = float(MIN_V + 10.0)
    n = len(ModelConstants.T_IDXS)
    curvature_bleed = (_CURRENT_LAT_ACC_BLEED_TH + 0.2) / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_bleed)

    pred_low = np.full(n, np.float32(0.5), dtype=np.float32)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_low, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    # Override: the bleed fallback is blocked; state stays overriding.
    self.scc_v.update(self.sm, True, True, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, True, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.overriding
    assert not self.scc_v.is_active

    # Remove override; bleed fallback is allowed next frame from enabled.
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.is_active

    # Below MIN_V with high current lat acc still stays enabled.
    self.scc_v = SmartCruiseControlVision()
    self.scc_v.enabled = True
    curvature_at_min = (_CURRENT_LAT_ACC_BLEED_TH + 0.2) / max(float(MIN_V) ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_at_min)
    self.scc_v.update(self.sm, True, False, float(MIN_V), 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, float(MIN_V), 0.0, 0.0)
    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active

  def test_current_lat_acc_bleed_handles_zero_prediction(self):
    v_ego = float(MIN_V + 10.0)
    n = len(ModelConstants.T_IDXS)
    curvature_bleed = (_CURRENT_LAT_ACC_BLEED_TH + 0.2) / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_bleed)

    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [0.0] * n
    mdl.modelV2.orientationRate.z = [0.0] * n
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.is_active

    # v_target should be derived from the current curve, not floored to MIN_V or unbounded.
    expected_v_target = float((2.0 / curvature_bleed) ** 0.5)
    a_target = float(self.scc_v.output_a_target)
    expected_output_v = max(expected_v_target, MIN_V) + a_target * 4.0
    assert self.scc_v.output_v_target == pytest.approx(expected_output_v, abs=0.01)
    assert MIN_V < self.scc_v.output_v_target < v_ego * 1.5

  def test_leaving_suppresses_accel_with_upcoming_curve(self):
    v_ego = float(MIN_V + 10.0)
    n = len(ModelConstants.T_IDXS)

    # Enter turning via high current lateral acceleration.
    curvature_turning = 3.0 / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_turning)
    pred_low = np.full(n, np.float32(0.5), dtype=np.float32)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_low, v_ego)
    self.sm["modelV2"] = mdl.modelV2

    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.turning

    # Drop current lat acc into leaving range (between _FINISH and _LEAVING).
    curvature_leaving = 1.2 / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_leaving)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.leaving
    assert self.scc_v.output_a_target == pytest.approx(_LEAVING_ACC, abs=1e-6)
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

    # Upcoming predicted curve should suppress positive leaving acceleration.
    pred_high = np.full(n, np.float32(2.0), dtype=np.float32)
    mdl = generate_modelV2()
    mdl = _set_modelV2_lat_acc(mdl, pred_high, v_ego)
    self.sm["modelV2"] = mdl.modelV2
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.leaving
    assert self.scc_v.output_a_target <= 0.0

  def test_leaving_with_zero_prediction_has_no_low_speed_cap(self):
    v_ego = float(MIN_V + 10.0)
    n = len(ModelConstants.T_IDXS)

    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [0.0] * n
    mdl.modelV2.orientationRate.z = [0.0] * n
    self.sm["modelV2"] = mdl.modelV2

    curvature_turning = 3.0 / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_turning)
    for _ in range(2):
      self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.turning
    assert self.scc_v.output_v_target != V_CRUISE_UNSET

    curvature_leaving = 1.2 / max(v_ego ** 2, 0.01)
    self.sm["controlsState"].curvature = float(curvature_leaving)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.leaving
    assert self.scc_v.output_a_target == pytest.approx(_LEAVING_ACC, abs=1e-6)
    assert self.scc_v.output_v_target == V_CRUISE_UNSET

  def test_invalid_model_arrays_clear_stale_state(self):
    self.scc_v.state = VisionState.entering
    self.scc_v.max_pred_lat_acc = 2.0
    self.scc_v.current_lat_acc = 1.7
    self.scc_v.v_target = 15.0
    self.scc_v._required_decel = -2.0
    self.scc_v.pre_entry_frames = 5
    self.scc_v.pre_entry_active = True

    # Mismatched-length model arrays are invalid and must fail closed.
    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [1.0] * (len(ModelConstants.T_IDXS) - 1)
    mdl.modelV2.orientationRate.z = [1.0] * len(ModelConstants.T_IDXS)
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.0
    assert self.scc_v.max_pred_lat_acc == 0.0
    assert self.scc_v.current_lat_acc == 0.0
    assert self.scc_v.v_target == 0.0
    assert self.scc_v._required_decel == 0.0
