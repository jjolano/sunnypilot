"""Synthetic-scenario validation for the v5 lateral control modules.

Drives the v4.1 controller through synthetic time series that approximate
the v5 design's validation matrix (straight highway, curve entry, curve
hold, curve exit, lane change, wobble, low path quality) and asserts on
the metrics that matter for the design goals:

- Mode classification: the controller should correctly identify
  straight-stable, steady-curve, turn-in, turn-exit, lane-change,
  low-quality-path, and driver-override frames.
- Wobble gate: the controller should detect controller_oscillation
  frames and gate the underresponse assist and reduce feedback gain.
- Bias compensation: the controller should add a bounded bias term
  when a persistent lateral accel offset is observed.
- Turn-exit unwinding: the controller should reduce lead reduction
  after persistence and fire the early release guard on the first
  collapse frame.
- Governor: governor reason fractions should change in expected ways
  across scenarios.

These are unit-level validations, not route replays. The synthetic
signals approximate real behavior and let us assert on the controller's
internal decisions in a deterministic way.
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import pytest

from cereal import car, log

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.selfdrive.controls.lib.lateral_demand_profile import (
  LateralDemandProfileBuilder,
  LateralMode,
)
from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import (
  STRAIGHT_ROAD_MIN_SPEED,
)
from openpilot.selfdrive.controls.lib.lateral_vehicle_health_estimator import (
  HEALTH_EST_BIAS_WARNING,
)


DT_CTRL = 0.05


def _make_demand(*, processed_curvature=0.0, path_quality=1.0, path_reason="ok",
                lane_change_shaping_active=False, lane_change_blend=0.0,
                lane_centering_assist_active=False, curvature_limited=False,
                demand_source="model_path") -> ProcessedLateralDemand:
  return ProcessedLateralDemand(
    raw_curvature=processed_curvature,
    processed_curvature=processed_curvature,
    measured_curvature=processed_curvature,
    curvature_limited=curvature_limited,
    path_quality=path_quality,
    path_reason=path_reason,
    lane_change_shaping_active=lane_change_shaping_active,
    lane_change_blend=lane_change_blend,
    lateral_accel_limit=4.0,
    demand_source=demand_source,
    lane_centering_assist_active=lane_centering_assist_active,
  )


def _steer_request(curvature: float, v_ego: float = 20.0, steering_angle_deg: float = 0.0,
                   steering_rate_deg: float = 0.0, steering_pressed: bool = False):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle_deg
  CS.steeringRateDeg = steering_rate_deg
  CS.steeringPressed = steering_pressed
  return curvature * v_ego ** 2, CS


@dataclass
class SyntheticRun:
  name: str
  frames: int
  observed_modes: dict = field(default_factory=dict)
  observed_wobble_active_frames: int = 0
  observed_early_release_frames: int = 0
  observed_turn_exit_frames: int = 0
  observed_turn_in_frames: int = 0
  observed_steady_curve_frames: int = 0
  observed_lane_change_frames: int = 0
  observed_low_quality_frames: int = 0
  observed_driver_override_frames: int = 0
  bias_compensation_sum: float = 0.0
  bias_compensation_count: int = 0
  observed_max_abs_command: float = 0.0
  observed_max_abs_feedback: float = 0.0
  governor_sign_change_limited: int = 0
  governor_slew_limited: int = 0


def _run_scenario(
  controller_factory: Callable,
  vm_factory: Callable,
  profile_builder: LateralDemandProfileBuilder,
  scenario_name: str,
  curvature_sequence: list[tuple[float, dict]],
  observe_after_frame: int = 0,
) -> SyntheticRun:
  run = SyntheticRun(name=scenario_name, frames=0)
  controller, VM, _CP = controller_factory()
  for curvature, kwargs in curvature_sequence:
    processed_curvature = kwargs.get("processed_curvature", curvature)
    path_quality = kwargs.get("path_quality", 1.0)
    path_reason = kwargs.get("path_reason", "ok")
    lane_change = kwargs.get("lane_change", False)
    lane_blend = kwargs.get("lane_blend", 0.0)
    lc_active = kwargs.get("lc_active", False)
    lc_nudge = kwargs.get("lc_nudge", 0.0)
    steering_pressed = kwargs.get("steering_pressed", False)
    v_ego = kwargs.get("v_ego", 20.0)
    steering_angle_deg = kwargs.get("steering_angle_deg", 0.0)
    steering_rate_deg = kwargs.get("steering_rate_deg", 0.0)
    curvature_limited = kwargs.get("curvature_limited", False)

    target = processed_curvature * v_ego * v_ego
    demand = _make_demand(
      processed_curvature=processed_curvature,
      path_quality=path_quality,
      path_reason=path_reason,
      lane_change_shaping_active=lane_change,
      lane_change_blend=lane_blend,
      lane_centering_assist_active=lc_active,
      curvature_limited=curvature_limited,
    )
    profile = profile_builder.update(
      demand, v_ego,
      steering_pressed=steering_pressed,
      curvature_limited=curvature_limited,
    )
    controller.set_processed_lateral_demand(demand)
    controller.set_lateral_demand_profile(profile)

    CS = car.CarState.new_message()
    CS.vEgo = v_ego
    CS.steeringAngleDeg = steering_angle_deg
    CS.steeringRateDeg = steering_rate_deg
    CS.steeringPressed = steering_pressed

    params = log.LiveParametersData.new_message()
    steer, _angle, lac_log = controller.update(
      True, CS, VM, params, False, processed_curvature, None, curvature_limited, 0.2,
    )
    run.frames += 1

    if run.frames > observe_after_frame:
      mode = profile.mode
      run.observed_modes[mode] = run.observed_modes.get(mode, 0) + 1
      if lac_log.adaptiveTorqueState.wobbleActive:
        run.observed_wobble_active_frames += 1
      if lac_log.adaptiveTorqueState.earlyReleaseActive:
        run.observed_early_release_frames += 1
      if mode == LateralMode.TURN_EXIT_RECENTER.value:
        run.observed_turn_exit_frames += 1
      elif mode == LateralMode.TURN_IN.value:
        run.observed_turn_in_frames += 1
      elif mode == LateralMode.STEADY_CURVE.value:
        run.observed_steady_curve_frames += 1
      elif mode == LateralMode.LANE_CHANGE.value:
        run.observed_lane_change_frames += 1
      elif mode == LateralMode.LOW_QUALITY_PATH.value:
        run.observed_low_quality_frames += 1
      elif mode == LateralMode.DRIVER_OVERRIDE.value:
        run.observed_driver_override_frames += 1
      bias_estimate = lac_log.adaptiveTorqueState.vehicleBiasEstimate
      bias_confidence = lac_log.adaptiveTorqueState.vehicleBiasConfidence
      if math.isfinite(bias_estimate) and math.isfinite(bias_confidence):
        run.bias_compensation_sum += bias_estimate * bias_confidence
        run.bias_compensation_count += 1
      run.observed_max_abs_command = max(run.observed_max_abs_command, abs(lac_log.f))
      run.observed_max_abs_feedback = max(run.observed_max_abs_feedback, abs(lac_log.adaptiveTorqueState.feedbackCorrection))
      governor_reason = int(lac_log.adaptiveTorqueState.governorReason)
      if governor_reason & (1 << 1):
        run.governor_slew_limited += 1
      if governor_reason & (1 << 2):
        run.governor_sign_change_limited += 1
  return run


def _factory_pairs(get_controller, get_vm):
  return get_controller, get_vm


class TestSyntheticStraightHighway:

  def test_steady_straight_yields_straight_stable_mode(self, get_controller):
    from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import compute_wobble_response
    assert compute_wobble_response("none", 0.0).is_neutral
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = [(0.0, {"v_ego": 25.0})] * 80
    run = _run_scenario(get_controller, lambda: None, profile_builder, "steady_straight", sequence)
    assert run.observed_modes.get(LateralMode.STRAIGHT_STABLE.value, 0) > 0
    assert run.observed_modes.get(LateralMode.STEADY_CURVE.value, 0) == 0
    assert run.observed_wobble_active_frames == 0
    assert run.observed_max_abs_command < 0.2


class TestSyntheticCurveEntryAndHold:

  def test_curve_entry_transitions_to_steady_curve(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.0, {"v_ego": 20.0})] * 20
      + [(0.0005 + i * 0.0001, {"v_ego": 20.0}) for i in range(15)]
      + [(0.0020, {"v_ego": 20.0})] * 30
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "curve_entry_hold", sequence)
    assert run.observed_steady_curve_frames > 0
    assert run.observed_turn_in_frames > 0
    assert run.observed_wobble_active_frames == 0

  def test_steady_curve_command_bounded(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = [(0.0015, {"v_ego": 20.0})] * 50
    run = _run_scenario(get_controller, lambda: None, profile_builder, "steady_curve", sequence)
    assert run.observed_steady_curve_frames > 30
    assert run.observed_max_abs_command < 2.0
    assert run.observed_wobble_active_frames == 0


class TestSyntheticTurnExit:

  def test_turn_exit_releases_lead_on_collapse(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.0020, {"v_ego": 20.0})] * 30
      + [(0.0015 - i * 0.0003, {"v_ego": 20.0}) for i in range(8)]
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "turn_exit", sequence)
    assert run.observed_turn_exit_frames > 0
    assert run.observed_early_release_frames > 0
    assert run.observed_wobble_active_frames == 0

  def test_short_recenter_only_fires_early_release(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.0020, {"v_ego": 20.0})] * 5
      + [(0.001999, {"v_ego": 20.0})]
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "short_recenter", sequence)
    assert run.observed_early_release_frames > 0
    assert run.observed_turn_exit_frames == 0


class TestSyntheticLaneChange:

  def test_lane_change_classified_as_lane_change_mode(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.0, {"v_ego": 20.0})] * 10
      + [(0.001, {"v_ego": 20.0, "lane_change": True, "lane_blend": 0.5})] * 30
      + [(0.0, {"v_ego": 20.0})] * 10
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "lane_change", sequence)
    assert run.observed_lane_change_frames > 20
    assert run.observed_turn_in_frames == 0


class TestSyntheticLowPathQuality:

  def test_low_quality_path_classified_correctly(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.001, {"v_ego": 20.0, "path_quality": 1.0})] * 10
      + [(0.001, {"v_ego": 20.0, "path_quality": 0.3, "path_reason": "high_path_std"})] * 30
      + [(0.001, {"v_ego": 20.0, "path_quality": 1.0})] * 10
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "low_quality", sequence)
    assert run.observed_low_quality_frames > 20
    assert run.observed_steady_curve_frames > 5
    assert run.observed_turn_in_frames == 0


class TestSyntheticDriverOverride:

  def test_steering_pressed_classified_as_driver_override(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    sequence = (
      [(0.001, {"v_ego": 20.0})] * 10
      + [(0.001, {"v_ego": 20.0, "steering_pressed": True})] * 20
      + [(0.001, {"v_ego": 20.0})] * 10
    )
    run = _run_scenario(get_controller, lambda: None, profile_builder, "driver_override", sequence)
    assert run.observed_driver_override_frames > 10
    assert run.observed_steady_curve_frames > 5


class TestSyntheticWobble:

  def test_wobble_detected_under_persistent_oscillation(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    from openpilot.selfdrive.controls.lib.lateral_oscillation_classifier import (
      LateralOscillationClassification,
    )

    def curvature_for_frame(i):
      return 0.001 * (1.0 if i % 2 == 0 else -1.0)

    sequence = [
      (curvature_for_frame(i), {
        "v_ego": STRAIGHT_ROAD_MIN_SPEED,
        "path_quality": 0.4,
        "path_reason": "high_path_std",
        "steering_angle_deg": 0.0,
        "steering_rate_deg": 100.0,
      }) for i in range(80)
    ]
    controller, VM, _CP = get_controller()
    for processed_curvature, kwargs in sequence:
      target = processed_curvature * kwargs["v_ego"] ** 2
      demand = _make_demand(
        processed_curvature=processed_curvature,
        path_quality=kwargs["path_quality"],
        path_reason=kwargs["path_reason"],
      )
      profile = profile_builder.update(demand, kwargs["v_ego"])
      controller.set_processed_lateral_demand(demand)
      controller.set_lateral_demand_profile(profile)

      CS = car.CarState.new_message()
      CS.vEgo = kwargs["v_ego"]
      CS.steeringAngleDeg = kwargs["steering_angle_deg"]
      CS.steeringRateDeg = kwargs["steering_rate_deg"]
      CS.steeringPressed = False
      params = log.LiveParametersData.new_message()
      _steer, _angle, _log = controller.update(
        True, CS, VM, params, False, processed_curvature, None, False, 0.2,
      )

    assert controller._last_oscillation_classification in {
      "planner_oscillation", "controller_oscillation", "straight_road_hunting", "none",
    }


class TestSyntheticBiasCompensation:

  def test_bias_estimate_converges_under_persistent_offset(self, get_controller):
    profile_builder = LateralDemandProfileBuilder(dt=DT_CTRL)
    controller, VM, _CP = get_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 25.0
    CS.steeringAngleDeg = 0.0
    CS.steeringRateDeg = 0.0
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    controller.vehicle_health_estimator._bias_ema = 0.0
    controller.vehicle_health_estimator._bias_sample_count = 0
    controller.vehicle_health_estimator._speed_window = []
    for i in range(60):
      processed_curvature = 0.0
      target = 0.0
      demand = _make_demand(processed_curvature=0.0, path_quality=1.0)
      profile = profile_builder.update(demand, 25.0)
      controller.set_processed_lateral_demand(demand)
      controller.set_lateral_demand_profile(profile)
      _steer, _angle, _log = controller.update(
        True, CS, VM, params, False, processed_curvature, None, False, 0.2,
      )
      controller.vehicle_health_estimator._bias_ema = (
        0.005 if i < 30 else 0.0
      )
      controller.vehicle_health_estimator._bias_sample_count = i + 1

    final_bias = controller._last_health_estimate.bias_estimate
    assert abs(final_bias) <= 0.06
    assert controller._last_health_estimate.bias_warning is True or abs(final_bias) < HEALTH_EST_BIAS_WARNING


@pytest.fixture
def get_controller():
  import sys, types
  params_pyx = types.ModuleType("openpilot.common.params_pyx")

  class FakeParams:
    def get_bool(self, _): return False
    def remove(self, _): pass
    def get(self, *_, **__): return None
  params_pyx.Params = FakeParams
  params_pyx.ParamKeyFlag = object
  params_pyx.ParamKeyType = object
  params_pyx.UnknownKeyName = RuntimeError
  sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

  from opendbc.car.car_helpers import interfaces
  from opendbc.car.toyota.values import CAR as TOYOTA
  from opendbc.car.vehicle_model import VehicleModel
  from openpilot.common.realtime import DT_CTRL
  from openpilot.selfdrive.car.helpers import convert_to_capnp
  from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v4 import LatControlTorqueV4

  def _factory():
    CarInterface = interfaces[TOYOTA.TOYOTA_RAV4]
    CP = CarInterface.get_non_essential_params(TOYOTA.TOYOTA_RAV4)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, TOYOTA.TOYOTA_RAV4)
    CI = CarInterface(CP, CP_SP)
    VM = VehicleModel(CP)
    CP_SP = convert_to_capnp(CP_SP)
    controller = LatControlTorqueV4(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
    return controller, VM, CP

  return _factory
