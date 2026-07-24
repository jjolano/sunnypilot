import numpy as np

from openpilot.tools.drive_lab.stiction_lab import (
  StictionPlantConfig,
  _make_controller,
  compute_metrics,
  run_closed_loop,
  wander_demand,
)


def _run(amp: float, breakaway: float):
  cfg = StictionPlantConfig(breakaway_torque=breakaway, kinetic_torque=breakaway / 2)
  trace = run_closed_loop(wander_demand(duration_s=60.0, amp=amp), cfg)
  return compute_metrics(trace)


def test_stiction_reproduces_onroad_stick_slip_signature():
  m = _run(amp=0.3, breakaway=0.13)
  # route 000002a1 signature: smooth command, jagged wheel, discrete steps
  assert m.rate_hf_lf > 0.5
  assert m.cmd_hf_lf < 0.5
  assert m.dwell_jump_per_min > 10
  assert 0.2 < m.median_step_deg < 2.0


def test_ideal_rack_is_clean():
  m = _run(amp=0.3, breakaway=0.0)
  assert m.rate_hf_lf < 0.3
  assert m.dwell_jump_per_min < 2
  assert m.desired_actual_corr > 0.99


def test_presliding_keeps_sub_breakaway_band_measurable():
  # With pure binary Coulomb stick the rack is frozen solid below breakaway, so a
  # demand whose whole envelope sits under it produces zero motion and every metric
  # degenerates — exactly the small-correction band this lab is used to study, and
  # one the on-road logs show is *not* frozen. Pre-sliding compliance keeps it real.
  demand = wander_demand(duration_s=60.0, amp=0.07)
  frozen = StictionPlantConfig(breakaway_torque=0.286, kinetic_torque=0.143,
                               presliding_compliance=0.0)
  real = StictionPlantConfig(breakaway_torque=0.286, kinetic_torque=0.143)
  frozen_trace = run_closed_loop(demand, frozen)
  real_trace = run_closed_loop(demand, real)
  assert np.ptp(frozen_trace.actual_lat_accel) < 1e-6          # literally frozen
  assert np.ptp(real_trace.actual_lat_accel) > 0.02            # microslip tracks
  assert compute_metrics(real_trace).desired_actual_corr > 0.5


def test_deadband_lag_grows_as_amplitude_shrinks():
  small = _run(amp=0.15, breakaway=0.13)
  large = _run(amp=0.6, breakaway=0.13)
  assert small.desired_actual_lag_s > large.desired_actual_lag_s + 0.1


def test_trace_finite_and_bounded():
  cfg = StictionPlantConfig(breakaway_torque=0.13, kinetic_torque=0.065)
  trace = run_closed_loop(wander_demand(duration_s=30.0, amp=0.3), cfg)
  for arr in (trace.actual_lat_accel, trace.steering_angle_deg, trace.command_torque):
    assert np.all(np.isfinite(arr))
  assert np.abs(trace.command_torque).max() <= 1.0 + 1e-6


def _run_mode(mode: str):
  cfg = StictionPlantConfig(breakaway_torque=0.13, kinetic_torque=0.065)
  ctrl = _make_controller()
  ctrl.extension.friction_breakaway_mode = mode  # production wiring: extension attr -> floor
  trace = run_closed_loop(wander_demand(duration_s=30.0, amp=0.15), cfg, controller=ctrl)
  return trace, ctrl


def test_floor_wiring_shadow_is_exactly_non_actuating():
  off_trace, _ = _run_mode("off")
  shadow_trace, shadow_ctrl = _run_mode("shadow")
  apply_trace, _ = _run_mode("apply")
  np.testing.assert_array_equal(off_trace.command_torque, shadow_trace.command_torque)
  assert not np.array_equal(off_trace.command_torque, apply_trace.command_torque)
  # shadow still records what it would have done
  assert shadow_ctrl.friction_floor.mode == "shadow"


def _run_direction_gain_mode(mode: str):
  cfg = StictionPlantConfig(breakaway_torque=0.13, kinetic_torque=0.065)
  ctrl = _make_controller()
  ctrl.extension.friction_breakaway_mode = "off"
  ctrl.extension.direction_gain_mode = mode
  ctrl.extension.direction_gain_scales = {1: 1.1, -1: 0.9}
  return run_closed_loop(wander_demand(duration_s=30.0, amp=0.3), cfg, controller=ctrl)


def test_direction_gain_shadow_is_exactly_non_actuating():
  # scales exposed but mode != apply must be identity at the controller layer too
  off = _run_direction_gain_mode("off")
  shadow = _run_direction_gain_mode("shadow")
  applied = _run_direction_gain_mode("apply")
  np.testing.assert_array_equal(off.command_torque, shadow.command_torque)
  assert not np.array_equal(off.command_torque, applied.command_torque)
