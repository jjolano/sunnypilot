"""Route 0000025a fix: approach-cusp damping (#4) on the longitudinal finalizer.

Locks that the ACC-MPC's +/-0.3 m/s^2 approach-cusp limit cycle is jerk-limited inside the gentle
authority band, while strong accel/decel, stops, and stop-hold release (launch) ramps pass straight
through untouched so brake authority and launch responsiveness are never delayed.
"""
from __future__ import annotations

import numpy as np

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.tests.test_finalizer_characterization import make_cp


def test_approach_damp_smooths_in_band_limit_cycle():
  fin = CustomLongitudinalFinalizer(make_cp())
  dt = 0.05
  raw = [0.3 if i % 2 == 0 else -0.3 for i in range(40)]  # +/-0.3 alternating, worst-case cusp
  out = [fin._apply_approach_damp(a, should_stop=False, release_mpc_stop=False, dt=dt) for a in raw]
  out = np.array(out[5:])  # drop the seed transient
  # frame-to-frame jerk is bounded by the cap, and the residual swing is a fraction of the raw +/-0.3.
  assert np.max(np.abs(np.diff(out)) / dt) <= CustomLongitudinalFinalizer._APPROACH_DAMP_MAX_JERK + 1e-6
  assert np.ptp(out) < 0.6 * np.ptp(raw)


def test_approach_damp_passes_strong_and_stop_commands_through():
  fin = CustomLongitudinalFinalizer(make_cp())
  # A real brake (outside the band) is never delayed.
  assert fin._apply_approach_damp(-1.5, should_stop=False, release_mpc_stop=False, dt=0.05) == -1.5
  # Seed inside band, then a stop command must pass through untouched and drop the filter state.
  fin._apply_approach_damp(0.2, should_stop=False, release_mpc_stop=False, dt=0.05)
  assert fin._apply_approach_damp(-0.4, should_stop=True, release_mpc_stop=False, dt=0.05) == -0.4
  assert fin.approach_damp_a_prev is None
  # And a stop-hold release passes through untouched.
  assert fin._apply_approach_damp(0.25, should_stop=False, release_mpc_stop=True, dt=0.05) == 0.25


def test_approach_damp_never_damps_launch_release_ramp():
  fin = CustomLongitudinalFinalizer(make_cp())
  # Seed an in-band value so a naive damper would clamp the next step.
  fin._apply_approach_damp(0.05, should_stop=False, release_mpc_stop=False, dt=0.05)
  # A stop-hold release ramp is in progress: launch accel must pass through unclamped.
  fin.stop_hold_release_slew_a_target = 0.05
  assert fin._apply_approach_damp(0.35, should_stop=False, release_mpc_stop=False, dt=0.05) == 0.35
  assert fin.approach_damp_a_prev is None
