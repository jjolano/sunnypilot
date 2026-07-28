"""Follow coast band: steady-follow accel<->decel sign chatter suppression.

Routes 290/291: finalA crossed zero 12-15x/min in steady follow with median dither depth
-0.08 m/s^2, toggling the Toyota PCM between gas and brake regimes every ~4 s. Locks that
shallow negative demands clamp to 0 while following (HOLD), demands past the natural-coast
band floor pass through untouched from their first frame (DECEL), and any stop / release /
pedal / no-lead / downhill-collapsed-band context is pure passthrough.
"""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.tests.test_finalizer_characterization import (
  make_cp,
  make_custom_long,
  make_custom_output,
  make_lead,
  make_sm,
)


def make_snapshot(*, accel_coast: float = -0.25, has_lead: bool = True, v_ego: float = 15.0,
                  gas_pressed: bool = False, brake_pressed: bool = False, force_decel: bool = False,
                  lead_v_rel: float = 0.0, dt: float = 0.05):
  return SimpleNamespace(
    custom_long_output=SimpleNamespace(accel_coast=accel_coast),
    has_lead=has_lead, v_ego=v_ego,
    gas_pressed=gas_pressed, brake_pressed=brake_pressed, force_decel=force_decel,
    lead_v_rel=lead_v_rel, dt=dt,
  )


def band(fin, a, snap, should_stop=False, release_mpc_stop=False):
  return fin._apply_follow_coast_band(a, snap, should_stop, release_mpc_stop)


def test_shallow_dither_never_crosses_zero():
  fin = CustomLongitudinalFinalizer(make_cp())
  snap = make_snapshot()
  raw = [0.08 if i % 2 == 0 else -0.08 for i in range(40)]
  out = [band(fin, a, snap) for a in raw]
  assert all(a >= 0.0 for a in out)
  assert out[0] == 0.08 and out[1] == 0.0  # positives pass, shallow negatives clamp


def test_real_decel_passes_from_first_frame_and_hysteresis_exit():
  fin = CustomLongitudinalFinalizer(make_cp())
  snap = make_snapshot()
  assert band(fin, 0.05, snap) == 0.05          # seed HOLD
  assert band(fin, -0.5, snap) == -0.5          # past band floor: DECEL, untouched
  assert band(fin, -0.1, snap) == -0.1          # honest shallow recovery inside DECEL
  assert band(fin, 0.01, snap) == 0.01          # below exit hysteresis: still DECEL
  assert band(fin, 0.05, snap) == 0.05          # past exit: back to HOLD
  assert band(fin, -0.08, snap) == 0.0          # shallow dither clamps again


def test_public_finalizer_chain_passes_shallow_mpc_decel_immediately():
  fin = CustomLongitudinalFinalizer(make_cp())
  custom_long = make_custom_long()
  custom_output = make_custom_output(accel_coast=-0.25)
  sm = make_sm(
    v_ego=15.0,
    lead_one=make_lead(d_rel=30.0, v_lead=13.5, v_rel=-1.5),
    long_active=True,
  )

  def finalize(mpc_a_target: float) -> float:
    return fin.finalize(
      sm, custom_long, custom_output, False, False, 0.05,
      mpc_a_target, False, 0.0, False,
      lambda _sm, a_target, *_args: a_target, lambda: None,
    ).a_target

  assert finalize(0.05) == 0.05  # seed the old band's HOLD regime
  outputs = [finalize(-0.08) for _ in range(25)]
  assert outputs[0] == -0.08
  assert all(a_target < 0.0 for a_target in outputs)


def test_joining_mid_decel_is_not_clamped():
  fin = CustomLongitudinalFinalizer(make_cp())
  assert band(fin, -0.2, make_snapshot()) == -0.2
  assert fin.follow_band_regime == "decel"


def test_context_gates_are_pure_passthrough_and_reset():
  fin = CustomLongitudinalFinalizer(make_cp())
  # prime HOLD, then each broken gate passes the raw value and drops the state
  for snap, kwargs in [
    (make_snapshot(has_lead=False), {}),
    (make_snapshot(v_ego=1.0), {}),
    (make_snapshot(gas_pressed=True), {}),
    (make_snapshot(brake_pressed=True), {}),
    (make_snapshot(force_decel=True), {}),
    (make_snapshot(accel_coast=-0.005), {}),   # downhill: band collapses
    (make_snapshot(), {"should_stop": True}),
    (make_snapshot(), {"release_mpc_stop": True}),
  ]:
    band(fin, 0.05, make_snapshot())
    assert band(fin, -0.08, snap, **kwargs) == -0.08
    assert fin.follow_band_regime is None


def test_release_slew_in_progress_is_passthrough():
  fin = CustomLongitudinalFinalizer(make_cp())
  band(fin, 0.05, make_snapshot())
  fin.stop_hold_release_slew_a_target = 0.1
  assert band(fin, -0.08, make_snapshot()) == -0.08
  assert fin.follow_band_regime is None


def test_sustained_closing_exhausts_the_giveaway_budget():
  # Regression, openpilot_lead_decel_3ms2: HOLD clamped a shallow-but-real closing demand
  # to 0 for 10 s of steady following, ratcheting d_rel 32.5 -> 26.2 m. The lead's stop
  # then had 2.3 m less runway than upstream and the maneuver ended in contact.
  fin = CustomLongitudinalFinalizer(make_cp())
  snap = make_snapshot(v_ego=20.0, lead_v_rel=-0.69)   # closing 0.69 m/s, budget = 2.0 m
  assert band(fin, 0.05, snap) == 0.05                 # seed HOLD
  held = 0
  for _ in range(400):                                 # 20 s at 20 Hz
    if band(fin, -0.08, snap) == -0.08:
      break
    held += 1
  assert fin.follow_band_regime == "decel"
  given = (held + 1) * 0.69 * 0.05   # the crossing frame is the one that passes through
  assert 2.0 < given <= 2.1, f"released after giving away {given:.2f} m"


def test_equilibrium_dither_never_exhausts_the_budget():
  # The chatter case this band exists for: v_rel oscillates about zero, so the net gap
  # given away stays ~0 and HOLD must survive indefinitely.
  fin = CustomLongitudinalFinalizer(make_cp())
  assert band(fin, 0.05, make_snapshot(v_ego=20.0)) == 0.05
  for i in range(2000):
    snap = make_snapshot(v_ego=20.0, lead_v_rel=0.15 if i % 2 else -0.15)
    assert band(fin, -0.08, snap) == 0.0
  assert fin.follow_band_regime == "hold"


def test_regime_change_resets_the_budget():
  fin = CustomLongitudinalFinalizer(make_cp())
  snap = make_snapshot(v_ego=20.0, lead_v_rel=-0.69)
  band(fin, 0.05, snap)
  for _ in range(20):
    band(fin, -0.08, snap)
  assert fin.follow_band_given_m > 0.0
  band(fin, -0.5, snap)                                # past band floor -> DECEL
  assert fin.follow_band_given_m == 0.0


def test_band_floor_is_capped_uphill():
  fin = CustomLongitudinalFinalizer(make_cp())
  snap = make_snapshot(accel_coast=-1.0)  # steep uphill coast; floor caps at -0.35
  band(fin, 0.05, snap)
  assert band(fin, -0.3, snap) == 0.0     # within capped band: clamped
  assert band(fin, -0.4, snap) == -0.4    # past the -0.35 floor: honest decel
