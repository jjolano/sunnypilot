#!/usr/bin/env python3
"""Closed-loop stick-slip lab for the torque v2.1 controller.

Route 000002a1 (2026-07-17) showed the wheel moving in discrete 0.6-1.2 deg steps
while the torque command stayed smooth: rack/EPS stick-slip. This lab closes the
loop between the production ``LatControlTorqueV21`` and a torque-domain plant with
static-breakaway + kinetic friction + pre-sliding compliance, so friction-compensation
changes can be tuned offline against the same metrics used on the on-road rlogs
(dwell->jump rate, step size, desired-vs-actual lag).

The pre-sliding term matters more than it sounds: with pure binary Coulomb stick the
rack is frozen solid below breakaway, so any demand whose whole envelope sits under
the breakaway threshold produces literally zero motion and every metric degenerates.
That is precisely the small, precise-correction band this lab is most often used to
study, and on-road logs show it is *not* frozen. Tuning against the frozen plant
overstates the benefit of friction changes and hides their dither cost.

The plant works in lateral-acceleration space: ``rack_pos`` is the lat-accel the
current rack position would produce at steady state; commanded torque must exceed
the breakaway threshold (relative to the self-centering holding torque) before the
rack moves. This is deliberately minimal — enough to reproduce the on-road
signature, not a high-fidelity EPS model.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.torque_v2_1 import LatControlTorqueV21

DT = 0.01

# RAV4 TSS2-like params (live-learned values from route 000002a1)
RAV4ISH_TORQUE_PARAMS = dict(latAccelFactor=1.94, latAccelOffset=0.0, friction=0.126,
                             steeringAngleDeadzoneDeg=0.0)


def _make_controller(torque_params: dict[str, float] | None = None) -> LatControlTorqueV21:
  tp = SimpleNamespace(**(torque_params or RAV4ISH_TORQUE_PARAMS))
  torque = SimpleNamespace(as_builder=lambda: tp)
  cp = SimpleNamespace(steerLimitTimer=3.0, lateralTuning=SimpleNamespace(torque=torque))
  ci = SimpleNamespace(
    torque_from_lateral_accel=lambda: (lambda la, tp: la / tp.latAccelFactor),
    lateral_accel_from_torque=lambda: (lambda t, tp: t * tp.latAccelFactor),
  )

  class _NoOpExtension:
    @staticmethod
    def update_override_torque_params(torque_params, v_ego=None) -> bool:
      return False

    @staticmethod
    def update(CS, VM, pid, params, ff, pid_log, *rest):
      return pid_log, rest[-1]

  return LatControlTorqueV21(cp, SimpleNamespace(), ci, DT, extension=_NoOpExtension())


class _FakeVM:
  # Same simplified vehicle model as fuzz_lateral_controller / test_torque_v2_1.
  @staticmethod
  def calc_curvature(angle_rad, v_ego, roll):
    denom = max(10.0 + 0.05 * v_ego * v_ego, 1e-6)
    return angle_rad / denom - 0.02 * roll


@dataclass(frozen=True)
class StictionPlantConfig:
  v_ego: float = 21.0
  breakaway_torque: float = 0.13     # normalized cmd units; must exceed to unstick
  kinetic_torque: float = 0.065      # sliding friction once moving
  mobility: float = 40.0             # (m/s^2)/s of rack motion per unit net torque
  restick_rate: float = 0.03         # |rack rate| below this re-sticks (m/s^2/s)
  tire_lag_s: float = 0.15           # rack position -> lat accel first-order lag
  actuator_delay_s: float = 0.12     # command -> rack force delay
  crown_pull_torque: float = 0.0     # constant plant bias (road crown), cmd units
  roll: float = 0.0                  # roll reported to the controller (rad)
  lat_delay: float = 0.28            # what the controller compensates (liveDelay-like)
  # Pre-sliding (microslip) compliance: (m/s^2) of rack displacement per unit net
  # torque while still stuck. Pure Coulomb stick (0.0) freezes the rack completely
  # below breakaway, so demands whose whole envelope sits under breakaway produce
  # *zero* motion and every metric degenerates -- which is exactly the small,
  # precise-correction band we care about. A real contact deforms elastically
  # first: below breakaway it still moves a little, roughly proportional to force.
  # Calibrated against routes 2cd/2ce, where the tiny band (<0.15 m/s^2 p2p) is not
  # frozen at all -- it tracks with ~0.33 s lag over 152k engaged samples. At 0.15
  # the lab reproduces that band at 0.32 s; at 0.0 it is frozen and unmeasurable.
  # NOTE this calibrates the small-amplitude band only: the lab still does not
  # reproduce the on-road amplitude->lag profile at larger amplitudes (on-road
  # medium 0.4-1.0 p2p measures ~0.00 s, the lab does not). Treat absolute lag as
  # comparative between variants, not as a prediction of on-road lag.
  # 0.0 reproduces the legacy binary-Coulomb plant for comparison.
  presliding_compliance: float = 0.15

  def __post_init__(self) -> None:
    # The implicit pre-sliding solve is stable for every non-negative compliance,
    # so there is no upper bound to police — but a negative one is unphysical
    # (the contact would deflect *against* the applied force) and silently
    # produces a plausible-looking trace, so reject it here.
    if not (self.presliding_compliance >= 0.0):
      raise ValueError(f"presliding_compliance must be >= 0, got {self.presliding_compliance}")

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class StictionTrace:
  t: np.ndarray
  desired_lat_accel: np.ndarray
  actual_lat_accel: np.ndarray
  steering_angle_deg: np.ndarray
  steering_rate_deg: np.ndarray
  command_torque: np.ndarray   # controller internal sign (positive = positive lat accel)
  stuck: np.ndarray


def run_closed_loop(desired_lat_accel: np.ndarray, config: StictionPlantConfig,
                    controller: LatControlTorqueV21 | None = None) -> StictionTrace:
  cfg = config
  ctrl = controller or _make_controller()
  vm = _FakeVM()
  n = len(desired_lat_accel)
  v = cfg.v_ego
  denom = 10.0 + 0.05 * v * v

  def angle_deg_for(a: float) -> float:
    # invert measurement path: measured = -calc_curvature(angle_rad, v, roll) * v^2
    curv = -a / (v * v)
    return math.degrees(curv * denom + 0.02 * cfg.roll * denom)

  lat_accel_factor = RAV4ISH_TORQUE_PARAMS["latAccelFactor"]
  delay_frames = max(1, int(round(cfg.actuator_delay_s / DT)))
  cmd_hist = [0.0] * delay_frames

  rack_pos = 0.0     # lat-accel-equivalent rack position
  rack_rate = 0.0
  act = 0.0          # actual lat accel (tire lag on rack_pos)
  stuck = True
  stick_anchor: float | None = 0.0   # rack position at zero net force while stuck
  angle = angle_deg_for(0.0)

  params = SimpleNamespace(roll=cfg.roll, angleOffsetDeg=0.0)
  out = {k: np.zeros(n) for k in ("act", "ang", "rate", "cmd", "stuck")}

  for i in range(n):
    prev_angle = angle
    angle = angle_deg_for(act)
    rate = (angle - prev_angle) / DT if i else 0.0

    CS = SimpleNamespace(vEgo=v, steeringAngleDeg=angle, steeringRateDeg=rate,
                         steeringPressed=False, aEgo=0.0)
    desired_curvature = desired_lat_accel[i] / (v * v)
    actuator_torque, _, _ = ctrl.update(True, CS, vm, params, False, desired_curvature,
                                        None, False, cfg.lat_delay)
    cmd = -float(actuator_torque)  # back to internal sign: positive -> positive lat accel

    cmd_hist.append(cmd)
    forced = cmd_hist.pop(0)

    # net torque on the rack: command + crown pull - self-centering holding torque
    holding = rack_pos / lat_accel_factor
    net = forced + cfg.crown_pull_torque - holding

    if stuck:
      if abs(net) > cfg.breakaway_torque:
        stuck = False
        stick_anchor = None
    if not stuck:
      rack_rate = cfg.mobility * (net - math.copysign(min(cfg.kinetic_torque, abs(net)), net))
      if abs(rack_rate) < cfg.restick_rate and abs(net) < cfg.breakaway_torque:
        stuck = True
        rack_rate = 0.0
        # remember the zero-force rack position so pre-sliding is measured from it
        stick_anchor = rack_pos - cfg.presliding_compliance * net
      rack_pos += rack_rate * DT
    else:
      # Pre-sliding: still stuck, but the contact deforms elastically with force.
      # Solve the static relation implicitly. The naive explicit form
      #   x_next = anchor + c * (F - x_prev / latAccelFactor)
      # is a fixed-point iteration with pole -c/latAccelFactor, so it rings at
      # c == latAccelFactor and diverges above it (c=2.0 reached 56 m/s^2 against
      # 0.013 at the default). Substituting net = F - x/latAccelFactor and solving
      # for x has no such pole and is stable for every c >= 0:
      #   x = anchor + c*(F - x/L)  ->  x = (anchor + c*F) / (1 + c/L)
      # Limits check out: c=0 pins x to the anchor, c->inf drives net to zero.
      prev_pos = rack_pos
      force = forced + cfg.crown_pull_torque
      if stick_anchor is None:
        stick_anchor = rack_pos - cfg.presliding_compliance * net
      rack_pos = ((stick_anchor + cfg.presliding_compliance * force)
                  / (1.0 + cfg.presliding_compliance / lat_accel_factor))
      rack_rate = (rack_pos - prev_pos) / DT

    alpha = DT / (DT + cfg.tire_lag_s)
    act += alpha * (rack_pos - act)

    out["act"][i] = act
    out["ang"][i] = angle
    out["rate"][i] = rate
    out["cmd"][i] = cmd
    out["stuck"][i] = float(stuck)

  t = np.arange(n) * DT
  return StictionTrace(t=t, desired_lat_accel=np.asarray(desired_lat_accel, dtype=float),
                       actual_lat_accel=out["act"], steering_angle_deg=out["ang"],
                       steering_rate_deg=out["rate"], command_torque=out["cmd"], stuck=out["stuck"])


# ── metrics (mirrors the on-road analysis of route 000002a1) ─────────────────

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
  X = np.fft.rfft(x - x.mean())
  f = np.fft.rfftfreq(len(x), 1 / fs)
  X[(f < lo) | (f > hi)] = 0
  return np.fft.irfft(X, n=len(x))


def _xcorr_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float = 5.0) -> tuple[float, float]:
  # positive lag => a leads b
  a = a - a.mean()
  b = b - b.mean()
  n = len(a)
  lags = np.arange(-int(max_lag_s * fs), int(max_lag_s * fs) + 1)
  norm = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12
  c = np.array([np.dot(a[max(0, -l):n - max(0, l)], b[max(0, l):n - max(0, -l)]) / norm for l in lags])
  i = int(np.argmax(c))
  return lags[i] / fs, float(c[i])


@dataclass(frozen=True)
class StictionMetrics:
  dwell_jump_per_min: float
  median_step_deg: float
  desired_actual_lag_s: float
  desired_actual_corr: float
  cmd_hf_lf: float
  rate_hf_lf: float
  tracking_rmse: float

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def compute_metrics(trace: StictionTrace, fs: float = 1.0 / DT) -> StictionMetrics:
  rate = trace.steering_rate_deg
  ang = trace.steering_angle_deg
  minutes = len(rate) / fs / 60.0

  dwell = np.abs(rate) < 0.5
  idx = np.flatnonzero(np.diff(np.concatenate(([0], dwell.view(np.int8), [0]))))
  runs = idx.reshape(-1, 2)
  runs = runs[(runs[:, 1] - runs[:, 0]) >= int(0.15 * fs)]
  njumps = 0
  steps = []
  for _, s1 in runs:
    if s1 < len(rate) - int(0.2 * fs) and np.abs(rate[s1:s1 + int(0.2 * fs)]).max() >= 1.5:
      njumps += 1
      steps.append(abs(ang[min(len(ang) - 1, s1 + int(0.3 * fs))] - ang[s1]))

  lag, corr = _xcorr_lag(trace.desired_lat_accel, trace.actual_lat_accel, fs)

  def hf(x: np.ndarray) -> float:
    return float(np.std(_bandpass(x, fs, 0.8, 5.0)))

  def lf(x: np.ndarray) -> float:
    return float(np.std(_bandpass(x, fs, 0.05, 0.8))) + 1e-9
  return StictionMetrics(
    dwell_jump_per_min=njumps / minutes if minutes > 0 else 0.0,
    median_step_deg=float(np.median(steps)) if steps else 0.0,
    desired_actual_lag_s=lag,
    desired_actual_corr=corr,
    cmd_hf_lf=hf(trace.command_torque) / lf(trace.command_torque),
    rate_hf_lf=hf(rate) / lf(rate),
    tracking_rmse=float(np.sqrt(np.mean((trace.desired_lat_accel - trace.actual_lat_accel) ** 2))),
  )


# ── scenarios ────────────────────────────────────────────────────────────────

def wander_demand(duration_s: float = 120.0, amp: float = 0.3, freq_hz: float = 0.12) -> np.ndarray:
  t = np.arange(0.0, duration_s, DT)
  return amp * np.sin(2 * np.pi * freq_hz * t)


def curve_demand(duration_s: float = 120.0, amp: float = 1.5, freq_hz: float = 0.05) -> np.ndarray:
  # gentle sweeping curves: sustained demand with slow modulation (octagon territory)
  t = np.arange(0.0, duration_s, DT)
  return amp * np.sin(2 * np.pi * freq_hz * t)


SCENARIOS = {"wander": wander_demand, "curve": curve_demand}


def main() -> None:
  parser = argparse.ArgumentParser(description="Closed-loop stick-slip lab for torque v2.1.")
  parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="wander")
  parser.add_argument("--duration", type=float, default=120.0)
  parser.add_argument("--amp", type=float, default=None, help="demand lat-accel amplitude m/s^2")
  parser.add_argument("--breakaway", type=float, default=0.13)
  parser.add_argument("--no-stiction", action="store_true", help="ideal rack (breakaway 0)")
  parser.add_argument("--crown", type=float, default=0.0, help="constant plant pull, cmd units")
  parser.add_argument("--floor", choices=("off", "shadow", "apply"), default="off",
                      help="friction breakaway floor mode")
  parser.add_argument("--floor-frac", type=float, default=None)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  kwargs = {"duration_s": args.duration}
  if args.amp is not None:
    kwargs["amp"] = args.amp
  demand = SCENARIOS[args.scenario](**kwargs)
  cfg = StictionPlantConfig(
    breakaway_torque=0.0 if args.no_stiction else args.breakaway,
    kinetic_torque=0.0 if args.no_stiction else args.breakaway / 2,
    crown_pull_torque=args.crown,
  )
  ctrl = _make_controller()
  # production wiring path: mode flows extension attr -> controller -> floor
  ctrl.extension.friction_breakaway_mode = args.floor
  if args.floor_frac is not None:
    ctrl.friction_floor.floor_frac = args.floor_frac
  trace = run_closed_loop(demand, cfg, controller=ctrl)
  metrics = compute_metrics(trace)
  if args.json:
    print(json.dumps({"config": cfg.to_dict(), "metrics": metrics.to_dict()}, indent=2))
  else:
    print(f"scenario={args.scenario} breakaway={cfg.breakaway_torque} crown={cfg.crown_pull_torque}")
    for k, v in metrics.to_dict().items():
      print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
  main()
