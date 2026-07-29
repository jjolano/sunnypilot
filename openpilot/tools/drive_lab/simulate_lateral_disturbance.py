#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.lateral_disturbance_sim import (
  LateralDisturbanceConfig,
  render_lateral_disturbance_report,
  route_lateral_trace,
  save_lateral_disturbance_report,
  simulate_lateral_disturbance,
  synthetic_lateral_trace,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Simulate lateral disturbances and score steering sensitivity.")
  parser.add_argument("--route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--synthetic", choices=("straight", "sine", "reversal", "curve_entry"), default="straight")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--dt", type=float, default=0.05)
  parser.add_argument("--speed", type=float, default=20.0)
  parser.add_argument("--crown-curvature", type=float, default=0.0)
  parser.add_argument("--crosswind-curvature", type=float, default=0.0)
  parser.add_argument("--gust-curvature", type=float, default=0.0)
  parser.add_argument("--model-noise-std", type=float, default=0.0)
  parser.add_argument("--sensor-noise-std", type=float, default=0.0)
  parser.add_argument("--stiction-deg", type=float, default=0.0)
  parser.add_argument("--backlash-deg", type=float, default=0.0)
  parser.add_argument("--delay", type=float, default=0.0)
  parser.add_argument("--rate-limit", type=float, default=180.0)
  parser.add_argument("--tire-gain", type=float, default=1.0)
  parser.add_argument("--tire-lag", type=float, default=0.25)
  parser.add_argument("--authority-attenuation", type=float, default=0.0)
  parser.add_argument("--authority-start", type=float, default=0.0)
  parser.add_argument("--authority-end", type=float, default=0.0)
  args = parser.parse_args()

  config = LateralDisturbanceConfig(
    seed=args.seed,
    duration_s=args.duration,
    dt_s=args.dt,
    speed_mps=args.speed,
    crown_curvature=args.crown_curvature,
    crosswind_curvature=args.crosswind_curvature,
    crosswind_gust_curvature=args.gust_curvature,
    model_noise_std=args.model_noise_std,
    sensor_noise_std=args.sensor_noise_std,
    steering_stiction_deg=args.stiction_deg,
    steering_backlash_deg=args.backlash_deg,
    actuator_delay_s=args.delay,
    actuator_rate_limit_deg_s=args.rate_limit,
    tire_gain=args.tire_gain,
    tire_lag_s=args.tire_lag,
    authority_attenuation=args.authority_attenuation,
    authority_start_s=args.authority_start,
    authority_end_s=args.authority_end,
  )
  source = args.synthetic
  if args.route:
    trace = route_lateral_trace(load_route_msgs(args.route, qlog=args.qlog), source=args.route)
    source = args.route
  else:
    trace = synthetic_lateral_trace(config, args.synthetic)
  report = simulate_lateral_disturbance(trace, config, source=source)
  print(output_report(report, json_output=args.json, renderer=render_lateral_disturbance_report, output_path=args.output, save=save_lateral_disturbance_report))


if __name__ == "__main__":
  main()
