#!/usr/bin/env python3
"""§3 validation gate — run the real LongitudinalMpc over a route's logged lead-following frames
twice (raw leads vs §3 confidence-shaped leads) and compare the braking it commands. Answers the
ADR's question: does lead-motion anticipation cut reactive braking WITHOUT under-braking real decels?

For each radarState frame the MPC is anchored to the logged ego state (set_cur_state from
carState), so the per-frame a_target difference is purely the lead-shaping effect. v_cruise is set
high so the lead-follow constraint binds (we measure lead braking, not cruise).

Run: uv run python -m openpilot.tools.drive_lab.replay_lead_anticipation ROUTE [ROUTE ...]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.sunnypilot.custom.longitudinal.lead_anticipation import LeadAnticipation
from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.route_io import load_route_msgs
from openpilot.tools.drive_lab.timeline import safe_get

V_MIN = 8.0          # only score cruise-following at/above this speed
V_CRUISE_HIGH = 40.0  # m/s; keep cruise from binding so the lead-follow constraint dominates
BRAKE_A = -0.4       # a_target below this counts as braking


def _f(v: Any, d: float = 0.0) -> float:
  try:
    out = float(v)
  except (TypeError, ValueError):
    return d
  return out if math.isfinite(out) else d


class _AlwaysOn:
  def get_bool(self, k):
    return True

  def get(self, k, default=None, return_default=False):
    if k == "LeadAnticipationMode":
      return "apply"
    return default


@dataclass(frozen=True)
class LeadReplayRow:
  a_raw: float
  a_shaped: float
  a_lead: float
  v_rel: float
  d_rel: float


def summarize_rows(rows: list[LeadReplayRow], source: str) -> dict[str, Any]:
  if not rows:
    return {
      "source": source,
      "following_frames": 0,
      "note": f"no lead-following frames >= {V_MIN:g} m/s",
      "benefit_detected": False,
      "safety_pass": False,
      "invalid_metric": False,
    }

  a_raw = np.array([r.a_raw for r in rows])
  a_shaped = np.array([r.a_shaped for r in rows])
  a_lead = np.array([r.a_lead for r in rows])
  v_rel = np.array([r.v_rel for r in rows])
  d_rel = np.array([r.d_rel for r in rows])
  valid = all(np.all(np.isfinite(arr)) for arr in (a_raw, a_shaped, a_lead, v_rel, d_rel))
  delta = a_shaped - a_raw
  softened = delta > 0.02
  risky = softened & (v_rel < -1.5) & (delta > 0.3)
  braking = a_raw < BRAKE_A
  benefit_detected = bool(np.any(softened) and float(np.sum(delta[softened])) > 0.0)
  invalid_metric = not valid or any(not math.isfinite(x) for x in (float(np.min(a_raw)), float(np.min(a_shaped)), float(np.max(delta))))
  return {
    "brake_aleadk_noise": int(np.sum(braking & (a_lead < -1.0) & (v_rel > -1.0))),
    "brake_genuine_closing": int(np.sum(braking & (v_rel < -1.0))),
    "source": source,
    "following_frames": len(rows),
    "braking_frames": int(np.sum(braking)),
    "softened_frames": int(np.sum(softened)),
    "softened_pct": round(100.0 * np.mean(softened), 1),
    "decel_peak_raw": round(float(a_raw.min()), 3),
    "decel_peak_shaped": round(float(a_shaped.min()), 3),
    "mean_brake_reduction": round(float(delta[softened].mean()), 4) if softened.any() else 0.0,
    "p90_brake_reduction": round(float(np.percentile(delta[softened], 90)), 4) if softened.any() else 0.0,
    "max_brake_reduction": round(float(delta.max()), 4),
    "risky_softenings": int(np.sum(risky)),
    "benefit_detected": benefit_detected,
    "invalid_metric": invalid_metric,
    "safety_pass": bool(valid and int(np.sum(risky)) == 0),
  }


def _mpc():
  m = LongitudinalMpc()
  m.set_weights()
  return m


def analyze_route(msgs: list[Any], source: str) -> dict[str, Any]:
  mpc_raw, mpc_shaped = _mpc(), _mpc()
  la = LeadAnticipation(_AlwaysOn())
  latest: dict[str, Any] = {}
  rows: list[LeadReplayRow] = []
  for rec in build_route_messages(msgs):
    typ, payload = rec.typ, rec.payload
    if typ in ("carState", "carControl"):
      latest[typ] = payload
      continue
    if typ != "radarState":
      continue
    cs = latest.get("carState")
    if cs is None:
      continue
    v_ego = _f(safe_get(cs, "vEgo"))
    a_ego = _f(safe_get(cs, "aEgo"))
    # Force the offline apply-candidate context so replay still measures the §3 shaping effect after
    # live apply was made fail-closed without planner safety context.
    shaped = la.shape(
      payload, DT_MDL,
      long_active=True,
      brake_pressed=False,
      gas_pressed=False,
      force_decel=False,
      v_ego=v_ego,
    )                                                  # run every frame to keep track continuity warm
    lead = safe_get(payload, "leadOne")
    # score any moving frame with a real lead — the MPC solve is valid whether or not the device was
    # engaged (the radar aLeadK §3 shapes is present regardless); engagement only gates actuation.
    following = (v_ego >= V_MIN and lead is not None and bool(safe_get(lead, "status", False)))
    # solve both every frame to keep warm-starts aligned; only score following frames
    for mpc, rs in ((mpc_raw, payload), (mpc_shaped, shaped)):
      mpc.set_cur_state(v_ego, a_ego)
      mpc.update(rs, V_CRUISE_HIGH)
    if not following:
      continue
    rows.append(LeadReplayRow(float(mpc_raw.a_solution[0]), float(mpc_shaped.a_solution[0]),
                              _f(safe_get(lead, "aLeadK")), _f(safe_get(lead, "vRel")),
                              _f(safe_get(lead, "dRel"))))

  return summarize_rows(rows, source)


def render(r: dict[str, Any]) -> str:
  if r.get("following_frames", 0) == 0:
    return f"§3 lead-anticipation A/B: {r['source']}\n  note: {r.get('note', 'no data')}"
  return (
    f"§3 lead-anticipation A/B: {r['source']}\n"
    + f"  following frames {r['following_frames']} ({r['braking_frames']} braking: "
    + f"{r['brake_aleadk_noise']} aLeadK-noise, {r['brake_genuine_closing']} genuine closing)\n"
    + f"  §3 softened braking on {r['softened_frames']} frames ({r['softened_pct']}%): "
    + f"mean {r['mean_brake_reduction']:+.3f}, p90 {r['p90_brake_reduction']:.3f}, max {r['max_brake_reduction']:.3f} m/s^2\n"
    + f"  decel peak: raw {r['decel_peak_raw']} -> shaped {r['decel_peak_shaped']} m/s^2  "
    + f"(reactive-brake reduction = {r['decel_peak_shaped'] - r['decel_peak_raw']:+.3f})\n"
    + f"  SAFETY — risky softenings (eased braking while lead closing >1.5 m/s): {r['risky_softenings']}"
  )


def _json_payload(reports: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
  return reports[0] if len(reports) == 1 else reports


def render_reports(reports: list[dict[str, Any]], json_output: bool = False) -> str:
  if json_output:
    return json.dumps(_json_payload(reports), indent=2)
  return "\n\n".join(render(report) for report in reports)


def main() -> None:
  p = argparse.ArgumentParser(description="§3 lead-anticipation MPC A/B over lead-following frames.")
  p.add_argument("routes", nargs="+")
  p.add_argument("--output")
  p.add_argument("--json", action="store_true")
  p.add_argument("--qlog", action="store_true")
  args = p.parse_args()
  reports = [analyze_route(load_route_msgs(route, qlog=args.qlog), route) for route in args.routes]
  if args.output:
    Path(args.output).write_text(json.dumps(_json_payload(reports), indent=2, sort_keys=True) + "\n")
  print(render_reports(reports, json_output=args.json))


if __name__ == "__main__":
  main()
