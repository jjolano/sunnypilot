#!/usr/bin/env python3
"""§3 validation gate — run the real LongitudinalMpc over a route's logged following frames twice
(raw leads vs §3 confidence-shaped leads) and compare the braking it commands. Answers the ADR's
question: does lead-motion anticipation cut reactive braking WITHOUT under-braking real decels?

For each radarState frame the MPC is anchored to the logged ego state (set_cur_state from
carState), so the per-frame a_target difference is purely the lead-shaping effect. v_cruise is set
high so the lead-follow constraint binds (we measure lead braking, not cruise).

Run: uv run python -m openpilot.tools.drive_lab.replay_lead_anticipation ROUTE [ROUTE ...]
"""
from __future__ import annotations

import argparse
import math
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


def _mpc():
  m = LongitudinalMpc()
  m.set_weights()
  return m


def analyze_route(msgs: list[Any], source: str) -> dict[str, Any]:
  mpc_raw, mpc_shaped = _mpc(), _mpc()
  la = LeadAnticipation(_AlwaysOn())
  latest: dict[str, Any] = {}
  rows = []
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
    shaped = la.shape(payload, DT_MDL)                 # run every frame to keep track continuity warm
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
    a_raw = float(mpc_raw.a_solution[0])
    a_shaped = float(mpc_shaped.a_solution[0])
    rows.append((a_raw, a_shaped, _f(safe_get(lead, "aLeadK")), _f(safe_get(lead, "vRel")),
                 _f(safe_get(lead, "dRel"))))

  if not rows:
    return {"source": source, "following_frames": 0, "note": "no engaged following frames >= 8 m/s"}

  a_raw = np.array([r[0] for r in rows])
  a_shaped = np.array([r[1] for r in rows])
  a_lead = np.array([r[2] for r in rows])
  v_rel = np.array([r[3] for r in rows])
  delta = a_shaped - a_raw                              # >0 => §3 reduced braking
  softened = delta > 0.02
  risky = softened & (v_rel < -1.5) & (delta > 0.3)    # safety: eased braking while lead closing fast
  # is the braking aLeadK-noise (a decel spike on a non-closing lead, smoothable) or genuine
  # closing (slower/closing lead, NOT removable by any aLeadK shaping)?
  braking = a_raw < BRAKE_A
  return {
    "brake_aleadk_noise": int(np.sum(braking & (a_lead < -1.0) & (v_rel > -1.0))),
    "brake_genuine_closing": int(np.sum(braking & (v_rel < -1.0))),
    "source": source,
    "following_frames": len(rows),
    "braking_frames": int(np.sum(a_raw < BRAKE_A)),
    "softened_frames": int(np.sum(softened)),
    "softened_pct": round(100.0 * np.mean(softened), 1),
    "decel_peak_raw": round(float(a_raw.min()), 3),
    "decel_peak_shaped": round(float(a_shaped.min()), 3),
    "mean_brake_reduction": round(float(delta[softened].mean()), 4) if softened.any() else 0.0,
    "p90_brake_reduction": round(float(np.percentile(delta[softened], 90)), 4) if softened.any() else 0.0,
    "max_brake_reduction": round(float(delta.max()), 4),
    "risky_softenings": int(np.sum(risky)),   # SAFETY: softened a brake while the lead closed fast
  }


def render(r: dict[str, Any]) -> str:
  if r.get("following_frames", 0) == 0:
    return f"§3 A/B {r['source']}: {r.get('note', 'no data')}"
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


def main() -> None:
  p = argparse.ArgumentParser(description="§3 lead-anticipation MPC A/B over engaged following frames.")
  p.add_argument("routes", nargs="+")
  p.add_argument("--qlog", action="store_true")
  args = p.parse_args()
  for route in args.routes:
    print(render(analyze_route(load_route_msgs(route, qlog=args.qlog), route)))


if __name__ == "__main__":
  main()
