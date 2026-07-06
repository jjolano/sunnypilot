#!/usr/bin/env python3
"""Extract compact per-channel numpy arrays from a full route of rlogs.

Usage (from repo root, after pulling rlogs per device-route-log-analysis):

  uv run --extra testing --extra tools python \
    .agents/skills/route-drive-diagnosis/scripts/extract_route_npz.py \
    '/tmp/opencode/sunnypilot-route-logs/ROUTE--*/rlog.zst' /path/out/route

Writes <out>.npz (arrays, prefixed cs_/cc_/rs_/lp_/sp_/mv_/ss_) and
<out>_events.json (onroadEvents tuples + intent/reason code tables).
~22 segments of rlogs fit in ~1.5 GB RAM; stream per-segment if larger.
"""
import glob
import json
import sys

import numpy as np

from openpilot.tools.lib.logreader import LogReader, ReadMode


def main() -> None:
  route_glob, out_prefix = sys.argv[1], sys.argv[2]
  paths = sorted(glob.glob(route_glob), key=lambda p: int(p.split("--")[-1].split("/")[0]))
  if not paths:
    raise SystemExit(f"no files match {route_glob}")
  print(f"loading {len(paths)} segments...", flush=True)
  msgs = list(LogReader(paths, default_mode=ReadMode.RLOG, sort_by_time=True))
  print(f"{len(msgs)} msgs", flush=True)

  cs = {k: [] for k in "t vEgo aEgo gasPressed brakePressed standstill steeringPressed leftBlinker rightBlinker cruiseEnabled".split()}
  cc = {k: [] for k in "t enabled longActive latActive accel".split()}
  rs = {k: [] for k in "t status dRel vRel yRel vLead aLeadK".split()}
  lp = {k: [] for k in "t aTarget shouldStop hasLead".split()}
  sp = {k: [] for k in ("t vTarget aTarget decActive decState clMode clActive clShouldStop clIntent clReason "
                        "dbgCustomA dbgMpcA dbgModelA dbgMpcStop dbgModelStop dbgCustomStop dbgFinalA dbgFinalStop "
                        "dbgClipMin dbgClipMax dbgE2e "
                        "sccVisState sccVisVTarget sccVisATarget sccVisCurLat sccVisMaxPredLat sccVisActive "
                        "sccMapState sccMapVTarget sccMapActive").split()}
  mv = {k: [] for k in "t llInnerLeftY llInnerRightY llProbInnerLeft llProbInnerRight llOuterLeftY llOuterRightY roadEdgeLY roadEdgeRY pathY10 desCurv".split()}
  ss = {k: [] for k in "t enabled active state".split()}
  events = []
  intent_codes: dict[str, int] = {}
  reason_codes: dict[str, int] = {}

  def code(d: dict[str, int], s: str) -> int:
    if s not in d:
      d[s] = len(d)
    return d[s]

  for m in msgs:
    w = m.which()
    t = m.logMonoTime * 1e-9
    if w == "carState":
      x = m.carState
      cs["t"].append(t); cs["vEgo"].append(x.vEgo); cs["aEgo"].append(x.aEgo)
      cs["gasPressed"].append(x.gasPressed); cs["brakePressed"].append(x.brakePressed)
      cs["standstill"].append(x.standstill); cs["steeringPressed"].append(x.steeringPressed)
      cs["leftBlinker"].append(x.leftBlinker); cs["rightBlinker"].append(x.rightBlinker)
      cs["cruiseEnabled"].append(x.cruiseState.enabled)
    elif w == "carControl":
      x = m.carControl
      cc["t"].append(t); cc["enabled"].append(x.enabled); cc["longActive"].append(x.longActive)
      cc["latActive"].append(x.latActive); cc["accel"].append(x.actuators.accel)
    elif w == "radarState":
      x = m.radarState.leadOne
      rs["t"].append(t); rs["status"].append(x.status); rs["dRel"].append(x.dRel)
      rs["vRel"].append(x.vRel); rs["yRel"].append(x.yRel); rs["vLead"].append(x.vLead); rs["aLeadK"].append(x.aLeadK)
    elif w == "longitudinalPlan":
      x = m.longitudinalPlan
      lp["t"].append(t); lp["aTarget"].append(x.aTarget); lp["shouldStop"].append(x.shouldStop); lp["hasLead"].append(x.hasLead)
    elif w == "longitudinalPlanSP":
      x = m.longitudinalPlanSP; d = x.longitudinalDebug; c = x.customLongitudinal
      sp["t"].append(t); sp["vTarget"].append(x.vTarget); sp["aTarget"].append(x.aTarget)
      sp["decActive"].append(x.dec.active); sp["decState"].append(int(x.dec.state.raw))
      sp["clMode"].append(int(c.mode.raw)); sp["clActive"].append(c.active); sp["clShouldStop"].append(c.shouldStop)
      sp["clIntent"].append(code(intent_codes, str(c.selectedIntent))); sp["clReason"].append(code(reason_codes, str(c.reason)))
      sp["dbgCustomA"].append(d.customATarget); sp["dbgMpcA"].append(d.mpcATarget); sp["dbgModelA"].append(d.modelATarget)
      sp["dbgMpcStop"].append(d.mpcShouldStop); sp["dbgModelStop"].append(d.modelShouldStop); sp["dbgCustomStop"].append(d.customShouldStop)
      sp["dbgFinalA"].append(d.finalATargetClipped); sp["dbgFinalStop"].append(d.finalShouldStop)
      sp["dbgClipMin"].append(d.accelClipMin); sp["dbgClipMax"].append(d.accelClipMax); sp["dbgE2e"].append(d.e2eSource)
      scc = x.smartCruiseControl; sv = scc.vision; smap = scc.map
      sp["sccVisState"].append(int(sv.state.raw)); sp["sccVisVTarget"].append(sv.vTarget); sp["sccVisATarget"].append(sv.aTarget)
      sp["sccVisCurLat"].append(sv.currentLateralAccel); sp["sccVisMaxPredLat"].append(sv.maxPredictedLateralAccel)
      sp["sccVisActive"].append(sv.active)
      sp["sccMapState"].append(int(smap.state.raw)); sp["sccMapVTarget"].append(smap.vTarget); sp["sccMapActive"].append(smap.active)
    elif w == "modelV2":
      x = m.modelV2
      lls = x.laneLines; probs = x.laneLineProbs; re_ = x.roadEdges
      if len(lls) >= 4 and len(lls[1].y) > 0:
        mv["t"].append(t)
        mv["llOuterLeftY"].append(lls[0].y[0]); mv["llInnerLeftY"].append(lls[1].y[0])
        mv["llInnerRightY"].append(lls[2].y[0]); mv["llOuterRightY"].append(lls[3].y[0])
        mv["llProbInnerLeft"].append(probs[1]); mv["llProbInnerRight"].append(probs[2])
        mv["roadEdgeLY"].append(re_[0].y[0] if len(re_) > 0 and len(re_[0].y) else np.nan)
        mv["roadEdgeRY"].append(re_[1].y[0] if len(re_) > 1 and len(re_[1].y) else np.nan)
        py = x.position.y
        mv["pathY10"].append(py[3] if len(py) > 3 else np.nan)
        mv["desCurv"].append(x.action.desiredCurvature)
    elif w == "selfdriveState":
      x = m.selfdriveState
      ss["t"].append(t); ss["enabled"].append(x.enabled); ss["active"].append(x.active); ss["state"].append(int(x.state.raw))
    elif w == "onroadEvents":
      for e in m.onroadEvents:
        events.append((t, str(e.name), bool(e.enable), bool(e.noEntry), bool(e.userDisable),
                       bool(e.softDisable), bool(e.immediateDisable), bool(e.overrideLongitudinal), bool(e.overrideLateral)))

  arrs = {}
  for prefix, d in [("cs", cs), ("cc", cc), ("rs", rs), ("lp", lp), ("sp", sp), ("mv", mv), ("ss", ss)]:
    for k, v in d.items():
      arrs[f"{prefix}_{k}"] = np.asarray(v, dtype=np.float64 if k == "t" else np.float32)
  np.savez_compressed(f"{out_prefix}.npz", **arrs)
  with open(f"{out_prefix}_events.json", "w") as f:
    json.dump({"events": events, "intent_codes": intent_codes, "reason_codes": reason_codes}, f)
  print("extraction done:", {k: len(v["t"]) for k, v in [("cs", cs), ("rs", rs), ("lp", lp), ("sp", sp), ("mv", mv)]}, flush=True)


if __name__ == "__main__":
  main()
