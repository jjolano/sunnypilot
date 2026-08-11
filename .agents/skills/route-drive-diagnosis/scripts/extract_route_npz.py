#!/usr/bin/env python3
"""Extract compact per-channel numpy arrays from a full route of rlogs.

Usage (from repo root, after pulling rlogs per device-route-log-analysis):

  uv run --extra testing --extra tools python \
    .agents/skills/route-drive-diagnosis/scripts/extract_route_npz.py \
    '/tmp/opencode/sunnypilot-route-logs/ROUTE--*/rlog.zst' /path/out/route

Writes <out>.npz (arrays, prefixed cs_/cc_/co_/rs_/lp_/sp_/pose_/calib_/mv_/ss_, plus can_) and
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

  cs = {k: [] for k in ("t vEgo aEgo gasPressed brakePressed standstill steeringPressed leftBlinker rightBlinker cruiseEnabled "
                        "steeringAngleDeg steeringRateDeg steeringTorque steeringTorqueEps yawRate").split()}
  cc = {k: [] for k in "t enabled longActive latActive accel pitch torque curvature".split()}
  ct = {k: [] for k in ("t curvature desiredCurvature "
                        "tqActive tqError tqErrorRate tqP tqI tqD tqF tqOutput tqSaturated "
                        "tqActualLatAccel tqDesiredLatAccel tqDesiredLatJerk "
                        "mpActive mpGated mpQuality mpRawCurv mpProcCurv mpCondCurv mpPathCurv "
                        "mpLcActive mpLcNudge mpLcLatErr mpLcConf mpLaneChangeBlend "
                        "mpPaActive mpPaApplied mpPaNudge mpPaSlewLimited mpPaAyDelta "
                        "mpDemandSource mpReason "
                        # output-governor telemetry: names which cap bound, and separates an
                        # upstream PID/FF step (govNominal steps too) from a governor-made one.
                        # govNominal is PRE-governor, govPreSlew is post-cap/target-arrival but
                        # pre-slew, and carControl torque is the final post-governor output --
                        # the three together separate cap chatter from stateful slew catch-up.
                        "govReason govNominal govPreSlew govCap govAuthority govRelease govSignConflict "
                        # NOT the output governor: these three are steer_limited_by_safety, i.e.
                        # downstream requested-vs-applied actuator mismatch (controlsd.py:215 ->
                        # torque_v2_1.py:315). Named gov* until 2026-08-11, which produced wrong
                        # conclusions about how often the governor binds. Do not conflate.
                        "steerLimitSafety steerLimitSameDir steerLimitUnwind "
                        "govFrictionActive govFrictionDelta "
                        "govOscillation").split()}
  co = {k: [] for k in "t accel".split()}
  rs = {k: [] for k in "t present trackId dRel vRel yRel vLead aLeadK modelProb radar".split()}
  lp = {k: [] for k in "t aTarget shouldStop hasLead".split()}
  sp = {k: [] for k in ("t vTarget aTarget decActive decState clMode clActive clShouldStop clIntent clReason "
                        "dbgCustomA dbgMpcA dbgModelA dbgMpcStop dbgModelStop dbgCustomStop dbgFinalA dbgFinalStop "
                        "dbgClipMin dbgClipMax dbgE2e "
                        "upMode upEffectiveMode upEligible upWouldCap upApplied upBlockReason upRegime upSourceAge "
                        "upCarPitch upLivePitch upPitchZero upRelativePitch upGradePercent upProfileReady "
                        "upFitSlope upFitScore upFitSpan upFitMad upFitSamples upBandSpread upCeiling upGradeEnter "
                        "upGradeExit upGradeAccel upBefore upCap upAfter upRequestedNet upDelta upGradeLoadExceeds "
                        "upHeld upResearchAllowed upHasLead "
                        "sccVisState sccVisVTarget sccVisATarget sccVisCurLat sccVisMaxPredLat sccVisActive "
                        "sccMapState sccMapVTarget sccMapActive").split()}
  pose = {k: [] for k in "t pitch pitchStd valid inputsOK sensorsOK posenetOK".split()}
  calib = {k: [] for k in "t status roll pitch yaw".split()}
  mv = {k: [] for k in "t llInnerLeftY llInnerRightY llProbInnerLeft llProbInnerRight llOuterLeftY llOuterRightY roadEdgeLY roadEdgeRY pathY10 desCurv".split()}
  ss = {k: [] for k in "t enabled active state".split()}
  can = {k: [] for k in "t src dat".split()}
  events = []
  intent_codes: dict[str, int] = {}
  reason_codes: dict[str, int] = {}
  demand_source_codes: dict[str, int] = {}
  mp_reason_codes: dict[str, int] = {}
  cap_mode_codes: dict[str, int] = {}
  cap_reason_codes: dict[str, int] = {}
  cap_regime_codes: dict[str, int] = {}

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
      cs["steeringAngleDeg"].append(x.steeringAngleDeg); cs["steeringRateDeg"].append(x.steeringRateDeg)
      cs["steeringTorque"].append(x.steeringTorque); cs["steeringTorqueEps"].append(x.steeringTorqueEps)
      cs["yawRate"].append(x.yawRate)
    elif w == "carControl":
      x = m.carControl
      cc["t"].append(t); cc["enabled"].append(x.enabled); cc["longActive"].append(x.longActive)
      cc["latActive"].append(x.latActive); cc["accel"].append(x.actuators.accel)
      cc["pitch"].append(x.orientationNED[1] if len(x.orientationNED) == 3 else np.nan)
      cc["torque"].append(x.actuators.torque); cc["curvature"].append(x.actuators.curvature)
    elif w == "controlsState":
      x = m.controlsState
      ct["t"].append(t); ct["curvature"].append(x.curvature); ct["desiredCurvature"].append(x.desiredCurvature)
      q = x.lateralControlState
      tq = q.torqueState if q.which() == "torqueState" else None
      ct["tqActive"].append(tq.active if tq else False)
      for key, attr in (("tqError", "error"), ("tqErrorRate", "errorRate"), ("tqP", "p"), ("tqI", "i"),
                        ("tqD", "d"), ("tqF", "f"), ("tqOutput", "output"),
                        ("tqActualLatAccel", "actualLateralAccel"), ("tqDesiredLatAccel", "desiredLateralAccel"),
                        ("tqDesiredLatJerk", "desiredLateralJerk")):
        ct[key].append(getattr(tq, attr) if tq else np.nan)
      ct["tqSaturated"].append(tq.saturated if tq else False)
      p = x.modelPathState
      ct["mpActive"].append(p.active); ct["mpGated"].append(p.gated); ct["mpQuality"].append(p.quality)
      ct["mpRawCurv"].append(p.rawDesiredCurvature); ct["mpProcCurv"].append(p.processedDesiredCurvature)
      ct["mpCondCurv"].append(p.conditionedDesiredCurvature); ct["mpPathCurv"].append(p.modelPathCurvature)
      ct["mpLcActive"].append(p.laneCenteringActive); ct["mpLcNudge"].append(p.laneCenteringCurvatureNudge)
      ct["mpLcLatErr"].append(p.laneCenteringLateralError); ct["mpLcConf"].append(p.laneCenteringConfidence)
      ct["mpLaneChangeBlend"].append(p.laneChangeBlend)
      ct["mpPaActive"].append(p.previewAssistActive); ct["mpPaApplied"].append(p.previewAssistApplied)
      ct["mpPaNudge"].append(p.previewAssistCurvatureNudge); ct["mpPaSlewLimited"].append(p.previewAssistSlewLimited)
      ct["mpPaAyDelta"].append(p.previewAssistAyDelta)
      ct["mpDemandSource"].append(code(demand_source_codes, str(p.demandSource)))
      ct["mpReason"].append(code(mp_reason_codes, str(p.reason)))
      a = tq.adaptiveTorqueState if tq else None
      ct["govReason"].append(a.governorReason if a else 0)
      for key, attr in (("govNominal", "nominalOutput"), ("govPreSlew", "preSlewTarget"),
                        ("govCap", "outputCap"),
                        ("govAuthority", "authorityScale"), ("govFrictionDelta", "frictionFloorDelta")):
        ct[key].append(getattr(a, attr) if a else np.nan)
      for key, attr in (("govRelease", "releaseActive"), ("govSignConflict", "signConflictActive"),
                        ("steerLimitSafety", "steerLimitLimited"), ("steerLimitSameDir", "steerLimitSameDirection"),
                        ("steerLimitUnwind", "steerLimitUnwind"), ("govFrictionActive", "frictionFloorActive")):
        ct[key].append(bool(getattr(a, attr)) if a else False)
      ct["govOscillation"].append(a.oscillationClassification if a else 0)
    elif w == "carOutput":
      x = m.carOutput
      co["t"].append(t); co["accel"].append(x.actuatorsOutput.accel)
    elif w == "radarState":
      x = m.radarState.leadOne
      rs["t"].append(t); rs["present"].append(x.present); rs["trackId"].append(x.radarTrackId); rs["dRel"].append(x.dRel)
      rs["vRel"].append(x.vRel); rs["yRel"].append(x.yRel); rs["vLead"].append(x.vLead); rs["aLeadK"].append(x.aLeadK)
      rs["modelProb"].append(x.modelProb); rs["radar"].append(x.radar)
    elif w == "longitudinalPlan":
      x = m.longitudinalPlan
      lp["t"].append(t); lp["aTarget"].append(x.aTarget); lp["shouldStop"].append(x.shouldStop); lp["hasLead"].append(x.hasLead)
    elif w == "longitudinalPlanSP":
      x = m.longitudinalPlanSP; d = x.longitudinalDebug; c = x.customLongitudinal
      u = d.uphillNetDemandCap
      sp["t"].append(t); sp["vTarget"].append(x.vTarget); sp["aTarget"].append(x.aTarget)
      sp["decActive"].append(x.dec.active); sp["decState"].append(int(x.dec.state.raw))
      sp["clMode"].append(int(c.mode.raw)); sp["clActive"].append(c.active); sp["clShouldStop"].append(c.shouldStop)
      sp["clIntent"].append(code(intent_codes, str(c.selectedIntent))); sp["clReason"].append(code(reason_codes, str(c.reason)))
      sp["dbgCustomA"].append(d.customATarget); sp["dbgMpcA"].append(d.mpcATarget); sp["dbgModelA"].append(d.modelATarget)
      sp["dbgMpcStop"].append(d.mpcShouldStop); sp["dbgModelStop"].append(d.modelShouldStop); sp["dbgCustomStop"].append(d.customShouldStop)
      sp["dbgFinalA"].append(d.finalATargetClipped); sp["dbgFinalStop"].append(d.finalShouldStop)
      sp["dbgClipMin"].append(d.accelClipMin); sp["dbgClipMax"].append(d.accelClipMax); sp["dbgE2e"].append(d.e2eSource)
      sp["upMode"].append(code(cap_mode_codes, str(u.mode))); sp["upEffectiveMode"].append(code(cap_mode_codes, str(u.effectiveMode)))
      sp["upEligible"].append(u.eligible); sp["upWouldCap"].append(u.wouldCap); sp["upApplied"].append(u.applied)
      sp["upBlockReason"].append(code(cap_reason_codes, str(u.blockReason))); sp["upRegime"].append(code(cap_regime_codes, str(u.regime)))
      sp["upSourceAge"].append(u.sourceAgeS); sp["upCarPitch"].append(u.carPitch); sp["upLivePitch"].append(u.livePosePitch)
      sp["upPitchZero"].append(u.pitchZero); sp["upRelativePitch"].append(u.relativePitch)
      sp["upGradePercent"].append(u.filteredGradePercent); sp["upProfileReady"].append(u.profileReady)
      sp["upFitSlope"].append(u.fitSlope); sp["upFitScore"].append(u.fitScore); sp["upFitSpan"].append(u.fitPitchSpan)
      sp["upFitMad"].append(u.fitResidualMad); sp["upFitSamples"].append(u.fitSampleCount)
      sp["upBandSpread"].append(u.fitSpeedBandSpread); sp["upCeiling"].append(u.ceiling)
      sp["upGradeEnter"].append(u.gradeEnterPercent); sp["upGradeExit"].append(u.gradeExitPercent)
      sp["upGradeAccel"].append(u.gradeAccel); sp["upBefore"].append(u.aTargetBefore); sp["upCap"].append(u.aTargetCap)
      sp["upAfter"].append(u.aTargetAfter); sp["upRequestedNet"].append(u.requestedNetDemand); sp["upDelta"].append(u.deltaA)
      sp["upGradeLoadExceeds"].append(u.gradeLoadExceedsCeiling); sp["upHeld"].append(u.gradeHeld)
      sp["upResearchAllowed"].append(u.researchActuationAllowed); sp["upHasLead"].append(u.hasLead)
      scc = x.smartCruiseControl; sv = scc.vision; smap = scc.map
      sp["sccVisState"].append(int(sv.state.raw)); sp["sccVisVTarget"].append(sv.vTarget); sp["sccVisATarget"].append(sv.aTarget)
      sp["sccVisCurLat"].append(sv.currentLateralAccel); sp["sccVisMaxPredLat"].append(sv.maxPredictedLateralAccel)
      sp["sccVisActive"].append(sv.active)
      sp["sccMapState"].append(int(smap.state.raw)); sp["sccMapVTarget"].append(smap.vTarget); sp["sccMapActive"].append(smap.active)
    elif w == "livePose":
      x = m.livePose; orientation = x.orientationNED
      pose["t"].append(t); pose["pitch"].append(orientation.y); pose["pitchStd"].append(orientation.yStd)
      pose["valid"].append(orientation.valid); pose["inputsOK"].append(x.inputsOK)
      pose["sensorsOK"].append(x.sensorsOK); pose["posenetOK"].append(x.posenetOK)
    elif w == "liveCalibration":
      x = m.liveCalibration; rpy = x.rpyCalib
      calib["t"].append(t); calib["status"].append(int(x.calStatus.raw))
      calib["roll"].append(rpy[0] if len(rpy) == 3 else np.nan)
      calib["pitch"].append(rpy[1] if len(rpy) == 3 else np.nan)
      calib["yaw"].append(rpy[2] if len(rpy) == 3 else np.nan)
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
    elif w == "can":
      for frame in m.can:
        if frame.address == 452:
          payload = list(bytes(frame.dat)[:8])
          can["t"].append(t); can["src"].append(frame.src); can["dat"].append(payload + [0] * (8 - len(payload)))
    elif w == "onroadEvents":
      for e in m.onroadEvents:
        events.append((t, str(e.name), bool(e.enable), bool(e.noEntry), bool(e.userDisable),
                       bool(e.softDisable), bool(e.immediateDisable), bool(e.overrideLongitudinal), bool(e.overrideLateral)))

  arrs = {}
  for prefix, d in [("cs", cs), ("cc", cc), ("co", co), ("rs", rs), ("lp", lp), ("sp", sp),
                    ("ct", ct), ("pose", pose), ("calib", calib), ("mv", mv), ("ss", ss)]:
    for k, v in d.items():
      arrs[f"{prefix}_{k}"] = np.asarray(v, dtype=np.float64 if k == "t" else np.float32)
  arrs["can_t"] = np.asarray(can["t"], dtype=np.float64)
  arrs["can_src"] = np.asarray(can["src"], dtype=np.int16)
  arrs["can_dat"] = np.asarray(can["dat"], dtype=np.uint8).reshape((-1, 8))
  np.savez_compressed(f"{out_prefix}.npz", **arrs)
  with open(f"{out_prefix}_events.json", "w") as f:
    json.dump({
      "events": events,
      "intent_codes": intent_codes,
      "reason_codes": reason_codes,
      "cap_mode_codes": cap_mode_codes,
      "cap_reason_codes": cap_reason_codes,
      "cap_regime_codes": cap_regime_codes,
      "demand_source_codes": demand_source_codes,
      "mp_reason_codes": mp_reason_codes,
    }, f)
  print("extraction done:", {k: len(v["t"]) for k, v in [("cs", cs), ("co", co), ("rs", rs), ("lp", lp), ("sp", sp), ("ct", ct), ("pose", pose), ("mv", mv)]}, flush=True)


if __name__ == "__main__":
  main()
