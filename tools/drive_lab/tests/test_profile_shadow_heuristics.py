from __future__ import annotations

from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_shadow_heuristics import build_shadow_heuristics_report


class Msg(SimpleNamespace):
  def which(self) -> str:
    return self.kind


def _msg(kind: str, payload: object, t: int) -> Msg:
  return Msg(kind=kind, logMonoTime=t, **{kind: payload})


def test_shadow_heuristics_report_summarizes_lateral_and_grade() -> None:
  msgs = [
    _msg("controlsState", SimpleNamespace(
      desiredCurvature=0.01,
      modelPathState=SimpleNamespace(
        sensorConfidenceAvailable=True,
        sensorSuppressCandidate=True,
        sensorDisagreementLevel="high",
        sensorResponseClassification="underresponse_candidate",
        sensorConfidenceBlockReason="ok",
        sensorModelMeasuredLatAccelDelta=0.4,
        sensorModelYawLatAccelDelta=1.2,
        sensorSteeringYawLatAccelDelta=0.8,
        sensorModelYawLatAccelSignedDelta=1.2,
        sensorSteeringYawLatAccelSignedDelta=0.8,
        dtleEstimate=0.1,
      ),
    ), 0),
    _msg("longitudinalPlanSP", SimpleNamespace(
      longitudinalDebug=SimpleNamespace(scenarioContext=SimpleNamespace(
        scenario="uphill_recovery",
        roadGrade="uphill",
        blockReason="ok",
        accelCoast=-0.8,
        gradeConfidence=0.5,
        estimatedAccelBias=-0.5,
        proposedCompensation=0.15,
      )),
    ), 10),
  ]

  report = build_shadow_heuristics_report(msgs).to_dict()

  assert report["lateral"]["availableSamples"] == 1
  assert report["lateral"]["suppressCandidates"] == 1
  assert report["lateral"]["responseClassCounts"] == {"underresponse_candidate": 1}
  assert report["grade"]["scenarioCounts"] == {"uphill_recovery": 1}
  assert report["grade"]["roadGradeCounts"] == {"uphill": 1}
  assert report["grade"]["proposedCompensationStats"]["maxAbs"] == 0.15


def test_shadow_heuristics_report_compares_control_invariants() -> None:
  baseline = [
    _msg("controlsState", SimpleNamespace(desiredCurvature=0.01), 0),
    _msg("carControl", SimpleNamespace(actuators=SimpleNamespace(accel=0.2, steer=0.1, steeringAngleDeg=1.0)), 1),
    _msg("longitudinalPlan", SimpleNamespace(aTarget=0.2), 2),
  ]
  candidate = [
    _msg("controlsState", SimpleNamespace(desiredCurvature=0.01), 0),
    _msg("carControl", SimpleNamespace(actuators=SimpleNamespace(accel=0.2, steer=0.1, steeringAngleDeg=1.0)), 1),
    _msg("longitudinalPlan", SimpleNamespace(aTarget=0.2), 2),
  ]

  report = build_shadow_heuristics_report(candidate, baseline_msgs=baseline).to_dict()

  assert report["invariants"]["baselineCompared"] is True
  assert report["invariants"]["pass"] is True
  assert all(report["invariants"]["timeAligned"].values())
  assert all(v == 0 for v in report["invariants"]["diffCounts"].values())


def test_shadow_heuristics_report_marks_misaligned_invariants() -> None:
  baseline = [_msg("controlsState", SimpleNamespace(desiredCurvature=0.01), 0)]
  candidate = [_msg("controlsState", SimpleNamespace(desiredCurvature=0.01), 99)]

  report = build_shadow_heuristics_report(candidate, baseline_msgs=baseline).to_dict()

  assert report["invariants"]["pass"] is False
  assert report["invariants"]["timeAligned"]["controlsState.desiredCurvature"] is False
