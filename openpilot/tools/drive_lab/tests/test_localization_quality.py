from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab.profile_localization_quality import _extract_report, render_report


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _meas(valid=True, std=(1.0, 2.0, 3.0), x=0.0, y=0.0, z=0.0):
  x_std, y_std, z_std = std
  return SimpleNamespace(valid=valid, xStd=x_std, yStd=y_std, zStd=z_std, x=x, y=y, z=z)


def test_flags_stds_and_frequency_extraction():
  msgs = [
    msg("cameraOdometry", 0.0, transStd=[1.0, 2.0, 8.0], rotStd=[0.1, 0.2, 0.3], rot=[0.0, 0.0, 0.02]),
    msg("livePose", 0.0, inputsOK=True, posenetOK=True, sensorsOK=False, orientationNED=_meas(std=(0.1, 0.2, 0.3), z=0.1), velocityDevice=_meas(std=(0.4, 0.5, 0.6)), angularVelocityDevice=_meas(std=(0.7, 0.8, 0.9), z=0.01), accelerationDevice=_meas(std=(1.0, 1.1, 1.2))),
    msg("cameraOdometry", 0.6, transStd=[2.0, 3.0, 9.0], rotStd=[0.2, 0.3, 0.4], rot=[0.0, 0.0, 0.03]),
    msg("livePose", 0.5, inputsOK=True, posenetOK=False, sensorsOK=True, orientationNED=_meas(std=(0.2, 0.3, 0.4), z=0.2), velocityDevice=_meas(std=(0.5, 0.6, 0.7)), angularVelocityDevice=_meas(std=(0.8, 0.9, 1.0), z=0.03), accelerationDevice=_meas(std=(1.3, 1.4, 1.5))),
  ]

  report = _extract_report(msgs, source="route-a")

  assert report.cameraOdometry_frequency.samples == 2
  assert report.livePose_frequency.samples == 2
  assert report.livePose_flags.all_ok_count == 0
  assert report.cameraOdometry_invalid_missing_vector_count == 0
  assert report.cameraOdometry_std["transStd"].p95 == pytest.approx((2.0**2 + 3.0**2 + 9.0**2) ** 0.5)
  assert report.cameraOdometry_high_trans_std_count == 2
  assert report.livePose_measurement_std["orientationNED"].max == pytest.approx(0.4)


def test_consistency_and_serialization():
  msgs = [
    msg("cameraOdometry", 0.0, transStd=[1.0, 1.0, 1.0], rotStd=[0.1, 0.1, 0.1], rot=[0.0, 0.0, 0.10]),
    msg("livePose", 0.05, inputsOK=True, posenetOK=True, sensorsOK=True, orientationNED=_meas(valid=True, z=0.09), velocityDevice=_meas(), angularVelocityDevice=_meas(valid=True, z=0.11), accelerationDevice=_meas()),
    msg("gpsLocation", 0.08, hasFix=True, speed=5.0, bearingAccuracyDeg=5.0, bearingDeg=8.0),
    msg("livePose", 0.18, inputsOK=True, posenetOK=True, sensorsOK=True, orientationNED=_meas(valid=True, z=0.08), velocityDevice=_meas(), angularVelocityDevice=_meas(valid=True, z=0.10), accelerationDevice=_meas()),
    msg("gpsLocationExternal", 0.22, valid=True, speed=6.0, accuracy=5.0, bearingDeg=7.0),
    msg("cameraOdometry", 0.20, transStd=[1.0, 1.0, 1.0], rotStd=[0.1, 0.1, 0.1], rot=[0.0, 0.0, 0.12]),
    msg("livePose", 0.22, inputsOK=True, posenetOK=True, sensorsOK=True, orientationNED=_meas(valid=True, z=0.13), velocityDevice=_meas(), angularVelocityDevice=_meas(valid=True, z=0.12), accelerationDevice=_meas()),
    msg("gpsLocation", 0.26, hasFix=True, speed=5.5, bearingAccuracyDeg=4.0, bearingDeg=7.5),
  ]

  report = _extract_report(msgs, source="route-b")
  data = report.to_dict()

  assert report.consistency["cameraOdometry_yaw_rate_z_vs_livePose_angularVelocityDevice.z"].pair_count >= 2
  assert report.consistency["gps_bearing_vs_livePose_orientationNED.z"].pair_count >= 3
  assert report.consistency["gps_bearing_vs_livePose_orientationNED.z"].p95_abs_error is not None
  assert report.health.ok is True
  assert report.health.degraded_reasons == []
  assert data["health"]["ok"] is True
  assert "Localization quality" in render_report(report)
  assert data["source"] == "route-b"


def test_degraded_health_reports_missing_gps_pairs_as_note_when_unavailable():
  msgs = [
    msg("cameraOdometry", 0.0, transStd=[10.0, 0.0, 0.0], rotStd=[0.1, 0.1, 0.1], rot=[0.0, 0.0, 0.0]),
    msg("livePose", 0.0, inputsOK=True, posenetOK=True, sensorsOK=True, orientationNED=_meas(valid=True, z=0.0), velocityDevice=_meas(), angularVelocityDevice=_meas(valid=True, z=0.0), accelerationDevice=_meas()),
  ]

  report = _extract_report(msgs, source="route-c")

  assert report.health.ok is False
  assert any("cameraOdometry" in reason for reason in report.health.degraded_reasons)
  assert "GPS messages present" not in " ".join(report.notes)
