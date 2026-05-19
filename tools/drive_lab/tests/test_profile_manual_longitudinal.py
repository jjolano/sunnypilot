from types import SimpleNamespace

from openpilot.tools.drive_lab import profile_manual_longitudinal as profile_cli
from openpilot.tools.lib.logreader import ReadMode


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def test_extract_manual_samples_persists_radar_lead_fields(monkeypatch):
  msgs = [
    msg("selfdriveState", 0.0, active=False),
    msg("radarState", 0.1, leadOne=SimpleNamespace(
      status=True,
      dRel=7.0,
      vRel=-0.2,
      vLeadK=0.1,
      aLeadK=0.05,
      modelProb=0.9,
    )),
    msg("carState", 0.2, vEgo=0.3, aEgo=0.0, gasPressed=False, brakePressed=False),
  ]
  monkeypatch.setattr(profile_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = profile_cli.extract_manual_samples("route-a", ReadMode.AUTO)

  assert len(samples) == 1
  assert samples[0].lead_d_rel == 7.0
  assert samples[0].lead_v_rel == -0.2
  assert samples[0].lead_v_lead == 0.1
  assert samples[0].lead_a_lead == 0.05
  assert samples[0].lead_model_prob == 0.9


def test_extract_manual_samples_persists_model_stop_context(monkeypatch):
  msgs = [
    msg("selfdriveState", 0.0, active=False),
    msg("modelV2", 0.1,
        action=SimpleNamespace(desiredAcceleration=-0.8, shouldStop=True),
        position=SimpleNamespace(x=[5.0, 20.0, 35.0])),
    msg("carState", 0.2, vEgo=8.0, aEgo=-0.4, gasPressed=False, brakePressed=True),
  ]
  monkeypatch.setattr(profile_cli, "LogReader", lambda route, default_mode, sort_by_time: msgs)

  samples = profile_cli.extract_manual_samples("route-a", ReadMode.AUTO)

  assert len(samples) == 1
  assert samples[0].model_should_stop
  assert samples[0].model_desired_accel == -0.8
  assert samples[0].model_stop_distance == 35.0
