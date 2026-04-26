from types import SimpleNamespace

from openpilot.tools.drive_lab.log_profile import build_longitudinal_profile, load_profile, save_profile


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def test_build_longitudinal_profile_extracts_route_ranges(tmp_path):
  msgs = []
  for i in range(10):
    t = float(i)
    standstill = i < 3
    msgs.append(msg("carState", t, vEgo=0.0 if standstill else 12.0 + i, vCruise=72.0, standstill=standstill))
    msgs.append(msg("radarState", t + 0.1, leadOne=SimpleNamespace(status=True, dRel=6.0 + i * 4.0, vRel=-1.0 - i * 0.1, vLead=max(0.0, 3.0 - i * 0.2))))

  profile = build_longitudinal_profile(msgs, source="test-route")
  path = tmp_path / "profile.json"
  save_profile(profile, path)
  loaded = load_profile(path)

  assert loaded.source == "test-route"
  assert loaded.sample_count == len(msgs)
  assert loaded.ego_speed.high > loaded.ego_speed.low
  assert loaded.lead_gap.high > loaded.lead_gap.low
  assert loaded.lead_decel.high > 0.0
