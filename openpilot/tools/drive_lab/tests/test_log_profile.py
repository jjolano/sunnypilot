from types import SimpleNamespace

from openpilot.tools.drive_lab import log_profile
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
    msgs.append(msg("radarState", t + 0.1, leadOne=SimpleNamespace(present=True, dRel=6.0 + i * 4.0, vRel=-1.0 - i * 0.1, vLead=max(0.0, 3.0 - i * 0.2))))

  profile = build_longitudinal_profile(msgs, source="test-route")
  path = tmp_path / "profile.json"
  save_profile(profile, path)
  loaded = load_profile(path)

  assert loaded.source == "test-route"
  assert loaded.sample_count == len(msgs)
  assert loaded.ego_speed.high > loaded.ego_speed.low
  assert loaded.lead_gap.high > loaded.lead_gap.low
  assert loaded.lead_decel.high > 0.0


def test_nearest_car_state_uses_indexed_lookup(monkeypatch):
  samples = [(float(i), float(i), i == 0) for i in range(100)]

  def fail_if_full_scan(*args, **kwargs):
    raise AssertionError("nearest lookup should not use a full min() scan")

  monkeypatch.setattr(log_profile, "min", fail_if_full_scan, raising=False)

  assert log_profile._nearest_car_state(samples, 42.1) == (42.0, False)


def test_build_longitudinal_profile_can_skip_sort_for_ordered_messages(monkeypatch):
  msgs = [
    msg("carState", 0.0, vEgo=0.0, vCruise=72.0, standstill=True),
    msg("radarState", 0.1, leadOne=SimpleNamespace(present=True, dRel=8.0, vRel=-1.0, vLead=0.0)),
  ]

  def fail_if_sorted(*args, **kwargs):
    raise AssertionError("ordered profile input should not be sorted again")

  monkeypatch.setattr(log_profile, "sorted", fail_if_sorted, raising=False)

  profile = build_longitudinal_profile(msgs, source="ordered", already_sorted=True)

  assert profile.source == "ordered"
