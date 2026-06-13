from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_launch_delays import analyze_route


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def build_stream(*, engaged=True, lead=False):
  """Cruise, decelerate to a near-stop, then launch away — one recoverable launch event."""
  msgs = []
  clock = [0.0]

  def emit(v, a):
    t = clock[0]
    msgs.append(msg("carState", t, vEgo=v, aEgo=a))
    msgs.append(msg("selfdriveState", t + 0.001, enabled=engaged))
    d_rel = 8.0 if lead else 250.0
    msgs.append(msg("radarState", t + 0.002, leadOne=SimpleNamespace(dRel=d_rel, vLead=v, status=lead)))
    clock[0] += 0.1

  for _ in range(20):
    emit(8.0, 0.0)          # cruising above threshold
  for _ in range(10):
    emit(0.2, -0.5)         # near-stop dip
  for i in range(30):
    emit(0.2 + 0.3 * i, 1.0)  # launch away
  return msgs


def test_detects_engaged_launch():
  report = analyze_route(build_stream(engaged=True, lead=False), source="launch")
  assert report.total_events == 1
  assert report.op_engaged_events == 1
  assert report.op_nolead_events == 1
  event = report.events[0]
  assert event.min_speed < 2.0
  assert event.recovery_time > 0.0
  assert event.accel_peak > 0.0


def test_lead_split_and_manual():
  assert analyze_route(build_stream(engaged=True, lead=True)).op_lead_events == 1
  manual = analyze_route(build_stream(engaged=False))
  assert manual.total_events == 1
  assert manual.op_engaged_events == 0
  assert manual.manual_events == 1


def test_no_carstate_returns_note():
  report = analyze_route([msg("selfdriveState", 0.0, enabled=True)], source="empty")
  assert report.total_events == 0
  assert any("carState" in note for note in report.notes)
