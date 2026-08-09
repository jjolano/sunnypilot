from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_launch_delays import analyze_route


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def build_stream(*, engaged=True, lead=False, long_active=None, gas_pressed=False, brake_pressed=False):
  """Cruise, decelerate to a near-stop, then launch away — one recoverable launch event."""
  if long_active is None:
    long_active = engaged
  msgs = []
  clock = [0.0]

  def emit(v, a, should_stop=False, a_target=0.0):
    t = clock[0]
    msgs.append(msg("carState", t, vEgo=v, aEgo=a, gasPressed=gas_pressed, brakePressed=brake_pressed))
    msgs.append(msg("selfdriveState", t + 0.001, enabled=engaged))
    msgs.append(msg("carControl", t + 0.0015, longActive=long_active))
    d_rel = 8.0 if lead else 250.0
    msgs.append(msg("radarState", t + 0.002, leadOne=SimpleNamespace(dRel=d_rel, vLead=v, present=lead)))
    msgs.append(msg("longitudinalPlanSP", t + 0.003, aTarget=a_target, customLongitudinal=SimpleNamespace(shouldStop=should_stop)))
    clock[0] += 0.1

  for _ in range(20):
    emit(8.0, 0.0)  # cruising above threshold
  for _ in range(10):
    emit(0.2, -0.5, should_stop=True, a_target=-0.5)  # near-stop dip
  for i in range(30):
    emit(0.2 + 0.3 * i, 1.0, should_stop=False, a_target=1.0)  # launch away
  return msgs


def build_following_launch(*, lead_moves: bool):
  msgs = []
  clock = [0.0]

  def emit(v, a, lead_v, lead_status=True, should_stop=True, a_target=0.0):
    t = clock[0]
    msgs.append(msg("carState", t, vEgo=v, aEgo=a))
    msgs.append(msg("selfdriveState", t + 0.001, enabled=True))
    msgs.append(msg("radarState", t + 0.002, leadOne=SimpleNamespace(dRel=8.0, vLead=lead_v, present=lead_status)))
    msgs.append(msg("longitudinalPlanSP", t + 0.003, aTarget=a_target, customLongitudinal=SimpleNamespace(shouldStop=should_stop)))
    clock[0] += 0.1

  for _ in range(10):
    emit(0.2, -0.1, 0.0)
  for _ in range(100):
    emit(0.2, -0.1, 0.0 if not lead_moves else 0.0)
  for i in range(10):
    lead_v = 0.0 if not lead_moves else 1.0
    ego_v = 0.2 if i < 3 else 0.5 + 0.2 * i
    ego_a = 0.1 if i < 3 else 0.3
    # First shouldStop=False sample is still holding a negative target; planner release should wait
    # until the target is near-zero/positive.
    should_stop = not (lead_moves and i >= 1)
    a_target = 0.3 if lead_moves and i >= 2 else -0.2
    emit(ego_v, ego_a, lead_v, should_stop=should_stop, a_target=a_target)
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


def test_lead_wait_and_reaction_are_split():
  report = analyze_route(build_following_launch(lead_moves=True), source="lead-launch")
  event = report.events[0]
  assert event.lead_wait_time is not None and event.lead_wait_time >= 9.5
  assert event.reaction_time is not None and event.reaction_time < 1.0
  assert event.planner_release_time is not None
  assert event.planner_release_delay is not None and 0.1 < event.planner_release_delay < 0.4
  assert event.ego_start_time is not None
  assert event.ego_start_delay is not None
  assert event.ego_start_delay == event.reaction_time
  assert report.op_lead_median_reaction_s is not None
  assert report.op_lead_median_planner_release_s is not None
  assert report.op_lead_median_ego_start_s is not None


def test_no_lead_motion_leaves_reaction_empty():
  report = analyze_route(build_following_launch(lead_moves=False), source="lead-stopped")
  event = report.events[0]
  assert event.lead_move_time is None
  assert event.reaction_time is None
  assert event.planner_release_time is None
  assert event.planner_release_delay is None
  assert event.ego_start_time is None
  assert event.ego_start_delay is None


def test_no_carstate_returns_note():
  report = analyze_route([msg("selfdriveState", 0.0, enabled=True)], source="empty")
  assert report.total_events == 0
  assert any("carState" in note for note in report.notes)


def test_driver_override_prefers_long_active_over_enabled():
  report = analyze_route(build_stream(engaged=True, long_active=False, gas_pressed=True), source="override")
  assert report.total_events == 1
  assert report.op_engaged_events == 0
  assert report.manual_events == 1
  event = report.events[0]
  assert event.enabled
  assert not event.long_active
  assert event.driver_override
  assert not event.op_engaged
