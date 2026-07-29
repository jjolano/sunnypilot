from types import SimpleNamespace

from openpilot.tools.drive_lab.replay_cut_in_advisory import analyze_route


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _lead(status=True, dRel=15.0, yRel=0.0, vRel=-3.0, vLead=8.0, vLeadK=8.0,
          aLeadK=0.0, radarTrackId=42, modelProb=0.9, radar=True):
  return SimpleNamespace(status=status, dRel=dRel, yRel=yRel, vRel=vRel, vLead=vLead,
                         vLeadK=vLeadK, aLeadK=aLeadK, radarTrackId=radarTrackId,
                         modelProb=modelProb, radar=radar)


def build_cut_in_scenario(*, op_engaged: bool, mpc_brake_delay_s: float):
  """A new lead cuts in with closing speed; MPC brakes after a delay."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead=None, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    if lead is not None:
      msgs.append(msg("radarState", t[0] + 0.002, leadOne=lead, leadTwo=None))
    if op_engaged and a_target is not None:
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=a_target))
    t[0] += 0.1

  # No lead, cruising
  for _ in range(10):
    emit(12.0, 0.0)

  # Cut-in: new lead appears with closing speed
  brake_step = int(mpc_brake_delay_s / 0.1)
  for i in range(40):
    lead = _lead(dRel=20.0 - 0.3 * i, vRel=-3.0, vLead=9.0, radarTrackId=42, yRel=0.5)
    if i >= brake_step:
      emit(12.0, -0.5, lead, a_target=-0.5)
    else:
      emit(12.0, 0.0, lead, a_target=0.0)

  return msgs


def build_track_swap_scenario():
  """Same object gets a new radarTrackId — should NOT fire advisory."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead=None, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=True))
    if lead is not None:
      msgs.append(msg("radarState", t[0] + 0.002, leadOne=lead, leadTwo=None))
    msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=a_target or 0.0))
    t[0] += 0.1

  # Following a lead
  for i in range(15):
    emit(10.0, 0.0, _lead(dRel=12.0, vRel=0.0, vLead=10.0, radarTrackId=7, yRel=0.0))

  # Same object, new ID (track swap)
  for i in range(10):
    emit(10.0, 0.0, _lead(dRel=12.5, vRel=0.0, vLead=10.0, radarTrackId=99, yRel=0.0))

  return msgs


def build_off_path_scenario():
  """Lead in adjacent lane — should NOT fire advisory."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead=None, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=True))
    if lead is not None:
      msgs.append(msg("radarState", t[0] + 0.002, leadOne=lead, leadTwo=None))
    msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=a_target or 0.0))
    t[0] += 0.1

  # Lead far off-path
  for i in range(20):
    emit(12.0, 0.0, _lead(dRel=15.0, vRel=-3.0, vLead=9.0, radarTrackId=42, yRel=3.0))

  return msgs


def test_cut_in_advisory_fires_before_mpc():
  report = analyze_route(build_cut_in_scenario(op_engaged=True, mpc_brake_delay_s=1.0), source="test")
  assert len(report.events) > 0
  event = report.events[0]
  assert event.stage in ("suspect", "confirmed")
  assert event.advisory_a < 0.0
  # Advisory should fire before MPC brakes
  assert event.timing_advantage_s is not None
  assert event.timing_advantage_s > 0.0
  assert not event.false_positive


def test_track_swap_does_not_fire():
  report = analyze_route(build_track_swap_scenario(), source="swap")
  # Track swap should be rejected — no events
  assert len(report.events) == 0
  assert report.rejection_reasons.get("track_swap", 0) > 0


def test_off_path_does_not_fire():
  report = analyze_route(build_off_path_scenario(), source="off-path")
  assert len(report.events) == 0
  # Should be rejected by path plausibility gate
  assert report.rejection_reasons.get("off_path", 0) > 0


def test_false_positive_when_mpc_never_brakes():
  """Advisory fires but MPC never brakes within window."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead=None, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=True))
    if lead is not None:
      msgs.append(msg("radarState", t[0] + 0.002, leadOne=lead, leadTwo=None))
    msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=a_target or 0.0))
    t[0] += 0.1

  for _ in range(10):
    emit(12.0, 0.0)
  for i in range(40):
    emit(12.0, 0.0, _lead(dRel=20.0 - 0.3 * i, vRel=-3.0, vLead=9.0, radarTrackId=42, yRel=0.5), a_target=0.0)

  report = analyze_route(msgs, source="fp")
  assert len(report.events) > 0
  assert report.events[0].false_positive is True


def test_no_radar_returns_note():
  report = analyze_route([msg("carState", 0.0, vEgo=10.0, aEgo=0.0)], source="empty")
  assert len(report.notes) > 0


def test_report_json_serializable():
  import json
  report = analyze_route(build_cut_in_scenario(op_engaged=True, mpc_brake_delay_s=1.0), source="json")
  d = report.to_dict()
  json.dumps(d)


def test_render_report_does_not_crash():
  report = analyze_route(build_cut_in_scenario(op_engaged=True, mpc_brake_delay_s=1.0), source="render")
  from openpilot.tools.drive_lab.replay_cut_in_advisory import render_report
  text = render_report(report)
  assert "Cut-in advisory replay" in text
