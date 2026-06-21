from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_lead_reaction import analyze_route


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _lead(status=True, dRel=10.0, vLead=5.0, vRel=0.0, aLeadK=0.0, radarTrackId=7, modelProb=0.9):
  return SimpleNamespace(status=status, dRel=dRel, vLead=vLead, vLeadK=vLead, vRel=vRel,
                         aLeadK=aLeadK, radarTrackId=radarTrackId, modelProb=modelProb)


def build_lead_decel_reaction(*, op_engaged: bool, reaction_delay_s: float):
  """Lead decelerates then accelerates; ego reacts after a delay from the direction change."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_v, lead_a, a_target=None, lead_id=7, lead_status=True):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(vLead=lead_v, aLeadK=lead_a, radarTrackId=lead_id, status=lead_status)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # Steady cruise with lead (aLeadK=0, below threshold)
  for _ in range(10):
    emit(10.0, 0.0, 10.0, 0.0)

  # Lead decelerates (aLeadK negative, sustained > 0.3s)
  for _ in range(6):
    emit(10.0, 0.0, 9.5, -1.0)

  # Lead accelerates (aLeadK positive) — direction change detected here
  # Ego reacts after reaction_delay_s from this point
  react_step = int(reaction_delay_s / 0.1)
  for i in range(30):
    lead_a = 1.0
    if i >= react_step:
      emit(10.0 + 0.1 * (i - react_step), 0.5, 10.5 + 0.1 * i, lead_a, a_target=0.5)
    else:
      emit(10.0, 0.0, 10.5 + 0.1 * i, lead_a, a_target=0.0)

  return msgs


def build_lead_exit(*, op_engaged: bool, accel_delay_s: float):
  """Lead disappears; ego accelerates after a delay from the exit."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, a_target=None):
    d_rel = 12.0 if lead_status else 0.0
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # Following a lead
  for _ in range(15):
    emit(10.0, 0.0, True)

  # Lead exits — ego accelerates after delay from dropout start
  react_step = int(accel_delay_s / 0.1)
  for i in range(20):
    if i >= react_step:
      emit(10.0 + 0.1 * (i - react_step), 0.5, False, a_target=0.5)
    else:
      emit(10.0, 0.0, False, a_target=0.0)

  return msgs


def build_cut_in(*, op_engaged: bool, brake_delay_s: float):
  """New lead cuts in close; ego brakes after a delay."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=50.0, v_rel=0.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False)

  # Cut-in: new lead appears close
  react_step = int(brake_delay_s / 0.1)
  for i in range(30):
    if i >= react_step:
      emit(12.0, -0.5, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=-0.5)
    else:
      emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.0)

  return msgs


def build_creep_while_braked():
  """brakePressed + vEgo > creep_speed should be counted as creep, not stopped."""
  msgs = []
  t = [0.0]

  def emit(v, a, brake):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=brake, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=False))
    t[0] += 0.1

  # Creeping with brake pressed at low speed
  for _ in range(20):
    emit(0.5, -0.1, True)
  return msgs


def test_lead_decel_to_accel_reaction_measured():
  report = analyze_route(build_lead_decel_reaction(op_engaged=True, reaction_delay_s=0.5), source="op")
  assert len(report.lead_speed_changes) > 0
  # Should find at least one accel_to_decel and one decel_to_accel
  directions = [c.direction for c in report.lead_speed_changes]
  assert "decel_to_accel" in directions
  op_reactions = [r for r in report.op_reactions if r.reaction_time is not None]
  assert len(op_reactions) > 0
  # Reaction should be around 0.5s (plus detection overhead from sustained_duration gate)
  rt = op_reactions[0].reaction_time
  assert rt is not None and 0.3 <= rt <= 1.2


def test_manual_reaction_uses_aego():
  report = analyze_route(build_lead_decel_reaction(op_engaged=False, reaction_delay_s=0.4), source="manual")
  manual_reactions = [r for r in report.manual_reactions if r.reaction_time is not None]
  assert len(manual_reactions) > 0
  rt = manual_reactions[0].reaction_time
  assert rt is not None and 0.2 <= rt <= 1.0


def test_lead_exit_accel_measured():
  report = analyze_route(build_lead_exit(op_engaged=True, accel_delay_s=0.6), source="op-exit")
  assert len(report.lead_exits) > 0
  op_exits = [e for e in report.lead_exits if e.op_engaged and e.accel_reaction_time is not None]
  assert len(op_exits) > 0
  rt = op_exits[0].accel_reaction_time
  assert rt is not None and 0.4 <= rt <= 1.2


def test_cut_in_brake_measured():
  report = analyze_route(build_cut_in(op_engaged=True, brake_delay_s=0.3), source="op-cutin")
  assert len(report.cut_ins) > 0
  op_cutins = [e for e in report.cut_ins if e.op_engaged and e.brake_reaction_time is not None]
  assert len(op_cutins) > 0
  rt = op_cutins[0].brake_reaction_time
  assert rt is not None and 0.1 <= rt <= 0.6


def test_creep_filtering_counts_brake_at_low_speed():
  report = analyze_route(build_creep_while_braked(), source="creep")
  assert report.creep_filtered_samples > 0
  assert report.manual_moving_s > 0


def test_no_radar_returns_note():
  report = analyze_route([msg("carState", 0.0, vEgo=10.0, aEgo=0.0)], source="empty")
  assert len(report.notes) > 0
  assert report.duration_s == 0.0


def test_report_json_serializable():
  import json
  report = analyze_route(build_lead_decel_reaction(op_engaged=True, reaction_delay_s=0.5), source="json")
  d = report.to_dict()
  json.dumps(d)  # must not raise


def test_render_report_does_not_crash():
  report = analyze_route(build_lead_decel_reaction(op_engaged=True, reaction_delay_s=0.5), source="render")
  from openpilot.tools.drive_lab.profile_lead_reaction import render_report
  text = render_report(report)
  assert "Lead-reaction profile" in text
