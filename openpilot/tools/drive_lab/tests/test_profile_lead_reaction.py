from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_lead_reaction import analyze_route


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _lead(status=True, dRel=10.0, vLead=5.0, vRel=0.0, aLeadK=0.0, radarTrackId=7, modelProb=0.9):
  return SimpleNamespace(present=status, dRel=dRel, vLead=vLead, vLeadK=vLead, vRel=vRel, aLeadK=aLeadK, radarTrackId=radarTrackId, modelProb=modelProb)


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


def build_lead_accel_to_decel_reaction(*, op_engaged: bool, braking_at_event: bool):
  """Lead transitions from accel to decel; ego may already be braking at the event."""
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

  # Steady cruise with lead
  for _ in range(10):
    emit(10.0, 0.0, 10.0, 0.0, a_target=0.0)

  # Lead accelerates first to establish a positive sign
  for _ in range(3):
    emit(10.0, 0.0, 10.5, 1.0, a_target=0.0)

  # Next positive-accel samples pre-arm the planner target if we want already-responding.
  for _ in range(3):
    at = -0.5 if braking_at_event else 0.0
    emit(10.0, 0.0, 10.5, 1.0, a_target=at)

  # Lead decelerates — direction change detected here
  for i in range(30):
    if braking_at_event:
      at = -0.5
    else:
      at = 0.0 if i < 3 else -0.5
    emit(10.0, 0.0, 9.5 + 0.1 * i, -1.0, a_target=at)

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


def build_cut_in_with_lead_id_churn(*, n_churn: int = 5):
  """Repeated radar lead-id changes inside the cluster window should count as one cut-in."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=-2.0, a_target=0.0):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=True))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=a_target))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False)

  # Initial cut-in, stable for two samples before churn starts
  emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.0)
  emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.0)

  # Rapid lead-id churn inside the cluster window
  for i in range(n_churn):
    emit(12.0, 0.0, True, lead_id=100 + i, d_rel=15.0, v_rel=-2.0, a_target=0.0)

  # Stable lead afterward
  for _ in range(10):
    emit(12.0, -0.5, True, lead_id=200, d_rel=15.0, v_rel=-2.0, a_target=-0.5)

  return msgs


def build_cut_in_already_braking(*, op_engaged: bool = True):
  """Cut-in appears while the ego is already braking; should not count as a reaction event."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=-2.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead, already braking
  for _ in range(10):
    emit(12.0, -0.5, False, a_target=-0.5)

  # Cut-in appears while already braking
  for _ in range(20):
    emit(12.0, -0.5, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=-0.5)

  return msgs


def build_soft_cut_in(*, op_engaged: bool = True):
  """Lead appears nearly co-speed (soft following); not a real closing cut-in."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=0.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False, a_target=0.0)

  # Soft "cut-in" with near-zero closing velocity
  for _ in range(20):
    emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-0.1, a_target=0.0)

  return msgs


def build_distant_cut_in(*, op_engaged: bool = True):
  """Lead closes slowly from far away; TTC too long to count as an immediate cut-in."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=0.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False, a_target=0.0)

  # Closing lead is far enough that TTC exceeds the validity window
  for _ in range(20):
    emit(12.0, 0.0, True, lead_id=42, d_rel=30.0, v_rel=-2.0, a_target=0.0)

  return msgs


def build_cut_in_repeated_detections(*, op_engaged: bool = True):
  """Two detected cut-ins 2s apart should collapse into one summary cluster."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=-2.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False, a_target=0.0)

  # First cut-in, then brief disappearance (>1s from last True to re-detect)
  for _ in range(11):
    emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.0)
  for _ in range(10):
    emit(12.0, 0.0, False, a_target=0.0)
  for _ in range(15):
    emit(12.0, -0.5, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=-0.5)

  return msgs


def build_cut_in_no_brake_response(*, op_engaged: bool = True):
  """Plausible cut-in, but OP/manual never brakes; valid cluster but no reaction metric."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=-2.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # No lead
  for _ in range(10):
    emit(12.0, 0.0, False, a_target=0.0)

  # Cut-in with no ego braking response
  for _ in range(25):
    emit(12.0, 0.0, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.0)

  return msgs


def build_cut_in_with_positive_accel(*, op_engaged: bool = True):
  """Plausible cut-in but ego accelerates; peak decel is positive and must not be median'd."""
  msgs = []
  t = [0.0]

  def emit(v, a, lead_status, lead_id=7, d_rel=15.0, v_rel=-2.0, a_target=None):
    msgs.append(msg("carState", t[0], vEgo=v, aEgo=a, brakePressed=False, gasPressed=False))
    msgs.append(msg("carControl", t[0] + 0.001, longActive=op_engaged))
    msgs.append(msg("radarState", t[0] + 0.002, leadOne=_lead(status=lead_status, dRel=d_rel, vLead=v + v_rel, vRel=v_rel, radarTrackId=lead_id)))
    if op_engaged:
      at = a_target if a_target is not None else a
      msgs.append(msg("longitudinalPlanSP", t[0] + 0.003, aTarget=at))
    t[0] += 0.1

  # Establish a mildly positive baseline so peak_decel stays positive
  for _ in range(10):
    emit(12.0, 0.1, False, a_target=0.1)

  # Cut-in, but ego accelerates instead of braking
  for _ in range(25):
    emit(12.0, 0.5, True, lead_id=42, d_rel=15.0, v_rel=-2.0, a_target=0.5)

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


def test_cut_in_lead_id_churn_clustered_to_one_event():
  report = analyze_route(build_cut_in_with_lead_id_churn(n_churn=5), source="churn")
  assert len(report.cut_ins) == 1
  assert report.cut_ins[0].lead_id == 42
  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_count"] == 1
  assert summary["op_cut_in_valid_count"] == 1
  assert summary["op_cut_in_cluster_count"] == 1
  assert summary["op_cut_in_valid_cluster_count"] == 1


def test_cut_in_already_braking_excluded_from_valid_reaction():
  report = analyze_route(build_cut_in_already_braking(op_engaged=True), source="already-braking")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.op_engaged
  assert event.already_braking
  assert not event.valid_reaction
  assert event.brake_reaction_time is None

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_count"] == 1
  assert summary["op_cut_in_valid_count"] == 0
  assert summary["op_cut_in_already_braking_count"] == 1
  assert summary["op_cut_in_brake_median_s"] is None
  assert summary["op_cut_in_peak_decel_median"] is None


def test_cut_in_immediate_response_not_classified_as_already_braking():
  report = analyze_route(build_cut_in(op_engaged=True, brake_delay_s=0.0), source="immediate-cutin")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.op_engaged
  assert not event.already_braking
  assert event.valid_reaction
  assert event.brake_reaction_time is not None
  assert event.brake_reaction_time < 0.2

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_valid_count"] == 1
  assert summary["op_cut_in_already_braking_count"] == 0


def test_soft_cut_in_is_invalid_reaction():
  report = analyze_route(build_soft_cut_in(op_engaged=True), source="soft-cutin")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.op_engaged
  assert not event.valid_reaction

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_count"] == 1
  assert summary["op_cut_in_valid_count"] == 0
  assert summary["op_cut_in_valid_cluster_count"] == 0
  assert summary["op_cut_in_brake_median_s"] is None


def test_distant_cut_in_is_invalid_reaction():
  report = analyze_route(build_distant_cut_in(op_engaged=True), source="distant-cutin")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.op_engaged
  assert not event.valid_reaction

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_count"] == 1
  assert summary["op_cut_in_valid_count"] == 0
  assert summary["op_cut_in_valid_cluster_count"] == 0
  assert summary["op_cut_in_brake_median_s"] is None


def test_cut_in_repeated_detections_form_one_summary_cluster():
  report = analyze_route(build_cut_in_repeated_detections(op_engaged=True), source="repeated")
  assert len(report.cut_ins) == 2
  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_count"] == 2
  assert summary["op_cut_in_cluster_count"] == 1
  assert summary["op_cut_in_valid_cluster_count"] == 1


def test_cut_in_cluster_without_brake_response_has_no_median():
  report = analyze_route(build_cut_in_no_brake_response(op_engaged=True), source="no-brake")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.valid_reaction
  assert event.brake_reaction_time is None

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_valid_cluster_count"] == 1
  assert summary["op_cut_in_brake_median_s"] is None
  assert summary["op_cut_in_peak_decel_median"] is None


def test_cut_in_positive_peak_decel_not_in_median():
  report = analyze_route(build_cut_in_with_positive_accel(op_engaged=True), source="positive-peak")
  assert len(report.cut_ins) == 1
  event = report.cut_ins[0]
  assert event.valid_reaction
  assert event.brake_reaction_time is None
  assert event.peak_decel is None or event.peak_decel >= 0

  summary = report.to_dict()["summary"]
  assert summary["op_cut_in_valid_cluster_count"] == 1
  assert summary["op_cut_in_peak_decel_median"] is None


def test_op_decel_reaction_ignores_already_responding():
  report = analyze_route(build_lead_accel_to_decel_reaction(op_engaged=True, braking_at_event=True), source="already-resp")
  decel_events = [r for r in report.op_reactions if r.lead_change.direction == "accel_to_decel"]
  assert len(decel_events) == 1
  event = decel_events[0]
  assert event.already_responding
  assert not event.valid_reaction
  assert event.reaction_time is None

  summary = report.to_dict()["summary"]
  assert summary["op_already_responding_count"] == 1
  assert summary["op_reaction_count"] == 0
  assert summary["op_reaction_median_s"] is None
