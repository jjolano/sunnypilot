from types import SimpleNamespace

from openpilot.tools.drive_lab.profile_corner_recovery import (
  DetectorParams,
  analyze_route,
  extract_samples,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def build_stream(approach_curv, recover_curv, *, stop_s=3.0, steering_on_launch=False,
                 steering_on_approach=False, engaged=True):
  """Synthesise a route: 2 s approach at 10 m/s on a curve, a stop, then a 3 s launch+recover.

  Each 0.05 s tick emits carState, carControl, modelV2, then controlsState (so the controlsState
  trigger sees the same-tick state as the latest values), matching the live wiring. Actual curvature
  tracks the command, so a continuing curve (recover_curv != 0) is confirmable.
  """
  msgs = []
  clock = [0.0]

  def emit(v, cmd, model, act, standstill, *, lat_active=True, steering=False, lane_change="off"):
    t = clock[0]
    msgs.append(msg("carState", t, vEgo=v, standstill=standstill, steeringPressed=steering))
    msgs.append(msg("carControl", t + 0.0005, latActive=lat_active))
    msgs.append(msg("modelV2", t + 0.001,
                    action=SimpleNamespace(desiredCurvature=model),
                    meta=SimpleNamespace(laneChangeState=lane_change)))
    msgs.append(msg("controlsState", t + 0.002, desiredCurvature=cmd, curvature=act))
    clock[0] += 0.05

  # Approach the curve, moving and confident (cmd == actual == the curve).
  for _ in range(40):
    emit(10.0, approach_curv, approach_curv, approach_curv, False, lat_active=engaged, steering=steering_on_approach)
  # Stop mid-corner: speed zero, standstill set (lat deactivates), command collapses (the "amnesia").
  for _ in range(int(stop_s / 0.05)):
    emit(0.0, approach_curv * 0.05, 0.0, 0.0, True, lat_active=False)
  # Launch: speed ramps; command sits near zero for 0.6 s (forgot), then snaps to recover_curv. The
  # car actually follows the command (act == cmd), so a non-zero recover_curv is a continuing curve.
  tt = 0.0
  for _ in range(60):
    v = 0.5 + 6.0 * tt
    cmd = 0.002 if tt < 0.6 else recover_curv
    steering = steering_on_launch and tt < 1.0
    emit(v, cmd, cmd, cmd, False, lat_active=engaged, steering=steering)
    tt += 0.05
  return msgs


def test_stopped_mid_corner_is_flagged_as_amnesia():
  report = analyze_route(build_stream(0.02, 0.02), source="amnesia")

  assert report.total_stops == 1
  assert report.corner_stops == 1
  assert report.amnesia_events == 1
  assert len(report.worst_events) == 1

  event = report.worst_events[0]
  assert event.classification == "amnesia"
  assert event.is_amnesia
  assert event.pre_stop_curv > 0.015           # Z: knew the curve
  assert abs(event.launch_curv) < 0.006         # X: forgot it on launch (command)
  assert abs(event.launch_curv_vision) < 0.006  # X_vis: perception forgot it too
  assert event.continuation_curv > 0.01         # the curve actually continued
  assert event.deficit > 0.01
  assert event.deficit_frac > 0.7
  assert event.deficit_vision > 0.01            # perception amnesia, isolated
  assert 0.0 < event.recovery_lag_s < 2.0
  assert event.recovery_lag_m > 0.0
  assert not event.recovery_censored


def test_straight_stop_is_not_a_corner():
  report = analyze_route(build_stream(0.0, 0.0), source="straight")

  assert report.total_stops == 1
  assert report.corner_stops == 0
  assert report.amnesia_events == 0
  assert report.worst_events == []


def test_curve_that_ends_during_stop_is_not_amnesia():
  # Held a curve on approach, but the road is straight after the stop — legitimate, not forgetting.
  report = analyze_route(build_stream(0.02, 0.0), source="curve_ended")

  assert report.corner_stops == 1
  assert report.amnesia_events == 0
  assert report.curve_ended == 1


def test_driver_corrected_launch_is_amnesia():
  # Driver grabs the wheel at launch (correcting the under-steer) — still amnesia, flagged corrected.
  report = analyze_route(build_stream(0.02, 0.02, steering_on_launch=True), source="corrected")

  assert report.corner_stops == 1
  assert report.amnesia_events == 1
  assert report.amnesia_driver_corrected == 1
  assert "driver_corrected" in report.worst_events[0].flags


def test_human_driven_approach_is_not_attributed():
  # Human steered the approach — openpilot never had the corner, so it is not scored as amnesia.
  report = analyze_route(build_stream(0.02, 0.02, steering_on_approach=True), source="manual_corner")

  assert report.corner_stops == 1
  assert report.amnesia_events == 0
  assert report.manual_corner == 1


def test_manual_approach_is_ignored():
  # Same corner stop, but openpilot was not steering — must not be scored at all.
  report = analyze_route(build_stream(0.02, 0.02, engaged=False), source="manual")

  assert report.corner_stops == 0
  assert report.amnesia_events == 0


def test_no_controls_state_returns_note():
  msgs = [msg("carState", 0.0, vEgo=0.0, standstill=True)]
  report = analyze_route(msgs, source="empty")

  assert report.sample_count == 0
  assert report.total_stops == 0
  assert any("controlsState" in note for note in report.notes)


def test_extract_samples_aligns_latest_services():
  msgs = [
    msg("carState", 0.0, vEgo=3.0, standstill=False, steeringPressed=False),
    msg("carControl", 0.0005, latActive=True),
    msg("modelV2", 0.001, action=SimpleNamespace(desiredCurvature=0.01),
        meta=SimpleNamespace(laneChangeState="off")),
    msg("controlsState", 0.002, desiredCurvature=0.011, curvature=0.009),
  ]
  samples, saw_carcontrol = extract_samples(msgs)

  assert saw_carcontrol
  assert len(samples) == 1
  sample = samples[0]
  assert sample.v_ego == 3.0
  assert sample.lat_active
  assert sample.cmd_curv == 0.011
  assert sample.act_curv == 0.009
  assert sample.model_curv == 0.01


def test_params_are_overridable():
  # With a high corner threshold, the 0.02 curve no longer counts as a corner.
  report = analyze_route(build_stream(0.02, 0.02), params=DetectorParams(kappa_corner=0.05))
  assert report.corner_stops == 0
