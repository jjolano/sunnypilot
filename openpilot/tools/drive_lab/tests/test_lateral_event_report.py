from types import SimpleNamespace

from openpilot.tools.drive_lab.lateral_event_report import (
  build_lateral_event_report,
  load_lateral_event_report,
  render_lateral_event_report,
  save_lateral_event_report,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return "torqueState"


class FakeEnum:
  def __init__(self, name: str):
    self.name = name


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def controls_msg(t_s, *, raw_curvature: float, processed_curvature: float | None = None,
                 steering_output: float = 0.0, actual_curvature: float | None = None,
                 quality: float = 1.0, gated: bool = False):
  processed = raw_curvature if processed_curvature is None else processed_curvature
  torque_state = SimpleNamespace(active=True, output=steering_output)
  return msg(
    "controlsState",
    t_s,
    curvature=raw_curvature * 0.95 if actual_curvature is None else actual_curvature,
    desiredCurvature=processed,
    lateralControlState=FakeUnion(torqueState=torque_state),
    modelPathState=SimpleNamespace(
      rawDesiredCurvature=raw_curvature,
      processedDesiredCurvature=processed,
      gated=gated,
      quality=quality,
    ),
  )


def sample_msgs(t_s: float, raw: float, *, processed: float | None = None, steering_angle: float | None = None,
                steering_output: float | None = None, actual: float | None = None, quality: float = 1.0,
                gated: bool = False):
  steering = -raw * 3200.0 if steering_angle is None else steering_angle
  output = -raw * 250.0 if steering_output is None else steering_output
  return [
    msg("carState", t_s, vEgo=18.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, steeringAngleDeg=steering),
    msg("carControl", t_s, latActive=True),
    msg("carOutput", t_s, actuatorsOutput=SimpleNamespace(torque=output)),
    msg("modelV2", t_s, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
    controls_msg(t_s, raw_curvature=raw, processed_curvature=processed, steering_output=output, actual_curvature=actual, quality=quality, gated=gated),
  ]


def test_lateral_event_report_classifies_distinct_lateral_events(tmp_path):
  msgs = []
  for i in range(900):
    t = i * 0.1
    raw = 0.0
    processed = None
    steering = None
    output = None
    actual = None
    quality = 1.0
    gated = False
    if 5.0 <= t < 40.0:
      raw = 0.0012 if int((t - 5.0) // 4.0) % 2 == 0 else -0.0012
    elif 45.0 <= t < 47.0:
      raw = 0.0009
      processed = 0.00025
      actual = 0.00005
      steering = -0.15
    elif 47.25 <= t < 51.0:
      raw = 0.0023 if int((t - 47.25) * 3.0) % 2 == 0 else -0.0023
      processed = raw
      actual = raw * 1.1
      steering = -7.0 if raw > 0.0 else 7.0
    elif 60.0 <= t < 68.0:
      raw = 0.00025 if int((t - 60.0) * 5.0) % 2 == 0 else -0.00025
      steering = 0.8 if int((t - 60.0) * 5.0) % 2 == 0 else -0.8
      output = 0.08 if raw > 0 else -0.08
      actual = raw * 0.9
    msgs.extend(sample_msgs(t, raw, processed=processed, steering_angle=steering, steering_output=output, actual=actual, quality=quality, gated=gated))

  report = build_lateral_event_report(msgs, source="synthetic", max_events=10)
  path = tmp_path / "lateral-events.json"
  save_lateral_event_report(report, path)
  loaded = load_lateral_event_report(path)
  rendered = render_lateral_event_report(loaded)
  kinds = {event.kind for event in loaded.top_events}

  assert "slow_wander" in kinds
  assert "rebound" in kinds
  assert "fast_reversal" in kinds
  assert loaded.sample_count > 0
  assert "Lateral event report" in rendered


def test_lateral_event_report_filters_inactive_samples():
  msgs = []
  for i in range(20):
    t = i * 0.1
    msgs.extend([
      msg("carState", t, vEgo=18.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, steeringAngleDeg=10.0),
      msg("carControl", t, latActive=False),
      msg("carOutput", t, actuatorsOutput=SimpleNamespace(torque=0.0)),
      msg("modelV2", t, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
      controls_msg(t, raw_curvature=0.001),
    ])

  report = build_lateral_event_report(msgs)

  assert report.active_percent == 0.0
  assert not report.top_events
