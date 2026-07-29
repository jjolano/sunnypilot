import numpy as np

from openpilot.tools.drive_lab.lateral_comfort_imu import ComfortSeries, find_events, jerks, summarize


def _burst(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
  return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _series() -> ComfortSeries:
  t = np.arange(0.0, 60.0, 0.05)  # 20 Hz, one minute
  v = np.full_like(t, 15.0)
  # control-caused event at t=20: commanded lat accel swings and the body follows.
  cmd = _burst(t, 20.0, 0.4, 1.5)
  meas = _burst(t, 20.05, 0.4, 1.4)
  # road-caused event at t=40: the body jerks while the command stays flat.
  meas = meas + _burst(t, 40.0, 0.3, 1.6)
  return ComfortSeries(t=t, measured_lat_accel=meas, commanded_lat_accel=cmd,
                       v_ego=v, mask=np.ones_like(t, dtype=bool))


def test_events_found_and_attributed():
  events = find_events(_series(), top=4)
  assert len(events) >= 2
  by_time = {round(e.t / 10) * 10: e for e in events}
  assert by_time[20].label == "control"
  assert by_time[20].correlation > 0.9
  assert by_time[40].label == "road/disturbance"
  assert abs(by_time[40].commanded_jerk) < 0.2


def test_summary_and_jerk_shapes():
  s = _series()
  meas_jerk, cmd_jerk = jerks(s)
  assert meas_jerk.shape == s.t.shape and cmd_jerk.shape == s.t.shape
  stats = summarize(s)
  assert stats["masked_duration_s"] > 55.0
  assert stats["measured_jerk_max"] > stats["measured_jerk_p95"] > stats["measured_jerk_p50"] >= 0.0
  # quiet outside the two bursts -> p50 near zero
  assert stats["measured_jerk_p50"] < 0.05


def test_empty_mask_degrades():
  s = _series()
  empty = ComfortSeries(t=s.t, measured_lat_accel=s.measured_lat_accel,
                        commanded_lat_accel=s.commanded_lat_accel, v_ego=s.v_ego,
                        mask=np.zeros_like(s.t, dtype=bool))
  assert summarize(empty) == {"masked_duration_s": 0.0}
  assert find_events(empty) == []
