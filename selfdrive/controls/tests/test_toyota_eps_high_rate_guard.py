import sys
import types

import pytest
from cereal import car
from opendbc.car.toyota.values import CAR as TOYOTA

msgq = types.ModuleType("msgq")
msgq.fake_event_handle = object()
msgq.drain_sock_raw = lambda *args, **kwargs: []
msgq.MultiplePublishersError = RuntimeError
msgq.IpcError = RuntimeError
msgq.Context = object
msgq.Poller = object
msgq.SubSocket = object
msgq.PubSocket = object
msgq.SocketEventHandle = object
msgq.toggle_fake_events = lambda *args, **kwargs: None
msgq.set_fake_prefix = lambda *args, **kwargs: None
msgq.get_fake_prefix = lambda *args, **kwargs: ""
msgq.delete_fake_prefix = lambda *args, **kwargs: None
msgq.wait_for_one_event = lambda *args, **kwargs: None
msgq.pub_sock = lambda *args, **kwargs: None
msgq.sub_sock = lambda *args, **kwargs: None
msgq.context = None
sys.modules.setdefault("msgq", msgq)

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from openpilot.selfdrive.controls.controlsd import (  # noqa: E402
  TOYOTA_EPS_HIGH_RATE_CUT_FRAMES,
  TOYOTA_EPS_HIGH_RATE_FRAMES,
  apply_toyota_eps_high_rate_guard,
)


def make_cp(car_fingerprint=TOYOTA.TOYOTA_RAV4, steer_control_type=car.CarParams.SteerControlType.torque):
  cp = car.CarParams.new_message()
  cp.carFingerprint = car_fingerprint
  cp.steerControlType = steer_control_type
  return cp


def make_cc(lat_active=True, torque=0.4):
  cc = car.CarControl.new_message()
  cc.latActive = lat_active
  cc.actuators.torque = torque
  return cc


def make_cs(steering_rate=0.0, steer_fault_temporary=False, steer_fault_permanent=False):
  cs = car.CarState.new_message()
  cs.steeringRateDeg = steering_rate
  cs.steerFaultTemporary = steer_fault_temporary
  cs.steerFaultPermanent = steer_fault_permanent
  return cs


def test_toyota_high_rate_guard_cuts_after_threshold_without_resetting_beforehand():
  cp = make_cp()
  cc = make_cc()
  cs = make_cs(steering_rate=140.0)
  high_rate_frames = 0
  cut_frames = 0

  for _ in range(TOYOTA_EPS_HIGH_RATE_FRAMES - 1):
    high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(cp, cc, cs, high_rate_frames, cut_frames)
    assert cc.latActive
    assert cc.actuators.torque == pytest.approx(0.4)

  high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(cp, cc, cs, high_rate_frames, cut_frames)

  assert not cc.latActive
  assert cc.actuators.torque == 0.0
  assert high_rate_frames == 0
  assert cut_frames == TOYOTA_EPS_HIGH_RATE_CUT_FRAMES - 1


def test_toyota_high_rate_guard_recovers_after_cut_window_and_normal_rate():
  cp = make_cp()
  cc = make_cc()
  cs = make_cs(steering_rate=140.0)
  high_rate_frames = TOYOTA_EPS_HIGH_RATE_FRAMES - 1
  cut_frames = 0

  high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(cp, cc, cs, high_rate_frames, cut_frames)
  assert not cc.latActive
  assert cut_frames == TOYOTA_EPS_HIGH_RATE_CUT_FRAMES - 1

  cc = make_cc()
  cs = make_cs(steering_rate=20.0)
  high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(cp, cc, cs, high_rate_frames, cut_frames)
  assert not cc.latActive
  assert cc.actuators.torque == 0.0
  assert cut_frames == 0

  cc = make_cc()
  high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(cp, cc, cs, high_rate_frames, cut_frames)
  assert cc.latActive
  assert cc.actuators.torque == pytest.approx(0.4)
  assert high_rate_frames == 0
  assert cut_frames == 0


def test_toyota_high_rate_guard_ignores_non_toyota_and_angle_control():
  for cp in (
    make_cp(car_fingerprint="HONDA_CIVIC"),
    make_cp(steer_control_type=car.CarParams.SteerControlType.angle),
  ):
    cc = make_cc()
    cs = make_cs(steering_rate=140.0)
    high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(
      cp, cc, cs, TOYOTA_EPS_HIGH_RATE_FRAMES - 1, TOYOTA_EPS_HIGH_RATE_CUT_FRAMES
    )

    assert cc.latActive
    assert cc.actuators.torque == pytest.approx(0.4)
    assert high_rate_frames == 0
    assert cut_frames == 0


def test_toyota_high_rate_guard_resets_on_existing_steer_fault_or_inactive_lateral():
  cp = make_cp()
  for cc, cs in (
    (make_cc(lat_active=False), make_cs(steering_rate=140.0)),
    (make_cc(), make_cs(steering_rate=140.0, steer_fault_temporary=True)),
    (make_cc(), make_cs(steering_rate=140.0, steer_fault_permanent=True)),
  ):
    high_rate_frames, cut_frames = apply_toyota_eps_high_rate_guard(
      cp, cc, cs, TOYOTA_EPS_HIGH_RATE_FRAMES - 1, TOYOTA_EPS_HIGH_RATE_CUT_FRAMES
    )

    assert high_rate_frames == 0
    assert cut_frames == 0
