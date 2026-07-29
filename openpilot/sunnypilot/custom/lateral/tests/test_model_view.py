"""Schema-drift guard for ModelView: a renamed/missing field degrades to valid=False,
never raises. Full message reproduces the fields the torque jerk evidence reads."""
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.sunnypilot.custom.lateral.model_view import ModelView


def _msg(orientation_len, accel_y):
  return SimpleNamespace(
    orientation=SimpleNamespace(x=tuple(range(orientation_len))),
    acceleration=SimpleNamespace(y=tuple(accel_y)),
  )


def test_full_message():
  accel_y = tuple(0.1 * i for i in range(CONTROL_N))
  mv = ModelView.from_msg(_msg(CONTROL_N, accel_y))
  assert mv.valid is True
  assert mv.acceleration_y == accel_y


def test_short_orientation_is_invalid():
  mv = ModelView.from_msg(_msg(CONTROL_N - 1, [0.0] * (CONTROL_N - 1)))
  assert mv.valid is False


def test_none_message():
  mv = ModelView.from_msg(None)
  assert mv.valid is False
  assert mv.acceleration_y == ()


def test_missing_orientation_field_fails_closed():
  # A comma schema rename: 'orientation' is gone. Must not raise.
  msg = SimpleNamespace(acceleration=SimpleNamespace(y=(1.0, 2.0)))
  mv = ModelView.from_msg(msg)
  assert mv.valid is False
  assert mv.acceleration_y == (1.0, 2.0)


def test_missing_acceleration_field_fails_closed():
  msg = SimpleNamespace(orientation=SimpleNamespace(x=tuple(range(CONTROL_N))))
  mv = ModelView.from_msg(msg)
  assert mv.valid is True
  assert mv.acceleration_y == ()


if __name__ == "__main__":
  test_full_message()
  test_short_orientation_is_invalid()
  test_none_message()
  test_missing_orientation_field_fails_closed()
  test_missing_acceleration_field_fails_closed()
  print("ok")
