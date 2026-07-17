from types import SimpleNamespace

from openpilot.sunnypilot.custom.lateral.friction_breakaway_floor import FrictionBreakawayFloor
from openpilot.sunnypilot.custom.lateral.torque_safety import validate_friction_breakaway_mode

TP = SimpleNamespace(latAccelFactor=1.94, friction=0.126)
FULL = TP.latAccelFactor * TP.friction


def _engaged_floor(mode="apply"):
  floor = FrictionBreakawayFloor()
  floor.mode = mode
  return floor


def test_validator_fails_closed():
  assert validate_friction_breakaway_mode("shadow") == "shadow"
  assert validate_friction_breakaway_mode("apply") == "apply"
  assert validate_friction_breakaway_mode(None) == "off"
  assert validate_friction_breakaway_mode("banana") == "off"


def test_off_mode_is_passthrough():
  floor = _engaged_floor("off")
  for _ in range(100):
    assert floor.shape(0.01, 0.1, TP) == 0.01
  assert not floor.debug.active


def test_shadow_computes_but_never_applies():
  floor = _engaged_floor("shadow")
  out = [floor.shape(0.005, 0.1, TP) for _ in range(100)]
  assert all(v == 0.005 for v in out)          # command unchanged
  assert floor.debug.active                     # but the would-be boost is recorded
  assert floor.debug.delta > 0


def test_apply_boosts_persistent_small_error():
  floor = _engaged_floor()
  out = 0.005
  for _ in range(100):
    out = floor.shape(0.005, 0.1, TP)
  assert out > 0.005
  assert out <= floor.floor_frac * FULL + 1e-9


def test_noise_level_error_never_boosted():
  floor = _engaged_floor()
  for _ in range(100):
    out = floor.shape(0.001, 0.02, TP)  # |error| below MIN_ERROR
  assert out == 0.001
  assert not floor.debug.active


def test_sign_flip_resets_persistence():
  floor = _engaged_floor()
  for _ in range(100):
    floor.shape(0.005, 0.1, TP)
  # flip: persistence resets, boost slews out instead of flipping instantly
  out = floor.shape(-0.005, -0.1, TP)
  assert out >= -0.005 - floor.slew_per_frame


def test_boost_is_slew_limited():
  floor = _engaged_floor()
  prev = 0.0
  for _ in range(60):
    out = floor.shape(0.0, 0.2, TP)
    assert abs(out - prev) <= floor.slew_per_frame + 1e-12
    prev = out


def test_never_fights_larger_base_term():
  floor = _engaged_floor()
  for _ in range(100):
    out = floor.shape(0.5, 0.2, TP)  # base already above any floor target
  assert out == 0.5
