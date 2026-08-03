import json
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.lateral.direction_gain_learning import (
  DIRECTION_GAIN_PARAMS_VERSION,
  DIRECTION_GAIN_SPEED_BANDS,
  direction_scales,
)
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import (
  ROLL_COMP_PARAMS_VERSION,
  ROLL_COMP_PRIMARY_BAND,
  roll_gain_at,
)
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint='test', lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def _roll_profile(gain=0.7):
  band = {
    'vLo': ROLL_COMP_PRIMARY_BAND[0],
    'vHi': ROLL_COMP_PRIMARY_BAND[1],
    'gain': gain,
    'points': 2000,
    'span': 0.5,
    'confidence': 0.5,
    'blockCount': 12,
    'slopeRelSe': 0.05,
  }
  return {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': {
      'carFingerprint': 'test', 'lateralTuning': 'torque', 'latAccelFactor': 2.0, 'friction': 0.2,
    },
    **{field: band[field] for field in ('gain', 'points', 'span', 'confidence', 'blockCount', 'slopeRelSe')},
    'bands': [band],
  }


def _direction_profile(ratio=1.1):
  bands = []
  for v_lo, v_hi in DIRECTION_GAIN_SPEED_BANDS:
    bands.append({
      'vLo': v_lo,
      'vHi': v_hi,
      'ratio': ratio,
      'pointsLeft': 400,
      'pointsRight': 400,
      'blocksLeft': 12,
      'blocksRight': 12,
      'ratioBlocks': 12,
      'leftSlopeRelSe': 0.05,
      'rightSlopeRelSe': 0.05,
      'ratioRelSe': 0.05,
    })
  return {
    'version': DIRECTION_GAIN_PARAMS_VERSION,
    'restoreKey': {
      'carFingerprint': 'test', 'lateralTuning': 'torque', 'latAccelFactor': 2.0, 'friction': 0.2,
    },
    'ratio': ratio,
    'points': 1600,
    'blockCount': 12,
    'maxRelSe': 0.05,
    'bands': bands,
  }


def _low_roll_profile(gain=0.4):
  return {
    'version': ROLL_COMP_PARAMS_VERSION,
    'restoreKey': {
      'carFingerprint': 'test', 'lateralTuning': 'torque', 'latAccelFactor': 2.0, 'friction': 0.2,
    },
    'bands': [{
      'vLo': 5.0, 'vHi': 10.0, 'gain': gain, 'points': 2000, 'span': 0.5,
      'confidence': 0.5, 'blockCount': 12, 'slopeRelSe': 0.05,
    }],
  }


class ParamStub:
  def __init__(self, roll_mode='apply', roll_profile=None, direction_mode='apply', direction_profile=None):
    self.roll_mode = roll_mode
    self.roll_profile = roll_profile
    self.direction_mode = direction_mode
    self.direction_profile = direction_profile

  def get_bool(self, key):
    return key == 'EnforceTorqueControl'

  def get(self, key, return_default=True):
    return {
      'RollCompGainMode': self.roll_mode,
      'RollCompGainParams': self.roll_profile,
      'LatDirectionGainMode': self.direction_mode,
      'LatDirectionGainParams': self.direction_profile,
    }.get(key, '')


def _update(ext, params):
  ext.params = params
  ext.update_override_torque_params(SimpleNamespace(latAccelFactor=2.0, friction=0.2), 25.0)


def test_apply_uses_valid_roll_and_direction_snapshots():
  ext = LatControlTorqueExtOverride(cp())
  params = ParamStub(
    roll_profile=json.dumps(_roll_profile()),
    direction_profile=json.dumps(_direction_profile()),
  )
  _update(ext, params)

  assert ext.learned_roll_gain == pytest.approx(0.7)
  assert ext.direction_gain_scales == direction_scales(_direction_profile())


def test_invalid_apply_profiles_keep_fixed_roll_and_identity_direction():
  ext = LatControlTorqueExtOverride(cp())
  bad_roll = _roll_profile()
  bad_roll['version'] = 1
  bad_direction = _direction_profile()
  bad_direction['version'] = 2
  _update(ext, ParamStub(roll_profile=json.dumps(bad_roll), direction_profile=json.dumps(bad_direction)))

  assert ext.learned_roll_gain is None
  assert ext.learned_roll_gain_at(25.0, 0.55) is None
  assert roll_gain_at(None, 25.0, 0.55) == pytest.approx(0.55)
  assert ext.direction_gain_scales == {1: 1.0, -1: 1.0}


def test_malformed_and_out_of_range_apply_profiles_fail_closed():
  for roll_profile in ('not-json', json.dumps({**_roll_profile(), 'gain': 0.2}),
                       json.dumps({**_roll_profile(), 'confidence': 0.4})):
    ext = LatControlTorqueExtOverride(cp())
    _update(ext, ParamStub(roll_profile=roll_profile))
    assert ext.learned_roll_gain is None


def test_partial_apply_profile_pins_unlearned_speed_bands_to_base():
  ext = LatControlTorqueExtOverride(cp())
  _update(ext, ParamStub(roll_profile=json.dumps(_low_roll_profile())))

  assert ext.learned_roll_gain is None
  assert ext.learned_roll_gain_at(7.5, 0.55) == pytest.approx(0.4)
  assert ext.learned_roll_gain_at(12.5, 0.55) == pytest.approx(0.55)
  assert ext.learned_roll_gain_at(20.0, 0.55) == pytest.approx(0.55)
  assert ext.learned_roll_gain_at(30.0, 0.55) == pytest.approx(0.55)


@pytest.mark.parametrize('mode', ['off', 'shadow'])
def test_non_apply_modes_fail_closed(mode):
  ext = LatControlTorqueExtOverride(cp())
  _update(ext, ParamStub(roll_mode=mode, roll_profile=json.dumps(_roll_profile()), direction_mode=mode,
                         direction_profile=json.dumps(_direction_profile())))
  assert ext.learned_roll_gain is None
  assert ext.direction_gain_scales == {1: 1.0, -1: 1.0}


def test_engaged_refresh_remains_deferred_until_allowed():
  ext = LatControlTorqueExtOverride(cp())
  params = ParamStub(roll_profile=json.dumps(_roll_profile(0.7)))
  _update(ext, params)
  assert ext.learned_roll_gain == pytest.approx(0.7)

  params.roll_profile = json.dumps(_roll_profile(0.8))
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  _update(ext, params)
  assert ext.learned_roll_gain == pytest.approx(0.7)

  ext.set_torque_override_refresh_allowed(True)
  _update(ext, params)
  assert ext.learned_roll_gain == pytest.approx(0.8)


def test_engaged_direction_refresh_remains_deferred_until_allowed():
  ext = LatControlTorqueExtOverride(cp())
  params = ParamStub(direction_profile=json.dumps(_direction_profile(1.1)))
  _update(ext, params)
  original = ext.direction_gain_scales

  params.direction_profile = json.dumps(_direction_profile(1.2))
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  _update(ext, params)
  assert ext.direction_gain_scales == original

  ext.set_torque_override_refresh_allowed(True)
  _update(ext, params)
  assert ext.direction_gain_scales == direction_scales(_direction_profile(1.2))


def test_wrong_restore_key_is_rejected_for_apply_profile():
  bad = _roll_profile()
  bad['restoreKey'] = {**bad['restoreKey'], 'carFingerprint': 'other'}
  ext = LatControlTorqueExtOverride(cp())
  _update(ext, ParamStub(roll_profile=json.dumps(bad)))
  assert ext.learned_roll_gain is None
