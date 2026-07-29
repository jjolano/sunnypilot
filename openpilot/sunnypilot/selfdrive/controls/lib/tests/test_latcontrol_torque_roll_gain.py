import json
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.custom.lateral.roll_comp_learning import parse_roll_comp_profile


def roll_comp_profile_payload(gain=0.7, points=2000, span=0.5, confidence=0.6):
  return json.dumps({
    "version": 1,
    "restoreKey": {"carFingerprint": "test", "lateralTuning": "torque", "latAccelFactor": 2.0, "friction": 0.2},
    "gain": gain,
    "points": points,
    "span": span,
    "confidence": confidence,
  })


PROFILE = roll_comp_profile_payload()


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint='test', lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def test_apply_mode_exposes_valid_learned_roll_gain():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return PROFILE
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain == 0.7


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_modes_fail_closed_to_none(mode):
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return mode
      if k == 'RollCompGainParams':
        return PROFILE
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain is None


def test_malformed_roll_comp_profile_rejected():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return 'not-json'
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain is None


def test_out_of_clamp_and_low_confidence_rejected():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return roll_comp_profile_payload(gain=0.2)
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain is None

  ext2 = LatControlTorqueExtOverride(cp())
  class P2:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return roll_comp_profile_payload(confidence=0.4)
      return ''
  ext2.params = P2()
  ext2.enforce_torque_control_toggle = True
  ext2.update_override_torque_params(tp, 25.0)
  assert ext2.learned_roll_gain is None


def test_roll_comp_profile_changes_deferred_while_refresh_disallowed():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    profile = PROFILE
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return self.profile
      return ''
  p = P()
  ext.params = p
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.set_torque_override_refresh_allowed(True)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain == 0.7

  p.profile = roll_comp_profile_payload(gain=0.8)
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain == 0.7

  ext.set_torque_override_refresh_allowed(True)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain == 0.8


def banded_profile_payload(low_gain=0.4):
  return json.dumps({
    "version": 1,
    "restoreKey": {"carFingerprint": "test", "lateralTuning": "torque", "latAccelFactor": 2.0, "friction": 0.2},
    "bands": [{"vLo": 5.0, "vHi": 10.0, "gain": low_gain, "points": 4000, "span": 0.5, "confidence": 1.0}],
  })


def test_apply_mode_exposes_speed_resolved_gain():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'apply'
      if k == 'RollCompGainParams':
        return banded_profile_payload(low_gain=0.4)
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  # city-only profile: no scalar mirror, but the speed-resolved gain applies at low
  # speed and pins the unlearned highway band to the caller's base gain
  assert ext.learned_roll_gain is None
  assert ext.learned_roll_gain_at(7.5, 0.55) == pytest.approx(0.4)
  assert ext.learned_roll_gain_at(25.0, 0.55) == pytest.approx(0.55)


def test_off_mode_speed_resolved_gain_is_none():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl',)
    def get(self, k, return_default=True):
      if k == 'RollCompGainMode':
        return 'off'
      if k == 'RollCompGainParams':
        return banded_profile_payload()
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.update_override_torque_params(tp, 25.0)
  assert ext.learned_roll_gain_at(7.5, 0.55) is None


def test_parse_rejects_wrong_restore_key():
  payload = json.loads(PROFILE)
  bad_cp = SimpleNamespace(
    carFingerprint="other",
    lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=SimpleNamespace(latAccelFactor=2.0, friction=0.2)),
  )
  assert parse_roll_comp_profile(bad_cp, payload) is None
