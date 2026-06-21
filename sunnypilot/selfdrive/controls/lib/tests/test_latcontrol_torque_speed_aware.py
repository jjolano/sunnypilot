import json
from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.custom.lateral.speed_aware_torque import parse_speed_aware_torque_profile


def profile_payload(anchors=None, ratios=None, confidence=None, points=None):
  return json.dumps({
    "version": 1,
    "restoreKey": {"carFingerprint": "test", "lateralTuning": "torque", "latAccelFactor": 2.0, "friction": 0.2},
    "anchors": anchors or [20.0, 30.0],
    "ratios": ratios or [1.1, 1.2],
    "confidence": confidence or [1.0, 1.0],
    "points": points or [500, 500],
    "globalLatAccelFactor": 2.0,
    "globalFriction": 0.2,
  })


PROFILE = profile_payload()


def cp():
  torque = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  return SimpleNamespace(carFingerprint='test', lateralTuning=SimpleNamespace(which=lambda: 'torque', torque=torque))


def test_override_off_and_manual_priority():
  ext = LatControlTorqueExtOverride(cp())
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled', 'LiveTorqueParamsToggle', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveParams':
        return profile_payload(anchors=[20.0], ratios=[1.1], confidence=[1.0], points=[1])
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return 'apply'
      if 'TorqueParamsOverride' in k:
        return '2.0' if 'LatAccelFactor' in k else '0.2'
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  ext.torque_override_enabled = True
  assert ext.update_override_torque_params(tp, 25.0)


def test_restore_base_on_low_speed_and_no_compounding():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'LiveTorqueParamsToggle')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return 'apply'
      if k == 'LiveTorqueSpeedAdaptiveParams':
        return PROFILE
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  ext.base_latAccelFactor = 2.0
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 10.0) is False
  assert tp.latAccelFactor == 2.0
  assert ext.update_override_torque_params(tp, 25.0) is True
  applied = tp.latAccelFactor
  tp.latAccelFactor = applied
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == applied


def test_mode_off_restores_previous_speed_apply():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    mode = 'apply'
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'LiveTorqueParamsToggle')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return self.mode
      if k == 'LiveTorqueSpeedAdaptiveParams':
        return PROFILE
      return ''
  p = P()
  ext.params = p
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor != 2.0
  p.mode = 'off'
  ext.frame = 299
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 2.0


def test_profile_payload_polled_not_read_every_tick():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    profile_reads = 0
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'LiveTorqueParamsToggle')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return 'apply'
      if k == 'LiveTorqueSpeedAdaptiveParams':
        self.profile_reads += 1
        return PROFILE
      return ''
  p = P()
  ext.params = p
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert p.profile_reads == 1


def test_speed_apply_does_not_freeze_live_friction_update():
  ext = LatControlTorqueExtOverride(cp())
  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'LiveTorqueParamsToggle')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return 'apply'
      if k == 'LiveTorqueSpeedAdaptiveParams':
        return PROFILE
      return ''
  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 25.0) is True

  # Simulate controlsd applying a fresh normal live-torque update before the next override tick.
  tp.latAccelFactor = 2.0
  tp.friction = 0.25
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.friction == 0.25


def test_manual_override_accepts_valid_range_edges():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return '0.1'
      if k == 'TorqueParamsOverrideFriction':
        return '1.0'
      return 'off'

  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 0.1
  assert tp.friction == 1.0


def test_invalid_manual_override_values_do_not_apply_or_clamp():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return 'nan'
      if k == 'TorqueParamsOverrideFriction':
        return '2.0'
      return 'off'

  ext.params = P()
  ext.enforce_torque_control_toggle = True
  ext.last_manual_applied = 3.0
  ext.last_manual_friction_applied = 0.5
  ext.base_latAccelFactor = 2.0
  ext.base_friction = 0.2
  tp = SimpleNamespace(latAccelFactor=3.0, friction=0.5)

  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2
  assert ext.last_manual_applied is None
  assert ext.last_manual_friction_applied is None


def test_manual_override_not_applied_without_custom_torque_params():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled')
    def get(self, k, return_default=True):
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return '0.5'
      if k == 'TorqueParamsOverrideFriction':
        return '0.3'
      return 'off'

  ext.params = P()
  ext.enforce_torque_control_toggle = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  assert ext.update_override_torque_params(tp, 25.0) is False
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2


def test_malformed_speed_aware_profile_rejected_for_non_monotonic_anchors():
  payload = json.loads(profile_payload(anchors=[30.0, 20.0], ratios=[1.0, 1.0], confidence=[1.0, 1.0], points=[500, 500]))
  assert parse_speed_aware_torque_profile(cp(), payload) is None


def test_malformed_speed_aware_profile_rejected_for_non_finite_global():
  payload = json.loads(PROFILE)
  payload['globalLatAccelFactor'] = float('nan')
  assert parse_speed_aware_torque_profile(cp(), payload) is None


def test_malformed_speed_aware_profile_rejected_for_out_of_bounds_ratio():
  payload = json.loads(profile_payload(ratios=[0.5, 1.0]))
  assert parse_speed_aware_torque_profile(cp(), payload) is None


def test_manual_override_changes_deferred_while_refresh_disallowed():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    factor = '2.0'
    friction = '0.2'
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return self.factor
      if k == 'TorqueParamsOverrideFriction':
        return self.friction
      return 'off'

  p = P()
  ext.params = p
  tp = SimpleNamespace(latAccelFactor=2.5, friction=0.1)
  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2

  p.factor = '4.0'
  p.friction = '0.5'
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2

  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 4.0
  assert tp.friction == 0.5


def test_disabling_manual_override_restores_only_after_refresh_allowed():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    enabled = True
    def get_bool(self, k):
      if k == 'TorqueParamsOverrideEnabled':
        return self.enabled
      return k in ('EnforceTorqueControl', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return '3.0'
      if k == 'TorqueParamsOverrideFriction':
        return '0.4'
      return 'off'

  p = P()
  ext.params = p
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 3.0
  assert tp.friction == 0.4

  p.enabled = False
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 3.0
  assert tp.friction == 0.4

  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2


def test_no_control_affecting_param_reads_while_refresh_disallowed():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    reads = 0
    def get_bool(self, k):
      self.reads += 1
      return k in ('EnforceTorqueControl', 'TorqueParamsOverrideEnabled', 'CustomTorqueParams')
    def get(self, k, return_default=True):
      self.reads += 1
      if k == 'TorqueParamsOverrideLatAccelFactor':
        return '4.0'
      if k == 'TorqueParamsOverrideFriction':
        return '0.5'
      return 'off'

  p = P()
  ext.params = p
  ext.enforce_torque_control_toggle = True
  ext.torque_override_enabled = True
  ext._custom_torque_params = True
  ext._manual_latAccelFactor = 2.0
  ext._manual_friction = 0.2
  ext._manual_override_values_valid = True
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert p.reads == 0
  assert tp.latAccelFactor == 2.0
  assert tp.friction == 0.2


def test_speed_aware_profile_changes_deferred_while_refresh_disallowed():
  ext = LatControlTorqueExtOverride(cp())

  class P:
    profile = profile_payload(anchors=[20.0], ratios=[1.1], confidence=[1.0], points=[1])
    def get_bool(self, k): return k in ('EnforceTorqueControl', 'LiveTorqueParamsToggle')
    def get(self, k, return_default=True):
      if k == 'LiveTorqueSpeedAdaptiveMode':
        return 'apply'
      if k == 'LiveTorqueSpeedAdaptiveParams':
        return self.profile
      return ''

  p = P()
  ext.params = p
  tp = SimpleNamespace(latAccelFactor=2.0, friction=0.2)
  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  applied = tp.latAccelFactor

  p.profile = profile_payload(anchors=[20.0], ratios=[1.2], confidence=[1.0], points=[1])
  ext.set_torque_override_refresh_allowed(False)
  ext.frame = 299
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor == applied

  ext.set_torque_override_refresh_allowed(True)
  assert ext.update_override_torque_params(tp, 25.0) is True
  assert tp.latAccelFactor != applied
