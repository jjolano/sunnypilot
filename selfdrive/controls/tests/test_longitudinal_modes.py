from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LEGACY_LONGITUDINAL_MODE_PARAMS,
  LONGITUDINAL_MODE_MIGRATION_VERSION,
  LONGITUDINAL_MODE_PARAM,
  LONGITUDINAL_MODE_MIGRATION_PARAM,
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolver,
  ResolvedLongitudinalImplementation,
  filter_legacy_longitudinal_mode_params,
  legacy_longitudinal_mode_params_ignored,
  migrate_longitudinal_mode_params,
  resolve_longitudinal_mode,
  should_skip_legacy_longitudinal_restore,
)


class FakeParams:
  def __init__(self):
    self.values = {}

  def get(self, key, *args, **kwargs):
    return self.values.get(key)

  def get_bool(self, key):
    value = self.values.get(key)
    if isinstance(value, bool):
      return value
    return str(value).lower() in ("1", "true", "yes")

  def put(self, key, value):
    self.values[key] = value


def test_fresh_install_migrates_to_acc():
  params = FakeParams()

  assert migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == str(int(LongitudinalMode.ACC))
  assert params.values[LONGITUDINAL_MODE_MIGRATION_PARAM] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_experimental_dec_migrates_to_scc():
  params = FakeParams()
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", True)

  migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == str(int(LongitudinalMode.SCC))


def test_experimental_without_dec_migrates_to_e2e():
  params = FakeParams()
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", False)

  migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == str(int(LongitudinalMode.E2E))


def test_existing_longitudinal_mode_is_never_overridden():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.ACC))
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", True)

  assert not migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == int(LongitudinalMode.ACC)
  assert params.values[LONGITUDINAL_MODE_MIGRATION_PARAM] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_migration_runs_once():
  params = FakeParams()

  assert migrate_longitudinal_mode_params(params)
  assert not migrate_longitudinal_mode_params(params)


def test_legacy_params_are_ignored_after_migration():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.ACC))
  params.put(LONGITUDINAL_MODE_MIGRATION_PARAM, LONGITUDINAL_MODE_MIGRATION_VERSION)

  assert legacy_longitudinal_mode_params_ignored(params)
  assert should_skip_legacy_longitudinal_restore("DynamicExperimentalControl", params)
  assert not should_skip_legacy_longitudinal_restore("SpeedLimitMode", params)

  incoming = {key: "1" for key in LEGACY_LONGITUDINAL_MODE_PARAMS}
  incoming["SpeedLimitMode"] = "3"
  assert filter_legacy_longitudinal_mode_params(incoming, params) == {"SpeedLimitMode": "3"}


def test_resolver_returns_hardware_acc_for_acc_with_radar():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.ACC))
  cp = SimpleNamespace(radarUnavailable=False, openpilotLongitudinalControl=True)

  resolution = LongitudinalModeResolver.resolve(params, cp)

  assert resolution.requested_mode == LongitudinalMode.ACC
  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.HARDWARE_ACC
  assert resolution.actuation_type == LongitudinalActuationType.DIRECT


def test_resolver_returns_model_acc_for_radarless_acc():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.ACC))
  cp = SimpleNamespace(radarUnavailable=True, openpilotLongitudinalControl=True)

  resolution = resolve_longitudinal_mode(params, cp)

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.MODEL_ACC


def test_resolver_returns_set_speed_advisory_without_direct_longitudinal():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.SCC))
  cp = SimpleNamespace(radarUnavailable=False, openpilotLongitudinalControl=False)

  resolution = resolve_longitudinal_mode(params, cp)

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.ICBM_ADVISORY
  assert resolution.actuation_type == LongitudinalActuationType.SET_SPEED_ADVISORY


def test_resolver_keeps_dec_compatibility_alias_sane():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.E2E))

  resolution = resolve_longitudinal_mode(params, SimpleNamespace(openpilotLongitudinalControl=True))

  assert resolution.compatibility_alias_state.value == "blended"
