from types import SimpleNamespace
from tempfile import TemporaryDirectory

from openpilot.common.params import Params

from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LEGACY_LONGITUDINAL_MODE_PARAMS,
  LONGITUDINAL_MODE_MIGRATION_VERSION,
  LONGITUDINAL_MODE_PARAM,
  LONGITUDINAL_MODE_MIGRATION_PARAM,
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolver,
  ResolvedLongitudinalImplementation,
  SccModeEvidence,
  filter_legacy_longitudinal_mode_params,
  legacy_longitudinal_mode_params_ignored,
  migrate_longitudinal_mode_params,
  requested_mode_from_params,
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

  assert params.values[LONGITUDINAL_MODE_PARAM] == int(LongitudinalMode.ACC)
  assert params.values[LONGITUDINAL_MODE_MIGRATION_PARAM] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_experimental_dec_migrates_to_scc():
  params = FakeParams()
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", True)

  migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == int(LongitudinalMode.SCC)


def test_experimental_without_dec_migrates_to_e2e():
  params = FakeParams()
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", False)

  migrate_longitudinal_mode_params(params)

  assert params.values[LONGITUDINAL_MODE_PARAM] == int(LongitudinalMode.E2E)


def test_migration_writes_typed_params_with_real_params():
  with TemporaryDirectory() as params_dir:
    params = Params(params_dir)
    params.put("ExperimentalMode", True)
    params.put("DynamicExperimentalControl", True)

    assert migrate_longitudinal_mode_params(params)

    assert params.get(LONGITUDINAL_MODE_PARAM) == int(LongitudinalMode.SCC)
    assert params.get(LONGITUDINAL_MODE_MIGRATION_PARAM) == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_migration_copies_legacy_scc_curve_params():
  params = FakeParams()
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", True)
  params.put("SmartCruiseControlVision", False)
  params.put("SmartCruiseControlMap", True)

  migrate_longitudinal_mode_params(params)

  assert params.values["SccCurveVisionEnabled"] is False
  assert params.values["SccCurveMapEnabled"] is True


def test_migration_does_not_override_existing_scc_curve_params():
  params = FakeParams()
  params.put("SmartCruiseControlVision", False)
  params.put("SccCurveVisionEnabled", True)

  migrate_longitudinal_mode_params(params)

  assert params.values["SccCurveVisionEnabled"] is True


def test_v1_migrated_devices_still_copy_legacy_scc_curve_params():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.SCC))
  params.put(LONGITUDINAL_MODE_MIGRATION_PARAM, "1.0")
  params.put("SmartCruiseControlVision", False)
  params.put("SmartCruiseControlMap", False)

  assert migrate_longitudinal_mode_params(params)

  assert params.values["SccCurveVisionEnabled"] is False
  assert params.values["SccCurveMapEnabled"] is False
  assert params.values[LONGITUDINAL_MODE_MIGRATION_PARAM] == LONGITUDINAL_MODE_MIGRATION_VERSION


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


def test_legacy_params_are_ignored_after_migration_even_if_source_param_is_missing():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_MIGRATION_PARAM, LONGITUDINAL_MODE_MIGRATION_VERSION)
  params.put("ExperimentalMode", True)
  params.put("DynamicExperimentalControl", True)

  assert requested_mode_from_params(params) == LongitudinalMode.ACC
  assert legacy_longitudinal_mode_params_ignored(params)
  assert should_skip_legacy_longitudinal_restore("ExperimentalMode", params)

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


def test_scc_resolver_promotes_model_stop_evidence_to_e2e_like():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.SCC))

  resolution = resolve_longitudinal_mode(params, scc_evidence=SccModeEvidence(model_stop=True))

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_E2E
  assert resolution.compatibility_alias_state.value == "blended"
  assert resolution.debug["reason"] == "scc_model_stop"


def test_scc_resolver_keeps_confirmed_lead_acc_like_even_with_model_stop():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.SCC))

  resolution = resolve_longitudinal_mode(
    params, scc_evidence=SccModeEvidence(confirmed_lead=True, model_stop=True)
  )

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert resolution.compatibility_alias_state.value == "acc"
  assert resolution.debug["reason"] == "scc_confirmed_lead"


def test_scc_resolver_tracks_curve_evidence_without_promoting_e2e():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.SCC))

  resolution = resolve_longitudinal_mode(params, scc_evidence=SccModeEvidence(curve_control=True))

  assert resolution.resolved_implementation == ResolvedLongitudinalImplementation.SCC_ACC
  assert resolution.debug["reason"] == "scc_curve"


def test_resolver_keeps_dec_compatibility_alias_sane():
  params = FakeParams()
  params.put(LONGITUDINAL_MODE_PARAM, int(LongitudinalMode.E2E))

  resolution = resolve_longitudinal_mode(params, SimpleNamespace(openpilotLongitudinalControl=True))

  assert resolution.compatibility_alias_state.value == "blended"
