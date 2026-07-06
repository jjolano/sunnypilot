"""Focused tests for the first DEC -> SCC migration slice."""
from __future__ import annotations

from types import SimpleNamespace
import math
import time

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal import wiring as long_wiring
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter, CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource


class FakeParams:
  def __init__(self, **vals): self._v = vals
  def get_bool(self, k): return bool(self._v.get(k, False))
  def get(self, k): return self._v.get(k)


class FakeSubMaster(dict):
  def __init__(self, *args, model_age_s=0.0, **kwargs):
    super().__init__(*args, **kwargs)
    self.recv_time = {'modelV2': time.monotonic() - float(model_age_s)}


def fake_sm(exp_mode=False, brake=False, gas=False, force_decel=False, model_age_s=0.0):
  return FakeSubMaster({  # type: ignore[return-value]
    'selfdriveState': SimpleNamespace(experimentalMode=exp_mode),
    'carState': SimpleNamespace(brakePressed=brake, gasPressed=gas),
    'controlsState': SimpleNamespace(forceDecel=force_decel),
  }, model_age_s=model_age_s)


def fake_cp():
  return SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)


def fake_planner(mode=LongitudinalMode.SCC, should_stop=False, sources=SourceToggles(), release=False,
                 curve_mode="off", standstill_mode="off", cut_in_mode="off", traffic_mode="off"):
  sp = object.__new__(LongitudinalPlannerSP)
  sp.__dict__['CP'] = fake_cp()
  sp.__dict__['custom_long'] = SimpleNamespace(enabled=True, mode=mode, sources=sources,
                                                  curve_speed_confidence_mode=curve_mode,
                                                  standstill_release_confidence_mode=standstill_mode,
                                                  cut_in_brake_assist_mode=cut_in_mode,
                                                  curve_traffic_advisor_mode=traffic_mode,
                                                  maybe_refresh_params=lambda: None)
  sp.__dict__['dec'] = SimpleNamespace(active=lambda: True, mode=lambda: 'blended')
  sp.__dict__['dt'] = 0.05
  # Finalizer owns the stop-hold/release state; keep the fake planner consistent with the
  # real __init__ by constructing it explicitly here.
  sp.__dict__['custom_long_finalizer'] = CustomLongitudinalFinalizer(sp.CP)
  sp.__dict__['custom_long_output'] = CustomLongitudinalOutput(
    a_target=0.0, should_stop=should_stop, enabled=True, mode=mode,
    selected_intent=("lead_pullaway" if release else None), reason=("trusted" if release else None),
    standstill_release_allowed=release, standstill_release_source=("lead_pullaway" if release else ""),
    standstill_release_a_target=(0.2 if release else 0.0), standstill_release_reason=("trusted" if release else ""),
    research_actuation_allowed=release,
    debug={})
  return sp


def lead(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7):
  return SimpleNamespace(status=status, dRel=dRel, vLead=vLead, vRel=vRel, radarTrackId=radarTrackId)


def test_adapter_evaluate_and_apply_keep_float_api():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode='e2e'))
  stop_x = [0.0, 13.0, 24.0, 32.0, 37.0, 38.0, 38.0, 38.0]
  stop_v = [15.0, 12.0, 9.0, 6.0, 3.0, 0.2, 0.0, 0.0]
  sm = {
    'radarState': SimpleNamespace(leadOne=None, leadTwo=None),
    'carState': SimpleNamespace(brakePressed=False, gasPressed=False),
    'modelV2': SimpleNamespace(action=SimpleNamespace(shouldStop=True, desiredAcceleration=-2.0), position=SimpleNamespace(x=stop_x), velocity=SimpleNamespace(x=stop_v)),
    'carControl': SimpleNamespace(orientationNED=[0.0, 0.0, 0.0]),
    'controlsState': SimpleNamespace(forceDecel=False),
  }
  a._stack.update = lambda *args, **kwargs: SimpleNamespace(a_target=-1.5, should_stop=True, standstill_release_allowed=False, standstill_release_source='', standstill_release_a_target=0.0, standstill_release_reason='', debug={'intent': 'e2e', 'reason': 'trusted'}, decision=SimpleNamespace(selected_intent='e2e', reason='trusted'))  # type: ignore[assignment]
  out = None
  for _ in range(20):
    out = a.evaluate(sm, 10.0, 0.0, 12.0, 0.3, SimpleNamespace(vision=SimpleNamespace(is_active=False, output_a_target=0.0), map=SimpleNamespace(is_active=False, output_a_target=0.0)), SimpleNamespace(is_active=False, output_v_target=0.0, output_a_target=0.0))
  assert out is not None
  assert out.should_stop is True
  assert isinstance(a.apply(sm, 10.0, 0.0, 12.0, 0.3, SimpleNamespace(vision=SimpleNamespace(is_active=False, output_a_target=0.0), map=SimpleNamespace(is_active=False, output_a_target=0.0)), SimpleNamespace(is_active=False, output_v_target=0.0, output_a_target=0.0)), float)


def test_is_e2e_uses_custom_mode_only_when_enabled():
  sp = fake_planner(LongitudinalMode.ACC)
  assert sp.is_e2e(fake_sm(True)) is False  # type: ignore[arg-type]
  sp.custom_long.mode = LongitudinalMode.SCC
  assert sp.is_e2e(fake_sm(True)) is False  # type: ignore[arg-type]
  sp.custom_long.mode = LongitudinalMode.E2E
  assert sp.is_e2e(fake_sm(False)) is True  # type: ignore[arg-type]


def test_custom_should_stop_ownership():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False)
  assert sp.custom_longitudinal_should_stop(True, True) is True
  assert sp.custom_longitudinal_should_stop(False, True) is False
  sp.custom_long_output = CustomLongitudinalOutput(a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC, selected_intent=None, reason=None, debug={})  # type: ignore[assignment]
  assert sp.custom_longitudinal_should_stop(False, False) is True
  sp.custom_long.mode = LongitudinalMode.ACC
  assert sp.custom_longitudinal_should_stop(True, True) is True
  assert sp.custom_longitudinal_should_stop(False, True) is False
  sp.custom_long.mode = LongitudinalMode.E2E
  assert sp.custom_longitudinal_should_stop(True, False) is True
  assert sp.custom_longitudinal_should_stop(False, True) is True
  assert sp.custom_longitudinal_should_stop(False, True, model_stale=True) is False


def test_final_output_selection_does_not_raw_or_model_stop_in_scc_or_acc():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False)
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, False, -3.0, True)  # type: ignore[arg-type]
  assert a == -0.2
  assert should_stop is False
  assert e2e_source is False

  sp.custom_long_output = CustomLongitudinalOutput(a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC, selected_intent=None, reason=None, debug={})  # type: ignore[assignment]
  assert sp.final_longitudinal_output(fake_sm(True), -0.2, False, -3.0, True)[1] is True  # type: ignore[arg-type]

  sp.custom_long_output = CustomLongitudinalOutput(a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.ACC, selected_intent=None, reason=None, debug={})  # type: ignore[assignment]
  sp.custom_long.mode = LongitudinalMode.ACC
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, False, -3.0, True)  # type: ignore[arg-type]
  assert a == -0.2
  assert should_stop is False
  assert e2e_source is False

  sp.custom_long.mode = LongitudinalMode.E2E
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(False), -0.2, False, -3.0, True)  # type: ignore[arg-type]
  assert a == -3.0
  assert should_stop is True
  assert e2e_source is True

  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(False, model_age_s=0.30), -0.2, False, -3.0, True)  # type: ignore[arg-type]
  assert a == -0.2
  assert should_stop is False
  assert e2e_source is False


def test_custom_disabled_finalizer_bypasses_custom_stop_hold_and_clears_latch():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.custom_long.enabled = False
  sm = FakeSubMaster({
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False),
    'controlsState': SimpleNamespace(forceDecel=False),
    'radarState': SimpleNamespace(leadOne=lead(dRel=5.0, vLead=0.0, vRel=0.0), leadTwo=None),
  })

  a, should_stop, e2e_source = sp.final_longitudinal_output(sm, 0.2, False, -3.0, True)  # type: ignore[arg-type]
  assert a == 0.2
  assert should_stop is False
  assert e2e_source is False
  assert sp._lead_stop_hold_active is False

  sp._lead_stop_hold_active = True
  sp._lead_stop_hold_lead_id = 7
  sp._lead_stop_hold_gap_baseline_d_rel = 5.0
  sp._lead_stop_hold_gap_prev_d_rel = 5.0
  sp._stop_hold_release_slew_a_target = 0.15

  a, should_stop, e2e_source = sp.final_longitudinal_output(sm, -0.1, False, -3.0, True)  # type: ignore[arg-type]
  assert a == -0.1
  assert should_stop is False
  assert e2e_source is False
  assert sp._lead_stop_hold_active is False
  assert sp._stop_hold_release_slew_a_target is None


def test_custom_disabled_finalizer_preserves_legacy_dec_e2e_output():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.custom_long.enabled = False

  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, False, -3.0, True)  # type: ignore[arg-type]

  assert a == -3.0
  assert should_stop is True
  assert e2e_source is True


def test_e2e_raw_model_stale_threshold_boundary(monkeypatch):
  monkeypatch.setattr(long_wiring.time, "monotonic", lambda: 100.0)
  for age, expected_stop in ((0.19, True), (0.199, True), (0.201, False), (0.21, False)):
    sp = fake_planner(LongitudinalMode.E2E)
    sm = fake_sm(False)
    sm.recv_time['modelV2'] = 100.0 - age
    a, should_stop, e2e_source = sp.final_longitudinal_output(
      sm, -0.2, False, -3.0, True)  # type: ignore[arg-type]
    assert should_stop is expected_stop
    assert e2e_source is expected_stop
    assert a == (-3.0 if expected_stop else -0.2)


def test_scc_stop_approach_custom_cap_applies_before_full_stop_commitment():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=-1.2, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="stop_approach", reason="model_stop", debug={},
  )  # type: ignore[assignment]
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, False, -0.3, False)  # type: ignore[arg-type]
  assert a == -1.2
  assert should_stop is False
  assert e2e_source is False


def test_scc_ignores_custom_stop_cap_when_not_stop_approach():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=-1.2, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_follow", reason="trusted", debug={},
  )  # type: ignore[assignment]
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, True, -0.3, True)  # type: ignore[arg-type]
  assert a == -0.2
  assert should_stop is True
  assert e2e_source is False


def test_scc_curve_confidence_apply_conservative_caps_final_accel():
  sp = fake_planner(LongitudinalMode.SCC, curve_mode="apply_conservative")
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    research_actuation_allowed=True,
    debug={
      "curve_speed_confidence_mode": "apply_conservative",
      "curve_speed_confidence_effective_mode": "apply_conservative",
      "curve_speed_confidence_apply_supported": True,
      "curve_speed_confidence_eligible": True,
      "curve_speed_confidence_confidence": 0.75,
      "curve_speed_confidence_proposed_cap": -0.4,
    },
  )  # type: ignore[assignment]
  sm = FakeSubMaster({
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
    'controlsState': SimpleNamespace(forceDecel=False),
  })
  a, should_stop, e2e_source = sp.final_longitudinal_output(sm, 0.2, False, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.4
  assert should_stop is False
  assert e2e_source is False


def test_scc_curve_confidence_apply_conservative_noops_without_research_actuation():
  sp = fake_planner(LongitudinalMode.SCC, curve_mode="apply_conservative")
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    research_actuation_allowed=False,
    debug={
      "curve_speed_confidence_apply_supported": True,
      "curve_speed_confidence_eligible": True,
      "curve_speed_confidence_confidence": 0.75,
      "curve_speed_confidence_proposed_cap": -0.4,
    },
  )  # type: ignore[assignment]
  sm = FakeSubMaster({
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
    'controlsState': SimpleNamespace(forceDecel=False),
  })
  a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 0.2
  assert should_stop is False


def test_scc_cut_in_brake_assist_cap_noops_without_research_actuation():
  sp = fake_planner(LongitudinalMode.SCC, cut_in_mode="apply")
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    research_actuation_allowed=False,
    debug={
      "path_shadow_model_path_available": True,
      "cut_in_brake_assist_eligible": True,
      "cut_in_brake_assist_apply_supported": True,
      "cut_in_brake_assist_confidence": 0.75,
      "cut_in_brake_assist_path_y_rel": 0.5,
      "cut_in_brake_assist_proposed_cap": -1.0,
    },
  )  # type: ignore[assignment]
  sm = FakeSubMaster({
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
    'controlsState': SimpleNamespace(forceDecel=False),
  })
  a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 0.2
  assert should_stop is False


def test_scc_curve_traffic_advisor_cap_noops_without_research_actuation():
  sp = fake_planner(LongitudinalMode.SCC, traffic_mode="apply_conservative")
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    research_actuation_allowed=False,
    debug={
      "curve_traffic_eligible": True,
      "curve_traffic_apply_supported": True,
      "curve_traffic_confidence": 0.75,
      "curve_traffic_traffic_block_reason": "",
      "curve_traffic_a_curve_cap_proposed": -1.0,
      "model_stale": False,
    },
  )  # type: ignore[assignment]
  sm = FakeSubMaster({
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
    'controlsState': SimpleNamespace(forceDecel=False),
  })
  a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 0.2
  assert should_stop is False


def test_curve_confidence_shadow_and_low_speed_do_not_cap_final_accel():
  for curve_mode, v_ego in (("shadow", 15.0), ("apply_conservative", 5.0)):
    sp = fake_planner(LongitudinalMode.SCC, curve_mode=curve_mode)
    sp.custom_long_output = CustomLongitudinalOutput(
      a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
      selected_intent="cruise", reason="cruise",
      debug={
        "curve_speed_confidence_apply_supported": curve_mode == "apply_conservative",
        "curve_speed_confidence_eligible": True,
        "curve_speed_confidence_confidence": 0.85,
        "curve_speed_confidence_proposed_cap": -0.4,
      },
    )  # type: ignore[assignment]
    sm = FakeSubMaster({
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'carState': SimpleNamespace(vEgo=v_ego, brakePressed=False, gasPressed=False),
      'controlsState': SimpleNamespace(forceDecel=False),
    })
    a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.3, False)  # type: ignore[arg-type]
    assert a == 0.2
    assert should_stop is False


def test_curve_confidence_apply_conservative_requires_scc_and_healthy_output():
  for mode, enabled in ((LongitudinalMode.ACC, True), (LongitudinalMode.E2E, True), (LongitudinalMode.SCC, False)):
    sp = fake_planner(mode, curve_mode="apply_conservative")
    sp.custom_long_output = CustomLongitudinalOutput(
      a_target=0.0, should_stop=False, enabled=enabled, mode=mode,
      selected_intent="cruise", reason="cruise",
      debug={
        "curve_speed_confidence_apply_supported": True,
        "curve_speed_confidence_eligible": True,
        "curve_speed_confidence_confidence": 0.85,
        "curve_speed_confidence_proposed_cap": -0.4,
      },
    )  # type: ignore[assignment]
    sm = FakeSubMaster({
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
      'controlsState': SimpleNamespace(forceDecel=False),
    })
    a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.3, False)  # type: ignore[arg-type]
    assert a == 0.2
    assert should_stop is False


def test_curve_confidence_apply_conservative_rejects_nonfinite_and_clamps_floor():
  for confidence, proposed, expected in ((float('nan'), -0.4, 0.2), (0.85, float('nan'), 0.2), (0.85, -2.0, -0.85)):
    sp = fake_planner(LongitudinalMode.SCC, curve_mode="apply_conservative")
    sp.custom_long_output = CustomLongitudinalOutput(
      a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
      selected_intent="cruise", reason="cruise",
      research_actuation_allowed=True,
      debug={
        "curve_speed_confidence_apply_supported": True,
        "curve_speed_confidence_eligible": True,
        "curve_speed_confidence_confidence": confidence,
        "curve_speed_confidence_proposed_cap": proposed,
      },
    )  # type: ignore[assignment]
    sm = FakeSubMaster({
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'carState': SimpleNamespace(vEgo=15.0, brakePressed=False, gasPressed=False),
      'controlsState': SimpleNamespace(forceDecel=False),
    })
    a, should_stop, _ = sp.final_longitudinal_output(sm, 0.2, False, 0.0, False)  # type: ignore[arg-type]
    assert a == expected
    assert should_stop is False


def test_acc_ignores_custom_stop_cap():
  sp = fake_planner(LongitudinalMode.ACC, should_stop=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=-1.2, should_stop=True, enabled=True, mode=LongitudinalMode.ACC,
    selected_intent="stop_approach", reason="model_stop", debug={},
  )  # type: ignore[assignment]
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, True, -0.3, True)  # type: ignore[arg-type]
  assert a == -0.2
  assert should_stop is True
  assert e2e_source is False


def test_standstill_release_planner_clears_only_mpc_stop_and_applies_floor():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False, release=True)
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, -3.0, False)  # type: ignore[arg-type]
  assert a >= 0.15
  assert should_stop is False


def test_standstill_release_planner_clears_mpc_stop_without_research_actuation():
  # Normal standstill release (custom permission + valid source) is fork baseline and must
  # clear regardless of the research actuation switch.
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False, release=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted", debug={},
    standstill_release_allowed=True, standstill_release_source='lead_pullaway',
    standstill_release_a_target=0.2, standstill_release_reason='trusted',
    research_actuation_allowed=False,
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, -3.0, False)  # type: ignore[arg-type]
  assert should_stop is False
  assert a >= 0.15


def test_standstill_release_clamps_high_target():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False, release=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent=None, reason=None, debug={},
    standstill_release_allowed=True, standstill_release_source='lead_pullaway',
    standstill_release_a_target=2.0, standstill_release_reason='trusted',
    research_actuation_allowed=True,
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a <= 0.50
  assert should_stop is False


def test_standstill_release_vetoes_real_mpc_brake_and_raw_model_stop():
  sp = fake_planner(LongitudinalMode.E2E, should_stop=False, release=True)
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, -3.0, True)  # type: ignore[arg-type]
  assert a == -3.0
  assert should_stop is True
  a2, should_stop2, _ = sp.final_longitudinal_output(fake_sm(force_decel=True), 0.0, True, -3.0, True)  # type: ignore[arg-type]
  assert a2 == -3.0
  assert should_stop2 is True

  sp.custom_long.mode = LongitudinalMode.SCC
  a3, should_stop3, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, -3.0, True)  # type: ignore[arg-type]
  assert a3 == 0.0
  assert should_stop3 is True


def test_standstill_release_vetoes_timid_e2e_model_accel():
  sp = fake_planner(LongitudinalMode.E2E, should_stop=False, release=True)
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == 0.0
  assert should_stop is True

  a2, should_stop2, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert a2 >= 0.15
  assert should_stop2 is False


def test_sticky_lead_stop_hold_latches_and_releases_on_pullaway():
  sp = fake_planner(LongitudinalMode.ACC, release=True)
  v_ego = 0.0
  x_ego = 0.0
  x_lead = 6.2
  v_lead = 0.0
  lead_v_rel = 0.0
  a_seen = []

  for _ in range(12):
    sm = {
      'carState': SimpleNamespace(vEgo=v_ego, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=x_lead - x_ego, vLead=v_lead, vRel=lead_v_rel, radarTrackId=7,
      )),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    a_seen.append(a_target)
    assert should_stop is True
    assert a_target <= -0.5
    v_ego = max(0.0, v_ego + a_target * 0.05)
    x_ego += v_ego * 0.05

  assert sp._lead_stop_hold_active is True
  assert min(a_seen) <= -0.5

  released = False
  for _ in range(20):
    v_lead = min(2.0, v_lead + 0.4)
    lead_v_rel = v_lead - v_ego
    x_lead += v_lead * 0.05
    sm = {
      'carState': SimpleNamespace(vEgo=v_ego, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=x_lead - x_ego, vLead=v_lead, vRel=lead_v_rel, radarTrackId=7,
      )),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    v_ego = max(0.0, v_ego + a_target * 0.05)
    x_ego += v_ego * 0.05
    if not sp._lead_stop_hold_active:
      assert a_target > -0.5
      released = True
      break

  assert released is True
  assert sp._lead_stop_hold_active is False


def test_stop_hold_rejects_nan_lead_kinematics_and_mpc_targets():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False, release=True)
  sp._lead_stop_hold_active = True
  sp._lead_stop_hold_lead_id = 7
  sm = {
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=lead(dRel=6.5, vLead=float('nan'), vRel=0.2)),
  }
  a, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert math.isfinite(a)

  sm['radarState'] = SimpleNamespace(leadOne=lead(dRel=6.5, vLead=0.4, vRel=float('nan')))
  a, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert math.isfinite(a)

  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), float('nan'), True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert math.isfinite(a)


def test_stop_hold_uses_lead_two_when_lead_one_missing():
  sp = fake_planner(LongitudinalMode.ACC)
  for _ in range(12):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(
        leadOne=SimpleNamespace(status=False, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7),
        leadTwo=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=8),
      ),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    assert should_stop is True
    assert a_target <= -0.5

  assert sp._lead_stop_hold_active is True


def test_stop_hold_survives_brief_lead_dropout():
  sp = fake_planner(LongitudinalMode.ACC)
  # Arm latch with a stopped lead.
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7,
      )),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True

  # Brief dropout (< 0.5 s) keeps latch active.
  for _ in range(3):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    assert sp._lead_stop_hold_active is True

  # Prolonged dropout (> 0.5 s) releases latch.
  for _ in range(15):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False


def test_stop_hold_survives_transient_different_moving_lead():
  """A one-frame moving leadTwo does not clear a stopped-lead latch (finding #2)."""
  sp = fake_planner(LongitudinalMode.ACC)
  # Arm with stopped leadOne id 7.
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7,
      )),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True

  # Brief transient: leadOne gone, moving leadTwo appears (different id 8).
  for _ in range(3):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(
        leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0, vRel=0.0, radarTrackId=7),
        leadTwo=SimpleNamespace(status=True, dRel=15.0, vLead=5.0, vRel=5.0, radarTrackId=8),
      ),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    assert sp._lead_stop_hold_active is True, "transient moving leadTwo must not clear latch"

  # Prolonged dropout (> 0.5 s without any stopped lead) releases.
  for _ in range(15):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False


def test_stop_hold_transfers_to_new_stopped_lead_without_drop():
  """Arm on leadOne (id 7), switch to stopped leadTwo (id 8) — latch transfers immediately (finding #2)."""
  sp = fake_planner(LongitudinalMode.ACC)
  # Arm with stopped leadOne id 7.
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7,
      )),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert sp._lead_stop_hold_lead_id == 7

  # Switch to stopped leadTwo id 8 — latch must stay active on the same tick (no one-cycle gap).
  sm = {
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(
      leadOne=SimpleNamespace(status=False, dRel=0.0, vLead=0.0, vRel=0.0, radarTrackId=7),
      leadTwo=SimpleNamespace(status=True, dRel=6.0, vLead=0.0, vRel=0.0, radarTrackId=8),
    ),
  }
  a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True, "latch must not drop on stopped lead ID change"
  assert sp._lead_stop_hold_lead_id == 8
  assert should_stop is True
  assert a_target <= -0.5


def test_same_id_tiny_gap_increases_do_not_release():
  sp = fake_planner(LongitudinalMode.ACC)
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True

  for d_rel in (6.21, 6.22):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=d_rel, vLead=0.4, vRel=0.2, radarTrackId=7)),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    assert should_stop is True
    assert a_target <= -0.5
  assert sp._lead_stop_hold_active is True


def test_same_id_pullaway_noise_without_release_permission_keeps_latch():
  sp = fake_planner(LongitudinalMode.ACC, release=False)
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=5.8, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True


def test_same_id_pullaway_gate_mode_releases_with_research_actuation():
  sp = fake_planner(LongitudinalMode.SCC, release=False, standstill_mode="gate")
  _arm_stop_hold(sp, d_rel=6.2)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
    research_actuation_allowed=True,
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.20, True, 0.05, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_same_id_pullaway_gate_mode_blocked_without_research_actuation():
  sp = fake_planner(LongitudinalMode.SCC, release=False, standstill_mode="gate")
  _arm_stop_hold(sp, d_rel=6.2)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
    research_actuation_allowed=False,
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.20, True, 0.05, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5


def test_same_id_pullaway_gate_mode_requires_positive_planner_evidence():
  sp = fake_planner(LongitudinalMode.SCC, release=False, standstill_mode="gate")
  _arm_stop_hold(sp, d_rel=6.2)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.00, True, -0.05, True)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5


def test_same_id_pullaway_gate_mode_preserves_raw_model_stop_veto():
  sp = fake_planner(LongitudinalMode.SCC, release=False, standstill_mode="gate")
  _arm_stop_hold(sp, d_rel=6.2)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="cruise",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.20, True, 0.05, True)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5


def test_same_id_pullaway_gate_mode_is_scc_only():
  for mode in (LongitudinalMode.ACC, LongitudinalMode.E2E):
    sp = fake_planner(mode, release=False, standstill_mode="gate")
    _arm_stop_hold(sp, d_rel=6.2)
    sp._lead_stop_hold_gap_increasing_s = 0.15
    sp.custom_long_output = CustomLongitudinalOutput(
      a_target=0.0, should_stop=False, enabled=True, mode=mode,
      selected_intent="cruise", reason="cruise",
      standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
      standstill_release_reason="", debug={},
    )  # type: ignore[assignment]
    a, should_stop, _ = sp.final_longitudinal_output(
      _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.20, True, 0.05, False)  # type: ignore[arg-type]
    assert sp._lead_stop_hold_active is True
    assert should_stop is True
    assert a <= -0.5


def test_same_id_pullaway_gate_mode_requires_healthy_custom_output():
  sp = fake_planner(LongitudinalMode.SCC, release=False, standstill_mode="gate")
  _arm_stop_hold(sp, d_rel=6.2)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=False, mode=LongitudinalMode.SCC,
    selected_intent="cruise", reason="fault",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.85, v_lead=0.55, v_rel=0.35), 0.20, True, 0.05, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5

  sp.custom_long_output = None  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(
    _release_sm(d_rel=6.90, v_lead=0.60, v_rel=0.40), 0.20, True, 0.05, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5

  for d_rel in (9.81, 9.83, 9.84, 9.86):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=d_rel, vLead=0.51, vRel=0.21, radarTrackId=7)),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, False, 0.0, False)  # type: ignore[arg-type]
    assert should_stop is True
    assert a_target <= -0.5

  assert sp._lead_stop_hold_active is True


def test_same_id_sustained_pullaway_releases():
  sp = fake_planner(LongitudinalMode.ACC, release=True)
  for _ in range(6):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True

  released = False
  for d_rel in (7.00, 7.06, 7.12, 7.18, 7.24, 7.30):
    sm = {
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=lead(dRel=d_rel, vLead=0.8, vRel=0.25, radarTrackId=7)),
    }
    a_target, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
    if not sp._lead_stop_hold_active:
      assert a_target > -0.5
      released = True
      break
  assert released is True


def test_same_id_stale_release_permission_never_overrides_current_veto():
  sp = fake_planner(LongitudinalMode.ACC, release=False)
  _arm_stop_hold(sp)

  # First pullaway tick: permission appears but release conditions are not yet met.
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.2, standstill_release_reason="trusted", debug={},
  )  # type: ignore[assignment]
  first_sm = _release_sm(d_rel=6.55, v_lead=0.35, v_rel=0.18)
  a1, should_stop1, _ = sp.final_longitudinal_output(first_sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop1 is True
  assert a1 <= -0.5
  assert sp._lead_stop_hold_active is True

  # Current tick now explicitly vetoes release; stale permission must not leak through.
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_follow", reason="veto",
    standstill_release_allowed=False, standstill_release_source="", standstill_release_a_target=0.0,
    standstill_release_reason="", debug={},
  )  # type: ignore[assignment]
  second_sm = _release_sm(d_rel=7.35, v_lead=0.8, v_rel=0.32)
  a2, should_stop2, _ = sp.final_longitudinal_output(second_sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop2 is True
  assert a2 <= -0.5
  assert sp._lead_stop_hold_active is True


def test_stop_hold_telemetry_shows_latch_intent():
  sp = fake_planner(LongitudinalMode.ACC)
  sm = {
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(
      status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7,
    )),
  }
  sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  telemetry = getattr(sp, "_custom_long_output_telemetry", None)
  assert telemetry is not None
  assert telemetry.selected_intent == "lead_stop_hold"
  assert telemetry.reason == "stopped_lead_latch"
  assert telemetry.should_stop is True


def test_release_block_reason_logged_when_lead_not_moving():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.10, v_rel=0.05), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._last_release_block_reason == "lead_not_moving"


def test_release_block_reason_cleared_on_successful_release():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.32, v_rel=0.16), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._last_release_block_reason == ""


def test_release_block_reason_distance_gate():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=4.56)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp.final_longitudinal_output(_release_sm(d_rel=4.46, v_lead=0.70, v_rel=0.70), 0.0, False, 0.2, False)  # type: ignore[arg-type]
  assert sp._last_release_block_reason == "distance_gate"


def test_release_block_reason_mpc_brake_veto():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.32, v_rel=0.16), -0.2, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._last_release_block_reason == "mpc_brake_veto"


def test_release_block_reason_driver_brake():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.32, v_rel=0.16, brake=True), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._last_release_block_reason == "driver_brake"


def test_standstill_release_vetoes_mpc_brake_driver_and_custom_stop():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=False, release=True)
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), -0.04, True, -3.0, False)  # type: ignore[arg-type]
  assert a == -0.04
  assert should_stop is True

  for sm in (fake_sm(brake=True), fake_sm(gas=True)):
    a, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, -3.0, False)  # type: ignore[arg-type]
    assert a == 0.0
    assert should_stop is True

  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.2, standstill_release_reason="trusted", debug={},
  )
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, -3.0, False)  # type: ignore[arg-type]
  assert a == 0.0
  assert should_stop is True


def test_stale_e2e_raw_model_accel_does_not_harden_stopped_lead_latch():
  sp = fake_planner(LongitudinalMode.E2E)
  _arm_stop_hold(sp)

  fresh_a, fresh_should_stop, fresh_e2e_source = sp.final_longitudinal_output(
    _release_sm(model_age_s=0.0), 0.0, True, -3.0, True)  # type: ignore[arg-type]
  assert fresh_a == -3.0
  assert fresh_should_stop is True
  assert fresh_e2e_source is True

  stale_a, stale_should_stop, stale_e2e_source = sp.final_longitudinal_output(
    _release_sm(model_age_s=0.30), 0.0, True, -3.0, True)  # type: ignore[arg-type]
  assert stale_a == -0.5
  assert stale_should_stop is True  # radar/MPC stopped-lead latch still binds
  assert stale_e2e_source is False


def _arm_stop_hold(sp, d_rel=6.2):
  for _ in range(6):
    sm = FakeSubMaster({
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=d_rel, vLead=0.0, vRel=0.0, radarTrackId=7,
      )),
    })
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True


def _release_sm(d_rel=6.5, v_lead=0.4, v_rel=0.2, radar_id=7, brake=False, gas=False, force_decel=False, model_age_s=0.0):
  return FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=brake, gasPressed=gas, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=force_decel),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(
      status=True, dRel=d_rel, vLead=v_lead, vRel=v_rel, radarTrackId=radar_id,
    )),
  }, model_age_s=model_age_s)


def _set_lead_pullaway_release(sp):
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=sp.custom_long.mode,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )


# First commanded accel on a latch-release frame: the -0.5 standstill hold plus one
# release up-jerk slew step; later frames ramp toward the release accel.
RELEASE_FIRST_STEP = (CustomLongitudinalFinalizer._STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET
                      + CustomLongitudinalFinalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * 0.05)


def test_latch_release_same_lead_clears_earlier_with_bounded_accel():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.32, v_rel=0.16), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_valid_source_crawl_releases_below_deadband_with_cap():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  # Valid source with a moving lead no longer waits for the 0.5 m crawl deadband,
  # but the crawl launch cap still limits authority.
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  # bypass the release-step ramp (covered by the slew tests) so the crawl cap stays observable
  sp.custom_long_finalizer.final_a_prev = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=6.6, v_lead=0.55, v_rel=0.35), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, 0.35, abs_tol=1e-9)


def test_stop_hold_does_not_latch_beyond_arm_envelope():
  # Route 261: latching at 8-10 m froze a 9.3 m gap; beyond the arm envelope the MPC
  # keeps creeping toward its stop buffer instead.
  sp = fake_planner(LongitudinalMode.ACC)
  for _ in range(6):
    sm = FakeSubMaster({
      'carState': SimpleNamespace(vEgo=0.0, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(
        status=True, dRel=8.0, vLead=0.0, vRel=0.0, radarTrackId=7,
      )),
    })
    sp.final_longitudinal_output(sm, 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False


def test_stop_hold_caps_baseline_at_five_meters():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=6.2)
  assert sp._lead_stop_hold_active is True
  assert sp._lead_stop_hold_gap_baseline_d_rel == 5.0


def test_early_stop_beyond_baseline_uses_capped_gap_target_for_crawl():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=6.2)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  # bypass the release-step ramp (covered by the slew tests) so the crawl cap stays observable
  sp.custom_long_finalizer.final_a_prev = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=6.9, v_lead=0.55, v_rel=0.35), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.05 <= a <= 0.35


def test_latch_release_crawl_gap_error_caps_positive_accel():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  # bypass the release-step ramp (covered by the slew tests) so the crawl cap stays observable
  sp.custom_long_finalizer.final_a_prev = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.55, v_rel=0.35), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.05 <= a <= 0.35


def test_latch_release_crawl_pullaway_waits_for_valid_gap_time():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  # Valid-source same-lead release now uses a shorter gap-confirm timer (0.10 s), but it
  # still requires evidence; starting from zero with an opening gap gives only one tick
  # (0.05 s) of evidence and must not release.
  sp._lead_stop_hold_gap_increasing_s = 0.0
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=0.55, v_rel=0.35), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.20  # still held; prep may be softening if above its own threshold


def test_latch_release_routine_breakout_uses_short_gap_time():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.15
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.35, v_lead=2.0, v_rel=1.1), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_latch_release_same_lead_allows_slightly_negative_mpc_with_tighter_gap_confirm():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.05, v_lead=0.32, v_rel=0.16), -0.08, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_latch_release_same_lead_uses_baseline_aware_close_gap_gate():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=4.56)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=5.32, v_lead=1.08, v_rel=1.08), 0.0, False, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_latch_release_same_lead_close_gap_keeps_absolute_floor():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=3.96)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=4.46, v_lead=0.70, v_rel=0.70), 0.0, False, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5


def test_latch_release_no_id_keeps_conservative_distance_gate():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp, d_rel=4.56)
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(radar_id=None, d_rel=5.32, v_lead=1.08, v_rel=1.08), 0.0, False, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= -0.5


def test_latch_release_same_lead_rejects_too_negative_mpc_or_non_opening_or_below_distance():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  assert sp.final_longitudinal_output(_release_sm(d_rel=7.05, v_lead=0.32, v_rel=0.16), -0.2, True, 0.2, False)[1] is True  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert sp.final_longitudinal_output(_release_sm(d_rel=7.05, v_lead=0.29, v_rel=0.16), -0.08, True, 0.2, False)[1] is True  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert sp.final_longitudinal_output(_release_sm(d_rel=7.0, v_lead=0.32, v_rel=0.16), -0.08, True, 0.2, False)[1] is True  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True


def test_latch_release_rejects_no_lead_launch():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="no_lead_launch", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="no_lead_launch",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp.final_longitudinal_output(_release_sm(), 0.0, False, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True


def test_latch_release_rejects_different_moving_lead():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp.final_longitudinal_output(_release_sm(radar_id=8, d_rel=15.0, v_lead=5.0, v_rel=5.0), 0.0, False, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True


def test_latch_release_rejects_raw_model_stop_and_low_mpc_accel_and_driver_inputs():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.30
  assert sp.final_longitudinal_output(_release_sm(), 0.0, True, 0.2, True)[1] is True  # type: ignore[arg-type]  # raw model stop
  assert sp._lead_stop_hold_active is True
  _set_lead_pullaway_release(sp)
  assert sp.final_longitudinal_output(_release_sm(), -0.04, True, 0.2, False)[1] is True  # type: ignore[arg-type]  # mpc brake veto
  assert sp._lead_stop_hold_active is True


def test_latch_release_rejects_driver_and_force_decel_inputs():
  for sm, expect_latch_active in (
    (_release_sm(brake=True), True),
    (_release_sm(force_decel=True), True),
    # Gas is a driver override: preserve the original behavior that cancels the latch, while
    # still vetoing any automatic standstill-release clear of the MPC stop bit.
    (_release_sm(gas=True), False),
  ):
    sp = fake_planner(LongitudinalMode.ACC)
    _arm_stop_hold(sp)
    sp.custom_long_output = CustomLongitudinalOutput(
      a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
      selected_intent="lead_pullaway", reason="trusted",
      standstill_release_allowed=True, standstill_release_source="lead_pullaway",
      standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
    )
    sp._lead_stop_hold_gap_increasing_s = 0.30
    _, should_stop, _ = sp.final_longitudinal_output(sm, 0.0, True, 0.2, False)  # type: ignore[arg-type]
    assert should_stop is True
    assert sp._lead_stop_hold_active is expect_latch_active


def test_latch_release_rejects_timid_e2e_model_accel():
  sp = fake_planner(LongitudinalMode.E2E)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.E2E,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(), 0.0, True, 0.1, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert should_stop is True
  assert a <= 0.0


def test_latch_release_requires_sustained_gap_increase_without_ids():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_lead_id = None
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.05
  sp.final_longitudinal_output(_release_sm(radar_id=None, d_rel=6.45), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  _set_lead_pullaway_release(sp)
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(radar_id=None, d_rel=6.55), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert math.isclose(a, RELEASE_FIRST_STEP)


def test_custom_target_filtering_by_mode_and_scc_source_toggles():
  targets = {
    LongitudinalPlanSource.cruise: (20.0, 0.1),
    LongitudinalPlanSource.sccVision: (5.0, -0.5),
    LongitudinalPlanSource.sccMap: (4.0, -0.7),
    LongitudinalPlanSource.speedLimitAssist: (3.0, -1.0),
  }
  assert set(fake_planner(LongitudinalMode.ACC).custom_longitudinal_targets(targets)) == {LongitudinalPlanSource.cruise}
  assert set(fake_planner(LongitudinalMode.E2E).custom_longitudinal_targets(targets)) == {LongitudinalPlanSource.cruise}
  assert set(fake_planner(LongitudinalMode.SCC, sources=SourceToggles()).custom_longitudinal_targets(targets)) == {
    LongitudinalPlanSource.cruise, LongitudinalPlanSource.speedLimitAssist}
  assert LongitudinalPlanSource.sccVision in fake_planner(LongitudinalMode.SCC, sources=SourceToggles(True, False)).custom_longitudinal_targets(targets)
  assert LongitudinalPlanSource.sccMap not in fake_planner(LongitudinalMode.SCC, sources=SourceToggles(False, True)).custom_longitudinal_targets(targets)


def test_stop_hold_release_slew_first_release_seeds_state_and_stays_floored():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert 0.15 <= a <= 0.50
  assert should_stop is False
  assert sp._stop_hold_release_slew_a_target == a


def test_stop_hold_release_slew_release_step_capped_from_prior_hold():
  # The release frame must not step from the prior hold command straight to the release
  # accel: that bypassed the up-jerk slew entirely (fuzz comfort lead-pullaway failures,
  # seeds 3/7). The seed is bounded at prior command + one slew step.
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  dt = 0.05
  release_output = sp.custom_long_output
  sp.__dict__['custom_long_output'] = fake_planner(LongitudinalMode.SCC, should_stop=True).custom_long_output
  held, _, _ = sp.final_longitudinal_output(fake_sm(), -0.5, True, 0.0, False)  # type: ignore[arg-type]
  assert held == -0.5
  assert sp.custom_long_finalizer.final_a_prev == -0.5
  sp.__dict__['custom_long_output'] = release_output
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is False
  assert a <= -0.5 + sp.custom_long_finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * dt + 1e-9
  assert sp._stop_hold_release_slew_a_target == a


def test_stop_hold_release_slew_caps_second_upward_jump():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  dt = 0.05
  first, _, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert math.isclose(a, first + sp.custom_long_finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * dt)
  assert should_stop is False


def test_stop_hold_release_slew_clears_when_upward_target_catches_without_cap():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  first, _, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), first + 0.1, False, 0.0, False)  # type: ignore[arg-type]
  assert math.isclose(a, first + 0.1)
  assert should_stop is False
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_downward_braking_passes_through_and_clears():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  sp._stop_hold_release_slew_a_target = 1.0  # artificial high-water state
  a, _, _ = sp.final_longitudinal_output(fake_sm(), -0.5, False, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_positive_dip_then_upward_jump_still_capped():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  first, _, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  # tiny positive dip passes through but keeps the slew active
  a_dip, _, _ = sp.final_longitudinal_output(fake_sm(), 0.05, False, 0.0, False)  # type: ignore[arg-type]
  assert a_dip == 0.05
  assert sp._stop_hold_release_slew_a_target is not None
  # subsequent upward jump is still capped
  a, _, _ = sp.final_longitudinal_output(fake_sm(), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert math.isclose(a, 0.05 + sp.custom_long_finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * 0.05)


def test_stop_hold_release_slew_brake_clears_and_passes_through():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, _, _ = sp.final_longitudinal_output(fake_sm(brake=True), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 2.0
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_gas_clears_and_passes_through():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, _, _ = sp.final_longitudinal_output(fake_sm(gas=True), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 2.0
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_force_decel_clears_and_passes_through():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, _, _ = sp.final_longitudinal_output(fake_sm(force_decel=True), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 2.0
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_raw_model_stop_clears_and_passes_through():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  a, _, _ = sp.final_longitudinal_output(fake_sm(), 2.0, False, -3.0, True)  # type: ignore[arg-type]
  assert a == 2.0
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_custom_should_stop_clears_and_passes_through():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.2, standstill_release_reason="trusted", debug={},
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 2.0, False, 0.0, False)  # type: ignore[arg-type]
  assert a == 2.0
  assert should_stop is True
  assert sp._stop_hold_release_slew_a_target is None


def test_stop_hold_release_slew_e2e_preserves_source_when_limited():
  sp = fake_planner(LongitudinalMode.E2E, release=True)
  first, _, _ = sp.final_longitudinal_output(fake_sm(False), 0.0, True, 2.0, False)  # type: ignore[arg-type]
  assert 0.15 <= first <= 0.50
  a, _, e2e_source = sp.final_longitudinal_output(fake_sm(False), 0.6, False, 2.0, False)  # type: ignore[arg-type]
  assert math.isclose(a, first + sp.custom_long_finalizer._STOP_HOLD_RELEASE_MAX_UP_JERK * 0.05)
  assert e2e_source is False  # source decided before slew limiting


def test_target_filtering_keeps_cruise_fallback():
  sp = object.__new__(LongitudinalPlannerSP)
  sp.__dict__['custom_long'] = SimpleNamespace(enabled=True, mode=LongitudinalMode.ACC, sources=SourceToggles(),
                                               maybe_refresh_params=lambda: None)
  sp.scc = SimpleNamespace(vision=SimpleNamespace(output_v_target=5.0, output_a_target=-0.5), map=SimpleNamespace(output_v_target=4.0, output_a_target=-0.7))  # type: ignore[assignment]
  sp.sla = SimpleNamespace(output_v_target=3.0, output_a_target=-1.0)  # type: ignore[assignment]
  sp.__dict__['resolver'] = SimpleNamespace(update=lambda *a, **k: None, speed_limit_valid=False, speed_limit_last_valid=False,
                                            speed_limit=0.0, speed_limit_final_last=0.0, distance=0.0)
  sp.scc.update = lambda *a, **k: None  # type: ignore[attr-defined]
  sp.sla.update = lambda *a, **k: None  # type: ignore[attr-defined]
  sp.output_a_target = 0.0
  sp.output_v_target = 0.0
  sp.source = LongitudinalPlanSource.cruise
  sp.__dict__['events_sp'] = SimpleNamespace(clear=lambda: None, to_msg=lambda: None)
  sp.custom_long.evaluate = lambda *a, **k: CustomLongitudinalOutput(a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.ACC, selected_intent=None, reason=None, debug={})  # type: ignore[assignment]
  sp.custom_long_output = CustomLongitudinalOutput(a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.ACC, selected_intent=None, reason=None, debug={})  # type: ignore[assignment]
  sm = {
    'carState': SimpleNamespace(vCruiseCluster=100.0, vCruise=100.0, aEgo=0.0),
    'carControl': SimpleNamespace(enabled=False, cruiseControl=SimpleNamespace(override=False)),
    'selfdriveState': SimpleNamespace(enabled=False),
  }
  # ACC mode excludes curve/sla evidence, leaving cruise as the only admissible fallback.
  v, a = sp.update_targets(sm, 10.0, 0.0, 8.0)  # type: ignore[arg-type]
  assert math.isclose(v, 8.0)
  assert math.isclose(a, 0.0)


def _prep_sm(d_rel=6.25, v_lead=0.20, v_rel=0.10, **kwargs):
  return _release_sm(d_rel=d_rel, v_lead=v_lead, v_rel=v_rel, **kwargs)


def _prep_planner(stop_accel=-2.0, mode=LongitudinalMode.SCC, release=True):
  sp = fake_planner(mode, release=release)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=stop_accel, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  return sp


def test_stop_hold_release_prep_relaxes_harsh_hold_toward_target():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  # Prime the gap-opening timer to just below the prep threshold while keeping lead below full-release speed.
  for i in range(2):
    sp.final_longitudinal_output(_prep_sm(d_rel=6.25 + i * 0.01, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  # Lead is moving/opening enough for prep, but not enough for full release (vLead < 0.30).
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.28, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert sp._lead_stop_hold_active is True
  assert -2.0 < a <= -0.20
  assert sp._stop_hold_release_prep_a_target == a


def test_stop_hold_release_prep_upward_ramp_bounded_by_jerk():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  for i in range(2):
    sp.final_longitudinal_output(_prep_sm(d_rel=6.25 + i * 0.01, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  prev = float(sp._stop_hold_release_prep_a_target) if sp._stop_hold_release_prep_a_target is not None else sp.custom_long_finalizer._STOP_HOLD_STANDSTILL_NORMALIZED_A_TARGET
  for i in range(6):
    sm = _prep_sm(d_rel=6.27 + i * 0.02, v_lead=0.25, v_rel=0.10)
    a, should_stop, _ = sp.final_longitudinal_output(sm, -0.05, True, 0.0, False)  # type: ignore[arg-type]
    assert should_stop is True
    assert a - prev <= sp.custom_long_finalizer._STOP_HOLD_RELEASE_PREP_MAX_UP_JERK * sp.dt + 1e-9
    assert a >= prev
    if math.isclose(a, -0.20, abs_tol=1e-6):
      break
    prev = a


def test_stop_hold_release_prep_vetoes_raw_model_stop_and_driver_inputs():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2

  # Raw model stop keeps harsh hold and clears prep state.
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(), -0.05, True, 0.0, True)  # type: ignore[arg-type]
  assert should_stop is True
  assert a == -2.0
  assert sp._stop_hold_release_prep_a_target is None

  # Driver brake.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(brake=True), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -2.0
  assert sp._stop_hold_release_prep_a_target is None

  # Force decel.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(force_decel=True), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -2.0
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_vetoes_driver_gas():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(gas=True), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_vetoes_weak_release_evidence():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2

  # Lead not moving enough.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(v_lead=0.15, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None

  # Lead not opening.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.2, v_lead=0.25, v_rel=0.0), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None

  # Gap opening too briefly.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.21, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None

  # Too close.
  a, _, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.05, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None

  # Hard-brake situation (MPC target too negative).
  a, _, _ = sp.final_longitudinal_output(_prep_sm(), -1.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_vetoes_different_moving_lead():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp._lead_stop_hold_gap_increasing_s = 0.30

  a, should_stop, _ = sp.final_longitudinal_output(
    _prep_sm(d_rel=7.0, v_lead=0.35, v_rel=0.10, radar_id=8), -0.05, True, 0.0, False)  # type: ignore[arg-type]

  assert a == -2.0
  assert should_stop is True
  assert sp._lead_stop_hold_active is True
  assert sp._lead_stop_hold_lead_id == 7
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_vetoes_bad_release_source_and_custom_stop():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2

  # Wrong release source.
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="no_lead_launch", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="no_lead_launch",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )  # type: ignore[assignment]
  a, _, _ = sp.final_longitudinal_output(_prep_sm(), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert sp._stop_hold_release_prep_a_target is None

  # Custom should_stop vetoes prep even with release permission.
  _set_lead_pullaway_release(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=True, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )  # type: ignore[assignment]
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5
  assert should_stop is True
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_handles_non_finite_targets():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(), float('nan'), True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert math.isfinite(a)
  assert a <= -0.5
  assert sp._stop_hold_release_prep_a_target is None

  sp._stop_hold_release_prep_a_target = float('nan')
  a2, should_stop2, _ = sp.final_longitudinal_output(_prep_sm(), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop2 is True
  assert math.isfinite(a2)
  assert sp._stop_hold_release_prep_a_target is None


def test_stop_hold_release_prep_downward_braking_passes_through():
  sp = _prep_planner(stop_accel=-0.6)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  # Prime the gap-opening timer to just below the prep threshold.
  for i in range(2):
    sp.final_longitudinal_output(_prep_sm(d_rel=6.25 + i * 0.01, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  sm = _prep_sm(d_rel=6.28, v_lead=0.25, v_rel=0.10)
  a1, _, _ = sp.final_longitudinal_output(sm, -0.05, True, 0.0, False)  # type: ignore[arg-type]
  # raw_hold is normalized to the mild stopped-lead hold; first active prep tick ramps upward.
  assert math.isclose(a1, -0.20)
  a2, _, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.29, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert math.isclose(a2, -0.20)
  # Make the hold command more negative while keeping MPC above the hard-brake veto.
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-1.0, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  a3, _, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.30, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert math.isclose(a3, -0.2)
  assert sp._stop_hold_release_prep_a_target == -0.2


def test_stop_hold_release_prep_does_not_block_first_positive_release():
  sp = _prep_planner(stop_accel=-2.0)
  _arm_stop_hold(sp)
  sp._lead_stop_hold_gap_baseline_d_rel = 6.2
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_prep_sm(d_rel=7.0, v_lead=0.8, v_rel=0.3), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert math.isclose(a, RELEASE_FIRST_STEP)
  assert should_stop is False
  assert sp._lead_stop_hold_active is False
  assert sp._stop_hold_release_prep_a_target is None
  assert sp._stop_hold_release_slew_a_target == a


def test_stop_hold_release_prep_does_not_interfere_with_standstill_release_slew_seed():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  # No lead stop-hold latch; MPC still says stop; custom release allowed clears it.
  a, should_stop, _ = sp.final_longitudinal_output(fake_sm(), 0.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a > 0.0
  assert should_stop is False
  assert sp._stop_hold_release_slew_a_target == a
  assert sp._stop_hold_release_prep_a_target is None
  assert sp._lead_stop_hold_active is False


def test_stop_hold_release_prep_state_resets_with_stop_hold():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp._stop_hold_release_prep_a_target = -0.5
  sp._stop_hold_release_prep_raw_prev = -0.5
  sp._reset_lead_stop_hold()
  assert sp._stop_hold_release_prep_a_target is None
  assert sp._stop_hold_release_prep_raw_prev is None


def test_stop_hold_standstill_normalize_same_id_harsh_to_mild_target():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-2.0, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
  })
  a, should_stop, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert a == -0.5


def test_stop_hold_standstill_normalize_v_ego_fallback():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-2.0, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.015, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
  })
  a, _, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5


def test_stop_hold_standstill_normalize_creeping_landing_clamped():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.05, standstill=False, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
  })
  a, _, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -0.5


def test_stop_hold_standstill_normalize_vetoes_different_lead_id():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  # Use a moving different lead so the latch does not auto-transfer.
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=1.0, vRel=1.0, radarTrackId=8)),
  })
  a, _, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -2.0


def test_stop_hold_standstill_normalize_vetoes_missing_lead_id():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=None)),
  })
  a, _, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a == -2.0


def test_stop_hold_standstill_normalize_vetoes_raw_model_stop():
  sp = fake_planner(LongitudinalMode.SCC)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  sm = FakeSubMaster({
    'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
  })
  a, _, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, True)  # type: ignore[arg-type]
  assert a == -2.0


def test_stop_hold_standstill_normalize_vetoes_driver_and_force_inputs():
  for brake, gas, force in ((True, False, False), (False, True, False), (False, False, True)):
    sp = fake_planner(LongitudinalMode.SCC)
    sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
    sp.custom_long_finalizer.CP = sp.CP
    _arm_stop_hold(sp, d_rel=6.2)
    sm = FakeSubMaster({
      'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=brake, gasPressed=gas, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=force),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
    })
    a, should_stop, _ = sp.final_longitudinal_output(sm, -2.0, True, 0.0, False)  # type: ignore[arg-type]
    assert a == -2.0, f"brake={brake} gas={gas} force={force}"
    assert should_stop is True


def test_stop_hold_standstill_normalize_allows_prep_to_relax():
  """Normalization to stop_accel does not prevent release prep from ramping further."""
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.0)
  # harsh MPC while stopped gets normalized to the mild stopped-lead hold
  a1, _, _ = sp.final_longitudinal_output(
    FakeSubMaster({
      'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.0, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }), -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a1 == -0.5
  # prep can still relax from the normalized hold
  for i in range(2):
    sp.final_longitudinal_output(_prep_sm(d_rel=6.21 + i * 0.01, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  a2, should_stop, _ = sp.final_longitudinal_output(_prep_sm(d_rel=6.23, v_lead=0.25, v_rel=0.10), -0.05, True, 0.0, False)  # type: ignore[arg-type]
  assert should_stop is True
  assert -0.5 < a2 <= -0.20


def test_stop_hold_standstill_normalize_first_positive_release_still_clears():
  sp = fake_planner(LongitudinalMode.SCC, release=True)
  sp.CP = SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)
  sp.custom_long_finalizer.CP = sp.CP
  _arm_stop_hold(sp, d_rel=6.2)
  a_hold, _, _ = sp.final_longitudinal_output(
    FakeSubMaster({
      'carState': SimpleNamespace(vEgo=0.0, standstill=True, brakePressed=False, gasPressed=False, vCruise=12.0),
      'controlsState': SimpleNamespace(forceDecel=False),
      'selfdriveState': SimpleNamespace(experimentalMode=False),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=6.2, vLead=0.0, vRel=0.0, radarTrackId=7)),
    }), -2.0, True, 0.0, False)  # type: ignore[arg-type]
  assert a_hold == -0.5
  sp._lead_stop_hold_gap_increasing_s = 0.30
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=7.0, v_lead=0.8, v_rel=0.3), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert math.isclose(a, RELEASE_FIRST_STEP)
  assert should_stop is False
  assert sp._lead_stop_hold_active is False
  assert sp._stop_hold_release_slew_a_target == a
