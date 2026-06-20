"""Focused tests for the first DEC -> SCC migration slice."""
from __future__ import annotations

from types import SimpleNamespace
import math

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter, CustomLongitudinalOutput
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource


class FakeParams:
  def __init__(self, **vals): self._v = vals
  def get_bool(self, k): return bool(self._v.get(k, False))
  def get(self, k): return self._v.get(k)


def fake_sm(exp_mode=False, brake=False, gas=False, force_decel=False):
  return {  # type: ignore[return-value]
    'selfdriveState': SimpleNamespace(experimentalMode=exp_mode),
    'carState': SimpleNamespace(brakePressed=brake, gasPressed=gas),
    'controlsState': SimpleNamespace(forceDecel=force_decel),
  }


def fake_cp():
  return SimpleNamespace(vEgoStopping=0.5, stoppingDistance=6.0, stopAccel=-0.5, openpilotLongitudinalControl=True)


def fake_planner(mode=LongitudinalMode.SCC, should_stop=False, sources=SourceToggles(), release=False):
  sp = object.__new__(LongitudinalPlannerSP)
  sp.__dict__['CP'] = fake_cp()
  sp.__dict__['custom_long'] = SimpleNamespace(enabled=True, mode=mode, sources=sources,
                                                maybe_refresh_params=lambda: None)
  sp.__dict__['dec'] = SimpleNamespace(active=lambda: True, mode=lambda: 'blended')
  sp.__dict__['dt'] = 0.05
  sp.__dict__['_lead_stop_hold_active'] = False
  sp.__dict__['_lead_stop_hold_gap_increasing_s'] = 0.0
  sp.__dict__['_lead_stop_hold_missing_s'] = 0.0
  sp.__dict__['_lead_stop_hold_lead_id'] = None
  sp.__dict__['_lead_stop_hold_gap_prev_d_rel'] = None
  sp.__dict__['custom_long_output'] = CustomLongitudinalOutput(
    a_target=0.0, should_stop=should_stop, enabled=True, mode=mode,
    selected_intent=("lead_pullaway" if release else None), reason=("trusted" if release else None),
    standstill_release_allowed=release, standstill_release_source=("lead_pullaway" if release else ""),
    standstill_release_a_target=(0.2 if release else 0.0), standstill_release_reason=("trusted" if release else ""),
    debug={})
  return sp


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
  a._stack.update = lambda *args, **kwargs: SimpleNamespace(a_target=-1.5, should_stop=True, standstill_release_allowed=False, standstill_release_source='', standstill_release_a_target=0.0, standstill_release_reason='', debug={'intent': 'e2e', 'reason': 'trusted'})  # type: ignore[assignment]
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


def test_scc_stop_approach_custom_cap_applies_only_for_trusted_stop_intent():
  sp = fake_planner(LongitudinalMode.SCC, should_stop=True)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=-1.2, should_stop=True, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="stop_approach", reason="model_stop", debug={},
  )  # type: ignore[assignment]
  a, should_stop, e2e_source = sp.final_longitudinal_output(fake_sm(True), -0.2, True, -0.3, True)  # type: ignore[arg-type]
  assert a == -1.2
  assert should_stop is True
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
  sp = fake_planner(LongitudinalMode.ACC)
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
  assert sp.custom_long_output is not None
  assert sp.custom_long_output.selected_intent == "lead_stop_hold"
  assert sp.custom_long_output.reason == "stopped_lead_latch"
  assert sp.custom_long_output.should_stop is True


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


def _arm_stop_hold(sp):
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


def _release_sm(d_rel=6.5, v_lead=0.4, v_rel=0.2, radar_id=7, brake=False, gas=False, force_decel=False):
  return {
    'carState': SimpleNamespace(vEgo=0.0, brakePressed=brake, gasPressed=gas, vCruise=12.0),
    'controlsState': SimpleNamespace(forceDecel=force_decel),
    'selfdriveState': SimpleNamespace(experimentalMode=False),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(
      status=True, dRel=d_rel, vLead=v_lead, vRel=v_rel, radarTrackId=radar_id,
    )),
  }


def _set_lead_pullaway_release(sp):
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=sp.custom_long.mode,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )


def test_latch_release_same_lead_clears_earlier_with_bounded_accel():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.20
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=6.35, v_lead=0.32, v_rel=0.16), 0.0, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.15 <= a <= 0.35


def test_latch_release_same_lead_allows_slightly_negative_mpc_with_tighter_gap_confirm():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.10
  a, should_stop, _ = sp.final_longitudinal_output(_release_sm(d_rel=6.21, v_lead=0.32, v_rel=0.16), -0.08, True, 0.2, False)  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is False
  assert should_stop is False
  assert 0.15 <= a <= 0.35


def test_latch_release_same_lead_rejects_too_negative_mpc_or_non_opening_or_below_distance():
  sp = fake_planner(LongitudinalMode.ACC)
  _arm_stop_hold(sp)
  sp.custom_long_output = CustomLongitudinalOutput(
    a_target=0.0, should_stop=False, enabled=True, mode=LongitudinalMode.SCC,
    selected_intent="lead_pullaway", reason="trusted",
    standstill_release_allowed=True, standstill_release_source="lead_pullaway",
    standstill_release_a_target=0.4, standstill_release_reason="trusted", debug={},
  )
  sp._lead_stop_hold_gap_increasing_s = 0.10
  assert sp.final_longitudinal_output(_release_sm(d_rel=6.21, v_lead=0.32, v_rel=0.16), -0.2, True, 0.2, False)[1] is True  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert sp.final_longitudinal_output(_release_sm(d_rel=6.21, v_lead=0.29, v_rel=0.16), -0.08, True, 0.2, False)[1] is True  # type: ignore[arg-type]
  assert sp._lead_stop_hold_active is True
  assert sp.final_longitudinal_output(_release_sm(d_rel=6.0, v_lead=0.32, v_rel=0.16), -0.08, True, 0.2, False)[1] is True  # type: ignore[arg-type]
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
  assert 0.15 <= a <= 0.35


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
  assert LongitudinalPlanSource.sccMap in fake_planner(LongitudinalMode.SCC, sources=SourceToggles(False, True)).custom_longitudinal_targets(targets)


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
