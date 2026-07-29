"""Active custom-longitudinal fault policy: Degraded Evidence vs Fail-closed internal faults.

CONTEXT.md: missing/stale/non-finite source data is Degraded Evidence and only withholds
Custom Authority; an unexpected internal fault after Custom Authority begins is Fail-closed —
it latches a stable Fault Class, requests immediateDisable through the planner event path,
and resets automatically at the next engagement.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.cereal import custom
from openpilot.sunnypilot.custom.longitudinal.wiring import FAULT_CLASS_INTERNAL, CustomLongitudinalAdapter

EventNameSP = custom.OnroadEventSP.EventName


class FakeParams:
  def __init__(self, **vals):
    self._v = vals

  def get_bool(self, k):
    return bool(self._v.get(k, False))

  def get(self, k):
    return self._v.get(k)


def make_adapter() -> CustomLongitudinalAdapter:
  return CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc"))


def good_sm(long_active=True):
  return {
    'radarState': SimpleNamespace(leadOne=None, leadTwo=None),
    'carState': SimpleNamespace(brakePressed=False, gasPressed=False, standstill=False,
                                steeringAngleDeg=0.0, steeringTorque=0.0),
    'modelV2': SimpleNamespace(action=SimpleNamespace(shouldStop=False, desiredAcceleration=0.0),
                               position=SimpleNamespace(), velocity=SimpleNamespace(x=None), leadsV3=[]),
    'carControl': SimpleNamespace(orientationNED=[0.0, 0.0, 0.0], longActive=long_active),
    'controlsState': SimpleNamespace(forceDecel=False),
  }


def fake_scc():
  return SimpleNamespace(
    vision=SimpleNamespace(is_active=False, output_a_target=0.0, state=0,
                           current_lat_acc=0.0, max_pred_lat_acc=0.0, pre_entry_active=False),
    map=SimpleNamespace(is_active=False, output_a_target=0.0, state=0, target_lat=0.0, target_lon=0.0),
  )


def fake_sla():
  return SimpleNamespace(is_active=False, output_v_target=0.0, output_a_target=0.0)


def run(a, sm, seed=0.3):
  return a.evaluate(sm, 10.0, 0.0, 12.0, seed, fake_scc(), fake_sla(), dt=0.05)


class Boom:
  calls = 0

  def __call__(self, *args, **kwargs):
    type(self).calls += 1
    raise RuntimeError("internal contract violation")


def test_missing_source_is_degraded_evidence_not_a_fault():
  a = make_adapter()
  out = run(a, {})  # every service missing
  assert out.selected_intent == "degraded_evidence"
  assert out.enabled is False
  assert out.a_target == pytest.approx(0.3)  # authority withheld: baseline seed unchanged
  assert out.fault_class == ""
  assert a.fault_class == ""


def test_non_finite_core_evidence_is_degraded_not_a_fault():
  a = make_adapter()
  out = a.evaluate(good_sm(), float('nan'), 0.0, 12.0, 0.3, fake_scc(), fake_sla(), dt=0.05)
  assert out.selected_intent == "degraded_evidence"
  assert out.reason == "non_finite_source"
  assert a.fault_class == ""


def test_internal_fault_before_authority_keeps_baseline_compat():
  a = make_adapter()
  a._stack.update = Boom()
  out = run(a, good_sm(long_active=False))
  assert out.selected_intent == "fault"
  assert out.fault_class == ""     # pre-authority: consumer-local baseline compatibility case
  assert a.fault_class == ""
  assert out.a_target == pytest.approx(0.3)


def test_internal_fault_after_authority_is_fail_closed_and_latched():
  a = make_adapter()
  # Authority begins: an engaged, successful evaluate.
  out = run(a, good_sm(long_active=True))
  assert out.enabled is True

  a._stack.update = Boom()
  Boom.calls = 0
  out = run(a, good_sm(long_active=True))
  assert out.selected_intent == "fault"
  assert out.fault_class == FAULT_CLASS_INTERNAL   # stable Fault Class, never raw exception text
  assert a.fault_class == FAULT_CLASS_INTERNAL
  assert out.a_target == pytest.approx(0.3)

  # Latched: no silent resume onto the baseline within the same engagement — the stack is
  # not even consulted again.
  Boom.calls = 0
  out = run(a, good_sm(long_active=True))
  assert out.selected_intent == "fault"
  assert Boom.calls == 0


def test_fault_latch_resets_at_next_engagement():
  a = make_adapter()
  run(a, good_sm(long_active=True))
  a._stack.update = Boom()
  run(a, good_sm(long_active=True))
  assert a.fault_class == FAULT_CLASS_INTERNAL

  # Disengaged: latch holds (still no custom authority).
  del a._stack.update  # drop the instance-level Boom override; the real stack resumes
  out = run(a, good_sm(long_active=False))
  assert a.fault_class == FAULT_CLASS_INTERNAL
  assert out.selected_intent == "fault"

  # Next engagement (longActive rising edge) resets the latch automatically.
  out = run(a, good_sm(long_active=True))
  assert a.fault_class == ""
  assert out.enabled is True
  assert out.selected_intent != "fault"


def test_fault_event_and_alert_mapping():
  from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP
  from openpilot.sunnypilot.selfdrive.selfdrived.events_base import ET

  mapping = EVENTS_SP[EventNameSP.customLongitudinalFault]
  assert set(mapping) == {ET.IMMEDIATE_DISABLE}    # requests the existing immediateDisable path
  assert mapping[ET.IMMEDIATE_DISABLE].alert_text_2 == "Custom Longitudinal Fault"


def test_state_machine_events_bridges_only_the_fault():
  from openpilot.selfdrive.selfdrived.events import Events
  from openpilot.selfdrive.selfdrived.state import StateMachine, State
  from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP, StateMachineEvents
  from openpilot.sunnypilot.selfdrive.selfdrived.events_base import ET

  events, events_sp = Events(), EventsSP()
  view = StateMachineEvents(events, events_sp)
  assert view.contains(ET.IMMEDIATE_DISABLE) is False

  # MADS-owned SP event types never cross into the main state machine.
  events_sp.add(EventNameSP.lkasEnable)
  assert view.contains(ET.ENABLE) is False
  assert view.contains(ET.IMMEDIATE_DISABLE) is False

  events_sp.add(EventNameSP.customLongitudinalFault)
  assert view.contains(ET.IMMEDIATE_DISABLE) is True

  sm = StateMachine()
  sm.state = State.enabled
  enabled, active = sm.update(StateMachineEvents(events, events_sp))
  assert enabled is False and active is False
  assert ET.IMMEDIATE_DISABLE in sm.current_alert_types
