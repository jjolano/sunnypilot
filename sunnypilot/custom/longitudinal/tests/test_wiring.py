"""Tests for the plannerd wiring adapter (opt-in custom longitudinal)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode, SourceToggles
from openpilot.sunnypilot.custom.longitudinal.policy_tables import Personality
from openpilot.sunnypilot.custom.longitudinal.wiring import (
  DEFAULT_ACCEL_LIMITS,
  CustomLongitudinalAdapter,
  build_stack_inputs,
)


def lead(d_rel=30.0, v_lead=12.0, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=0.0,
                         yRel=0.0, radarTrackId=3, radar=True, modelProb=0.9, aLeadTau=1.0)


def fake_sm(lead_one=None, brake=False, gas=False):
  return {
    'radarState': SimpleNamespace(leadOne=lead_one, leadTwo=None),
    'carState': SimpleNamespace(brakePressed=brake, gasPressed=gas),
  }


def fake_scc(vision_active=False, vision_a=0.0, map_active=False, map_a=0.0):
  return SimpleNamespace(
    vision=SimpleNamespace(is_active=vision_active, output_a_target=vision_a),
    map=SimpleNamespace(is_active=map_active, output_a_target=map_a),
  )


def fake_sla(active=False, v_target=0.0, a_target=0.0):
  return SimpleNamespace(is_active=active, output_v_target=v_target, output_a_target=a_target)


class FakeParams:
  def __init__(self, **vals):
    self._v = vals
  def get_bool(self, k):
    return bool(self._v.get(k, False))
  def get(self, k):
    return self._v.get(k)
  def all_keys(self):
    return [k.encode() for k in self._v]


def test_build_stack_inputs_maps_evidence():
  inp = build_stack_inputs(
    v_ego=20.0, a_ego=0.1, v_cruise=22.0, seed_a_target=0.4, accel_limits=DEFAULT_ACCEL_LIMITS,
    lead_one=lead(), lead_two=None,
    scc_vision_active=True, scc_vision_a_target=-0.7, scc_map_active=False, scc_map_a_target=0.0,
    sla_active=True, sla_v_target=18.0, sla_a_target=-0.5,
    mode=LongitudinalMode.SCC, personality=Personality.STANDARD, sources=SourceToggles(True, False),
  )
  assert inp.curve_active is True and inp.curve_a_target == pytest.approx(-0.7)
  assert inp.speed_limit_active is True and inp.speed_limit_a_target == pytest.approx(-0.5)
  assert inp.lead_a_target == pytest.approx(0.4)   # MPC baseline carried as lead-follow accel
  assert inp.model_should_stop is False            # conservatively defaulted (harness-gated)


def test_adapter_disabled_passthrough():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=False))
  out = a.apply(fake_sm(lead()), 20.0, 0.0, 22.0, 0.42, fake_scc(), fake_sla())
  assert out == 0.42


def test_adapter_enabled_shapes_and_bounds():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="scc",
                                           LongitudinalPersonality="1"))
  # curve cap active -> shaped target should brake relative to the cruise seed
  out = a.apply(fake_sm(lead(d_rel=40.0)), 20.0, 0.0, 22.0, 0.5,
                fake_scc(vision_active=True, vision_a=-0.6), fake_sla())
  assert DEFAULT_ACCEL_LIMITS[0] <= out <= DEFAULT_ACCEL_LIMITS[1]


def test_adapter_fail_closed_on_bad_sm():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True))
  # sm missing radarState -> apply must return the seed unchanged, never raise
  out = a.apply({}, 20.0, 0.0, 22.0, 0.33, fake_scc(), fake_sla())
  assert out == 0.33


def test_adapter_acc_ignores_curve():
  a = CustomLongitudinalAdapter(FakeParams(CustomLongitudinalEnabled=True, CustomLongitudinalMode="acc"))
  out = a.apply(fake_sm(), 20.0, 0.0, 22.0, 0.4, fake_scc(vision_active=True, vision_a=-1.0), fake_sla())
  assert out == pytest.approx(0.4)  # ACC excludes curve evidence -> cruise stands
