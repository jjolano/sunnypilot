"""Research longitudinal actuation gate."""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.custom.longitudinal.research_actuation import research_actuation_allowed


class FakeParams:
  def __init__(self, **vals):
    self._v = vals

  def get_bool(self, k):
    return bool(self._v.get(k, False))


def make_cp(openpilot_longitudinal_control=True):
  return SimpleNamespace(openpilotLongitudinalControl=openpilot_longitudinal_control)


def test_gate_true_only_with_all_conditions():
  params = FakeParams(CustomLongitudinalEnabled=True, AllowLongitudinalResearchActuation=True)
  cp = make_cp(True)
  assert research_actuation_allowed(params, cp) is True


def test_gate_false_without_openpilot_longitudinal_control():
  params = FakeParams(CustomLongitudinalEnabled=True, AllowLongitudinalResearchActuation=True)
  cp = make_cp(False)
  assert research_actuation_allowed(params, cp) is False


def test_gate_false_without_custom_longitudinal_enabled():
  params = FakeParams(CustomLongitudinalEnabled=False, AllowLongitudinalResearchActuation=True)
  cp = make_cp(True)
  assert research_actuation_allowed(params, cp) is False


def test_gate_false_without_research_param():
  params = FakeParams(CustomLongitudinalEnabled=True, AllowLongitudinalResearchActuation=False)
  cp = make_cp(True)
  assert research_actuation_allowed(params, cp) is False


def test_gate_fail_closed_on_param_error():
  class BadParams:
    def get_bool(self, _k):
      raise RuntimeError("param fault")

  assert research_actuation_allowed(BadParams(), make_cp(True)) is False
