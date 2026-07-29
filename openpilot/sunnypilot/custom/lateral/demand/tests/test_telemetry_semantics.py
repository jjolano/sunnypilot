"""Raw / Conditioned / Processed lateral demand telemetry semantics (CONTEXT.md).

- rawDesiredCurvature          = raw model/maneuver desired curvature (pre-pipeline input)
- conditionedDesiredCurvature  = pipeline result before hard caps (Conditioned Lateral Demand)
- processedDesiredCurvature    = post-cap controller-facing curvature (Processed Lateral Demand)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.cereal import log
from openpilot.sunnypilot.custom.lateral.demand.telemetry import publish_model_path_state


class FakeSM:
  def __init__(self):
    self._d = {'longitudinalPlan': SimpleNamespace(speeds=[], accels=[])}
    self.valid = {'longitudinalPlan': False}
    self.alive = {'longitudinalPlan': False}
    self.freq_ok = {'longitudinalPlan': False}
    self.logMonoTime = {'longitudinalPlan': 0}

  def __getitem__(self, key):
    return self._d[key]


def make_model_path_state():
  return log.ControlsState.new_message().modelPathState


def make_demand(raw: float, conditioned: float):
  zeros = dict.fromkeys((
    'lane_centering_lateral_error', 'lane_centering_heading_error', 'lane_centering_predicted_error',
    'lane_centering_curvature_nudge', 'lane_centering_confidence', 'lane_centering_relax_envelope',
    'lane_centering_relax_lateral_error', 'lane_centering_relax_predicted_error', 'lane_centering_relax_age',
    'lane_centering_relax_nudge_flip_score', 'lane_centering_relax_error_cross_score',
  ), 0.0)
  return SimpleNamespace(
    raw_curvature=raw, processed_curvature=conditioned,
    lane_centering_assist_active=False, lane_centering_reason="off",
    lane_centering_relax_active=False, lane_centering_relax_reason_bits=0,
    **zeros,
  )


def make_lateral_demand(raw: float, conditioned: float):
  last_result = SimpleNamespace(
    demand=make_demand(raw, conditioned),
    model_path_result=SimpleNamespace(gated=False, quality=1.0, reason="ok"),
    debug={},
  )
  return SimpleNamespace(enabled=True, last_result=last_result, last_debug={}, clear=lambda: None)


def test_raw_conditioned_processed_are_distinct_and_unambiguous():
  mps = make_model_path_state()
  lateral_demand = make_lateral_demand(raw=0.010, conditioned=0.020)
  publish_model_path_state(mps, FakeSM(), lateral_demand, 10.0, 0.0,
                           raw_desired_curvature=0.010, processed_desired_curvature=0.015,
                           lat_delay=0.2)
  assert mps.rawDesiredCurvature == pytest.approx(0.010)          # raw input
  assert mps.conditionedDesiredCurvature == pytest.approx(0.020)  # pipeline result pre-cap
  assert mps.processedDesiredCurvature == pytest.approx(0.015)    # post-cap controller input


def test_disabled_defaults_keep_processed_as_post_cap_value():
  mps = make_model_path_state()
  lateral_demand = SimpleNamespace(enabled=False, last_result=None, last_debug={}, clear=lambda: None)
  publish_model_path_state(mps, FakeSM(), lateral_demand, 10.0, 0.0,
                           raw_desired_curvature=0.010, processed_desired_curvature=0.012,
                           lat_delay=0.2)
  assert mps.active is False
  assert mps.rawDesiredCurvature == pytest.approx(0.010)
  # No conditioning happened: conditioned == raw, processed is still the controller input.
  assert mps.conditionedDesiredCurvature == pytest.approx(0.010)
  assert mps.processedDesiredCurvature == pytest.approx(0.012)
