from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.evidence.builder import build_snapshot
from openpilot.sunnypilot.custom.evidence.types import EvidenceSnapshot, SourceHealth, SourceStatus


def _healthy_model(**action_overrides):
  action = dict(shouldStop=True, desiredAcceleration=-1.2, desiredCurvature=0.004)
  action.update(action_overrides)
  return SimpleNamespace(
    position=SimpleNamespace(x=tuple(float(i) for i in range(17)), y=tuple(float(i) * 0.1 for i in range(17)), yStd=tuple(0.1 for _ in range(17))),
    orientation=SimpleNamespace(z=[0.01, 0.02, 0.03]),
    orientationRate=SimpleNamespace(z=[0.0, 0.01, 0.02]),
    laneLineProbs=[0.9, 0.8], frameDropPerc=1.5,
    action=SimpleNamespace(**action),
    velocity=SimpleNamespace(x=tuple([5.0] * 16 + [0.1])),
  )


def test_healthy_sources_build_finite_snapshot():
  car = SimpleNamespace(vEgo=12.3, aEgo=-0.4, standstill=False, brakePressed=True, gasPressed=False, steeringPressed=True)
  model = _healthy_model()
  lead = SimpleNamespace(status=True, dRel=25.0, yRel=0.5, vRel=-3.0, vLeadK=8.0, vLead=9.0, aLeadK=-0.1, radarTrackId=7, modelProb=0.6)
  snap = build_snapshot(car_state=car, model_v2=model, radar_state=SimpleNamespace(leadOne=lead, leadTwo=None), source_status={"carState": SourceStatus(SourceHealth.HEALTHY, ("ok",)), "modelV2": SourceStatus(SourceHealth.HEALTHY, ("ok",)), "radarState": SourceStatus(SourceHealth.HEALTHY, ("ok",))})
  assert isinstance(snap, EvidenceSnapshot)
  assert snap.ego.v_ego_mps == pytest.approx(12.3)
  assert snap.model_path.source.health is SourceHealth.HEALTHY
  assert snap.model_action.source.health is SourceHealth.HEALTHY
  assert snap.lead.source.health is SourceHealth.HEALTHY
  assert snap.model_action.model_stop_distance_m == pytest.approx(16.0)
  assert snap.lead.lead_one.v_lead_mps == pytest.approx(8.0)
  assert snap.lead.lead_one.ttc_s == pytest.approx(25.0 / 3.0)


def test_missing_sources_are_unknown_and_empty():
  snap = build_snapshot()
  assert snap.ego.source.health is SourceHealth.UNAVAILABLE
  assert snap.model_path.source.health is SourceHealth.UNAVAILABLE
  assert snap.model_action.source.health is SourceHealth.UNAVAILABLE
  assert snap.lead.source.health is SourceHealth.UNAVAILABLE
  assert snap.lead.lead_one.source.health is SourceHealth.UNAVAILABLE
  assert snap.model_path.position_x_m == ()
  assert snap.ego.v_ego_mps is None
  assert snap.ego.source.reasons == ("missing_source",)


def test_partial_source_status_keeps_present_sources_unknown_not_unavailable():
  car = SimpleNamespace(vEgo=1.0)
  snap = build_snapshot(car_state=car, source_status={"modelV2": SourceStatus(SourceHealth.HEALTHY, ("ok",))})
  assert snap.ego.source.health is SourceHealth.UNKNOWN
  assert snap.ego.source.reasons == ("unknown_freshness",)
  assert dict(snap.source_statuses)["carState"].health is SourceHealth.UNKNOWN


def test_valid_model_without_metadata_remains_unknown_not_healthy():
  snap = build_snapshot(model_v2=_healthy_model())
  assert snap.model_path.source.health is SourceHealth.UNKNOWN
  assert snap.model_path.source.reasons == ("unknown_freshness",)


def test_nonfinite_values_become_none():
  car = SimpleNamespace(vEgo=float("nan"), aEgo=float("inf"), standstill=1, brakePressed=None)
  model = SimpleNamespace(position=SimpleNamespace(x=[0.0, float("nan")], y=[float("inf")], yStd=[0.1]), orientation=SimpleNamespace(z=[float("nan")]), orientationRate=SimpleNamespace(z=[float("inf")]), laneLineProbs=[float("nan")], frameDropPerc=float("inf"), action=SimpleNamespace(shouldStop=None, desiredAcceleration=float("nan"), desiredCurvature=float("inf")), velocity=SimpleNamespace(x=[1.0, 0.0]))
  snap = build_snapshot(car_state=car, model_v2=model, source_status={"carState": SourceStatus(SourceHealth.HEALTHY, ()), "modelV2": SourceStatus(SourceHealth.HEALTHY, ()), "radarState": SourceStatus(SourceHealth.HEALTHY, ())})
  assert snap.ego.v_ego_mps is None
  assert snap.ego.source.health is SourceHealth.DEGRADED
  assert "nonfinite_field" in snap.ego.source.reasons
  assert snap.model_path.position_x_m == (0.0, None)
  assert snap.model_path.source.health is SourceHealth.DEGRADED
  assert "nonfinite_field" in snap.model_path.source.reasons
  assert snap.model_action.desired_acceleration_mps2 is None
  assert snap.model_action.model_stop_distance_m is None


def test_nonfinite_desired_curvature_degrades_path_not_model_action():
  snap = build_snapshot(
    model_v2=_healthy_model(desiredCurvature=float("inf")),
    source_status={"modelV2": SourceStatus(SourceHealth.HEALTHY, ())},
  )
  assert snap.model_path.desired_curvature_1_m is None
  assert snap.model_path.source.health is SourceHealth.DEGRADED
  assert "nonfinite_field" in snap.model_path.source.reasons
  assert snap.model_action.desired_acceleration_mps2 == pytest.approx(-1.2)
  assert snap.model_action.source.health is SourceHealth.HEALTHY


def test_optional_booleans_missing_are_none():
  snap = build_snapshot(car_state=SimpleNamespace(), model_v2=SimpleNamespace(), radar_state=SimpleNamespace())
  assert snap.ego.brake_pressed is None
  assert snap.model_action.should_stop is None


def test_snapshot_is_frozen_and_not_raw_message():
  snap = build_snapshot(car_state=SimpleNamespace(vEgo=1.0), model_v2=SimpleNamespace(), radar_state=SimpleNamespace())
  with pytest.raises(Exception):
    snap.ego.v_ego_mps = 2.0
  assert not hasattr(snap, "carState")
  assert isinstance(snap.source_statuses, tuple)
  assert not hasattr(snap, "source_status")


def test_ttc_and_time_headway_policy_free():
  lead = SimpleNamespace(status=True, dRel=20.0, vRel=-5.0)
  snap = build_snapshot(car_state=SimpleNamespace(vEgo=10.0), radar_state=SimpleNamespace(leadOne=lead, leadTwo=lead))
  assert snap.lead.lead_one.ttc_s == pytest.approx(4.0)
  assert snap.lead.lead_one.time_headway_s == pytest.approx(2.0)


def test_lead_false_or_missing_is_unavailable_and_no_metrics():
  snap_false = build_snapshot(car_state=SimpleNamespace(vEgo=10.0), radar_state=SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=10.0, vRel=-2.0), leadTwo=None))
  assert snap_false.lead.lead_one.status is False
  assert snap_false.lead.lead_one.source.health is SourceHealth.UNAVAILABLE
  assert snap_false.lead.lead_one.ttc_s is None
  snap_bad = build_snapshot(car_state=SimpleNamespace(vEgo=10.0), radar_state=SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=float("nan"), vRel=-2.0), leadTwo=None))
  assert snap_bad.lead.lead_one.source.health is SourceHealth.DEGRADED
  assert snap_bad.lead.lead_one.ttc_s is None
  snap_missing = build_snapshot(car_state=SimpleNamespace(vEgo=10.0), radar_state=SimpleNamespace())
  assert snap_missing.lead.lead_one.status is None
  assert snap_missing.lead.lead_one.source.health is SourceHealth.UNAVAILABLE


def test_iterable_model_arrays_and_explicit_unknown_metadata():
  class FakeArray:
    def __init__(self, values): self._values = values
    def __iter__(self): return iter(self._values)
    def __len__(self): return len(self._values)
    def __bool__(self): raise AssertionError("builder must not use raw array truthiness")
  snap = build_snapshot(car_state=SimpleNamespace(), model_v2=SimpleNamespace(position=SimpleNamespace(x=FakeArray([0.0]*17), y=FakeArray([0.0]*17), yStd=FakeArray([0.1]*17)), orientation=SimpleNamespace(z=FakeArray([0.0]*17)), orientationRate=SimpleNamespace(z=FakeArray([0.0]*17)), laneLineProbs=FakeArray([0.5, 0.6]), action=SimpleNamespace(), velocity=SimpleNamespace(x=FakeArray([1.0]*17))), source_status={"modelV2": SourceStatus(SourceHealth.HEALTHY, ()), "carState": None, "radarState": None})
  assert len(snap.model_path.position_x_m) == 17
  assert snap.model_path.source.health is SourceHealth.HEALTHY
