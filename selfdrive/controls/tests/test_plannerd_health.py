from types import SimpleNamespace

from openpilot.selfdrive.controls import plannerd


class FakeSubMaster:
  frame = 42

  def __init__(self):
    self.recv_time = {"modelV2": 9.90, "radarState": 9.80, "carState": 9.75}


def test_planner_health_snapshot_defaults_safely_for_missing_debug():
  snapshot = plannerd._planner_health_snapshot(
    FakeSubMaster(), SimpleNamespace(), update_reason="modelV2_updated", now=10.0,
  )

  assert snapshot["model_age_ms"] == 100.0
  assert snapshot["radar_age_ms"] == 200.0
  assert snapshot["car_state_age_ms"] == 250.0
  assert snapshot["update_reason"] == "modelV2_updated"
  assert snapshot["skipped_reason"] == ""
  assert snapshot["last_valid_plan_age_ms"] == 0.0
  assert snapshot["primary_lead"] == {}
  assert snapshot["speed_limit_handoff"] == {}


def test_planner_health_snapshot_includes_primary_lead_and_speed_limit_debug():
  primary_context = SimpleNamespace(debug_dict=lambda: {"primary_physical_lead_idx": 1, "shadow_lead_active": True})
  planner = SimpleNamespace(
    primary_lead_context=primary_context,
    speed_limit_handoff_debug={"speed_limit_handoff_active": True, "reason": "handoff_active"},
  )

  snapshot = plannerd._planner_health_snapshot(
    FakeSubMaster(), planner, skipped_reason="modelV2_not_updated", last_valid_plan_time=8.5, now=10.0,
  )

  assert snapshot["skipped_reason"] == "modelV2_not_updated"
  assert snapshot["last_valid_plan_age_ms"] == 1500.0
  assert snapshot["primary_lead"]["primary_physical_lead_idx"] == 1
  assert snapshot["speed_limit_handoff"]["speed_limit_handoff_active"] is True


def test_planner_health_debug_logs_on_signature_change_and_rate_limit(monkeypatch):
  events = []
  times = iter([10.0, 10.1, 16.0])
  monkeypatch.setattr(plannerd.time, "monotonic", lambda: next(times))
  monkeypatch.setattr(plannerd.cloudlog, "event", lambda name, **kwargs: events.append((name, kwargs)))

  signature, log_time = plannerd._log_planner_health_debug(
    FakeSubMaster(), SimpleNamespace(), None, 0.0, update_reason="modelV2_updated", min_interval=5.0,
  )
  same_signature, same_log_time = plannerd._log_planner_health_debug(
    FakeSubMaster(), SimpleNamespace(), signature, log_time, update_reason="modelV2_updated", min_interval=5.0,
  )
  _signature, later_log_time = plannerd._log_planner_health_debug(
    FakeSubMaster(), SimpleNamespace(), same_signature, same_log_time, update_reason="modelV2_updated", min_interval=5.0,
  )

  assert len(events) == 2
  assert events[0][0] == "plannerd_health_debug"
  assert later_log_time == 16.0
