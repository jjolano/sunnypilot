from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import replay_sps_mode


class FakeMsg(SimpleNamespace):
  def which(self) -> str:
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self) -> str:
    return "torqueState"


def _msg(kind: str, t_s: float, payload: SimpleNamespace) -> FakeMsg:
  return FakeMsg(kind=kind, logMonoTime=round(t_s * 1e9), **{kind: payload})


def _model_v2(curvature: float = 0.0003) -> SimpleNamespace:
  xs = [float(i) for i in range(33)]
  return SimpleNamespace(
    position=SimpleNamespace(x=xs, y=[0.5 * curvature * x * x for x in xs], yStd=[0.05] * len(xs)),
    orientation=SimpleNamespace(z=[curvature * x for x in xs]),
    orientationRate=SimpleNamespace(z=[curvature * 20.0] * len(xs)),
    laneLineProbs=[0.9] * 4,
    frameDropPerc=0.0,
    meta=SimpleNamespace(laneChangeState=0, laneChangeDirection=0),
  )


def _model_path(mode: str, curvature: float = 0.0003, *, complete: bool = False,
                missing_sps_mode: bool = False) -> SimpleNamespace:
  values: dict[str, object]
  if not complete:
    values = {
      "active": True,
      "rawDesiredCurvature": curvature,
      "spsMode": mode,
      "previewAssistMode": "off",
      "laneRateDampingMode": "off",
      "laneFitSourceMode": "off",
      "laneCenteringOneLineMode": "off",
    }
  else:
    values = {field: False for field in replay_sps_mode.BOOLEAN_FIELDS}
    values.update({field: "disabled" for field in replay_sps_mode.TEXT_FIELDS})
    values.update({field: 0.0 for field in replay_sps_mode.CURVATURE_FIELDS})
    values.update({field: 0.0 for field in replay_sps_mode.QUALITY_FIELDS})
    values.update({field: 0.0 for field in replay_sps_mode.STAGE_FIELDS})
    values.update({
      "active": True,
      "rawDesiredCurvature": curvature,
      "conditionedDesiredCurvature": curvature,
      "processedDesiredCurvature": curvature,
      "modelPathCurvature": curvature,
      "spsMode": mode,
      "spsActive": mode != "off",
      "spsApplied": mode == "apply",
      "spsReason": "ok" if mode != "off" else "disabled",
      "previewAssistMode": "off",
      "laneRateDampingMode": "off",
      "laneFitSourceMode": "off",
      "laneCenteringOneLineMode": "off",
    })
  if missing_sps_mode:
    values.pop("spsMode", None)
  return SimpleNamespace(**values)


def _controls_state(mode: str, curvature: float = 0.0003, *, complete: bool = False,
                    missing_sps_mode: bool = False) -> SimpleNamespace:
  return SimpleNamespace(
    curvature=curvature,
    modelPathState=_model_path(mode, curvature, complete=complete, missing_sps_mode=missing_sps_mode),
    lateralControlState=FakeUnion(
      torqueState=SimpleNamespace(
        adaptiveTorqueState=SimpleNamespace(steerLimitLimited=False),
      ),
    ),
  )


def _init_data() -> SimpleNamespace:
  return SimpleNamespace(
    params=SimpleNamespace(entries=[
      SimpleNamespace(key="LaneCenteringAssistEnabled", value=b"0"),
      # This route-start value must not override runtime modelPathState telemetry.
      SimpleNamespace(key="StraightPathStabilizationMode", value=b"shadow"),
    ]),
  )


def _route(
  modes: list[str],
  *,
  include_init: bool = True,
  include_model: bool = True,
  complete_path: bool = False,
  missing_sps_mode: bool = False,
  model_hz: int = 20,
  control_hz: int = 100,
) -> list[FakeMsg]:
  assert control_hz % model_hz == 0
  messages: list[FakeMsg] = []
  if include_init:
    messages.append(_msg("initData", 0.0, _init_data()))

  model_stride = control_hz // model_hz
  for index, mode in enumerate(modes):
    t_s = index / control_hz
    if include_model and index % model_stride == 0:
      messages.append(_msg("modelV2", t_s, _model_v2()))
    messages.extend((
      _msg("carState", t_s, SimpleNamespace(
        vEgo=20.0,
        steeringPressed=False,
        steeringRateDeg=0.0,
        leftBlinker=False,
        rightBlinker=False,
      )),
      _msg("liveParameters", t_s, SimpleNamespace(roll=0.0)),
      _msg("controlsState", t_s, _controls_state(
        mode, complete=complete_path, missing_sps_mode=missing_sps_mode,
      )),
      _msg("carControl", t_s + 0.0003, SimpleNamespace(latActive=True)),
    ))
  return messages


def _contract_init_data(*, lagd_toggle: bytes = b"0", lagd_value_cache: bytes = b"0.0") -> SimpleNamespace:
  payload = _init_data()
  payload.params.entries.extend((
    SimpleNamespace(key="LagdToggle", value=lagd_toggle),
    SimpleNamespace(key="LagdValueCache", value=lagd_value_cache),
  ))
  return payload


def _marked_model(frame_id: str) -> SimpleNamespace:
  model = _model_v2()
  model.frameId = frame_id
  return model


def _live_pose(yaw_rate: float) -> SimpleNamespace:
  measurement = dict(x=0.0, y=0.0, z=0.0, xStd=1.0, yStd=1.0, zStd=1.0)
  return SimpleNamespace(
    orientationNED=SimpleNamespace(**measurement),
    velocityDevice=SimpleNamespace(**measurement),
    accelerationDevice=SimpleNamespace(**measurement),
    angularVelocityDevice=SimpleNamespace(**{**measurement, "z": yaw_rate}),
  )


def _contract_frame(
  t_s: float,
  *,
  plan_time: int,
  models: tuple[tuple[float, SimpleNamespace], ...] = (),
  controls_active: bool = True,
  car_control_active: bool = False,
  model_path_active: bool = True,
  include_car_control: bool = True,
) -> list[FakeMsg]:
  messages = [_msg("modelV2", model_t, model) for model_t, model in models]
  messages.extend((
    _msg("carState", t_s, SimpleNamespace(
      vEgo=20.0,
      steeringPressed=False,
      steeringRateDeg=0.0,
      leftBlinker=False,
      rightBlinker=False,
    )),
    _msg("liveParameters", t_s, SimpleNamespace(roll=0.0)),
  ))
  controls = _controls_state("apply")
  controls.modelPathState.active = model_path_active
  controls.lateralPlanMonoTime = plan_time
  controls.lateralControlState.torqueState.active = controls_active
  messages.append(_msg("controlsState", t_s, controls))
  # Deliberately after controlsState and inside its timestamp interval.
  if include_car_control:
    messages.append(_msg("carControl", t_s + 0.0003, SimpleNamespace(latActive=car_control_active)))
  return messages


def _capture_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
  rows: list[dict] = []
  original = replay_sps_mode._forced_shadow_report

  def capture(selected_rows: list[dict]) -> dict:
    rows.extend(selected_rows)
    return original(selected_rows)

  monkeypatch.setattr(replay_sps_mode, "_forced_shadow_report", capture)
  return rows


def _stub_replay(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                 previous_curvature):
  del adapter, params, previous_curvature_limited, previous_curvature
  return 0.0, 0.0, False, replay_sps_mode._default_telemetry(context.raw_curvature, 0.0)


def test_replay_live_delay_retains_last_finite_value_after_invalid_sample(monkeypatch: pytest.MonkeyPatch):
  delays: list[float] = []

  def capture_delay(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                    previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      delays.append(context.lat_delay)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_delay)
  messages = [_msg("initData", 0.0, _contract_init_data(lagd_toggle=b"1"))]
  messages.append(_msg("liveDelay", 0.05, SimpleNamespace(lateralDelay=0.27)))
  messages.extend(_contract_frame(
    0.1,
    models=((0.0, _marked_model("delay")),),
    plan_time=0,
  ))
  messages.append(_msg("liveDelay", 0.15, SimpleNamespace(lateralDelay=float("nan"))))
  messages.extend(_contract_frame(0.2, plan_time=0))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.1, window_end_s=0.2)

  assert delays == [0.27, 0.27]
  assert report.context["scored_frames_replayed"] == 2
  assert report.context["passed"] is True


def test_replay_uses_logged_delay_cache_until_a_valid_live_delay_exists(monkeypatch: pytest.MonkeyPatch):
  delays: list[float] = []

  def capture_delay(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                    previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      delays.append(context.lat_delay)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_delay)
  messages = [_msg(
    "initData", 0.0,
    _contract_init_data(lagd_toggle=b"1", lagd_value_cache=b"0.31"),
  )]
  messages.extend(_contract_frame(
    0.1,
    models=((0.0, _marked_model("cache")),),
    plan_time=0,
  ))
  messages.append(_msg("liveDelay", 0.15, SimpleNamespace(lateralDelay=float("nan"))))
  messages.extend(_contract_frame(0.2, plan_time=0))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.1, window_end_s=0.2)

  assert delays == [0.31, 0.31]
  assert report.context["initData_latency_params"] == {"LagdToggle": True, "LagdValueCache": 0.31}
  assert report.context["scored_frames_replayed"] == 2
  assert report.context["passed"] is True


def test_car_control_yaw_pairs_by_timestamp_interval_when_serialization_is_batched():
  cs0_t, cs1_t = 1.0, 2.0
  records = replay_sps_mode.build_route_messages([
    _msg("carControl", cs0_t + 0.0003, SimpleNamespace(angularVelocity=[0.0, 0.0, 0.11])),
    _msg("carControl", cs1_t + 0.0003, SimpleNamespace(angularVelocity=[0.0, 0.0, 0.22])),
    _msg("controlsState", cs0_t, SimpleNamespace()),
    _msg("controlsState", cs1_t, SimpleNamespace()),
  ])

  yaw_by_controls = replay_sps_mode._car_control_yaw_by_controls(records)

  assert yaw_by_controls[round(cs0_t * 1e9)] == 0.11
  assert yaw_by_controls[round(cs1_t * 1e9)] == 0.22


def test_car_control_yaw_does_not_borrow_from_the_next_timestamp_interval():
  cs0_t, cs1_t = 1.0, 2.0
  records = replay_sps_mode.build_route_messages([
    _msg("carControl", cs1_t + 0.0003, SimpleNamespace(angularVelocity=[0.0, 0.0, 0.22])),
    _msg("controlsState", cs0_t, SimpleNamespace()),
    _msg("controlsState", cs1_t, SimpleNamespace()),
  ])

  yaw_by_controls = replay_sps_mode._car_control_yaw_by_controls(records)

  assert yaw_by_controls[round(cs0_t * 1e9)] is replay_sps_mode._MISSING
  assert yaw_by_controls[round(cs1_t * 1e9)] == 0.22


def test_car_control_yaw_pairing_has_linear_property_reads():
  reads = {"typ": 0, "log_mono_time": 0}

  class CountingRecord:
    def __init__(self, typ: str, log_mono_time: int, payload: SimpleNamespace):
      self._typ = typ
      self._log_mono_time = log_mono_time
      self.payload = payload

    @property
    def typ(self) -> str:
      reads["typ"] += 1
      return self._typ

    @property
    def log_mono_time(self) -> int:
      reads["log_mono_time"] += 1
      return self._log_mono_time

  count = 128
  period_ns = 1_000_000
  records: list[CountingRecord] = []
  for index in range(count):
    controls_time = index * period_ns
    records.extend((
      CountingRecord("controlsState", controls_time, SimpleNamespace()),
      CountingRecord(
        "carControl", controls_time + 300_000,
        SimpleNamespace(angularVelocity=[0.0, 0.0, float(index)]),
      ),
    ))

  yaw_by_controls = replay_sps_mode._car_control_yaw_by_controls(records)

  assert [yaw_by_controls[index * period_ns] for index in range(count)] == [float(index) for index in range(count)]
  assert reads["typ"] + reads["log_mono_time"] <= 12 * len(records)


def test_replay_binds_controls_state_to_lateral_plan_model_not_latest_model(monkeypatch: pytest.MonkeyPatch):
  bound_frame_ids: list[str] = []

  def capture_model(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                    previous_curvature):
    del adapter, params, forced_shadow, previous_curvature_limited, previous_curvature
    bound_frame_ids.append(context.model_v2.frameId)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_model)
  model_n = _marked_model("N")
  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(_contract_frame(
    0.3,
    models=((0.1, model_n), (0.2, _marked_model("N+1"))),
    plan_time=round(0.1 * 1e9),
  ))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.3, window_end_s=0.3)

  assert report.context["passed"] is True
  assert bound_frame_ids == ["N", "N"]


def test_replay_uses_controls_lateral_state_when_car_control_follows_it(monkeypatch: pytest.MonkeyPatch):
  active_values: list[bool] = []

  def capture_active(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                     previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      active_values.append(context.lat_active)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_active)
  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(_contract_frame(
    0.1,
    models=((0.0, _marked_model("0")),),
    plan_time=0,
    controls_active=False,
    car_control_active=False,
  ))
  messages.extend(_contract_frame(
    0.2,
    models=((0.1, _marked_model("1")),),
    plan_time=round(0.1 * 1e9),
    controls_active=True,
    car_control_active=False,
  ))

  replay_sps_mode.analyze_route(messages, window_start_s=0.0, window_end_s=0.2)

  assert active_values == [False, True]


def test_replay_uses_same_cycle_car_control_yaw_over_misordered_live_pose(monkeypatch: pytest.MonkeyPatch):
  yaw_rates: list[float | None] = []

  def capture_yaw(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                  previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      yaw_rates.append(context.yaw_rate)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_yaw)
  car_control_yaw = 0.37
  frame = _contract_frame(
    0.2,
    models=((0.1, _marked_model("bound")),),
    plan_time=round(0.1 * 1e9),
  )

  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(frame[:3])
  # Deliberately out of order: this future, contradictory pose must not win.
  messages.append(_msg("livePose", 0.25, _live_pose(-7.0)))
  messages.extend(frame[3:])
  messages[-1].carControl.angularVelocity = [0.0, 0.0, car_control_yaw]

  replay_sps_mode.analyze_route(messages, window_start_s=0.2, window_end_s=0.2)

  assert yaw_rates == [car_control_yaw]


def test_empty_same_cycle_car_control_yaw_is_none_and_keeps_replay_frame_valid(monkeypatch: pytest.MonkeyPatch):
  controls_t = 0.2
  controls = _msg("controlsState", controls_t, SimpleNamespace())
  records = replay_sps_mode.build_route_messages([
    controls,
    _msg("carControl", controls_t + 0.0003, SimpleNamespace(angularVelocity=[])),
  ])
  yaw_by_controls = replay_sps_mode._car_control_yaw_by_controls(records)
  assert yaw_by_controls[controls.logMonoTime] is None
  assert yaw_by_controls[controls.logMonoTime] is not replay_sps_mode._MISSING

  yaw_rates: list[float | None] = []

  def capture_yaw(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                  previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      yaw_rates.append(context.yaw_rate)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_yaw)
  frame = _contract_frame(
    controls_t,
    models=((0.1, _marked_model("bound")),),
    plan_time=round(0.1 * 1e9),
  )
  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(frame[:3])
  messages.append(_msg("livePose", 0.25, _live_pose(-7.0)))
  messages.extend(frame[3:])
  messages[-1].carControl.angularVelocity = []

  report = replay_sps_mode.analyze_route(messages, window_start_s=controls_t, window_end_s=controls_t)

  assert yaw_rates == [None]
  assert report.context["scored_frames_replayed"] == 1
  assert report.context["passed"] is True


def test_replay_rejects_a_future_lateral_plan_model_binding(monkeypatch: pytest.MonkeyPatch):
  bound_frame_ids: list[str] = []

  def capture_model(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                    previous_curvature):
    del adapter, params, forced_shadow, previous_curvature_limited, previous_curvature
    bound_frame_ids.append(context.model_v2.frameId)
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_model)
  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(_contract_frame(
    0.1,
    models=((0.0, _marked_model("past")),),
    plan_time=0,
  ))
  messages.extend(_contract_frame(
    0.2,
    models=((0.3, _marked_model("future")),),
    plan_time=round(0.3 * 1e9),
  ))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.0, window_end_s=0.2)

  assert bound_frame_ids == ["past", "past"]
  assert report.context["scored_frames_replayed"] == 1
  assert report.context["issues"]
  assert report.context["issues"][0]["t_s"] == pytest.approx(0.2)
  assert any("model" in field.lower() for field in report.context["missing_by_field"])
  assert report.context["passed"] is False


def test_replay_tolerates_missing_context_until_both_lateral_flags_are_active(monkeypatch: pytest.MonkeyPatch):
  replayed: list[tuple[str, bool, bool]] = []

  def capture_replay(adapter, context, params, *, forced_shadow, previous_curvature_limited,
                     previous_curvature):
    del adapter, params, previous_curvature_limited, previous_curvature
    if not forced_shadow:
      replayed.append((
        context.model_v2.frameId,
        context.lat_active,
        bool(replay_sps_mode._get(context.model_path_state, "active", False)),
      ))
    return _stub_replay(
      None, context, None, forced_shadow=False,
      previous_curvature_limited=False, previous_curvature=0.0,
    )

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", capture_replay)
  messages = [_msg("initData", 0.0, _contract_init_data())]
  messages.extend(_contract_frame(
    0.0,
    models=((0.0, _marked_model("waiting")),),
    plan_time=0,
    controls_active=False,
    model_path_active=True,
    include_car_control=False,
  ))
  messages.extend(_contract_frame(
    0.1,
    models=((0.1, _marked_model("reset")),),
    plan_time=round(0.1 * 1e9),
    controls_active=False,
    model_path_active=False,
  ))
  messages.extend(_contract_frame(
    0.2,
    models=((0.2, _marked_model("active")),),
    plan_time=round(0.2 * 1e9),
    controls_active=True,
    model_path_active=True,
  ))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.0, window_end_s=0.2)

  assert replayed == [("reset", False, False), ("active", True, True)]
  assert report.context["startup_missing_scored_frames"] == 1
  assert report.context["scored_frames_replayed"] == 2
  assert report.context["issues"] == []
  assert report.context["passed"] is True


def test_replay_invalidates_active_startup_gap_but_tolerates_preprocessing_gap(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(replay_sps_mode, "_run_adapter", _stub_replay)

  preprocessing_gap = [_msg("initData", 0.0, _contract_init_data())]
  preprocessing_gap.extend(_contract_frame(
    0.0,
    plan_time=0,
    model_path_active=False,
  ))
  preprocessing_gap.extend(_contract_frame(
    0.1,
    models=((0.1, _marked_model("complete")),),
    plan_time=round(0.1 * 1e9),
  ))
  tolerated = replay_sps_mode.analyze_route(
    preprocessing_gap, window_start_s=0.0, window_end_s=0.1,
  )

  active_gap = [_msg("initData", 0.0, _contract_init_data())]
  active_gap.extend(_contract_frame(
    0.0,
    models=((0.0, _marked_model("active")),),
    plan_time=0,
    include_car_control=False,
  ))
  active_gap.extend(_contract_frame(
    0.1,
    models=((0.1, _marked_model("complete")),),
    plan_time=round(0.1 * 1e9),
  ))
  invalidated = replay_sps_mode.analyze_route(
    active_gap, window_start_s=0.0, window_end_s=0.1,
  )

  assert tolerated.context["startup_missing_scored_frames"] == 1
  assert tolerated.context["passed"] is True
  assert invalidated.context["startup_missing_scored_frames"] == 0
  assert invalidated.context["issues"][0]["t_s"] == pytest.approx(0.0)
  assert invalidated.context["missing_by_field"]["carControl"] == 1
  assert invalidated.context["passed"] is False


def test_replay_does_not_fall_back_to_live_pose_without_same_cycle_car_control():
  controls = _msg("controlsState", 0.2, SimpleNamespace())
  records = replay_sps_mode.build_route_messages([
    _msg("livePose", 0.1, _live_pose(-7.0)),
    controls,
  ])

  yaw_by_controls = replay_sps_mode._car_control_yaw_by_controls(records)

  assert yaw_by_controls.get(controls.logMonoTime) is replay_sps_mode._MISSING


def test_replay_tolerates_startup_gap_but_invalidates_gap_after_active_replay(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(replay_sps_mode, "_run_adapter", _stub_replay)
  messages = [_msg("initData", 0.0, _contract_init_data())]
  # No model is available for the first scored controlsState.
  messages.extend(_contract_frame(0.0, plan_time=0, model_path_active=False))
  messages.extend(_contract_frame(
    0.1,
    models=((0.1, _marked_model("complete")),),
    plan_time=round(0.1 * 1e9),
  ))
  # The model remains available, but this binding is invalid after replay starts.
  messages.extend(_contract_frame(
    0.2,
    plan_time=999_000_000,
  ))

  report = replay_sps_mode.analyze_route(messages, window_start_s=0.0, window_end_s=0.2)

  context = report.context
  assert context["startup_missing_scored_frames"] == 1
  assert context["required_scored_frames"] == 2
  assert context["scored_frames_replayed"] == 1
  assert context["issues"]
  assert context["issues"][0]["t_s"] == pytest.approx(0.2)
  assert context["passed"] is False
  assert report.valid is False


def test_replay_uses_runtime_sps_schedule_not_route_start_param(monkeypatch: pytest.MonkeyPatch):
  modes = ["shadow"] * 20 + ["apply"] * 60 + ["shadow"] * 20
  rows = _capture_rows(monkeypatch)

  report = replay_sps_mode.analyze_route(
    _route(modes), window_start_s=0.0, window_end_s=0.99,
  )

  assert report.context["initData_params"] == {
    "LaneCenteringAssistEnabled": False,
  }
  assert len(rows) == 100
  assert report.context["scored_frames_replayed"] == 100
  assert [row["recorded"]["spsMode"] for row in rows] == modes
  assert {row["forced"]["spsMode"] for row in rows} == {"shadow"}
  assert any(row["recorded"]["spsApplied"] for row in rows if row["recorded"]["spsMode"] == "apply")


def test_replay_warms_state_before_scoring_late_window(monkeypatch: pytest.MonkeyPatch):
  rows = _capture_rows(monkeypatch)

  report = replay_sps_mode.analyze_route(
    _route(["apply"] * 100), window_start_s=0.8, window_end_s=0.9,
  )

  assert report.context["controls_state_seen"] == 100
  assert report.context["scored_frames"] == 11
  assert report.context["scored_frames_replayed"] == 11
  assert len(rows) == 11
  assert rows[0]["recorded"]["spsActive"] is True
  assert rows[0]["recorded"]["spsApplied"] is True
  assert rows[0]["recorded"]["spsReason"] == "ok"


def test_replay_keeps_apply_authority_and_shadow_state_independent(monkeypatch: pytest.MonkeyPatch):
  rows = _capture_rows(monkeypatch)

  replay_sps_mode.analyze_route(
    _route(["apply"] * 100), window_start_s=0.8, window_end_s=0.9,
  )

  applied = [row for row in rows if row["recorded"]["spsApplied"]]
  assert applied
  assert all(row["recorded"]["spsMode"] == "apply" for row in applied)
  assert all(row["forced"]["spsMode"] == "shadow" for row in rows)
  assert all(row["forced"]["spsApplied"] is False for row in rows)


@pytest.mark.parametrize(("missing", "expected_field"), (
  ("sps", "modelPathState.spsMode"),
  ("model", "modelV2"),
  ("init", "initData.params.LaneCenteringAssistEnabled"),
))
def test_replay_invalidates_report_when_required_context_is_missing(missing: str, expected_field: str):
  report = replay_sps_mode.analyze_route(
    _route(
      ["apply"],
      include_init=missing != "init",
      include_model=missing != "model",
      missing_sps_mode=missing == "sps",
    ),
    window_start_s=0.0,
    window_end_s=0.0,
  )

  assert report.valid is False
  assert report.context["passed"] is False
  assert report.context["scored_frames"] == 1
  assert report.context["scored_frames_replayed"] == 0
  assert report.context["missing_by_field"][expected_field] == 1
  assert report.recorded_arm["parity"]["sample_count"] == 0


def test_replay_aligns_20hz_models_to_100hz_controls_and_passes_parity(monkeypatch: pytest.MonkeyPatch):
  def stub_run_adapter(_adapter, context, _params, *, forced_shadow, previous_curvature_limited,
                       previous_curvature):
    del previous_curvature_limited, previous_curvature
    mode = "shadow" if forced_shadow else context.model_path_state.spsMode
    telemetry = vars(_model_path(mode, context.raw_curvature, complete=True)).copy()
    return context.raw_curvature, context.raw_curvature, False, telemetry

  monkeypatch.setattr(replay_sps_mode, "_run_adapter", stub_run_adapter)
  report = replay_sps_mode.analyze_route(
    _route(["apply"] * 10, complete_path=True, model_hz=20, control_hz=100),
    window_start_s=0.0,
    window_end_s=0.09,
  )

  parity = report.recorded_arm["parity"]
  assert report.valid is True
  assert report.context["scored_frames"] == 10
  assert report.context["scored_frames_replayed"] == 10
  assert parity["sample_count"] == 10
  assert parity["telemetry_exact"]["mismatch_count"] == 0
  assert parity["curvature_errors"]["rawDesiredCurvature"]["passed"] is True
  assert parity["quality_errors"]["quality"]["passed"] is True
  assert parity["tolerances"]["curvature_max"] == replay_sps_mode.CURVATURE_TOLERANCE
