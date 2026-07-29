from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab.route_analysis import (
  build_route_messages,
  correlation,
  finite_list,
  finite_or_none,
  format_counts,
  format_optional,
  iter_route_messages,
  route_duration,
  route_identity,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def test_route_identity_handles_segment_paths_and_log_suffixes():
  assert route_identity("/tmp/00000187--ea39892416/4/rlog.zst") == ("00000187--ea39892416", 4)
  assert route_identity("/tmp/00000188--249e4349c3--2/qlog.zst") == ("00000188--249e4349c3", 2)
  assert route_identity("/tmp/00000188--249e4349c3--2.rlog.zst") == ("00000188--249e4349c3", 2)
  assert route_identity("route-without-segment") == ("route-without-segment", None)


def test_build_route_messages_attaches_type_payload_and_relative_time():
  records = build_route_messages([
    msg("carState", 10.0, vEgo=5.0),
    msg("longitudinalPlan", 10.25, aTarget=-0.5),
  ])

  assert [record.typ for record in records] == ["carState", "longitudinalPlan"]
  assert records[0].t == pytest.approx(0.0)
  assert records[1].t == pytest.approx(0.25)
  assert records[0].payload.vEgo == pytest.approx(5.0)
  assert records[1].payload.aTarget == pytest.approx(-0.5)


def test_iter_route_messages_is_lazy_and_uses_log_reader_factory():
  consumed = []

  def fake_log_reader(route, default_mode, sort_by_time):
    assert route == "route-a"
    assert default_mode == "auto"
    assert sort_by_time is True
    consumed.append("first")
    yield msg("carState", 10.0, vEgo=5.0)
    consumed.append("second")
    yield msg("longitudinalPlan", 10.5, aTarget=-0.2)

  records = iter_route_messages("route-a", "auto", log_reader_factory=fake_log_reader)

  assert consumed == []
  first = next(records)
  assert consumed == ["first"]
  assert first.typ == "carState"
  assert first.t == pytest.approx(0.0)
  second = next(records)
  assert consumed == ["first", "second"]
  assert second.typ == "longitudinalPlan"
  assert second.t == pytest.approx(0.5)


def test_numeric_helpers_keep_drive_lab_empty_input_defaults():
  assert finite_or_none(float("nan")) is None
  assert finite_list([1.0, float("inf"), "bad", 2]) == [1.0, 2.0]
  assert correlation([1.0], [1.0]) is None
  assert route_duration([]) == 0.0
  assert format_counts({"b": 1, "a": 2}) == "a=2, b=1"
  assert format_optional(None) == "n/a"
  assert format_optional(1.23456, precision=2, suffix="s") == "1.23s"
