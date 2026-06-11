import io
import json
import sys

import pytest

sys.path.insert(0, "tools/route_validation")
import scan_route  # type: ignore[import-not-found]  # noqa: E402


SAMPLE_LOG = """
2026-01-01 12:00:00 INFO lateral_demand_stack resolved=custom-2.0 requested=custom-2.0 version=2.0 fallback=False reason=
2026-01-01 12:00:01 INFO lateral_demand_stack resolved=custom-2.0 requested=custom-2.0 version=2.0 fallback=False reason=
2026-01-01 12:00:02 INFO lateral_demand_stack resolved=sunnypilot-current requested=custom-2.0 version= fallback=True reason=unavailable_stack
2026-01-01 12:00:03 INFO lateral_demand_stack resolved=custom-2.0 requested=custom-2.0 version=2.0 fallback=False reason=
"""


def test_scan_route_text_finds_all_lines():
  matches = scan_route.scan_route_text(SAMPLE_LOG)
  assert len(matches) == 4
  assert matches[0]["resolved"] == "custom-2.0"
  assert matches[0]["version"] == "2.0"
  assert matches[0]["fallback"] == "False"
  assert matches[2]["resolved"] == "sunnypilot-current"
  assert matches[2]["fallback"] == "True"
  assert matches[2]["reason"] == "unavailable_stack"


def test_summarize_empty_input():
  summary = scan_route.summarize([])
  assert summary["total_transitions"] == 0
  assert summary["stacks_seen"] == []
  assert summary["versions_seen"] == []
  assert summary["fallback_events"] == 0


def test_summarize_populated_input():
  matches = scan_route.scan_route_text(SAMPLE_LOG)
  summary = scan_route.summarize(matches)
  assert summary["total_transitions"] == 4
  assert summary["stacks_seen"] == ["custom-2.0", "sunnypilot-current"]
  assert set(summary["versions_seen"]) == {"2.0", ""}
  assert summary["fallback_events"] == 1
  assert summary["distinct_fallback_reasons"] == ["unavailable_stack"]


def test_main_with_expected_stack_present(monkeypatch, capsys, tmp_path):
  log = tmp_path / "log.txt"
  log.write_text(SAMPLE_LOG)
  rc = scan_route.main([str(log), "--expected-stack", "custom-2.0", "--max-fallbacks", "10", "--json"])
  out = json.loads(capsys.readouterr().out)
  assert rc == 0
  assert out["expected_stack_seen"] is True
  assert out["fallback_events"] == 1


def test_main_with_expected_stack_missing(monkeypatch, capsys, tmp_path):
  log = tmp_path / "log.txt"
  log.write_text(SAMPLE_LOG)
  rc = scan_route.main([str(log), "--expected-stack", "custom-experimental", "--max-fallbacks", "10"])
  assert rc == 2


def test_main_with_too_many_fallbacks(monkeypatch, tmp_path):
  log = tmp_path / "log.txt"
  log.write_text(SAMPLE_LOG)
  rc = scan_route.main([str(log), "--max-fallbacks", "0"])
  assert rc == 3


def test_main_no_telemetry_lines(tmp_path):
  log = tmp_path / "empty.log"
  log.write_text("")
  rc = scan_route.main([str(log)])
  assert rc == 0
