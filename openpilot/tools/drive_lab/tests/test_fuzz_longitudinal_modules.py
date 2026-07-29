import contextlib
import io
import json
import sys

import pytest

from openpilot.tools.drive_lab import fuzz_longitudinal_modules
from openpilot.tools.drive_lab.fuzz_longitudinal_modules import (
  DEFAULT_KINDS,
  ModuleCase,
  ModuleFuzzerConfig,
  ModuleResult,
  evaluate_case,
  generate_cases,
  main,
)


# ---------- determinism ----------


def test_generate_cases_is_seeded():
  assert generate_cases(ModuleFuzzerConfig(seed=42, cases=20)) == generate_cases(ModuleFuzzerConfig(seed=42, cases=20))


def test_evaluate_case_is_deterministic():
  config = ModuleFuzzerConfig(seed=7, cases=1, kind="mode_matrix")
  case = generate_cases(config)[0]
  first = evaluate_case(case)
  second = evaluate_case(case)
  assert first.valid == second.valid
  assert [f["check"] for f in first.failures] == [f["check"] for f in second.failures]


def test_case_round_trips_through_dict():
  config = ModuleFuzzerConfig(seed=3, cases=1)
  case = generate_cases(config)[0]
  restored = ModuleCase.from_dict(case.to_dict())
  assert restored == case


# ---------- per-kind zero-failure smoke ----------


@pytest.mark.parametrize("kind", DEFAULT_KINDS)
def test_kind_has_zero_failures(kind):
  cases = generate_cases(ModuleFuzzerConfig(seed=1, cases=10, kind=kind))
  failure_count = sum(1 for c in cases if not evaluate_case(c).valid)
  assert failure_count == 0


# ---------- injected failure detection ----------


def test_monkeypatched_mode_function_produces_failures():
  original = fuzz_longitudinal_modules.modes_mod.admitted_evidence
  config = ModuleFuzzerConfig(seed=1, cases=1, kind="mode_matrix")
  case = generate_cases(config)[0]
  try:
    fuzz_longitudinal_modules.modes_mod.admitted_evidence = lambda mode, sources: frozenset()
    result = evaluate_case(case)
    assert not result.valid
    assert result.failure_count > 0
  finally:
    fuzz_longitudinal_modules.modes_mod.admitted_evidence = original


# ---------- CLI smoke ----------


def test_main_text_output_runs_and_reports_zero_failures():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal_modules.py", "--seed", "1", "--cases", "5", "--kind", "mode_matrix"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab longitudinal modules fuzz" in output
  assert "failures=0" in output


def test_main_json_output_is_stable():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal_modules.py", "--seed", "2", "--cases", "3", "--json"]
    with contextlib.redirect_stdout(stdout):
      main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["seed"] == 2
  assert payload["cases"] == 3
  assert "failures" in payload
  assert payload["metrics"]["total"] == 3


def test_main_exits_nonzero_on_injected_failure():
  original_evaluate = fuzz_longitudinal_modules.evaluate_case
  config = ModuleFuzzerConfig(seed=1, cases=1, kind="mode_matrix")
  case = generate_cases(config)[0]
  baseline = evaluate_case(case)
  failed = ModuleResult(
    case=baseline.case,
    failures=baseline.failures + [{"check": "injected", "detail": "injected failure"}],
    metrics=baseline.metrics,
  )

  def _forced_evaluate(c):
    return failed

  fuzz_longitudinal_modules.evaluate_case = _forced_evaluate
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal_modules.py", "--seed", "1", "--cases", "1"]
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 1
  finally:
    fuzz_longitudinal_modules.evaluate_case = original_evaluate
    sys.argv = previous_argv


def test_main_fail_fast_stops_after_first_injected_failure():
  original_evaluate = fuzz_longitudinal_modules.evaluate_case
  call_count = 0

  def _forced_evaluate(case):
    nonlocal call_count
    call_count += 1
    baseline = original_evaluate(case)
    return ModuleResult(
      case=baseline.case,
      failures=baseline.failures + [{"check": "injected", "detail": "injected failure"}],
      metrics=baseline.metrics,
    )

  fuzz_longitudinal_modules.evaluate_case = _forced_evaluate
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal_modules.py", "--seed", "1", "--cases", "50", "--fail-fast"]
    with pytest.raises(SystemExit) as exc:
      with contextlib.redirect_stdout(stdout):
        main()
    assert exc.value.code == 1
  finally:
    fuzz_longitudinal_modules.evaluate_case = original_evaluate
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "failures=1" in output
  assert "cases=1" in output
  assert call_count == 1
