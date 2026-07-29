from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec


REQUIRED_TAGS = {
  "lateral": ("route-derived", "lateral"),
  "longitudinal": ("route-derived", "longitudinal"),
  "any": ("route-derived",),
  "lateral-synthetic": ("lateral",),
  "longitudinal-synthetic": ("longitudinal",),
}


@dataclass(frozen=True)
class BehaviorReadinessResult:
  domain: str
  ready: bool
  scenario_count: int
  matching_count: int
  required_tags: tuple[str, ...]
  reasons: tuple[str, ...]


def assess_behavior_change_readiness(specs: list[ScenarioSpec], domain: str) -> BehaviorReadinessResult:
  if domain not in REQUIRED_TAGS:
    raise ValueError(f"unsupported domain {domain!r}")
  required_tags = REQUIRED_TAGS[domain]
  scenario_count = len(specs)
  matching = [spec for spec in specs if _matches(spec, required_tags)]
  tagged = [spec for spec in specs if _has_required_tags(spec, required_tags)]
  reasons: list[str] = []
  if scenario_count == 0:
    reasons.append("no_scenarios")
  if scenario_count > 0 and not tagged:
    reasons.append(_missing_reason(required_tags))
  if scenario_count > 0 and any(_has_required_tags(spec, required_tags) and not _has_oracle_checks(spec) for spec in specs):
    reasons.append("missing_oracle_checks")
  if scenario_count > 0 and any(_has_required_tags(spec, required_tags) and not _has_events(spec) for spec in specs):
    reasons.append("missing_events")
  return BehaviorReadinessResult(
    domain=domain,
    ready=len(matching) > 0,
    scenario_count=scenario_count,
    matching_count=len(matching),
    required_tags=required_tags,
    reasons=tuple(reasons),
  )


def load_scenario_specs(path: str | Path) -> list[ScenarioSpec]:
  payload = json.loads(Path(path).read_text())
  if isinstance(payload, list):
    return [ScenarioSpec.from_dict(item) for item in payload]
  if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
    return [ScenarioSpec.from_dict(item) for item in payload["scenarios"]]
  raise ValueError("expected a JSON list or an object with a scenarios list")


def main() -> None:
  parser = argparse.ArgumentParser(description="Assess whether route-derived scenario evidence is ready for behavior tuning.")
  parser.add_argument("scenario_json", help="Path to scenario JSON list or {scenarios:[...]} payload")
  parser.add_argument("--domain", choices=tuple(REQUIRED_TAGS), default="any")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  result = assess_behavior_change_readiness(load_scenario_specs(args.scenario_json), args.domain)
  payload = {
    "domain": result.domain,
    "ready": result.ready,
    "scenario_count": result.scenario_count,
    "matching_count": result.matching_count,
    "required_tags": list(result.required_tags),
    "reasons": list(result.reasons),
  }
  print(json.dumps(payload, indent=2) if args.json else _render_result(result))
  raise SystemExit(0 if result.ready else 1)


def _matches(spec: ScenarioSpec, required_tags: tuple[str, ...]) -> bool:
  return _has_required_tags(spec, required_tags) and _has_oracle_checks(spec) and _has_events(spec)


def _has_required_tags(spec: ScenarioSpec, required_tags: tuple[str, ...]) -> bool:
  tags = set(spec.tags)
  return all(tag in tags for tag in required_tags)


def _has_oracle_checks(spec: ScenarioSpec) -> bool:
  checks = spec.oracle.get("checks", ())
  return bool(tuple(checks))


def _has_events(spec: ScenarioSpec) -> bool:
  return bool(spec.events)


def _missing_reason(required_tags: tuple[str, ...]) -> str:
  if required_tags == REQUIRED_TAGS["lateral"]:
    return "no_route_derived_lateral_specs"
  if required_tags == REQUIRED_TAGS["longitudinal"]:
    return "no_route_derived_longitudinal_specs"
  return "no_route_derived_specs"


def _render_result(result: BehaviorReadinessResult) -> str:
  return (
    f"behavior readiness domain={result.domain} ready={result.ready} "
    f"matching={result.matching_count}/{result.scenario_count} "
    f"required_tags={','.join(result.required_tags)} reasons={','.join(result.reasons) or 'none'}"
  )


if __name__ == "__main__":
  main()
