"""Route validation for the controls profile system.

Given a route log, this tool:
  - extracts the cloudlog lines containing
    'lateral_demand_stack resolved=' to find per-frame stack identity
  - groups them by (resolved_stack, version, fallback_active, reason)
  - prints a summary of the stacks that were active during the route
  - exits with a non-zero status if the route crossed a fallback event

This is the offline counterpart to the on-device cloudlog telemetry:
- on-device: cloudlog.info("lateral_demand_stack resolved=...")
- offline: tools/route_validation/scan_route.py ROUTE --expected-stack NAME

Usage:
  tools/route_validation/scan_route.py ROUTE [--expected-stack custom-2.0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


LINE_RE = re.compile(
  r"lateral_demand_stack resolved=(?P<resolved>[^ ]+)"
  r" requested=(?P<requested>[^ ]+)"
  r" version=(?P<version>[^ ]*)"
  r" fallback=(?P<fallback>[^ ]+)"
  r" reason=(?P<reason>.*)$"
)


def scan_route_text(text: str) -> list[dict[str, str]]:
  matches: list[dict[str, str]] = []
  for line in text.splitlines():
    m = LINE_RE.search(line)
    if m is None:
      continue
    matches.append({
      "resolved": m.group("resolved"),
      "requested": m.group("requested"),
      "version": m.group("version"),
      "fallback": m.group("fallback"),
      "reason": m.group("reason"),
    })
  return matches


def summarize(matches: list[dict[str, str]]) -> dict[str, object]:
  if not matches:
    return {
      "total_transitions": 0,
      "stacks_seen": [],
      "versions_seen": [],
      "fallback_events": 0,
      "distinct_fallback_reasons": [],
    }
  by_key = Counter()
  fallback_reasons: Counter[str] = Counter()
  for m in matches:
    by_key[(m["resolved"], m["version"], m["fallback"], m["reason"])] += 1
    if m["fallback"].lower() == "true":
      fallback_reasons[m["reason"]] += 1
  return {
    "total_transitions": len(matches),
    "stacks_seen": sorted({m["resolved"] for m in matches}),
    "versions_seen": sorted({m["version"] for m in matches}),
    "fallback_events": sum(fallback_reasons.values()),
    "distinct_fallback_reasons": sorted(fallback_reasons.keys()),
    "transitions": [
      {"resolved": k[0], "version": k[1], "fallback": k[2], "reason": k[3], "count": v}
      for k, v in sorted(by_key.items(), key=lambda item: -item[1])
    ],
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Validate a route against the controls profile telemetry contract.")
  parser.add_argument("route", help="Path to a route log file (or '-' for stdin).")
  parser.add_argument("--expected-stack", default="", help="Expected primary resolved stack name. Non-empty triggers a check.")
  parser.add_argument("--max-fallbacks", type=int, default=0, help="Maximum allowed fallback events (default 0).")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
  args = parser.parse_args(argv)

  if args.route == "-":
    text = sys.stdin.read()
  else:
    text = Path(args.route).read_text(encoding="utf-8", errors="replace")

  matches = scan_route_text(text)
  report = summarize(matches)

  if args.expected_stack:
    report["expected_stack"] = args.expected_stack
    report["expected_stack_seen"] = args.expected_stack in (report.get("stacks_seen") or [])

  if args.json:
    print(json.dumps(report, indent=2))
  else:
    print(f"Total transitions: {report['total_transitions']}")
    print(f"Stacks seen: {', '.join(report['stacks_seen']) or '(none)'}")
    print(f"Versions seen: {', '.join(report['versions_seen']) or '(none)'}")
    print(f"Fallback events: {report['fallback_events']}")
    if report['distinct_fallback_reasons']:
      print(f"Fallback reasons: {', '.join(report['distinct_fallback_reasons'])}")
    if args.expected_stack:
      print(f"Expected stack: {args.expected_stack}  -> seen: {report.get('expected_stack_seen')}")

  if args.expected_stack and not report.get("expected_stack_seen"):
    return 2
  if report["fallback_events"] > args.max_fallbacks:
    return 3
  return 0


if __name__ == "__main__":
  sys.exit(main())
