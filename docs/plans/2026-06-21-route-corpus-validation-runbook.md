# Route corpus validation runbook

## Purpose

Use this after collecting new on-road routes to decide what, if anything, should change next.
The recent commits added mostly shadow telemetry, stale-signal hardening, manual override bounds,
torque-tune selection gating, and ACC-envelope auditing. They should be validated with real route
evidence before promoting any behavior change.

Primary question: are the new routes showing a repeatable comfort or tracking issue, and can it be
attributed to path-state inputs, demand shaping, controller actuation, or longitudinal policy?

## Inputs

- New engaged routes from the current `master` branch.
- Comparable old-fork or earlier-restart routes when available.
- Route bookmarks or notes describing notable events: lane changes, bends, hills, lead approach,
  stop-and-go, manual torque override, or driver intervention.
- Current local commit SHA and deployed device SHA.

## Intake checklist

For each route, record:

```text
route:
branch/commit:
vehicle:
date/time:
segment(s):
conditions: dry/wet, day/night, traffic, road type
notable events:
driver interventions:
initial hypothesis: lateral / longitudinal / mixed / unknown
```

Keep raw route identifiers unchanged. Use short local filenames only for derived outputs.

Suggested derived-output layout:

```text
/tmp/drive_lab_runs/YYYY-MM-DD/
  routes.yaml
  lateral/<route-slug>/profile.json
  lateral/<route-slug>/replay.json
  longitudinal/<route-slug>/profile.json
  summaries/
```

## Pass 1 — route profiling

Run broad profiling before making any code changes.

Lateral profile:

```bash
uv run python tools/drive_lab/profile_lateral.py ROUTE --output /tmp/drive_lab_runs/YYYY-MM-DD/lateral/ROUTE/profile.json
```

Longitudinal/route event profile, as applicable:

```bash
uv run python -m openpilot.tools.drive_lab.explain_route_event ROUTE --nearest-bookmark
uv run python -m openpilot.tools.drive_lab.profile_route ROUTE --output /tmp/drive_lab_runs/YYYY-MM-DD/longitudinal/ROUTE/profile.json
```

If the route contains lead-following behavior, also run the lead-following profile tooling used by
the recent longitudinal ADR gates.

## Pass 2 — replay gates

Run replay against the current branch first. Do not tune from a single hand-picked event.

Lateral route replay:

```bash
uv run python tools/drive_lab/fuzz_lateral_route_replay.py --route ROUTE --perturbation noise
```

Profile-guided lateral fuzzing:

```bash
uv run python tools/drive_lab/fuzz_lateral_demand.py --preset fuzz --profile /tmp/drive_lab_runs/YYYY-MM-DD/lateral/ROUTE/profile.json --cases 100
```

Longitudinal fuzz presets for relevant route classes:

```bash
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --preset openpilot-acc --cases 100
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal --preset ncap-acc --cases 100
uv run python -m openpilot.tools.drive_lab.fuzz_longitudinal_route_replay --route ROUTE
```

## Pass 3 — classify failures

Classify each issue before proposing a fix.

### Lateral classes

- **Path-state input problem:** stale model, low sensor confidence, bad curvature/lane-line input,
  route conditions outside expected model confidence.
- **Demand-shaping problem:** desired demand is too weak/late/oscillatory before it reaches the
  torque controller.
- **Governor/shaper problem:** controller has enough evidence but output shaping suppresses needed
  same-direction correction or underresponse recovery.
- **Actuation/safety problem:** torque demand would exceed safe bounds or manual override behavior
  shows driver disagreement.
- **Attribution unknown:** needs more routes or instrumentation; do not tune yet.

### Longitudinal classes

- **Lead speed/alignment problem:** lead evidence or speed alignment causes late or early response.
- **ACC envelope problem:** shadow audit shows desired behavior outside accepted envelope.
- **Comfort policy problem:** response is safe but rough, late, or inefficient.
- **Route/source problem:** map/model/lead signal is missing, stale, or not trustworthy.

## Decision rules

- Prefer no behavior change when evidence is isolated, route-specific, or instrumentation is unclear.
- Promote behavior only when the same failure class repeats across comparable routes.
- Preserve lateral tracking accuracy over comfort smoothing; avoid live adaptation that can create
  wrong-direction or runaway torque corrections.
- If a lateral fix is needed, prefer the narrowest guarded point:
  1. demand-shaping adjustment if requested demand is wrong,
  2. governor/shaper adjustment if requested demand is right but suppressed,
  3. safety/override adjustment only if driver-disagreement evidence supports it.
- If a longitudinal fix is needed, keep it behind the relevant replay gate and compare against the
  documented ACC/lead-following envelopes.
- Write a new ADR before adding a new learned state, changing default-on behavior, or introducing a
  second implementation path.

## Suggested next implementation candidates

Only consider these after the route evidence is classified:

1. **Guarded lateral underresponse recovery** — if routes show repeatable same-direction tracking
   deficit where path-state confidence is good and driver overrides do not object.
2. **Sensor-confidence promotion gate** — if shadow confidence clearly separates good and bad route
   sections without false positives.
3. **ACC envelope follow-up** — if shadow audit identifies recurring desired-accel outliers tied to
   lead approach or pullaway behavior.
4. **Baseline corpus expansion** — if evidence is too thin; archive profiles and make them part of
   future behavior gates before tuning.

## Output summary template

After the run, write a short summary:

```text
routes analyzed:
commits compared:
lateral findings:
longitudinal findings:
repeated failure classes:
single-route anomalies:
recommended action: no-op / instrument more / implement guarded fix / write ADR
tests or gates to preserve:
```

## Exit criteria

- New routes have profiles and replay outputs archived locally.
- Any proposed behavior change has at least one replay/profile gate that would fail before the fix
  or demonstrate the measured improvement after it.
- No proposed lateral change weakens safety bounds, wrong-direction prevention, manual override
  handling, or stale-model gating.
- No proposed longitudinal change bypasses the ACC envelope or lead-following replay gates.
