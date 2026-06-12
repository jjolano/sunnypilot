# Longitudinal candidate authority contract

Planner candidates are algebraic claims, not route-tuning preferences.

- `PHYSICAL_HAZARD` can always restrict. It outranks advisory caps, relaxation/progress floors, and driver intent. It must not be softened by advisory or relaxation candidates.
- `ADVISORY_CAP` can lower target speed and/or accel. It cannot increase accel and cannot authorize launch, creep, gap-fill, pullaway, or other progress.
- `RELAXATION` can raise accel only after physical hazards and advisory caps are absent. Candidate builders must also block it for driver brake/gas, force-slow, active stop threats, and mode boundaries.
- `DRIVER_INTENT` has exactly one effective candidate. One-pedal policy replaces driver intent instead of adding a second driver candidate.
- Scene-derived progress floors require explicit progress authority. `has_lead` by itself is never enough.

Tests in `selfdrive/controls/tests/test_longitudinal_candidate_authority.py` pin these contracts and the rejected-candidate telemetry used by custom-v2 debug output.
