# ADR: Bidirectional lead-speed alignment

## Status

Accepted, Phase 1 implemented.

## Context

Custom longitudinal should react earlier to a relevant lead in both directions:

- when the lead slows, ego should lift/coast or gently bleed closing speed before hard braking is needed;
- when the lead pulls away, ego should release/accelerate sooner, including stop-and-go launch;
- far, noisy, new, flickery, or alternate-threat leads must not dominate the drive.

The MPC remains the hard lead-follow safety authority. The custom policy may add soft desires or caps, but must not remove physical hazards.

## Decision

Implement lead-speed alignment as a pure policy helper feeding the existing candidate/decision system:

- `sunnypilot/custom/longitudinal/lead_speed_alignment.py` computes a guarded recommendation: ignore, coast, gentle brake, moving pullaway, or standstill launch.
- `policy.py` converts recommendations into `ADVISORY_CAP` or authorized `PROGRESS` candidates.
- `stack.py` wires primary lead confidence/stability and only authorizes alignment when the confidence source matches the lead0 kinematics used by policy.
- `longitudinal_planner.py` only extends the accepted stop-release source for the explicit standstill launch intent.

No changes are made to `longcontrol.py` or `long_mpc.py` in Phase 1.

## Safety invariants

- Fail closed on non-finite data, invalid lead kinematics, driver override, force slow, model stop, shadow lead, or alternate threat.
- Slowdown alignment requires a stable, confident lead and enough excess gap; high required decel returns `ignore` so MPC remains authoritative.
- Pullaway requires a stable lead, progress authorization, real opening, cruise headroom, and a safe gap; positive accel is capped by `lead_speedup_guard()`.
- Standstill launch is separate from general moving pullaway and can be cancelled by the existing planner-level model-stop, driver, force-decel, and MPC-braking guards.

## Validation

Initial gates:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/custom/longitudinal/tests/test_lead_speed_alignment.py sunnypilot/custom/longitudinal/tests/test_policy.py sunnypilot/custom/longitudinal/tests/test_stack.py sunnypilot/custom/longitudinal/tests/test_wiring.py
uv run --extra testing --extra tools python -m pytest sunnypilot/custom/longitudinal/tests
```

Before default-confidence expansion or pre-MPC smoothing, validate with route replay and drive-lab lead-follow profiles for cut-ins, cut-outs, stopped leads, lead flicker, far-lead false braking, and stop-and-go launches.
