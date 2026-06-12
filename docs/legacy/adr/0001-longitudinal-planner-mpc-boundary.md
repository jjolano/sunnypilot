# Longitudinal Planner and MPC Boundary

Accepted: Longitudinal Planner owns scene assembly, stateful stop/go intent, and candidate arbitration; Longitudinal MPC owns physical cruise and lead-follow trajectory feasibility, including closing-rate risk, danger gaps, time-gap/runway behavior, and lead braking. Custom stacks may choose among valid policy envelopes and cap non-lead speed-up seeds, but they must not become a parallel lead-physics model or change `sunnypilot-current` behavior; this keeps safety-critical lead physics centralized while still allowing custom-2.0 to express progress and comfort policy.

Rejected alternatives: putting all closing-rate behavior into custom-2.0 would duplicate lead physics outside MPC, while putting every planner seed guard into MPC would miss cases where cruise or speed-limit targets accelerate before lead MPC is selected.
