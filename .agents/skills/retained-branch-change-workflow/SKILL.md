---
name: retained-branch-change-workflow
description: Guides code changes in this sunnypilot fork onto the correct retained branch and worktree, including ownership selection, setup, testing, commit readiness, and propagation reminders. Use when the user asks to implement or fix product behavior, add retained feature code, choose a branch, create a worktree, prepare a retained branch for rebuild/deploy, or avoid putting long-term changes on custom.
---

# Retained Branch Change Workflow

## Quick Start

Use this skill before editing product code. `custom` is the admin/deploy integration branch, not the source of truth for long-term behavior.

```bash
git branch --show-current
git status --short
git worktree list --porcelain
git worktree add .worktrees/<branch-suffix> feat/<owning-branch>
uv run --extra testing --extra tools python -m pytest <owning test paths>
```

## Guardrails

Do not commit directly to `master`.
Do not put long-term product behavior on `custom`.
Do not create a new retained branch unless the user explicitly asks and `.sync-config`, `AGENTS.md`, and the workflow are updated together.
Do not remove or force-remove user-opened worktrees; only remove clean agent-created blockers when the documented workflow allows it.
Do not revert, overwrite, or stage unrelated user changes in dirty worktrees.
Do not commit or push unless the user asks, or unless the user has requested a workflow that explicitly requires committed retained-branch changes before propagation, rebuild, or deploy.

## Workflow

1. Read `AGENTS.md` ownership sections and identify the retained branch that owns the requested behavior.
2. If ownership is ambiguous, ask one short question before editing.
3. Check the current branch and worktree state; if on `custom`, create or reuse an appropriate retained-branch worktree under `.worktrees/<branch-suffix>`.
4. Before adding a worktree, inspect `git worktree list --porcelain` and do not reuse a branch already checked out elsewhere.
5. In fresh worktrees, initialize/build required submodules and generated Python extensions before treating missing modules as product failures.
6. Make the smallest branch-owned code and test change needed for the request.
7. Run the owning tests with `uv run --extra testing --extra tools python -m pytest <paths>`; include additional integration tests when touching shared planner/controller interfaces.
8. Review `git diff` for only intended files and ownership boundaries.
9. If committing is requested, stage only intended files, inspect status/diff/log, commit with repo-style message, and push the retained branch when needed for rebuild/deploy.
10. If `feat/retained-baseline` changed and the change is intended for deploy, switch to the rebuild/deploy workflow and propagate from `feat/retained-baseline` before rebuilding `custom`.

## Ownership Cues

Use `feat/retained-baseline` only for shared upstream compatibility fixes every domain should inherit.
Use `feat/device-admin` for Tailscale, startup comm health, process wiring, deploy/runtime support, params, and UI.
Use `feat/longitudinal-control` for follow gap, lead transition, engage bootstrap, e2e stop approach, cruise coast, FCW, launch, decision-layer, and longitudinal stack selector changes.
Use `feat/speed-map-control` for SCC vision/map, OSM/mapd, speed-limit resolver, speed-limit assist, and speed-limit auto-cruise changes.
Use `feat/lateral-control` for controller-side torque behavior, torque selector wiring, lane-change path shaping, model path processing, steering actuator feedback, and lateral controller tuning.
Use `feat/control-learning-stats` for speed-aware torque learning, mass/drag learning, response learning, params/UI/metadata, roll, lateral acceleration, curvature-stat correctness, learned-cache invalidation, and accurate lateral-accel control paths.
Use `feat/offline-drive-analysis` for offline analysis, route explanation, fuzzing, replay/test generation tooling, scenario tools, and route-to-regression helpers.

## Fresh Worktree Setup

Run this before Python tests in new or recently rebased worktrees when generated modules are missing:

```bash
git submodule update --init --recursive
uv run --extra testing --extra tools scons -j$(nproc) common/params_pyx.so msgq_repo/msgq/ipc_pyx.so cereal selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so
```

Then run tests with:

```bash
uv run --extra testing --extra tools python -m pytest <test paths>
```

## Completion Criteria

A retained-branch change is ready when it lives on the owning branch, has focused tests, leaves unrelated work untouched, passes the relevant verification, and has a clear propagation/rebuild path if it is intended for deployment.
