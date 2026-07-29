# Spike Conventions

## Stack

Use the repository's Python, `drive_lab`, `uv`, and cached-rlog stack. Add no experiment
dependencies.

## Structure

Each control experiment keeps a self-checking `experiment.py`, full-route `results.json`,
and evidence-focused `README.md` under its numbered spike directory.

## Patterns

- Pin behavior experiments to a clean detached checkout when the shared worktree is
  mid-migration.
- Pair one fast logic self-check with full cached-rlog replay.
- Keep experiments offline; do not add Params, production hooks, or deployments.

## Tools & Libraries

Reuse repository processors, log readers, and route-extraction helpers instead of copying
control logic.
