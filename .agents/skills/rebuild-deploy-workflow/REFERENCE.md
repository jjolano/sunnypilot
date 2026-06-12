# Rebuild Deploy Reference

## Propagation Decision Rules

Use propagation as rebuild preflight whenever `feat/retained-baseline` has new commits that domain branches should inherit. Domain branches are independent siblings; do not propagate from one domain branch into another.

Decision table:

| Situation | Action |
|---|---|
| Changed branch is `feat/retained-baseline` | Run `scripts/propagate-retained.sh --from feat/retained-baseline`. |
| Changed branch is a domain branch only | Rebuild can proceed without propagation. |
| Baseline and domain branches changed | Propagate from `feat/retained-baseline`, then rebuild after all intended domain commits are present. |
| User asks for local-only test rebuild | State that propagation is intentionally skipped, then rebuild locally only. |
| Unsure whether domains contain the baseline change | Propagate; do not use `custom` rebuild as the compatibility test. |

Before propagation:

```bash
git worktree list --porcelain
```

If a target domain branch is checked out in an agent-created clean worktree, remove that worktree before propagation. If it is dirty, user-opened, or ambiguous, stop and ask.

## Propagation Conflicts

Conflict policy:

- Resolve conflicts on the retained branch where the conflict occurs, not on `custom`.
- Prefer the target domain branch's ownership rules when resolving compatibility conflicts.
- Commit and push the resolved domain branch.
- Rerun propagation from `feat/retained-baseline`; already-resolved domains should be skipped.

Typical recovery:

```bash
scripts/propagate-retained.sh --from feat/retained-baseline
# conflict occurs in a domain branch and script prints preserved worktree path
uv run --extra testing --extra tools python -m pytest <domain owning tests>
git status --short
git add <resolved files>
git commit -m "scope: resolve baseline propagation"
git push origin <domain-branch>
scripts/propagate-retained.sh --from feat/retained-baseline
```

Do not commit conflict-only product fixes to `custom`; they will be lost on the next rebuild.

## Custom Rebuild Failures

If `scripts/rebuild-custom.sh` fails while merging a retained branch, treat it as a retained-branch compatibility issue.

Checklist:

- Capture the failing branch and conflict files from script output.
- Abort fixing in `custom` unless the conflict is strictly in a custom-only workflow file.
- Move the fix to the domain branch that owns the long-term code, or to `feat/retained-baseline` only when every domain should inherit it.
- Push the retained branch.
- Propagate from `feat/retained-baseline` if that branch changed.
- Rerun `scripts/rebuild-custom.sh` from a clean `custom` worktree.

## Deploy Health Checks

After deploy, wait for SSH and then check the deployed commit:

```bash
ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "uptime"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git log -1 --oneline"
```

Check manager and process state:

```bash
ssh "$DEPLOY_HOST" "pgrep -af manager"
ssh "$DEPLOY_HOST" "pgrep -af 'selfdrive.ui.ui|pandad|loggerd|modeld|controlsd|selfdrived|locationd|paramsd|radard|manage_tailscaled'"
```

If the device is offroad, `controlsd`, `modeld`, `radard`, and other onroad-only processes may be absent. Confirm onroad state before treating that as a failure:

```bash
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && PYTHONPATH='$DEPLOY_PATH' /usr/local/venv/bin/python3 -c 'from openpilot.common.params import Params; print(Params().get_bool(\"IsOnroad\"))'"
```

Check recent errors and retained imports:

```bash
ssh "$DEPLOY_HOST" "journalctl --since '5 min ago' 2>/dev/null | rg -i 'traceback|ImportError|ModuleNotFoundError|exception|crash' || true"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && PYTHONPATH='$DEPLOY_PATH' /usr/local/venv/bin/python3 -c 'from openpilot.sunnypilot.system.tailscale.manage_tailscaled import TailscaleDaemon; from openpilot.system.ui.sunnypilot.widgets.tailscale_pairing_dialog import TailscalePairingDialog; print(\"tailscale-import-ok\")'"
```

Healthy offroad result:

- Deployed commit matches local `custom` HEAD.
- Manager, UI, pandad, loggerd helpers, and `manage_tailscaled` are present.
- No recent crash/import log matches.
- Retained-feature import sanity check passes.

Healthy onroad result additionally expects core control processes such as `controlsd`, `modeld`, `selfdrived`, `locationd`, `paramsd`, and `radard`.

## Partial Deploy Recovery

If failure happens before `git push --force-with-lease`, do not retry blindly. Report the failing command and inspect local `custom` state.

If push succeeds but device update fails:

```bash
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git status --short && git log -1 --oneline"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git fetch jjolano custom"
ssh "$DEPLOY_HOST" "cd '$DEPLOY_PATH' && git rev-parse HEAD && git rev-parse jjolano/custom"
```

If the device HEAD differs from `jjolano/custom` and the worktree is clean, rerun `scripts/deploy.sh` rather than issuing ad hoc reset commands. If the device worktree is dirty or submodule update failed, report the exact state before making changes.

If reboot was requested but SSH does not return, keep retrying `uptime` for a bounded period and report timeout duration. Do not assume the branch is bad without checking whether the device is reachable through the expected network path.

## Rollback Guidance

Rollback is a deployment action and requires explicit user approval.

Safe rollback checklist:

- Identify the known-good `custom` commit from local reflog, remote history, or user-provided SHA.
- Verify the commit is in this repository and is a `custom` integration commit.
- Confirm no retained branch source-of-truth changes need reverting unless the user explicitly requests that.
- Use the standard deploy script after preparing `custom`; do not manually patch the device into an untracked state.
- Run the same health checks after rollback.

If the user asks for rollback but does not provide a target commit, ask for confirmation before selecting one.
