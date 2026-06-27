---
name: device-route-log-analysis
description: Access route logs from the configured deployment target and analyze longitudinal/lateral behavior with drive_lab. Use when the user asks to inspect device route logs, analyze recent drives, compare lateral or longitudinal control from logs, or pull qlogs/rlogs from the comma deployment target.
---

# Device Route Log Analysis

## Quick start

1. Read `.deploy-config` for `DEPLOY_HOST` and `DEPLOY_PATH`.
2. Confirm SSH:
   ```bash
   ssh -o ConnectTimeout=20 "$DEPLOY_HOST" "uptime"
   ```
3. List recent routes on-device:
   ```bash
   ssh "$DEPLOY_HOST" 'python3 - <<'"'"'PY'"'"'
import os, time
root="/data/media/0/realdata"
routes={}
for name in os.listdir(root):
  p=os.path.join(root,name)
  if not os.path.isdir(p): continue
  parts=name.rsplit("--",1)
  if not (len(parts)==2 and parts[1].isdigit()): continue
  base, seg = parts[0], int(parts[1])
  rec=routes.setdefault(base,{"mtime":0,"segs":set(),"q":0,"r":0,"qcount":0,"rcount":0})
  rec["mtime"]=max(rec["mtime"], os.stat(p).st_mtime)
  rec["segs"].add(seg)
  for fn,key,count in (("qlog.zst","q","qcount"),("rlog.zst","r","rcount")):
    fp=os.path.join(p,fn)
    if os.path.exists(fp): rec[key]+=os.path.getsize(fp); rec[count]+=1
for base, rec in sorted(routes.items(), key=lambda kv:kv[1]["mtime"], reverse=True)[:12]:
  nums=sorted(rec["segs"])
  print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(rec['mtime']))} {base} segs={len(nums)} first={nums[0]} last={nums[-1]} qlogs={rec['qcount']} {rec['q']/1e6:.1f}MB rlogs={rec['rcount']} {rec['r']/1e6:.1f}MB")
PY'
   ```

## Pull logs safely

- Choose the lightest log that can answer the question:
  - Use `qlog.zst` for quick route triage: events, timelines, engagement windows, broad lead-following/launch-delay summaries, and “what happened?” checks.
  - Use `rlog.zst` for high-fidelity analysis: exact controller/model/planner timing, replay, rate/validity debugging, signal-frequency checks, or any tuning decision where downsampling could hide behavior.
- Start with qlogs only when the question is summary-level; escalate to rlogs as soon as a needed signal is missing, downsampled, or timing-sensitive.
- Use `/tmp/opencode/sunnypilot-route-logs` for local copies.
- Device SSH over Tailscale can drop. Copy file-by-file with size checks/retries instead of one huge tar stream.

Example one-route qlog copy:

```bash
python3 - <<'PY'
import os, pathlib, subprocess, sys, time
host = "comma@100.94.10.12"  # or source .deploy-config before running
remote_root = "/data/media/0/realdata"
local_root = pathlib.Path("/tmp/opencode/sunnypilot-route-logs")
route = "ROUTE_BASE"  # e.g. 000001d3--d52c54a6fc
remote_py = f'''
import os
root={remote_root!r}; route={route!r}; names=[]
for name in os.listdir(root):
  if not name.startswith(route+'--'): continue
  try: seg=int(name.rsplit('--',1)[1])
  except Exception: continue
  p=os.path.join(root,name,'qlog.zst')
  if os.path.exists(p): names.append((seg,name,p))
for seg,name,p in sorted(names): print(name + '/qlog.zst' + '\\t' + str(os.path.getsize(p)))
'''
res = subprocess.run(["ssh", host, "python3 -"], input=remote_py, text=True, capture_output=True, timeout=60)
res.check_returncode()
for line in res.stdout.splitlines():
  rel, size_s = line.split("\t"); size = int(size_s)
  dest = local_root / rel
  if dest.exists() and dest.stat().st_size == size: continue
  dest.parent.mkdir(parents=True, exist_ok=True)
  src = f"{host}:{remote_root}/{rel}"
  for attempt in range(1, 5):
    try:
      rc = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", src, str(dest)], timeout=60).returncode
    except subprocess.TimeoutExpired:
      rc = 124
    if rc == 0 and dest.exists() and dest.stat().st_size == size: break
    time.sleep(3)
  else:
    raise SystemExit(f"failed: {rel}")
PY
```

## Analyze locally

Use `--qlog` only for qlog-derived summaries. Drop `--qlog` and point tools/harnesses at `rlog.zst` files when validating exact control behavior.

Run local tools with repo deps:

```bash
uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.profile_lead_following ROUTE_OR_FILE --qlog
uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.profile_launch_delays ROUTE_OR_FILE --qlog
uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.profile_route ROUTE_OR_FILE --qlog
```

For device route bases like `000001d3--...`, local route strings are not canonical comma route IDs. Use a Python harness with `LogReader([list_of_qlog_paths], default_mode=ReadMode.QLOG, sort_by_time=True)`.

## Notes

- On-device `python3` may lack `numpy`/`capnp`; do not rely on running drive_lab there.
- Avoid `uv run` on the device unless disk space is confirmed; it can try to build a venv and fill `/home`.
- Do not use qlogs to make fine controller-tuning conclusions unless confirmed against rlogs.
- If all-at-once analysis is killed on very large routes, stream one qlog segment at a time and aggregate carState/radarState/plan signals.
- Record which routes had `carControl.longActive`; long routes with zero engagement are not useful for longitudinal-control behavior.
