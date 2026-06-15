#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-.deploy-config}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  printf 'ERROR: config file not found: %s\n' "$CONFIG_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

for var in DEPLOY_HOST DEPLOY_PATH; do
  if [[ -z "${!var:-}" ]]; then
    printf 'ERROR: missing %s in %s\n' "$var" "$CONFIG_FILE" >&2
    exit 1
  fi
done

printf '== target ==\n'
printf 'DEPLOY_HOST=%s\nDEPLOY_PATH=%s\n' "$DEPLOY_HOST" "$DEPLOY_PATH"

ssh -o ConnectTimeout=10 "$DEPLOY_HOST" "DEPLOY_PATH=$(printf '%q' "$DEPLOY_PATH") bash -s" <<'REMOTE'
set -euo pipefail

printf '\n== uptime ==\n'
uptime || true

printf '\n== checkout ==\n'
if cd "$DEPLOY_PATH"; then
  pwd
  git log -1 --oneline || true
  git status --short || true
else
  printf 'ERROR: cannot cd to DEPLOY_PATH=%s\n' "$DEPLOY_PATH" >&2
fi

printf '\n== process snapshot ==\n'
python3 - <<'PY'
import subprocess

patterns = (
  'manager.py', 'selfdrive.ui.ui', 'pandad', 'loggerd', 'logmessaged',
  'selfdrived', 'controlsd', 'plannerd', 'modeld', 'radard', 'locationd', 'paramsd',
)
try:
  out = subprocess.check_output(
    ['ps', '-eo', 'pid,ppid,stat,pcpu,pmem,comm,args'],
    text=True,
    errors='replace',
  )
except Exception as exc:
  print('process snapshot failed:', exc)
else:
  for line in out.splitlines():
    if any(pattern in line for pattern in patterns):
      print(line)
PY

printf '\n== ipc/log paths ==\n'
python3 - <<'PY'
import glob
import os
import stat
import time

paths = ['/tmp/logmessage', '/data/log'] + sorted(glob.glob('/tmp/msgq_*'))[:40]
for path in paths:
  try:
    st = os.stat(path)
  except FileNotFoundError:
    print(path, 'missing')
    continue
  mode = stat.filemode(st.st_mode)
  mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))
  print(mode, st.st_size, mtime, path)
PY

printf '\n== recent selfdrived communication swaglogs ==\n'
python3 - <<'PY'
import glob
import json
import os
import time

files = sorted(glob.glob('/data/log/swaglog.*'))[-400:]
print('files_scanned', len(files), files[0] if files else None, files[-1] if files else None)

# Static entries from selfdrive/selfdrived/selfdrived.py's ignore list. GPS service can vary,
# so include known GPS service names too. The collector still prints ignored entries, but separates
# them from likely causal services.
ignored_services = {
  'accelerometer', 'gyroscope',
  'gpsLocation', 'gpsLocationExternal', 'qcomGnss', 'ubloxGnss',
  'alertDebug', 'lateralManeuverPlan', 'modelDataV2SP',
}

def split_ignored(items):
  causal = [item for item in items if item not in ignored_services]
  ignored = [item for item in items if item in ignored_services]
  return causal, ignored

matches = 0
for fn in files:
  try:
    fh = open(fn, errors='replace')
  except FileNotFoundError:
    continue
  with fh:
    for lineno, line in enumerate(fh, 1):
      if ('commIssue' not in line and 'process_not_running' not in line and
          'commIssueAvgFreq' not in line and 'selfdrived.initialized' not in line):
        continue
      matches += 1
      try:
        obj = json.loads(line)
      except Exception as exc:
        print(f'{os.path.basename(fn)}:{lineno} parse_error={exc} raw={line[:500]!r}')
        continue

      msg = obj.get('msg') if isinstance(obj.get('msg'), dict) else {}
      created = obj.get('created')
      ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created)) if isinstance(created, (int, float)) else ''
      invalid = msg.get('invalid$a') or msg.get('invalid') or []
      not_alive = msg.get('not_alive$a') or msg.get('not_alive') or []
      not_freq_ok = msg.get('not_freq_ok$a') or msg.get('not_freq_ok') or []
      not_running = msg.get('not_running$a') or msg.get('not_running') or []
      causal_invalid, ignored_invalid = split_ignored(invalid)
      causal_not_alive, ignored_not_alive = split_ignored(not_alive)
      causal_not_freq_ok, ignored_not_freq_ok = split_ignored(not_freq_ok)
      print(
        f'{os.path.basename(fn)}:{lineno}',
        ts,
        'level=' + str(obj.get('level')),
        'daemon=' + str(obj.get('ctx', {}).get('daemon')),
        'pid=' + str(obj.get('process')),
        'event=' + str(msg.get('event$s') or msg.get('event')),
        'causal_invalid=' + repr(causal_invalid),
        'ignored_invalid=' + repr(ignored_invalid),
        'causal_not_alive=' + repr(causal_not_alive),
        'ignored_not_alive=' + repr(ignored_not_alive),
        'causal_not_freq_ok=' + repr(causal_not_freq_ok),
        'ignored_not_freq_ok=' + repr(ignored_not_freq_ok),
        'not_running=' + repr(not_running),
        'timeout=' + str(msg.get('timeout$b') or msg.get('timeout')),
        'canValid=' + str(msg.get('canValid$b') or msg.get('canValid')),
      )
print('matches', matches)
PY
REMOTE
