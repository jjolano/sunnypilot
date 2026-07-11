from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get


LOG_FILE_NAMES = frozenset(("qlog.zst", "rlog.zst", "qlog.bz2", "rlog.bz2"))
LOG_FILE_SUFFIXES = (".qlog.zst", ".rlog.zst", ".qlog.bz2", ".rlog.bz2")
LATERAL_DEMAND_SCHEMA_LEGACY = "legacy"
LATERAL_DEMAND_SCHEMA_SPLIT = "split"
_LATERAL_DEMAND_SPLIT_COMMIT = "c63e4b14d33b37b0dd01797b699b3b172dbd9c0f"


@dataclass(frozen=True)
class RouteMessage:
  raw: Any
  typ: str
  payload: Any
  t: float
  log_mono_time: int


@cache
def _commit_has_split_lateral_demand(commit: str) -> bool:
  if not commit:
    return False
  try:
    result = subprocess.run(
      ["git", "merge-base", "--is-ancestor", _LATERAL_DEMAND_SPLIT_COMMIT, commit],
      cwd=BASEDIR,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )
  except OSError:
    return False
  return result.returncode == 0


def lateral_demand_schema(messages: Iterable[Any]) -> str:
  """Classify the pre/post-c63 telemetry boundary from a route's source commit.

  Descendant checks require full local Git history. Missing or unknown commits fall back
  to legacy semantics so old logs never read a default-zero conditioned field.
  """
  for msg in messages:
    if msg_type(msg) == "initData":
      init_data = msg_payload(msg)
      for field in ("gitCommit", "gitSrcCommit"):
        commit = str(safe_get(init_data, field, "")).strip()
        if _commit_has_split_lateral_demand(commit):
          return LATERAL_DEMAND_SCHEMA_SPLIT
  return LATERAL_DEMAND_SCHEMA_LEGACY


def conditioned_desired_curvature(model_path_state: Any, schema: str) -> Any:
  field = "conditionedDesiredCurvature" if schema == LATERAL_DEMAND_SCHEMA_SPLIT else "processedDesiredCurvature"
  return safe_get(model_path_state, field)


def route_identity(route: str) -> tuple[str, int | None]:
  path = Path(str(route))
  name = path.name
  if name in LOG_FILE_NAMES:
    if path.parent.name.isdigit() and "--" in path.parent.parent.name:
      name = f"{path.parent.parent.name}--{path.parent.name}"
    else:
      name = path.parent.name
  for suffix in LOG_FILE_SUFFIXES:
    if name.endswith(suffix):
      name = name[:-len(suffix)]
      break
  if "--" not in name:
    return str(route), None
  prefix, segment_text = name.rsplit("--", 1)
  try:
    return prefix, int(segment_text)
  except ValueError:
    return name, None


def _route_messages_from_iterable(messages: Iterable[Any]) -> Iterator[RouteMessage]:
  base_mono_time: int | None = None
  for msg in messages:
    log_mono_time = int(getattr(msg, "logMonoTime", 0))
    if base_mono_time is None:
      base_mono_time = log_mono_time
    yield RouteMessage(
      raw=msg,
      typ=msg_type(msg),
      payload=msg_payload(msg),
      t=msg_time_s(msg, base_mono_time),
      log_mono_time=log_mono_time,
    )


def build_route_messages(messages: Iterable[Any]) -> list[RouteMessage]:
  return list(_route_messages_from_iterable(messages))


def iter_route_messages(route: str, read_mode: Any, *, log_reader_factory: Callable[..., Iterable[Any]] | None = None) -> Iterator[RouteMessage]:
  if log_reader_factory is None:
    from openpilot.tools.lib.logreader import LogReader

    log_reader_factory = LogReader
  yield from _route_messages_from_iterable(log_reader_factory(route, default_mode=read_mode, sort_by_time=True))


def route_duration(samples: Iterable[Any]) -> float:
  times = [float(sample.t) for sample in samples]
  return max(times, default=0.0) - min(times, default=0.0)


def finite_or_none(value: Any) -> float | None:
  if isinstance(value, int | float):
    numeric = float(value)
    if isfinite(numeric):
      return numeric
  return None


def finite_list(values: Any) -> list[float]:
  if values is None:
    return []
  try:
    iterable = list(values)
  except TypeError:
    finite_value = finite_or_none(values)
    return [] if finite_value is None else [finite_value]
  return [finite_value for value in iterable if (finite_value := finite_or_none(value)) is not None]


def min_optional(values: Iterable[float | None]) -> float | None:
  finite_values = [value for value in values if value is not None and isfinite(value)]
  return min(finite_values) if finite_values else None


def correlation(xs: list[float], ys: list[float]) -> float | None:
  if len(xs) < 2 or len(ys) < 2:
    return None
  x = np.asarray(xs, dtype=float)
  y = np.asarray(ys, dtype=float)
  if float(np.std(x)) <= 1e-9 or float(np.std(y)) <= 1e-9:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def mean(values: list[float]) -> float:
  return float(np.mean(values)) if values else 0.0


def optional_mean(values: list[float]) -> float | None:
  return float(np.mean(values)) if values else None


def percentile(values: list[float], percentile_value: float) -> float:
  return float(np.percentile(values, percentile_value)) if values else 0.0


def ratio(count: int, total: int) -> float:
  return count / total if total else 0.0


def format_counts(counts: dict[str, int]) -> str:
  if not counts:
    return "none"
  return ", ".join(f"{key}={value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def format_optional(value: float | None, *, precision: int = 3, suffix: str = "") -> str:
  return "n/a" if value is None else f"{value:.{precision}f}{suffix}"
