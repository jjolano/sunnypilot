from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_route_msgs(route: str, qlog: bool = False) -> list[Any]:
  from openpilot.tools.lib.logreader import LogReader, ReadMode

  read_mode = ReadMode.QLOG if qlog else ReadMode.AUTO
  return list(LogReader(route, default_mode=read_mode, sort_by_time=True))


def output_report(
  report: Any,
  *,
  json_output: bool,
  renderer: Callable[[Any], str],
  output_path: str | Path | None = None,
  save: Callable[[Any, str | Path], None] | None = None,
) -> str:
  rendered = json.dumps(report.to_dict(), indent=2) if json_output else renderer(report)
  if output_path is not None:
    if save is not None:
      save(report, output_path)
    else:
      Path(output_path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
  return rendered
