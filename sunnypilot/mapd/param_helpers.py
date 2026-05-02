import json
from collections.abc import Iterable, Mapping
from typing import Any


def parse_mapd_json(value: Any) -> Any | None:
  if value is None or value == "":
    return None

  if isinstance(value, bytes):
    value = value.decode("utf-8")

  if isinstance(value, str):
    try:
      return json.loads(value)
    except json.JSONDecodeError:
      return None

  if isinstance(value, (dict, list, int, float)):
    return value

  return None


def get_mapd_json(params, key: str) -> Any | None:
  try:
    return parse_mapd_json(params.get(key))
  except Exception:
    return None


def get_first_mapd_json(params, keys: Iterable[str]) -> Any | None:
  for key in keys:
    value = get_mapd_json(params, key)
    if value not in (None, {}, []):
      return value
  return None


def mapd_section_float(section: Mapping[str, Any], key: str, default: float | None = 0.0) -> float | None:
  value = section.get(key)
  if value is None or value == "":
    return default

  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def mapd_section_int(section: Mapping[str, Any], key: str, default: int = 0) -> int:
  value = section.get(key)
  if value is None or value == "":
    return default

  try:
    return int(value)
  except (TypeError, ValueError):
    return default
