"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Small shared helpers for schema-driven action factories.
"""
from __future__ import annotations

import datetime
from collections.abc import Callable

from openpilot.common.time_helpers import system_time_valid
from openpilot.system.ui.lib.multilang import tr, trn


def deferred_tr(text: str) -> Callable[[], str]:
  """Defer translation to render time (titles/descriptions are callables)."""
  return lambda: tr(text)


def time_ago(date_str: str | None) -> str:
  """Return a human-readable relative time string from an ISO datetime string."""
  if not date_str:
    return tr("never")
  try:
    date = datetime.datetime.fromisoformat(date_str)
  except (ValueError, TypeError):
    return tr("never")

  if not system_time_valid():
    return date.strftime("%a %b %d %Y")

  now = datetime.datetime.now(datetime.UTC)
  if date.tzinfo is None:
    date = date.replace(tzinfo=datetime.UTC)

  diff_seconds = int((now - date).total_seconds())
  if diff_seconds < 60:
    return tr("now")
  if diff_seconds < 3600:
    m = diff_seconds // 60
    return trn("{} minute ago", "{} minutes ago", m).format(m)
  if diff_seconds < 86400:
    h = diff_seconds // 3600
    return trn("{} hour ago", "{} hours ago", h).format(h)
  if diff_seconds < 604800:
    d = diff_seconds // 86400
    return trn("{} day ago", "{} days ago", d).format(d)
  return date.strftime("%a %b %d %Y")
