from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CurveSpeedConfidenceInputs:
  """SCC curve evidence carried into the stack.

  The curve-speed-confidence *predictor* that once consumed this was deleted on 2026-07-24
  after the shadow harvest measured it eligible on 0.07% of 101,741 engaged frames (see
  docs/research/natural-feel-gap-analysis.md). This input carrier survives because the curve
  traffic advisor reads the same SCC vision evidence from it.
  """

  vision_active: bool = False
  vision_a_target: float = 0.0
  vision_state: Any = None
  vision_current_lat_acc: float = 0.0
  vision_max_pred_lat_acc: float = 0.0
  vision_pre_entry_active: bool = False
  map_active: bool = False
  map_a_target: float = 0.0
  map_state: Any = None
  map_target_lat: float = 0.0
  map_target_lon: float = 0.0
