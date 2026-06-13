"""controlsd wiring for the lateral demand pipeline (opt-in).

``LateralDemandAdapter`` is held by the controller loop; when ``CustomLateralDemandEnabled``
is set it processes the raw model curvature through the demand pipeline before clipping.
Default off => the stock model curvature is used, so this can never change default steering.

Evidence mapped from verified ``modelV2`` fields (position x/y/yStd, orientation z + rate,
laneLineProbs). CONSERVATIVELY DEFAULTED pending wiring (so they stay inert until verified):
the lane-change state/direction and the lane-line y0 offsets that the lane-change shaper
needs — until those are wired the pipeline runs model-path processing (+ optional lane
centering) only. See docs/touch-points.md.
"""
from __future__ import annotations

from typing import Any

from openpilot.sunnypilot.custom.lateral.demand.pipeline import (
  LateralDemandPipeline,
  LateralDemandPipelineInputs,
)

PARAMS_REFRESH_PERIOD = 100  # control ticks (100Hz -> ~1s)


def build_pipeline_inputs(*, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
                          measured_curvature: float, model_v2: Any,
                          lane_centering_assist_enabled: bool) -> LateralDemandPipelineInputs:
  pos = getattr(model_v2, "position", None)
  ori = getattr(model_v2, "orientation", None)
  ori_rate = getattr(model_v2, "orientationRate", None)
  return LateralDemandPipelineInputs(
    lat_active=lat_active, v_ego=v_ego, roll=roll,
    desired_curvature=raw_curvature, measured_curvature=measured_curvature,
    position_x=tuple(getattr(pos, "x", ()) or ()),
    position_y=tuple(getattr(pos, "y", ()) or ()),
    position_y_std=tuple(getattr(pos, "yStd", ()) or ()),
    orientation_z=tuple(getattr(ori, "z", ()) or ()),
    orientation_rate_z=tuple(getattr(ori_rate, "z", ()) or ()),
    lane_line_probs=tuple(getattr(model_v2, "laneLineProbs", ()) or ()),
    frame_drop_perc=float(getattr(model_v2, "frameDropPerc", 0.0) or 0.0),
    # conservative until wired (harness-gated):
    lane_change_state=0, lane_change_direction=0,
    lane_centering_assist_enabled=bool(lane_centering_assist_enabled),
  )


class LateralDemandAdapter:
  def __init__(self, params: Any = None):
    self._params = params
    self._pipeline = LateralDemandPipeline()
    self._tick = 0
    self.enabled = False
    self.lane_centering_assist_enabled = False
    if params is not None:
      self.refresh_params()

  def refresh_params(self) -> None:
    p = self._params
    if p is None:
      return
    try:
      self.enabled = bool(p.get_bool("CustomLateralDemandEnabled"))
      self.lane_centering_assist_enabled = bool(p.get_bool("LaneCenteringAssistEnabled"))
    except Exception:
      self.enabled = False

  def process(self, lat_active: bool, v_ego: float, roll: float, raw_curvature: float,
              measured_curvature: float, model_v2: Any) -> float:
    """Return the processed desired curvature, or the unchanged raw curvature when disabled
    or on any fault (fail-closed)."""
    self._tick += 1
    if self._params is not None and self._tick % PARAMS_REFRESH_PERIOD == 0:
      self.refresh_params()
    if not self.enabled or model_v2 is None:
      return raw_curvature
    try:
      inputs = build_pipeline_inputs(
        lat_active=lat_active, v_ego=v_ego, roll=roll, raw_curvature=raw_curvature,
        measured_curvature=measured_curvature, model_v2=model_v2,
        lane_centering_assist_enabled=self.lane_centering_assist_enabled,
      )
      return float(self._pipeline.update(inputs).demand.processed_curvature)
    except Exception:
      return raw_curvature
