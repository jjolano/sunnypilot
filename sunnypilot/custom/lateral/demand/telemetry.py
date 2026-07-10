"""Fork-owned serialization of custom lateral telemetry into ControlsState.modelPathState.

Moved out of ``controlsd`` so the upstream file keeps hook-sized calls. Telemetry semantics
(CONTEXT.md):
  - ``rawDesiredCurvature``          — raw model/maneuver desired curvature (pre-pipeline input)
  - ``conditionedDesiredCurvature``  — Conditioned Lateral Demand: pipeline result before hard caps
  - ``processedDesiredCurvature``    — Processed Lateral Demand: post-cap controller-facing curvature
"""
from __future__ import annotations

import math

import numpy as np

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDXS = ModelConstants.T_IDXS[:CONTROL_N]


def set_model_path_state_geometry(model_path_state, debug: dict | None = None) -> None:
  debug = debug or {}
  model_path_state.geometryMode = bool(debug.get('lane_centering_geometry_mode', False))
  model_path_state.geometryValid = bool(debug.get('lane_centering_geometry_valid', False))
  model_path_state.geometryReason = str(debug.get('lane_centering_geometry_reason', 'disabled'))
  model_path_state.geometryConfidence = float(debug.get('lane_centering_geometry_confidence', 0.0))
  model_path_state.geometryOffsetNear = float(debug.get('lane_centering_geometry_offset_near', 0.0))
  model_path_state.geometryOffsetPreview = float(debug.get('lane_centering_geometry_offset_preview', 0.0))
  model_path_state.geometryWidthNear = float(debug.get('lane_centering_geometry_width_near', 0.0))
  model_path_state.geometryWidthPreview = float(debug.get('lane_centering_geometry_width_preview', 0.0))
  model_path_state.laneCenteringGeometryHoldActive = bool(debug.get('lane_centering_geometry_hold_active', False))


def set_model_path_state_sps(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.spsMode = str(debug.get('straight_path_stabilization_mode', 'off'))
  model_path_state.spsActive = bool(debug.get('straight_path_stabilization_active', False))
  model_path_state.spsApplied = bool(debug.get('straight_path_stabilization_applied', False))
  model_path_state.spsCandidateCurvature = float(debug.get('straight_path_stabilization_candidate_curvature', 0.0))
  model_path_state.spsAnchorLatAccel = float(debug.get('straight_path_stabilization_anchor_lat_accel', 0.0))
  model_path_state.spsReason = str(debug.get('straight_path_stabilization_reason', default_reason))


def set_model_path_state_one_line(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.laneCenteringOneLineMode = str(debug.get('lane_centering_one_line_mode', 'off'))
  model_path_state.laneCenteringOneLineActive = bool(debug.get('lane_centering_one_line_active', False))
  model_path_state.laneCenteringOneLineApplied = bool(debug.get('lane_centering_one_line_applied', False))
  model_path_state.laneCenteringOneLineReason = str(debug.get('lane_centering_one_line_reason', default_reason))
  model_path_state.laneCenteringOneLineLateralError = float(debug.get('lane_centering_one_line_lateral_error', 0.0))
  model_path_state.laneCenteringOneLinePredictedError = float(debug.get('lane_centering_one_line_predicted_error', 0.0))
  model_path_state.laneCenteringOneLineCandidateNudge = float(debug.get('lane_centering_one_line_candidate_nudge', 0.0))
  model_path_state.laneCenteringOneLineLearnedWidth = float(debug.get('lane_centering_one_line_learned_width', 0.0))
  model_path_state.laneCenteringOneLineConfidence = float(debug.get('lane_centering_one_line_confidence', 0.0))


def set_model_path_state_preview(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.previewAssistMode = str(debug.get('lateral_preview_assist_mode', 'off'))
  model_path_state.previewAssistActive = bool(debug.get('lateral_preview_assist_active', False))
  model_path_state.previewAssistApplied = bool(debug.get('lateral_preview_assist_applied', False))
  model_path_state.previewAssistReason = str(debug.get('lateral_preview_assist_reason', default_reason))
  model_path_state.previewAssistConfidence = float(debug.get('lateral_preview_assist_confidence', 0.0))
  model_path_state.previewAssistTPreview = float(debug.get('lateral_preview_assist_t_preview', 0.0))
  model_path_state.previewAssistBaseCurvature = float(debug.get('lateral_preview_assist_base_curvature', 0.0))
  model_path_state.previewAssistPreviewCurvature = float(debug.get('lateral_preview_assist_preview_curvature', 0.0))
  model_path_state.previewAssistCurvatureNudge = float(debug.get('lateral_preview_assist_curvature_nudge', 0.0))
  model_path_state.previewAssistAyBase = float(debug.get('lateral_preview_assist_ay_base', 0.0))
  model_path_state.previewAssistAyPreview = float(debug.get('lateral_preview_assist_ay_preview', 0.0))
  model_path_state.previewAssistAyDelta = float(debug.get('lateral_preview_assist_ay_delta', 0.0))
  model_path_state.previewAssistSlewLimited = bool(debug.get('lateral_preview_assist_slew_limited', False))


def set_model_path_state_lane_rate_damping(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.laneRateDampingMode = str(debug.get('lane_rate_damping_mode', 'off'))
  model_path_state.laneRateDampingActive = bool(debug.get('lane_rate_damping_active', False))
  model_path_state.laneRateDampingApplied = bool(debug.get('lane_rate_damping_applied', False))
  model_path_state.laneRateDampingReason = str(debug.get('lane_rate_damping_reason', default_reason))
  model_path_state.laneRateDampingLaneCenter = float(debug.get('lane_rate_damping_lane_center', 0.0))
  model_path_state.laneRateDampingLaneCenterRate = float(debug.get('lane_rate_damping_lane_center_rate', 0.0))
  model_path_state.laneRateDampingLatAccel = float(debug.get('lane_rate_damping_lat_accel', 0.0))
  model_path_state.laneRateDampingCurvature = float(debug.get('lane_rate_damping_curvature', 0.0))
  model_path_state.laneRateDampingCapLatAccel = float(debug.get('lane_rate_damping_cap_lat_accel', 0.05))


def set_model_path_state_lane_fit_source(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.laneFitSourceMode = str(debug.get('lane_fit_source_mode', 'off'))
  model_path_state.laneFitSourceActive = bool(debug.get('lane_fit_source_active', False))
  model_path_state.laneFitSourceApplied = bool(debug.get('lane_fit_source_applied', False))
  model_path_state.laneFitSourceReason = str(debug.get('lane_fit_source_reason', default_reason))
  model_path_state.laneFitSourceCandidateCurvature = float(debug.get('lane_fit_source_candidate_curvature', 0.0))
  model_path_state.laneFitSourceAppliedCurvature = float(debug.get('lane_fit_source_applied_curvature', 0.0))
  model_path_state.laneFitSourceLatAccelDelta = float(debug.get('lane_fit_source_lat_accel_delta', 0.0))
  model_path_state.laneFitSourceConfidence = float(debug.get('lane_fit_source_confidence', 0.0))
  model_path_state.laneFitSourceSlewLimited = bool(debug.get('lane_fit_source_slew_limited', False))


def set_model_path_state_sensor_confidence(model_path_state, debug: dict | None = None, *, default_reason: str = "disabled") -> None:
  debug = debug or {}
  model_path_state.sensorConfidenceAvailable = bool(debug.get('sensor_confidence_available', False))
  model_path_state.sensorConfidenceBlockReason = str(debug.get('sensor_confidence_block_reason', default_reason))
  model_path_state.sensorConfidenceScore = float(debug.get('sensor_confidence_score', 0.0))
  model_path_state.sensorDisagreementLevel = str(debug.get('sensor_disagreement_level', 'blocked'))
  model_path_state.sensorSuppressCandidate = bool(debug.get('sensor_suppress_candidate', False))
  model_path_state.sensorResponseClassification = str(debug.get('sensor_response_classification', 'blocked'))
  model_path_state.sensorModelMeasuredCurvatureDelta = float(debug.get('sensor_model_measured_curvature_delta', float('nan')))
  model_path_state.sensorModelMeasuredLatAccelDelta = float(debug.get('sensor_model_measured_lat_accel_delta', float('nan')))
  model_path_state.sensorYawCurvature = float(debug.get('sensor_yaw_curvature', float('nan')))
  model_path_state.sensorModelYawLatAccelDelta = float(debug.get('sensor_model_yaw_lat_accel_delta', float('nan')))
  model_path_state.sensorSteeringYawLatAccelDelta = float(debug.get('sensor_steering_yaw_lat_accel_delta', float('nan')))
  model_path_state.sensorModelYawLatAccelSignedDelta = float(debug.get('sensor_model_yaw_lat_accel_signed_delta', float('nan')))
  model_path_state.sensorSteeringYawLatAccelSignedDelta = float(debug.get('sensor_steering_yaw_lat_accel_signed_delta', float('nan')))


def fill_model_path_state_disabled_defaults(model_path_state, raw_curvature_for_log: float,
                                            processed_curvature_for_log: float | None = None) -> None:
  if processed_curvature_for_log is None:
    processed_curvature_for_log = raw_curvature_for_log
  model_path_state.active = False
  model_path_state.gated = False
  model_path_state.quality = 0.0
  model_path_state.reason = "disabled"
  model_path_state.rawDesiredCurvature = raw_curvature_for_log
  # No conditioning happened: Conditioned Lateral Demand equals the raw input; Processed
  # Lateral Demand is still the post-cap controller input.
  model_path_state.conditionedDesiredCurvature = raw_curvature_for_log
  model_path_state.processedDesiredCurvature = processed_curvature_for_log
  model_path_state.modelPathCurvature = raw_curvature_for_log
  model_path_state.laneCenteringActive = False
  model_path_state.laneCenteringReason = "disabled"
  model_path_state.laneCenteringLateralError = 0.0
  model_path_state.laneCenteringHeadingError = 0.0
  model_path_state.laneCenteringPredictedError = 0.0
  model_path_state.laneCenteringCurvatureNudge = 0.0
  model_path_state.laneCenteringConfidence = 0.0
  model_path_state.laneCenteringRelaxActive = False
  model_path_state.laneCenteringRelaxReasonBits = 0
  model_path_state.laneCenteringRelaxEnvelope = 0.0
  model_path_state.laneCenteringRelaxLateralError = 0.0
  model_path_state.laneCenteringRelaxPredictedError = 0.0
  model_path_state.laneCenteringRelaxAge = 0.0
  model_path_state.laneCenteringRelaxNudgeFlipScore = 0.0
  model_path_state.laneCenteringRelaxErrorCrossScore = 0.0
  model_path_state.curveMemoryActive = False
  model_path_state.curveMemoryRemembered = float('nan')
  model_path_state.laneChangeBlend = 0.0
  model_path_state.laneChangeShapingActive = False
  model_path_state.demandSource = "disabled"
  model_path_state.dtleEstimate = float('nan')
  set_model_path_state_geometry(model_path_state)
  set_model_path_state_lane_fit_source(model_path_state)
  set_model_path_state_lane_rate_damping(model_path_state)
  set_model_path_state_sps(model_path_state)
  set_model_path_state_one_line(model_path_state)
  set_model_path_state_preview(model_path_state)
  set_model_path_state_sensor_confidence(model_path_state)


_speed_shadow_cache: tuple | None = None  # (key, v_delay, v_05, v_10, a_delay)


def set_model_path_state_speed_shadow(model_path_state, curvature: float, v_ego: float, a_ego: float,
                                      plan_speeds, plan_accels, lat_delay: float, *, plan_valid: bool = True,
                                      plan_key=None) -> None:
  def finite_or_nan(value) -> float:
    try:
      value = float(value)
    except (TypeError, ValueError):
      return float('nan')
    return value if math.isfinite(value) else float('nan')

  curvature = finite_or_nan(curvature)
  v_ego = finite_or_nan(v_ego)
  a_ego = finite_or_nan(a_ego)

  def predicted(seq, t: float) -> float:
    t = finite_or_nan(t)
    if not plan_valid or not math.isfinite(t) or len(seq) != len(CONTROL_N_T_IDXS):
      return float('nan')
    values = np.asarray(seq, dtype=float)
    if not np.all(np.isfinite(values)):
      return float('nan')
    return float(np.interp(max(0.0, float(t)), CONTROL_N_T_IDXS, values))

  model_path_state.shadowCurrentLatAccel = curvature * v_ego ** 2
  model_path_state.shadowCurrentJerkSpeedTerm = 2.0 * curvature * v_ego * a_ego

  # the four interpolations depend only on the 20Hz plan + lat_delay; cache across 100Hz cycles
  global _speed_shadow_cache
  cache_key = (plan_key, lat_delay, plan_valid) if plan_key is not None else None
  if cache_key is not None and _speed_shadow_cache is not None and _speed_shadow_cache[0] == cache_key:
    v_delay, v_05, v_10, a_delay = _speed_shadow_cache[1:]
  else:
    v_delay = predicted(plan_speeds, lat_delay)
    v_05 = predicted(plan_speeds, 0.5)
    v_10 = predicted(plan_speeds, 1.0)
    a_delay = predicted(plan_accels, lat_delay)
    if cache_key is not None:
      _speed_shadow_cache = (cache_key, v_delay, v_05, v_10, a_delay)

  model_path_state.shadowLatDelayLatAccel = curvature * v_delay ** 2
  model_path_state.shadow05sLatAccel = curvature * v_05 ** 2
  model_path_state.shadow10sLatAccel = curvature * v_10 ** 2
  model_path_state.shadowLatDelayJerkSpeedTerm = 2.0 * curvature * v_delay * a_delay


def publish_model_path_state(model_path_state, sm, lateral_demand, v_ego: float, a_ego: float,
                             raw_desired_curvature: float, processed_desired_curvature: float,
                             lat_delay: float) -> None:
  """Serialize the full custom lateral telemetry block (100Hz, controlsd publish path)."""
  long_plan = sm['longitudinalPlan']
  long_plan_valid = (
    sm.valid['longitudinalPlan'] and
    sm.alive['longitudinalPlan'] and
    sm.freq_ok['longitudinalPlan']
  )
  # ponytail: no unconditional default pre-fill; the success path overwrites every default, so
  # defaults are written only on the else/exception paths (was ~73 redundant capnp sets at 100Hz)
  set_model_path_state_speed_shadow(model_path_state, processed_desired_curvature, v_ego, a_ego,
                                    long_plan.speeds, long_plan.accels, lat_delay,
                                    plan_valid=long_plan_valid,
                                    plan_key=sm.logMonoTime['longitudinalPlan'])
  last_result = getattr(lateral_demand, 'last_result', None) if lateral_demand is not None else None
  if last_result is not None:
    try:
      d = last_result.demand
      model_path = last_result.model_path_result
      debug = getattr(last_result, 'debug', {}) or {}
      model_path_state.active = bool(getattr(lateral_demand, 'enabled', False))
      model_path_state.gated = bool(model_path.gated)
      model_path_state.quality = float(model_path.quality)
      model_path_state.reason = str(model_path.reason)
      model_path_state.rawDesiredCurvature = float(d.raw_curvature)
      # Conditioned Lateral Demand: the pipeline result before hard curvature / lat-accel caps.
      model_path_state.conditionedDesiredCurvature = float(d.processed_curvature)
      # Processed Lateral Demand: the post-cap controller-facing curvature.
      model_path_state.processedDesiredCurvature = float(processed_desired_curvature)
      model_path_state.modelPathCurvature = float(debug.get('model_path_curvature', d.processed_curvature))
      model_path_state.laneCenteringActive = bool(d.lane_centering_assist_active)
      model_path_state.laneCenteringReason = str(d.lane_centering_reason)
      model_path_state.laneCenteringLateralError = float(d.lane_centering_lateral_error)
      model_path_state.laneCenteringHeadingError = float(d.lane_centering_heading_error)
      model_path_state.laneCenteringPredictedError = float(d.lane_centering_predicted_error)
      model_path_state.laneCenteringCurvatureNudge = float(d.lane_centering_curvature_nudge)
      model_path_state.laneCenteringConfidence = float(d.lane_centering_confidence)
      model_path_state.laneCenteringRelaxActive = bool(d.lane_centering_relax_active)
      model_path_state.laneCenteringRelaxReasonBits = int(d.lane_centering_relax_reason_bits)
      model_path_state.laneCenteringRelaxEnvelope = float(d.lane_centering_relax_envelope)
      model_path_state.laneCenteringRelaxLateralError = float(d.lane_centering_relax_lateral_error)
      model_path_state.laneCenteringRelaxPredictedError = float(d.lane_centering_relax_predicted_error)
      model_path_state.laneCenteringRelaxAge = float(d.lane_centering_relax_age)
      model_path_state.laneCenteringRelaxNudgeFlipScore = float(d.lane_centering_relax_nudge_flip_score)
      model_path_state.laneCenteringRelaxErrorCrossScore = float(d.lane_centering_relax_error_cross_score)
      model_path_state.curveMemoryActive = bool(debug.get('curve_memory_active', False))
      model_path_state.curveMemoryRemembered = float(debug.get('curve_memory_remembered', float('nan')))
      model_path_state.laneChangeBlend = float(debug.get('lane_change_blend', 0.0))
      model_path_state.laneChangeShapingActive = bool(debug.get('lane_change_shaping_active', False))
      model_path_state.demandSource = str(debug.get('demand_source', 'model_path'))
      model_path_state.dtleEstimate = float(debug.get('dtle_estimate', float('nan')))
      set_model_path_state_geometry(model_path_state, debug)
      set_model_path_state_lane_fit_source(model_path_state, debug, default_reason="missing")
      set_model_path_state_preview(model_path_state, debug, default_reason="missing")
      set_model_path_state_sensor_confidence(model_path_state, debug, default_reason="missing")
      set_model_path_state_lane_rate_damping(model_path_state, debug, default_reason="missing")
      set_model_path_state_sps(model_path_state, debug, default_reason="missing")
      set_model_path_state_one_line(model_path_state, debug, default_reason="missing")
    except Exception:
      cloudlog.exception("failed to publish lateral modelPathState telemetry")
      lateral_demand.clear()
      fill_model_path_state_disabled_defaults(model_path_state, raw_desired_curvature, processed_desired_curvature)
  else:
    fill_model_path_state_disabled_defaults(model_path_state, raw_desired_curvature, processed_desired_curvature)
    debug = getattr(lateral_demand, 'last_debug', {}) or {}
    if debug:
      set_model_path_state_lane_fit_source(model_path_state, debug, default_reason="missing")
      if 'lateral_preview_assist_mode' in debug:
        set_model_path_state_preview(model_path_state, debug, default_reason="missing")
      else:
        set_model_path_state_preview(model_path_state)
      set_model_path_state_sensor_confidence(model_path_state, debug, default_reason="missing")
    else:
      set_model_path_state_lane_fit_source(model_path_state)
      set_model_path_state_preview(model_path_state)
    set_model_path_state_lane_rate_damping(model_path_state, debug, default_reason="missing")
    set_model_path_state_sps(model_path_state, debug, default_reason="missing")
    set_model_path_state_one_line(model_path_state, debug, default_reason="missing")
