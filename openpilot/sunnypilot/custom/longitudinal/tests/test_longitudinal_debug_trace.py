from __future__ import annotations

import math
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from openpilot.cereal import custom
from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.departure_prediction import DeparturePredictionTrace
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.net_demand_cap import NetDemandCapTrace
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalAdapter
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


class DummyPm:
  def __init__(self):
    self.msg = None

  def send(self, name, msg):
    assert name == 'longitudinalPlanSP'
    self.msg = msg


def fake_sm():
  class SM:
    def all_checks(self, service_list=None):
      return True

    def __getitem__(self, key):
      return {
        'carState': SimpleNamespace(vEgo=12.3, vCruise=123.0),
        'controlsState': SimpleNamespace(),
      }[key]
  return SM()


def custom_output(**kwargs):
  defaults = dict(enabled=True, a_target=0.25, should_stop=True, selected_intent="intent", reason="reason", debug={})
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


def make_planner(debug_trace_mode: str, custom_long_output: Any | None = None) -> Any:
  planner: Any = object.__new__(LongitudinalPlannerSP)
  planner.CP = SimpleNamespace(stoppingDistance=6.0, vEgoStopping=0.5, stopAccel=-0.5, openpilotLongitudinalControl=True)
  planner.custom_long_finalizer = CustomLongitudinalFinalizer(planner.CP)
  planner.custom_long = SimpleNamespace(enabled=True, debug_trace_mode=debug_trace_mode, mode=LongitudinalMode.ACC)
  planner._custom_long_output_telemetry = None
  planner.custom_long_output = custom_long_output if custom_long_output is not None else custom_output()
  planner._last_longitudinal_debug = {}
  planner._sm_item = LongitudinalPlannerSP._sm_item
  planner._custom_longitudinal_mode_to_telemetry = lambda: 0
  planner.events_sp = SimpleNamespace(to_msg=list)
  planner.source = 0
  planner.output_v_target = 0.0
  planner.output_a_target = 0.0
  planner.dec = SimpleNamespace(mode=lambda: 'acc', enabled=lambda: False, active=lambda: False)
  planner.scc = SimpleNamespace(
    vision=SimpleNamespace(state=0, output_v_target=0.0, output_a_target=0.0, current_lat_acc=0.0,
                           max_pred_lat_acc=0.0, is_enabled=False, is_active=False),
    map=SimpleNamespace(state=0, output_v_target=0.0, output_a_target=0.0, is_enabled=False, is_active=False),
  )
  planner.resolver = SimpleNamespace(speed_limit=0.0, speed_limit_last=0.0, speed_limit_final=0.0,
                                     speed_limit_final_last=0.0, speed_limit_valid=False,
                                     speed_limit_last_valid=False, speed_limit_offset=0.0,
                                     distance=0.0, source=0)
  planner.sla = SimpleNamespace(state=0, is_enabled=False, is_active=False, output_v_target=0.0,
                                output_a_target=0.0)
  planner.e2e_alerts_helper = SimpleNamespace(green_light_alert=False, lead_depart_alert=False)
  return planner


def publish(planner: Any):
  pm = DummyPm()
  planner.publish_longitudinal_plan_sp(fake_sm(), pm)
  assert pm.msg is not None
  return pm.msg.longitudinalPlanSP


def test_absent_confidence_trace_uses_negative_one_schema_defaults():
  # A pre-trace/old-style message has no confidenceTrace payload at all. The schema defaults
  # must preserve "not measured" sentinels instead of making an absent lead/model look valid.
  debug = custom.LongitudinalPlanSP.LongitudinalDebug.new_message()
  assert debug.confidenceTrace.selectedTrackId == -1
  assert debug.confidenceTrace.modelAgeS == pytest.approx(-1.0)

  plan = custom.LongitudinalPlanSP.new_message()
  with custom.LongitudinalPlanSP.from_bytes(plan.to_bytes()) as decoded:
    assert decoded.longitudinalDebug.confidenceTrace.selectedTrackId == -1
    assert decoded.longitudinalDebug.confidenceTrace.modelAgeS == pytest.approx(-1.0)


def test_publisher_without_model_age_debug_marks_confidence_trace_stale():
  confidence = publish(make_planner("log", custom_output(debug={}))).longitudinalDebug.confidenceTrace

  assert confidence.selectedTrackId == -1
  assert confidence.modelAgeS == pytest.approx(-1.0)
  assert confidence.modelStale is True


def test_publish_trace_off_leaves_debug_disabled_and_preserves_custom_telemetry():
  planner = make_planner("off", custom_output(a_target=-0.5, should_stop=True, selected_intent="hold", reason="lead"))

  plan = publish(planner)

  assert plan.longitudinalDebug.enabled is False
  assert plan.customLongitudinal.enabled is True
  assert plan.customLongitudinal.active is True
  assert plan.customLongitudinal.shouldStop is True
  assert plan.customLongitudinal.selectedIntent == "hold"
  assert plan.customLongitudinal.reason == "lead"


def test_trace_mode_does_not_mutate_custom_telemetry():
  output = custom_output(a_target=-0.5, should_stop=True, selected_intent="hold", reason="lead")
  off = publish(make_planner("off", output)).customLongitudinal
  logged = publish(make_planner("log", output)).customLongitudinal

  for field in ("enabled", "active", "shouldStop", "selectedIntent", "reason"):
    assert getattr(logged, field) == getattr(off, field)
  assert off is not logged


def test_debug_trace_populates_whitelisted_fields():
  planner = make_planner("log", custom_output(debug={
    'actual_primary_lead_authority': 'physical',
    'actual_primary_lead_d_rel': 20.0,
    'actual_primary_lead_v_rel': -2.0,
    'actual_primary_lead_y_rel': 0.1,
    'cut_in_brake_assist_mode': 'shadow',
    'cut_in_brake_assist_effective_mode': 'shadow',
    'cut_in_brake_assist_apply_supported': False,
    'cut_in_brake_assist_eligible': True,
    'cut_in_brake_assist_block_reason': '',
    'cut_in_brake_assist_lead_idx': 0,
    'cut_in_brake_assist_path_y_rel': 0.3,
    'cut_in_brake_assist_lateral_velocity': 0.0,
    'cut_in_brake_assist_ttc': 3.5,
    'cut_in_brake_assist_required_decel': 0.4,
    'cut_in_brake_assist_proposed_cap': -0.7,
    'cut_in_brake_assist_confidence': 0.8,
    'standstill_release_confidence_mode': 'shadow',
    'standstill_release_confidence_effective_mode': 'shadow',
    'standstill_release_confidence_apply_supported': False,
    'standstill_release_confidence_eligible': True,
    'standstill_release_confidence_block_reason': '',
    'standstill_release_confidence_confidence': 0.9,
    'standstill_release_confidence_release_allowed': True,
    'standstill_release_confidence_release_source': 'lead_pullaway',
    'standstill_release_confidence_release_reason': 'lead_opening',
    'standstill_release_confidence_release_a_target': 0.25,
    'acc_envelope_active': True,
    'acc_envelope_would_cap': True,
    'acc_envelope_cap_reason': 'inside_time_gap,ttc_low',
    'acc_envelope_allowed_a_target': -0.4,
    'acc_envelope_delta_a': -0.6,
    'acc_envelope_desired_gap': 18.0,
    'acc_envelope_time_gap': 1.2,
    'acc_envelope_ttc': 2.5,
    'acc_envelope_usable_stopping_gap': -3.0,
    'acc_envelope_required_stopping_decel': 1.4,
    'acc_envelope_closing_speed_decel': 1.4,
    'acc_envelope_jerk_limited_a_target': 0.3,
    'map_coast_mode': 'shadow',
    'map_coast_v_target': 15.0,
    'map_coast_distance': 320.0,
    'map_coast_eligible': True,
    'map_coast_cap': -0.25,
    'map_coast_applied': False,
    'map_coast_fault': False,
    'map_coast_accel_coast': -0.28,
    'confidence_selected_lead_slot': 'leadTwo',
    'confidence_selected_track_id': 42,
    'confidence_selected_authority': 'physical',
    'confidence_acquisition_timer_s': 0.15,
    'confidence_flicker_guard_timer_s': 1.5,
    'confidence_model_age_s': 0.12,
    'confidence_model_stale': False,
    'confidence_model_service_healthy': True,
    'confidence_radar_service_healthy': True,
    'confidence_corroboration_hold_remaining_s': 2.5,
    'confidence_corroboration_refresh_source': 'radar',
    'confidence_anchor_travel_corroborated': True,
    'confidence_stationary_radar_correlation_applied': True,
    'confidence_effective_caution_floor': -1.25,
    'confidence_cut_out_remaining_s': 2.5,
    'confidence_stop_trust': 0.77,
  }))
  planner._last_longitudinal_debug = {'v_cruise': 15.2, 'mpc_a_target': 1.0, 'mpc_should_stop': True, 'model_a_target': -1.0,
                                      'model_should_stop': False, 'final_a_target_unclipped': 0.9,
                                      'final_a_target_clipped': 0.4, 'final_should_stop': True,
                                      'accel_clip_min': -2.0, 'accel_clip_max': 0.4, 'e2e_source': True}

  msg = publish(planner).longitudinalDebug

  assert msg.enabled is True
  assert msg.traceMode == 'log'
  assert msg.vEgo == pytest.approx(12.3)
  assert msg.vCruise == pytest.approx(15.2)
  assert msg.customATarget == pytest.approx(0.25)
  assert msg.customATargetPlan == pytest.approx(0.25)   # no separate plan -> mirrors command
  assert msg.customShouldStop is True
  assert msg.customIntent == "intent"
  assert msg.customReason == "reason"
  assert msg.mpcATarget == pytest.approx(1.0)
  assert msg.mpcShouldStop is True
  assert msg.modelATarget == pytest.approx(-1.0)
  assert msg.modelShouldStop is False
  assert msg.finalATargetUnclipped == pytest.approx(0.9)
  assert msg.finalATargetClipped == pytest.approx(0.4)
  assert msg.finalShouldStop is True
  assert msg.accelClipMin == pytest.approx(-2.0)
  assert msg.accelClipMax == pytest.approx(0.4)
  assert msg.e2eSource is True
  assert msg.cutInBrakeAssist.mode == 'shadow'
  assert msg.cutInBrakeAssist.applySupported is False
  assert msg.cutInBrakeAssist.eligible is True
  assert msg.cutInBrakeAssist.proposedCap == pytest.approx(-0.7)
  assert msg.cutInBrakeAssist.ttc == pytest.approx(3.5)
  assert msg.standstillReleaseConfidence.mode == 'shadow'
  assert msg.standstillReleaseConfidence.releaseAllowed is True
  assert msg.standstillReleaseConfidence.releaseSource == 'lead_pullaway'
  assert msg.standstillReleaseConfidence.releaseATarget == pytest.approx(0.25)
  assert msg.accEnvelope.active is True
  assert msg.accEnvelope.wouldCap is True
  assert msg.accEnvelope.capReason == 'inside_time_gap,ttc_low'
  assert msg.accEnvelope.allowedATarget == pytest.approx(-0.4)
  assert msg.accEnvelope.deltaA == pytest.approx(-0.6)
  assert msg.accEnvelope.desiredGap == pytest.approx(18.0)
  assert msg.accEnvelope.timeGap == pytest.approx(1.2)
  assert msg.accEnvelope.ttc == pytest.approx(2.5)
  assert msg.accEnvelope.usableStoppingGap == pytest.approx(-3.0)
  assert msg.accEnvelope.requiredStoppingDecel == pytest.approx(1.4)
  assert msg.accEnvelope.closingSpeedDecel == pytest.approx(1.4)
  assert msg.accEnvelope.jerkLimitedATarget == pytest.approx(0.3)
  assert msg.mapCoast.mode == 'shadow'
  assert msg.mapCoast.vTarget == pytest.approx(15.0)
  assert msg.mapCoast.distance == pytest.approx(320.0)
  assert msg.mapCoast.eligible is True
  assert msg.mapCoast.cap == pytest.approx(-0.25)
  assert msg.mapCoast.applied is False
  assert msg.mapCoast.fault is False
  assert msg.mapCoast.coastDecel == pytest.approx(-0.28)
  confidence = msg.confidenceTrace
  assert confidence.selectedLeadSlot == 'leadTwo'
  assert confidence.selectedTrackId == 42
  assert confidence.selectedAuthority == 'physical'
  assert confidence.acquisitionTimerS == pytest.approx(0.15)
  assert confidence.flickerGuardTimerS == pytest.approx(1.5)
  assert confidence.modelAgeS == pytest.approx(0.12)
  assert confidence.modelStale is False
  assert confidence.modelServiceHealthy is True
  assert confidence.radarServiceHealthy is True
  assert confidence.corroborationHoldRemainingS == pytest.approx(2.5)
  assert confidence.corroborationRefreshSource == 'radar'
  assert confidence.anchorTravelCorroborated is True
  assert confidence.stationaryRadarCorrelationApplied is True
  assert confidence.effectiveCautionFloor == pytest.approx(-1.25)
  assert confidence.cutOutRemainingS == pytest.approx(2.5)
  assert confidence.stopTrust == pytest.approx(0.77)


def test_confidence_trace_defaults_missing_model_age_to_negative_one_and_stale():
  planner = make_planner("log", custom_output(debug={
    'confidence_selected_lead_slot': 'leadOne',
    'confidence_selected_track_id': 7,
    'confidence_selected_authority': 'physical',
    'confidence_model_stale': True,
    'confidence_corroboration_refresh_source': 'vision',
    # No model age/health fields: the serializer must distinguish "not measured" from age 0.
  }))

  confidence = publish(planner).longitudinalDebug.confidenceTrace

  assert confidence.selectedLeadSlot == 'leadOne'
  assert confidence.selectedTrackId == 7
  assert confidence.modelAgeS == pytest.approx(-1.0)
  assert confidence.modelStale is True
  assert confidence.modelServiceHealthy is False
  assert confidence.radarServiceHealthy is False
  assert confidence.corroborationRefreshSource == 'vision'


def test_confidence_trace_sanitizes_non_finite_values_to_safe_defaults():
  planner = make_planner("log", custom_output(debug={
    'confidence_selected_lead_slot': 'bad',
    'confidence_selected_track_id': math.inf,
    'confidence_selected_authority': 'bad',
    'confidence_acquisition_timer_s': math.nan,
    'confidence_flicker_guard_timer_s': math.inf,
    'confidence_model_age_s': math.nan,
    'confidence_model_stale': None,
    'confidence_model_service_healthy': None,
    'confidence_radar_service_healthy': None,
    'confidence_corroboration_hold_remaining_s': math.nan,
    'confidence_corroboration_refresh_source': 'bad',
    'confidence_anchor_travel_corroborated': None,
    'confidence_stationary_radar_correlation_applied': None,
    'confidence_effective_caution_floor': math.inf,
    'confidence_cut_out_remaining_s': -math.inf,
    'confidence_stop_trust': math.nan,
  }))

  confidence = publish(planner).longitudinalDebug.confidenceTrace

  assert confidence.selectedLeadSlot == 'none'
  assert confidence.selectedTrackId == -1
  assert confidence.selectedAuthority == 'none'
  assert confidence.acquisitionTimerS == pytest.approx(0.0)
  assert confidence.flickerGuardTimerS == pytest.approx(0.0)
  assert confidence.modelAgeS == pytest.approx(-1.0)
  assert confidence.modelStale is True
  assert confidence.modelServiceHealthy is False
  assert confidence.radarServiceHealthy is False
  assert confidence.corroborationHoldRemainingS == pytest.approx(0.0)
  assert confidence.corroborationRefreshSource == 'none'
  assert confidence.anchorTravelCorroborated is False
  assert confidence.stationaryRadarCorrelationApplied is False
  assert confidence.effectiveCautionFloor == pytest.approx(0.0)
  assert confidence.cutOutRemainingS == pytest.approx(0.0)
  assert confidence.stopTrust == pytest.approx(0.0)


def test_debug_trace_sanitizes_non_finite_values_without_throwing():
  planner = make_planner("log", custom_output(a_target=math.nan, debug={
    'cut_in_brake_assist_mode': 'shadow',
    'cut_in_brake_assist_lead_idx': math.nan,
    'cut_in_brake_assist_proposed_cap': math.nan,
    'standstill_release_confidence_mode': 'shadow',
    'standstill_release_confidence_release_a_target': math.nan,
    'acc_envelope_active': True,
    'acc_envelope_allowed_a_target': math.nan,
    'acc_envelope_ttc': math.inf,
    'acc_envelope_required_stopping_decel': -math.inf,
  }))
  planner._last_longitudinal_debug = {
    'mpc_a_target': math.nan,
    'model_a_target': math.inf,
    'final_a_target_unclipped': math.nan,
    'final_a_target_clipped': math.inf,
    'accel_clip_min': -math.inf,
    'accel_clip_max': math.inf,
  }

  msg = publish(planner).longitudinalDebug

  assert msg.enabled is True
  assert msg.customATarget == pytest.approx(0.0)
  assert msg.mpcATarget == pytest.approx(0.0)
  assert msg.modelATarget == pytest.approx(0.0)
  assert msg.finalATargetUnclipped == pytest.approx(0.0)
  assert msg.finalATargetClipped == pytest.approx(0.0)
  assert msg.accelClipMin == pytest.approx(0.0)
  assert msg.accelClipMax == pytest.approx(0.0)
  assert msg.cutInBrakeAssist.leadIdx == -1
  assert msg.cutInBrakeAssist.proposedCap == pytest.approx(0.0)
  assert msg.standstillReleaseConfidence.releaseATarget == pytest.approx(0.0)
  assert msg.accEnvelope.active is True
  assert msg.accEnvelope.allowedATarget == pytest.approx(0.0)
  assert msg.accEnvelope.ttc == pytest.approx(0.0)
  assert msg.accEnvelope.requiredStoppingDecel == pytest.approx(0.0)


def test_uphill_net_demand_trace_is_structured() -> None:
  trace = NetDemandCapTrace(
    mode="apply",
    effective_mode="apply",
    eligible=True,
    would_cap=True,
    applied=True,
    regime="cap",
    source_age_s=0.05,
    car_pitch=0.10,
    pitch_zero=0.04,
    filtered_grade_percent=6.0,
    profile_ready=True,
    ceiling=1.2,
    grade_accel=0.6,
    a_target_before=0.8,
    a_target_cap=0.6,
    a_target_after=0.6,
    requested_net_demand=1.4,
    delta_a=-0.2,
    research_actuation_allowed=True,
  )
  planner = make_planner("log", custom_output(uphill_net_demand_trace=trace))

  msg = publish(planner).longitudinalDebug.uphillNetDemandCap

  assert msg.mode == "apply"
  assert msg.effectiveMode == "apply"
  assert msg.applied is True
  assert msg.regime == "cap"
  assert msg.profileReady is True
  assert msg.ceiling == pytest.approx(1.2)
  assert msg.aTargetBefore == pytest.approx(0.8)
  assert msg.aTargetAfter == pytest.approx(0.6)
  assert msg.deltaA == pytest.approx(-0.2)


def test_debug_trace_dynamic_safety_floor_serializes_fields():
  planner = make_planner("log", custom_output(debug={
    'dynamic_safety_floor_active': True,
    'dynamic_safety_floor_block_reason': 'pitch_unavailable',
    'dynamic_safety_floor_current_safe_distance': 18.5,
    'dynamic_safety_floor_proposed_safe_distance': 21.2,
    'dynamic_safety_floor_delta_safe_distance': 2.7,
    'dynamic_safety_floor_dynamic_floor_value': 4.0,
    'dynamic_safety_floor_kinematic_floor_violation': True,
    'dynamic_safety_floor_comfort_brake_effective': 2.1,
    'dynamic_safety_floor_latency_s': 0.35,
    'dynamic_safety_floor_lat_accel': 1.2,
    'dynamic_safety_floor_pitch': -0.03,
  }))

  msg = publish(planner).longitudinalDebug.dynamicSafetyFloor

  assert msg.active is True
  assert msg.blockReason == 'pitch_unavailable'
  assert msg.currentSafeDistance == pytest.approx(18.5)
  assert msg.proposedSafeDistance == pytest.approx(21.2)
  assert msg.deltaSafeDistance == pytest.approx(2.7)
  assert msg.dynamicFloorValue == pytest.approx(4.0)
  assert msg.kinematicFloorViolation is True
  assert msg.comfortBrakeEffective == pytest.approx(2.1)
  assert msg.latencyS == pytest.approx(0.35)
  assert msg.latAccel == pytest.approx(1.2)
  assert msg.pitch == pytest.approx(-0.03)


def test_debug_trace_dynamic_safety_floor_defaults_safely():
  planner = make_planner("log", custom_output(debug={
    'dynamic_safety_floor_active': True,
    # all numeric fields intentionally missing
  }))

  msg = publish(planner).longitudinalDebug.dynamicSafetyFloor

  assert msg.active is True
  assert msg.blockReason == ''
  assert msg.currentSafeDistance == pytest.approx(0.0)
  assert msg.proposedSafeDistance == pytest.approx(0.0)
  assert msg.deltaSafeDistance == pytest.approx(0.0)
  assert msg.dynamicFloorValue == pytest.approx(0.0)
  assert msg.kinematicFloorViolation is False
  assert msg.comfortBrakeEffective == pytest.approx(0.0)
  assert msg.latencyS == pytest.approx(0.0)
  assert msg.latAccel == pytest.approx(0.0)
  assert msg.pitch == pytest.approx(0.0)


def test_debug_trace_serializes_departure_prediction_trace():
  planner = make_planner("log", custom_output(
    departure_prediction_trace=DeparturePredictionTrace(
      mode="apply", effective_mode="shadow", apply_supported=False,
      phase="predicted", eligible=False, block_reason="mpc_brake_or_stop",
      track_id=17, evidence_s=0.2, age_s=0.35, predicted_gap_delta=0.8,
      would_coast=False, applied=False, a_target_before=-0.1,
      a_target_proposed=-0.1, a_target_after=-0.1, delta_a=0.0,
      research_actuation_allowed=False,
    ),
  ))

  trace = publish(planner).longitudinalDebug.departurePrediction
  assert trace.mode == "apply"
  assert trace.effectiveMode == "shadow"
  assert trace.applySupported is False
  assert trace.phase == "predicted"
  assert trace.eligible is False
  assert trace.blockReason == "mpc_brake_or_stop"
  assert trace.trackId == 17
  assert trace.evidenceS == pytest.approx(0.2)
  assert trace.ageS == pytest.approx(0.35)
  assert trace.predictedGapDelta == pytest.approx(0.8)
  assert trace.wouldCoast is False
  assert trace.applied is False
  assert trace.aTargetBefore == pytest.approx(-0.1)
  assert trace.aTargetProposed == pytest.approx(-0.1)
  assert trace.aTargetAfter == pytest.approx(-0.1)
  assert trace.deltaA == pytest.approx(0.0)
  assert trace.researchActuationAllowed is False


def test_plan_target_is_logged_separately_from_the_smoothed_command():
  # Since the a_desired plan/command split it is a_target_unsmoothed, not a_target, that
  # reaches the MPC's initial acceleration state. Logging only the smoothed value made the
  # solver's actual input invisible (route 000002dc jerk investigation).
  planner = make_planner("log", custom_output(a_target=-0.80, a_target_unsmoothed=-0.30))
  planner._last_longitudinal_debug = {'v_cruise': 15.2}
  msg = publish(planner).longitudinalDebug
  assert msg.customATarget == pytest.approx(-0.80)
  assert msg.customATargetPlan == pytest.approx(-0.30)


def test_plan_target_falls_back_to_the_command_when_absent():
  # NaN is the "no separate plan" sentinel and _safe_float would log it as a real 0.0.
  # The trace must show what actually reached a_desired.
  planner = make_planner("log", custom_output(a_target=-0.80, a_target_unsmoothed=float("nan")))
  planner._last_longitudinal_debug = {'v_cruise': 15.2}
  msg = publish(planner).longitudinalDebug
  assert msg.customATargetPlan == pytest.approx(-0.80)


class _TraceParams:
  def __init__(self, trace_mode: str):
    self._values = {
      "CustomLongitudinalEnabled": True,
      "CustomLongitudinalMode": "scc",
      "LongitudinalDebugTraceMode": trace_mode,
    }

  def get_bool(self, key):
    return bool(self._values.get(key, False))

  def get(self, key):
    return self._values.get(key)


class _TraceSubMaster(dict):
  recv_time: dict
  valid: dict
  alive: dict
  freq_ok: dict


def _trace_lead(d_rel=6.2, v_lead=0.0, v_rel=0.0, track_id=7, status=True, radar=True):
  return SimpleNamespace(
    status=status, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, vRel=v_rel,
    aLeadK=0.0, yRel=0.0, radarTrackId=track_id, radar=radar, modelProb=0.9, aLeadTau=1.0,
  )


def _trace_sm(lead_one=None, lead_two=None, *, model_stop=False, model_a=0.0,
              model_age=0.0, radar_ok=True):
  sm = _TraceSubMaster(
    radarState=SimpleNamespace(leadOne=lead_one, leadTwo=lead_two),
    carState=SimpleNamespace(
      vEgo=0.0, aEgo=0.0, brakePressed=False, gasPressed=False, standstill=True, vCruise=12.0,
    ),
    modelV2=SimpleNamespace(
      action=SimpleNamespace(shouldStop=model_stop, desiredAcceleration=model_a),
      position=None, velocity=None,
    ),
    carControl=SimpleNamespace(longActive=True, orientationNED=[0.0, 0.0, 0.0]),
    controlsState=SimpleNamespace(forceDecel=False),
    selfdriveState=SimpleNamespace(experimentalMode=False),
  )
  now = time.monotonic()
  sm.recv_time = {"modelV2": now - model_age, "radarState": now}
  sm.valid = {"modelV2": True, "radarState": radar_ok}
  sm.alive = {"modelV2": True, "radarState": radar_ok}
  sm.freq_ok = {"modelV2": True, "radarState": radar_ok}
  return sm


def _trace_pipeline(trace_mode: str):
  adapter = CustomLongitudinalAdapter(_TraceParams(trace_mode))
  # Keep the typed verdict path live in both pairs; with all optional modes off, the stack
  # intentionally omits those debug-only verdict records when collection is disabled.
  adapter.cut_in_brake_assist_mode = "apply"
  cp = SimpleNamespace(
    stoppingDistance=6.0, vEgoStopping=0.5, stopAccel=-0.5, openpilotLongitudinalControl=True,
  )
  planner: Any = object.__new__(LongitudinalPlannerSP)
  planner.CP = cp
  planner.custom_long = adapter
  planner.custom_long_finalizer = CustomLongitudinalFinalizer(cp)
  planner.dt = 0.05
  planner.custom_long_output = None
  return adapter, planner


def _trace_output_behavior(output):
  if output is None:
    return None
  return replace(output, debug={})


def _trace_release_behavior(output):
  if output is None:
    return None
  return tuple(getattr(output, field) for field in (
    "standstill_release_allowed", "standstill_release_source", "standstill_release_a_target",
    "standstill_release_reason",
  ))


def _trace_finalizer_behavior(result, telemetry, planner):
  return (
    result, planner._last_release_block_reason,
    _trace_output_behavior(telemetry), planner.custom_long_finalizer.departure_prediction_trace,
  )


def _trace_state_snapshot(adapter, planner):
  stack = adapter._stack
  finalizer = planner.custom_long_finalizer
  lead_confidence = tuple(
    (
      tracker.track_id, tracker.age, tracker.guard_timer, tracker.was_status,
    )
    for tracker in stack._lead_confidence
  )
  adapter_state = (
    adapter._stop_trust.confidence,
    adapter._caution_ramp.floor,
    adapter._corroboration_hold.hold_s,
    adapter._stop_anchor.remaining,
    adapter._stop_anchor.committed_s,
    adapter._stop_anchor.corroborated,
    adapter._stop_anchor._missing_s,
  )
  stack_state = (
    stack._prev_smoothed_a_target,
    stack._lead_gap_compression_active,
    lead_confidence,
  )
  finalizer_fields = (
    "lead_stop_hold_active", "lead_stop_hold_gap_increasing_s", "lead_stop_hold_missing_s",
    "lead_stop_hold_lead_id", "lead_stop_hold_gap_prev_d_rel", "lead_stop_hold_gap_baseline_d_rel",
    "lead_stop_hold_release_carry_s", "stop_hold_release_sustain_s", "final_a_prev",
  )
  return adapter_state, stack_state, tuple(getattr(finalizer, field) for field in finalizer_fields)


def test_trace_off_and_log_are_exactly_paired_through_adapter_stack_and_finalizer():
  # This deliberately uses the typed output from the real adapter/stack and feeds it through
  # the real post-MPC finalizer. Only ``debug`` is excluded from equality: it is the trace
  # serialization, not an input to control or state transitions.
  sequence = [
    ("clear", {}),
    ("clear", {}),
    ("new_lead", {"lead_one": _trace_lead(d_rel=30.0, v_lead=12.0, track_id=7)}),
    ("id_churn", {"lead_one": _trace_lead(d_rel=20.0, v_lead=8.0, v_rel=-4.0, track_id=11)}),
    ("status_churn", {"lead_one": _trace_lead(d_rel=20.0, v_lead=8.0, v_rel=-4.0, track_id=11, status=False)}),
    ("slot_churn", {"lead_two": _trace_lead(d_rel=18.0, v_lead=8.0, v_rel=-4.0, track_id=22)}),
    ("model_slowdown", {"lead_one": _trace_lead(d_rel=10.0, v_lead=5.0, v_rel=-5.0, track_id=22), "model_a": -0.8}),
    ("model_stop", {"lead_one": _trace_lead(track_id=22), "model_stop": True, "model_a": -1.5}),
    ("id_churn", {"lead_one": _trace_lead(track_id=11), "model_stop": True, "model_a": -1.5}),
    ("id_churn", {"lead_one": _trace_lead(track_id=22), "model_stop": True, "model_a": -1.5}),
    ("radar_dropout", {"model_stop": True, "model_a": -2.0, "model_age": 0.3, "radar_ok": False}),
    ("radar_dropout", {"model_stop": True, "model_a": -2.0, "model_age": 0.3, "radar_ok": False}),
    ("lead_pullaway", {"lead_one": _trace_lead(d_rel=6.3, v_lead=0.35, v_rel=0.35, track_id=22)}),
    ("lead_pullaway", {"lead_one": _trace_lead(d_rel=6.45, v_lead=0.5, v_rel=0.5, track_id=22)}),
    ("lead_pullaway", {"lead_one": _trace_lead(d_rel=6.65, v_lead=0.7, v_rel=0.7, track_id=22)}),
    ("released", {"lead_one": _trace_lead(d_rel=6.85, v_lead=0.8, v_rel=0.8, track_id=22)}),
  ]
  paired = (_trace_pipeline("off"), _trace_pipeline("log"))
  seen = set()
  release_seen = False
  previous_hold = False

  for label, frame in sequence:
    seen.add(label)
    results = []
    for adapter, planner in paired:
      sm = _trace_sm(**frame)
      model_a = frame.get("model_a", 0.0)
      model_stop = frame.get("model_stop", False)
      output = adapter.evaluate(
        sm, 0.0, 0.0, 12.0, 0.0,
        SimpleNamespace(
          vision=SimpleNamespace(is_active=False, output_a_target=0.0),
          map=SimpleNamespace(is_active=False, output_a_target=0.0),
        ),
        SimpleNamespace(is_active=False, output_v_target=0.0, output_a_target=0.0),
        dt=0.05, collect_debug=adapter.debug_trace_mode == "log",
      )
      planner.custom_long_output = output
      final = planner.final_longitudinal_output(
        sm, output.a_target, output.should_stop, model_a, model_stop,
      )
      results.append((output, final, planner._custom_long_output_telemetry,
                     _trace_state_snapshot(adapter, planner)))

    off, logged = results
    assert off[0].debug == {}
    assert logged[0].debug
    assert _trace_output_behavior(off[0]) == _trace_output_behavior(logged[0])
    assert _trace_release_behavior(off[0]) == _trace_release_behavior(logged[0])
    assert _trace_finalizer_behavior(off[1], off[2], paired[0][1]) == _trace_finalizer_behavior(
      logged[1], logged[2], paired[1][1],
    )
    assert _trace_release_behavior(off[2]) == _trace_release_behavior(logged[2])
    assert off[3] == logged[3]

    hold_active = off[3][2][0]
    release_seen = release_seen or (previous_hold and not hold_active)
    previous_hold = hold_active

  assert {"new_lead", "id_churn", "status_churn", "slot_churn", "model_slowdown",
          "model_stop", "radar_dropout", "lead_pullaway", "released"} <= seen
  assert release_seen
  assert paired[0][0].debug_trace_mode == "off"
  assert paired[1][0].debug_trace_mode == "log"
