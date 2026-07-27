from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer
from openpilot.sunnypilot.custom.longitudinal.departure_prediction import DeparturePredictionTrace
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.net_demand_cap import NetDemandCapTrace
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


def test_publish_trace_off_leaves_debug_disabled_and_preserves_custom_telemetry():
  planner = make_planner("off", custom_output(a_target=-0.5, should_stop=True, selected_intent="hold", reason="lead"))

  plan = publish(planner)

  assert plan.longitudinalDebug.enabled is False
  assert plan.customLongitudinal.enabled is True
  assert plan.customLongitudinal.active is True
  assert plan.customLongitudinal.shouldStop is True
  assert plan.customLongitudinal.selectedIntent == "hold"
  assert plan.customLongitudinal.reason == "lead"


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
