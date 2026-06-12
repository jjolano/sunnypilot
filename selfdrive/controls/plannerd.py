#!/usr/bin/env python3
from cereal import car, custom
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
import cereal.messaging as messaging
import time
from typing import Any, TypeAlias, TypedDict


class PlannerServiceCheck(TypedDict):
  service: str
  alive: bool
  freq_ok: bool
  valid: bool
  updated: bool
  recv_frame: int
  recv_age_ms: float
  log_mono_time: int


class PlannerCheckSnapshot(TypedDict):
  all_alive: bool
  all_freq_ok: bool
  all_valid: bool
  all_checks: bool
  services: list[PlannerServiceCheck]


PlannerCheckSignature: TypeAlias = tuple[tuple[str, tuple[tuple[str, bool, bool, bool], ...]], ...]
PlannerHealthSignature: TypeAlias = tuple[tuple[str, str], ...]


PLANNER_VALIDITY_CHECKS = {
  'longitudinalPlan': ['carState', 'controlsState', 'selfdriveState', 'radarState'],
  'longitudinalPlanSP': ['carState', 'controlsState'],
  'driverAssistance': ['carState', 'carControl', 'modelV2', 'liveParameters'],
}


def _planner_check_snapshot(sm: messaging.SubMaster, services: list[str]) -> PlannerCheckSnapshot:
  now = time.monotonic()
  service_checks: list[PlannerServiceCheck] = []
  for service in services:
    service_checks.append({
      'service': service,
      'alive': sm.alive[service],
      'freq_ok': sm.freq_ok[service],
      'valid': sm.valid[service],
      'updated': sm.updated[service],
      'recv_frame': sm.recv_frame[service],
      'recv_age_ms': round((now - sm.recv_time[service]) * 1000.0, 1),
      'log_mono_time': sm.logMonoTime[service],
    })

  return {
    'all_alive': sm.all_alive(services),
    'all_freq_ok': sm.all_freq_ok(services),
    'all_valid': sm.all_valid(services),
    'all_checks': sm.all_checks(services),
    'services': service_checks,
  }


def _planner_check_signature(checks: dict[str, PlannerCheckSnapshot]) -> PlannerCheckSignature:
  return tuple(
    (message, tuple((s['service'], s['alive'], s['freq_ok'], s['valid']) for s in snapshot['services']))
    for message, snapshot in sorted(checks.items())
    if not snapshot['all_checks']
  )


def _log_invalid_planner_checks(sm: messaging.SubMaster, previous_signature: PlannerCheckSignature | None) -> PlannerCheckSignature | None:
  checks = {message: _planner_check_snapshot(sm, services) for message, services in PLANNER_VALIDITY_CHECKS.items()}
  signature = _planner_check_signature(checks)

  if signature and signature != previous_signature:
    cloudlog.event('plannerd_invalid_output_checks', error=True, frame=sm.frame, checks=checks)
  elif not signature and previous_signature is not None:
    cloudlog.event('plannerd_invalid_output_checks_recovered', frame=sm.frame, checks=checks)

  return signature or None


def _safe_service_age_ms(sm: messaging.SubMaster, service: str, now: float | None = None) -> float:
  now = time.monotonic() if now is None else now
  try:
    age = (now - sm.recv_time[service]) * 1000.0
  except (KeyError, TypeError, AttributeError):
    return 0.0
  return round(age, 1) if age >= 0.0 else 0.0


def _stable_signature_value(value: Any) -> str:
  if isinstance(value, float):
    return f"{value:.1f}"
  if isinstance(value, dict):
    return str(tuple(sorted((str(k), _stable_signature_value(v)) for k, v in value.items())))
  return str(value)


def _planner_health_signature(snapshot: dict[str, Any]) -> PlannerHealthSignature:
  signature_keys = (
    "update_reason",
    "skipped_reason",
    "primary_lead",
    "speed_limit_handoff",
  )
  return tuple((key, _stable_signature_value(snapshot.get(key, ""))) for key in signature_keys)


def _planner_health_snapshot(sm: messaging.SubMaster, planner=None, *, update_reason: str = "", skipped_reason: str = "",
                             last_valid_plan_time: float | None = None, now: float | None = None) -> dict[str, Any]:
  now = time.monotonic() if now is None else now
  primary_lead_context = getattr(planner, "primary_lead_context", None)
  primary_lead_debug = primary_lead_context.debug_dict() if hasattr(primary_lead_context, "debug_dict") else {}
  if last_valid_plan_time is None:
    last_valid_plan_age_ms = 0.0
  else:
    last_valid_plan_age_ms = round(max(0.0, now - last_valid_plan_time) * 1000.0, 1)
  return {
    "model_age_ms": _safe_service_age_ms(sm, "modelV2", now),
    "radar_age_ms": _safe_service_age_ms(sm, "radarState", now),
    "car_state_age_ms": _safe_service_age_ms(sm, "carState", now),
    "update_reason": str(update_reason),
    "skipped_reason": str(skipped_reason),
    "last_valid_plan_age_ms": last_valid_plan_age_ms,
    "primary_lead": primary_lead_debug,
    "speed_limit_handoff": dict(getattr(planner, "speed_limit_handoff_debug", {}) or {}),
  }


def _log_planner_health_debug(sm: messaging.SubMaster, planner, previous_signature: PlannerHealthSignature | None,
                              previous_log_time: float, *, update_reason: str = "", skipped_reason: str = "",
                              last_valid_plan_time: float | None = None, min_interval: float = 5.0) -> tuple[PlannerHealthSignature, float]:
  now = time.monotonic()
  snapshot = _planner_health_snapshot(
    sm, planner, update_reason=update_reason, skipped_reason=skipped_reason,
    last_valid_plan_time=last_valid_plan_time, now=now,
  )
  signature = _planner_health_signature(snapshot)
  if signature != previous_signature or now - previous_log_time >= min_interval:
    cloudlog.event('plannerd_health_debug', frame=sm.frame, health=snapshot)
    previous_log_time = now
  return signature, previous_log_time


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  cloudlog.info("plannerd is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("plannerd got CarParamsSP")

  gps_location_service = get_gps_location_service(params)

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP, CP_SP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'liveMapDataSP', 'carStateSP', gps_location_service],
                           poll='carState', ignore_avg_freq=['carState'])
  invalid_planner_check_signature = None
  planner_health_signature = None
  planner_health_last_log_time = 0.0
  last_valid_plan_time = None

  while True:
    sm.update()
    longitudinal_planner.sla.update_car_state(sm['carState'])
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      invalid_planner_check_signature = _log_invalid_planner_checks(sm, invalid_planner_check_signature)
      if sm.all_checks(['carState', 'controlsState', 'selfdriveState', 'radarState']):
        last_valid_plan_time = time.monotonic()
      planner_health_signature, planner_health_last_log_time = _log_planner_health_debug(
        sm, longitudinal_planner, planner_health_signature, planner_health_last_log_time,
        update_reason="modelV2_updated", last_valid_plan_time=last_valid_plan_time,
      )
      longitudinal_planner.publish(sm, pm)

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)
    else:
      planner_health_signature, planner_health_last_log_time = _log_planner_health_debug(
        sm, longitudinal_planner, planner_health_signature, planner_health_last_log_time,
        skipped_reason="modelV2_not_updated", last_valid_plan_time=last_valid_plan_time,
      )


if __name__ == "__main__":
  main()
