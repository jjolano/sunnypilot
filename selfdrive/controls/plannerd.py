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
from typing import TypeAlias, TypedDict


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

  while True:
    sm.update()
    longitudinal_planner.sla.update_car_state(sm['carState'])
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      invalid_planner_check_signature = _log_invalid_planner_checks(sm, invalid_planner_check_signature)
      longitudinal_planner.publish(sm, pm)

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
