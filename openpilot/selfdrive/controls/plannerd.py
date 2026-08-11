#!/usr/bin/env python3
from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
import openpilot.cereal.messaging as messaging


PLANNER_VALIDITY_CHECKS = {
  'longitudinalPlan': ['carControl', 'carState', 'carStateSP', 'controlsState', 'liveParameters', 'modelV2', 'radarState', 'selfdriveState', 'selfdriveStateSP'],
  'longitudinalPlanSP': ['carState', 'controlsState'],
  'driverAssistance': ['carState', 'carControl', 'modelV2', 'liveParameters'],
}
PLANNER_VALIDITY_SERVICES = sorted({s for services in PLANNER_VALIDITY_CHECKS.values() for s in services})


def _freq_tracker_snapshot(sm, service):
  tracker = sm.freq_tracker[service]

  def _freq(avg):
    if avg.count == 0:
      return None
    average_dt = avg.get_average()
    return None if average_dt <= 0 else float(1. / average_dt)

  return {
    'avgCount': int(tracker.avg_dt.count),
    'recentCount': int(tracker.recent_avg_dt.count),
    'avgFreq': _freq(tracker.avg_dt),
    'recentFreq': _freq(tracker.recent_avg_dt),
    'minFreq': float(tracker.min_freq),
    'maxFreq': float(tracker.max_freq),
  }


def _planner_validity_diag(sm):
  output_valid = {name: sm.all_checks(service_list=services) for name, services in PLANNER_VALIDITY_CHECKS.items()}
  services = {
    s: {
      'valid': bool(sm.valid[s]),
      'alive': bool(sm.alive[s]),
      'freqOk': bool(sm.freq_ok[s]),
      'updated': bool(sm.updated[s]),
      'recvFrame': int(sm.recv_frame[s]),
      'frameAge': int(sm.frame - sm.recv_frame[s]),
      'logMonoTime': int(sm.logMonoTime[s]),
      'freq': _freq_tracker_snapshot(sm, s),
    }
    for s in PLANNER_VALIDITY_SERVICES
  }
  failed = {
    name: {
      'invalid': [s for s in check_services if not sm.valid[s]],
      'notAlive': [s for s in check_services if not sm.alive[s]],
      'notFreqOk': [s for s in check_services if sm._check_avg_freq(s) and not sm.freq_ok[s]],
    }
    for name, check_services in PLANNER_VALIDITY_CHECKS.items()
    if not output_valid[name]
  }
  return {
    'outputs': output_valid,
    'failed': failed,
    'services': services,
    'frame': int(sm.frame),
    'modelV2LogMonoTime': int(sm.logMonoTime['modelV2']),
  }


def _planner_validity_signature(diag):
  if all(diag['outputs'].values()):
    return ('ok',)
  return tuple(
    (name,
     tuple(failed['invalid']),
     tuple(failed['notAlive']),
     tuple(failed['notFreqOk']))
    for name, failed in sorted(diag['failed'].items())
  )


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
  ignore_services = ["liveMapDataSP", "carStateSP", "selfdriveStateSP", gps_location_service]

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP, CP_SP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'livePose', 'liveCalibration',
                            'radarState', 'modelV2', 'selfdriveState', 'liveMapDataSP', 'carStateSP', 'selfdriveStateSP', gps_location_service],
                           poll='carState', ignore_alive=ignore_services, ignore_avg_freq=ignore_services,
                           ignore_valid=ignore_services)
  last_validity_signature = None
  last_validity_failed = False

  while True:
    sm.update()
    longitudinal_planner.sla.update_car_state(sm['carState'])
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      validity_diag = _planner_validity_diag(sm)
      validity_signature = _planner_validity_signature(validity_diag)
      validity_failed = not all(validity_diag['outputs'].values())
      if (validity_failed or last_validity_failed) and validity_signature != last_validity_signature:
        if validity_failed:
          cloudlog.event("plannerd_validity", error=True, **validity_diag)
        else:
          cloudlog.event("plannerd_validity", **validity_diag)
      last_validity_signature = validity_signature
      last_validity_failed = validity_failed
      longitudinal_planner.publish(sm, pm)

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(['carState', 'carControl', 'modelV2', 'liveParameters'])
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
