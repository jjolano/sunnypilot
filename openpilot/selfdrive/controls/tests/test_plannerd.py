from types import SimpleNamespace

import pytest
import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls import plannerd
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


def test_plannerd_consumes_car_state_events_but_plans_once_per_model_frame(monkeypatch):
  planners = []
  submasters = []

  class FakeSLA:
    def __init__(self):
      self.events = []

    def update_car_state(self, car_state):
      self.events.append(car_state.buttonEvents[0].sequence)

  class FakePlanner:
    def __init__(self, *_):
      self.sla = FakeSLA()
      self.update_count = 0
      self.publish_count = 0
      planners.append(self)

    def update(self, _sm):
      self.update_count += 1

    def publish(self, _sm, _pm):
      self.publish_count += 1

  class FakeSubMaster:
    def __init__(self, _services, poll=None, **_kwargs):
      self.poll = poll
      self.frame = -1
      self.index = 0
      self.updated = {'modelV2': False}
      self.data = {
        'carState': SimpleNamespace(buttonEvents=[]),
        'modelV2': SimpleNamespace(),
        'carControl': SimpleNamespace(),
      }
      submasters.append(self)

    def update(self):
      if self.index == 6:
        raise StopIteration

      self.frame = self.index
      model_frame = self.index in (0, 5)
      if self.poll == 'carState' or model_frame:
        self.data['carState'] = SimpleNamespace(
          buttonEvents=[SimpleNamespace(pressed=False, sequence=self.index)])
      self.updated = {'modelV2': model_frame}
      self.index += 1

    def __getitem__(self, service):
      return self.data[service]

    def all_checks(self, _services=None, service_list=None):
      return True

  class FakePubMaster:
    def __init__(self, _services):
      self.sent = []

    def send(self, service, _msg):
      self.sent.append(service)

  monkeypatch.setattr(plannerd, 'config_realtime_process', lambda *_: None)
  monkeypatch.setattr(plannerd, 'Params', lambda: SimpleNamespace(get=lambda *_args, **_kwargs: b''))
  monkeypatch.setattr(plannerd, 'get_gps_location_service', lambda _params: 'gpsLocation')
  monkeypatch.setattr(plannerd.messaging, 'log_from_bytes', lambda *_args: SimpleNamespace(brand='test'))
  monkeypatch.setattr(plannerd.messaging, 'SubMaster', FakeSubMaster)
  monkeypatch.setattr(plannerd.messaging, 'PubMaster', FakePubMaster)
  monkeypatch.setattr(plannerd, 'LongitudinalPlanner', FakePlanner)
  monkeypatch.setattr(plannerd, 'LaneDepartureWarning', lambda: SimpleNamespace(left=False, right=False, update=lambda *_: None))
  monkeypatch.setattr(plannerd, '_planner_validity_diag', lambda _sm: {
    'outputs': dict.fromkeys(plannerd.PLANNER_VALIDITY_CHECKS, True), 'failed': {},
  })

  with pytest.raises(StopIteration):
    plannerd.main()

  sm = submasters[0]
  planner = planners[0]
  assert sm.poll == 'carState'
  assert planner.sla.events == list(range(6))
  assert planner.update_count == planner.publish_count == 2


def test_lane_departure_warning_blinker_suppression_expires_at_five_seconds():
  desire = [0.0] * 5
  desire[int(log.Desire.laneChangeLeft)] = 1.0
  desire[int(log.Desire.laneChangeRight)] = 1.0
  model = SimpleNamespace(
    meta=SimpleNamespace(desirePrediction=desire),
    laneLineProbs=[0.0, 1.0, 1.0, 0.0],
    laneLines=[SimpleNamespace(y=[0.0]), SimpleNamespace(y=[-1.0]), SimpleNamespace(y=[1.0])],
  )
  car_state = SimpleNamespace(leftBlinker=True, rightBlinker=False, vEgo=40.0)
  car_control = SimpleNamespace(latActive=False)
  ldw = LaneDepartureWarning()

  ldw.update(100, model, car_state, car_control)
  assert not ldw.warning

  car_state.leftBlinker = False
  ldw.update(100 + int(5.0 / DT_CTRL) - 1, model, car_state, car_control)
  assert not ldw.warning

  ldw.update(100 + int(5.0 / DT_CTRL), model, car_state, car_control)
  assert ldw.warning


REQUIRED_LONGITUDINAL_PLAN_SERVICES = (
  'carControl', 'carState', 'carStateSP', 'controlsState', 'liveParameters',
  'modelV2', 'radarState', 'selfdriveState', 'selfdriveStateSP',
)


class _ValiditySubMaster:
  def __init__(self, invalid_service):
    services = set(plannerd.PLANNER_VALIDITY_SERVICES) | {'livePose', 'liveCalibration'}
    self.services = sorted(services)
    self.valid = {service: service != invalid_service for service in services}
    self.alive = dict.fromkeys(services, True)
    self.freq_ok = dict.fromkeys(services, True)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)
    self.logMonoTime = dict.fromkeys(services, 1)
    self.frame = 1
    tracker = SimpleNamespace(
      avg_dt=SimpleNamespace(count=1, get_average=lambda: 0.01),
      recent_avg_dt=SimpleNamespace(count=1, get_average=lambda: 0.01),
      min_freq=1.0,
      max_freq=100.0,
    )
    self.freq_tracker = dict.fromkeys(services, tracker)
    self.data = {'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=False))}

  def all_checks(self, service_list=None):
    return all(self.valid[service] and self.alive[service] and self.freq_ok[service]
               for service in (service_list or self.services))

  def _check_avg_freq(self, _service):
    return False

  def __getitem__(self, service):
    return self.data[service]


@pytest.mark.parametrize('invalid_service', REQUIRED_LONGITUDINAL_PLAN_SERVICES + ('livePose', 'liveCalibration'))
def test_longitudinal_plan_validity_scope_and_publish(invalid_service):
  sm = _ValiditySubMaster(invalid_service)
  diag = plannerd._planner_validity_diag(sm)
  required = invalid_service in REQUIRED_LONGITUDINAL_PLAN_SERVICES
  assert diag['outputs']['longitudinalPlan'] is not required
  assert diag['outputs']['longitudinalPlanSP'] is not (invalid_service in ('carState', 'controlsState'))
  assert diag['outputs']['driverAssistance'] is not (invalid_service in ('carState', 'carControl', 'modelV2', 'liveParameters'))

  planner = object.__new__(LongitudinalPlanner)
  planner.v_desired_trajectory = np.array([0.0])
  planner.a_desired_trajectory = np.array([0.0])
  planner.j_desired_trajectory = np.array([0.0])
  planner.mpc = SimpleNamespace(solve_time=0.0, source=0)
  planner.fcw = False
  planner.output_a_target = 0.0
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.publish_longitudinal_plan_sp = lambda _sm, _pm: None

  class FakePubMaster:
    def __init__(self):
      self.sent = []

    def send(self, service, msg):
      self.sent.append((service, msg))

  pm = FakePubMaster()
  planner.publish(sm, pm)
  assert pm.sent[0][1].valid is not required
