import random
import numpy as np
import pytest

from cereal import log, messaging
from openpilot.selfdrive.locationd.helpers import PoseCalibrator
from openpilot.selfdrive.locationd.paramsd import VehicleParamsLearner, retrieve_initial_vehicle_params, migrate_cached_vehicle_params_if_needed
from openpilot.selfdrive.locationd.models.car_kf import CarKalman, States
from openpilot.selfdrive.locationd.models.constants import ObservationKind
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader


def get_random_live_parameters(CP):
  msg = messaging.new_message("liveParameters")
  msg.liveParameters.steerRatio = (random.random() + 0.5) * CP.steerRatio
  msg.liveParameters.stiffnessFactor = random.random()
  msg.liveParameters.angleOffsetAverageDeg = random.random()
  msg.liveParameters.debugFilterState.std = [random.random() for _ in range(CarKalman.P_initial.shape[0])]
  return msg


class TestParamsd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes(), block=True)
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes(), block=True)

    migrate_cached_vehicle_params_if_needed(params) # this is not tested here but should not mess anything up or throw an error
    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    np.testing.assert_equal(p_init.shape, CarKalman.P_initial.shape)
    np.testing.assert_allclose(np.sqrt(np.diagonal(p_init)), msg.liveParameters.debugFilterState.std)

  def test_observed_roll_uses_calibrated_pose_frame(self):
    class FakeKF:
      def __init__(self):
        self.x = CarKalman.initial_x.copy()
        self.observations = []

      def predict_and_observe(self, _t, kind, value, *_args):
        self.observations.append((kind, np.array(value, copy=True)))

    learner = VehicleParamsLearner.__new__(VehicleParamsLearner)
    learner.calibrator = PoseCalibrator()
    learner.observed_yaw_rate = 0.0
    learner.observed_roll = 0.0
    learner.active = True
    learner.kf = FakeKF()
    learner.kf.x[States.STIFFNESS] = 1.0
    learner.kf.x[States.STEER_RATIO] = 15.0

    mount_roll = 0.05
    learner.handle_log(0.0, "liveCalibration", log.LiveCalibrationData(
      rpyCalib=[mount_roll, 0.0, 0.0],
      calStatus=log.LiveCalibrationData.Status.calibrated,
    ))
    learner.handle_log(1.0, "livePose", log.LivePose(
      timestamp=int(1e9),
      orientationNED=log.LivePose.XYZMeasurement(x=-mount_roll, xStd=0.001, valid=True),
      angularVelocityDevice=log.LivePose.XYZMeasurement(z=0.1, zStd=0.001, valid=True),
      inputsOK=True,
      posenetOK=True,
      sensorsOK=True,
    ))

    observed_rolls = [float(value.item()) for kind, value in learner.kf.observations if kind == ObservationKind.ROAD_ROLL]
    assert observed_rolls[-1] == pytest.approx(0.0, abs=1e-4)

  # TODO Remove this test after the support for old format is removed
  def test_read_saved_params_old_format(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParameters", msg.liveParameters.to_dict(), block=True)
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes(), block=True)
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    sr, sf, offset, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    assert params.get("LiveParametersV2") is not None

  def test_read_saved_params_corrupted_old_format(self):
    params = Params()
    params.put("LiveParameters", {}, block=True)
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    assert params.get("LiveParameters") is None
    assert params.get("LiveParametersV2") is None
