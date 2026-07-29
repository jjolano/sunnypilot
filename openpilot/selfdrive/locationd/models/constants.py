import os

GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))


def require_generated_ekf(generated_dir: str, name: str) -> None:
  """Fail loudly if a generated EKF model library is missing, before rednose dlopens it.

  rednose's ekf_load.cc does `assert(handle)` on the dlopen result, so an unbuilt model
  library takes the whole process down with SIGABRT rather than raising. Under test that
  means a ~126 MB core dump per crash and a parent left waiting on a handshake that will
  never arrive — 30 of 37 cores found in this tree on 2026-07-29 carried exactly that
  assert. `generated/` is gitignored build output, so a fresh checkout or worktree hits
  this every time until scons has run.

  A missing build artifact is a setup problem, and it should read like one.
  """
  path = os.path.join(generated_dir, f"lib{name}.so")
  if not os.path.exists(path):
    raise FileNotFoundError(
      f"generated EKF model library missing: {path}\n"
      f"This is a build/setup problem, not a runtime fault. Build it with:\n"
      f"    scons selfdrive/locationd\n"
      f"(Without this check rednose would dlopen the missing file and abort the process.)"
    )


class ObservationKind:
  UNKNOWN = 0
  NO_OBSERVATION = 1
  GPS_NED = 2
  ODOMETRIC_SPEED = 3
  PHONE_GYRO = 4
  GPS_VEL = 5
  PSEUDORANGE_GPS = 6
  PSEUDORANGE_RATE_GPS = 7
  SPEED = 8
  NO_ROT = 9
  PHONE_ACCEL = 10
  ORB_POINT = 11
  ECEF_POS = 12
  CAMERA_ODO_TRANSLATION = 13
  CAMERA_ODO_ROTATION = 14
  ORB_FEATURES = 15
  MSCKF_TEST = 16
  FEATURE_TRACK_TEST = 17
  LANE_PT = 18
  IMU_FRAME = 19
  PSEUDORANGE_GLONASS = 20
  PSEUDORANGE_RATE_GLONASS = 21
  PSEUDORANGE = 22
  PSEUDORANGE_RATE = 23
  ECEF_VEL = 35
  ECEF_ORIENTATION_FROM_GPS = 32
  NO_ACCEL = 33
  ORB_FEATURES_WIDE = 34

  ROAD_FRAME_XY_SPEED = 24  # (x, y) [m/s]
  ROAD_FRAME_YAW_RATE = 25  # [rad/s]
  STEER_ANGLE = 26  # [rad]
  ANGLE_OFFSET_FAST = 27  # [rad]
  STIFFNESS = 28  # [-]
  STEER_RATIO = 29  # [-]
  ROAD_FRAME_X_SPEED = 30  # (x) [m/s]
  ROAD_ROLL = 31  # [rad]

  names = [
    'Unknown',
    'No observation',
    'GPS NED',
    'Odometric speed',
    'Phone gyro',
    'GPS velocity',
    'GPS pseudorange',
    'GPS pseudorange rate',
    'Speed',
    'No rotation',
    'Phone acceleration',
    'ORB point',
    'ECEF pos',
    'camera odometric translation',
    'camera odometric rotation',
    'ORB features',
    'MSCKF test',
    'Feature track test',
    'Lane ecef point',
    'imu frame eulers',
    'GLONASS pseudorange',
    'GLONASS pseudorange rate',
    'pseudorange',
    'pseudorange rate',

    'Road Frame x,y speed',
    'Road Frame yaw rate',
    'Steer Angle',
    'Fast Angle Offset',
    'Stiffness',
    'Steer Ratio',
    'Road Frame x speed',
    'Road Roll',
    'ECEF orientation from GPS',
    'NO accel',
    'ORB features wide camera',
    'ECEF_VEL',
  ]

  @classmethod
  def to_string(cls, kind):
    return cls.names[kind]


SAT_OBS = [ObservationKind.PSEUDORANGE_GPS,
           ObservationKind.PSEUDORANGE_RATE_GPS,
           ObservationKind.PSEUDORANGE_GLONASS,
           ObservationKind.PSEUDORANGE_RATE_GLONASS]
