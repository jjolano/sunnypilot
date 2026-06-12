from openpilot.common.params import Params


def test_controls_profile_param_key_registered():
  assert bool(Params().check_key("ControlsProfile")) is True


def test_lateral_demand_stack_param_key_registered():
  assert bool(Params().check_key("LateralDemandStack")) is True


def test_torque_control_tune_param_key_registered():
  assert bool(Params().check_key("TorqueControlTune")) is True


def test_controls_profile_migration_version_param_key_registered():
  assert bool(Params().check_key("ControlsProfileMigrationVersion")) is True
