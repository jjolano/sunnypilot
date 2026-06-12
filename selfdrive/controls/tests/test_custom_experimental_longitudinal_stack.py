from openpilot.selfdrive.controls.lib.longitudinal_stacks import (
  CUSTOM_EXPERIMENTAL,
  CUSTOM_V2,
  CustomExperimentalLongitudinalStack,
  CustomLongitudinalStackV2,
  make_custom_longitudinal_stack,
)


def test_factory_returns_experimental_instance():
  s = make_custom_longitudinal_stack(CUSTOM_EXPERIMENTAL)
  assert isinstance(s, CustomExperimentalLongitudinalStack)
  assert s.NAME == CUSTOM_EXPERIMENTAL
  assert s.VERSION == "experimental"


def test_factory_returns_v2_instance():
  s = make_custom_longitudinal_stack(CUSTOM_V2)
  assert isinstance(s, CustomLongitudinalStackV2)
  assert not isinstance(s, CustomExperimentalLongitudinalStack)


def test_experimental_stage_is_v2_baseline():
  s = CustomExperimentalLongitudinalStack()
  assert s.stage == "v2_baseline"


def test_experimental_stack_inherits_v2_behavior():
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomLongitudinalStackV2
  s = CustomExperimentalLongitudinalStack()
  assert isinstance(s, CustomLongitudinalStackV2)
