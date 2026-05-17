from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import (
  COMFORT_RELAX_ACCEL_MIN,
  CUSTOM_V2_INTENTS,
  CustomLongitudinalStackV2,
  CustomV2Scene,
  LEAD_PULLAWAY_ACCEL_MAX,
  MAP_ONLY_CAUTION_ACCEL_MIN,
  MPH_TO_MS,
  NO_LEAD_LAUNCH_ACCEL_MAX,
  dynamic_cruise_overspeed_leeway,
  lead_evidence_releases_stop,
  no_lead_stop_clear,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput, validate_stack_output


def make_output(a_target=0.0, should_stop=False):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=should_stop,
    has_lead=False,
    source="cruise",
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(0.0 for _ in range(CONTROL_N)),
    accels=tuple(a_target for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
  )


def test_custom_v2_intent_taxonomy_is_complete():
  assert CUSTOM_V2_INTENTS == (
    "driver_cruise",
    "lead_follow",
    "stop_approach",
    "launch",
    "speed_policy",
    "curve_policy",
    "map_caution",
    "comfort_relax",
    "safety_cap",
  )


def test_no_lead_launch_requires_stop_clear():
  clear_scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=30.0, model_desired_accel=0.0)
  blocked_scene = CustomV2Scene(v_ego=0.2, v_cruise=5.0, model_stop_distance=10.0, model_desired_accel=0.0)
  stack = CustomLongitudinalStackV2()

  clear_output = stack.update(make_output(), clear_scene, accel_limits=(-2.0, 2.0))
  blocked_output = stack.update(make_output(), blocked_scene, accel_limits=(-2.0, 2.0))

  assert no_lead_stop_clear(clear_scene)
  assert not no_lead_stop_clear(blocked_scene)
  assert clear_output.a_target == NO_LEAD_LAUNCH_ACCEL_MAX
  assert clear_output.debug["custom_v2_selected_intent"] == "launch"
  assert blocked_output.a_target == 0.0
  assert "model_stop_not_clear" in blocked_output.debug["custom_v2_rejected_reasons"]


def test_confirmed_lead_pullaway_releases_lagging_stop_prediction():
  scene = CustomV2Scene(
    v_ego=0.3,
    v_cruise=6.0,
    has_lead=True,
    lead_v=0.2,
    lead_confirmed_pullaway=True,
    stop_threat=True,
    model_should_stop=True,
    model_stop_distance=5.0,
    model_desired_accel=-1.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(-0.4, should_stop=True), scene, accel_limits=(-2.0, 2.0))

  assert lead_evidence_releases_stop(scene)
  assert output.a_target == LEAD_PULLAWAY_ACCEL_MAX
  assert not output.should_stop
  assert output.debug["custom_v2_selected_reason"] == "confirmed_lead_pullaway"
  assert "lead_pullaway_release" in output.debug["custom_v2_rejected_reasons"]


def test_speed_policy_is_coast_biased_for_speed_reductions():
  scene = CustomV2Scene(
    v_ego=22.0,
    v_cruise=24.0,
    accel_coast=-0.25,
    speed_limit_active=True,
    speed_limit_v_target=18.0,
    speed_limit_a_target=-1.2,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.8), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.25
  assert output.debug["custom_v2_selected_intent"] == "speed_policy"
  assert output.debug["custom_v2_selected_reason"] == "coast_biased_speed_reduction"


def test_map_only_caution_is_preparatory_not_full_stop_authority():
  scene = CustomV2Scene(
    v_ego=12.0,
    v_cruise=18.0,
    map_caution_active=True,
    map_caution_confirmed=False,
    map_caution_a_target=-1.0,
  )

  output = CustomLongitudinalStackV2().update(make_output(0.5), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == MAP_ONLY_CAUTION_ACCEL_MIN
  assert not output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "map_caution"
  assert output.debug["custom_v2_selected_reason"] == "map_only_preparation"


def test_comfort_relax_only_softens_advisory_decel():
  scene = CustomV2Scene(v_ego=12.0, v_cruise=18.0, curve_active=True, curve_a_target=-1.0)

  output = CustomLongitudinalStackV2().update(make_output(0.4), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == COMFORT_RELAX_ACCEL_MIN
  assert output.debug["custom_v2_selected_intent"] == "comfort_relax"
  assert output.debug["custom_v2_selected_reason"] == "clear_margin_advisory_softening"


def test_dynamic_plain_cruise_leeway_allows_downhill_coasting():
  scene = CustomV2Scene(v_ego=20.0 + 6.0 * MPH_TO_MS, v_cruise=20.0, accel_coast=0.25)

  output = CustomLongitudinalStackV2().update(make_output(-0.8), scene, accel_limits=(-2.0, 2.0))

  assert dynamic_cruise_overspeed_leeway(scene.accel_coast) == 7.0 * MPH_TO_MS
  assert output.a_target == 0.0
  assert output.debug["custom_v2_selected_reason"] == "dynamic_overspeed_coast_leeway"


def test_force_slow_decel_is_safety_cap():
  scene = CustomV2Scene(v_ego=10.0, v_cruise=20.0, force_slow_decel=True)

  output = CustomLongitudinalStackV2().update(make_output(0.5), scene, accel_limits=(-2.0, 2.0))

  assert output.a_target == -0.2
  assert output.should_stop
  assert output.debug["custom_v2_selected_intent"] == "safety_cap"
  assert validate_stack_output(output, accel_limits=(-2.0, 2.0)).valid
