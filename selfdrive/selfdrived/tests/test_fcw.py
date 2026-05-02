from types import SimpleNamespace

from openpilot.selfdrive.selfdrived.fcw import get_fcw_active_lead, should_suppress_model_fcw


LEAD0_SOURCE = "lead0"
LEAD1_SOURCE = "lead1"
CRUISE_SOURCE = "cruise"


def make_lead(*, status=False, model_prob=0.0, d_rel=20.0):
  return SimpleNamespace(status=status, modelProb=model_prob, dRel=d_rel)


def make_radar_state(*, lead_one=None, lead_two=None):
  return SimpleNamespace(
    leadOne=lead_one or make_lead(),
    leadTwo=lead_two or make_lead(),
  )


def make_longitudinal_plan(*, source=CRUISE_SOURCE, has_lead=False, a_target=0.0):
  return SimpleNamespace(
    longitudinalPlanSource=source,
    hasLead=has_lead,
    aTarget=a_target,
  )


def make_car_state(*, a_ego=0.0):
  return SimpleNamespace(aEgo=a_ego)


def test_get_fcw_active_lead_matches_selected_source():
  radar_state = make_radar_state(
    lead_one=make_lead(status=True, model_prob=0.95),
    lead_two=make_lead(status=True, model_prob=0.99),
  )

  assert get_fcw_active_lead(LEAD0_SOURCE, LEAD0_SOURCE, LEAD1_SOURCE, radar_state) is radar_state.leadOne
  assert get_fcw_active_lead(LEAD1_SOURCE, LEAD0_SOURCE, LEAD1_SOURCE, radar_state) is radar_state.leadTwo
  assert get_fcw_active_lead(CRUISE_SOURCE, LEAD0_SOURCE, LEAD1_SOURCE, radar_state) is None


def test_model_fcw_is_suppressed_when_op_is_already_braking_hard_on_confirmed_lead():
  radar_state = make_radar_state(lead_one=make_lead(status=True, model_prob=0.98))
  longitudinal_plan = make_longitudinal_plan(source=LEAD0_SOURCE, has_lead=True, a_target=-2.5)
  car_state = make_car_state(a_ego=-2.0)

  assert should_suppress_model_fcw(
    True,
    True,
    car_state.aEgo,
    longitudinal_plan.aTarget,
    longitudinal_plan.hasLead,
    longitudinal_plan.longitudinalPlanSource,
    LEAD0_SOURCE,
    LEAD1_SOURCE,
    radar_state,
  )


def test_model_fcw_is_not_suppressed_when_plan_has_no_lead():
  radar_state = make_radar_state(lead_two=make_lead(status=True, model_prob=0.98))
  longitudinal_plan = make_longitudinal_plan(source=LEAD1_SOURCE, has_lead=False, a_target=-2.5)
  car_state = make_car_state(a_ego=-2.0)

  assert not should_suppress_model_fcw(
    True,
    True,
    car_state.aEgo,
    longitudinal_plan.aTarget,
    longitudinal_plan.hasLead,
    longitudinal_plan.longitudinalPlanSource,
    LEAD0_SOURCE,
    LEAD1_SOURCE,
    radar_state,
  )


def test_model_fcw_is_not_suppressed_for_far_selected_lead():
  radar_state = make_radar_state(lead_one=make_lead(status=True, model_prob=0.98, d_rel=80.0))
  longitudinal_plan = make_longitudinal_plan(source=LEAD0_SOURCE, has_lead=True, a_target=-2.5)
  car_state = make_car_state(a_ego=-2.0)

  assert not should_suppress_model_fcw(
    True,
    True,
    car_state.aEgo,
    longitudinal_plan.aTarget,
    longitudinal_plan.hasLead,
    longitudinal_plan.longitudinalPlanSource,
    LEAD0_SOURCE,
    LEAD1_SOURCE,
    radar_state,
  )


def test_model_fcw_is_not_suppressed_without_a_confirmed_lead():
  radar_state = make_radar_state(lead_one=make_lead(status=True, model_prob=0.6))
  longitudinal_plan = make_longitudinal_plan(source=LEAD0_SOURCE, has_lead=True, a_target=-2.5)
  car_state = make_car_state(a_ego=-2.0)

  assert not should_suppress_model_fcw(
    True,
    True,
    car_state.aEgo,
    longitudinal_plan.aTarget,
    longitudinal_plan.hasLead,
    longitudinal_plan.longitudinalPlanSource,
    LEAD0_SOURCE,
    LEAD1_SOURCE,
    radar_state,
  )


def test_model_fcw_is_not_suppressed_for_cruise_source_or_mild_braking():
  radar_state = make_radar_state(lead_one=make_lead(status=True, model_prob=0.99))
  strong_plan = make_longitudinal_plan(source=CRUISE_SOURCE, has_lead=True, a_target=-2.5)
  mild_plan = make_longitudinal_plan(source=LEAD0_SOURCE, has_lead=True, a_target=-1.0)
  car_state = make_car_state(a_ego=-2.0)

  assert not should_suppress_model_fcw(
    True, True, car_state.aEgo, strong_plan.aTarget, strong_plan.hasLead, strong_plan.longitudinalPlanSource, LEAD0_SOURCE, LEAD1_SOURCE, radar_state
  )
  assert not should_suppress_model_fcw(
    True, True, car_state.aEgo, mild_plan.aTarget, mild_plan.hasLead, mild_plan.longitudinalPlanSource, LEAD0_SOURCE, LEAD1_SOURCE, radar_state
  )
