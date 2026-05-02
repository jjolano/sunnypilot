FCW_SUPPRESS_COMMAND_ACCEL = -2.0
FCW_SUPPRESS_MEASURED_ACCEL = -1.5
FCW_SUPPRESS_LEAD_PROB = 0.9
FCW_SUPPRESS_MAX_LEAD_DISTANCE = 60.0


def get_fcw_active_lead(longitudinal_plan_source, lead0_source, lead1_source, radar_state):
  if longitudinal_plan_source == lead0_source and radar_state.leadOne.status:
    return radar_state.leadOne
  if longitudinal_plan_source == lead1_source and radar_state.leadTwo.status:
    return radar_state.leadTwo
  return None


def should_suppress_model_fcw(
  enabled, openpilot_longitudinal_control, a_ego, a_target, longitudinal_plan_has_lead, longitudinal_plan_source, lead0_source, lead1_source, radar_state
):
  if not (enabled and openpilot_longitudinal_control):
    return False
  if not longitudinal_plan_has_lead:
    return False

  lead = get_fcw_active_lead(longitudinal_plan_source, lead0_source, lead1_source, radar_state)
  if lead is None or lead.modelProb < FCW_SUPPRESS_LEAD_PROB or lead.dRel > FCW_SUPPRESS_MAX_LEAD_DISTANCE:
    return False

  return a_target <= FCW_SUPPRESS_COMMAND_ACCEL and a_ego <= FCW_SUPPRESS_MEASURED_ACCEL
