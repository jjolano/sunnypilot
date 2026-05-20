from cereal import custom
from openpilot.sunnypilot.selfdrive.selfdrived.events import AlertSize, AlertStatus, AudibleAlert, ET, EVENTS_SP, Priority, VisualAlert


def test_custom_longitudinal_stack_fault_immediate_disable_alert():
  alert = EVENTS_SP[custom.OnroadEventSP.EventName.customLongitudinalStackFault][ET.IMMEDIATE_DISABLE]

  assert alert.alert_text_2 == "Custom Longitudinal Stack Fault"
  assert alert.alert_status == AlertStatus.critical
  assert alert.alert_size == AlertSize.full
  assert alert.priority == Priority.HIGHEST


def test_custom_longitudinal_stack_fault_event_schema_exists():
  event = custom.OnroadEventSP.Event.new_message()
  event.name = custom.OnroadEventSP.EventName.customLongitudinalStackFault
  event.immediateDisable = True

  assert event.name == custom.OnroadEventSP.EventName.customLongitudinalStackFault
  assert event.immediateDisable
  assert event.name.raw >= 0


def test_low_traction_warning_is_visual_only():
  alert = EVENTS_SP[custom.OnroadEventSP.EventName.lowTraction][ET.WARNING]

  assert alert.alert_text_1 == "Low Traction Detected"
  assert alert.alert_text_2 == "Driving softened"
  assert alert.alert_status == AlertStatus.userPrompt
  assert alert.alert_size == AlertSize.small
  assert alert.priority == Priority.LOW
  assert alert.visual_alert == VisualAlert.none
  assert alert.audible_alert == AudibleAlert.none


def test_low_traction_event_schema_exists():
  event = custom.OnroadEventSP.Event.new_message()
  event.name = custom.OnroadEventSP.EventName.lowTraction
  event.warning = True

  assert event.name == custom.OnroadEventSP.EventName.lowTraction
  assert event.warning
  assert event.name.raw >= 0
