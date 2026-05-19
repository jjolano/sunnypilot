from cereal import custom
from openpilot.sunnypilot.selfdrive.selfdrived.events import AlertSize, AlertStatus, AudibleAlert, ET, EVENTS_SP, Priority


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
