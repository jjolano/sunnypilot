from cereal import custom
from openpilot.sunnypilot.selfdrive.selfdrived.events import AlertSize, AlertStatus, AudibleAlert, ET, EVENTS_SP, Priority


def test_custom_longitudinal_fallback_warning_alert():
  alert = EVENTS_SP[custom.OnroadEventSP.EventName.customLongitudinalFallback][ET.WARNING]

  assert alert.alert_text_1 == "Custom Longitudinal Fallback"
  assert alert.alert_text_2 == "Using sunnypilot longitudinal"
  assert alert.alert_status == AlertStatus.userPrompt
  assert alert.alert_size == AlertSize.mid
  assert alert.priority == Priority.MID
  assert alert.audible_alert == AudibleAlert.none


def test_custom_longitudinal_fallback_event_schema_exists():
  event = custom.OnroadEventSP.Event.new_message()
  event.name = custom.OnroadEventSP.EventName.customLongitudinalFallback
  event.warning = True

  assert event.name == custom.OnroadEventSP.EventName.customLongitudinalFallback
  assert event.warning
  assert event.name.raw >= 0
