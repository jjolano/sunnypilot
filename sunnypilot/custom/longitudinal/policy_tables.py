"""Personality tables and policy constants (personality-as-data).

Values are the legacy custom-2.0 constants (docs/legacy/tuned-constants.yaml /
custom_v2.py). The ADR's "personality as data" decision: standard anchors comfort/progress,
other personalities scale comfort/progress only — never safety caps.
"""
from __future__ import annotations

from enum import Enum

MPH_TO_MS = 0.44704


class Personality(Enum):
  RELAXED = "relaxed"
  STANDARD = "standard"
  AGGRESSIVE = "aggressive"

  @classmethod
  def from_value(cls, value: object, default: "Personality" = None) -> "Personality":
    default = default if default is not None else cls.STANDARD
    if isinstance(value, cls):
      return value
    if isinstance(value, bytes):
      value = value.decode(errors="ignore")
    text = str(value if value is not None else "").strip().lower()
    for p in cls:
      if text in (p.value, p.name.lower()):
        return p
    # legacy cereal int encoding: relaxed=0, standard=1, aggressive=2
    return {"0": cls.RELAXED, "1": cls.STANDARD, "2": cls.AGGRESSIVE}.get(text, default)


# Progress / comfort scale with personality (never safety caps).
NO_LEAD_LAUNCH_ACCEL_MAX = {
  Personality.RELAXED: 1.10,
  Personality.STANDARD: 1.35,
  Personality.AGGRESSIVE: 1.55,
}
STOP_APPROACH_COMFORT_DECEL = {
  Personality.RELAXED: -0.30,
  Personality.STANDARD: -0.38,
  Personality.AGGRESSIVE: -0.45,
}

# Policy constants (personality-independent).
NO_LEAD_LAUNCH_MAX_V_EGO = 3.0
PROGRESS_CRUISE_SPEED_MARGIN = 0.2
# Lead pull-away launch (off-the-line behind an opening lead, e.g. stop-and-go): below this ego
# speed and when the lead is genuinely opening, key the pull-away on a lead-tracking launch accel
# (match the lead's speed over LEAD_LAUNCH_TAU, capped by the personality launch accel and the
# speedup guard) instead of waiting for a far 25 m gap excess. Gentle for a crawling lead, brisk
# when it genuinely goes — never a fixed lurch.
LEAD_LAUNCH_MAX_V_EGO = 8.0
LEAD_LAUNCH_TAU = 1.0               # s; time constant to match the lead's speed off the line
LEAD_PULLAWAY_MIN_V_LEAD = 0.2     # m/s; lead considered moving (mirror of the close-stop gate)
LEAD_PULLAWAY_MIN_OPENING = 0.15   # m/s; lead considered opening (mirror of the close-stop gate)
# Close, low-speed stop-and-go is damped so we don't chase every lead twitch in an accordion.
# A stronger opening or faster lead is treated as a normal launch breakout instead.
LEAD_CRAWL_MAX_V_EGO = 5.0
LEAD_CRAWL_MAX_V_LEAD = 5.0
LEAD_CRAWL_MAX_D_REL = 25.0
# Route 282: the lead was clearly pulling away at 0.7-0.95 m/s opening while the old
# 1.0 threshold kept launch in the accordion-damped crawl branch until driver override.
LEAD_CRAWL_BREAKOUT_MIN_OPENING = 0.7
LEAD_CRAWL_LAUNCH_TAU = 2.5
# Route 261: 0.55 capped the first pull-away frames of an engaged launch while the driver
# launches at ~1.0 immediately; the accordion case stays damped by the tau'd gentle branch.
LEAD_CRAWL_ACCEL_MAX = 0.8
NO_LEAD_STOP_CLEAR_DISTANCE = 20.0
NO_LEAD_STOP_CLEAR_ACCEL_MIN = -0.5
MAP_ONLY_CAUTION_ACCEL_MIN = -0.3
COMFORT_RELAX_ACCEL_MIN = -0.5
CRUISE_LEEWAY_MIN = 5.0 * MPH_TO_MS
CRUISE_LEEWAY_MAX = 10.0 * MPH_TO_MS
CRUISE_LEEWAY_HIGHWAY_MAX = 12.0 * MPH_TO_MS
CRUISE_LEEWAY_HIGHWAY_MIN_V_EGO = 28.0  # ~63 mph; highway-only expansion
CRUISE_LEEWAY_DOWNHILL_ACCEL = 0.25
CRUISE_LEEWAY_RECOVERY = 3.5 * MPH_TO_MS
FLAT_COAST_BASELINE = -0.3
GRADE_COMPENSATION_MAX_MS2 = 0.15
GRADE_FLAT_BAND_HALF_WIDTH = 0.35
STOP_APPROACH_DECEL_MIN = -1.5
# Final low-speed landing floor. The normal stop-approach floor remains available
# above walking speed and for genuinely hard-stop kinematics; this only prevents a
# trusted stop approach from carrying the full -1.5 m/s^2 floor into the last few
# mph of a routine landing.
STOP_LANDING_SOFTEN_MAX_V_EGO = 2.5
STOP_LANDING_DECEL_MIN = -0.85
EXCESS_GAP_MIN = 1.0


def launch_accel_max(personality: Personality) -> float:
  return NO_LEAD_LAUNCH_ACCEL_MAX[personality]


def stop_approach_comfort_decel(personality: Personality) -> float:
  return STOP_APPROACH_COMFORT_DECEL[personality]
