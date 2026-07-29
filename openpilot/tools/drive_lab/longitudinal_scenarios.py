from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from openpilot.cereal import log

from openpilot.tools.drive_lab.log_profile import LongitudinalProfile

MPH_TO_MS = 1.609 / 3.6
KPH_TO_MS = 1.0 / 3.6

REALISM_MODES = ("comfort", "emergency", "adversarial")
SCENARIO_PRESETS = (
    "fuzz", "udacity-acc", "openpilot-acc", "ncap-acc", "commonroad-acc",
    "nuscenes-acc", "iso15622-acc", "unr157-alks", "nhtsa-fcw", "cncap-ccrh",
    "iihs-acc",
)

LAUNCH_START_ORACLE_KINDS = (
  "lead_pullaway",
  "udacity_acc_green_light_launch",
  "openpilot_resume_from_stop",
)

_OPENPILOT_KIND_BY_TITLE = {
  "resume from a stop": "openpilot_resume_from_stop",
  "NaN recovery": "openpilot_nan_recovery",
  "cruising at 25 m/s while disabled": "openpilot_cruise_disabled",
  "stay stopped behind radar override lead": "openpilot_radar_override_stop",
  "approach stopped car at 25m/s, initial distance: 120m": "openpilot_stopped_lead_25ms_120m",
  "approach stopped car at 20m/s, initial distance 90m": "openpilot_stopped_lead_20ms_90m",
  "steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2": "openpilot_lead_decel_1ms2",
  "steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2": "openpilot_lead_decel_2ms2",
  "steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2": "openpilot_lead_decel_3ms2",
  "steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2": "openpilot_lead_decel_3plus_ms2",
  "approach stopped car at 20m/s, with prob_lead_values": "openpilot_stopped_lead_20ms_prob_lead",
  "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1": "openpilot_stopped_lead_20ms_throttle_downhill",
  "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1": "openpilot_stopped_lead_20ms_throttle_uphill",
  "approach slower cut-in car at 20m/s": "openpilot_slower_cut_in",
}

REGRESSION_ORACLE_KINDS = frozenset(_OPENPILOT_KIND_BY_TITLE.values())
UDACITY_REGRESSION_ORACLE_KINDS = frozenset({
  "udacity_acc_lead_decel_to_stop",
  "udacity_acc_lead_decel_to_stop_2ms2",
})


@dataclass(frozen=True)
class Scenario:
  mode: str
  kind: str
  title: str
  duration: float
  kwargs: dict[str, Any]
  oracle_profile: str = "comfort"
  provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class PresetRequest:
  preset: str
  mode: str = "comfort"
  seed: int = 1
  cases: int = 100
  profile: LongitudinalProfile | None = None
  e2e: bool = False
  force_decel: bool = False
  ncap_family: str | None = None
  ncap_sample: int | None = None
  commonroad_scenario: str | None = None
  nuscenes_scenario: str | None = None


def _validate_mode(mode: str) -> None:
  if mode not in REALISM_MODES:
    raise ValueError(f"unknown mode {mode!r}; expected one of {REALISM_MODES}")


def generate_scenarios(seed: int, cases: int, mode: str = "comfort", profile: LongitudinalProfile | None = None) -> list[Scenario]:
  _validate_mode(mode)
  rng = random.Random(seed)
  generators = [
    _stopped_lead_approach,
    _slower_cut_in,
    _lead_occlusion,
    _lead_pullaway,
    _cruise_coast,
  ]
  return [rng.choice(generators)(rng, idx, mode, profile) for idx in range(cases)]


def generate_udacity_acc_scenarios(mode: str = "comfort") -> list[Scenario]:
  """Return native Drive Lab scenarios inspired by Udacity's archived ACC challenge cases."""
  _validate_mode(mode)
  return [
    Scenario(mode, "udacity_acc_cruise_speed_step", "udacity acc inspired cruise speed step", 30.0, {
      "initial_speed": 40.0 * MPH_TO_MS,
      "lead_relevancy": False,
      "cruise_values": [40.0 * MPH_TO_MS, 40.0 * MPH_TO_MS, 50.0 * MPH_TO_MS, 50.0 * MPH_TO_MS],
      "breakpoints": [0.0, 10.0, 10.01, 30.0],
    }),
    Scenario(mode, "udacity_acc_cruise_speed_decrease", "udacity acc inspired cruise speed decrease", 30.0, {
      "initial_speed": 60.0 * MPH_TO_MS,
      "lead_relevancy": False,
      "cruise_values": [60.0 * MPH_TO_MS, 60.0 * MPH_TO_MS, 50.0 * MPH_TO_MS, 50.0 * MPH_TO_MS],
      "breakpoints": [0.0, 10.0, 10.01, 30.0],
    }),
    Scenario(mode, "udacity_acc_grade_change", "udacity acc inspired uphill grade change", 25.0, {
      "initial_speed": 20.0 * MPH_TO_MS,
      "lead_relevancy": False,
      "cruise_values": [20.0 * MPH_TO_MS] * 4,
      "pitch_values": [0.0, 0.0, 0.10, 0.10],
      "breakpoints": [0.0, 10.0, 11.0, 25.0],
    }),
    Scenario(mode, "udacity_acc_grade_downhill", "udacity acc inspired downhill grade change", 25.0, {
      "initial_speed": 20.0 * MPH_TO_MS,
      "lead_relevancy": False,
      "cruise_values": [20.0 * MPH_TO_MS] * 4,
      "pitch_values": [0.0, 0.0, -0.10, -0.10],
      "breakpoints": [0.0, 10.0, 11.0, 25.0],
    }),
    Scenario(mode, "udacity_acc_slower_lead", "udacity acc inspired slower lead approach", 30.0, {
      "initial_speed": 60.0 * MPH_TO_MS,
      "lead_relevancy": True,
      "initial_distance_lead": 100.0,
      "speed_lead_values": [40.0 * MPH_TO_MS, 40.0 * MPH_TO_MS],
      "prob_lead_values": [1.0, 1.0],
      "cruise_values": [60.0 * MPH_TO_MS, 60.0 * MPH_TO_MS],
      "breakpoints": [0.0, 30.0],
    }),
    Scenario(mode, "udacity_acc_stopped_lead", "udacity acc inspired stopped lead approach", 30.0, {
      "initial_speed": 40.0 * MPH_TO_MS,
      "lead_relevancy": True,
      "initial_distance_lead": 150.0,
      "speed_lead_values": [0.0, 0.0],
      "prob_lead_values": [1.0, 1.0],
      "cruise_values": [40.0 * MPH_TO_MS, 40.0 * MPH_TO_MS],
      "breakpoints": [0.0, 30.0],
    }),
    Scenario(mode, "udacity_acc_approach_from_stop", "udacity acc inspired approach from stop", 30.0, {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": 100.0,
      "speed_lead_values": [0.0, 0.0],
      "prob_lead_values": [1.0, 1.0],
      "cruise_values": [20.0, 20.0],
      "breakpoints": [0.0, 30.0],
    }),
    Scenario(mode, "udacity_acc_lead_decel_to_stop", "udacity acc inspired lead decel to stop", 50.0, {
      "initial_speed": 20.0,
      "lead_relevancy": True,
      "initial_distance_lead": 35.0,
      "speed_lead_values": [20.0 * MPH_TO_MS, 20.0 * MPH_TO_MS, 0.0, 0.0],
      "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
      "cruise_values": [20.0] * 4,
      "breakpoints": [0.0, 15.0, 35.0, 50.0],
    }, oracle_profile=_udacity_oracle_profile("udacity_acc_lead_decel_to_stop")),
    Scenario(mode, "udacity_acc_lead_decel_to_stop_2ms2", "udacity acc inspired lead decel to stop at 2 m/s2", 50.0, {
      "initial_speed": 20.0,
      "lead_relevancy": True,
      "initial_distance_lead": 35.0,
      "speed_lead_values": [20.0 * MPH_TO_MS, 20.0 * MPH_TO_MS, 0.0, 0.0],
      "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
      "cruise_values": [20.0] * 4,
      "breakpoints": [0.0, 15.0, 25.0, 50.0],
    }, oracle_profile=_udacity_oracle_profile("udacity_acc_lead_decel_to_stop_2ms2")),
    Scenario(mode, "udacity_acc_oscillating_lead", "udacity acc inspired oscillating lead speed", 25.0, {
      "initial_speed": 30.0,
      "lead_relevancy": True,
      "initial_distance_lead": 49.0,
      "speed_lead_values": [30.0, 30.0, 29.0, 31.0, 29.0, 31.0, 29.0],
      "prob_lead_values": [1.0] * 7,
      "cruise_values": [30.0] * 7,
      "breakpoints": [0.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0],
    }),
    Scenario(mode, "udacity_acc_stop_and_go_10mph", "udacity acc inspired stop and go at 10 mph", 70.0, {
      "initial_speed": 10.0 * MPH_TO_MS,
      "lead_relevancy": True,
      "initial_distance_lead": 20.0,
      "speed_lead_values": [10.0 * MPH_TO_MS, 0.0, 0.0, 10.0 * MPH_TO_MS, 0.0, 10.0 * MPH_TO_MS],
      "prob_lead_values": [1.0] * 6,
      "cruise_values": [10.0 * MPH_TO_MS] * 6,
      "breakpoints": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }),
    Scenario(mode, "udacity_acc_stop_and_go", "udacity acc inspired stop and go lead", 70.0, {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": 20.0,
      "speed_lead_values": [10.0 * MPH_TO_MS, 0.0, 0.0, 10.0 * MPH_TO_MS, 0.0, 0.0],
      "prob_lead_values": [1.0] * 6,
      "cruise_values": [30.0 * MPH_TO_MS] * 6,
      "breakpoints": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }),
    Scenario(mode, "udacity_acc_green_light_launch", "udacity acc inspired green light lead launch", 20.0, {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": 11.0,
      "speed_lead_values": [0.0, 0.0, 5.0, 12.0],
      "prob_lead_values": [1.0] * 4,
      "cruise_values": [15.0] * 4,
      "breakpoints": [0.0, 5.0, 8.0, 20.0],
    }),
    Scenario(mode, "udacity_acc_accel_while_lead_decel_mild", "udacity acc inspired accel while lead decel mild", 30.0, {
      "initial_speed": 10.0,
      "lead_relevancy": True,
      "initial_distance_lead": 10.0,
      "speed_lead_values": [20.0, 10.0],
      "prob_lead_values": [1.0, 1.0],
      "cruise_values": [20.0, 20.0],
      "breakpoints": [1.0, 11.0],
    }),
    Scenario(mode, "udacity_acc_accel_while_lead_decel_hard", "udacity acc inspired accel while lead decel hard", 30.0, {
      "initial_speed": 10.0,
      "lead_relevancy": True,
      "initial_distance_lead": 20.0,
      "speed_lead_values": [20.0, 0.0],
      "prob_lead_values": [1.0, 1.0],
      "cruise_values": [20.0, 20.0],
      "breakpoints": [1.0, 11.0],
    }),
  ]


def _openpilot_kind(title: str) -> str:
  if title in _OPENPILOT_KIND_BY_TITLE:
    return _OPENPILOT_KIND_BY_TITLE[title]
  slug = title.lower().replace(" ", "_").replace("/", "_").replace(",", "").replace(":", "")
  slug = slug.replace("__", "_").replace("+", "plus")
  return f"openpilot_{slug[:64]}"


def _openpilot_oracle_profile(kind: str) -> str:
  if kind in LAUNCH_START_ORACLE_KINDS:
    return "comfort"
  if kind in REGRESSION_ORACLE_KINDS:
    return "regression"
  return "comfort"


def _udacity_oracle_profile(kind: str) -> str:
  if kind in UDACITY_REGRESSION_ORACLE_KINDS:
    return "regression"
  return "comfort"


def _maneuver_to_kwargs(maneuver: Any) -> dict[str, Any]:
  kwargs: dict[str, Any] = {
    "initial_distance_lead": maneuver.distance_lead,
    "initial_speed": maneuver.speed,
    "lead_relevancy": bool(maneuver.lead_relevancy),
    "breakpoints": list(maneuver.breakpoints),
    "speed_lead_values": list(maneuver.speed_lead_values),
    "prob_lead_values": list(maneuver.prob_lead_values),
    "prob_throttle_values": list(maneuver.prob_throttle_values),
    "cruise_values": list(maneuver.cruise_values),
    "pitch_values": list(maneuver.pitch_values),
    "e2e": maneuver.e2e,
    "force_decel": maneuver.force_decel,
    "personality": maneuver.personality,
  }
  if maneuver.only_lead2:
    kwargs["only_lead2"] = True
  if maneuver.only_radar:
    kwargs["only_radar"] = True
  if maneuver.ensure_start:
    kwargs["ensure_start"] = True
  if maneuver.ensure_slowdown:
    kwargs["ensure_slowdown"] = True
  if not maneuver.enabled:
    kwargs["enabled"] = False
  return kwargs


def generate_openpilot_acc_scenarios(mode: str = "comfort", *, e2e: bool = False, force_decel: bool = False) -> list[Scenario]:
  from openpilot.selfdrive.test.longitudinal_maneuvers.scenarios import create_maneuvers

  _validate_mode(mode)
  maneuvers = create_maneuvers({"e2e": e2e, "force_decel": force_decel})
  scenarios: list[Scenario] = []
  for maneuver in maneuvers:
    kind = _openpilot_kind(maneuver.title)
    scenarios.append(Scenario(
      mode,
      kind,
      maneuver.title,
      maneuver.duration,
      _maneuver_to_kwargs(maneuver),
      oracle_profile=_openpilot_oracle_profile(kind),
    ))
  return scenarios


def generate_preset_scenarios(request: PresetRequest) -> list[Scenario]:
  if request.preset == "fuzz":
    return generate_scenarios(request.seed, request.cases, request.mode, request.profile)
  if request.preset == "udacity-acc":
    return generate_udacity_acc_scenarios(request.mode)
  if request.preset == "openpilot-acc":
    return generate_openpilot_acc_scenarios(request.mode, e2e=request.e2e, force_decel=request.force_decel)
  if request.preset == "ncap-acc":
    from openpilot.tools.drive_lab.ncap_acc_scenarios import generate_ncap_acc_scenarios
    return generate_ncap_acc_scenarios(
      request.mode,
      family=request.ncap_family,
      sample=request.ncap_sample,
      seed=request.seed,
    )
  if request.preset == "commonroad-acc":
    from openpilot.tools.drive_lab.commonroad_acc import generate_commonroad_acc_scenarios
    return generate_commonroad_acc_scenarios(request.mode, scenario_path=request.commonroad_scenario)
  if request.preset == "nuscenes-acc":
    from openpilot.tools.drive_lab.nuscenes_acc import generate_nuscenes_acc_scenarios
    if request.nuscenes_scenario is None:
      raise ValueError("nuscenes-acc preset requires --nuscenes-scenario")
    return generate_nuscenes_acc_scenarios(request.mode, scenario_path=request.nuscenes_scenario)
  if request.preset == "iso15622-acc":
    return generate_iso15622_acc_scenarios(request.mode)
  if request.preset == "unr157-alks":
    return generate_unr157_alks_scenarios(request.mode)
  if request.preset == "nhtsa-fcw":
    return generate_nhtsa_fcw_scenarios(request.mode)
  if request.preset == "cncap-ccrh":
    return generate_cncap_ccrh_scenarios(request.mode)
  if request.preset == "iihs-acc":
    return generate_iihs_acc_scenarios(request.mode)
  raise ValueError(f"unknown preset {request.preset!r}")


def scenario_maneuver_kwargs(scenario: Scenario) -> dict[str, Any]:
  kwargs = dict(scenario.kwargs)
  kwargs["personality"] = log.LongitudinalPersonality.standard
  if scenario.oracle_profile == "regression":
    return kwargs
  if scenario.kind in LAUNCH_START_ORACLE_KINDS:
    kwargs["ensure_start"] = False
  return kwargs


def _stopped_lead_approach(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  if mode == "comfort":
    v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 24.0), (5.0, 26.0))
    lead_decel = _sample_profile_range(rng, profile, "lead_decel", (1.5, 3.5), (1.0, 3.8))
    lead_distance = rng.uniform(max(55.0, v_ego * 3.5), max(110.0, v_ego * 6.0))
  elif mode == "emergency":
    v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 28.0), (5.0, 32.0))
    lead_decel = _sample_profile_range(rng, profile, "lead_decel", (3.5, 7.0), (3.0, 8.0))
    lead_distance = rng.uniform(max(35.0, v_ego * 2.2), max(90.0, v_ego * 4.5))
  else:
    v_ego = rng.uniform(8.0, 28.0)
    lead_decel = v_ego / rng.uniform(0.2, 2.5)
    lead_distance = rng.uniform(max(35.0, v_ego * 2.5), max(75.0, v_ego * 5.5))

  lead_stop_time = v_ego / lead_decel
  return Scenario(
    mode,
    "stopped_lead_approach",
    f"fuzz stopped lead approach #{idx}",
    max(rng.uniform(12.0, 28.0), lead_stop_time + 8.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(lead_distance, 3),
      "speed_lead_values": [round(v_ego, 3), 0.0, 0.0],
      "prob_lead_values": [1.0, 1.0, 1.0],
      "cruise_values": [round(max(v_ego, 12.0), 3)] * 3,
      "breakpoints": [0.0, round(lead_stop_time, 3), round(lead_stop_time + 0.01, 3)],
    },
  )


def _slower_cut_in(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (10.0, 25.0), (5.0, 30.0))
  cut_in_time = rng.uniform(1.0, 4.0)
  if mode == "comfort":
    closing_speed = _sample_profile_range(rng, profile, "closing_speed", (0.0, 4.0), (0.0, 4.5))
    v_lead = max(0.0, v_ego - closing_speed)
    closing_speed = max(0.0, v_ego - v_lead)
    detected_gap = rng.uniform(max(25.0, v_ego * 1.5, closing_speed * 4.0 + 10.0), max(65.0, v_ego * 3.0, closing_speed * 6.0 + 20.0))
    initial_distance_lead = detected_gap + closing_speed * cut_in_time
  elif mode == "emergency":
    closing_speed = _sample_profile_range(rng, profile, "closing_speed", (3.0, 9.0), (2.0, 11.0))
    v_lead = max(0.0, v_ego - closing_speed)
    closing_speed = max(0.0, v_ego - v_lead)
    detected_gap = rng.uniform(max(12.0, v_ego * 0.8, closing_speed * 1.5 + 6.0), max(45.0, v_ego * 1.8, closing_speed * 3.0 + 12.0))
    initial_distance_lead = detected_gap + closing_speed * cut_in_time
  else:
    v_lead = rng.uniform(max(0.0, v_ego - 8.0), v_ego + 1.0)
    closing_speed = max(0.0, v_ego - v_lead)
    detected_gap = rng.uniform(6.0, 30.0)
    initial_distance_lead = detected_gap + closing_speed * cut_in_time

  return Scenario(
    mode,
    "slower_cut_in",
    f"fuzz slower cut-in #{idx}",
    rng.uniform(8.0, 18.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [round(v_lead, 3)] * 3,
      "prob_lead_values": [0.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 3,
      "breakpoints": [0.0, round(cut_in_time, 3), round(cut_in_time + 0.01, 3)],
    },
  )


def _lead_occlusion(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (8.0, 22.0), (5.0, 28.0))
  occlusion_start = rng.uniform(2.0, 5.0)
  occlusion_end = occlusion_start + rng.uniform(0.2, 1.2)
  if mode == "comfort":
    initial_distance_lead = _sample_profile_range(
      rng, profile, "lead_gap", (max(35.0, v_ego * 2.0), max(80.0, v_ego * 4.0)), (max(30.0, v_ego * 1.5), max(90.0, v_ego * 5.0))
    )
    lead_delta = _sample_profile_range(rng, profile, "closing_speed", (0.0, 2.5), (0.0, 3.0))
  elif mode == "emergency":
    initial_distance_lead = _sample_profile_range(
      rng, profile, "lead_gap", (max(20.0, v_ego * 1.2), max(60.0, v_ego * 3.0)), (max(15.0, v_ego), max(70.0, v_ego * 4.0))
    )
    lead_delta = _sample_profile_range(rng, profile, "closing_speed", (1.0, 5.0), (0.5, 7.0))
  else:
    initial_distance_lead = rng.uniform(25.0, 70.0)
    lead_delta = rng.uniform(0.0, 4.0)

  return Scenario(
    mode,
    "lead_occlusion",
    f"fuzz lead occlusion #{idx}",
    rng.uniform(10.0, 20.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [round(max(0.0, v_ego - lead_delta), 3)] * 4,
      "prob_lead_values": [1.0, 1.0, 0.0, 1.0],
      "cruise_values": [round(v_ego, 3)] * 4,
      "breakpoints": [0.0, round(occlusion_start, 3), round(occlusion_end, 3), round(occlusion_end + 0.01, 3)],
    },
  )


def _lead_pullaway(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  pullaway_time = rng.uniform(3.0, 8.0)
  if mode == "comfort":
    v_lead = _sample_profile_range(rng, profile, "lead_pullaway_speed", (1.0, 3.5), (0.5, 4.0))
    initial_distance_lead = _sample_profile_range(rng, profile, "stopped_lead_gap", (4.5, 8.0), (3.5, 10.0))
    cruise = _sample_profile_range(rng, profile, "cruise_speed", (5.0, 12.0), (3.0, 15.0))
  elif mode == "emergency":
    v_lead = _sample_profile_range(rng, profile, "lead_pullaway_speed", (2.5, 5.0), (1.0, 6.0))
    initial_distance_lead = _sample_profile_range(rng, profile, "stopped_lead_gap", (4.0, 8.0), (3.0, 10.0))
    cruise = _sample_profile_range(rng, profile, "cruise_speed", (5.0, 15.0), (3.0, 20.0))
  else:
    v_lead = rng.uniform(1.0, 5.0)
    initial_distance_lead = rng.uniform(4.0, 8.0)
    cruise = rng.uniform(5.0, 15.0)

  return Scenario(
    mode,
    "lead_pullaway",
    f"fuzz lead pullaway #{idx}",
    rng.uniform(10.0, 18.0),
    {
      "initial_speed": 0.0,
      "lead_relevancy": True,
      "initial_distance_lead": round(initial_distance_lead, 3),
      "speed_lead_values": [0.0, 0.0, round(v_lead, 3), round(v_lead, 3)],
      "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
      "cruise_values": [round(cruise, 3)] * 4,
      "breakpoints": [0.0, round(pullaway_time, 3), round(pullaway_time + 1.0, 3), 18.0],
      "ensure_start": True,
    },
  )


def _cruise_coast(rng: random.Random, idx: int, mode: str, profile: LongitudinalProfile | None) -> Scenario:
  v_ego = _sample_profile_range(rng, profile, "ego_speed", (10.0, 25.0), (5.0, 30.0))
  cruise = min(v_ego, _sample_profile_range(rng, profile, "cruise_speed", (max(1.0, v_ego - 5.0), v_ego), (1.0, 32.0)))
  pitch = rng.uniform(-0.08, 0.08)
  if math.isclose(cruise, 0.0):
    cruise = 1.0
  return Scenario(
    mode,
    "cruise_coast",
    f"fuzz cruise coast #{idx}",
    rng.uniform(8.0, 18.0),
    {
      "initial_speed": round(v_ego, 3),
      "lead_relevancy": False,
      "cruise_values": [round(cruise, 3), round(cruise, 3)],
      "pitch_values": [round(pitch, 3), round(pitch, 3)],
      "breakpoints": [0.0, 18.0],
    },
  )


def _sample_profile_range(
  rng: random.Random, profile: LongitudinalProfile | None, attr: str, fallback: tuple[float, float], clamp: tuple[float, float]
) -> float:
  low, high = fallback
  if profile is not None:
    profile_range = getattr(profile, attr)
    profiled_low = max(float(profile_range.low), clamp[0])
    profiled_high = min(float(profile_range.high), clamp[1])
    if profiled_low <= profiled_high:
      low, high = profiled_low, profiled_high

  low = max(low, clamp[0])
  high = min(high, clamp[1])
  if high < low:
    low, high = high, low
  if math.isclose(low, high):
    return low
  return rng.uniform(low, high)


# ── ISO 15622 ACC ────────────────────────────────────────────────────────────

def generate_iso15622_acc_scenarios(mode: str = "comfort") -> list[Scenario]:
    """ISO 15622:2018 ACC performance requirements.

    Key parameters: auto-stop at 2.5 m/s² lead braking, minimum following
    time gap τ ≥ 0.8 s, target discrimination at 3.5 m lateral, curve
    R ≥ 500 m with lateral acceleration ≤ 2.0 m/s².
    """
    _validate_mode(mode)
    return [
        Scenario(mode, "iso15622_auto_stop", "ISO 15622 auto-stop test", 30.0, {
            "initial_speed": 80 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 50.0,
            "speed_lead_values": [80 * KPH_TO_MS, 80 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [80 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 5.0, 10.0, 30.0],  # ~4.44 m/s² decel (80 km/h -> 0 in 5 s)
        }, oracle_profile="iso"),
        Scenario(mode, "iso15622_steady_following", "ISO 15622 steady following", 20.0, {
            "initial_speed": 100 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 30.0,  # τ ≈ 1.1 s at 100 km/h
            "speed_lead_values": [90 * KPH_TO_MS] * 2,
            "prob_lead_values": [1.0] * 2,
            "cruise_values": [100 * KPH_TO_MS] * 2,
            "breakpoints": [0.0, 20.0],
        }, oracle_profile="iso"),
        Scenario(mode, "iso15622_approaching_convoy", "ISO 15622 approaching slower convoy", 25.0, {
            "initial_speed": 120 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 150.0,
            "speed_lead_values": [80 * KPH_TO_MS] * 2,
            "prob_lead_values": [1.0] * 2,
            "cruise_values": [120 * KPH_TO_MS] * 2,
            "breakpoints": [0.0, 25.0],
        }, oracle_profile="iso"),
        Scenario(mode, "iso15622_stop_and_go", "ISO 15622 stop and go", 70.0, {
            "initial_speed": 60 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 40.0,
            "speed_lead_values": [60 * KPH_TO_MS, 0.0, 0.0, 60 * KPH_TO_MS, 0.0, 60 * KPH_TO_MS],
            "prob_lead_values": [1.0] * 6,
            "cruise_values": [60 * KPH_TO_MS] * 6,
            "breakpoints": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }, oracle_profile="iso"),
        Scenario(mode, "iso15622_overtaking", "ISO 15622 overtaking after approach", 25.0, {
            "initial_speed": 100 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 80.0,
            "speed_lead_values": [70 * KPH_TO_MS, 70 * KPH_TO_MS, 100 * KPH_TO_MS],
            "prob_lead_values": [1.0, 1.0, 0.0],  # lead disappears
            "cruise_values": [100 * KPH_TO_MS] * 3,
            "breakpoints": [0.0, 10.0, 10.1],
        }, oracle_profile="iso"),
        Scenario(mode, "iso15622_cornering_follow", "ISO 15622 cornering follow R≥500m", 20.0, {
            "initial_speed": 80 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 40.0,
            "speed_lead_values": [70 * KPH_TO_MS] * 2,
            "prob_lead_values": [1.0] * 2,
            "cruise_values": [80 * KPH_TO_MS] * 2,
            # Lat accel ≤ 2.0 m/s² at R=500m → v ≤ sqrt(2*500) = 31.6 m/s ≈ 114 km/h
            "pitch_values": [0.0] * 2,
            "breakpoints": [0.0, 20.0],
        }, oracle_profile="iso"),
    ]


# ── UN R157 ALKS ─────────────────────────────────────────────────────────────

def generate_unr157_alks_scenarios(mode: str = "comfort") -> list[Scenario]:
    """UN R157 Automated Lane Keeping System longitudinal tests.

    Key: cut-in TTC formula, lead brake at ≥ 6 m/s², minimum following
    distance d_min = v × τ (τ from speed-dependent table).
    """
    _validate_mode(mode)
    return [
        Scenario(mode, "unr157_lead_brake_6ms2", "UN R157 lead brake 6 m/s²", 25.0, {
            "initial_speed": 90 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 60.0,
            "speed_lead_values": [90 * KPH_TO_MS, 90 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [90 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 2.0, 4.17, 25.0],  # 25 m/s → 0 in ~4.17s at −6 m/s²
        }),
        Scenario(mode, "unr157_cut_in_avoidable", "UN R157 cut-in avoidable", 20.0, {
            "initial_speed": 80 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 80.0,
            "speed_lead_values": [70 * KPH_TO_MS, 70 * KPH_TO_MS, 70 * KPH_TO_MS],
            "prob_lead_values": [0.0, 0.0, 1.0],  # cut-in at mid-scenario
            "cruise_values": [80 * KPH_TO_MS] * 3,
            "breakpoints": [0.0, 5.0, 5.1],
        }),
        Scenario(mode, "unr157_cut_out_obstacle", "UN R157 cut-out reveal obstacle", 25.0, {
            "initial_speed": 90 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 100.0,
            "speed_lead_values": [80 * KPH_TO_MS, 80 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],  # lead suddenly brakes hard
            "cruise_values": [90 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 4.0, 6.5, 25.0],  # hard brake at 6 m/s²
        }),
        Scenario(mode, "unr157_min_following", "UN R157 minimum following distance", 20.0, {
            "initial_speed": 60 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 15.0,  # τ ≈ 0.9 s, near minimum
            "speed_lead_values": [55 * KPH_TO_MS] * 2,
            "prob_lead_values": [1.0] * 2,
            "cruise_values": [60 * KPH_TO_MS] * 2,
            "breakpoints": [0.0, 20.0],
        }),
        Scenario(mode, "unr157_emergency_decel", "UN R157 emergency decel ≥5 m/s²", 20.0, {
            "initial_speed": 60 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 30.0,
            "speed_lead_values": [60 * KPH_TO_MS, 60 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [60 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 1.5, 3.33, 20.0],  # 16.67 m/s → 0 in ~3.33s at −5 m/s²
        }),
    ]


# ── NHTSA FCW ────────────────────────────────────────────────────────────────

def generate_nhtsa_fcw_scenarios(mode: str = "comfort") -> list[Scenario]:
    """NHTSA NCAP Forward Collision Warning tests.

    Three scenarios: LVS (lead vehicle stopped, TTC ≥ 2.1s),
    LVD (lead decel at 0.3g, TTC ≥ 2.4s), LVM (lead moving slower, TTC ≥ 2.0s).
    Test speed: 45 mph (72.4 km/h).
    """
    _validate_mode(mode)
    v = 45.0 * MPH_TO_MS  # 72.4 km/h
    v_slower = 20.0 * MPH_TO_MS  # LVM target speed
    return [
        Scenario(mode, "nhtsa_lvs_stopped", "NHTSA LVS lead vehicle stopped", 20.0, {
            "initial_speed": v,
            "lead_relevancy": True,
            "initial_distance_lead": 120.0,
            "speed_lead_values": [v, v, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [v] * 4,
            "breakpoints": [0.0, 5.0, 5.1, 20.0],
        }),
        Scenario(mode, "nhtsa_lvd_decel", "NHTSA LVD lead decel 0.3g", 20.0, {
            "initial_speed": v,
            "lead_relevancy": True,
            "initial_distance_lead": 30.0,  # 30 m headway per spec
            "speed_lead_values": [v, v, 16.0],  # 0.3g decel from 20.1 to ~16 m/s
            "prob_lead_values": [1.0, 1.0, 1.0],
            "cruise_values": [v] * 3,
            "breakpoints": [0.0, 1.5, 2.9],  # Δv ≈ 4.1 m/s at -2.94 m/s² → ~1.4s
        }),
        Scenario(mode, "nhtsa_lvm_slower", "NHTSA LVM lead vehicle moving slower", 25.0, {
            "initial_speed": v,
            "lead_relevancy": True,
            "initial_distance_lead": 100.0,
            "speed_lead_values": [v_slower, v_slower],
            "prob_lead_values": [1.0, 1.0],
            "cruise_values": [v, v],
            "breakpoints": [0.0, 25.0],
        }),
        Scenario(mode, "nhtsa_false_positive_trench", "NHTSA false positive trench plate", 15.0, {
            "initial_speed": v,
            "lead_relevancy": False,  # no lead — must not false-positive brake
            "cruise_values": [v, v],
            "breakpoints": [0.0, 15.0],
        }),
    ]


# ── C-NCAP CCRh ──────────────────────────────────────────────────────────────

def generate_cncap_ccrh_scenarios(mode: str = "comfort") -> list[Scenario]:
    """C-NCAP high-speed stationary approach (CCRh).

    Tests at 80 and 120 km/h toward stationary target — higher speeds
    than typical Euro NCAP CCRs tests.
    """
    _validate_mode(mode)
    return [
        Scenario(mode, "cncap_ccrh_80kph", "C-NCAP CCRh 80 km/h", 20.0, {
            "initial_speed": 80 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 200.0,
            "speed_lead_values": [80 * KPH_TO_MS, 80 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [80 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 3.0, 3.1, 20.0],
        }),
        Scenario(mode, "cncap_ccrh_120kph", "C-NCAP CCRh 120 km/h", 25.0, {
            "initial_speed": 120 * KPH_TO_MS,
            "lead_relevancy": True,
            "initial_distance_lead": 300.0,
            "speed_lead_values": [120 * KPH_TO_MS, 120 * KPH_TO_MS, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [120 * KPH_TO_MS] * 4,
            "breakpoints": [0.0, 3.0, 3.1, 25.0],
        }),
    ]


# ── IIHS ACC ─────────────────────────────────────────────────────────────────

def generate_iihs_acc_scenarios(mode: str = "comfort") -> list[Scenario]:
    """IIHS 2018 track test inspired ACC scenarios.

    Cut-out reveal at TTC ≈ 4.3 s (lead exits, stationary target appears).
    ACC stationary approach at 31 mph with ACC engaged.
    """
    _validate_mode(mode)
    v31 = 31.0 * MPH_TO_MS  # 49.9 km/h
    return [
        Scenario(mode, "iihs_cut_out_reveal", "IIHS cut-out reveal TTC 4.3s", 25.0, {
            "initial_speed": v31,
            "lead_relevancy": True,
            "initial_distance_lead": 60.0,
            "speed_lead_values": [v31, v31, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],  # lead hard brakes at TTC 4.3s
            "cruise_values": [v31] * 4,
            "breakpoints": [0.0, 4.3, 6.3, 25.0],  # brake to stop in ~2s at moderate decel
        }),
        Scenario(mode, "iihs_stopped_acc_on", "IIHS stopped vehicle ACC on", 25.0, {
            "initial_speed": v31,
            "lead_relevancy": True,
            "initial_distance_lead": 150.0,
            "speed_lead_values": [v31, v31, 0.0, 0.0],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [v31] * 4,
            "breakpoints": [0.0, 5.0, 5.1, 25.0],
        }),
        Scenario(mode, "iihs_stop_and_go_smooth", "IIHS stop-and-go smooth decel", 40.0, {
            "initial_speed": v31,
            "lead_relevancy": True,
            "initial_distance_lead": 30.0,
            "speed_lead_values": [v31, 0.0, 0.0, v31],
            "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
            "cruise_values": [v31, v31, v31, v31],
            "breakpoints": [0.0, 10.0, 20.0, 30.0],
        }),
    ]
