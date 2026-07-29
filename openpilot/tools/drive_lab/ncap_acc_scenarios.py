from __future__ import annotations

import random
from typing import Any

from openpilot.tools.drive_lab.longitudinal_scenarios import REALISM_MODES, Scenario, _validate_mode


def _kmh_to_ms(kmh: float) -> float:
  return kmh / 3.6


CCRS_VUT_KMH = (70, 90, 110, 130)
CCRM_GRID = (
  (80, 20), (80, 60),
  (90, 20), (90, 60),
  (100, 20), (100, 60),
  (110, 20), (110, 60),
  (120, 20), (120, 60),
  (130, 20), (130, 60),
)
CCRb_CASES = (
  (55, 50, 4.0),
  (80, 70, 4.0),
)
CCRM_MIN_INITIAL_DISTANCE_M = 80.0
CCRM_MIN_INITIAL_TTC_S = 4.5


def generate_ncap_acc_scenarios(
  mode: str = "comfort",
  *,
  family: str | None = None,
  sample: int | None = None,
  seed: int = 1,
) -> list[Scenario]:
  _validate_mode(mode)
  if family is None and sample is None:
    return _curated_ncap_scenarios(mode)
  if family is None:
    raise ValueError("ncap family is required when using ncap sample mode")
  return _sample_ncap_scenarios(mode, family=family, sample=sample or 10, seed=seed)


def _curated_ncap_scenarios(mode: str) -> list[Scenario]:
  scenarios: list[Scenario] = []
  for vut_kmh in CCRS_VUT_KMH:
    vut = _kmh_to_ms(vut_kmh)
    scenarios.append(Scenario(
      mode,
      f"ncap_ccrs_{vut_kmh}",
      f"ncap acc ccrs stationary target at {vut_kmh} km/h",
      35.0,
      {
        "initial_speed": round(vut, 3),
        "lead_relevancy": True,
        "initial_distance_lead": 200.0,
        "speed_lead_values": [0.0, 0.0],
        "prob_lead_values": [1.0, 1.0],
        "cruise_values": [round(vut, 3), round(vut, 3)],
        "breakpoints": [0.0, 35.0],
      },
      oracle_profile="safety",
    ))
  for vut_kmh, target_kmh in CCRM_GRID:
    vut = _kmh_to_ms(vut_kmh)
    target = _kmh_to_ms(target_kmh)
    initial_distance = _ccrm_initial_distance(vut, target)
    scenarios.append(Scenario(
      mode,
      f"ncap_ccrm_{vut_kmh}_{target_kmh}",
      f"ncap acc ccrm moving target {target_kmh} km/h at {vut_kmh} km/h",
      40.0,
      {
        "initial_speed": round(vut, 3),
        "lead_relevancy": True,
        "initial_distance_lead": initial_distance,
        "speed_lead_values": [round(target, 3), round(target, 3)],
        "prob_lead_values": [1.0, 1.0],
        "cruise_values": [round(vut, 3), round(vut, 3)],
        "breakpoints": [0.0, 40.0],
      },
      oracle_profile="safety",
    ))
  for vut_kmh, target_kmh, decel in CCRb_CASES:
    vut = _kmh_to_ms(vut_kmh)
    target = _kmh_to_ms(target_kmh)
    decel_time = max(0.01, target / decel)
    scenarios.append(Scenario(
      mode,
      f"ncap_ccrb_{vut_kmh}",
      f"ncap acc ccrb braking target at {vut_kmh} km/h",
      max(30.0, decel_time + 10.0),
      {
        "initial_speed": round(vut, 3),
        "lead_relevancy": True,
        "initial_distance_lead": 60.0,
        "speed_lead_values": [round(target, 3), round(target, 3), 0.0, 0.0],
        "prob_lead_values": [1.0, 1.0, 1.0, 1.0],
        "cruise_values": [round(vut, 3)] * 4,
        "breakpoints": [0.0, 5.0, round(5.0 + decel_time, 3), round(5.0 + decel_time + 0.01, 3)],
      },
      oracle_profile="safety",
    ))
  return scenarios


def _ccrm_initial_distance(vut_speed: float, target_speed: float) -> float:
  closing_speed = max(0.0, vut_speed - target_speed)
  return round(max(CCRM_MIN_INITIAL_DISTANCE_M, closing_speed * CCRM_MIN_INITIAL_TTC_S), 3)


def _sample_ncap_scenarios(mode: str, *, family: str, sample: int, seed: int) -> list[Scenario]:
  curated = _curated_ncap_scenarios(mode)
  if family == "CCRs":
    pool = [s for s in curated if s.kind.startswith("ncap_ccrs_")]
  elif family == "CCRm":
    pool = [s for s in curated if s.kind.startswith("ncap_ccrm_")]
  elif family == "CCRb":
    pool = [s for s in curated if s.kind.startswith("ncap_ccrb_")]
  else:
    raise ValueError(f"unknown ncap family {family!r}; expected CCRs, CCRm, or CCRb")
  rng = random.Random(seed)
  if sample >= len(pool):
    return list(pool)
  return rng.sample(pool, sample)
