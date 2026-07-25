from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleProfile:
  name: str
  checks: tuple[str, ...]
  use_launch_oracle: bool
  respect_plant_ensure_flags: bool
  max_jerk_override: float | None = None
  max_impact_speed_ms: float | None = None
  use_best_effort_collision: bool = True
  skip_jerk: bool = False


ORACLE_PROFILES: dict[str, OracleProfile] = {
  "comfort": OracleProfile(
    name="comfort",
    checks=("valid", "finite", "speed", "collision", "jerk", "launch"),
    use_launch_oracle=True,
    respect_plant_ensure_flags=False,
    use_best_effort_collision=True,
  ),
  # ISO 15622 is a performance standard: contact is a failure, full stop. It still
  # wants the comfort checks (speed, jerk, launch), so it cannot just reuse
  # "regression" (which drops them) or "safety" (which tolerates a 5 km/h impact and
  # disables the jerk bound). Previously these scenarios fell through to "comfort",
  # where best-effort braking could excuse a collision outright.
  "iso": OracleProfile(
    name="iso",
    checks=("valid", "finite", "speed", "collision", "jerk", "launch"),
    use_launch_oracle=True,
    respect_plant_ensure_flags=False,
    use_best_effort_collision=False,
  ),
  "safety": OracleProfile(
    name="safety",
    checks=("valid", "finite", "collision", "jerk"),
    use_launch_oracle=False,
    respect_plant_ensure_flags=False,
    max_jerk_override=100.0,
    max_impact_speed_ms=5.0 / 3.6,
    use_best_effort_collision=False,
    skip_jerk=False,
  ),
  "regression": OracleProfile(
    name="regression",
    checks=("valid", "finite", "collision"),
    use_launch_oracle=False,
    respect_plant_ensure_flags=True,
    use_best_effort_collision=False,
    skip_jerk=True,
  ),
  "baseline_only": OracleProfile(
    name="baseline_only",
    checks=("finite",),
    use_launch_oracle=False,
    respect_plant_ensure_flags=False,
    use_best_effort_collision=True,
    skip_jerk=True,
  ),
}


def get_oracle_profile(name: str) -> OracleProfile:
  if name not in ORACLE_PROFILES:
    raise ValueError(f"unknown oracle profile {name!r}; expected one of {tuple(ORACLE_PROFILES)}")
  return ORACLE_PROFILES[name]
