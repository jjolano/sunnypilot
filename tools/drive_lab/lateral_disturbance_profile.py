from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.custom.lateral.disturbance_classifier import (
  DisturbanceClassifier,
  DisturbanceReason,
  LateralSample,
  LearningDecision,
  decision_name,
  reason_names,
)
from openpilot.tools.drive_lab.lateral_torque_event_report import _extract_torque_samples


@dataclass(frozen=True)
class LateralDisturbanceProfile:
  source: str
  sample_count: int
  eligible_sample_count: int
  inactive_excluded: int
  duration_s: float
  decision_counts: dict[str, int] = field(default_factory=dict)
  reason_counts: dict[str, int] = field(default_factory=dict)
  shadow_accepted: int = 0
  shadow_quarantined: int = 0
  shadow_rejected: int = 0
  shadow_quarantine_percent: float = 0.0
  shadow_reject_percent: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "LateralDisturbanceProfile":
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", 0)),
      eligible_sample_count=int(data.get("eligible_sample_count", data.get("sample_count", 0))),
      inactive_excluded=int(data.get("inactive_excluded", 0)),
      duration_s=float(data.get("duration_s", 0.0)),
      decision_counts=dict(data.get("decision_counts", {})),
      reason_counts=dict(data.get("reason_counts", {})),
      shadow_accepted=int(data.get("shadow_accepted", 0)),
      shadow_quarantined=int(data.get("shadow_quarantined", 0)),
      shadow_rejected=int(data.get("shadow_rejected", 0)),
      shadow_quarantine_percent=float(data.get("shadow_quarantine_percent", 0.0)),
      shadow_reject_percent=float(data.get("shadow_reject_percent", 0.0)),
    )


def _is_lane_change_active(sample: Any) -> bool:
  state = getattr(sample, "lane_change_state", "off")
  return state not in ("off", "unknown")


def _torque_sample_to_classifier_input(sample: Any) -> LateralSample:
  return LateralSample(
    t=sample.t,
    v_ego=sample.v_ego,
    lat_active=sample.lat_active,
    steering_pressed=sample.steering_pressed,
    blinker_active=sample.blinker_active,
    lane_change_active=_is_lane_change_active(sample),
    steering_rate_deg=sample.steering_rate_deg if np.isfinite(sample.steering_rate_deg) else None,
    output=sample.output if np.isfinite(sample.output) else None,
    unshaped_output=sample.unshaped_output if np.isfinite(sample.unshaped_output) else None,
    applied_torque=sample.applied_torque if np.isfinite(sample.applied_torque) else None,
    desired_lateral_accel=sample.desired_lateral_accel if np.isfinite(sample.desired_lateral_accel) else None,
    actual_lateral_accel=sample.actual_lateral_accel if np.isfinite(sample.actual_lateral_accel) else None,
    output_sat=False,
    steer_limited=sample.steer_limited,
    model_path_quality=sample.model_path_quality if np.isfinite(sample.model_path_quality) else None,
    model_path_gated=sample.model_path_gated,
    shaping_reason=sample.shaping_reason,
    governor_reason=sample.governor_reason,
  )


def build_lateral_disturbance_profile(
  msgs: list[Any],
  source: str = "unknown",
  already_sorted: bool = False,
  classifier: DisturbanceClassifier | None = None,
) -> LateralDisturbanceProfile:
  ordered_msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  samples = _extract_torque_samples(ordered_msgs)
  if not samples:
    return LateralDisturbanceProfile(source=source, sample_count=0, eligible_sample_count=0, inactive_excluded=0, duration_s=0.0)

  clf = classifier if classifier is not None else DisturbanceClassifier()
  decision_counts: dict[str, int] = {}
  reason_counts: dict[str, int] = {}
  accepted = 0
  quarantined = 0
  rejected = 0
  inactive_excluded = 0

  prev: LateralSample | None = None
  for sample in samples:
    inp = _torque_sample_to_classifier_input(sample)
    if not inp.lat_active:
      inactive_excluded += 1
      prev = None
      continue
    dt = (inp.t - prev.t) if prev is not None else None
    result = clf.classify(inp, prev_sample=prev, dt=dt)
    prev = inp

    name = decision_name(result.decision)
    decision_counts[name] = decision_counts.get(name, 0) + 1
    if result.decision == LearningDecision.ACCEPT:
      accepted += 1
    elif result.decision == LearningDecision.QUARANTINE:
      quarantined += 1
    elif result.decision == LearningDecision.REJECT_SHADOW:
      rejected += 1

    for reason_name in reason_names(result.reasons):
      reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1

  total = len(samples)
  eligible = accepted + quarantined + rejected
  duration_s = float(samples[-1].t - samples[0].t) if len(samples) > 1 else 0.0
  return LateralDisturbanceProfile(
    source=source,
    sample_count=total,
    eligible_sample_count=eligible,
    inactive_excluded=inactive_excluded,
    duration_s=duration_s,
    decision_counts=decision_counts,
    reason_counts=reason_counts,
    shadow_accepted=accepted,
    shadow_quarantined=quarantined,
    shadow_rejected=rejected,
    shadow_quarantine_percent=quarantined / eligible * 100.0 if eligible else 0.0,
    shadow_reject_percent=rejected / eligible * 100.0 if eligible else 0.0,
  )


def render_lateral_disturbance_profile(profile: LateralDisturbanceProfile) -> str:
  lines = [
    f"Lateral disturbance profile: {profile.source}",
    f"samples: {profile.sample_count}",
    f"eligible samples: {profile.eligible_sample_count}",
    f"inactive excluded: {profile.inactive_excluded}",
    f"duration: {profile.duration_s:.1f} s",
    f"shadow accepted: {profile.shadow_accepted}",
    f"shadow quarantined: {profile.shadow_quarantined} ({profile.shadow_quarantine_percent:.1f}%)",
    f"shadow rejected: {profile.shadow_rejected} ({profile.shadow_reject_percent:.1f}%)",
    "decisions:",
  ]
  for name, count in sorted(profile.decision_counts.items()):
    lines.append(f"  {name}: {count}")
  lines.append("reasons:")
  for name, count in sorted(profile.reason_counts.items()):
    lines.append(f"  {name}: {count}")
  return "\n".join(lines)


def save_lateral_disturbance_profile(profile: LateralDisturbanceProfile, path: str | Path) -> None:
  Path(path).write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")


def load_lateral_disturbance_profile(path: str | Path) -> LateralDisturbanceProfile:
  return LateralDisturbanceProfile.from_dict(json.loads(Path(path).read_text()))
