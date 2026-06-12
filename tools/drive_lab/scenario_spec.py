from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_ORACLE_CHECKS = ("valid", "finite", "speed", "collision", "jerk")


@dataclass(frozen=True)
class ScenarioSpec:
  scenario_id: str
  kind: str
  title: str
  mode: str
  duration: float
  source: str
  maneuver_kwargs: dict[str, Any]
  ego: dict[str, Any] = field(default_factory=dict)
  actors: dict[str, Any] = field(default_factory=dict)
  events: tuple[str, ...] = ()
  oracle: dict[str, Any] = field(default_factory=dict)
  tags: tuple[str, ...] = ()
  seed: int | None = None
  index: int | None = None
  provenance: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def from_maneuver_kwargs(cls, kind: str, title: str, mode: str, duration: float, kwargs: dict[str, Any],
                           source: str = "generated", seed: int | None = None, index: int | None = None,
                           provenance: dict[str, Any] | None = None) -> ScenarioSpec:
    scenario_id = _scenario_id(source, mode, kind, title, seed, index)
    return cls(
      scenario_id=scenario_id,
      kind=str(kind),
      title=str(title),
      mode=str(mode),
      duration=float(duration),
      source=str(source),
      maneuver_kwargs=dict(kwargs),
      ego=_ego_from_kwargs(kwargs),
      actors=_actors_from_kwargs(kwargs),
      events=(str(kind),),
      oracle={"checks": DEFAULT_ORACLE_CHECKS},
      tags=tuple(tag for tag in (source, mode, kind) if tag),
      seed=seed,
      index=index,
      provenance=dict(provenance or {}),
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "scenario_id": self.scenario_id,
      "kind": self.kind,
      "title": self.title,
      "mode": self.mode,
      "duration": self.duration,
      "source": self.source,
      "maneuver_kwargs": self.maneuver_kwargs,
      "ego": self.ego,
      "actors": self.actors,
      "events": list(self.events),
      "oracle": _oracle_to_dict(self.oracle),
      "tags": list(self.tags),
      "seed": self.seed,
      "index": self.index,
      "provenance": self.provenance,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> ScenarioSpec:
    oracle = dict(data.get("oracle", {}))
    if "checks" in oracle:
      oracle["checks"] = tuple(oracle["checks"])
    return cls(
      scenario_id=str(data["scenario_id"]),
      kind=str(data["kind"]),
      title=str(data["title"]),
      mode=str(data["mode"]),
      duration=float(data["duration"]),
      source=str(data["source"]),
      maneuver_kwargs=dict(data.get("maneuver_kwargs", {})),
      ego=dict(data.get("ego", {})),
      actors=dict(data.get("actors", {})),
      events=tuple(data.get("events", ())),
      oracle=oracle,
      tags=tuple(data.get("tags", ())),
      seed=data.get("seed"),
      index=data.get("index"),
      provenance=dict(data.get("provenance", {})),
    )


def route_window_provenance(route_id: str, segment: int | None, start_s: float, end_s: float, source_tool: str) -> dict[str, Any]:
  return {
    "route_id": route_id,
    "segment": segment,
    "start_s": start_s,
    "end_s": end_s,
    "source_tool": source_tool,
  }


def _scenario_id(source: str, mode: str, kind: str, title: str, seed: int | None, index: int | None) -> str:
  if seed is not None and index is not None:
    return f"{source}:{mode}:{kind}:{seed}:{index}"
  return f"{source}:{mode}:{kind}:{_slug(title)}"


def _slug(value: str) -> str:
  return "_".join(str(value).strip().lower().replace("#", " ").split())


def _ego_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
  ego: dict[str, Any] = {}
  if "initial_speed" in kwargs:
    ego["initial_speed"] = kwargs["initial_speed"]
  if "cruise_values" in kwargs:
    ego["cruise_values"] = kwargs["cruise_values"]
  if "pitch_values" in kwargs:
    ego["pitch_values"] = kwargs["pitch_values"]
  return ego


def _actors_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
  if not kwargs.get("lead_relevancy"):
    return {}
  lead: dict[str, Any] = {}
  if "initial_distance_lead" in kwargs:
    lead["initial_distance"] = kwargs["initial_distance_lead"]
  if "speed_lead_values" in kwargs:
    lead["speed_values"] = kwargs["speed_lead_values"]
  if "prob_lead_values" in kwargs:
    lead["probability_values"] = kwargs["prob_lead_values"]
  return {"lead": lead}


def _oracle_to_dict(oracle: dict[str, Any]) -> dict[str, Any]:
  payload = dict(oracle)
  if "checks" in payload:
    payload["checks"] = list(payload["checks"])
  return payload
