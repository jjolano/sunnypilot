import numpy as np

from openpilot.tools.drive_lab.fuzz_longitudinal_route_replay import (
  RouteReplayFuzzerConfig,
  _apply_recipe,
  _generate_recipe,
  generate_route_replay_scenarios,
  route_replay_to_scenario,
)
from openpilot.tools.drive_lab.longitudinal_route_extract import (
  LongitudinalRouteFrame,
  frames_to_maneuver_kwargs,
  max_d_rel_error,
)


def _sample_frames(n: int = 20) -> tuple[LongitudinalRouteFrame, ...]:
  return tuple(
    LongitudinalRouteFrame(
      t=i * 0.05,
      source_t=float(i),
      v_ego=15.0,
      v_cruise=15.0,
      pitch=0.0,
      lead_active=True,
      d_rel=40.0 - i * 0.1,
      v_lead=14.0,
      prob_lead=1.0,
    )
    for i in range(n)
  )


def test_frames_to_maneuver_kwargs():
  frames = _sample_frames(5)
  kwargs = frames_to_maneuver_kwargs(frames)
  assert kwargs["initial_speed"] == 15.0
  assert kwargs["lead_relevancy"] is True
  assert len(kwargs["breakpoints"]) == 5
  assert kwargs["initial_distance_lead"] == 40.0


def test_generate_recipe_is_seeded():
  frames = _sample_frames()
  import random
  r1 = _generate_recipe(random.Random(1), len(frames), "dropout")
  r2 = _generate_recipe(random.Random(1), len(frames), "dropout")
  assert r1 == r2


def test_dropout_perturbation_zeros_prob_lead():
  frames = _sample_frames()
  recipe = _generate_recipe(__import__("random").Random(2), len(frames), "dropout")
  perturbed = _apply_recipe(recipe, frames)
  for i in range(recipe.start_frame, recipe.end_frame):
    assert perturbed[i].prob_lead == 0.0


def test_generate_route_replay_scenarios_count():
  frames = _sample_frames()
  config = RouteReplayFuzzerConfig(seed=3, cases=4, perturbation="none")
  scenarios = generate_route_replay_scenarios(frames, config)
  assert len(scenarios) == 4


def test_route_replay_to_scenario():
  frames = _sample_frames()
  config = RouteReplayFuzzerConfig(seed=1, cases=1, perturbation="none")
  replay = generate_route_replay_scenarios(frames, config)[0]
  scenario = route_replay_to_scenario(replay, "comfort")
  assert scenario.kind == "route_replay_none"
  assert scenario.duration > 0.0


def test_max_d_rel_error():
  frames = _sample_frames(3)
  output = np.array([
    [0.0, 0.0, 0.0, 15.0, 14.0, 0.0, 40.0],
    [0.05, 0.75, 0.0, 15.0, 14.0, 0.0, 39.5],
    [0.10, 1.5, 0.0, 15.0, 14.0, 0.0, 39.0],
  ])
  err = max_d_rel_error(frames, output)
  assert err is not None
  assert err >= 0.0


def test_extract_longitudinal_route_frames_from_fake_messages():
  from types import SimpleNamespace
  from openpilot.tools.drive_lab.longitudinal_route_extract import DT, extract_longitudinal_route_frames

  class _FakeMsg(SimpleNamespace):
    def which(self):
      return self.kind

  def _route_msg(kind: str, t_s: float, payload: SimpleNamespace) -> _FakeMsg:
    return _FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: payload})

  messages = [
    _route_msg("carState", 0.0, SimpleNamespace(vEgo=10.0, vCruise=50.0)),
    _route_msg("liveParameters", 0.0, SimpleNamespace(pitch=0.01)),
    _route_msg("radarState", 0.0, SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=30.0, vLead=9.0, modelProb=0.9))),
    _route_msg("controlsState", 0.0, SimpleNamespace()),
    _route_msg("carState", DT, SimpleNamespace(vEgo=10.1, vCruise=50.0)),
    _route_msg("radarState", DT, SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=29.5, vLead=9.0, modelProb=0.9))),
    _route_msg("controlsState", DT, SimpleNamespace()),
  ]
  frames = extract_longitudinal_route_frames(messages)
  assert len(frames) == 2
  assert frames[0].v_ego == 10.0
  assert frames[0].d_rel == 30.0
