"""Tests for the uncorroborated SCC-vision slowdown softener (route 261)."""
from __future__ import annotations

from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def _planner_shell(vision_active, map_active, v_target, a_target):
  shell = SimpleNamespace(
    scc=SimpleNamespace(
      vision=SimpleNamespace(is_active=vision_active, output_v_target=v_target, output_a_target=a_target),
      map=SimpleNamespace(is_active=map_active),
    ),
  )
  shell._soften_uncorroborated_vision_slowdown = (
    LongitudinalPlannerSP._soften_uncorroborated_vision_slowdown.__get__(shell))
  shell._SCC_VISION_UNCORROBORATED_A_MIN = LongitudinalPlannerSP._SCC_VISION_UNCORROBORATED_A_MIN
  shell._SCC_VISION_SOFTEN_TAU_S = LongitudinalPlannerSP._SCC_VISION_SOFTEN_TAU_S
  return shell


def test_vision_only_slowdown_is_rate_limited():
  # Route 261 t=1052: v_ego 12.5, vision vTarget pinned at 5.6, aTarget -1.0, map inactive.
  p = _planner_shell(True, False, v_target=5.6, a_target=-1.0)
  p._soften_uncorroborated_vision_slowdown(12.5)
  assert p.scc.vision.output_v_target == 12.0   # v_ego - 0.5*tau
  assert p.scc.vision.output_a_target == -0.5


def test_map_corroborated_slowdown_keeps_full_authority():
  p = _planner_shell(True, True, v_target=5.6, a_target=-1.0)
  p._soften_uncorroborated_vision_slowdown(12.5)
  assert p.scc.vision.output_v_target == 5.6
  assert p.scc.vision.output_a_target == -1.0


def test_inactive_vision_untouched():
  p = _planner_shell(False, False, v_target=255.0, a_target=0.0)
  p._soften_uncorroborated_vision_slowdown(12.5)
  assert p.scc.vision.output_v_target == 255.0


def test_gentle_vision_target_passes_through():
  # Target above the rate-limit floor is not raised.
  p = _planner_shell(True, False, v_target=12.2, a_target=-0.3)
  p._soften_uncorroborated_vision_slowdown(12.5)
  assert p.scc.vision.output_v_target == 12.2
  assert p.scc.vision.output_a_target == -0.3


def test_glide_converges_to_turn_speed_as_ego_slows():
  # As ego slows, the rate-limited floor follows it down: the turn speed is still reached,
  # just via a bounded-decel glide.
  p = _planner_shell(True, False, v_target=5.6, a_target=-1.0)
  v = 12.5
  for _ in range(40):
    p.scc.vision.output_v_target = 5.6
    p._soften_uncorroborated_vision_slowdown(v)
    v = max(5.6, v - 0.5 * 0.5)  # ego tracking the softened target at ~0.5 m/s^2, 0.5 s ticks
  assert p.scc.vision.output_v_target == 5.6  # floor no longer binds once ego is at turn speed
