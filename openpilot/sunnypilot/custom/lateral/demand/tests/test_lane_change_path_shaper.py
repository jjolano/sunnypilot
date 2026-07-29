import numpy as np

from openpilot.sunnypilot.custom.lateral.demand.lane_change_path_shaper import (
  LaneChangeDirection,
  LaneChangePathShaper,
  LaneChangePathShaperInputs,
  LaneChangeState,
)


def _inputs(state, direction=LaneChangeDirection.left, model_curvature=0.0):
  return LaneChangePathShaperInputs(
    lat_active=True,
    v_ego=20.0,
    left_blinker=direction == LaneChangeDirection.left,
    right_blinker=direction == LaneChangeDirection.right,
    steering_pressed=False,
    lane_change_state=state,
    lane_change_direction=direction,
    model_curvature=model_curvature,
    prev_desired_curvature=0.0,
    lane_line_probs=[0.9, 0.9, 0.9, 0.9],
    left_lane_y0=-1.8,
    right_lane_y0=1.8,
  )


def _run(shaper, state, frames):
  return [shaper.update(_inputs(state)).blend for _ in range(frames)]


def test_blend_reaches_full_authority_and_returns_to_zero():
  shaper = LaneChangePathShaper(dt=0.01)
  entry = _run(shaper, LaneChangeState.laneChangeStarting, 200)
  assert max(entry) > 0.99
  exit_ = _run(shaper, LaneChangeState.laneChangeFinishing, 150)
  assert exit_[-1] < 0.01


def test_entry_and_exit_endpoints_are_eased_not_linear():
  # A raw linear ramp has near-constant per-frame deltas (first step ~= peak step). Smootherstep
  # easing drives the per-frame delta to ~0 at both endpoints, so first/last steps << peak step.
  shaper = LaneChangePathShaper(dt=0.01)
  entry = np.array(_run(shaper, LaneChangeState.laneChangeStarting, 200))
  full_idx = int(np.argmax(entry >= 0.99))
  ramp = np.diff(entry[: full_idx + 1])
  assert ramp.max() > 0.0
  assert ramp[0] < 0.3 * ramp.max()     # soft onset (the "very start")
  assert ramp[-1] < 0.3 * ramp.max()    # soft approach to full authority

  exit_ = np.array(_run(shaper, LaneChangeState.laneChangeFinishing, 150))
  zero_idx = int(np.argmax(exit_ <= 0.01))
  decay = np.abs(np.diff(exit_[: zero_idx + 1]))
  assert decay.max() > 0.0
  assert decay[0] < 0.3 * decay.max()   # soft exit onset (the "very end")
  assert decay[-1] < 0.3 * decay.max()  # soft landing at zero


def test_inactive_state_passes_model_curvature_through_unchanged():
  shaper = LaneChangePathShaper(dt=0.01)
  result = shaper.update(_inputs(LaneChangeState.off, model_curvature=0.004))
  assert result.desired_curvature == 0.004
  assert result.blend == 0.0
  assert not result.active
