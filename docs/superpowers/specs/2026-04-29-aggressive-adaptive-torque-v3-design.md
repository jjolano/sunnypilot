# Aggressive Adaptive Torque V3 Design

Status: superseded by `docs/adr/0001-redesign-torque-v3.md`

This document describes the old universal adaptive Torque v3 direction. It is retained as historical context only; fresh Torque v3 is native-torque-only, follows processed curvature, and uses bounded session response-scale/trim learning instead of synthetic torque model authority.

## Summary

Torque v3 is an opt-in lateral torque controller for every non-angle-control vehicle. It keeps the proven torque v2 safety envelope, but replaces v2's mostly fixed torque model with an adaptive model that can learn normalized torque response online. Native torque-tuned cars start from their existing car-interface torque conversion. PID or otherwise non-torque non-angle cars start from a conservative synthetic model, then earn authority as the online estimator proves stable behavior.

The long-term goal is universal non-angle compatibility, including full normalized torque authority after convergence. Full authority is not available by default. It is earned through stable bidirectional evidence and can be revoked immediately when confidence drops.

## Goals

- Support all non-angle cars through one torque-controller path.
- Preserve native torque behavior for cars with valid `CP.lateralTuning.torque` and car-interface torque callbacks.
- Allow PID/non-torque non-angle cars to start from a conservative synthetic model and learn toward full normalized torque authority.
- Keep v2's conservative safety shaping active at all authority levels.
- Make authority confidence-driven, reversible, and visible in telemetry.
- Fall back or demote quickly when behavior is not explainable by the learned model.
- Keep v3 opt-in until route/replay and platform validation justify broader rollout.

## Non-Goals

- Do not support angle-control cars with torque v3.
- Do not bypass platform safety, controller rate limits, EPS limits, or car-controller torque limiters.
- Do not make v3 the default controller in the first implementation.
- Do not rely on brand-specific CAN torque units inside v3; v3 outputs normalized torque only.
- Do not require persisted learned state for the first version. Runtime learning is sufficient for the initial rollout.

## Current Torque V2 Baseline

Torque v2 is the retained conservative torque controller in `sunnypilot/selfdrive/controls/lib/latcontrol_torque_v2.py`. It combines these layers:

- Lateral-accel-space PID with car-specific torque conversion.
- Angle-derived lateral acceleration smoothing.
- Delay-aware desired lateral acceleration and jerk calculation.
- Existing sunnypilot torque extension and NNLC hook.
- Guarded response assist for bounded under-response correction.
- Conservative output shaping for override, release, sign conflict, over-response, near-ISO lateral accel, bump disturbance, low-speed steer limits, and same-sign unwind.
- Adaptive torque telemetry in `ControlsState.LateralTorqueState.AdaptiveTorqueState`.

V2's main limitations for universal compatibility are:

- It assumes usable `CP.lateralTuning.torque` parameters.
- Some runtime support in `controlsd.py` is gated by `CP.lateralTuning.which() == 'torque'`.
- It can add assist and shaping, but it does not own a general online torque-response model.
- Its constants are fixed and not confidence-aware across native and synthetic platforms.

## Architecture

Torque v3 should be implemented as a separate controller rather than a patch to v2.

### `LatControlTorqueV3`

`LatControlTorqueV3` is the controller entry point and preserves the existing `LatControl.update(...)` signature. It owns the control loop, telemetry, fallback handling, and the component pipeline.

Responsibilities:

- Compute desired and measured lateral acceleration using the v2 delay-aware path.
- Build a conservative baseline torque request.
- Update the adaptive estimator with clean samples.
- Ask the torque model for feedforward torque.
- Apply authority scaling and safety shaping.
- Emit torque-state telemetry.
- Fall back or demote when compatibility or confidence checks fail.

### `TorqueModelAdapter`

`TorqueModelAdapter` converts desired lateral acceleration into normalized torque and provides the inverse estimate used by limits and diagnostics.

Modes:

- `native`: use car-interface torque conversion callbacks and native torque params.
- `synthetic`: use conservative generic starting parameters when native torque tuning is unavailable.
- `learned`: use the online learned model once confidence permits.
- `fallback`: keep low-authority conservative behavior or delegate to the original controller when v3 is not safe to run.

The adapter should support a simple linear model first, then an optional bounded residual term for nonlinear response. GM-style nonlinear native mappings must remain valid because openpilot car interfaces can expose nonlinear torque callbacks.

### `AdaptiveTorqueEstimator`

`AdaptiveTorqueEstimator` learns the vehicle's normalized torque response from clean control frames.

Estimated state:

- `latAccelFactor`: normalized torque-to-lateral-acceleration gain.
- `friction`: torque deadband/friction estimate.
- `latAccelOffset`: roll/calibration residual offset.
- `responseDelay`: observed actuator response delay.
- `residual`: optional bounded nonlinear correction by speed, sign, or torque magnitude.

The estimator must learn only from clean frames:

- Lateral control is active.
- Driver is not steering.
- Command is not saturated.
- `steer_limited_by_safety` is false.
- `curvature_limited` is false.
- Speed is above the learning threshold.
- Steering rate, measured lateral acceleration, and lateral jerk are finite and plausible.
- The measured response sign is explainable by the command and desired path.

Learning should be separated by speed range and torque sign so a bad or sparse region does not poison the whole model. Confidence should build slowly from repeated clean samples and decay quickly on residual spikes or safety events.

### `AuthorityManager`

`AuthorityManager` converts estimator confidence into allowed normalized torque authority.

Authority bands:

- `limited`: synthetic startup and low-confidence operation.
- `partial`: enough clean evidence for moderate torque but not full output.
- `near_full`: stable bidirectional evidence, still with additional margin.
- `full`: full `[-1.0, 1.0]` normalized torque authority allowed.

Synthetic/non-native cars can reach `full`, but only after convergence. Full authority requires stable evidence across both torque signs and enough speed/curvature/torque ranges. Native torque cars may start at higher authority, but still demote on unexplained behavior.

Demotion must be faster than promotion. Any of these should reduce authority immediately:

- Residual spike beyond the current model's tolerance.
- Sign conflict between desired response and measured response.
- Repeated safety limiting or actuator mismatch.
- Saturation while the estimator is still trying to learn in the same direction.
- Driver override.
- Bump/noise disturbance.
- Stale or invalid model/pose/live-parameter data.
- Non-finite output or measurement.

### `TorqueSafetyEnvelope`

`TorqueSafetyEnvelope` should keep the existing v2 conservative shaping behavior active after adaptive torque generation.

Always-preserved shaping reasons:

- Steering override and release.
- Sign conflict.
- Over-response.
- Near or over ISO lateral acceleration margin.
- Bump disturbance.
- Low-speed steer-limited high output.
- Same-sign unwind.
- Recovery rate limiting after shaping.

V3 may add confidence-based caps before or inside the safety envelope, but it should not remove v2's safety caps.

### `FallbackController`

V3 must preserve the originally selected non-angle lateral controller for fallback. For native torque cars this may be the stock/sunnypilot torque controller. For PID cars this is the PID controller that would have run without v3.

Fallback behavior:

- If v3 cannot initialize a valid model, keep the original controller.
- If runtime confidence collapses, blend or step down to conservative torque/fallback behavior.
- If v3 produces non-finite internal state, output zero or fallback output according to the safest available path and log the reason.

## Data Flow

Each control frame follows four passes.

### 1. Baseline Request

- Convert desired curvature to future desired lateral acceleration.
- Use the v2 request buffer and `lat_delay` to compute expected lateral acceleration, setpoint, and desired lateral jerk.
- Convert steering angle and vehicle model state to measured lateral acceleration.
- Condition measured acceleration with the v2 smoother.
- Build a conservative baseline feedback request so v3 always has a bounded control path.

### 2. Adaptive Model Update

- Build a candidate observation from command, measured response, speed, roll, steering angle/rate, desired lateral acceleration, and safety flags.
- Reject the observation unless every clean-frame gate passes.
- Update factor/friction/offset/delay/residual estimates using bounded rates.
- Update confidence for the relevant speed/sign region.
- Decay confidence on unsafe, stale, or unexplained frames.

### 3. Authority Decision

- Query model confidence and coverage from the estimator.
- Select authority band and normalized torque cap.
- Blend baseline feedback torque with learned feedforward torque according to confidence.
- Permit full normalized torque only in `full` authority.

### 4. Safety Envelope

- Apply confidence caps and v2 conservative shaping to the unshaped adaptive output.
- Check saturation using the final shaped output.
- Emit telemetry for model mode, confidence, authority band, cap, fallback state, shaping state, and learned parameters.

## Integration Changes

Current `controlsd.py` only updates torque-specific live params and model extension state when `CP.lateralTuning.which() == 'torque'`. V3 needs controller-type-aware plumbing because it can run on PID/non-torque non-angle cars.

Required integration changes:

- Add `LatControlTorqueV3` as a first-class tune version selected by `TorqueControlTune == 3.0`.
- Stop coercing `3.0` to `2.0` in `ControlsExt.initialize_lateral_control(...)`.
- Keep angle-control cars excluded.
- Let v3 receive model and lateral-lag updates even when the original lateral tuning is not torque.
- Let v3 publish `torqueState` telemetry when selected, even if the original car tune was PID.
- Keep `TorqueControlTune` default at `2.0` for initial rollout.
- Add `v3.0` to `latcontrol_torque_versions.json` and sync Sunnylink metadata.

## Telemetry

Existing adaptive torque telemetry should remain populated. V3 needs additional fields or a compact extension to expose model and authority state.

Recommended telemetry:

- `modelMode`: native, synthetic, learned, fallback.
- `modelConfidence`: aggregate confidence.
- `authorityBand`: limited, partial, near-full, full.
- `authorityScale`: current output cap from confidence.
- `fallbackActive`: whether fallback/blended fallback is active.
- `learnedLatAccelFactor`.
- `learnedFriction`.
- `learnedLatAccelOffset`.
- `learnedResponseDelay`.
- `residualError`.
- `sampleAccepted`: whether the current frame updated the estimator.
- `sampleRejectReason`: bitmask for rejected learning samples.

If schema expansion is too large for the first implementation, these can start as a smaller bitmask plus core confidence/authority fields.

## Failure Handling

V3 should fail toward lower authority.

Rules:

- Non-finite measurement, command, model output, or learned parameter resets the affected update and demotes authority.
- Driver steering immediately freezes learning and starts release behavior.
- Safety limiting freezes learning and demotes the affected sign/speed region.
- Repeated residual spikes force fallback or limited authority.
- Stale model/lateral-lag data disables model-based extras but keeps conservative baseline control.
- Inactive control resets transient assist/shaping state and prevents estimator updates.
- Confidence recovery requires new clean evidence; it does not recover just because time passes.

## Testing Plan

Testing must prove both pre-convergence boundedness and post-convergence authority.

### Unit Tests

- `TorqueModelAdapter`: native, synthetic, learned, nonlinear residual, invalid-param fallback.
- `AdaptiveTorqueEstimator`: accepts clean samples, rejects unsafe samples, learns factor/friction/offset/delay, separates speed/sign regions, demotes on residual spikes.
- `AuthorityManager`: starts synthetic cars capped, promotes through bands, reaches full authority only with stable bidirectional evidence, demotes immediately on faults.
- `TorqueSafetyEnvelope`: preserves v2 caps and rate-limited recovery.

### Controller Tests

- V3 initializes on native torque cars.
- V3 initializes on PID/non-angle cars with synthetic model.
- V3 refuses angle-control cars.
- Outputs and telemetry remain finite.
- Pre-convergence synthetic mode cannot exceed limited authority.
- Learned synthetic mode can reach full normalized authority after convergence.
- Assist and learned residuals are disabled or capped before confidence.
- Confidence collapse reduces authority or activates fallback.
- Sign conflict, near-ISO lateral acceleration, bump disturbance, override, release, and same-sign unwind still shape output.

### Integration Tests

- `TorqueControlTune=3.0` selects v3.
- `TorqueControlTune=3.0` is no longer rewritten to `2.0`.
- `TorqueControlTune=2.0` still selects v2.
- Invalid tune values preserve existing fallback behavior.
- `latcontrol_torque_versions.json` and Sunnylink metadata stay synchronized.
- `controlsd` updates v3 model/lateral-lag state even for PID-origin vehicles.
- `controlsState.lateralControlState.torqueState` is published when v3 is selected.

### Platform And Replay Validation

- Native torque Toyota smoke test.
- Native/nonlinear GM smoke test.
- Honda/Hyundai/Rivian live-params smoke tests where available.
- At least one PID/non-angle synthetic-mode test.
- Drive Lab route/replay validation for real steering events before making v3 default.
- Seeded lateral fuzzing for residual spikes, oscillation, sign conflict, and authority demotion.

## Rollout Plan

1. Hidden/dev-only `v3.0`; default remains `2.0`.
2. Exposed opt-in with telemetry warnings and conservative authority ramp.
3. Full normalized authority after convergence, still opt-in.
4. Consider default only after route/replay evidence across multiple native and synthetic platforms.

## Open Implementation Notes

- Keep the first implementation modular. `LatControlTorqueV3` should orchestrate small components rather than become a large monolithic file.
- Prefer simple bounded estimators before nonlinear residuals. The nonlinear term should be optional and capped.
- Avoid persisted learned state until runtime learning behavior is validated.
- Use existing torque v2 shaper and assist modules where possible, but gate assist more strictly by model confidence.
- Keep the old controller available for fallback instead of trying to synthesize every behavior in v3 on day one.
