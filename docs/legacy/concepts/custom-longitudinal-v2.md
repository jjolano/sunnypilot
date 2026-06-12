# Custom Longitudinal v2 Spec

## Goal

Add selectable `custom-2.0` as a full custom longitudinal stack that prioritizes assertive progress without relaxing explicit safety caps.

## Non-Goals

- Do not change the unset/default `LongitudinalStack` resolution away from `sunnypilot-current`.
- Do not require parity with the removed v1 stack.
- Do not let `sunnypilot-current` consume custom-only tuning or arbitration.
- Do not change FCW/AEB policy as part of custom-v2 style tuning.
- Do not add autonomous no-lead creep in ambiguous low-speed spaces.
- Do not treat one-pedal mode as a follow-gap override or a parallel lead-physics model.

## Stack Selection

- `LongitudinalStack` values are `sunnypilot-current`, `custom-recommended`, and `custom-2.0`.
- Unset or unknown values resolve to `sunnypilot-current`.
- `custom-recommended` resolves globally to `custom-2.0`; this is an opt-in alias and not the default baseline stack.
- Stack selection is latched and changes require an onroad cycle.
- `AlphaLongitudinalEnabled` remains the gas/brake takeover gate.

## V2 Intents

- `safety_cap`: hard restrictions from validated safety constraints.
- `stop_approach`: comfort-first stop-threat handling for true no-lead stops.
- `lead_follow`: confirmed-lead following and lead safety behavior.
- `launch`: no-lead and planner-seeded lead-pullaway progress behavior.
- `speed_policy`: coast-biased speed-limit handling.
- `curve_policy`: driver-like curve-speed behavior inside lateral-accel and path-confidence limits.
- `map_caution`: model-confirmed OSM/mapd caution only; raw map-only stops and hazards do not affect control.
- `comfort_relax`: small relax of advisory braking inside clear safety margins.
- `driver_cruise`: driver set-speed tracking with dynamic downhill/coast leeway.
- `one_pedal`: opt-in lift-off coast, terminal creep, and low-speed full-stop behavior where cruise is an acceleration ceiling.

## First-Release Tunings

- `Standard` driving personality is the manual-derived custom-v2 anchor.
- `Relaxed` and `Aggressive` scale custom-v2 comfort/progress envelopes and jerk shaping only; safety caps stay fixed.
- Routine stop comfort favors early mild decel around `-0.30` to `-0.45 m/s^2` while runway margin exists.
- Urgent stop capability requires confirmed stop evidence plus finite runway shortage; it may use strong decel clipped to planner limits and should ramp by remaining margin when possible.
- Clear Launch Pulse: brief no-lead launch pulse around `1.4-1.7 m/s^2` when fresh clear-path evidence supports progress, tapered down when weak distant model-stop ambiguity exists.
- Lead Pullaway Pulse: manual-like peak allowed only when a confirmed lead is moving away and the gap is opening; close or unstable leads use lower lead-matched behavior.
- Positive jerk is personality-scoped: Relaxed ramps softly, Standard uses aggressive positive jerk for launch/pullaway pulses, and Aggressive may also use faster progress ramps for moving speed-up and excess-gap closure.
- Normal negative jerk should be softer than the initial `-5.0 m/s^3` retreat; urgent stops, safety caps, and preserved planner lead restrictions may ramp faster when required.
- Lead motion gate uses confirmed lead motion, opening-gap evidence, or trusted lead speed; lead acceleration can relax closing guards only when corroborated by stable opening evidence.
- Slower Lead Approach uses bounded moving-follow runway comfort to coast first, then apply routine light braking when closing speed would consume the available runway to the bounded target.
- Launch speed caps: `3.0 m/s` no-lead and `5.0 m/s` lead pullaway.
- Stop Target Buffer: tight stopped-lead crawl starts around `+1.0 m` above Stop Target, follows around `+0.3 m`, and blocks positive creep at or below `0.0 m`.
- Stopped-lead approach blends from normal geometry above `3.0 m/s` to the tight Stop Target Buffer below `1.0 m/s`.
- Downhill crawl may loosen the Stop Target Buffer using existing grade or response proxies, but must not tighten below the base `+0.3 m` follow target.
- Excess Gap Closure is speed-aware and planner-approved only; it does not lower steady moving-follow gaps or create an independent lead-physics model.
- Lead-loss or transition occlusion guard is evidence-sensitive, roughly `1.0-1.5 s`.
- Free Coast plain-cruise overspeed leeway is dynamic by grade/coast context, bounded around `+5 to +10 mph`; manual set-speed reductions get prompt smooth response instead.

## Scene Behavior

- Lead-follow and lead-pullaway actuation is planner/MPC-seeded first; planner seeds and raw physical fallback candidates enter `LongitudinalDecisionCore` before `custom-2.0` shapes the selected output.
- Planner seed telemetry publishes seed context/candidate and maps seed reasons into v2 intents while preserving the raw seed reason as `selectedReason`.
- Classification-only planner seeds preserve their incoming speed, accel, and jerk trajectories.
- No-lead launch uses a tapered clear-path gate; missing or stale evidence is not clear-path permission.
- No-lead stop approach is comfort-bounded by default; `custom-2.0` may exceed the comfort bound only when confirmed stop evidence and finite stop distance require harder decel.
- Weak no-lead model slowdowns use coast or light decel first, then escalate only as runway shortage becomes real.
- After a no-lead model stop, Clear Launch Pulse remains blocked until stop evidence clears.
- Speed-limit reductions remain coast-biased; stronger braking must come from lead, stop, curve, or confirmed map-caution evidence.
- SCC vision/map curve policy may relax advisory decel only inside lateral-accel and path-confidence limits.
- OSM traffic-control prior `active` is treated as confirmed because that prior already requires model-distance confirmation. Raw map-only traffic-control or hazard cues are ignored for control.
- Confirmed map caution is cap-only and does not set stop intent or get softened by comfort relax.
- Driver brake/gas input blocks progress floors and comfort relax, while safety caps and conservative advisory caps may still apply.
- One-pedal mode replaces driver intent inside the custom-v2 decision candidates and suppresses non-hazard progress floors and advisory braking unless Temporary Cruise Hold is active.
- One-pedal Lift-Off Coast does not change Lead MPC follow-gap, danger-gap, TTC, stop-runway, FCW, or AEB behavior.
- One-pedal Creep may hold a small crawl target once rolling or when existing clear evidence authorizes movement; it must not autonomously launch from ambiguous standstill.
- One-pedal Full Stop may gently stop and hold below parking-lot speed without treating that terminal policy as Stop Approach evidence.
- Any cruise speed adjustment button enters Temporary Cruise Hold, restoring normal custom-v2 cruise/advisory behavior until gas, brake, or disengagement.
- New-lead and cut-in handling is severity-tiered: close or closing leads suppress progress immediately, while leads already opening the gap transition smoothly.
- Slower Lead Approach remains `lead_follow`, not `stop_approach`; raw radar lead evidence may cap/coast, but routine braking needs stable lead evidence or severe closing runway/TTC evidence.
- Low-confidence, flickering, or sensor-disagreed leads may suppress acceleration without custom hard braking unless Lead MPC or planner lead safety requires it.
- Driver gas/brake input suppresses applying Slower Lead Approach routine comfort shaping; lead safety constraints remain available.
- Independent stop threats block lead-pullaway progress until the stop threat clears.
- Clear Launch Pulse may fire while turning, but lateral-accel or traction caps may still restrict it.
- Low-speed under-response may receive a bounded follow-up increase inside existing accel limits when clear or lead evidence remains valid.
- Core non-finite scene inputs fail closed; invalid optional speed, curve, or map advisory targets are ignored for that cycle.
- Normal v2-owned accel changes are jerk-limited by personality-scaled tunings. Hard model stops, safety caps, and preserved planner lead restrictions may bypass comfort jerk limiting when safety requires it.

## Fail-Closed Behavior

If `custom-2.0` produces an invalid output or raises internally while enabled, latch a custom stack fault and request immediate disable. The latch resets after disengagement.

## Validation

- Selector, UI, metadata, and schema tests cover `custom-2.0` exposure.
- Unit tests cover intent names, personality scaling, progress-core caps, planner seed classification, trajectory preservation, stop tiers, launch gates, Stop Target Buffer boundaries, speed-policy coast bias, map-caution authority, driver-like curve caps, dynamic cruise leeway, scene validation, jerk limiting, and fail-closed behavior.
- One-pedal unit tests cover lift-off coast, advisory suppression, physical hazard preservation, terminal creep, low-speed full stop, Temporary Cruise Hold, and button/pedal state transitions.
- Drive Lab profiling should classify routine-vs-urgent stops by runway/required-decel evidence before route-derived stop comfort targets are changed.
- Manual-vs-custom route analysis is primary for style tuning; minimal baseline-vs-custom checks remain necessary for regression detection.
- Tests should assert scenario invariants rather than exact route-derived manual numbers.
