# sunnypilot Longitudinal Context

This file records the project language for longitudinal planning, stack selection, and custom longitudinal behavior.

## Language

**Longitudinal Planner**:
The scene-and-arbitration boundary that turns model, radar, map, speed, and stack evidence into a longitudinal plan.
_Avoid_: MPC, controller, policy stack

**Longitudinal MPC**:
The physical trajectory solver used for cruise and lead-follow feasibility.
_Avoid_: planner, decision layer, policy engine

**Lead MPC**:
Longitudinal MPC output for confirmed lead-following physical hazard handling.
_Avoid_: advisory candidate, progress policy

**Custom Stack**:
A selectable longitudinal policy boundary that can shape planner behavior without redefining the baseline stack.
_Avoid_: MPC, controller tune

**Longitudinal Decision Layer**:
The planner-internal candidate arbiter used by custom stacks.
_Avoid_: user mode, stack selector, product toggle

**sunnypilot-current**:
The baseline stack that preserves current sunnypilot behavior and must not be changed by custom-stack internals.
_Avoid_: upstream, stock, openpilot-current

**custom-recommended**:
The recommended custom-stack alias, distinct from the default baseline stack.
_Avoid_: default custom, latest custom

**Promotion Gate**:
The decision threshold for changing which custom stack the recommended alias names.
_Avoid_: deploy check, smoke test

**custom-2.0**:
The selectable custom stack whose product promise is assertive progress without relaxing explicit safety caps.
_Avoid_: v1, recommended

**AlphaLongitudinalEnabled**:
The user-facing gas/brake takeover gate.
_Avoid_: stack selector, policy toggle

**Stack Selection**:
The latched choice of longitudinal stack for an onroad cycle.
_Avoid_: live stack switch, mode toggle

**Longitudinal Mode**:
The top-level user behavior choice among **ACC**, **E2E**, and **SCC**. It decides which classes of evidence may affect actuation before planner candidates are built.
_Avoid_: legacy DEC toggle, stack selection

**SCC Mode**:
The public smart-cruise mode that can select ACC-like or E2E-like behavior from explicit evidence while using curve, map, speed, or stop cues only inside mode-specific boundaries.
_Avoid_: renamed DEC, SCC Vision toggle

**One-Pedal Longitudinal**:
A custom-2.0 mode where driver lift-off makes cruise speed an acceleration ceiling instead of a speed-hold target.
_Avoid_: regen mode, follow-gap mode

**Lift-Off Coast**:
The one-pedal no-hazard policy that suppresses progress acceleration and commanded braking so the vehicle rolls down naturally.
_Avoid_: Free Coast, brake-light-safe regen

**Terminal Creep**:
The one-pedal low-speed crawl policy that holds a small crawl target only when rolling or clear evidence authorizes movement.
_Avoid_: launch, autonomous creep

**Low-Speed Terminal Stop**:
The one-pedal full-stop policy that gently stops and holds below parking-lot speed without declaring a stop hazard.
_Avoid_: Stop Approach, model stop

**Temporary Cruise Hold**:
A driver-requested one-pedal escape state where normal custom-2.0 cruise and advisory behavior resumes until pedal input or disengagement.
_Avoid_: disabling one-pedal, stack switch

**Fail-closed**:
Custom-stack fault handling that requests immediate disable instead of silently falling back to baseline output.
_Avoid_: fallback, fail-open

**Intent**:
A named longitudinal objective used by custom-stack arbitration.
_Avoid_: source, event, mode

**Safety Cap**:
A hard cap that can only restrict output.
_Avoid_: comfort target, advisory target

**Stop Approach**:
High-confidence stop-threat handling from model or radar-confirmed evidence.
_Avoid_: slowdown, braking case

**Map Caution**:
Preparatory response to OSM or mapd hazards before model or radar confirmation.
_Avoid_: map stop, traffic-control stop

**Comfort Relax**:
Conservative softening of advisory braking when runway and safety margins are clear.
_Avoid_: ignore advisory, disable braking

**Progress Core**:
The custom-2.0 policy envelope for progress that is already authorized by lead-confirmed or runway-confirmed evidence.
_Avoid_: launch only, acceleration boost

**Stop/go Intent**:
Planner-owned state for stopped-gap creep, gap fill, and launch candidate generation.
_Avoid_: controller launch, MPC launch

**Planner Seed Candidate**:
A planner-owned candidate used to seed custom-stack arbitration from existing retained behavior.
_Avoid_: custom v1 candidate, legacy stack candidate

**Runway-confirmed**:
Evidence that there is enough clear distance or time to apply positive progress acceleration.
_Avoid_: clear road, no obstacle

**Claim Type**:
The kind of longitudinal assertion a change makes before it is assigned to a boundary.
_Avoid_: file choice, feature label

**Lead-confirmed Progress**:
Positive progress authorized by stable or confirmed lead evidence.
_Avoid_: lead status, weak lead progress

**Closing-rate Risk**:
Lead-follow risk caused by ego speed consuming the lead gap faster than the lead trajectory can safely absorb.
_Avoid_: speed-up annoyance, custom policy preference

**Lead Speed-up Guard**:
A planner-level cap that blocks non-lead speed-up seeds when a close lead is being closed on.
_Avoid_: lead braking, MPC replacement

**Lead Flicker Safety Cap**:
A custom-stack **Safety Cap** that blocks positive acceleration during low-confidence or recently lost risky lead evidence without treating that evidence as **Lead-confirmed Progress**.
_Avoid_: lead persistence, hidden lead MPC, flicker progress

**Slower Lead Approach**:
Lead-follow runway comfort for approaching a slower moving lead without declaring stop intent.
_Avoid_: Stop Approach, stopped-lead crawl, lead speed-up guard

**Standard Personality**:
The driving-personality anchor for custom-2.0 comfort and progress behavior.
_Avoid_: default mode, manual profile

**Routine Stop Comfort**:
The non-urgent stop style that favors early mild deceleration while runway margin exists.
_Avoid_: weak braking, urgent stop

**Urgent Stop Capability**:
The stop behavior that can use stronger braking when confirmed evidence and finite runway shortage require it.
_Avoid_: panic braking, routine stop

**Stop Target**:
The desired terminal gap from a confirmed stopped lead.
_Avoid_: model endpoint, zero gap

**Stop Target Buffer**:
Positive extra gap above the Stop Target used to shape crawl, follow, and soft-stop behavior.
_Avoid_: following distance, bumper gap

**Clear Launch Pulse**:
A brief no-lead launch acceleration when fresh clear-path evidence supports progress.
_Avoid_: no-lead creep, general speed-up

**Lead Pullaway Pulse**:
A brief confirmed-lead launch acceleration when the lead is moving away and the gap is opening.
_Avoid_: lead-follow target, crawl

**Excess Gap Closure**:
Driver-like progress that closes a gap larger than the active follow target without lowering the steady-state target.
_Avoid_: tighter following, tailgating

**Free Coast**:
Harmless overspeed coasting that lets speed bleed off naturally before smooth recovery braking is needed.
_Avoid_: speed-limit policy, uncontrolled overspeed

**Driver-Like Curve Speed**:
Curve-speed behavior that may relax advisory deceleration only inside lateral-accel and path-confidence limits.
_Avoid_: ignoring curves, lateral safety

## Relationships

- A **Longitudinal Planner** may use **Longitudinal MPC** as a lower-level solver.
- **Lead MPC** is authoritative physical-hazard evidence when lead confidence is sufficient.
- Low-confidence, flickering, or transitioning lead evidence may restrict or hold acceleration, but only **Lead-confirmed Progress** may authorize positive lead progress.
- **Closing-rate Risk** belongs to **Longitudinal MPC** or planner lead guards rather than a competing **Custom Stack** physics model.
- A **Lead Speed-up Guard** caps planner seeds; **Longitudinal MPC** owns lead braking, time-gap, danger-gap, and stop-runway shaping.
- A **Lead Flicker Safety Cap** may hold acceleration during risky flicker; it does not create **Lead-confirmed Progress** or a **Lead MPC** physical hazard.
- A **Slower Lead Approach** is **Closing-rate Risk** handling inside lead-follow behavior; it may use bounded moving-follow runway comfort, but it is not **Stop Approach**.
- **Stop/go Intent** belongs to the **Longitudinal Planner**; **Longitudinal MPC** owns physical lead/runway constraints and controller logic only shapes release.
- A **Planner Seed Candidate** is planner-owned input to custom-stack arbitration, not a selectable stack version.
- A **Custom Stack** may shape **Longitudinal Planner** behavior, but **sunnypilot-current** remains behavior-isolated.
- A **Lead Speed-up Guard** is custom-stack behavior unless a separate baseline change is explicitly accepted.
- **custom-2.0** is the only selectable **Custom Stack** until **custom-recommended** is promoted.
- **custom-2.0** progress helpers express policy over confirmed evidence; they are not independent lead-physics authority.
- **One-Pedal Longitudinal** is a **Custom Stack** mode inside **custom-2.0**, not a separate **Stack Selection** value.
- **Lift-Off Coast** suppresses non-hazard **Progress Core** behavior and advisory braking; **Lead MPC** and confident stop evidence remain authoritative for physical hazards.
- **Terminal Creep** and **Low-Speed Terminal Stop** are terminal one-pedal policies; they are not **Stop Approach** and do not change moving-follow target gaps.
- **Temporary Cruise Hold** restores normal **custom-2.0** behavior without changing **Stack Selection**.
- A **Promotion Gate** can change the **custom-recommended** alias without changing the default **sunnypilot-current** baseline.
- **Standard Personality** anchors custom-2.0 behavior; other personalities may scale comfort and progress, but not **Safety Caps**.
- **Routine Stop Comfort** and **Urgent Stop Capability** are separate: routine stops favor comfort, while urgent stops preserve strong braking capability.
- A **Stop Target Buffer** belongs to terminal stopped-lead crawl behavior and does not lower normal moving-follow gaps; **Slower Lead Approach** uses a separate bounded moving-follow floor.
- **Clear Launch Pulse**, **Lead Pullaway Pulse**, **Excess Gap Closure**, and **Free Coast** are progress or comfort policies that yield to **Safety Caps**, confirmed stop evidence, and **Lead MPC** hazard handling.
- **Driver-Like Curve Speed** cannot override lateral-accel or path-confidence limits.
- **AlphaLongitudinalEnabled** gates gas/brake takeover separately from **Stack Selection**.
- SCC is a **Longitudinal Mode**; SCC Vision, SCC Map, SLA, and OSM are signal sources whose outputs are only built when the active mode allows them.
- The **Longitudinal Decision Layer** is internal to custom-stack behavior and disabled for **sunnypilot-current**.
- A **Claim Type** is classified before code ownership: physical feasibility belongs to MPC or lead guards, stop/go state belongs to the planner, policy trade-offs belong to custom-stack arbitration, actuator feel belongs to controller logic, and source interpretation belongs to signal providers.

## Example Dialogue

> **Dev:** "Should this closing-rate rule live in **Longitudinal MPC** or **custom-2.0**?"
> **Domain expert:** "If it changes physical lead-follow feasibility, it belongs near **Longitudinal MPC**; if it chooses between valid policy envelopes, it belongs in the **Custom Stack**."
