# sunnypilot Custom Planning Context

This context records project language for custom planning evidence, planner boundaries, and safety-related interpretation.

## Language

**Evidence Snapshot**:
A per-tick, non-authoritative summary of current ego, model-path, lead, map, and source-health evidence used by custom planner features. It is observational only and does not choose actuation targets.
_Avoid_: World model, persistent scene memory, SLAM map, route memory, planner state

**Ego Evidence**:
Current observed vehicle-state evidence about the controlled car. It excludes consumer-specific control-loop context unless that context is explicitly modeled as evidence.
_Avoid_: Control mode, planner state, lateral demand context

**Source Health**:
The trust and availability status of an evidence source at the moment an **Evidence Snapshot** is built.
_Avoid_: Sanitized default, assumed valid input

**Degraded Evidence**:
Evidence whose source is missing, stale, invalid, inconsistent, or otherwise not trustworthy enough for normal authority. Degraded Evidence may only reduce or withhold **Custom Authority**.
_Avoid_: Fallback value, fake normal reading

**Derived Evidence**:
Evidence calculated deterministically from current source evidence while preserving where it came from and how trustworthy it is. Derived Evidence may explain risk or quality, but it does not select planner intent or actuation targets.
_Avoid_: Planner output, target command, policy decision

**Custom Authority**:
The ability of a custom feature to move planner- or controller-facing behavior away from that feature's consumer-local baseline. It is not platform safety authority or actuator permission.
_Avoid_: Safety limit, panda authority, actuator command

**Fail-closed**:
Fault handling after **Custom Authority** begins that requests immediate disengagement rather than continuing with a consumer-local baseline.
_Avoid_: Silent fallback, best-effort recovery

**Fault Class**:
A stable, driver- and telemetry-safe category for a **Fail-closed** fault. It is distinct from raw exception text, which remains log-only diagnostic detail.
_Avoid_: Raw traceback, implementation-specific UI reason

**Actuation Verdict**:
A typed policy result that states whether a gated custom restriction applies to final planner output. It is independent of optional diagnostic collection.
_Avoid_: Debug dictionary, telemetry payload, log-only result

**Consumer-Local Baseline**:
The input behavior a custom feature receives before applying its own shaping. For lateral demand this is the pre-custom desired-curvature path; for custom longitudinal this is the pre-custom seed or planner output.
_Avoid_: Global stock behavior, platform limit, final command

**Model-Path Evidence**:
Current model-predicted path geometry and source quality before lateral demand processing. It is not **Processed Lateral Demand** or lateral path gating.
_Avoid_: Processed curvature, controller-facing demand, lateral command

**Processed Lateral Demand**:
The controller-facing curvature after custom conditioning and hard curvature or lateral-acceleration caps are resolved.
_Avoid_: Raw model path, pre-cap curvature, actuator command

**Conditioned Lateral Demand**:
The intermediate curvature after custom path conditioning and before hard curvature or lateral-acceleration caps. It is diagnostic evidence, not controller-facing demand.
_Avoid_: Processed Lateral Demand, actuator command

**Lead Evidence**:
Current lead-vehicle evidence summarized from the existing lead-fusion boundary. It describes lead observations and derived risk, but it is not a competing lead-physics model.
_Avoid_: Lead MPC, duplicate lead fusion, lead authority

**Model-Action Evidence**:
Current model-proposed longitudinal action evidence, such as stop or acceleration hints, along with its trust status. It is not a planner stop commitment.
_Avoid_: Stop Approach, should-stop decision, planner intent

**Engagement-Cycle Latch**:
A setting captured when controls engage and held unchanged until disengagement. Writes while engaged are accepted for the next engagement but do not alter current behavior.
_Avoid_: Live control toggle, runtime mode switch

## Relationships

- An **Evidence Snapshot** represents current-tick evidence only; volatile continuity and memory belong to **Scene Memory** or feature-specific trackers.
- **Scene Memory** may consume **Evidence Snapshot** later, but **Evidence Snapshot** does not become **Scene Memory**.
- **Ego Evidence** describes observed vehicle state; feature activity, measured demand tracking, and planner/controller context remain consumer-owned unless explicitly added as evidence.
- **Model-Path Evidence** may feed lateral demand processing, but the lateral pipeline owns **Processed Lateral Demand**.
- Hard curvature and lateral-acceleration caps transform **Conditioned Lateral Demand** into **Processed Lateral Demand**.
- Better **Model-Path Evidence** source quality may permit normal custom behavior, but poorer or unknown source quality may only reduce or withhold **Custom Authority**.
- Missing, stale, or non-finite source data is **Degraded Evidence**: custom longitudinal withholds authority without treating source health as a **Fail-closed** fault.
- A custom longitudinal fault after **Custom Authority** begins is **Fail-closed** and requests `immediateDisable`; consumer-local baseline continuation is only a pre-authority compatibility case.
- A **Fail-closed** custom-longitudinal fault shows a distinct driver alert and retains its diagnostic reason in logs.
- A **Fail-closed** alert uses a **Fault Class**; raw exception detail does not cross its interface.
- A **Fail-closed** fault ends only the current engagement; its latch resets automatically at the next engagement.
- An **Actuation Verdict** may restrict final planner output inside its existing gates; diagnostic collection neither creates nor suppresses that verdict.
- Trace logging is optional in every apply mode and changes diagnostics only.
- Existing research **Actuation Verdicts** preserve their current restrictive precedence and gate ordering while moving out of diagnostics.
- Lateral path quality, hysteresis, smoothing, and gating remain owned by lateral demand processing.
- The generic controls loop owns hard lateral caps; custom lateral modules own **Conditioned Lateral Demand**, custom evidence, and telemetry that reports raw, conditioned, and **Processed Lateral Demand** separately.
- **Model-Path Evidence** source quality covers basic source structure; richer path validity and lane-confidence thresholds remain lateral demand concerns unless explicitly refactored.
- Model-proposed desired curvature is **Model-Path Evidence**; model-proposed stop and acceleration hints are **Model-Action Evidence**.
- An **Evidence Snapshot** may feed longitudinal planning, but the longitudinal planner and custom stack own scenes, candidates, intents, and target acceleration.
- **Lead Evidence** comes from the existing radar/model lead-fusion boundary; **Evidence Snapshot** does not independently re-fuse model leads.
- Custom longitudinal lead trackers consume **Lead Evidence** only from that existing fusion; raw **Model-Path Evidence** cannot be relabeled as **Lead Evidence** to authorize **Custom Authority**.
- **Lead Evidence** may include same-tick derived risk metrics, but lead confidence, flicker, shadow, and continuity remain owned by lead trackers.
- Generic lead risk metrics may be **Derived Evidence**, but policy thresholds, actions, and lead-alignment reasons remain longitudinal policy concerns.
- **Lead Evidence** metrics must not depend on policy follow targets, personality, or comfort thresholds.
- **Model-Action Evidence** may inform longitudinal planning only after the active mode admits that evidence class.
- **Longitudinal Mode** is an **Engagement-Cycle Latch**: its evidence admission does not change while engaged, and an onroad write applies only after the next engagement.
- `selfdrived` owns the **Longitudinal Mode** capture lifecycle and publishes the active value; planner consumers do not reread its setting during an engagement.
- Model-derived stop distance is **Derived Evidence** within **Model-Action Evidence**, not a planner stop commitment.
- Degraded **Model-Action Evidence** may support caution through longitudinal trust policy, but it does not create automatic stop commitment.
- Stateful model-stop trust learning and stop commitment remain longitudinal policy concerns, not **Evidence Snapshot** concerns.
- **Evidence Snapshot** is mode-agnostic; **Longitudinal Mode** remains the authority for evidence admission.
- **Evidence Snapshot** describes observed evidence; feature toggles, user modes, and personality remain consumer-owned behavior inputs.
- Planner outputs, controller outputs, and seed targets are not **Evidence Snapshot** evidence.
- Evidence construction should prefer **Degraded Evidence** over runtime failure; unexpected evidence faults are handled by the consuming feature's existing fault policy.
- Unknown **Source Health** may support no-change evidence extraction, but it does not authorize additional **Custom Authority**.
- **Degraded Evidence** is source-specific; one degraded source does not invalidate healthier independent evidence.
- Missing or weak **Lead Evidence** does not authorize lead-confirmed progress; absence of a lead is not by itself runway confirmation.
