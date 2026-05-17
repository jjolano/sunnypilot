# sunnypilot Longitudinal Stack Context

This file records the project language used by the retained longitudinal stack selector branch.

## Glossary

- `sunnypilot-current`: Baseline stack. It preserves current sunnypilot behavior and must not be changed by custom-stack internals.
- `custom-recommended`: Alias that resolves per platform to a promoted custom stack. It must remain unchanged until promotion gates pass.
- `custom-1.0`: Current retained custom longitudinal stack. It keeps the existing custom candidate/fallback behavior.
- `custom-2.0`: Full replacement custom longitudinal stack. Its product promise is assertive progress without relaxing explicit safety caps.
- `AlphaLongitudinalEnabled`: User-facing gas/brake takeover gate. Stack selection does not replace this safety boundary.
- Stack selection: The latched choice stored in `LongitudinalStack`. Changes require an onroad cycle.
- Fail-closed: Custom stack fault handling that requests immediate disable instead of silently falling back to baseline output.
- Intent: A named longitudinal objective used by `custom-2.0` arbitration.
- Safety cap: A hard cap that can only restrict output. It outranks every progress or comfort intent.
- Stop approach: High-confidence stop-threat handling from model/radar-confirmed evidence.
- Map caution: Preparatory response to OSM/mapd hazards or traffic controls before model/radar confirmation.
- Comfort relax: Conservative softening of advisory braking when runway and safety margins are clear.
- Progress core: The first custom-2.0 envelope for no-lead launch, lead pullaway, excess-gap closing, and lead-loss recovery.
- Runway-confirmed: Evidence that there is enough clear distance/time to apply positive progress acceleration.

## Stack Boundary

- `sunnypilot-current` is the only baseline stack and must stay behavior-isolated.
- `custom-1.0` keeps the retained custom behavior and baseline shadow/fallback fields during the first v2 rollout stage.
- `custom-2.0` owns policy, envelopes, scoring, fail-closed validation, and selected-candidate telemetry.
- SCC, SLA, and OSM modules remain signal sources and user-facing feature switches; custom stack selection changes how their outputs are used.
- `LongitudinalDecisionLayer` has no user toggle. It is internal to custom-stack behavior and disabled for `sunnypilot-current`.
