# Settings UI: information-architecture reimagining

Status: proposed
Date: 2026-06-13
Relates to: [schema-driven rendering ADR](2026-06-13-settings-ui-schema-driven-rendering.md),
[restart plan](../plans/2026-06-12-fork-restart-reimplementation.md).

> This is a design proposal to react to, not an implementation plan. It assumes the
> schema-driven rendering ADR (steering / visuals / cruise / display already live) as
> the foundation. Open questions for the owner are collected at the end.

## Context

The settings UI is **16 flat top-level panels** (Device, Network, sunnylink, Toggles,
Software, Models, Steering, Cruise, Visuals, Display, OSM, Trips, Vehicle, Firehose,
Developer, + a disabled Navigation). That flat list has structural problems we hit
directly while converting it:

1. **Duplication.** Longitudinal settings (`ExperimentalMode`, `DisengageOnAccelerator`,
   `LongitudinalPersonality`) live in **both** Toggles and Cruise — the exact thing that
   blocked mounting schema-cruise.
2. **Split concerns.** Visuals and Display are two panels for "the screen." Device,
   Network, and Software are three panels for "the system."
3. **Top-level overload.** 7 of 16 panels are device/system plumbing competing for
   first-class space with the driving features people actually tune.
4. **No global search.** You must already know which panel a setting lives in.
5. **Inconsistent custom panels** — each device/system panel is its own hand-rolled UI.

What changed: settings are now **data** (the schema-driven foundation). Reorganizing,
consolidating, and folding pages is now *editing the schema + a small amount of nav/
search code* — not rewriting fifteen Python panels. That is what makes a reimagining
tractable rather than a year-long rewrite.

## Decision

### 1. Group the 16 panels into ~7 categories with drill-down

| Category | Contents (folded from) |
|---|---|
| **Driving** | **Lateral** — MADS, lane change, torque, NNLC (Steering). **Longitudinal** — cruise, ACC + custom-ACC, speed limits, personality, experimental mode, disengage-on-accel, ICBM, SCC (Cruise + the longitudinal half of Toggles). |
| **Interface** | **On-road** — chevron metrics, dev UI, alerts, blind-spot, rainbow (Visuals). **Screen** — brightness, timeouts, interactivity (Display). **General** — LDW, recording, units (the rest of Toggles). |
| **Vehicle** | brand / platform selection (Vehicle) |
| **Models** | driving model selection + download (Models) |
| **Navigation** | maps (OSM), trips, nav |
| **System** | device, network / wifi, software / updates, storage |
| **Cloud & Developer** | sunnylink, firehose / data sharing, Tailscale, advanced/experimental |

The wins are concrete: longitudinal is **unified** (duplication gone), "the screen" is
**one** place, and the system plumbing collapses from 7 top-level entries to 1 category.

### 2. Global search as a first-class entry point

Settings are data, so a search index over every schema item (title, description,
keywords, the param key) is cheap. A search field at the top of settings → results jump
straight to the setting and highlight it in place. **This is what makes depth navigable**
— grouping into categories is only comfortable if search bypasses the hierarchy when you
know what you want.

### 3. Revamp the device/system panels as a custom-widget library

The non-declarative panels become reusable `widget: custom` components — **rebuilt
cleanly**, not ported as-is:

- WiFi manager (Network) · Model picker + download progress (Models) · Branch selector
  (Software) · Tailscale install/login flow (Developer) · Brand/platform selector
  (Vehicle) · Driver-camera preview + confirm/reset dialogs (Device) · pairing/backup
  (sunnylink).

Each is built once, registered, and rendered inside the same schema-driven shell, so the
whole settings UI finally looks and behaves like one thing.

### 4. The schema gains a category grouping

Add a `category` to each panel (or a top-level `categories` block) in `settings_ui_src/`.
The compiled schema then expresses the full IA — and since the cloud/mobile frontend
consumes the same schema, **the reorganized IA is shared** device↔cloud for free.

### 5. Navigation shell

A category landing (7 cards/rows) replaces the 16-item sidebar; each opens a category
page that reuses `SchemaNavLayout` for its sections + sub-panels; a search overlay sits
above both. The existing per-panel renderers (`SchemaPanelLayout` / `SchemaNavLayout`)
are the building blocks — this is mostly a new top-level shell + search, not new
per-control rendering.

## Prerequisite: a visual loop

A reimagining is a **visual** project (hierarchy, density, look) and the renderer has no
headless way to verify pixels — every conversion so far shipped on parity tests + process
health checks. Before building IA/visual changes we must close that loop:

- **Preferred:** render the settings UI headless locally (Xvfb + software GL) and
  screenshot it — fast iteration, and it retroactively verifies the four panels already
  shipped blind.
- **Fallback:** add a `take_screenshot` trigger to the device UI; screenshots come back
  over SCP (one deploy cycle per look).

Treat this as phase 0 — non-negotiable for design work.

## Rollout (phased, flag-gated, visually validated)

0. **Visual loop** — establish local render or device screenshot.
1. **Nav shell** — category landing + drill-down, behind a flag, alongside today's flat
   sidebar (no settings move yet).
2. **Search** — index + overlay.
3. **Regroup the declarative panels** — they're already schema-driven; this is schema
   `category` edits + dropping them into the shell. Dedup longitudinal here.
4. **Custom-widget library** — one component at a time, each converting a device/system
   panel into the shell, validated on-device.
5. **Flip the flag**, retire the flat sidebar and the hand-coded panels.

Each phase is independently shippable and reversible.

## Open questions (owner)

- **Category set + names** — is the 7-way split above right? (e.g. should Navigation fold
  into Interface? Cloud and Developer split or merged?)
- **Longitudinal home** — confirm longitudinal consolidates under Driving and is *removed*
  from Toggles (vs. the current temporary overlap).
- **Scope** — full reimagining, or stop at the high-value consolidations (unify Driving,
  merge Visuals+Display, add search) and leave the system panels as-is?
- **Visual loop** — is the local-render investment worth it, or is the device-screenshot
  cycle acceptable?

## Consequences

- **Pro:** a coherent IA; global search; one consistent renderer; duplication gone; adding
  a setting is a schema edit; the cloud frontend inherits the same IA.
- **Con:** large effort; the custom widgets are genuinely new UI work; needs visual
  iteration throughout; settings move, so users must relearn locations; the schema needs a
  grouping concept and the cloud frontend must honor it.
- **Risk:** the visual-verification gap (mitigated by phase 0); scope creep; churn during a
  long migration (mitigated by flag-gating and shipping each phase).

## Alternatives

- **Keep the flat IA, just finish converting panels.** Lowest effort; leaves the
  duplication, overload, and no-search problems in place. Rejected as the *end* state, fine
  as a stopping point.
- **Targeted consolidation only** — unify Driving (kill the Toggles/Cruise split), merge
  Visuals+Display, add search; leave the system panels flat and hand-coded. ~60% of the
  value for ~20% of the effort, and no custom-widget library needed. A strong middle option.
- **Full reimagining** (this ADR) — the complete vision; highest value, highest cost.
