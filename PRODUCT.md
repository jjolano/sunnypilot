# Product

## Register

product

## Users

sunnypilot serves drivers and owners running a dedicated comma device in supported cars, plus installers and contributors who configure, validate, and maintain the fork. They use the interface in high-attention contexts: in-car setup, offroad configuration, route review, and safety-critical driving-assist workflows where controls must be legible, predictable, and hard to misread.

## Product Purpose

This fork provides a refined sunnypilot/openpilot driving-assistance experience with custom behavior implemented cleanly on top of upstream. Success means the system feels trustworthy in the car, keeps safety constraints obvious, makes custom settings understandable, and lets contributors evolve lateral/longitudinal behavior without turning the UI into a grab bag of tuning knobs.

## Brand Personality

Distinctive, precise, safety-minded. The fork should feel recognizably sunnypilot rather than anonymous upstream, but its identity must support confidence and control instead of competing for attention.

## Anti-references

Do not make safety-critical surfaces flashy, gamified, or decorative. Avoid generic SaaS/AI visual tropes, ambiguous custom controls, low-contrast gray-on-dark text, novelty motion, and fork branding that distracts from driving context. Avoid settings layouts that make experimental or dangerous adjustments look casual.

## Design Principles

- Safety before expression: visual identity is welcome only when it preserves calm, legible, driving-safe decision-making.
- Distinction with restraint: sunnypilot-specific surfaces can feel ownable without abandoning familiar openpilot-style affordances.
- State clarity over ornament: every enabled, disabled, active, warning, and error state should be immediately distinguishable without relying on color alone.
- Predictable controls: settings and tuning affordances should make consequence, reversibility, and current state obvious before the user acts.
- Contributor-readable systems: UI patterns should be documented and reusable so custom behavior stays clean instead of becoming branch-specific clutter.

## Accessibility & Inclusion

Target WCAG AA contrast as a baseline for user-facing text and controls, with larger touch targets and conservative spacing for in-car use. Support reduced motion, avoid distracting animation, ensure color-blind-safe state communication, and prefer high-legibility choices that work under glare, vibration, quick glances, and mixed ambient light.
