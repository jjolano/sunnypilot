---
name: sunnypilot
description: Calibrated, safety-minded in-car UI for custom sunnypilot driving-assistance workflows.
colors:
  cockpit-black: "#121212"
  dialog-black: "#1B1B1B"
  surface-low: "#1E1E1E"
  surface-disabled: "#272727"
  surface-neutral: "#333333"
  surface-base: "#393939"
  surface-pressed: "#4A4A4A"
  border-muted: "#969696C8"
  text-primary: "#FFFFFF"
  text-strong: "#E4E4E4"
  text-dialog: "#C9C9C9"
  text-secondary: "#808080"
  text-disabled: "#585858"
  primary: "#465DEA"
  primary-pressed: "#3049F4"
  state-on: "#1C65BA"
  state-on-hover: "#114E96"
  state-on-disabled: "#25466B"
  success: "#00F100"
  info: "#0086E9"
  warning: "#FFD500"
  danger: "#E22C2C"
  danger-pressed: "#FF2424"
  link: "#1E79E8"
typography:
  display:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "90px"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "0"
  headline:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "70px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "0"
  title:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "60px"
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: "0"
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "50px"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0"
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "40px"
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: "0"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "40px"
    fontWeight: 500
    lineHeight: 1.2
rounded:
  control: "10px"
  control-large: "20px"
  dialog-search: "32px"
  pill: "999px"
spacing:
  hairline: "1px"
  xs: "10px"
  sm: "20px"
  md: "40px"
  lg: "50px"
  xl: "70px"
  xxl: "100px"
  dialog: "200px"
  row-height: "170px"
  control-height: "120px"
  action-height: "150px"
  dialog-action-height: "160px"
components:
  button-neutral:
    backgroundColor: "{colors.surface-neutral}"
    textColor: "{colors.text-strong}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 48px"
    height: "{spacing.control-height}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.text-strong}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 48px"
    height: "{spacing.control-height}"
  button-primary-pressed:
    backgroundColor: "{colors.primary-pressed}"
    textColor: "{colors.text-strong}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 48px"
    height: "{spacing.control-height}"
  button-danger:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.text-strong}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 48px"
    height: "{spacing.control-height}"
  toggle-on:
    backgroundColor: "{colors.state-on}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.pill}"
    width: "210px"
    height: "100px"
  toggle-off:
    backgroundColor: "{colors.surface-base}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.pill}"
    width: "210px"
    height: "100px"
  list-row:
    backgroundColor: "{colors.cockpit-black}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    padding: "0 20px"
    height: "{spacing.row-height}"
  dialog-panel:
    backgroundColor: "{colors.dialog-black}"
    textColor: "{colors.text-dialog}"
    padding: "50px"
---

# Design System: sunnypilot

## Overview

**Creative North Star: "Calibrated Cockpit"**

sunnypilot's UI is a native raylib product surface for a car computer, not a decorative web app. It should feel like a calibrated instrument panel: dark, legible, direct, and tuned for the moment when a driver or installer needs to understand state quickly. The interface earns its sunnypilot distinction through confident blue action states, clear settings patterns, and consistent control geometry rather than ornamental branding.

The system is large-scale and touch-first. Rows, buttons, toggles, dialogs, and search controls use generous physical targets, solid fills, and immediate pressed/disabled feedback. Typography is Inter-led, high-contrast, and practical; display personality is reserved for non-control moments, never for safety-critical labels.

This design system explicitly rejects the PRODUCT.md anti-references: flashy, gamified, or decorative safety-critical surfaces; generic SaaS/AI tropes; ambiguous custom controls; low-contrast gray-on-dark body text; novelty motion; fork branding that distracts from driving context; and settings layouts that make experimental or dangerous adjustments look casual.

**Key Characteristics:**

- Dark cockpit surfaces with high-contrast text and restrained blue identity/action states.
- Large touch targets built around 120–170px control rhythms for in-car use.
- Flat by default: state, selection, overlays, and tonal layers do the work of depth.
- Conservative component behavior: every setting should show current state, enabled state, and consequence before action.
- sunnypilot distinction stays inside the control language; identity never competes with safety.

## Colors

The palette is **Instrument blues**: charcoal instrument-panel neutrals, crisp action blues, and status colors used only when the state carries meaning.

### Primary

- **Sunnypilot Action Blue** (`#465DEA`): primary dialog buttons, selected rows, and decisive actions that move a workflow forward.
- **Pressed Action Blue** (`#3049F4`): tactile pressed feedback for primary actions. It should appear only during interaction, not as resting decoration.
- **Drive State Blue** (`#1C65BA`): sunnypilot toggle-on and active-state identity. Use it for an on/enabled state where the user needs instant recognition.
- **Deep State Blue** (`#114E96`): hover/pressed darkening for active blue controls.
- **Disabled State Blue** (`#25466B`): on-but-disabled state; keep it visibly muted so it cannot be mistaken for actionable.

### Secondary

- **Proceed Green** (`#00F100`): positive vehicle/status indication only. Do not use it for generic decoration.
- **Route Info Blue** (`#0086E9`): vehicle-description/info blue and progress/link-adjacent affordances.
- **Caution Yellow** (`#FFD500`): warning, attention, or classification signal.
- **Critical Red** (`#E22C2C`) and **Pressed Critical Red** (`#FF2424`): destructive or safety-critical actions. Red must be rare and never share a layout with casual affordances.

### Neutral

- **Cockpit Black** (`#121212`): deepest system background and disabled button floor.
- **Dialog Black** (`#1B1B1B`): confirm-dialog panel surface.
- **Surface Low** (`#1E1E1E`): disabled/dark component fill.
- **Disabled Surface** (`#272727`): disabled-off fills and inactive regions.
- **Control Charcoal** (`#333333`): neutral buttons and keyboard keys.
- **Base Charcoal** (`#393939`): sunnypilot list controls, toggle-off track, and base panel areas.
- **Pressed Charcoal** (`#4A4A4A`): pressed feedback for neutral controls.
- **Muted Border** (`#969696C8`): search borders and thin outline affordances.
- **Primary Text** (`#FFFFFF`), **Strong Text** (`#E4E4E4`), and **Dialog Text** (`#C9C9C9`): the normal readable text ramp.
- **Secondary Text** (`#808080`) and **Disabled Text** (`#585858`): metadata/disabled treatment only. Never use these for explanatory body copy unless contrast is verified at the rendered size.

### Named Rules

**The Blue Means State Rule.** Blue is for primary action, active selection, or on-state identity. If blue is used only to decorate a surface, remove it.

**The Status Colors Are Evidence Rule.** Green, yellow, and red are reserved for actual status, warning, or destructive meaning. They are not a palette expansion.

**The Gray Text Audit Rule.** `#808080` on dark charcoal is acceptable only for large, secondary metadata; body text and safety explanations must use `#C9C9C9`, `#E4E4E4`, or `#FFFFFF`.

## Typography

**Display Font:** Inter Bold, with system sans fallback.
**Body Font:** Inter Regular/Medium, with system sans fallback.
**Label/Mono Font:** JetBrains Mono for diagnostic or code-like content only; Unifont/Noto Color Emoji provide fallback coverage.

**Character:** The type system is product-native: large, direct, and readable at glance distance. It uses weight and scale for hierarchy, not decorative pairing. Audiowide exists in assets but must not be used for settings labels, buttons, or data.

### Hierarchy

- **Display** (700, 90px, 1.05): rare top-level or keyboard/dialog title moments where the UI needs a single dominant statement.
- **Headline** (700, 70–75px, 1.1): dialog titles, confirm prompts, and major modal headings.
- **Title** (500–700, 60px, 1.15): button labels, prominent control labels, and action-heavy copy.
- **Body** (400–500, 47–50px, 1.2): list item titles, right-aligned values, search text, and in-dialog explanation.
- **Label** (500, 40px, 1.15): segmented controls, compact list states, and secondary row metadata.

### Named Rules

**The Inter Carries the Product Rule.** Use Inter for product UI. Do not introduce display fonts into controls, labels, values, settings, or safety text.

**The Fixed Scale Rule.** This is a raylib device UI; use fixed pixel scales and the app's existing `SCALE` behavior instead of fluid web-style type ramps.

## Elevation

This system is **flat by default**. The existing native UI uses solid fills, rounded rectangles, thin borders, and overlays rather than shadows, blur, or glass. Depth is conveyed by panel boundaries, modal occlusion, selected-state color, pressed-state tonal shifts, and row/list structure.

### Named Rules

**The No Decorative Shadow Rule.** Shadows are prohibited as ambient decoration. If a future surface needs separation, use a darker overlay, a clear panel edge, a border, or a state color first.

**The Pressed Tone Rule.** Interaction feedback changes tone immediately: neutral controls move from `#333333` to `#4A4A4A`, primary controls from `#465DEA` to `#3049F4`, and danger controls from `#E22C2C` to `#FF2424`.

## Components

The component language is **tactile and conservative**: large controls, clear states, and no invented affordances for standard settings tasks.

### Buttons

- **Shape:** gently rounded rectangles, usually 10px radius; sunnypilot simple/list actions can use a larger 20px radius.
- **Primary:** Sunnypilot Action Blue fill with Strong Text, medium Inter label, 120–160px height depending context.
- **Neutral:** Control Charcoal at rest, Pressed Charcoal on touch, dimmed text on disabled.
- **Danger:** Critical Red only for destructive or safety-critical actions; pressed state becomes brighter, not darker, to make commitment unmistakable.
- **Hover / Focus:** native raylib primarily uses pressed feedback. Browser/live snippets should add a visible focus outline using Muted Border and avoid motion beyond short state transitions.

### Toggles

- **Shape:** pill track with circular knob. The sunnypilot toggle is 210px wide with a 100px visual track and a padded knob.
- **On / Off:** on blends from Base Charcoal to Drive State Blue; off rests on Base Charcoal. Disabled on uses Disabled State Blue and a muted knob.
- **Motion:** only state motion, never decorative entrance. The knob may animate quickly between states; reduced motion should snap without choreography.

### Option Controls

- **Style:** three-part inline control: minus button, centered value, plus button, contained inside a Base Charcoal rounded background.
- **Sizing:** 150px button height, 150px button width, 350px default value label width, 25px internal spacing, 20px container padding.
- **State:** disabled endpoints must visibly mute the minus/plus text; pressed state fills only the touched segment.

### Segmented Selectors

- **Style:** transparent track with a 2px muted charcoal border and an animated selected highlight.
- **Text:** 40px Inter Medium; disabled segments use alpha-muted text.
- **State:** selected highlight uses the checked neutral fill (`#696868` in source) rather than primary blue unless selection is safety-critical.

### List Rows

- **Structure:** 170px base row height with 20px padding. Title text sits left; values align right; toggle/simple-button actions may occupy the left and shift text to the right.
- **Descriptions:** secondary copy lives below the primary line with explicit expanded height; do not squeeze descriptions into the title row.
- **Selection:** selected tree/list rows use Action Blue; unselected options sit on disabled/dark surfaces.

### Dialogs

- **Confirm Dialog:** full dark panel inset by 200px, 50px internal margins, 160px bottom actions. Text is centered and bold for irreversible decisions.
- **Tree Dialog:** black content panel inset by 50px, 70px title, 110px search field, 135px option rows, and bottom actions. Search uses a transparent fill, 3px muted border, and magnifier icon.
- **Pairing Dialogs:** allowed light-surface exception for QR/instruction flows. Use black text, numbered dark step circles, and link blue; do not generalize this light panel into the driving UI.

### Progress Bars

- **Style:** thin blue fill over a dark chip background. Use for actual progress only, not as a static accent line.

## Do's and Don'ts

### Do:

- **Do** preserve the Calibrated Cockpit feel: dark, legible, direct, and tuned for fast comprehension.
- **Do** use existing raylib tokens from `system/ui/sunnypilot/lib/styles.py` before inventing new values.
- **Do** keep primary actions blue (`#465DEA`) and active/on states blue (`#1C65BA`) so users learn one consistent state language.
- **Do** verify contrast for every body/explanatory text placement; use `#C9C9C9`, `#E4E4E4`, or `#FFFFFF` when copy carries meaning.
- **Do** make experimental, dangerous, or tuning-related controls communicate consequence and current state before action.
- **Do** include reduced-motion behavior for any animated toggle, segmented-selector, dialog, or progress transition.

### Don't:

- **Don't** make safety-critical surfaces flashy, gamified, or decorative.
- **Don't** use generic SaaS/AI visual tropes, gradient text, glassmorphism, hero metrics, side-stripe accents, or identical decorative card grids.
- **Don't** use ambiguous custom controls for settings that can affect driving behavior.
- **Don't** use low-contrast gray-on-dark body text; `#808080` and `#585858` are secondary/disabled treatments, not paragraph colors.
- **Don't** add novelty motion, bounce, elastic easing, or choreographed page-load sequences.
- **Don't** let fork branding distract from driving context.
- **Don't** make experimental or dangerous adjustments look casual.
