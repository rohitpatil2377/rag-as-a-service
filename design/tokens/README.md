# Design Tokens

**Authority:** `tokens.json` in this directory. These tokens govern **user-facing / visual surfaces only** — sovereignty, security, tenancy and code standards are engineering constraints and live in [`../../docs/`](../../docs/) and [`../../CLAUDE.md`](../../CLAUDE.md) instead.

`tokens.json` is the **single source of truth**. The other files are derived views:

| File | Consumed by |
|---|---|
| `tokens.json` | source of truth (Style-Dictionary-style) |
| `tokens.css` | any web UI — CSS custom properties, light + dark semantic sets |
| `tailwind.preset.js` | `apps/console` (Phase-1) Tailwind config |

## Rules baked in
- **Palette** 50-30-12-6-2; Heritage Saffron = primary action, Emerald = success.
- **Type** Sora headings (H1–H6), Inter body/UI, JetBrains Mono for code/meta.
- **Spacing** 4px base scale; **radii** 6/8/12px (max 12).
- **Theme parity (§9)** switching themes remaps only semantic tokens; structure/spacing/radii never change. Never pure black — dark base is Midnight Ink `#0D1B3A`.
- **Accessibility (§21)** WCAG 2.1 AA, ≥44px targets (`--tap-target-min`), visible saffron focus ring (`--focus-ring`), never colour alone.
- **Data-viz (§20)** fixed series order (see `tokens.json → dataviz.series`); **Teal reserved for AI/predicted series only**. Applied in `infra/grafana`.

Phase-0 has no console UI, so the live consumers are `infra/grafana` (dashboard palette) and this package as the ready-to-use foundation for the Phase-1 console.
