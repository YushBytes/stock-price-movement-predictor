# Stock Price Movement Predictor — Frontend

A static presentation dashboard for the [Stock Price Movement Predictor](../README.md) Option A
machine-learning experiment. It presents the project's real, already-computed results — dataset
summary, baselines, raw and engineered models, technical indicators, leakage-prevention design, and
the final four-way comparison — in a form a recruiter or evaluator can skim in 1–2 minutes.

**This is a presentation layer only. It performs no live ML inference and makes no prediction on
its own.** Every number and the one embedded figure (`predicted_vs_actual.png`) are point-in-time
transcriptions/copies of the verified results in the root project's `results/*.csv` and
`figures/predicted_vs_actual.png` — see the source comments at the top of each file in `src/data/`
for exactly where each value came from.

## Technology

- React 19
- Vite
- Tailwind CSS 3

No chart library, state manager, or router is used — the data is simple enough to render directly
with styled HTML, and the site is a single scrollable page with anchor navigation.

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

Starts a local dev server (default `http://localhost:5173`) with hot module reload.

## Production Build

```bash
npm run build
```

Outputs a static site to `dist/`, deployable to any static host.

## Preview the Production Build

```bash
npm run preview
```

Serves the contents of `dist/` locally, exactly as it would be served in production.

## Updating the Data

If the underlying ML experiment's results ever change (they should not, without a documented new
experiment run), update the corresponding file in `src/data/` by hand, re-transcribing the new
values from the root project's `results/*.csv` — never compute or estimate a replacement value in
the frontend itself. If the prediction figure changes, replace
`src/assets/predicted_vs_actual.png` with a fresh copy of the root project's
`figures/predicted_vs_actual.png` — never redraw or regenerate it from within the frontend.
