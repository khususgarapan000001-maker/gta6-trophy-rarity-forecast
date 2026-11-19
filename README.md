# GTA VI Trophy Rarity Forecaster v3 — "Recent-window fit + saturation-aware earners"

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles (5/25/50/75/95%)
of its Exophase earn rate at the case target, for make.fun
`gta6-trophy-rarity-forecast-01` ($250, 20 paid places).

## How it works

The forecast is built in **earners/players space**, not in rate space:
`rate = 100 * E(target) / P(target)`. History reads are first collapsed into
**refresh events** (identical consecutive reads are one stale page, dated at its
last read); null rates carry forward.

- **Flat branch** (0–1 distinct events — the practice cases): assume one refresh
  before the target. Players jump by `G = 6.06` (3,868/638, the first launch-night
  refresh); earners grow by a per-group factor (≥50%: `G`, so the rate holds;
  0.5–50%: 1.28; 0.1–0.5%: 1.15; <0.1%: 1.02).
- **Rich branch** (≥2 events — the final case): **all fits use only the recent
  24-hour window** of events, so the current growth regime is extrapolated.
  Players follow a power law in hours since release with one MAD outlier
  rejection. Earners per trophy are fitted in **two shapes** on that window and
  the shape with the clearly smaller log-space residual wins:
  - power law in players: `log E = a + b * log P`
  - saturating in time: `E = E_inf * (1 - exp(-t/tau))`, tau on a grid
    (6–288 h), chosen only when its residual is below `0.8 *` the power-law
    residual, and only with ≥4 window points.
  A v1-style time fit on the same window cross-checks each estimate; a >3x
  disagreement widens that trophy's spread by +0.25 sigma. Earners are clamped
  to never fall below 98% of the last value and never exceed players.
- **Spread**: quantiles on the logit scale around the median,
  `sigma = sqrt(0.15² + 0.25²·days + sigma_refresh²)`,
  `sigma_refresh = 0.35` (flat) / `0.10` (rich).

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> node main.ts
```

Node 22, standard library only, deterministic, no network. Reads every `*.in`
file in the input directory and writes `<id>.out` per case. Never crashes on a
readable case: a per-case try/catch falls back to the house baseline
(last rate + random-walk spread).

## Validation (design doc v3 §5)

- Scorer unit test: RULES worked example → loss 0.403404.
- Practice p00/p01: 55 rows, monotone quantiles, median/last-rate ≈ 0.211 for
  the early group, ≈ 1.0 for the 100% trophy.
- Synthetic rich-branch gate: `GRID=top50 node run-tests.mjs` in the harness
  directory (outside this repo); grid log as a diagnostic.
