# Refresh-aware earners/players forecaster

Forecast for every GTA VI PS5 trophy its earn rate on Exophase at the target
time, as five quantiles (5%, 25%, 50%, 75%, 95%).

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01` ($250, 20 paid
places). Program: `node main.ts` (Node 22, standard library only).

## Why earners / players, not rates

Exophase shows rates as of its last refresh, not live truth. On launch night
the page sat unchanged for hours (638 players tracked), then refreshed:
3,868 players tracked and every rate but the 100% one dropped. A refresh is
an event: players jump, rates fall because the new players have not earned
the trophies yet.

So this program works in earners/players space:

- earners `E = rate * P / 100`, rate at target = `100 * E(target) / P(target)`.

## How it works

1. **Collapse history into refresh events.** Consecutive identical reads are
   one stale page; they become a single event stamped at the last read of the
   run. A read is a new event when `players_tracked` changes or any rate
   changes. Null rates carry forward from the last known value.
2. **Rich history (2+ distinct events, the final):** fit power laws —
   `log P` and `log E_i` per trophy against `log hours since release`
   (19 Nov 2026 00:00 UTC) — by least squares and extrapolate to the target.
   Implied daily player growth is capped at 3x/day. Trophy slopes shrink
   toward the cross-trophy median slope when a trophy has fewer than 4
   events, so thin trophies cannot explode.
3. **Flat history (1 event, the practice cases):** assume one refresh before
   the target. Players jump by `G = 6.06` (3868/638, the first launch-night
   refresh) while earners grow by a per-group factor `g_i` from the trophy's
   last rate: >= 50% → `G` (near-saturated, earners track players), 2–50% →
   2.0, 0.5–2% → 1.5 (early story), 0.1–0.5% → 1.15 (mid), < 0.1% → 1.02
   (rare / late-game). Median = `rate_last * g_i / G`.
4. **Quantile spread on the logit scale:** `q = 100 * sigmoid(logit(m) +
   z_tau * sigma)` with `z = (-1.645, -0.674, 0, 0.674, 1.645)` and
   `sigma = sqrt(0.15^2 + 0.25^2 * days + sigma_refresh^2)`, where
   `sigma_refresh` is 0.35 for the flat branch (refresh anticipated, size
   unknown) and 0.10 for the rich branch (refreshes already visible).
   Quantiles are rounded to 4 decimals, clamped to [0.01, 99.99], and forced
   non-decreasing.

## Robustness

The whole per-case forecast runs inside a try/catch: any failure falls back
to the house baseline (last rate + random-walk spread), so a case is never
invalid. Missing or unreadable case files are skipped, never fatal. The
program is deterministic — same input, same output bytes — and runs in well
under the 8-second limit (tens of milliseconds per case).

## Run locally

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<tmp> node main.ts
```

Reads every `*.in` in the input directory (name order) and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}`.
