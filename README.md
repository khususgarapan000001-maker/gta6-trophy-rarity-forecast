# GTA VI Trophy Rarity Forecaster — v2

Forecasts, for each of GTA VI's 55 PlayStation 5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case target.

## How it works

Everything is modelled in **earners / players** space, not in rates:
`E = rate * P / 100`, and the forecast rate is `100 * E(target) / P(target)`.
Rates are ratios of two quantities that move on their own (earners rise as
players earn trophies; players rise as new players get tracked), so separating
them is the core of the design.

**History is first collapsed into refresh events.** Exophase's page can sit
unchanged for hours, so consecutive identical reads are one stale page: a new
event starts only when `players_tracked` changes or any rate changes, and the
event is dated at the last stale read of the run. Null rates carry forward.

**Rich history (2+ events — the final):** players follow a power law in
hours since release, `log P` on `log t`, with one round of MAD outlier
rejection and alpha clamped so implied daily growth stays ≤ 3x/day. Each
trophy's earners are fit **against players** (`log E` on `log P`), not against
time — both quantities are driven by the same player inflow, so the relation
is steadier than either one's relation to time, and there is only one time
extrapolation (through `P(target)`) instead of two. Slopes shrink toward the
cross-trophy median slope when a trophy has fewer than 4 events. A second,
time-based fit (the v1 formulation) cross-checks each trophy: if the two
extrapolations disagree by more than 3x, the players-fit estimate is kept but
the trophy's sigma is widened by 0.25. Earners are clamped to `[0.5,
0.999 * P(target)]` and never below 98% of the last observed earners.

**Flat history (1 event — the practice cases):** one refresh is anticipated
before the target. Players jump by `G = 6.06` (the first launch-night refresh,
3868 / 638), earners grow by `g_i` depending on the trophy's last rate:
≥ 50% → `G` (rate holds near 100%), 0.5–50% → **1.28**, 0.1–0.5% → 1.15,
< 0.1% → 1.02. The 1.28 comes from v1's dev loss (0.167 ≈ ln(1.5 / g_true)
→ g_true ≈ 1.27); the 2–50% band, undefined in v1, is closed at 1.28.

**Spread:** quantiles sit on the logit scale,
`q = 100 * sigmoid(logit(m) + z_tau * sigma)`, with
`sigma = sqrt(0.15² + 0.25² * days + sigma_refresh²)`,
`sigma_refresh = 0.35` when a refresh is anticipated (flat) and `0.10` when
refreshes are already visible in the history (rich).

**Robustness:** every case runs inside a try/catch; on any failure the case
falls back to the house baseline (last rate, random-walk spread), so a case
is never invalid. The program is deterministic, uses only the Node standard
library, and runs in well under a second per case.

## Run

```sh
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> node main.ts
```

Reads every `*.in` in the input directory (sorted by name) and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## v2 changes vs v1

- Rich branch rebuilt: earners fit against players instead of against time,
  plus MAD outlier rejection on the players fit, shrinkage, a time-fit
  cross-check with sigma widening, and earners sanity clamps.
- Flat branch retuned: `g_early` 1.5 → 1.28; the 2–50% group gap closed
  (was an ad-hoc 2.0 in v1, now 1.28).
