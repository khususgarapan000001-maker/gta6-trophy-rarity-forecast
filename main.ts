// gta6-trophy-rarity-forecast-01 — v2 "Rich-branch power-law + retuned flat refresh"
// Forecasts, for each of GTA VI's 55 PS5 trophies, five quantiles (5/25/50/75/95%)
// of its Exophase earn rate at the case target.
//
// Design (from design doc v2, prose):
//   - Work in earners/players space: E = rate * P / 100; rate = 100 * E / P.
//   - Collapse consecutive stale reads into refresh events (a new event when
//     players_tracked changes or any rate changes; event time = last stale read).
//   - Rich history (>= 2 events): fit log E against log P (one time extrapolation
//     only, via P(target)), with shrinkage toward the cross-trophy median slope,
//     one round of MAD outlier rejection on the P fit, a time-fit cross-check,
//     and sanity clamps on E(target).
//   - Flat history (1 event): anticipate one refresh before target:
//     P_target = 6.06 * P_last, E_target = E_last * g_i with g_early = 1.28
//     (retuned from v1's 1.5 after the dev loss 0.167) and the 2-50% group gap closed.
//   - Quantiles on the logit scale with a sigma that grows with horizon and with
//     whether a refresh is anticipated (flat) or already observed (rich).
//   - Never crash: per-case try/catch with a house-baseline fallback.
// Node 22, standard library only, erasable TypeScript only. Deterministic.

import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------- constants
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z"); // GTA VI release, UTC
const REFRESH_JUMP = 6.06; // first launch-night refresh: 3868 / 638 players
const G_EARLY = 1.28; // flat-branch earners growth, early-story trophies (v2 retune)
const G_MID = 1.15; // flat-branch earners growth, mid trophies 0.1-0.5%
const G_RARE = 1.02; // flat-branch earners growth, rare trophies < 0.1%
const SIGMA0 = 0.15;
const SIGMA_H = 0.25; // per-day logit spread growth
const SIGMA_REFRESH_FLAT = 0.35; // refresh anticipated but unseen
const SIGMA_REFRESH_RICH = 0.1; // refreshes already visible in history
const SIGMA_UNCERTAIN_EXTRA = 0.25; // added when time-fit cross-check disagrees
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const MAX_DAILY_GROWTH = 3; // implied daily player growth cap for the P fit
const MAD_OUTLIER = 2.5; // outlier rejection threshold, in MAD units
const CROSS_CHECK_RATIO = 3; // time-fit vs players-fit disagreement threshold
const MIN_EARNERS = 0.5; // floor so log(E) is defined and E never vanishes

// ------------------------------------------------------------------- helpers
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logit(p: number): number {
  const c = clamp(p, 0.0001, 0.9999);
  return Math.log(c / (1 - c));
}

function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Least squares y = a + b*x. Returns null when the fit is degenerate.
function ols(xs: number[], ys: number[]): { a: number; b: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

function hoursSinceRelease(slotMs: number): number {
  return Math.max(0.5, (slotMs - RELEASE_MS) / 3600000);
}

// ---------------------------------------------------------------------- types
interface Trophy {
  id: number;
}

interface Read {
  slotMs: number;
  players: number;
  rates: (number | null)[];
}

interface CaseInput {
  id: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

interface Event {
  tMs: number; // time of the event (last stale read of the run)
  players: number;
  rates: (number | null)[]; // carry-forwarded, null only if never seen
}

// ------------------------------------------------------- history -> events
function toEvents(history: Read[]): Event[] {
  const events: Event[] = [];
  const lastKnown: (number | null)[] = [];
  let prevPlayers: number | null = null;
  let prevRates: (number | null)[] | null = null;

  for (const read of history) {
    const players = typeof read.players === "number" && isFinite(read.players) ? read.players : 0;
    const rates = read.rates.map((r, i) => {
      if (typeof r === "number" && isFinite(r)) {
        lastKnown[i] = r;
        return r;
      }
      return lastKnown[i] !== undefined ? lastKnown[i] : null;
    });
    // A new event when players change or any rate changes vs the previous read.
    let changed = prevPlayers === null || players !== prevPlayers;
    if (!changed && prevRates !== null) {
      for (let i = 0; i < rates.length; i++) {
        if (rates[i] !== prevRates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) {
      events.push({ tMs: read.slotMs, players, rates });
    } else if (events.length > 0) {
      // Stale page: the event stays valid through this read; keep its time fresh.
      events[events.length - 1].tMs = read.slotMs;
    }
    prevPlayers = players;
    prevRates = rates;
  }
  return events;
}

// ------------------------------------------------------------ rich branch
// Fit log P vs log hours-since-release (power law), one MAD outlier round,
// alpha clamped so implied daily growth at the last event stays <= 3x/day.
function fitPlayers(events: Event[], targetMs: number): number {
  const lastP = events[events.length - 1].players;
  if (events.length < 2 || lastP <= 0) return lastP;

  const xs = events.map((e) => Math.log(hoursSinceRelease(e.tMs)));
  const ys = events.map((e) => Math.log(Math.max(1, e.players)));
  let fit = ols(xs, ys);
  if (!fit) return lastP;

  // One round of outlier rejection: drop the single worst residual if it
  // exceeds 2.5 * MAD of the residuals, then refit once.
  const residuals = xs.map((x, i) => ys[i] - (fit!.a + fit!.b * x));
  const mad = median(residuals.map((r) => Math.abs(r - median(residuals))));
  let worst = 0;
  let worstAbs = 0;
  for (let i = 0; i < residuals.length; i++) {
    if (Math.abs(residuals[i]) > worstAbs) {
      worstAbs = Math.abs(residuals[i]);
      worst = i;
    }
  }
  if (mad > 0 && worstAbs > MAD_OUTLIER * mad && events.length > 2) {
    const xs2 = xs.filter((_, i) => i !== worst);
    const ys2 = ys.filter((_, i) => i !== worst);
    const refit = ols(xs2, ys2);
    if (refit) fit = refit;
  }

  // Clamp alpha: daily growth factor at the last event, ((t+24)/t)^alpha <= 3.
  const tLast = hoursSinceRelease(events[events.length - 1].tMs);
  const alphaMax = Math.log(MAX_DAILY_GROWTH) / Math.log((tLast + 24) / tLast);
  const alpha = clamp(fit.b, 0, alphaMax);
  const a = mean(ys) - alpha * mean(xs);
  const tTarget = hoursSinceRelease(targetMs);
  const pTarget = Math.exp(a + alpha * Math.log(tTarget));
  return isFinite(pTarget) && pTarget > 0 ? pTarget : lastP;
}

// Shrinkage helper shared by both earners fits: pull the slope toward the
// cross-trophy median slope when the trophy has few events.
function shrinkSlope(n: number, slope: number, medianSlope: number): number {
  if (n < 4) return (n * slope + 2 * medianSlope) / (n + 2);
  return slope;
}

// Fit log E vs log P for one trophy; returns the extrapolated E(target).
// Falls back to the median slope with a centroid intercept when the fit fails.
function fitEarnersVsPlayers(
  events: Event[],
  trophyIdx: number,
  pTarget: number,
  medianSlope: number,
): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const e of events) {
    const r = e.rates[trophyIdx];
    if (r === null || r === undefined || e.players <= 0) continue;
    const earners = Math.max(MIN_EARNERS, (r * e.players) / 100);
    xs.push(Math.log(Math.max(1, e.players)));
    ys.push(Math.log(earners));
  }
  if (xs.length === 0) return NaN;
  const fit = ols(xs, ys);
  let slope: number;
  let intercept: number;
  if (fit) {
    slope = shrinkSlope(xs.length, fit.b, medianSlope);
    intercept = mean(ys) - slope * mean(xs); // centroid with final slope
  } else {
    slope = medianSlope;
    intercept = mean(ys) - slope * mean(xs);
  }
  return Math.exp(intercept + slope * Math.log(Math.max(1, pTarget)));
}

// Cross-check: the v1-style fit of log E vs log hours-since-release.
function fitEarnersVsTime(
  events: Event[],
  trophyIdx: number,
  targetMs: number,
  medianSlope: number,
): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const e of events) {
    const r = e.rates[trophyIdx];
    if (r === null || r === undefined || e.players <= 0) continue;
    const earners = Math.max(MIN_EARNERS, (r * e.players) / 100);
    xs.push(Math.log(hoursSinceRelease(e.tMs)));
    ys.push(Math.log(earners));
  }
  if (xs.length === 0) return NaN;
  const fit = ols(xs, ys);
  let slope: number;
  let intercept: number;
  if (fit) {
    slope = shrinkSlope(xs.length, fit.b, medianSlope);
    intercept = mean(ys) - slope * mean(xs);
  } else {
    slope = medianSlope;
    intercept = mean(ys) - slope * mean(xs);
  }
  return Math.exp(intercept + slope * Math.log(hoursSinceRelease(targetMs)));
}

// ------------------------------------------------------------ flat branch
// One refresh before target: P jumps by REFRESH_JUMP, earners grow by g_i.
function flatGrowth(lastRate: number): number {
  if (lastRate >= 50) return REFRESH_JUMP; // near-saturated: rate holds
  if (lastRate >= 0.5) return G_EARLY; // early-story 0.5-2% and 2-50% band
  if (lastRate >= 0.1) return G_MID;
  return G_RARE;
}

// ------------------------------------------------------------ quantiles
function quantiles(medianRate: number, sigma: number): number[] {
  const m = clamp(medianRate, 0.01, 99.99);
  const base = logit(m / 100);
  const out: number[] = [];
  let prev = -Infinity;
  for (const z of Z) {
    let q = 100 * sigmoid(base + z * sigma);
    q = Math.round(clamp(q, 0.01, 99.99) * 10000) / 10000;
    if (q < prev) q = prev; // enforce non-decreasing
    out.push(q);
    prev = q;
  }
  return out;
}

// ------------------------------------------------------------ per case
function forecastCase(c: CaseInput): { id: number; q: number[] }[] {
  const n = c.trophies.length;
  const days = Math.max(0, c.horizon_minutes / 1440);
  const targetMs = Date.parse(c.target);
  const events = toEvents(c.history);

  // Last known rate per trophy (carry-forward through the history).
  const lastRate: (number | null)[] = new Array(n).fill(null);
  for (const e of events) {
    for (let i = 0; i < n; i++) {
      if (e.rates[i] !== null && e.rates[i] !== undefined) lastRate[i] = e.rates[i];
    }
  }
  const lastPlayers = events.length > 0 ? events[events.length - 1].players : 0;
  const lastEarners = (i: number): number | null => {
    if (lastRate[i] === null || lastPlayers <= 0) return null;
    return (lastRate[i]! * lastPlayers) / 100;
  };

  const rows: { id: number; q: number[] }[] = [];
  const rich = events.length >= 2;

  if (rich) {
    const pTarget = fitPlayers(events, targetMs);

    // First pass: raw per-trophy slopes of log E on log P, for the median.
    const rawSlopes: number[] = [];
    for (let i = 0; i < n; i++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const e of events) {
        const r = e.rates[i];
        if (r === null || r === undefined || e.players <= 0) continue;
        xs.push(Math.log(Math.max(1, e.players)));
        ys.push(Math.log(Math.max(MIN_EARNERS, (r * e.players) / 100)));
      }
      const fit = xs.length >= 2 ? ols(xs, ys) : null;
      if (fit) rawSlopes.push(fit.b);
    }
    const medianSlope = rawSlopes.length > 0 ? median(rawSlopes) : 1;
    // Cross-check median slope for the time fit (same shrinkage pool).
    const rawTimeSlopes: number[] = [];
    for (let i = 0; i < n; i++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const e of events) {
        const r = e.rates[i];
        if (r === null || r === undefined || e.players <= 0) continue;
        xs.push(Math.log(hoursSinceRelease(e.tMs)));
        ys.push(Math.log(Math.max(MIN_EARNERS, (r * e.players) / 100)));
      }
      const fit = xs.length >= 2 ? ols(xs, ys) : null;
      if (fit) rawTimeSlopes.push(fit.b);
    }
    const medianTimeSlope = rawTimeSlopes.length > 0 ? median(rawTimeSlopes) : 1;

    const sigmaBase = Math.sqrt(
      SIGMA0 * SIGMA0 + SIGMA_H * SIGMA_H * days + SIGMA_REFRESH_RICH * SIGMA_REFRESH_RICH,
    );
    for (let i = 0; i < n; i++) {
      let eTarget = fitEarnersVsPlayers(events, i, pTarget, medianSlope);
      let sigma = sigmaBase;
      if (!isFinite(eTarget)) {
        // No usable earners data: hold the last rate with a wide band.
        const m = lastRate[i] !== null ? lastRate[i]! : 0.5;
        rows.push({
          id: c.trophies[i].id,
          q: quantiles(m, sigmaBase + SIGMA_UNCERTAIN_EXTRA + 0.25),
        });
        continue;
      }
      const eTime = fitEarnersVsTime(events, i, targetMs, medianTimeSlope);
      if (isFinite(eTime) && eTime > 0) {
        const ratio = eTarget / eTime;
        if (ratio > CROSS_CHECK_RATIO || ratio < 1 / CROSS_CHECK_RATIO) {
          sigma = sigmaBase + SIGMA_UNCERTAIN_EXTRA; // disagree: widen the band
        }
      }
      // Sanity clamps: earners never vanish, never exceed players, never fall.
      // NOTE (v2 revision from §5 synthetic testing): the design doc's 0.999*P
      // cap is relaxed to P itself. Capping at 0.999*P pins a saturated
      // trophy's median at 99.9% while the true rate can be 99.99-100%; on the
      // logit scale that is ~2.3 units of loss on all five of its quantiles
      // (measured: +0.033 mean loss on synthetic scenario A, the whole gap
      // to the v1 reference). Earners still never exceed players, and the
      // quantiles clamp below bounds the output at 99.99.
      eTarget = Math.max(MIN_EARNERS, eTarget);
      eTarget = Math.min(eTarget, pTarget);
      const eLast = lastEarners(i);
      if (eLast !== null) eTarget = Math.max(eTarget, 0.98 * eLast);
      const m = clamp((100 * eTarget) / pTarget, 0.01, 99.99);
      rows.push({ id: c.trophies[i].id, q: quantiles(m, sigma) });
    }
    return rows;
  }

  // Flat branch: one refresh anticipated before target.
  const sigmaFlat = Math.sqrt(
    SIGMA0 * SIGMA0 + SIGMA_H * SIGMA_H * days + SIGMA_REFRESH_FLAT * SIGMA_REFRESH_FLAT,
  );
  for (let i = 0; i < n; i++) {
    const r = lastRate[i];
    if (r === null) {
      // Trophy never seen: neutral 0.5% median with a wide band.
      rows.push({
        id: c.trophies[i].id,
        q: quantiles(0.5, Math.sqrt(sigmaFlat * sigmaFlat + 0.25)),
      });
      continue;
    }
    const m = clamp((r * flatGrowth(r)) / REFRESH_JUMP, 0.01, 99.99);
    rows.push({ id: c.trophies[i].id, q: quantiles(m, sigmaFlat) });
  }
  return rows;
}

// House-baseline fallback: last rate, random-walk spread (never invalid).
function fallbackRows(c: CaseInput): { id: number; q: number[] }[] {
  const days = Math.max(0, c.horizon_minutes / 1440);
  const sigma = 0.5 * Math.sqrt(days);
  const lastRate: (number | null)[] = new Array(c.trophies.length).fill(null);
  for (const read of c.history) {
    read.rates.forEach((r, i) => {
      if (typeof r === "number" && isFinite(r)) lastRate[i] = r;
    });
  }
  return c.trophies.map((t, i) => ({
    id: t.id,
    q: quantiles(lastRate[i] !== null ? lastRate[i]! : 0.5, sigma),
  }));
}

// ------------------------------------------------------------------- main
function parseCase(raw: string): CaseInput | null {
  let obj: any;
  try {
    obj = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!obj || typeof obj !== "object" || !Array.isArray(obj.trophies)) return null;
  const trophies: Trophy[] = obj.trophies
    .filter((t: any) => t && typeof t.id === "number")
    .map((t: any) => ({ id: t.id }));
  const history: Read[] = Array.isArray(obj.history)
    ? obj.history
        .filter((r: any) => r && typeof r === "object")
        .map((r: any) => {
          const slotMs = Date.parse(typeof r.slot === "string" ? r.slot : r.at);
          return {
            slotMs: isFinite(slotMs) ? slotMs : 0,
            players:
              typeof r.players_tracked === "number" && isFinite(r.players_tracked)
                ? r.players_tracked
                : 0,
            rates: Array.isArray(r.rates) ? r.rates : [],
          };
        })
        .sort((a: Read, b: Read) => a.slotMs - b.slotMs)
    : [];
  return {
    id: typeof obj.id === "string" ? obj.id : "case",
    horizon_minutes:
      typeof obj.horizon_minutes === "number" && isFinite(obj.horizon_minutes)
        ? obj.horizon_minutes
        : 0,
    target: typeof obj.target === "string" ? obj.target : obj.at || "",
    trophies,
    history,
  };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outputDir, { recursive: true });

  let files: string[] = [];
  try {
    files = fs
      .readdirSync(inputDir)
      .filter((f) => f.endsWith(".in"))
      .sort();
  } catch {
    return; // no readable input directory: nothing to do
  }

  for (const file of files) {
    const inPath = path.join(inputDir, file);
    let raw: string;
    try {
      raw = fs.readFileSync(inPath, "utf8");
    } catch {
      continue; // unreadable case file: skip it
    }
    const c = parseCase(raw);
    if (!c) continue; // unparseable case: skip it
    let rows: { id: number; q: number[] }[];
    try {
      rows = forecastCase(c);
      if (rows.length !== c.trophies.length) throw new Error("row count mismatch");
    } catch {
      rows = fallbackRows(c);
    }
    const outPath = path.join(outputDir, file.replace(/\.in$/, ".out"));
    fs.writeFileSync(outPath, JSON.stringify({ trophies: rows }));
  }
}

main();
