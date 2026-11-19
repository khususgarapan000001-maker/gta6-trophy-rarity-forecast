// Refresh-aware earners/players forecaster — gta6-trophy-rarity-forecast-01 v1.
//
// Idea (see design-doc-v1.md): Exophase shows rates as of its last refresh, not
// live truth. A refresh jumps players_tracked and drops every rate, because new
// players have not earned the trophies yet. So forecast in earners/players space:
//   earners E = rate * P / 100, rate(target) = 100 * E(target) / P(target).
// Consecutive identical reads are one stale page; collapse them into a single
// refresh event at the last read of the run. With >= 2 distinct events, fit
// power laws log P and log E_i against log hours since release and extrapolate
// to the target (with slope shrinkage for thin trophies). With a single flat
// event (launch night), assume one refresh before the target: players jump by
// G = 6.06 (3868/638, the first launch-night refresh) while earners grow by a
// per-group factor g_i. Quantiles spread on the logit scale around the median.
//
// Node 22, standard library only, erasable TypeScript syntax only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// --- Tunable constants (retune from dev-board feedback) ---------------------
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z"); // GTA VI release
const FLAT_REFRESH_GROWTH = 6.06; // 3868 / 638: first launch-night refresh jump
const Z_SCORES = [-1.645, -0.674, 0, 0.674, 1.645]; // levels .05/.25/.5/.75/.95
const SIGMA0 = 0.15; // base logit spread
const SIGMA_H = 0.25; // per-day random-walk growth of the spread
const SIGMA_REFRESH_FLAT = 0.35; // refresh anticipated but its size unknown
const SIGMA_REFRESH_RICH = 0.1; // refreshes already visible in history
const MIN_RATE = 0.01;
const MAX_RATE = 99.99;
const MIN_EARNERS = 0.5; // floor so log(E) is defined
const MAX_DAILY_GROWTH = 3; // implied player growth may not exceed 3x/day

// --- Types ------------------------------------------------------------------
interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: unknown;
  trophies: Trophy[];
  history: HistoryRead[];
  notes: unknown;
}

interface QuantileRow {
  id: number;
  q: number[];
}

// One distinct page state: a run of identical consecutive reads collapsed to
// the last read of the run (the last moment that page was still live).
interface RefreshEvent {
  timeMs: number;
  players: number;
  rates: (number | null)[];
}

// --- Small numeric helpers --------------------------------------------------
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logitOfRate(ratePct: number): number {
  const p = clamp(ratePct / 100, 0.0001, 0.9999);
  return Math.log(p / (1 - p));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Ordinary least squares on (xs, ys). Null when the fit is undefined.
function leastSquares(
  xs: number[],
  ys: number[],
): { slope: number; intercept: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (!(sxx > 1e-12) || !Number.isFinite(sxx) || !Number.isFinite(sxy)) {
    return null;
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  if (!Number.isFinite(slope) || !Number.isFinite(intercept)) return null;
  return { slope, intercept };
}

function hoursSinceRelease(timeMs: number): number {
  return Math.max(0.5, (timeMs - RELEASE_MS) / 3600000);
}

function daysOfHorizon(c: CaseFile): number {
  const h = typeof c.horizon_minutes === "number" ? c.horizon_minutes : 0;
  return Math.max(0, h / 1440);
}

function targetMs(c: CaseFile): number {
  const t = Date.parse(c.target);
  if (Number.isFinite(t)) return t;
  const at = Date.parse(c.at);
  const h = typeof c.horizon_minutes === "number" ? c.horizon_minutes : 0;
  if (Number.isFinite(at)) return at + h * 60000;
  return Date.now();
}

// Quantiles around a median rate (percent), spread on the logit scale.
function quantilesFromMedian(medianPct: number, sigma: number): number[] {
  const m = clamp(medianPct, MIN_RATE, MAX_RATE);
  const base = logitOfRate(m);
  const qs = Z_SCORES.map((z) =>
    round4(clamp(100 * sigmoid(base + z * sigma), MIN_RATE, MAX_RATE)),
  );
  // Enforce non-decreasing explicitly (sigmoid already guarantees it).
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

function spreadSigma(days: number, refreshSigma: number): number {
  return Math.sqrt(
    SIGMA0 * SIGMA0 + SIGMA_H * SIGMA_H * days + refreshSigma * refreshSigma,
  );
}

// --- History handling -------------------------------------------------------
// Carry null rates forward; collapse runs of identical reads into one event
// stamped at the last read of the run.
function buildEvents(c: CaseFile): RefreshEvent[] {
  const n = c.trophies.length;
  const lastKnown: (number | null)[] = new Array(n).fill(null);
  const events: RefreshEvent[] = [];
  let prevPlayers: number | null = null;
  let prevRates: (number | null)[] | null = null;

  for (const read of c.history ?? []) {
    const rates: (number | null)[] = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
      const r = read.rates?.[i];
      if (typeof r === "number" && Number.isFinite(r)) lastKnown[i] = r;
      rates[i] = lastKnown[i];
    }
    const players = read.players_tracked;
    const changed =
      prevRates === null ||
      players !== prevPlayers ||
      rates.some((r, i) => r !== prevRates?.[i]);
    if (changed) {
      const t = Date.parse(read.slot);
      const tMs = Number.isFinite(t) ? t : Date.parse(read.at);
      if (Number.isFinite(tMs) && typeof players === "number" && players > 0) {
        events.push({ timeMs: tMs, players, rates });
      }
      prevPlayers = players;
      prevRates = rates;
    } else if (events.length > 0) {
      // Same page again: the event stays live until this read.
      const t = Date.parse(read.slot);
      if (Number.isFinite(t)) events[events.length - 1].timeMs = t;
    }
  }
  return events;
}

// Last known rate per trophy (carry-forward over the whole history).
function lastRates(c: CaseFile): (number | null)[] {
  const n = c.trophies.length;
  const last: (number | null)[] = new Array(n).fill(null);
  for (const read of c.history ?? []) {
    for (let i = 0; i < n; i++) {
      const r = read.rates?.[i];
      if (typeof r === "number" && Number.isFinite(r)) last[i] = r;
    }
  }
  return last;
}

// --- Flat-history branch (practice; also the fallback) ----------------------
// One refresh before the target: players jump by G, earners grow by g_i,
// where g_i depends on the trophy's last rate (how fast it is earned).
function flatGrowthFactor(lastRate: number): number {
  if (lastRate >= 50) return FLAT_REFRESH_GROWTH; // near-saturated: earners track players
  if (lastRate >= 2) return 2.0; // fairly common early trophy
  if (lastRate >= 0.5) return 1.5; // early story
  if (lastRate >= 0.1) return 1.15; // mid
  return 1.02; // rare / late-game
}

function flatBranch(c: CaseFile, last: (number | null)[]): number[] {
  return c.trophies.map((_, i) => {
    const r = last[i];
    if (r === null || !Number.isFinite(r)) return 0.5;
    return clamp((r * flatGrowthFactor(r)) / FLAT_REFRESH_GROWTH, MIN_RATE, MAX_RATE);
  });
}

// --- Rich-history branch (final) --------------------------------------------
// Power-law fits in log hours since release, per trophy with shrinkage.
function richBranch(
  c: CaseFile,
  events: RefreshEvent[],
  last: (number | null)[],
): number[] {
  const n = c.trophies.length;
  const hTarget = hoursSinceRelease(targetMs(c));

  // Players: log P vs log hours. Cap the implied daily growth at 3x/day.
  const xs = events.map((e) => Math.log(hoursSinceRelease(e.timeMs)));
  const ysP = events.map((e) => Math.log(Math.max(1, e.players)));
  const pFit = leastSquares(xs, ysP);
  let pTarget: number;
  const lastP = events[events.length - 1].players;
  if (pFit) {
    const maxAlpha = Math.log(MAX_DAILY_GROWTH) / Math.log((hTarget + 24) / hTarget);
    const alpha = clamp(pFit.slope, 0, maxAlpha);
    const est = Math.exp(pFit.intercept + alpha * Math.log(hTarget));
    pTarget = Number.isFinite(est) && est > 0 ? est : lastP;
  } else {
    pTarget = lastP;
  }
  // Players tracked does not shrink during launch week: a noisy fit must
  // never extrapolate below the count already observed.
  pTarget = Math.max(pTarget, lastP);

  // Earners per trophy: collect each trophy's successful slope first, so the
  // median slope is available for shrinkage and for failed fits.
  const slopes: number[] = [];
  const fits: ({ slope: number; intercept: number } | null)[] = [];
  const counts: number[] = [];
  for (let i = 0; i < n; i++) {
    const ex: number[] = [];
    const ey: number[] = [];
    for (const e of events) {
      const r = e.rates[i];
      if (r === null) continue;
      const earners = (r * e.players) / 100;
      if (!(earners > 0)) continue; // E_i = 0 carries no log signal
      ex.push(Math.log(hoursSinceRelease(e.timeMs)));
      ey.push(Math.log(Math.max(MIN_EARNERS, earners)));
    }
    const fit = leastSquares(ex, ey);
    fits.push(fit);
    counts.push(ex.length);
    if (fit) slopes.push(fit.slope);
  }
  const medianSlope = medianOf(slopes);

  const medians: number[] = [];
  for (let i = 0; i < n; i++) {
    const fit = fits[i];
    let eTarget: number | null = null;
    if (fit) {
      let slope = fit.slope;
      if (counts[i] < 4) {
        // Shrink thin trophies toward the cross-trophy median slope.
        slope = (counts[i] * slope + 2 * medianSlope) / (counts[i] + 2);
      }
      const est = Math.exp(fit.intercept + slope * Math.log(hTarget));
      if (Number.isFinite(est) && est > 0) eTarget = Math.max(MIN_EARNERS, est);
    }
    if (eTarget === null) {
      // No usable fit: anchor on the last known earners, grow at median slope.
      const r = last[i];
      if (r !== null && Number.isFinite(r) && lastP > 0) {
        const eLast = Math.max(MIN_EARNERS, (r * lastP) / 100);
        const hLast = hoursSinceRelease(events[events.length - 1].timeMs);
        const est = eLast * Math.pow(hTarget / hLast, medianSlope);
        if (Number.isFinite(est) && est > 0) eTarget = Math.max(MIN_EARNERS, est);
      }
    }
    if (eTarget === null) {
      medians.push(0.5);
    } else {
      medians.push(clamp((100 * eTarget) / pTarget, MIN_RATE, MAX_RATE));
    }
  }
  return medians;
}

// --- House-baseline fallback (never let a case go invalid) ------------------
function baselineRows(c: CaseFile): QuantileRow[] {
  const days = daysOfHorizon(c);
  const sigma = 0.5 * Math.sqrt(days);
  const last = lastRates(c);
  return c.trophies.map((t, i) => {
    const r = last[i];
    const median = r !== null && Number.isFinite(r) ? r : 0.5;
    return { id: t.id, q: quantilesFromMedian(median, sigma) };
  });
}

// --- Per-case forecast -------------------------------------------------------
function forecastCase(c: CaseFile): QuantileRow[] {
  try {
    if (!c || !Array.isArray(c.trophies)) return [];
    const days = daysOfHorizon(c);
    const last = lastRates(c);
    const events = buildEvents(c);
    let medians: number[];
    let refreshSigma: number;
    if (events.length >= 2) {
      medians = richBranch(c, events, last);
      refreshSigma = SIGMA_REFRESH_RICH;
    } else if ((c.history ?? []).length > 0) {
      medians = flatBranch(c, last);
      refreshSigma = SIGMA_REFRESH_FLAT;
    } else {
      // No history at all: no rate known, wide spread around 0.5%.
      medians = c.trophies.map(() => 0.5);
      refreshSigma = SIGMA_REFRESH_FLAT;
    }
    const sigma = spreadSigma(days, refreshSigma);
    return c.trophies.map((t, i) => ({
      id: t.id,
      q: quantilesFromMedian(medians[i], sigma),
    }));
  } catch {
    return baselineRows(c);
  }
}

// --- Entry point -------------------------------------------------------------
function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output/";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    let parsed: CaseFile;
    try {
      parsed = JSON.parse(readFileSync(join(inputDir, file), "utf8")) as CaseFile;
    } catch {
      continue; // unreadable case file: skip it, never crash the run
    }
    if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.trophies)) {
      continue;
    }
    const outId = typeof parsed.id === "string" && parsed.id ? parsed.id : file.replace(/\.in$/, "");
    try {
      const rows = forecastCase(parsed);
      writeFileSync(
        join(outputDir, `${outId}.out`),
        JSON.stringify({ trophies: rows }),
        "utf8",
      );
    } catch {
      // If even the fallback write fails, there is nothing else to do.
    }
  }
}

main();
