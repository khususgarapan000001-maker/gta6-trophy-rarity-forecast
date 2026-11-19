// GTA VI trophy rarity forecaster v3 — "Recent-window fit + saturation-aware earners"
//
// Forecast, for each of the 55 GTA VI PS5 trophies, five quantiles (5/25/50/75/95%)
// of its Exophase earn rate at the case target. Works in earners/players space:
//   rate = 100 * E(target) / P(target)
// History is first collapsed into refresh events (stale identical reads merged),
// then either the flat branch (a single refresh is assumed before the target,
// practice cases) or the rich branch (many refreshes seen, final case).
//
// Rich branch (v3, new): ALL fits use only the recent 24h window of events,
// so the current growth regime — not the launch average — is extrapolated.
// Earners per trophy are fitted in TWO shapes on that window and the shape
// with the clearly smaller log-space residual wins:
//   P — power law in players:  log E = a + b * log P
//   S — saturating in time:    E = E_inf * (1 - exp(-t / tau)), tau on a grid
// A time-fit (v1 style, also on the recent window) cross-checks the chosen
// estimate; disagreement beyond 3x widens that trophy's spread.
//
// Only the Node standard library is used. Deterministic, no network, no clock.

import * as fs from "node:fs";
import * as path from "node:path";

// --- Named constants (tunable, see design doc v3) ---
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const WINDOW_HOURS = 24; // recent window for every rich-branch fit
const SAT_MARGIN = 0.8; // saturating shape wins only if R_S < SAT_MARGIN * R_P
const TAU_GRID = [6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288]; // hours
const G_REFRESH = 6.06; // first launch-night refresh jump: 3868 / 638 players
const G_EARLY = 1.28; // earners growth over one refresh, early-story trophies
const G_MID = 1.15;
const G_RARE = 1.02;
const SIGMA0 = 0.15;
const SIGMA_H = 0.25;
const SIGMA_REFRESH_FLAT = 0.35;
const SIGMA_REFRESH_RICH = 0.1;
const SIGMA_DISAGREE = 0.25; // extra sigma when the two fits disagree > 3x
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

interface TrophyDef {
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

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: unknown;
  trophies: TrophyDef[];
  history: HistoryRead[];
  notes: string;
}

interface RefreshEvent {
  tMs: number;
  players: number;
  rates: (number | null)[];
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logitFrac(p: number): number {
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
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

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

function hoursSinceRelease(ms: number): number {
  return Math.max(0.5, (ms - RELEASE_MS) / 3600000);
}

// Collapse the raw reads into refresh events: a read is a new event when
// players_tracked changed or any rate changed vs the previous read; a run of
// identical reads is one event dated at its LAST read. Null rates carry
// forward from the last known value for that trophy.
function toEvents(history: HistoryRead[]): RefreshEvent[] {
  const events: RefreshEvent[] = [];
  const lastKnown: (number | null)[] = [];
  let prevPlayers: number | null = null;
  let prevRates: (number | null)[] | null = null;
  const sorted = [...history].sort(
    (a, b) => Date.parse(a.slot || a.at) - Date.parse(b.slot || b.at),
  );
  for (const read of sorted) {
    const players =
      typeof read.players_tracked === "number" && isFinite(read.players_tracked)
        ? read.players_tracked
        : 0;
    const rates: (number | null)[] = (read.rates || []).map((r, i) => {
      if (typeof r === "number" && isFinite(r)) {
        lastKnown[i] = r;
        return r;
      }
      return lastKnown[i] !== undefined && lastKnown[i] !== null ? lastKnown[i] : null;
    });
    let changed = prevPlayers === null || players !== prevPlayers;
    if (!changed && prevRates !== null) {
      for (let i = 0; i < rates.length; i++) {
        if (rates[i] !== prevRates[i]) {
          changed = true;
          break;
        }
      }
    }
    const slotMs = Date.parse(read.slot || read.at);
    if (changed) {
      events.push({ tMs: slotMs, players, rates });
    } else if (events.length > 0) {
      events[events.length - 1].tMs = slotMs;
    }
    prevPlayers = players;
    prevRates = rates;
  }
  return events;
}

function quantiles(medianPct: number, sigma: number): number[] {
  const base = logitFrac(clamp(medianPct, 0.01, 99.99) / 100);
  const out: number[] = [];
  let prev = -Infinity;
  for (const z of Z) {
    let q = Math.round(clamp(100 * sigmoid(base + z * sigma), 0.01, 99.99) * 10000) / 10000;
    if (q < prev) q = prev;
    out.push(q);
    prev = q;
  }
  return out;
}

function earners(rate: number | null, players: number): number {
  if (rate === null || rate === undefined || players <= 0) return 0;
  return (rate * players) / 100;
}

// Players power law in time over the given events, with one round of
// MAD outlier rejection. Returns P(target), or NaN when it cannot fit.
function fitPlayers(events: RefreshEvent[], targetMs: number): number {
  const fitOnce = (evs: RefreshEvent[]): { a: number; b: number } | null => {
    const xs = evs.map((e) => Math.log(hoursSinceRelease(e.tMs)));
    const ys = evs.map((e) => Math.log(Math.max(1, e.players)));
    return ols(xs, ys);
  };
  let fit = fitOnce(events);
  if (!fit) return NaN;
  // One MAD rejection round: drop the single worst residual if it exceeds
  // 2.5x the median absolute deviation, then refit once.
  const xsAll = events.map((e) => Math.log(hoursSinceRelease(e.tMs)));
  const resid = events.map((e, i) => Math.log(Math.max(1, e.players)) - (fit!.a + fit!.b * xsAll[i]));
  const medResid = median(resid);
  const absDev = resid.map((r) => Math.abs(r - medResid));
  const mad = median(absDev);
  let worst = 0;
  let worstAbs = -1;
  for (let i = 0; i < resid.length; i++) {
    if (Math.abs(resid[i]) > worstAbs) {
      worstAbs = Math.abs(resid[i]);
      worst = i;
    }
  }
  if (mad > 0 && worstAbs > 2.5 * mad && events.length > 2) {
    const kept = events.filter((_, i) => i !== worst);
    const refit = fitOnce(kept);
    if (refit) fit = refit;
  }
  const tLast = hoursSinceRelease(events[events.length - 1].tMs);
  const alphaMax = Math.log(3) / Math.log((tLast + 24) / tLast);
  const alpha = clamp(fit.b, 0, alphaMax);
  const xs = events.map((e) => Math.log(hoursSinceRelease(e.tMs)));
  const ys = events.map((e) => Math.log(Math.max(1, e.players)));
  const a = mean(ys) - alpha * mean(xs);
  const p = Math.exp(a + alpha * Math.log(hoursSinceRelease(targetMs)));
  return isFinite(p) && p > 0 ? p : NaN;
}

// Per-trophy rich-branch estimate on the recent-window events.
// Returns { eTarget, uncertain }.
function forecastTrophyRich(
  idx: number,
  events: RefreshEvent[],
  pTarget: number,
  targetMs: number,
  medSlopeP: number,
  medSlopeT: number,
): { eTarget: number; uncertain: boolean } {
  // Collect window points with E > 0 for this trophy.
  const pts: { logP: number; logT: number; logE: number; tH: number }[] = [];
  for (const e of events) {
    const r = e.rates[idx];
    if (r === null || r === undefined || e.players <= 0) continue;
    const rawE = earners(r, e.players);
    if (rawE <= 0) continue;
    pts.push({
      logP: Math.log(e.players),
      logT: Math.log(hoursSinceRelease(e.tMs)),
      logE: Math.log(Math.max(0.5, rawE)),
      tH: hoursSinceRelease(e.tMs),
    });
  }
  if (pts.length === 0) return { eTarget: NaN, uncertain: true };

  const n = pts.length;
  const logPs = pts.map((p) => p.logP);
  const logTs = pts.map((p) => p.logT);
  const logEs = pts.map((p) => p.logE);

  // --- Shape P: power law in players ---
  const fitP = ols(logPs, logEs);
  let slopeP: number;
  let interceptP: number;
  let rP = Infinity;
  if (fitP) {
    slopeP = n < 4 ? (n * fitP.b + 2 * medSlopeP) / (n + 2) : fitP.b;
    interceptP = mean(logEs) - slopeP * mean(logPs);
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = interceptP + slopeP * logPs[i] - logEs[i];
      ss += d * d;
    }
    rP = ss / n;
  } else {
    slopeP = medSlopeP;
    interceptP = mean(logEs) - slopeP * mean(logPs);
  }
  const eFromP = Math.exp(interceptP + slopeP * Math.log(pTarget));

  // --- Shape S: saturating in time, tau on a grid ---
  let bestS: { eTarget: number; resid: number } | null = null;
  if (n >= 4) {
    for (const tau of TAU_GRID) {
      const fs: number[] = [];
      let usable = true;
      for (const p of pts) {
        const f = 1 - Math.exp(-p.tH / tau);
        if (f <= 0) {
          usable = false;
          break;
        }
        fs.push(f);
      }
      if (!usable) continue;
      const logEInf = mean(logEs.map((le, i) => le - Math.log(fs[i])));
      const eInf = Math.exp(logEInf);
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const pred = Math.max(0.5, eInf * fs[i]);
        const d = Math.log(pred) - logEs[i];
        ss += d * d;
      }
      const rS = ss / n;
      const tTarget = hoursSinceRelease(targetMs);
      const eTargetS = eInf * (1 - Math.exp(-tTarget / tau));
      if (!bestS || rS < bestS.resid) bestS = { eTarget: eTargetS, resid: rS };
    }
  }

  // --- Selection: saturating only on a clear residual win ---
  let eChosen = eFromP;
  if (bestS && bestS.resid < SAT_MARGIN * rP) {
    eChosen = bestS.eTarget;
  }

  // --- Cross-check: v1-style time fit, also on the recent window ---
  let eTime = NaN;
  const fitT = ols(logTs, logEs);
  if (fitT) {
    const slopeT = n < 4 ? (n * fitT.b + 2 * medSlopeT) / (n + 2) : fitT.b;
    const interceptT = mean(logEs) - slopeT * mean(logTs);
    eTime = Math.exp(interceptT + slopeT * Math.log(hoursSinceRelease(targetMs)));
  }
  let uncertain = false;
  if (isFinite(eTime) && isFinite(eChosen) && eTime > 0 && eChosen > 0) {
    const ratio = eChosen / eTime;
    if (ratio > 3 || ratio < 1 / 3) uncertain = true;
  } else {
    uncertain = true;
  }
  return { eTarget: eChosen, uncertain };
}

function forecastCase(c: Case): { id: number; q: number[] }[] {
  const n = c.trophies.length;
  const days = Math.max(0, (c.horizon_minutes || 0) / 1440);
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  const events = toEvents(c.history || []);

  // Last known rate per trophy, and last players.
  const lastRate: (number | null)[] = new Array(n).fill(null);
  for (const e of events) {
    for (let i = 0; i < n; i++) {
      if (e.rates[i] !== null && e.rates[i] !== undefined) lastRate[i] = e.rates[i];
    }
  }
  const lastP = events.length > 0 ? events[events.length - 1].players : 0;
  const lastE: number[] = lastRate.map((r, i) => earners(r, lastP));

  const sigmaFlat = Math.sqrt(SIGMA0 * SIGMA0 + SIGMA_H * SIGMA_H * days + SIGMA_REFRESH_FLAT * SIGMA_REFRESH_FLAT);
  const sigmaRich = Math.sqrt(SIGMA0 * SIGMA0 + SIGMA_H * SIGMA_H * days + SIGMA_REFRESH_RICH * SIGMA_REFRESH_RICH);

  // --- Flat branch: 0 or 1 distinct events (practice / empty history) ---
  if (events.length < 2) {
    return c.trophies.map((t, i) => {
      const r = lastRate[i];
      if (r === null) {
        return { id: t.id, q: quantiles(0.5, Math.sqrt(sigmaFlat * sigmaFlat + 0.25)) };
      }
      let g: number;
      if (r >= 50) g = G_REFRESH;
      else if (r >= 2) g = G_EARLY;
      else if (r >= 0.5) g = G_EARLY;
      else if (r >= 0.1) g = G_MID;
      else g = G_RARE;
      return { id: t.id, q: quantiles(clamp((r * g) / G_REFRESH, 0.01, 99.99), sigmaFlat) };
    });
  }

  // --- Rich branch ---
  // Recent window: events after at - WINDOW_HOURS; fall back to all events
  // when the window holds fewer than 4 events.
  const windowStart = atMs - WINDOW_HOURS * 3600000;
  let windowEvents = events.filter((e) => e.tMs > windowStart);
  if (windowEvents.length < 4) windowEvents = events;
  if (windowEvents.length < 2) {
    // Degenerate: fall back to the flat branch shape with last rates.
    return c.trophies.map((t, i) => {
      const r = lastRate[i];
      if (r === null) {
        return { id: t.id, q: quantiles(0.5, Math.sqrt(sigmaFlat * sigmaFlat + 0.25)) };
      }
      let g: number;
      if (r >= 50) g = G_REFRESH;
      else if (r >= 2) g = G_EARLY;
      else if (r >= 0.5) g = G_EARLY;
      else if (r >= 0.1) g = G_MID;
      else g = G_RARE;
      return { id: t.id, q: quantiles(clamp((r * g) / G_REFRESH, 0.01, 99.99), sigmaFlat) };
    });
  }

  let pTarget = fitPlayers(windowEvents, targetMs);
  if (!isFinite(pTarget) || pTarget <= 0) pTarget = lastP > 0 ? lastP : 1;

  // Cross-trophy median slopes (players-shape and time-shape) for shrinkage.
  const rawSlopesP: number[] = [];
  const rawSlopesT: number[] = [];
  for (let i = 0; i < n; i++) {
    const xsP: number[] = [];
    const ys: number[] = [];
    const xsT: number[] = [];
    for (const e of windowEvents) {
      const r = e.rates[i];
      if (r === null || r === undefined || e.players <= 0) continue;
      const rawE = earners(r, e.players);
      if (rawE <= 0) continue;
      xsP.push(Math.log(e.players));
      xsT.push(Math.log(hoursSinceRelease(e.tMs)));
      ys.push(Math.log(Math.max(0.5, rawE)));
    }
    const fP = xsP.length >= 2 ? ols(xsP, ys) : null;
    if (fP) rawSlopesP.push(fP.b);
    const fT = xsT.length >= 2 ? ols(xsT, ys) : null;
    if (fT) rawSlopesT.push(fT.b);
  }
  const medSlopeP = rawSlopesP.length > 0 ? median(rawSlopesP) : 1;
  const medSlopeT = rawSlopesT.length > 0 ? median(rawSlopesT) : 1;

  return c.trophies.map((t, i) => {
    const { eTarget, uncertain } = forecastTrophyRich(i, windowEvents, pTarget, targetMs, medSlopeP, medSlopeT);
    let sigma = sigmaRich + (uncertain ? SIGMA_DISAGREE : 0);
    if (!isFinite(eTarget)) {
      // No usable earners data: keep the last rate with a wide band.
      const r = lastRate[i];
      return { id: t.id, q: quantiles(r !== null ? r : 0.5, sigma + 0.5) };
    }
    // Sanity clamps: earners never fall (rounding noise only), never exceed players.
    // Ceiling is 0.9999 * P, not 0.999 * P: a saturated trophy's truth scores as
    // 99.99% (the logit clamp), and 99.9% sits a full 2.3 logit units away from
    // it — on the log grid that one trophy alone cost ~0.035 of mean loss.
    let e = Math.max(0.5, eTarget);
    e = Math.min(e, 0.9999 * pTarget);
    if (lastE[i] > 0) e = Math.max(e, 0.98 * lastE[i]);
    const m = clamp((100 * e) / pTarget, 0.01, 99.99);
    return { id: t.id, q: quantiles(m, sigma) };
  });
}

// House-baseline fallback for a case whose model throws: last rate with a
// random-walk spread. Never lets a readable case go invalid.
function fallbackRows(c: Case): { id: number; q: number[] }[] {
  const days = Math.max(0, (c.horizon_minutes || 0) / 1440);
  const sigma = 0.5 * Math.sqrt(Math.max(days, 0));
  const events = toEvents(c.history || []);
  const lastRate: (number | null)[] = new Array(c.trophies.length).fill(null);
  for (const e of events) {
    for (let i = 0; i < c.trophies.length; i++) {
      if (e.rates[i] !== null && e.rates[i] !== undefined) lastRate[i] = e.rates[i];
    }
  }
  return c.trophies.map((t, i) => {
    const r = lastRate[i];
    return { id: t.id, q: quantiles(r !== null ? r : 0.5, sigma + 0.5) };
  });
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    let c: Case;
    try {
      c = JSON.parse(fs.readFileSync(path.join(inputDir, f), "utf8")) as Case;
    } catch {
      continue; // unreadable case file: skip it, never crash
    }
    let rows: { id: number; q: number[] }[];
    try {
      rows = forecastCase(c);
    } catch {
      try {
        rows = fallbackRows(c);
      } catch {
        rows = c.trophies.map((t) => ({ id: t.id, q: quantiles(0.5, 1) }));
      }
    }
    const outPath = path.join(outputDir, f.replace(/\.in$/, ".out"));
    fs.writeFileSync(outPath, JSON.stringify({ trophies: rows }));
  }
}

main();
