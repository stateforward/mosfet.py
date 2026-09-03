// Orthogonal router on a sparse Hanan grid whose lanes are the edges of inflated
// obstacle rects. Pure geometry: no DOM, no graph host, no app state.
//
// route() returns world-coord waypoints [start, ...bends..., end] or null when the
// trivial case applies (the caller then leaves the edge to the spline renderer).

import { type HandlePosition, type Rect, type XYPosition } from "../types.ts";

const BEND_COST = 24;
// Best-effort (R1): a blocked grid edge is not forbidden, it costs its length times
// this. The search then minimises hidden stretch instead of failing.
const BLOCKED_MULT = 8;
// Clearance is advisory, not solid: a lane inside an obstacle's INFLATED rect but
// outside the real node costs only this. Without the distinction a tight-but-clear
// corridor (deliberately aligned pins with a neighbour within 16px) is priced as
// fully blocked, and a wrap around the whole node beats the straight line (QA
// round). Nearly nominal on purpose: the toll only tie-breaks toward roomier lanes
// at equal geometry. Anything bigger taxes long tight corridors until a parallel
// lane a few px off the pin line wins, turning a deliberate alignment into a
// whole-run shimmy (measured upstream: at 1.1 a 436px soft stretch still lost to a
// 6px detour).
const SOFT_MULT = 1.02;
const EPS = 0.01;
// The real node boxes: crossing these (inflated by this much) is the 8x sin. Lanes
// may ride exactly on inflated-clearance edges, but never graze the drawn node.
const HARD_MARGIN = 4;
// Problem-size cap: a pathological grid (hundreds of nodes = dense Hanan lines)
// must not freeze the main thread on every obstacle change. Bailing returns null
// and the caller falls back to the plain spline for that edge -- degraded, alive.
const EXPANSION_CAP = 20000;

// Vector semantics per face, screen coords (y grows downward): "top" exits upward.
const DIR: Record<HandlePosition, readonly [number, number]> = {
  right: [1, 0],
  left: [-1, 0],
  top: [0, -1],
  bottom: [0, 1],
};

/** Closed axis-aligned box in lane space: `[x0, y0, x1, y1]`. */
type Box = [number, number, number, number];

export type RouteArgs = {
  /** Pin the wire leaves from, in world pixels. */
  readonly start: XYPosition;
  /** Pin the wire enters, in world pixels. */
  readonly end: XYPosition;
  /** Node/gate boxes the wire must respect, in world pixels. */
  readonly obstacles: readonly Rect[];
  /** Face the wire leaves the start pin toward; default `"right"`. */
  readonly startDir?: HandlePosition;
  /** Face the wire enters the end pin from; default `"left"`. */
  readonly endDir?: HandlePosition;
  /** Obstacle inflation in px that lanes form around; default `16`. */
  readonly clearance?: number;
  /** Straight stub length leaving the start pin in px; default `24`. */
  readonly stubStart?: number;
  /** Straight stub length entering the end pin in px; default `24`. */
  readonly stubEnd?: number;
  /** Ban retracing out of the start stub (binding comb-gate faces); default `false`. */
  readonly enforceStart?: boolean;
  /** Ban entering the end stub from the endpoint's side; default `false`. */
  readonly enforceEnd?: boolean;
  /** Extra grid-only vertical obstacle inflation (drop-shadow standoff); default `0`. */
  readonly vPad?: number;
  /** Colinearity tolerance in px under which a near-straight link returns null; default `0`. */
  readonly flatTol?: number;
};

/**
 * Route one orthogonal wire over a sparse Hanan grid built from the edges of
 * `args.obstacles` inflated by `clearance`.
 *
 * Inputs: `start`/`end` pin positions in world px; `startDir`/`endDir` exit and
 * entry faces; obstacle rects in world px; tuning numbers documented on
 * `RouteArgs`. Outputs: waypoints `[start, ...bends..., end]`, or `null` when the
 * trivial case applies (facing stubs, near-colinear within `flatTol`, clear
 * corridor) or when the search budget is exhausted -- callers fall back to the
 * spline renderer on `null`. Ownership: pure; the caller owns the returned array;
 * inputs are not retained. Lifetime: one call. Concurrency: synchronous/pure.
 * Failure modes: `null` for trivial-flat corridors and for grids exceeding
 * `EXPANSION_CAP` expansions; never throws for geometric inputs. Units: world
 * pixels throughout; costs are px-equivalents (`BEND_COST`, multipliers).
 * Classification: runtime-safe.
 */
export function route(args: RouteArgs): readonly XYPosition[] | null {
  const clearance = args.clearance ?? 16;
  const stubStart = args.stubStart ?? 24;
  const stubEnd = args.stubEnd ?? 24;
  const vPad = args.vPad ?? 0;
  const flatTol = args.flatTol ?? 0;
  const sv = DIR[args.startDir ?? "right"];
  const ev = DIR[args.endDir ?? "left"];
  const sStub: XYPosition = {
    x: args.start.x + sv[0] * stubStart,
    y: args.start.y + sv[1] * stubStart,
  };
  const eStub: XYPosition = {
    x: args.end.x + ev[0] * stubEnd,
    y: args.end.y + ev[1] * stubEnd,
  };

  // Local obstacle set, inflated. The corridor bbox is padded so lanes exist around
  // obstacles sitting just outside the endpoints' span.
  const pad = clearance + 80;
  const bb: Box = [
    Math.min(sStub.x, eStub.x, args.start.x, args.end.x) - pad,
    Math.min(sStub.y, eStub.y, args.start.y, args.end.y) - pad,
    Math.max(sStub.x, eStub.x, args.start.x, args.end.x) + pad,
    Math.max(sStub.y, eStub.y, args.start.y, args.end.y) + pad,
  ];
  const rects: Box[] = [];
  const hard: Box[] = []; // the real node boxes: crossing these is the 8x sin
  for (const r of args.obstacles) {
    // vPad makes every obstacle taller FOR THE GRID ONLY (shadow rule): horizontal
    // lanes form that much further from node tops/bottoms, so wires stand clear of
    // the drop shadow instead of grazing it. Vertical lanes stay put -- pin-column
    // corridors are deliberate geometry.
    const x0 = r.x - clearance;
    const y0 = r.y - clearance - vPad;
    const x1 = r.x + r.width + clearance;
    const y1 = r.y + r.height + clearance + vPad;
    if (x1 < bb[0] || x0 > bb[2] || y1 < bb[1] || y0 > bb[3]) continue;
    rects.push([x0, y0, x1, y1]);
    hard.push([
      r.x - HARD_MARGIN,
      r.y - HARD_MARGIN,
      r.x + r.width + HARD_MARGIN,
      r.y + r.height + HARD_MARGIN,
    ]);
  }

  // Trivial forward case: straight stub-to-stub corridor with nothing in the way.
  // ONLY for canonical exit-right/enter-left directions -- with any other faces
  // (flipped gates), a clear straight corridor still exists geometrically but runs
  // THROUGH the body against the stub direction, and the shortcut made flips appear
  // to do nothing unless the layout already forced a wrap (QA find). Non-canonical
  // dirs always take the full search, which honours the stubs.
  // flatTol (straightening rule): endpoints within the edge stroke's thickness of
  // colinear fall back to the spline too -- a jog smaller than the line is drawing
  // noise, and the near-flat spline reads as a straight wire.
  if (
    sv[0] === 1 &&
    ev[0] === -1 &&
    eStub.x - sStub.x > EPS &&
    Math.abs(sStub.y - eStub.y) < Math.max(EPS, flatTol) &&
    clearH(sStub.y, sStub.x, eStub.x, rects) &&
    clearH(eStub.y, sStub.x, eStub.x, rects)
  ) {
    return null;
  }

  // Lanes: obstacle edges plus the two stub coordinates.
  const xs = [sStub.x, eStub.x];
  const ys = [sStub.y, eStub.y];
  for (const [x0, y0, x1, y1] of rects) {
    xs.push(x0, x1);
    ys.push(y0, y1);
  }
  const X = dedupe(xs);
  const Y = dedupe(ys);

  const si: readonly [number, number] = [idx(X, sStub.x), idx(Y, sStub.y)];
  const ei: readonly [number, number] = [idx(X, eStub.x), idx(Y, eStub.y)];

  const path = astar(
    X,
    Y,
    si,
    ei,
    rects,
    hard,
    sv,
    ev,
    args.enforceStart ?? false,
    args.enforceEnd ?? false,
  );
  if (path === null) return null;

  // Waypoints: real endpoints, stubs, grid path. Collinear runs collapsed.
  const pts: XYPosition[] = [args.start, sStub];
  for (const [i, j] of path) {
    // Invariant: astar only emits indices over this exact grid, so the hit below
    // cannot miss.
    pts.push({ x: laneAt(X, i), y: laneAt(Y, j) });
  }
  pts.push(eStub, args.end);
  return simplify(pts);
}

export type OffsetStrandArgs = {
  /**
   * Template polyline `[start, ...bends..., end]` with alternating orthogonal
   * segments and at least four points (typically the median sibling's route).
   */
  readonly template: readonly XYPosition[];
  /** This member's own start pin; the first segment re-anchors on it. */
  readonly start: XYPosition;
  /** This member's own end pin; the last segment re-anchors on it. */
  readonly end: XYPosition;
  /** Signed offset in px along the template path's LEFT normal. */
  readonly offset: number;
};

/**
 * Reuse a sibling's routed shape for a link between the same two nodes, translated
 * to this link's own slots. Sibling stub tips differ only in y (same nodes, stacked
 * slots), so the translation is pure-y plus one orthogonal tail alignment.
 * Identical shapes cannot cross each other -- this is what kills same-pair homotopy
 * braids.
 *
 * Ribbon-cable bundle construction. A bundle of links between the same two nodes
 * must share one homotopy AND nest correctly around wrapped obstacles -- uniform
 * y-shifts copy the source-side stacking onto every run, which inverts the required
 * nesting on the far side of a wrap (measured upstream: 6 inherent-looking crossings
 * that a perpendicular offset eliminates entirely). Each strand is the template
 * offset along the path's LEFT NORMAL by `offset`; orthogonal corner joins come
 * free (a vertex takes x from its vertical neighbour, y from its horizontal one).
 * The first and last segments re-anchor on the member's own pins.
 *
 * Inputs: orthogonal template with >= 4 points, member pins, signed px offset.
 * Outputs: the member's polyline `[start, ...bends..., end]`, or `null` when the
 * template is degenerate (fewer than 4 points, non-orthogonal or non-alternating
 * segments) or the simplified strand has fewer than 3 points. Ownership: pure; the
 * caller owns the returned array; inputs are not retained. Lifetime: one call.
 * Concurrency: synchronous/pure. Failure modes: `null` as described; never throws.
 * Units: world pixels. Classification: runtime-safe.
 */
export function offsetStrand(args: OffsetStrandArgs): readonly XYPosition[] | null {
  const { template, start, end, offset } = args;
  const n = template.length;
  if (n < 4) return null;
  // Per-segment free coordinate: vertical segments carry x, horizontal carry y.
  type Strand = { readonly vert: boolean; coord: number; readonly nx: number; readonly ny: number };
  const strands: Strand[] = [];
  for (let k = 0; k < n - 1; k++) {
    const a = template[k];
    const b = template[k + 1];
    if (a === undefined || b === undefined) return null;
    const vert = Math.abs(a.x - b.x) < 0.1;
    const horiz = Math.abs(a.y - b.y) < 0.1;
    if (!vert && !horiz) return null;
    // Left normal of travel: (dy, -dx). Offsetting a vertical shifts x, a horizontal
    // shifts y, signed by travel direction.
    const dx = Math.sign(b.x - a.x);
    const dy = Math.sign(b.y - a.y);
    strands.push({ vert, coord: vert ? a.x : a.y, nx: dy, ny: -dx });
  }
  // Pin re-anchor on the attachment segments; perpendicular offset on the mid ones.
  for (let k = 0; k < strands.length; k++) {
    const s = strands[k];
    if (s === undefined) continue; // unreachable: strands has exactly n - 1 items
    if (k === 0) s.coord = s.vert ? start.x : start.y;
    else if (k === strands.length - 1) s.coord = s.vert ? end.x : end.y;
    else s.coord += offset * (s.vert ? s.nx : s.ny);
  }
  // Consecutive segments must alternate orientation for corner joins to be defined.
  const own: XYPosition[] = [{ x: start.x, y: start.y }];
  for (let k = 1; k < strands.length; k++) {
    const a = strands[k - 1];
    const b = strands[k];
    if (a === undefined || b === undefined) return null;
    if (a.vert === b.vert) return null;
    own.push(a.vert ? { x: a.coord, y: b.coord } : { x: b.coord, y: a.coord });
  }
  own.push({ x: end.x, y: end.y });
  const pts = simplify(own);
  return pts.length >= 3 ? pts : null;
}

function dedupe(vals: number[]): number[] {
  const s = [...vals].sort((a, b) => a - b);
  const first = s[0];
  if (first === undefined) return [];
  const out = [first];
  for (const v of s) {
    const last = out[out.length - 1];
    if (last !== undefined && v - last > EPS) out.push(v);
  }
  return out;
}

function idx(lanes: readonly number[], v: number): number {
  for (let i = 0; i < lanes.length; i++) {
    const lane = lanes[i];
    if (lane !== undefined && Math.abs(lane - v) <= EPS) return i;
  }
  return 0;
}

// Grid lanes are built once per route() call; every index handed here comes from
// dedupe()/astar() bounds checks, so a miss is a programming error, not data.
function laneAt(lanes: readonly number[], index: number): number {
  const lane = lanes[index];
  if (lane === undefined) throw new Error("pathing grid lane index out of range");
  return lane;
}

// A horizontal run at y crossing [xa,xb] is blocked by a rect when y lies strictly
// inside it and the spans overlap with positive length. Strict bounds let paths ride
// exactly on inflated edges (the lanes ARE those edges).
function clearH(y: number, xa: number, xb: number, rects: readonly Box[]): boolean {
  const a = Math.min(xa, xb);
  const b = Math.max(xa, xb);
  for (const [x0, y0, x1, y1] of rects) {
    if (y > y0 + EPS && y < y1 - EPS && b > x0 + EPS && a < x1 - EPS) return false;
  }
  return true;
}

function clearV(x: number, ya: number, yb: number, rects: readonly Box[]): boolean {
  const a = Math.min(ya, yb);
  const b = Math.max(ya, yb);
  for (const [x0, y0, x1, y1] of rects) {
    if (x > x0 + EPS && x < x1 - EPS && b > y0 + EPS && a < y1 - EPS) return false;
  }
  return true;
}

function astar(
  X: readonly number[],
  Y: readonly number[],
  si: readonly [number, number],
  ei: readonly [number, number],
  rects: readonly Box[],
  hard: readonly Box[],
  sv: readonly [number, number],
  ev: readonly [number, number],
  enforceStart: boolean,
  enforceEnd: boolean,
): readonly (readonly [number, number])[] | null {
  const W = X.length;
  const H = Y.length;
  const id = (i: number, j: number): number => j * W + i;
  // State carries entry axis (0 h, 1 v, -1 start) for the bend penalty.
  const g = new Map<number, number>();
  const from = new Map<number, number>();
  const open: [number, number, number, number][] = [[0, si[0], si[1], -1]];
  const ex = laneAt(X, ei[0]);
  const ey = laneAt(Y, ei[1]);
  const hCost = (i: number, j: number): number =>
    Math.abs(laneAt(X, i) - ex) + Math.abs(laneAt(Y, j) - ey);
  g.set(id(si[0], si[1]) * 4 + 0, 0); // axis -1 -> offset (axis+1) = 0, matching the pop

  let expansions = 0;
  while (open.length > 0) {
    if (++expansions > EXPANSION_CAP) return null;
    let bi = 0;
    for (let k = 1; k < open.length; k++) {
      const cand = open[k];
      const best = open[bi];
      if (cand !== undefined && best !== undefined && cand[0] < best[0]) bi = k;
    }
    const popped = open.splice(bi, 1)[0];
    if (popped === undefined) break;
    const [, i, j, axis] = popped;
    const xi = laneAt(X, i);
    const yi = laneAt(Y, j);
    const skey = id(i, j) * 4 + (axis + 1);
    const base = g.get(skey);
    if (base === undefined) continue;
    if (i === ei[0] && j === ei[1]) {
      const pts: [number, number][] = [[i, j]];
      let cur: number | undefined = skey;
      while (cur !== undefined && from.has(cur)) {
        cur = from.get(cur);
        if (cur === undefined) break;
        pts.push([Math.floor(cur / 4) % W, Math.floor(cur / 4 / W)]);
      }
      return pts.reverse();
    }
    const steps: [number, number, number][] = [
      [i - 1, j, 0],
      [i + 1, j, 0],
      [i, j - 1, 1],
      [i, j + 1, 1],
    ];
    for (const [ni, nj, nax] of steps) {
      if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
      const xn = laneAt(X, ni);
      const yn = laneAt(Y, nj);
      // ENFORCED stubs (gate faces) are binding, not cosmetic: retracing one is free
      // on this grid (same-lane travel, no bend), so A* folded straight back over
      // any stub the layout found inconvenient and simplify() collapsed the fold --
      // flipped gates looked like no-ops (QA find). Ban the one step that reverses
      // out of the start stub / enters the end stub from the endpoint's side.
      // OPT-IN per end: for node pins the fold is load-bearing -- tight
      // side-by-side pairs route THROUGH the collapsed fold, and a global ban
      // turned their doglegs into colliding shimmies (measured upstream: adjacent
      // 0 -> 17 crossings).
      if (enforceStart || enforceEnd) {
        const dx = Math.sign(xn - xi);
        const dy = Math.sign(yn - yi);
        if (enforceStart && i === si[0] && j === si[1] && dx === -sv[0] && dy === -sv[1]) continue;
        if (enforceEnd && ni === ei[0] && nj === ei[1] && dx === ev[0] && dy === ev[1]) continue;
      }
      const len = nax === 0 ? Math.abs(xn - xi) : Math.abs(yn - yi);
      const free =
        nax === 0 ? clearH(yi, xi, xn, rects) : clearV(xi, yi, yn, rects);
      let mult = 1;
      if (!free) {
        const overNode =
          nax === 0 ? !clearH(yi, xi, xn, hard) : !clearV(xi, yi, yn, hard);
        mult = overNode ? BLOCKED_MULT : SOFT_MULT;
      }
      // Vertical travel carries an epsilon surcharge so equal-cost bend placements
      // resolve identically for every link (horizontal-first). Without it, sibling
      // links tie-break by expansion order into different shapes and braid. The
      // distance-from-target term additionally orders the two L-shapes of a plain
      // dogleg (identical H and V totals): descending AT the allocated stub lane
      // always prices under descending early at the source -- without it the choice
      // fell to expansion order and single links ignored their entry lane.
      const cost =
        len * mult * (nax === 1 ? 1.0005 + Math.abs(xi - ex) * 1e-6 : 1) +
        (axis !== -1 && axis !== nax ? BEND_COST : 0);
      const nkey = id(ni, nj) * 4 + (nax + 1);
      const ng = base + cost;
      if (ng < (g.get(nkey) ?? Infinity)) {
        g.set(nkey, ng);
        from.set(nkey, skey);
        open.push([ng + hCost(ni, nj), ni, nj, nax]);
      }
    }
  }
  return null;
}

// Collapse collinear runs so waypoints carry only real bends.
function simplify(pts: readonly XYPosition[]): XYPosition[] {
  const first = pts[0];
  const last = pts[pts.length - 1];
  if (first === undefined || last === undefined) return [];
  const out: XYPosition[] = [first];
  for (let k = 1; k < pts.length - 1; k++) {
    const a = out[out.length - 1];
    const b = pts[k];
    const c = pts[k + 1];
    if (a === undefined || b === undefined || c === undefined) continue;
    const abx = Math.abs(a.x - b.x) < EPS;
    const aby = Math.abs(a.y - b.y) < EPS;
    const bcx = Math.abs(b.x - c.x) < EPS;
    const bcy = Math.abs(b.y - c.y) < EPS;
    if ((abx && bcx) || (aby && bcy)) continue;
    if (abx && aby) continue;
    out.push(b);
  }
  out.push(last);
  return out;
}
