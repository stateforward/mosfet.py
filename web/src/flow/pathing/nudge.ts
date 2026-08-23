// Collinear-overlap separation (R2) and, as a direct consequence, collapsed-node
// fan-out (R5): interior segments sharing a lane are spread into parallel strands.
// Stub segments (first and last) never move -- they must land on the pin; shifting
// the first interior turn staggers the stub LENGTH instead, which is the fan-out.
// Pure geometry: no DOM, no graph host, no app state.

import { type Rect, type XYPosition } from "../types.ts";

const LANE = 8;
// Segments within this many px share a lane cluster. Must stay BELOW the pin ladder
// pitch: allocated neighbour lanes are deliberate geometry, and clustering them
// would re-spread the pin ladder back out to 8px.
const CLUSTER = 4;
const MIN_STUB = 12;
// Spans must share more than this along the lane to collide.
const OVERLAP = 2;
// Spread strands keep this many px off a node box.
const CLEAR = 6;

export type RouteEntry = {
  /**
   * Stable identity for this route (e.g. the edge id). Used only to break ordering
   * ties deterministically; two passes over equal inputs must order equally.
   */
  readonly key: string;
  /** Shared origin pin when the strand leaves a pin other strands also leave; omit otherwise. */
  readonly start?: XYPosition;
  /** Routed shape the pass copies from. Never mutated by `nudgePass`. */
  readonly raw: readonly XYPosition[];
  /**
   * Working shape: `nudgePass` replaces it with a fresh copy of `raw`, then
   * mutates those points in place. Writable because `XYPosition` is readonly;
   * assignable wherever `XYPosition[]` is consumed.
   */
  pts: { x: number; y: number }[];
};

/**
 * Separate interior segments that share a lane into parallel strands.
 *
 * Inputs: live route entries (`raw` read-only templates plus mutable working
 * `pts`) and obstacle rects in world px. Outputs: none -- each entry's `pts` is
 * replaced with a fresh copy of its `raw` and then shifted in place where clusters
 * collide. Ownership: mutates only `entry.pts` (property and point objects);
 * `raw` and `obstacles` are never touched. Lifetime: one call per layout change.
 * Concurrency: synchronous; not re-entrant on the same entries.
 * Failure modes: degenerate polylines contribute no segments and are left alone;
 * never throws for geometric inputs. Units: world pixels; shifts are multiples of
 * `LANE` (8px) around the cluster centre, biased into the free corridor.
 * Classification: runtime-safe.
 */
export function nudgePass(entries: readonly RouteEntry[], obstacles: readonly Rect[] = []): void {
  for (const e of entries) e.pts = e.raw.map((p) => ({ x: p.x, y: p.y }));

  spread(entries, 0, obstacles); // vertical runs, shift x
  spread(entries, 1, obstacles); // horizontal runs, shift y
}

/** One collected interior segment: its owner, waypoint index, lane coord, span. */
type Seg = {
  readonly e: RouteEntry;
  readonly k: number;
  readonly coord: number;
  readonly span: readonly [number, number];
};

function coordOf(p: { readonly x: number; readonly y: number }, axis: 0 | 1): number {
  return axis === 0 ? p.x : p.y;
}

function setCoord(p: { x: number; y: number }, axis: 0 | 1, value: number): void {
  if (axis === 0) p.x = value;
  else p.y = value;
}

function spread(entries: readonly RouteEntry[], axis: 0 | 1, obstacles: readonly Rect[]): void {
  // axis 0: vertical segments (constant x, shift x). axis 1: horizontal (shift y).
  const along: 0 | 1 = axis === 0 ? 1 : 0;
  const segs: Seg[] = [];
  for (const e of entries) {
    const pts = e.pts;
    for (let k = 1; k < pts.length - 2; k++) {
      const a = pts[k];
      const b = pts[k + 1];
      if (a === undefined || b === undefined) continue;
      const isV = Math.abs(a.x - b.x) < 0.1;
      if ((axis === 0 && isV) || (axis === 1 && !isV && Math.abs(a.y - b.y) < 0.1)) {
        const lo = coordOf(a, along);
        const hi = coordOf(b, along);
        segs.push({
          e,
          k,
          coord: coordOf(a, axis),
          span: [Math.min(lo, hi), Math.max(lo, hi)],
        });
      }
    }
  }
  segs.sort((s, t) => s.coord - t.coord || s.e.key.localeCompare(t.e.key));

  // Collision is PAIRWISE: within CLUSTER px on the lane axis AND longitudinally
  // overlapped by more than OVERLAP px. Groups form only through actual collision
  // edges -- proximity chains must not transfer membership. A span-disjoint segment
  // sitting between two properly spaced runs would otherwise bridge them into one
  // cluster and re-spread a ribbon it never touched (QA: the lora_stack kink opened
  // a gap between clip and vae strands whose own coords are a full lane apart).
  const parent: number[] = segs.map((_seg, i) => i);
  const find = (i: number): number => {
    const root = parent[i];
    if (root === undefined || root === i) return root ?? i;
    const top = find(root);
    parent[i] = top;
    return top;
  };
  for (let i = 0; i < segs.length; i++) {
    const si = segs[i];
    if (si === undefined) continue;
    for (let j = i + 1; j < segs.length; j++) {
      const sj = segs[j];
      if (sj === undefined) continue;
      if (sj.coord - si.coord > CLUSTER) break;
      const a = si.span;
      const b = sj.span;
      if (Math.min(a[1], b[1]) - Math.max(a[0], b[0]) > OVERLAP) parent[find(j)] = find(i);
    }
  }
  const groups = new Map<number, Seg[]>();
  segs.forEach((s, i) => {
    const root = find(i);
    const group = groups.get(root);
    if (group === undefined) groups.set(root, [s]);
    else group.push(s);
  });
  for (const comp of groups.values()) {
    if (comp.length > 1) spreadComponent(comp, axis, along, obstacles);
  }
}

// Free corridor around a cluster's lane: nearest obstacle edge on each side of the
// shift axis, among obstacles that longitudinally overlap the cluster's union span.
// A rect the lane already runs THROUGH straddles the centre and updates neither
// bound -- the router priced that; no sideways shuffle inside a node helps.
function corridor(
  cluster: readonly Seg[],
  axis: 0 | 1,
  obstacles: readonly Rect[],
  center: number,
): [number, number] {
  let s0 = Infinity;
  let s1 = -Infinity;
  for (const s of cluster) {
    if (s.span[0] < s0) s0 = s.span[0];
    if (s.span[1] > s1) s1 = s.span[1];
  }
  let lo = -Infinity;
  let hi = Infinity;
  for (const r of obstacles) {
    const rl = axis === 0 ? r.y : r.x;
    const rh = rl + (axis === 0 ? r.height : r.width);
    if (Math.min(rh, s1) - Math.max(rl, s0) <= OVERLAP) continue;
    const b0 = axis === 0 ? r.x : r.y;
    const b1 = b0 + (axis === 0 ? r.width : r.height);
    if (b1 <= center && b1 + CLEAR > lo) lo = b1 + CLEAR;
    else if (b0 >= center && b0 - CLEAR < hi) hi = b0 - CLEAR;
  }
  return [lo, hi];
}

/** A same-pin bundle of segments plus its union span and attachment directions. */
type StrandMeta = {
  readonly list: readonly Seg[];
  readonly span: readonly [number, number];
  readonly att: readonly { readonly y: number; readonly dir: number }[];
};

function spreadComponent(
  cluster: readonly Seg[],
  axis: 0 | 1,
  along: 0 | 1,
  obstacles: readonly Rect[],
): void {
  // Same-pin sticking (#3, Barney's rule): links leaving one shared pin -- a node
  // output or a reroute exit -- overlap while colinear and only separate where they
  // branch; the rule ends at the next anchor, which is where the entry's polyline
  // ends anyway. All of a strand's segments take ONE lane and one offset; strangers
  // still get spread away from the bundle.
  const strandKey = (s: Seg): string =>
    s.e.start !== undefined ? `${Math.trunc(s.e.start.x)},${Math.trunc(s.e.start.y)}` : s.e.key;
  const strands = new Map<string, Seg[]>();
  for (const s of cluster) {
    const key = strandKey(s);
    const list = strands.get(key);
    if (list === undefined) strands.set(key, [s]);
    else list.push(s);
  }
  if (strands.size < 2) return;
  // Anti-braid lane order, pairwise: for members L and R of one corridor, placing
  // L on the -perp side costs one crossing for every R attachment (the horizontal
  // entering/leaving R's run) that extends toward -perp THROUGH L's span, and vice
  // versa. Sorting by that comparator picks the right order for both the staircase
  // pattern (stepped entries, stepped exits) and the nested pattern (entries
  // inside each other's spans) -- no single-key ordering covers both. Comparator
  // may be non-transitive in pathological mixes; "prefer to avoid" is the spec.
  const meta: StrandMeta[] = [...strands.values()].map((list) => {
    const span: [number, number] = [Infinity, -Infinity];
    const att: { y: number; dir: number }[] = [];
    for (const s of list) {
      if (s.span[0] < span[0]) span[0] = s.span[0];
      if (s.span[1] > span[1]) span[1] = s.span[1];
      const pts = s.e.pts;
      const a = pts[s.k];
      const b = pts[s.k + 1];
      if (a === undefined || b === undefined) continue;
      const lane = coordOf(a, axis);
      const prev = pts[s.k - 1];
      const next = pts[s.k + 2];
      if (prev !== undefined) {
        att.push({ y: coordOf(a, along), dir: Math.sign(coordOf(prev, axis) - lane) || 0 });
      }
      if (next !== undefined) {
        att.push({ y: coordOf(b, along), dir: Math.sign(coordOf(next, axis) - lane) || 0 });
      }
    }
    return { list, span, att };
  });
  const covers = (span: readonly [number, number], y: number): boolean =>
    y > span[0] + 1 && y < span[1] - 1;
  const cost = (L: StrandMeta, R: StrandMeta): number => {
    let c = 0;
    for (const a of R.att) {
      if (a.dir < 0 && covers(L.span, a.y)) c++;
    }
    for (const a of L.att) {
      if (a.dir > 0 && covers(R.span, a.y)) c++;
    }
    return c;
  };
  meta.sort(
    (p, q) =>
      cost(p, q) - cost(q, p) ||
      (p.list[0]?.e.key ?? "").localeCompare(q.list[0]?.e.key ?? ""),
  );
  const mid = (meta.length - 1) / 2;
  // Bias the fan away from the nearest node instead of centring it blindly on the
  // lane -- the lane is an inflated node edge, so half the spread otherwise walks
  // INTO the node (polish round: bunched gap runs pushed under neighbours). Minimal
  // shift when the corridor fits the fan; corridor centre when it cannot (overflow
  // splits evenly instead of one side eating all of it).
  let cmin = Infinity;
  let cmax = -Infinity;
  for (const s of cluster) {
    if (s.coord < cmin) cmin = s.coord;
    if (s.coord > cmax) cmax = s.coord;
  }
  const centre = (cmin + cmax) / 2;
  const half = mid * LANE;
  const [lo, hi] = corridor(cluster, axis, obstacles, centre);
  let shift = 0;
  if (hi - lo >= 2 * half) {
    if (centre - half < lo) shift = lo - (centre - half);
    else if (centre + half > hi) shift = hi - (centre + half);
  } else {
    shift = (lo + hi) / 2 - centre; // narrow corridor: both bounds are finite here
  }
  meta.forEach((m, i) => {
    const offset = (i - mid) * LANE + shift;
    if (offset === 0) return;
    for (const s of m.list) {
      const pts = s.e.pts;
      const a = pts[s.k];
      const b = pts[s.k + 1];
      if (a === undefined || b === undefined) continue;
      const orig = coordOf(a, axis);
      const nv = orig + offset;
      // Keep the adjacent stubs honest: same side as the pin, never shorter than MIN_STUB.
      // (Perpendicularity is guaranteed -- simplify() merged collinear runs -- so a shift
      // only ever changes a stub's length, not its orientation.)
      if (s.k === 1) {
        const pin = pts[0];
        if (pin === undefined || !stubOk(coordOf(pin, axis), orig, nv)) continue;
      }
      if (s.k === pts.length - 3) {
        const pin = pts[pts.length - 1];
        if (pin === undefined || !stubOk(coordOf(pin, axis), orig, nv)) continue;
      }
      setCoord(a, axis, nv);
      setCoord(b, axis, nv);
    }
  });
}

function stubOk(pin: number, orig: number, nv: number): boolean {
  const d = nv - pin;
  return Math.sign(d) === Math.sign(orig - pin) && Math.abs(d) >= MIN_STUB;
}
