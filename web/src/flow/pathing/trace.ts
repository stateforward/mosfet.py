// Trace geometry helpers: serialize a filleted orthogonal polyline to an SVG path
// string, and arc-length math for the centre point and point-at-t (arrows, flow
// dots). Pure geometry: no DOM, no renderer state.

import { type XYPosition } from "../types.ts";

const RADIUS = 10;

export type MeasuredPath = {
  /** Segment lengths in px; `seg[k]` joins `pts[k]` to `pts[k + 1]`. */
  readonly seg: readonly number[];
  /** Total polyline length in px (fillets ignored). */
  readonly total: number;
};

export type PathSample = {
  readonly x: number;
  readonly y: number;
  /** Direction of the segment the sample lands on, radians, atan2 screen-space. */
  readonly angle: number;
};

function dist(a: XYPosition, b: XYPosition): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Arc-length measure of a polyline.
 *
 * Inputs: waypoints `[start, ...bends..., end]`. Outputs: per-segment lengths and
 * the total. Ownership: pure; the caller owns the result; `pts` is not retained.
 * Lifetime: one call -- pass the same `pts`/`MeasuredPath` pair to `pointAt`.
 * Concurrency: synchronous/pure. Failure modes: empty input measures `{ seg: [],
 * total: 0 }`. Units: world pixels. Classification: runtime-safe.
 */
export function measure(pts: readonly XYPosition[]): MeasuredPath {
  const seg: number[] = [];
  let total = 0;
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1];
    const b = pts[k];
    if (a === undefined || b === undefined) continue;
    const d = dist(a, b);
    seg.push(d);
    total += d;
  }
  return { seg, total };
}

/**
 * Point and direction at fraction `t` of total polyline length (fillets ignored;
 * the error is bounded by the fillet radius, fine for markers).
 *
 * Inputs: the same `pts` array `m` was measured from; `t` clamped to `[0, 1]`.
 * Outputs: world position plus segment direction in radians. Ownership: pure; the
 * caller owns the sample. Lifetime: one call. Concurrency: synchronous/pure.
 * Failure modes: degenerate input (no segments) samples the first point at
 * angle 0; out-of-range `t` clamps to an endpoint. Units: world px, radians.
 * Classification: runtime-safe.
 */
export function pointAt(pts: readonly XYPosition[], m: MeasuredPath, t: number): PathSample {
  const first = pts[0];
  const last = pts[pts.length - 1];
  if (first === undefined || last === undefined || m.seg.length === 0) {
    const anchor = first ?? last;
    return anchor === undefined ? { x: 0, y: 0, angle: 0 } : { x: anchor.x, y: anchor.y, angle: 0 };
  }
  let target = Math.max(0, Math.min(1, t)) * m.total;
  for (let k = 0; k < m.seg.length; k++) {
    const len = m.seg[k];
    const a = pts[k];
    const b = pts[k + 1];
    if (len === undefined || a === undefined || b === undefined) break;
    if (target <= len || k === m.seg.length - 1) {
      const f = len > 0 ? target / len : 0;
      return {
        x: a.x + (b.x - a.x) * f,
        y: a.y + (b.y - a.y) * f,
        angle: Math.atan2(b.y - a.y, b.x - a.x),
      };
    }
    target -= len;
  }
  return { x: last.x, y: last.y, angle: 0 };
}

/**
 * Centre point for an edge label: the midpoint of the longest segment.
 *
 * Inputs: waypoints `[start, ...bends..., end]`. Outputs: the midpoint of the
 * first longest segment (ties keep the earliest). Ownership: pure; the caller
 * owns the result; `pts` is not retained. Lifetime: one call. Concurrency:
 * synchronous/pure. Failure modes: an empty polyline samples the origin at
 * `{ x: 0, y: 0 }` and a single point samples that point, mirroring `pointAt`.
 * Units: world pixels. Classification: runtime-safe.
 */
export function labelPoint(pts: readonly XYPosition[]): XYPosition {
  const first = pts[0];
  if (first === undefined) return { x: 0, y: 0 };
  if (pts.length < 2) return { x: first.x, y: first.y };
  let best: XYPosition = { x: first.x, y: first.y };
  let bestLength = -1;
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1];
    const b = pts[k];
    if (a === undefined || b === undefined) continue;
    const length = dist(a, b);
    if (length > bestLength) {
      bestLength = length;
      best = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }
  }
  return best;
}

/**
 * Serialize a polyline to an SVG path string with rounded corners.
 *
 * Inputs: waypoints `[start, ...bends..., end]`; `radius` upper bound for corner
 * fillets in px (default 10). Outputs: `"M..."` path using L for straight runs and
 * Q for corners; `""` for an empty polyline. Each corner's radius is clamped to a
 * THIRD of the adjacent segments, not half: dense bundles have 12-30px segments,
 * and half-length fillets turn the whole trace into curve -- it then reads as a
 * spline instead of a routed trace. A 0.5px floor mirrors the reference emitter's
 * arc minimum so corners stay well-formed. Ownership: pure; the caller owns the
 * string; `pts` is not retained. Lifetime: one call. Concurrency:
 * synchronous/pure. Failure modes: zero-length adjacent segments emit a plain L
 * through the corner; non-finite coordinates propagate into the string verbatim.
 * Units: world pixels (caller maps to view space). Classification: runtime-safe.
 */
export function polylinePath(pts: readonly XYPosition[], radius: number = RADIUS): string {
  const first = pts[0];
  if (first === undefined) return "";
  const commands = [`M${first.x},${first.y}`];
  for (let k = 1; k < pts.length - 1; k++) {
    const p = pts[k - 1];
    const c = pts[k];
    const n = pts[k + 1];
    if (p === undefined || c === undefined || n === undefined) continue;
    const inLen = dist(p, c);
    const outLen = dist(c, n);
    if (inLen === 0 || outLen === 0) {
      commands.push(`L${c.x},${c.y}`);
      continue;
    }
    const r = Math.max(0.5, Math.min(radius, inLen / 3, outLen / 3));
    const before = {
      x: c.x - ((c.x - p.x) / inLen) * r,
      y: c.y - ((c.y - p.y) / inLen) * r,
    };
    const after = {
      x: c.x + ((n.x - c.x) / outLen) * r,
      y: c.y + ((n.y - c.y) / outLen) * r,
    };
    commands.push(`L${before.x},${before.y}`, `Q${c.x},${c.y} ${after.x},${after.y}`);
  }
  const last = pts[pts.length - 1];
  if (last !== undefined && pts.length > 1) commands.push(`L${last.x},${last.y}`);
  return commands.join(" ");
}
