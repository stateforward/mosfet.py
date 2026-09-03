import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { nudgePass } from "../src/flow/pathing/nudge.ts";
import type { RouteEntry } from "../src/flow/pathing/nudge.ts";
import { route } from "../src/flow/pathing/router.ts";
import type { Rect, XYPosition } from "../src/flow/types.ts";

const pt = (x: number, y: number): XYPosition => ({ x, y });
const box = (x: number, y: number, width: number, height: number): Rect => ({ x, y, width, height });
const entry = (key: string, raw: XYPosition[], start?: XYPosition): RouteEntry => ({
  key,
  ...(start === undefined ? {} : { start }),
  raw,
  pts: [],
});

describe("nudgePass", () => {
  test("near-collinear runs with disjoint spans are never spread (overlapgate A)", () => {
    // Two identical stepped links far apart in x: their long horizontal runs land on
    // the SAME lane y with disjoint longitudinal spans. Lane proximity alone is not
    // a collision.
    const rawA = [pt(100, 200), pt(140, 200), pt(140, 330), pt(420, 330), pt(420, 360), pt(440, 360)];
    const rawB = rawA.map((p) => pt(p.x + 900, p.y));
    const a = entry("a", rawA);
    const b = entry("b", rawB);
    nudgePass([a, b], []);
    assert.deepEqual(a.pts, rawA);
    assert.deepEqual(b.pts, rawB);
    assert.notEqual(a.pts, a.raw); // working copy, never the raw template
  });

  test("a span-disjoint bridge between two spaced runs does not transfer cluster membership", () => {
    // P and R collide pairwise (2px apart on the lane axis, spans overlapped) and
    // form a legitimate two-strand cluster; Q sits between their lanes
    // coordinate-wise (within CLUSTER of both) but its span is disjoint from each
    // -- a proximity chain must not pull Q into the fan.
    const p = entry("P", [pt(0, 80), pt(0, 100), pt(500, 100), pt(500, 120)], pt(0, 80));
    const r = entry("R", [pt(0, 130), pt(0, 102), pt(500, 102), pt(500, 80)], pt(0, 130));
    const q = entry("Q", [pt(560, 101), pt(600, 101), pt(700, 101), pt(740, 101)]);
    nudgePass([p, r, q], []);
    assert.deepEqual(q.pts, q.raw);
    const pLane = p.pts[1]?.y;
    const rLane = r.pts[1]?.y;
    assert.equal(pLane, 96); // 100 - LANE/2
    assert.equal(rLane, 106); // 102 + LANE/2, stub guards honoured (same side as pin)
    assert.notEqual(pLane, rLane);
  });

  test("forced gap crossings ride the gap then fan into distinct lanes clear of both blockers", () => {
    // gapbias scenario: five links forced through the 90px gap between stacked
    // blocker walls all pick the same lane (the upper wall's inflated bottom edge)
    // and the biased fan spreads them without walking any strand into a box.
    const upper = box(700, -400, 220, 830); // occupies y [-400, 430]
    const lower = box(700, 520, 220, 380); // occupies y [520, 900]
    const obstacles = [upper, lower];
    const entries: RouteEntry[] = [];
    for (let i = 0; i < 5; i += 1) {
      const sy = 60 + i * 70;
      const routed = route({ start: pt(60, sy), end: pt(1500, sy), obstacles });
      if (routed === null) assert.fail(`link ${i} failed to route through the gap`);
      entries.push(entry(`l${i}`, [...routed], pt(60, sy)));
    }
    // Pre-nudge: every link's long mid run shares one lane -- the upper blocker's
    // inflated bottom edge (430 + clearance 16).
    const gapLane = 446;
    for (const [i, e] of entries.entries()) {
      const ys = crossingLaneYs(e.raw);
      assert.ok(ys.length > 0, `link ${i} has no blocker-crossing run`);
      for (const y of ys) {
        assert.ok(
          Math.abs(y - gapLane) < 0.01,
          `link ${i} lane ${String(y)} missed the shared gap lane ${gapLane}`,
        );
      }
    }

    nudgePass(entries, obstacles);

    const lanes = new Set<number>();
    for (const e of entries) {
      for (const y of crossingLaneYs(e.pts)) lanes.add(Math.round(y));
    }
    assert.equal(lanes.size, 5); // spread survived the corridor bias
    const minLane = Math.min(...lanes);
    const maxLane = Math.max(...lanes);
    // CLEAR=6px off both boxes: strands live inside [436, 514].
    assert.ok(minLane >= 430 + 6 - 0.5, `top strand ${minLane} grazed the upper blocker`);
    assert.ok(maxLane <= 520 - 6 + 0.5, `bottom strand ${maxLane} grazed the lower blocker`);

    // No waypoint sits strictly inside either blocker inflated by the router's 4px
    // hard margin.
    const inflated = obstacles.map((r) => ({
      x0: r.x - 4,
      y0: r.y - 4,
      x1: r.x + r.width + 4,
      y1: r.y + r.height + 4,
    }));
    for (const e of entries) {
      for (const p of e.pts) {
        for (const b of inflated) {
          const strictlyInside =
            p.x > b.x0 + 1e-9 && p.x < b.x1 - 1e-9 && p.y > b.y0 + 1e-9 && p.y < b.y1 - 1e-9;
          assert.ok(!strictlyInside, `waypoint ${JSON.stringify(p)} entered an inflated blocker`);
        }
      }
    }
  });

  test("same-pin strands stick at the shared pin while strangers get spread away", () => {
    const pin = pt(500, 40);
    // Both strands leave ONE shared pin with lanes 2px apart -- close enough to
    // collide pairwise. Same-pin sticking keeps the bundle intact (one strand);
    // keyed per-entry they would have fanned apart.
    const first = entry("f", [pt(500, 40), pt(500, 100), pt(800, 100), pt(800, 140)], pin);
    const second = entry("s", [pt(500, 40), pt(500, 102), pt(800, 102), pt(800, 180)], pin);
    const stranger = entry("x", [pt(900, 101), pt(920, 101), pt(1020, 101), pt(1040, 101)]);
    nudgePass([first, second, stranger], []);
    assert.deepEqual(first.pts, first.raw);
    assert.deepEqual(second.pts, second.raw);
    assert.deepEqual(stranger.pts, stranger.raw);
  });
});

// Longitudinal-interior horizontal runs whose x-span overlaps the blocker column by
// at least 20px -- "the wire actually crosses the walls".
function crossingLaneYs(pts: readonly XYPosition[]): number[] {
  const ys: number[] = [];
  for (let k = 1; k < pts.length - 2; k++) {
    const a = pts[k];
    const b = pts[k + 1];
    if (a === undefined || b === undefined) continue;
    const horiz = Math.abs(a.y - b.y) < 0.1;
    const lo = Math.min(a.x, b.x);
    const hi = Math.max(a.x, b.x);
    if (horiz && Math.min(hi, 920) - Math.max(lo, 700) >= 20) ys.push(a.y);
  }
  return ys;
}
