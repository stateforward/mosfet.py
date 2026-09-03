import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { offsetStrand, route } from "../src/flow/pathing/router.ts";
import type { Rect, XYPosition } from "../src/flow/types.ts";

const pt = (x: number, y: number): XYPosition => ({ x, y });
const box = (x: number, y: number, width: number, height: number): Rect => ({ x, y, width, height });

describe("route", () => {
  test("clear facing stubs return null so callers keep the straight spline", () => {
    const routed = route({ start: pt(100, 300), end: pt(900, 300), obstacles: [] });
    assert.equal(routed, null);
  });

  test("dead-level pins under flatTol return null (overlapgate F: straightening rule)", () => {
    const level = route({ start: pt(100, 300), end: pt(900, 300), obstacles: [] });
    assert.equal(level, null);
    const withinStroke = route({
      start: pt(100, 300),
      end: pt(900, 302),
      obstacles: [],
      flatTol: 3,
    });
    assert.equal(withinStroke, null);
  });

  test("a jog past flatTol still routes with a single end jog", () => {
    const routed = route({ start: pt(100, 300), end: pt(900, 302), obstacles: [], flatTol: 1 });
    assert.deepEqual(routed, [pt(100, 300), pt(876, 300), pt(876, 302), pt(900, 302)]);
  });

  test("tight corridor beside a crowding node routes straight inside the pin band (overlapgate C)", () => {
    // The crowder's real top edge (y=316) sits 8px under the lower pin's band --
    // inside the default 16px clearance ring but outside the 4px hard margin -- so
    // the shared lane is only soft-tolled and the wire must not detour around it.
    const crowder = box(420, 316, 220, 100);
    const routed = route({ start: pt(100, 300), end: pt(900, 308), obstacles: [crowder] });
    assert.notEqual(routed, null);
    if (routed === null) return;
    assert.ok(routed.length <= 4, `want <= 4 waypoints, got ${JSON.stringify(routed)}`);
    for (const p of routed) {
      assert.ok(p.y >= 299 && p.y <= 309, `waypoint ${JSON.stringify(p)} left the pin band`);
    }
  });

  test("a fully blocked corridor wraps the obstacle without entering its inflated rect", () => {
    const wall = box(150, -50, 100, 100);
    const routed = route({ start: pt(0, 0), end: pt(400, 0), obstacles: [wall] });
    assert.notEqual(routed, null);
    if (routed === null) return;
    let wrapped = false;
    for (const p of routed) {
      const strictlyInside =
        p.x > 133.01 && p.x < 266.99 && p.y > -65.99 && p.y < 65.99;
      assert.ok(!strictlyInside, `waypoint ${JSON.stringify(p)} entered the inflated wall`);
      if (Math.abs(p.y) >= 65) wrapped = true;
      assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), `non-finite waypoint ${JSON.stringify(p)}`);
    }
    assert.ok(wrapped, `no wrap around the wall: ${JSON.stringify(routed)}`);
    assert.equal(routed[0]?.x, 0);
    assert.equal(routed[routed.length - 1]?.x, 400);
  });

  test("identical inputs produce identical waypoint arrays (CORE-DET-001)", () => {
    const args = {
      start: pt(60, 60),
      end: pt(1500, 340),
      obstacles: [box(700, -400, 220, 830), box(700, 520, 220, 380)],
    };
    const first = route(args);
    const second = route(args);
    assert.notEqual(first, null);
    assert.deepEqual(first, second);
  });
});

describe("offsetStrand", () => {
  test("shifts perpendicular mid segments only and keeps endpoints pinned", () => {
    const template = [
      pt(0, 0),
      pt(24, 0),
      pt(24, 100),
      pt(200, 100),
      pt(200, 130),
      pt(224, 130),
    ];
    const start = pt(0, 0);
    const end = pt(224, 130);
    const plus = offsetStrand({ template, start, end, offset: 8 });
    assert.deepEqual(plus, [
      pt(0, 0),
      pt(32, 0),
      pt(32, 92),
      pt(208, 92),
      pt(208, 130),
      pt(224, 130),
    ]);
    const minus = offsetStrand({ template, start, end, offset: -8 });
    assert.deepEqual(minus, [
      pt(0, 0),
      pt(16, 0),
      pt(16, 108),
      pt(192, 108),
      pt(192, 130),
      pt(224, 130),
    ]);
  });

  test("returns null for degenerate templates", () => {
    const tooShort = [pt(0, 0), pt(10, 0), pt(10, 20)];
    assert.equal(offsetStrand({ template: tooShort, start: pt(0, 0), end: pt(30, 20), offset: 8 }), null);
    const notAlternating = [pt(0, 0), pt(10, 0), pt(20, 0), pt(30, 10)];
    assert.equal(
      offsetStrand({ template: notAlternating, start: pt(0, 0), end: pt(30, 10), offset: 8 }),
      null,
    );
    const diagonal = [pt(0, 0), pt(10, 10), pt(10, 20), pt(30, 20)];
    assert.equal(
      offsetStrand({ template: diagonal, start: pt(0, 0), end: pt(30, 20), offset: 8 }),
      null,
    );
  });

  test("re-anchors attachment segments on the member's own pins", () => {
    const template = [pt(0, 0), pt(24, 0), pt(24, 100), pt(200, 100), pt(200, 130), pt(224, 130)];
    const start = pt(0, 40);
    const end = pt(224, 90);
    const strand = offsetStrand({ template, start, end, offset: 0 });
    assert.notEqual(strand, null);
    if (strand === null) return;
    assert.deepEqual(strand[0], start);
    assert.deepEqual(strand[strand.length - 1], end);
    // First and last segments carry the member's own pin coordinates.
    assert.equal(strand[1]?.y, 40);
    assert.equal(strand[strand.length - 2]?.y, 90);
  });
});
