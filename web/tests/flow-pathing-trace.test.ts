import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { labelPoint, measure, pointAt, polylinePath } from "../src/flow/pathing/trace.ts";
import type { XYPosition } from "../src/flow/types.ts";

const pt = (x: number, y: number): XYPosition => ({ x, y });

// All numeric literals appearing in a serialized path, in order.
const numbersIn = (path: string): number[] =>
  [...path.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));

describe("measure", () => {
  test("segment lengths sum to the total", () => {
    const m = measure([pt(0, 0), pt(3, 4), pt(3, 10)]);
    assert.deepEqual(m.seg, [5, 6]);
    assert.equal(m.total, 11);
  });

  test("degenerate polylines measure to zero", () => {
    assert.deepEqual(measure([]), { seg: [], total: 0 });
    assert.deepEqual(measure([pt(1, 2)]), { seg: [], total: 0 });
  });
});

describe("pointAt", () => {
  test("t=0 is the first point and t=1 the last", () => {
    const pts = [pt(0, 0), pt(10, 0), pt(10, 20)];
    const m = measure(pts);
    const first = pointAt(pts, m, 0);
    assert.equal(first.x, 0);
    assert.equal(first.y, 0);
    const last = pointAt(pts, m, 1);
    assert.equal(last.x, 10);
    assert.equal(last.y, 20);
  });

  test("midpoint of a straight run lands at half length", () => {
    const pts = [pt(0, 0), pt(10, 0)];
    const m = measure(pts);
    const mid = pointAt(pts, m, 0.5);
    assert.equal(mid.x, 5);
    assert.equal(mid.y, 0);
    assert.equal(mid.angle, 0);
  });

  test("samples land on the carrying segment with its direction", () => {
    const pts = [pt(0, 0), pt(10, 0), pt(10, 20)];
    const m = measure(pts); // total 30
    const sample = pointAt(pts, m, 0.5); // target 15 -> 5 into the 20px riser
    assert.equal(sample.x, 10);
    assert.equal(sample.y, 5);
    assert.ok(Math.abs(sample.angle - Math.PI / 2) < 1e-12);
  });

  test("out-of-range t clamps to the endpoints", () => {
    const pts = [pt(0, 0), pt(10, 0)];
    const m = measure(pts);
    assert.equal(pointAt(pts, m, -1).x, 0);
    assert.equal(pointAt(pts, m, 2).x, 10);
  });

  test("degenerate input samples the first point at angle 0", () => {
    assert.deepEqual(pointAt([], measure([]), 0.5), { x: 0, y: 0, angle: 0 });
    assert.deepEqual(pointAt([pt(7, 8)], measure([pt(7, 8)]), 0.5), {
      x: 7,
      y: 8,
      angle: 0,
    });
  });
});

describe("labelPoint", () => {
  test("is the midpoint of the longest segment", () => {
    const pts = [pt(0, 0), pt(10, 0), pt(10, 40)]; // segments 10 and 40
    assert.deepEqual(labelPoint(pts), { x: 10, y: 20 });
  });

  test("ties keep the earliest longest segment", () => {
    assert.deepEqual(labelPoint([pt(0, 0), pt(4, 0), pt(8, 0)]), { x: 2, y: 0 });
    assert.deepEqual(labelPoint([pt(0, 0), pt(6, 0), pt(6, 6)]), { x: 3, y: 0 });
  });

  test("degenerate input mirrors pointAt fallbacks", () => {
    assert.deepEqual(labelPoint([]), { x: 0, y: 0 });
    assert.deepEqual(labelPoint([pt(7, 8)]), { x: 7, y: 8 });
  });
});

describe("polylinePath", () => {
  test("single point emits a lone moveto", () => {
    assert.equal(polylinePath([pt(3, 4)]), "M3,4");
    assert.equal(polylinePath([]), "");
  });

  test("two points emit one straight segment", () => {
    const path = polylinePath([pt(0, 0), pt(10, 0)]);
    assert.equal(path, "M0,0 L10,0");
  });

  test("a three-point right angle fillets through the corner", () => {
    const path = polylinePath([pt(0, 0), pt(10, 0), pt(10, 10)], 10);
    assert.ok(path.startsWith("M"), path);
    assert.ok(path.includes("Q"), path);
    const nums = numbersIn(path);
    // M(2) + before(2) + corner(2) + after(2) + last(2)
    assert.equal(nums.length, 10);
    const [, , beforeX, beforeY, cornerX, cornerY, afterX, afterY] = nums;
    // Corner radius clamps to a THIRD of each adjacent segment (10px here ->
    // 10/3), so the trim never overshoots the segment halves (5px).
    assert.ok(beforeX !== undefined && beforeX > 5 && beforeX < 10, String(beforeX));
    assert.equal(beforeY, 0);
    assert.equal(cornerX, 10);
    assert.equal(cornerY, 0);
    assert.equal(afterX, 10);
    assert.ok(afterY !== undefined && afterY > 0 && afterY < 5, String(afterY));
  });

  test("corner radius never overshoots tight segment halves", () => {
    const path = polylinePath([pt(0, 0), pt(2, 0), pt(2, 2)], 10);
    const nums = numbersIn(path);
    const beforeX = nums[2];
    const afterY = nums[7];
    // Segments are 2px; the fillet may consume at most 2/3 <= half of each.
    assert.ok(beforeX !== undefined && beforeX >= 1 - 1e-9 && beforeX < 2, String(beforeX));
    assert.ok(afterY !== undefined && afterY >= 2 / 3 - 1e-9 && afterY < 2, String(afterY));
  });

  test("radius parameter bounds the fillet size", () => {
    const wide = polylinePath([pt(0, 0), pt(30, 0), pt(30, 30)], 3);
    const wideNums = numbersIn(wide);
    // r = min(3, 10, 10) = 3 -> before sits 27px along the incoming run.
    assert.ok(wideNums[2] !== undefined && Math.abs(wideNums[2] - 27) < 1e-9);
    const tiny = polylinePath([pt(0, 0), pt(30, 0), pt(30, 30)], 0);
    const tinyNums = numbersIn(tiny);
    // A sub-0.5px radius hits the 0.5px floor that mirrors the reference emitter's
    // arc minimum.
    assert.ok(tinyNums[2] !== undefined && Math.abs(tinyNums[2] - 29.5) < 1e-9);
  });
});
