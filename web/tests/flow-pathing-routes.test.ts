import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { cableEnds, Routes } from "../src/flow/pathing/routes.ts";
import type { CableEdgeData, CableEnds, NodeRectData, SyncData } from "../src/flow/pathing/routes.ts";
import type { Rect, XYPosition } from "../src/flow/types.ts";

const YIELD_MS = 0;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for routes");
}

const pt = (x: number, y: number): XYPosition => ({ x, y });
const box = (x: number, y: number, width: number, height: number): Rect => ({ x, y, width, height });
const nodeRect = (id: string, rect: Rect, parentId?: string): NodeRectData => ({
  id,
  x: rect.x,
  y: rect.y,
  width: rect.width,
  height: rect.height,
  ...(parentId === undefined ? {} : { parentId }),
});
const edgeOf = (id: string, source: string, target: string): CableEdgeData => ({ id, source, target });

// Source and target face each other across a wall no straight corridor crosses.
// Short enough that wrapping stays clearly cheaper than any punch-through under
// the shadow-pad standoff (the reference routes with vPad>=8 always on).
const SRC = box(0, 0, 80, 40);
const WALL = box(200, -140, 60, 380);
const TGT = box(400, 0, 80, 40);

const NODES: readonly NodeRectData[] = [
  nodeRect("src", SRC),
  nodeRect("wall", WALL),
  nodeRect("tgt", TGT),
];

function syncOf(edges: readonly CableEdgeData[], dragging: readonly string[]): SyncData {
  return { nodes: NODES, edges, draggingNodeIds: dragging };
}

// Transitive containment: outer ⊃ shell ⊃ {a, blocker, b}. For the a→b edge
// BOTH container rects are rooms (their interior lanes stay open) while the
// sibling blocker stays furniture -- before ancestor exclusion the blanket
// rects taxed every interior lane uniformly and the wire ran straight through
// the blocker.
const OUTER = box(-120, -120, 1440, 1040);
const SHELL = box(0, 0, 1000, 800);
const NESTED_A = box(80, 380, 80, 40);
const SIBLING_BLOCKER = box(460, 300, 60, 360);
const NESTED_B = box(840, 380, 80, 40);

const NESTED_NODES: readonly NodeRectData[] = [
  nodeRect("outer", OUTER),
  nodeRect("shell", SHELL, "outer"),
  nodeRect("a", NESTED_A, "shell"),
  nodeRect("blocker", SIBLING_BLOCKER, "shell"),
  nodeRect("b", NESTED_B, "shell"),
];

function nestedSync(edges: readonly CableEdgeData[]): SyncData {
  return { nodes: NESTED_NODES, edges, draggingNodeIds: [] };
}

// Cross-container edge x∈s1 → y∈s2: each shell is a room along its own
// descendants' chains, so the route enters s2 to reach y -- but the unrelated
// machine shell m between the two machines and the sibling blocker d inside s2
// are furniture for this edge either way.
const SHELL_ONE = box(0, 0, 400, 400);
const CROSS_X = box(60, 180, 80, 40);
const FOREIGN_SHELL = box(500, -350, 200, 1100);
const SHELL_TWO = box(800, 0, 400, 400);
const INNER_BLOCKER = box(880, 100, 50, 250);
const CROSS_Y = box(1060, 180, 80, 40);

const CROSS_NODES: readonly NodeRectData[] = [
  nodeRect("s1", SHELL_ONE),
  nodeRect("x", CROSS_X, "s1"),
  nodeRect("m", FOREIGN_SHELL),
  nodeRect("s2", SHELL_TWO),
  nodeRect("d", INNER_BLOCKER, "s2"),
  nodeRect("y", CROSS_Y, "s2"),
];

function crossSync(edges: readonly CableEdgeData[]): SyncData {
  return { nodes: CROSS_NODES, edges, draggingNodeIds: [] };
}

type Fixture = {
  machine: Routes;
  inner: (event: hsm.DispatchEvent) => ReturnType<Routes["dispatch"]>;
  /** Narrowed `routes` maps, one per routed notification attempt, in order. */
  routed: Array<Record<string, readonly XYPosition[]>>;
  holdAck: () => void;
  releaseAck: () => void;
  rejectAcks: () => void;
  allowAcks: () => void;
};

function routedRoutesOf(value: unknown): Record<string, readonly XYPosition[]> | null {
  if (!hsm.isRecord(value) || !hsm.isRecord(value["routes"])) return null;
  const routes: Record<string, readonly XYPosition[]> = {};
  for (const [key, pts] of Object.entries(value["routes"])) {
    if (!Array.isArray(pts)) return null;
    routes[key] = pts;
  }
  return routes;
}

function startFixture(): Fixture {
  const machine = hsm.start({ instance: new Routes(), model: Routes.model });
  const inner = machine.dispatch.bind(machine);
  const routed: Array<Record<string, readonly XYPosition[]>> = [];
  let hold = false;
  let held: hsm.DispatchEvent | null = null;
  let fail = false;
  machine.dispatch = ((event: hsm.DispatchEvent) => {
    if (event.name === Routes.routedEvent.name) {
      const data = routedRoutesOf(event.data);
      if (data !== null) routed.push(data);
      if (fail) return Promise.reject(new Error("routed host drop"));
      if (hold && held === null) {
        held = event;
        return new Promise<never>(() => {});
      }
    }
    return inner(event);
  }) as Routes["dispatch"];
  return {
    machine,
    inner,
    routed,
    holdAck: () => {
      hold = true;
    },
    releaseAck: () => {
      hold = false;
      const pending = held;
      held = null;
      if (pending !== null) void inner(pending).catch(hsm.catchFailure());
    },
    rejectAcks: () => {
      fail = true;
    },
    allowAcks: () => {
      fail = false;
    },
  };
}

/** No waypoint may sit strictly inside `wall` inflated by the router's hard margin. */
function assertAvoidsWall(pts: readonly XYPosition[], wall: Rect = WALL): void {
  const hard = { x0: wall.x - 4, y0: wall.y - 4, x1: wall.x + wall.width + 4, y1: wall.y + wall.height + 4 };
  for (const p of pts) {
    const inside = p.x > hard.x0 && p.x < hard.x1 && p.y > hard.y0 && p.y < hard.y1;
    assert.ok(!inside, `waypoint ${JSON.stringify(p)} entered the wall`);
  }
}

/** No orthogonal SEGMENT may cross `rect` inflated by the router's hard margin. */
function assertSegmentsAvoid(pts: readonly XYPosition[], rect: Rect): void {
  const left = rect.x - 4;
  const right = rect.x + rect.width + 4;
  const top = rect.y - 4;
  const bottom = rect.y + rect.height + 4;
  for (let index = 1; index < pts.length; index += 1) {
    const start = pts[index - 1];
    const end = pts[index];
    assert.ok(start !== undefined && end !== undefined);
    assert.ok(
      start.x === end.x || start.y === end.y,
      `segment ${JSON.stringify(start)}->${JSON.stringify(end)} is not orthogonal`,
    );
    const crosses = start.x === end.x
      ? start.x > left && start.x < right && Math.max(start.y, end.y) > top && Math.min(start.y, end.y) < bottom
      : start.y > top && start.y < bottom && Math.max(start.x, end.x) > left && Math.min(start.x, end.x) < right;
    assert.ok(!crosses, `segment crosses ${JSON.stringify(rect)} at ${JSON.stringify([start, end])}`);
  }
}

// --- Polyline intersection helpers (pure geometry, mirrors the painter's view). ---

type Seg = readonly [XYPosition, XYPosition];

function ccw(a: XYPosition, b: XYPosition, c: XYPosition): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function onSegment(p: XYPosition, a: XYPosition, c: XYPosition): boolean {
  return (
    Math.min(a.x, c.x) <= p.x + 1e-9 &&
    p.x <= Math.max(a.x, c.x) + 1e-9 &&
    Math.min(a.y, c.y) <= p.y + 1e-9 &&
    p.y <= Math.max(a.y, c.y) + 1e-9 &&
    Math.abs(ccw(a, c, p)) < 1e-9
  );
}

function segmentsIntersect(p1: XYPosition, p2: XYPosition, p3: XYPosition, p4: XYPosition): boolean {
  const d1 = ccw(p3, p4, p1);
  const d2 = ccw(p3, p4, p2);
  const d3 = ccw(p1, p2, p3);
  const d4 = ccw(p1, p2, p4);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }
  if (Math.abs(d1) < 1e-9 && onSegment(p1, p3, p4)) return true;
  if (Math.abs(d2) < 1e-9 && onSegment(p2, p3, p4)) return true;
  if (Math.abs(d3) < 1e-9 && onSegment(p3, p1, p2)) return true;
  if (Math.abs(d4) < 1e-9 && onSegment(p4, p1, p2)) return true;
  return false;
}

/** Segments strictly between the two pin stubs: the routed mid-path. */
function interiorSegments(pts: readonly XYPosition[]): Seg[] {
  const out: Seg[] = [];
  for (let k = 1; k < pts.length - 2; k += 1) {
    const a = pts[k];
    const b = pts[k + 1];
    if (a === undefined || b === undefined) continue;
    if (Math.abs(a.x - b.x) < 0.01 && Math.abs(a.y - b.y) < 0.01) continue;
    out.push([a, b]);
  }
  return out;
}

/** Intersections between two polylines' interior segments. */
function interiorCrossCount(a: readonly XYPosition[], b: readonly XYPosition[]): number {
  let count = 0;
  for (const [p1, p2] of interiorSegments(a)) {
    for (const [p3, p4] of interiorSegments(b)) {
      if (segmentsIntersect(p1, p2, p3, p4)) count += 1;
    }
  }
  return count;
}

/** Non-adjacent self-intersections across one polyline's full segment list. */
function selfIntersectionCount(pts: readonly XYPosition[]): number {
  const all: Seg[] = [];
  for (let k = 0; k < pts.length - 1; k += 1) {
    const a = pts[k];
    const b = pts[k + 1];
    if (a === undefined || b === undefined) continue;
    if (Math.abs(a.x - b.x) < 0.01 && Math.abs(a.y - b.y) < 0.01) continue;
    all.push([a, b]);
  }
  let count = 0;
  for (let i = 0; i < all.length; i += 1) {
    for (let j = i + 2; j < all.length; j += 1) {
      if (i === 0 && j === all.length - 1) continue;
      const s1 = all[i];
      const s2 = all[j];
      if (s1 === undefined || s2 === undefined) continue;
      if (segmentsIntersect(s1[0], s1[1], s2[0], s2[1])) count += 1;
    }
  }
  return count;
}

describe("Routes", () => {
  test("emits an obstacle-avoiding multipoint route on sync and rests at idle after the ack", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined, "no routed payload captured");
    const e1 = routes["e1"];
    assert.ok(e1 !== undefined && e1.length >= 4, `expected multipoint waypoints, got ${JSON.stringify(e1)}`);
    assert.deepEqual(e1[0], pt(80, 20)); // source right-border anchor
    assert.deepEqual(e1[e1.length - 1], pt(400, 20)); // target left-border anchor
    assertAvoidsWall(e1);
    // Some bend must leave the wall's inflated band: the wire went around, not
    // through.
    assert.ok(
      e1.some((p) => p.y < WALL.y - 16 || p.y > WALL.y + WALL.height + 16),
      `route never left the wall band: ${JSON.stringify(e1)}`,
    );
    await hsm.stop(fix.machine);
  });

  test("defers a sync sent mid-pass and runs exactly one follow-up pass after the ack", async () => {
    const fix = startFixture();
    fix.holdAck();
    void fix.inner(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/routing$/.test(fix.machine.state()));
    assert.equal(fix.routed.length, 1);
    // Mid-pass sync: deferred behind the held ack, never a concurrent pass.
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => true);
    assert.equal(fix.routed.length, 1, "deferred sync started a pass before the ack");
    fix.releaseAck();
    await waitFor(() => fix.routed.length >= 2 && /\/idle$/.test(fix.machine.state()));
    assert.equal(fix.routed.length, 2, "expected exactly one follow-up pass");
    const followUp = fix.routed[1];
    assert.ok(followUp !== undefined);
    // The fixture stores each pass's narrowed edge->waypoints map; the
    // follow-up consumed the SECOND sync, which dropped every edge.
    assert.deepEqual(followUp, {}, "follow-up pass must consume the SECOND sync, not replay the first");
    await hsm.stop(fix.machine);
  });

  test("stopping mid-held-ack cancels cleanly without unhandled rejection", async () => {
    const fix = startFixture();
    fix.holdAck();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/routing$/.test(fix.machine.state()));
    assert.equal(fix.routed.length, 1);
    // Queue a second sync behind the held ack, then cancel the machine while
    // the notification is still outstanding.
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([], []) }))
      .catch(hsm.catchFailure());
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      await hsm.stop(fix.machine);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    // Yield so any late rejection microtask would surface before judging.
    await waitFor(() => true);
    assert.deepEqual(unhandled, [], "cancellation must not surface an unhandled rejection");
    assert.equal(fix.machine.state(), "", "stopped machine must report an empty state");
    assert.equal(hsm.hostWasStopped(fix.machine), true);
    assert.equal(fix.routed.length, 1, "no further pass may run after stop");
    // Post-stop ingress classifies as a host drop instead of resurrecting a
    // pass; the rejection is consumed here by the assertion itself.
    await assert.rejects(() =>
      fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
    );
  });

  test("omits dragged-endpoint edges from routes and restores them post-drag", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], ["src"]) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const during = fix.routed[0];
    assert.ok(during !== undefined);
    assert.equal(
      during["e1"],
      undefined,
      "an edge touching a dragging node must be absent (paint fallback signal)",
    );
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 2 && /\/idle$/.test(fix.machine.state()));
    const restored = fix.routed[fix.routed.length - 1]?.["e1"];
    assert.ok(restored !== undefined && restored.length >= 4, "post-drag sync must restore the edge route");
    assertAvoidsWall(restored);
    await hsm.stop(fix.machine);
  });

  test("a rejected routed notification enters failed and the next valid sync recovers", async () => {
    const fix = startFixture();
    fix.rejectAcks();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/failed$/.test(fix.machine.state()));
    fix.allowAcks();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/idle$/.test(fix.machine.state()) && fix.routed.length >= 2);
    const recovered = fix.routed[fix.routed.length - 1]?.["e1"];
    assert.ok(recovered !== undefined && recovered.length >= 4, "recovery pass must emit routes again");
    await hsm.stop(fix.machine);
  });

  test("malformed sync payloads are ignored while idle and while failed", async () => {
    const fix = startFixture();
    const malformed: unknown[] = [
      undefined,
      "sync",
      42,
      {},
      { nodes: "nope", edges: [], draggingNodeIds: [] },
      { nodes: [], edges: [], draggingNodeIds: "nope" },
      { nodes: [{ id: "a", x: Number.NaN, y: 0, width: 10, height: 10 }], edges: [], draggingNodeIds: [] },
      { nodes: [{ id: "a", x: 0, y: 0, width: 10, height: 10, parentId: 7 }], edges: [], draggingNodeIds: [] },
      { nodes: [{ id: "a", x: 0, y: 0, width: 10, height: 10, parentId: null }], edges: [], draggingNodeIds: [] },
      { nodes: [{ x: 0, y: 0, width: 10, height: 10 }], edges: [], draggingNodeIds: [] },
      { nodes: [], edges: [{ id: "e", source: "a" }], draggingNodeIds: [] },
      { nodes: [], edges: [null], draggingNodeIds: [] },
      { nodes: [], edges: [], draggingNodeIds: [7] },
    ];
    for (const data of malformed) {
      void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data })).catch(hsm.catchFailure());
    }
    await waitFor(() => true);
    assert.match(fix.machine.state(), /\/idle$/, "malformed syncs must not move an idle machine");
    assert.equal(fix.routed.length, 0, "malformed syncs must not emit routes");

    // Same rule while failed: only a VALID sync recovers.
    fix.rejectAcks();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/failed$/.test(fix.machine.state()));
    for (const data of malformed) {
      void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data })).catch(hsm.catchFailure());
    }
    await waitFor(() => true);
    assert.match(fix.machine.state(), /\/failed$/, "malformed syncs must not move a failed machine");
    fix.allowAcks();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => /\/idle$/.test(fix.machine.state()) && fix.routed.length >= 2);
    await hsm.stop(fix.machine);
  });

  test("a first sync with no cable edges and an empty cache stays idle without emitting", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => true);
    assert.match(fix.machine.state(), /\/idle$/);
    assert.equal(fix.routed.length, 0);
    await hsm.stop(fix.machine);
  });

  test("routes around an intermediate sibling inside nested shells (containers are rooms)", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: nestedSync([edgeOf("e1", "a", "b")]) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    assert.ok(e1 !== undefined && e1.length >= 4, `expected a detour polyline, got ${JSON.stringify(e1)}`);
    assert.deepEqual(e1[0], pt(160, 400)); // a's right-border anchor
    assert.deepEqual(e1[e1.length - 1], pt(840, 400)); // b's left-border anchor
    // The sibling blocker is furniture even though both ancestor containers
    // are in the snapshot: the wire prices its way around it.
    assertSegmentsAvoid(e1, SIBLING_BLOCKER);
    assert.ok(
      e1.some((p) => p.y < 290 || p.y > 670),
      `route never left the blocker band: ${JSON.stringify(e1)}`,
    );
    await hsm.stop(fix.machine);
  });

  test("cross-container edges keep foreign shells as obstacles but enter the target's room", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: crossSync([edgeOf("e1", "x", "y")]) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    assert.ok(e1 !== undefined && e1.length >= 4, `expected a detour polyline, got ${JSON.stringify(e1)}`);
    assert.deepEqual(e1[0], pt(140, 200)); // x's right-border anchor
    assert.deepEqual(e1[e1.length - 1], pt(1060, 200)); // y's left-border anchor
    // The unrelated machine shell m is on neither endpoint's ancestor chain,
    // so it still bites; the wire leaves m's band to get across.
    assertAvoidsWall(e1, FOREIGN_SHELL);
    assert.ok(
      e1.some((p) => Math.abs(p.y) > 350),
      `route never left the foreign shell band: ${JSON.stringify(e1)}`,
    );
    // s2 IS an ancestor of y: the route legitimately enters its interior --
    // and the sibling blocker d inside s2 is still furniture to dodge.
    assert.ok(
      e1.some((p) => p.x > 946 && p.y > 4 && p.y < 396),
      `route never entered s2's interior: ${JSON.stringify(e1)}`,
    );
    assertSegmentsAvoid(e1, INNER_BLOCKER);
    await hsm.stop(fix.machine);
  });

  test("fans a same-node-pair bundle into distinct ribbon strands", async () => {
    const fix = startFixture();
    const edges = [edgeOf("e1", "src", "tgt"), edgeOf("e2", "src", "tgt")];
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf(edges, []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    const e2 = routes["e2"];
    assert.ok(e1 !== undefined && e1.length >= 4, "bundle member e1 missing a strand");
    assert.ok(e2 !== undefined && e2.length >= 4, "bundle member e2 missing a strand");
    assert.notEqual(JSON.stringify(e1), JSON.stringify(e2), "bundle strands must differ mid-path");
    assert.deepEqual(e1[0], pt(80, 20));
    assert.deepEqual(e2[0], pt(80, 20));
    assertAvoidsWall(e1);
    assertAvoidsWall(e2);
    await hsm.stop(fix.machine);
  });

  // Two stacked source nodes wire to two stacked targets on the same side while a
  // pair of furniture blocks sits between the stacks. With stub folds globally
  // banned each cable shimmies onto its sibling's lanes (the upstream "adjacent
  // crossing shimmy"); with folds allowed at node pins both cables nest cleanly.
  const SHIMMY_S1 = box(0, 0, 80, 40);
  const SHIMMY_S2 = box(0, 63, 80, 40);
  const SHIMMY_T1 = box(228, 1, 80, 40);
  const SHIMMY_T2 = box(228, 68, 80, 40);
  const SHIMMY_NODES: readonly NodeRectData[] = [
    nodeRect("s1", SHIMMY_S1),
    nodeRect("s2", SHIMMY_S2),
    nodeRect("t1", SHIMMY_T1),
    nodeRect("t2", SHIMMY_T2),
    nodeRect("b1", box(108, -116, 86, 88)),
    nodeRect("b2", box(105, -12, 57, 134)),
  ];

  test("keeps parallel same-side cables between stacked nodes crossing-free", async () => {
    const fix = startFixture();
    const edges = [edgeOf("e1", "s1", "t1"), edgeOf("e2", "s2", "t2")];
    const data: SyncData = { nodes: SHIMMY_NODES, edges, draggingNodeIds: [] };
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    const e2 = routes["e2"];
    assert.ok(e1 !== undefined && e1.length >= 4, "parallel cable e1 missing");
    assert.ok(e2 !== undefined && e2.length >= 4, "parallel cable e2 missing");
    assert.equal(
      interiorCrossCount(e1, e2),
      0,
      `same-side cables must not cross: ${JSON.stringify(e1)} vs ${JSON.stringify(e2)}`,
    );
    await hsm.stop(fix.machine);
  });

  test("stands horizontal lanes off obstacle tops by clearance plus the shadow pad", async () => {
    const fix = startFixture();
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: syncOf([edgeOf("e1", "src", "tgt")], []) }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    assert.ok(e1 !== undefined && e1.length >= 4, "expected a wrap-around polyline");
    // The wrap lane is an inflated wall edge; the shadow pad widens the inflation so
    // horizontal runs stand vPad further off the top/bottom than bare clearance.
    for (const p of e1.slice(1, -1)) {
      const d = Math.min(Math.abs(p.y - WALL.y), Math.abs(p.y - (WALL.y + WALL.height)));
      assert.ok(
        d >= 24 - 1,
        `waypoint ${JSON.stringify(p)} rides ${d.toFixed(1)}px off the wall edge; want >= ${24 - 1}`,
      );
    }
    await hsm.stop(fix.machine);
  });

  // Same-pair bundles whose median template degenerates (a straight aligned pair
  // collapses to fewer than four usable points): every strand fails, so members
  // keep their own raws instead of ribbon geometry. The painter contract is that
  // each member still carries its full pin-to-pin polyline untouched.
  const FLAT_S = box(0, 0, 80, 40);
  const FLAT_T = box(400, 0, 80, 40);

  test("keeps failed-strand bundle members on their own pin-to-pin raws", async () => {
    const fix = startFixture();
    const flatNodes: readonly NodeRectData[] = [nodeRect("s", FLAT_S), nodeRect("t", FLAT_T)];
    const edges = [edgeOf("e1", "s", "t"), edgeOf("e2", "s", "t"), edgeOf("e3", "s", "t")];
    const data: SyncData = { nodes: flatNodes, edges, draggingNodeIds: [] };
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    for (const id of ["e1", "e2", "e3"]) {
      const pts: readonly XYPosition[] | undefined = routes[id];
      assert.ok(pts !== undefined && pts.length >= 2, `member ${id} missing its polyline`);
      assert.deepEqual(pts[0], pt(80, 20), `member ${id} must start on s's border anchor`);
      assert.deepEqual(pts[pts.length - 1], pt(400, 20), `member ${id} must end on t's border anchor`);
    }
    await hsm.stop(fix.machine);
  });

  // A forward edge plus reverse edges between the same unordered node pair share
  // one ribbon (pairKey normalizes direction). Their border anchors differ, so the
  // ladder order must come from projecting member starts onto the template's first
  // mid-segment travel direction -- a border-stack sort puts the forward member on
  // the wrong side and its lane crosses the reversed strands.
  const STAIR_A = box(17, 105, 78, 42);
  const STAIR_B = box(487, 28, 60, 29);
  const STAIR_NODES: readonly NodeRectData[] = [
    nodeRect("a", STAIR_A),
    nodeRect("b", STAIR_B),
    nodeRect("w1", box(237, -57, 86, 88)),
    nodeRect("w2", box(-23, -28, 22, 109)),
  ];

  test("ranks mixed-direction ribbon members by travel-direction projection without crossings", async () => {
    const fix = startFixture();
    const edges = [edgeOf("f1", "a", "b"), edgeOf("r1", "b", "a"), edgeOf("r2", "b", "a")];
    const data: SyncData = { nodes: STAIR_NODES, edges, draggingNodeIds: [] };
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const f1 = routes["f1"];
    const r1 = routes["r1"];
    const r2 = routes["r2"];
    for (const pts of [f1, r1, r2]) {
      assert.ok(pts !== undefined && pts.length >= 4, "every ribbon member needs a routed polyline");
    }
    assert.ok(f1 !== undefined && r1 !== undefined && r2 !== undefined);
    assert.equal(interiorCrossCount(f1, r1), 0, "f1 vs r1 lanes cross");
    assert.equal(interiorCrossCount(f1, r2), 0, "f1 vs r2 lanes cross");
    assert.equal(interiorCrossCount(r1, r2), 0, "r1 vs r2 lanes cross");
    await hsm.stop(fix.machine);
  });

  // Facing anchors closer than the combined stub depth make the stub tips cross;
  // unsqueezed, A* folds a loop around them (measured upstream: one self-crossing
  // per link). Both stubs scale into the gap preserving their ratio.
  const SHORT_GAP_NODES: readonly NodeRectData[] = [
    nodeRect("s", box(0, 0, 80, 40)),
    nodeRect("t", box(104, 0, 80, 40)),
    nodeRect("block", box(84, -30, 12, 48)),
  ];

  test("squeezes facing stubs into short gaps so short-gap wires never self-intersect", async () => {
    const fix = startFixture();
    const edges = [edgeOf("e1", "s", "t")];
    const data: SyncData = { nodes: SHORT_GAP_NODES, edges, draggingNodeIds: [] };
    void fix.machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data }))
      .catch(hsm.catchFailure());
    await waitFor(() => fix.routed.length >= 1 && /\/idle$/.test(fix.machine.state()));
    const routes = fix.routed[0];
    assert.ok(routes !== undefined);
    const e1 = routes["e1"];
    assert.ok(e1 !== undefined && e1.length >= 2, "short-gap cable missing");
    assert.equal(
      selfIntersectionCount(e1),
      0,
      `short-gap cable self-intersects: ${JSON.stringify(e1)}`,
    );
    await hsm.stop(fix.machine);
  });
});

describe("cableEnds facing inference", () => {
  const cases: Array<[string, Rect, Rect, CableEnds]> = [
    ["target right exits right, enters left", SRC, TGT, {
      start: pt(80, 20),
      end: pt(400, 20),
      startDir: "right",
      endDir: "left",
    }],
    ["target left exits left, enters right", box(400, 0, 80, 40), box(0, 0, 80, 40), {
      start: pt(400, 20),
      end: pt(80, 20),
      startDir: "left",
      endDir: "right",
    }],
    ["target below exits bottom, enters top", box(0, 0, 80, 40), box(0, 300, 80, 40), {
      start: pt(40, 40),
      end: pt(40, 300),
      startDir: "bottom",
      endDir: "top",
    }],
    ["target above exits top, enters bottom", box(0, 300, 80, 40), box(0, 0, 80, 40), {
      start: pt(40, 300),
      end: pt(40, 40),
      startDir: "top",
      endDir: "bottom",
    }],
    ["shallow diagonal takes the horizontal faces", box(0, 0, 80, 40), box(100, 20, 80, 40), {
      start: pt(80, 20),
      end: pt(100, 40),
      startDir: "right",
      endDir: "left",
    }],
    ["steep diagonal takes the vertical faces", box(0, 0, 80, 40), box(20, 100, 80, 40), {
      start: pt(40, 40),
      end: pt(60, 100),
      startDir: "bottom",
      endDir: "top",
    }],
    ["equal dominance ties horizontal toward +x", box(0, 0, 80, 40), box(50, 50, 80, 40), {
      start: pt(80, 20),
      end: pt(50, 70),
      startDir: "right",
      endDir: "left",
    }],
    ["overlapping centres take the deterministic default", box(0, 0, 80, 40), box(0, 0, 80, 40), {
      start: pt(80, 20),
      end: pt(0, 20),
      startDir: "right",
      endDir: "left",
    }],
  ];
  for (const [name, source, target, expected] of cases) {
    test(name, () => {
      assert.deepEqual(cableEnds(source, target), expected);
    });
  }

  test("anchors always sit ON the rect border mid-side", () => {
    const source = box(10, 20, 30, 40); // centre (25, 40)
    const target = box(500, 60, 70, 80); // centre (535, 100); dominant axis horizontal
    const ends = cableEnds(source, target);
    assert.ok(ends.start.x === source.x || ends.start.x === source.x + source.width);
    assert.equal(ends.start.y, 40); // source vertical midpoint
    assert.equal(ends.end.x, 500); // target left border
    assert.equal(ends.end.y, 100); // target horizontal midpoint
  });
});
