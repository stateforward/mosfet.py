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
const nodeRect = (id: string, rect: Rect): NodeRectData => ({
  id,
  x: rect.x,
  y: rect.y,
  width: rect.width,
  height: rect.height,
});
const edgeOf = (id: string, source: string, target: string): CableEdgeData => ({ id, source, target });

// Source and target face each other across a wall no straight corridor crosses.
const SRC = box(0, 0, 80, 40);
const WALL = box(200, -300, 60, 600);
const TGT = box(400, 0, 80, 40);

const NODES: readonly NodeRectData[] = [
  nodeRect("src", SRC),
  nodeRect("wall", WALL),
  nodeRect("tgt", TGT),
];

function syncOf(edges: readonly CableEdgeData[], dragging: readonly string[]): SyncData {
  return { nodes: NODES, edges, draggingNodeIds: dragging };
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

/** No waypoint may sit strictly inside the wall inflated by the router's hard margin. */
function assertAvoidsWall(pts: readonly XYPosition[]): void {
  const hard = { x0: WALL.x - 4, y0: WALL.y - 4, x1: WALL.x + WALL.width + 4, y1: WALL.y + WALL.height + 4 };
  for (const p of pts) {
    const inside = p.x > hard.x0 && p.x < hard.x1 && p.y > hard.y0 && p.y < hard.y1;
    assert.ok(!inside, `waypoint ${JSON.stringify(p)} entered the wall`);
  }
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
    // Some bend must leave the wall's band: the wire went around, not through.
    assert.ok(
      e1.some((p) => Math.abs(p.y) > 305),
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
