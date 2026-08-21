import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { FlowGraph } from "../src/flow/graph.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { getBezierPath, getNodesBounds, getStraightPath, getViewportForBounds } from "../src/flow/path.ts";
import type { PointerSampleData } from "../src/flow/types.ts";

registerFlowElements();

async function ticks(count = 3): Promise<void> {
  for (let i = 0; i < count; i += 1) {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  }
}

function pointerData(overrides: Partial<PointerSampleData> = {}): PointerSampleData {
  const origin = overrides.origin ?? { x: 10, y: 10 };
  const client = overrides.client ?? origin;
  return {
    pointerId: 1,
    client,
    viewport: overrides.viewport ?? client,
    world: overrides.world ?? client,
    buttons: 1,
    button: 0,
    pointerType: "mouse",
    shiftKey: false,
    metaKey: false,
    ctrlKey: false,
    origin,
    hit: overrides.hit ?? { kind: "empty" },
    eventType: overrides.eventType ?? "pointerdown",
    ...overrides,
  };
}

describe("flow-graph", () => {
  test("admits nodes and edges and fits the viewport via a defined element", async () => {
    const graph = document.createElement("flow-graph");
    assert.ok(graph instanceof FlowGraph);
    document.body.append(graph);
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 },
      { id: "b", position: { x: 200, y: 0 }, data: { label: "B" }, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b", type: "bezier" }];
    graph.fitView();
    await ticks();
    const viewport = graph.getViewport();
    assert.equal(graph.nodes.length, 2);
    assert.equal(graph.edges.length, 1);
    assert.ok(viewport.zoom > 0);
    const before = graph.getViewport();
    graph.setViewport({ x: 12, y: 8, zoom: 1.1 });
    await ticks();
    const after = graph.getViewport();
    assert.notEqual(`${before.x},${before.y},${before.zoom}`, `${after.x},${after.y},${after.zoom}`);
    assert.equal(after.zoom, 1.1);
    graph.remove();
  });

  test("pans through dispatched pointer events", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    const before = graph.getViewport();
    graph.dispatch(hsm.typedEvent(FlowGraph.pointerDownEvent, pointerData({
      eventType: "pointerdown",
      origin: { x: 40, y: 40 },
      client: { x: 40, y: 40 },
      viewport: { x: 40, y: 40 },
      hit: { kind: "empty" },
    })));
    graph.dispatch(hsm.typedEvent(FlowGraph.pointerSampleEvent, pointerData({
      eventType: "pointermove",
      origin: { x: 40, y: 40 },
      client: { x: 80, y: 90 },
      viewport: { x: 80, y: 90 },
      hit: { kind: "empty" },
    })));
    graph.dispatch(hsm.typedEvent(FlowGraph.pointerUpEvent, pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 40, y: 40 },
      client: { x: 80, y: 90 },
      viewport: { x: 80, y: 90 },
      hit: { kind: "empty" },
    })));
    await ticks();
    const after = graph.getViewport();
    assert.ok(Math.abs(after.x - before.x) + Math.abs(after.y - before.y) > 0);
    graph.remove();
  });

  test("path helpers cover bezier, straight, bounds, and viewport", () => {
    const [bezier] = getBezierPath({ sourceX: 0, sourceY: 0, targetX: 100, targetY: 80 });
    const [straight] = getStraightPath({ sourceX: 0, sourceY: 0, targetX: 10, targetY: 10 });
    assert.match(bezier, /^M0,0 C/);
    assert.equal(straight, "M0,0 L10,10");
    const bounds = getNodesBounds([
      { id: "a", position: { x: 10, y: 20 }, data: {}, width: 40, height: 20 },
      { id: "b", position: { x: 80, y: 40 }, data: {}, width: 20, height: 10 },
    ]);
    assert.deepEqual(bounds, { x: 10, y: 20, width: 90, height: 30 });
    const viewport = getViewportForBounds(bounds, 180, 90, 0.1, 2, 0);
    assert.equal(viewport.zoom, 2);
    assert.equal(viewport.x, 180 / 2 - (10 + 90 / 2) * 2);
    assert.equal(viewport.y, 90 / 2 - (20 + 30 / 2) * 2);
  });
});
