import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { FlowGraph } from "../src/flow/graph.ts";
import { FlowEdge } from "../src/flow/edge.ts";
import { FlowNode } from "../src/flow/node.ts";
import { FlowNodeResizer } from "../src/flow/node-resizer.ts";
import { RESIZE_DIRECTIONS } from "../src/flow/resize-control.ts";
import { resizedBounds } from "../src/flow/resizer.ts";
import { Selection } from "../src/flow/selection.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { getBezierPath, getNodesBounds, getSmoothStepPath, getStraightPath, getViewportForBounds } from "../src/flow/path.ts";
import {
  FIT_PADDING_RATIO,
  MAX_FLOW_EDGES,
  MAX_FLOW_NODES,
  MAX_JSON_DEPTH,
  MAX_ZOOM,
  MIN_ZOOM,
  type Node,
  type PointerSampleData,
} from "../src/flow/types.ts";
import { getByRole } from "./by-role.ts";
import { labelPoint } from "../src/flow/pathing/trace.ts";

registerFlowElements();

const POINTER_ID = 1;
const POINTER_BUBBLES = true;
const POINTER_COMPOSED = true;

function pointerInit(args: { clientX: number; clientY: number }): PointerEventInit {
  return {
    clientX: args.clientX,
    clientY: args.clientY,
    pointerId: POINTER_ID,
    bubbles: POINTER_BUBBLES,
    composed: POINTER_COMPOSED,
  };
}

async function selectFirstNode(nodeEl: FlowNode): Promise<void> {
  const button = nodeEl.shadowRoot?.querySelector("button");
  assert.ok(button instanceof HTMLButtonElement);
  const clientX = 10;
  const clientY = 10;
  button.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX, clientY })));
  button.dispatchEvent(new PointerEvent("pointerup", pointerInit({ clientX, clientY })));
  await waitUntil(() => nodeEl.node?.selected === true);
}

function offeredResizer(nodeEl: FlowNode): FlowNodeResizer {
  const resizer = nodeEl.shadowRoot?.querySelector("flow-node-resizer");
  assert.ok(resizer instanceof FlowNodeResizer);
  assert.equal(resizer.hidden, false);
  return resizer;
}

function controlOf(args: { resizer: FlowNodeResizer; direction: string }): HTMLElement {
  const control = args.resizer.shadowRoot?.querySelector(`flow-node-resize-control[direction="${args.direction}"]`);
  assert.ok(control instanceof HTMLElement);
  return control;
}

function keyDownOn(args: { target: Element; key: string }): void {
  args.target.dispatchEvent(new KeyboardEvent("keydown", {
    key: args.key,
    bubbles: POINTER_BUBBLES,
    composed: POINTER_COMPOSED,
    cancelable: true,
  }));
}

function controlButtonOf(control: HTMLElement): HTMLButtonElement {
  const button = control.shadowRoot?.querySelector("button");
  assert.ok(button instanceof HTMLButtonElement);
  return button;
}

const YIELD_MS = 0;
const publicEventBubbles = true;
const publicEventComposed = true;
const publicEventCancelable = false;

function assertPublicCustomEvent(event: Event): void {
  assert.equal(event.cancelable, publicEventCancelable);
  assert.equal(event.bubbles, publicEventBubbles);
  assert.equal(event.composed, publicEventComposed);
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function ownedActors(host: { context(): hsm.Context }): hsm.Instance[] {
  const instances = host.context().Value(hsm.Keys.Instances);
  if (typeof instances !== "object" || instances === null) return [];
  const actors: hsm.Instance[] = [];
  for (const value of Object.values(instances as Record<string, unknown>)) {
    if (value === host || typeof value !== "object" || value === null) continue;
    if (typeof (value as { context?: unknown }).context !== "function") continue;
    if (typeof (value as { state?: unknown }).state !== "function") continue;
    if ((value as hsm.Instance).context().Value(hsm.Keys.Owner) !== host) continue;
    actors.push(value as hsm.Instance);
  }
  return actors;
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for flow-graph");
}

function pointerData(overrides: Partial<PointerSampleData> = {}): PointerSampleData {
  const origin = overrides.origin ?? { x: 10, y: 10 };
  const client = overrides.client ?? origin;
  const eventType = overrides.eventType ?? "pointerdown";
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
    eventType,
    originalEvent: overrides.originalEvent ?? {
      pointerId: 1,
      clientX: client.x,
      clientY: client.y,
      type: eventType,
    },
    ...overrides,
  };
}

/**
 * Painted `d` of the graph's first flow-edge path, or null before any paint.
 */
function edgePathDOf(graph: FlowGraph): string | null {
  const element = graph.querySelector("flow-edge");
  if (!(element instanceof FlowEdge)) return null;
  return element.path.getAttribute("d");
}

/**
 * Waypoints recovered from a `polylinePath` serialization.
 *
 * Interior corners survive as Q control points -- fillets only trim their
 * surrounding L/Q endpoints -- so the recovered set equals the routed
 * waypoint list. Returns an empty list while no polyline is painted.
 */
function cableWaypointsOf(d: string | null): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = [];
  if (d === null || !d.startsWith("M")) return points;
  const start = d.match(/^M(-?[0-9.]+),(-?[0-9.]+)/);
  if (start === null) return points;
  points.push({ x: Number(start[1]), y: Number(start[2]) });
  for (const match of d.matchAll(/Q(-?[0-9.]+),(-?[0-9.]+)/g)) {
    points.push({ x: Number(match[1]), y: Number(match[2]) });
  }
  const lineTos = [...d.matchAll(/L(-?[0-9.]+),(-?[0-9.]+)/g)];
  const last = lineTos[lineTos.length - 1];
  if (last !== undefined) points.push({ x: Number(last[1]), y: Number(last[2]) });
  return points;
}

async function waitForCablePolyline(
  graph: FlowGraph,
  start: { x: number; y: number },
): Promise<Array<{ x: number; y: number }>> {
  for (let i = 0; i < 50; i += 1) {
    const pts = cableWaypointsOf(edgePathDOf(graph));
    // Gate on the routed border anchor: a painted smoothstep fallback also
    // recovers >=4 points but starts at the node-bottom spline anchor.
    if (pts.length >= 4 && pts[0]?.x === start.x && pts[0]?.y === start.y) return pts;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for a routed cable polyline");
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
    await flush();
    const viewport = graph.getViewport();
    assert.equal(graph.nodes.length, 2);
    assert.equal(graph.edges.length, 1);
    assert.ok(viewport.zoom > 0);
    const before = graph.getViewport();
    graph.setViewport({ x: 12, y: 8, zoom: 1.1 });
    await flush();
    const after = graph.getViewport();
    assert.notEqual(`${before.x},${before.y},${before.zoom}`, `${after.x},${after.y},${after.zoom}`);
    assert.equal(after.zoom, 1.1);
    graph.remove();
  });

  test("remove then append still admits nodes", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await flush();
    assert.equal(graph.nodes[0]?.id, "a");
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    graph.nodes = [{ id: "b", position: { x: 8, y: 8 }, data: { label: "B" }, width: 80, height: 40 }];
    await flush();
    assert.equal(graph.nodes[0]?.id, "b");
    graph.remove();
  });

  test("pans through dispatched pointer events", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    const before = graph.getViewport();
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 40, clientY: 40, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/pan$/.test(graph.state()));
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 80, clientY: 90, pointerId: 1, bubbles: true, composed: true,
    }));
    graph.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 80, clientY: 90, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => {
      const now = graph.getViewport();
      return Math.abs(now.x - before.x) + Math.abs(now.y - before.y) > 0;
    });
    const after = graph.getViewport();
    assert.ok(Math.abs(after.x - before.x) + Math.abs(after.y - before.y) > 0);
    graph.remove();
  });

  test("pointermove while panning writes the world transform in the same synchronous turn", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 20, clientY: 20, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/pan$/.test(graph.state()));
    const viewport = graph.querySelector('[part="viewport"]');
    const world = viewport?.querySelector("div") as unknown as { style: { transform: string } } | null;
    assert.ok(world !== null);
    const before = world.style.transform ?? "";
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 60, clientY: 70, pointerId: 1, bubbles: true, composed: true,
    }));
    // Same turn: no setTimeout(0) flush and no host dispatch hop between the
    // pointermove and the world transform write.
    const after = world.style.transform ?? "";
    assert.notEqual(after, before);
    assert.match(after, /translate\(40px, 50px\) scale\(1\)/);
    graph.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 60, clientY: 70, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    graph.remove();
  });

  test("interleaved nodes_set during pan does not delay the pan transform", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.nodes.length === 1);
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 20, clientY: 20, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/pan$/.test(graph.state()));
    // Host dispatch tail in flight: nodes_set queued on the host while panning.
    graph.nodes = [{ id: "b", position: { x: 0, y: 0 }, data: { label: "B" }, width: 80, height: 40 }];
    const viewport = graph.querySelector('[part="viewport"]');
    const world = viewport?.querySelector("div") as unknown as { style: { transform: string } } | null;
    assert.ok(world !== null);
    const before = world.style.transform ?? "";
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 70, clientY: 80, pointerId: 1, bubbles: true, composed: true,
    }));
    // The panner dispatch is independent of the host tail: the transform still
    // lands in the same synchronous turn as the pointermove.
    const after = world.style.transform ?? "";
    assert.notEqual(after, before);
    assert.match(after, /translate\(50px, 60px\) scale\(1\)/);
    await flush();
    graph.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 70, clientY: 80, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    graph.remove();
  });

  test("setNodes during pan stays in pan and pointer_up still ends pan", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 20, y: 20 },
      client: { x: 20, y: 20 },
      viewport: { x: 20, y: 20 },
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/pan$/);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    assert.match(graph.state(), /\/pan$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 20, y: 20 },
      client: { x: 40, y: 40 },
      viewport: { x: 40, y: 40 },
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    const admittedNodeCount = 1;
    assert.equal(graph.nodes.length, admittedNodeCount);
    graph.remove();
  });

  test("shift+empty pointer_down boxes and does not pan", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    const before = graph.getViewport();
    const selected: string[] = [];
    graph.addEventListener("flow-selection-change", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail) || !Array.isArray(event.detail["nodes"])) return;
      for (const node of event.detail["nodes"]) {
        if (hsm.isRecord(node) && typeof node["id"] === "string") selected.push(node["id"]);
      }
    });
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      shiftKey: true,
      origin: { x: 10, y: 10 },
      client: { x: 10, y: 10 },
      viewport: { x: 10, y: 10 },
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/box$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      shiftKey: true,
      origin: { x: 10, y: 10 },
      client: { x: 120, y: 80 },
      viewport: { x: 120, y: 80 },
      hit: { kind: "empty" },
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      shiftKey: true,
      buttons: 0,
      origin: { x: 10, y: 10 },
      client: { x: 120, y: 80 },
      viewport: { x: 120, y: 80 },
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    assert.deepEqual(graph.getViewport(), before);
    assert.ok(selected.includes("a"));
    graph.remove();
  });

  test("empty pointer_down pans and does not box when panOnDrag", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    const before = graph.getViewport();
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 20, y: 20 },
      client: { x: 20, y: 20 },
      viewport: { x: 20, y: 20 },
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/pan$/);
    assert.doesNotMatch(graph.state(), /\/box$/);
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 60, clientY: 70, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 20, y: 20 },
      client: { x: 60, y: 70 },
      viewport: { x: 60, y: 70 },
      hit: { kind: "empty" },
    }) }));
    await flush();
    const after = graph.getViewport();
    assert.ok(Math.abs(after.x - before.x) + Math.abs(after.y - before.y) > 0);
    graph.remove();
  });

  test("node pointer_down past click threshold drags and does not pan", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const node = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [node];
    graph.nodesDraggable = true;
    graph.panOnDrag = true;
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof HTMLElement);
    const nodeX = 0;
    nodeEl.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 10, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/click$/);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 50, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/drag$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    // The drag-start sample arrives before drag_start, so the Dragger ignores
    // it: no node move in the same turn.
    assert.equal(graph.nodes[0]?.position.x, nodeX);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 80, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    // The first sample while dragging establishes the dedupe baseline.
    await flush();
    assert.equal(graph.nodes[0]?.position.x, nodeX);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 110, clientY: 20, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => graph.nodes[0]?.position.x !== nodeX);
    nodeEl.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 80, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    assert.notEqual(graph.nodes[0]?.position.x, 0);
    graph.remove();
  });

  test("pointer samples drive drag_moved per sample without a timer poll", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodesDraggable = true;
    graph.panOnDrag = true;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof HTMLElement);
    const noMove = 0;
    nodeEl.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 10, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/click$/);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 50, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/drag$/);
    // The drag-start sample arrives before drag_start, so the Dragger ignores
    // it: a held pointer does not move the node yet.
    assert.equal(graph.nodes[0]?.position.x, noMove);
    assert.equal(graph.nodes[0]?.position.y, noMove);
    // The first sample while dragging establishes the dedupe baseline.
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 80, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    assert.equal(graph.nodes[0]?.position.x, noMove);
    assert.equal(graph.nodes[0]?.position.y, noMove);
    // Each later sample past the epsilon moves the node per sample: no 16ms
    // frame floor and no wall-clock wait between the sample and the move.
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 90, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => graph.nodes[0]?.position.x !== noMove || graph.nodes[0]?.position.y !== noMove);
    assert.equal(graph.nodes[0]?.position.x, 40);
    assert.equal(graph.nodes[0]?.position.y, 20);
    // A sub-epsilon sample is deduped and does not move the node.
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 90.005, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    assert.equal(graph.nodes[0]?.position.x, 40);
    assert.equal(graph.nodes[0]?.position.y, 20);
    // The next past-epsilon sample moves the node again per sample.
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 95, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => graph.nodes[0]?.position.x === 45 && graph.nodes[0]?.position.y === 20);
    assert.equal(graph.nodes[0]?.position.x, 45);
    assert.equal(graph.nodes[0]?.position.y, 20);
    nodeEl.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 95, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    const settled = { x: graph.nodes[0]?.position.x, y: graph.nodes[0]?.position.y };
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 120, clientY: 40, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    assert.deepEqual({ x: graph.nodes[0]?.position.x, y: graph.nodes[0]?.position.y }, settled);
    graph.remove();
  });

  test("kind-only node pointer_down does not enter drag", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodesDraggable = true;
    const kindOnlyNode = { kind: "node" };
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: { ...pointerData(), hit: kindOnlyNode },
    }));
    assert.doesNotMatch(graph.state(), /\/drag$/);
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerSampleEvent,
      data: {
        ...pointerData({
          eventType: "pointermove",
          origin: { x: 10, y: 10 },
          client: { x: 40, y: 10 },
          viewport: { x: 40, y: 10 },
          world: { x: 40, y: 10 },
        }),
        hit: kindOnlyNode,
      },
    }));
    assert.doesNotMatch(graph.state(), /\/drag$/);
    graph.remove();
  });

  test("kind-only edge pointer_down does not emit a click path into drag", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const kindOnlyEdge = { kind: "edge" };
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: { ...pointerData(), hit: kindOnlyEdge },
    }));
    assert.doesNotMatch(graph.state(), /\/drag$/);
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerSampleEvent,
      data: {
        ...pointerData({
          eventType: "pointermove",
          origin: { x: 10, y: 10 },
          client: { x: 40, y: 10 },
          viewport: { x: 40, y: 10 },
          world: { x: 40, y: 10 },
        }),
        hit: kindOnlyEdge,
      },
    }));
    assert.doesNotMatch(graph.state(), /\/drag$/);
    graph.remove();
  });

  test("incomplete handle pointer_down does not enter connect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    const pointerId = 1;
    const eventType = "pointerdown";
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: {
        pointerId,
        eventType,
        hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
      },
    }));
    assert.doesNotMatch(graph.state(), /\/connect$/);
    graph.remove();
  });

  test("handle pointer_down missing position does not enter connect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    const completeWithoutPosition = {
      ...pointerData(),
      hit: { kind: "handle", node: source, handleKind: "source" },
    };
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: completeWithoutPosition,
    }));
    assert.doesNotMatch(graph.state(), /\/connect$/);
    graph.remove();
  });

  test("handle pointer_down missing buttons does not enter connect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    const sample = pointerData({
      hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
    });
    const { buttons: _omittedButtons, ...withoutButtons } = sample;
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: withoutButtons,
    }));
    assert.doesNotMatch(graph.state(), /\/connect$/);
    graph.remove();
  });

  test("complete handle pointer_down enters connect without reconstructing in the guard", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: pointerData({
        hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
      }),
    }));
    assert.match(graph.state(), /\/connect$/);
    graph.remove();
  });

  test("handle pointer_down connects and finishes on a target handle", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 },
      { id: "b", position: { x: 200, y: 0 }, data: { label: "B" }, width: 80, height: 40 },
    ];
    const source = graph.nodes[0];
    const target = graph.nodes[1];
    assert.ok(source !== undefined);
    assert.ok(target !== undefined);
    const connected: Array<{ source: string; target: string }> = [];
    graph.addEventListener("flow-connect", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail)) return;
      const sourceId = event.detail["source"];
      const targetId = event.detail["target"];
      if (typeof sourceId === "string" && typeof targetId === "string") connected.push({ source: sourceId, target: targetId });
    });
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 80, y: 20 },
      client: { x: 80, y: 20 },
      world: { x: 80, y: 20 },
      hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
    }) }));
    assert.match(graph.state(), /\/connect$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 80, y: 20 },
      client: { x: 200, y: 20 },
      world: { x: 200, y: 20 },
      hit: { kind: "handle", node: target, handleKind: "target", position: "left" },
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 80, y: 20 },
      client: { x: 200, y: 20 },
      world: { x: 200, y: 20 },
      hit: { kind: "handle", node: target, handleKind: "target", position: "left" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    assert.deepEqual(connected, [{ source: "a", target: "b" }]);
    graph.remove();
  });

  test("detach reaches disconnected and reconnects on the defined element", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    assert.match(graph.state(), /\/connected\//);
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    document.body.append(graph);
    await flush();
    assert.match(graph.state(), /\/connected\//);
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
  });

  test("dropping a connect on empty cancels without flow-connect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    const connected: string[] = [];
    graph.addEventListener("flow-connect", () => {
      connected.push("connected");
    });
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
    }) }));
    assert.match(graph.state(), /\/connect$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    assert.deepEqual(connected, []);
    graph.remove();
  });

  test("overflow and invalid admits reject without mutating the graph", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const rejected: Array<{ reason: string; nodeCount: number; edgeCount: number }> = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail)) return;
      const reason = event.detail["reason"];
      const nodeCount = event.detail["nodeCount"];
      const edgeCount = event.detail["edgeCount"];
      if (typeof reason === "string" && typeof nodeCount === "number" && typeof edgeCount === "number") {
        rejected.push({ reason, nodeCount, edgeCount });
      }
    });
    const valid = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [valid];
    assert.equal(graph.nodes.length, 1);
    graph.nodes = Array.from({ length: MAX_FLOW_NODES + 1 }, (_item, index) => ({
      id: `n${index}`,
      position: { x: index, y: 0 },
      data: { label: `${index}` },
    }));
    assert.equal(graph.nodes.length, 1);
    graph.edges = Array.from({ length: MAX_FLOW_EDGES + 1 }, (_item, index) => ({
      id: `e${index}`,
      source: "a",
      target: "a",
    }));
    assert.equal(graph.edges.length, 0);
    graph.nodes = [{ id: "bad" } as never];
    assert.equal(graph.nodes.length, 1);
    assert.deepEqual(rejected.map((item) => item.reason), ["too_many_nodes", "too_many_edges", "invalid"]);
    const snapshot = [...graph.nodes];
    snapshot[0] = { id: "mutated", position: { x: 1, y: 1 }, data: {} };
    assert.equal(graph.nodes[0]?.id, "a");
    graph.remove();
  });

  test("poison nodes_set after copy success emits reject and keeps admitted nodes", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const valid = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [valid];
    await waitUntil(() => graph.nodes.length === 1);
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.setNodesEvent,
      data: { nodes: [{ id: "bad" }] },
    }));
    await flush();
    assert.equal(graph.nodes.length, 1);
    assert.equal(graph.nodes[0]?.id, "a");
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("poison nodes_set while panning stays in pan and keeps admitted nodes", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    const valid = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [valid];
    await waitUntil(() => graph.nodes.length === 1);
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 20,
      clientY: 20,
      pointerId: 1,
      bubbles: true,
      composed: true,
    }));
    assert.match(graph.state(), /\/pan$/);
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.setNodesEvent,
      data: { nodes: [{ id: "bad" }] },
    }));
    await flush();
    assert.match(graph.state(), /\/pan$/);
    const admittedNodeCount = 1;
    assert.equal(graph.nodes.length, admittedNodeCount);
    assert.equal(graph.nodes[0]?.id, "a");
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("Event-path cyclic nodes_set rejects without throw and keeps admitted nodes", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const valid = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [valid];
    await waitUntil(() => graph.nodes.length === 1);
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    const cyclic: Record<string, unknown> = { label: "cycle" };
    cyclic["self"] = cyclic;
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.setNodesEvent,
      data: { nodes: [{ id: "cycle", position: { x: 0, y: 0 }, data: cyclic }] },
    }));
    await flush();
    const admittedNodeCount = 1;
    assert.equal(graph.nodes.length, admittedNodeCount);
    assert.equal(graph.nodes[0]?.id, "a");
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("poison edges_set after copy success emits reject and keeps admitted edges", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const node = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    const valid = { id: "e", source: "a", target: "a" };
    graph.nodes = [node];
    graph.edges = [valid];
    await waitUntil(() => graph.edges.length === 1);
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.setEdgesEvent,
      data: { edges: [{ id: "bad" }] },
    }));
    await flush();
    assert.equal(graph.edges.length, 1);
    assert.equal(graph.edges[0]?.id, "e");
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("PointerEvent pan and box run on the defined element", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    const before = graph.getViewport();
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 20,
      clientY: 20,
      pointerId: 1,
      bubbles: true,
      composed: true,
    }));
    assert.match(graph.state(), /\/pan$/);
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 60,
      clientY: 70,
      pointerId: 1,
      bubbles: true,
      composed: true,
    }));
    await waitUntil(() => {
      const now = graph.getViewport();
      return Math.abs(now.x - before.x) + Math.abs(now.y - before.y) > 0;
    });
    graph.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 60,
      clientY: 70,
      pointerId: 1,
      bubbles: true,
      composed: true,
    }));
    await flush();
    const after = graph.getViewport();
    assert.ok(Math.abs(after.x - before.x) + Math.abs(after.y - before.y) > 0);
    graph.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 10,
      clientY: 10,
      pointerId: 1,
      shiftKey: true,
      bubbles: true,
      composed: true,
    }));
    assert.match(graph.state(), /\/box$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    graph.remove();
  });

  test("edge paint nodes remount after detach and reconnect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 },
      { id: "b", position: { x: 200, y: 0 }, data: { label: "B" }, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b", type: "bezier" }];
    await flush();
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    document.body.append(graph);
    await waitUntil(() => graph.nodes.length === 2 && graph.edges.length === 1);
    assert.equal(graph.nodes.length, 2);
    assert.equal(graph.edges.length, 1);
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
    const viewportWidth = 180;
    const viewportHeight = 90;
    const minZoom = 0.1;
    const maxZoom = 2;
    const noPadding = 0;
    const viewport = getViewportForBounds({
      bounds: {
        left: bounds.x,
        right: bounds.x + bounds.width,
        top: bounds.y,
        bottom: bounds.y + bounds.height,
      },
      origin: { x: 0, y: 0 },
      width: viewportWidth,
      height: viewportHeight,
      minZoom,
      maxZoom,
      padding: noPadding,
    });
    assert.equal(viewport.zoom, 2);
    assert.equal(viewport.x, 180 / 2 - (10 + 90 / 2) * 2);
    assert.equal(viewport.y, 90 / 2 - (20 + 30 / 2) * 2);
  });

  test("non-finite node coordinates reject as invalid", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    const valid = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [valid];
    assert.equal(graph.nodes.length, 1);
    graph.nodes = [{ id: "nan", position: { x: Number.NaN, y: 0 }, data: {} }];
    graph.nodes = [{ id: "inf", position: { x: 0, y: Number.POSITIVE_INFINITY }, data: {} }];
    assert.equal(graph.nodes.length, 1);
    assert.equal(graph.nodes[0]?.id, "a");
    assert.deepEqual(rejected, ["invalid", "invalid"]);
    graph.remove();
  });

  test("constructor does not start the host", () => {
    const graph = document.createElement("flow-graph");
    assert.equal(graph.state(), "");
  });

  test("policy setter before connect does not start the host", () => {
    const graph = document.createElement("flow-graph");
    const policyOff = false;
    graph.nodesDraggable = policyOff;
    graph.nodesResizable = policyOff;
    graph.panOnDrag = policyOff;
    assert.equal(graph.state(), "");
    assert.equal(graph.nodesDraggable, policyOff);
    assert.equal(graph.nodesResizable, policyOff);
    assert.equal(graph.panOnDrag, policyOff);
  });

  test("unstarted fitView and focusTarget emit host-drop unstarted", async () => {
    const graph = document.createElement("flow-graph");
    const atLeastOneDrop = 1;
    const unstarted = "unstarted";
    const drops: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({
          cancelable: event.cancelable,
          bubbles: event.bubbles,
          composed: event.composed,
          reason: event.detail["reason"],
        });
      }
    });
    graph.fitView();
    graph.focusTarget({
      kind: "viewport",
      bounds: { left: 0, right: 1, top: 0, bottom: 1 },
    });
    await flush();
    assert.equal(graph.state(), "");
    assert.ok(drops.length >= atLeastOneDrop);
    for (const drop of drops) {
      assert.equal(drop.cancelable, publicEventCancelable);
      assert.equal(drop.bubbles, publicEventBubbles);
      assert.equal(drop.composed, publicEventComposed);
      assert.equal(drop.reason, unstarted);
    }
    graph.remove();
  });

  test("fitView after stop emits host-drop stopped", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    const atLeastOneDrop = 1;
    const stopped = "stopped";
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    await graph.stop();
    graph.fitView();
    await flush();
    assert.equal(graph.state(), "");
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    graph.remove();
  });

  test("nodes write before connect applies after the child starts", async () => {
    const graph = document.createElement("flow-graph");
    const noNodes = 0;
    const admitted = 1;
    const drops: Event[] = [];
    graph.addEventListener("host-drop", (event: Event) => {
      drops.push(event);
    });
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await flush();
    assert.equal(graph.nodes.length, noNodes);
    assert.equal(drops.length, noNodes);
    assert.equal(graph.state(), "");
    document.body.append(graph);
    await waitUntil(() => graph.nodes.length === admitted);
    assert.equal(graph.nodes[0]?.id, "a");
    assert.equal(drops.length, noNodes);
    graph.remove();
  });

  test("mutating the caller nodes array after set does not change admitted nodes", async () => {
    const graph = document.createElement("flow-graph");
    const admitted = 1;
    const nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    graph.nodes = nodes;
    nodes.push({ id: "b", position: { x: 1, y: 1 }, data: { label: "B" }, width: 80, height: 40 });
    const spliceFromStart = 0;
    nodes.splice(spliceFromStart, admitted);
    document.body.append(graph);
    await waitUntil(() => graph.nodes.length === admitted);
    assert.equal(graph.nodes.length, admitted);
    assert.equal(graph.nodes[0]?.id, "a");
    graph.remove();
  });

  test("mutating the caller edges array after set does not change admitted edges", async () => {
    const graph = document.createElement("flow-graph");
    const admitted = 1;
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    const edges = [{ id: "a-b", source: "a", target: "b" }];
    graph.edges = edges;
    edges.push({ id: "extra", source: "a", target: "b" });
    const spliceFromStart = 0;
    edges.splice(spliceFromStart, admitted);
    document.body.append(graph);
    await waitUntil(() => graph.edges.length === admitted);
    assert.equal(graph.edges.length, admitted);
    assert.equal(graph.edges[0]?.id, "a-b");
    graph.remove();
  });

  test("edges write before connect applies after the child starts", async () => {
    const graph = document.createElement("flow-graph");
    const noEdges = 0;
    const admitted = 1;
    const drops: Event[] = [];
    graph.addEventListener("host-drop", (event: Event) => {
      drops.push(event);
    });
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b" }];
    await flush();
    assert.equal(graph.edges.length, noEdges);
    assert.equal(drops.length, noEdges);
    document.body.append(graph);
    await waitUntil(() => graph.edges.length === admitted);
    assert.equal(graph.edges[0]?.id, "a-b");
    assert.equal(drops.length, noEdges);
    graph.remove();
  });

  test("set nodes while disconnected apply after reconnect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: {} }];
    await flush();
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    graph.nodes = [{ id: "b", position: { x: 1, y: 1 }, data: {} }];
    document.body.append(graph);
    await waitUntil(() => graph.nodes[0]?.id === "b");
    assert.equal(graph.nodes.length, 1);
    graph.remove();
  });

  test("viewport is a labeled group landmark", async () => {
    const graph = document.createElement("flow-graph");
    const roleMissingBeforeConnect = false;
    assert.equal(graph.hasAttribute("role"), roleMissingBeforeConnect);
    document.body.append(graph);
    assert.equal(graph.getAttribute("role"), "group");
    assert.equal(graph.getAttribute("aria-label"), "Machine graph");
    graph.remove();
  });

  test("author display survives flow-edge connect", async () => {
    const edge = document.createElement("flow-edge");
    const displayUnset = "";
    assert.equal(edge.style.display ?? "", displayUnset);
    edge.style.display = "block";
    document.body.append(edge);
    assert.equal(edge.style.display, "block");
    edge.remove();
  });

  test("author accessible name survives connect defaults", async () => {
    const graph = document.createElement("flow-graph");
    graph.setAttribute("aria-label", "Custom graph");
    graph.setAttribute("role", "group");
    document.body.append(graph);
    assert.equal(graph.getAttribute("aria-label"), "Custom graph");
    assert.equal(graph.getAttribute("role"), "group");
    graph.remove();
  });

  test("application keys zoom pan and fit through the public viewport", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 },
      { id: "b", position: { x: 200, y: 0 }, data: { label: "B" }, width: 80, height: 40 },
    ];
    await flush();
    graph.focus();
    const before = graph.getViewport();
    const keyConsumed = true;
    const zoomKey = new KeyboardEvent("keydown", { key: "+", bubbles: true, cancelable: true });
    graph.dispatchEvent(zoomKey);
    assert.equal(zoomKey.defaultPrevented, keyConsumed);
    await flush();
    const zoomed = graph.getViewport();
    assert.ok(zoomed.zoom > before.zoom);
    const panKey = new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true });
    graph.dispatchEvent(panKey);
    assert.equal(panKey.defaultPrevented, keyConsumed);
    await flush();
    const panned = graph.getViewport();
    assert.ok(panned.x !== zoomed.x);
    const fitKey = new KeyboardEvent("keydown", { key: "f", bubbles: true, cancelable: true });
    graph.dispatchEvent(fitKey);
    assert.equal(fitKey.defaultPrevented, keyConsumed);
    await flush();
    const fitted = graph.getViewport();
    assert.notEqual(fitted.zoom, panned.zoom);
    const ignored = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    graph.dispatchEvent(ignored);
    const keyNotConsumed = false;
    assert.equal(ignored.defaultPrevented, keyNotConsumed);
    graph.remove();
  });

  test("application keys leave slotted native controls alone", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const zoom = document.createElement("button");
    graph.append(zoom);
    await flush();
    const panKey = new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true });
    zoom.dispatchEvent(panKey);
    const keyNotConsumed = false;
    assert.equal(panKey.defaultPrevented, keyNotConsumed);
    graph.remove();
  });

  test("clickable nodes are keyboard and AT focusable", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const nodeWidth = 80;
    const nodeHeight = 40;
    const originLeft = 0;
    const originTop = 0;
    const expectedTabIndex = 0;
    const keyNotConsumed = false;
    const eventBubbles = true;
    const eventCancelable = true;
    const eventComposed = true;
    const enterKey = "Enter";
    const spaceKey = " ";
    graph.nodes = [
      { id: "a", position: { x: originLeft, y: originTop }, data: { label: "A", path: "/A", machineName: "/A" }, width: nodeWidth, height: nodeHeight },
    ];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const node = graph.querySelector("flow-node");
    assert.ok(node instanceof HTMLElement);
    const control = getByRole(graph, "button", "A");
    assert.ok(control instanceof HTMLButtonElement);
    assert.equal(node.getAttribute("role"), null);
    assert.equal(control.getAttribute("aria-label"), "A");
    assert.equal(node.getAttribute("data-testid"), "state-node");
    assert.equal(control.tabIndex, expectedTabIndex);
    const noActivedescendant = null;
    graph.focusTarget({
      kind: "machine",
      machineName: "/A",
      bounds: { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight },
    });
    await flush();
    assert.notEqual(document.activeElement, control);
    assert.equal(graph.getAttribute("aria-activedescendant"), noActivedescendant);
    graph.focusTarget({
      kind: "viewport",
      bounds: { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight },
    });
    await flush();
    assert.notEqual(document.activeElement, control);
    graph.focusTarget({
      kind: "node",
      nodeId: "a",
      nodePath: "/A",
      machineName: "/A",
      bounds: { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight },
    });
    await flush();
    assert.equal(document.activeElement, control);
    const origins: unknown[] = [];
    graph.addEventListener("flow-node-click", (event) => {
      if (event instanceof CustomEvent) origins.push(event.detail.originalEvent);
    });
    const keyInit = { bubbles: eventBubbles, cancelable: eventCancelable, composed: eventComposed };
    const enter = new KeyboardEvent("keydown", { ...keyInit, key: enterKey });
    control.dispatchEvent(enter);
    await flush();
    const noneActivated = 0;
    assert.equal(enter.defaultPrevented, keyNotConsumed);
    assert.equal(origins.length, noneActivated);
    const space = new KeyboardEvent("keydown", { ...keyInit, key: spaceKey });
    control.dispatchEvent(space);
    await flush();
    assert.equal(space.defaultPrevented, keyNotConsumed);
    assert.equal(origins.length, noneActivated);
    const clickInit = { bubbles: eventBubbles, cancelable: eventCancelable, composed: eventComposed };
    control.dispatchEvent(new Event("click", clickInit));
    await flush();
    const activated = 1;
    assert.equal(origins.length, activated);
    assert.deepEqual(origins[0], { type: "click" });
    graph.remove();
  });

  test("node_activate_click and node_activate_key types accepted keys and rejects unknown keys", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const nodeWidth = 80;
    const nodeHeight = 40;
    const originLeft = 0;
    const originTop = 0;
    graph.nodes = [
      { id: "a", position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
    ];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const origins: unknown[] = [];
    graph.addEventListener("flow-node-click", (event) => {
      if (event instanceof CustomEvent) origins.push(event.detail.originalEvent);
    });
    const nodeId = "a";
    const enterKey = "Enter";
    const spaceKey = " ";
    const tabKey = "Tab";
    const unknownKey = "Escape";
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateClickEvent, data: { nodeId } }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateKeyEvent, data: { nodeId, key: enterKey } }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateKeyEvent, data: { nodeId, key: spaceKey } }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateKeyEvent, data: { nodeId, key: tabKey } }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateKeyEvent, data: { nodeId, key: unknownKey } }));
    const accepted = 3;
    assert.equal(origins.length, accepted);
    assert.deepEqual(origins[0], { type: "click" });
    assert.deepEqual(origins[1], { type: "keydown", key: enterKey });
    assert.deepEqual(origins[2], { type: "keydown", key: spaceKey });
    graph.remove();
  });

  test("focus_machine, focus_node, and focus_viewport during pan keep fitted viewport through pointer_up", async () => {
    const originLeft = 0;
    const originTop = 0;
    const nodeWidth = 80;
    const nodeHeight = 40;
    const pointerOriginX = 20;
    const pointerOriginY = 20;
    const panClientX = 60;
    const panClientY = 70;
    const panDeltaX = panClientX - pointerOriginX;
    const panDeltaY = panClientY - pointerOriginY;
    const buttonsReleased = 0;
    const machineName = "/A";
    const nodeId = "a";
    const nodePath = "/A";
    const bounds = { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight };
    const pointerOrigin = { x: pointerOriginX, y: pointerOriginY };
    const panClient = { x: panClientX, y: panClientY };
    const kinds = [
      { kind: "machine" as const, machineName, bounds },
      { kind: "node" as const, nodeId, nodePath, machineName, bounds },
      { kind: "viewport" as const, bounds },
    ];
    for (const target of kinds) {
      const graph = document.createElement("flow-graph");
      document.body.append(graph);
      graph.panOnDrag = true;
      graph.nodes = [
        { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
      ];
      await waitUntil(() => graph.querySelector("flow-node") !== null);
      const before = graph.getViewport();
      await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
        eventType: "pointerdown",
        origin: pointerOrigin,
        client: pointerOrigin,
        viewport: pointerOrigin,
        hit: { kind: "empty" },
      }) }));
      assert.match(graph.state(), /\/pan$/);
      graph.focusTarget(target);
      await flush();
      assert.match(graph.state(), /\/pan$/);
      const fitted = graph.getViewport();
      assert.notEqual(`${fitted.x},${fitted.y},${fitted.zoom}`, `${before.x},${before.y},${before.zoom}`);
      assert.equal(graph.classList.contains("is-focused"), true);
      graph.dispatchEvent(new PointerEvent("pointermove", {
        clientX: panClientX, clientY: panClientY, pointerId: 1, bubbles: true, composed: true,
      }));
      await flush();
      const continued = graph.getViewport();
      assert.equal(continued.x, fitted.x + panDeltaX);
      assert.equal(continued.y, fitted.y + panDeltaY);
      assert.equal(continued.zoom, fitted.zoom);
      await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
        eventType: "pointerup",
        buttons: buttonsReleased,
        origin: pointerOrigin,
        client: panClient,
        viewport: panClient,
        hit: { kind: "empty" },
      }) }));
      await flush();
      assert.match(graph.state(), /\/idle$/);
      assert.deepEqual(graph.getViewport(), continued);
      graph.remove();
    }
  });

  test("focus_machine during pan after a cursor move rebases getViewport from the live pointer", async () => {
    const originLeft = 0;
    const originTop = 0;
    const nodeWidth = 80;
    const nodeHeight = 40;
    const pointerOriginX = 20;
    const pointerOriginY = 20;
    const midClientX = 40;
    const midClientY = 48;
    const panClientX = 60;
    const panClientY = 70;
    const buttonsReleased = 0;
    const machineName = "/A";
    const nodeId = "a";
    const bounds = { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight };
    const pointerOrigin = { x: pointerOriginX, y: pointerOriginY };
    const panClient = { x: panClientX, y: panClientY };
    const postFitDeltaX = panClientX - midClientX;
    const postFitDeltaY = panClientY - midClientY;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    graph.nodes = [
      { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
    ];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: pointerOrigin,
      client: pointerOrigin,
      viewport: pointerOrigin,
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/pan$/);
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: midClientX, clientY: midClientY, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    graph.focusTarget({ kind: "machine", machineName, bounds });
    await flush();
    assert.match(graph.state(), /\/pan$/);
    const fitted = graph.getViewport();
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: midClientX, clientY: midClientY, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    assert.deepEqual(graph.getViewport(), fitted);
    graph.dispatchEvent(new PointerEvent("pointermove", {
      clientX: panClientX, clientY: panClientY, pointerId: 1, bubbles: true, composed: true,
    }));
    await flush();
    const continued = graph.getViewport();
    assert.equal(continued.x, fitted.x + postFitDeltaX);
    assert.equal(continued.y, fitted.y + postFitDeltaY);
    assert.equal(continued.zoom, fitted.zoom);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: buttonsReleased,
      origin: pointerOrigin,
      client: panClient,
      viewport: panClient,
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    assert.deepEqual(graph.getViewport(), continued);
    graph.remove();
  });

  test("fitView, fitBounds, and focusTarget write one getViewport for the same bounds", async () => {
    const originLeft = 0;
    const originTop = 0;
    const nodeWidth = 80;
    const nodeHeight = 40;
    const nodeId = "a";
    const machineName = "/A";
    const bounds = { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight };
    const expected = getViewportForBounds({
      bounds,
      origin: { x: 0, y: 0 },
      width: 1000,
      height: 600,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      padding: FIT_PADDING_RATIO,
    });
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [
      { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
    ];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    graph.fitView();
    await flush();
    const fitViewPort = graph.getViewport();
    graph.fitBounds(bounds);
    await flush();
    const fitBoundsPort = graph.getViewport();
    graph.focusTarget({ kind: "machine", machineName, bounds });
    await flush();
    const focusPort = graph.getViewport();
    assert.deepEqual(fitViewPort, expected);
    assert.deepEqual(fitBoundsPort, expected);
    assert.deepEqual(focusPort, expected);
    graph.remove();
  });

  test("fitBounds inverted and zero-span bounds share getViewportForBounds", async () => {
    const viewportWidth = 1000;
    const viewportHeight = 600;
    const cases: readonly { left: number; right: number; top: number; bottom: number }[] = [
      { left: 80, right: 0, top: 40, bottom: 0 },
      { left: 20, right: 20, top: 10, bottom: 10 },
    ];
    for (const bounds of cases) {
      const expected = getViewportForBounds({
        bounds,
        origin: { x: 0, y: 0 },
        width: viewportWidth,
        height: viewportHeight,
        minZoom: MIN_ZOOM,
        maxZoom: MAX_ZOOM,
        padding: FIT_PADDING_RATIO,
      });
      const graph = document.createElement("flow-graph");
      document.body.append(graph);
      graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
      await waitUntil(() => graph.querySelector("flow-node") !== null);
      graph.fitBounds(bounds);
      await flush();
      assert.deepEqual(graph.getViewport(), expected);
      graph.remove();
    }
  });

  test("fitView and fitBounds during pan rebase getViewport from the live pointer", async () => {
    const originLeft = 0;
    const originTop = 0;
    const nodeWidth = 80;
    const nodeHeight = 40;
    const pointerOriginX = 20;
    const pointerOriginY = 20;
    const midClientX = 40;
    const midClientY = 48;
    const panClientX = 60;
    const panClientY = 70;
    const buttonsReleased = 0;
    const nodeId = "a";
    const bounds = { left: originLeft, right: nodeWidth, top: originTop, bottom: nodeHeight };
    const pointerOrigin = { x: pointerOriginX, y: pointerOriginY };
    const panClient = { x: panClientX, y: panClientY };
    const postFitDeltaX = panClientX - midClientX;
    const postFitDeltaY = panClientY - midClientY;
    const expected = getViewportForBounds({
      bounds,
      origin: { x: 0, y: 0 },
      width: 1000,
      height: 600,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      padding: FIT_PADDING_RATIO,
    });
    const applies = [
      (graph: FlowGraph) => {
        graph.fitView();
      },
      (graph: FlowGraph) => {
        graph.fitBounds(bounds);
      },
    ];
    for (const apply of applies) {
      const graph = document.createElement("flow-graph");
      document.body.append(graph);
      graph.panOnDrag = true;
      graph.nodes = [
        { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
      ];
      await waitUntil(() => graph.querySelector("flow-node") !== null);
      await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
        eventType: "pointerdown",
        origin: pointerOrigin,
        client: pointerOrigin,
        viewport: pointerOrigin,
        hit: { kind: "empty" },
      }) }));
      assert.match(graph.state(), /\/pan$/);
      graph.dispatchEvent(new PointerEvent("pointermove", {
        clientX: midClientX, clientY: midClientY, pointerId: 1, bubbles: true, composed: true,
      }));
      await flush();
      apply(graph);
      await flush();
      const fitted = graph.getViewport();
      assert.deepEqual(fitted, expected);
      graph.dispatchEvent(new PointerEvent("pointermove", {
        clientX: midClientX, clientY: midClientY, pointerId: 1, bubbles: true, composed: true,
      }));
      await flush();
      assert.deepEqual(graph.getViewport(), fitted);
      graph.dispatchEvent(new PointerEvent("pointermove", {
        clientX: panClientX, clientY: panClientY, pointerId: 1, bubbles: true, composed: true,
      }));
      await flush();
      const continued = graph.getViewport();
      assert.equal(continued.x, fitted.x + postFitDeltaX);
      assert.equal(continued.y, fitted.y + postFitDeltaY);
      assert.equal(continued.zoom, fitted.zoom);
      await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
        eventType: "pointerup",
        buttons: buttonsReleased,
        origin: pointerOrigin,
        client: panClient,
        viewport: panClient,
        hit: { kind: "empty" },
      }) }));
      await flush();
      assert.match(graph.state(), /\/idle$/);
      assert.deepEqual(graph.getViewport(), continued);
      graph.remove();
    }
  });

  test("node_activate_click during pan stays in pan and pointer_up still ends pan", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.panOnDrag = true;
    const nodeId = "a";
    const originLeft = 0;
    const originTop = 0;
    const nodeWidth = 80;
    const nodeHeight = 40;
    const pointerOriginX = 20;
    const pointerOriginY = 20;
    const panClientX = 40;
    const panClientY = 40;
    const buttonsReleased = 0;
    const pointerOrigin = { x: pointerOriginX, y: pointerOriginY };
    const panClient = { x: panClientX, y: panClientY };
    graph.nodes = [
      { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
    ];
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: pointerOrigin,
      client: pointerOrigin,
      viewport: pointerOrigin,
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/pan$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateClickEvent, data: { nodeId } }));
    assert.match(graph.state(), /\/pan$/);
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: buttonsReleased,
      origin: pointerOrigin,
      client: panClient,
      viewport: panClient,
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    graph.remove();
  });

  test("public flow CustomEvents dispatch non-cancelable with bubbles and composed", async () => {
    const nodeWidth = 80;
    const nodeHeight = 40;
    const sourceNodeX = 0;
    const targetNodeX = 200;
    const nodeY = 0;
    const handleSourceX = 80;
    const handleY = 20;
    const buttonsReleased = 0;
    const oneNode = 1;
    const noEdges = 0;
    const atLeastOne = 1;
    const priorNodeId = "a";
    const connectPair = "a->b";

    // flow-admit-rejected: non-finite node coordinates reject as invalid.
    // Postcondition: the prior graph remains committed.
    const rejectedGraph = document.createElement("flow-graph");
    document.body.append(rejectedGraph);
    const rejected: string[] = [];
    rejectedGraph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail)) {
        rejected.push(String(event.detail["reason"] ?? ""));
        assertPublicCustomEvent(event);
      }
    });
    const validNode = { id: priorNodeId, position: { x: nodeY, y: nodeY }, data: { label: "A" }, width: nodeWidth, height: nodeHeight };
    rejectedGraph.nodes = [validNode];
    assert.equal(rejectedGraph.nodes.length, oneNode);
    const invalidNode = { id: "nan", position: { x: Number.NaN, y: nodeY }, data: {} };
    rejectedGraph.nodes = [invalidNode];
    assert.equal(rejectedGraph.nodes.length, oneNode);
    assert.equal(rejectedGraph.nodes[0]?.id, priorNodeId);
    assert.deepEqual(rejected, ["invalid"]);

    // flow-selection-change: box select covering a node commits that node.
    const selectGraph = document.createElement("flow-graph");
    document.body.append(selectGraph);
    const selected: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; nodeCount: number }> = [];
    selectGraph.addEventListener("flow-selection-change", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail) || !Array.isArray(event.detail["nodes"])) return;
      selected.push({ cancelable: event.cancelable, bubbles: event.bubbles, composed: event.composed, nodeCount: event.detail["nodes"].length });
    });
    selectGraph.nodes = [validNode];
    const boxOrigin = { x: nodeY, y: nodeY };
    const boxEnd = { x: nodeWidth * 2, y: nodeHeight * 2 };
    await selectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      shiftKey: true,
      origin: boxOrigin,
      client: boxOrigin,
      viewport: boxOrigin,
      hit: { kind: "empty" },
    }) }));
    await selectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      shiftKey: true,
      origin: boxOrigin,
      client: boxEnd,
      viewport: boxEnd,
      hit: { kind: "empty" },
    }) }));
    await selectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      shiftKey: true,
      buttons: buttonsReleased,
      origin: boxOrigin,
      client: boxEnd,
      viewport: boxEnd,
      hit: { kind: "empty" },
    }) }));
    await flush();
    assert.ok(selected.length >= atLeastOne);
    for (const seen of selected) {
      assert.equal(seen.cancelable, publicEventCancelable);
      assert.equal(seen.bubbles, publicEventBubbles);
      assert.equal(seen.composed, publicEventComposed);
    }
    assert.ok(selected.some((seen) => seen.nodeCount === oneNode));

    // flow-connect: pointer down on a source handle, up on a target handle.
    // Postcondition: the dispatcher does not add an edge.
    const connectGraph = document.createElement("flow-graph");
    document.body.append(connectGraph);
    connectGraph.nodes = [
      { id: "a", position: { x: sourceNodeX, y: nodeY }, data: { label: "A" }, width: nodeWidth, height: nodeHeight },
      { id: "b", position: { x: targetNodeX, y: nodeY }, data: { label: "B" }, width: nodeWidth, height: nodeHeight },
    ];
    const sourceNode = connectGraph.nodes[0];
    const targetNode = connectGraph.nodes[1];
    assert.ok(sourceNode !== undefined && targetNode !== undefined);
    const connected: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; pair: string }> = [];
    connectGraph.addEventListener("flow-connect", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail)) return;
      const sourceId = event.detail["source"];
      const targetId = event.detail["target"];
      if (typeof sourceId === "string" && typeof targetId === "string") {
        connected.push({ cancelable: event.cancelable, bubbles: event.bubbles, composed: event.composed, pair: `${sourceId}->${targetId}` });
      }
    });
    await connectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: handleSourceX, y: handleY },
      client: { x: handleSourceX, y: handleY },
      world: { x: handleSourceX, y: handleY },
      hit: { kind: "handle", node: sourceNode, handleKind: "source", position: "right" },
    }) }));
    assert.match(connectGraph.state(), /\/connect$/);
    const releaseClient = { x: targetNodeX, y: handleY };
    await connectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: handleSourceX, y: handleY },
      client: releaseClient,
      world: releaseClient,
      hit: { kind: "handle", node: targetNode, handleKind: "target", position: "left" },
    }) }));
    await connectGraph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: buttonsReleased,
      origin: { x: handleSourceX, y: handleY },
      client: releaseClient,
      world: releaseClient,
      hit: { kind: "handle", node: targetNode, handleKind: "target", position: "left" },
    }) }));
    await flush();
    assert.deepEqual(connected.map((item) => item.pair), [connectPair]);
    assert.equal(connectGraph.edges.length, noEdges);
    for (const item of connected) {
      assert.equal(item.cancelable, publicEventCancelable);
      assert.equal(item.bubbles, publicEventBubbles);
      assert.equal(item.composed, publicEventComposed);
    }

    rejectedGraph.remove();
    selectGraph.remove();
    connectGraph.remove();
  });

  test("node click, edge click, and viewport change CustomEvents are non-cancelable", async () => {
    const nodeWidth = 80;
    const nodeHeight = 40;
    const originLeft = 0;
    const originTop = 0;
    const targetNodeX = 200;
    const buttonsReleased = 0;
    const viewportX = 12;
    const viewportY = 8;
    const viewportZoom = 1.1;
    const atLeastOne = 1;
    const nodeId = "a";
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const sourceNode = { id: nodeId, position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight };
    const targetNode = { id: "b", position: { x: targetNodeX, y: originTop }, data: { label: "B" }, width: nodeWidth, height: nodeHeight };
    const edge = { id: "a-b", source: nodeId, target: "b", type: "bezier" as const };
    graph.nodes = [sourceNode, targetNode];
    graph.edges = [edge];
    const clicks: Event[] = [];
    const edgeClicks: Event[] = [];
    const viewports: Event[] = [];
    graph.addEventListener("flow-node-click", (event) => clicks.push(event));
    graph.addEventListener("flow-edge-click", (event) => edgeClicks.push(event));
    graph.addEventListener("flow-viewport-change", (event) => viewports.push(event));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.activateClickEvent, data: { nodeId } }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: { kind: "edge", edge },
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: buttonsReleased,
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: { kind: "edge", edge },
    }) }));
    graph.setViewport({ x: viewportX, y: viewportY, zoom: viewportZoom });
    await flush();
    assert.ok(clicks.length >= atLeastOne);
    assert.ok(edgeClicks.length >= atLeastOne);
    assert.ok(viewports.length >= atLeastOne);
    for (const event of [...clicks, ...edgeClicks, ...viewports]) {
      assertPublicCustomEvent(event);
    }
    const viewport = graph.getViewport();
    assert.equal(viewport.x, viewportX);
    assert.equal(viewport.y, viewportY);
    assert.equal(viewport.zoom, viewportZoom);
    graph.remove();
  });

  test("nodes write after stop emits host-drop stopped and drops the write", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const priorNodeId = "a";
    const rejectedNodeId = "b";
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    graph.nodes = [{ id: priorNodeId, position: { x: 0, y: 0 }, data: {} }];
    await waitUntil(() => graph.nodes[0]?.id === priorNodeId);
    const drops: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({
          cancelable: event.cancelable,
          bubbles: event.bubbles,
          composed: event.composed,
          reason: event.detail["reason"],
        });
      }
    });
    await graph.stop();
    graph.nodes = [{ id: rejectedNodeId, position: { x: 1, y: 1 }, data: {} }];
    await flush();
    assert.equal(graph.nodes[0]?.id, priorNodeId);
    assert.ok(drops.length >= atLeastOneDrop);
    for (const drop of drops) {
      assert.equal(drop.cancelable, publicEventCancelable);
      assert.equal(drop.bubbles, publicEventBubbles);
      assert.equal(drop.composed, publicEventComposed);
    }
    assert.ok(drops.some((drop) => drop.reason === stopped));
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => graph.nodes[0]?.id === priorNodeId);
    assert.equal(graph.nodes[0]?.id, priorNodeId);
    const droppedWriteAbsent = false;
    assert.equal(graph.nodes.some((node) => node.id === rejectedNodeId), droppedWriteAbsent);
    graph.remove();
  });

  test("edges write after stop emits host-drop stopped and drops the write", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const priorEdgeId = "a-b";
    const rejectedEdgeId = "dropped";
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: priorEdgeId, source: "a", target: "b" }];
    await waitUntil(() => graph.edges[0]?.id === priorEdgeId);
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    await graph.stop();
    graph.edges = [{ id: rejectedEdgeId, source: "a", target: "b" }];
    await flush();
    assert.equal(graph.edges[0]?.id, priorEdgeId);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => graph.edges[0]?.id === priorEdgeId);
    assert.equal(graph.edges[0]?.id, priorEdgeId);
    const droppedWriteAbsent = false;
    assert.equal(graph.edges.some((edge) => edge.id === rejectedEdgeId), droppedWriteAbsent);
    graph.remove();
  });

  test("nodes write during in-flight stop emits host-drop stopped and drops the write", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const priorNodeId = "a";
    const rejectedNodeId = "dropped";
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    const position = { x: 1, y: 1 };
    graph.nodes = [{ id: priorNodeId, position: { x: 0, y: 0 }, data: {} }];
    await waitUntil(() => graph.nodes[0]?.id === priorNodeId);
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    const stopping = graph.stop();
    graph.nodes = [{ id: rejectedNodeId, position, data: { label: "dropped" } }];
    position.x = 999;
    await flush();
    assert.equal(graph.nodes[0]?.id, priorNodeId);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    await stopping;
    assert.equal(graph.nodes[0]?.id, priorNodeId);
    const droppedWriteAbsent = false;
    assert.equal(graph.nodes.some((node) => node.id === rejectedNodeId), droppedWriteAbsent);
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => graph.nodes[0]?.id === priorNodeId);
    assert.equal(graph.nodes[0]?.id, priorNodeId);
    assert.equal(graph.nodes.some((node) => node.id === rejectedNodeId), droppedWriteAbsent);
    graph.remove();
  });

  test("edges write during in-flight stop emits host-drop stopped and drops the write", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const priorEdgeId = "a-b";
    const rejectedEdgeId = "dropped";
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: priorEdgeId, source: "a", target: "b" }];
    await waitUntil(() => graph.edges[0]?.id === priorEdgeId);
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    const stopping = graph.stop();
    graph.edges = [{ id: rejectedEdgeId, source: "a", target: "b" }];
    await flush();
    assert.equal(graph.edges[0]?.id, priorEdgeId);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    await stopping;
    assert.equal(graph.edges[0]?.id, priorEdgeId);
    const droppedWriteAbsent = false;
    assert.equal(graph.edges.some((edge) => edge.id === rejectedEdgeId), droppedWriteAbsent);
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => graph.edges[0]?.id === priorEdgeId);
    assert.equal(graph.edges[0]?.id, priorEdgeId);
    assert.equal(graph.edges.some((edge) => edge.id === rejectedEdgeId), droppedWriteAbsent);
    graph.remove();
  });

  test("nodesDraggable and panOnDrag writes after stop emit host-drop stopped and drop the write", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const policyOn = true;
    const policyOff = false;
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    graph.nodesDraggable = policyOn;
    graph.nodesResizable = policyOn;
    graph.panOnDrag = policyOn;
    await waitUntil(() => /\/connected\//.test(graph.state()));
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    await graph.stop();
    graph.nodesDraggable = policyOff;
    graph.nodesResizable = policyOff;
    graph.panOnDrag = policyOff;
    await flush();
    assert.equal(graph.nodesDraggable, policyOn);
    assert.equal(graph.nodesResizable, policyOn);
    assert.equal(graph.panOnDrag, policyOn);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    assert.equal(graph.nodesDraggable, policyOn);
    assert.equal(graph.nodesResizable, policyOn);
    assert.equal(graph.panOnDrag, policyOn);
    graph.remove();
  });

  test("repairing nested fields after an invalid nodes write does not admit the write", async () => {
    const graph = document.createElement("flow-graph");
    const repairedId = "repaired";
    const nanWidthId = "nan-width";
    const noNodes = 0;
    const position = { x: Number.NaN, y: 0 };
    const data = { label: "bad" };
    graph.nodes = [{ id: repairedId, position, data }];
    position.x = 0;
    data.label = "mutated";
    document.body.append(graph);
    await flush();
    assert.equal(graph.nodes.length, noNodes);
    const droppedWriteAbsent = false;
    assert.equal(graph.nodes.some((node) => node.id === repairedId), droppedWriteAbsent);
    graph.remove();
    const nanWidthPosition = { x: 4, y: 5 };
    const nanWidthData = { label: "nan-width" };
    const again = document.createElement("flow-graph");
    again.nodes = [{ id: nanWidthId, position: nanWidthPosition, data: nanWidthData, width: Number.NaN }];
    nanWidthPosition.x = 999;
    nanWidthData.label = "mutated-width";
    document.body.append(again);
    await flush();
    assert.equal(again.nodes.length, noNodes);
    assert.equal(again.nodes.some((node) => node.id === nanWidthId), droppedWriteAbsent);
    again.remove();
  });

  test("mutating FlowGraph.nodes and edges getter nested fields does not change host state", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originX = 0;
    const originalLabel = "A";
    const originalEdgeLabel = "e";
    const mutatedX = 999;
    const mutatedLabel = "mutated";
    const admittedNodes = 2;
    const admittedEdges = 1;
    graph.nodes = [
      { id: "a", position: { x: originX, y: 0 }, data: { label: originalLabel }, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b", data: { label: originalEdgeLabel } }];
    await waitUntil(() => graph.nodes.length === admittedNodes && graph.edges.length === admittedEdges);
    const first = graph.nodes[0];
    assert.ok(first !== undefined);
    const nodePosition = first.position as { x: number };
    nodePosition.x = mutatedX;
    first.data["label"] = mutatedLabel;
    const edge = graph.edges[0];
    assert.ok(edge !== undefined);
    assert.ok(edge.data !== undefined);
    edge.data["label"] = mutatedLabel;
    assert.equal(graph.nodes[0]?.position.x, originX);
    assert.equal(graph.nodes[0]?.data["label"], originalLabel);
    assert.equal(graph.edges[0]?.data?.["label"], originalEdgeLabel);
    graph.remove();
  });

  test("flow-node cyclic and over-deep data does not throw and does not paint", () => {
    const element = document.createElement("flow-node");
    const originX = 0;
    const originalLabel = "A";
    const valid: Node = {
      id: "a",
      position: { x: originX, y: 0 },
      data: { label: originalLabel },
      width: 80,
      height: 40,
    };
    element.node = valid;
    assert.equal(element.node?.id, valid.id);
    assert.equal(element.style.left, `${originX}px`);
    const cyclicData: Record<string, unknown> = { label: "cycle", className: "boom" };
    cyclicData["self"] = cyclicData;
    element.node = { id: "cycle", position: { x: 9, y: 9 }, data: cyclicData };
    assert.equal(element.node?.id, valid.id);
    assert.equal(element.node?.data["label"], originalLabel);
    assert.equal(element.style.left, `${originX}px`);
    let nested: Record<string, unknown> = { label: "deep" };
    const overDepth = MAX_JSON_DEPTH + 1;
    for (let depth = 0; depth < overDepth; depth += 1) {
      nested = { child: nested };
    }
    element.node = { id: "deep", position: { x: 9, y: 9 }, data: nested };
    assert.equal(element.node?.id, valid.id);
    assert.equal(element.style.left, `${originX}px`);
    const unset = document.createElement("flow-node");
    const rejectedLeft = "0px";
    unset.node = { id: "cycle", position: { x: 0, y: 0 }, data: cyclicData };
    assert.equal(unset.node, null);
    assert.notEqual(unset.style.left, rejectedLeft);
  });

  test("flow-node non-finite position does not throw and does not paint", () => {
    const element = document.createElement("flow-node");
    const originX = 0;
    const originalLabel = "A";
    const invalidCoordinate = Number.NaN;
    const infiniteCoordinate = Number.POSITIVE_INFINITY;
    const valid: Node = {
      id: "a",
      position: { x: originX, y: 0 },
      data: { label: originalLabel },
      width: 80,
      height: 40,
    };
    element.node = valid;
    assert.equal(element.style.left, `${originX}px`);
    element.node = { id: "nan", position: { x: invalidCoordinate, y: 0 }, data: { label: "nan" } };
    assert.equal(element.node?.id, valid.id);
    assert.equal(element.style.left, `${originX}px`);
    element.node = { id: "inf", position: { x: 0, y: infiniteCoordinate }, data: { label: "inf" } };
    assert.equal(element.node?.id, valid.id);
    assert.equal(element.style.left, `${originX}px`);
    const unset = document.createElement("flow-node");
    const rejectedLeft = "0px";
    unset.node = { id: "nan", position: { x: invalidCoordinate, y: 0 }, data: {} };
    assert.equal(unset.node, null);
    assert.notEqual(unset.style.left, rejectedLeft);
  });

  test("flow-edge cyclic data does not throw and leaves stored paint unchanged", () => {
    const element = document.createElement("flow-edge");
    const originalLabel = "e";
    const valid = { id: "a-b", source: "a", target: "b", data: { label: originalLabel } };
    element.edge = valid;
    assert.equal(element.edge?.id, valid.id);
    const cyclicData: Record<string, unknown> = { label: "cycle" };
    cyclicData["self"] = cyclicData;
    element.edge = { id: "cycle", source: "a", target: "b", data: cyclicData };
    assert.equal(element.edge?.id, valid.id);
    assert.equal(element.edge?.data?.["label"], originalLabel);
    const unset = document.createElement("flow-edge");
    unset.edge = { id: "cycle", source: "a", target: "b", data: cyclicData };
    assert.equal(unset.edge, null);
  });

  test("flow-node paint does not alias graph node nested fields", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originX = 0;
    const mutatedX = 999;
    const originalLabel = "A";
    const originLeft = `${originX}px`;
    graph.nodes = [{ id: "a", position: { x: originX, y: 0 }, data: { label: originalLabel }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const element = graph.querySelector("flow-node");
    assert.ok(element instanceof FlowNode);
    const painted = element.node as { position: { x: number }; data: Record<string, unknown> } | null;
    assert.ok(painted !== null);
    painted.position.x = mutatedX;
    painted.data["label"] = "mutated";
    assert.equal(graph.nodes[0]?.position.x, originX);
    assert.equal(graph.nodes[0]?.data["label"], originalLabel);
    assert.equal(element.node?.position.x, originX);
    assert.equal(element.node?.data["label"], originalLabel);
    assert.equal(element.style.left, originLeft);
    const badge = element.shadowRoot?.querySelector('[part="badge"]');
    assert.equal(badge?.textContent, originalLabel);
    graph.remove();
  });

  test("flow-edge getter copies nested data", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originalLabel = "e";
    const mutatedLabel = "mutated";
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "b", position: { x: 80, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b", data: { label: originalLabel } }];
    await waitUntil(() => graph.querySelector("flow-edge") !== null);
    const element = graph.querySelector("flow-edge");
    assert.ok(element instanceof FlowEdge);
    const painted = element.edge;
    assert.ok(painted !== null);
    assert.ok(painted.data !== undefined);
    painted.data["label"] = mutatedLabel;
    assert.equal(graph.edges[0]?.data?.["label"], originalLabel);
    assert.equal(element.edge?.data?.["label"], originalLabel);
    graph.remove();
  });

  test("cable edges route around blockers, fall back to smoothstep during drags, and resume after drag_end", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    // Source and target face each other across a wall no straight corridor
    // crosses; the routed polyline must leave the wall's band to reach around.
    // Short enough that wrapping stays clearly cheaper than any punch-through
    // under the shadow-pad standoff (the reference routes with vPad>=8 on).
    const wall = { x: 200, y: -140, width: 60, height: 380 };
    const strictlyInsideWallPlus16 = (p: { x: number; y: number }): boolean =>
      p.x > wall.x - 16 && p.x < wall.x + wall.width + 16
      && p.y > wall.y - 16 && p.y < wall.y + wall.height + 16;
    graph.nodes = [
      { id: "src", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 },
      { id: "wall", position: { x: wall.x, y: wall.y }, data: {}, width: wall.width, height: wall.height },
      { id: "tgt", position: { x: 400, y: 0 }, data: {}, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "wire", source: "src", target: "tgt", type: "cable", label: "W" }];
    await waitUntil(() => graph.querySelectorAll("flow-node").length === 3);
    const edgeEl = graph.querySelector("flow-edge");
    assert.ok(edgeEl instanceof FlowEdge);

    const routed = await waitForCablePolyline(graph, { x: 80, y: 20 });
    assert.ok(routed.length >= 4, `expected a multipoint route, got ${JSON.stringify(routed)}`);
    assert.deepEqual(routed[0], { x: 80, y: 20 }); // source right-border anchor
    for (const p of routed) {
      assert.ok(!strictlyInsideWallPlus16(p), `waypoint ${JSON.stringify(p)} entered blocker+16`);
    }
    assert.ok(
      routed.some((p) => p.y < wall.y - 16 || p.y > wall.y + wall.height + 16),
      `route never left the wall band: ${JSON.stringify(routed)}`,
    );
    // The label sits at the longest-segment midpoint of the routed polyline,
    // and the painted path carries the cable class token.
    const expectedLabel = labelPoint(routed);
    assert.equal(edgeEl.label.getAttribute("x"), String(expectedLabel.x));
    assert.equal(edgeEl.label.getAttribute("y"), String(expectedLabel.y));
    const routedClass = edgeEl.path.getAttribute("class") ?? "";
    assert.match(routedClass, /(^| )cable( |$)/);

    // Drag the target node: while it drags its edge drops out of the routed
    // map and paint falls back to the smoothstep spline.
    const tgtEl = Array.from(graph.querySelectorAll("flow-node"))
      .find((el) => el instanceof FlowNode && el.node?.id === "tgt");
    assert.ok(tgtEl instanceof FlowNode);
    tgtEl.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 420, clientY: 20 })));
    assert.match(graph.state(), /\/click$/);
    tgtEl.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 460, clientY: 20 })));
    assert.match(graph.state(), /\/drag$/);
    // The first sample while dragging establishes the dedupe baseline...
    tgtEl.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 500, clientY: 30 })));
    await flush();
    assert.equal(graph.nodes.find((node) => node.id === "tgt")?.position.x, 400);
    // ...the next past-epsilon sample moves the node. The grab offset came
    // from the entering-drag sample: (460,20) - (400,0) = (60,20), so the
    // node lands at (540-60, 40-20) = (480, 20).
    tgtEl.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 540, clientY: 40 })));
    await waitUntil(() => graph.nodes.find((node) => node.id === "tgt")?.position.x === 480);
    assert.equal(graph.nodes.find((node) => node.id === "tgt")?.position.y, 20);
    // Node at (480, 20); the fallback spline runs bottom-center of src to
    // top-center of the moved target.
    const fallbackD = getSmoothStepPath({ sourceX: 40, sourceY: 40, targetX: 520, targetY: 20 })[0];
    await waitUntil(() => edgePathDOf(graph) === fallbackD);
    assert.equal(edgePathDOf(graph), fallbackD);

    // drag_end clears draggingNodeIds; the post-drag sync restores a routed
    // polyline for the moved geometry.
    tgtEl.dispatchEvent(new PointerEvent("pointerup", pointerInit({ clientX: 540, clientY: 40 })));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    const resumed = await waitForCablePolyline(graph, { x: 80, y: 20 });
    assert.deepEqual(resumed[0], { x: 80, y: 20 });
    assert.deepEqual(resumed[resumed.length - 1], { x: 480, y: 40 }); // moved target left-border anchor
    for (const p of resumed) {
      assert.ok(!strictlyInsideWallPlus16(p), `resumed waypoint ${JSON.stringify(p)} entered blocker+16`);
    }
    graph.remove();
  });

  test("mutating caller nested node fields after a valid set does not change admitted nodes", async () => {
    const graph = document.createElement("flow-graph");
    const originX = 0;
    const originalLabel = "A";
    const mutatedX = 999;
    const position = { x: originX, y: 0 };
    const data = { label: originalLabel };
    const admitted = 1;
    graph.nodes = [{ id: "a", position, data, width: 80, height: 40 }];
    position.x = mutatedX;
    data.label = "mutated";
    document.body.append(graph);
    await waitUntil(() => graph.nodes.length === admitted);
    assert.equal(graph.nodes[0]?.position.x, originX);
    assert.equal(graph.nodes[0]?.data["label"], originalLabel);
    graph.remove();
  });

  test("omitted or non-record node data is not admitted as empty data", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const noNodes = 0;
    const droppedWriteAbsent = false;
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    const omittedId = "omitted";
    const nullId = "null-data";
    const arrayId = "array-data";
    graph.nodes = [{ id: omittedId, position: { x: 0, y: 0 } } as Node];
    graph.nodes = [{ id: nullId, position: { x: 0, y: 0 }, data: null } as unknown as Node];
    graph.nodes = [{ id: arrayId, position: { x: 0, y: 0 }, data: [] } as unknown as Node];
    await flush();
    assert.equal(graph.nodes.length, noNodes);
    assert.equal(graph.nodes.some((node) => node.id === omittedId), droppedWriteAbsent);
    assert.equal(graph.nodes.some((node) => node.id === nullId), droppedWriteAbsent);
    assert.equal(graph.nodes.some((node) => node.id === arrayId), droppedWriteAbsent);
    assert.deepEqual(rejected, ["invalid", "invalid", "invalid"]);
    graph.remove();
  });

  test("cyclic nested node data does not throw and is not admitted", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const noNodes = 0;
    const droppedWriteAbsent = false;
    const cyclicId = "cycle";
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    const data: Record<string, unknown> = { label: "cycle" };
    data["self"] = data;
    graph.nodes = [{ id: cyclicId, position: { x: 0, y: 0 }, data }];
    await flush();
    assert.equal(graph.nodes.length, noNodes);
    assert.equal(graph.nodes.some((node) => node.id === cyclicId), droppedWriteAbsent);
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("over-deep nested node data does not throw and is not admitted", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const noNodes = 0;
    const droppedWriteAbsent = false;
    const deepId = "deep";
    const rejected: string[] = [];
    graph.addEventListener("flow-admit-rejected", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        rejected.push(event.detail["reason"]);
      }
    });
    let nested: Record<string, unknown> = { label: "deep" };
    const overDepth = MAX_JSON_DEPTH + 1;
    for (let depth = 0; depth < overDepth; depth += 1) {
      nested = { child: nested };
    }
    graph.nodes = [{ id: deepId, position: { x: 0, y: 0 }, data: nested }];
    await flush();
    assert.equal(graph.nodes.length, noNodes);
    assert.equal(graph.nodes.some((node) => node.id === deepId), droppedWriteAbsent);
    assert.deepEqual(rejected, ["invalid"]);
    graph.remove();
  });

  test("fitView and zoomIn during in-flight stop emit host-drop stopped and do not change viewport", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const stopped = "stopped";
    const atLeastOneDrop = 1;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: {}, width: 80, height: 40 }];
    await waitUntil(() => /\/connected\//.test(graph.state()));
    const prior = graph.getViewport();
    const drops: Array<{ reason: string }> = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push({ reason: event.detail["reason"] });
      }
    });
    const stopping = graph.stop();
    graph.fitView();
    graph.zoomIn();
    graph.zoomOut();
    graph.setViewport({ x: 9, y: 9, zoom: 2 });
    graph.focusTarget({
      kind: "viewport",
      bounds: { left: 0, right: 1, top: 0, bottom: 1 },
    });
    await flush();
    assert.deepEqual(graph.getViewport(), prior);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    await stopping;
    assert.deepEqual(graph.getViewport(), prior);
    graph.remove();
  });

  test("Host.stop stops nested graph actors", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    const actors = ownedActors(graph);
    const childCount = 8;
    assert.equal(actors.length, childCount);
    for (const actor of actors) {
      assert.notEqual(actor.state(), "");
    }
    // Detach must carry every actor -- including Routes -- into stopActors via
    // #childActors(); a missing entry leaks one live router per attach/detach
    // cycle even though Host.stop's owned-children teardown masks it here.
    const routes = actors.find((actor) => actor.state().startsWith("/Routes"));
    assert.ok(routes !== undefined, "Routes missing from graph-owned actors");
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    assert.equal(routes.state(), "", "detach must stop the Routes actor with the others");
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    await graph.stop();
    assert.equal(graph.state(), "");
    for (const actor of ownedActors(graph)) {
      assert.equal(actor.state(), "");
    }
    graph.remove();
  });

  test("handle hit with non-string id does not enter connect", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const source = { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 };
    graph.nodes = [source];
    const numericHandleId = 7;
    const invalidHit = {
      kind: "handle",
      node: source,
      handleKind: "source",
      position: "right",
      id: numericHandleId,
    };
    await graph.dispatch(hsm.typedEvent({
      event: FlowGraph.pointerDownEvent,
      data: { ...pointerData(), hit: invalidHit },
    }));
    assert.doesNotMatch(graph.state(), /\/connect$/);
    graph.remove();
  });

  test("meta or ctrl node click is additive and copies the clicked node", async () => {
    const nodeWidth = 80;
    const nodeHeight = 40;
    const originLeft = 0;
    const originTop = 0;
    const secondLeft = 120;
    const buttonsReleased = 0;
    const nodeA = { id: "a", position: { x: originLeft, y: originTop }, data: { label: "A" }, width: nodeWidth, height: nodeHeight };
    const nodeB = { id: "b", position: { x: secondLeft, y: originTop }, data: { label: "B" }, width: nodeWidth, height: nodeHeight };
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [nodeA, nodeB];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const selectedIds: string[][] = [];
    const clicks: CustomEvent[] = [];
    graph.addEventListener("flow-selection-change", (event: Event) => {
      if (!(event instanceof CustomEvent) || !hsm.isRecord(event.detail) || !Array.isArray(event.detail["nodes"])) return;
      selectedIds.push(event.detail["nodes"].map((node: { id?: string }) => String(node.id ?? "")));
    });
    graph.addEventListener("flow-node-click", (event: Event) => {
      if (event instanceof CustomEvent) clicks.push(event);
    });
    const hitA = { kind: "node" as const, node: nodeA };
    const hitB = { kind: "node" as const, node: nodeB };
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: hitA,
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: buttonsReleased,
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: hitA,
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      metaKey: true,
      origin: { x: secondLeft, y: originTop },
      client: { x: secondLeft, y: originTop },
      hit: hitB,
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      metaKey: true,
      buttons: buttonsReleased,
      origin: { x: secondLeft, y: originTop },
      client: { x: secondLeft, y: originTop },
      hit: hitB,
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      ctrlKey: true,
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: hitA,
    }) }));
    await graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      ctrlKey: true,
      buttons: buttonsReleased,
      origin: { x: originLeft, y: originTop },
      client: { x: originLeft, y: originTop },
      hit: hitA,
    }) }));
    await flush();
    const twoNodes = 2;
    const oneNode = 1;
    const atLeastTwoClicks = 2;
    assert.ok(clicks.length >= atLeastTwoClicks);
    assert.ok(selectedIds.some((ids) => ids.length === oneNode && ids[0] === nodeA.id));
    assert.ok(selectedIds.some((ids) => ids.length === twoNodes && ids.includes(nodeA.id) && ids.includes(nodeB.id)));
    assert.ok(selectedIds.some((ids) => ids.length === oneNode && ids[0] === nodeB.id));
    const click = clicks[clicks.length - 1];
    assert.ok(click instanceof CustomEvent);
    const detail = click.detail as { node: { id: string; position: { x: number } } };
    const clickedId = detail.node.id;
    const clicked = graph.nodes.find((node) => node.id === clickedId);
    const priorX = clicked?.position.x;
    const mutatedX = 999;
    detail.node.position.x = mutatedX;
    const after = graph.nodes.find((node) => node.id === clickedId);
    assert.equal(after?.position.x, priorX);
    assert.notEqual(after?.position.x, mutatedX);
    graph.remove();
  });

  test("nodesResizable defaults true", () => {
    const graph = document.createElement("flow-graph");
    assert.equal(graph.nodesResizable, true);
  });

  test("selected node exposes eight resize controls in the resizer shadow", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    const directions = RESIZE_DIRECTIONS.map((direction) => {
      const control = resizer.shadowRoot?.querySelector(`flow-node-resize-control[direction="${direction}"]`);
      assert.ok(control instanceof HTMLElement, direction);
      return direction;
    });
    assert.equal(directions.length, RESIZE_DIRECTIONS.length);
    graph.remove();
  });

  test("se control pointer_down enters resize, not drag or pan", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    const clientX = 80;
    const clientY = 40;
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX, clientY })));
    assert.match(graph.state(), /\/resize\/pointer$/);
    assert.doesNotMatch(graph.state(), /\/drag$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    graph.remove();
  });

  test("se sample grows width and height and leaves position unchanged", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originX = 0;
    const originY = 0;
    const originWidth = 80;
    const originHeight = 40;
    graph.nodes = [{ id: "a", position: { x: originX, y: originY }, data: { label: "A" }, width: originWidth, height: originHeight }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    assert.match(graph.state(), /\/resize\/pointer$/);
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 100, clientY: 50 })));
    await flush();
    assert.equal(graph.nodes[0]?.width, originWidth);
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 120, clientY: 70 })));
    await waitUntil(() => graph.nodes[0]?.width === 120 && graph.nodes[0]?.height === 70);
    assert.equal(graph.nodes[0]?.position.x, originX);
    assert.equal(graph.nodes[0]?.position.y, originY);
    assert.equal(graph.nodes[0]?.width, 120);
    assert.equal(graph.nodes[0]?.height, 70);
    graph.remove();
  });

  test("nw sample keeps the se corner fixed", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originX = 40;
    const originY = 20;
    const originWidth = 80;
    const originHeight = 40;
    graph.nodes = [{ id: "a", position: { x: originX, y: originY }, data: { label: "A" }, width: originWidth, height: originHeight }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const nw = controlOf({ resizer: offeredResizer(nodeEl), direction: "nw" });
    nw.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 40, clientY: 20 })));
    nw.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 30, clientY: 10 })));
    await flush();
    nw.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 20, clientY: 0 })));
    await waitUntil(() => graph.nodes[0]?.width === 100 && graph.nodes[0]?.height === 60);
    const node = graph.nodes[0];
    assert.ok(node !== undefined);
    assert.equal(node.position.x, 20);
    assert.equal(node.position.y, 0);
    assert.equal(node.position.x + (node.width ?? 0), originX + originWidth);
    assert.equal(node.position.y + (node.height ?? 0), originY + originHeight);
    graph.remove();
  });

  test("min-width and min-height clamp a shrinking se resize", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    resizer.setAttribute("min-width", "50");
    resizer.setAttribute("min-height", "30");
    const se = controlOf({ resizer, direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 70, clientY: 30 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 0, clientY: 0 })));
    await waitUntil(() => graph.nodes[0]?.width === 50 && graph.nodes[0]?.height === 30);
    assert.equal(graph.nodes[0]?.width, 50);
    assert.equal(graph.nodes[0]?.height, 30);
    graph.remove();
  });

  test("max-width and max-height clamp a growing se resize", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    resizer.setAttribute("max-width", "100");
    resizer.setAttribute("max-height", "55");
    const se = controlOf({ resizer, direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 90, clientY: 50 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 200, clientY: 200 })));
    await waitUntil(() => graph.nodes[0]?.width === 100 && graph.nodes[0]?.height === 55);
    assert.equal(graph.nodes[0]?.width, 100);
    assert.equal(graph.nodes[0]?.height, 55);
    graph.remove();
  });

  test("keep-aspect-ratio preserves the origin ratio on se resize", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const originWidth = 80;
    const originHeight = 40;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: originHeight }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    resizer.setAttribute("keep-aspect-ratio", "");
    const se = controlOf({ resizer, direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 90, clientY: 40 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 120, clientY: 40 })));
    await waitUntil(() => graph.nodes[0]?.width === 120);
    assert.equal(graph.nodes[0]?.width, 120);
    assert.equal(graph.nodes[0]?.height, 60);
    graph.remove();
  });

  test("nodesResizable=false hides chrome and does not enter resize", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodesResizable = false;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = nodeEl.shadowRoot?.querySelector("flow-node-resizer");
    assert.ok(resizer instanceof FlowNodeResizer);
    assert.equal(resizer.hidden, true);
    const se = resizer.shadowRoot?.querySelector('flow-node-resize-control[direction="se"]');
    assert.ok(se instanceof HTMLElement);
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    assert.doesNotMatch(graph.state(), /\/resize/);
    graph.remove();
  });

  test("public resize CustomEvents are non-cancelable with bubbles and composed", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const seen: Array<{ name: string; cancelable: boolean; bubbles: boolean; composed: boolean }> = [];
    for (const name of ["flow-node-resize-start", "flow-node-resize", "flow-node-resize-end"]) {
      graph.addEventListener(name, (event: Event) => {
        seen.push({
          name,
          cancelable: event.cancelable,
          bubbles: event.bubbles,
          composed: event.composed,
        });
        assertPublicCustomEvent(event);
      });
    }
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 100, clientY: 50 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 110, clientY: 55 })));
    await waitUntil(() => graph.nodes[0]?.width === 110);
    se.dispatchEvent(new PointerEvent("pointerup", pointerInit({ clientX: 110, clientY: 55 })));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    assert.ok(seen.some((event) => event.name === "flow-node-resize-start"));
    assert.ok(seen.some((event) => event.name === "flow-node-resize"));
    assert.ok(seen.some((event) => event.name === "flow-node-resize-end"));
    for (const event of seen) {
      assert.equal(event.cancelable, publicEventCancelable);
      assert.equal(event.bubbles, publicEventBubbles);
      assert.equal(event.composed, publicEventComposed);
    }
    graph.remove();
  });

  test("each direction keeps the opposite edge fixed", async () => {
    const origin = { x: 40, y: 20, width: 80, height: 40 };
    const cases = [
      { direction: "n", clientX: 80, clientY: 20, moveX: 80, moveY: 0, east: 120, south: 60 },
      { direction: "s", clientX: 80, clientY: 60, moveX: 80, moveY: 80, east: 120, south: 80 },
      { direction: "e", clientX: 120, clientY: 40, moveX: 140, moveY: 40, east: 140, south: 60 },
      { direction: "w", clientX: 40, clientY: 40, moveX: 20, moveY: 40, east: 120, south: 60 },
      { direction: "ne", clientX: 120, clientY: 20, moveX: 140, moveY: 0, east: 140, south: 60 },
      { direction: "sw", clientX: 40, clientY: 60, moveX: 20, moveY: 80, east: 120, south: 80 },
    ] as const;
    for (const item of cases) {
      const graph = document.createElement("flow-graph");
      document.body.append(graph);
      graph.nodes = [{ id: "a", position: { x: origin.x, y: origin.y }, data: { label: "A" }, width: origin.width, height: origin.height }];
      await waitUntil(() => graph.querySelector("flow-node") !== null);
      const nodeEl = graph.querySelector("flow-node");
      assert.ok(nodeEl instanceof FlowNode);
      await selectFirstNode(nodeEl);
      const control = controlOf({ resizer: offeredResizer(nodeEl), direction: item.direction });
      control.dispatchEvent(new PointerEvent("pointerdown", pointerInit({
        clientX: item.clientX,
        clientY: item.clientY,
      })));
      control.dispatchEvent(new PointerEvent("pointermove", pointerInit({
        clientX: (item.clientX + item.moveX) / 2,
        clientY: (item.clientY + item.moveY) / 2,
      })));
      await flush();
      control.dispatchEvent(new PointerEvent("pointermove", pointerInit({
        clientX: item.moveX,
        clientY: item.moveY,
      })));
      await waitUntil(() => {
        const node = graph.nodes[0];
        if (node === undefined) return false;
        return node.position.x + (node.width ?? 0) === item.east
          && node.position.y + (node.height ?? 0) === item.south;
      });
      const node = graph.nodes[0];
      assert.ok(node !== undefined);
      assert.equal(node.position.x + (node.width ?? 0), item.east, item.direction);
      assert.equal(node.position.y + (node.height ?? 0), item.south, item.direction);
      graph.remove();
    }
  });

  test("resize detail is a frozen copy and end bounds match graph.nodes", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const details: Array<{ name: string; width: number; nodeWidth: number | undefined }> = [];
    for (const name of ["flow-node-resize-start", "flow-node-resize", "flow-node-resize-end"]) {
      graph.addEventListener(name, (event: Event) => {
        assert.ok(event instanceof CustomEvent);
        const detail = event.detail as { node: { width?: number; position: { x: number } }; width: number };
        details.push({ name, width: detail.width, nodeWidth: graph.nodes[0]?.width });
        assert.throws(() => {
          detail.width = 1;
        });
        assert.throws(() => {
          detail.node.position.x = 999;
        });
      });
    }
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 100, clientY: 50 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 110, clientY: 55 })));
    await waitUntil(() => graph.nodes[0]?.width === 110);
    se.dispatchEvent(new PointerEvent("pointerup", pointerInit({ clientX: 110, clientY: 55 })));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    const end = details.find((item) => item.name === "flow-node-resize-end");
    assert.ok(end !== undefined);
    assert.equal(end.width, graph.nodes[0]?.width);
    assert.equal(end.nodeWidth, graph.nodes[0]?.width);
    graph.remove();
  });

  test("unstarted and stopped hosts do not emit resize events", async () => {
    const graph = document.createElement("flow-graph");
    const seen: string[] = [];
    graph.addEventListener("flow-node-resize-start", () => seen.push("start"));
    assert.equal(seen.length, 0);
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    await graph.stop();
    const resizer = nodeEl.shadowRoot?.querySelector("flow-node-resizer");
    const se = resizer?.shadowRoot?.querySelector('flow-node-resize-control[direction="se"]');
    if (se instanceof HTMLElement) {
      se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    }
    await flush();
    assert.equal(seen.length, 0);
    graph.remove();
  });

  test("stop in flight drops further resize apply and emit", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const seen: string[] = [];
    graph.addEventListener("flow-node-resize", () => seen.push("move"));
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    const stopping = graph.stop();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 100, clientY: 50 })));
    await flush();
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 160, clientY: 90 })));
    await stopping;
    assert.equal(graph.nodes[0]?.width, 80);
    assert.equal(seen.length, 0);
    graph.remove();
  });

  test("keyboard Enter starts an se resize, arrow keys step 1px, Escape ends and applies", async () => {
    const originWidth = 80;
    const originHeight = 40;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: originHeight }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const seen: Array<{ name: string; width: number; height: number }> = [];
    for (const name of ["flow-node-resize-start", "flow-node-resize", "flow-node-resize-end"] as const) {
      graph.addEventListener(name, (event: Event) => {
        assert.ok(event instanceof CustomEvent);
        const detail = event.detail as { width: number; height: number };
        seen.push({ name, width: detail.width, height: detail.height });
      });
    }
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    const button = controlButtonOf(se);
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    const start = seen.find((item) => item.name === "flow-node-resize-start");
    assert.ok(start !== undefined);
    assert.equal(start.width, originWidth);
    assert.equal(start.height, originHeight);
    keyDownOn({ target: button, key: "ArrowRight" });
    keyDownOn({ target: button, key: "ArrowRight" });
    keyDownOn({ target: button, key: "ArrowDown" });
    await waitUntil(() => graph.nodes[0]?.width === originWidth + 2 && graph.nodes[0]?.height === originHeight + 1);
    assert.equal(graph.nodes[0]?.width, originWidth + 2);
    assert.equal(graph.nodes[0]?.height, originHeight + 1);
    assert.ok(seen.some((item) => item.name === "flow-node-resize"));
    keyDownOn({ target: button, key: "Escape" });
    await waitUntil(() => seen.some((item) => item.name === "flow-node-resize-end"));
    const end = seen.filter((item) => item.name === "flow-node-resize-end");
    assert.equal(end.length, 1);
    const ended = end[0];
    assert.ok(ended !== undefined);
    assert.equal(ended.width, originWidth + 2);
    assert.equal(ended.height, originHeight + 1);
    assert.equal(graph.nodes[0]?.width, originWidth + 2);
    assert.equal(graph.nodes[0]?.height, originHeight + 1);
    assert.equal(graph.nodes[0]?.position.x, 0);
    graph.remove();
  });

  test("second Enter on a resize control ends the keyboard resize", async () => {
    const originWidth = 80;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const ends: number[] = [];
    graph.addEventListener("flow-node-resize-end", (event: Event) => {
      assert.ok(event instanceof CustomEvent);
      ends.push((event.detail as { width: number }).width);
    });
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    const button = controlButtonOf(se);
    keyDownOn({ target: button, key: " " });
    await flush();
    keyDownOn({ target: button, key: "ArrowRight" });
    await waitUntil(() => graph.nodes[0]?.width === originWidth + 1);
    keyDownOn({ target: button, key: "Enter" });
    await waitUntil(() => ends.length === 1);
    assert.equal(ends[0], originWidth + 1);
    assert.equal(graph.nodes[0]?.width, originWidth + 1);
    graph.remove();
  });

  test("keyboard step on w moves the west edge and clamps at min-width", async () => {
    const originX = 40;
    const originWidth = 80;
    const minWidth = 78;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: originX, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    resizer.setAttribute("min-width", String(minWidth));
    const w = controlOf({ resizer, direction: "w" });
    const button = controlButtonOf(w);
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    const east = originX + originWidth;
    // Three rightward steps of 1px: width 80 → 78 (clamped at min-width);
    // the east edge stays fixed at 120.
    const steps = 3;
    for (let i = 0; i < steps; i += 1) {
      keyDownOn({ target: button, key: "ArrowRight" });
    }
    await waitUntil(() => graph.nodes[0]?.width === minWidth);
    const node = graph.nodes[0];
    assert.ok(node !== undefined);
    assert.equal(node.width, minWidth);
    assert.equal(node.position.x + (node.width ?? 0), east);
    graph.remove();
  });

  test("click on a resize control button does not emit flow-node-click or activate", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const clicks: Event[] = [];
    graph.addEventListener("flow-node-click", (event) => clicks.push(event));
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    const button = controlButtonOf(se);
    button.dispatchEvent(new MouseEvent("click", {
      bubbles: POINTER_BUBBLES,
      composed: POINTER_COMPOSED,
      cancelable: true,
      detail: 0,
    }));
    await flush();
    assert.equal(clicks.length, 0);
    graph.remove();
  });

  test("detail-0 click on the node's own button still activates the node", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    const clicks: Event[] = [];
    graph.addEventListener("flow-node-click", (event) => clicks.push(event));
    const nodeButton = nodeEl.shadowRoot?.querySelector("button");
    assert.ok(nodeButton instanceof HTMLButtonElement);
    nodeButton.dispatchEvent(new MouseEvent("click", {
      bubbles: POINTER_BUBBLES,
      composed: POINTER_COMPOSED,
      cancelable: true,
      detail: 0,
    }));
    await waitUntil(() => clicks.length === 1);
    const click = clicks[0];
    assert.ok(click instanceof CustomEvent);
    const detail = click.detail as { node: { id: string } };
    assert.equal(detail.node.id, "a");
    await waitUntil(() => nodeEl.node?.selected === true);
    assert.ok(nodeEl.node?.selected === true);
    graph.remove();
  });

  test("unknown and orthogonal keys are no-ops during keyboard resize", async () => {
    const originWidth = 80;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const moves: number[] = [];
    graph.addEventListener("flow-node-resize", () => moves.push(1));
    const e = controlOf({ resizer: offeredResizer(nodeEl), direction: "e" });
    const button = controlButtonOf(e);
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    keyDownOn({ target: button, key: "x" });
    keyDownOn({ target: button, key: "ArrowUp" });
    await flush();
    assert.equal(moves.length, 0);
    assert.equal(graph.nodes[0]?.width, originWidth);
    keyDownOn({ target: button, key: "Escape" });
    graph.remove();
  });

  test("idle arrows do not start a resize", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const starts: Event[] = [];
    graph.addEventListener("flow-node-resize-start", (event) => starts.push(event));
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    keyDownOn({ target: controlButtonOf(se), key: "ArrowRight" });
    await flush();
    assert.equal(starts.length, 0);
    assert.doesNotMatch(graph.state(), /\/resize/);
    assert.equal(graph.nodes[0]?.width, 80);
    graph.remove();
  });

  test("stop in flight drops further keyboard resize apply and emit", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const seen: string[] = [];
    graph.addEventListener("flow-node-resize", () => seen.push("move"));
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const button = controlButtonOf(controlOf({ resizer: offeredResizer(nodeEl), direction: "se" }));
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    const stopping = graph.stop();
    keyDownOn({ target: button, key: "ArrowRight" });
    keyDownOn({ target: button, key: "ArrowRight" });
    await stopping;
    assert.equal(graph.nodes[0]?.width, 80);
    assert.equal(seen.length, 0);
    graph.remove();
  });

  test("keys on another control are ignored while a keyboard resize is active", async () => {
    const originWidth = 80;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const resizer = offeredResizer(nodeEl);
    const seButton = controlButtonOf(controlOf({ resizer, direction: "se" }));
    const nwButton = controlButtonOf(controlOf({ resizer, direction: "nw" }));
    keyDownOn({ target: seButton, key: "Enter" });
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    keyDownOn({ target: nwButton, key: "Enter" });
    keyDownOn({ target: nwButton, key: "ArrowLeft" });
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    assert.equal(graph.nodes[0]?.width, originWidth);
    keyDownOn({ target: seButton, key: "Escape" });
    await waitUntil(() => !/\/resize/.test(graph.state()));
    graph.remove();
  });

  test("removing the node ends the keyboard resize session", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const ends: Event[] = [];
    graph.addEventListener("flow-node-resize-end", (event) => ends.push(event));
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const button = controlButtonOf(controlOf({ resizer: offeredResizer(nodeEl), direction: "se" }));
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    graph.nodes = [];
    await waitUntil(() => !/\/resize/.test(graph.state()));
    assert.doesNotMatch(graph.state(), /\/resize/);
    graph.remove();
  });

  test("deselecting the node ends the keyboard resize session", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const button = controlButtonOf(controlOf({ resizer: offeredResizer(nodeEl), direction: "se" }));
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    await graph.dispatch(hsm.typedEvent({
      event: Selection.changedEvent,
      data: { nodeIds: [], edgeIds: [] },
    }));
    await waitUntil(() => !/\/resize/.test(graph.state()));
    assert.doesNotMatch(graph.state(), /\/resize/);
    graph.remove();
  });

  test("pointer-then-Enter does not jump the box or start a keyboard session", async () => {
    const originWidth = 80;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 90, clientY: 45 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 100, clientY: 50 })));
    await waitUntil(() => graph.nodes[0]?.width === 100);
    assert.match(graph.state(), /\/resize\/pointer$/);
    keyDownOn({ target: controlButtonOf(se), key: "Enter" });
    keyDownOn({ target: controlButtonOf(se), key: "ArrowRight" });
    await flush();
    assert.match(graph.state(), /\/resize\/pointer$/);
    assert.equal(graph.nodes[0]?.width, 100);
    graph.remove();
  });

  test("keyboard-then-pointerdown does not apply world-as-delta", async () => {
    const originWidth = 80;
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: originWidth, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const se = controlOf({ resizer: offeredResizer(nodeEl), direction: "se" });
    const button = controlButtonOf(se);
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    keyDownOn({ target: button, key: "ArrowRight" });
    await waitUntil(() => graph.nodes[0]?.width === originWidth + 1);
    assert.match(graph.state(), /\/resize\/keyboard$/);
    se.dispatchEvent(new PointerEvent("pointerdown", pointerInit({ clientX: 80, clientY: 40 })));
    se.dispatchEvent(new PointerEvent("pointermove", pointerInit({ clientX: 200, clientY: 90 })));
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    assert.equal(graph.nodes[0]?.width, originWidth + 1);
    keyDownOn({ target: button, key: "Escape" });
    graph.remove();
  });

  test("disconnect and reconnect mid-keyboard resize does not leave a stale session", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof FlowNode);
    await selectFirstNode(nodeEl);
    const button = controlButtonOf(controlOf({ resizer: offeredResizer(nodeEl), direction: "se" }));
    keyDownOn({ target: button, key: "Enter" });
    await flush();
    assert.match(graph.state(), /\/resize\/keyboard$/);
    graph.remove();
    await waitUntil(() => /\/disconnected$/.test(graph.state()));
    document.body.append(graph);
    await flush();
    assert.match(graph.state(), /\/connected\//);
    assert.doesNotMatch(graph.state(), /\/resize/);
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const reattached = graph.querySelector("flow-node");
    assert.ok(reattached instanceof FlowNode);
    await selectFirstNode(reattached);
    const next = controlButtonOf(controlOf({ resizer: offeredResizer(reattached), direction: "se" }));
    const starts: Event[] = [];
    graph.addEventListener("flow-node-resize-start", (event) => starts.push(event));
    keyDownOn({ target: next, key: "Enter" });
    await flush();
    assert.equal(starts.length, 1);
    assert.match(graph.state(), /\/resize\/keyboard$/);
    graph.remove();
  });
});

describe("resizedBounds", () => {
  test("clamps and fixes the opposite edge", () => {
    const origin = { x: 10, y: 20, width: 80, height: 40 };
    const constraints = { minWidth: 10, minHeight: 10, keepAspectRatio: false };
    const se = resizedBounds({ origin, direction: "se", dx: 20, dy: 10, constraints });
    assert.deepEqual(se, { x: 10, y: 20, width: 100, height: 50 });
    const nw = resizedBounds({ origin, direction: "nw", dx: -10, dy: -10, constraints });
    assert.equal(nw.x + nw.width, origin.x + origin.width);
    assert.equal(nw.y + nw.height, origin.y + origin.height);
    const n = resizedBounds({ origin, direction: "n", dx: 0, dy: -10, constraints });
    assert.equal(n.y + n.height, origin.y + origin.height);
    const s = resizedBounds({ origin, direction: "s", dx: 0, dy: 10, constraints });
    assert.equal(s.x, origin.x);
    const e = resizedBounds({ origin, direction: "e", dx: 10, dy: 0, constraints });
    assert.equal(e.y, origin.y);
    const w = resizedBounds({ origin, direction: "w", dx: -10, dy: 0, constraints });
    assert.equal(w.x + w.width, origin.x + origin.width);
    const ne = resizedBounds({ origin, direction: "ne", dx: 10, dy: -10, constraints });
    assert.equal(ne.x + ne.width, origin.x + origin.width + 10);
    const sw = resizedBounds({ origin, direction: "sw", dx: -10, dy: 10, constraints });
    assert.equal(sw.y + sw.height, origin.y + origin.height + 10);
    const clamped = resizedBounds({
      origin,
      direction: "se",
      dx: -100,
      dy: -100,
      constraints: { minWidth: 30, minHeight: 20, keepAspectRatio: false },
    });
    assert.equal(clamped.width, 30);
    assert.equal(clamped.height, 20);
  });
});

describe("flow-controls", () => {
  test("clicking a control emits non-cancelable flow-control without applying zoom", async () => {
    const zoomIn = "zoom-in";
    const oneAction = 1;
    const controls = document.createElement("flow-controls");
    const actions: string[] = [];
    const events: Event[] = [];
    controls.addEventListener("flow-control", (event: Event) => {
      events.push(event);
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["action"] === "string") {
        actions.push(event.detail["action"]);
      }
    });
    document.body.append(controls);
    const zoomInButton = getByRole(controls, "button", "zoom in");
    assert.ok(zoomInButton instanceof HTMLButtonElement);
    // The fit control keeps a distinct stable name that cannot collide with
    // the dashboard's "Fit environment map" button case-insensitively.
    assert.ok(getByRole(controls, "button", "Fit flow view") instanceof HTMLButtonElement);
    assert.ok(getByRole(controls, "button", "zoom out") instanceof HTMLButtonElement);
    const clickBubbles = true;
    const clickComposed = true;
    zoomInButton.dispatchEvent(new Event("click", { bubbles: clickBubbles, composed: clickComposed }));
    assert.equal(actions.length, oneAction);
    assert.equal(actions[0], zoomIn);
    for (const event of events) {
      assertPublicCustomEvent(event);
    }
    controls.remove();
  });
});

describe("flow-node accessible names", () => {
  test("node buttons without a label fall back to the node id for the accessible name", () => {
    const nodeEl = document.createElement("flow-node");
    document.body.append(nodeEl);
    const unlabeled: Node = { id: "initial:/Phone:entry", position: { x: 0, y: 0 }, data: { label: "" }, width: 24, height: 24 };
    nodeEl.node = unlabeled;
    const button = nodeEl.shadowRoot?.querySelector("button");
    assert.ok(button instanceof HTMLButtonElement);
    assert.equal(button.getAttribute("aria-label"), unlabeled.id);
    const labeled: Node = { id: "labeled", position: { x: 0, y: 0 }, data: { label: "A" }, width: 40, height: 20 };
    nodeEl.node = labeled;
    assert.equal(button.getAttribute("aria-label"), "A");
    nodeEl.remove();
  });
});

describe("flow-graph agent accessibility", () => {
  test("named edge buttons emit flow-edge-click", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    const eventName = "phone.ring";
    graph.nodes = [
      { id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 },
      { id: "b", position: { x: 200, y: 0 }, data: { label: "B" }, width: 80, height: 40 },
    ];
    graph.edges = [{ id: "a-b", source: "a", target: "b", data: { eventName } }];
    await waitUntil(() => {
      try {
        getByRole(graph, "button", eventName);
        return true;
      } catch {
        return false;
      }
    });
    assert.ok(getByRole(graph, "group", "Edges") instanceof HTMLElement);
    const edge = getByRole(graph, "button", eventName);
    assert.ok(edge instanceof HTMLButtonElement);
    const clicks: Event[] = [];
    graph.addEventListener("flow-edge-click", (event) => clicks.push(event));
    const clickBubbles = true;
    const clickComposed = true;
    const oneClick = 1;
    edge.dispatchEvent(new Event("click", { bubbles: clickBubbles, composed: clickComposed }));
    await flush();
    assert.equal(clicks.length, oneClick);
    const click = clicks[0];
    assert.ok(click instanceof CustomEvent);
    assertPublicCustomEvent(click);
    assert.ok(hsm.isRecord(click.detail) && hsm.isRecord(click.detail["edge"]));
    assert.equal(click.detail["edge"]["id"], "a-b");
    graph.remove();
  });

  test("background is decorative", () => {
    const background = document.createElement("flow-background");
    document.body.append(background);
    assert.equal(background.getAttribute("aria-hidden"), "true");
    background.remove();
  });
});
