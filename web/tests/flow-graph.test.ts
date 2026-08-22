import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { FlowGraph } from "../src/flow/graph.ts";
import { FlowEdge } from "../src/flow/edge.ts";
import { FlowNode } from "../src/flow/node.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { getBezierPath, getNodesBounds, getStraightPath, getViewportForBounds } from "../src/flow/path.ts";
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

registerFlowElements();

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
    // No drag_move sample feeds the Dragger: no node move in the same turn.
    assert.equal(graph.nodes[0]?.position.x, nodeX);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 80, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => graph.nodes[0]?.position.x !== nodeX);
    nodeEl.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 80, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    assert.notEqual(graph.nodes[0]?.position.x, 0);
    graph.remove();
  });

  test("drag_start polls on the 16ms frame and produces drag_moved without drag_move samples", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodesDraggable = true;
    graph.panOnDrag = true;
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    await waitUntil(() => graph.querySelector("flow-node") !== null);
    const nodeEl = graph.querySelector("flow-node");
    assert.ok(nodeEl instanceof HTMLElement);
    const frameMs = 16;
    const cadenceFloorMs = 10;
    const noMove = 0;
    nodeEl.dispatchEvent(new PointerEvent("pointerdown", {
      clientX: 10, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/click$/);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 50, clientY: 10, pointerId: 1, bubbles: true, composed: true,
    }));
    assert.match(graph.state(), /\/drag$/);
    const enteredDragAt = Date.now();
    // No drag_move event ingress: a held pointer does not move the node yet.
    assert.equal(graph.nodes[0]?.position.x, noMove);
    assert.equal(graph.nodes[0]?.position.y, noMove);
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 90, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => graph.nodes[0]?.position.x !== noMove || graph.nodes[0]?.position.y !== noMove);
    assert.ok(Date.now() - enteredDragAt >= cadenceFloorMs, "move arrives on the frame cadence, not per sample");
    assert.equal(graph.nodes[0]?.position.x, 40);
    assert.equal(graph.nodes[0]?.position.y, 20);
    nodeEl.dispatchEvent(new PointerEvent("pointerup", {
      clientX: 90, clientY: 30, pointerId: 1, bubbles: true, composed: true,
    }));
    await waitUntil(() => /\/idle$/.test(graph.state()));
    const settled = { x: graph.nodes[0]?.position.x, y: graph.nodes[0]?.position.y };
    nodeEl.dispatchEvent(new PointerEvent("pointermove", {
      clientX: 120, clientY: 40, pointerId: 1, bubbles: true, composed: true,
    }));
    await new Promise<void>((resolve) => { globalThis.setTimeout(resolve, frameMs * 3); });
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
    graph.panOnDrag = policyOff;
    assert.equal(graph.state(), "");
    assert.equal(graph.nodesDraggable, policyOff);
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
    graph.panOnDrag = policyOff;
    await flush();
    assert.equal(graph.nodesDraggable, policyOn);
    assert.equal(graph.panOnDrag, policyOn);
    assert.ok(drops.length >= atLeastOneDrop);
    assert.ok(drops.some((drop) => drop.reason === stopped));
    graph.remove();
    document.body.append(graph);
    await waitUntil(() => /\/connected\//.test(graph.state()));
    assert.equal(graph.nodesDraggable, policyOn);
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
    const childCount = 6;
    assert.equal(actors.length, childCount);
    for (const actor of actors) {
      assert.notEqual(actor.state(), "");
    }
    await graph.stop();
    assert.equal(graph.state(), "");
    for (const actor of actors) {
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
