import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { FlowGraph } from "../src/flow/graph.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { getBezierPath, getNodesBounds, getStraightPath, getViewportForBounds } from "../src/flow/path.ts";
import { MAX_FLOW_EDGES, MAX_FLOW_NODES, type PointerSampleData } from "../src/flow/types.ts";

registerFlowElements();

const YIELD_MS = 0;

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
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

  test("pans through dispatched pointer events", async () => {
    const graph = document.createElement("flow-graph");
    document.body.append(graph);
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: { label: "A" }, width: 80, height: 40 }];
    const before = graph.getViewport();
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 40, y: 40 },
      client: { x: 40, y: 40 },
      viewport: { x: 40, y: 40 },
      hit: { kind: "empty" },
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 40, y: 40 },
      client: { x: 80, y: 90 },
      viewport: { x: 80, y: 90 },
      hit: { kind: "empty" },
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 40, y: 40 },
      client: { x: 80, y: 90 },
      viewport: { x: 80, y: 90 },
      hit: { kind: "empty" },
    }) }));
    await flush();
    const after = graph.getViewport();
    assert.ok(Math.abs(after.x - before.x) + Math.abs(after.y - before.y) > 0);
    const viewport = graph.shadowRoot?.childNodes[0] as { childNodes?: Array<{ className?: string; style?: { transform?: string } }> } | undefined;
    const world = viewport?.childNodes?.find((node) => node.className === "world");
    assert.match(world?.style?.transform ?? "", /translate\(/);
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
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      shiftKey: true,
      origin: { x: 10, y: 10 },
      client: { x: 10, y: 10 },
      viewport: { x: 10, y: 10 },
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/box$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      shiftKey: true,
      origin: { x: 10, y: 10 },
      client: { x: 120, y: 80 },
      viewport: { x: 120, y: 80 },
      hit: { kind: "empty" },
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
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
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 20, y: 20 },
      client: { x: 20, y: 20 },
      viewport: { x: 20, y: 20 },
      hit: { kind: "empty" },
    }) }));
    assert.match(graph.state(), /\/pan$/);
    assert.doesNotMatch(graph.state(), /\/box$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 20, y: 20 },
      client: { x: 60, y: 70 },
      viewport: { x: 60, y: 70 },
      hit: { kind: "empty" },
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
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
    const admitted = graph.nodes[0];
    assert.ok(admitted !== undefined);
    const hit = { kind: "node" as const, node: admitted };
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 10, y: 10 },
      client: { x: 10, y: 10 },
      viewport: { x: 10, y: 10 },
      world: { x: 10, y: 10 },
      hit,
    }) }));
    assert.match(graph.state(), /\/click$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 10, y: 10 },
      client: { x: 40, y: 10 },
      viewport: { x: 40, y: 10 },
      world: { x: 40, y: 10 },
      hit,
    }) }));
    assert.match(graph.state(), /\/drag$/);
    assert.doesNotMatch(graph.state(), /\/pan$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 10, y: 10 },
      client: { x: 80, y: 10 },
      viewport: { x: 80, y: 10 },
      world: { x: 80, y: 10 },
      hit,
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
      eventType: "pointerup",
      buttons: 0,
      origin: { x: 10, y: 10 },
      client: { x: 80, y: 10 },
      viewport: { x: 80, y: 10 },
      world: { x: 80, y: 10 },
      hit,
    }) }));
    await flush();
    assert.match(graph.state(), /\/idle$/);
    assert.notEqual(graph.nodes[0]?.position.x, 0);
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
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      origin: { x: 80, y: 20 },
      client: { x: 80, y: 20 },
      world: { x: 80, y: 20 },
      hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
    }) }));
    assert.match(graph.state(), /\/connect$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerSampleEvent, data: pointerData({
      eventType: "pointermove",
      origin: { x: 80, y: 20 },
      client: { x: 200, y: 20 },
      world: { x: 200, y: 20 },
      hit: { kind: "handle", node: target, handleKind: "target", position: "left" },
    }) }));
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
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
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerDownEvent, data: pointerData({
      eventType: "pointerdown",
      hit: { kind: "handle", node: source, handleKind: "source", position: "right" },
    }) }));
    assert.match(graph.state(), /\/connect$/);
    graph.dispatch(hsm.typedEvent({ event: FlowGraph.pointerUpEvent, data: pointerData({
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
    const viewport = getViewportForBounds(bounds, 180, 90, 0.1, 2, 0);
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

  test("nodes write before connect emits host-drop", async () => {
    const graph = document.createElement("flow-graph");
    const drops: string[] = [];
    graph.addEventListener("host-drop", (event: Event) => {
      if (event instanceof CustomEvent && hsm.isRecord(event.detail) && typeof event.detail["reason"] === "string") {
        drops.push(event.detail["reason"]);
      }
    });
    graph.nodes = [{ id: "a", position: { x: 0, y: 0 }, data: {} }];
    await flush();
    assert.equal(graph.nodes.length, 0);
    assert.ok(drops.includes("unstarted") || drops.length >= 1);
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

  test("viewport is an application landmark", async () => {
    const graph = document.createElement("flow-graph");
    const roleMissingBeforeConnect = false;
    assert.equal(graph.hasAttribute("role"), roleMissingBeforeConnect);
    document.body.append(graph);
    assert.equal(graph.getAttribute("role"), "application");
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
    graph.dispatchEvent(new KeyboardEvent("keydown", { key: "+", bubbles: true }));
    await flush();
    const zoomed = graph.getViewport();
    assert.ok(zoomed.zoom > before.zoom);
    graph.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    await flush();
    const panned = graph.getViewport();
    assert.ok(panned.x !== zoomed.x);
    graph.dispatchEvent(new KeyboardEvent("keydown", { key: "f", bubbles: true }));
    await flush();
    const fitted = graph.getViewport();
    assert.notEqual(fitted.zoom, panned.zoom);
    graph.remove();
  });
});
