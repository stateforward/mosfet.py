import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { BotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { registerBotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { FlowGraph } from "../src/flow/index.ts";

registerFlowElements();
registerBotMachineGraph();

const YIELD_MS = 0;
const FRAME_PART = '[part="frame"]';
const NESTED_NODES_DRAGGABLE = false;
const NESTED_PAN_ON_DRAG = true;

function flowFrame(host: BotMachineGraph): FlowGraph {
  const flow = host.shadowRoot?.querySelector(FRAME_PART);
  assert.ok(flow instanceof FlowGraph);
  return flow;
}

function spyFitView(flow: FlowGraph): { count: () => number; restore: () => void } {
  let fitViewCount = 0;
  const originalFitView = flow.fitView.bind(flow);
  flow.fitView = () => {
    fitViewCount += 1;
    originalFitView();
  };
  return {
    count: () => fitViewCount,
    restore: () => {
      flow.fitView = originalFitView;
    },
  };
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for bot-machine-graph");
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

function graphFor(name: string) {
  return {
    name,
    componentName: name.slice(1),
    currentState: `${name}/ready`,
    lastEventName: "",
    observationCount: 1,
    nodes: [
      { path: name, parent: null, label: name.slice(1) },
      { path: `${name}/ready`, parent: name, label: "ready" },
    ],
    edges: [],
  };
}

describe("bot-machine-graph flow host", () => {
  test("constructor does not start the host", () => {
    const host = document.createElement("bot-machine-graph");
    assert.equal(host.state(), "");
  });

  test("constructor does not start the nested flow-graph", () => {
    const host = document.createElement("bot-machine-graph");
    const flow = flowFrame(host);
    assert.equal(host.state(), "");
    assert.equal(flow.state(), "");
    assert.equal(flow.nodesDraggable, NESTED_NODES_DRAGGABLE);
    assert.equal(flow.panOnDrag, NESTED_PAN_ON_DRAG);
  });

  test("nested connect admits nodesDraggable false after the child starts", async () => {
    const host = document.createElement("bot-machine-graph");
    document.body.append(host);
    const flow = flowFrame(host);
    await waitUntil(() => flow.isConnected && /connected|pointer/.test(flow.state()));
    assert.match(flow.state(), /connected|pointer/);
    assert.equal(flow.nodesDraggable, NESTED_NODES_DRAGGABLE);
    assert.equal(flow.panOnDrag, NESTED_PAN_ON_DRAG);
    host.remove();
  });

  test("Host.stop stops nested Graph actor", async () => {
    const host = document.createElement("bot-machine-graph");
    document.body.append(host);
    await waitUntil(() => host.state().includes("/connected"));
    const flow = flowFrame(host);
    const actors = ownedActors(host);
    const graph = actors.find((actor) => !(actor instanceof FlowGraph));
    assert.ok(graph !== undefined);
    assert.ok(actors.some((actor) => actor instanceof FlowGraph));
    assert.notEqual(graph.state(), "");
    assert.notEqual(flow.state(), "");
    await host.stop();
    assert.equal(host.state(), "");
    assert.equal(graph.state(), "");
    assert.equal(flow.state(), "");
    host.remove();
  });

  test("graphs write before append applies after nested connect without dropping fit", async () => {
    const host = document.createElement("bot-machine-graph");
    const flow = flowFrame(host);
    const spy = spyFitView(flow);
    const drops: Event[] = [];
    flow.addEventListener("host-drop", (event: Event) => {
      drops.push(event);
    });
    try {
      const graphs = [graphFor("/Phone")];
      const noNodes = 0;
      const nodeCount = graphs.reduce((count, graph) => count + graph.nodes.length, noNodes);
      host.graphs = graphs;
      assert.equal(host.getAttribute("data-node-count"), null);
      document.body.append(host);
      const readyState = "/ready";
      await waitUntil(() => host.getAttribute("data-node-count") === String(nodeCount) && host.state().endsWith(readyState));
      await waitUntil(() => /connected|pointer/.test(flow.state()));
      const oneFit = 1;
      assert.equal(spy.count(), oneFit);
      assert.equal(drops.length, noNodes);
      host.remove();
    } finally {
      spy.restore();
    }
  });

  test("admits graphs and focuses a machine on a defined element", async () => {
    const host = document.createElement("bot-machine-graph");
    assert.ok(host instanceof BotMachineGraph);
    document.body.append(host);
    const graphs = [graphFor("/Phone")];
    host.graphs = graphs;
    assert.equal(host.graphs.length, 1);
    assert.equal(host.graphs[0]?.name, "/Phone");
    const noNodes = 0;
    const phoneNodeCount = graphs.reduce((count, graph) => count + graph.nodes.length, noNodes);
    await waitUntil(() => host.getAttribute("data-node-count") === String(phoneNodeCount));
    assert.equal(host.getAttribute("data-node-count"), String(phoneNodeCount));
    assert.match(host.state(), /\/ready$/);
    const focused: string[] = [];
    const originalDispatch = host.dispatch.bind(host);
    host.dispatch = ((eventOrCtx: hsm.Event | hsm.Context, maybeEvent?: hsm.Event) => {
      const event = maybeEvent ?? (eventOrCtx instanceof hsm.Context ? undefined : eventOrCtx);
      if (event !== undefined && event.name === BotMachineGraph.focusEvent.name && hsm.isRecord(event.data)) {
        const machineName = event.data["machineName"];
        if (typeof machineName === "string") focused.push(machineName);
      }
      return maybeEvent === undefined
        ? originalDispatch(eventOrCtx as hsm.Event)
        : originalDispatch(eventOrCtx as hsm.Context, maybeEvent);
    }) as BotMachineGraph["dispatch"];
    const flow = flowFrame(host);
    let fitBoundsCount = 0;
    const originalFitBounds = flow.fitBounds.bind(flow);
    flow.fitBounds = (bounds) => {
      fitBoundsCount += 1;
      originalFitBounds(bounds);
    };
    try {
      host.focusMachine("/Phone");
      host.focusMachine("/Missing");
      await waitUntil(() => focused.length === 2);
      assert.deepEqual(focused, ["/Phone", "/Missing"]);
      const oneFit = 1;
      assert.equal(fitBoundsCount, oneFit);
      assert.equal(host.getAttribute("data-node-count"), String(phoneNodeCount));
      host.graphs = [{
        name: "/Empty",
        componentName: "Empty",
        currentState: "/Empty",
        lastEventName: "",
        observationCount: 0,
        nodes: [],
        edges: [],
      }];
      await waitUntil(() => host.getAttribute("data-node-count") === "0");
      assert.equal(host.getAttribute("data-node-count"), "0");
      const emptyFitBefore = fitBoundsCount;
      host.focusMachine("/Empty");
      await waitUntil(() => focused.length === 3);
      assert.equal(fitBoundsCount, emptyFitBefore);
    } finally {
      host.dispatch = originalDispatch;
      flow.fitBounds = originalFitBounds;
    }
    await host.dispatch(hsm.typedEvent({ event: BotMachineGraph.nodeClickEvent, data: {
      machineName: "/Phone",
      path: "/Phone/ready",
      bounds: { left: 0, right: 40, top: 0, bottom: 20 },
    } }));
    await host.dispatch(hsm.typedEvent({
      event: { name: "graph.drawn", kind: hsm.Kinds.Event },
      data: { graphs: [{ not: "a graph" }] },
    }));
    assert.equal(host.graphs[0]?.name, "/Empty");
    assert.match(host.state(), /\/connected/);
    host.dispatchEvent(new CustomEvent("flow-node-click", { detail: { node: {} }, bubbles: true, composed: true }));
    host.dispatchEvent(new CustomEvent("flow-node-click", {
      detail: {
        node: {
          id: "n",
          position: { x: 0, y: 0 },
          data: { path: "/Phone/ready", machineName: "/Phone" },
          width: 40,
          height: 20,
        },
        originalEvent: { pointerId: 1, clientX: 0, clientY: 0, type: "pointerup" },
      },
      bubbles: true,
      composed: true,
    }));
    host.remove();
  });

  test("first admit on an empty model ends ready and runs fitView", async () => {
    const host = document.createElement("bot-machine-graph");
    assert.ok(host instanceof BotMachineGraph);
    const spy = spyFitView(flowFrame(host));
    try {
      document.body.append(host);
      await waitUntil(() => host.state().includes("/connected"));
      const graphs = [graphFor("/Phone")];
      const noNodes = 0;
      const nodeCount = graphs.reduce((count, graph) => count + graph.nodes.length, noNodes);
      host.graphs = graphs;
      const readyState = "/ready";
      await waitUntil(() => host.getAttribute("data-node-count") === String(nodeCount) && host.state().endsWith(readyState));
      assert.ok(host.state().endsWith(readyState));
      const oneFit = 1;
      assert.equal(spy.count(), oneFit);
      host.remove();
    } finally {
      spy.restore();
    }
  });

  test("second admit with the same node count ends ready without another fitView", async () => {
    const host = document.createElement("bot-machine-graph");
    assert.ok(host instanceof BotMachineGraph);
    const spy = spyFitView(flowFrame(host));
    try {
      document.body.append(host);
      await waitUntil(() => host.state().includes("/connected"));
      const graphs = [graphFor("/Phone")];
      const noNodes = 0;
      const nodeCount = graphs.reduce((count, graph) => count + graph.nodes.length, noNodes);
      host.graphs = graphs;
      const readyState = "/ready";
      await waitUntil(() => host.getAttribute("data-node-count") === String(nodeCount) && host.state().endsWith(readyState));
      const oneFit = 1;
      assert.equal(spy.count(), oneFit);
      const again = [graphFor("/Phone2")];
      const againNodeCount = again.reduce((count, graph) => count + graph.nodes.length, noNodes);
      assert.equal(againNodeCount, nodeCount);
      host.graphs = again;
      const admittedAgain = "/Phone2";
      await waitUntil(() => host.graphs[0]?.name === admittedAgain && host.state().endsWith(readyState));
      assert.ok(host.state().endsWith(readyState));
      assert.equal(spy.count(), oneFit);
      host.remove();
    } finally {
      spy.restore();
    }
  });

  test("echoes viewport zoom and edge clicks as non-cancelable host events", async () => {
    const host = document.createElement("bot-machine-graph");
    assert.ok(host instanceof BotMachineGraph);
    const publicEventCancelable = false;
    const publicEventBubbles = true;
    const publicEventComposed = true;
    const publicEventInit = {
      bubbles: publicEventBubbles,
      composed: publicEventComposed,
      cancelable: publicEventCancelable,
    };
    const originX = 0;
    const originY = 0;
    const zoom = 1.25;
    const eventName = "ping";
    const atLeastOne = 1;
    const zooms: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; zoom: number }> = [];
    const edges: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; eventName: string }> = [];
    host.addEventListener("bot-machine-graph-zoom", (event: Event) => {
      if (!(event instanceof CustomEvent) || typeof event.detail?.zoom !== "number") return;
      zooms.push({
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
        zoom: event.detail.zoom,
      });
    });
    host.addEventListener("bot-machine-graph-edge", (event: Event) => {
      if (!(event instanceof CustomEvent) || typeof event.detail?.eventName !== "string") return;
      edges.push({
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
        eventName: event.detail.eventName,
      });
    });
    document.body.append(host);
    await waitUntil(() => host.state().includes("/connected"));
    host.dispatchEvent(new CustomEvent("flow-viewport-change", {
      ...publicEventInit,
      detail: { viewport: { x: originX, y: originY, zoom } },
    }));
    host.dispatchEvent(new CustomEvent("flow-edge-click", {
      ...publicEventInit,
      detail: { edge: { id: "e", source: "a", target: "b", data: { eventName } } },
    }));
    await waitUntil(() => zooms.length >= atLeastOne && edges.length >= atLeastOne);
    assert.ok(zooms.length >= atLeastOne);
    assert.ok(edges.length >= atLeastOne);
    for (const seen of zooms) {
      assert.equal(seen.cancelable, publicEventCancelable);
      assert.equal(seen.bubbles, publicEventBubbles);
      assert.equal(seen.composed, publicEventComposed);
      assert.equal(seen.zoom, zoom);
    }
    for (const seen of edges) {
      assert.equal(seen.cancelable, publicEventCancelable);
      assert.equal(seen.bubbles, publicEventBubbles);
      assert.equal(seen.composed, publicEventComposed);
      assert.equal(seen.eventName, eventName);
    }
    host.remove();
  });
});
