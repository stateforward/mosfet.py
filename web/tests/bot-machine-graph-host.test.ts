import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { BotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { registerBotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";

registerFlowElements();
registerBotMachineGraph();

const YIELD_MS = 0;

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for bot-machine-graph");
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
  test("admits graphs and focuses a machine on a defined element", async () => {
    const host = document.createElement("bot-machine-graph");
    assert.ok(host instanceof BotMachineGraph);
    document.body.append(host);
    const graphs = [graphFor("/Phone")];
    host.graphs = graphs;
    assert.equal(host.graphs.length, 1);
    assert.equal(host.graphs[0]?.name, "/Phone");
    const phoneNodeCount = graphs.reduce((count, graph) => count + graph.nodes.length, 0);
    await waitUntil(() => host.getAttribute("data-node-count") === String(phoneNodeCount));
    assert.equal(host.getAttribute("data-node-count"), String(phoneNodeCount));
    assert.equal(host.focusMachine("/Phone"), true);
    assert.equal(host.focusMachine("/Missing"), false);
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
    assert.equal(host.focusMachine("/Empty"), false);
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
});
