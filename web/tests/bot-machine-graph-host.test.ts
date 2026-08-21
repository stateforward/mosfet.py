import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { BotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { registerBotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";

registerFlowElements();
registerBotMachineGraph();

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
  test("admits graphs and focuses a machine", () => {
    const host = new BotMachineGraph();
    host.connectedCallback();
    const graphs = [graphFor("/Phone")];
    host.graphs = graphs;
    assert.equal(host.graphs.length, 1);
    assert.equal(host.graphs[0]?.name, "/Phone");
    assert.equal(host.focusMachine("/Phone"), true);
    assert.equal(host.focusMachine("/Missing"), false);
    host.disconnectedCallback();
  });
});
