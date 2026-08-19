import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { graphsForVisibility } from "../src/dashboard-graphs.ts";
import { type MachineGraph } from "../src/otel/machines.ts";

function machine(name: string, owner?: string): MachineGraph {
  return {
    name,
    ...(owner === undefined ? {} : { owner }),
    componentName: name.slice(1),
    currentState: `${name}/ready`,
    lastEventName: "ready",
    nodes: [
      { path: name, parent: null, label: name.slice(1) },
      { path: `${name}/ready`, parent: name, label: "ready" },
    ],
    edges: [{ source: `${name}/ready`, target: name, eventName: "reset", count: 1, lastFired: true }],
    observationCount: 3,
  };
}

describe("dashboard render graph admission", () => {
  test("keeps visible descendants nested under root-only hidden owner shells", () => {
    const environment = machine("/Environment");
    const service = machine("/Service", "/Environment");
    const child = machine("/Child", "/Service");

    const graphs = graphsForVisibility(
      [environment, service, child],
      new Map([
        ["/Environment", false],
        ["/Service", false],
        ["/Child", true],
      ]),
    );

    assert.deepEqual(graphs.map((graph) => graph.name), ["/Environment", "/Service", "/Child"]);
    assert.equal(graphs[0]?.nodes.length, 1);
    assert.equal(graphs[1]?.nodes.length, 1);
    assert.equal(graphs[0]?.edges.length, 0);
    assert.equal(graphs[1]?.edges.length, 0);
    assert.equal(graphs[0]?.currentState, "");
    assert.equal(graphs[1]?.observationCount, 0);
    assert.equal(graphs[0]?.owner, undefined);
    assert.equal(graphs[1]?.owner, "/Environment");
    assert.equal(graphs[1]?.componentName, "Service");
    assert.deepEqual(graphs[1]?.nodes, [{ path: "/Service", parent: null, label: "Service" }]);
    assert.equal(graphs[2], child);
  });

  test("does not include hidden shells when no visible descendant needs them", () => {
    const owner = machine("/Environment");
    const child = machine("/Child", "/Environment");

    assert.deepEqual(graphsForVisibility([owner, child], new Map([["/Environment", false], ["/Child", false]])), []);
    assert.deepEqual(graphsForVisibility([owner, child], new Map([["/Environment", true], ["/Child", false]])).map((graph) => graph.name), ["/Environment"]);
  });
});
