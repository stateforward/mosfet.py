import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  environmentWorkspaceGraphs,
  environmentRootGraphs,
  graphsForVisibility,
  machineNamesInOwnedSubtree,
} from "../src/dashboard-graphs.ts";
import { type MachineGraph } from "../src/otel/machines.ts";

function machine(name: string, owner?: string | null): MachineGraph {
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
  test("lists only valid direct environment roots and keeps owned descendants in the subtree", () => {
    const environment = machine("/Environment", null);
    const explicitRoot = machine("/ExplicitRoot", null);
    const child = machine("/Child", "/Environment");
    const grandchild = machine("/Grandchild", "/Child");
    const malformedRoot = { ...machine("/MalformedRoot"), nodes: [] };
    const malformedChild = { ...machine("/MalformedChild", "/Environment"), nodes: [] };

    assert.deepEqual(
      environmentRootGraphs([environment, explicitRoot, child, grandchild, malformedRoot, malformedChild])
        .map((graph) => graph.name),
      ["/Environment", "/ExplicitRoot"],
    );
    assert.deepEqual(
      [...machineNamesInOwnedSubtree(
        [environment, explicitRoot, child, grandchild, malformedRoot, malformedChild],
        "/Environment",
      )],
      ["/Environment", "/Child", "/Grandchild"],
    );
  });

  test("keeps visible descendants nested under root-only hidden owner shells", () => {
    const environment = machine("/Environment", null);
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
    assert.equal(graphs[0]?.owner, null);
    assert.equal(graphs[1]?.owner, "/Environment");
    assert.equal(graphs[1]?.componentName, "Service");
    assert.deepEqual(graphs[1]?.nodes, [{ path: "/Service", parent: null, label: "Service" }]);
    assert.equal(graphs[2], child);
  });

  test("does not include hidden shells when no visible descendant needs them", () => {
    const owner = machine("/Environment", null);
    const child = machine("/Child", "/Environment");

    assert.deepEqual(graphsForVisibility([owner, child], new Map([["/Environment", false], ["/Child", false]])), []);
    assert.deepEqual(graphsForVisibility([owner, child], new Map([["/Environment", true], ["/Child", false]])).map((graph) => graph.name), ["/Environment"]);
  });

  test("treats explicit null ownership as an environment root", () => {
    const environment = machine("/Environment", null);

    assert.deepEqual(
      graphsForVisibility([environment], new Map([["/Environment", true]])).map((graph) => graph.name),
      ["/Environment"],
    );
  });

  test("excludes observed-only ownerless graphs even when visible", () => {
    const observedOnly = machine("/Ability");

    assert.deepEqual(environmentRootGraphs([observedOnly]), []);
    assert.deepEqual(
      [...machineNamesInOwnedSubtree([observedOnly], "/Ability")],
      [],
    );
    assert.deepEqual(graphsForVisibility([observedOnly], new Map([["/Ability", true]])), []);
    assert.deepEqual(environmentWorkspaceGraphs([observedOnly]), []);
  });

  test("requires ownership chains to terminate at an explicit null root", () => {
    const root = machine("/Environment", null);
    const child = machine("/Child", "/Environment");
    const ownerless = machine("/Ownerless");

    assert.deepEqual(
      graphsForVisibility(
        [root, child, ownerless],
        new Map([["/Environment", false], ["/Child", true], ["/Ownerless", true]]),
      ).map((graph) => graph.name),
      ["/Environment", "/Child"],
    );
  });

  test("excludes visible models whose owner is missing", () => {
    const missingOwner = machine("/Service", "/MissingOwner");

    assert.deepEqual(graphsForVisibility([missingOwner], new Map([["/Service", true]])), []);
  });

  test("excludes visible children whose owner has no root", () => {
    const owner = { ...machine("/Environment", null), nodes: [] };
    const child = machine("/Child", "/Environment");

    assert.deepEqual(
      graphsForVisibility(
        [owner, child],
        new Map([
          ["/Environment", false],
          ["/Child", true],
        ]),
      ),
      [],
    );
  });

  test("excludes visible children with no root", () => {
    const owner = machine("/Environment", null);
    const child = {
      ...machine("/Child", "/Environment"),
      nodes: [{ path: "/Child/ready", parent: "/Child", label: "ready" }],
    };

    assert.deepEqual(
      graphsForVisibility(
        [owner, child],
        new Map([
          ["/Environment", false],
          ["/Child", true],
        ]),
      ),
      [],
    );
  });

  test("excludes visible models in ownership cycles", () => {
    const first = machine("/First", "/Second");
    const second = machine("/Second", "/First");

    assert.deepEqual(
      graphsForVisibility(
        [first, second],
        new Map([
          ["/First", true],
          ["/Second", true],
        ]),
      ),
      [],
    );
  });
});
