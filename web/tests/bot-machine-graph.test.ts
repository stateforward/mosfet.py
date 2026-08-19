import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  CANVAS_FILL,
  compoundTitleStyle,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_FILL,
  INITIAL_SIZE,
  NOW_BORDER,
  NOW_FILL,
  NOW_INK,
  initialPosition,
  initialTargets,
  nodeClasses,
  nodeLabel,
  graphNodeStyle,
  graphEdgeIdentity,
  isRenderableGraphEdge,
  loopAnchorIdentity,
  machineKey,
  namespacedPath,
  structureKey,
} from "../src/machine-graph-view.ts";
import {
  environmentBoxPositions,
  machineOwnerIndex,
  ownershipLayout,
  renderableGraphs,
} from "../src/machine-graph-layout.ts";
import { foldMachines, graphFromPublishedModel } from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

describe("machine graph now theme and UML initial", () => {
  const model = (name: string, owner?: string | null) => ({
    name,
    ...(owner === undefined ? {} : { owner }),
    componentName: name.slice(1),
    currentState: `${name}/ready`,
    lastEventName: "",
    observationCount: 1,
    nodes: [
      { path: name, parent: null, label: name.slice(1) },
      { path: `${name}/ready`, parent: name, label: "ready" },
    ],
    edges: [],
  });

  test("owned model roots use the owner's namespaced compound id", () => {
    const owner = model("/Phone");
    const child = model("/PhoneService", "/Phone");
    const graphs = [owner, child];
    assert.equal(machineOwnerIndex(graphs, 1), 0);
    const ownerId = namespacedPath(machineKey(owner, 0), owner.name);
    assert.equal(ownershipLayout(graphs).childrenByIndex.get(0)?.[0], 1);
    assert.equal(ownerId, "machine:0:%2FPhone:%2FPhone");
    assert.notEqual(structureKey(graphs), structureKey([owner, model("/PhoneService")]));
  });

  test("duplicate transition occurrences receive distinct edge and loop identities", () => {
    const edge = { source: "/Machine/a", target: "/Machine/b", eventName: "go" };
    assert.notEqual(graphEdgeIdentity("machine:0", edge, 0), graphEdgeIdentity("machine:0", edge, 1));
    assert.notEqual(loopAnchorIdentity("machine:0", edge, 0), loopAnchorIdentity("machine:0", edge, 1));
  });

  test("invalid transition targets are excluded from rendered edge paths", () => {
    const knownPaths = new Set(["/Ability", "/Ability/ready"]);
    assert.equal(
      isRenderableGraphEdge(
        { source: "/Ability/ready", target: "", eventName: "invalid" },
        knownPaths,
      ),
      false,
    );
    assert.equal(
      isRenderableGraphEdge(
        { source: "/Ability/ready", target: "/Ability", eventName: "valid" },
        knownPaths,
      ),
      true,
    );
  });

  test("owned layout keeps the child graph inside the owner's combined box", () => {
    const owner = model("/Phone");
    const child = model("/PhoneService", "/Phone");
    const graphs = [owner, child];
    const positions = environmentBoxPositions(graphs);
    const topLevelPositions = environmentBoxPositions([owner, model("/PhoneService")]);
    const ownerPosition = positions.get(namespacedPath(machineKey(owner, 0), owner.name));
    const childPosition = positions.get(namespacedPath(machineKey(child, 1), child.name));
    assert.ok(ownerPosition !== undefined);
    assert.ok(childPosition !== undefined);
    const topLevelOwner = topLevelPositions.get(namespacedPath(machineKey(owner, 0), owner.name));
    const topLevelChild = topLevelPositions.get(namespacedPath(machineKey(child, 1), child.name));
    assert.ok(topLevelOwner !== undefined);
    assert.ok(topLevelChild !== undefined);
    assert.ok(ownerPosition.x > topLevelOwner.x);
    assert.ok(childPosition.x < topLevelChild.x);
  });

  test("missing owners and ownership cycles remain top-level", () => {
    const missing = [model("/PhoneService", "/Phone")];
    assert.equal(machineOwnerIndex(missing, 0), null);
    assert.equal(machineOwnerIndex([model("/Phone"), model("/PhoneService", null)], 1), null);
    const cycle = [model("/Phone", "/PhoneService"), model("/PhoneService", "/Phone")];
    assert.deepEqual([...ownershipLayout(cycle).ownerByIndex], []);
  });

  test("direct renderer admission skips unresolved and cyclic ownership graphs", () => {
    const owner = model("/Phone");
    const child = model("/PhoneService", "/Phone");
    const unresolved = model("/MissingOwnerChild", "/MissingOwner");
    const firstCycle = model("/FirstCycle", "/SecondCycle");
    const secondCycle = model("/SecondCycle", "/FirstCycle");
    const graphs = [unresolved, owner, child, firstCycle, secondCycle];

    assert.deepEqual(
      renderableGraphs(graphs).map((graph) => graph.name),
      ["/Phone", "/PhoneService"],
    );
    const positions = environmentBoxPositions(graphs);
    assert.ok([...positions.keys()].some((key) => key.includes(encodeURIComponent("/Phone"))));
    assert.ok([...positions.keys()].some((key) => key.includes(encodeURIComponent("/PhoneService"))));
    assert.ok(![...positions.keys()].some((key) => key.includes(encodeURIComponent("/MissingOwnerChild"))));
    assert.ok(![...positions.keys()].some((key) => key.includes(encodeURIComponent("/FirstCycle"))));
    assert.ok(![...positions.keys()].some((key) => key.includes(encodeURIComponent("/SecondCycle"))));
  });

  test("compound titles have a neutral padded backdrop that clears the border", () => {
    const title = compoundTitleStyle();
    assert.equal(title.backgroundColor, "#161b22");
    assert.equal(title.backgroundOpacity, 1);
    assert.equal(title.padding, 3);
    assert.equal(title.marginY, 8);
  });

  test("active graph states use neutral interiors and outline emphasis", () => {
    const active = graphNodeStyle("active-path");
    const current = graphNodeStyle("current");

    assert.equal(active.backgroundColor, "#161b22");
    assert.equal(active.backgroundOpacity, 1);
    assert.equal(active.borderColor, NOW_BORDER);
    assert.ok(active.borderWidth >= 2);
    assert.equal(current.backgroundColor, "#161b22");
    assert.equal(current.backgroundOpacity, 1);
    assert.equal(current.borderColor, NOW_BORDER);
    assert.ok(current.borderWidth > active.borderWidth);
    assert.equal(current.underlayOpacity, 0);
  });

  test("now paint stays in the teal charcoal family", () => {
    assert.equal(NOW_FILL, "#14b8a6");
    assert.equal(NOW_BORDER, "#2dd4bf");
    assert.equal(NOW_INK, "#042f2e");
    assert.doesNotMatch(NOW_FILL, /fbbf24|f59e0b/i);
    assert.doesNotMatch(NOW_BORDER, /fbbf24|f59e0b/i);
  });

  test("initial circle stays near-black with a contrasting ring, not fill-equals-canvas", () => {
    assert.match(INITIAL_FILL, /^#0{6}$|^#0a0a0a$/i);
    assert.match(INITIAL_BORDER, /^#e8eaef$|^#2dd4bf$/i);
    assert.equal(INITIAL_BORDER_WIDTH, 2);
    assert.notEqual(INITIAL_BORDER.toLowerCase(), CANVAS_FILL.toLowerCase());
    assert.notEqual(INITIAL_BORDER.toLowerCase(), INITIAL_FILL.toLowerCase());
    assert.ok(INITIAL_SIZE >= 16 && INITIAL_SIZE <= 18);
    assert.doesNotMatch(INITIAL_FILL, /fbbf24|f59e0b/i);
    assert.doesNotMatch(INITIAL_BORDER, /fbbf24|f59e0b/i);
  });

  test("only the exact current leaf is current; ancestors are active-path", () => {
    const current = "/PhoneBot/active/processing";
    const active = new Set(["/PhoneBot", "/PhoneBot/active", "/PhoneBot/active/processing"]);
    assert.equal(nodeClasses(current, current, active), "state current");
    assert.equal(nodeClasses("/PhoneBot", current, active), "state active-path");
    assert.equal(nodeClasses("/PhoneBot/active", current, active), "state active-path");
    assert.match(nodeLabel("processing", current, current), /●/);
    assert.equal(nodeLabel("PhoneBot", "/PhoneBot", current), "PhoneBot");
  });

  test("PhoneBot fixture gets one root initial and does not invent nested circles", () => {
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    const machines = foldMachines(parsed.spans);
    const phoneBot = machines.find((item) => item.name === "/PhoneBot");
    const phone = machines.find((item) => item.name === "/Phone");
    assert.ok(phoneBot !== undefined);
    assert.ok(phone !== undefined);
    assert.deepEqual(initialTargets(phoneBot), ["/PhoneBot"]);
    assert.deepEqual(initialTargets(phone), ["/Phone"]);
    assert.ok(!initialTargets(phoneBot).includes("/PhoneBot/active"));
    assert.ok(!initialTargets(phoneBot).includes("/PhoneBot/inactive"));
  });

  test("an incoming hsm/initial edge is the only circle when present", () => {
    const targets = initialTargets({
      name: "/Machine",
      componentName: "Machine",
      currentState: "/Machine/idle",
      lastEventName: "hsm/initial",
      observationCount: 1,
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/idle", parent: "/Machine", label: "idle" },
        { path: "/Machine/busy", parent: "/Machine", label: "busy" },
      ],
      edges: [
        {
          source: "/Machine/.initial",
          target: "/Machine/idle",
          eventName: "hsm/initial",
          count: 1,
          lastFired: true,
        },
      ],
    });
    assert.deepEqual(targets, ["/Machine/idle"]);
  });

  test("a region entered only from a real source does not invent a circle", () => {
    const targets = initialTargets({
      name: "/Machine",
      componentName: "Machine",
      currentState: "/Machine/outer/inner",
      lastEventName: "go",
      observationCount: 2,
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/outer", parent: "/Machine", label: "outer" },
        { path: "/Machine/outer/inner", parent: "/Machine/outer", label: "inner" },
      ],
      edges: [
        {
          source: "/Machine",
          target: "/Machine/outer/inner",
          eventName: "go",
          count: 1,
          lastFired: true,
        },
      ],
    });
    assert.deepEqual(targets, ["/Machine"]);
  });

  test("an invalid transition endpoint does not invent an initial circle", () => {
    const graph = {
      name: "/Machine",
      componentName: "Machine",
      currentState: "",
      lastEventName: "go",
      observationCount: 1,
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/idle", parent: "/Machine", label: "idle" },
      ],
      edges: [
        {
          source: "/Machine/idle",
          target: "",
          eventName: "go",
          count: 1,
          lastFired: true,
        },
      ],
    };
    assert.deepEqual(initialTargets(graph), []);
  });

  test("published topology still draws a UML initial onto idle before any visit", () => {
    const graph = graphFromPublishedModel({
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [
        { qualified_name: "/Demo", parent: "/", initial: "/Demo/.initial" },
        { qualified_name: "/Demo/idle", parent: "/Demo", initial: "" },
        { qualified_name: "/Demo/run", parent: "/Demo", initial: "" },
      ],
      transitions: [{ source: "/Demo/.initial", target: "/Demo/idle", events: ["hsm/initial"] }],
    });
    assert.deepEqual(initialTargets(graph), ["/Demo/idle"]);
    assert.equal(nodeClasses("/Demo/idle", "", new Set()), "state");
    assert.equal(nodeClasses("/Demo/run", "/Demo/run", new Set(["/Demo", "/Demo/run"])), "state current");
  });

  test("initial circle sits left of a leaf on the shared horizontal axis", () => {
    const point = initialPosition({ x: 100, y: 40 }, { width: 120, height: 40 });
    assert.equal(point.y, 40);
    assert.ok(point.x < 100 - 60);
  });
});
