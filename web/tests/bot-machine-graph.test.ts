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
  INITIAL_EVENT,
  INITIAL_FILL,
  INITIAL_SIZE,
  NOW_BORDER,
  NOW_FILL,
  NOW_INK,
  STATE_NODE_SIZE,
  initialPosition,
  initialTargets,
  nodeClasses,
  nodeLabel,
  graphNodeStyle,
  stateNodeStyle,
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
  measureState,
} from "../src/machine-graph-layout.ts";
import { foldMachines, graphFromPublishedModel, type MachineEdge, type MachineGraph } from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import * as hsm from "../src/hsm.ts";
import { flowModelFromGraphs } from "../src/elements/bot-machine-graph/flow-model.ts";
import { Routes, type SyncData } from "../src/flow/pathing/routes.ts";
import type { XYPosition } from "../src/flow/types.ts";

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

  test("independent machine roots remain horizontally separated", () => {
    const alice = model("/Alice");
    const bob = model("/Bob");
    const positions = environmentBoxPositions([alice, bob]);
    const alicePosition = positions.get(namespacedPath(machineKey(alice, 0), alice.name));
    const bobPosition = positions.get(namespacedPath(machineKey(bob, 1), bob.name));
    assert.ok(alicePosition !== undefined);
    assert.ok(bobPosition !== undefined);
    assert.ok(alicePosition.x < bobPosition.x);
    assert.equal(alicePosition.y, bobPosition.y);
  });

  test("leaf state geometry is square and balanced siblings use compact rows", () => {
    const graph = {
      ...model("/Machine"),
      nodes: [
        { path: "/Machine", parent: null, label: "Machine" },
        { path: "/Machine/one", parent: "/Machine", label: "one" },
        { path: "/Machine/two", parent: "/Machine", label: "two" },
        { path: "/Machine/three", parent: "/Machine", label: "three" },
        { path: "/Machine/four", parent: "/Machine", label: "four" },
      ],
    };
    const ownership = ownershipLayout([graph]);
    const leaf = measureState([graph], ownership, 0, "/Machine/one");
    const style = stateNodeStyle();
    assert.equal(style.width, style.height);
    assert.equal(style.width, STATE_NODE_SIZE);
    assert.equal(leaf.width, STATE_NODE_SIZE);
    assert.equal(leaf.height, STATE_NODE_SIZE);
    const positions = environmentBoxPositions([graph]);
    const machine = machineKey(graph, 0);
    const one = positions.get(namespacedPath(machine, "/Machine/one"));
    const two = positions.get(namespacedPath(machine, "/Machine/two"));
    const three = positions.get(namespacedPath(machine, "/Machine/three"));
    const four = positions.get(namespacedPath(machine, "/Machine/four"));
    assert.ok(one !== undefined);
    assert.ok(two !== undefined);
    assert.ok(three !== undefined);
    assert.ok(four !== undefined);
    assert.equal(one.y, two.y);
    assert.equal(three.y, four.y);
    assert.ok(one.x < two.x);
    assert.ok(one.y < three.y);
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

describe("machine graph flow model edge cutover", () => {
  const MACHINE = "/Machine";
  const leafPath = (leaf: string): string => `${MACHINE}/${leaf}`;
  const machineOf = (graph: MachineGraph): string => machineKey(graph, 0);

  function leafGraph(
    leaves: readonly string[],
    edges: readonly MachineEdge[],
    currentState = leafPath("b"),
  ): MachineGraph {
    return {
      name: MACHINE,
      componentName: "Machine",
      currentState,
      lastEventName: "",
      observationCount: 1,
      nodes: [
        { path: MACHINE, parent: null, label: "Machine" },
        ...leaves.map((leaf) => ({ path: leafPath(leaf), parent: MACHINE, label: leaf })),
      ],
      edges: [...edges],
    };
  }

  const edgeInputOf = (edge: MachineEdge): Pick<MachineEdge, "source" | "target" | "eventName"> => ({
    source: edge.source,
    target: edge.target,
    eventName: edge.eventName,
  });

  function cableEdgesOf(model: ReturnType<typeof flowModelFromGraphs>) {
    return model.edges.filter((edge) => edge.type === "cable");
  }

  test("taxi and aligned transitions ship as engine-routed cables without precomputed paths", () => {
    const longEvent = "r".repeat(31);
    const alignedEdge: MachineEdge = {
      source: leafPath("a"),
      target: leafPath("b"),
      eventName: "sibling_hop",
      count: 3,
      lastFired: false,
    };
    const taxiEdge: MachineEdge = {
      source: leafPath("b"),
      target: leafPath("c"),
      eventName: longEvent,
      count: 1,
      lastFired: true,
    };
    const graph = leafGraph(["a", "b", "c", "d"], [alignedEdge, taxiEdge]);
    const model = flowModelFromGraphs([graph]);
    const cables = cableEdgesOf(model);
    assert.equal(cables.length, 2);

    // Row siblings share the layout axis: aligned.
    const aligned = cables.find((edge) => edge.data?.["eventName"] === "sibling_hop");
    assert.ok(aligned !== undefined);
    assert.equal(aligned.type, "cable");
    assert.ok(aligned.data !== undefined);
    assert.equal("d" in aligned.data, false);
    assert.equal("labelPosition" in aligned.data, false);
    assert.equal(aligned.className, "aligned");
    assert.equal(aligned.label, "sibling_hop");
    assert.equal(aligned.id, graphEdgeIdentity(machineOf(graph), edgeInputOf(alignedEdge), 0));

    // Diagonal pair across rows and columns: taxi, last-fired emphasis kept.
    const taxi = cables.find((edge) => edge.data?.["eventName"] === longEvent);
    assert.ok(taxi !== undefined);
    assert.equal(taxi.type, "cable");
    assert.ok(taxi.data !== undefined);
    assert.equal("d" in taxi.data, false);
    assert.equal(taxi.className, "taxi last-fired");
    assert.deepEqual(taxi.data["lastFired"], true);
    assert.equal(taxi.label, `${longEvent.slice(0, 29)}…`);
    assert.equal(taxi.id, graphEdgeIdentity(machineOf(graph), edgeInputOf(taxiEdge), 0));
  });

  test("loop-class transitions keep their dedicated precomputed stub shapes", () => {
    const edges: readonly MachineEdge[] = [
      { source: leafPath("c"), target: leafPath("c"), eventName: "self_loop", count: 2, lastFired: false },
      { source: MACHINE, target: leafPath("a"), eventName: "descend_loop", count: 1, lastFired: true },
    ];
    const graph = leafGraph(["a", "b", "c", "d"], [...edges]);
    const model = flowModelFromGraphs([graph]);
    const loops = model.edges.filter((edge) => edge.className?.startsWith("loop"));
    assert.equal(loops.length, 2);
    for (const loop of loops) {
      assert.equal(loop.type, "step");
      assert.ok(loop.data !== undefined);
      const d = loop.data["d"];
      assert.ok(typeof d === "string");
      assert.match(d, /^M /);
      assert.match(d, /Q /);
      const labelAt = loop.data["labelPosition"];
      assert.ok(hsm.isRecord(labelAt));
      assert.equal(typeof labelAt["x"], "number");
      assert.equal(typeof labelAt["y"], "number");
    }
    const selfLoop = loops.find((edge) => edge.data?.["eventName"] === "self_loop");
    assert.equal(selfLoop?.className, "loop");
    const descend = loops.find((edge) => edge.data?.["eventName"] === "descend_loop");
    assert.equal(descend?.className, "loop last-fired");
  });

  test("initial arrows stay straight precomputed shapes", () => {
    const graph = leafGraph(["a", "b"], [
      { source: leafPath("a"), target: leafPath("b"), eventName: "go", count: 1, lastFired: false },
    ]);
    const model = flowModelFromGraphs([graph]);
    const initial = model.edges.find((edge) => edge.className === "initial");
    assert.ok(initial !== undefined);
    assert.equal(initial.type, "straight");
    assert.ok(initial.data !== undefined);
    const d = initial.data["d"];
    assert.ok(typeof d === "string");
    assert.match(d, /^M /);
    assert.equal(initial.data["eventName"], INITIAL_EVENT);
  });

  function routedPayloadOf(value: unknown): Record<string, readonly XYPosition[]> | null {
    if (!hsm.isRecord(value) || !hsm.isRecord(value["routes"])) return null;
    const routes: Record<string, readonly XYPosition[]> = {};
    for (const [id, points] of Object.entries(value["routes"])) {
      if (!Array.isArray(points)) return null;
      const waypoints: XYPosition[] = [];
      for (const point of points) {
        if (!hsm.isRecord(point) || typeof point["x"] !== "number" || typeof point["y"] !== "number") return null;
        waypoints.push({ x: point["x"], y: point["y"] });
      }
      routes[id] = waypoints;
    }
    return routes;
  }

  async function routedRoutesOf(sync: SyncData): Promise<Record<string, readonly XYPosition[]>> {
    const machine = hsm.start({ instance: new Routes(), model: Routes.model });
    const inner = machine.dispatch.bind(machine);
    const routed: Array<Record<string, readonly XYPosition[]>> = [];
    machine.dispatch = ((event: hsm.DispatchEvent) => {
      const payload = routedPayloadOf(event.data);
      if (event.name === Routes.routedEvent.name && payload !== null) routed.push(payload);
      return inner(event);
    }) as Routes["dispatch"];
    try {
      void machine.dispatch(hsm.typedEvent({ event: Routes.syncEvent, data: sync })).catch(hsm.catchFailure());
      for (let attempt = 0; attempt < 50 && routed.length === 0; attempt += 1) {
        await new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 0);
        });
      }
      assert.ok(routed.length >= 1, "timed out waiting for routed routes");
      return routed[0] ?? {};
    } finally {
      await hsm.stop(machine);
    }
  }

  type WorldRect = { x: number; y: number; width: number; height: number };

  function assertSegmentsAvoid(points: readonly XYPosition[], rect: WorldRect, margin: number): void {
    const left = rect.x - margin;
    const right = rect.x + rect.width + margin;
    const top = rect.y - margin;
    const bottom = rect.y + rect.height + margin;
    for (let index = 1; index < points.length; index += 1) {
      const start = points[index - 1];
      const end = points[index];
      assert.ok(start !== undefined && end !== undefined);
      assert.ok(
        start.x === end.x || start.y === end.y,
        `cable segment ${JSON.stringify(start)}->${JSON.stringify(end)} is not orthogonal`,
      );
      const crosses = start.x === end.x
        ? start.x > left && start.x < right && Math.max(start.y, end.y) > top && Math.min(start.y, end.y) < bottom
        : start.y > top && start.y < bottom && Math.max(start.x, end.x) > left && Math.min(start.x, end.x) < right;
      assert.ok(!crosses, `cable segment crosses the intermediate node rect at ${JSON.stringify([start, end])}`);
    }
  }

  test("a transition forced around an intermediate node paints a polyline avoiding its rect", async () => {
    // Five root children force a three-column layout: `left` | `wide` | `right`
    // on one row, so the straight pin line between the inner leaves crosses
    // `wide`. The routing snapshot carries EVERY laid-out rect, machine shells
    // included: the shell transitively contains both endpoints, so Routes
    // excludes it for this edge as a room -- including it must still yield the
    // avoiding polyline (that is exactly what the parentId stamp buys).
    const graph: MachineGraph = {
      name: MACHINE,
      componentName: "Machine",
      currentState: "",
      lastEventName: "",
      observationCount: 1,
      nodes: [
        { path: MACHINE, parent: null, label: "Machine" },
        { path: leafPath("left"), parent: MACHINE, label: "left" },
        { path: leafPath("left/main"), parent: leafPath("left"), label: "main" },
        { path: leafPath("wide"), parent: MACHINE, label: "wide" },
        { path: leafPath("wide/w1"), parent: leafPath("wide"), label: "w1" },
        { path: leafPath("wide/w2"), parent: leafPath("wide"), label: "w2" },
        { path: leafPath("wide/w3"), parent: leafPath("wide"), label: "w3" },
        { path: leafPath("right"), parent: MACHINE, label: "right" },
        { path: leafPath("right/main"), parent: leafPath("right"), label: "main" },
        { path: leafPath("z1"), parent: MACHINE, label: "z1" },
        { path: leafPath("z2"), parent: MACHINE, label: "z2" },
      ],
      edges: [{ source: leafPath("left/main"), target: leafPath("right/main"), eventName: "cross", count: 1, lastFired: false }],
    };
    const model = flowModelFromGraphs([graph]);
    const cross = cableEdgesOf(model).find((edge) => edge.data?.["eventName"] === "cross");
    assert.ok(cross !== undefined);
    assert.equal(cross.type, "cable");

    const worldRectOf = (leaf: string): WorldRect => {
      const node = model.nodes.find((item) => item.id === namespacedPath(machineOf(graph), leafPath(leaf)));
      assert.ok(node !== undefined, `missing layout node for ${leaf}`);
      return {
        x: node.position.x,
        y: node.position.y,
        width: node.width ?? 0,
        height: node.height ?? 0,
      };
    };

    const routes = await routedRoutesOf({
      nodes: model.nodes.map((node) => ({
        id: node.id,
        x: node.position.x,
        y: node.position.y,
        width: node.width ?? 0,
        height: node.height ?? 0,
        ...(node.parentId === undefined ? {} : { parentId: node.parentId }),
      })),
      edges: [{ id: cross.id, source: cross.source, target: cross.target }],
      draggingNodeIds: [],
    });
    const points = routes[cross.id];
    assert.ok(points !== undefined && points.length >= 4, `expected a detour polyline, got ${JSON.stringify(points)}`);

    const fromLeft = worldRectOf("left/main");
    const toRight = worldRectOf("right/main");
    const first = points[0];
    const last = points[points.length - 1];
    assert.ok(first !== undefined && last !== undefined);
    assert.deepEqual(first, { x: fromLeft.x + fromLeft.width, y: fromLeft.y + fromLeft.height / 2 });
    assert.deepEqual(last, { x: toRight.x, y: toRight.y + toRight.height / 2 });

    // The router's hard margin: lanes may ride the inflated edge, never graze
    // the drawn intermediate node (mirrors router HARD_MARGIN).
    assertSegmentsAvoid(points, worldRectOf("wide"), 4);
  });
});
