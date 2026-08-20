import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  documentFromOtlp,
  documentFromSpans,
  foldMachines,
  graphFromPublishedModel,
  mergePublishedModel,
  overlayObserve,
  parseLiveModel,
  parseMachineGraph,
  parsePublishedModel,
  type PublishedModel,
} from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { graphsForVisibility } from "../src/dashboard-graphs.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

describe("otel observe fold", () => {
  test("OTLP ExportTraceServiceRequest observe spans become a machine graph", () => {
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    assert.ok(parsed.spans.length >= 1);
    const last = parsed.spans[parsed.spans.length - 1];
    assert.ok(last !== undefined);
    const machines = foldMachines(parsed.spans);
    const machine = machines.find((item) => item.name === last.attributes["hsm.machine.name"]);
    assert.ok(machine !== undefined);
    assert.equal(machine.currentState, last.attributes["hsm.machine.state"]);
    assert.equal(machine.name, "/PhoneBot");
    assert.equal(machine.currentState, "/PhoneBot/active/processing");
    assert.equal(machine.lastEventName, "bot.processing.completed");
    assert.ok(machine.nodes.some((node) => node.path === "/PhoneBot/active"));
    assert.ok(machine.nodes.some((node) => node.parent === "/PhoneBot"));
    assert.ok(machine.edges.some((edge) => edge.eventName === "bot.activate"));
    const lastEdge = machine.edges.find((edge) => edge.lastFired);
    assert.ok(lastEdge !== undefined);
    assert.equal(lastEdge.target, "/PhoneBot/active/processing");
    assert.equal(lastEdge.eventName, "bot.processing.completed");
    const document = documentFromOtlp(request);
    assert.ok(document !== null);
    assert.equal(document.observeCount, parsed.spans.length);
  });

  test("skips non-observe and invalid OTLP spans", () => {
    const request = {
      resourceSpans: [
        {
          scopeSpans: [
            {
              spans: [
                { name: "bot.reasoning.stage", attributes: [] },
                {
                  name: "bot.hsm.observe",
                  attributes: [{ key: "hsm.machine.name", value: { stringValue: "" } }],
                },
              ],
            },
          ],
        },
      ],
    };
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    assert.equal(parsed.spans.length, 0);
    assert.equal(parsed.skipped, 2);
    const document = documentFromOtlp(request);
    assert.ok(document !== null);
    assert.equal(document.machines.length, 0);
    assert.equal(document.selectedMachine, null);
  });

  test("rejects a malformed ExportTraceServiceRequest", () => {
    assert.equal(parseExportTraceServiceRequest("not-json-object"), null);
    assert.equal(parseExportTraceServiceRequest({ resourceSpans: "nope" }), null);
    assert.equal(documentFromOtlp({ resourceSpans: "nope" }), null);
  });

  test("observed-only ownerless graphs are not admitted to the workspace", () => {
    const document = documentFromSpans([
      {
        name: "bot.hsm.observe",
        timestamp: "2026-08-16T00:00:00.000000000Z",
        start_time: "2026-08-16T00:00:00.000000000Z",
        attributes: {
          "hsm.machine.name": "/Ability",
          "hsm.machine.state": "/Ability/ready",
          "bot.component.name": "Ability",
          "hsm.event.name": "hsm/initial",
          "hsm.event.kind": "event",
          "hsm.observation.occurrence": "event",
          "bot.outcome": "observed",
        },
      },
    ], 0);

    assert.equal(document.machines[0]?.owner, undefined);
    assert.deepEqual(
      graphsForVisibility(document.machines, new Map([["/Ability", true]])),
      [],
    );
  });
});

const demoModel: PublishedModel = {
  name: "/Demo",
  initial: "/Demo/.initial",
  states: [
    { qualified_name: "/Demo", parent: "/", initial: "/Demo/.initial" },
    { qualified_name: "/Demo/idle", parent: "/Demo", initial: "" },
    { qualified_name: "/Demo/run", parent: "/Demo", initial: "" },
  ],
  transitions: [
    { source: "/Demo/.initial", target: "/Demo/idle", events: ["hsm/initial"] },
    { source: "/Demo/idle", target: "/Demo/run", events: ["go"] },
  ],
};

describe("published model topology", () => {
  test("rejects explicitly present malformed ownership fields", () => {
    const malformedOwners: unknown[] = ["", 0, false, {}, [], undefined];
    for (const owner of malformedOwners) {
      assert.equal(
        parsePublishedModel({ ...demoModel, owner }),
        null,
        `published owner ${String(owner)} should be rejected`,
      );
      assert.equal(
        parseLiveModel({
          name: "/PhoneService",
          component: "PhoneService",
          state: "",
          live: true,
          owner,
        }),
        null,
        `live owner ${String(owner)} should be rejected`,
      );
      assert.equal(
        parseMachineGraph({
          name: "/PhoneService",
          componentName: "PhoneService",
          currentState: "",
          lastEventName: "",
          observationCount: 0,
          nodes: [{ path: "/PhoneService", parent: null, label: "PhoneService" }],
          edges: [],
          owner,
        }),
        null,
        `graph owner ${String(owner)} should be rejected`,
      );
    }
  });

  test("distinguishes missing ownership from explicit ownerless ownership", () => {
    const missing = parsePublishedModel(demoModel);
    const explicitNull = parsePublishedModel({ ...demoModel, owner: null });
    assert.ok(missing !== null);
    assert.ok(explicitNull !== null);
    assert.equal(Object.hasOwn(missing, "owner"), false);
    assert.equal(explicitNull.owner, null);
  });

  test("published model renders idle and run with no observe spans", () => {
    const parsed = parsePublishedModel(demoModel);
    assert.ok(parsed !== null);
    const graph = graphFromPublishedModel(parsed);
    assert.equal(graph.currentState, "");
    assert.ok(graph.nodes.some((node) => node.path === "/Demo/idle"));
    assert.ok(graph.nodes.some((node) => node.path === "/Demo/run"));
    assert.ok(graph.edges.some((edge) => edge.eventName === "go" && edge.source === "/Demo/idle"));
    const document = documentFromSpans([], 0, null, [parsed]);
    assert.equal(document.machines.length, 1);
    assert.equal(document.machines[0]?.currentState, "");
    assert.equal(document.observeCount, 0);
  });

  test("a live payload on a published model sets current leaf without observe spans", () => {
    const document = documentFromSpans([], 0, null, [
      { ...demoModel, live: true, state: "/Demo/idle", component: "Demo" },
    ]);
    assert.equal(document.machines.length, 1);
    assert.equal(document.machines[0]?.currentState, "/Demo/idle");
    assert.equal(document.machines[0]?.componentName, "Demo");
    assert.ok(document.machines[0]?.nodes.some((node) => node.path === "/Demo/run"));
    assert.equal(document.observeCount, 0);
  });

  test("published and live ownership survives graph creation and overlays", () => {
    const service = parsePublishedModel({
      name: "/PhoneService",
      owner: "/Phone",
      initial: "/PhoneService/.initial",
      states: [{ qualified_name: "/PhoneService/ready", parent: "/PhoneService", initial: "" }],
      transitions: [],
      live: true,
      state: "/PhoneService/ready",
      component: "PhoneService",
    });
    assert.ok(service !== null);
    const graph = graphFromPublishedModel(service);
    assert.equal(graph.owner, "/Phone");
    assert.equal(overlayObserve(graph, []).owner, "/Phone");
    const document = documentFromSpans([], 0, null, [service]);
    assert.equal(document.machines[0]?.owner, "/Phone");
  });

  test("explicit null ownership survives parsing and overlays as a clear", () => {
    const service = parsePublishedModel({
      name: "/PhoneService",
      owner: null,
      initial: "/PhoneService/.initial",
      states: [],
      transitions: [],
    });
    const live = parseLiveModel({
      name: "/PhoneService",
      component: "PhoneService",
      state: "",
      live: true,
      owner: null,
    });
    assert.ok(service !== null);
    assert.ok(live !== null);
    assert.equal(service.owner, null);
    assert.equal(live.owner, null);
    assert.equal(graphFromPublishedModel(service).owner, null);
    assert.equal(overlayObserve(graphFromPublishedModel(service), []).owner, null);
  });

  test("live owner omission clears stale ownership while explicit null remains a root", () => {
    const topology = parsePublishedModel({ ...demoModel, owner: null });
    assert.ok(topology !== null);
    const omittedOwnerLive = mergePublishedModel(topology, {
      name: "/Demo",
      states: [],
      transitions: [],
      initial: "",
      live: true,
      state: "/Demo/idle",
      component: "Demo",
    });
    assert.equal(Object.hasOwn(omittedOwnerLive, "owner"), false);
    assert.deepEqual(
      graphsForVisibility(
        [graphFromPublishedModel(omittedOwnerLive)],
        new Map([["/Demo", true]]),
      ),
      [],
    );

    const explicitNullLive = mergePublishedModel(topology, {
      name: "/Demo",
      states: [],
      transitions: [],
      initial: "",
      live: true,
      state: "/Demo/idle",
      component: "Demo",
      owner: null,
    });
    assert.equal(explicitNullLive.owner, null);
    assert.deepEqual(
      graphsForVisibility(
        [graphFromPublishedModel(explicitNullLive)],
        new Map([["/Demo", true]]),
      ).map((graph) => graph.name),
      ["/Demo"],
    );
  });

  test("observe spans overlay the current leaf without dropping model states", () => {
    const graph = overlayObserve(graphFromPublishedModel(demoModel), [
      {
        name: "bot.hsm.observe",
        timestamp: "2026-08-16T00:00:00.000000000Z",
        start_time: "2026-08-16T00:00:00.000000000Z",
        attributes: {
          "hsm.machine.name": "/Demo",
          "hsm.machine.state": "/Demo/idle",
          "bot.component.name": "Demo",
          "hsm.event.name": "hsm/initial",
          "hsm.event.kind": "event",
          "hsm.observation.occurrence": "event",
          "bot.outcome": "observed",
        },
      },
      {
        name: "bot.hsm.observe",
        timestamp: "2026-08-16T00:00:01.000000000Z",
        start_time: "2026-08-16T00:00:01.000000000Z",
        attributes: {
          "hsm.machine.name": "/Demo",
          "hsm.machine.state": "/Demo/run",
          "bot.component.name": "Demo",
          "hsm.event.name": "go",
          "hsm.event.kind": "event",
          "hsm.observation.occurrence": "event",
          "bot.outcome": "observed",
        },
      },
    ]);
    assert.equal(graph.currentState, "/Demo/run");
    assert.ok(graph.nodes.some((node) => node.path === "/Demo/idle"));
    const last = graph.edges.find((edge) => edge.lastFired);
    assert.ok(last !== undefined);
    assert.equal(last.eventName, "go");
    assert.equal(last.target, "/Demo/run");
  });
});
