import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as hsm from "@stateforward/hsm.ts";

import { DashboardController, type DashboardSnapshot } from "../src/dashboard-hsm.ts";
import { Focuser } from "../src/focuser-hsm.ts";
import { From, reportHsmFailure, startMachine, stopMachine } from "../src/hsm-runtime.ts";
import { Graph } from "../src/machine-graph-hsm.ts";
import { structureKey } from "../src/machine-graph-view.ts";
import { Panner } from "../src/panner-hsm.ts";
import { Renderer } from "../src/renderer-hsm.ts";
import { documentFromOtlp } from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { streamSource, type OtelStreamHandlers, type OtelStreamSubscription } from "../src/otel/source.ts";
import { OtelSourceController } from "../src/otel-source-hsm.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 1000;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error("timed out waiting for dashboard stream update");
    }
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  }
}

function fakeWorld(): HTMLElement {
  const style = { transform: "", transformOrigin: "" };
  return {
    style,
    classList: { toggle(): void { return; } },
  } as unknown as HTMLElement;
}

function startAdmittedGraph(hooks: ConstructorParameters<typeof Graph>[0]): Graph {
  return startMachine(new Graph(hooks), Graph.model);
}

function graphFor(name: string, admitted = true) {
  return {
    name,
    componentName: name,
    currentState: `${name}/idle`,
    lastEventName: "",
    observationCount: 0,
    nodes: admitted ? [{ path: name, parent: null, label: name }] : [],
    edges: [],
  };
}

describe("companion-style HSM controllers", () => {
  test("graph structure identity includes node parent and label metadata", () => {
    const node = { path: "/Demo", parent: null, label: "Demo" };
    const graph = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [node],
      edges: [],
    };
    const parentChanged = { ...graph, nodes: [{ path: node.path, parent: "/Root", label: node.label }] };
    const labelChanged = { ...graph, nodes: [{ path: node.path, parent: node.parent, label: "Renamed" }] };
    assert.notEqual(structureKey([graph]), structureKey([parentChanged]));
    assert.notEqual(structureKey([graph]), structureKey([labelChanged]));
  });

  test("malformed and empty graph sets remain empty and clear a drawing", async () => {
    let draws = 0;
    let destroys = 0;
    const graph = startAdmittedGraph({
      onDraw: () => {
        draws += 1;
      },
      onDestroy: () => {
        destroys += 1;
      },
    });
    const valid = graphFor("/Demo");

    const empty = graph.setGraphs([]);
    assert.equal(empty.phase, "empty");
    assert.equal(empty.graphs.length, 0);
    assert.equal(draws, 0);
    const drawing = graph.setGraphs([valid]);
    assert.equal(drawing.phase, "drawing");
    assert.equal(draws, 1);
    const malformed = graph.setGraphs([{}]);
    assert.equal(malformed.phase, "empty");
    assert.equal(malformed.graphs.length, 0);
    assert.ok(destroys >= 1);
    const malformedPayload = graph.setGraphs("invalid");
    assert.equal(malformedPayload.phase, "empty");
    assert.equal(malformedPayload.graphs.length, 0);
    const redraw = graph.setGraphs([valid]);
    assert.equal(redraw.phase, "drawing");
    assert.equal(draws, 2);
    await stopMachine(graph);
  });

  test("panner writes transform synchronously and stays off the graph model", async () => {
    const applied: string[] = [];
    const world = fakeWorld();
    const panner = startMachine(new Panner(world, {
      onTransform: (transform) => {
        applied.push(world.style.transform);
        applied.push(`${transform.scale}:${transform.pan.x},${transform.pan.y}`);
      },
    }), Panner.model);
    const graph = startAdmittedGraph({
      onDraw: () => undefined,
      onDestroy: () => undefined,
    });

    const drawing = graph.setGraphs([graphFor("/Demo")]);
    assert.equal(drawing.phase, "drawing");
    assert.match(drawing.statePath, /\/drawing$/);
    panner.fit({
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      metrics: {
        width: 1000,
        height: 600,
        bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
        origin: { x: 0, y: 0 },
      },
    });
    panner.panStart({ pointerId: 1, point: { x: 10, y: 10 } });
    assert.match(panner.state(), /\/panning$/);
    assert.match(graph.state(), /\/drawing$/);
    panner.cursorMove({ pan: { x: 12, y: 8 } });
    panner.zoom({ scale: 1.1, point: { x: 20, y: 20 } });
    panner.panEnd({ pointerId: 1 });
    assert.match(panner.state(), /\/fixed$/);
    assert.ok(applied.length >= 3);
    await stopMachine(panner);
    await stopMachine(graph);
  });

  test("graph updates repaint while the viewport is panning", async () => {
    const drawn: string[][] = [];
    const world = fakeWorld();
    const panner = startMachine(new Panner(world), Panner.model);
    const graph = startAdmittedGraph({
      onDraw: (graphs) => {
        drawn.push(graphs.map((value) => value.name));
      },
      onDestroy: () => undefined,
    });

    graph.setGraphs([graphFor("/A")]);
    graph.setGraphs([graphFor("/B")]);
    panner.panStart({ pointerId: 1, point: { x: 10, y: 10 } });
    assert.match(panner.state(), /\/panning$/);
    graph.setGraphs([graphFor("/C")]);

    assert.deepEqual(drawn, [["/A"], ["/B"], ["/C"]]);
    assert.deepEqual(graph.snapshot().graphs.map((value) => value.name), ["/C"]);
    await stopMachine(panner);
    await stopMachine(graph);
  });

  test("normalized viewport intents update panner-owned transform", async () => {
    const world = fakeWorld();
    const panner = startMachine(new Panner(world), Panner.model);
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      origin: { x: 0, y: 0 },
    };

    panner.fit({ reason: "initial", bounds: metrics.bounds, metrics });
    panner.panStart({ pointerId: 1, point: { x: 10, y: 10 } });
    assert.match(panner.state(), /\/panning$/);
    panner.cursorMove({ pointerId: 1, point: { x: 30, y: 24 } });
    panner.panEnd({ pointerId: 1 });
    panner.zoom({ deltaY: -100, point: { x: 30, y: 24 } });
    panner.fit({ bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics });
    assert.deepEqual(panner.transform, { scale: 1.2, pan: { x: 442.4, y: 242.4 } });
    await stopMachine(panner);
  });

  test("node viewport focus uses exact bounds and stays focused across resize", async () => {
    const world = fakeWorld();
    const panner = startMachine(new Panner(world), Panner.model);
    let focusKind = "";
    let focusPath = "";
    const focuser = startMachine(new Focuser(null, {
      onFocus: (target) => {
        focusKind = target.kind;
        focusPath = target.nodePath ?? "";
        panner.fit({
          bounds: target.bounds,
          metrics: {
            width: 1000,
            height: 600,
            bounds: { left: 0, right: 400, top: 0, bottom: 300 },
            origin: { x: 0, y: 0 },
          },
        });
      },
      onClear: () => {
        focusKind = "";
        focusPath = "";
      },
    }), Focuser.model);
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 400, top: 0, bottom: 300 },
      origin: { x: 0, y: 0 },
    };

    panner.fit({ reason: "initial", bounds: metrics.bounds, metrics });
    focuser.focus({ kind: "machine", machineName: "/Demo", bounds: { left: 0, right: 400, top: 0, bottom: 300 } });
    focuser.focus({
      kind: "node",
      nodePath: "/Demo/idle",
      bounds: { left: 40, right: 136, top: 80, bottom: 176 },
    });
    const nodeTransform = panner.transform;
    assert.deepEqual(nodeTransform, { scale: 1.2, pan: { x: 394.4, y: 146.4 } });
    assert.equal(focusKind, "node");
    assert.equal(focusPath, "/Demo/idle");
    assert.deepEqual(panner.transform, nodeTransform);
    await stopMachine(focuser);
    await stopMachine(panner);
  });

  test("clearing focus fits the remaining graphs", async () => {
    const world = fakeWorld();
    const panner = startMachine(new Panner(world), Panner.model);
    const focuser = startMachine(new Focuser(), Focuser.model);
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      origin: { x: 0, y: 0 },
    };
    focuser.focus({ kind: "machine", machineName: "/A", bounds: { left: 0, right: 96, top: 0, bottom: 96 } });
    panner.fit({ bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics });
    assert.deepEqual(panner.transform, { scale: 1.2, pan: { x: 442.4, y: 242.4 } });
    focuser.clear();
    panner.fit({ bounds: metrics.bounds, metrics });
    assert.deepEqual(panner.transform, {
      scale: 600 / 656,
      pan: { x: 500 - (500 * 600) / 656, y: 300 - (300 * 600) / 656 },
    });
    await stopMachine(focuser);
    await stopMachine(panner);
  });

  test("From mixin starts an element-shaped host as an HSM instance", async () => {
    class Base {
      label = "host";
    }
    class Host extends From(Base) {}
    const host = startMachine(new Host(), hsm.define(
      "Host",
      hsm.initial(hsm.target("active")),
      hsm.state("active"),
    ));
    assert.equal(host.label, "host");
    assert.equal(typeof host.dispatch, "function");
    assert.match(host.state(), /\/active$/);
    await stopMachine(host);
  });

  test("renderer paints after mark_dirty", async () => {
    let paints = 0;
    const renderer = startMachine(new Renderer(() => {
      paints += 1;
    }), Renderer.model);
    renderer.markDirty();
    await waitFor(() => paints === 1);
    assert.match(renderer.state(), /\/clean$/);
    await stopMachine(renderer);
  });

  test("graph admission is synchronous and stop leaves no unhandled rejection", async () => {
    const graph = startAdmittedGraph({
      onDraw: () => undefined,
      onDestroy: () => undefined,
    });
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      graph.setGraphs([graphFor("/Demo")]);
      graph.setGraphs([graphFor("/Demo")]);
      await stopMachine(graph);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assert.deepEqual(unhandled, []);
  });

  test("source and dashboard dispatch-stop races suppress expected shutdown failures", async () => {
    const source = new OtelSourceController();
    const dashboard = new DashboardController();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      const sourceDispatch = source.dispatch("source.connect.requested").catch(reportHsmFailure);
      const dashboardDispatch = dashboard.dispatch("dashboard.replay.next").catch(reportHsmFailure);
      await Promise.all([source.stop(), dashboard.stop()]);
      await Promise.all([sourceDispatch, dashboardDispatch]);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assert.deepEqual(unhandled, []);
  });

  test("HSM failure reporting suppresses shutdown and reports unexpected errors", () => {
    const reports: unknown[] = [];
    const globalWithReportError = globalThis as typeof globalThis & {
      reportError?: (value: unknown) => void;
    };
    const previous = globalWithReportError.reportError;
    globalWithReportError.reportError = (error) => {
      reports.push(error);
    };
    try {
      reportHsmFailure(new Error("dispatch requires a started HSM"));
      const unexpected = new Error("unexpected HSM failure");
      reportHsmFailure(unexpected);
      assert.deepEqual(reports, [unexpected]);
    } finally {
      if (previous === undefined) {
        delete globalWithReportError.reportError;
      } else {
        globalWithReportError.reportError = previous;
      }
    }
  });

  test("stopping the dashboard cancels replay callbacks", async () => {
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    const snapshots: DashboardSnapshot[] = [];
    const dashboard = new DashboardController({
      onSnapshot: (snapshot) => {
        snapshots.push(snapshot);
      },
    });
    dashboard.applySpans(parsed.spans, 0, "replace");
    dashboard.enterReplay();
    dashboard.playReplay();
    assert.ok(dashboard.snapshot().replay.total > 0);
    assert.equal(dashboard.snapshot().replay.playing, true);
    const snapshotsBeforeStop = snapshots.length;

    await dashboard.stop();
    await new Promise<void>((resolve) => setTimeout(resolve, 750));

    assert.equal(snapshots.length, snapshotsBeforeStop);
  });

  test("each controller starts an hsm.ts machine whose snapshot state path is hierarchical", async () => {
    const dashboard = new DashboardController();
    const source = new OtelSourceController();
    const draws: string[] = [];
    const graph = startAdmittedGraph({
      onDraw: (values) => {
        draws.push(values.map((value) => value.currentState).join(","));
      },
      onDestroy: () => {
        draws.push("destroy");
      },
    });

    assert.equal(dashboard instanceof hsm.Instance, true);
    assert.equal(source instanceof hsm.Instance, true);
    assert.equal(graph instanceof hsm.Instance, true);
    assert.match(dashboard.snapshot().statePath, /^\//);
    assert.match(source.snapshot().statePath, /^\//);
    assert.match(graph.snapshot().statePath, /^\//);
    assert.equal(dashboard.snapshot().phase, "idle");
    assert.equal(source.snapshot().phase, "idle");
    assert.equal(graph.snapshot().phase, "empty");

    const ready: string[] = [];
    const emitting = new OtelSourceController({
      onReady: (otelSource) => {
        ready.push(otelSource.label);
      },
    });
    const afterConnect = await emitting.dispatch("source.connect.requested");
    assert.equal(afterConnect.phase, "live");
    assert.ok(afterConnect.statePath.startsWith("/"));
    assert.deepEqual(ready, ["OTLP"]);
    assert.equal(afterConnect.source?.kind, "stream");
    assert.equal(afterConnect.source?.url, "/v1/traces/stream");

    const document = documentFromOtlp(JSON.parse(readFileSync(fixturePath, "utf8")));
    assert.ok(document !== null);
    const phone = document.machines.find((machine) => machine.name === "/Phone");
    assert.ok(phone !== undefined);
    const phoneBot = document.machines.find((machine) => machine.name === "/PhoneBot");
    assert.ok(phoneBot !== undefined);
    const afterDraw = graph.setGraphs([phone, phoneBot]);
    assert.equal(afterDraw.phase, "drawing");
    assert.ok(afterDraw.statePath.startsWith("/"));
    assert.ok(draws.includes("/Phone,/PhoneBot/active/processing"));
    assert.equal(afterDraw.graphs.length, 2);

    await dashboard.stop();
    await source.stop();
    await emitting.stop();
    await stopMachine(graph);
  });

  test("stream source incrementally folds spans into live dashboard state", async () => {
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    assert.ok(parsed.spans.length >= 2);
    const first = parsed.spans[0];
    const rest = parsed.spans.slice(1);
    assert.ok(first !== undefined);

    const captured: { handlers?: OtelStreamHandlers } = {};
    let closed = 0;
    const live = new DashboardController({
      connectStream: (url, next): OtelStreamSubscription => {
        assert.equal(url, "/v1/traces/stream");
        captured.handlers = next;
        return {
          close(): void {
            closed += 1;
          },
        };
      },
    });

    const afterSelect = await live.dispatch("dashboard.source.selected", { source: streamSource() });
    assert.equal(afterSelect.phase, "live");
    assert.equal(afterSelect.source?.kind, "stream");
    const handlers = captured.handlers;
    assert.ok(handlers !== undefined);

    handlers.onSnapshot({ observeSpans: [first], skipped: 0 });
    await waitFor(() => live.snapshot().document?.observeCount === 1);
    assert.equal(live.snapshot().phase, "live");

    handlers.onSpans({ observeSpans: rest, skipped: 1 });
    await waitFor(() => live.snapshot().document?.observeCount === parsed.spans.length);
    const afterIncremental = live.snapshot();
    assert.equal(afterIncremental.phase, "live");
    assert.equal(afterIncremental.document?.skippedCount, 1);
    const phone = afterIncremental.document?.machines.find((machine) => machine.name === "/PhoneBot");
    assert.ok(phone !== undefined);
    assert.equal(phone.currentState, "/PhoneBot/active/processing");

    await live.stop();
    assert.ok(closed >= 1);
  });

  test("a published model appears before any observe spans and overlay highlights the leaf", async () => {
    const dashboard = new DashboardController({
      connectStream: () => ({
        close(): void {
          return;
        },
      }),
    });
    const afterSource = await dashboard.dispatch("dashboard.source.selected", { source: streamSource() });
    assert.equal(afterSource.phase, "live");
    const afterModel = await dashboard.dispatch("dashboard.model.published", {
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [
        { qualified_name: "/Demo", parent: "/", initial: "/Demo/.initial" },
        { qualified_name: "/Demo/idle", parent: "/Demo", initial: "" },
        { qualified_name: "/Demo/run", parent: "/Demo", initial: "" },
      ],
      transitions: [{ source: "/Demo/idle", target: "/Demo/run", events: ["go"] }],
    });
    assert.equal(afterModel.selectedGraph?.name, "/Demo");
    assert.equal(afterModel.selectedGraph?.currentState, "");
    assert.ok(afterModel.selectedGraph?.nodes.some((node) => node.path === "/Demo/run"));

    const afterLive = await dashboard.dispatch("dashboard.model.published", {
      name: "/Demo",
      initial: "/Demo/.initial",
      states: [
        { qualified_name: "/Demo", parent: "/", initial: "/Demo/.initial" },
        { qualified_name: "/Demo/idle", parent: "/Demo", initial: "" },
        { qualified_name: "/Demo/run", parent: "/Demo", initial: "" },
      ],
      transitions: [{ source: "/Demo/idle", target: "/Demo/run", events: ["go"] }],
      live: true,
      state: "/Demo/idle",
      component: "Demo",
    });
    assert.equal(afterLive.selectedGraph?.currentState, "/Demo/idle");
    assert.equal(afterLive.selectedGraph?.componentName, "Demo");
    assert.ok(afterLive.selectedGraph?.nodes.some((node) => node.path === "/Demo/run"));

    const afterObserve = await dashboard.dispatch("dashboard.load.completed", {
      mode: "replace",
      skipped: 0,
      observeSpans: [
        {
          name: "bot.hsm.observe",
          timestamp: "2026-08-16T00:00:00.000000000Z",
          start_time: "2026-08-16T00:00:00.000000000Z",
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
      ],
    });
    assert.equal(afterObserve.selectedGraph?.currentState, "/Demo/run");
    assert.ok(afterObserve.selectedGraph?.nodes.some((node) => node.path === "/Demo/idle"));
    await dashboard.stop();
  });

  test("send event posts the named command and records the gateway result", async () => {
    const posted: Array<{ eventName: string; dataJson: string }> = [];
    const dashboard = new DashboardController({
      postCommand: async (command) => {
        posted.push(command);
        return { result: "no_subscriber", detail: "no subscriber" };
      },
    });
    const afterPrefill = await dashboard.dispatch("dashboard.command.prefill", { eventName: "phone.ring" });
    assert.equal(afterPrefill.commandEventName, "phone.ring");
    const afterSend = await dashboard.dispatch("dashboard.command.send", {
      eventName: "phone.ring",
      dataJson: "",
    });
    assert.deepEqual(posted, [{ eventName: "phone.ring", dataJson: "" }]);
    assert.equal(afterSend.commandResult?.result, "no_subscriber");
    await dashboard.stop();
  });
});
