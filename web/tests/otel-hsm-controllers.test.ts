import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import { DashboardController } from "../src/dashboard-hsm.ts";
import { MachineGraphController } from "../src/machine-graph-hsm.ts";
import { structureKey } from "../src/machine-graph-view.ts";
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
    const graph = new MachineGraphController({
      renderer: {
        draw: () => {
          draws += 1;
        },
        destroy: () => {
          destroys += 1;
        },
      },
    });
    const valid = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: "/Demo", parent: null, label: "Demo" }],
      edges: [],
    };

    const empty = await graph.dispatch("graph.set", { graphs: [] });
    assert.equal(empty.phase, "empty");
    assert.equal(empty.graphs.length, 0);
    assert.equal(draws, 0);
    const drawing = await graph.dispatch("graph.set", { graphs: [valid] });
    assert.equal(drawing.phase, "drawing");
    assert.equal(draws, 1);
    const malformed = await graph.dispatch("graph.set", { graphs: [{}] });
    assert.equal(malformed.phase, "empty");
    assert.equal(malformed.graphs.length, 0);
    assert.ok(destroys >= 1);
    const malformedPayload = await graph.dispatch("graph.set", { graphs: "invalid" });
    assert.equal(malformedPayload.phase, "empty");
    assert.equal(malformedPayload.graphs.length, 0);
    const redraw = await graph.dispatch("graph.set", { graphs: [valid] });
    assert.equal(redraw.phase, "drawing");
    assert.equal(draws, 2);
    await graph.stop();
  });

  test("viewport gestures use modeled fit, pan, and zoom transitions", async () => {
    const applied: unknown[] = [];
    const graph = new MachineGraphController({
      renderer: {
        draw: () => undefined,
        destroy: () => undefined,
        applyViewport: (data) => {
          applied.push(data);
        },
      },
    });
    const valid = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: "/Demo", parent: null, label: "Demo" }],
      edges: [],
    };

    await graph.dispatch("graph.set", { graphs: [valid] });
    await graph.dispatch("viewport.fit");
    assert.equal(applied.length, 1);
    const panning = await graph.dispatch("viewport.pan.start");
    assert.equal(panning.phase, "drawing");
    assert.match(panning.statePath, /\/panning$/);
    await graph.dispatch("viewport.pan", { pan: { x: 12, y: 8 } });
    await graph.dispatch("viewport.zoom", { scale: 1.1, point: { x: 20, y: 20 } });
    assert.equal(applied.length, 3);
    const drawing = await graph.dispatch("viewport.pan.end");
    assert.match(drawing.statePath, /\/drawing$/);
    await graph.stop();
  });

  test("graph updates repaint while the viewport is panning", async () => {
    const drawn: string[][] = [];
    const graphFor = (name: string) => ({
      name,
      componentName: name,
      currentState: `${name}/idle`,
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: name, parent: null, label: name }],
      edges: [],
    });
    const graph = new MachineGraphController({
      renderer: {
        draw: (graphs) => {
          drawn.push(graphs.map((value) => value.name));
          return true;
        },
        destroy: () => undefined,
      },
    });

    await graph.dispatch("graph.set", { graphs: [graphFor("/A")] });
    await graph.dispatch("graph.set", { graphs: [graphFor("/B")] });
    const panning = await graph.dispatch("viewport.pan.start", { pointerId: 1, point: { x: 10, y: 10 } });
    assert.match(panning.statePath, /\/panning$/);
    await graph.dispatch("graph.set", { graphs: [graphFor("/C")] });

    assert.deepEqual(drawn, [["/A"], ["/B"], ["/C"]]);
    assert.deepEqual(graph.snapshot().graphs.map((value) => value.name), ["/C"]);
    await graph.stop();
  });

  test("normalized viewport intents update controller-owned transform and gesture state", async () => {
    const applied: unknown[] = [];
    const graph = new MachineGraphController({
      renderer: {
        draw: () => false,
        destroy: () => undefined,
        viewportMetrics: () => ({
          width: 1000,
          height: 600,
          bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
          origin: { x: 0, y: 0 },
        }),
        focusBounds: (machineName) => machineName === "/Demo" ? { left: 0, right: 96, top: 0, bottom: 96 } : null,
        applyViewport: (data) => {
          applied.push(data);
        },
      },
    });
    const valid = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: "/Demo", parent: null, label: "Demo" }],
      edges: [],
    };

    await graph.dispatch("graph.set", { graphs: [valid] });
    await graph.dispatch("viewport.fit", { reason: "initial" });
    const panning = await graph.dispatch("viewport.pan.start", { pointerId: 1, point: { x: 10, y: 10 } });
    assert.match(panning.statePath, /\/panning$/);
    await graph.dispatch("viewport.pan", { pointerId: 1, point: { x: 30, y: 24 } });
    await graph.dispatch("viewport.pan.end", { pointerId: 1 });
    const drawing = await graph.dispatch("viewport.zoom", { deltaY: -100, point: { x: 30, y: 24 } });
    assert.match(drawing.statePath, /\/drawing$/);
    await graph.dispatch("viewport.focus", { machineName: "/Demo" });
    assert.deepEqual(applied.at(-1), { scale: 1.2, pan: { x: 442.4, y: 242.4 } });
    await graph.stop();
  });

  test("removing the focused machine clears focus and fits the remaining graphs", async () => {
    const applied: unknown[] = [];
    let available = new Set<string>();
    const graphFor = (name: string) => ({
      name,
      componentName: name,
      currentState: `${name}/idle`,
      lastEventName: "",
      observationCount: 1,
      nodes: [{ path: name, parent: null, label: name }],
      edges: [],
    });
    const graph = new MachineGraphController({
      renderer: {
        draw: (graphs) => {
          available = new Set(graphs.map((value) => value.name));
          return true;
        },
        destroy: () => undefined,
        viewportMetrics: () => ({
          width: 1000,
          height: 600,
          bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
          origin: { x: 0, y: 0 },
        }),
        focusBounds: (machineName) => available.has(machineName)
          ? { left: 0, right: 96, top: 0, bottom: 96 }
          : null,
        applyViewport: (data) => {
          applied.push(data);
        },
      },
    });

    await graph.dispatch("graph.set", { graphs: [graphFor("/A")] });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    assert.equal(graph.focusMachine("/A"), true);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await graph.dispatch("graph.set", { graphs: [graphFor("/B")] });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    const transforms = applied.filter((value) =>
      typeof value === "object" && value !== null && "scale" in value,
    );
    assert.notDeepEqual(transforms.at(-1), { scale: 1.2, pan: { x: 442.4, y: 242.4 } });
    assert.deepEqual(transforms.at(-1), {
      scale: 600 / 656,
      pan: { x: 500 - (500 * 600) / 656, y: 300 - (300 * 600) / 656 },
    });
    await graph.stop();
  });

  test("filtered focused machines clear focus and fit the admitted graphs", async () => {
    const applied: unknown[] = [];
    let available = new Set<string>();
    const graphFor = (name: string, admitted: boolean) => ({
      name,
      componentName: name,
      currentState: `${name}/idle`,
      lastEventName: "",
      observationCount: 1,
      nodes: admitted ? [{ path: name, parent: null, label: name }] : [],
      edges: [],
    });
    const graph = new MachineGraphController({
      renderer: {
        draw: (graphs) => {
          available = new Set(graphs.filter((value) => value.nodes.length > 0).map((value) => value.name));
          return true;
        },
        destroy: () => undefined,
        viewportMetrics: () => ({
          width: 1000,
          height: 600,
          bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
          origin: { x: 0, y: 0 },
        }),
        focusBounds: (machineName) => available.has(machineName)
          ? { left: 0, right: 96, top: 0, bottom: 96 }
          : null,
        applyViewport: (data) => {
          applied.push(data);
        },
      },
    });

    await graph.dispatch("graph.set", { graphs: [graphFor("/A", true)] });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    assert.equal(graph.focusMachine("/A"), true);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await graph.dispatch("graph.set", { graphs: [graphFor("/A", false), graphFor("/B", true)] });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    const transforms = applied.filter((value) =>
      typeof value === "object" && value !== null && "scale" in value,
    );
    assert.deepEqual(transforms.at(-1), {
      scale: 600 / 656,
      pan: { x: 500 - (500 * 600) / 656, y: 300 - (300 * 600) / 656 },
    });
    await graph.stop();
  });

  test("stopping a graph controller gates queued admission before the HSM stops", async () => {
    let draws = 0;
    const graph = new MachineGraphController({
      renderer: {
        draw: () => {
          draws += 1;
          return true;
        },
        destroy: () => undefined,
      },
    });
    const valid = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: "/Demo", parent: null, label: "Demo" }],
      edges: [],
    };

    const admission = graph.dispatch("graph.set", { graphs: [valid] });
    await graph.stop();
    await assert.rejects(admission, /MachineGraphController is stopped/);
    assert.equal(draws, 0);
  });

  test("deferred initial fit does not reject during serialized shutdown", async () => {
    const graph = new MachineGraphController({
      renderer: {
        draw: () => true,
        destroy: () => undefined,
      },
    });
    const valid = {
      name: "/Demo",
      componentName: "Demo",
      currentState: "/Demo/idle",
      lastEventName: "",
      observationCount: 0,
      nodes: [{ path: "/Demo", parent: null, label: "Demo" }],
      edges: [],
    };
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      const first = graph.dispatch("graph.set", { graphs: [valid] });
      const second = graph.dispatch("graph.set", { graphs: [valid] });
      await first;
      await graph.stop();
      await second;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assert.deepEqual(unhandled, []);
  });

  test("each controller starts an hsm.ts machine whose snapshot state path is hierarchical", async () => {
    const dashboard = new DashboardController();
    const source = new OtelSourceController();
    const draws: string[] = [];
    const graph = new MachineGraphController({
      renderer: {
        draw: (values) => {
          draws.push(values.map((value) => value.currentState).join(","));
        },
        destroy: () => {
          draws.push("destroy");
        },
      },
    });

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
    const afterDraw = await graph.dispatch("graph.set", { graphs: [phone, phoneBot] });
    assert.equal(afterDraw.phase, "drawing");
    assert.ok(afterDraw.statePath.startsWith("/"));
    assert.ok(draws.includes("/Phone,/PhoneBot/active/processing"));
    assert.equal(afterDraw.graphs.length, 2);

    await dashboard.stop();
    await source.stop();
    await emitting.stop();
    await graph.stop();
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
