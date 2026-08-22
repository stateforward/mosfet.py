import "./dom.ts";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as hsm from "../src/hsm.ts";

import { commandEventNameLegal, Dashboard, postCommandHttp, type DashboardSnapshot } from "../src/dashboard.ts";
import { Focuser } from "../src/flow/focuser.ts";
import { getViewportForBounds } from "../src/flow/path.ts";
import { Panner } from "../src/flow/panner.ts";
import { FIT_PADDING_RATIO, MAX_ZOOM, MIN_ZOOM } from "../src/flow/types.ts";
import { Renderer } from "../src/flow/renderer.ts";
import { Graph, graphsFromEvent, parseGraphs } from "../src/machine-graph.ts";
import { isMachineGraph } from "../src/otel/machines.ts";
import { structureKey } from "../src/machine-graph-view.ts";
import { documentFromOtlp } from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { streamSource, type OtelStreamHandlers, type OtelStreamSubscription } from "../src/otel/source.ts";
import {
  collectorUrl,
  eventWithSourceConnect,
  isSourceConnectPayload,
  OtelSource,
  sourceConnectFrom,
} from "../src/otel-source.ts";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "hsm-observe-spans.otlp.json",
);

const YIELD_MS = 0;

function viewportForFit(args: {
  bounds: { left: number; right: number; top: number; bottom: number };
  metrics: { width: number; height: number; origin: { x: number; y: number } };
}): { x: number; y: number; zoom: number } {
  return getViewportForBounds({
    bounds: args.bounds,
    origin: args.metrics.origin,
    width: args.metrics.width,
    height: args.metrics.height,
    minZoom: MIN_ZOOM,
    maxZoom: MAX_ZOOM,
    padding: FIT_PADDING_RATIO,
  });
}

const HOST_STOPPED = "stopped";
const STARTED_RUNTIME_ERROR = new Error("dispatch requires a started HSM");

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for dashboard stream update");
}

async function stopDashboard(dashboard: Dashboard): Promise<void> {
  await dashboard.stop();
  assert.equal(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: dashboard })?.reason, HOST_STOPPED);
}

function streamView(source = streamSource(), origin = "http://localhost"): {
  source: ReturnType<typeof streamSource>;
  origin: string;
} {
  return { source, origin };
}

function startAdmittedGraph(): Graph {
  return hsm.start({ instance: new Graph(), model: Graph.model });
}

async function admitGraphs(graph: Graph, value: unknown) {
  await graph.dispatch(hsm.typedEvent({ event: Graph.setEvent, data: { graphs: value } }));
  return graph.snapshot();
}

function countGraphSignals(graph: Graph): { draws: string[][]; destroys: number } {
  const signals = { draws: [] as string[][], destroys: 0 };
  const inner = graph.dispatch.bind(graph);
  graph.dispatch = ((event: hsm.DispatchEvent) => {
    if (event.name === Graph.drawnEvent.name && hsm.isRecord(event.data) && Array.isArray(event.data["graphs"])) {
      signals.draws.push(event.data["graphs"].map((value: { name?: string }) => String(value.name ?? "")));
    }
    if (event.name === Graph.clearedEvent.name) signals.destroys += 1;
    return inner(event);
  }) as Graph["dispatch"];
  return signals;
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

function bootDashboard(options: {
  onSnapshot?: (snapshot: DashboardSnapshot) => void;
  connectStream?: Dashboard["connectStream"];
  postCommand?: Dashboard["postCommand"];
} = {}): Dashboard {
  const dashboard = new Dashboard();
  dashboard.origin = "http://localhost";
  if (options.onSnapshot !== undefined) dashboard.onSnapshot = options.onSnapshot;
  if (options.connectStream !== undefined) dashboard.connectStream = options.connectStream;
  if (options.postCommand !== undefined) dashboard.postCommand = options.postCommand;
  dashboard.boot();
  return dashboard;
}

async function dispatchLoad(
  dashboard: Dashboard,
  data: unknown,
  name: "dashboard.load.completed" | "dashboard.load.failed" = "dashboard.load.completed",
): Promise<DashboardSnapshot> {
  const kind = name.endsWith("failed") ? hsm.Kinds.ErrorEvent : hsm.Kinds.CompletionEvent;
  await dashboard.dispatch(hsm.typedEvent({ event: { name, kind }, data: data }));
  return dashboard.snapshot();
}

function bootSource(options: { onReady?: OtelSource["onReady"] } = {}): OtelSource {
  const source = new OtelSource();
  if (options.onReady !== undefined) source.onReady = options.onReady;
  source.boot();
  return source;
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

  test("parseGraphs and graphsFromEvent reject malformed payloads", () => {
    const empty: unknown[] = [];
    assert.deepEqual(parseGraphs(empty), []);
    assert.equal(parseGraphs(null), null);
    assert.equal(parseGraphs([{}]), null);
    assert.equal(isMachineGraph({}), false);
    assert.equal(isMachineGraph(graphFor("/Demo")), true);
    const missingData = graphsFromEvent(hsm.typedEvent({ event: Graph.setEvent }));
    assert.equal(missingData, null);
    const malformed = graphsFromEvent(hsm.typedEvent({ event: Graph.setEvent, data: { graphs: [{}] } }));
    assert.equal(malformed, null);
    const valid = graphsFromEvent(hsm.typedEvent({ event: Graph.setEvent, data: { graphs: [graphFor("/Demo")] } }));
    assert.equal(valid?.[0]?.name, "/Demo");
  });

  test("malformed and empty graph sets remain empty and clear a drawing", async () => {
    const graph = startAdmittedGraph();
    const signals = countGraphSignals(graph);
    const valid = graphFor("/Demo");

    const empty = await admitGraphs(graph, []);
    assert.equal(empty.phase, "empty");
    assert.equal(empty.graphs.length, 0);
    const drawing = await admitGraphs(graph, [valid]);
    assert.equal(drawing.phase, "drawing");
    assert.equal(signals.draws.length, 1);
    const malformed = await admitGraphs(graph, [{}]);
    assert.equal(malformed.phase, "empty");
    assert.equal(malformed.graphs.length, 0);
    assert.ok(signals.destroys >= 1);
    const malformedPayload = await admitGraphs(graph, "invalid");
    assert.equal(malformedPayload.phase, "empty");
    assert.equal(malformedPayload.graphs.length, 0);
    const redraw = await admitGraphs(graph, [valid]);
    assert.equal(redraw.phase, "drawing");
    assert.equal(signals.draws.length, 2);
    await hsm.stop(graph);
  });

  test("drawing entry always notifies graph.drawn", async () => {
    const graph = startAdmittedGraph();
    const signals = countGraphSignals(graph);
    const drawing = await admitGraphs(graph, [graphFor("/Demo")]);
    assert.equal(drawing.phase, "drawing");
    assert.equal(drawing.graphs.length, 1);
    assert.equal(signals.draws.length, 1);
    assert.deepEqual(signals.draws[0], ["/Demo"]);
    Graph.notifyDrawn(graph.context(), graph, hsm.typedEvent({ event: Graph.setEvent }));
    await waitFor(() => signals.draws.length === 2);
    assert.equal(signals.draws.length, 2);
    assert.deepEqual(signals.draws[1], ["/Demo"]);
    const empty = startAdmittedGraph();
    const emptySignals = countGraphSignals(empty);
    Graph.notifyDrawn(empty.context(), empty, hsm.typedEvent({ event: Graph.setEvent }));
    await waitFor(() => emptySignals.draws.length === 1);
    assert.equal(emptySignals.draws.length, 1);
    assert.deepEqual(emptySignals.draws[0], []);
    assert.throws(
      () => Graph.remember(empty.context(), empty, hsm.typedEvent({ event: Graph.setEvent, data: { graphs: "invalid" } })),
      TypeError,
    );
    assert.equal(empty.snapshot().graphs.length, 0);
    await hsm.stop(graph);
    await hsm.stop(empty);
  });

  test("panner writes transform synchronously and stays off the graph model", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const graph = startAdmittedGraph();

    const drawing = await admitGraphs(graph, [graphFor("/Demo")]);
    assert.equal(drawing.phase, "drawing");
    assert.match(drawing.statePath, /\/drawing$/);
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: {
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      metrics: {
        width: 1000,
        height: 600,
        bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
        origin: { x: 0, y: 0 },
      },
    } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId: 1, point: { x: 10, y: 10 } } }));
    assert.match(panner.state(), /\/single$/);
    assert.match(graph.state(), /\/drawing$/);
    await panner.dispatch(hsm.typedEvent({ event: Panner.viewportEvent, data: { x: 12, y: 8, zoom: panner.viewport.zoom } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.zoomEvent, data: { scale: 1.1, point: { x: 20, y: 20 } } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.panEndEvent, data: { pointerId: 1 } }));
    assert.match(panner.state(), /\/fixed$/);
    assert.ok(Number.isFinite(panner.transform.scale));
    await hsm.stop(panner);
    await hsm.stop(graph);
  });

  test("graph updates repaint while the viewport is panning", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const graph = startAdmittedGraph();
    const signals = countGraphSignals(graph);

    await admitGraphs(graph, [graphFor("/A")]);
    await admitGraphs(graph, [graphFor("/B")]);
    await panner.dispatch(hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId: 1, point: { x: 10, y: 10 } } }));
    assert.match(panner.state(), /\/single$/);
    await admitGraphs(graph, [graphFor("/C")]);

    assert.deepEqual(signals.draws, [["/A"], ["/B"], ["/C"]]);
    assert.deepEqual(graph.snapshot().graphs.map((value) => value.name), ["/C"]);
    const snap = graph.snapshot();
    (snap.graphs as unknown as Array<{ name: string }>).push({ name: "/hijack" });
    assert.deepEqual(graph.snapshot().graphs.map((value) => value.name), ["/C"]);
    const live = graph.graphs as unknown as Array<{ name: string; nodes: Array<{ label: string }> }>;
    const first = live[0];
    if (first !== undefined) {
      first.name = "/hijacked";
      const node = first.nodes[0];
      if (node !== undefined) node.label = "MUT";
    }
    assert.deepEqual(graph.snapshot().graphs.map((value) => value.name), ["/C"]);
    assert.equal(graph.snapshot().graphs[0]?.nodes[0]?.label, "/C");
    await hsm.stop(panner);
    await hsm.stop(graph);
  });

  test("normalized viewport intents update panner-owned transform", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      origin: { x: 0, y: 0 },
    };

    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { reason: "initial", bounds: metrics.bounds, metrics } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId: 1, point: { x: 10, y: 10 } } }));
    assert.match(panner.state(), /\/single$/);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId: 1, point: { x: 30, y: 24 } } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.panEndEvent, data: { pointerId: 1 } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.zoomEvent, data: { deltaY: -100, point: { x: 30, y: 24 } } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics } }));
    assert.deepEqual(panner.viewport, viewportForFit({ bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics }));
    await hsm.stop(panner);
  });

  test("fit during pan rebases the next cursor delta from the fitted pan", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const pointerId = 1;
    const startPoint = { x: 10, y: 10 };
    const midPoint = { x: 30, y: 24 };
    const endPoint = { x: 50, y: 44 };
    const metricsWidth = 1000;
    const metricsHeight = 600;
    const bounds = { left: 0, right: 96, top: 0, bottom: 96 };
    const metrics = {
      width: metricsWidth,
      height: metricsHeight,
      bounds: { left: 0, right: metricsWidth, top: 0, bottom: metricsHeight },
      origin: { x: 0, y: 0 },
    };
    await panner.dispatch(hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId, point: startPoint } }));
    assert.match(panner.state(), /\/single$/);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: midPoint } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { bounds, metrics } }));
    const fitted = panner.viewport;
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: midPoint } }));
    const held = panner.viewport;
    const pixelTolerance = 1e-9;
    assert.ok(Math.abs(held.x - fitted.x) < pixelTolerance);
    assert.ok(Math.abs(held.y - fitted.y) < pixelTolerance);
    assert.equal(held.zoom, fitted.zoom);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: endPoint } }));
    const continued = panner.viewport;
    assert.equal(continued.x, fitted.x + (endPoint.x - midPoint.x));
    assert.equal(continued.y, fitted.y + (endPoint.y - midPoint.y));
    assert.equal(continued.zoom, fitted.zoom);
    await hsm.stop(panner);
  });

  test("viewport_set during pan rebases the next cursor delta from the set pan", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const pointerId = 1;
    const startPoint = { x: 10, y: 10 };
    const midPoint = { x: 30, y: 24 };
    const endPoint = { x: 50, y: 44 };
    const setViewport = { x: 12, y: 8, zoom: 1.1 };
    await panner.dispatch(hsm.typedEvent({ event: Panner.panStartEvent, data: { pointerId, point: startPoint } }));
    assert.match(panner.state(), /\/single$/);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: midPoint } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.viewportEvent, data: setViewport }));
    assert.deepEqual(panner.viewport, setViewport);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: midPoint } }));
    assert.deepEqual(panner.viewport, setViewport);
    await panner.dispatch(hsm.typedEvent({ event: Panner.cursorMoveEvent, data: { pointerId, point: endPoint } }));
    const continued = panner.viewport;
    assert.equal(continued.x, setViewport.x + (endPoint.x - midPoint.x));
    assert.equal(continued.y, setViewport.y + (endPoint.y - midPoint.y));
    assert.equal(continued.zoom, setViewport.zoom);
    await hsm.stop(panner);
  });

  test("node viewport focus uses exact bounds and stays focused across resize", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    let focusKind = "";
    let focusPath = "";
    const focuser = hsm.start({ instance: new Focuser(), model: Focuser.model });
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 400, top: 0, bottom: 300 },
      origin: { x: 0, y: 0 },
    };

    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { reason: "initial", bounds: metrics.bounds, metrics } }));
    await focuser.dispatch(hsm.typedEvent({ event: Focuser.focusEvent, data: { kind: "machine", machineName: "/Demo", bounds: { left: 0, right: 400, top: 0, bottom: 300 } } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: {
      bounds: { left: 0, right: 400, top: 0, bottom: 300 },
      metrics: {
        width: 1000,
        height: 600,
        bounds: { left: 0, right: 400, top: 0, bottom: 300 },
        origin: { x: 0, y: 0 },
      },
    } }));
    await focuser.dispatch(hsm.typedEvent({ event: Focuser.focusEvent, data: {
      kind: "node",
      nodePath: "/Demo/idle",
      bounds: { left: 40, right: 136, top: 80, bottom: 176 },
    } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: {
      bounds: { left: 40, right: 136, top: 80, bottom: 176 },
      metrics: {
        width: 1000,
        height: 600,
        bounds: { left: 0, right: 400, top: 0, bottom: 300 },
        origin: { x: 0, y: 0 },
      },
    } }));
    const nodeTransform = panner.viewport;
    assert.deepEqual(nodeTransform, viewportForFit({
      bounds: { left: 40, right: 136, top: 80, bottom: 176 },
      metrics: { width: 1000, height: 600, origin: { x: 0, y: 0 } },
    }));
    focusKind = focuser.current?.kind ?? "";
    focusPath = focuser.current?.nodePath ?? "";
    assert.equal(focusKind, "node");
    assert.equal(focusPath, "/Demo/idle");
    assert.deepEqual(panner.viewport, nodeTransform);
    await hsm.stop(focuser);
    await hsm.stop(panner);
  });

  test("clearing focus fits the remaining graphs", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const focuser = hsm.start({ instance: new Focuser(), model: Focuser.model });
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      origin: { x: 0, y: 0 },
    };
    await focuser.dispatch(hsm.typedEvent({ event: Focuser.focusEvent, data: { kind: "machine", machineName: "/A", bounds: { left: 0, right: 96, top: 0, bottom: 96 } } }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics } }));
    assert.deepEqual(panner.viewport, viewportForFit({ bounds: { left: 0, right: 96, top: 0, bottom: 96 }, metrics }));
    await focuser.dispatch(hsm.typedEvent({ event: Focuser.clearEvent }));
    await panner.dispatch(hsm.typedEvent({ event: Panner.fitEvent, data: { bounds: metrics.bounds, metrics } }));
    assert.deepEqual(panner.viewport, viewportForFit({ bounds: metrics.bounds, metrics }));
    await hsm.stop(focuser);
    await hsm.stop(panner);
  });

  test("fitEvent inverted, zero-span, and nonzero-origin bounds share getViewportForBounds", async () => {
    const viewportWidth = 1000;
    const viewportHeight = 600;
    const originX = 48;
    const originY = -12;
    const cases: readonly {
      readonly bounds: { left: number; right: number; top: number; bottom: number };
      readonly origin: { x: number; y: number };
    }[] = [
      { bounds: { left: 80, right: 0, top: 40, bottom: 0 }, origin: { x: 0, y: 0 } },
      { bounds: { left: 20, right: 20, top: 10, bottom: 10 }, origin: { x: 0, y: 0 } },
      { bounds: { left: 0, right: 80, top: 0, bottom: 40 }, origin: { x: originX, y: originY } },
    ];
    for (const fitCase of cases) {
      const panner = hsm.start({ instance: new Panner(), model: Panner.model });
      const metrics = {
        width: viewportWidth,
        height: viewportHeight,
        bounds: { left: 0, right: viewportWidth, top: 0, bottom: viewportHeight },
        origin: fitCase.origin,
      };
      await panner.dispatch(hsm.typedEvent({
        event: Panner.fitEvent,
        data: { bounds: fitCase.bounds, metrics },
      }));
      assert.deepEqual(panner.viewport, viewportForFit({ bounds: fitCase.bounds, metrics }));
      await hsm.stop(panner);
    }
  });

  test("fitEvent with non-finite bounds leaves the identity viewport", async () => {
    const panner = hsm.start({ instance: new Panner(), model: Panner.model });
    const identity = panner.viewport;
    const metrics = {
      width: 1000,
      height: 600,
      bounds: { left: 0, right: 1000, top: 0, bottom: 600 },
      origin: { x: 0, y: 0 },
    };
    await panner.dispatch(hsm.typedEvent({
      event: Panner.fitEvent,
      data: {
        bounds: { left: 0, right: 80, top: 0, bottom: 40 },
        metrics: { ...metrics, width: Number.NaN },
      },
    }));
    assert.deepEqual(panner.viewport, identity);
    await hsm.stop(panner);
  });

  test("From mixin starts an element-shaped host as an HSM instance", async () => {
    class Base {
      label = "host";
    }
    class Host extends hsm.from(Base) {}
    const host = hsm.start({
      instance: new Host(),
      model: hsm.define(
        "Host",
        hsm.initial(hsm.target("active")),
        hsm.state("active"),
      ),
    });
    assert.equal(host.label, "host");
    assert.equal(typeof host.dispatch, "function");
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
  });

  test("renderer paints after mark_dirty", async () => {
    let paints = 0;
    const renderer = hsm.start({ instance: new Renderer(), model: Renderer.model });
    const inner = renderer.dispatch.bind(renderer);
    renderer.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Renderer.paintEvent.name) paints += 1;
      return inner(event);
    }) as Renderer["dispatch"];
    void renderer.dispatch(hsm.typedEvent({ event: Renderer.markDirtyEvent })).catch(hsm.catchFailure());
    await waitFor(() => paints === 1);
    assert.match(renderer.state(), /\/clean$/);
    await hsm.stop(renderer);
  });

  test("graph admission is synchronous and stop leaves no unhandled rejection", async () => {
    const graph = startAdmittedGraph();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      await admitGraphs(graph, [graphFor("/Demo")]);
      await admitGraphs(graph, [graphFor("/Demo")]);
      await hsm.stop(graph);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assert.deepEqual(unhandled, []);
  });

  test("Dashboard Event-path dispatch rejects non-Events", async () => {
    const dashboard = bootDashboard();
    const eventPathMessage = "dispatch(event) requires an Event";
    const ctxPathMessage = "dispatch(ctx, event) requires an Event";
    const notEvent = { name: "dashboard.reset" };
    assert.throws(
      () => {
        void dashboard.dispatch(notEvent as hsm.Event);
      },
      (error: unknown) => error instanceof TypeError && error.message === eventPathMessage,
    );
    assert.throws(
      () => {
        void dashboard.dispatch(dashboard.context(), notEvent as hsm.Event);
      },
      (error: unknown) => error instanceof TypeError && error.message === ctxPathMessage,
    );
    const snapshot = await dashboard.dispatch("dashboard.reset");
    assert.equal(typeof snapshot.statePath, "string");
    await dashboard.dispatch(hsm.typedEvent({ event: { name: "dashboard.reset", kind: hsm.Kinds.Event } }));
    await dashboard.stop();
  });

  test("OtelSource Event-path dispatch rejects non-Events", async () => {
    const source = bootSource();
    const eventPathMessage = "dispatch(event) requires an Event";
    const ctxPathMessage = "dispatch(ctx, event) requires an Event";
    const notEvent = { name: "source.attach" };
    assert.throws(
      () => {
        void source.dispatch(notEvent as hsm.Event);
      },
      (error: unknown) => error instanceof TypeError && error.message === eventPathMessage,
    );
    assert.throws(
      () => {
        void source.dispatch(source.context(), notEvent as hsm.Event);
      },
      (error: unknown) => error instanceof TypeError && error.message === ctxPathMessage,
    );
    const snapshot = await source.dispatch("source.attach");
    assert.equal(typeof snapshot.statePath, "string");
    await source.dispatch(hsm.typedEvent({ event: { name: "source.attach", kind: hsm.Kinds.Event } }));
    await source.stop();
  });

  test("source and dashboard dispatch-stop races suppress expected shutdown failures", async () => {
    const source = bootSource();
    const dashboard = bootDashboard();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      const sourceDispatch = source.dispatch("source.connect.requested", { origin: "http://localhost" }).catch(hsm.catchFailure(source));
      const dashboardDispatch = dashboard.dispatch("dashboard.replay.next").catch(hsm.catchFailure(dashboard));
      await Promise.all([source.stop(), stopDashboard(dashboard)]);
      await Promise.all([sourceDispatch, dashboardDispatch]);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assert.deepEqual(unhandled, []);
  });

  test("failure reporting surfaces unstarted dispatch as HostDropError", () => {
    const reports: unknown[] = [];
    const globalWithReportError = globalThis as typeof globalThis & {
      reportError?: (value: unknown) => void;
    };
    const previous = globalWithReportError.reportError;
    globalWithReportError.reportError = (error) => {
      reports.push(error);
    };
    const startedRuntimeError = new Error("dispatch requires a started HSM");
    const unstarted = "unstarted";
    const host = document.createElement("div");
    try {
      assert.throws(
        () => hsm.reportFailure({ error: startedRuntimeError, host }),
        (error: unknown) => error instanceof hsm.HostDropError && error.reason === unstarted,
      );
      assert.throws(
        () => hsm.reportFailure({ error: startedRuntimeError }),
        (error: unknown) => error instanceof hsm.HostRequiredError,
      );
      assert.throws(
        () => hsm.catchFailure()(startedRuntimeError),
        (error: unknown) => error instanceof hsm.HostRequiredError,
      );
      const unexpected = new Error("unexpected HSM failure");
      hsm.reportFailure({ error: unexpected });
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
    const dashboard = bootDashboard({
      onSnapshot: (snapshot) => {
        snapshots.push(snapshot);
      },
    });
    await dashboard.dispatch("dashboard.replay.enter");
    await dispatchLoad(dashboard, {
      mode: "replace",
      skipped: parsed.skipped,
      observeSpans: parsed.spans,
    });
    await dashboard.dispatch("dashboard.replay.play");
    assert.ok(dashboard.snapshot().replay.total > 0);
    assert.equal(dashboard.snapshot().replay.playing, true);
    const snapshotsBeforeStop = snapshots.length;

    await stopDashboard(dashboard);
    await Promise.resolve();
    await Promise.resolve();

    assert.equal(snapshots.length, snapshotsBeforeStop);
  });

  test("each controller starts an hsm.ts machine whose snapshot state path is hierarchical", async () => {
    const dashboard = bootDashboard();
    const source = bootSource();
    const draws: string[] = [];
    const graph = startAdmittedGraph();
    const innerGraph = graph.dispatch.bind(graph);
    graph.dispatch = ((event: hsm.DispatchEvent) => {
      if (event.name === Graph.drawnEvent.name && hsm.isRecord(event.data) && Array.isArray(event.data["graphs"])) {
        draws.push(event.data["graphs"].map((value: { currentState?: string }) => String(value.currentState ?? "")).join(","));
      }
      if (event.name === Graph.clearedEvent.name) draws.push("destroy");
      return innerGraph(event);
    }) as Graph["dispatch"];

    assert.equal(typeof dashboard.dispatch, "function");
    assert.equal(typeof source.dispatch, "function");
    const graphIsInstance = true;
    assert.equal(graph instanceof hsm.Instance, graphIsInstance);
    assert.match(dashboard.snapshot().statePath, /^\//);
    assert.match(source.snapshot().statePath, /^\//);
    assert.match(graph.snapshot().statePath, /^\//);
    assert.equal(dashboard.snapshot().phase, "idle");
    assert.equal(source.snapshot().phase, "idle");
    assert.equal(graph.snapshot().phase, "empty");

    const ready: string[] = [];
    const emitting = bootSource({
      onReady: (otelSource) => {
        ready.push(otelSource.label);
      },
    });
    const afterConnect = await emitting.dispatch("source.connect.requested", { origin: "http://localhost" });
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
    const afterDraw = await admitGraphs(graph, [phone, phoneBot]);
    assert.equal(afterDraw.phase, "drawing");
    assert.ok(afterDraw.statePath.startsWith("/"));
    assert.ok(draws.includes("/Phone,/PhoneBot/active/processing"));
    assert.equal(afterDraw.graphs.length, 2);

    await stopDashboard(dashboard);
    await source.stop();
    await emitting.stop();
    await hsm.stop(graph);
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
    const live = bootDashboard({
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

    const afterSelect = await live.dispatch("dashboard.source.selected", streamView());
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

    await stopDashboard(live);
    assert.ok(closed >= 1);
  });

  test("a published model appears before any observe spans and overlay highlights the leaf", async () => {
    const dashboard = bootDashboard({
      connectStream: () => ({
        close(): void {
          return;
        },
      }),
    });
    const afterSource = await dashboard.dispatch("dashboard.source.selected", streamView());
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

    const afterObserve = await dispatchLoad(dashboard, {
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
    await stopDashboard(dashboard);
  });

  test("replay play without events stays idle", async () => {
    const dashboard = bootDashboard();
    const after = await dashboard.dispatch("dashboard.replay.play");
    const replayNotPlaying = false;
    assert.equal(after.replay.playing, replayNotPlaying);
    assert.match(after.statePath, /\/idle$/);
    await stopDashboard(dashboard);
  });

  test("dashboard stop unbinds like Host.stop", async () => {
    const dashboard = bootDashboard();
    await dashboard.stop();
    assert.equal(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: dashboard })?.reason, HOST_STOPPED);
    assert.equal(dashboard.state(), "");
  });

  test("dashboard stop then boot attaches a new Command", async () => {
    const posted: string[] = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push(command.eventName);
        return { result: "accepted", detail: "ok" };
      },
    });
    await dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "{}" });
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.deepEqual(posted, ["phone.ring"]);
    const prior = ownedActors(dashboard);
    const commandCount = 1;
    assert.equal(prior.length, commandCount);
    const stoppedCommand = prior[0];
    assert.ok(stoppedCommand !== undefined);
    assert.notEqual(stoppedCommand.state(), "");
    await dashboard.stop();
    assert.equal(dashboard.state(), "");
    assert.equal(stoppedCommand.state(), "");
    assert.equal(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: stoppedCommand })?.reason, HOST_STOPPED);
    dashboard.boot();
    await dashboard.dispatch("dashboard.command.send", { eventName: "phone.hangup", dataJson: "{}" });
    await waitFor(() => posted.length === 2);
    assert.deepEqual(posted, ["phone.ring", "phone.hangup"]);
    const next = ownedActors(dashboard);
    assert.equal(next.length, commandCount);
    assert.notEqual(next[0], stoppedCommand);
    assert.notEqual(next[0]?.state(), "");
    await stopDashboard(dashboard);
  });

  test("overlapping dashboard stop both complete without hanging", async () => {
    const dashboard = bootDashboard();
    const first = dashboard.stop();
    const second = dashboard.stop();
    await Promise.all([first, second]);
    assert.equal(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: dashboard })?.reason, HOST_STOPPED);
    assert.equal(dashboard.state(), "");
  });

  test("requestDetach stops command then stays bound", async () => {
    const posted: string[] = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push(command.eventName);
        return { result: "accepted", detail: "ok" };
      },
    });
    dashboard.requestDetach();
    await waitFor(() => dashboard.state().endsWith("/disconnected"));
    assert.notEqual(hsm.hostDropFrom({ error: STARTED_RUNTIME_ERROR, host: dashboard })?.reason, HOST_STOPPED);
    dashboard.requestAttach();
    await waitFor(() => dashboard.state().includes("/session"));
    await dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "{}" });
    await waitFor(() => posted.length === 1);
    assert.deepEqual(posted, ["phone.ring"]);
    await stopDashboard(dashboard);
  });

  test("command event-name charset rejects empty, whitespace, overlong, and non-letter start", async () => {
    const maxLegal = 128;
    const overlong = 129;
    const letter = "a";
    const legalMax = letter.repeat(maxLegal);
    const tooLong = letter.repeat(overlong);
    const leadingSpace = " phone.ring";
    const trailingSpace = "phone.ring ";
    const digitFirst = "1phone";
    const nameIllegal = false;
    const nameLegal = true;
    assert.equal(commandEventNameLegal(""), nameIllegal);
    assert.equal(commandEventNameLegal(legalMax), nameLegal);
    assert.equal(commandEventNameLegal(tooLong), nameIllegal);
    assert.equal(commandEventNameLegal(leadingSpace), nameIllegal);
    assert.equal(commandEventNameLegal(trailingSpace), nameIllegal);
    assert.equal(commandEventNameLegal(digitFirst), nameIllegal);
  });

  test("empty command name fails without posting", async () => {
    const posted: string[] = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push(command.eventName);
        return { result: "accepted", detail: "ok" };
      },
    });
    await dashboard.dispatch("dashboard.command.send", { eventName: "", dataJson: "" });
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.deepEqual(posted, []);
    assert.equal(dashboard.snapshot().commandResult?.detail, "event_name is required");
    await stopDashboard(dashboard);
  });

  test("whitespace and digit-first command names fail as not allowed, not required", async () => {
    const posted: string[] = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push(command.eventName);
        return { result: "accepted", detail: "ok" };
      },
    });
    const notAllowed = "event_name is not an allowed command";
    await dashboard.dispatch("dashboard.command.send", { eventName: " ", dataJson: "" });
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.deepEqual(posted, []);
    assert.equal(dashboard.snapshot().commandResult?.detail, notAllowed);
    await dashboard.dispatch("dashboard.command.send", { eventName: "1go", dataJson: "" });
    await waitFor(() => dashboard.snapshot().commandResult?.detail === notAllowed && dashboard.snapshot().commandEventName === "1go");
    assert.deepEqual(posted, []);
    const httpEmpty = await postCommandHttp({ eventName: "", dataJson: "" });
    const httpSpace = await postCommandHttp({ eventName: " ", dataJson: "" });
    assert.equal(httpEmpty.detail, "event_name is required");
    assert.equal(httpSpace.detail, notAllowed);
    await stopDashboard(dashboard);
  });

  test("send event posts the named command and records the gateway result", async () => {
    const posted: Array<{ eventName: string; dataJson: string }> = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push({ eventName: command.eventName, dataJson: command.dataJson });
        return { result: "no_subscriber", detail: "no subscriber" };
      },
    });
    const afterPrefill = await dashboard.dispatch("dashboard.command.prefill", { eventName: "phone.ring" });
    assert.equal(afterPrefill.commandEventName, "phone.ring");
    await dashboard.dispatch("dashboard.command.send", {
      eventName: "phone.ring",
      dataJson: "",
    });
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.deepEqual(posted, [{ eventName: "phone.ring", dataJson: "" }]);
    assert.equal(dashboard.snapshot().commandResult?.result, "no_subscriber");
    await stopDashboard(dashboard);
  });

  test("command send failures complete through dashboard.command.failed", async () => {
    const dashboard = bootDashboard({
      postCommand: async () => {
        return { result: "error", detail: "gateway down" };
      },
    });
    await dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "" });
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.equal(dashboard.snapshot().commandResult?.result, "error");
    assert.equal(dashboard.snapshot().commandResult?.detail, "gateway down");
    await stopDashboard(dashboard);
  });

  test("error.replay.enter enters replay paused", async () => {
    const dashboard = bootDashboard({
      connectStream: () => ({
        close(): void {
          return;
        },
      }),
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    await dispatchLoad(dashboard, { message: "stream failed" }, "dashboard.load.failed");
    assert.equal(dashboard.snapshot().phase, "error");
    await dashboard.dispatch("dashboard.replay.enter");
    assert.match(dashboard.snapshot().statePath, /\/replay\/paused$/);
    assert.equal(dashboard.snapshot().replay.active, true);
    await stopDashboard(dashboard);
  });

  test("collectorUrl admits only the exact same-origin stream path", () => {
    const origin = "http://localhost:8080";
    assert.equal(collectorUrl({ origin }), "/v1/traces/stream");
    assert.equal(collectorUrl({ requested: "/v1/traces/stream", origin }), "/v1/traces/stream");
    assert.equal(collectorUrl({ requested: "http://localhost:8080/v1/traces/stream", origin }), "/v1/traces/stream");
    assert.equal(collectorUrl({ requested: "/v1/traces/stream/../../../v1/commands", origin }), null);
    assert.equal(collectorUrl({ requested: "/v1/traces/streamevil", origin }), null);
    assert.equal(collectorUrl({ requested: "/v1/traces/stream?token=1", origin }), null);
    assert.equal(collectorUrl({ requested: "javascript:alert(1)", origin }), null);
    assert.equal(collectorUrl({ requested: "https://evil.example/v1/traces/stream", origin }), null);
    assert.equal(collectorUrl({ requested: "http://169.254.169.254/v1/traces/stream", origin }), null);
  });

  test("hostile dashboard source URLs fail before EventSource", async () => {
    const opened: string[] = [];
    const dashboard = bootDashboard({
      connectStream: (url) => {
        opened.push(url);
        return { close(): void { return; } };
      },
    });
    dashboard.origin = "http://localhost";
    await dashboard.dispatch("dashboard.source.selected", {
      source: { kind: "stream", url: "https://evil.example/sse", label: "evil" },
      origin: "http://localhost",
    });
    await waitFor(() => dashboard.snapshot().phase === "error");
    assert.deepEqual(opened, []);
    assert.match(dashboard.snapshot().errorMessage ?? "", /collector url is not allowed/);
    await stopDashboard(dashboard);
  });

  test("sourceConnectFrom stamps urlAllowed from source.url and rejects split url fields", () => {
    const origin = "http://localhost";
    const allowed = "/v1/traces/stream";
    const disallowed = "https://evil.example/sse";
    const notAllowed = false;
    const allowedStamp = true;
    const splitDisallowedSource = sourceConnectFrom({
      origin,
      url: allowed,
      source: { kind: "stream", url: disallowed, label: "evil" },
    });
    assert.equal(splitDisallowedSource.urlAllowed, notAllowed);
    assert.equal(splitDisallowedSource.url, allowed);
    assert.equal(splitDisallowedSource.source?.url, disallowed);
    const splitDisallowedUrl = sourceConnectFrom({
      origin,
      url: disallowed,
      source: { kind: "stream", url: allowed, label: "ok" },
    });
    assert.equal(splitDisallowedUrl.urlAllowed, notAllowed);
    const splitSamePathDifferentString = sourceConnectFrom({
      origin,
      url: allowed,
      source: { kind: "stream", url: `${origin}${allowed}`, label: "ok" },
    });
    assert.equal(splitSamePathDifferentString.urlAllowed, notAllowed);
    const matched = sourceConnectFrom({
      origin,
      url: allowed,
      source: { kind: "stream", url: allowed, label: "ok" },
    });
    assert.equal(matched.urlAllowed, allowedStamp);
    const sourceOnly = sourceConnectFrom({
      origin,
      source: { kind: "stream", url: allowed, label: "ok" },
    });
    assert.equal(sourceOnly.urlAllowed, allowedStamp);
    const urlOnly = sourceConnectFrom({ origin, url: allowed });
    assert.equal(urlOnly.urlAllowed, allowedStamp);
  });

  test("allowed url with disallowed source.url fails closed before viewing", async () => {
    const opened: string[] = [];
    const dashboard = bootDashboard({
      connectStream: (url) => {
        opened.push(url);
        return { close(): void { return; } };
      },
    });
    dashboard.origin = "http://localhost";
    const viewing = "viewing";
    await dashboard.dispatch("dashboard.source.selected", {
      origin: "http://localhost",
      url: "/v1/traces/stream",
      source: { kind: "stream", url: "https://evil.example/sse", label: "evil" },
    });
    await waitFor(() => dashboard.snapshot().phase === "error");
    assert.equal(dashboard.snapshot().phase, "error");
    const viewingAbsent = false;
    assert.equal(dashboard.snapshot().statePath.includes(viewing), viewingAbsent);
    assert.deepEqual(opened, []);
    assert.match(dashboard.snapshot().errorMessage ?? "", /collector url is not allowed/);
    await stopDashboard(dashboard);
  });

  test("disallowed url with allowed source.url fails closed before viewing", async () => {
    const opened: string[] = [];
    const dashboard = bootDashboard({
      connectStream: (url) => {
        opened.push(url);
        return { close(): void { return; } };
      },
    });
    dashboard.origin = "http://localhost";
    const viewing = "viewing";
    await dashboard.dispatch("dashboard.source.selected", {
      origin: "http://localhost",
      url: "https://evil.example/sse",
      source: streamSource(),
    });
    await waitFor(() => dashboard.snapshot().phase === "error");
    assert.equal(dashboard.snapshot().phase, "error");
    const viewingAbsent = false;
    assert.equal(dashboard.snapshot().statePath.includes(viewing), viewingAbsent);
    assert.deepEqual(opened, []);
    assert.match(dashboard.snapshot().errorMessage ?? "", /collector url is not allowed/);
    await stopDashboard(dashboard);
  });

  test("typed dashboard.command.send posts through the modeled event", async () => {
    const posted: string[] = [];
    const kinds: unknown[] = [];
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted.push(command.eventName);
        return { result: "accepted", detail: "ok" };
      },
    });
    const inner = dashboard.dispatch.bind(dashboard) as (event: unknown, data?: unknown) => unknown;
    dashboard.dispatch = ((eventOrContext: unknown, data?: unknown) => {
      if (typeof eventOrContext === "object" && eventOrContext !== null && "kind" in eventOrContext) {
        kinds.push((eventOrContext as { kind: unknown }).kind);
      }
      return inner(eventOrContext, data);
    }) as unknown as Dashboard["dispatch"];
    await dashboard.dispatch(hsm.typedEvent({ event: { name: "dashboard.command.send", kind: hsm.Kinds.Event }, data: {
      eventName: "phone.ring",
      dataJson: "",
    } }));
    await waitFor(() => dashboard.snapshot().commandResult !== null);
    assert.deepEqual(posted, ["phone.ring"]);
    assert.equal(dashboard.snapshot().commandResult?.result, "accepted");
    assert.ok(kinds.includes(hsm.Kinds.Event));
    await stopDashboard(dashboard);
  });

  test("overlapping command sends ignore stale completion ids", async () => {
    let releaseFirst = (): void => {
      return;
    };
    const firstHold = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let calls = 0;
    const dashboard = bootDashboard({
      postCommand: async () => {
        calls += 1;
        if (calls === 1) {
          await firstHold;
          return { result: "accepted", detail: "first" };
        }
        return { result: "accepted", detail: "second" };
      },
    });
    const first = dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "1" });
    await waitFor(() => calls === 1);
    const second = dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "2" });
    await waitFor(() => dashboard.snapshot().commandResult?.detail === "second");
    releaseFirst();
    await first;
    await second;
    assert.equal(dashboard.snapshot().commandResult?.detail, "second");
    await stopDashboard(dashboard);
  });

  test("resolved accepted command is not relabeled canceled by later stop", async () => {
    const seen: DashboardSnapshot[] = [];
    const dashboard = bootDashboard({
      postCommand: async () => ({ result: "accepted", detail: "committed" }),
    });
    dashboard.onSnapshot = (snapshot) => {
      seen.push(snapshot);
    };
    await dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "" });
    await waitFor(() => dashboard.snapshot().commandResult?.result === "accepted");
    await stopDashboard(dashboard);
    assert.equal(seen.some((snapshot) => snapshot.commandResult?.result === "canceled"), false);
    assert.equal(seen.some((snapshot) => snapshot.commandResult?.detail === "committed"), true);
  });

  test("stale in-flight send cannot cancel the successor send", async () => {
    const holds: Array<{
      resolve: (result: { result: "accepted"; detail: string }) => void;
    }> = [];
    const dashboard = bootDashboard({
      postCommand: async () => {
        return await new Promise((resolve) => {
          holds.push({ resolve });
        });
      },
    });
    void dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "1" });
    await waitFor(() => holds.length === 1);
    const second = dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "2" });
    await waitFor(() => holds.length === 2);
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
    assert.notEqual(dashboard.snapshot().commandResult?.result, "canceled");
    holds[1]?.resolve({ result: "accepted", detail: "second" });
    await waitFor(() => dashboard.snapshot().commandResult?.detail === "second");
    await second;
    holds[0]?.resolve({ result: "accepted", detail: "first" });
    assert.equal(dashboard.snapshot().commandResult?.result, "accepted");
    assert.equal(dashboard.snapshot().commandResult?.detail, "second");
    await stopDashboard(dashboard);
  });

  test("replay live without a stream fails sourceCheck", async () => {
    const dashboard = bootDashboard();
    await dashboard.dispatch("dashboard.replay.enter");
    assert.equal(dashboard.snapshot().replay.active, true);
    const after = await dashboard.dispatch("dashboard.replay.live");
    assert.equal(after.phase, "error");
    assert.match(after.errorMessage ?? "", /no otel stream selected/);
    await stopDashboard(dashboard);
  });

  test("replay live with a stream re-enters viewing through sourceCheck", async () => {
    let connects = 0;
    const dashboard = bootDashboard({
      connectStream: () => {
        connects += 1;
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    assert.equal(connects, 1);
    await dashboard.dispatch("dashboard.replay.enter");
    const after = await dashboard.dispatch("dashboard.replay.live", streamView());
    assert.equal(after.phase, "live");
    const replayInactive = false;
    assert.equal(after.replay.active, replayInactive);
    assert.match(after.statePath, /\/viewing$/);
    await waitFor(() => connects === 2);
    await stopDashboard(dashboard);
  });

  test("replay live without payload uses dashboard-owned source", async () => {
    let connects = 0;
    const urls: string[] = [];
    const firstConnect = 1;
    const secondConnect = 2;
    const dashboard = bootDashboard({
      connectStream: (url) => {
        connects += 1;
        urls.push(url);
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    assert.equal(connects, firstConnect);
    await dashboard.dispatch("dashboard.replay.enter");
    const after = await dashboard.dispatch("dashboard.replay.live");
    assert.equal(after.phase, "live");
    const replayInactive = false;
    assert.equal(after.replay.active, replayInactive);
    assert.match(after.statePath, /\/viewing$/);
    await waitFor(() => connects === secondConnect);
    const collectorPath = "/v1/traces/stream";
    assert.equal(urls[1], collectorPath);
    await stopDashboard(dashboard);
  });

  test("event-path payload-less replay live is not restamped by name", async () => {
    let connects = 0;
    const firstConnect = 1;
    const dashboard = bootDashboard({
      connectStream: () => {
        connects += 1;
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    assert.equal(connects, firstConnect);
    await dashboard.dispatch("dashboard.replay.enter");
    await dashboard.dispatch(hsm.typedEvent({
      event: { name: "dashboard.replay.live", kind: hsm.Kinds.Event },
    }));
    assert.equal(dashboard.snapshot().phase, "error");
    assert.match(dashboard.snapshot().errorMessage ?? "", /no otel stream selected/);
    assert.equal(connects, firstConnect);
    await stopDashboard(dashboard);
  });

  test("event-path replay live uses producer-stamped connect data", async () => {
    let connects = 0;
    const urls: string[] = [];
    const firstConnect = 1;
    const secondConnect = 2;
    const dashboard = bootDashboard({
      connectStream: (url) => {
        connects += 1;
        urls.push(url);
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    assert.equal(connects, firstConnect);
    await dashboard.dispatch("dashboard.replay.enter");
    const owned = dashboard.ownedSourceConnect();
    await dashboard.dispatch(hsm.typedEvent({
      event: { name: "dashboard.replay.live", kind: hsm.Kinds.Event },
      data: owned,
    }));
    const after = dashboard.snapshot();
    assert.equal(after.phase, "live");
    const replayInactive = false;
    assert.equal(after.replay.active, replayInactive);
    assert.match(after.statePath, /\/viewing$/);
    await waitFor(() => connects === secondConnect);
    const collectorPath = "/v1/traces/stream";
    const secondUrlIndex = 1;
    assert.equal(urls[secondUrlIndex], collectorPath);
    await stopDashboard(dashboard);
  });

  test("replay live with source and no origin does not fill instance origin", async () => {
    let connects = 0;
    const firstConnect = 1;
    const dashboard = bootDashboard({
      connectStream: () => {
        connects += 1;
        return { close(): void { return; } };
      },
    });
    dashboard.origin = "http://localhost";
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(connects, firstConnect);
    await dashboard.dispatch("dashboard.replay.enter");
    const after = await dashboard.dispatch("dashboard.replay.live", { source: streamSource() });
    assert.equal(after.phase, "error");
    assert.match(after.errorMessage ?? "", /collector url is not allowed/);
    assert.equal(connects, firstConnect);
    await stopDashboard(dashboard);
  });

  test("replay live with origin and no source does not fill owned source", async () => {
    let connects = 0;
    const firstConnect = 1;
    const dashboard = bootDashboard({
      connectStream: () => {
        connects += 1;
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(connects, firstConnect);
    await dashboard.dispatch("dashboard.replay.enter");
    const after = await dashboard.dispatch("dashboard.replay.live", { origin: "http://example.com" });
    assert.equal(after.phase, "error");
    assert.match(after.errorMessage ?? "", /no otel stream selected/);
    assert.equal(connects, firstConnect);
    await stopDashboard(dashboard);
  });

  test("liveConnectData matches stamped origin and source pairing", async () => {
    const dashboard = bootDashboard({
      connectStream: () => ({ close(): void { return; } }),
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    const owned = dashboard.ownedSourceConnect();
    const ownedAllowed = true;
    assert.equal(owned.urlAllowed, ownedAllowed);
    assert.equal(owned.origin, dashboard.origin);
    assert.equal(owned.source?.kind, dashboard.snapshot().source?.kind);
    assert.equal(owned.source?.url, dashboard.snapshot().source?.url);
    const liveEvent = {
      name: "dashboard.replay.live",
      kind: hsm.Kinds.Event,
      data: owned,
    };
    assert.deepEqual(dashboard.liveConnectData(liveEvent), owned);
    const sourceOnly = {
      name: "dashboard.replay.live",
      kind: hsm.Kinds.Event,
      data: { source: streamSource() },
    };
    assert.equal(dashboard.liveConnectData(sourceOnly), null);
    const originStamp = sourceConnectFrom({ origin: "http://example.com" });
    const originOnly = {
      name: "dashboard.replay.live",
      kind: hsm.Kinds.Event,
      data: originStamp,
    };
    const originRead = dashboard.liveConnectData(originOnly);
    assert.equal(originRead?.origin, originStamp.origin);
    assert.equal(originRead?.urlAllowed, originStamp.urlAllowed);
    assert.equal(originRead?.source, undefined);
    await stopDashboard(dashboard);
  });

  test("postCommandHttp maps abort to canceled", async () => {
    const originalFetch = globalThis.fetch;
    let fetched = false;
    const abort = new AbortController();
    globalThis.fetch = (async (_input: unknown, init?: { signal?: AbortSignal }) => {
      fetched = true;
      const signal = init?.signal;
      return await new Promise((_resolve, reject) => {
        const fail = (): void => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        };
        if (signal?.aborted === true) {
          fail();
          return;
        }
        signal?.addEventListener("abort", fail);
      });
    }) as unknown as typeof fetch;
    abort.abort();
    try {
      const result = await postCommandHttp({
        eventName: "phone.ring",
        dataJson: "",
        signal: abort.signal,
      });
      const fetchNotEntered = false;
      assert.equal(result.result, "canceled");
      assert.equal(fetched, fetchNotEntered);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("postCommandHttp maps abort after fetch is entered to interrupted", async () => {
    const originalFetch = globalThis.fetch;
    const abort = new AbortController();
    let entered = false;
    globalThis.fetch = (async (_input: unknown, init?: { signal?: AbortSignal }) => {
      entered = true;
      const signal = init?.signal;
      return await new Promise((_resolve, reject) => {
        const fail = (): void => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        };
        if (signal?.aborted === true) {
          fail();
          return;
        }
        signal?.addEventListener("abort", fail);
      });
    }) as unknown as typeof fetch;
    try {
      const pending = postCommandHttp({
        eventName: "phone.ring",
        dataJson: "",
        signal: abort.signal,
      });
      await waitFor(() => entered);
      abort.abort();
      const result = await pending;
      assert.equal(result.result, "error");
      assert.equal(result.detail, "command reply interrupted");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("postCommandHttp admits gateway canceled", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      return {
        json: async () => ({ result: "canceled", detail: "gateway canceled" }),
      };
    }) as unknown as typeof fetch;
    try {
      const result = await postCommandHttp({ eventName: "phone.ring", dataJson: "" });
      assert.equal(result.result, "canceled");
      assert.equal(result.detail, "gateway canceled");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("postCommandHttp returns parsed accepted when the signal is already aborted", async () => {
    const originalFetch = globalThis.fetch;
    const abort = new AbortController();
    globalThis.fetch = (async () => {
      abort.abort();
      return {
        json: async () => ({ result: "accepted", detail: "committed" }),
      };
    }) as unknown as typeof fetch;
    try {
      const result = await postCommandHttp({
        eventName: "phone.ring",
        dataJson: "",
        signal: abort.signal,
      });
      assert.equal(result.result, "accepted");
      assert.equal(result.detail, "committed");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("postCommandHttp maps json AbortError after fetch resolve to interrupted", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      return {
        json: async () => {
          const error = new Error("aborted");
          error.name = "AbortError";
          throw error;
        },
      };
    }) as unknown as typeof fetch;
    try {
      const result = await postCommandHttp({ eventName: "phone.ring", dataJson: "" });
      assert.equal(result.result, "error");
      assert.equal(result.detail, "command reply interrupted");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("Command.post abort after fetch is entered is interrupted", async () => {
    const originalFetch = globalThis.fetch;
    let entered = false;
    globalThis.fetch = (async (_input: unknown, init?: { signal?: AbortSignal }) => {
      entered = true;
      const signal = init?.signal;
      return await new Promise((_resolve, reject) => {
        const fail = (): void => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        };
        if (signal?.aborted === true) {
          fail();
          return;
        }
        signal?.addEventListener("abort", fail);
      });
    }) as unknown as typeof fetch;
    let posted: DashboardSnapshot["commandResult"] | undefined;
    const dashboard = bootDashboard({
      postCommand: async (command) => {
        posted = await postCommandHttp(command);
        return posted;
      },
    });
    try {
      void dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "" });
      await waitFor(() => entered);
      await stopDashboard(dashboard);
      await waitFor(() => posted !== undefined);
      assert.equal(posted?.result, "error");
      assert.equal(posted?.detail, "command reply interrupted");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("Command.post through postCommandHttp keeps accepted after stop", async () => {
    const originalFetch = globalThis.fetch;
    let fetched = false;
    globalThis.fetch = (async () => {
      fetched = true;
      return {
        json: async () => ({ result: "accepted", detail: "http committed" }),
      };
    }) as unknown as typeof fetch;
    const dashboard = bootDashboard();
    const results: DashboardSnapshot["commandResult"][] = [];
    dashboard.onSnapshot = (snapshot) => {
      results.push(snapshot.commandResult);
    };
    try {
      void dashboard.dispatch("dashboard.command.send", { eventName: "phone.ring", dataJson: "" });
      await waitFor(() => fetched && results.some((result) => result?.result === "accepted"));
      await stopDashboard(dashboard);
      const accepted = results.find((result) => result?.result === "accepted");
      assert.equal(accepted?.result, "accepted");
      assert.equal(accepted?.detail, "http committed");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("source.selected without origin does not fall back to instance origin", async () => {
    const dashboard = bootDashboard({
      connectStream: () => ({ close(): void { return; } }),
    });
    dashboard.origin = "http://localhost";
    const after = await dashboard.dispatch("dashboard.source.selected", { source: streamSource() });
    assert.equal(after.phase, "error");
    assert.match(after.errorMessage ?? "", /collector url is not allowed/);
    await stopDashboard(dashboard);
  });

  test("dashboard detach then attach still admits a stream", async () => {
    let connects = 0;
    const dashboard = bootDashboard({
      connectStream: () => {
        connects += 1;
        return { close(): void { return; } };
      },
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    dashboard.requestDetach();
    await waitFor(() => dashboard.snapshot().statePath.includes("/disconnected"));
    dashboard.requestAttach();
    await waitFor(() => dashboard.snapshot().statePath.includes("/connected"));
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.equal(dashboard.snapshot().phase, "live");
    assert.ok(connects >= 2);
    await stopDashboard(dashboard);
  });

  test("late stream batches after leaving viewing do not dispatch stream canceled", async () => {
    const captured: { handlers?: OtelStreamHandlers } = {};
    const kinds: string[] = [];
    const dashboard = bootDashboard({
      connectStream: (_url, next): OtelStreamSubscription => {
        captured.handlers = next;
        return { close(): void { return; } };
      },
    });
    const inner = dashboard.dispatch.bind(dashboard) as Dashboard["dispatch"];
    dashboard.dispatch = ((eventOrContext: unknown, data?: unknown) => {
      if (typeof eventOrContext === "object" && eventOrContext !== null && "name" in eventOrContext) {
        kinds.push((eventOrContext as { name: string }).name);
      }
      return inner(eventOrContext as never, data);
    }) as Dashboard["dispatch"];
    await dashboard.dispatch("dashboard.source.selected", streamView());
    await dashboard.dispatch("dashboard.replay.enter");
    captured.handlers?.onSpans({ observeSpans: [], skipped: 0 });
    await Promise.resolve();
    await Promise.resolve();
    const streamCanceled = false;
    const loadFailed = false;
    const streamDropped = true;
    const productsDispatched = false;
    assert.equal(kinds.includes("dashboard.stream.canceled"), streamCanceled);
    assert.equal(kinds.includes("dashboard.load.failed"), loadFailed);
    assert.equal(kinds.includes("dashboard.stream.dropped"), streamDropped);
    assert.equal(kinds.includes("dashboard.load.completed"), productsDispatched);
    assert.match(dashboard.snapshot().statePath, /\/replay\//);
    await stopDashboard(dashboard);
  });

  test("replay play while viewing stays in viewing", async () => {
    const dashboard = bootDashboard({
      connectStream: () => ({ close(): void { return; } }),
    });
    await dashboard.dispatch("dashboard.source.selected", streamView());
    assert.match(dashboard.snapshot().statePath, /\/viewing$/);
    const request: unknown = JSON.parse(readFileSync(fixturePath, "utf8"));
    const parsed = parseExportTraceServiceRequest(request);
    assert.ok(parsed !== null);
    await dispatchLoad(dashboard, {
      mode: "replace",
      skipped: parsed.skipped,
      observeSpans: parsed.spans,
    });
    await dashboard.dispatch("dashboard.replay.play");
    assert.match(dashboard.snapshot().statePath, /\/viewing$/);
    const replayNotPlaying = false;
    assert.equal(dashboard.snapshot().replay.playing, replayNotPlaying);
    await stopDashboard(dashboard);
  });

  test("SourceConnect restamp requires origin or urlAllowed, not url alone", () => {
    const commandPayload = { eventName: "phone.ring", dataJson: "{}" };
    const commandWithUrl = { eventName: "phone.ring", dataJson: "{}", url: "/v1/traces/stream" };
    const spoofAllowed = true;
    const payloadIsConnect = true;
    const payloadIsNotConnect = false;
    const restampFailClosed = false;
    assert.equal(isSourceConnectPayload({ origin: "http://localhost" }), payloadIsConnect);
    assert.equal(isSourceConnectPayload({ urlAllowed: spoofAllowed }), payloadIsConnect);
    assert.equal(isSourceConnectPayload(commandPayload), payloadIsNotConnect);
    assert.equal(isSourceConnectPayload(commandWithUrl), payloadIsNotConnect);
    assert.equal(isSourceConnectPayload({ url: "/v1/traces/stream" }), payloadIsNotConnect);
    const spoofEvent = eventWithSourceConnect({
      name: "source.connect.requested",
      kind: hsm.Kinds.Event,
      data: { urlAllowed: spoofAllowed },
    });
    const restamped = sourceConnectFrom(spoofEvent.data);
    assert.equal(restamped.urlAllowed, restampFailClosed);
    const kept = eventWithSourceConnect({
      name: "dashboard.command.send",
      kind: hsm.Kinds.Event,
      data: commandWithUrl,
    });
    assert.deepEqual(kept.data, commandWithUrl);
  });

  test("event-path connect without origin fails closed instead of staying connecting", async () => {
    const source = bootSource();
    const spoofAllowed = true;
    await source.dispatch(hsm.typedEvent({
      event: { name: "source.connect.requested", kind: hsm.Kinds.Event },
      data: { urlAllowed: spoofAllowed },
    }));
    await waitFor(() => source.snapshot().phase === "error" || source.snapshot().phase === "live");
    assert.equal(source.snapshot().phase, "error");
    assert.notEqual(source.snapshot().phase, "connecting");
    await source.stop();
  });

  test("otel source connect completions keep declared kinds", async () => {
    const kinds: unknown[] = [];
    const source = bootSource();
    const inner = source.dispatch.bind(source) as (event: unknown, data?: unknown) => unknown;
    source.dispatch = ((eventOrContext: unknown, data?: unknown) => {
      if (typeof eventOrContext === "object" && eventOrContext !== null && "kind" in eventOrContext) {
        kinds.push((eventOrContext as { kind: unknown }).kind);
      }
      return inner(eventOrContext, data);
    }) as unknown as OtelSource["dispatch"];
    await source.dispatch("source.connect.requested", { origin: "http://localhost" });
    await waitFor(() => source.snapshot().phase === "live" || source.snapshot().phase === "error");
    assert.equal(source.snapshot().phase, "live");
    assert.ok(kinds.includes(hsm.Kinds.CompletionEvent));
    await source.stop();
  });
});
