import * as hsm from "@stateforward/hsm.ts";

import { isRecord, namedEvent, startMachine, stopMachine } from "./hsm-runtime.ts";
import { parseMachineGraph, type MachineGraph } from "./otel/machines.ts";

const graphEvents = {
  "graph.set": { name: "graph.set", kind: hsm.Kinds.Event },
  "graph.clear": { name: "graph.clear", kind: hsm.Kinds.Event },
  "viewport.fit": { name: "viewport.fit", kind: hsm.Kinds.Event },
  "viewport.focus": { name: "viewport.focus", kind: hsm.Kinds.Event },
  "viewport.pan.start": { name: "viewport.pan.start", kind: hsm.Kinds.Event },
  "viewport.pan": { name: "viewport.pan", kind: hsm.Kinds.Event },
  "viewport.pan.end": { name: "viewport.pan.end", kind: hsm.Kinds.Event },
  "viewport.zoom": { name: "viewport.zoom", kind: hsm.Kinds.Event },
} as const;

const STOPPED_CONTROLLER_ERROR = "MachineGraphController is stopped";
const FIT_PADDING = 28;
const MIN_ZOOM = 0.12;
const MAX_ZOOM = 2.4;
const MAX_FIT_ZOOM = 1.2;
const ZOOM_STEP = 0.0015;

function isExpectedControllerStop(error: unknown): boolean {
  return error instanceof Error && error.message === STOPPED_CONTROLLER_ERROR;
}

export function reportMachineGraphFailure(error: unknown): void {
  if (isExpectedControllerStop(error)) return;
  const reportError = (globalThis as typeof globalThis & {
    reportError?: (value: unknown) => void;
  }).reportError;
  if (reportError !== undefined) {
    reportError(error);
    return;
  }
  setTimeout(() => {
    throw error;
  }, 0);
}

function recordOf(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function pointOf(value: unknown): ViewportPoint | null {
  const record = recordOf(value);
  const x = record?.["x"];
  const y = record?.["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
    ? { x, y }
    : null;
}

function pointerIdOf(value: unknown): number | null {
  const pointerId = recordOf(value)?.["pointerId"];
  return typeof pointerId === "number" && Number.isInteger(pointerId) ? pointerId : null;
}

function boundsOf(value: unknown): ViewportBounds | null {
  const record = recordOf(value);
  const left = record?.["left"];
  const right = record?.["right"];
  const top = record?.["top"];
  const bottom = record?.["bottom"];
  return typeof left === "number" && Number.isFinite(left)
    && typeof right === "number" && Number.isFinite(right)
    && typeof top === "number" && Number.isFinite(top)
    && typeof bottom === "number" && Number.isFinite(bottom)
    ? { left, right, top, bottom }
    : null;
}

export type MachineGraphEventName = keyof typeof graphEvents;

export type MachineGraphPhase = "empty" | "drawing";

export type ViewportPoint = { readonly x: number; readonly y: number };
export type ViewportBounds = { readonly left: number; readonly right: number; readonly top: number; readonly bottom: number };
export type ViewportMetrics = {
  readonly width: number;
  readonly height: number;
  readonly bounds: ViewportBounds;
  readonly origin: ViewportPoint;
};

export type GraphRenderer = {
  draw(graphs: readonly MachineGraph[]): boolean | void;
  destroy(): void;
  applyViewport?(data: unknown): void;
  viewportMetrics?(): ViewportMetrics | null;
  focusBounds?(machineName: string): ViewportBounds | null;
};

export type MachineGraphSnapshot = {
  readonly phase: MachineGraphPhase;
  readonly statePath: string;
  readonly graphs: readonly MachineGraph[];
};

export type MachineGraphControllerOptions = {
  readonly renderer?: GraphRenderer;
  readonly onSnapshot?: (snapshot: MachineGraphSnapshot) => void;
};

class MachineGraphRuntime extends hsm.Instance {
  controller: MachineGraphController | null = null;
}

function controllerOf(instance: hsm.Instance): MachineGraphController | null {
  return instance instanceof MachineGraphRuntime ? instance.controller : null;
}

function parseGraphs(value: unknown): MachineGraph[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const graphs: MachineGraph[] = [];
  for (const item of value) {
    const graph = parseMachineGraph(item);
    if (graph === null) {
      return null;
    }
    graphs.push(graph);
  }
  return graphs;
}

function graphsFromEvent(event: hsm.Event): MachineGraph[] | null {
  return isRecord(event.data) ? parseGraphs(event.data["graphs"]) : null;
}

function rememberGraph(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const graphs = graphsFromEvent(event);
  if (graphs === null) {
    return;
  }
  controllerOf(instance)?.rememberGraphs(graphs);
}

function rememberAndDrawGraph(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  rememberGraph(ctx, instance, event);
  controllerOf(instance)?.draw(true);
}

function clearGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.rememberGraphs([]);
}

function drawGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.draw();
}

function destroyGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.destroyRenderer();
}

function applyViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.applyViewport(event.name, event.data);
}

function beginPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.beginPan(event.data);
}

function endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.endPan(event.data);
}

function canEndPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
  return controllerOf(instance)?.canEndPan(event.data) ?? true;
}

function ignoreViewport(): void {
  // Empty graph surfaces accept viewport events without re-entering their lifecycle.
}

const machineGraphModel = hsm.define(
  "MachineGraph",
  hsm.initial(hsm.target("empty")),
  hsm.state(
    "empty",
    hsm.entry(destroyGraph),
    hsm.transition(hsm.on("graph.clear"), hsm.target("."), hsm.effect(clearGraph)),
    hsm.transition(hsm.on("graph.set"), hsm.target("../drawing"), hsm.effect(rememberGraph)),
    hsm.transition(hsm.on("viewport.fit"), hsm.effect(ignoreViewport)),
    hsm.transition(hsm.on("viewport.focus"), hsm.effect(ignoreViewport)),
    hsm.transition(hsm.on("viewport.pan.start"), hsm.effect(ignoreViewport)),
    hsm.transition(hsm.on("viewport.pan"), hsm.effect(ignoreViewport)),
    hsm.transition(hsm.on("viewport.pan.end"), hsm.effect(ignoreViewport)),
    hsm.transition(hsm.on("viewport.zoom"), hsm.effect(ignoreViewport)),
  ),
  hsm.state(
    "drawing",
    hsm.entry(drawGraph),
    hsm.exit(destroyGraph),
    hsm.transition(hsm.on("graph.set"), hsm.effect(rememberAndDrawGraph)),
    hsm.transition(hsm.on("graph.clear"), hsm.target("../empty"), hsm.effect(clearGraph)),
    hsm.transition(hsm.on("viewport.fit"), hsm.effect(applyViewport)),
    hsm.transition(hsm.on("viewport.focus"), hsm.effect(applyViewport)),
    hsm.transition(hsm.on("viewport.pan.start"), hsm.target("../panning"), hsm.effect(beginPan)),
    hsm.transition(hsm.on("viewport.zoom"), hsm.effect(applyViewport)),
  ),
  hsm.state(
    "panning",
    hsm.transition(hsm.on("graph.set"), hsm.target("."), hsm.effect(rememberAndDrawGraph)),
    hsm.transition(hsm.on("graph.clear"), hsm.target("../empty"), hsm.effect(clearGraph)),
    hsm.transition(hsm.on("viewport.pan.start"), hsm.effect(beginPan)),
    hsm.transition(hsm.on("viewport.pan"), hsm.effect(applyViewport)),
    hsm.transition(hsm.on("viewport.pan.end"), hsm.guard(canEndPan), hsm.target("../drawing"), hsm.effect(endPan)),
    hsm.transition(hsm.on("viewport.pan.end"), hsm.effect(endPan)),
    hsm.transition(hsm.on("viewport.fit"), hsm.effect(applyViewport)),
    hsm.transition(hsm.on("viewport.focus"), hsm.effect(applyViewport)),
    hsm.transition(hsm.on("viewport.zoom"), hsm.effect(applyViewport)),
  ),
);

function phaseFromStatePath(statePath: string): MachineGraphPhase {
  if (statePath.endsWith("/drawing") || statePath.endsWith("/panning")) {
    return "drawing";
  }
  return "empty";
}

export function isMachineGraphEventName(value: string): value is MachineGraphEventName {
  return Object.hasOwn(graphEvents, value);
}

export class MachineGraphController {
  #runtime = new MachineGraphRuntime();
  #machine: MachineGraphRuntime;
  #graphs: readonly MachineGraph[] = [];
  #renderer: GraphRenderer | null;
  #onSnapshot: ((snapshot: MachineGraphSnapshot) => void) | null;
  #initialViewPending = false;
  #dispatchTail: Promise<void> = Promise.resolve();
  #stopping = false;
  #stopPromise: Promise<void> | null = null;
  #scale = 1;
  #pan: ViewportPoint = { x: 0, y: 0 };
  #focusedMachine: string | undefined;
  #hasRealDimensions = false;
  #pointers = new Map<number, ViewportPoint>();
  #dragStart: { pointerId: number; point: ViewportPoint; pan: ViewportPoint } | null = null;
  #pinchStart: { distance: number; scale: number } | null = null;

  constructor(options: MachineGraphControllerOptions = {}) {
    this.#renderer = options.renderer ?? null;
    this.#onSnapshot = options.onSnapshot ?? null;
    this.#runtime.controller = this;
    this.#machine = startMachine(this.#runtime, machineGraphModel);
  }

  snapshot(): MachineGraphSnapshot {
    const statePath = this.#machine.takeSnapshot().state;
    return {
      phase: phaseFromStatePath(statePath),
      statePath,
      graphs: this.#graphs,
    };
  }

  async dispatch(eventName: MachineGraphEventName, data?: unknown): Promise<MachineGraphSnapshot> {
    if (this.#stopping) {
      throw new Error("MachineGraphController is stopped");
    }
    const dispatch = this.#dispatchTail.then(() => this.#dispatchNow(eventName, data));
    this.#dispatchTail = dispatch.then(() => undefined, () => undefined);
    return dispatch;
  }

  async #dispatchNow(eventName: MachineGraphEventName, data?: unknown): Promise<MachineGraphSnapshot> {
    if (this.#stopping) {
      throw new Error("MachineGraphController is stopped");
    }
    let nextGraphs: MachineGraph[] | null = null;
    if (eventName === "graph.set") {
      nextGraphs = isRecord(data) ? parseGraphs(data["graphs"]) : null;
      if (nextGraphs === null || nextGraphs.length === 0) {
        await this.#machine.dispatch(namedEvent(graphEvents["graph.clear"].name));
        this.#emit();
        return this.snapshot();
      }
    }
    let focusRemoved = eventName === "graph.set"
      && this.#focusedMachine !== undefined
      && nextGraphs !== null
      && !nextGraphs.some((graph) => graph.name === this.#focusedMachine);
    if (focusRemoved) this.#focusedMachine = undefined;
    await this.#machine.dispatch(namedEvent(graphEvents[eventName].name, data));
    if (eventName === "graph.set" && !focusRemoved && this.#focusedMachine !== undefined) {
      const renderer = this.#renderer;
      if (renderer?.focusBounds !== undefined && renderer.focusBounds(this.#focusedMachine) === null) {
        this.#focusedMachine = undefined;
        focusRemoved = true;
      }
    }
    this.#emit();
    const shouldRefocus = eventName === "graph.set" && this.#focusedMachine !== undefined;
    if (this.#initialViewPending || shouldRefocus || focusRemoved) {
      this.#initialViewPending = false;
      const renderer = this.#renderer;
      queueMicrotask(() => {
        if (!this.#stopping && this.#renderer === renderer && renderer !== null) {
          const eventName = this.#focusedMachine === undefined ? "viewport.fit" : "viewport.focus";
          const data = this.#focusedMachine === undefined
            ? focusRemoved ? undefined : { reason: "initial" }
            : { machineName: this.#focusedMachine };
          void this.dispatch(eventName, data).catch(reportMachineGraphFailure);
        }
      });
    }
    return this.snapshot();
  }

  applyViewport(eventName: string, data: unknown): void {
    const renderer = this.#renderer;
    const metrics = renderer?.viewportMetrics === undefined ? undefined : renderer.viewportMetrics();
    if (eventName === "viewport.fit") {
      if (metrics === undefined) {
        renderer?.applyViewport?.(data);
        return;
      }
      if (metrics === null) return;
      const reason = recordOf(data)?.["reason"];
      if (reason === "resize" || reason === "initial") {
        const firstDimensions = !this.#hasRealDimensions;
        this.#hasRealDimensions = true;
        if (this.#focusedMachine !== undefined && this.#applyFocusedViewport(metrics, data)) return;
        if (reason === "initial" || firstDimensions) this.#fitBounds(metrics.bounds, metrics);
        return;
      }
      this.#focusedMachine = undefined;
      this.#hasRealDimensions = true;
      this.#fitBounds(metrics.bounds, metrics);
      return;
    }
    if (eventName === "viewport.focus") {
      if (metrics === undefined) {
        renderer?.applyViewport?.(data);
        return;
      }
      if (metrics !== null) this.#applyFocusedViewport(metrics, data);
      return;
    }
    if (eventName === "viewport.pan") {
      this.#applyPan(data);
      return;
    }
    if (eventName === "viewport.zoom") {
      this.#applyZoom(data);
    }
  }

  fit(): void {
    this.#focusedMachine = undefined;
    this.#requestViewport("viewport.fit");
  }

  focusMachine(machineName: string): boolean {
    const renderer = this.#renderer;
    if (renderer?.focusBounds === undefined || renderer.focusBounds(machineName) === null) {
      return false;
    }
    this.#focusedMachine = machineName;
    this.#requestViewport("viewport.focus", { machineName });
    return true;
  }

  stop(): Promise<void> {
    if (this.#stopPromise !== null) return this.#stopPromise;
    this.#stopping = true;
    this.#initialViewPending = false;
    this.#stopPromise = this.#finishStop();
    return this.#stopPromise;
  }

  async #finishStop(): Promise<void> {
    await this.#dispatchTail;
    this.destroyRenderer();
    this.#runtime.controller = null;
    await stopMachine(this.#machine);
  }

  rememberGraphs(graphs: readonly MachineGraph[]): void {
    this.#graphs = graphs;
    if (graphs.length === 0) this.#focusedMachine = undefined;
    this.#emit();
  }

  draw(preserveViewport = false): void {
    if (this.#graphs.length === 0) {
      this.#renderer?.destroy();
      this.#initialViewPending = false;
      return;
    }
    if (this.#renderer?.draw(this.#graphs) === true && !preserveViewport) {
      this.#initialViewPending = true;
    }
  }

  destroyRenderer(): void {
    this.#renderer?.destroy();
    this.#initialViewPending = false;
    this.#pointers.clear();
    this.#dragStart = null;
    this.#pinchStart = null;
    this.#applyPanning(false);
  }

  beginPan(data: unknown): void {
    const pointerId = pointerIdOf(data);
    const point = pointOf(recordOf(data)?.["point"]);
    if (pointerId === null || point === null) return;
    this.#pointers.set(pointerId, point);
    if (this.#pointers.size === 1) {
      this.#dragStart = { pointerId, point, pan: { ...this.#pan } };
      this.#applyPanning(true);
      return;
    }
    this.#dragStart = null;
    const points = [...this.#pointers.values()];
    const first = points[0];
    const second = points[1];
    if (first !== undefined && second !== undefined) {
      this.#pinchStart = {
        distance: Math.max(1, Math.hypot(second.x - first.x, second.y - first.y)),
        scale: this.#scale,
      };
    }
    this.#applyPanning(false);
  }

  canEndPan(data: unknown): boolean {
    const pointerId = pointerIdOf(data);
    return pointerId === null || (this.#pointers.has(pointerId) && this.#pointers.size <= 1);
  }

  endPan(data: unknown): void {
    const pointerId = pointerIdOf(data);
    if (pointerId !== null) this.#pointers.delete(pointerId);
    if (this.#pointers.size < 2) this.#pinchStart = null;
    if (this.#pointers.size === 0) this.#dragStart = null;
    this.#applyPanning(this.#dragStart !== null);
  }

  #requestViewport(eventName: MachineGraphEventName, data?: unknown): void {
    void this.dispatch(eventName, data).catch(reportMachineGraphFailure);
  }

  #applyFocusedViewport(metrics: ViewportMetrics, data: unknown): boolean {
    const record = recordOf(data);
    const machineName = typeof record?.["machineName"] === "string" ? record["machineName"] : this.#focusedMachine;
    const bounds = boundsOf(record?.["bounds"])
      ?? (machineName === undefined ? null : this.#renderer?.focusBounds?.(machineName) ?? null);
    if (bounds === null) return false;
    if (machineName !== undefined) this.#focusedMachine = machineName;
    this.#fitBounds(bounds, metrics);
    return true;
  }

  #fitBounds(bounds: ViewportBounds, metrics: ViewportMetrics): void {
    const width = Math.max(1, bounds.right - bounds.left + FIT_PADDING * 2);
    const height = Math.max(1, bounds.bottom - bounds.top + FIT_PADDING * 2);
    const scale = Math.min(
      MAX_FIT_ZOOM,
      Math.max(MIN_ZOOM, Math.min(metrics.width / width, metrics.height / height)),
    );
    const center = {
      x: (bounds.left + bounds.right) / 2 + metrics.origin.x,
      y: (bounds.top + bounds.bottom) / 2 + metrics.origin.y,
    };
    this.#setTransform(scale, {
      x: metrics.width / 2 - center.x * scale,
      y: metrics.height / 2 - center.y * scale,
    });
  }

  #applyPan(data: unknown): void {
    const record = recordOf(data);
    const directPan = pointOf(record?.["pan"]);
    if (directPan !== null) {
      const scale = typeof record?.["scale"] === "number" ? record["scale"] : this.#scale;
      this.#setTransform(scale, directPan);
      return;
    }
    const pointerId = pointerIdOf(data);
    const point = pointOf(record?.["point"]);
    if (pointerId === null || point === null) return;
    if (!this.#pointers.has(pointerId)) return;
    this.#pointers.set(pointerId, point);
    if (this.#pointers.size >= 2 && this.#pinchStart !== null) {
      const points = [...this.#pointers.values()];
      const first = points[0];
      const second = points[1];
      if (first !== undefined && second !== undefined) {
        const midpoint = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
        const scale = this.#pinchStart.scale
          * Math.hypot(second.x - first.x, second.y - first.y)
          / this.#pinchStart.distance;
        this.#setZoom(scale, midpoint);
      }
      return;
    }
    if (this.#dragStart?.pointerId === pointerId) {
      this.#setTransform(this.#scale, {
        x: this.#dragStart.pan.x + point.x - this.#dragStart.point.x,
        y: this.#dragStart.pan.y + point.y - this.#dragStart.point.y,
      });
    }
  }

  #applyZoom(data: unknown): void {
    const record = recordOf(data);
    const scale = typeof record?.["scale"] === "number"
      ? record["scale"]
      : typeof record?.["deltaY"] === "number" ? this.#scale * Math.exp(-record["deltaY"] * ZOOM_STEP) : null;
    if (scale === null) return;
    const point = pointOf(record?.["point"]);
    if (point === null) {
      this.#setTransform(scale, this.#pan);
      return;
    }
    this.#setZoom(scale, point);
  }

  #setZoom(scale: number, point: ViewportPoint): void {
    const worldPoint = {
      x: (point.x - this.#pan.x) / this.#scale,
      y: (point.y - this.#pan.y) / this.#scale,
    };
    this.#setTransform(scale, {
      x: point.x - worldPoint.x * scale,
      y: point.y - worldPoint.y * scale,
    });
  }

  #setTransform(scale: number, pan: ViewportPoint): void {
    this.#scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
    this.#pan = { ...pan };
    this.#renderer?.applyViewport?.({ scale: this.#scale, pan: this.#pan });
  }

  #applyPanning(panning: boolean): void {
    if (this.#renderer?.viewportMetrics !== undefined) {
      this.#renderer.applyViewport?.({ panning });
    }
  }

  #emit(): void {
    this.#onSnapshot?.(this.snapshot());
  }
}
