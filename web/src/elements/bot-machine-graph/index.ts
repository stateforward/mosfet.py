import * as hsm from "@stateforward/hsm.ts";

import { startFocuser, type FocusTarget, type Focuser } from "../../focuser-hsm.ts";
import { From, startMachine, stopMachine } from "../../hsm-runtime.ts";
import { Graph, reportMachineGraphFailure, startGraph } from "../../machine-graph-hsm.ts";
import { type MachineGraph } from "../../otel/machines.ts";
import { startPanner, type Panner } from "../../panner-hsm.ts";
import { startRenderer, type Renderer } from "../../renderer-hsm.ts";
import { applyStyles } from "../styles.ts";
import { NativeGraphRenderer, type GraphHit } from "./renderer.ts";
import { graphStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";

export type GraphZoomDetail = { zoom: number };
export type GraphEdgeDetail = { eventName: string };

export class BotMachineGraph extends From(HTMLElement) {
  static readonly model = hsm.define(
    "BotMachineGraph",
    hsm.initial(hsm.target("active")),
    hsm.state("active"),
  );

  readonly #root: ShadowRoot;
  readonly #frame: HTMLDivElement;
  #paint: NativeGraphRenderer | null = null;
  #graph: Graph | null = null;
  #renderer: Renderer | null = null;
  #panner: Panner | null = null;
  #focuser: Focuser | null = null;
  #pending: readonly MachineGraph[] | undefined;
  #pendingFocus: string | undefined;
  #focusTarget: FocusTarget | null = null;
  #connectionGeneration = 0;
  #resizeObserver: ResizeObserver | null = null;
  #clickCandidate: { pointerId: number; hit: GraphHit; x: number; y: number } | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, graphStyles);
    this.#frame = document.createElement("div");
    this.#frame.className = "frame";
    this.#frame.part.add("frame");
    this.#frame.setAttribute("data-testid", "frame");
    this.#root.append(this.#frame);
  }

  get graphs(): readonly MachineGraph[] {
    return this.#graph?.graphs ?? this.#pending ?? [];
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#pending = value;
    this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0)));
    this.#admit(value);
  }

  fit(): void {
    this.#pendingFocus = undefined;
    this.#focuser?.clear();
    this.#fitAdmitted();
  }

  focusMachine(machineName: string): boolean {
    const bounds = this.#paint?.focusBounds(machineName);
    if (bounds === null || bounds === undefined) return false;
    this.#pendingFocus = undefined;
    this.#focuser?.focus({ kind: "machine", machineName, bounds });
    return true;
  }

  connectedCallback(): void {
    const generation = ++this.#connectionGeneration;
    if (this.#paint === null) {
      this.#paint = new NativeGraphRenderer(this.#frame);
    }
    startMachine(this, BotMachineGraph.model);
    const ctx = this.context();
    const paint = this.#paint;
    this.#renderer = startRenderer(ctx, () => this.#paintNow());
    this.#panner = startPanner(ctx, paint.world, {
      frame: this.#frame,
      onTransform: (transform) => this.#emitZoom(transform.scale),
    });
    this.#focuser = startFocuser(ctx, this.#frame, {
      onFocus: (target) => {
        this.#focusTarget = target;
        this.#fitBounds(target.bounds);
      },
      onClear: () => {
        this.#focusTarget = null;
      },
    });
    this.#graph = startGraph(ctx, {
      onDraw: () => this.#renderer?.markDirty(),
      onDestroy: () => this.#paint?.destroy(),
    });
    paint.viewport.addEventListener("pointerdown", this.#onPointerDown);
    paint.viewport.addEventListener("pointermove", this.#onPointerMove);
    paint.viewport.addEventListener("pointerup", this.#onPointerUp, true);
    paint.viewport.addEventListener("pointercancel", this.#onPointerUp, true);
    paint.viewport.addEventListener("wheel", this.#onWheel, { passive: false });
    paint.viewport.addEventListener("click", this.#onClick);
    this.#ensureResizeObserver();
    if (generation === this.#connectionGeneration && this.#pending !== undefined) this.#admit(this.#pending);
  }

  disconnectedCallback(): void {
    this.#connectionGeneration += 1;
    this.#clickCandidate = null;
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    const paint = this.#paint;
    paint?.viewport.removeEventListener("pointerdown", this.#onPointerDown);
    paint?.viewport.removeEventListener("pointermove", this.#onPointerMove);
    paint?.viewport.removeEventListener("pointerup", this.#onPointerUp, true);
    paint?.viewport.removeEventListener("pointercancel", this.#onPointerUp, true);
    paint?.viewport.removeEventListener("wheel", this.#onWheel);
    paint?.viewport.removeEventListener("click", this.#onClick);
    const graph = this.#graph;
    const renderer = this.#renderer;
    const panner = this.#panner;
    const focuser = this.#focuser;
    this.#graph = null;
    this.#renderer = null;
    this.#panner = null;
    this.#focuser = null;
    this.#paint = null;
    this.#focusTarget = null;
    paint?.dispose();
    this.#frame.replaceChildren();
    void Promise.all([
      graph === null ? undefined : stopMachine(graph),
      renderer === null ? undefined : stopMachine(renderer),
      panner === null ? undefined : stopMachine(panner),
      focuser === null ? undefined : stopMachine(focuser),
      stopMachine(this),
    ]).catch(reportMachineGraphFailure);
  }

  #admit(value: readonly MachineGraph[]): void {
    const graph = this.#graph;
    if (graph === null) return;
    graph.setGraphs(value);
    if (this.#pendingFocus !== undefined && this.focusMachine(this.#pendingFocus)) {
      this.#pendingFocus = undefined;
    }
  }

  #paintNow(): void {
    const paint = this.#paint;
    const graph = this.#graph;
    if (paint === null || graph === null) return;
    const structureChanged = paint.draw(graph.graphs);
    const focus = this.#focusTarget;
    if (focus?.kind === "machine" && focus.machineName !== undefined) {
      const bounds = paint.focusBounds(focus.machineName);
      if (bounds === null) {
        this.#focuser?.clear();
        this.#fitAdmitted();
        return;
      }
      this.#fitBounds(bounds);
      return;
    }
    if (structureChanged && focus === null) this.#fitAdmitted();
  }

  #fitAdmitted(): void {
    const metrics = this.#paint?.viewportMetrics();
    if (metrics === undefined || metrics === null) return;
    this.#panner?.fit({ bounds: metrics.bounds, metrics });
  }

  #fitBounds(bounds: FocusTarget["bounds"]): void {
    const metrics = this.#paint?.viewportMetrics();
    if (metrics === undefined || metrics === null) return;
    this.#panner?.fit({ bounds, metrics });
  }

  #emitZoom(zoom: number): void {
    this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
      detail: { zoom },
      bubbles: true,
      composed: true,
    }));
  }

  #emitEdge(eventName: string): void {
    this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
      detail: { eventName },
      bubbles: true,
      composed: true,
    }));
  }

  #pointerPoint(event: PointerEvent | WheelEvent): { x: number; y: number } {
    const rect = this.#paint?.viewport.getBoundingClientRect();
    return {
      x: event.clientX - (rect?.left ?? 0),
      y: event.clientY - (rect?.top ?? 0),
    };
  }

  #onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    const paint = this.#paint;
    const panner = this.#panner;
    if (paint === null || panner === null) return;
    const hit = paint.hitTestNode(event);
    this.#clickCandidate = hit === null
      ? null
      : { pointerId: event.pointerId, hit, x: event.clientX, y: event.clientY };
    if (hit === null) paint.viewport.setPointerCapture(event.pointerId);
    panner.panStart({ pointerId: event.pointerId, point: this.#pointerPoint(event) });
  };

  #onPointerMove = (event: PointerEvent): void => {
    const paint = this.#paint;
    const panner = this.#panner;
    if (paint === null || panner === null) return;
    const candidate = this.#clickCandidate;
    if (candidate?.pointerId === event.pointerId) {
      if (Math.hypot(event.clientX - candidate.x, event.clientY - candidate.y) <= 4) return;
      this.#clickCandidate = null;
      paint.viewport.setPointerCapture(event.pointerId);
    }
    if (event.buttons === 0 && !paint.viewport.hasPointerCapture(event.pointerId)) return;
    if (event.buttons !== 0 && !paint.viewport.hasPointerCapture(event.pointerId)) {
      paint.viewport.setPointerCapture(event.pointerId);
    }
    panner.cursorMove({ pointerId: event.pointerId, point: this.#pointerPoint(event) });
  };

  #onPointerUp = (event: PointerEvent): void => {
    const candidate = this.#clickCandidate?.pointerId === event.pointerId ? this.#clickCandidate : null;
    this.#clickCandidate = null;
    this.#panner?.panEnd({ pointerId: event.pointerId });
    if (event.type === "pointerup" && candidate !== null) {
      this.#focuser?.focus({
        kind: "node",
        nodePath: candidate.hit.path,
        machineName: candidate.hit.machineName,
        bounds: candidate.hit.bounds,
      });
    }
  };

  #onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.#panner?.zoom({ deltaY: event.deltaY, point: this.#pointerPoint(event) });
  };

  #onClick = (event: MouseEvent): void => {
    const paint = this.#paint;
    if (paint === null) return;
    const eventName = paint.hitTestEdge(event);
    if (eventName !== null) this.#emitEdge(eventName);
    if (event.detail > 0) return;
    const hit = paint.hitTestNode(event);
    if (hit === null) return;
    this.#focuser?.focus({
      kind: "node",
      nodePath: hit.path,
      machineName: hit.machineName,
      bounds: hit.bounds,
    });
  };

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return;
    this.#resizeObserver = new ResizeObserver(() => {
      if (this.#focusTarget !== null) return;
      this.#fitAdmitted();
    });
    this.#resizeObserver.observe(this.#frame);
  }
}

export function registerBotMachineGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, BotMachineGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-machine-graph": BotMachineGraph;
  }
}
