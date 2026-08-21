import * as hsm from "../../hsm.ts";

import { FlowGraph, type NodeClickDetail } from "../../flow/index.ts";
import { copyGraphs, Graph, graphsFromEvent } from "../../machine-graph.ts";
import { type MachineGraph } from "../../otel/machines.ts";
import { replaceStyles } from "../styles.ts";
import { flowModelFromGraphs, focusBoundsForMachine, type FlowGraphModel } from "./flow-model.ts";
import { graphStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";
/** Public attribute. This host is the only writer; value is `graphNodeCount(#held)`. */
const NODE_COUNT_ATTR = "data-node-count";

/**
 * Detail of the `bot-machine-graph-zoom` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Side-effect owner: the listener; the dispatcher does not interpret
 * `preventDefault()` and the event cannot be canceled.
 */
export type GraphZoomDetail = { zoom: number };
/**
 * Detail of the `bot-machine-graph-edge` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Side-effect owner: the listener; the dispatcher does not interpret
 * `preventDefault()` and the event cannot be canceled.
 */
export type GraphEdgeDetail = { eventName: string };

type GraphsAdmitData = { readonly graphs: readonly MachineGraph[] };
type FocusData = { readonly machineName: string };

export class BotMachineGraph extends hsm.from(HTMLElement) {
  static readonly attachEvent = { name: "host_attach", kind: hsm.Kinds.Event } as const;
  static readonly detachEvent = { name: "host_detach", kind: hsm.Kinds.Event } as const;
  static readonly stoppedEvent = { name: "host_stopped", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly graphsEvent = { name: "graphs_admit", kind: hsm.Kinds.Event } as const;
  static readonly focusEvent = { name: "focus_machine", kind: hsm.Kinds.Event } as const;
  static readonly fitEvent = { name: "fit_view", kind: hsm.Kinds.Event } as const;
  static readonly nodeClickEvent = { name: "node_click", kind: hsm.Kinds.Event } as const;
  static readonly resizeEvent = { name: "host_resize", kind: hsm.Kinds.Event } as const;
  static readonly drawnAppliedEvent = { name: "drawn_applied", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly fitDoneEvent = { name: "fit_done", kind: hsm.Kinds.CompletionEvent } as const;

  static readonly model = hsm.define(
    "BotMachineGraph",
    hsm.initial(hsm.target("disconnected")),
    hsm.state(
      "disconnected",
      hsm.defer(BotMachineGraph.graphsEvent.name),
      hsm.defer(BotMachineGraph.focusEvent.name),
      hsm.defer(BotMachineGraph.fitEvent.name),
      hsm.defer(BotMachineGraph.nodeClickEvent.name),
      hsm.transition(hsm.on(BotMachineGraph.attachEvent.name), hsm.target("../connected")),
    ),
    hsm.state(
      "connected",
      hsm.initial(hsm.target("ready")),
      hsm.entry(BotMachineGraph.onConnected),
      hsm.exit(BotMachineGraph.onConnectedExit),
      hsm.transition(hsm.on(BotMachineGraph.detachEvent.name), hsm.target("../stopping")),
      hsm.transition(hsm.on(BotMachineGraph.graphsEvent.name), hsm.effect(BotMachineGraph.admitGraphs)),
      hsm.transition(hsm.on(Graph.drawnEvent.name), hsm.target("afterDraw")),
      hsm.transition(hsm.on(Graph.clearedEvent.name), hsm.effect(BotMachineGraph.applyCleared)),
      hsm.transition(hsm.on(BotMachineGraph.focusEvent.name), hsm.effect(BotMachineGraph.applyFocus)),
      hsm.transition(hsm.on(BotMachineGraph.fitEvent.name), hsm.effect(BotMachineGraph.applyFit)),
      hsm.transition(hsm.on(BotMachineGraph.nodeClickEvent.name), hsm.effect(BotMachineGraph.applyNodeClick)),
      hsm.transition(hsm.on(BotMachineGraph.resizeEvent.name), hsm.effect(BotMachineGraph.applyFit)),
      hsm.state("ready"),
      hsm.choice(
        "afterDraw",
        hsm.transition(hsm.guard(BotMachineGraph.needsFit), hsm.target("drawingFit")),
        hsm.transition(hsm.target("drawingSkip")),
      ),
      hsm.state(
        "drawingSkip",
        hsm.entry(BotMachineGraph.applyDrawnThenSignal),
        hsm.transition(hsm.on(BotMachineGraph.drawnAppliedEvent.name), hsm.target("../ready")),
      ),
      hsm.state(
        "drawingFit",
        hsm.entry(BotMachineGraph.applyDrawnThenSignal),
        hsm.transition(hsm.on(BotMachineGraph.drawnAppliedEvent.name), hsm.target("../fitting")),
      ),
      hsm.state(
        "fitting",
        hsm.entry(BotMachineGraph.applyFitThenSignal),
        hsm.transition(hsm.on(BotMachineGraph.fitDoneEvent.name), hsm.target("../ready")),
      ),
    ),
    hsm.state(
      "stopping",
      hsm.defer(BotMachineGraph.attachEvent.name),
      hsm.defer(BotMachineGraph.graphsEvent.name),
      hsm.defer(BotMachineGraph.focusEvent.name),
      hsm.defer(BotMachineGraph.nodeClickEvent.name),
      hsm.activity(BotMachineGraph.stopActors),
      hsm.transition(
        hsm.on(BotMachineGraph.stoppedEvent.name),
        hsm.target("../disconnected"),
        hsm.effect(BotMachineGraph.clearActors),
      ),
    ),
  );

  readonly #root: ShadowRoot;
  readonly #flow: FlowGraph;
  #graph: Graph | null = null;
  #model: FlowGraphModel | null = null;
  #held: readonly MachineGraph[] = [];
  #resizeObserver: ResizeObserver | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    replaceStyles(this.#root, `:host { display: block; width: 100%; height: 100%; min-height: 16rem; }`);
    this.#flow = document.createElement("flow-graph");
    this.#flow.adoptStyles(graphStyles);
    this.#flow.style.width = "100%";
    this.#flow.style.height = "100%";
    this.#flow.setAttribute("data-testid", "frame");
    this.#flow.part.add("frame");
    const background = document.createElement("flow-background");
    const controls = document.createElement("flow-controls");
    this.#flow.append(background, controls);
    this.#root.append(this.#flow);
  }

  get graphs(): readonly MachineGraph[] {
    return copyGraphs(this.#held);
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#live(hsm.typedEvent({ event: BotMachineGraph.graphsEvent, data: { graphs: [...value] } satisfies GraphsAdmitData }));
  }

  fit(): void {
    this.#live(hsm.typedEvent({ event: BotMachineGraph.fitEvent }));
  }

  focusMachine(machineName: string): boolean {
    const model = this.#model ?? flowModelFromGraphs(this.#held);
    const bounds = focusBoundsForMachine(this.#held, machineName, model);
    if (bounds === null) return false;
    this.#live(hsm.typedEvent({ event: BotMachineGraph.focusEvent, data: { machineName } satisfies FocusData }));
    return true;
  }

  connectedCallback(): void {
    hsm.start(this, BotMachineGraph.model);
    this.#live(hsm.typedEvent({ event: BotMachineGraph.attachEvent }));
  }

  disconnectedCallback(): void {
    this.#live(hsm.typedEvent({
      event: BotMachineGraph.detachEvent,
      data: { graph: this.#graph },
    }));
  }

  #live(event: hsm.DispatchEvent): void {
    void this.dispatch(event).catch(hsm.catchFailure(this));
  }

  static onConnected(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    const ctx = instance.context();
    instance.#flow.nodesDraggable = false;
    instance.#flow.panOnDrag = true;
    instance.#graph = hsm.start(ctx, new Graph(), Graph.model);
    instance.addEventListener("flow-node-click", instance.#onNodeClick);
    instance.addEventListener("flow-edge-click", instance.#onEdgeClick);
    instance.addEventListener("flow-viewport-change", instance.#onViewport);
    instance.#ensureResizeObserver();
  }

  static onConnectedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#resizeObserver?.disconnect();
    instance.#resizeObserver = null;
    instance.removeEventListener("flow-node-click", instance.#onNodeClick);
    instance.removeEventListener("flow-edge-click", instance.#onEdgeClick);
    instance.removeEventListener("flow-viewport-change", instance.#onViewport);
  }

  static async stopActors(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
    if (!(instance instanceof BotMachineGraph)) return;
    const graph = graphFromEvent(event);
    if (graph !== null) await hsm.stop(graph);
    await instance.dispatch(hsm.typedEvent({ event: BotMachineGraph.stoppedEvent }));
  }

  static clearActors(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#graph = null;
    instance.#model = null;
  }

  static admitGraphs(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph) || instance.#graph === null || !hsm.isRecord(event.data)) return;
    void instance.#graph.dispatch(hsm.typedEvent({ event: Graph.setEvent, data: { graphs: event.data["graphs"] } })).catch(hsm.catchFailure(instance));
  }

  static needsFit(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof BotMachineGraph)) return false;
    const graphs = graphsFromEvent(event);
    if (graphs === null) return false;
    if (instance.#model === null) return true;
    return graphNodeCount(instance.#held) !== graphNodeCount(graphs);
  }

  static applyDrawnThenSignal(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    BotMachineGraph.applyDrawn(ctx, instance, event);
    void instance.dispatch(hsm.typedEvent({ event: BotMachineGraph.drawnAppliedEvent })).catch(hsm.catchFailure(instance));
  }

  static applyFitThenSignal(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    BotMachineGraph.applyFit(ctx, instance, event);
    void instance.dispatch(hsm.typedEvent({ event: BotMachineGraph.fitDoneEvent })).catch(hsm.catchFailure(instance));
  }

  static applyDrawn(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    const graphs = graphsFromEvent(event);
    if (graphs === null) return;
    instance.#applyDrawn(graphs);
  }

  static applyCleared(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#held = [];
    instance.#model = null;
    instance.#flow.nodes = [];
    instance.#flow.edges = [];
    instance.setAttribute(NODE_COUNT_ATTR, String(graphNodeCount(instance.#held)));
  }

  static applyFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph) || !hsm.isRecord(event.data)) return;
    const machineName = event.data["machineName"];
    if (typeof machineName !== "string") return;
    const model = instance.#model ?? flowModelFromGraphs(instance.#held);
    const bounds = focusBoundsForMachine(instance.#held, machineName, model);
    if (bounds === null) return;
    instance.#flow.fitBounds(bounds);
    instance.#flow.focusTarget({ kind: "machine", machineName, bounds });
  }

  static applyFit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#flow.fitView();
  }

  static applyNodeClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph) || !hsm.isRecord(event.data)) return;
    const machineName = event.data["machineName"];
    const path = event.data["path"];
    const bounds = boundsOf(event.data["bounds"]);
    if (typeof machineName !== "string" || bounds === null) return;
    instance.#flow.fitBounds(bounds);
    instance.#flow.focusTarget({
      kind: "node",
      machineName,
      bounds,
      ...(typeof path === "string" ? { nodePath: path } : {}),
    });
  }

  #applyDrawn(graphs: readonly MachineGraph[]): void {
    this.#held = copyGraphs(graphs);
    const model = flowModelFromGraphs(this.#held);
    this.#model = model;
    this.#flow.nodes = model.nodes;
    this.#flow.edges = model.edges;
    this.setAttribute(NODE_COUNT_ATTR, String(graphNodeCount(this.#held)));
  }

  #onNodeClick = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const node = nodeFromClickDetail(event.detail);
    if (node === null) return;
    const path = node.data["path"];
    const machineName = node.data["machineName"];
    if (typeof path !== "string" || typeof machineName !== "string") return;
    this.#live(hsm.typedEvent({ event: BotMachineGraph.nodeClickEvent, data: {
      machineName,
      path,
      bounds: {
        left: node.position.x,
        right: node.position.x + (node.width ?? 0),
        top: node.position.y,
        bottom: node.position.y + (node.height ?? 0),
      },
    } }));
  };

  #onEdgeClick = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const eventName = eventNameFromEdgeDetail(event.detail);
    if (eventName === null) return;
    this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
      detail: { eventName },
      bubbles: true,
      composed: true,
      cancelable: false,
    }));
  };

  #onViewport = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const detail = event.detail;
    if (!hsm.isRecord(detail) || !hsm.isRecord(detail["viewport"])) return;
    const viewport = detail["viewport"];
    const x = viewport["x"];
    const y = viewport["y"];
    const zoom = viewport["zoom"];
    if (typeof x !== "number" || typeof y !== "number" || typeof zoom !== "number") return;
    this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
      detail: { zoom },
      bubbles: true,
      composed: true,
      cancelable: false,
    }));
  };

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return;
    this.#resizeObserver = new ResizeObserver(() => this.#live(hsm.typedEvent({ event: BotMachineGraph.resizeEvent })));
    this.#resizeObserver.observe(this);
  }
}

export function registerBotMachineGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, BotMachineGraph);
}

function nodeFromClickDetail(value: unknown): NodeClickDetail["node"] | null {
  if (!hsm.isRecord(value) || !hsm.isRecord(value["node"])) return null;
  const node = value["node"];
  if (typeof node["id"] !== "string" || !hsm.isRecord(node["position"]) || !hsm.isRecord(node["data"])) return null;
  const x = node["position"]["x"];
  const y = node["position"]["y"];
  if (typeof x !== "number" || typeof y !== "number") return null;
  return node as NodeClickDetail["node"];
}

function eventNameFromEdgeDetail(value: unknown): string | null {
  if (!hsm.isRecord(value) || !hsm.isRecord(value["edge"])) return null;
  const data = value["edge"]["data"];
  if (!hsm.isRecord(data) || typeof data["eventName"] !== "string" || data["eventName"].length === 0) return null;
  return data["eventName"];
}

function graphFromEvent(event: hsm.Event): Graph | null {
  if (!hsm.isRecord(event.data) || !(event.data["graph"] instanceof Graph)) return null;
  return event.data["graph"];
}

function graphNodeCount(graphs: readonly MachineGraph[]): number {
  let count = 0;
  for (const graph of graphs) count += graph.nodes.length;
  return count;
}

function boundsOf(value: unknown): { left: number; right: number; top: number; bottom: number } | null {
  if (!hsm.isRecord(value)) return null;
  const left = value["left"];
  const right = value["right"];
  const top = value["top"];
  const bottom = value["bottom"];
  return typeof left === "number" && Number.isFinite(left)
    && typeof right === "number" && Number.isFinite(right)
    && typeof top === "number" && Number.isFinite(top)
    && typeof bottom === "number" && Number.isFinite(bottom)
    ? { left, right, top, bottom }
    : null;
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-machine-graph": BotMachineGraph;
  }
}
