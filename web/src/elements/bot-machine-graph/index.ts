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
const NESTED_NODES_DRAGGABLE = false;
const NESTED_PAN_ON_DRAG = true;

/**
 * Detail of the `bot-machine-graph-zoom` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the viewport zoom is already applied on the inner graph; the
 * host only echoes the committed `zoom` out, so listeners observe and cannot
 * intervene (`preventDefault()` has no effect, the event cannot be canceled).
 */
export type GraphZoomDetail = { readonly zoom: number };
/**
 * Detail of the `bot-machine-graph-edge` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the click did not mutate the graph. The host only re-exposes
 * the clicked edge's `eventName` from its detail; selecting or following the
 * edge is the listener's job, if it chooses to. `preventDefault()` has no
 * effect because the event cannot be canceled.
 */
export type GraphEdgeDetail = { readonly eventName: string };

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
      hsm.transition(
        hsm.on(BotMachineGraph.focusEvent.name),
        hsm.guard(BotMachineGraph.machineFocusable),
        hsm.effect(BotMachineGraph.applyFocus),
      ),
      hsm.transition(
        hsm.on(BotMachineGraph.focusEvent.name),
        hsm.effect(BotMachineGraph.ignoreFocus),
      ),
      hsm.transition(hsm.on(BotMachineGraph.fitEvent.name), hsm.effect(BotMachineGraph.applyFit)),
      hsm.transition(hsm.on(BotMachineGraph.nodeClickEvent.name), hsm.effect(BotMachineGraph.applyNodeClick)),
      hsm.transition(hsm.on(BotMachineGraph.resizeEvent.name), hsm.effect(BotMachineGraph.applyFit)),
      hsm.state("ready"),
      hsm.choice(
        "afterDraw",
        hsm.transition(hsm.guard(BotMachineGraph.needsFit), hsm.target("drawing/fit")),
        hsm.transition(hsm.target("drawing/skip")),
      ),
      hsm.state(
        "drawing",
        hsm.entry(BotMachineGraph.applyDrawnThenSignal),
        hsm.initial(hsm.target("skip")),
        hsm.state(
          "skip",
          hsm.transition(hsm.on(BotMachineGraph.drawnAppliedEvent.name), hsm.target("../../ready")),
        ),
        hsm.state(
          "fit",
          hsm.transition(hsm.on(BotMachineGraph.drawnAppliedEvent.name), hsm.target("../../fitting")),
        ),
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
  #graphsWrite: readonly MachineGraph[] | undefined;
  #focusableNames: ReadonlySet<string> = new Set();
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
    this.#flow.nodesDraggable = NESTED_NODES_DRAGGABLE;
    this.#flow.panOnDrag = NESTED_PAN_ON_DRAG;
    this.#root.append(this.#flow);
  }

  get graphs(): readonly MachineGraph[] {
    return copyGraphs(this.#held);
  }

  /**
   * Stage a copy of `value` and admit it when this host is started.
   *
   * Inputs: caller `value`. Copied with `copyGraphs` at write time; later
   * mutation of the caller array or of graph items (identity fields, `nodes`,
   * `edges`) does not change staged or admitted graphs. Graph items are copied,
   * not shared. Before connect this is a write buffer replayed from
   * `connectedCallback` after `start`; that staging is pre-start local state,
   * not a dropped dispatch, and emits no host-drop. After start, `graphs_admit`
   * is dispatched with the copy. After stop, including while `stop()` is in
   * flight, this setter emits `host-drop` with reason `"stopped"` and does not
   * dispatch or retain `value` (the write buffer is unchanged; a later `start`
   * from `connectedCallback` replays the last staged write, not the dropped
   * one). Does not call `start`.
   * Outputs: getter returns copies of admitted graphs.
   * Ownership: this host owns the copy. Lifetime: until the next staged graphs
   * write. Stopped writes do not replace the buffer. Concurrency: runtime-safe.
   * Failure modes: unstarted staging is not a failure; stopped writes emit
   * `host-drop` with reason `"stopped"` and are not retained.
   * Classification: runtime-safe.
   */
  set graphs(value: readonly MachineGraph[]) {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: BotMachineGraph.graphsEvent.name });
      return;
    }
    const graphs = copyGraphs(value);
    this.#graphsWrite = graphs;
    if (!hsm.hostWasStarted(this)) return;
    this.#live(hsm.typedEvent({ event: BotMachineGraph.graphsEvent, data: { graphs } satisfies GraphsAdmitData }));
  }

  /**
   * Dispatch `fit_view`.
   *
   * Inputs: none. Outputs: void; does not report whether a fit ran.
   * Ownership: this host owns the dispatch; the nested `flow-graph` applies
   * the fit.
   * Lifetime: unstarted hosts surface host-drop through `catchFailure(this)`.
   * After stop, including while `stop()` is in flight, this method emits
   * `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; the viewport is unchanged.
   * Units: none. Classification: runtime-safe.
   */
  fit(): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: BotMachineGraph.fitEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: BotMachineGraph.fitEvent }));
  }

  /**
   * Dispatch `focus_machine` with `machineName`.
   *
   * Inputs: `machineName` is the machine path to focus. Outputs: void; does
   * not report whether a machine exists.
   * Ownership: this host owns the dispatch; `machineFocusable` applies or the
   * unguarded ignore keeps the current state. Lifetime: safe after
   * `connectedCallback`/`start`; unstarted hosts surface host-drop through
   * `catchFailure(this)`. After stop, including while `stop()` is in flight,
   * this method emits `host-drop` with reason `"stopped"` and does not dispatch.
   * Concurrency: `#live` queues overlapping calls as HSM events.
   * Failure modes: unstarted and stopped hosts emit `host-drop` with reason
   * `"unstarted"` or `"stopped"`; a missing machine or a held machine without
   * painted fit bounds takes the unguarded `focus_machine` ignore (no
   * `fitBounds`, `data-node-count` unchanged).
   * Callers observe `data-node-count` and viewport/`fitBounds` effects rather
   * than a boolean return.
   * Classification: runtime-safe.
   */
  focusMachine(machineName: string): void {
    if (hsm.hostWasStopped(this)) {
      this.#dropStopped({ operation: BotMachineGraph.focusEvent.name });
      return;
    }
    this.#live(hsm.typedEvent({ event: BotMachineGraph.focusEvent, data: { machineName } satisfies FocusData }));
  }

  connectedCallback(): void {
    // `role` is set on connect, never in the constructor: a constructor that
    // sets attributes makes real browsers refuse to upgrade
    // `document.createElement` results.
    if (this.getAttribute("role") === null) this.setAttribute("role", "presentation");
    hsm.start({ instance: this, model: BotMachineGraph.model });
    hsm.start({ ctx: this.context(), instance: this.#flow, model: FlowGraph.model });
    if (this.#graphsWrite !== undefined) {
      this.#live(hsm.typedEvent({
        event: BotMachineGraph.graphsEvent,
        data: { graphs: this.#graphsWrite } satisfies GraphsAdmitData,
      }));
    }
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

  #dropStopped(args: { operation: string }): void {
    hsm.catchFailure(this)(new hsm.HostDropError({ reason: "stopped", operation: args.operation }));
  }

  static onConnected(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    const ctx = instance.context();
    instance.#graph = hsm.start({ ctx, instance: new Graph(), model: Graph.model });
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
    const incoming = eventGraphNodeCount(event);
    if (incoming === null) return false;
    if (instance.#model === null) return true;
    return graphNodeCount(instance.#held) !== incoming;
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
    instance.#focusableNames = new Set();
    instance.#flow.nodes = [];
    instance.#flow.edges = [];
    instance.setAttribute(NODE_COUNT_ATTR, String(graphNodeCount(instance.#held)));
  }

  static ignoreFocus(_ctx: hsm.Context, _instance: hsm.Instance, _event: hsm.Event): void {
    return;
  }

  static machineFocusable(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof BotMachineGraph) || !hsm.isRecord(event.data)) return false;
    const machineName = event.data["machineName"];
    if (typeof machineName !== "string" || machineName.length === 0) return false;
    return instance.#focusableNames.has(machineName);
  }

  static applyFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) {
      throw new TypeError("focus_machine taken without graph");
    }
    if (!hsm.isRecord(event.data) || typeof event.data["machineName"] !== "string") {
      throw new TypeError("focus_machine taken without machineName");
    }
    const machineName = event.data["machineName"];
    const model = instance.#model ?? flowModelFromGraphs(instance.#held);
    const bounds = focusBoundsForMachine({ graphs: instance.#held, machineName, model });
    if (bounds === null) {
      throw new TypeError("focus_machine taken without focus bounds");
    }
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
    this.#focusableNames = focusableMachineNames({ graphs: this.#held, model });
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

function focusableMachineNames(args: {
  graphs: readonly MachineGraph[];
  model: FlowGraphModel;
}): ReadonlySet<string> {
  const names = new Set<string>();
  for (const graph of args.graphs) {
    if (focusBoundsForMachine({ graphs: args.graphs, machineName: graph.name, model: args.model }) !== null) {
      names.add(graph.name);
    }
  }
  return names;
}

function graphNodeCount(graphs: readonly MachineGraph[]): number {
  let count = 0;
  for (const graph of graphs) count += graph.nodes.length;
  return count;
}

function eventGraphNodeCount(event: hsm.Event): number | null {
  if (!hsm.isRecord(event.data) || !Array.isArray(event.data["graphs"])) return null;
  let count = 0;
  for (const graph of event.data["graphs"]) {
    if (!hsm.isRecord(graph) || !Array.isArray(graph["nodes"])) return null;
    count += graph["nodes"].length;
  }
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
