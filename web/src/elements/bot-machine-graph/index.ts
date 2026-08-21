import * as hsm from "../../hsm.ts";

import { FlowGraph, type EdgeClickDetail, type NodeClickDetail, type ViewportChangeDetail } from "../../flow/index.ts";
import { Graph } from "../../machine-graph.ts";
import { type MachineGraph } from "../../otel/machines.ts";
import { replaceStyles } from "../styles.ts";
import { flowModelFromGraphs, focusBoundsForMachine, type FlowGraphModel } from "./flow-model.ts";
import { graphStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";

export type GraphZoomDetail = { zoom: number };
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
  static readonly resizeEvent = { name: "host_resize", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "BotMachineGraph",
    hsm.initial(hsm.target("disconnected")),
    hsm.state(
      "disconnected",
      hsm.defer(BotMachineGraph.graphsEvent.name),
      hsm.defer(BotMachineGraph.focusEvent.name),
      hsm.defer(BotMachineGraph.fitEvent.name),
      hsm.transition(hsm.on(BotMachineGraph.attachEvent.name), hsm.target("../connected")),
    ),
    hsm.state(
      "connected",
      hsm.entry(BotMachineGraph.onConnected),
      hsm.exit(BotMachineGraph.onConnectedExit),
      hsm.transition(hsm.on(BotMachineGraph.detachEvent.name), hsm.target("../stopping")),
      hsm.transition(hsm.on(BotMachineGraph.graphsEvent.name), hsm.effect(BotMachineGraph.admitGraphs)),
      hsm.transition(hsm.on(BotMachineGraph.focusEvent.name), hsm.effect(BotMachineGraph.applyFocus)),
      hsm.transition(hsm.on(BotMachineGraph.fitEvent.name), hsm.effect(BotMachineGraph.applyFit)),
      hsm.transition(hsm.on(BotMachineGraph.resizeEvent.name), hsm.effect(BotMachineGraph.applyFit)),
    ),
    hsm.state(
      "stopping",
      hsm.defer(BotMachineGraph.attachEvent.name),
      hsm.defer(BotMachineGraph.graphsEvent.name),
      hsm.defer(BotMachineGraph.focusEvent.name),
      hsm.activity(BotMachineGraph.stopActors),
      hsm.transition(hsm.on(BotMachineGraph.stoppedEvent.name), hsm.target("../disconnected")),
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
    this.#flow.nodesDraggable = false;
    this.#flow.panOnDrag = true;
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
    return this.#graph?.graphs ?? this.#held;
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#held = value;
    this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0)));
    this.#live(hsm.typedEvent(BotMachineGraph.graphsEvent, { graphs: value } satisfies GraphsAdmitData));
  }

  fit(): void {
    this.#live(hsm.typedEvent(BotMachineGraph.fitEvent));
  }

  focusMachine(machineName: string): boolean {
    const model = this.#model ?? flowModelFromGraphs(this.graphs);
    const bounds = focusBoundsForMachine(this.graphs, machineName, model);
    if (bounds === null) return false;
    this.#live(hsm.typedEvent(BotMachineGraph.focusEvent, { machineName } satisfies FocusData));
    return true;
  }

  connectedCallback(): void {
    hsm.start(this, BotMachineGraph.model);
    this.#live(hsm.typedEvent(BotMachineGraph.attachEvent));
  }

  disconnectedCallback(): void {
    this.#live(hsm.typedEvent(BotMachineGraph.detachEvent));
  }

  #live(event: hsm.DispatchEvent): void {
    try {
      this.dispatch(event);
    } catch (error) {
      hsm.catchFailure(this)(error);
    }
  }

  static onConnected(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    const ctx = instance.context();
    instance.#graph = hsm.start(ctx, new Graph({
      onDraw: (graphs) => instance.#draw(graphs),
      onDestroy: () => {
        instance.#flow.nodes = [];
        instance.#flow.edges = [];
        instance.#model = null;
      },
    }), Graph.model);
    instance.#flow.addEventListener("flow-node-click", instance.#onNodeClick);
    instance.#flow.addEventListener("flow-edge-click", instance.#onEdgeClick);
    instance.#flow.addEventListener("flow-viewport-change", instance.#onViewport);
    instance.#ensureResizeObserver();
  }

  static onConnectedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#resizeObserver?.disconnect();
    instance.#resizeObserver = null;
    instance.#flow.removeEventListener("flow-node-click", instance.#onNodeClick);
    instance.#flow.removeEventListener("flow-edge-click", instance.#onEdgeClick);
    instance.#flow.removeEventListener("flow-viewport-change", instance.#onViewport);
  }

  static async stopActors(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
    if (!(instance instanceof BotMachineGraph) || ctx.done) return;
    const graph = instance.#graph;
    instance.#graph = null;
    instance.#model = null;
    if (graph !== null) await hsm.stop(graph);
    if (ctx.done) return;
    instance.dispatch(hsm.typedEvent(BotMachineGraph.stoppedEvent));
  }

  static admitGraphs(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph) || instance.#graph === null || !hsm.isRecord(event.data)) return;
    const graphs = event.data["graphs"];
    if (!Array.isArray(graphs)) return;
    instance.#graph.setGraphs(graphs);
  }

  static applyFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph) || !hsm.isRecord(event.data)) return;
    const machineName = event.data["machineName"];
    if (typeof machineName !== "string") return;
    const model = instance.#model ?? flowModelFromGraphs(instance.graphs);
    const bounds = focusBoundsForMachine(instance.graphs, machineName, model);
    if (bounds === null) return;
    instance.#flow.fitBounds(bounds);
    instance.#flow.focusTarget({ kind: "machine", machineName, bounds });
  }

  static applyFit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof BotMachineGraph)) return;
    instance.#flow.fitView();
  }

  #draw(graphs: readonly MachineGraph[]): void {
    const previous = this.#model;
    const model = flowModelFromGraphs(graphs);
    this.#model = model;
    this.#flow.nodes = model.nodes;
    this.#flow.edges = model.edges;
    if (previous === null || previous.nodes.length !== model.nodes.length) {
      this.#flow.fitView();
    }
  }

  #onNodeClick = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const detail = event.detail as NodeClickDetail | undefined;
    const node = detail?.node;
    if (node === undefined) return;
    const path = node.data["path"];
    const machineName = node.data["machineName"];
    if (typeof path !== "string" || typeof machineName !== "string") return;
    this.#flow.fitBounds({
      left: node.position.x,
      right: node.position.x + (node.width ?? 0),
      top: node.position.y,
      bottom: node.position.y + (node.height ?? 0),
    });
  };

  #onEdgeClick = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const detail = event.detail as EdgeClickDetail | undefined;
    const eventName = detail?.edge.data?.["eventName"];
    if (typeof eventName !== "string" || eventName.length === 0) return;
    this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
      detail: { eventName },
      bubbles: true,
      composed: true,
    }));
  };

  #onViewport = (event: Event): void => {
    if (!(event instanceof CustomEvent)) return;
    const detail = event.detail;
    if (!hsm.isRecord(detail) || !hsm.isRecord(detail["viewport"]) || typeof detail["viewport"]["zoom"] !== "number") {
      return;
    }
    const viewport = detail["viewport"] as ViewportChangeDetail["viewport"];
    this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
      detail: { zoom: viewport.zoom },
      bubbles: true,
      composed: true,
    }));
  };

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return;
    this.#resizeObserver = new ResizeObserver(() => this.#live(hsm.typedEvent(BotMachineGraph.resizeEvent)));
    this.#resizeObserver.observe(this);
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
