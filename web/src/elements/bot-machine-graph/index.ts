import * as hsm from "../../hsm.ts";

import { FlowGraph, type NodeClickDetail, type EdgeClickDetail, type ViewportChangeDetail } from "../../flow/index.ts";
import { Graph, reportMachineGraphFailure } from "../../machine-graph-hsm.ts";
import { type MachineGraph } from "../../otel/machines.ts";
import { replaceStyles } from "../styles.ts";
import { flowModelFromGraphs, focusBoundsForMachine, type FlowGraphModel } from "./flow-model.ts";
import { graphStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";

export type GraphZoomDetail = { zoom: number };
export type GraphEdgeDetail = { eventName: string };

export class BotMachineGraph extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "BotMachineGraph",
    hsm.initial(hsm.target("active")),
    hsm.state("active"),
  );

  readonly #root: ShadowRoot;
  readonly #flow: FlowGraph;
  #graph: Graph | null = null;
  #pending: readonly MachineGraph[] | undefined;
  #pendingFocus: string | undefined;
  #model: FlowGraphModel | null = null;
  #connectionGeneration = 0;
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
    return this.#graph?.graphs ?? this.#pending ?? [];
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#pending = value;
    this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0)));
    this.#admit(value);
  }

  fit(): void {
    this.#pendingFocus = undefined;
    this.#flow.fitView();
  }

  focusMachine(machineName: string): boolean {
    const model = this.#model ?? flowModelFromGraphs(this.graphs);
    const bounds = focusBoundsForMachine(this.graphs, machineName, model);
    if (bounds === null) return false;
    this.#pendingFocus = undefined;
    this.#flow.fitBounds(bounds);
    return true;
  }

  connectedCallback(): void {
    const generation = ++this.#connectionGeneration;
    hsm.start(this, BotMachineGraph.model);
    const ctx = this.context();
    this.#graph = hsm.start(ctx, new Graph({
      onDraw: (graphs) => this.#draw(graphs),
      onDestroy: () => {
        this.#flow.nodes = [];
        this.#flow.edges = [];
        this.#model = null;
      },
    }), Graph.model);
    this.#flow.addEventListener("flow-node-click", this.#onNodeClick);
    this.#flow.addEventListener("flow-edge-click", this.#onEdgeClick);
    this.#flow.addEventListener("flow-viewport-change", this.#onViewport);
    this.#ensureResizeObserver();
    if (generation === this.#connectionGeneration && this.#pending !== undefined) this.#admit(this.#pending);
  }

  disconnectedCallback(): void {
    this.#connectionGeneration += 1;
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    this.#flow.removeEventListener("flow-node-click", this.#onNodeClick);
    this.#flow.removeEventListener("flow-edge-click", this.#onEdgeClick);
    this.#flow.removeEventListener("flow-viewport-change", this.#onViewport);
    const graph = this.#graph;
    this.#graph = null;
    this.#model = null;
    void Promise.all([
      graph === null ? undefined : hsm.stop(graph),
      hsm.stop(this),
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
    const detail = (event as CustomEvent<NodeClickDetail>).detail;
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
    const detail = (event as CustomEvent<EdgeClickDetail>).detail;
    const eventName = detail?.edge.data?.["eventName"];
    if (typeof eventName !== "string" || eventName.length === 0) return;
    this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
      detail: { eventName },
      bubbles: true,
      composed: true,
    }));
  };

  #onViewport = (event: Event): void => {
    const detail = (event as CustomEvent<ViewportChangeDetail>).detail;
    if (detail === undefined) return;
    this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
      detail: { zoom: detail.viewport.zoom },
      bubbles: true,
      composed: true,
    }));
  };

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return;
    this.#resizeObserver = new ResizeObserver(() => this.#flow.fitView());
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
