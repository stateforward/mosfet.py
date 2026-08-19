import cytoscape from "cytoscape";

import { MachineGraphController, type GraphRenderer } from "../machine-graph-hsm.ts";
import {
  environmentBoxPositions,
  machineOwnerIndex,
  measureState,
  ownershipLayout,
} from "../machine-graph-layout.ts";
import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_EVENT,
  INITIAL_FILL,
  INITIAL_SIZE,
  compoundTitleStyle,
  graphNodeStyle,
  graphEdgeIdentity,
  graphEdgeSignature,
  initialNodeId,
  initialPosition,
  initialTargets,
  isRenderableGraphEdge,
  machineKey,
  namespacedPath,
  nodeClasses,
  nodeLabel,
  loopAnchorIdentity,
  structureKey,
  type Point,
} from "../machine-graph-view.ts";
import { type MachineGraph } from "../otel/machines.ts";
import { applyStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";

export type GraphZoomDetail = {
  zoom: number;
};

export type GraphEdgeDetail = {
  eventName: string;
};

const cssText = `
:host {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 16rem;
}
.frame {
  width: 100%;
  height: 100%;
  min-height: 16rem;
  background-color: ${CANVAS_FILL};
  background-image: radial-gradient(rgba(232, 234, 239, 0.07) 1px, transparent 1px);
  background-size: 16px 16px;
}
`;

const COMPOUND_TITLE = compoundTitleStyle();

function ancestorSet(path: string): Set<string> {
  const parts = path.split("/").filter((part) => part.length > 0);
  const values = new Set<string>();
  for (let index = 0; index < parts.length; index += 1) {
    values.add(`/${parts.slice(0, index + 1).join("/")}`);
  }
  return values;
}

const AXIS_EPS = 0.5;
const LOOP_STUB_X = 28;
const LOOP_STUB_Y = 36;

type EdgeKind = "taxi" | "aligned" | "loop";

function shareAxis(left: Point, right: Point): boolean {
  return Math.abs(left.x - right.x) <= AXIS_EPS || Math.abs(left.y - right.y) <= AXIS_EPS;
}

function edgeOccurrenceIndex(
  occurrences: Map<string, number>,
  edge: { source: string; target: string; eventName: string },
): number {
  const signature = graphEdgeSignature(edge);
  const occurrence = occurrences.get(signature) ?? 0;
  occurrences.set(signature, occurrence + 1);
  return occurrence;
}

function isDescendantPath(descendant: string, ancestor: string): boolean {
  return descendant.startsWith(`${ancestor}/`);
}

function needsOrthogonalWaypoint(source: string, target: string): boolean {
  return source === target || isDescendantPath(source, target) || isDescendantPath(target, source);
}

function waypointPosition(source: string, target: string, positions: Map<string, Point>): Point {
  if (source === target) {
    const origin = positions.get(source) ?? { x: 0, y: 0 };
    return { x: origin.x + LOOP_STUB_X, y: origin.y - LOOP_STUB_Y };
  }
  const inner = isDescendantPath(source, target) ? source : target;
  const origin = positions.get(inner) ?? { x: 0, y: 0 };
  return { x: origin.x - LOOP_STUB_X, y: origin.y - LOOP_STUB_Y };
}

function classifyEdge(source: string, target: string, positions: Map<string, Point>): EdgeKind {
  if (needsOrthogonalWaypoint(source, target)) {
    return "loop";
  }
  const from = positions.get(source);
  const to = positions.get(target);
  if (from !== undefined && to !== undefined && shareAxis(from, to)) {
    return "aligned";
  }
  return "taxi";
}

function edgeClasses(kind: EdgeKind, lastFired: boolean, extra: readonly string[] = []): string {
  const classes = [kind, ...extra];
  if (lastFired) {
    classes.push("last-fired");
  }
  return classes.join(" ");
}

class CytoscapeRenderer implements GraphRenderer {
  #cy: cytoscape.Core | null = null;
  #structure: string | null = null;
  #resizeObserver: ResizeObserver | null = null;
  #hasRealDimensions = false;

  constructor(
    private readonly container: HTMLElement,
    private readonly onZoom: (zoom: number) => void,
    private readonly onEdge: (eventName: string) => void,
  ) {
    this.#ensureResizeObserver();
  }

  draw(graphs: readonly MachineGraph[]): void {
    this.#ensureResizeObserver();
    const nextStructure = structureKey(graphs);
    if (this.#cy !== null && this.#structure === nextStructure) {
      this.#paint(graphs);
      this.#resizeForContainer();
      return;
    }
    this.#structure = nextStructure;
    const positions = environmentBoxPositions(graphs);
    const ownership = ownershipLayout(graphs);
    const elements: cytoscape.ElementDefinition[] = [];
    for (const [index, graph] of graphs.entries()) {
      const machine = machineKey(graph, index);
      const active = ancestorSet(graph.currentState);
      const localPositions = new Map<string, Point>();
      const knownPaths = new Set(graph.nodes.map((node) => node.path));
      const edgeOccurrences = new Map<string, number>();
      for (const node of graph.nodes) {
        const id = namespacedPath(machine, node.path);
        const data: Record<string, string> = {
          id,
          machine,
          path: node.path,
          name: node.label,
          label: nodeLabel(node.label, node.path, graph.currentState),
        };
        if (node.parent !== null) {
          data["parent"] = namespacedPath(machine, node.parent);
        } else if (node.path === graph.name) {
          const ownerIndex = machineOwnerIndex(graphs, index);
          if (ownerIndex !== null) {
            const ownerGraph = graphs[ownerIndex];
            if (ownerGraph !== undefined) {
              data["parent"] = namespacedPath(machineKey(ownerGraph, ownerIndex), ownerGraph.name);
            }
          }
        }
        const position = positions.get(id);
        if (position !== undefined) {
          localPositions.set(node.path, position);
        }
        elements.push({
          group: "nodes",
          data,
          classes: nodeClasses(node.path, graph.currentState, active),
          ...(position === undefined ? {} : { position }),
        });
      }
      for (const target of initialTargets(graph)) {
        const targetId = namespacedPath(machine, target);
        const targetPos = localPositions.get(target) ?? { x: 0, y: 0 };
        const size = measureState(graphs, ownership, index, target);
        const position = initialPosition(targetPos, size);
        const sourceId = `${machine}:${initialNodeId(target)}`;
        elements.push({
          group: "nodes",
          data: { id: sourceId, machine, label: "" },
          classes: "initial",
          position,
        });
        const kind: EdgeKind = shareAxis(position, targetPos) ? "aligned" : "taxi";
        elements.push({
          group: "edges",
          data: {
            id: `${sourceId}->${targetId}`,
            source: sourceId,
            target: targetId,
            label: "",
            eventName: INITIAL_EVENT,
            edgeKey: `${machine}:initial->${target}`,
            machine,
          },
          classes: edgeClasses(kind, false, ["initial"]),
        });
      }
      for (const edge of graph.edges) {
        if (edge.eventName === INITIAL_EVENT) {
          continue;
        }
        if (!isRenderableGraphEdge(edge, knownPaths)) {
          continue;
        }
        const occurrence = edgeOccurrenceIndex(edgeOccurrences, edge);
        const sourceId = namespacedPath(machine, edge.source);
        const targetId = namespacedPath(machine, edge.target);
        const kind = classifyEdge(edge.source, edge.target, localPositions);
        const key = graphEdgeIdentity(machine, edge, occurrence);
        if (kind === "loop") {
          const anchor = waypointPosition(edge.source, edge.target, localPositions);
          const anchorId = loopAnchorIdentity(machine, edge, occurrence);
          elements.push({
            group: "nodes",
            data: { id: anchorId, machine, label: "" },
            classes: "loop-anchor",
            position: anchor,
          });
          elements.push({
            group: "edges",
            data: {
              id: `${sourceId}->${anchorId}:${edge.eventName}`,
              source: sourceId,
              target: anchorId,
              label: edge.eventName,
              eventName: edge.eventName,
              edgeKey: key,
              machine,
            },
            classes: edgeClasses(kind, edge.lastFired, ["loop-out"]),
          });
          elements.push({
            group: "edges",
            data: {
              id: `${anchorId}->${targetId}:${edge.eventName}`,
              source: anchorId,
              target: targetId,
              label: "",
              eventName: edge.eventName,
              edgeKey: key,
              machine,
            },
            classes: edgeClasses(kind, edge.lastFired, ["loop-in"]),
          });
          continue;
        }
        elements.push({
          group: "edges",
          data: {
            id: key,
            source: sourceId,
            target: targetId,
            label: edge.eventName,
            eventName: edge.eventName,
            edgeKey: key,
            machine,
          },
          classes: edgeClasses(kind, edge.lastFired),
        });
      }
    }
    if (this.#cy === null) {
      this.#cy = cytoscape({
        container: this.container,
        elements,
        layout: { name: "preset", fit: true, padding: 28 },
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              color: "#d5dbe8",
              "background-color": "#161b22",
              "border-width": 1,
              "border-color": "#3d4a5c",
              "font-size": 11,
              "font-family": "IBM Plex Sans, Segoe UI, system-ui, sans-serif",
              "text-wrap": "wrap",
              "text-max-width": "100",
              "text-valign": "center",
              "text-halign": "center",
              padding: "10px",
              width: "label",
              height: "label",
              shape: "round-rectangle",
            },
          },
          {
            selector: "node:parent",
            style: {
              "background-opacity": 1,
              "background-color": "#161b22",
              "border-color": "#3f4b5f",
              "border-width": 1.2,
              color: "#9aa3b5",
              "text-valign": "top",
              "text-halign": "center",
              "text-margin-y": COMPOUND_TITLE.marginY,
              "text-background-color": COMPOUND_TITLE.backgroundColor,
              "text-background-opacity": COMPOUND_TITLE.backgroundOpacity,
              "text-background-padding": `${COMPOUND_TITLE.padding}px`,
              "font-weight": 650,
              "font-size": 10,
              padding: "18px",
            },
          },
          {
            selector: "node.active-path",
            style: {
              "background-color": graphNodeStyle("active-path").backgroundColor,
              "background-opacity": graphNodeStyle("active-path").backgroundOpacity,
              "border-color": graphNodeStyle("active-path").borderColor,
              "border-width": graphNodeStyle("active-path").borderWidth,
              color: graphNodeStyle("active-path").textColor,
              "font-weight": graphNodeStyle("active-path").fontWeight,
              "z-index": graphNodeStyle("active-path").zIndex,
            },
          },
          {
            selector: "node.current",
            style: {
              "background-color": graphNodeStyle("current").backgroundColor,
              "background-opacity": graphNodeStyle("current").backgroundOpacity,
              "border-color": graphNodeStyle("current").borderColor,
              "border-width": graphNodeStyle("current").borderWidth,
              color: graphNodeStyle("current").textColor,
              "font-weight": graphNodeStyle("current").fontWeight,
              "z-index": graphNodeStyle("current").zIndex,
            },
          },
          {
            selector: "node.initial",
            style: {
              width: INITIAL_SIZE,
              height: INITIAL_SIZE,
              shape: "ellipse",
              label: "",
              padding: "0px",
              "background-color": INITIAL_FILL,
              "background-opacity": 1,
              "border-color": INITIAL_BORDER,
              "border-width": INITIAL_BORDER_WIDTH,
              "underlay-color": INITIAL_BORDER,
              "underlay-padding": 3,
              "underlay-opacity": 0.22,
              "underlay-shape": "ellipse",
              events: "no",
              "z-index": 8,
            },
          },
          {
            selector: "node.loop-anchor",
            style: {
              width: 1,
              height: 1,
              padding: "0px",
              opacity: 0,
              label: "",
              events: "no",
            },
          },
          {
            selector: "edge",
            style: {
              label: "data(label)",
              color: "#7b8498",
              "font-size": 8,
              width: 1.2,
              "line-color": "#5b6578",
              "target-arrow-color": "#5b6578",
              "target-arrow-shape": "triangle",
              "arrow-scale": 0.8,
              "curve-style": "taxi",
              "taxi-direction": "horizontal",
              "taxi-turn": 24,
              "taxi-turn-min-distance": 12,
              "text-rotation": "none",
              "text-margin-y": -8,
              "text-background-color": CANVAS_FILL,
              "text-background-opacity": 0.72,
              "text-background-padding": "2px",
            },
          },
          {
            selector: "edge.aligned",
            style: {
              "curve-style": "straight",
            },
          },
          {
            selector: "edge.loop",
            style: {
              "curve-style": "taxi",
              "taxi-direction": "horizontal",
              "taxi-turn": 24,
              "taxi-turn-min-distance": 8,
            },
          },
          {
            selector: "edge.loop-out",
            style: {
              "target-arrow-shape": "none",
            },
          },
          {
            selector: "edge.initial",
            style: {
              label: "",
              width: 1.2,
              "line-color": "#c5ccd8",
              "target-arrow-color": "#c5ccd8",
              "text-opacity": 0,
            },
          },
          {
            selector: "edge.last-fired",
            style: {
              width: 1.5,
              color: "#9aa6b8",
              "line-color": "#7b8698",
              "target-arrow-color": "#7b8698",
            },
          },
        ],
      });
      this.#cy.on("zoom", () => {
        this.onZoom(this.#cy?.zoom() ?? 1);
      });
      this.#cy.on("tap", "edge", (event) => {
        const eventName = event.target.data("eventName");
        if (typeof eventName === "string" && eventName.length > 0) {
          this.onEdge(eventName);
        }
      });
    } else {
      this.#cy.json({ elements });
      this.#cy.layout({ name: "preset", fit: true, padding: 28 }).run();
    }
    this.#fitCapped();
  }

  #paint(graphs: readonly MachineGraph[]): void {
    const cy = this.#cy;
    if (cy === null) {
      return;
    }
    const byMachine = new Map(graphs.map((graph, index) => [machineKey(graph, index), graph]));
    cy.nodes().forEach((node) => {
      if (node.hasClass("loop-anchor") || node.hasClass("initial")) {
        return;
      }
      const machine = node.data("machine");
      const graph = typeof machine === "string" ? byMachine.get(machine) : undefined;
      if (graph === undefined) {
        return;
      }
      const path = node.data("path");
      if (typeof path !== "string") {
        return;
      }
      const active = ancestorSet(graph.currentState);
      const name = node.data("name");
      const label = typeof name === "string" && name.length > 0 ? name : path.split("/").pop() ?? path;
      node.classes(nodeClasses(path, graph.currentState, active));
      node.data("label", nodeLabel(label, path, graph.currentState));
    });
    cy.edges().forEach((edge) => {
      const machine = edge.data("machine");
      const graph = typeof machine === "string" ? byMachine.get(machine) : undefined;
      if (graph === undefined) {
        return;
      }
      const key = edge.data("edgeKey");
      const edgeOccurrences = new Map<string, number>();
      const lastFired = new Set<string>();
      for (const item of graph.edges) {
        if (item.eventName === INITIAL_EVENT) {
          continue;
        }
        const occurrence = edgeOccurrenceIndex(edgeOccurrences, item);
        if (item.lastFired) {
          lastFired.add(graphEdgeIdentity(machine, item, occurrence));
        }
      }
      const last = typeof key === "string" && lastFired.has(key);
      edge.toggleClass("last-fired", last);
    });
  }

  fit(): void {
    this.#fitCapped();
  }

  #fitCapped(): void {
    const cy = this.#cy;
    if (cy === null) {
      return;
    }
    cy.resize();
    cy.fit(undefined, 36);
    if (cy.zoom() > 1.2) {
      cy.zoom(1.2);
      cy.center();
    }
    this.onZoom(cy.zoom());
  }

  #resizeForContainer(): void {
    const cy = this.#cy;
    if (cy === null || this.container.clientWidth <= 0 || this.container.clientHeight <= 0) {
      return;
    }
    if (!this.#hasRealDimensions) {
      this.#hasRealDimensions = true;
      this.#fitCapped();
      return;
    }
    const zoom = cy.zoom();
    const pan = cy.pan();
    cy.resize();
    cy.zoom(zoom);
    cy.pan(pan);
    this.onZoom(cy.zoom());
  }

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") {
      return;
    }
    this.#resizeObserver = new ResizeObserver(() => {
      this.#resizeForContainer();
    });
    this.#resizeObserver.observe(this.container);
  }

  zoom(): number {
    return this.#cy?.zoom() ?? 1;
  }

  destroy(): void {
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    this.#cy?.destroy();
    this.#cy = null;
    this.#structure = null;
    this.#hasRealDimensions = false;
  }
}

export class BotMachineGraph extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #frame: HTMLDivElement;
  #controller: MachineGraphController | null = null;
  #renderer: CytoscapeRenderer | null = null;
  #pending: readonly MachineGraph[] | undefined;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, cssText);
    this.#frame = document.createElement("div");
    this.#frame.className = "frame";
    this.#frame.part.add("frame");
    this.#frame.setAttribute("data-testid", "frame");
    this.#root.append(this.#frame);
  }

  get graphs(): readonly MachineGraph[] {
    return this.#controller?.snapshot().graphs ?? this.#pending ?? [];
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#pending = value;
    this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0)));
    void this.#admit(value);
  }

  fit(): void {
    this.#renderer?.fit();
  }

  connectedCallback(): void {
    if (this.#renderer === null) {
      this.#renderer = new CytoscapeRenderer(
        this.#frame,
        (zoom) => {
          this.dispatchEvent(
            new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
              detail: { zoom },
              bubbles: true,
              composed: true,
            }),
          );
        },
        (eventName) => {
          this.dispatchEvent(
            new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
              detail: { eventName },
              bubbles: true,
              composed: true,
            }),
          );
        },
      );
    }
    if (this.#controller === null) {
      this.#controller = new MachineGraphController({
        renderer: this.#renderer,
      });
    }
    if (this.#pending !== undefined) {
      void this.#admit(this.#pending);
    }
  }

  disconnectedCallback(): void {
    const controller = this.#controller;
    this.#controller = null;
    this.#renderer = null;
    if (controller !== null) {
      void controller.stop();
    }
  }

  async #admit(value: readonly MachineGraph[]): Promise<void> {
    const controller = this.#controller;
    if (controller === null) {
      return;
    }
    if (value.length === 0) {
      await controller.dispatch("graph.clear");
      return;
    }
    await controller.dispatch("graph.set", { graphs: value });
  }
}

export function registerBotMachineGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) {
    customElements.define(ELEMENT_NAME, BotMachineGraph);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-machine-graph": BotMachineGraph;
  }
}
