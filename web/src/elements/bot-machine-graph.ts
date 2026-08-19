import cytoscape from "cytoscape";

import { MachineGraphController, type GraphRenderer } from "../machine-graph-hsm.ts";
import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_EVENT,
  INITIAL_FILL,
  INITIAL_SIZE,
  graphNodeStyle,
  initialNodeId,
  initialPosition,
  initialTargets,
  nodeClasses,
  nodeLabel,
  type Point,
  type Size,
} from "../machine-graph-view.ts";
import { type MachineGraph, type MachineStateNode } from "../otel/machines.ts";
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

const LEAF_WIDTH = 120;
const LEAF_HEIGHT = 40;
const GAP = 22;
const PAD_X = 22;
const PAD_Y = 36;

function childrenOf(nodes: readonly MachineStateNode[], parent: string | null): MachineStateNode[] {
  return nodes.filter((node) => node.parent === parent);
}

function measureNode(nodes: readonly MachineStateNode[], path: string | null): Size {
  const kids = childrenOf(nodes, path);
  if (kids.length === 0) {
    return { width: LEAF_WIDTH, height: LEAF_HEIGHT };
  }
  let width = PAD_X;
  let height = 0;
  for (const kid of kids) {
    const size = measureNode(nodes, kid.path);
    width += size.width + GAP;
    height = Math.max(height, size.height);
  }
  return { width: width - GAP + PAD_X, height: height + PAD_Y + PAD_X };
}

function nestedBoxPositions(graph: MachineGraph): Map<string, Point> {
  const positions = new Map<string, Point>();
  const roots = childrenOf(graph.nodes, null);
  let originX = 0;
  for (const root of roots) {
    const size = measureNode(graph.nodes, root.path);
    placeNode(graph.nodes, root.path, originX, 0, positions);
    originX += size.width + GAP * 2;
  }
  return positions;
}

function placeNode(
  nodes: readonly MachineStateNode[],
  path: string,
  originX: number,
  originY: number,
  positions: Map<string, Point>,
): Size {
  const size = measureNode(nodes, path);
  positions.set(path, { x: originX + size.width / 2, y: originY + size.height / 2 });
  const kids = childrenOf(nodes, path);
  let childX = originX + PAD_X;
  const childY = originY + PAD_Y;
  for (const kid of kids) {
    const childSize = placeNode(nodes, kid.path, childX, childY, positions);
    childX += childSize.width + GAP;
  }
  return size;
}

function structureKey(graph: MachineGraph): string {
  const nodes = graph.nodes.map((node) => node.path).sort();
  const edges = graph.edges.map((edge) => `${edge.source}->${edge.target}:${edge.eventName}`).sort();
  return `${graph.name}|${nodes.join(",")}|${edges.join(",")}`;
}

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

function graphEdgeKey(edge: { source: string; target: string; eventName: string }): string {
  return `${edge.source}->${edge.target}:${edge.eventName}`;
}

function loopAnchorId(edge: { source: string; target: string; eventName: string }): string {
  return `loop:${graphEdgeKey(edge)}`;
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

  constructor(
    private readonly container: HTMLElement,
    private readonly onZoom: (zoom: number) => void,
    private readonly onEdge: (eventName: string) => void,
  ) {}

  draw(graph: MachineGraph): void {
    const nextStructure = structureKey(graph);
    if (this.#cy !== null && this.#structure === nextStructure) {
      this.#paint(graph);
      return;
    }
    this.#structure = nextStructure;
    const positions = nestedBoxPositions(graph);
    const active = ancestorSet(graph.currentState);
    const elements: cytoscape.ElementDefinition[] = [];
    for (const node of graph.nodes) {
      const data: Record<string, string> = {
        id: node.path,
        name: node.label,
        label: nodeLabel(node.label, node.path, graph.currentState),
      };
      if (node.parent !== null) {
        data["parent"] = node.parent;
      }
      const position = positions.get(node.path);
      elements.push({
        group: "nodes",
        data,
        classes: nodeClasses(node.path, graph.currentState, active),
        ...(position === undefined ? {} : { position }),
      });
    }
    for (const target of initialTargets(graph)) {
      const targetPos = positions.get(target) ?? { x: 0, y: 0 };
      const size = measureNode(graph.nodes, target);
      const position = initialPosition(targetPos, size);
      const sourceId = initialNodeId(target);
      elements.push({
        group: "nodes",
        data: { id: sourceId, label: "" },
        classes: "initial",
        position,
      });
      const kind: EdgeKind = shareAxis(position, targetPos) ? "aligned" : "taxi";
      elements.push({
        group: "edges",
        data: {
          id: `initial->${target}`,
          source: sourceId,
          target,
          label: "",
          eventName: INITIAL_EVENT,
          edgeKey: `initial->${target}`,
        },
        classes: edgeClasses(kind, false, ["initial"]),
      });
    }
    for (const edge of graph.edges) {
      if (edge.eventName === INITIAL_EVENT) {
        continue;
      }
      const kind = classifyEdge(edge.source, edge.target, positions);
      const key = graphEdgeKey(edge);
      if (kind === "loop") {
        const anchor = waypointPosition(edge.source, edge.target, positions);
        const anchorId = loopAnchorId(edge);
        elements.push({
          group: "nodes",
          data: { id: anchorId, label: "" },
          classes: "loop-anchor",
          position: anchor,
        });
        elements.push({
          group: "edges",
          data: {
            id: `${edge.source}->${anchorId}:${edge.eventName}`,
            source: edge.source,
            target: anchorId,
            label: edge.eventName,
            eventName: edge.eventName,
            edgeKey: key,
          },
          classes: edgeClasses(kind, edge.lastFired, ["loop-out"]),
        });
        elements.push({
          group: "edges",
          data: {
            id: `${anchorId}->${edge.target}:${edge.eventName}`,
            source: anchorId,
            target: edge.target,
            label: "",
            eventName: edge.eventName,
            edgeKey: key,
          },
          classes: edgeClasses(kind, edge.lastFired, ["loop-in"]),
        });
        continue;
      }
      elements.push({
        group: "edges",
        data: {
          id: key,
          source: edge.source,
          target: edge.target,
          label: edge.eventName,
          eventName: edge.eventName,
          edgeKey: key,
        },
        classes: edgeClasses(kind, edge.lastFired),
      });
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
              "text-margin-y": 8,
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

  #paint(graph: MachineGraph): void {
    const cy = this.#cy;
    if (cy === null) {
      return;
    }
    const active = ancestorSet(graph.currentState);
    cy.nodes().forEach((node) => {
      if (node.hasClass("loop-anchor") || node.hasClass("initial")) {
        return;
      }
      const path = node.id();
      const name = node.data("name");
      const label = typeof name === "string" && name.length > 0 ? name : path.split("/").pop() ?? path;
      node.classes(nodeClasses(path, graph.currentState, active));
      node.data("label", nodeLabel(label, path, graph.currentState));
    });
    cy.edges().forEach((edge) => {
      const key = edge.data("edgeKey");
      const last = graph.edges.some((item) => item.lastFired && graphEdgeKey(item) === key);
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

  zoom(): number {
    return this.#cy?.zoom() ?? 1;
  }

  destroy(): void {
    this.#cy?.destroy();
    this.#cy = null;
    this.#structure = null;
  }
}

export class BotMachineGraph extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #frame: HTMLDivElement;
  #controller: MachineGraphController | null = null;
  #renderer: CytoscapeRenderer | null = null;
  #pending: MachineGraph | null | undefined;

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

  get graph(): MachineGraph | null {
    return this.#controller?.snapshot().graph ?? this.#pending ?? null;
  }

  set graph(value: MachineGraph | null) {
    this.#pending = value;
    this.setAttribute("data-node-count", value === null ? "0" : String(value.nodes.length));
    this.setAttribute("data-current-state", value?.currentState ?? "");
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

  async #admit(value: MachineGraph | null): Promise<void> {
    const controller = this.#controller;
    if (controller === null) {
      return;
    }
    if (value === null) {
      await controller.dispatch("graph.clear");
      return;
    }
    await controller.dispatch("graph.set", { graph: value });
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
