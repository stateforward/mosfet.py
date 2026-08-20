import type { GraphRenderer, ViewportBounds, ViewportMetrics } from "../../machine-graph-hsm.ts";
import {
  environmentBoxPositions,
  machineOwnerIndex,
  measureState,
  ownershipLayout,
  renderableGraphs,
} from "../../machine-graph-layout.ts";
import {
  INITIAL_EVENT,
  INITIAL_SIZE,
  STATE_NODE_SIZE,
  graphEdgeIdentity,
  initialPosition,
  initialTargets,
  isRenderableGraphEdge,
  machineKey,
  namespacedPath,
  nodeClasses,
  nodeLabel,
  structureKey,
  type Point,
  type Size,
} from "../../machine-graph-view.ts";
import type { MachineGraph, MachineStateNode } from "../../otel/machines.ts";
import {
  ancestorSet,
  boundaryPoint,
  classifyEdge,
  edgeOccurrenceIndex,
  edgePoints,
  labelPosition,
  offsetPoints,
  pathFor,
  rectFor,
  shareAxis,
  shiftedPoint,
  type Bounds,
  type EdgeKind,
  type Rect,
} from "./geometry.ts";

const SVG_NS = "http://www.w3.org/2000/svg";
const WORLD_PADDING = 56;
const EDGE_LABEL_LIMIT = 30;
const STOPPED_CONTROLLER_ERROR = "MachineGraphController is stopped";

type LayoutNode = {
  id: string;
  machine: string;
  machineName: string;
  path: string;
  graph: MachineGraph;
  source: MachineStateNode;
  center: Point;
  size: Size;
  element: HTMLDivElement;
};

type RenderedEdge = {
  key: string;
  eventName: string;
  points: readonly Point[];
  path: SVGPathElement;
  hit: SVGPathElement;
  label: SVGTextElement | null;
};

function addSvgElement<K extends keyof SVGElementTagNameMap>(parent: SVGElement, name: K): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, name);
  parent.append(element);
  return element;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pointOf(value: unknown): Point | null {
  if (!isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
    ? { x, y }
    : null;
}

function reportInteractionFailure(error: unknown): void {
  if (error instanceof Error && error.message === STOPPED_CONTROLLER_ERROR) return;
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

export class NativeGraphRenderer implements GraphRenderer {
  #viewport: HTMLDivElement;
  #world: HTMLDivElement;
  #edgeLayer: SVGSVGElement;
  #nodeLayer: HTMLDivElement;
  #container: HTMLElement;
  #onZoom: (zoom: number) => void;
  #onEdge: (eventName: string) => void;
  #structure: string | null = null;
  #graphs: readonly MachineGraph[] = [];
  #resizeObserver: ResizeObserver | null = null;
  #nodes = new Map<string, LayoutNode>();
  #edges: RenderedEdge[] = [];
  #initialNodes: Array<{ element: HTMLDivElement; point: Point }> = [];
  #bounds: Bounds = { left: 0, right: 0, top: 0, bottom: 0 };
  #origin: Point = { x: WORLD_PADDING, y: WORLD_PADDING };
  #worldSize: Size = { width: 1, height: 1 };

  constructor(container: HTMLElement, onZoom: (zoom: number) => void, onEdge: (eventName: string) => void) {
    this.#container = container;
    this.#onZoom = onZoom;
    this.#onEdge = onEdge;
    this.#viewport = document.createElement("div");
    this.#viewport.className = "viewport";
    this.#world = document.createElement("div");
    this.#world.className = "world";
    this.#edgeLayer = document.createElementNS(SVG_NS, "svg");
    this.#edgeLayer.classList.add("edge-layer");
    this.#edgeLayer.setAttribute("aria-hidden", "true");
    this.#nodeLayer = document.createElement("div");
    this.#nodeLayer.className = "node-layer";
    this.#world.append(this.#edgeLayer, this.#nodeLayer);
    this.#viewport.append(this.#world);
    this.#container.append(this.#viewport);
    this.#viewport.addEventListener("pointerdown", this.#onPointerDown);
    this.#viewport.addEventListener("pointermove", this.#onPointerMove);
    this.#viewport.addEventListener("pointerup", this.#onPointerUp);
    this.#viewport.addEventListener("pointercancel", this.#onPointerUp);
    this.#viewport.addEventListener("wheel", this.#onWheel, { passive: false });
    this.#edgeLayer.addEventListener("click", this.#onEdgeClick);
    this.#ensureResizeObserver();
  }

  draw(graphs: readonly MachineGraph[]): boolean {
    const renderable = renderableGraphs(graphs);
    this.#graphs = renderable;
    const nextStructure = structureKey(renderable);
    if (this.#structure === nextStructure && this.#nodes.size > 0) {
      this.#paint(renderable);
      return false;
    }
    this.#structure = nextStructure;
    this.#renderStructure(renderable);
    return true;
  }

  destroy(): void {
    this.#edgeLayer.replaceChildren();
    this.#nodeLayer.replaceChildren();
    this.#world.style.width = "1px";
    this.#world.style.height = "1px";
    this.#structure = null;
    this.#graphs = [];
    this.#nodes.clear();
    this.#edges = [];
    this.#initialNodes = [];
    this.#world.style.transform = "none";
    this.#container.classList.remove("is-dragging");
  }

  dispose(): void {
    this.destroy();
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    this.#viewport.removeEventListener("pointerdown", this.#onPointerDown);
    this.#viewport.removeEventListener("pointermove", this.#onPointerMove);
    this.#viewport.removeEventListener("pointerup", this.#onPointerUp);
    this.#viewport.removeEventListener("pointercancel", this.#onPointerUp);
    this.#viewport.removeEventListener("wheel", this.#onWheel);
    this.#edgeLayer.removeEventListener("click", this.#onEdgeClick);
    this.#interactionDispatch = null;
    this.#onZoom = () => undefined;
    this.#onEdge = () => undefined;
  }

  #renderStructure(graphs: readonly MachineGraph[]): void {
    this.#nodes.clear();
    this.#edges = [];
    this.#initialNodes = [];
    this.#edgeLayer.replaceChildren();
    this.#nodeLayer.replaceChildren();
    const defs = addSvgElement(this.#edgeLayer, "defs");
    const marker = addSvgElement(defs, "marker");
    marker.id = "graph-arrow";
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrow = addSvgElement(marker, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.setAttribute("fill", "#5b6578");

    const positions = environmentBoxPositions(graphs);
    const ownership = ownershipLayout(graphs);
    const bounds: Bounds = {
      left: Number.POSITIVE_INFINITY,
      right: Number.NEGATIVE_INFINITY,
      top: Number.POSITIVE_INFINITY,
      bottom: Number.NEGATIVE_INFINITY,
    };
    const include = (rect: Rect): void => {
      bounds.left = Math.min(bounds.left, rect.left);
      bounds.right = Math.max(bounds.right, rect.right);
      bounds.top = Math.min(bounds.top, rect.top);
      bounds.bottom = Math.max(bounds.bottom, rect.bottom);
    };

    for (const [index, graph] of graphs.entries()) {
      const machine = machineKey(graph, index);
      const active = ancestorSet(graph.currentState);
      const ownerIndex = machineOwnerIndex(graphs, index);
      for (const source of graph.nodes) {
        const center = positions.get(namespacedPath(machine, source.path));
        if (center === undefined) continue;
        const size = measureState(graphs, ownership, index, source.path);
        const machineRoot = source.path === graph.name;
        const compound = graph.nodes.some((node) => node.parent === source.path)
          || (machineRoot && (ownership.childrenByIndex.get(index)?.length ?? 0) > 0);
        const element = document.createElement("div");
        element.className = this.#nodeClass(source.path, graph.currentState, active, compound, machineRoot, ownerIndex !== null);
        element.dataset["machine"] = machine;
        element.dataset["machineName"] = graph.name;
        element.dataset["path"] = source.path;
        element.style.width = `${Math.max(size.width, STATE_NODE_SIZE)}px`;
        element.style.height = `${Math.max(size.height, STATE_NODE_SIZE)}px`;
        const badge = document.createElement("span");
        badge.className = "node-badge";
        badge.textContent = nodeLabel(source.label, source.path, graph.currentState);
        element.append(badge);
        this.#nodeLayer.append(element);
        this.#nodes.set(namespacedPath(machine, source.path), {
          id: namespacedPath(machine, source.path),
          machine,
          machineName: graph.name,
          path: source.path,
          graph,
          source,
          center,
          size,
          element,
        });
        include(rectFor(center, size));
      }
    }

    if (!Number.isFinite(bounds.left)) {
      this.#bounds = { left: 0, right: 1, top: 0, bottom: 1 };
      this.#worldSize = { width: 1, height: 1 };
      return;
    }

    const edgeGroups: Array<{
      edge: MachineGraph["edges"][number];
      machine: string;
      positions: Map<string, Point>;
      nodes: Map<string, LayoutNode>;
      occurrence: number;
    }> = [];
    for (const [index, graph] of graphs.entries()) {
      const machine = machineKey(graph, index);
      const known = new Set(graph.nodes.map((node) => node.path));
      const localPositions = new Map<string, Point>();
      const localNodes = new Map<string, LayoutNode>();
      for (const node of graph.nodes) {
        const layout = this.#nodes.get(namespacedPath(machine, node.path));
        if (layout !== undefined) {
          localPositions.set(node.path, layout.center);
          localNodes.set(node.path, layout);
        }
      }
      const occurrences = new Map<string, number>();
      for (const target of initialTargets(graph)) {
        const layout = localNodes.get(target);
        const targetPosition = localPositions.get(target);
        if (layout === undefined || targetPosition === undefined) continue;
        const initial = initialPosition(targetPosition, layout.size);
        const initialElement = document.createElement("div");
        initialElement.className = "initial-node";
        this.#nodeLayer.append(initialElement);
        this.#initialNodes.push({ element: initialElement, point: initial });
        include(rectFor(initial, { width: INITIAL_SIZE, height: INITIAL_SIZE }));
        const start = initial;
        const end = boundaryPoint(rectFor(targetPosition, layout.size), start);
        const points = shareAxis(start, end)
          ? [start, end]
          : [start, { x: (start.x + end.x) / 2, y: start.y }, { x: (start.x + end.x) / 2, y: end.y }, end];
        this.#appendEdge(`${machine}:initial->${target}`, INITIAL_EVENT, points, "initial", false);
      }
      for (const edge of graph.edges) {
        if (edge.eventName === INITIAL_EVENT || !isRenderableGraphEdge(edge, known)) continue;
        edgeGroups.push({
          edge,
          machine,
          positions: localPositions,
          nodes: localNodes,
          occurrence: edgeOccurrenceIndex(occurrences, edge),
        });
      }
    }

    for (const { edge, machine, positions: localPositions, nodes: localNodes, occurrence } of edgeGroups) {
      const source = localNodes.get(edge.source);
      const target = localNodes.get(edge.target);
      if (source === undefined || target === undefined) continue;
      const kind = classifyEdge(edge.source, edge.target, localPositions);
      const offset = kind === "loop" ? 0 : (occurrence % 3 - 1) * 12;
      const points = offsetPoints(
        edgePoints(rectFor(source.center, source.size), rectFor(target.center, target.size), kind, edge.source, edge.target, offset),
        offset,
      );
      for (const point of points) {
        bounds.left = Math.min(bounds.left, point.x);
        bounds.right = Math.max(bounds.right, point.x);
        bounds.top = Math.min(bounds.top, point.y);
        bounds.bottom = Math.max(bounds.bottom, point.y);
      }
      this.#appendEdge(graphEdgeIdentity(machine, edge, occurrence), edge.eventName, points, kind, edge.lastFired);
    }

    this.#bounds = bounds;
    this.#origin = { x: WORLD_PADDING - bounds.left, y: WORLD_PADDING - bounds.top };
    this.#worldSize = {
      width: Math.max(1, bounds.right - bounds.left + WORLD_PADDING * 2),
      height: Math.max(1, bounds.bottom - bounds.top + WORLD_PADDING * 2),
    };
    this.#world.style.width = `${this.#worldSize.width}px`;
    this.#world.style.height = `${this.#worldSize.height}px`;
    this.#edgeLayer.setAttribute("viewBox", `0 0 ${this.#worldSize.width} ${this.#worldSize.height}`);
    this.#edgeLayer.setAttribute("width", String(this.#worldSize.width));
    this.#edgeLayer.setAttribute("height", String(this.#worldSize.height));
    this.#positionElements();
  }

  #appendEdge(
    key: string,
    eventName: string,
    points: readonly Point[],
    kind: EdgeKind | "initial",
    lastFired: boolean,
  ): void {
    const path = addSvgElement(this.#edgeLayer, "path");
    path.classList.add("edge-path", kind);
    if (lastFired) path.classList.add("last-fired");
    if (kind === "initial") {
      path.classList.add("initial");
      path.removeAttribute("marker-end");
    }
    const hit = addSvgElement(this.#edgeLayer, "path");
    hit.classList.add("edge-hit");
    if (eventName !== INITIAL_EVENT) hit.dataset["eventName"] = eventName;
    let label: SVGTextElement | null = null;
    if (eventName !== INITIAL_EVENT) {
      label = addSvgElement(this.#edgeLayer, "text");
      label.classList.add("edge-label");
      if (lastFired) label.classList.add("last-fired");
      label.textContent = eventName.length > EDGE_LABEL_LIMIT ? `${eventName.slice(0, EDGE_LABEL_LIMIT - 1)}…` : eventName;
      const title = addSvgElement(label, "title");
      title.textContent = eventName;
    }
    this.#edges.push({ key, eventName, points, path, hit, label });
    path.setAttribute("d", pathFor(points));
    hit.setAttribute("d", pathFor(points));
    if (label !== null) {
      const point = labelPosition(points);
      label.setAttribute("x", String(point.x));
      label.setAttribute("y", String(point.y));
    }
  }

  #nodeClass(path: string, currentState: string, active: ReadonlySet<string>, compound: boolean, machineRoot: boolean, owned: boolean): string {
    const classes = ["state-node", ...nodeClasses(path, currentState, active).split(" ")];
    if (compound) classes.push("compound");
    if (machineRoot) classes.push("machine-shell");
    if (owned) classes.push("owned-machine");
    return classes.join(" ");
  }

  #paint(graphs: readonly MachineGraph[]): void {
    const byMachine = new Map(graphs.map((graph, index) => [machineKey(graph, index), graph]));
    for (const node of this.#nodes.values()) {
      const graph = byMachine.get(node.machine);
      if (graph === undefined) continue;
      const compound = node.element.classList.contains("compound");
      const machineRoot = node.path === graph.name;
      const index = graphs.indexOf(graph);
      node.element.className = this.#nodeClass(
        node.path,
        graph.currentState,
        ancestorSet(graph.currentState),
        compound,
        machineRoot,
        machineRoot && machineOwnerIndex(graphs, index) !== null,
      );
      const badge = node.element.querySelector(".node-badge");
      if (badge !== null) badge.textContent = nodeLabel(node.source.label, node.path, graph.currentState);
    }
    const fired = new Set<string>();
    for (const [index, graph] of graphs.entries()) {
      const occurrences = new Map<string, number>();
      for (const edge of graph.edges) {
        if (edge.eventName === INITIAL_EVENT) continue;
        const occurrence = edgeOccurrenceIndex(occurrences, edge);
        if (edge.lastFired) fired.add(graphEdgeIdentity(machineKey(graph, index), edge, occurrence));
      }
    }
    for (const edge of this.#edges) {
      const active = fired.has(edge.key);
      edge.path.classList.toggle("last-fired", active);
      edge.label?.classList.toggle("last-fired", active);
    }
  }

  viewportMetrics(): ViewportMetrics | null {
    if (!this.#hasViewport()) return null;
    return {
      width: this.#viewport.clientWidth,
      height: this.#viewport.clientHeight,
      bounds: { ...this.#bounds },
      origin: { ...this.#origin },
    };
  }

  focusBounds(machineName: string): ViewportBounds | null {
    if (!this.#graphs.some((graph) => graph.name === machineName)) return null;
    const names = new Set([machineName]);
    const byName = new Map(this.#graphs.map((graph) => [graph.name, graph]));
    for (const graph of this.#graphs) {
      let owner = graph.owner;
      const visited = new Set<string>();
      while (owner !== null && owner !== undefined && !visited.has(owner)) {
        if (owner === machineName) {
          names.add(graph.name);
          break;
        }
        visited.add(owner);
        owner = byName.get(owner)?.owner;
      }
    }
    const focused = [...this.#nodes.values()].filter((node) => names.has(node.machineName));
    if (focused.length === 0) return null;
    const bounds: Bounds = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
    for (const node of focused) {
      const rect = rectFor(node.center, node.size);
      bounds.left = Math.min(bounds.left, rect.left);
      bounds.right = Math.max(bounds.right, rect.right);
      bounds.top = Math.min(bounds.top, rect.top);
      bounds.bottom = Math.max(bounds.bottom, rect.bottom);
    }
    return bounds;
  }

  #hasViewport(): boolean {
    return this.#viewport.clientWidth > 0 && this.#viewport.clientHeight > 0;
  }

  #positionElements(): void {
    for (const node of this.#nodes.values()) {
      node.element.style.left = `${node.center.x + this.#origin.x}px`;
      node.element.style.top = `${node.center.y + this.#origin.y}px`;
    }
    for (const initial of this.#initialNodes) {
      initial.element.style.left = `${initial.point.x + this.#origin.x}px`;
      initial.element.style.top = `${initial.point.y + this.#origin.y}px`;
    }
    for (const edge of this.#edges) {
      const points = edge.points.map((point) => shiftedPoint(point, this.#origin));
      const d = pathFor(points);
      edge.path.setAttribute("d", d);
      edge.hit.setAttribute("d", d);
      if (edge.label !== null) {
        const point = shiftedPoint(labelPosition(edge.points), this.#origin);
        edge.label.setAttribute("x", String(point.x));
        edge.label.setAttribute("y", String(point.y));
      }
    }
  }

  #ensureResizeObserver(): void {
    if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return;
    this.#resizeObserver = new ResizeObserver(() => this.#resizeForContainer());
    this.#resizeObserver.observe(this.#container);
  }

  #resizeForContainer(): void {
    if (!this.#hasViewport()) return;
    this.#dispatchInteraction("viewport.fit", { reason: "resize" });
  }

  #dispatchInteraction(
    eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom",
    data?: unknown,
  ): void {
    const dispatch = this.#interactionDispatch?.(eventName, data);
    if (dispatch !== undefined) void dispatch.catch(reportInteractionFailure);
  }

  #interactionDispatch: ((
    eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom",
    data?: unknown,
  ) => void | Promise<void>) | null = null;

  setInteractionDispatcher(dispatch: (
    eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom",
    data?: unknown,
  ) => void | Promise<void>): void {
    this.#interactionDispatch = dispatch;
  }

  applyViewport(data: unknown): void {
    const record = isRecord(data) ? data : null;
    const panning = record?.["panning"];
    if (typeof panning === "boolean") this.#container.classList.toggle("is-dragging", panning);
    const scale = record?.["scale"];
    const pan = pointOf(record?.["pan"]);
    if (typeof scale !== "number" || pan === null) return;
    this.#world.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${scale})`;
    this.#onZoom(scale);
  }

  #onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    this.#viewport.setPointerCapture(event.pointerId);
    const rect = this.#viewport.getBoundingClientRect();
    this.#dispatchInteraction("viewport.pan.start", {
      pointerId: event.pointerId,
      point: { x: event.clientX - rect.left, y: event.clientY - rect.top },
    });
  };

  #onPointerMove = (event: PointerEvent): void => {
    const rect = this.#viewport.getBoundingClientRect();
    this.#dispatchInteraction("viewport.pan", {
      pointerId: event.pointerId,
      point: { x: event.clientX - rect.left, y: event.clientY - rect.top },
    });
  };

  #onPointerUp = (event: PointerEvent): void => {
    this.#dispatchInteraction("viewport.pan.end", { pointerId: event.pointerId });
  };

  #onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const rect = this.#viewport.getBoundingClientRect();
    this.#dispatchInteraction("viewport.zoom", {
      deltaY: event.deltaY,
      point: { x: event.clientX - rect.left, y: event.clientY - rect.top },
    });
  };

  #onEdgeClick = (event: MouseEvent): void => {
    const target = event.target;
    if (!(target instanceof SVGElement)) return;
    const eventName = target.closest<SVGPathElement>(".edge-hit")?.dataset["eventName"];
    if (eventName !== undefined && eventName.length > 0) this.#onEdge(eventName);
  };
}
