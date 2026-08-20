import { MachineGraphController, type GraphRenderer } from "../machine-graph-hsm.ts";
import {
  environmentBoxPositions,
  machineOwnerIndex,
  measureState,
  ownershipLayout,
  renderableGraphs,
} from "../machine-graph-layout.ts";
import {
  CANVAS_FILL,
  INITIAL_BORDER,
  INITIAL_BORDER_WIDTH,
  INITIAL_EVENT,
  INITIAL_FILL,
  INITIAL_SIZE,
  graphEdgeIdentity,
  graphEdgeSignature,
  initialPosition,
  initialTargets,
  isRenderableGraphEdge,
  machineKey,
  namespacedPath,
  nodeClasses,
  nodeLabel,
  structureKey,
  STATE_NODE_SIZE,
  type Point,
  type Size,
} from "../machine-graph-view.ts";
import { type MachineGraph, type MachineStateNode } from "../otel/machines.ts";
import { applyStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";
const SVG_NS = "http://www.w3.org/2000/svg";
const WORLD_PADDING = 56;
const FIT_PADDING = 28;
const MIN_ZOOM = 0.12;
const MAX_FIT_ZOOM = 1.2;
const MAX_ZOOM = 2.4;
const EDGE_LABEL_LIMIT = 30;
const LOOP_STUB_X = 28;
const LOOP_STUB_Y = 36;
const AXIS_EPS = 0.5;

export type GraphZoomDetail = { zoom: number };
export type GraphEdgeDetail = { eventName: string };

const cssText = `
:host { display: block; width: 100%; height: 100%; min-height: 16rem; }
.frame {
  width: 100%; height: 100%; min-height: 16rem; overflow: hidden; position: relative;
  isolation: isolate; background-color: ${CANVAS_FILL};
  background-image: radial-gradient(rgba(232, 234, 239, 0.07) 1px, transparent 1px);
  background-size: 16px 16px; cursor: grab; touch-action: none; user-select: none;
}
.frame.is-dragging { cursor: grabbing; }
.viewport, .world, .edge-layer, .node-layer { position: absolute; inset: 0; }
.viewport { overflow: hidden; }
.world { inset: auto; transform-origin: 0 0; will-change: transform; }
.edge-layer { overflow: visible; pointer-events: none; }
.node-layer { pointer-events: none; }
.state-node {
  position: absolute; box-sizing: border-box; display: grid; place-items: center;
  transform: translate(-50%, -50%); border: 1px solid #3d4a5c; border-radius: 12px;
  background: #161b22; color: #d5dbe8;
  font: 500 11px/1.18 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  text-align: center; overflow: visible;
}
.state-node.machine-shell { border-color: #536176; border-radius: 16px; color: #9aa3b5; font-size: 10px; font-weight: 650; z-index: 1; }
.state-node.owned-machine { border-color: #596b78; border-style: dashed; }
.state-node.compound:not(.machine-shell) { border-color: #455166; border-radius: 14px; color: #a8b2c2; font-size: 10px; font-weight: 650; z-index: 2; }
.state-node:not(.compound) { z-index: 4; }
.state-node.active-path { border-color: #2dd4bf; border-width: 2px; color: #d5dbe8; font-weight: 650; }
.state-node.current { border-color: #2dd4bf; border-width: 3px; color: #d5dbe8; font-weight: 800; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.13); }
.node-badge { max-width: calc(100% - 12px); padding: 2px 5px; overflow: hidden; color: inherit; background: #161b22; border-radius: 5px; text-overflow: ellipsis; white-space: nowrap; }
.state-node.compound > .node-badge, .state-node.machine-shell > .node-badge {
  position: absolute; top: -13px; left: 12px; max-width: calc(100% - 24px);
  border: 1px solid currentColor; background: #0b0d12; letter-spacing: 0.01em;
}
.state-node.current > .node-badge { background: #123b3a; }
.initial-node { position: absolute; width: ${INITIAL_SIZE}px; height: ${INITIAL_SIZE}px; box-sizing: border-box; transform: translate(-50%, -50%); border: ${INITIAL_BORDER_WIDTH}px solid ${INITIAL_BORDER}; border-radius: 50%; background: ${INITIAL_FILL}; box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2); z-index: 5; }
.edge-path { fill: none; stroke: #5b6578; stroke-width: 1.35; vector-effect: non-scaling-stroke; marker-end: url(#graph-arrow); }
.edge-path.last-fired { stroke: #95a3b8; stroke-width: 1.8; }
.edge-path.initial { stroke: #c5ccd8; stroke-width: 1.25; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 14; pointer-events: stroke; cursor: pointer; }
.edge-label { fill: #aeb8c9; font: 500 10px/1 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; paint-order: stroke; stroke: ${CANVAS_FILL}; stroke-width: 6px; stroke-linejoin: round; pointer-events: none; text-anchor: middle; }
.edge-label.last-fired { fill: #d5dbe8; font-weight: 650; }
`;

type EdgeKind = "taxi" | "aligned" | "loop";
type Rect = { left: number; right: number; top: number; bottom: number; center: Point };
type Bounds = { left: number; right: number; top: number; bottom: number };
type LayoutNode = {
  id: string; machine: string; machineName: string; path: string; graph: MachineGraph;
  source: MachineStateNode; center: Point; size: Size; element: HTMLDivElement;
};
type RenderedEdge = {
  key: string; eventName: string; points: readonly Point[]; path: SVGPathElement;
  hit: SVGPathElement; label: SVGTextElement | null;
};

function ancestorSet(path: string): Set<string> {
  const parts = path.split("/").filter((part) => part.length > 0);
  const values = new Set<string>();
  for (let index = 0; index < parts.length; index += 1) values.add(`/${parts.slice(0, index + 1).join("/")}`);
  return values;
}
function shareAxis(left: Point, right: Point): boolean { return Math.abs(left.x - right.x) <= AXIS_EPS || Math.abs(left.y - right.y) <= AXIS_EPS; }
function isDescendantPath(descendant: string, ancestor: string): boolean { return descendant.startsWith(`${ancestor}/`); }
function edgeOccurrenceIndex(occurrences: Map<string, number>, edge: { source: string; target: string; eventName: string }): number {
  const signature = graphEdgeSignature(edge); const occurrence = occurrences.get(signature) ?? 0;
  occurrences.set(signature, occurrence + 1); return occurrence;
}
function classifyEdge(source: string, target: string, positions: Map<string, Point>): EdgeKind {
  if (source === target || isDescendantPath(source, target) || isDescendantPath(target, source)) return "loop";
  const from = positions.get(source); const to = positions.get(target);
  return from !== undefined && to !== undefined && shareAxis(from, to) ? "aligned" : "taxi";
}
function rectFor(center: Point, size: Size): Rect {
  return { left: center.x - size.width / 2, right: center.x + size.width / 2, top: center.y - size.height / 2, bottom: center.y + size.height / 2, center };
}
function boundaryPoint(rect: Rect, toward: Point): Point {
  const dx = toward.x - rect.center.x; const dy = toward.y - rect.center.y;
  if (dx === 0 && dy === 0) return { x: rect.right, y: rect.center.y };
  const width = rect.right - rect.left;
  const height = rect.bottom - rect.top;
  if (Math.abs(dx) * height >= Math.abs(dy) * width) {
    return { x: dx < 0 ? rect.left : rect.right, y: rect.center.y + (dy * (width / 2)) / Math.max(Math.abs(dx), 1) };
  }
  return { x: rect.center.x + (dx * (height / 2)) / Math.max(Math.abs(dy), 1), y: dy < 0 ? rect.top : rect.bottom };
}
function pathFor(points: readonly Point[], radius = 9): string {
  const first = points[0]; if (first === undefined) return "";
  if (points.length === 1) return `M ${first.x} ${first.y}`;
  const commands = [`M ${first.x} ${first.y}`];
  for (let index = 1; index < points.length; index += 1) {
    const point = points[index]; const previous = points[index - 1]; const next = points[index + 1];
    if (point === undefined || previous === undefined) continue;
    if (next === undefined) { commands.push(`L ${point.x} ${point.y}`); continue; }
    const incoming = Math.hypot(point.x - previous.x, point.y - previous.y);
    const outgoing = Math.hypot(next.x - point.x, next.y - point.y);
    const trim = Math.min(radius, incoming / 2, outgoing / 2);
    const before = { x: point.x - ((point.x - previous.x) / Math.max(incoming, 1)) * trim, y: point.y - ((point.y - previous.y) / Math.max(incoming, 1)) * trim };
    const after = { x: point.x + ((next.x - point.x) / Math.max(outgoing, 1)) * trim, y: point.y + ((next.y - point.y) / Math.max(outgoing, 1)) * trim };
    commands.push(`L ${before.x} ${before.y}`, `Q ${point.x} ${point.y} ${after.x} ${after.y}`);
  }
  return commands.join(" ");
}
function labelPosition(points: readonly Point[]): Point {
  let best = { length: 0, midpoint: points[0] ?? { x: 0, y: 0 }, horizontal: true };
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1]; const end = points[index]; if (start === undefined || end === undefined) continue;
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    if (length > best.length) best = { length, midpoint: { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 }, horizontal: Math.abs(end.x - start.x) >= Math.abs(end.y - start.y) };
  }
  return { x: best.midpoint.x + (best.horizontal ? 0 : 8), y: best.midpoint.y + (best.horizontal ? -8 : 0) };
}
function offsetPoints(points: readonly Point[], offset: number): Point[] {
  if (offset === 0 || points.length < 2) return [...points];
  const first = points[0]; const second = points[1]; if (first === undefined || second === undefined) return [...points];
  const horizontal = Math.abs(second.x - first.x) >= Math.abs(second.y - first.y);
  return points.map((point) => ({ x: point.x + (horizontal ? 0 : offset), y: point.y + (horizontal ? offset : 0) }));
}
function edgePoints(source: Rect, target: Rect, kind: EdgeKind, sourcePath: string, targetPath: string, offset: number): Point[] {
  if (kind === "loop" && sourcePath === targetPath) return [
    { x: source.right, y: source.center.y }, { x: source.right + LOOP_STUB_X, y: source.center.y },
    { x: source.right + LOOP_STUB_X, y: source.top - LOOP_STUB_Y }, { x: source.center.x, y: source.top - LOOP_STUB_Y },
    { x: source.center.x, y: source.top },
  ];
  const start = boundaryPoint(source, target.center); const end = boundaryPoint(target, source.center);
  if (kind === "aligned") return [start, end];
  if (kind === "loop") {
    const inner = isDescendantPath(sourcePath, targetPath) ? source : target; const channelX = inner.left - LOOP_STUB_X;
    return [start, { x: channelX, y: start.y }, { x: channelX, y: end.y }, end];
  }
  if (Math.abs(end.x - start.x) >= Math.abs(end.y - start.y)) {
    const channelX = (start.x + end.x) / 2 + offset; return [start, { x: channelX, y: start.y }, { x: channelX, y: end.y }, end];
  }
  const channelY = (start.y + end.y) / 2 + offset; return [start, { x: start.x, y: channelY }, { x: end.x, y: channelY }, end];
}
function addSvgElement<K extends keyof SVGElementTagNameMap>(parent: SVGElement, name: K): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, name); parent.append(element); return element;
}
function shiftedPoint(point: Point, origin: Point): Point { return { x: point.x + origin.x, y: point.y + origin.y }; }

class NativeGraphRenderer implements GraphRenderer {
  #viewport: HTMLDivElement; #world: HTMLDivElement; #edgeLayer: SVGSVGElement; #nodeLayer: HTMLDivElement; #container: HTMLElement;
  #onZoom: (zoom: number) => void; #onEdge: (eventName: string) => void;
  #structure: string | null = null; #graphs: readonly MachineGraph[] = []; #focusedMachine: string | undefined;
  #resizeObserver: ResizeObserver | null = null; #nodes = new Map<string, LayoutNode>(); #edges: RenderedEdge[] = [];
  #initialNodes: Array<{ element: HTMLDivElement; point: Point }> = [];
  #bounds: Bounds = { left: 0, right: 0, top: 0, bottom: 0 }; #origin: Point = { x: WORLD_PADDING, y: WORLD_PADDING };
  #worldSize: Size = { width: 1, height: 1 }; #scale = 1; #pan: Point = { x: 0, y: 0 }; #hasRealDimensions = false;
  #pointers = new Map<number, Point>(); #dragStart: { pointerId: number; point: Point; pan: Point } | null = null;
  #pinchStart: { distance: number; scale: number } | null = null;

  constructor(container: HTMLElement, onZoom: (zoom: number) => void, onEdge: (eventName: string) => void) {
    this.#container = container; this.#onZoom = onZoom; this.#onEdge = onEdge;
    this.#viewport = document.createElement("div"); this.#viewport.className = "viewport";
    this.#world = document.createElement("div"); this.#world.className = "world";
    this.#edgeLayer = document.createElementNS(SVG_NS, "svg"); this.#edgeLayer.classList.add("edge-layer"); this.#edgeLayer.setAttribute("aria-hidden", "true");
    this.#nodeLayer = document.createElement("div"); this.#nodeLayer.className = "node-layer";
    this.#world.append(this.#edgeLayer, this.#nodeLayer); this.#viewport.append(this.#world); this.#container.append(this.#viewport);
    this.#viewport.addEventListener("pointerdown", this.#onPointerDown); this.#viewport.addEventListener("pointermove", this.#onPointerMove);
    this.#viewport.addEventListener("pointerup", this.#onPointerUp); this.#viewport.addEventListener("pointercancel", this.#onPointerUp);
    this.#viewport.addEventListener("wheel", this.#onWheel, { passive: false }); this.#edgeLayer.addEventListener("click", this.#onEdgeClick); this.#ensureResizeObserver();
  }

  draw(graphs: readonly MachineGraph[]): boolean {
    const renderable = renderableGraphs(graphs); this.#graphs = renderable; const nextStructure = structureKey(renderable);
    if (this.#structure === nextStructure && this.#nodes.size > 0) { this.#paint(renderable); return false; }
    this.#structure = nextStructure; this.#renderStructure(renderable); return true;
  }
  focusMachine(machineName: string): boolean { this.#focusedMachine = machineName; return this.#focusMachine(machineName); }
  fit(): void { this.#focusedMachine = undefined; this.#dispatchInteraction("viewport.fit"); }
  zoom(): number { return this.#scale; }
  destroy(): void {
    this.#edgeLayer.replaceChildren(); this.#nodeLayer.replaceChildren();
    this.#world.style.width = "1px"; this.#world.style.height = "1px"; this.#structure = null; this.#graphs = [];
    this.#nodes.clear(); this.#edges = []; this.#initialNodes = []; this.#focusedMachine = undefined; this.#hasRealDimensions = false;
  }

  #renderStructure(graphs: readonly MachineGraph[]): void {
    this.#nodes.clear(); this.#edges = []; this.#initialNodes = []; this.#edgeLayer.replaceChildren(); this.#nodeLayer.replaceChildren();
    const defs = addSvgElement(this.#edgeLayer, "defs"); const marker = addSvgElement(defs, "marker"); marker.id = "graph-arrow";
    marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "8"); marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "6"); marker.setAttribute("markerHeight", "6"); marker.setAttribute("orient", "auto-start-reverse");
    const arrow = addSvgElement(marker, "path"); arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); arrow.setAttribute("fill", "#5b6578");
    const positions = environmentBoxPositions(graphs); const ownership = ownershipLayout(graphs);
    const bounds: Bounds = { left: Number.POSITIVE_INFINITY, right: Number.NEGATIVE_INFINITY, top: Number.POSITIVE_INFINITY, bottom: Number.NEGATIVE_INFINITY };
    const include = (rect: Rect): void => { bounds.left = Math.min(bounds.left, rect.left); bounds.right = Math.max(bounds.right, rect.right); bounds.top = Math.min(bounds.top, rect.top); bounds.bottom = Math.max(bounds.bottom, rect.bottom); };
    for (const [index, graph] of graphs.entries()) {
      const machine = machineKey(graph, index); const active = ancestorSet(graph.currentState); const ownerIndex = machineOwnerIndex(graphs, index);
      for (const source of graph.nodes) {
        const center = positions.get(namespacedPath(machine, source.path)); if (center === undefined) continue;
        const size = measureState(graphs, ownership, index, source.path); const machineRoot = source.path === graph.name;
        const compound = graph.nodes.some((node) => node.parent === source.path) || (machineRoot && (ownership.childrenByIndex.get(index)?.length ?? 0) > 0);
        const element = document.createElement("div"); element.className = this.#nodeClass(source.path, graph.currentState, active, compound, machineRoot, ownerIndex !== null);
        element.dataset["machine"] = machine; element.dataset["machineName"] = graph.name; element.dataset["path"] = source.path; element.style.width = `${Math.max(size.width, STATE_NODE_SIZE)}px`; element.style.height = `${Math.max(size.height, STATE_NODE_SIZE)}px`;
        const badge = document.createElement("span"); badge.className = "node-badge"; badge.textContent = nodeLabel(source.label, source.path, graph.currentState); element.append(badge); this.#nodeLayer.append(element);
        this.#nodes.set(namespacedPath(machine, source.path), { id: namespacedPath(machine, source.path), machine, machineName: graph.name, path: source.path, graph, source, center, size, element }); include(rectFor(center, size));
      }
    }
    if (!Number.isFinite(bounds.left)) { this.#bounds = { left: 0, right: 1, top: 0, bottom: 1 }; this.#worldSize = { width: 1, height: 1 }; return; }
    const edgeGroups: Array<{ edge: MachineGraph["edges"][number]; machine: string; positions: Map<string, Point>; nodes: Map<string, LayoutNode>; occurrence: number }> = [];
    for (const [index, graph] of graphs.entries()) {
      const machine = machineKey(graph, index); const known = new Set(graph.nodes.map((node) => node.path)); const localPositions = new Map<string, Point>(); const localNodes = new Map<string, LayoutNode>();
      for (const node of graph.nodes) { const layout = this.#nodes.get(namespacedPath(machine, node.path)); if (layout !== undefined) { localPositions.set(node.path, layout.center); localNodes.set(node.path, layout); } }
      const occurrences = new Map<string, number>();
      for (const target of initialTargets(graph)) {
        const layout = localNodes.get(target); const targetPosition = localPositions.get(target); if (layout === undefined || targetPosition === undefined) continue;
        const initial = initialPosition(targetPosition, layout.size); const initialElement = document.createElement("div"); initialElement.className = "initial-node"; this.#nodeLayer.append(initialElement); this.#initialNodes.push({ element: initialElement, point: initial }); include(rectFor(initial, { width: INITIAL_SIZE, height: INITIAL_SIZE }));
        const start = initial; const end = boundaryPoint(rectFor(targetPosition, layout.size), start); const points = shareAxis(start, end) ? [start, end] : [start, { x: (start.x + end.x) / 2, y: start.y }, { x: (start.x + end.x) / 2, y: end.y }, end];
        this.#appendEdge(`${machine}:initial->${target}`, INITIAL_EVENT, points, "initial", false);
      }
      for (const edge of graph.edges) { if (edge.eventName === INITIAL_EVENT || !isRenderableGraphEdge(edge, known)) continue; edgeGroups.push({ edge, machine, positions: localPositions, nodes: localNodes, occurrence: edgeOccurrenceIndex(occurrences, edge) }); }
    }
    for (const { edge, machine, positions: localPositions, nodes: localNodes, occurrence } of edgeGroups) {
      const source = localNodes.get(edge.source); const target = localNodes.get(edge.target); if (source === undefined || target === undefined) continue;
      const kind = classifyEdge(edge.source, edge.target, localPositions); const offset = kind === "loop" ? 0 : (occurrence % 3 - 1) * 12; const points = offsetPoints(edgePoints(rectFor(source.center, source.size), rectFor(target.center, target.size), kind, edge.source, edge.target, offset), offset);
      for (const point of points) { bounds.left = Math.min(bounds.left, point.x); bounds.right = Math.max(bounds.right, point.x); bounds.top = Math.min(bounds.top, point.y); bounds.bottom = Math.max(bounds.bottom, point.y); }
      this.#appendEdge(graphEdgeIdentity(machine, edge, occurrence), edge.eventName, points, kind, edge.lastFired);
    }
    this.#bounds = bounds; this.#origin = { x: WORLD_PADDING - bounds.left, y: WORLD_PADDING - bounds.top }; this.#worldSize = { width: Math.max(1, bounds.right - bounds.left + WORLD_PADDING * 2), height: Math.max(1, bounds.bottom - bounds.top + WORLD_PADDING * 2) };
    this.#world.style.width = `${this.#worldSize.width}px`; this.#world.style.height = `${this.#worldSize.height}px`; this.#edgeLayer.setAttribute("viewBox", `0 0 ${this.#worldSize.width} ${this.#worldSize.height}`); this.#edgeLayer.setAttribute("width", String(this.#worldSize.width)); this.#edgeLayer.setAttribute("height", String(this.#worldSize.height)); this.#positionElements();
  }

  #appendEdge(key: string, eventName: string, points: readonly Point[], kind: EdgeKind | "initial", lastFired: boolean): void {
    const path = addSvgElement(this.#edgeLayer, "path"); path.classList.add("edge-path", kind); if (lastFired) path.classList.add("last-fired"); if (kind === "initial") { path.classList.add("initial"); path.removeAttribute("marker-end"); }
    const hit = addSvgElement(this.#edgeLayer, "path"); hit.classList.add("edge-hit"); if (eventName !== INITIAL_EVENT) hit.dataset["eventName"] = eventName;
    let label: SVGTextElement | null = null;
    if (eventName !== INITIAL_EVENT) { label = addSvgElement(this.#edgeLayer, "text"); label.classList.add("edge-label"); if (lastFired) label.classList.add("last-fired"); label.textContent = eventName.length > EDGE_LABEL_LIMIT ? `${eventName.slice(0, EDGE_LABEL_LIMIT - 1)}…` : eventName; const title = addSvgElement(label, "title"); title.textContent = eventName; }
    this.#edges.push({ key, eventName, points, path, hit, label });
  }

  #nodeClass(path: string, currentState: string, active: ReadonlySet<string>, compound: boolean, machineRoot: boolean, owned: boolean): string {
    const classes = ["state-node", ...nodeClasses(path, currentState, active).split(" ")]; if (compound) classes.push("compound"); if (machineRoot) classes.push("machine-shell"); if (owned) classes.push("owned-machine"); return classes.join(" ");
  }
  #paint(graphs: readonly MachineGraph[]): void {
    const byMachine = new Map(graphs.map((graph, index) => [machineKey(graph, index), graph]));
    for (const node of this.#nodes.values()) { const graph = byMachine.get(node.machine); if (graph === undefined) continue; const compound = node.element.classList.contains("compound"); const machineRoot = node.path === graph.name; const index = graphs.indexOf(graph); node.element.className = this.#nodeClass(node.path, graph.currentState, ancestorSet(graph.currentState), compound, machineRoot, machineRoot && machineOwnerIndex(graphs, index) !== null); const badge = node.element.querySelector(".node-badge"); if (badge !== null) badge.textContent = nodeLabel(node.source.label, node.path, graph.currentState); }
    const fired = new Set<string>(); for (const [index, graph] of graphs.entries()) { const occurrences = new Map<string, number>(); for (const edge of graph.edges) { if (edge.eventName === INITIAL_EVENT) continue; const occurrence = edgeOccurrenceIndex(occurrences, edge); if (edge.lastFired) fired.add(graphEdgeIdentity(machineKey(graph, index), edge, occurrence)); } }
    for (const edge of this.#edges) { const active = fired.has(edge.key); edge.path.classList.toggle("last-fired", active); edge.label?.classList.toggle("last-fired", active); }
  }
  #focusMachine(machineName: string): boolean {
    if (!this.#graphs.some((graph) => graph.name === machineName)) return false;
    const names = new Set([machineName]); const byName = new Map(this.#graphs.map((graph) => [graph.name, graph]));
    for (const graph of this.#graphs) { let owner = graph.owner; const visited = new Set<string>(); while (owner !== null && owner !== undefined && !visited.has(owner)) { if (owner === machineName) { names.add(graph.name); break; } visited.add(owner); owner = byName.get(owner)?.owner; } }
    const focused = [...this.#nodes.values()].filter((node) => names.has(node.machineName)); if (focused.length === 0) return false;
    const bounds: Bounds = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity }; for (const node of focused) { const rect = rectFor(node.center, node.size); bounds.left = Math.min(bounds.left, rect.left); bounds.right = Math.max(bounds.right, rect.right); bounds.top = Math.min(bounds.top, rect.top); bounds.bottom = Math.max(bounds.bottom, rect.bottom); }
    this.#dispatchInteraction("viewport.focus", { bounds }); return true;
  }
  #applyFocusedMachine(): void { if (this.#focusedMachine !== undefined) this.#focusMachine(this.#focusedMachine); }
  #fitCapped(): void { if (this.#hasViewport()) this.#dispatchInteraction("viewport.fit"); }
  #fitBounds(bounds: Bounds, padding: number, maxZoom = MAX_FIT_ZOOM): void { const width = Math.max(1, bounds.right - bounds.left + padding * 2); const height = Math.max(1, bounds.bottom - bounds.top + padding * 2); const scale = Math.min(maxZoom, Math.max(MIN_ZOOM, Math.min(this.#viewport.clientWidth / width, this.#viewport.clientHeight / height))); const center = shiftedPoint({ x: (bounds.left + bounds.right) / 2, y: (bounds.top + bounds.bottom) / 2 }, this.#origin); this.#setTransform(scale, { x: this.#viewport.clientWidth / 2 - center.x * scale, y: this.#viewport.clientHeight / 2 - center.y * scale }); }
  #hasViewport(): boolean { return this.#viewport.clientWidth > 0 && this.#viewport.clientHeight > 0; }
  #setTransform(scale: number, pan: Point): void { this.#scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale)); this.#pan = pan; this.#world.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${this.#scale})`; this.#onZoom(this.#scale); }
  #positionElements(): void { for (const node of this.#nodes.values()) { node.element.style.left = `${node.center.x + this.#origin.x}px`; node.element.style.top = `${node.center.y + this.#origin.y}px`; } for (const initial of this.#initialNodes) { initial.element.style.left = `${initial.point.x + this.#origin.x}px`; initial.element.style.top = `${initial.point.y + this.#origin.y}px`; } for (const edge of this.#edges) { const points = edge.points.map((point) => shiftedPoint(point, this.#origin)); const d = pathFor(points); edge.path.setAttribute("d", d); edge.hit.setAttribute("d", d); if (edge.label !== null) { const point = shiftedPoint(labelPosition(edge.points), this.#origin); edge.label.setAttribute("x", String(point.x)); edge.label.setAttribute("y", String(point.y)); } } }
  #ensureResizeObserver(): void { if (this.#resizeObserver !== null || typeof ResizeObserver === "undefined") return; this.#resizeObserver = new ResizeObserver(() => this.#resizeForContainer()); this.#resizeObserver.observe(this.#container); }
  #resizeForContainer(): void { if (!this.#hasViewport()) return; if (!this.#hasRealDimensions) { this.#hasRealDimensions = true; this.#fitCapped(); this.#applyFocusedMachine(); return; } this.#fitCapped(); this.#applyFocusedMachine(); }

  #dispatchInteraction(eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom", data?: unknown): void { this.#interactionDispatch?.(eventName, data); }
  #interactionDispatch: ((eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom", data?: unknown) => void) | null = null;
  setInteractionDispatcher(dispatch: (eventName: "viewport.fit" | "viewport.focus" | "viewport.pan.start" | "viewport.pan" | "viewport.pan.end" | "viewport.zoom", data?: unknown) => void): void { this.#interactionDispatch = dispatch; }
  applyViewport(data: unknown): void { const record = data as { bounds?: Bounds; scale?: number; pan?: Point; point?: Point } | null; if (record?.bounds !== undefined) { this.#fitBounds(record.bounds, FIT_PADDING); return; } if (record?.scale !== undefined && record.pan !== undefined) { this.#setTransform(record.scale, record.pan); return; } if (record?.point !== undefined && record.scale !== undefined) { const worldPoint = { x: (record.point.x - this.#pan.x) / this.#scale, y: (record.point.y - this.#pan.y) / this.#scale }; this.#setTransform(record.scale, { x: record.point.x - worldPoint.x * record.scale, y: record.point.y - worldPoint.y * record.scale }); return; } this.#fitBounds(this.#bounds, FIT_PADDING); }

  #onPointerDown = (event: PointerEvent): void => { if (event.button !== 0 && event.pointerType === "mouse") return; this.#pointers.set(event.pointerId, { x: event.clientX, y: event.clientY }); this.#viewport.setPointerCapture(event.pointerId); if (this.#pointers.size === 1) { this.#dragStart = { pointerId: event.pointerId, point: { x: event.clientX, y: event.clientY }, pan: { ...this.#pan } }; this.#container.classList.add("is-dragging"); this.#dispatchInteraction("viewport.pan.start"); } else { this.#dragStart = null; this.#container.classList.remove("is-dragging"); const first = [...this.#pointers.values()][0]; const second = [...this.#pointers.values()][1]; if (first !== undefined && second !== undefined) this.#pinchStart = { distance: Math.max(1, Math.hypot(second.x - first.x, second.y - first.y)), scale: this.#scale }; } };
  #onPointerMove = (event: PointerEvent): void => { if (!this.#pointers.has(event.pointerId)) return; this.#pointers.set(event.pointerId, { x: event.clientX, y: event.clientY }); if (this.#pointers.size >= 2 && this.#pinchStart !== null) { const points = [...this.#pointers.values()]; const first = points[0]; const second = points[1]; if (first !== undefined && second !== undefined) { const midpoint = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 }; const scale = this.#pinchStart.scale * Math.hypot(second.x - first.x, second.y - first.y) / this.#pinchStart.distance; this.#dispatchInteraction("viewport.zoom", { scale, point: midpoint }); } return; } if (this.#dragStart?.pointerId === event.pointerId) this.#dispatchInteraction("viewport.pan", { pan: { x: this.#dragStart.pan.x + event.clientX - this.#dragStart.point.x, y: this.#dragStart.pan.y + event.clientY - this.#dragStart.point.y } }); };
  #onPointerUp = (event: PointerEvent): void => { this.#pointers.delete(event.pointerId); if (this.#pointers.size < 2) this.#pinchStart = null; if (this.#pointers.size === 0) { this.#dragStart = null; this.#container.classList.remove("is-dragging"); this.#dispatchInteraction("viewport.pan.end"); } };
  #onWheel = (event: WheelEvent): void => { event.preventDefault(); const rect = this.#viewport.getBoundingClientRect(); this.#dispatchInteraction("viewport.zoom", { scale: this.#scale * Math.exp(-event.deltaY * 0.0015), point: { x: event.clientX - rect.left, y: event.clientY - rect.top } }); };
  #onEdgeClick = (event: MouseEvent): void => { const target = event.target; if (!(target instanceof SVGElement)) return; const eventName = target.closest<SVGPathElement>(".edge-hit")?.dataset["eventName"]; if (eventName !== undefined && eventName.length > 0) this.#onEdge(eventName); };
}

export class BotMachineGraph extends HTMLElement {
  readonly #root: ShadowRoot; readonly #frame: HTMLDivElement; #controller: MachineGraphController | null = null; #renderer: NativeGraphRenderer | null = null; #pending: readonly MachineGraph[] | undefined; #pendingFocus: string | undefined;
  constructor() { super(); this.#root = this.attachShadow({ mode: "open" }); applyStyles(this.#root, cssText); this.#frame = document.createElement("div"); this.#frame.className = "frame"; this.#frame.part.add("frame"); this.#frame.setAttribute("data-testid", "frame"); this.#root.append(this.#frame); }
  get graphs(): readonly MachineGraph[] { return this.#controller?.snapshot().graphs ?? this.#pending ?? []; }
  set graphs(value: readonly MachineGraph[]) { this.#pending = value; this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0))); void this.#admit(value); }
  fit(): void { this.#pendingFocus = undefined; this.#renderer?.fit(); }
  focusMachine(machineName: string): void { this.#pendingFocus = machineName; this.#renderer?.focusMachine(machineName); }
  connectedCallback(): void {
    if (this.#renderer === null) { this.#renderer = new NativeGraphRenderer(this.#frame, (zoom) => this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", { detail: { zoom }, bubbles: true, composed: true })), (eventName) => this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", { detail: { eventName }, bubbles: true, composed: true }))); }
    if (this.#controller === null) { this.#controller = new MachineGraphController({ renderer: this.#renderer }); this.#renderer.setInteractionDispatcher((eventName, data) => void this.#controller?.dispatch(eventName, data)); }
    if (this.#pending !== undefined) void this.#admit(this.#pending);
  }
  disconnectedCallback(): void { const controller = this.#controller; this.#controller = null; this.#renderer = null; if (controller !== null) void controller.stop(); this.#frame.replaceChildren(); }
  async #admit(value: readonly MachineGraph[]): Promise<void> { const controller = this.#controller; if (controller === null) return; if (value.length === 0) { await controller.dispatch("graph.clear"); return; } await controller.dispatch("graph.set", { graphs: value }); if (this.#pendingFocus !== undefined && this.#renderer?.focusMachine(this.#pendingFocus)) this.#pendingFocus = undefined; }
}
export function registerBotMachineGraph(): void { if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, BotMachineGraph); }
declare global { interface HTMLElementTagNameMap { "bot-machine-graph": BotMachineGraph; } }
