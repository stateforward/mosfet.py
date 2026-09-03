import type { Edge, Node } from "../../flow/types.ts";
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
  graphEdgeSignature,
  initialPosition,
  initialTargets,
  isRenderableGraphEdge,
  machineKey,
  namespacedPath,
  nodeClasses,
  nodeLabel,
  type Point,
  type Size,
} from "../../machine-graph-view.ts";
import type { MachineGraph, MachineStateNode } from "../../otel/machines.ts";

const WORLD_PADDING = 56;
const EDGE_LABEL_LIMIT = 30;
// Centers within this distance of a shared axis classify as "aligned".
const AXIS_EPS = 0.5;
// Standoff of the semantic loop-channel wire from the state rect it wraps.
const LOOP_STUB_X = 28;
const LOOP_STUB_Y = 36;

export type GraphHit = {
  readonly machineName: string;
  readonly path: string;
  readonly bounds: { left: number; right: number; top: number; bottom: number };
};

type Bounds = { left: number; right: number; top: number; bottom: number };

type Rect = { left: number; right: number; top: number; bottom: number; center: Point };

/**
 * Wire class of a rendered transition. "loop" edges keep their dedicated
 * semantic stub shapes; "taxi" and "aligned" edges ship as flow-engine cables.
 */
type EdgeKind = "taxi" | "aligned" | "loop";

export type FlowGraphModel = {
  readonly nodes: Node[];
  readonly edges: Edge[];
  readonly bounds: Bounds;
  readonly origin: Point;
};

type LayoutNode = {
  id: string;
  machine: string;
  machineName: string;
  path: string;
  source: MachineStateNode;
  center: Point;
  size: Size;
};

function nodeClass(
  path: string,
  currentState: string,
  active: ReadonlySet<string>,
  compound: boolean,
  machineRoot: boolean,
  owned: boolean,
): string {
  const classes = ["state-node", ...nodeClasses(path, currentState, active).split(" ")];
  if (compound) classes.push("compound");
  if (machineRoot) classes.push("machine-shell");
  if (owned) classes.push("owned-machine");
  return classes.join(" ");
}

function rectFor(center: Point, size: Size): Rect {
  return {
    left: center.x - size.width / 2,
    right: center.x + size.width / 2,
    top: center.y - size.height / 2,
    bottom: center.y + size.height / 2,
    center,
  };
}

function boundaryPoint(rect: Rect, toward: Point): Point {
  const dx = toward.x - rect.center.x;
  const dy = toward.y - rect.center.y;
  if (dx === 0 && dy === 0) return { x: rect.right, y: rect.center.y };
  const width = rect.right - rect.left;
  const height = rect.bottom - rect.top;
  if (Math.abs(dx) * height >= Math.abs(dy) * width) {
    return {
      x: dx < 0 ? rect.left : rect.right,
      y: rect.center.y + (dy * (width / 2)) / Math.max(Math.abs(dx), 1),
    };
  }
  return {
    x: rect.center.x + (dx * (height / 2)) / Math.max(Math.abs(dy), 1),
    y: dy < 0 ? rect.top : rect.bottom,
  };
}

function shareAxis(left: Point, right: Point): boolean {
  return Math.abs(left.x - right.x) <= AXIS_EPS || Math.abs(left.y - right.y) <= AXIS_EPS;
}

function isDescendantPath(descendant: string, ancestor: string): boolean {
  return descendant.startsWith(`${ancestor}/`);
}

function classifyEdge(source: string, target: string, positions: Map<string, Point>): EdgeKind {
  if (source === target || isDescendantPath(source, target) || isDescendantPath(target, source)) {
    return "loop";
  }
  const from = positions.get(source);
  const to = positions.get(target);
  return from !== undefined && to !== undefined && shareAxis(from, to) ? "aligned" : "taxi";
}

function shiftedPoint(point: Point, origin: Point): Point {
  return { x: point.x + origin.x, y: point.y + origin.y };
}

function pathFor(points: readonly Point[], radius = 9): string {
  const first = points[0];
  if (first === undefined) return "";
  if (points.length === 1) return `M ${first.x} ${first.y}`;
  const commands = [`M ${first.x} ${first.y}`];
  for (let index = 1; index < points.length; index += 1) {
    const point = points[index];
    const previous = points[index - 1];
    const next = points[index + 1];
    if (point === undefined || previous === undefined) continue;
    if (next === undefined) {
      commands.push(`L ${point.x} ${point.y}`);
      continue;
    }
    const incoming = Math.hypot(point.x - previous.x, point.y - previous.y);
    const outgoing = Math.hypot(next.x - point.x, next.y - point.y);
    const trim = Math.min(radius, incoming / 2, outgoing / 2);
    const before = {
      x: point.x - ((point.x - previous.x) / Math.max(incoming, 1)) * trim,
      y: point.y - ((point.y - previous.y) / Math.max(incoming, 1)) * trim,
    };
    const after = {
      x: point.x + ((next.x - point.x) / Math.max(outgoing, 1)) * trim,
      y: point.y + ((next.y - point.y) / Math.max(outgoing, 1)) * trim,
    };
    commands.push(`L ${before.x} ${before.y}`, `Q ${point.x} ${point.y} ${after.x} ${after.y}`);
  }
  return commands.join(" ");
}

function labelPosition(points: readonly Point[]): Point {
  let best = { length: 0, midpoint: points[0] ?? { x: 0, y: 0 }, horizontal: true };
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    if (start === undefined || end === undefined) continue;
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    if (length > best.length) {
      best = {
        length,
        midpoint: { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
        horizontal: Math.abs(end.x - start.x) >= Math.abs(end.y - start.y),
      };
    }
  }
  return {
    x: best.midpoint.x + (best.horizontal ? 0 : 8),
    y: best.midpoint.y + (best.horizontal ? -8 : 0),
  };
}

function loopEdgePoints(source: Rect, target: Rect, sourcePath: string, targetPath: string): Point[] {
  if (sourcePath === targetPath) {
    return [
      { x: source.right, y: source.center.y },
      { x: source.right + LOOP_STUB_X, y: source.center.y },
      { x: source.right + LOOP_STUB_X, y: source.top - LOOP_STUB_Y },
      { x: source.center.x, y: source.top - LOOP_STUB_Y },
      { x: source.center.x, y: source.top },
    ];
  }
  const start = boundaryPoint(source, target.center);
  const end = boundaryPoint(target, source.center);
  const channelX = (isDescendantPath(sourcePath, targetPath) ? source : target).left - LOOP_STUB_X;
  return [start, { x: channelX, y: start.y }, { x: channelX, y: end.y }, end];
}

export function flowModelFromGraphs(graphs: readonly MachineGraph[]): FlowGraphModel {
  const renderable = renderableGraphs(graphs);
  const positions = environmentBoxPositions(renderable);
  const ownership = ownershipLayout(renderable);
  const layout = new Map<string, LayoutNode>();
  const bounds: Bounds = {
    left: Number.POSITIVE_INFINITY,
    right: Number.NEGATIVE_INFINITY,
    top: Number.POSITIVE_INFINITY,
    bottom: Number.NEGATIVE_INFINITY,
  };
  const include = (rect: { left: number; right: number; top: number; bottom: number }): void => {
    bounds.left = Math.min(bounds.left, rect.left);
    bounds.right = Math.max(bounds.right, rect.right);
    bounds.top = Math.min(bounds.top, rect.top);
    bounds.bottom = Math.max(bounds.bottom, rect.bottom);
  };

  for (const [index, graph] of renderable.entries()) {
    const machine = machineKey(graph, index);
    for (const source of graph.nodes) {
      const center = positions.get(namespacedPath(machine, source.path));
      if (center === undefined) continue;
      const size = measureState(renderable, ownership, index, source.path);
      const id = namespacedPath(machine, source.path);
      layout.set(id, {
        id,
        machine,
        machineName: graph.name,
        path: source.path,
        source,
        center,
        size: { width: Math.max(size.width, STATE_NODE_SIZE), height: Math.max(size.height, STATE_NODE_SIZE) },
      });
      include(rectFor(center, size));
    }
  }

  if (!Number.isFinite(bounds.left)) {
    return { nodes: [], edges: [], bounds: { left: 0, right: 1, top: 0, bottom: 1 }, origin: { x: 0, y: 0 } };
  }

  const origin = { x: WORLD_PADDING - bounds.left, y: WORLD_PADDING - bounds.top };
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  for (const node of layout.values()) {
    const graph = renderable.find((item) => item.name === node.machineName);
    if (graph === undefined) continue;
    const index = renderable.indexOf(graph);
    const machine = machineKey(graph, index);
    const machineRoot = node.path === graph.name;
    const compound = node.size.width > STATE_NODE_SIZE || node.size.height > STATE_NODE_SIZE
      || graph.nodes.some((item) => item.parent === node.path);
    const activeParts = graph.currentState.split("/").filter((part) => part.length > 0);
    const active = new Set<string>();
    for (let part = 0; part < activeParts.length; part += 1) {
      active.add(`/${activeParts.slice(0, part + 1).join("/")}`);
    }
    // Ownership namespacing: a child state's parentId is the id of the
    // compound/machine-shell node whose interior contains it, when that
    // container is rendered. Standalone machine roots stay unparented.
    const container = node.source.parent === null
      ? undefined
      : layout.get(namespacedPath(machine, node.source.parent));
    const className = nodeClass(
      node.path,
      graph.currentState,
      active,
      compound,
      machineRoot,
      machineRoot && machineOwnerIndex(renderable, index) !== null,
    );
    nodes.push({
      id: node.id,
      position: {
        x: node.center.x - node.size.width / 2 + origin.x,
        y: node.center.y - node.size.height / 2 + origin.y,
      },
      width: node.size.width,
      height: node.size.height,
      className,
      ...(container === undefined ? {} : { parentId: container.id }),
      data: {
        className,
        label: nodeLabel(node.source.label, node.path, graph.currentState),
        path: node.path,
        machineName: node.machineName,
        machine: node.machine,
      },
    });
  }

  for (const [index, graph] of renderable.entries()) {
    const machine = machineKey(graph, index);
    const known = new Set(graph.nodes.map((node) => node.path));
    const local = new Map<string, LayoutNode>();
    for (const node of graph.nodes) {
      const item = layout.get(namespacedPath(machine, node.path));
      if (item !== undefined) local.set(node.path, item);
    }
    const occurrences = new Map<string, number>();
    for (const target of initialTargets(graph)) {
      const layoutNode = local.get(target);
      if (layoutNode === undefined) continue;
      const initial = initialPosition(layoutNode.center, layoutNode.size);
      include(rectFor(initial, { width: INITIAL_SIZE, height: INITIAL_SIZE }));
      const initialId = `initial:${machine}:${target}`;
      nodes.push({
        id: initialId,
        position: { x: initial.x - INITIAL_SIZE / 2 + origin.x, y: initial.y - INITIAL_SIZE / 2 + origin.y },
        width: INITIAL_SIZE,
        height: INITIAL_SIZE,
        className: "initial-node",
        data: { className: "initial-node", label: "", path: "", machineName: graph.name },
        type: "initial",
      });
      const start = initial;
      const end = boundaryPoint(rectFor(layoutNode.center, layoutNode.size), start);
      const points = shareAxis(start, end)
        ? [start, end]
        : [start, { x: (start.x + end.x) / 2, y: start.y }, { x: (start.x + end.x) / 2, y: end.y }, end];
      const shifted = points.map((point) => shiftedPoint(point, origin));
      edges.push({
        id: `${machine}:initial->${target}`,
        source: initialId,
        target: layoutNode.id,
        type: "straight",
        className: "initial",
        data: { d: pathFor(shifted), eventName: INITIAL_EVENT, labelPosition: labelPosition(shifted) },
      });
    }
    for (const edge of graph.edges) {
      if (edge.eventName === INITIAL_EVENT || !isRenderableGraphEdge(edge, known)) continue;
      const signature = graphEdgeSignature(edge);
      const occurrence = occurrences.get(signature) ?? 0;
      occurrences.set(signature, occurrence + 1);
      const source = local.get(edge.source);
      const target = local.get(edge.target);
      if (source === undefined || target === undefined) continue;
      const kind = classifyEdge(edge.source, edge.target, new Map([...local].map(([path, node]) => [path, node.center])));
      const label = edge.eventName.length > EDGE_LABEL_LIMIT
        ? `${edge.eventName.slice(0, EDGE_LABEL_LIMIT - 1)}…`
        : edge.eventName;
      if (kind === "loop") {
        // Loop-class transitions keep their dedicated semantic stub shapes.
        const points = loopEdgePoints(
          rectFor(source.center, source.size),
          rectFor(target.center, target.size),
          edge.source,
          edge.target,
        );
        const shifted = points.map((point) => shiftedPoint(point, origin));
        edges.push({
          id: graphEdgeIdentity(machine, edge, occurrence),
          source: source.id,
          target: target.id,
          type: "step",
          label,
          className: `${kind}${edge.lastFired ? " last-fired" : ""}`,
          data: {
            d: pathFor(shifted),
            eventName: edge.eventName,
            lastFired: edge.lastFired,
            labelPosition: labelPosition(shifted),
          },
        });
        continue;
      }
      // Taxi/aligned transitions route live through FlowGraph's cable engine
      // (Routes actor, smoothstep fallback until routed, auto-placed labels),
      // so the model ships identity and metadata without a precomputed path.
      edges.push({
        id: graphEdgeIdentity(machine, edge, occurrence),
        source: source.id,
        target: target.id,
        type: "cable",
        label,
        className: `${kind}${edge.lastFired ? " last-fired" : ""}`,
        data: { eventName: edge.eventName, lastFired: edge.lastFired },
      });
    }
  }

  return { nodes, edges, bounds, origin };
}

export function focusBoundsForMachine(args: {
  graphs: readonly MachineGraph[];
  machineName: string;
  model: FlowGraphModel;
}): GraphHit["bounds"] | null {
  const graphs = args.graphs;
  const machineName = args.machineName;
  const model = args.model;
  if (!graphs.some((graph) => graph.name === machineName)) return null;
  const names = new Set([machineName]);
  const byName = new Map(graphs.map((graph) => [graph.name, graph]));
  for (const graph of graphs) {
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
  const focused = model.nodes.filter((node) => typeof node.data["machineName"] === "string" && names.has(node.data["machineName"]));
  if (focused.length === 0) return null;
  const bounds: Bounds = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
  for (const node of focused) {
    bounds.left = Math.min(bounds.left, node.position.x);
    bounds.right = Math.max(bounds.right, node.position.x + (node.width ?? 0));
    bounds.top = Math.min(bounds.top, node.position.y);
    bounds.bottom = Math.max(bounds.bottom, node.position.y + (node.height ?? 0));
  }
  return bounds;
}
