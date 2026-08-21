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
} from "./geometry.ts";

const WORLD_PADDING = 56;
const EDGE_LABEL_LIMIT = 30;

export type GraphHit = {
  readonly machineName: string;
  readonly path: string;
  readonly bounds: { left: number; right: number; top: number; bottom: number };
};

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
    const machineRoot = node.path === graph.name;
    const compound = node.size.width > STATE_NODE_SIZE || node.size.height > STATE_NODE_SIZE
      || graph.nodes.some((item) => item.parent === node.path);
    const className = nodeClass(
      node.path,
      graph.currentState,
      ancestorSet(graph.currentState),
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
      const occurrence = edgeOccurrenceIndex(occurrences, edge);
      const source = local.get(edge.source);
      const target = local.get(edge.target);
      if (source === undefined || target === undefined) continue;
      const kind = classifyEdge(edge.source, edge.target, new Map([...local].map(([path, node]) => [path, node.center])));
      const offset = kind === "loop" ? 0 : (occurrence % 3 - 1) * 12;
      const points = offsetPoints(
        edgePoints(rectFor(source.center, source.size), rectFor(target.center, target.size), kind, edge.source, edge.target, offset),
        offset,
      );
      const shifted = points.map((point) => shiftedPoint(point, origin));
      const label = edge.eventName.length > EDGE_LABEL_LIMIT
        ? `${edge.eventName.slice(0, EDGE_LABEL_LIMIT - 1)}…`
        : edge.eventName;
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
    }
  }

  return { nodes, edges, bounds, origin };
}

export function focusBoundsForMachine(graphs: readonly MachineGraph[], machineName: string, model: FlowGraphModel): GraphHit["bounds"] | null {
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
