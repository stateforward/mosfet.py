import { type MachineGraph, type MachineStateNode } from "./otel/machines.ts";
import {
  machineKey,
  namespacedPath,
  STATE_NODE_SIZE,
  type Point,
  type Size,
} from "./machine-graph-view.ts";

const LEAF_WIDTH = STATE_NODE_SIZE;
const LEAF_HEIGHT = STATE_NODE_SIZE;
const GAP = 22;
const PAD_X = 22;
const PAD_Y = 36;

function childrenOf(nodes: readonly MachineStateNode[], parent: string | null): MachineStateNode[] {
  return nodes.filter((node) => node.parent === parent);
}

export type OwnershipLayout = {
  ownerByIndex: Map<number, number>;
  childrenByIndex: Map<number, number[]>;
};

function hasRoot(graph: MachineGraph): boolean {
  return graph.nodes.some((node) => node.path === graph.name && node.parent === null);
}

function hasRenderableOwnerChain(
  graphs: readonly MachineGraph[],
  byName: ReadonlyMap<string, number>,
  index: number,
): boolean {
  const visited = new Set<number>();
  let current: number | undefined = index;
  while (current !== undefined) {
    if (visited.has(current)) {
      return false;
    }
    visited.add(current);
    const graph = graphs[current];
    if (graph === undefined || !hasRoot(graph)) {
      return false;
    }
    const owner = graph.owner;
    if (owner === undefined || owner === null) {
      return true;
    }
    if (typeof owner !== "string") {
      return false;
    }
    current = byName.get(owner);
  }
  return false;
}

/**
 * Graphs admitted to the renderer must have a root state and, when owned,
 * remain nested under an existing owner through a complete acyclic chain.
 */
export function renderableGraphs(graphs: readonly MachineGraph[]): MachineGraph[] {
  const byName = new Map<string, number>();
  graphs.forEach((graph, index) => {
    if (!byName.has(graph.name)) {
      byName.set(graph.name, index);
    }
  });
  return graphs.filter((_graph, index) => hasRenderableOwnerChain(graphs, byName, index));
}

export function ownershipLayout(graphs: readonly MachineGraph[]): OwnershipLayout {
  const byName = new Map<string, number>();
  graphs.forEach((graph, index) => {
    if (!byName.has(graph.name)) {
      byName.set(graph.name, index);
    }
  });
  const candidates = new Map<number, number>();
  graphs.forEach((graph, index) => {
    if (graph.owner === undefined || graph.owner === null || graph.owner === graph.name || !hasRoot(graph)) {
      return;
    }
    const ownerIndex = byName.get(graph.owner);
    const ownerGraph = ownerIndex === undefined ? undefined : graphs[ownerIndex];
    if (ownerIndex !== undefined && ownerGraph !== undefined && hasRoot(ownerGraph)) {
      candidates.set(index, ownerIndex);
    }
  });
  const ownerByIndex = new Map<number, number>();
  for (const [index, candidate] of candidates) {
    const visited = new Set<number>();
    let current: number | undefined = index;
    let valid = true;
    while (current !== undefined) {
      if (visited.has(current)) {
        valid = false;
        break;
      }
      visited.add(current);
      current = candidates.get(current);
    }
    if (valid) {
      ownerByIndex.set(index, candidate);
    }
  }
  const childrenByIndex = new Map<number, number[]>();
  for (const [index, owner] of ownerByIndex) {
    const children = childrenByIndex.get(owner) ?? [];
    children.push(index);
    childrenByIndex.set(owner, children);
  }
  return { ownerByIndex, childrenByIndex };
}

type MachineLayout = {
  graph: MachineGraph;
  index: number;
  key: string;
};

function graphLayout(graphs: readonly MachineGraph[], index: number): MachineLayout {
  const graph = graphs[index];
  if (graph === undefined) {
    throw new Error(`missing machine graph at index ${String(index)}`);
  }
  return { graph, index, key: machineKey(graph, index) };
}

type GridMetrics = {
  columns: number;
  columnWidths: number[];
  rowHeights: number[];
  width: number;
  height: number;
};

function gridColumns(itemCount: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(itemCount)));
}

function gridMetrics(
  sizes: readonly Size[],
  paddingX: number,
  paddingY: number,
  paddingBottom = PAD_X,
): GridMetrics {
  const columns = gridColumns(sizes.length);
  const rows = Math.ceil(sizes.length / columns);
  const columnWidths = Array.from({ length: columns }, () => 0);
  const rowHeights = Array.from({ length: rows }, () => 0);
  sizes.forEach((size, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    columnWidths[column] = Math.max(columnWidths[column] ?? 0, size.width);
    rowHeights[row] = Math.max(rowHeights[row] ?? 0, size.height);
  });
  const contentWidth = columnWidths.reduce((total, width) => total + width, 0) + GAP * (columns - 1);
  const contentHeight = rowHeights.reduce((total, height) => total + height, 0) + GAP * (rows - 1);
  return {
    columns,
    columnWidths,
    rowHeights,
    width: contentWidth + paddingX * 2,
    height: contentHeight + paddingY + paddingBottom,
  };
}

function placeGrid(
  sizes: readonly Size[],
  metrics: GridMetrics,
  originX: number,
  originY: number,
  place: (index: number, x: number, y: number) => void,
): void {
  const columnOffsets: number[] = [];
  let columnX = originX;
  for (const width of metrics.columnWidths) {
    columnOffsets.push(columnX);
    columnX += width + GAP;
  }
  const rowOffsets: number[] = [];
  let rowY = originY;
  for (const height of metrics.rowHeights) {
    rowOffsets.push(rowY);
    rowY += height + GAP;
  }
  sizes.forEach((size, index) => {
    const column = index % metrics.columns;
    const row = Math.floor(index / metrics.columns);
    const cellWidth = metrics.columnWidths[column] ?? size.width;
    const cellHeight = metrics.rowHeights[row] ?? size.height;
    place(
      index,
      (columnOffsets[column] ?? originX) + (cellWidth - size.width) / 2,
      (rowOffsets[row] ?? originY) + (cellHeight - size.height) / 2,
    );
  });
}

export function measureState(
  graphs: readonly MachineGraph[],
  ownership: OwnershipLayout,
  index: number,
  path: string,
): Size {
  const layout = graphLayout(graphs, index);
  const stateChildren = childrenOf(layout.graph.nodes, path);
  const machineChildren = path === layout.graph.name
    ? (ownership.childrenByIndex.get(index) ?? []).map((child) => graphLayout(graphs, child))
    : [];
  const childSizes = [
    ...stateChildren.map((child) => measureState(graphs, ownership, index, child.path)),
    ...machineChildren.map((child) => measureMachine(graphs, ownership, child.index)),
  ];
  if (childSizes.length === 0) {
    return { width: LEAF_WIDTH, height: LEAF_HEIGHT };
  }
  const metrics = gridMetrics(childSizes, PAD_X, PAD_Y);
  return { width: metrics.width, height: metrics.height };
}

function measureMachine(
  graphs: readonly MachineGraph[],
  ownership: OwnershipLayout,
  index: number,
): Size {
  const graph = graphLayout(graphs, index).graph;
  const roots = childrenOf(graph.nodes, null);
  if (roots.length === 0) {
    return { width: LEAF_WIDTH, height: LEAF_HEIGHT };
  }
  const sizes = roots.map((root) => measureState(graphs, ownership, index, root.path));
  const metrics = gridMetrics(sizes, 0, 0, 0);
  return { width: metrics.width, height: metrics.height };
}

function placeState(
  graphs: readonly MachineGraph[],
  ownership: OwnershipLayout,
  index: number,
  path: string,
  originX: number,
  originY: number,
  positions: Map<string, Point>,
): Size {
  const layout = graphLayout(graphs, index);
  const size = measureState(graphs, ownership, index, path);
  positions.set(namespacedPath(layout.key, path), {
    x: originX + size.width / 2,
    y: originY + size.height / 2,
  });
  const stateChildren = childrenOf(layout.graph.nodes, path);
  const machineChildren = path === layout.graph.name ? ownership.childrenByIndex.get(index) ?? [] : [];
  const placements = [
    ...stateChildren.map((child) => ({
      size: measureState(graphs, ownership, index, child.path),
      place: (x: number, y: number) => placeState(graphs, ownership, index, child.path, x, y, positions),
    })),
    ...machineChildren.map((childIndex) => ({
      size: measureMachine(graphs, ownership, childIndex),
      place: (x: number, y: number) => placeMachine(graphs, ownership, childIndex, x, y, positions),
    })),
  ];
  const metrics = gridMetrics(placements.map((item) => item.size), PAD_X, PAD_Y);
  placeGrid(
    placements.map((item) => item.size),
    metrics,
    originX + PAD_X,
    originY + PAD_Y,
    (index, x, y) => placements[index]?.place(x, y),
  );
  return size;
}

function placeMachine(
  graphs: readonly MachineGraph[],
  ownership: OwnershipLayout,
  index: number,
  originX: number,
  originY: number,
  positions: Map<string, Point>,
): Size {
  const graph = graphLayout(graphs, index).graph;
  const roots = childrenOf(graph.nodes, null);
  const size = measureMachine(graphs, ownership, index);
  const rootSizes = roots.map((root) => measureState(graphs, ownership, index, root.path));
  const metrics = gridMetrics(rootSizes, 0, 0);
  placeGrid(rootSizes, metrics, originX, originY, (rootIndex, x, y) => {
    const root = roots[rootIndex];
    if (root !== undefined) {
      placeState(graphs, ownership, index, root.path, x, y, positions);
    }
  });
  return size;
}

export function environmentBoxPositions(graphs: readonly MachineGraph[]): Map<string, Point> {
  const positions = new Map<string, Point>();
  const renderable = renderableGraphs(graphs);
  const ownership = ownershipLayout(renderable);
  let originX = 0;
  renderable.forEach((_graph, index) => {
    if (ownership.ownerByIndex.has(index)) {
      return;
    }
    const size = measureMachine(renderable, ownership, index);
    placeMachine(renderable, ownership, index, originX, 0, positions);
    originX += size.width + GAP * 3;
  });
  return positions;
}

export function machineOwnerIndex(graphs: readonly MachineGraph[], index: number): number | null {
  return ownershipLayout(graphs).ownerByIndex.get(index) ?? null;
}
