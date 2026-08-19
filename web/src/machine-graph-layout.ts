import { type MachineGraph, type MachineStateNode } from "./otel/machines.ts";
import { machineKey, namespacedPath, type Point, type Size } from "./machine-graph-view.ts";

const LEAF_WIDTH = 120;
const LEAF_HEIGHT = 40;
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
  const width = childSizes.reduce((total, size) => total + size.width + GAP, PAD_X) - GAP + PAD_X;
  const height = Math.max(...childSizes.map((size) => size.height)) + PAD_Y + PAD_X;
  return { width, height };
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
  return {
    width: sizes.reduce((total, size) => total + size.width + GAP * 2, 0),
    height: Math.max(...sizes.map((size) => size.height)),
  };
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
  let childX = originX + PAD_X;
  const childY = originY + PAD_Y;
  for (const child of stateChildren) {
    const childSize = placeState(graphs, ownership, index, child.path, childX, childY, positions);
    childX += childSize.width + GAP;
  }
  for (const childIndex of machineChildren) {
    const childSize = placeMachine(graphs, ownership, childIndex, childX, childY, positions);
    childX += childSize.width + GAP;
  }
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
  let childX = originX;
  for (const root of roots) {
    const childSize = placeState(graphs, ownership, index, root.path, childX, originY, positions);
    childX += childSize.width + GAP * 2;
  }
  return size;
}

export function environmentBoxPositions(graphs: readonly MachineGraph[]): Map<string, Point> {
  const positions = new Map<string, Point>();
  const ownership = ownershipLayout(graphs);
  let originX = 0;
  graphs.forEach((_graph, index) => {
    if (ownership.ownerByIndex.has(index)) {
      return;
    }
    const size = measureMachine(graphs, ownership, index);
    placeMachine(graphs, ownership, index, originX, 0, positions);
    originX += size.width + GAP * 3;
  });
  return positions;
}

export function machineOwnerIndex(graphs: readonly MachineGraph[], index: number): number | null {
  return ownershipLayout(graphs).ownerByIndex.get(index) ?? null;
}
