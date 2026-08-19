import { type MachineGraph, type MachineStateNode } from "./otel/machines.ts";

export const NOW_FILL = "#14b8a6";
export const NOW_INK = "#042f2e";
export const NOW_BORDER = "#2dd4bf";
export const NOW_MARKER = "  ●";
export const INITIAL_EVENT = "hsm/initial";
export const INITIAL_SIZE = 16;
export const INITIAL_GAP = 22;
export const INITIAL_FILL = "#000000";
export const INITIAL_BORDER = "#2dd4bf";
export const INITIAL_BORDER_WIDTH = 2;
export const CANVAS_FILL = "#0b0d12";

export type GraphNodeStyle = {
  backgroundColor: string;
  backgroundOpacity: number;
  borderColor: string;
  borderWidth: number;
  fontWeight: number;
  textColor: string;
  underlayOpacity: number;
  zIndex: number;
};

export type CompoundTitleStyle = {
  backgroundColor: string;
  backgroundOpacity: number;
  padding: number;
  marginY: number;
};

export function compoundTitleStyle(): CompoundTitleStyle {
  return {
    backgroundColor: "#161b22",
    backgroundOpacity: 1,
    padding: 3,
    marginY: 8,
  };
}

export function graphNodeStyle(kind: "active-path" | "current"): GraphNodeStyle {
  const current = kind === "current";
  return {
    backgroundColor: "#161b22",
    backgroundOpacity: 1,
    borderColor: NOW_BORDER,
    borderWidth: current ? 3 : 2,
    fontWeight: current ? 800 : 650,
    textColor: "#d5dbe8",
    underlayOpacity: 0,
    zIndex: current ? 10 : 0,
  };
}

export type Size = { width: number; height: number };
export type Point = { x: number; y: number };

function childrenOf(nodes: readonly MachineStateNode[], parent: string | null): MachineStateNode[] {
  return nodes.filter((node) => node.parent === parent);
}

export function nodeClasses(path: string, currentState: string, active: ReadonlySet<string>): string {
  const classes = ["state"];
  if (path === currentState) {
    classes.push("current");
    return classes.join(" ");
  }
  if (active.has(path)) {
    classes.push("active-path");
  }
  return classes.join(" ");
}

export function nodeLabel(label: string, path: string, currentState: string): string {
  if (path === currentState) {
    return `${label}${NOW_MARKER}`;
  }
  return label;
}

function isVisitedState(path: string, graph: MachineGraph): boolean {
  if (path === graph.currentState) {
    return true;
  }
  return graph.edges.some((edge) => edge.source === path || edge.target === path);
}

export function initialNodeId(target: string): string {
  return `initial:${target}`;
}

export function initialTargets(graph: MachineGraph): string[] {
  const known = new Set(graph.nodes.map((node) => node.path));
  const targets = new Set<string>();
  for (const edge of graph.edges) {
    if (edge.eventName === INITIAL_EVENT && known.has(edge.target)) {
      targets.add(edge.target);
    }
  }
  const realIncoming = new Set<string>();
  for (const edge of graph.edges) {
    if (edge.eventName === INITIAL_EVENT) {
      continue;
    }
    if (known.has(edge.source) && known.has(edge.target)) {
      realIncoming.add(edge.target);
    }
  }
  const regions = new Set<string | null>([null]);
  for (const node of graph.nodes) {
    if (childrenOf(graph.nodes, node.path).length > 0) {
      regions.add(node.path);
    }
  }
  for (const region of regions) {
    const regionHasInitial = [...targets].some((path) => {
      const node = graph.nodes.find((item) => item.path === path);
      return node !== undefined && (node.parent === region || path === region);
    });
    if (regionHasInitial) {
      continue;
    }
    const candidate = childrenOf(graph.nodes, region).find(
      (node) => isVisitedState(node.path, graph) && !realIncoming.has(node.path),
    );
    if (candidate !== undefined) {
      targets.add(candidate.path);
    }
  }
  return [...targets];
}

export function initialPosition(target: Point, size: Size): Point {
  return {
    x: target.x - size.width / 2 - INITIAL_GAP - INITIAL_SIZE / 2,
    y: target.y - size.height / 2 + Math.min(20, size.height / 2),
  };
}
