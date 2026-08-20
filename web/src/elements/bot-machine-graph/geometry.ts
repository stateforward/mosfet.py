import { graphEdgeSignature, type Point, type Size } from "../../machine-graph-view.ts";
import type { MachineEdge } from "../../otel/machines.ts";

export type EdgeKind = "taxi" | "aligned" | "loop";
export type Rect = { left: number; right: number; top: number; bottom: number; center: Point };
export type Bounds = { left: number; right: number; top: number; bottom: number };

const AXIS_EPS = 0.5;
const LOOP_STUB_X = 28;
const LOOP_STUB_Y = 36;

export function ancestorSet(path: string): Set<string> {
  const parts = path.split("/").filter((part) => part.length > 0);
  const values = new Set<string>();
  for (let index = 0; index < parts.length; index += 1) {
    values.add(`/${parts.slice(0, index + 1).join("/")}`);
  }
  return values;
}

export function shareAxis(left: Point, right: Point): boolean {
  return Math.abs(left.x - right.x) <= AXIS_EPS || Math.abs(left.y - right.y) <= AXIS_EPS;
}

export function isDescendantPath(descendant: string, ancestor: string): boolean {
  return descendant.startsWith(`${ancestor}/`);
}

export function edgeOccurrenceIndex(occurrences: Map<string, number>, edge: MachineEdge): number {
  const signature = graphEdgeSignature(edge);
  const occurrence = occurrences.get(signature) ?? 0;
  occurrences.set(signature, occurrence + 1);
  return occurrence;
}

export function classifyEdge(source: string, target: string, positions: Map<string, Point>): EdgeKind {
  if (source === target || isDescendantPath(source, target) || isDescendantPath(target, source)) {
    return "loop";
  }
  const from = positions.get(source);
  const to = positions.get(target);
  return from !== undefined && to !== undefined && shareAxis(from, to) ? "aligned" : "taxi";
}

export function rectFor(center: Point, size: Size): Rect {
  return {
    left: center.x - size.width / 2,
    right: center.x + size.width / 2,
    top: center.y - size.height / 2,
    bottom: center.y + size.height / 2,
    center,
  };
}

export function boundaryPoint(rect: Rect, toward: Point): Point {
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

export function pathFor(points: readonly Point[], radius = 9): string {
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

export function labelPosition(points: readonly Point[]): Point {
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

export function offsetPoints(points: readonly Point[], offset: number): Point[] {
  if (offset === 0 || points.length < 2) return [...points];
  const first = points[0];
  const second = points[1];
  if (first === undefined || second === undefined) return [...points];
  const horizontal = Math.abs(second.x - first.x) >= Math.abs(second.y - first.y);
  return points.map((point) => ({
    x: point.x + (horizontal ? 0 : offset),
    y: point.y + (horizontal ? offset : 0),
  }));
}

export function edgePoints(
  source: Rect,
  target: Rect,
  kind: EdgeKind,
  sourcePath: string,
  targetPath: string,
  offset: number,
): Point[] {
  if (kind === "loop" && sourcePath === targetPath) {
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
  if (kind === "aligned") return [start, end];
  if (kind === "loop") {
    const inner = isDescendantPath(sourcePath, targetPath) ? source : target;
    const channelX = inner.left - LOOP_STUB_X;
    return [start, { x: channelX, y: start.y }, { x: channelX, y: end.y }, end];
  }
  if (Math.abs(end.x - start.x) >= Math.abs(end.y - start.y)) {
    const channelX = (start.x + end.x) / 2 + offset;
    return [start, { x: channelX, y: start.y }, { x: channelX, y: end.y }, end];
  }
  const channelY = (start.y + end.y) / 2 + offset;
  return [start, { x: start.x, y: channelY }, { x: end.x, y: channelY }, end];
}

export function shiftedPoint(point: Point, origin: Point): Point {
  return { x: point.x + origin.x, y: point.y + origin.y };
}
