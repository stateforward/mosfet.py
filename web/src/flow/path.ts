import {
  DEFAULT_NODE_HEIGHT,
  DEFAULT_NODE_WIDTH,
  MIN_BOUNDS_SPAN,
  type EdgeType,
  type HandlePosition,
  type Node,
  type Rect,
  type Viewport,
  type ViewportBounds,
  type XYPosition,
} from "./types.ts";

export type BezierPathParams = {
  readonly sourceX: number;
  readonly sourceY: number;
  readonly sourcePosition?: HandlePosition;
  readonly targetX: number;
  readonly targetY: number;
  readonly targetPosition?: HandlePosition;
  readonly curvature?: number;
};

export type StraightPathParams = {
  readonly sourceX: number;
  readonly sourceY: number;
  readonly targetX: number;
  readonly targetY: number;
};

export type SmoothStepPathParams = StraightPathParams & {
  readonly sourcePosition?: HandlePosition;
  readonly targetPosition?: HandlePosition;
  readonly borderRadius?: number;
  readonly offset?: number;
};

function controlOffset(distance: number, curvature: number): number {
  if (distance >= 0) return 0.5 * distance * curvature;
  return curvature * 25 * Math.sqrt(Math.abs(distance));
}

function controlPoint(
  position: HandlePosition,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  curvature: number,
): XYPosition {
  switch (position) {
    case "left":
      return { x: x1 - controlOffset(x1 - x2, curvature), y: y1 };
    case "right":
      return { x: x1 + controlOffset(x2 - x1, curvature), y: y1 };
    case "top":
      return { x: x1, y: y1 - controlOffset(y1 - y2, curvature) };
    case "bottom":
      return { x: x1, y: y1 + controlOffset(y2 - y1, curvature) };
  }
}

export function getBezierPath(params: BezierPathParams): [string, number, number, number, number] {
  const sourcePosition = params.sourcePosition ?? "bottom";
  const targetPosition = params.targetPosition ?? "top";
  const curvature = params.curvature ?? 0.25;
  const sourceControl = controlPoint(
    sourcePosition,
    params.sourceX,
    params.sourceY,
    params.targetX,
    params.targetY,
    curvature,
  );
  const targetControl = controlPoint(
    targetPosition,
    params.targetX,
    params.targetY,
    params.sourceX,
    params.sourceY,
    curvature,
  );
  const labelX = (params.sourceX + 3 * sourceControl.x + 3 * targetControl.x + params.targetX) / 8;
  const labelY = (params.sourceY + 3 * sourceControl.y + 3 * targetControl.y + params.targetY) / 8;
  return [
    `M${params.sourceX},${params.sourceY} C${sourceControl.x},${sourceControl.y} ${targetControl.x},${targetControl.y} ${params.targetX},${params.targetY}`,
    labelX,
    labelY,
    Math.abs(labelX - params.sourceX),
    Math.abs(labelY - params.sourceY),
  ];
}

export function getStraightPath(params: StraightPathParams): [string, number, number, number, number] {
  const labelX = (params.sourceX + params.targetX) / 2;
  const labelY = (params.sourceY + params.targetY) / 2;
  return [
    `M${params.sourceX},${params.sourceY} L${params.targetX},${params.targetY}`,
    labelX,
    labelY,
    Math.abs(labelX - params.sourceX),
    Math.abs(labelY - params.sourceY),
  ];
}

function handleVector(position: HandlePosition, offset: number): XYPosition {
  switch (position) {
    case "left":
      return { x: -offset, y: 0 };
    case "right":
      return { x: offset, y: 0 };
    case "top":
      return { x: 0, y: -offset };
    case "bottom":
      return { x: 0, y: offset };
  }
}

export function getSmoothStepPath(params: SmoothStepPathParams): [string, number, number, number, number] {
  const sourcePosition = params.sourcePosition ?? "bottom";
  const targetPosition = params.targetPosition ?? "top";
  const radius = Math.max(0, params.borderRadius ?? 5);
  const offset = params.offset ?? 20;
  const sourceLeave = handleVector(sourcePosition, offset);
  const targetLeave = handleVector(targetPosition, offset);
  const start = { x: params.sourceX + sourceLeave.x, y: params.sourceY + sourceLeave.y };
  const end = { x: params.targetX + targetLeave.x, y: params.targetY + targetLeave.y };
  const horizontal = Math.abs(end.x - start.x) >= Math.abs(end.y - start.y);
  const mid = horizontal
    ? { x: (start.x + end.x) / 2, y: start.y }
    : { x: start.x, y: (start.y + end.y) / 2 };
  const points: XYPosition[] = [
    { x: params.sourceX, y: params.sourceY },
    start,
    mid,
    horizontal ? { x: mid.x, y: end.y } : { x: end.x, y: mid.y },
    end,
    { x: params.targetX, y: params.targetY },
  ];
  const path = roundedPolyline(points, radius);
  const labelX = (params.sourceX + params.targetX) / 2;
  const labelY = (params.sourceY + params.targetY) / 2;
  return [path, labelX, labelY, Math.abs(labelX - params.sourceX), Math.abs(labelY - params.sourceY)];
}

export function getStepPath(params: SmoothStepPathParams): [string, number, number, number, number] {
  return getSmoothStepPath({ ...params, borderRadius: 0 });
}

function roundedPolyline(points: readonly XYPosition[], radius: number): string {
  const first = points[0];
  if (first === undefined) return "";
  if (points.length === 1 || radius <= 0) {
    return points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`).join(" ");
  }
  const commands = [`M${first.x},${first.y}`];
  for (let index = 1; index < points.length; index += 1) {
    const point = points[index];
    const previous = points[index - 1];
    const next = points[index + 1];
    if (point === undefined || previous === undefined) continue;
    if (next === undefined) {
      commands.push(`L${point.x},${point.y}`);
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
    commands.push(`L${before.x},${before.y}`, `Q${point.x},${point.y} ${after.x},${after.y}`);
  }
  return commands.join(" ");
}

export function edgePath(
  type: EdgeType | string | undefined,
  params: SmoothStepPathParams,
): [string, number, number, number, number] {
  if (type === "straight") return getStraightPath(params);
  if (type === "step") return getStepPath(params);
  if (type === "smoothstep") return getSmoothStepPath(params);
  return getBezierPath(params);
}

export function getNodesBounds(nodes: readonly Node[]): Rect {
  if (nodes.length === 0) return { x: 0, y: 0, width: MIN_BOUNDS_SPAN, height: MIN_BOUNDS_SPAN };
  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  for (const node of nodes) {
    const width = node.width ?? DEFAULT_NODE_WIDTH;
    const height = node.height ?? DEFAULT_NODE_HEIGHT;
    left = Math.min(left, node.position.x);
    top = Math.min(top, node.position.y);
    right = Math.max(right, node.position.x + width);
    bottom = Math.max(bottom, node.position.y + height);
  }
  return {
    x: left,
    y: top,
    width: Math.max(MIN_BOUNDS_SPAN, right - left),
    height: Math.max(MIN_BOUNDS_SPAN, bottom - top),
  };
}

/**
 * Fit a world `bounds` rectangle into a pixel viewport.
 *
 * Inputs: `bounds` `{left,right,top,bottom}` in world units (finite numbers;
 * inverted or zero span is accepted). `origin` is the world-to-viewport
 * translation in CSS pixels. `width`/`height` are the viewport size in CSS
 * pixels. `minZoom`/`maxZoom` are unitless zoom clamps. `padding` is a
 * fraction of the fitted bounds, not pixels: each span is clamped to
 * `MIN_BOUNDS_SPAN`, then multiplied by `(1 + padding)`.
 * Outputs: `Viewport` `{x, y, zoom}`. `x`/`y` are CSS pixels placing the
 * padded bounds in the viewport; `zoom` is unitless and clamped to
 * `[minZoom, maxZoom]`.
 * Ownership: pure; no retained state. The caller owns the returned object.
 * Lifetime: the returned viewport does not alias `args`.
 * Concurrency: runtime-safe/pure.
 * Failure modes: non-finite inputs yield a non-finite viewport; `padding <= -1`
 * yields a non-positive padded span and a non-finite or clamped zoom.
 * Units: bounds and origin in world/CSS pixels; padding is a bounds ratio;
 * zoom is unitless.
 * Classification: runtime-safe.
 */
export function getViewportForBounds(args: {
  bounds: ViewportBounds;
  origin: XYPosition;
  width: number;
  height: number;
  minZoom: number;
  maxZoom: number;
  padding: number;
}): Viewport {
  const spanWidth = Math.max(MIN_BOUNDS_SPAN, args.bounds.right - args.bounds.left);
  const spanHeight = Math.max(MIN_BOUNDS_SPAN, args.bounds.bottom - args.bounds.top);
  const x = args.bounds.left + args.origin.x;
  const y = args.bounds.top + args.origin.y;
  const paddedWidth = spanWidth * (1 + args.padding);
  const paddedHeight = spanHeight * (1 + args.padding);
  const zoom = Math.min(
    args.maxZoom,
    Math.max(args.minZoom, Math.min(args.width / paddedWidth, args.height / paddedHeight)),
  );
  return {
    x: args.width / 2 - (x + spanWidth / 2) * zoom,
    y: args.height / 2 - (y + spanHeight / 2) * zoom,
    zoom,
  };
}

export function handlePoint(
  node: Node,
  position: HandlePosition,
): XYPosition {
  const width = node.width ?? DEFAULT_NODE_WIDTH;
  const height = node.height ?? DEFAULT_NODE_HEIGHT;
  switch (position) {
    case "left":
      return { x: node.position.x, y: node.position.y + height / 2 };
    case "right":
      return { x: node.position.x + width, y: node.position.y + height / 2 };
    case "top":
      return { x: node.position.x + width / 2, y: node.position.y };
    case "bottom":
      return { x: node.position.x + width / 2, y: node.position.y + height };
  }
}
