export const MAX_FLOW_NODES = 1024;
export const MAX_FLOW_EDGES = 2048;
export const CLICK_THRESHOLD = 4;
export const MIN_ZOOM = 0.12;
export const MAX_ZOOM = 2.4;
export const ZOOM_FACTOR = 1.2;
export const FIT_PADDING_RATIO = 0.1;

export type XYPosition = { readonly x: number; readonly y: number };

export type Viewport = { readonly x: number; readonly y: number; readonly zoom: number };

export type Rect = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

export type HandlePosition = "top" | "right" | "bottom" | "left";

export type HandleKind = "source" | "target";

export type EdgeType = "bezier" | "straight" | "step" | "smoothstep";

export type Node = {
  readonly id: string;
  readonly position: XYPosition;
  readonly data: Record<string, unknown>;
  readonly selected?: boolean;
  readonly parentId?: string;
  readonly type?: string;
  readonly width?: number;
  readonly height?: number;
  readonly className?: string;
};

export type Edge = {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
  readonly type?: EdgeType | string;
  readonly label?: string;
  readonly selected?: boolean;
  readonly data?: Record<string, unknown>;
  readonly className?: string;
};

export type NodeClickDetail = { readonly node: Node; readonly originalEvent: Event };
export type EdgeClickDetail = { readonly edge: Edge; readonly originalEvent: Event };
export type ViewportChangeDetail = { readonly viewport: Viewport };
export type ConnectDetail = {
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
};
export type SelectionChangeDetail = {
  readonly nodes: readonly Node[];
  readonly edges: readonly Edge[];
};

export type HandleHit = {
  readonly kind: "handle";
  readonly node: Node;
  readonly handleKind: HandleKind;
  readonly position: HandlePosition;
  readonly id?: string;
};

export type NodeHit = { readonly kind: "node"; readonly node: Node };
export type EdgeHit = { readonly kind: "edge"; readonly edge: Edge };
export type EmptyHit = { readonly kind: "empty" };
export type PointerHit = HandleHit | NodeHit | EdgeHit | EmptyHit;

export type PointerSampleData = {
  readonly pointerId: number;
  readonly client: XYPosition;
  readonly viewport: XYPosition;
  readonly world: XYPosition;
  readonly buttons: number;
  readonly button: number;
  readonly pointerType: string;
  readonly shiftKey: boolean;
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  readonly origin: XYPosition;
  readonly hit: PointerHit;
  readonly eventType: "pointerdown" | "pointermove" | "pointerup" | "pointercancel";
};

export type WheelSampleData = {
  readonly deltaY: number;
  readonly point: XYPosition;
};

export type ViewportBounds = {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
};

export type AdmitRejectedDetail = {
  readonly reason: "too_many_nodes" | "too_many_edges" | "invalid";
  readonly nodeCount: number;
  readonly edgeCount: number;
};

export const DEFAULT_NODE_WIDTH = 150;
export const DEFAULT_NODE_HEIGHT = 40;

export function copyJson(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(copyJson);
  const record = value as Record<string, unknown>;
  const copy: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(record)) {
    if (key === "__proto__" || key === "constructor" || key === "prototype") continue;
    copy[key] = copyJson(entry);
  }
  return copy;
}

export function copyNode(node: Node): Node {
  const data = copyJson(node.data);
  return {
    ...node,
    position: { x: node.position.x, y: node.position.y },
    data: data !== null && typeof data === "object" && !Array.isArray(data)
      ? data as Record<string, unknown>
      : {},
  };
}

export function copyEdge(edge: Edge): Edge {
  if (edge.data === undefined) return { ...edge };
  const data = copyJson(edge.data);
  if (data === null || typeof data !== "object" || Array.isArray(data)) return { ...edge };
  return { ...edge, data: data as Record<string, unknown> };
}
