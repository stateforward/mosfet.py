export const MAX_FLOW_NODES = 1024;
export const MAX_FLOW_EDGES = 2048;
export const CLICK_THRESHOLD = 4;
export const MIN_ZOOM = 0.12;
export const MAX_ZOOM = 2.4;
export const ZOOM_FACTOR = 1.2;
export const FIT_PADDING_RATIO = 0.1;
export const MIN_BOUNDS_SPAN = 1;

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

export type ResizeDirection = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";

export const MIN_RESIZE_WIDTH = 10;
export const MIN_RESIZE_HEIGHT = 10;

export type ResizeConstraints = {
  readonly minWidth: number;
  readonly minHeight: number;
  readonly maxWidth?: number;
  readonly maxHeight?: number;
  readonly keepAspectRatio: boolean;
};

export type ResizeBounds = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

export type EdgeType = "bezier" | "straight" | "step" | "smoothstep" | "cable";

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

export type PointerOrigin = {
  readonly pointerId: number;
  readonly clientX: number;
  readonly clientY: number;
  readonly type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel";
};

export type KeyboardOrigin = {
  readonly type: "keydown";
  readonly key: "Enter" | " ";
};

export type ClickOrigin = {
  readonly type: "click";
};

export type ActivationOrigin = PointerOrigin | KeyboardOrigin | ClickOrigin;

export type NodeActivateData = {
  readonly nodeId: string;
  readonly key?: KeyboardOrigin["key"];
};

/**
 * Detail of the `flow-node-click` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the click already ran on the graph — exclusive selection of
 * the clicked node is already requested unless the originating pointer sample
 * is meta/ctrl additive, in which case membership of that node is toggled, and
 * `node` is a copy (mutating `detail.node` does not mutate `graph.nodes`).
 * Listeners observe; they do not apply the click. `preventDefault()` has no
 * effect because the event cannot be canceled.
 */
export type NodeClickDetail = { readonly node: Node; readonly originalEvent: ActivationOrigin };
/**
 * Detail of the `flow-edge-click` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the click already ran on the graph — selection of the
 * clicked edge is already requested, and `edge` is a copy. The graph
 * topology is unchanged. `preventDefault()` has no effect because the event
 * cannot be canceled.
 */
export type EdgeClickDetail = { readonly edge: Edge; readonly originalEvent: PointerOrigin };
/**
 * Detail of the `flow-viewport-change` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the viewport is already applied on the graph (the committed
 * transform and `getViewport()` value). Listeners observe the outcome; they
 * neither apply nor roll back the change. `preventDefault()` has no effect
 * because the event cannot be canceled.
 */
export type ViewportChangeDetail = { readonly viewport: Viewport };
/**
 * Detail of the `flow-connect` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the dispatcher does NOT add an edge. The finished connection
 * gesture is reported here with its `source`/`target` handles; persisting a
 * new edge is the listener's job, if it chooses to. `preventDefault()` has no
 * effect because the event cannot be canceled.
 */
export type ConnectDetail = {
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
};
/**
 * Detail of the `flow-selection-change` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the selection is already committed when this event is
 * dispatched — the graph has already replaced its selected node/edge sets,
 * and `nodes`/`edges` are copies of the committed selection. Listeners observe
 * the outcome; they neither apply nor roll back the change, and
 * `preventDefault()` has no effect because the event cannot be canceled.
 */
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

export type ResizeHit = {
  readonly kind: "resize";
  readonly node: Node;
  readonly direction: ResizeDirection;
  readonly minWidth: number;
  readonly minHeight: number;
  readonly keepAspectRatio: boolean;
  readonly maxWidth?: number;
  readonly maxHeight?: number;
};

/** Active resize input channel. Pointer and keyboard cannot share one session. */
export type ResizeChannel = "pointer" | "keyboard";

/** Payload of `FlowGraph.resize_key`. `hit` is null for Escape with no control under the path. */
export type ResizeKeyData = {
  readonly key: string;
  readonly hit: ResizeHit | null;
};

/**
 * Single offer predicate for resize chrome, hit-test, and start.
 * Canonical policy is `FlowGraph.nodesResizable`. `visible` is an author
 * override on `flow-node-resizer` (`true` force-show, `false` force-hide,
 * `undefined` auto). Auto offers only when policy is on and the node is selected.
 */
export function resizeOffered(args: {
  readonly policy: boolean;
  readonly selected: boolean;
  readonly visible: boolean | undefined;
}): boolean {
  if (!args.policy) return false;
  if (args.visible === false) return false;
  if (args.visible === true) return true;
  return args.selected;
}

export type NodeHit = { readonly kind: "node"; readonly node: Node };
export type EdgeHit = { readonly kind: "edge"; readonly edge: Edge };
export type EmptyHit = { readonly kind: "empty" };
export type PointerHit = HandleHit | ResizeHit | NodeHit | EdgeHit | EmptyHit;

/**
 * Detail of `flow-node-resize-start`, `flow-node-resize`, and
 * `flow-node-resize-end`.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the bounds are already applied on the graph for move/end
 * (start reports the origin bounds). `node` is a copy. Listeners observe;
 * they do not apply the resize. `preventDefault()` has no effect.
 */
export type NodeResizeDetail = {
  readonly node: Node;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

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
  readonly originalEvent: PointerOrigin;
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

/**
 * Detail of the `flow-admit-rejected` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the rejected write was NOT applied — the graph's prior
 * `nodes`/`edges` are unchanged and remain committed. `reason` identifies why
 * the write was dropped, and `nodeCount`/`edgeCount` describe the rejected
 * write, not the committed graph. Listeners only observe; there is nothing to
 * roll back, and `preventDefault()` has no effect because the event cannot be
 * canceled.
 */
export type AdmitRejectedDetail = {
  readonly reason: "too_many_nodes" | "too_many_edges" | "invalid";
  readonly nodeCount: number;
  readonly edgeCount: number;
};

export const DEFAULT_NODE_WIDTH = 150;
export const DEFAULT_NODE_HEIGHT = 40;
/** Maximum object/array nesting `copyJson` will copy. Deeper or cyclic values fail the copy. */
export const MAX_JSON_DEPTH = 32;

export type CopyResult<T> = { readonly ok: true; readonly value: T } | { readonly ok: false };

type CopyJsonResult = CopyResult<unknown>;

const initialJsonDepth = 0;

function isCopiedRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function jsonIsCopyable(args: { value: unknown; stack: object[]; depth: number }): boolean {
  const { value, stack, depth } = args;
  if (value === null || typeof value !== "object") return true;
  if (depth >= MAX_JSON_DEPTH) return false;
  if (stack.includes(value)) return false;
  stack.push(value);
  const nestedDepth = depth + 1;
  if (Array.isArray(value)) {
    for (const entry of value) {
      if (!jsonIsCopyable({ value: entry, stack, depth: nestedDepth })) {
        stack.pop();
        return false;
      }
    }
    stack.pop();
    return true;
  }
  const record = value as Record<string, unknown>;
  for (const [key, entry] of Object.entries(record)) {
    if (key === "__proto__" || key === "constructor" || key === "prototype") continue;
    if (!jsonIsCopyable({ value: entry, stack, depth: nestedDepth })) {
      stack.pop();
      return false;
    }
  }
  stack.pop();
  return true;
}

function copyJsonResult(args: { value: unknown; stack: object[]; depth: number }): CopyJsonResult {
  const { value, stack, depth } = args;
  if (value === null || typeof value !== "object") return { ok: true, value };
  if (depth >= MAX_JSON_DEPTH) return { ok: false };
  if (stack.includes(value)) return { ok: false };
  stack.push(value);
  const nestedDepth = depth + 1;
  if (Array.isArray(value)) {
    const items: unknown[] = [];
    for (const entry of value) {
      const copied = copyJsonResult({ value: entry, stack, depth: nestedDepth });
      if (!copied.ok) {
        stack.pop();
        return copied;
      }
      items.push(copied.value);
    }
    stack.pop();
    return { ok: true, value: items };
  }
  const record = value as Record<string, unknown>;
  const copy: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(record)) {
    if (key === "__proto__" || key === "constructor" || key === "prototype") continue;
    const copied = copyJsonResult({ value: entry, stack, depth: nestedDepth });
    if (!copied.ok) {
      stack.pop();
      return copied;
    }
    copy[key] = copied.value;
  }
  stack.pop();
  return { ok: true, value: copy };
}

/**
 * Deep-copy JSON-like `value`, skipping `__proto__` / `constructor` / `prototype` keys.
 *
 * Inputs: unknown nested objects and arrays. Outputs: `{ ok: true, value }`
 * with a new tree of the same shape, including `{ ok: true, value: undefined }`
 * when `value` is undefined. Ownership: the caller owns `value` on success;
 * the input is not retained. Lifetime: one call. Concurrency: synchronous.
 * Failure modes: cyclic or over-deep values return `{ ok: false }` and do not
 * throw. Classification: runtime-safe.
 */
export function copyJson(value: unknown): CopyResult<unknown> {
  const initialStack: object[] = [];
  return copyJsonResult({ value, stack: initialStack, depth: initialJsonDepth });
}

/**
 * True when `copyJson(value)` would succeed.
 *
 * Inputs: unknown nested objects and arrays. Outputs: `true` when the value
 * is acyclic and nested at most `MAX_JSON_DEPTH`, including `true` for
 * primitives and `undefined`. Ownership: `value` is not retained and no
 * copied tree is allocated; only a visit stack is used. Lifetime: one call.
 * Concurrency: synchronous. Failure modes: cyclic or over-deep values return
 * `false` and do not throw. Classification: runtime-safe.
 */
export function jsonCopyable(value: unknown): boolean {
  const visitStack: object[] = [];
  return jsonIsCopyable({ value, stack: visitStack, depth: initialJsonDepth });
}

/**
 * Copy `node` with owned `position` and JSON `data`.
 *
 * Inputs: a Node-shaped record. Outputs: `{ ok: true, value }` with copied
 * finite `position` and record `data`, or `{ ok: false }` when `position` is
 * missing, not a record, or has non-finite `x`/`y`, or when `data` is omitted,
 * null, a non-record, cyclic, or nested deeper than `MAX_JSON_DEPTH`.
 * Ownership: the caller owns `value`; `node` is not retained. Lifetime: one
 * call. Concurrency: synchronous. Failure modes: `{ ok: false }` and does not
 * throw. Classification: runtime-safe.
 */
export function copyNode(node: Node): CopyResult<Node> {
  const rawPosition: unknown = node.position;
  if (!isCopiedRecord(rawPosition) || !isFiniteNumber(rawPosition["x"]) || !isFiniteNumber(rawPosition["y"])) {
    return { ok: false };
  }
  const position = { x: rawPosition["x"], y: rawPosition["y"] };
  const initialStack: object[] = [];
  const copied = copyJsonResult({ value: node.data, stack: initialStack, depth: initialJsonDepth });
  if (!copied.ok || !isCopiedRecord(copied.value)) return { ok: false };
  return { ok: true, value: { ...node, position, data: copied.value } };
}

/**
 * Copy `edge` with owned JSON `data` when present.
 *
 * Inputs: an Edge-shaped record. Outputs: `{ ok: true, value }` with copied
 * `data`, or `{ ok: true, value }` without `data` when it was omitted, or
 * `{ ok: false }` when `data` is null, a non-record, cyclic, or nested deeper
 * than `MAX_JSON_DEPTH`. Ownership: the caller owns `value`; `edge` is not
 * retained. Lifetime: one call. Concurrency: synchronous. Failure modes:
 * `{ ok: false }` and does not throw. Classification: runtime-safe.
 */
export function copyEdge(edge: Edge): CopyResult<Edge> {
  if (edge.data === undefined) return { ok: true, value: { ...edge } };
  const initialStack: object[] = [];
  const copied = copyJsonResult({ value: edge.data, stack: initialStack, depth: initialJsonDepth });
  if (!copied.ok || !isCopiedRecord(copied.value)) return { ok: false };
  return { ok: true, value: { ...edge, data: copied.value } };
}
