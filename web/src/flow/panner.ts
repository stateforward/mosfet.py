import * as hsm from "../hsm.ts";

import { MAX_ZOOM, MIN_ZOOM, type Viewport, type XYPosition } from "./types.ts";

export type ViewportPoint = XYPosition;
export type ViewportBounds = {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
};
export type ViewportMetrics = {
  readonly width: number;
  readonly height: number;
  readonly bounds: ViewportBounds;
  readonly origin: ViewportPoint;
};
export type ViewportTransform = { readonly scale: number; readonly pan: ViewportPoint };

export type PanPointerData = {
  readonly pointerId: number;
  readonly point: ViewportPoint;
};
export type ZoomData = {
  readonly scale?: number;
  readonly deltaY?: number;
  readonly point?: ViewportPoint;
};
export type FitData = {
  readonly bounds: ViewportBounds;
  readonly metrics: ViewportMetrics;
  readonly reason?: string;
};
export type ViewportData = Viewport;

const FIT_PADDING = 28;
const MAX_FIT_ZOOM = 1.2;
const ZOOM_STEP = 0.0015;

export class Panner extends hsm.Instance {
  static readonly panStartEvent = { name: "pan_start", kind: hsm.Kinds.Event } as const;
  static readonly cursorMoveEvent = { name: "cursor_move", kind: hsm.Kinds.Event } as const;
  static readonly panEndEvent = { name: "pan_end", kind: hsm.Kinds.Event } as const;
  static readonly zoomEvent = { name: "zoom", kind: hsm.Kinds.Event } as const;
  static readonly fitEvent = { name: "fit", kind: hsm.Kinds.Event } as const;
  static readonly viewportEvent = { name: "viewport_set", kind: hsm.Kinds.Event } as const;
  static readonly transformEvent = { name: "transform_changed", kind: hsm.Kinds.Event } as const;
  static readonly panningEvent = { name: "panning_changed", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Panner",
    hsm.initial(hsm.target("ready")),
    hsm.state(
      "ready",
      hsm.initial(hsm.target("fixed")),
      hsm.transition(hsm.on(Panner.zoomEvent.name), hsm.effect(Panner.applyZoom)),
      hsm.transition(hsm.on(Panner.fitEvent.name), hsm.effect(Panner.applyFit)),
      hsm.transition(hsm.on(Panner.viewportEvent.name), hsm.effect(Panner.applyViewport)),
      hsm.state(
        "fixed",
        hsm.transition(
          hsm.on(Panner.panStartEvent.name),
          hsm.target("../single"),
          hsm.effect(Panner.startPan),
        ),
      ),
      hsm.state(
        "single",
        hsm.transition(hsm.on(Panner.panStartEvent.name), hsm.target("../pinch"), hsm.effect(Panner.startPan)),
        hsm.transition(hsm.on(Panner.panEndEvent.name), hsm.target("../fixed"), hsm.effect(Panner.endPan)),
        hsm.transition(hsm.on(Panner.cursorMoveEvent.name), hsm.effect(Panner.dragPan)),
      ),
      hsm.state(
        "pinch",
        hsm.transition(hsm.on(Panner.panStartEvent.name), hsm.effect(Panner.startPan)),
        hsm.transition(hsm.on(Panner.panEndEvent.name), hsm.target("../ending"), hsm.effect(Panner.endPan)),
        hsm.transition(hsm.on(Panner.cursorMoveEvent.name), hsm.effect(Panner.pinchPan)),
      ),
      hsm.choice(
        "ending",
        hsm.transition(hsm.guard(Panner.hasTwoPointers), hsm.target("pinch")),
        hsm.transition(
          hsm.guard(Panner.hasOnePointer),
          hsm.target("single"),
          hsm.effect(Panner.resumeSingle),
        ),
        hsm.transition(hsm.target("fixed")),
      ),
    ),
  );

  scale = 1;
  pan: ViewportPoint = { x: 0, y: 0 };
  #pointers = new Map<number, ViewportPoint>();
  #dragStart: { pointerId: number; point: ViewportPoint; pan: ViewportPoint } | null = null;
  #pinchStart: { distance: number; scale: number } | null = null;

  constructor() {
    super();
  }

  get transform(): ViewportTransform {
    return { scale: this.scale, pan: this.pan };
  }

  get viewport(): Viewport {
    return { x: this.pan.x, y: this.pan.y, zoom: this.scale };
  }

  static startPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointer = panPointerOf(event.data);
    if (pointer === null) return;
    instance.#pointers.set(pointer.pointerId, pointer.point);
    if (instance.#pointers.size === 1) {
      instance.#dragStart = { pointerId: pointer.pointerId, point: pointer.point, pan: { ...instance.pan } };
      instance.#setPanning(true);
      return;
    }
    instance.#dragStart = null;
    const points = [...instance.#pointers.values()];
    const first = points[0];
    const second = points[1];
    if (first !== undefined && second !== undefined) {
      instance.#pinchStart = {
        distance: Math.max(1, Math.hypot(second.x - first.x, second.y - first.y)),
        scale: instance.scale,
      };
    }
    instance.#setPanning(false);
  }

  static dragPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointer = panPointerOf(event.data);
    if (pointer === null || !instance.#pointers.has(pointer.pointerId)) return;
    instance.#pointers.set(pointer.pointerId, pointer.point);
    if (instance.#dragStart?.pointerId !== pointer.pointerId) return;
    instance.#setTransform(instance.scale, {
      x: instance.#dragStart.pan.x + pointer.point.x - instance.#dragStart.point.x,
      y: instance.#dragStart.pan.y + pointer.point.y - instance.#dragStart.point.y,
    });
  }

  static pinchPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner) || instance.#pinchStart === null) return;
    const pointer = panPointerOf(event.data);
    if (pointer === null || !instance.#pointers.has(pointer.pointerId)) return;
    instance.#pointers.set(pointer.pointerId, pointer.point);
    const points = [...instance.#pointers.values()];
    const first = points[0];
    const second = points[1];
    if (first === undefined || second === undefined) return;
    const midpoint = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
    const scale = instance.#pinchStart.scale
      * Math.hypot(second.x - first.x, second.y - first.y)
      / instance.#pinchStart.distance;
    instance.#setZoom(scale, midpoint);
  }

  static hasTwoPointers(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): boolean {
    return instance instanceof Panner && instance.#pointers.size >= 2;
  }

  static hasOnePointer(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): boolean {
    return instance instanceof Panner && instance.#pointers.size === 1;
  }

  static endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointerId = pointerIdOf(event.data);
    if (pointerId !== null) instance.#pointers.delete(pointerId);
    if (instance.#pointers.size < 2) instance.#pinchStart = null;
    if (instance.#pointers.size === 0) {
      instance.#dragStart = null;
      instance.#setPanning(false);
    }
  }

  static resumeSingle(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Panner) || instance.#pointers.size !== 1) return;
    const [pointerId, point] = [...instance.#pointers.entries()][0] ?? [];
    if (pointerId === undefined || point === undefined) return;
    instance.#dragStart = { pointerId, point, pan: { ...instance.pan } };
    instance.#setPanning(true);
  }

  static applyZoom(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const record = recordOf(event.data);
    const scale = typeof record?.["scale"] === "number"
      ? record["scale"]
      : typeof record?.["deltaY"] === "number" ? instance.scale * Math.exp(-record["deltaY"] * ZOOM_STEP) : null;
    if (scale === null) return;
    const point = pointOf(record?.["point"]);
    if (point === null) {
      instance.#setTransform(scale, instance.pan);
      return;
    }
    instance.#setZoom(scale, point);
  }

  static applyFit(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const record = recordOf(event.data);
    const metrics = metricsOf(record?.["metrics"]);
    const bounds = boundsOf(record?.["bounds"]) ?? metrics?.bounds ?? null;
    if (metrics === null || bounds === null) return;
    instance.#fitBounds(bounds, metrics);
  }

  static applyViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#setTransform(viewport.zoom, { x: viewport.x, y: viewport.y });
    if (instance.#dragStart !== null) {
      instance.#dragStart = { ...instance.#dragStart, pan: { ...instance.pan } };
    }
  }

  #fitBounds(bounds: ViewportBounds, metrics: ViewportMetrics): void {
    const width = Math.max(1, bounds.right - bounds.left + FIT_PADDING * 2);
    const height = Math.max(1, bounds.bottom - bounds.top + FIT_PADDING * 2);
    const scale = Math.min(
      MAX_FIT_ZOOM,
      Math.max(MIN_ZOOM, Math.min(metrics.width / width, metrics.height / height)),
    );
    const center = {
      x: (bounds.left + bounds.right) / 2 + metrics.origin.x,
      y: (bounds.top + bounds.bottom) / 2 + metrics.origin.y,
    };
    this.#setTransform(scale, {
      x: metrics.width / 2 - center.x * scale,
      y: metrics.height / 2 - center.y * scale,
    });
  }

  #setZoom(scale: number, point: ViewportPoint): void {
    const worldPoint = {
      x: (point.x - this.pan.x) / this.scale,
      y: (point.y - this.pan.y) / this.scale,
    };
    this.#setTransform(scale, {
      x: point.x - worldPoint.x * scale,
      y: point.y - worldPoint.y * scale,
    });
  }

  #setTransform(scale: number, pan: ViewportPoint): void {
    this.scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
    this.pan = { ...pan };
    void hsm.notifyOwner({
      instance: this,
      event: hsm.typedEvent({ event: Panner.transformEvent, data: this.viewport }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(this)));
  }

  #setPanning(panning: boolean): void {
    void hsm.notifyOwner({
      instance: this,
      event: hsm.typedEvent({ event: Panner.panningEvent, data: { panning } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(this)));
  }
}

export function startPanner(args: {
  ctx: hsm.Context;
}): Panner {
  return hsm.start(args.ctx, new Panner(), Panner.model);
}

function recordOf(value: unknown): Record<string, unknown> | null {
  return hsm.isRecord(value) ? value : null;
}

function pointOf(value: unknown): ViewportPoint | null {
  const record = recordOf(value);
  const x = record?.["x"];
  const y = record?.["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
    ? { x, y }
    : null;
}

function pointerIdOf(value: unknown): number | null {
  const pointerId = recordOf(value)?.["pointerId"];
  return typeof pointerId === "number" && Number.isInteger(pointerId) ? pointerId : null;
}

function panPointerOf(value: unknown): PanPointerData | null {
  const pointerId = pointerIdOf(value);
  const point = pointOf(recordOf(value)?.["point"]);
  if (pointerId === null || point === null) return null;
  return { pointerId, point };
}

function viewportOf(value: unknown): Viewport | null {
  const record = recordOf(value);
  const x = record?.["x"];
  const y = record?.["y"];
  const zoom = record?.["zoom"];
  return typeof x === "number" && Number.isFinite(x)
    && typeof y === "number" && Number.isFinite(y)
    && typeof zoom === "number" && Number.isFinite(zoom)
    ? { x, y, zoom }
    : null;
}

function boundsOf(value: unknown): ViewportBounds | null {
  const record = recordOf(value);
  const left = record?.["left"];
  const right = record?.["right"];
  const top = record?.["top"];
  const bottom = record?.["bottom"];
  return typeof left === "number" && Number.isFinite(left)
    && typeof right === "number" && Number.isFinite(right)
    && typeof top === "number" && Number.isFinite(top)
    && typeof bottom === "number" && Number.isFinite(bottom)
    ? { left, right, top, bottom }
    : null;
}

function metricsOf(value: unknown): ViewportMetrics | null {
  const record = recordOf(value);
  const width = record?.["width"];
  const height = record?.["height"];
  const bounds = boundsOf(record?.["bounds"]);
  const origin = pointOf(record?.["origin"]);
  return typeof width === "number" && Number.isFinite(width)
    && typeof height === "number" && Number.isFinite(height)
    && bounds !== null
    && origin !== null
    ? { width, height, bounds, origin }
    : null;
}
