import * as hsm from "../hsm.ts";

import { getViewportForBounds } from "./path.ts";
import { FIT_PADDING_RATIO, MAX_ZOOM, MIN_ZOOM, type Viewport, type ViewportBounds, type XYPosition } from "./types.ts";

export type ViewportPoint = XYPosition;
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

/**
 * Synchronous world-transform writer injected at `startPanner`.
 * The Panner paints the world element here, in the same turn as the cursor
 * or viewport event that changed the transform. `transform_changed` remains
 * owner bookkeeping only; it is never the path the DOM transform takes.
 * Inputs: the new `{ x, y, zoom }` viewport. Outputs: the injected world
 * element style update. Ownership: the Panner calls it; the host supplies
 * it. Lifetime: the Panner's lifetime. Concurrency: synchronous. Failure
 * modes: a writer that throws propagates from the Panner effect. Units: pan
 * in CSS pixels, zoom as a scale factor. Classification: external-system
 * (DOM paint).
 */
export type WorldTransformWriter = (viewport: Viewport) => void;

const MIN_PINCH_DISTANCE = 1;
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
  readonly paintWorld: WorldTransformWriter | null;
  #pointers = new Map<number, ViewportPoint>();
  #dragStart: { pointerId: number; point: ViewportPoint; pan: ViewportPoint } | null = null;
  #pinchStart: { distance: number; scale: number } | null = null;

  constructor(paintWorld?: WorldTransformWriter) {
    super();
    this.paintWorld = paintWorld ?? null;
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
      instance.#setPanning({ panning: true });
      return;
    }
    instance.#dragStart = null;
    const points = [...instance.#pointers.values()];
    const first = points[0];
    const second = points[1];
    if (first !== undefined && second !== undefined) {
      instance.#pinchStart = {
        distance: Math.max(MIN_PINCH_DISTANCE, Math.hypot(second.x - first.x, second.y - first.y)),
        scale: instance.scale,
      };
    }
    instance.#setPanning({ panning: false });
  }

  static dragPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointer = panPointerOf(event.data);
    if (pointer === null || !instance.#pointers.has(pointer.pointerId)) return;
    instance.#pointers.set(pointer.pointerId, pointer.point);
    if (instance.#dragStart?.pointerId !== pointer.pointerId) return;
    instance.#setTransform({
      scale: instance.scale,
      pan: {
        x: instance.#dragStart.pan.x + pointer.point.x - instance.#dragStart.point.x,
        y: instance.#dragStart.pan.y + pointer.point.y - instance.#dragStart.point.y,
      },
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
    instance.#setZoom({ scale, point: midpoint });
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
      instance.#setPanning({ panning: false });
    }
  }

  static resumeSingle(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Panner) || instance.#pointers.size !== 1) return;
    const [pointerId, point] = [...instance.#pointers.entries()][0] ?? [];
    if (pointerId === undefined || point === undefined) return;
    instance.#dragStart = { pointerId, point, pan: { ...instance.pan } };
    instance.#setPanning({ panning: true });
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
      instance.#setTransform({ scale, pan: instance.pan });
    } else {
      instance.#setZoom({ scale, point });
    }
    instance.#rebaseGestureOrigin();
  }

  static applyFit(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const record = recordOf(event.data);
    const metrics = metricsOf(record?.["metrics"]);
    const bounds = boundsOf(record?.["bounds"]) ?? metrics?.bounds ?? null;
    if (metrics === null || bounds === null) return;
    instance.#fitBounds(bounds, metrics);
    instance.#rebaseGestureOrigin();
  }

  static applyViewport(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const viewport = viewportOf(event.data);
    if (viewport === null) return;
    instance.#setTransform({ scale: viewport.zoom, pan: { x: viewport.x, y: viewport.y } });
    instance.#rebaseGestureOrigin();
  }

  #rebaseGestureOrigin(): void {
    if (this.#dragStart !== null) {
      const live = this.#pointers.get(this.#dragStart.pointerId);
      this.#dragStart = {
        pointerId: this.#dragStart.pointerId,
        point: live ?? this.#dragStart.point,
        pan: { ...this.pan },
      };
    }
    if (this.#pinchStart === null) return;
    const points = [...this.#pointers.values()];
    const first = points[0];
    const second = points[1];
    if (first === undefined || second === undefined) return;
    this.#pinchStart = {
      distance: Math.max(MIN_PINCH_DISTANCE, Math.hypot(second.x - first.x, second.y - first.y)),
      scale: this.scale,
    };
  }

  #fitBounds(bounds: ViewportBounds, metrics: ViewportMetrics): void {
    const viewport = getViewportForBounds({
      bounds,
      origin: metrics.origin,
      width: metrics.width,
      height: metrics.height,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      padding: FIT_PADDING_RATIO,
    });
    this.#setTransform({ scale: viewport.zoom, pan: { x: viewport.x, y: viewport.y } });
  }

  #setZoom(args: { scale: number; point: ViewportPoint }): void {
    const worldPoint = {
      x: (args.point.x - this.pan.x) / this.scale,
      y: (args.point.y - this.pan.y) / this.scale,
    };
    this.#setTransform({
      scale: args.scale,
      pan: {
        x: args.point.x - worldPoint.x * args.scale,
        y: args.point.y - worldPoint.y * args.scale,
      },
    });
  }

  #setTransform(args: { scale: number; pan: ViewportPoint }): void {
    this.scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, args.scale));
    this.pan = { ...args.pan };
    this.paintWorld?.(this.viewport);
    void hsm.notifyOwner({
      instance: this,
      event: hsm.typedEvent({ event: Panner.transformEvent, data: this.viewport }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(this)));
  }

  #setPanning(args: { panning: boolean }): void {
    void hsm.notifyOwner({
      instance: this,
      event: hsm.typedEvent({ event: Panner.panningEvent, data: { panning: args.panning } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(this)));
  }
}

/**
 * Start a Panner under `ctx`.
 *
 * Inputs: `ctx` — owner context used as the HSM parent environment;
 * `paintWorld` — the synchronous world-transform writer the Panner calls in
 * every transform-changing effect (cursor move, zoom, fit, viewport set)
 * before `transform_changed` notifies the owner for bookkeeping.
 * Outputs: a started Panner in `/Panner/ready/fixed` with identity transform.
 * Ownership: caller owns the returned actor and must `hsm.stop` it.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one viewport per instance; pointer ids are the pan keys.
 * Failure modes: malformed pan/zoom payloads are ignored.
 * Units: pan in CSS pixels, scale unitless.
 * Classification: runtime-safe.
 */
export function startPanner(args: {
  ctx: hsm.Context;
  paintWorld: WorldTransformWriter;
}): Panner {
  return hsm.start({ ctx: args.ctx, instance: new Panner(args.paintWorld), model: Panner.model });
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
