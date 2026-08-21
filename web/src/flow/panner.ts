import * as hsm from "../hsm.ts";

export type ViewportPoint = { readonly x: number; readonly y: number };
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
export type Viewport = { readonly x: number; readonly y: number; readonly zoom: number };

const FIT_PADDING = 28;
const MIN_ZOOM = 0.12;
const MAX_ZOOM = 2.4;
const MAX_FIT_ZOOM = 1.2;
const ZOOM_STEP = 0.0015;

export class Panner extends hsm.Instance {
  static readonly panStartEvent = { name: "pan_start", kind: hsm.Kinds.Event } as const;
  static readonly cursorMoveEvent = { name: "cursor_move", kind: hsm.Kinds.Event } as const;
  static readonly panEndEvent = { name: "pan_end", kind: hsm.Kinds.Event } as const;
  static readonly zoomEvent = { name: "zoom", kind: hsm.Kinds.Event } as const;
  static readonly fitEvent = { name: "fit", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Panner",
    hsm.initial(hsm.target("fixed")),
    hsm.state(
      "fixed",
      hsm.transition(
        hsm.on(Panner.panStartEvent.name),
        hsm.target("../panning"),
        hsm.effect(Panner.startPan),
      ),
      hsm.transition(hsm.on(Panner.zoomEvent.name), hsm.effect(Panner.applyZoom)),
      hsm.transition(hsm.on(Panner.fitEvent.name), hsm.effect(Panner.applyFit)),
    ),
    hsm.state(
      "panning",
      hsm.transition(hsm.on(Panner.panStartEvent.name), hsm.effect(Panner.startPan)),
      hsm.transition(hsm.on(Panner.cursorMoveEvent.name), hsm.effect(Panner.updatePan)),
      hsm.transition(
        hsm.on(Panner.panEndEvent.name),
        hsm.guard(Panner.canEndPan),
        hsm.target("../fixed"),
        hsm.effect(Panner.endPan),
      ),
      hsm.transition(hsm.on(Panner.panEndEvent.name), hsm.effect(Panner.endPan)),
      hsm.transition(hsm.on(Panner.zoomEvent.name), hsm.effect(Panner.applyZoom)),
      hsm.transition(hsm.on(Panner.fitEvent.name), hsm.effect(Panner.applyFit)),
    ),
  );

  readonly world: HTMLElement;
  readonly frame: HTMLElement | null;
  readonly onTransform: ((transform: ViewportTransform) => void) | null;
  scale = 1;
  pan: ViewportPoint = { x: 0, y: 0 };
  #pointers = new Map<number, ViewportPoint>();
  #dragStart: { pointerId: number; point: ViewportPoint; pan: ViewportPoint } | null = null;
  #pinchStart: { distance: number; scale: number } | null = null;

  constructor(
    world: HTMLElement,
    options: { frame?: HTMLElement; onTransform?: (transform: ViewportTransform) => void } = {},
  ) {
    super();
    this.world = world;
    this.frame = options.frame ?? null;
    this.onTransform = options.onTransform ?? null;
  }

  get transform(): ViewportTransform {
    return { scale: this.scale, pan: this.pan };
  }

  get viewport(): Viewport {
    return { x: this.pan.x, y: this.pan.y, zoom: this.scale };
  }

  panStart(data: unknown): void {
    this.dispatch(hsm.namedEvent(Panner.panStartEvent.name, data));
  }

  cursorMove(data: unknown): void {
    this.dispatch(hsm.namedEvent(Panner.cursorMoveEvent.name, data));
  }

  panEnd(data: unknown): void {
    this.dispatch(hsm.namedEvent(Panner.panEndEvent.name, data));
  }

  zoom(data: unknown): void {
    this.dispatch(hsm.namedEvent(Panner.zoomEvent.name, data));
  }

  fit(data: unknown): void {
    this.dispatch(hsm.namedEvent(Panner.fitEvent.name, data));
  }

  setViewport(viewport: Viewport): void {
    this.#setTransform(viewport.zoom, { x: viewport.x, y: viewport.y });
  }

  static startPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointerId = pointerIdOf(event.data);
    const point = pointOf(recordOf(event.data)?.["point"]);
    if (pointerId === null || point === null) return;
    instance.#pointers.set(pointerId, point);
    if (instance.#pointers.size === 1) {
      instance.#dragStart = { pointerId, point, pan: { ...instance.pan } };
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

  static updatePan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const record = recordOf(event.data);
    const directPan = pointOf(record?.["pan"]);
    if (directPan !== null) {
      const scale = typeof record?.["scale"] === "number" ? record["scale"] : instance.scale;
      instance.#setTransform(scale, directPan);
      return;
    }
    const pointerId = pointerIdOf(event.data);
    const point = pointOf(record?.["point"]);
    if (pointerId === null || point === null) return;
    if (!instance.#pointers.has(pointerId)) return;
    instance.#pointers.set(pointerId, point);
    if (instance.#pointers.size >= 2 && instance.#pinchStart !== null) {
      const points = [...instance.#pointers.values()];
      const first = points[0];
      const second = points[1];
      if (first !== undefined && second !== undefined) {
        const midpoint = { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
        const scale = instance.#pinchStart.scale
          * Math.hypot(second.x - first.x, second.y - first.y)
          / instance.#pinchStart.distance;
        instance.#setZoom(scale, midpoint);
      }
      return;
    }
    if (instance.#dragStart?.pointerId === pointerId) {
      instance.#setTransform(instance.scale, {
        x: instance.#dragStart.pan.x + point.x - instance.#dragStart.point.x,
        y: instance.#dragStart.pan.y + point.y - instance.#dragStart.point.y,
      });
    }
  }

  static canEndPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof Panner)) return true;
    const pointerId = pointerIdOf(event.data);
    return pointerId === null || (instance.#pointers.has(pointerId) && instance.#pointers.size <= 1);
  }

  static endPan(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Panner)) return;
    const pointerId = pointerIdOf(event.data);
    if (pointerId !== null) instance.#pointers.delete(pointerId);
    if (instance.#pointers.size < 2) instance.#pinchStart = null;
    if (instance.#pointers.size === 0) instance.#dragStart = null;
    instance.#setPanning(instance.#dragStart !== null);
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
    this.world.style.transformOrigin = "0 0";
    this.world.style.transform = `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.scale})`;
    this.onTransform?.(this.transform);
  }

  #setPanning(panning: boolean): void {
    this.frame?.classList.toggle("is-dragging", panning);
  }
}

export function startPanner(
  ctx: hsm.Context,
  world: HTMLElement,
  options: { frame?: HTMLElement; onTransform?: (transform: ViewportTransform) => void } = {},
): Panner {
  return hsm.start(ctx, new Panner(world, options), Panner.model);
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
