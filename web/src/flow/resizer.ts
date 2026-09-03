import * as hsm from "../hsm.ts";

import { isResizeDirection } from "./resize-control.ts";
import {
  MIN_RESIZE_HEIGHT,
  MIN_RESIZE_WIDTH,
  type ResizeBounds,
  type ResizeChannel,
  type ResizeConstraints,
  type ResizeDirection,
} from "./types.ts";

export type ResizeStartData = {
  readonly nodeId: string;
  readonly direction: ResizeDirection;
  readonly channel: ResizeChannel;
  readonly origin: ResizeBounds;
  readonly pointer: { readonly x: number; readonly y: number };
  readonly minWidth: number;
  readonly minHeight: number;
  readonly maxWidth?: number;
  readonly maxHeight?: number;
  readonly keepAspectRatio: boolean;
};

export type ResizeSampleData = { readonly world: { readonly x: number; readonly y: number } };
export type ResizeKeyStepData = { readonly key: string };
export type ResizeMovedData = { readonly nodeId: string } & ResizeBounds;

const RESIZE_MOVE_EPSILON = 0.01;
/** Documented keyboard step: 1 world px of edge travel per keypress. */
const KEYBOARD_RESIZE_STEP = 1;

export class Resizer extends hsm.Instance {
  static readonly resizeStartEvent = { name: "resize_start", kind: hsm.Kinds.Event } as const;
  static readonly resizeSampleEvent = { name: "resize_sample", kind: hsm.Kinds.Event } as const;
  static readonly resizeKeyStepEvent = { name: "resize_key_step", kind: hsm.Kinds.Event } as const;
  static readonly resizeEndEvent = { name: "resize_end", kind: hsm.Kinds.Event } as const;
  static readonly movedEvent = { name: "resize_moved", kind: hsm.Kinds.Event } as const;
  static readonly finishedEvent = { name: "resize_finished", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Resizer",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.transition(
        hsm.on(Resizer.resizeStartEvent.name),
        hsm.guard(Resizer.isKeyboardChannel),
        hsm.target("../resizing/keyboard"),
        hsm.effect(Resizer.startResize),
      ),
      hsm.transition(
        hsm.on(Resizer.resizeStartEvent.name),
        hsm.target("../resizing/pointer"),
        hsm.effect(Resizer.startResize),
      ),
    ),
    hsm.state(
      "resizing",
      hsm.initial(hsm.target("pointer")),
      hsm.transition(
        hsm.on(Resizer.resizeEndEvent.name),
        hsm.target("../idle"),
        hsm.effect(Resizer.endResize),
      ),
      hsm.state(
        "pointer",
        hsm.transition(hsm.on(Resizer.resizeSampleEvent.name), hsm.effect(Resizer.applySample)),
      ),
      hsm.state(
        "keyboard",
        hsm.transition(hsm.on(Resizer.resizeKeyStepEvent.name), hsm.effect(Resizer.applyKeyStep)),
      ),
    ),
  );

  #nodeId: string | null = null;
  #direction: ResizeDirection = "se";
  #origin: ResizeBounds = { x: 0, y: 0, width: 0, height: 0 };
  #pointer: { x: number; y: number } = { x: 0, y: 0 };
  #constraints: ResizeConstraints = {
    minWidth: MIN_RESIZE_WIDTH,
    minHeight: MIN_RESIZE_HEIGHT,
    keepAspectRatio: false,
  };
  #last: { x: number; y: number } | null = null;
  #current: ResizeBounds = { x: 0, y: 0, width: 0, height: 0 };
  #keyDx: number = 0;
  #keyDy: number = 0;

  static isKeyboardChannel(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    return resizeStartOf(event.data)?.channel === "keyboard";
  }

  static startResize(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    const data = resizeStartOf(event.data);
    if (!(instance instanceof Resizer) || data === null) return;
    instance.#nodeId = data.nodeId;
    instance.#direction = data.direction;
    instance.#origin = data.origin;
    instance.#pointer = data.pointer;
    instance.#constraints = {
      minWidth: data.minWidth,
      minHeight: data.minHeight,
      keepAspectRatio: data.keepAspectRatio,
      ...(data.maxWidth !== undefined ? { maxWidth: data.maxWidth } : {}),
      ...(data.maxHeight !== undefined ? { maxHeight: data.maxHeight } : {}),
    };
    instance.#last = null;
    instance.#current = data.origin;
    instance.#keyDx = 0;
    instance.#keyDy = 0;
  }

  static applySample(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Resizer) || instance.#nodeId === null) return;
    if (!hsm.isRecord(event.data)) return;
    const world = pointOf(event.data["world"]);
    if (world === null) return;
    const last = instance.#last;
    if (last === null) {
      instance.#last = world;
      return;
    }
    if (Math.abs(world.x - last.x) + Math.abs(world.y - last.y) <= RESIZE_MOVE_EPSILON) return;
    instance.#last = world;
    const bounds = resizedBounds({
      origin: instance.#origin,
      direction: instance.#direction,
      dx: world.x - instance.#pointer.x,
      dy: world.y - instance.#pointer.y,
      constraints: instance.#constraints,
    });
    instance.#current = bounds;
    const nodeId = instance.#nodeId;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Resizer.movedEvent, data: { nodeId, ...bounds } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  /**
   * One typed keyboard step while `resizing`.
   *
   * Contract: each keypress moves the resized edge(s) by 1 world px in the
   * arrow's direction (ArrowRight grows an east edge or shrinks a west edge,
   * ArrowDown grows a south edge or shrinks a north edge, and symmetrically
   * for the other two). The cumulative delta is re-applied through the shared
   * `resizedBounds` clamp logic, so min/max and aspect constraints apply
   * exactly as they do for pointer samples. The owner is notified with
   * `resize_moved` (bounds already clamped) and nothing else; no timers and
   * no element reads live on this path.
   *
   * Inputs: `resize_key_step` with `data.key` one of the four arrow keys.
   * Outputs: `resize_moved` to the owner when the step changed the box.
   * Ownership: the instance owns the accumulated key deltas. Lifetime: one
   * step. Concurrency: steps are serialized by the machine. Failure modes:
   * unknown keys or keys orthogonal to the direction are no-ops.
   * Classification: runtime-safe.
   */
  static applyKeyStep(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Resizer) || instance.#nodeId === null) return;
    if (!hsm.isRecord(event.data)) return;
    const key = event.data["key"];
    if (typeof key !== "string") return;
    const direction = instance.#direction;
    // The arrow key moves the resized edge in the arrow's direction:
    // `dx`/`dy` are the edge's right/down travel, so the key-to-delta mapping
    // is the same on every axis; `resizedBounds` flips the delta itself for
    // west/north edges (fixed east/south), exactly as it does for pointers.
    const movesHorizontally = direction === "e" || direction === "w"
      || direction === "ne" || direction === "nw" || direction === "se" || direction === "sw";
    const movesVertically = direction === "n" || direction === "s"
      || direction === "ne" || direction === "nw" || direction === "se" || direction === "sw";
    let dx = 0;
    let dy = 0;
    if (movesHorizontally) {
      if (key === "ArrowRight") dx = KEYBOARD_RESIZE_STEP;
      else if (key === "ArrowLeft") dx = -KEYBOARD_RESIZE_STEP;
    }
    if (movesVertically) {
      if (key === "ArrowDown") dy = KEYBOARD_RESIZE_STEP;
      else if (key === "ArrowUp") dy = -KEYBOARD_RESIZE_STEP;
    }
    if (dx === 0 && dy === 0) return;
    instance.#keyDx += dx;
    instance.#keyDy += dy;
    const bounds = resizedBounds({
      origin: instance.#origin,
      direction: instance.#direction,
      dx: instance.#keyDx,
      dy: instance.#keyDy,
      constraints: instance.#constraints,
    });
    instance.#current = bounds;
    const nodeId = instance.#nodeId;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Resizer.movedEvent, data: { nodeId, ...bounds } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static endResize(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Resizer)) return;
    const nodeId = instance.#nodeId;
    const bounds = instance.#current;
    instance.#nodeId = null;
    instance.#last = null;
    if (nodeId === null) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Resizer.finishedEvent, data: { nodeId, ...bounds } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }
}

/**
 * Start a Resizer under `ctx`.
 *
 * Inputs: `ctx` — owner context used as the HSM parent environment. Pointer
 * samples arrive as `resize_sample` events carrying the world point while
 * `resizing/pointer`; keyboard steps arrive as `resize_key_step` events
 * carrying one arrow key while `resizing/keyboard` (each step moves the
 * edge by 1 world px through the shared clamp logic). The start payload's
 * `channel` selects the exclusive nested state.
 * Outputs: a started Resizer in `/Resizer/idle`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one resize at a time; pointer and keyboard cannot share a
 * session. Samples and key steps are serialized by the machine.
 * Failure modes: malformed start or sample payloads are ignored; samples
 * with non-finite world coordinates are skipped; a second start while
 * resizing is ignored.
 * Units: positions and sizes in world coordinates. Classification: runtime-safe.
 */
export function startResizer(args: { ctx: hsm.Context }): Resizer {
  return hsm.start({ ctx: args.ctx, instance: new Resizer(), model: Resizer.model });
}

/**
 * Compute the next node box for one resize sample.
 *
 * Inputs: finite `origin` box, a `ResizeDirection`, pointer delta, and
 * `constraints` with `minWidth`/`minHeight` > 0 and optional max. Outputs: a
 * new box whose opposite edge stays fixed. Ownership: caller owns the result.
 * Lifetime: one call. Concurrency: synchronous. Failure modes: non-finite
 * inputs are not validated here; callers must pass parsed start data.
 * Units: world coordinates. Classification: runtime-safe.
 */
export function resizedBounds(args: {
  origin: ResizeBounds;
  direction: ResizeDirection;
  dx: number;
  dy: number;
  constraints: ResizeConstraints;
}): ResizeBounds {
  const { origin, direction, dx, dy, constraints } = args;
  const east = origin.x + origin.width;
  const south = origin.y + origin.height;
  const moveE = direction === "e" || direction === "ne" || direction === "se";
  const moveW = direction === "w" || direction === "nw" || direction === "sw";
  const moveS = direction === "s" || direction === "se" || direction === "sw";
  const moveN = direction === "n" || direction === "ne" || direction === "nw";
  let width = origin.width;
  let height = origin.height;
  if (moveE) width = origin.width + dx;
  if (moveW) width = origin.width - dx;
  if (moveS) height = origin.height + dy;
  if (moveN) height = origin.height - dy;
  if (constraints.keepAspectRatio && origin.height > 0) {
    const ratio = origin.width / origin.height;
    if ((moveE || moveW) && (moveN || moveS)) {
      if (Math.abs(dx) * origin.height >= Math.abs(dy) * origin.width) height = width / ratio;
      else width = height * ratio;
    } else if (moveE || moveW) {
      height = width / ratio;
    } else if (moveN || moveS) {
      width = height * ratio;
    }
  }
  width = clampSize({ value: width, min: constraints.minWidth, max: constraints.maxWidth });
  height = clampSize({ value: height, min: constraints.minHeight, max: constraints.maxHeight });
  if (constraints.keepAspectRatio && origin.height > 0) {
    const ratio = origin.width / origin.height;
    if ((moveE || moveW) && !(moveN || moveS)) {
      height = clampSize({
        value: width / ratio,
        min: constraints.minHeight,
        max: constraints.maxHeight,
      });
    } else if ((moveN || moveS) && !(moveE || moveW)) {
      width = clampSize({
        value: height * ratio,
        min: constraints.minWidth,
        max: constraints.maxWidth,
      });
    }
  }
  return {
    x: moveW ? east - width : origin.x,
    y: moveN ? south - height : origin.y,
    width,
    height,
  };
}

function clampSize(args: { value: number; min: number; max: number | undefined }): number {
  const next = Math.max(args.min, args.value);
  return args.max === undefined ? next : Math.min(args.max, next);
}

function pointOf(value: unknown): { x: number; y: number } | null {
  if (!hsm.isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
    ? { x, y }
    : null;
}

function resizeStartOf(value: unknown): ResizeStartData | null {
  if (!hsm.isRecord(value)) return null;
  const nodeId = value["nodeId"];
  const direction = value["direction"];
  const origin = boundsOf(value["origin"]);
  const pointer = pointOf(value["pointer"]);
  const minWidth = value["minWidth"];
  const minHeight = value["minHeight"];
  if (typeof nodeId !== "string" || origin === null || pointer === null) return null;
  if (!isResizeDirection(direction)) return null;
  if (typeof minWidth !== "number" || !Number.isFinite(minWidth)) return null;
  if (typeof minHeight !== "number" || !Number.isFinite(minHeight)) return null;
  const maxWidth = value["maxWidth"];
  const maxHeight = value["maxHeight"];
  const channel = value["channel"] === "keyboard" ? "keyboard" : "pointer";
  return {
    nodeId,
    direction,
    channel,
    origin,
    pointer,
    minWidth,
    minHeight,
    keepAspectRatio: value["keepAspectRatio"] === true,
    ...(typeof maxWidth === "number" && Number.isFinite(maxWidth) ? { maxWidth } : {}),
    ...(typeof maxHeight === "number" && Number.isFinite(maxHeight) ? { maxHeight } : {}),
  };
}

function boundsOf(value: unknown): ResizeBounds | null {
  if (!hsm.isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  const width = value["width"];
  const height = value["height"];
  if (
    typeof x !== "number" || !Number.isFinite(x)
    || typeof y !== "number" || !Number.isFinite(y)
    || typeof width !== "number" || !Number.isFinite(width)
    || typeof height !== "number" || !Number.isFinite(height)
  ) {
    return null;
  }
  return { x, y, width, height };
}

