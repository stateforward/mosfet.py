import * as hsm from "../hsm.ts";

export type DragPosition = { readonly x: number; readonly y: number };
export type DragStartData = { readonly nodeId: string; readonly offset: DragPosition };
export type DragSampleData = { readonly world: DragPosition };
export type DragMovedData = { readonly nodeId: string; readonly position: DragPosition };

const DRAG_MOVE_EPSILON = 0.01;

export class Dragger extends hsm.Instance {
  static readonly dragStartEvent = { name: "drag_start", kind: hsm.Kinds.Event } as const;
  static readonly dragSampleEvent = { name: "drag_sample", kind: hsm.Kinds.Event } as const;
  static readonly dragEndEvent = { name: "drag_end", kind: hsm.Kinds.Event } as const;
  static readonly movedEvent = { name: "drag_moved", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Dragger",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.transition(
        hsm.on(Dragger.dragStartEvent.name),
        hsm.target("../dragging"),
        hsm.effect(Dragger.startDrag),
      ),
    ),
    hsm.state(
      "dragging",
      hsm.transition(hsm.on(Dragger.dragSampleEvent.name), hsm.effect(Dragger.applySample)),
      hsm.transition(
        hsm.on(Dragger.dragEndEvent.name),
        hsm.target("../idle"),
        hsm.effect(Dragger.endDrag),
      ),
    ),
  );

  #nodeId: string | null = null;
  #offset: DragPosition = { x: 0, y: 0 };
  #last: DragPosition | null = null;

  static startDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    const data = dragStartOf(event.data);
    if (!(instance instanceof Dragger) || data === null) return;
    instance.#nodeId = data.nodeId;
    instance.#offset = data.offset;
    // The first sample after drag start establishes the dedupe baseline, so a
    // held pointer does not jump; motion arrives only on real pointer movement.
    instance.#last = null;
  }

  static applySample(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Dragger) || instance.#nodeId === null) return;
    const world = positionOf((event.data as { world?: unknown } | null)?.world);
    if (world === null) return;
    const last = instance.#last;
    if (last === null) {
      instance.#last = world;
      return;
    }
    if (Math.abs(world.x - last.x) + Math.abs(world.y - last.y) <= DRAG_MOVE_EPSILON) return;
    instance.#last = world;
    const nodeId = instance.#nodeId;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Dragger.movedEvent, data: {
        nodeId,
        position: {
          x: world.x - instance.#offset.x,
          y: world.y - instance.#offset.y,
        },
      } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Dragger)) return;
    instance.#nodeId = null;
    instance.#offset = { x: 0, y: 0 };
    instance.#last = null;
  }
}

/**
 * Start a Dragger under `ctx`.
 *
 * Inputs: `ctx` — owner context used as the HSM parent environment. Pointer
 * samples arrive as `drag_sample` events carrying the world point, so no
 * position source is injected or polled.
 * Outputs: a started Dragger in `/Dragger/idle`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one drag at a time; samples are serialized by the machine.
 * Failure modes: malformed drag-start or sample payloads are ignored; samples
 * with non-finite world coordinates are skipped.
 * Units: positions in world coordinates. Classification: runtime-safe.
 */
export function startDragger(args: { ctx: hsm.Context }): Dragger {
  return hsm.start({ ctx: args.ctx, instance: new Dragger(), model: Dragger.model });
}

function positionOf(value: unknown): DragPosition | null {
  if (!hsm.isRecord(value)) return null;
  const x = value["x"];
  const y = value["y"];
  return typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)
    ? { x, y }
    : null;
}

function dragStartOf(value: unknown): DragStartData | null {
  if (!hsm.isRecord(value)) return null;
  const nodeId = value["nodeId"];
  const offset = positionOf(value["offset"]);
  if (typeof nodeId !== "string" || offset === null) return null;
  return { nodeId, offset };
}
