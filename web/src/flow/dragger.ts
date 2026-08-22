import * as hsm from "../hsm.ts";

export type DragPosition = { readonly x: number; readonly y: number };
export type DragStartData = { readonly nodeId: string; readonly offset: DragPosition };
export type DragMovedData = { readonly nodeId: string; readonly position: DragPosition };
export type PositionSource = () => DragPosition;

const DRAG_FRAME_MS = 16;
const DRAG_MOVE_EPSILON = 0.01;

export class Dragger extends hsm.Instance {
  static readonly dragStartEvent = { name: "drag_start", kind: hsm.Kinds.Event } as const;
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
      hsm.transition(hsm.every(Dragger.dragFrameInterval), hsm.effect(Dragger.pollPosition)),
      hsm.transition(
        hsm.on(Dragger.dragEndEvent.name),
        hsm.target("../idle"),
        hsm.effect(Dragger.endDrag),
      ),
    ),
  );

  /**
   * ~60fps poll cadence while dragging, per the gogo dragger frame interval.
   *
   * Inputs: none; HSM calls it as a time expression on each `dragging` entry.
   * Outputs: the frame delay in milliseconds (`DRAG_FRAME_MS` = 16).
   * Ownership: Dragger. Lifetime: one `dragging` episode. Concurrency:
   * synchronous. Failure modes: none. Classification: runtime-safe.
   */
  static dragFrameInterval(): number {
    return DRAG_FRAME_MS;
  }

  readonly getPosition: PositionSource;
  #nodeId: string | null = null;
  #offset: DragPosition = { x: 0, y: 0 };
  #last: DragPosition | null = null;

  constructor(getPosition: PositionSource) {
    super();
    this.getPosition = getPosition;
  }

  static startDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    const data = dragStartOf(event.data);
    if (!(instance instanceof Dragger) || data === null) return;
    instance.#nodeId = data.nodeId;
    instance.#offset = data.offset;
    // Seed the last-polled position so a held pointer does not jump on the
    // first frame tick; motion arrives only on real pointer movement.
    instance.#last = instance.getPosition();
  }

  static pollPosition(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Dragger) || instance.#nodeId === null) return;
    const position = instance.getPosition();
    const validated = positionOf(position);
    if (validated === null) return;
    const last = instance.#last;
    if (last === null) {
      instance.#last = validated;
      return;
    }
    if (Math.abs(validated.x - last.x) + Math.abs(validated.y - last.y) <= DRAG_MOVE_EPSILON) return;
    instance.#last = validated;
    const nodeId = instance.#nodeId;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Dragger.movedEvent, data: {
        nodeId,
        position: {
          x: validated.x - instance.#offset.x,
          y: validated.y - instance.#offset.y,
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
 * Inputs: `ctx` — owner context used as the HSM parent environment;
 * `getPosition` — injected live pointer-position source (world coordinates),
 * polled on every frame tick while dragging. The Dragger never reaches into
 * the host for it; it is supplied here as a dependency.
 * Outputs: a started Dragger in `/Dragger/idle`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one drag at a time; frame polls are serialized by the
 * machine. Failure modes: malformed drag-start payloads are ignored; a
 * position source returning non-finite values skips that frame.
 * Units: positions in world coordinates. Classification: runtime-safe.
 */
export function startDragger(args: { ctx: hsm.Context; getPosition: PositionSource }): Dragger {
  return hsm.start({ ctx: args.ctx, instance: new Dragger(args.getPosition), model: Dragger.model });
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
