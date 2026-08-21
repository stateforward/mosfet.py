import * as hsm from "../hsm.ts";

export type DragPosition = { readonly x: number; readonly y: number };
export type DragStartData = { readonly nodeId: string; readonly offset: DragPosition };
export type DragMoveData = { readonly position: DragPosition };
export type DragMovedData = { readonly nodeId: string; readonly position: DragPosition };

export class Dragger extends hsm.Instance {
  static readonly dragStartEvent = { name: "drag_start", kind: hsm.Kinds.Event } as const;
  static readonly dragMoveEvent = { name: "drag_move", kind: hsm.Kinds.Event } as const;
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
      hsm.transition(hsm.on(Dragger.dragMoveEvent.name), hsm.effect(Dragger.applyMove)),
      hsm.transition(
        hsm.on(Dragger.dragEndEvent.name),
        hsm.target("../idle"),
        hsm.effect(Dragger.endDrag),
      ),
    ),
  );

  #nodeId: string | null = null;
  #offset: DragPosition = { x: 0, y: 0 };

  static startDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    const data = dragStartOf(event.data);
    if (!(instance instanceof Dragger) || data === null) return;
    instance.#nodeId = data.nodeId;
    instance.#offset = data.offset;
  }

  static applyMove(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Dragger) || instance.#nodeId === null) return;
    const position = positionOf(hsm.isRecord(event.data) ? event.data["position"] : null);
    if (position === null) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Dragger.movedEvent, data: {
        nodeId: instance.#nodeId,
        position: {
          x: position.x - instance.#offset.x,
          y: position.y - instance.#offset.y,
        },
      } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Dragger)) return;
    instance.#nodeId = null;
  }
}

export function startDragger(args: { ctx: hsm.Context }): Dragger {
  return hsm.start(args.ctx, new Dragger(), Dragger.model);
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
