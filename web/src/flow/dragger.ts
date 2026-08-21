import * as hsm from "../hsm.ts";

export type DragPosition = { readonly x: number; readonly y: number };

export class Dragger extends hsm.Instance {
  static readonly dragStartEvent = { name: "drag_start", kind: hsm.Kinds.Event } as const;
  static readonly dragEndEvent = { name: "drag_end", kind: hsm.Kinds.Event } as const;

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
      hsm.transition(hsm.every(Dragger.frameInterval), hsm.effect(Dragger.samplePosition)),
      hsm.transition(
        hsm.on(Dragger.dragEndEvent.name),
        hsm.target("../idle"),
        hsm.effect(Dragger.endDrag),
      ),
    ),
  );

  readonly getCurrentPosition: () => DragPosition;
  readonly onPosition: (position: DragPosition) => void;
  nodeId: string | null = null;
  #offset: DragPosition = { x: 0, y: 0 };
  #last: DragPosition | null = null;

  constructor(getCurrentPosition: () => DragPosition, onPosition: (position: DragPosition) => void) {
    super();
    this.getCurrentPosition = getCurrentPosition;
    this.onPosition = onPosition;
  }

  get dragging(): boolean {
    return this.nodeId !== null;
  }

  dragStart(data: { nodeId: string; offset: DragPosition }): void {
    this.dispatch(hsm.namedEvent(Dragger.dragStartEvent.name, data));
  }

  dragEnd(): void {
    this.dispatch(hsm.namedEvent(Dragger.dragEndEvent.name));
  }

  static frameInterval(): number {
    return 16;
  }

  static startDrag(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Dragger) || !hsm.isRecord(event.data)) return;
    const nodeId = event.data["nodeId"];
    const offset = event.data["offset"];
    if (typeof nodeId !== "string" || !hsm.isRecord(offset)) return;
    const x = offset["x"];
    const y = offset["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    instance.nodeId = nodeId;
    instance.#offset = { x, y };
    instance.#last = instance.getCurrentPosition();
  }

  static samplePosition(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Dragger) || ctx.done || instance.nodeId === null) return;
    const mouse = instance.getCurrentPosition();
    const previous = instance.#last;
    if (previous !== null && Math.abs(mouse.x - previous.x) + Math.abs(mouse.y - previous.y) < 0.01) return;
    instance.#last = mouse;
    instance.onPosition({
      x: mouse.x - instance.#offset.x,
      y: mouse.y - instance.#offset.y,
    });
  }

  static endDrag(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Dragger)) return;
    instance.nodeId = null;
    instance.#last = null;
  }
}

export function startDragger(
  ctx: hsm.Context,
  getCurrentPosition: () => DragPosition,
  onPosition: (position: DragPosition) => void,
): Dragger {
  return hsm.start(ctx, new Dragger(getCurrentPosition, onPosition), Dragger.model);
}
