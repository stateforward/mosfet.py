import * as hsm from "../hsm.ts";

export class Renderer extends hsm.Instance {
  static readonly markDirtyEvent = { name: "mark_dirty", kind: hsm.Kinds.Event } as const;
  static readonly paintEvent = { name: "paint", kind: hsm.Kinds.Event } as const;
  static readonly renderCompleteEvent = { name: "render_complete", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly renderCanceledEvent = { name: "render_canceled", kind: hsm.Kinds.CompletionEvent } as const;

  static readonly model = hsm.define(
    "Renderer",
    hsm.initial(hsm.target("clean")),
    hsm.state(
      "clean",
      hsm.transition(hsm.on(Renderer.markDirtyEvent.name), hsm.target("../dirty")),
    ),
    hsm.state(
      "dirty",
      hsm.transition(hsm.after(Renderer.frameDelay), hsm.target("../rendering")),
    ),
    hsm.state(
      "rendering",
      hsm.entry(Renderer.addRenderingClass),
      hsm.exit(Renderer.removeRenderingClass),
      hsm.defer(Renderer.markDirtyEvent.name),
      hsm.activity(Renderer.performRender),
      hsm.transition(hsm.on(Renderer.renderCompleteEvent.name), hsm.target("../clean")),
      hsm.transition(hsm.on(Renderer.renderCanceledEvent.name), hsm.target("../clean")),
      hsm.transition(hsm.on(hsm.ErrorEvent.name), hsm.target("../failed")),
    ),
    hsm.state(
      "failed",
      hsm.transition(hsm.on(Renderer.markDirtyEvent.name), hsm.target("../dirty")),
    ),
  );

  readonly host: HTMLElement | null;

  constructor(host: HTMLElement | null = null) {
    super();
    this.host = host;
  }

  markDirty(): void {
    this.dispatch(hsm.typedEvent(Renderer.markDirtyEvent));
  }

  static frameDelay(): number {
    return 0;
  }

  static addRenderingClass(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Renderer)) return;
    instance.host?.classList.add("is-rendering");
  }

  static removeRenderingClass(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Renderer)) return;
    instance.host?.classList.remove("is-rendering");
  }

  static async performRender(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
    if (!(instance instanceof Renderer)) return;
    const cancel = (): hsm.Completion => instance.dispatch(hsm.typedEvent(Renderer.renderCanceledEvent));
    if (ctx.done) {
      await cancel();
      return;
    }
    try {
      await Promise.resolve();
      await Promise.resolve();
      if (ctx.done) {
        await cancel();
        return;
      }
      await hsm.notifyOwner({ instance, event: hsm.typedEvent(Renderer.paintEvent) });
      if (ctx.done) {
        await cancel();
        return;
      }
      await instance.dispatch(hsm.typedEvent(Renderer.renderCompleteEvent));
    } catch (error) {
      await instance.dispatch({ ...hsm.ErrorEvent, data: error });
    }
  }
}

export function startRenderer(args: {
  ctx: hsm.Context;
  host?: HTMLElement | null;
}): Renderer {
  return hsm.start(args.ctx, new Renderer(args.host ?? null), Renderer.model);
}
