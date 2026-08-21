import * as hsm from "../hsm.ts";

export class Renderer extends hsm.Instance {
  static readonly markDirtyEvent = { name: "mark_dirty", kind: hsm.Kinds.Event } as const;
  static readonly renderCompleteEvent = { name: "render_complete", kind: hsm.Kinds.CompletionEvent } as const;

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
      hsm.transition(hsm.on(hsm.ErrorEvent.name), hsm.target("../failed")),
    ),
    hsm.state(
      "failed",
      hsm.transition(hsm.on(Renderer.markDirtyEvent.name), hsm.target("../dirty")),
    ),
  );

  readonly host: HTMLElement | null;
  readonly onRender: () => void | Promise<void>;

  constructor(onRender: () => void | Promise<void>, host: HTMLElement | null = null) {
    super();
    this.onRender = onRender;
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
    if (!(instance instanceof Renderer) || ctx.done) return;
    try {
      await instance.onRender();
      if (ctx.done) return;
      instance.dispatch(hsm.typedEvent(Renderer.renderCompleteEvent));
    } catch (error) {
      instance.dispatch({ ...hsm.ErrorEvent, data: error });
    }
  }
}

export function startRenderer(args: {
  ctx: hsm.Context;
  onRender: () => void | Promise<void>;
  host?: HTMLElement | null;
}): Renderer {
  return hsm.start(args.ctx, new Renderer(args.onRender, args.host ?? null), Renderer.model);
}
