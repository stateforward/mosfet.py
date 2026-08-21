import * as hsm from "../hsm.ts";

export class Renderer extends hsm.Instance {
  static readonly markDirtyEvent = { name: "mark_dirty", kind: hsm.Kinds.Event } as const;
  static readonly paintEvent = { name: "paint", kind: hsm.Kinds.Event } as const;
  static readonly renderCanceledEvent = { name: "render_canceled", kind: hsm.Kinds.CompletionEvent } as const;
  static readonly renderingStartedEvent = { name: "rendering_started", kind: hsm.Kinds.Event } as const;
  static readonly renderingStoppedEvent = { name: "rendering_stopped", kind: hsm.Kinds.Event } as const;

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
      hsm.entry(Renderer.notifyStarted),
      hsm.exit(Renderer.notifyStopped),
      hsm.defer(Renderer.markDirtyEvent.name),
      hsm.initial(hsm.target("waiting")),
      hsm.transition(hsm.on(Renderer.renderCanceledEvent.name), hsm.target("../clean")),
      hsm.transition(hsm.on(hsm.ErrorEvent.name), hsm.target("../failed")),
      hsm.state(
        "waiting",
        hsm.transition(hsm.after(Renderer.frameDelay), hsm.target("../painting")),
      ),
      hsm.state(
        "painting",
        hsm.activity(Renderer.paintFrame),
        hsm.transition(hsm.on(Renderer.paintEvent.name), hsm.target("../../clean")),
      ),
    ),
    hsm.state(
      "failed",
      hsm.transition(hsm.on(Renderer.markDirtyEvent.name), hsm.target("../dirty")),
    ),
  );

  constructor() {
    super();
  }

  static frameDelay(): number {
    return 0;
  }

  static notifyStarted(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Renderer)) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Renderer.renderingStartedEvent }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static notifyStopped(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Renderer)) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Renderer.renderingStoppedEvent }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static async paintFrame(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
    if (!(instance instanceof Renderer)) return;
    if (ctx.done) {
      await instance.dispatch(hsm.typedEvent({ event: Renderer.renderCanceledEvent }));
      return;
    }
    try {
      await hsm.notifyOwner({ instance, event: hsm.typedEvent({ event: Renderer.paintEvent }) });
    } catch (error) {
      if (ctx.done) return;
      await instance.dispatch({ ...hsm.ErrorEvent, data: error });
    }
  }
}

/**
 * Start a Renderer under `ctx`.
 *
 * Inputs: `ctx` — owner context used as the HSM parent environment.
 * Outputs: a started Renderer in `/Renderer/clean`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it; the renderer
 * does not retain `ctx` beyond start.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one dirty/frame protocol per instance; overlapping mark_dirty is
 * deferred while rendering.
 * Failure modes: owner paint notify rejection while still painting enters
 * `/failed`; activity cancel before notify emits `render_canceled` and returns
 * to `/clean`. Units: none.
 * Classification: runtime-safe.
 */
export function startRenderer(args: {
  ctx: hsm.Context;
}): Renderer {
  return hsm.start(args.ctx, new Renderer(), Renderer.model);
}
