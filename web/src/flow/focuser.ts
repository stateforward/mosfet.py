import * as hsm from "../hsm.ts";

import { type ViewportBounds } from "./types.ts";

export type FocusTarget = {
  readonly kind: "node" | "machine" | "viewport";
  readonly machineName?: string;
  readonly nodePath?: string;
  readonly nodeId?: string;
  readonly bounds: ViewportBounds;
};

export class Focuser extends hsm.Instance {
  static readonly focusEvent = { name: "focus", kind: hsm.Kinds.Event } as const;
  static readonly clearEvent = { name: "clear", kind: hsm.Kinds.Event } as const;
  static readonly changedEvent = { name: "focus_changed", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Focuser",
    hsm.initial(hsm.target("unfocused")),
    hsm.state(
      "unfocused",
      hsm.transition(
        hsm.on(Focuser.focusEvent.name),
        hsm.target("../focused"),
        hsm.effect(Focuser.setFocus),
      ),
    ),
    hsm.state(
      "focused",
      hsm.entry(Focuser.onFocusedEntry),
      hsm.exit(Focuser.onFocusedExit),
      hsm.transition(
        hsm.on(Focuser.focusEvent.name),
        hsm.effect(Focuser.setFocus),
      ),
      hsm.transition(
        hsm.on(Focuser.clearEvent.name),
        hsm.target("../unfocused"),
        hsm.effect(Focuser.clearFocus),
      ),
    ),
  );

  current: FocusTarget | null = null;

  constructor() {
    super();
  }

  static setFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Focuser)) return;
    instance.current = focusTargetOf(event.data);
  }

  static clearFocus(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Focuser)) return;
    instance.current = null;
  }

  static onFocusedEntry(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Focuser) || instance.current === null) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Focuser.changedEvent, data: { focused: true } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static onFocusedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Focuser)) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Focuser.changedEvent, data: { focused: false } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }
}

/**
 * Start a Focuser under `ctx`.
 *
 * Inputs: `ctx` — owner context used as the HSM parent environment.
 * Outputs: a started Focuser in `/Focuser/unfocused`.
 * Ownership: caller owns the returned actor and must `hsm.stop` it.
 * Lifetime: until `hsm.stop` or owner context cancel.
 * Concurrency: one current focus target per instance.
 * Failure modes: malformed focus payloads leave `current` null and stay unfocused.
 * Units: bounds in world pixels.
 * Classification: runtime-safe.
 */
export function startFocuser(args: {
  ctx: hsm.Context;
}): Focuser {
  return hsm.start({ ctx: args.ctx, instance: new Focuser(), model: Focuser.model });
}

function focusTargetOf(value: unknown): FocusTarget | null {
  if (!hsm.isRecord(value)) return null;
  const bounds = value["bounds"];
  if (!hsm.isRecord(bounds)) return null;
  const left = bounds["left"];
  const right = bounds["right"];
  const top = bounds["top"];
  const bottom = bounds["bottom"];
  if (
    typeof left !== "number" || !Number.isFinite(left)
    || typeof right !== "number" || !Number.isFinite(right)
    || typeof top !== "number" || !Number.isFinite(top)
    || typeof bottom !== "number" || !Number.isFinite(bottom)
  ) {
    return null;
  }
  const kind = value["kind"] === "node" || value["kind"] === "viewport" ? value["kind"] : "machine";
  const target: FocusTarget = { kind, bounds: { left, right, top, bottom } };
  const machineName = value["machineName"];
  const nodePath = value["nodePath"];
  const nodeId = value["nodeId"];
  return {
    ...target,
    ...(typeof machineName === "string" ? { machineName } : {}),
    ...(typeof nodePath === "string" ? { nodePath } : {}),
    ...(typeof nodeId === "string" ? { nodeId } : {}),
  };
}
