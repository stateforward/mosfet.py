import * as hsm from "../hsm.ts";

export type FocusTarget = {
  readonly kind: "node" | "machine" | "viewport";
  readonly machineName?: string;
  readonly nodePath?: string;
  readonly nodeId?: string;
  readonly bounds: ViewportBounds;
};

export type ViewportBounds = {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
};

export class Focuser extends hsm.Instance {
  static readonly focusEvent = { name: "focus", kind: hsm.Kinds.Event } as const;
  static readonly clearEvent = { name: "clear", kind: hsm.Kinds.Event } as const;

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

  readonly host: HTMLElement | null;
  current: FocusTarget | null = null;

  constructor(host: HTMLElement | null = null) {
    super();
    this.host = host;
  }

  focus(target: FocusTarget): void {
    this.dispatch(hsm.typedEvent(Focuser.focusEvent, target));
  }

  clear(): void {
    this.dispatch(hsm.typedEvent(Focuser.clearEvent));
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
    instance.host?.classList.add("is-focused");
  }

  static onFocusedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Focuser)) return;
    instance.host?.classList.remove("is-focused");
  }
}

export function startFocuser(args: {
  ctx: hsm.Context;
  host?: HTMLElement | null;
}): Focuser {
  return hsm.start(args.ctx, new Focuser(args.host ?? null), Focuser.model);
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
