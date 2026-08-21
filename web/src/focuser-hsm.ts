import * as hsm from "@stateforward/hsm.ts";

import { namedEvent, startMachine } from "./hsm-runtime.ts";

export type FocusTarget = {
  readonly kind: "node" | "machine";
  readonly machineName?: string;
  readonly nodePath?: string;
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
        hsm.target("."),
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
  readonly onFocus: ((target: FocusTarget) => void) | null;
  readonly onClear: (() => void) | null;
  current: FocusTarget | null = null;

  constructor(
    host: HTMLElement | null = null,
    config: { onFocus?: (target: FocusTarget) => void; onClear?: () => void } = {},
  ) {
    super();
    this.host = host;
    this.onFocus = config.onFocus ?? null;
    this.onClear = config.onClear ?? null;
  }

  focus(target: FocusTarget): void {
    this.dispatch(namedEvent(Focuser.focusEvent.name, target));
  }

  clear(): void {
    this.dispatch(namedEvent(Focuser.clearEvent.name));
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
    instance.onFocus?.(instance.current);
  }

  static onFocusedExit(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Focuser)) return;
    instance.host?.classList.remove("is-focused");
    instance.onClear?.();
  }
}

export function startFocuser(
  ctx: hsm.Context,
  host: HTMLElement | null,
  config: { onFocus?: (target: FocusTarget) => void; onClear?: () => void } = {},
): Focuser {
  return startMachine(ctx, new Focuser(host, config), Focuser.model);
}

function focusTargetOf(value: unknown): FocusTarget | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const bounds = record["bounds"];
  if (typeof bounds !== "object" || bounds === null) return null;
  const box = bounds as Record<string, unknown>;
  const left = box["left"];
  const right = box["right"];
  const top = box["top"];
  const bottom = box["bottom"];
  if (
    typeof left !== "number" || !Number.isFinite(left)
    || typeof right !== "number" || !Number.isFinite(right)
    || typeof top !== "number" || !Number.isFinite(top)
    || typeof bottom !== "number" || !Number.isFinite(bottom)
  ) {
    return null;
  }
  const kind = record["kind"] === "node" || record["target"] === "node" ? "node" : "machine";
  const target: FocusTarget = { kind, bounds: { left, right, top, bottom } };
  if (typeof record["machineName"] === "string") {
    return typeof record["nodePath"] === "string"
      ? { ...target, machineName: record["machineName"], nodePath: record["nodePath"] }
      : { ...target, machineName: record["machineName"] };
  }
  if (typeof record["nodePath"] === "string") {
    return { ...target, nodePath: record["nodePath"] };
  }
  return target;
}
