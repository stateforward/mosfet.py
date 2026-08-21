import * as hsm from "../hsm.ts";

import type { HandleKind, HandlePosition, XYPosition } from "./types.ts";

export type ConnectionDraft = {
  readonly source: string;
  readonly sourceHandle?: string;
  readonly sourcePosition: HandlePosition;
  readonly start: XYPosition;
  readonly cursor: XYPosition;
};

export type ConnectionComplete = {
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
};

export class Connection extends hsm.Instance {
  static readonly startEvent = { name: "connect_start", kind: hsm.Kinds.Event } as const;
  static readonly moveEvent = { name: "connect_move", kind: hsm.Kinds.Event } as const;
  static readonly completeEvent = { name: "connect_complete", kind: hsm.Kinds.Event } as const;
  static readonly cancelEvent = { name: "connect_cancel", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Connection",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.transition(
        hsm.on(Connection.startEvent.name),
        hsm.target("../connecting"),
        hsm.effect(Connection.begin),
      ),
    ),
    hsm.state(
      "connecting",
      hsm.transition(hsm.on(Connection.moveEvent.name), hsm.effect(Connection.move)),
      hsm.transition(
        hsm.on(Connection.completeEvent.name),
        hsm.target("../idle"),
        hsm.effect(Connection.finish),
      ),
      hsm.transition(
        hsm.on(Connection.cancelEvent.name),
        hsm.target("../idle"),
        hsm.effect(Connection.reset),
      ),
    ),
  );

  draft: ConnectionDraft | null = null;
  readonly onDraft: ((draft: ConnectionDraft | null) => void) | null;
  readonly onComplete: ((connection: ConnectionComplete) => boolean) | null;

  constructor(
    hooks: {
      onDraft?: (draft: ConnectionDraft | null) => void;
      onComplete?: (connection: ConnectionComplete) => boolean;
    } = {},
  ) {
    super();
    this.onDraft = hooks.onDraft ?? null;
    this.onComplete = hooks.onComplete ?? null;
  }

  beginFrom(data: {
    source: string;
    sourceHandle?: string;
    sourcePosition: HandlePosition;
    start: XYPosition;
  }): void {
    this.dispatch(hsm.namedEvent(Connection.startEvent.name, data));
  }

  cursorMove(point: XYPosition): void {
    this.dispatch(hsm.namedEvent(Connection.moveEvent.name, point));
  }

  complete(data: { target: string; targetHandle?: string; kind?: HandleKind }): void {
    this.dispatch(hsm.namedEvent(Connection.completeEvent.name, data));
  }

  cancel(): void {
    this.dispatch(hsm.namedEvent(Connection.cancelEvent.name));
  }

  static begin(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Connection) || !hsm.isRecord(event.data)) return;
    const source = event.data["source"];
    const start = event.data["start"];
    const sourcePosition = event.data["sourcePosition"];
    if (typeof source !== "string" || !hsm.isRecord(start)) return;
    if (sourcePosition !== "top" && sourcePosition !== "right" && sourcePosition !== "bottom" && sourcePosition !== "left") {
      return;
    }
    const x = start["x"];
    const y = start["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    const sourceHandle = event.data["sourceHandle"];
    instance.draft = {
      source,
      sourcePosition,
      start: { x, y },
      cursor: { x, y },
      ...(typeof sourceHandle === "string" ? { sourceHandle } : {}),
    };
    instance.onDraft?.(instance.draft);
  }

  static move(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Connection) || instance.draft === null || !hsm.isRecord(event.data)) return;
    const x = event.data["x"];
    const y = event.data["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    instance.draft = { ...instance.draft, cursor: { x, y } };
    instance.onDraft?.(instance.draft);
  }

  static finish(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Connection) || instance.draft === null || !hsm.isRecord(event.data)) {
      if (instance instanceof Connection) Connection.reset(_ctx, instance, event);
      return;
    }
    const target = event.data["target"];
    if (typeof target !== "string" || target === instance.draft.source) {
      Connection.reset(_ctx, instance, event);
      return;
    }
    const targetHandle = event.data["targetHandle"];
    const completed: ConnectionComplete = {
      source: instance.draft.source,
      target,
      ...(instance.draft.sourceHandle !== undefined ? { sourceHandle: instance.draft.sourceHandle } : {}),
      ...(typeof targetHandle === "string" ? { targetHandle } : {}),
    };
    const valid = instance.onComplete?.(completed) ?? true;
    if (!valid) {
      Connection.reset(_ctx, instance, event);
      return;
    }
    instance.draft = null;
    instance.onDraft?.(null);
  }

  static reset(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Connection)) return;
    instance.draft = null;
    instance.onDraft?.(null);
  }
}

export function startConnection(
  ctx: hsm.Context,
  hooks: {
    onDraft?: (draft: ConnectionDraft | null) => void;
    onComplete?: (connection: ConnectionComplete) => boolean;
  } = {},
): Connection {
  return hsm.start(ctx, new Connection(hooks), Connection.model);
}
