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

export type ConnectStartData = {
  readonly source: string;
  readonly sourceHandle?: string;
  readonly sourcePosition: HandlePosition;
  readonly start: XYPosition;
};

export class Connection extends hsm.Instance {
  static readonly startEvent = { name: "connect_start", kind: hsm.Kinds.Event } as const;
  static readonly moveEvent = { name: "connect_move", kind: hsm.Kinds.Event } as const;
  static readonly completeEvent = { name: "connect_complete", kind: hsm.Kinds.Event } as const;
  static readonly cancelEvent = { name: "connect_cancel", kind: hsm.Kinds.Event } as const;
  static readonly draftEvent = { name: "connect_draft", kind: hsm.Kinds.Event } as const;
  static readonly finishedEvent = { name: "connect_finished", kind: hsm.Kinds.Event } as const;

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

  #draft: ConnectionDraft | null = null;

  beginFrom(data: ConnectStartData): void {
    this.dispatch(hsm.typedEvent(Connection.startEvent, data));
  }

  cursorMove(point: XYPosition): void {
    this.dispatch(hsm.typedEvent(Connection.moveEvent, point));
  }

  complete(data: { target: string; targetHandle?: string; kind?: HandleKind }): void {
    this.dispatch(hsm.typedEvent(Connection.completeEvent, data));
  }

  cancel(): void {
    this.dispatch(hsm.typedEvent(Connection.cancelEvent));
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
    instance.#draft = {
      source,
      sourcePosition,
      start: { x, y },
      cursor: { x, y },
      ...(typeof sourceHandle === "string" ? { sourceHandle } : {}),
    };
    instance.#emitDraft();
  }

  static move(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Connection) || instance.#draft === null || !hsm.isRecord(event.data)) return;
    const x = event.data["x"];
    const y = event.data["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    instance.#draft = { ...instance.#draft, cursor: { x, y } };
    instance.#emitDraft();
  }

  static finish(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Connection) || instance.#draft === null || !hsm.isRecord(event.data)) {
      if (instance instanceof Connection) Connection.reset(_ctx, instance, event);
      return;
    }
    const target = event.data["target"];
    if (typeof target !== "string" || target === instance.#draft.source) {
      Connection.reset(_ctx, instance, event);
      return;
    }
    const targetHandle = event.data["targetHandle"];
    const completed: ConnectionComplete = {
      source: instance.#draft.source,
      target,
      ...(instance.#draft.sourceHandle !== undefined ? { sourceHandle: instance.#draft.sourceHandle } : {}),
      ...(typeof targetHandle === "string" ? { targetHandle } : {}),
    };
    instance.#draft = null;
    hsm.notifyOwner({ instance, event: hsm.typedEvent(Connection.finishedEvent, completed) });
    instance.#emitDraft();
  }

  static reset(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Connection)) return;
    instance.#draft = null;
    instance.#emitDraft();
  }

  #emitDraft(): void {
    hsm.notifyOwner({ instance: this, event: hsm.typedEvent(Connection.draftEvent, this.#draft) });
  }
}

export function startConnection(args: { ctx: hsm.Context }): Connection {
  return hsm.start(args.ctx, new Connection(), Connection.model);
}
