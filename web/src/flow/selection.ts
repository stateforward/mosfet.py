import * as hsm from "../hsm.ts";

export type SelectionBox = {
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
};

export type SelectionClickData = { readonly id: string; readonly kind: "node" | "edge"; readonly additive: boolean };
export type SelectionPointData = { readonly x: number; readonly y: number };
export type SelectionBoxEndData = { readonly ids: readonly string[] };
export type SelectionSnapshot = {
  readonly nodeIds: readonly string[];
  readonly edgeIds: readonly string[];
  readonly box: SelectionBox | null;
};

export class Selection extends hsm.Instance {
  static readonly clickEvent = { name: "selection_click", kind: hsm.Kinds.Event } as const;
  static readonly boxStartEvent = { name: "box_start", kind: hsm.Kinds.Event } as const;
  static readonly boxMoveEvent = { name: "box_move", kind: hsm.Kinds.Event } as const;
  static readonly boxEndEvent = { name: "box_end", kind: hsm.Kinds.Event } as const;
  static readonly clearEvent = { name: "selection_clear", kind: hsm.Kinds.Event } as const;
  static readonly changedEvent = { name: "selection_changed", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Selection",
    hsm.initial(hsm.target("idle")),
    hsm.state(
      "idle",
      hsm.initial(hsm.target("none")),
      hsm.transition(
        hsm.on(Selection.boxStartEvent.name),
        hsm.target("../box"),
        hsm.effect(Selection.startBox),
      ),
      hsm.state(
        "none",
        hsm.transition(
          hsm.on(Selection.clickEvent.name),
          hsm.target("../picking"),
          hsm.effect(Selection.applyClick),
        ),
      ),
      hsm.state(
        "picking",
        hsm.transition(hsm.on(Selection.clickEvent.name), hsm.effect(Selection.applyClick)),
        hsm.transition(
          hsm.on(Selection.clearEvent.name),
          hsm.target("../none"),
          hsm.effect(Selection.clearAll),
        ),
      ),
    ),
    hsm.state(
      "box",
      hsm.transition(hsm.on(Selection.boxMoveEvent.name), hsm.effect(Selection.moveBox)),
      hsm.transition(
        hsm.on(Selection.boxEndEvent.name),
        hsm.target("../idle/picking"),
        hsm.effect(Selection.endBox),
      ),
      hsm.transition(
        hsm.on(Selection.clearEvent.name),
        hsm.target("../idle/none"),
        hsm.effect(Selection.clearAll),
      ),
    ),
  );

  #nodeIds = new Set<string>();
  #edgeIds = new Set<string>();
  #box: SelectionBox | null = null;
  #boxOrigin: { x: number; y: number } | null = null;

  snapshot(): SelectionSnapshot {
    return {
      nodeIds: [...this.#nodeIds],
      edgeIds: [...this.#edgeIds],
      box: this.#box,
    };
  }

  static applyClick(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Selection) || !hsm.isRecord(event.data)) return;
    const id = event.data["id"];
    const kind = event.data["kind"];
    const additive = event.data["additive"] === true;
    if (typeof id !== "string" || (kind !== "node" && kind !== "edge")) return;
    const bucket = kind === "node" ? instance.#nodeIds : instance.#edgeIds;
    if (!additive) {
      instance.#nodeIds.clear();
      instance.#edgeIds.clear();
      bucket.add(id);
    } else if (bucket.has(id)) {
      bucket.delete(id);
    } else {
      bucket.add(id);
    }
    instance.#emit();
  }

  static startBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Selection) || !hsm.isRecord(event.data)) return;
    const x = event.data["x"];
    const y = event.data["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    instance.#boxOrigin = { x, y };
    instance.#box = { left: x, top: y, right: x, bottom: y };
    instance.#emit();
  }

  static moveBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Selection) || instance.#boxOrigin === null || !hsm.isRecord(event.data)) return;
    const x = event.data["x"];
    const y = event.data["y"];
    if (typeof x !== "number" || typeof y !== "number") return;
    instance.#box = {
      left: Math.min(instance.#boxOrigin.x, x),
      top: Math.min(instance.#boxOrigin.y, y),
      right: Math.max(instance.#boxOrigin.x, x),
      bottom: Math.max(instance.#boxOrigin.y, y),
    };
    instance.#emit();
  }

  static endBox(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Selection)) return;
    instance.#boxOrigin = null;
    instance.#box = null;
    if (hsm.isRecord(event.data) && Array.isArray(event.data["ids"])) {
      instance.#nodeIds = new Set(event.data["ids"].filter((id): id is string => typeof id === "string"));
    }
    instance.#emit();
  }

  static clearAll(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Selection)) return;
    instance.#nodeIds.clear();
    instance.#edgeIds.clear();
    instance.#box = null;
    instance.#boxOrigin = null;
    instance.#emit();
  }

  #emit(): void {
    void hsm.notifyOwner({
      instance: this,
      event: hsm.typedEvent({ event: Selection.changedEvent, data: this.snapshot() }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(this)));
  }
}

export function startSelection(args: { ctx: hsm.Context }): Selection {
  return hsm.start({ ctx: args.ctx, instance: new Selection(), model: Selection.model });
}
