import * as hsm from "./hsm.ts";
import { isOtelSource, streamSource, type OtelSource as StreamSource } from "./otel/source.ts";

const sourceCommands = {
  "source.connect.requested": { name: "source.connect.requested", kind: hsm.Kinds.Event },
  "source.disconnect.requested": { name: "source.disconnect.requested", kind: hsm.Kinds.Event },
  "source.attach": { name: "source.attach", kind: hsm.Kinds.Event },
  "source.detach": { name: "source.detach", kind: hsm.Kinds.Event },
} as const;

const sourceCompletions = {
  "source.connected": { name: "source.connected", kind: hsm.Kinds.CompletionEvent },
  "source.connect.failed": { name: "source.connect.failed", kind: hsm.Kinds.ErrorEvent },
  "source.stopped": { name: "source.stopped", kind: hsm.Kinds.CompletionEvent },
} as const;

export type OtelSourceEventName = keyof typeof sourceCommands;

export type OtelSourcePhase = "idle" | "connecting" | "live" | "error";

export type OtelSourceSnapshot = {
  readonly phase: OtelSourcePhase;
  readonly statePath: string;
  readonly source: StreamSource | null;
  readonly errorMessage: string | null;
};

const ALLOWED_COLLECTOR_PATH = "/v1/traces/stream";

function controllerOf(instance: hsm.Instance): OtelSource | null {
  return instance instanceof OtelSource ? instance : null;
}

function sourceFromEvent(event: hsm.Event): StreamSource | null {
  if (!hsm.isRecord(event.data) || !isOtelSource(event.data["source"])) {
    return null;
  }
  return event.data["source"];
}

function rememberReadySource(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const source = sourceFromEvent(event);
  if (source === null) {
    return;
  }
  controllerOf(instance)?.rememberSource(source);
}

function rememberConnectFailure(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const message =
    hsm.isRecord(event.data) && typeof event.data["message"] === "string" ? event.data["message"] : "connect failed";
  controllerOf(instance)?.rememberError(message);
}

function clearSource(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.rememberSource(null);
}

function emitReady(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.emitReady();
}

async function connectCollector(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
  const controller = controllerOf(instance);
  if (controller === null) {
    return;
  }
  await controller.connect(event);
}

function urlAllowed(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  if (!hsm.isRecord(event.data) || typeof event.data["origin"] !== "string") return false;
  const requested = typeof event.data["url"] === "string" ? event.data["url"] : undefined;
  return collectorUrl({
    origin: event.data["origin"],
    ...(requested !== undefined ? { requested } : {}),
  }) !== null;
}

async function stopHost(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
  await instance.dispatch(hsm.typedEvent({ event: sourceCompletions["source.stopped"] }));
}

const otelSourceModel = hsm.define(
  "OtelSource",
  hsm.initial(hsm.target("connected")),
  hsm.state(
    "disconnected",
    hsm.defer("source.connect.requested"),
    hsm.transition(hsm.on("source.attach"), hsm.target("../connected")),
  ),
  hsm.state(
    "connected",
    hsm.initial(hsm.target("idle")),
    hsm.transition(hsm.on("source.detach"), hsm.target("../stopping")),
    hsm.state(
      "idle",
      hsm.transition(hsm.on("source.connect.requested"), hsm.target("../validate")),
    ),
    hsm.choice(
      "validate",
      hsm.transition(hsm.guard(urlAllowed), hsm.target("connecting")),
      hsm.transition(hsm.target("error"), hsm.effect(rememberConnectFailure)),
    ),
    hsm.state(
      "connecting",
      hsm.activity(connectCollector),
      hsm.transition(hsm.on("source.connected"), hsm.target("../live"), hsm.effect(rememberReadySource)),
      hsm.transition(hsm.on("source.connect.failed"), hsm.target("../error"), hsm.effect(rememberConnectFailure)),
    ),
    hsm.state(
      "live",
      hsm.entry(emitReady),
      hsm.transition(hsm.on("source.disconnect.requested"), hsm.target("../idle"), hsm.effect(clearSource)),
      hsm.transition(hsm.on("source.connect.requested"), hsm.target("../validate")),
    ),
    hsm.state(
      "error",
      hsm.transition(hsm.on("source.connect.requested"), hsm.target("../validate")),
    ),
  ),
  hsm.state(
    "stopping",
    hsm.defer("source.attach"),
    hsm.defer("source.connect.requested"),
    hsm.activity(stopHost),
    hsm.transition(hsm.on("source.stopped"), hsm.target("../disconnected")),
  ),
);

function phaseFromStatePath(statePath: string): OtelSourcePhase {
  if (statePath.endsWith("/connecting")) {
    return "connecting";
  }
  if (statePath.endsWith("/live")) {
    return "live";
  }
  if (statePath.endsWith("/error")) {
    return "error";
  }
  return "idle";
}

export function isOtelSourceEventName(value: string): value is OtelSourceEventName {
  return Object.hasOwn(sourceCommands, value);
}

export class OtelSource extends hsm.from(HTMLElement) {
  #source: StreamSource | null = null;
  #errorMessage: string | null = null;
  onSnapshot: ((snapshot: OtelSourceSnapshot) => void) | null = null;
  onReady: ((source: StreamSource) => void) | null = null;

  constructor() {
    super();
  }

  boot(): void {
    hsm.start(this, otelSourceModel);
  }

  /**
   * Request host attach.
   *
   * Inputs: none. Dispatches `source.attach`.
   * Outputs: none directly. Topology moves `disconnected` to `connected` only
   * when the host is already disconnected. Attach is ignored unless
   * disconnected, and deferred while stopping so reconnect cannot start until
   * `source.stopped` has completed the previous detach.
   * Ownership: this collector host owns the dispatch. Lifetime: one attach
   * request; the connected session lasts until detach/stop.
   * Concurrency: runtime-safe. Overlapping attach while connected is ignored.
   * While stopping, attach is deferred, not dropped.
   * Failure modes: dispatch rejection is classified by `catchFailure` as a
   * host-drop when the runtime is unstarted or stopped; otherwise reported.
   * Units: none.
   * Classification: runtime-safe.
   */
  requestAttach(): void {
    void super.dispatch(hsm.typedEvent({ event: sourceCommands["source.attach"] })).catch(hsm.catchFailure(this));
  }

  /**
   * Request host detach through stopping.
   *
   * Inputs: none. Dispatches `source.detach`.
   * Outputs: none directly. Topology moves `connected` to `stopping`, then
   * `disconnected` on `source.stopped`.
   * Ownership: this collector host owns the dispatch. Lifetime: one detach
   * request; stopping lasts until `source.stopped`.
   * Concurrency: runtime-safe. Detach while already disconnected is ignored.
   * Failure modes: dispatch rejection is classified by `catchFailure` as a
   * host-drop when the runtime is unstarted or stopped; otherwise reported.
   * Units: none.
   * Classification: runtime-safe.
   */
  requestDetach(): void {
    void super.dispatch(hsm.typedEvent({ event: sourceCommands["source.detach"] })).catch(hsm.catchFailure(this));
  }

  snapshot(): OtelSourceSnapshot {
    const statePath = this.takeSnapshot().state;
    return {
      phase: phaseFromStatePath(statePath),
      statePath,
      source: this.#source,
      errorMessage: this.#errorMessage,
    };
  }

  override dispatch(eventName: OtelSourceEventName, data?: unknown): Promise<OtelSourceSnapshot>;
  override dispatch(event: hsm.Event): hsm.Completion;
  override dispatch(ctx: hsm.Context, event: hsm.Event): hsm.Completion;
  override dispatch(eventOrContext: OtelSourceEventName | hsm.Event | hsm.Context, data?: unknown): hsm.Completion | Promise<OtelSourceSnapshot> {
    if (typeof eventOrContext !== "string") {
      return eventOrContext instanceof hsm.Context
        ? super.dispatch(eventOrContext, data as hsm.Event)
        : super.dispatch(eventOrContext);
    }
    return this.#dispatchController(eventOrContext, data);
  }

  async #dispatchController(eventName: OtelSourceEventName, data?: unknown): Promise<OtelSourceSnapshot> {
    await super.dispatch(
      data === undefined
        ? hsm.typedEvent({ event: sourceCommands[eventName] })
        : hsm.typedEvent({ event: sourceCommands[eventName], data }),
    );
    this.#emit();
    return this.snapshot();
  }

  override async stop(): Promise<void> {
    await hsm.stop(this);
  }

  rememberSource(source: StreamSource | null): void {
    this.#source = source;
    this.#errorMessage = null;
    this.#emit();
  }

  rememberError(message: string): void {
    this.#source = null;
    this.#errorMessage = message;
    this.#emit();
  }

  emitReady(): void {
    const source = this.#source;
    if (source === null) {
      return;
    }
    this.onReady?.(source);
  }

  async connect(event: hsm.Event): Promise<void> {
    if (!hsm.isRecord(event.data) || typeof event.data["origin"] !== "string") {
      return;
    }
    const requested = typeof event.data["url"] === "string" ? event.data["url"] : undefined;
    const url = collectorUrl({
      origin: event.data["origin"],
      ...(requested !== undefined ? { requested } : {}),
    });
    if (url === null) {
      return;
    }
    const source = streamSource(url);
    await this.dispatch(hsm.typedEvent({ event: sourceCompletions["source.connected"], data: { source } }));
  }

  #emit(): void {
    this.onSnapshot?.(this.snapshot());
  }
}

export function collectorUrl(args: { readonly requested?: string; readonly origin: string }): string | null {
  const requested = args.requested;
  const value = requested === undefined || requested.length === 0 ? ALLOWED_COLLECTOR_PATH : requested;
  try {
    const origin = new URL(args.origin);
    const url = new URL(value, origin);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (url.origin !== origin.origin) return null;
    if (url.pathname !== ALLOWED_COLLECTOR_PATH) return null;
    if (url.username.length > 0 || url.password.length > 0) return null;
    if (url.hash.length > 0 || url.search.length > 0) return null;
    return url.pathname;
  } catch {
    return null;
  }
}


