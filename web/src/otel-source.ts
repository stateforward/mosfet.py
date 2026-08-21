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

/**
 * Typed URL-bearing ingress for source connect / dashboard stream selection.
 *
 * Inputs: producer `origin`, optional `url` and `source`. `urlAllowed` is
 * computed here from `collectorUrl` and is never trusted from the caller.
 * Outputs: the payload guards and activities read. Ownership: returned record
 * is owned by the dispatch. Lifetime: one admission. Concurrency: synchronous.
 * Failure modes: missing/invalid origin or URL yields `urlAllowed: false`.
 * Classification: runtime-safe.
 */
export type SourceConnectData = {
  readonly urlAllowed: boolean;
  readonly origin: string;
  readonly url?: string;
  readonly source?: StreamSource;
};

const SOURCE_CONNECT_EVENT = sourceCommands["source.connect.requested"].name;

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

async function connectCollector(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
  const controller = controllerOf(instance);
  if (controller === null) {
    if (!ctx.done) {
      await instance.dispatch(hsm.typedEvent({
        event: sourceCompletions["source.connect.failed"],
        data: { message: "source host missing" },
      }));
    }
    return;
  }
  await controller.connect({ ctx, event });
}

function urlAllowed(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  return hsm.isRecord(event.data) && event.data["urlAllowed"] === true;
}

function sourceConnectOf(event: hsm.Event): SourceConnectData | null {
  if (!hsm.isRecord(event.data) || typeof event.data["urlAllowed"] !== "boolean" || typeof event.data["origin"] !== "string") {
    return null;
  }
  const url = event.data["url"];
  const source = event.data["source"];
  return {
    urlAllowed: event.data["urlAllowed"],
    origin: event.data["origin"],
    ...(typeof url === "string" ? { url } : {}),
    ...(isOtelSource(source) ? { source } : {}),
  };
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
        ? super.dispatch(eventOrContext, eventWithSourceConnect(data as hsm.Event))
        : super.dispatch(eventWithSourceConnect(eventOrContext));
    }
    return this.#dispatchController(eventOrContext, data);
  }

  async #dispatchController(eventName: OtelSourceEventName, data?: unknown): Promise<OtelSourceSnapshot> {
    const admitted = eventName === SOURCE_CONNECT_EVENT ? sourceConnectFrom(data) : data;
    await super.dispatch(
      admitted === undefined
        ? hsm.typedEvent({ event: sourceCommands[eventName] })
        : hsm.typedEvent({ event: sourceCommands[eventName], data: admitted }),
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

  async connect(args: { ctx: hsm.Context; event: hsm.Event }): Promise<void> {
    const fail = async (message: string): Promise<void> => {
      if (args.ctx.done) return;
      await this.dispatch(hsm.typedEvent({
        event: sourceCompletions["source.connect.failed"],
        data: { message },
      }));
    };
    const admitted = sourceConnectOf(args.event);
    if (admitted === null || admitted.origin.length === 0) {
      await fail("collector origin is missing");
      return;
    }
    const requested = admitted.url ?? admitted.source?.url;
    const url = collectorUrl({
      origin: admitted.origin,
      ...(requested !== undefined ? { requested } : {}),
    });
    if (url === null) {
      await fail("collector url is not allowed");
      return;
    }
    if (args.ctx.done) return;
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

/**
 * Build typed URL-bearing ingress so choice guards compare `urlAllowed` and
 * never construct `URL` objects.
 *
 * Inputs: the command payload (`origin`, optional `url` or `source.url`).
 * Caller-supplied `urlAllowed` is ignored and recomputed. Outputs:
 * `SourceConnectData`. Ownership: returns a new record; does not retain the
 * input. Lifetime: consumed by the following dispatch. Concurrency:
 * synchronous. Failure modes: missing/invalid origin or URL yields
 * `urlAllowed: false`. Classification: runtime-safe.
 */
export function sourceConnectFrom(data: unknown): SourceConnectData {
  const record = hsm.isRecord(data) ? data : {};
  const origin = typeof record["origin"] === "string" ? record["origin"] : "";
  const url = typeof record["url"] === "string" ? record["url"] : undefined;
  const source = isOtelSource(record["source"]) ? record["source"] : undefined;
  const requested = url ?? source?.url;
  const urlAllowed = origin.length > 0 && collectorUrl({
    origin,
    ...(requested !== undefined ? { requested } : {}),
  }) !== null;
  return {
    urlAllowed,
    origin,
    ...(url !== undefined ? { url } : {}),
    ...(source !== undefined ? { source } : {}),
  };
}

export function eventWithSourceConnect(event: hsm.Event): hsm.Event {
  if (event.name !== SOURCE_CONNECT_EVENT) return event;
  return { ...event, data: sourceConnectFrom(event.data) };
}

const DASHBOARD_SOURCE_SELECTED = "dashboard.source.selected";
const DASHBOARD_REPLAY_LIVE = "dashboard.replay.live";

export function eventWithUrlAdmission(event: hsm.Event): hsm.Event {
  if (
    event.name !== SOURCE_CONNECT_EVENT
    && event.name !== DASHBOARD_SOURCE_SELECTED
    && event.name !== DASHBOARD_REPLAY_LIVE
  ) {
    return event;
  }
  return { ...event, data: sourceConnectFrom(event.data) };
}


