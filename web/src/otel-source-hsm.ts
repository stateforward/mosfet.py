import * as hsm from "./hsm.ts";
import { isOtelSource, streamSource, type OtelSource as StreamSource } from "./otel/source.ts";

const sourceEvents = {
  "source.connect.requested": { name: "source.connect.requested", kind: hsm.Kinds.Event },
  "source.disconnect.requested": { name: "source.disconnect.requested", kind: hsm.Kinds.Event },
  "source.connected": { name: "source.connected", kind: hsm.Kinds.CompletionEvent },
  "source.connect.failed": { name: "source.connect.failed", kind: hsm.Kinds.ErrorEvent },
} as const;

export type OtelSourceEventName = keyof typeof sourceEvents;

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

const otelSourceModel = hsm.define(
  "OtelSource",
  hsm.initial(hsm.target("idle")),
  hsm.state(
    "idle",
    hsm.transition(hsm.on("source.connect.requested"), hsm.target("../connecting")),
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
    hsm.transition(hsm.on("source.connect.requested"), hsm.target("../connecting")),
  ),
  hsm.state(
    "error",
    hsm.transition(hsm.on("source.connect.requested"), hsm.target("../connecting")),
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
  return Object.hasOwn(sourceEvents, value);
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
    await super.dispatch(hsm.namedEvent(sourceEvents[eventName].name, data));
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
    const requested = hsm.isRecord(event.data) && typeof event.data["url"] === "string" ? event.data["url"] : undefined;
    const url = collectorUrl(requested);
    if (url === null) {
      await this.dispatch("source.connect.failed", { message: "collector url is not allowed" });
      return;
    }
    const source = streamSource(url);
    await this.dispatch("source.connected", { source });
  }

  #emit(): void {
    this.onSnapshot?.(this.snapshot());
  }
}

export function collectorUrl(requested: string | undefined): string | null {
  const value = requested === undefined || requested.length === 0 ? ALLOWED_COLLECTOR_PATH : requested;
  if (value === ALLOWED_COLLECTOR_PATH || value.startsWith(`${ALLOWED_COLLECTOR_PATH}?`)) {
    return value;
  }
  if (!value.startsWith("/")) {
    try {
      const url = new URL(value);
      if (url.protocol !== "http:" && url.protocol !== "https:") return null;
      const origin = globalThis.location?.origin;
      if (typeof origin === "string" && origin.length > 0 && url.origin !== origin) return null;
      if (url.pathname !== ALLOWED_COLLECTOR_PATH) return null;
      return `${url.pathname}${url.search}`;
    } catch {
      return null;
    }
  }
  if (!value.startsWith(ALLOWED_COLLECTOR_PATH)) return null;
  return value;
}
