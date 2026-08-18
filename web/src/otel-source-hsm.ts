import * as hsm from "@stateforward/hsm.ts";

import { isRecord, namedEvent, startMachine, stopMachine } from "./hsm-runtime.ts";
import { isOtelSource, streamSource, type OtelSource } from "./otel/source.ts";

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
  readonly source: OtelSource | null;
  readonly errorMessage: string | null;
};

export type OtelSourceControllerOptions = {
  readonly onSnapshot?: (snapshot: OtelSourceSnapshot) => void;
  readonly onReady?: (source: OtelSource) => void;
};

class OtelSourceRuntime extends hsm.Instance {
  controller: OtelSourceController | null = null;
}

function controllerOf(instance: hsm.Instance): OtelSourceController | null {
  return instance instanceof OtelSourceRuntime ? instance.controller : null;
}

function sourceFromEvent(event: hsm.Event): OtelSource | null {
  if (!isRecord(event.data) || !isOtelSource(event.data["source"])) {
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
    isRecord(event.data) && typeof event.data["message"] === "string" ? event.data["message"] : "connect failed";
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

export class OtelSourceController {
  #runtime = new OtelSourceRuntime();
  #machine: OtelSourceRuntime;
  #source: OtelSource | null = null;
  #errorMessage: string | null = null;
  #onSnapshot: ((snapshot: OtelSourceSnapshot) => void) | null;
  #onReady: ((source: OtelSource) => void) | null;

  constructor(options: OtelSourceControllerOptions = {}) {
    this.#onSnapshot = options.onSnapshot ?? null;
    this.#onReady = options.onReady ?? null;
    this.#runtime.controller = this;
    this.#machine = startMachine(this.#runtime, otelSourceModel);
  }

  snapshot(): OtelSourceSnapshot {
    const statePath = this.#machine.takeSnapshot().state;
    return {
      phase: phaseFromStatePath(statePath),
      statePath,
      source: this.#source,
      errorMessage: this.#errorMessage,
    };
  }

  async dispatch(eventName: OtelSourceEventName, data?: unknown): Promise<OtelSourceSnapshot> {
    await this.#machine.dispatch(namedEvent(sourceEvents[eventName].name, data));
    this.#emit();
    return this.snapshot();
  }

  async stop(): Promise<void> {
    this.#runtime.controller = null;
    await stopMachine(this.#machine);
  }

  rememberSource(source: OtelSource | null): void {
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
    this.#onReady?.(source);
  }

  async connect(event: hsm.Event): Promise<void> {
    const requested = isRecord(event.data) && typeof event.data["url"] === "string" ? event.data["url"] : undefined;
    const source = streamSource(requested);
    if (source.url.length === 0) {
      await this.dispatch("source.connect.failed", { message: "missing OTLP stream url" });
      return;
    }
    await this.dispatch("source.connected", { source });
  }

  #emit(): void {
    this.#onSnapshot?.(this.snapshot());
  }
}
