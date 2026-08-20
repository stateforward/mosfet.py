import * as hsm from "@stateforward/hsm.ts";

import { isRecord, namedEvent, startMachine, stopMachine } from "./hsm-runtime.ts";
import {
  documentFromSpans,
  machineByName,
  mergePublishedModel,
  parsePublishedModels,
  type MachineGraph,
  type OtelDocument,
  type PublishedModel,
} from "./otel/machines.ts";
import { parseOtelSpanBatch, type OtelSpanBatch } from "./otel/otlp.ts";
import {
  connectOtelStream,
  isOtelSource,
  type OtelSource,
  type OtelStreamConnect,
} from "./otel/source.ts";
import { type ObserveSpan } from "./otel/span.ts";
import { clampReplayPosition, replayEvents, replayPrefix, type ReplayEvent } from "./otel/replay.ts";

const dashboardEvents = {
  "dashboard.source.selected": { name: "dashboard.source.selected", kind: hsm.Kinds.Event },
  "dashboard.load.completed": { name: "dashboard.load.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.load.failed": { name: "dashboard.load.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.machine.selected": { name: "dashboard.machine.selected", kind: hsm.Kinds.Event },
  "dashboard.command.prefill": { name: "dashboard.command.prefill", kind: hsm.Kinds.Event },
  "dashboard.command.send": { name: "dashboard.command.send", kind: hsm.Kinds.Event },
  "dashboard.reset": { name: "dashboard.reset", kind: hsm.Kinds.Event },
  "dashboard.model.published": { name: "dashboard.model.published", kind: hsm.Kinds.Event },
  "dashboard.replay.enter": { name: "dashboard.replay.enter", kind: hsm.Kinds.Event },
  "dashboard.replay.play": { name: "dashboard.replay.play", kind: hsm.Kinds.Event },
  "dashboard.replay.pause": { name: "dashboard.replay.pause", kind: hsm.Kinds.Event },
  "dashboard.replay.previous": { name: "dashboard.replay.previous", kind: hsm.Kinds.Event },
  "dashboard.replay.next": { name: "dashboard.replay.next", kind: hsm.Kinds.Event },
  "dashboard.replay.seek": { name: "dashboard.replay.seek", kind: hsm.Kinds.Event },
  "dashboard.replay.live": { name: "dashboard.replay.live", kind: hsm.Kinds.Event },
} as const;

export type CommandResult = {
  readonly result: "accepted" | "no_subscriber" | "error";
  readonly detail: string;
};

export type CommandPost = (command: { eventName: string; dataJson: string }) => Promise<CommandResult>;

export type DashboardEventName = keyof typeof dashboardEvents;

export type DashboardPhase = "idle" | "live" | "error";

export type DashboardReplaySnapshot = {
  readonly active: boolean;
  readonly playing: boolean;
  readonly position: number;
  readonly total: number;
  readonly current: ObserveSpan | null;
};

export type DashboardSnapshot = {
  readonly phase: DashboardPhase;
  readonly statePath: string;
  readonly source: OtelSource | null;
  readonly document: OtelDocument | null;
  readonly errorMessage: string | null;
  readonly selectedGraph: MachineGraph | null;
  readonly commandEventName: string;
  readonly commandDataJson: string;
  readonly commandResult: CommandResult | null;
  readonly replay: DashboardReplaySnapshot;
};

export type DashboardControllerOptions = {
  readonly onSnapshot?: (snapshot: DashboardSnapshot) => void;
  readonly connectStream?: OtelStreamConnect;
  readonly postCommand?: CommandPost;
};

class DashboardRuntime extends hsm.Instance {
  controller: DashboardController | null = null;
}

function controllerOf(instance: hsm.Instance): DashboardController | null {
  return instance instanceof DashboardRuntime ? instance.controller : null;
}

function sourceFromEvent(event: hsm.Event): OtelSource | null {
  if (!isRecord(event.data) || !isOtelSource(event.data["source"])) {
    return null;
  }
  return event.data["source"];
}

function batchFromEvent(event: hsm.Event): (OtelSpanBatch & { mode: "replace" | "append" }) | null {
  if (!isRecord(event.data)) {
    return null;
  }
  const mode = event.data["mode"];
  if (mode !== "replace" && mode !== "append") {
    return null;
  }
  const batch = parseOtelSpanBatch(event.data);
  if (batch === null) {
    return null;
  }
  return { ...batch, mode };
}

function messageFromEvent(event: hsm.Event): string | null {
  if (!isRecord(event.data) || typeof event.data["message"] !== "string") {
    return null;
  }
  return event.data["message"];
}

function machineNameFromEvent(event: hsm.Event): string | null {
  if (!isRecord(event.data) || typeof event.data["machineName"] !== "string") {
    return null;
  }
  return event.data["machineName"];
}

function stringField(event: hsm.Event, key: string): string | null {
  if (!isRecord(event.data) || typeof event.data[key] !== "string") {
    return null;
  }
  return event.data[key];
}

function replayPositionFromEvent(event: hsm.Event): number | null {
  if (!isRecord(event.data) || typeof event.data["position"] !== "number") {
    return null;
  }
  return event.data["position"];
}

export async function postCommandHttp(command: { eventName: string; dataJson: string }): Promise<CommandResult> {
  try {
    const response = await fetch("/v1/commands", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event_name: command.eventName, data_json: command.dataJson }),
    });
    const payload: unknown = await response.json();
    if (!isRecord(payload) || typeof payload["result"] !== "string" || typeof payload["detail"] !== "string") {
      return { result: "error", detail: "invalid command reply" };
    }
    if (payload["result"] === "accepted" || payload["result"] === "no_subscriber" || payload["result"] === "error") {
      return { result: payload["result"], detail: payload["detail"] };
    }
    return { result: "error", detail: payload["detail"] };
  } catch {
    return { result: "error", detail: "command request failed" };
  }
}

function rememberSource(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const controller = controllerOf(instance);
  const source = sourceFromEvent(event);
  if (controller === null || source === null) {
    return;
  }
  controller.applySource(source);
}

function applySpans(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const controller = controllerOf(instance);
  const batch = batchFromEvent(event);
  if (controller === null || batch === null) {
    return;
  }
  controller.applySpans(batch.observeSpans, batch.skipped, batch.mode);
}

function applyError(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.applyError(messageFromEvent(event) ?? "load failed");
}

function applyMachine(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const name = machineNameFromEvent(event);
  if (name === null) {
    return;
  }
  controllerOf(instance)?.applyMachine(name);
}

function applyPrefill(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const eventName = stringField(event, "eventName");
  if (eventName === null) {
    return;
  }
  controllerOf(instance)?.applyPrefill(eventName, stringField(event, "dataJson") ?? "");
}

function modelsFromEvent(event: hsm.Event): PublishedModel[] | null {
  if (!isRecord(event.data)) {
    return null;
  }
  if (Array.isArray(event.data["models"])) {
    return parsePublishedModels(event.data["models"]);
  }
  return parsePublishedModels(event.data);
}

function applyModels(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const models = modelsFromEvent(event);
  if (models === null) {
    return;
  }
  controllerOf(instance)?.applyModels(models);
}

function applySend(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const controller = controllerOf(instance);
  if (controller === null) {
    return;
  }
  void controller.sendCommand(stringField(event, "eventName"), stringField(event, "dataJson"));
}

function clearView(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.clearView();
}

function enterReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.enterReplay();
}

function playReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.playReplay();
}

function pauseReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.pauseReplay();
}

function previousReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.previousReplay();
}

function nextReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.nextReplay();
}

function seekReplay(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const position = replayPositionFromEvent(event);
  if (position !== null) {
    controllerOf(instance)?.seekReplay(position);
  }
}

function returnToLive(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.returnToLive();
}

async function streamLive(ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): Promise<void> {
  const controller = controllerOf(instance);
  if (controller === null) {
    return;
  }
  await controller.streamSource(ctx);
}

const dashboardModel = hsm.define(
  "Dashboard",
  hsm.initial(hsm.target("idle")),
  hsm.state(
    "idle",
    hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("../live"), hsm.effect(rememberSource)),
    hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
    hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(applySend)),
    hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModels)),
    hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../live"), hsm.effect(enterReplay)),
  ),
  hsm.state(
    "live",
    hsm.activity(streamLive),
    hsm.transition(hsm.on("dashboard.load.completed"), hsm.effect(applySpans)),
    hsm.transition(hsm.on("dashboard.load.failed"), hsm.target("../error"), hsm.effect(applyError)),
    hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("."), hsm.effect(rememberSource)),
    hsm.transition(hsm.on("dashboard.machine.selected"), hsm.effect(applyMachine)),
    hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
    hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(applySend)),
    hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModels)),
    hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
    hsm.transition(hsm.on("dashboard.replay.enter"), hsm.effect(enterReplay)),
    hsm.transition(hsm.on("dashboard.replay.play"), hsm.effect(playReplay)),
    hsm.transition(hsm.on("dashboard.replay.pause"), hsm.effect(pauseReplay)),
    hsm.transition(hsm.on("dashboard.replay.previous"), hsm.effect(previousReplay)),
    hsm.transition(hsm.on("dashboard.replay.next"), hsm.effect(nextReplay)),
    hsm.transition(hsm.on("dashboard.replay.seek"), hsm.effect(seekReplay)),
    hsm.transition(hsm.on("dashboard.replay.live"), hsm.effect(returnToLive)),
  ),
  hsm.state(
    "error",
    hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("../live"), hsm.effect(rememberSource)),
    hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
    hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(applySend)),
    hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModels)),
    hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
    hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../live"), hsm.effect(enterReplay)),
  ),
);

function phaseFromStatePath(statePath: string): DashboardPhase {
  if (statePath.endsWith("/live")) {
    return "live";
  }
  if (statePath.endsWith("/error")) {
    return "error";
  }
  return "idle";
}

export function isDashboardEventName(value: string): value is DashboardEventName {
  return Object.hasOwn(dashboardEvents, value);
}

export class DashboardController {
  #runtime = new DashboardRuntime();
  #machine: DashboardRuntime;
  #source: OtelSource | null = null;
  #spans: ObserveSpan[] = [];
  #replayEvents: ReplayEvent[] = [];
  #replayActive = false;
  #replayPlaying = false;
  #replayPosition = 0;
  #replayTimer: ReturnType<typeof setTimeout> | null = null;
  #models: PublishedModel[] = [];
  #skipped = 0;
  #document: OtelDocument | null = null;
  #errorMessage: string | null = null;
  #commandEventName = "";
  #commandDataJson = "";
  #commandResult: CommandResult | null = null;
  #pendingSend: Promise<void> | null = null;
  #onSnapshot: ((snapshot: DashboardSnapshot) => void) | null;
  #connectStream: OtelStreamConnect;
  #postCommand: CommandPost;

  constructor(options: DashboardControllerOptions = {}) {
    this.#onSnapshot = options.onSnapshot ?? null;
    this.#connectStream = options.connectStream ?? connectOtelStream;
    this.#postCommand = options.postCommand ?? postCommandHttp;
    this.#runtime.controller = this;
    this.#machine = startMachine(this.#runtime, dashboardModel);
  }

  snapshot(): DashboardSnapshot {
    const statePath = this.#machine.takeSnapshot().state;
    const document = this.#document;
    return {
      phase: phaseFromStatePath(statePath),
      statePath,
      source: this.#source,
      document,
      errorMessage: this.#errorMessage,
      selectedGraph: document === null ? null : machineByName(document, document.selectedMachine),
      commandEventName: this.#commandEventName,
      commandDataJson: this.#commandDataJson,
      commandResult: this.#commandResult,
      replay: {
        active: this.#replayActive,
        playing: this.#replayPlaying,
        position: this.#replayPosition,
        total: this.#replayEvents.length,
        current: this.#replayActive ? this.#replayEvents[this.#replayPosition - 1]?.span ?? null : null,
      },
    };
  }

  async dispatch(eventName: DashboardEventName, data?: unknown): Promise<DashboardSnapshot> {
    await this.#machine.dispatch(namedEvent(dashboardEvents[eventName].name, data));
    if (this.#pendingSend !== null) {
      await this.#pendingSend;
    }
    this.#emit();
    return this.snapshot();
  }

  async stop(): Promise<void> {
    this.#runtime.controller = null;
    await stopMachine(this.#machine);
  }

  applySource(source: OtelSource): void {
    this.#source = source;
    this.#errorMessage = null;
    this.#emit();
  }

  applySpans(spans: readonly ObserveSpan[], skipped: number, mode: "replace" | "append"): void {
    if (mode === "replace") {
      this.#spans = [...spans];
      this.#skipped = skipped;
    } else {
      this.#spans = [...this.#spans, ...spans];
      this.#skipped += skipped;
    }
    this.#replayEvents = replayEvents(this.#spans);
    this.#replayPosition = clampReplayPosition(this.#replayPosition, this.#replayEvents.length);
    this.#rebuildDocument();
    this.#errorMessage = null;
    this.#emit();
  }

  applyModels(models: readonly PublishedModel[]): void {
    const byName = new Map(this.#models.map((model) => [model.name, model]));
    for (const model of models) {
      byName.set(model.name, mergePublishedModel(byName.get(model.name), model));
    }
    this.#models = [...byName.values()];
    this.#rebuildDocument();
    this.#errorMessage = null;
    this.#emit();
  }

  applyError(message: string): void {
    this.#errorMessage = message;
    this.#emit();
  }

  applyMachine(machineName: string): void {
    const document = this.#document;
    if (document === null || !document.machines.some((machine) => machine.name === machineName)) {
      return;
    }
    this.#document = { ...document, selectedMachine: machineName };
    this.#emit();
  }

  clearView(): void {
    this.#stopReplayTimer();
    this.#source = null;
    this.#spans = [];
    this.#replayEvents = [];
    this.#replayActive = false;
    this.#replayPlaying = false;
    this.#replayPosition = 0;
    this.#models = [];
    this.#skipped = 0;
    this.#document = null;
    this.#errorMessage = null;
    this.#commandEventName = "";
    this.#commandDataJson = "";
    this.#commandResult = null;
    this.#emit();
  }

  enterReplay(): void {
    this.#stopReplayTimer();
    this.#replayEvents = replayEvents(this.#spans);
    this.#replayActive = true;
    this.#replayPlaying = false;
    this.#replayPosition = 0;
    this.#rebuildDocument();
    this.#emit();
  }

  playReplay(): void {
    if (!this.#replayActive) {
      this.enterReplay();
    }
    if (this.#replayEvents.length === 0 || this.#replayPosition >= this.#replayEvents.length) {
      return;
    }
    this.#replayPlaying = true;
    this.#scheduleReplayStep();
    this.#emit();
  }

  pauseReplay(): void {
    this.#stopReplayTimer();
    this.#emit();
  }

  previousReplay(): void {
    if (!this.#replayActive) {
      return;
    }
    this.#stopReplayTimer();
    this.#replayPosition = clampReplayPosition(this.#replayPosition - 1, this.#replayEvents.length);
    this.#rebuildDocument();
    this.#emit();
  }

  nextReplay(): void {
    if (!this.#replayActive) {
      return;
    }
    this.#replayPosition = clampReplayPosition(this.#replayPosition + 1, this.#replayEvents.length);
    if (this.#replayPosition >= this.#replayEvents.length) {
      this.#stopReplayTimer();
    }
    this.#rebuildDocument();
    this.#emit();
  }

  seekReplay(position: number): void {
    if (!this.#replayActive) {
      return;
    }
    this.#replayPosition = clampReplayPosition(position, this.#replayEvents.length);
    if (this.#replayPosition >= this.#replayEvents.length) {
      this.#stopReplayTimer();
    }
    this.#rebuildDocument();
    this.#emit();
  }

  returnToLive(): void {
    this.#stopReplayTimer();
    this.#replayActive = false;
    this.#replayPosition = this.#replayEvents.length;
    this.#rebuildDocument();
    this.#emit();
  }

  applyPrefill(eventName: string, dataJson: string): void {
    this.#commandEventName = eventName;
    this.#commandDataJson = dataJson;
    this.#emit();
  }

  async sendCommand(eventName: string | null, dataJson: string | null): Promise<void> {
    const work = this.#send(eventName, dataJson);
    this.#pendingSend = work;
    try {
      await work;
    } finally {
      if (this.#pendingSend === work) {
        this.#pendingSend = null;
      }
    }
  }

  async #send(eventName: string | null, dataJson: string | null): Promise<void> {
    const name = (eventName ?? this.#commandEventName).trim();
    const payload = dataJson ?? this.#commandDataJson;
    this.#commandEventName = name;
    this.#commandDataJson = payload;
    if (name.length === 0) {
      this.#commandResult = { result: "error", detail: "event_name is required" };
      this.#emit();
      return;
    }
    this.#commandResult = await this.#postCommand({ eventName: name, dataJson: payload });
    this.#emit();
  }

  async streamSource(ctx: hsm.Context): Promise<void> {
    const source = this.#source;
    if (source === null || source.kind !== "stream") {
      await this.dispatch("dashboard.load.failed", { message: "no otel stream selected" });
      return;
    }
    const finished = new Promise<void>((resolve) => {
      const onDone = (): void => {
        ctx.removeEventListener("done", onDone);
        resolve();
      };
      ctx.addEventListener("done", onDone);
      if (ctx.done) {
        onDone();
      }
    });
    const subscription = this.#connectStream(source.url, {
      onSnapshot: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.completed", { ...batch, mode: "replace" });
      },
      onSpans: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.completed", { ...batch, mode: "append" });
      },
      onModels: (models) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.model.published", { models });
      },
      onError: (message) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.failed", { message });
      },
    });
    try {
      await finished;
    } finally {
      subscription.close();
    }
  }

  #rebuildDocument(): void {
    const previous = this.#document?.selectedMachine ?? null;
    const spans = this.#replayActive
      ? replayPrefix(this.#spans, this.#replayEvents, this.#replayPosition)
      : this.#spans;
    const currentMachine = this.#replayEvents[this.#replayPosition - 1]?.span.attributes["hsm.machine.name"] ?? null;
    this.#document = documentFromSpans(
      spans,
      this.#replayActive ? 0 : this.#skipped,
      this.#replayActive ? currentMachine : previous,
      this.#models,
    );
  }

  #scheduleReplayStep(): void {
    this.#stopReplayTimer();
    if (!this.#replayPlaying) {
      return;
    }
    this.#replayTimer = setTimeout(() => {
      this.#replayTimer = null;
      this.nextReplay();
      if (this.#replayPlaying) {
        this.#scheduleReplayStep();
      }
    }, 700);
  }

  #stopReplayTimer(): void {
    if (this.#replayTimer !== null) {
      clearTimeout(this.#replayTimer);
      this.#replayTimer = null;
    }
    this.#replayPlaying = false;
  }

  #emit(): void {
    this.#onSnapshot?.(this.snapshot());
  }
}
