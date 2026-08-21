import * as hsm from "./hsm.ts";
import {
  environmentWorkspaceGraphs,
  machineNamesInOwnedSubtree,
} from "./dashboard-graphs.ts";
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
  "dashboard.command.completed": { name: "dashboard.command.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.command.failed": { name: "dashboard.command.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.visibility.set": { name: "dashboard.visibility.set", kind: hsm.Kinds.Event },
  "dashboard.visibility.action": { name: "dashboard.visibility.action", kind: hsm.Kinds.Event },
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
  readonly visibleMachines: Readonly<Record<string, boolean>>;
};

export const COMMAND_EVENT_NAME = /^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$/;

function controllerOf(instance: hsm.Instance): Dashboard | null {
  return instance instanceof Dashboard ? instance : null;
}

function sourceFromEvent(event: hsm.Event): OtelSource | null {
  if (!hsm.isRecord(event.data) || !isOtelSource(event.data["source"])) {
    return null;
  }
  return event.data["source"];
}

function batchFromEvent(event: hsm.Event): (OtelSpanBatch & { mode: "replace" | "append" }) | null {
  if (!hsm.isRecord(event.data)) {
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
  if (!hsm.isRecord(event.data) || typeof event.data["message"] !== "string") {
    return null;
  }
  return event.data["message"];
}

function machineNameFromEvent(event: hsm.Event): string | null {
  if (!hsm.isRecord(event.data) || typeof event.data["machineName"] !== "string") {
    return null;
  }
  return event.data["machineName"];
}

function stringField(event: hsm.Event, key: string): string | null {
  if (!hsm.isRecord(event.data) || typeof event.data[key] !== "string") {
    return null;
  }
  return event.data[key];
}

function replayPositionFromEvent(event: hsm.Event): number | null {
  if (!hsm.isRecord(event.data) || typeof event.data["position"] !== "number") {
    return null;
  }
  return event.data["position"];
}

export async function postCommandHttp(command: { eventName: string; dataJson: string }): Promise<CommandResult> {
  if (!COMMAND_EVENT_NAME.test(command.eventName)) {
    return { result: "error", detail: "event_name is not an allowed command" };
  }
  try {
    const response = await fetch("/v1/commands", {
      method: "POST",
      headers: { "content-type": "application/json", "x-requested-with": "bot-dashboard" },
      body: JSON.stringify({ event_name: command.eventName, data_json: command.dataJson }),
    });
    const payload: unknown = await response.json();
    if (!hsm.isRecord(payload) || typeof payload["result"] !== "string" || typeof payload["detail"] !== "string") {
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
  if (!hsm.isRecord(event.data)) {
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
  controller.queueSend(stringField(event, "eventName"), stringField(event, "dataJson"));
}

function applyVisibility(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  if (!hsm.isRecord(event.data)) return;
  const machineName = event.data["machineName"];
  const visible = event.data["visible"];
  if (typeof machineName !== "string" || typeof visible !== "boolean") return;
  controllerOf(instance)?.applyVisibility(machineName, visible);
}

function applyVisibilityAction(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const action = stringField(event, "action");
  if (action !== "show-all" && action !== "hide-all" && action !== "hide-unobserved") return;
  controllerOf(instance)?.applyVisibilityAction(action);
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

function replayStep(): number {
  return 700;
}

const dashboardModel = hsm.define(
  "Dashboard",
  hsm.initial(hsm.target("session")),
  hsm.state(
    "session",
    hsm.initial(hsm.target("idle")),
    hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
    hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(applySend)),
    hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModels)),
    hsm.transition(hsm.on("dashboard.visibility.set"), hsm.effect(applyVisibility)),
    hsm.transition(hsm.on("dashboard.visibility.action"), hsm.effect(applyVisibilityAction)),
    hsm.state(
      "idle",
      hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("../live"), hsm.effect(rememberSource)),
      hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../live/replay_paused"), hsm.effect(enterReplay)),
      hsm.transition(hsm.on("dashboard.replay.play"), hsm.target("../live/replay_playing"), hsm.effect(playReplay)),
    ),
    hsm.state(
      "live",
      hsm.initial(hsm.target("viewing")),
      hsm.transition(hsm.on("dashboard.load.completed"), hsm.effect(applySpans)),
      hsm.transition(hsm.on("dashboard.load.failed"), hsm.target("../error"), hsm.effect(applyError)),
      hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("."), hsm.effect(rememberSource)),
      hsm.transition(hsm.on("dashboard.machine.selected"), hsm.effect(applyMachine)),
      hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
      hsm.state(
        "viewing",
        hsm.activity(streamLive),
        hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../replay_paused"), hsm.effect(enterReplay)),
        hsm.transition(hsm.on("dashboard.replay.play"), hsm.target("../replay_playing"), hsm.effect(playReplay)),
      ),
      hsm.state(
        "replay_paused",
        hsm.transition(hsm.on("dashboard.replay.play"), hsm.target("../replay_playing"), hsm.effect(playReplay)),
        hsm.transition(hsm.on("dashboard.replay.enter"), hsm.effect(enterReplay)),
        hsm.transition(hsm.on("dashboard.replay.previous"), hsm.effect(previousReplay)),
        hsm.transition(hsm.on("dashboard.replay.next"), hsm.effect(nextReplay)),
        hsm.transition(hsm.on("dashboard.replay.seek"), hsm.effect(seekReplay)),
        hsm.transition(hsm.on("dashboard.replay.live"), hsm.target("../viewing"), hsm.effect(returnToLive)),
        hsm.transition(hsm.on("dashboard.replay.pause"), hsm.effect(pauseReplay)),
      ),
      hsm.state(
        "replay_playing",
        hsm.transition(hsm.every(replayStep), hsm.effect(nextReplay)),
        hsm.transition(hsm.on("dashboard.replay.pause"), hsm.target("../replay_paused"), hsm.effect(pauseReplay)),
        hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../replay_paused"), hsm.effect(enterReplay)),
        hsm.transition(hsm.on("dashboard.replay.previous"), hsm.effect(previousReplay)),
        hsm.transition(hsm.on("dashboard.replay.next"), hsm.effect(nextReplay)),
        hsm.transition(hsm.on("dashboard.replay.seek"), hsm.effect(seekReplay)),
        hsm.transition(hsm.on("dashboard.replay.live"), hsm.target("../viewing"), hsm.effect(returnToLive)),
      ),
    ),
    hsm.state(
      "error",
      hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("../live"), hsm.effect(rememberSource)),
      hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
      hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("../live"), hsm.effect(enterReplay)),
    ),
  ),
);

function phaseFromStatePath(statePath: string): DashboardPhase {
  if (statePath.includes("/live")) {
    return "live";
  }
  if (statePath.includes("/error")) {
    return "error";
  }
  return "idle";
}

export function isDashboardEventName(value: string): value is DashboardEventName {
  return Object.hasOwn(dashboardEvents, value);
}

export class Dashboard extends hsm.from(HTMLElement) {
  #source: OtelSource | null = null;
  #spans: ObserveSpan[] = [];
  #replayEvents: ReplayEvent[] = [];
  #replayPosition = 0;
  #replaying = false;
  #models: PublishedModel[] = [];
  #skipped = 0;
  #document: OtelDocument | null = null;
  #errorMessage: string | null = null;
  #commandEventName = "";
  #commandDataJson = "";
  #commandResult: CommandResult | null = null;
  #visibleMachines = new Map<string, boolean>();
  #queuedSend: Promise<void> | null = null;
  onSnapshot: ((snapshot: DashboardSnapshot) => void) | null = null;
  connectStream: OtelStreamConnect = connectOtelStream;
  postCommand: CommandPost = postCommandHttp;

  constructor() {
    super();
  }

  boot(): void {
    hsm.start(this, dashboardModel);
  }

  snapshot(): DashboardSnapshot {
    const statePath = this.takeSnapshot().state;
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
        active: statePath.includes("/replay_"),
        playing: statePath.endsWith("/replay_playing"),
        position: this.#replayPosition,
        total: this.#replayEvents.length,
        current: statePath.includes("/replay_") ? this.#replayEvents[this.#replayPosition - 1]?.span ?? null : null,
      },
      visibleMachines: Object.fromEntries(this.#visibleMachines),
    };
  }

  override dispatch(eventName: DashboardEventName, data?: unknown): Promise<DashboardSnapshot>;
  override dispatch(event: hsm.Event): hsm.Completion;
  override dispatch(ctx: hsm.Context, event: hsm.Event): hsm.Completion;
  override dispatch(eventOrContext: DashboardEventName | hsm.Event | hsm.Context, data?: unknown): hsm.Completion | Promise<DashboardSnapshot> {
    if (typeof eventOrContext !== "string") {
      return eventOrContext instanceof hsm.Context
        ? super.dispatch(eventOrContext, data as hsm.Event)
        : super.dispatch(eventOrContext);
    }
    return this.#dispatchController(eventOrContext, data);
  }

  async #dispatchController(eventName: DashboardEventName, data?: unknown): Promise<DashboardSnapshot> {
    await super.dispatch(hsm.namedEvent(dashboardEvents[eventName].name, data));
    if (this.#queuedSend !== null) {
      await this.#queuedSend;
      this.#queuedSend = null;
    }
    this.#emit();
    return this.snapshot();
  }

  queueSend(eventName: string | null, dataJson: string | null): void {
    this.#queuedSend = this.sendCommand(eventName, dataJson);
  }

  override async stop(): Promise<void> {
    await hsm.stop(this);
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
    this.#source = null;
    this.#spans = [];
    this.#replayEvents = [];
    this.#replayPosition = 0;
    this.#replaying = false;
    this.#models = [];
    this.#skipped = 0;
    this.#document = null;
    this.#errorMessage = null;
    this.#commandEventName = "";
    this.#commandDataJson = "";
    this.#commandResult = null;
    this.#visibleMachines.clear();
    this.#emit();
  }

  enterReplay(): void {
    this.#replaying = true;
    this.#replayEvents = replayEvents(this.#spans);
    this.#replayPosition = 0;
    this.#rebuildDocument();
    this.#emit();
  }

  playReplay(): void {
    if (this.#replayEvents.length === 0) {
      this.enterReplay();
    }
    if (this.#replayEvents.length === 0 || this.#replayPosition >= this.#replayEvents.length) {
      return;
    }
    this.#emit();
  }

  pauseReplay(): void {
    this.#emit();
  }

  previousReplay(): void {
    this.#replayPosition = clampReplayPosition(this.#replayPosition - 1, this.#replayEvents.length);
    this.#rebuildDocument();
    this.#emit();
  }

  nextReplay(): void {
    this.#replayPosition = clampReplayPosition(this.#replayPosition + 1, this.#replayEvents.length);
    this.#rebuildDocument();
    this.#emit();
  }

  seekReplay(position: number): void {
    this.#replayPosition = clampReplayPosition(position, this.#replayEvents.length);
    this.#rebuildDocument();
    this.#emit();
  }

  returnToLive(): void {
    this.#replaying = false;
    this.#replayPosition = this.#replayEvents.length;
    this.#rebuildDocument();
    this.#emit();
  }

  applyVisibility(machineName: string, visible: boolean): void {
    const document = this.#document;
    const names = document === null
      ? [machineName]
      : [...machineNamesInOwnedSubtree(environmentWorkspaceGraphs(document.machines), machineName)];
    if (names.length === 0) names.push(machineName);
    for (const name of names) this.#visibleMachines.set(name, visible);
    this.#emit();
  }

  applyVisibilityAction(action: "show-all" | "hide-all" | "hide-unobserved"): void {
    const document = this.#document;
    if (document === null) return;
    for (const machine of document.machines) {
      const visible = action === "show-all" || (action === "hide-unobserved" && machine.observationCount > 0);
      this.#visibleMachines.set(machine.name, visible);
    }
    this.#emit();
  }

  applyPrefill(eventName: string, dataJson: string): void {
    this.#commandEventName = eventName;
    this.#commandDataJson = dataJson;
    this.#emit();
  }

  async sendCommand(eventName: string | null, dataJson: string | null): Promise<void> {
    const name = (eventName ?? this.#commandEventName).trim();
    const payload = dataJson ?? this.#commandDataJson;
    this.#commandEventName = name;
    this.#commandDataJson = payload;
    if (name.length === 0) {
      this.#commandResult = { result: "error", detail: "event_name is required" };
      this.#emit();
      return;
    }
    if (!COMMAND_EVENT_NAME.test(name)) {
      this.#commandResult = { result: "error", detail: "event_name is not an allowed command" };
      this.#emit();
      return;
    }
    this.#commandResult = await this.postCommand({ eventName: name, dataJson: payload });
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
    const subscription = this.connectStream(source.url, {
      onSnapshot: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.completed", { ...batch, mode: "replace" }).catch(hsm.catchFailure(this));
      },
      onSpans: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.completed", { ...batch, mode: "append" }).catch(hsm.catchFailure(this));
      },
      onModels: (models) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.model.published", { models }).catch(hsm.catchFailure(this));
      },
      onError: (message) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch("dashboard.load.failed", { message }).catch(hsm.catchFailure(this));
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
    const replaying = this.#replaying;
    const spans = replaying
      ? replayPrefix(this.#spans, this.#replayEvents, this.#replayPosition)
      : this.#spans;
    const currentMachine = this.#replayEvents[this.#replayPosition - 1]?.span.attributes["hsm.machine.name"] ?? null;
    this.#document = documentFromSpans(
      spans,
      replaying ? 0 : this.#skipped,
      replaying ? currentMachine : previous,
      this.#models,
    );
    if (this.#document !== null) {
      for (const machine of this.#document.machines) {
        if (!this.#visibleMachines.has(machine.name)) this.#visibleMachines.set(machine.name, true);
      }
    }
  }

  #emit(): void {
    this.onSnapshot?.(this.snapshot());
  }
}
