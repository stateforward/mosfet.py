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
import { collectorUrl } from "./otel-source.ts";
import { type ObserveSpan } from "./otel/span.ts";
import { clampReplayPosition, replayEvents, replayPrefix, type ReplayEvent } from "./otel/replay.ts";

const dashboardCommands = {
  "dashboard.source.selected": { name: "dashboard.source.selected", kind: hsm.Kinds.Event },
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
  "dashboard.visibility.set": { name: "dashboard.visibility.set", kind: hsm.Kinds.Event },
  "dashboard.visibility.action": { name: "dashboard.visibility.action", kind: hsm.Kinds.Event },
} as const;

const dashboardCompletions = {
  "dashboard.load.completed": { name: "dashboard.load.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.load.failed": { name: "dashboard.load.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.command.completed": { name: "dashboard.command.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.command.failed": { name: "dashboard.command.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.command.canceled": { name: "dashboard.command.canceled", kind: hsm.Kinds.CompletionEvent },
} as const;

export type CommandResult = {
  readonly result: "accepted" | "no_subscriber" | "error" | "canceled";
  readonly detail: string;
};

export type CommandPost = (command: {
  eventName: string;
  dataJson: string;
  signal?: AbortSignal;
}) => Promise<CommandResult>;

export type DashboardEventName = keyof typeof dashboardCommands;

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

function stringField(args: { event: hsm.Event; key: string }): string | null {
  if (!hsm.isRecord(args.event.data)) {
    return null;
  }
  const value = args.event.data[args.key];
  return typeof value === "string" ? value : null;
}

function replayPositionFromEvent(event: hsm.Event): number | null {
  if (!hsm.isRecord(event.data) || typeof event.data["position"] !== "number") {
    return null;
  }
  return event.data["position"];
}

/**
 * POST `/v1/commands`. AbortError, an aborted signal, and gateway
 * `result: "canceled"` are `canceled` (the HTTP side-effect did not commit).
 * A parsed `accepted` / `no_subscriber` / `error` / `canceled` payload is
 * returned as-is even if the signal later aborts (the gateway result stands).
 */
export async function postCommandHttp(command: {
  eventName: string;
  dataJson: string;
  signal?: AbortSignal;
}): Promise<CommandResult> {
  if (!COMMAND_EVENT_NAME.test(command.eventName)) {
    return { result: "error", detail: "event_name is not an allowed command" };
  }
  try {
    const response = await fetch("/v1/commands", {
      method: "POST",
      headers: { "content-type": "application/json", "x-requested-with": "bot-dashboard" },
      body: JSON.stringify({ event_name: command.eventName, data_json: command.dataJson }),
      ...(command.signal !== undefined ? { signal: command.signal } : {}),
    });
    const payload: unknown = await response.json();
    if (!hsm.isRecord(payload) || typeof payload["result"] !== "string" || typeof payload["detail"] !== "string") {
      return { result: "error", detail: "invalid command reply" };
    }
    if (
      payload["result"] === "accepted"
      || payload["result"] === "no_subscriber"
      || payload["result"] === "error"
      || payload["result"] === "canceled"
    ) {
      return { result: payload["result"], detail: payload["detail"] };
    }
    return { result: "error", detail: payload["detail"] };
  } catch (error) {
    if (command.signal?.aborted === true || (error instanceof Error && error.name === "AbortError")) {
      return { result: "canceled", detail: "command canceled" };
    }
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

function applySpansLive(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  applySpansAt({ instance, event, replay: false });
}

function applySpansReplay(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  applySpansAt({ instance, event, replay: true });
}

function applySpansAt(args: { instance: hsm.Instance; event: hsm.Event; replay: boolean }): void {
  const controller = controllerOf(args.instance);
  const batch = batchFromEvent(args.event);
  if (controller === null || batch === null) {
    return;
  }
  controller.applySpans({
    spans: batch.observeSpans,
    skipped: batch.skipped,
    mode: batch.mode,
    replay: args.replay,
  });
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
  const eventName = stringField({ event, key: "eventName" });
  if (eventName === null) {
    return;
  }
  controllerOf(instance)?.applyPrefill(eventName, stringField({ event, key: "dataJson" }) ?? "");
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

function applyModelsLive(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  applyModelsAt({ instance, event, replay: false });
}

function applyModelsReplay(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  applyModelsAt({ instance, event, replay: true });
}

function applyModelsAt(args: { instance: hsm.Instance; event: hsm.Event; replay: boolean }): void {
  const models = modelsFromEvent(args.event);
  if (models === null) {
    return;
  }
  controllerOf(args.instance)?.applyModels({ models, replay: args.replay });
}

function forwardCommand(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.forwardCommand(event);
}

function attachCommand(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.attachCommand();
}

function applyCommandCompleted(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.rememberCommand(commandResultFromEvent(event) ?? { result: "error", detail: "invalid command reply" });
}

function applyCommandFailed(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  controllerOf(instance)?.rememberCommand(commandResultFromEvent(event) ?? { result: "error", detail: "command request failed" });
}

function applyCommandCanceled(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const result = commandResultFromEvent(event);
  const id = result?.id ?? (hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : undefined);
  controllerOf(instance)?.rememberCommand({
    result: "canceled",
    detail: result?.detail ?? "command canceled",
    ...(id !== undefined ? { id } : {}),
  });
}

function commandResultFromEvent(event: hsm.Event): (CommandResult & { id?: number }) | null {
  if (!hsm.isRecord(event.data) || typeof event.data["detail"] !== "string") return null;
  const result = event.data["result"];
  if (result !== "accepted" && result !== "no_subscriber" && result !== "error" && result !== "canceled") return null;
  const id = event.data["id"];
  return {
    result,
    detail: event.data["detail"],
    ...(typeof id === "number" ? { id } : {}),
  };
}

function applyVisibility(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  if (!hsm.isRecord(event.data)) return;
  const machineName = event.data["machineName"];
  const visible = event.data["visible"];
  if (typeof machineName !== "string" || typeof visible !== "boolean") return;
  controllerOf(instance)?.applyVisibility({ machineName, visible });
}

function applyVisibilityAction(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const action = stringField({ event, key: "action" });
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
    controllerOf(instance)?.seekReplay({ position });
  }
}

function returnToLive(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.returnToLive();
}

async function streamLive(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
  const controller = controllerOf(instance);
  if (controller === null) {
    return;
  }
  await controller.streamSource({ ctx, event });
}

function hasPlayableReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): boolean {
  return controllerOf(instance)?.hasPlayableReplay() === true;
}

function streamUrlAllowed(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
  const controller = controllerOf(instance);
  if (controller === null) return false;
  const source = sourceFromEvent(event) ?? controller.selectedSource();
  if (source === null || source.kind !== "stream") return false;
  return collectorUrl({ requested: source.url, origin: controller.origin }) !== null;
}

function streamUrlDisallowed(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
  const controller = controllerOf(instance);
  if (controller === null) return false;
  const source = sourceFromEvent(event) ?? controller.selectedSource();
  if (source === null || source.kind !== "stream") return false;
  return collectorUrl({ requested: source.url, origin: controller.origin }) === null;
}

function failNoStream(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.applyError("no otel stream selected");
}

function failCollectorUrl(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.applyError("collector url is not allowed");
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
    hsm.entry(attachCommand),
    hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
    hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(forwardCommand)),
    hsm.transition(hsm.on("dashboard.command.completed"), hsm.effect(applyCommandCompleted)),
    hsm.transition(hsm.on("dashboard.command.failed"), hsm.effect(applyCommandFailed)),
    hsm.transition(hsm.on("dashboard.command.canceled"), hsm.effect(applyCommandCanceled)),
    hsm.transition(hsm.on("dashboard.visibility.set"), hsm.effect(applyVisibility)),
    hsm.transition(hsm.on("dashboard.visibility.action"), hsm.effect(applyVisibilityAction)),
    hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("live/sourceCheck"), hsm.effect(rememberSource)),
    hsm.transition(hsm.on("dashboard.replay.enter"), hsm.target("live/replay/paused"), hsm.effect(enterReplay)),
    hsm.state(
      "idle",
      hsm.transition(
        hsm.on("dashboard.replay.play"),
        hsm.guard(hasPlayableReplay),
        hsm.target("../live/replay/playing"),
        hsm.effect(playReplay),
      ),
    ),
    hsm.state(
      "live",
      hsm.initial(hsm.target("sourceCheck")),
      hsm.transition(hsm.on("dashboard.load.failed"), hsm.target("../error"), hsm.effect(applyError)),
      hsm.transition(hsm.on("dashboard.machine.selected"), hsm.effect(applyMachine)),
      hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
      hsm.choice(
        "sourceCheck",
        hsm.transition(hsm.guard(streamUrlAllowed), hsm.target("viewing")),
        hsm.transition(hsm.guard(streamUrlDisallowed), hsm.target("../error"), hsm.effect(failCollectorUrl)),
        hsm.transition(hsm.target("../error"), hsm.effect(failNoStream)),
      ),
      hsm.state(
        "viewing",
        hsm.activity(streamLive),
        hsm.transition(hsm.on("dashboard.load.completed"), hsm.effect(applySpansLive)),
        hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModelsLive)),
        hsm.transition(
          hsm.on("dashboard.replay.play"),
          hsm.guard(hasPlayableReplay),
          hsm.target("../replay/playing"),
          hsm.effect(playReplay),
        ),
      ),
      hsm.state(
        "replay",
        hsm.initial(hsm.target("paused")),
        hsm.transition(hsm.on("dashboard.load.completed"), hsm.effect(applySpansReplay)),
        hsm.transition(hsm.on("dashboard.model.published"), hsm.effect(applyModelsReplay)),
        hsm.transition(hsm.on("dashboard.replay.previous"), hsm.effect(previousReplay)),
        hsm.transition(hsm.on("dashboard.replay.next"), hsm.effect(nextReplay)),
        hsm.transition(hsm.on("dashboard.replay.seek"), hsm.effect(seekReplay)),
        hsm.transition(hsm.on("dashboard.replay.live"), hsm.target("../sourceCheck"), hsm.effect(returnToLive)),
        hsm.state(
          "paused",
          hsm.transition(
            hsm.on("dashboard.replay.play"),
            hsm.guard(hasPlayableReplay),
            hsm.target("../playing"),
            hsm.effect(playReplay),
          ),
          hsm.transition(hsm.on("dashboard.replay.pause"), hsm.effect(pauseReplay)),
        ),
        hsm.state(
          "playing",
          hsm.transition(hsm.every(replayStep), hsm.effect(nextReplay)),
          hsm.transition(hsm.on("dashboard.replay.pause"), hsm.target("../paused"), hsm.effect(pauseReplay)),
        ),
      ),
    ),
    hsm.state(
      "error",
      hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
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
  return Object.hasOwn(dashboardCommands, value);
}

function commandNameLegal(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  if (!hsm.isRecord(event.data) || typeof event.data["eventName"] !== "string") return false;
  const name = event.data["eventName"].trim();
  return name.length > 0 && COMMAND_EVENT_NAME.test(name);
}

class Command extends hsm.Instance {
  static readonly sendEvent = { name: "command.send", kind: hsm.Kinds.Event } as const;
  static readonly completedEvent = dashboardCompletions["dashboard.command.completed"];
  static readonly failedEvent = dashboardCompletions["dashboard.command.failed"];
  static readonly canceledEvent = dashboardCompletions["dashboard.command.canceled"];

  static readonly model = hsm.define(
    "Command",
    hsm.initial(hsm.target("idle")),
    hsm.transition(hsm.on(Command.sendEvent.name), hsm.target("validate"), hsm.effect(Command.claim)),
    hsm.choice(
      "validate",
      hsm.transition(hsm.guard(commandNameLegal), hsm.target("sending")),
      hsm.transition(hsm.target("idle"), hsm.effect(Command.reject)),
    ),
    hsm.state("idle"),
    hsm.state(
      "sending",
      hsm.activity(Command.post),
      hsm.transition(hsm.on(Command.completedEvent.name), hsm.guard(Command.matchesActive), hsm.target("../idle")),
      hsm.transition(hsm.on(Command.failedEvent.name), hsm.guard(Command.matchesActive), hsm.target("../idle")),
      hsm.transition(hsm.on(Command.canceledEvent.name), hsm.guard(Command.matchesActive), hsm.target("../idle")),
    ),
  );

  readonly post: CommandPost;
  #activeId: number | null = null;

  constructor(args: { post: CommandPost }) {
    super();
    this.post = args.post;
  }

  static claim(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Command)) return;
    instance.#activeId = hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : null;
  }

  static matchesActive(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): boolean {
    if (!(instance instanceof Command)) return false;
    const id = hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : null;
    return id !== null && id === instance.#activeId;
  }

  static reject(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Command)) return;
    const name = stringField({ event, key: "eventName" }) ?? "";
    const id = hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : undefined;
    const detail = name.trim().length === 0 ? "event_name is required" : "event_name is not an allowed command";
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({
        event: Command.failedEvent,
        data: { result: "error", detail, ...(id !== undefined ? { id } : {}) },
      }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  /**
   * Failure effect: abort before `instance.post` resolves means the HTTP
   * command did not commit (`canceled`). Once `post` resolves, that result is
   * committed — accepted/no_subscriber complete, error fails, canceled
   * cancels — and a later ctx.done/abort does not relabel or roll back.
   */
  static async post(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
    if (!(instance instanceof Command)) return;
    const name = (stringField({ event, key: "eventName" }) ?? "").trim();
    const payload = stringField({ event, key: "dataJson" }) ?? "";
    const id = hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : undefined;
    const abort = new AbortController();
    const onDone = (): void => {
      abort.abort();
    };
    ctx.addEventListener("done", onDone);
    if (ctx.done) abort.abort();
    const finish = async (args: {
      event: typeof Command.completedEvent | typeof Command.failedEvent | typeof Command.canceledEvent;
      result: CommandResult;
    }): Promise<void> => {
      await hsm.notifyOwner({
        instance,
        event: hsm.typedEvent({
          event: args.event,
          data: { ...args.result, ...(id !== undefined ? { id } : {}) },
        }),
      });
    };
    const canceled: CommandResult = { result: "canceled", detail: "command canceled" };
    let posted: CommandResult | undefined;
    try {
      if (ctx.done || abort.signal.aborted) {
        await finish({ event: Command.canceledEvent, result: canceled });
        return;
      }
      posted = await instance.post({ eventName: name, dataJson: payload, signal: abort.signal });
    } catch (error) {
      if (ctx.done || abort.signal.aborted || (error instanceof Error && error.name === "AbortError")) {
        await finish({ event: Command.canceledEvent, result: canceled });
        return;
      }
      await finish({ event: Command.failedEvent, result: { result: "error", detail: "command request failed" } });
      return;
    } finally {
      ctx.removeEventListener("done", onDone);
    }
    if (posted === undefined) return;
    if (posted.result === "accepted" || posted.result === "no_subscriber") {
      await finish({ event: Command.completedEvent, result: posted });
      return;
    }
    if (posted.result === "error") {
      await finish({ event: Command.failedEvent, result: posted });
      return;
    }
    await finish({ event: Command.canceledEvent, result: posted });
  }
}

export class Dashboard extends hsm.from(HTMLElement) {
  #source: OtelSource | null = null;
  #spans: ObserveSpan[] = [];
  #replayEvents: ReplayEvent[] = [];
  #replayPosition = 0;
  #models: PublishedModel[] = [];
  #skipped = 0;
  #document: OtelDocument | null = null;
  #errorMessage: string | null = null;
  #commandEventName = "";
  #commandDataJson = "";
  #commandResult: CommandResult | null = null;
  #visibleMachines = new Map<string, boolean>();
  origin = "";
  #commandSeq = 0;
  #command: Command | null = null;
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
        active: statePath.includes("/replay"),
        playing: statePath.endsWith("/playing") && statePath.includes("/replay"),
        position: this.#replayPosition,
        total: this.#replayEvents.length,
        current: statePath.includes("/replay") ? this.#replayEvents[this.#replayPosition - 1]?.span ?? null : null,
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
    await super.dispatch(
      data === undefined
        ? hsm.typedEvent({ event: dashboardCommands[eventName] })
        : hsm.typedEvent({ event: dashboardCommands[eventName], data }),
    );
    this.#emit();
    return this.snapshot();
  }

  override async stop(): Promise<void> {
    const command = this.#command;
    this.#command = null;
    if (command !== null) await hsm.stop(command);
    await hsm.stop(this);
  }

  attachCommand(): void {
    if (this.#command !== null) return;
    this.#command = hsm.start(this.context(), new Command({ post: (command) => this.postCommand(command) }), Command.model);
  }

  selectedSource(): OtelSource | null {
    return this.#source;
  }

  hasPlayableReplay(): boolean {
    return this.#replayEvents.length > 0 && this.#replayPosition < this.#replayEvents.length;
  }

  forwardCommand(event: hsm.Event): void {
    const command = this.#command;
    if (command === null) return;
    const name = (stringField({ event, key: "eventName" }) ?? this.#commandEventName).trim();
    const payload = stringField({ event, key: "dataJson" }) ?? this.#commandDataJson;
    this.#commandEventName = name;
    this.#commandDataJson = payload;
    const id = this.#commandSeq + 1;
    this.#commandSeq = id;
    void command.dispatch(hsm.typedEvent({
      event: Command.sendEvent,
      data: { eventName: name, dataJson: payload, id },
    })).catch(hsm.catchFailure(this));
  }

  applySource(source: OtelSource): void {
    this.#source = source;
    this.#errorMessage = null;
    this.#emit();
  }

  applySpans(args: {
    spans: readonly ObserveSpan[];
    skipped: number;
    mode: "replace" | "append";
    replay: boolean;
  }): void {
    if (args.mode === "replace") {
      this.#spans = [...args.spans];
      this.#skipped = args.skipped;
    } else {
      this.#spans = [...this.#spans, ...args.spans];
      this.#skipped += args.skipped;
    }
    this.#replayEvents = replayEvents(this.#spans);
    this.#replayPosition = clampReplayPosition(this.#replayPosition, this.#replayEvents.length);
    this.#rebuildDocument({ replay: args.replay });
    this.#errorMessage = null;
    this.#emit();
  }

  applyModels(args: { models: readonly PublishedModel[]; replay: boolean }): void {
    const byName = new Map(this.#models.map((model) => [model.name, model]));
    for (const model of args.models) {
      byName.set(model.name, mergePublishedModel(byName.get(model.name), model));
    }
    this.#models = [...byName.values()];
    this.#rebuildDocument({ replay: args.replay });
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
    this.#replayEvents = replayEvents(this.#spans);
    this.#replayPosition = 0;
    this.#rebuildDocument({ replay: true });
    this.#emit();
  }

  playReplay(): void {
    this.#emit();
  }

  pauseReplay(): void {
    this.#emit();
  }

  previousReplay(): void {
    this.#replayPosition = clampReplayPosition(this.#replayPosition - 1, this.#replayEvents.length);
    this.#rebuildDocument({ replay: true });
    this.#emit();
  }

  nextReplay(): void {
    this.#replayPosition = clampReplayPosition(this.#replayPosition + 1, this.#replayEvents.length);
    this.#rebuildDocument({ replay: true });
    this.#emit();
  }

  seekReplay(args: { position: number }): void {
    this.#replayPosition = clampReplayPosition(args.position, this.#replayEvents.length);
    this.#rebuildDocument({ replay: true });
    this.#emit();
  }

  returnToLive(): void {
    this.#replayPosition = this.#replayEvents.length;
    this.#rebuildDocument({ replay: false });
    this.#emit();
  }

  rememberCommand(result: CommandResult & { id?: number }): void {
    if (typeof result.id === "number" && result.id !== this.#commandSeq) return;
    this.#commandResult = { result: result.result, detail: result.detail };
    this.#emit();
  }

  applyVisibility(args: { machineName: string; visible: boolean }): void {
    const document = this.#document;
    const names = document === null
      ? [args.machineName]
      : [...machineNamesInOwnedSubtree(environmentWorkspaceGraphs(document.machines), args.machineName)];
    if (names.length === 0) names.push(args.machineName);
    for (const name of names) this.#visibleMachines.set(name, args.visible);
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

  async streamSource(args: { ctx: hsm.Context; event: hsm.Event }): Promise<void> {
    const source = sourceFromEvent(args.event) ?? this.#source;
    if (source === null || source.kind !== "stream") {
      await this.dispatch(hsm.typedEvent({
        event: dashboardCompletions["dashboard.load.failed"],
        data: { message: "no otel stream selected" },
      })).catch(hsm.catchFailure(this));
      return;
    }
    const url = collectorUrl({ requested: source.url, origin: this.origin });
    if (url === null) {
      await this.dispatch(hsm.typedEvent({
        event: dashboardCompletions["dashboard.load.failed"],
        data: { message: "collector url is not allowed" },
      })).catch(hsm.catchFailure(this));
      return;
    }
    const ctx = args.ctx;
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
    const subscription = this.connectStream(url, {
      onSnapshot: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.load.completed"], data: {
          ...batch,
          mode: "replace",
        } })).catch(hsm.catchFailure(this));
      },
      onSpans: (batch) => {
        if (ctx.done) {
          return;
        }
        void this.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.load.completed"], data: {
          ...batch,
          mode: "append",
        } })).catch(hsm.catchFailure(this));
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
        void this.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.load.failed"], data: {
          message,
        } })).catch(hsm.catchFailure(this));
      },
    });
    try {
      await finished;
    } finally {
      subscription.close();
    }
  }

  #rebuildDocument(args: { replay: boolean }): void {
    const previous = this.#document?.selectedMachine ?? null;
    const replaying = args.replay;
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


