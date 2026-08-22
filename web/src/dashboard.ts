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
import {
  collectorUrl,
  eventWithSourceConnect,
  isSourceConnectPayload,
  sourceConnectFrom,
  type SourceConnectData,
} from "./otel-source.ts";
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
  "dashboard.graph.focus": { name: "dashboard.graph.focus", kind: hsm.Kinds.Event },
  "dashboard.host.attach": { name: "dashboard.host.attach", kind: hsm.Kinds.Event },
  "dashboard.host.detach": { name: "dashboard.host.detach", kind: hsm.Kinds.Event },
} as const;

const dashboardCompletions = {
  "dashboard.load.completed": { name: "dashboard.load.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.load.failed": { name: "dashboard.load.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.stream.dropped": { name: "dashboard.stream.dropped", kind: hsm.Kinds.CompletionEvent },
  "dashboard.command.completed": { name: "dashboard.command.completed", kind: hsm.Kinds.CompletionEvent },
  "dashboard.command.failed": { name: "dashboard.command.failed", kind: hsm.Kinds.ErrorEvent },
  "dashboard.command.canceled": { name: "dashboard.command.canceled", kind: hsm.Kinds.CompletionEvent },
  "dashboard.host.stopped": { name: "dashboard.host.stopped", kind: hsm.Kinds.CompletionEvent },
} as const;

export type CommandResult = {
  readonly result: "accepted" | "no_subscriber" | "error" | "canceled";
  readonly detail: string;
};

/**
 * HTTP command adapter.
 *
 * Inputs: `eventName`, `dataJson`, optional `signal`.
 * Outputs: a `CommandResult`. A resolved result is final even if `signal` later
 * aborts.
 * Ownership: caller owns `signal` and the returned promise; this type does not
 * retain the command.
 * Lifetime: one HTTP round-trip; settling the promise ends the call.
 * Concurrency: overlapping posts are independent; the dashboard sequences them
 * with `id`.
 * Failure modes:
 * - abort observed before `fetch` is invoked => `canceled` / unsent
 * - abort after `fetch` is invoked => `error` / "command reply interrupted"
 *   (unknown commit); never relabel a sent POST as unsent `canceled`
 * - illegal `eventName` => `error`
 * Units: none.
 * Classification: external-system.
 */
export type CommandPost = (command: {
  eventName: string;
  dataJson: string;
  signal?: AbortSignal;
}) => Promise<CommandResult>;

export type VisibilityAction = "show-all" | "hide-all" | "hide-unobserved";

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

const COMMAND_NAME_MAX = 128;
const COMMAND_NAME_FIRST_INDEX = 0;
const LETTER_A = 65;
const LETTER_Z = 90;
const LETTER_a = 97;
const LETTER_z = 122;
const DIGIT_0 = 48;
const DIGIT_9 = 57;
const CHAR_UNDERSCORE = 95;
const CHAR_DOT = 46;
const CHAR_COLON = 58;
const CHAR_SLASH = 47;
const CHAR_DASH = 45;
const EVENT_NAME_REQUIRED = "event_name is required";
const EVENT_NAME_NOT_ALLOWED = "event_name is not an allowed command";

/**
 * Allocation-free command event-name charset: `A-Za-z` then up to 127 of
 * `A-Za-z0-9_.:/-`. Does not trim. Empty is distinct from whitespace/illegal.
 */
export function commandEventNameLegal(name: string): boolean {
  if (name.length === 0 || name.length > COMMAND_NAME_MAX) return false;
  const first = name.charCodeAt(COMMAND_NAME_FIRST_INDEX);
  if (!((first >= LETTER_A && first <= LETTER_Z) || (first >= LETTER_a && first <= LETTER_z))) return false;
  for (let index = 1; index < name.length; index += 1) {
    const code = name.charCodeAt(index);
    if ((code >= LETTER_A && code <= LETTER_Z) || (code >= LETTER_a && code <= LETTER_z)) continue;
    if (code >= DIGIT_0 && code <= DIGIT_9) continue;
    if (code === CHAR_UNDERSCORE || code === CHAR_DOT || code === CHAR_COLON || code === CHAR_SLASH || code === CHAR_DASH) {
      continue;
    }
    return false;
  }
  return true;
}

function commandNameFailureDetail(name: string): string {
  return name.length === 0 ? EVENT_NAME_REQUIRED : EVENT_NAME_NOT_ALLOWED;
}

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

function signalAborted(signal: AbortSignal | undefined): boolean {
  return signal !== undefined && signal.aborted;
}

/**
 * POST `/v1/commands`.
 *
 * Inputs: `eventName`, `dataJson`, optional `signal`.
 * Outputs: `CommandResult`.
 * Ownership: caller owns `signal`; this function does not retain the command.
 * Lifetime: one HTTP round-trip; the promise settling ends the call.
 * Concurrency: overlapping calls are independent fetches.
 * Failure modes:
 * - empty `eventName` => `error` / "event_name is required"
 * - illegal `eventName` => `error` / "event_name is not an allowed command"
 * - abort observed before `fetch` is invoked => `canceled` / "command canceled"
 *   (HTTP did not commit)
 * - abort after `fetch` is invoked, including AbortError before the response
 *   resolves => `error` / "command reply interrupted" (commit unknown)
 * - `fetch` resolves and JSON is a CommandResult, including gateway `canceled`
 *   => that payload as-is even if `signal` is already aborted (HTTP committed;
 *   gateway `canceled` is a domain cancel, not an unsent request)
 * - `fetch` resolves then `json()` throws AbortError => `error` /
 *   "command reply interrupted" (commit unknown)
 * - other fetch/parse failures => `error` / "command request failed" or
 *   "invalid command reply"
 * Classification: external-system.
 */
export async function postCommandHttp(command: {
  eventName: string;
  dataJson: string;
  signal?: AbortSignal;
}): Promise<CommandResult> {
  if (!commandEventNameLegal(command.eventName)) {
    return { result: "error", detail: commandNameFailureDetail(command.eventName) };
  }
  if (signalAborted(command.signal)) {
    return { result: "canceled", detail: "command canceled" };
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
    const aborted = signalAborted(command.signal) || (error instanceof Error && error.name === "AbortError");
    if (aborted) {
      return { result: "error", detail: "command reply interrupted" };
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
  controllerOf(instance)?.applyPrefill({
    eventName,
    dataJson: stringField({ event, key: "dataJson" }) ?? "",
  });
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
  controllerOf(instance)?.applyVisibilityAction({ action });
}

function clearView(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.clearView();
}

function enterReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.enterReplay();
}

function playReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.playReplay();
  raiseGraphFocus(instance);
}

function pauseReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.pauseReplay();
}

function previousReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.previousReplay();
  raiseGraphFocus(instance);
}

function nextReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.nextReplay();
  raiseGraphFocus(instance);
}

function seekReplay(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const position = replayPositionFromEvent(event);
  if (position !== null) {
    controllerOf(instance)?.seekReplay({ position });
  }
  raiseGraphFocus(instance);
}

function returnToLive(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.returnToLive();
}

function hasGraphFocusName(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  const machineName = machineNameFromEvent(event);
  return machineName !== null && machineName.length > 0;
}

function applyGraphFocus(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const machineName = machineNameFromEvent(event);
  if (machineName === null || machineName.length === 0) return;
  controllerOf(instance)?.applyGraphFocus({ machineName });
}

function raiseGraphFocus(instance: hsm.Instance): void {
  const machineName = controllerOf(instance)?.replayGraphName() ?? "";
  void instance.dispatch(hsm.typedEvent({
    event: dashboardCommands["dashboard.graph.focus"],
    data: { machineName },
  })).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
}

function raiseReplayNext(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  void instance.dispatch(hsm.typedEvent({ event: dashboardCommands["dashboard.replay.next"] })).catch(
    hsm.catchFailure(hsm.ownerTarget(instance)),
  );
}

async function streamLive(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
  const controller = controllerOf(instance);
  if (controller === null) {
    await instance.dispatch({
      ...hsm.ErrorEvent,
      data: { message: "dashboard host missing" },
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
    return;
  }
  await controller.streamSource({ ctx, event });
}

function hasPlayableReplay(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): boolean {
  return controllerOf(instance)?.hasPlayableReplay() === true;
}

function streamUrlAllowed(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  return hsm.isRecord(event.data)
    && event.data["urlAllowed"] === true
    && isOtelSource(event.data["source"]);
}

function streamUrlDisallowed(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
  return hsm.isRecord(event.data)
    && event.data["urlAllowed"] === false
    && isOtelSource(event.data["source"]);
}

function isConnectIngress(data: unknown): boolean {
  return isSourceConnectPayload(data) || (hsm.isRecord(data) && isOtelSource(data["source"]));
}

function failNoStream(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.applyError("no otel stream selected");
}

function failCollectorUrl(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.applyError("collector url is not allowed");
}

function consumeStreamDropped(_ctx: hsm.Context, _instance: hsm.Instance, _event: hsm.Event): void {
  return;
}

type HostDetachData = {
  readonly command: Command | null;
};

function isHostDetachData(value: unknown): value is HostDetachData {
  if (!hsm.isRecord(value)) return false;
  return value["command"] === null || value["command"] instanceof Command;
}

async function stopCommandActor(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
  const command = isHostDetachData(event.data) ? event.data.command : null;
  try {
    if (command !== null) await hsm.stop(command);
  } catch (error) {
    hsm.catchFailure(instance instanceof EventTarget ? instance : undefined)(error);
  } finally {
    try {
      await instance.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.host.stopped"] }));
    } catch (error) {
      hsm.catchFailure(instance instanceof EventTarget ? instance : undefined)(error);
    }
  }
}

function clearCommandActor(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.clearCommandActor();
}

function replayStep(): number {
  return 700;
}

const dashboardModel = hsm.define(
  "Dashboard",
  hsm.initial(hsm.target("connected")),
  hsm.state(
    "disconnected",
    hsm.defer("dashboard.source.selected"),
    hsm.defer("dashboard.command.send"),
    hsm.defer("dashboard.replay.play"),
    hsm.defer("dashboard.replay.enter"),
    hsm.transition(hsm.on("dashboard.host.attach"), hsm.target("../connected")),
  ),
  hsm.state(
    "connected",
    hsm.initial(hsm.target("session")),
    hsm.transition(hsm.on("dashboard.host.detach"), hsm.target("../stopping")),
    hsm.state(
      "session",
      hsm.initial(hsm.target("idle")),
      hsm.entry(attachCommand),
      hsm.exit(clearCommandActor),
      hsm.transition(hsm.on("dashboard.command.prefill"), hsm.effect(applyPrefill)),
      hsm.transition(hsm.on("dashboard.command.send"), hsm.effect(forwardCommand)),
      hsm.transition(hsm.on("dashboard.command.completed"), hsm.effect(applyCommandCompleted)),
      hsm.transition(hsm.on("dashboard.command.failed"), hsm.effect(applyCommandFailed)),
      hsm.transition(hsm.on("dashboard.command.canceled"), hsm.effect(applyCommandCanceled)),
      hsm.transition(hsm.on("dashboard.visibility.set"), hsm.effect(applyVisibility)),
      hsm.transition(hsm.on("dashboard.visibility.action"), hsm.effect(applyVisibilityAction)),
      hsm.transition(
        hsm.on("dashboard.graph.focus"),
        hsm.guard(hasGraphFocusName),
        hsm.effect(applyGraphFocus),
      ),
      hsm.transition(
        hsm.on("dashboard.machine.selected"),
        hsm.guard(hasGraphFocusName),
        hsm.effect(applyMachine, applyGraphFocus),
      ),
      hsm.transition(hsm.on("dashboard.source.selected"), hsm.target("live/sourceCheck"), hsm.effect(rememberSource)),
      hsm.transition(
        hsm.on("dashboard.replay.enter"),
        hsm.target("live/replay/paused"),
        hsm.effect(enterReplay),
      ),
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
        hsm.transition(hsm.on(hsm.ErrorEvent.name), hsm.target("../error"), hsm.effect(applyError)),
        hsm.transition(hsm.on("dashboard.stream.dropped"), hsm.effect(consumeStreamDropped)),
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
          hsm.transition(
            hsm.on("dashboard.replay.play"),
            hsm.guard(hasPlayableReplay),
            hsm.target("playing"),
            hsm.effect(playReplay),
          ),
          hsm.transition(hsm.on("dashboard.replay.pause"), hsm.target("paused"), hsm.effect(pauseReplay)),
          hsm.state("paused"),
          hsm.state(
            "playing",
            hsm.transition(hsm.every(replayStep), hsm.effect(raiseReplayNext)),
          ),
        ),
      ),
      hsm.state(
        "error",
        hsm.transition(hsm.on("dashboard.reset"), hsm.target("../idle"), hsm.effect(clearView)),
      ),
    ),
  ),
  hsm.state(
    "stopping",
    hsm.defer("dashboard.host.attach"),
    hsm.defer("dashboard.host.detach"),
    hsm.defer("dashboard.source.selected"),
    hsm.defer("dashboard.command.send"),
    hsm.activity(stopCommandActor),
    hsm.transition(
      hsm.on("dashboard.host.stopped"),
      hsm.target("../disconnected"),
      hsm.effect(clearCommandActor),
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
  return commandEventNameLegal(event.data["eventName"]);
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
    const detail = commandNameFailureDetail(name);
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({
        event: Command.failedEvent,
        data: { result: "error", detail, ...(id !== undefined ? { id } : {}) },
      }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  /**
   * Failure effect: abort observed before `instance.post` is invoked is unsent
   * `canceled`. `ctx.done` is wired to `CommandPost.signal`. Abort after the
   * adapter is invoked uses the adapter result: unsent `canceled` only when the
   * adapter reports that, otherwise unknown-commit `error` / interrupted.
   * A resolved `CommandResult` is final.
   */
  static async post(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): Promise<void> {
    if (!(instance instanceof Command)) return;
    const name = stringField({ event, key: "eventName" }) ?? "";
    const payload = stringField({ event, key: "dataJson" }) ?? "";
    const id = hsm.isRecord(event.data) && typeof event.data["id"] === "number" ? event.data["id"] : undefined;
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
    if (!commandEventNameLegal(name)) {
      await finish({ event: Command.failedEvent, result: { result: "error", detail: commandNameFailureDetail(name) } });
      return;
    }
    const canceled: CommandResult = { result: "canceled", detail: "command canceled" };
    const interrupted: CommandResult = { result: "error", detail: "command reply interrupted" };
    if (ctx.done) {
      await finish({ event: Command.canceledEvent, result: canceled });
      return;
    }
    const abort = new AbortController();
    const onDone = (): void => {
      abort.abort();
    };
    ctx.addEventListener("done", onDone);
    if (ctx.done) {
      onDone();
      ctx.removeEventListener("done", onDone);
      await finish({ event: Command.canceledEvent, result: canceled });
      return;
    }
    let invoked = false;
    let posted: CommandResult | undefined;
    try {
      invoked = true;
      posted = await instance.post({ eventName: name, dataJson: payload, signal: abort.signal });
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        await finish({
          event: invoked ? Command.failedEvent : Command.canceledEvent,
          result: invoked ? interrupted : canceled,
        });
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
    hsm.start({ instance: this, model: dashboardModel });
  }

  /**
   * Request host attach.
   *
   * Inputs: none. Dispatches `dashboard.host.attach`.
   * Outputs: none directly. Topology moves `disconnected` to `connected` only
   * when the host is already disconnected. Attach is ignored unless
   * disconnected, and deferred while stopping so reconnect cannot start until
   * `dashboard.host.stopped` has cleared the previous Command actor.
   * Ownership: this dashboard owns the dispatch and the Command actor it later
   * attaches under `session`. Lifetime: one attach request; the connected
   * session lasts until detach/stop.
   * Concurrency: runtime-safe. Overlapping attach while connected is ignored
   * (no transition). While stopping, attach is deferred, not dropped.
   * Failure modes: dispatch rejection is classified by `catchFailure` as a
   * host-drop when the runtime is unstarted or stopped; otherwise reported.
   * Units: none.
   * Classification: runtime-safe.
   */
  requestAttach(): void {
    void super.dispatch(hsm.typedEvent({ event: dashboardCommands["dashboard.host.attach"] })).catch(hsm.catchFailure(this));
  }

  /**
   * Request host detach through stopping.
   *
   * Inputs: none. Dispatches `dashboard.host.detach` with typed `HostDetachData`
   * (`command` is the live Command actor or `null`). Detach stamps that actor
   * so `stopCommandActor` stops the payload, not a field lookup.
   * Outputs: none directly. Topology moves `connected` to `stopping`, then
   * `disconnected` on `dashboard.host.stopped`. The host stays bound.
   * Ownership: this dashboard owns the dispatch; the stamped Command actor is
   * stopped by the stopping activity. Session exit and `host.stopped` null
   * `#command`.
   * Lifetime: one detach request; stopping lasts until `dashboard.host.stopped`.
   * Concurrency: runtime-safe. Detach while stopping is deferred. Detach while
   * disconnected is ignored. Public `stop()` is mixin Host.stop, not detach.
   * Failure modes: dispatch rejection is classified by `catchFailure` as a
   * host-drop when the runtime is unstarted or stopped; otherwise reported.
   * Units: none.
   * Classification: runtime-safe.
   */
  requestDetach(): void {
    const data: HostDetachData = { command: this.#command };
    void super.dispatch(hsm.typedEvent({
      event: dashboardCommands["dashboard.host.detach"],
      data,
    })).catch(hsm.catchFailure(this));
  }

  /**
   * Null the Command actor after session exit or `dashboard.host.stopped`.
   *
   * Inputs: none. Invoked from session exit and the `dashboard.host.stopped`
   * effect. Outputs: `#command` is `null`. Ownership: declaring Dashboard
   * class only. Lifetime: until the next `session` entry attaches a new
   * Command actor. Concurrency: runtime-safe.
   * Failure modes: none. Units: none.
   * Classification: runtime-safe.
   */
  clearCommandActor(): void {
    this.#command = null;
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

  /**
   * Admit Event-path and controller payloads. Connect-shaped data (`origin`
   * string, `urlAllowed` boolean, or `source`) is restamped through
   * `sourceConnectFrom` so choice guards read `urlAllowed` and never construct
   * URL objects. Payload-less Event-path live is not restamped by `event.name`;
   * producers stamp typed connect data, or callers use the controller overload
   * `dispatch("dashboard.replay.live")`, which stamps from this dashboard's
   * owned `#source` and `origin`, not from `snapshot()`.
   */
  override dispatch(eventName: DashboardEventName, data?: unknown): Promise<DashboardSnapshot>;
  override dispatch(event: hsm.Event): hsm.Completion;
  override dispatch(ctx: hsm.Context, event: hsm.Event): hsm.Completion;
  override dispatch(eventOrContext: DashboardEventName | hsm.Event | hsm.Context, data?: unknown): hsm.Completion | Promise<DashboardSnapshot> {
    if (typeof eventOrContext !== "string") {
      if (eventOrContext instanceof hsm.Context) {
        if (!hsm.isEvent(data)) {
          throw new TypeError("dispatch(ctx, event) requires an Event");
        }
        return super.dispatch(eventOrContext, this.#admitEvent(data));
      }
      if (!hsm.isEvent(eventOrContext)) {
        throw new TypeError("dispatch(event) requires an Event");
      }
      return super.dispatch(this.#admitEvent(eventOrContext));
    }
    return this.#dispatchController({ eventName: eventOrContext, data });
  }

  async #dispatchController(args: { eventName: DashboardEventName; data?: unknown }): Promise<DashboardSnapshot> {
    const admitted = this.#admitControllerData(args);
    await super.dispatch(
      admitted === undefined
        ? hsm.typedEvent({ event: dashboardCommands[args.eventName] })
        : hsm.typedEvent({ event: dashboardCommands[args.eventName], data: admitted }),
    );
    this.#emit();
    return this.snapshot();
  }

  #admitControllerData(args: { eventName: DashboardEventName; data?: unknown }): unknown {
    const data = args.data;
    if (isConnectIngress(data)) return sourceConnectFrom(data);
    if (args.eventName === "dashboard.replay.live" && data === undefined) {
      return this.ownedSourceConnect();
    }
    return data;
  }

  /**
   * Admit Event-path payloads. Connect-shaped data (`origin` string,
   * `urlAllowed` boolean, or `source`) is restamped through
   * `sourceConnectFrom`. Payload-less Event-path live is not restamped by
   * `event.name`; producers stamp typed connect data, or callers use the
   * controller overload.
   *
   * Inputs: an HSM Event. Outputs: the same event, or a clone whose `data` is
   * recomputed `SourceConnectData` when the payload is connect-shaped.
   * Does not read `event.name`. Ownership: returns a new event object only
   * when restamping. Classification: runtime-safe.
   */
  #admitEvent(event: hsm.Event): hsm.Event {
    if (isConnectIngress(event.data)) {
      return { ...event, data: sourceConnectFrom(event.data) };
    }
    return eventWithSourceConnect(event);
  }

  /**
   * Stamp stream connect data from this dashboard's owned `#source` and
   * `origin`.
   *
   * Inputs: none. Uses owned `#source` (omitted when null) and `origin`.
   * Outputs: `SourceConnectData` with recomputed `urlAllowed`. Missing source
   * yields a stamp without `source`; missing or invalid origin yields
   * `urlAllowed: false`. Ownership: this dashboard owns `#source`; the returned
   * record is a new connect payload. Lifetime: one dispatch admission.
   * Concurrency: runtime-safe. Classification: runtime-safe.
   */
  ownedSourceConnect(): SourceConnectData {
    return sourceConnectFrom({
      origin: this.origin,
      ...(this.#source !== null ? { source: this.#source } : {}),
    });
  }

  attachCommand(): void {
    if (this.#command !== null) return;
    this.#command = hsm.start({
      ctx: this.context(),
      instance: new Command({ post: (command) => this.postCommand(command) }),
      model: Command.model,
    });
  }

  hasPlayableReplay(): boolean {
    return this.#replayEvents.length > 0 && this.#replayPosition < this.#replayEvents.length;
  }

  /**
   * Read stamped stream connect data from the entering event.
   *
   * Inputs: an event whose producer already stamped `SourceConnectData`
   * (`urlAllowed` boolean and `origin` string). Does not fill instance
   * `#source` or `origin`, does not call `sourceConnectFrom` or `collectorUrl`,
   * and does not recompute `urlAllowed`. Outputs: the stamped pairing, or null
   * when the event is not a complete connect stamp. Ownership: returned record
   * aliases the event `source` object when present. Lifetime: one
   * choice/activity evaluation. Concurrency: runtime-safe. Failure modes:
   * missing `urlAllowed` or `origin` returns null. Classification: runtime-safe.
   */
  liveConnectData(event: hsm.Event): SourceConnectData | null {
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

  forwardCommand(event: hsm.Event): void {
    const command = this.#command;
    if (command === null) return;
    const name = stringField({ event, key: "eventName" }) ?? this.#commandEventName;
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

  /**
   * Forward a modeled graph-focus request to a host-owned graph.
   *
   * Inputs: `machineName` from `dashboard.graph.focus` after the non-empty
   * string guard, from `dashboard.machine.selected` which also carries
   * `machineName`, and from replay play/previous/next/seek after those
   * effects dispatch `dashboard.graph.focus` with the cursor `machineName`.
   * Outputs: none on this dashboard. Hosts that own a `BotMachineGraph`
   * dispatch `focus_machine`. Snapshot render must not call this.
   * Ownership: this dashboard. Lifetime: one focus request.
   * Concurrency: runtime-safe on the dashboard dispatch thread.
   * Failure modes: empty or non-string names never reach this method;
   * the `dashboard.graph.focus` and `dashboard.machine.selected` guards drop
   * them, including empty names stamped by replay cursor steps.
   * Classification: runtime-safe.
   */
  applyGraphFocus(_args: { machineName: string }): void {
    return;
  }

  /**
   * Machine name on the current replay cursor event, if any.
   *
   * Inputs: `#replayEvents` and `#replayPosition` after a replay-step effect.
   * Outputs: the `hsm.machine.name` string on the event at `position - 1`,
   * or null when the cursor is at 0 or the attribute is missing or empty.
   * Ownership: this dashboard. Lifetime: until the next replay step or clear.
   * Concurrency: runtime-safe on the dashboard dispatch thread.
   * Failure modes: missing or empty names return null; the replay step then
   * dispatches `dashboard.graph.focus` with an empty name, which
   * `hasGraphFocusName` drops.
   * Classification: runtime-safe.
   */
  replayGraphName(): string | null {
    const name = this.#replayEvents[this.#replayPosition - 1]?.span.attributes["hsm.machine.name"];
    return typeof name === "string" && name.length > 0 ? name : null;
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

  /**
   * Apply one machine's visibility flag and its owned subtree.
   *
   * Inputs: `machineName`, `visible`. Outputs: snapshot `visibleMachines`.
   * Ownership: this dashboard owns the map. Lifetime: until `clearView`/stop.
   * Concurrency: runtime-safe on the dashboard dispatch thread.
   * Failure modes: unknown names are still recorded on the map.
   * Classification: runtime-safe.
   */
  applyVisibility(args: { machineName: string; visible: boolean }): void {
    const document = this.#document;
    const names = document === null
      ? [args.machineName]
      : [...machineNamesInOwnedSubtree(environmentWorkspaceGraphs(document.machines), args.machineName)];
    if (names.length === 0) names.push(args.machineName);
    for (const name of names) this.#visibleMachines.set(name, args.visible);
    this.#emit();
  }

  /**
   * Apply a bulk visibility action to every machine in the current document.
   *
   * Inputs: `action` show-all | hide-all | hide-unobserved.
   * Outputs: snapshot `visibleMachines`. Ownership: this dashboard.
   * Lifetime: until `clearView`/stop. Concurrency: runtime-safe.
   * Failure modes: no document => no-op.
   * Classification: runtime-safe.
   */
  applyVisibilityAction(args: { action: VisibilityAction }): void {
    const document = this.#document;
    if (document === null) return;
    for (const machine of document.machines) {
      const visible = args.action === "show-all" || (args.action === "hide-unobserved" && machine.observationCount > 0);
      this.#visibleMachines.set(machine.name, visible);
    }
    this.#emit();
  }

  /**
   * Prefill the command compose fields.
   *
   * Inputs: `eventName`, `dataJson`. Outputs: snapshot command fields.
   * Ownership: this dashboard. Lifetime: until send/clearView/stop.
   * Concurrency: runtime-safe. Failure modes: none; strings are stored as given.
   * Classification: runtime-safe.
   */
  applyPrefill(args: { eventName: string; dataJson: string }): void {
    this.#commandEventName = args.eventName;
    this.#commandDataJson = args.dataJson;
    this.#emit();
  }

  /**
   * Viewing activity: connect the stream already admitted by `sourceCheck`.
   *
   * Inputs: activity `ctx` and the entering `dashboard.source.selected` /
   * `dashboard.replay.live` event. `source`, `origin`, and `urlAllowed` come
   * from that stamped payload. `urlAllowed` is
   * `collectorUrl({ requested: source.url, origin }) !== null`. This activity
   * opens that same `collectorUrl` result with `connectStream`, not raw
   * `source.url`.
   * `sourceCheck` already selected viewing, so this activity does not dispatch
   * `dashboard.load.failed` for missing source or disallowed URL.
   * Outputs: `dashboard.load.completed` / `dashboard.model.published` products
   * and `dashboard.load.failed` for later stream errors. Leaving viewing
   * (replay.enter, reset, detach, new source) cancels the stream by exiting
   * this activity. Late callbacks after that exit dispatch
   * `dashboard.stream.dropped` on live; they do not apply products or
   * `load.failed`.
   * Ownership: this dashboard owns the subscription and closes it on activity
   * exit. Lifetime: one viewing activity. Concurrency: one stream per viewing;
   * overlapping viewing is prevented by topology. Classification: external-system.
   */
  async streamSource(args: { ctx: hsm.Context; event: hsm.Event }): Promise<void> {
    const connect = this.liveConnectData(args.event);
    if (connect === null || connect.urlAllowed !== true || connect.source === undefined) {
      throw new TypeError("viewing entered without stamped stream connect");
    }
    const url = collectorUrl({ requested: connect.source.url, origin: connect.origin });
    if (url === null) {
      throw new TypeError("stamped urlAllowed is true but collector url is null");
    }
    const ctx = args.ctx;
    let subscription: { close(): void } | null = null;
    const drop = (): void => {
      subscription?.close();
      subscription = null;
    };
    const finished = new Promise<void>((resolve) => {
      const onDone = (): void => {
        ctx.removeEventListener("done", onDone);
        drop();
        resolve();
      };
      ctx.addEventListener("done", onDone);
      if (ctx.done) {
        onDone();
      }
    });
    const dropLate = (): void => {
      void this.dispatch(hsm.typedEvent({
        event: dashboardCompletions["dashboard.stream.dropped"],
      })).catch(hsm.catchFailure(this));
    };
    subscription = this.connectStream(url, {
      onSnapshot: (batch) => {
        if (ctx.done) {
          dropLate();
          return;
        }
        void this.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.load.completed"], data: {
          ...batch,
          mode: "replace",
        } })).catch(hsm.catchFailure(this));
      },
      onSpans: (batch) => {
        if (ctx.done) {
          dropLate();
          return;
        }
        void this.dispatch(hsm.typedEvent({ event: dashboardCompletions["dashboard.load.completed"], data: {
          ...batch,
          mode: "append",
        } })).catch(hsm.catchFailure(this));
      },
      onModels: (models) => {
        if (ctx.done) {
          dropLate();
          return;
        }
        void this.dispatch("dashboard.model.published", { models }).catch(hsm.catchFailure(this));
      },
      onError: (message) => {
        if (ctx.done) {
          dropLate();
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
      drop();
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


