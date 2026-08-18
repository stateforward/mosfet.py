import {
  DashboardController,
  isDashboardEventName,
  type DashboardSnapshot,
} from "../dashboard-hsm.ts";
import { type MachineGraph } from "../otel/machines.ts";
import { isOtelSource } from "../otel/source.ts";
import { BotMachineGraph } from "./bot-machine-graph.ts";
import { BotOtelSource, type OtelSourceDetail } from "./bot-otel-source.ts";
import { applyStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-dashboard";

const cssText = `
:host {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  height: 100dvh;
  min-height: 100dvh;
  background: var(--bot-bg, #0b0d12);
  color: var(--bot-ink, #e8eaef);
}
.topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem 0.75rem;
  min-height: 2.5rem;
  padding: 0.35rem 0.7rem;
  border-bottom: 1px solid var(--bot-line, #2a3140);
  background: var(--bot-panel, #12141a);
}
.title {
  margin: 0;
  font-size: 0.82rem;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.studio {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: 0;
  height: 100%;
}
.inspector {
  display: grid;
  align-content: start;
  gap: 0.85rem;
  padding: 0.85rem 0.8rem 1rem;
  border-right: 1px solid var(--bot-line, #2a3140);
  background: var(--bot-panel, #12141a);
  overflow: auto;
}
.canvas {
  min-width: 0;
  min-height: 16rem;
  height: 100%;
}
bot-machine-graph {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 16rem;
}
.section-label {
  margin: 0 0 0.35rem;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.68rem;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.machine-list {
  display: grid;
  gap: 0.25rem;
}
.machine {
  display: grid;
  gap: 0.1rem;
  width: 100%;
  margin: 0;
  padding: 0.4rem 0.5rem;
  border: 1px solid transparent;
  border-radius: 0.4rem;
  background: #161922;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.machine[aria-current="true"] {
  border-color: color-mix(in srgb, var(--bot-accent, #2dd4bf) 55%, #2a3140);
  background: color-mix(in srgb, var(--bot-active-fill, #134e4a) 55%, #161922);
}
.machine-name {
  font-weight: 600;
}
.machine-component {
  color: var(--bot-muted, #8b93a7);
  font-size: 0.75rem;
}
.path,
.last-event,
.observes,
.inspector-status {
  margin: 0;
  color: var(--bot-ink, #e8eaef);
  word-break: break-all;
}
.now-block {
  padding: 0.55rem 0.6rem;
  border: 2px solid var(--bot-now-border, #2dd4bf);
  border-radius: 0.5rem;
  background: color-mix(in srgb, var(--bot-now, #14b8a6) 16%, #161922);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--bot-now, #14b8a6) 28%, transparent);
}
.now-block[data-empty="true"] {
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.now-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
  margin: 0 0 0.35rem;
}
.now-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
  padding: 0.08rem 0.42rem;
  border-radius: 999px;
  background: var(--bot-now, #14b8a6);
  color: var(--bot-now-ink, #042f2e);
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  animation: now-pulse 1.8s ease-out infinite;
}
.now-chip[hidden] {
  display: none;
}
.now-chip::before {
  content: "";
  width: 0.38rem;
  height: 0.38rem;
  border-radius: 999px;
  background: var(--bot-now-ink, #042f2e);
}
@keyframes now-pulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--bot-now, #14b8a6) 55%, transparent);
  }
  50% {
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--bot-now, #14b8a6) 0%, transparent);
  }
}
.crumbs {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.2rem 0.25rem;
  margin: 0 0 0.25rem;
  color: var(--bot-ink, #e8eaef);
  font-size: 0.92rem;
  font-weight: 650;
}
.crumb-sep {
  opacity: 0.55;
}
.crumb-now {
  color: var(--bot-now-ink, #042f2e);
  background: var(--bot-now, #14b8a6);
  border-radius: 0.28rem;
  padding: 0.04rem 0.35rem;
  font-weight: 800;
}
.muted {
  color: var(--bot-muted, #8b93a7);
}
select,
button.tool {
  font: inherit;
  color: inherit;
  background: #161922;
  border: 1px solid var(--bot-line, #2a3140);
  border-radius: 0.35rem;
  padding: 0.22rem 0.45rem;
}
button.tool {
  cursor: pointer;
}
.zoom {
  color: var(--bot-muted, #8b93a7);
  min-width: 3rem;
}
.error {
  margin: 0;
  color: #fecaca;
  font-size: 0.78rem;
}
.command {
  display: grid;
  gap: 0.35rem;
}
.command label {
  display: grid;
  gap: 0.2rem;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.72rem;
}
.command input,
.command textarea {
  font: inherit;
  color: inherit;
  background: #161922;
  border: 1px solid var(--bot-line, #2a3140);
  border-radius: 0.35rem;
  padding: 0.22rem 0.45rem;
}
.command textarea {
  min-height: 3.2rem;
  resize: vertical;
}
.command-result {
  margin: 0;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.75rem;
}
@media (max-width: 720px) {
  .studio {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 1fr) auto;
  }
  .canvas {
    order: 1;
    min-height: 48vh;
  }
  .inspector {
    order: 2;
    max-height: 38vh;
    border-right: 0;
    border-top: 1px solid var(--bot-line, #2a3140);
  }
}
`;

function mark(element: HTMLElement, hook: string): void {
  element.part.add(hook);
  element.setAttribute("data-testid", hook);
}

function pathCrumbs(path: string): string[] {
  return path.split("/").filter((part) => part.length > 0);
}

export class BotDashboard extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #inspectorStatus: HTMLParagraphElement;
  readonly #picker: HTMLSelectElement;
  readonly #zoom: HTMLSpanElement;
  readonly #machines: HTMLDivElement;
  readonly #pathBlock: HTMLDivElement;
  readonly #nowChip: HTMLSpanElement;
  readonly #path: HTMLParagraphElement;
  readonly #lastEvent: HTMLParagraphElement;
  readonly #observes: HTMLParagraphElement;
  readonly #error: HTMLParagraphElement;
  readonly #eventName: HTMLInputElement;
  readonly #eventData: HTMLTextAreaElement;
  readonly #commandResult: HTMLParagraphElement;
  readonly #source: BotOtelSource;
  readonly #graph: BotMachineGraph;
  #controller: DashboardController | null = null;
  #abort: AbortController | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, cssText);

    const topbar = document.createElement("div");
    topbar.className = "topbar";
    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = "bot HSM";
    this.#source = document.createElement("bot-otel-source");
    this.#picker = document.createElement("select");
    this.#picker.setAttribute("aria-label", "Observed machine");
    this.#picker.dataset["event"] = "dashboard.machine.selected";
    const fit = document.createElement("button");
    fit.type = "button";
    fit.className = "tool";
    fit.textContent = "Fit";
    fit.addEventListener("click", () => {
      this.#graph.fit();
    });
    this.#zoom = document.createElement("span");
    this.#zoom.className = "zoom";
    this.#zoom.textContent = "100%";
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "tool";
    reset.dataset["event"] = "dashboard.reset";
    reset.textContent = "Reset";
    topbar.append(title, this.#source, this.#picker, fit, this.#zoom, reset);
    this.#root.addEventListener(BotOtelSource.eventName, this.#onSource);

    const studio = document.createElement("div");
    studio.className = "studio";
    const inspector = document.createElement("aside");
    inspector.className = "inspector";
    mark(inspector, "inspector");

    const statusBlock = document.createElement("div");
    const statusLabel = document.createElement("p");
    statusLabel.className = "section-label";
    statusLabel.textContent = "Status";
    this.#inspectorStatus = document.createElement("p");
    this.#inspectorStatus.className = "inspector-status";
    mark(this.#inspectorStatus, "inspector-status");
    statusBlock.append(statusLabel, this.#inspectorStatus);

    const machineBlock = document.createElement("div");
    const machineLabel = document.createElement("p");
    machineLabel.className = "section-label";
    machineLabel.textContent = "Machines";
    this.#machines = document.createElement("div");
    this.#machines.className = "machine-list";
    mark(this.#machines, "machine-list");
    machineBlock.append(machineLabel, this.#machines);

    this.#pathBlock = document.createElement("div");
    this.#pathBlock.className = "now-block";
    this.#pathBlock.dataset["empty"] = "true";
    this.#pathBlock.setAttribute("aria-live", "polite");
    const pathHead = document.createElement("div");
    pathHead.className = "now-head";
    const pathLabel = document.createElement("p");
    pathLabel.className = "section-label";
    pathLabel.textContent = "Current state";
    this.#nowChip = document.createElement("span");
    this.#nowChip.className = "now-chip";
    this.#nowChip.textContent = "now";
    this.#nowChip.hidden = true;
    mark(this.#nowChip, "now-chip");
    pathHead.append(pathLabel, this.#nowChip);
    this.#path = document.createElement("p");
    this.#path.className = "path";
    mark(this.#path, "current-path");
    this.#pathBlock.append(pathHead, this.#path);

    const eventBlock = document.createElement("div");
    const eventLabel = document.createElement("p");
    eventLabel.className = "section-label";
    eventLabel.textContent = "Last event";
    this.#lastEvent = document.createElement("p");
    this.#lastEvent.className = "last-event";
    mark(this.#lastEvent, "last-event");
    eventBlock.append(eventLabel, this.#lastEvent);

    const observeBlock = document.createElement("div");
    const observeLabel = document.createElement("p");
    observeLabel.className = "section-label";
    observeLabel.textContent = "Observes";
    this.#observes = document.createElement("p");
    this.#observes.className = "observes";
    mark(this.#observes, "observe-count");
    observeBlock.append(observeLabel, this.#observes);

    this.#error = document.createElement("p");
    this.#error.className = "error";

    const commandBlock = document.createElement("div");
    commandBlock.className = "command";
    const commandLabel = document.createElement("p");
    commandLabel.className = "section-label";
    commandLabel.textContent = "Send event";
    const nameLabel = document.createElement("label");
    nameLabel.textContent = "Event name";
    this.#eventName = document.createElement("input");
    this.#eventName.type = "text";
    this.#eventName.setAttribute("aria-label", "Event name");
    mark(this.#eventName, "event-name");
    nameLabel.append(this.#eventName);
    const dataLabel = document.createElement("label");
    dataLabel.textContent = "JSON data";
    this.#eventData = document.createElement("textarea");
    this.#eventData.setAttribute("aria-label", "Event JSON data");
    mark(this.#eventData, "event-data");
    dataLabel.append(this.#eventData);
    const send = document.createElement("button");
    send.type = "button";
    send.className = "tool";
    send.dataset["event"] = "dashboard.command.send";
    send.textContent = "Send";
    mark(send, "send-event");
    this.#commandResult = document.createElement("p");
    this.#commandResult.className = "command-result";
    mark(this.#commandResult, "command-result");
    commandBlock.append(commandLabel, nameLabel, dataLabel, send, this.#commandResult);

    inspector.append(
      statusBlock,
      machineBlock,
      this.#pathBlock,
      eventBlock,
      observeBlock,
      commandBlock,
      this.#error,
    );

    const canvas = document.createElement("div");
    canvas.className = "canvas";
    this.#graph = document.createElement("bot-machine-graph");
    mark(this.#graph, "canvas");
    canvas.append(this.#graph);
    studio.append(inspector, canvas);
    this.#root.append(topbar, studio);
  }

  connectedCallback(): void {
    if (this.#controller === null) {
      this.#controller = new DashboardController({
        onSnapshot: (snapshot) => {
          this.#render(snapshot);
        },
      });
    }
    this.#abort = new AbortController();
    const signal = this.#abort.signal;
    this.#root.addEventListener("click", this.#onGesture, { signal });
    this.#root.addEventListener("change", this.#onGesture, { signal });
    this.#graph.addEventListener("bot-machine-graph-zoom", this.#onZoom, { signal });
    this.#graph.addEventListener("bot-machine-graph-edge", this.#onEdge, { signal });
    this.#render(this.#controller.snapshot());
    this.#source.replayReady();
  }

  disconnectedCallback(): void {
    this.#abort?.abort();
    this.#abort = null;
    const controller = this.#controller;
    this.#controller = null;
    if (controller !== null) {
      void controller.stop();
    }
  }

  #render(snapshot: DashboardSnapshot): void {
    const live = snapshot.phase === "live";
    this.#inspectorStatus.textContent = live ? "live" : snapshot.phase === "error" ? "offline" : snapshot.phase;
    this.#error.textContent = snapshot.errorMessage ?? "";
    if (this.#eventName.value !== snapshot.commandEventName) {
      this.#eventName.value = snapshot.commandEventName;
    }
    if (this.#eventData.value !== snapshot.commandDataJson) {
      this.#eventData.value = snapshot.commandDataJson;
    }
    this.#commandResult.textContent =
      snapshot.commandResult === null
        ? ""
        : `${snapshot.commandResult.result.replaceAll("_", " ")}: ${snapshot.commandResult.detail}`;
    this.#picker.replaceChildren();
    this.#machines.replaceChildren();
    const view = snapshot.document;
    if (view === null || view.machines.length === 0) {
      this.#writeCurrentState(null);
      this.#lastEvent.textContent = "—";
      this.#observes.textContent = "0";
      this.#writeGraphHooks(null);
      this.#graph.graph = null;
      return;
    }
    for (const machine of view.machines) {
      const option = document.createElement("option");
      option.value = machine.name;
      option.textContent = machine.name;
      option.selected = machine.name === view.selectedMachine;
      this.#picker.append(option);

      const item = document.createElement("button");
      item.type = "button";
      item.className = "machine";
      item.dataset["event"] = "dashboard.machine.selected";
      item.dataset["machineName"] = machine.name;
      item.setAttribute("aria-current", machine.name === view.selectedMachine ? "true" : "false");
      const name = document.createElement("span");
      name.className = "machine-name";
      name.textContent = machine.name;
      const component = document.createElement("span");
      component.className = "machine-component";
      component.textContent = machine.componentName;
      item.append(name, component);
      this.#machines.append(item);
    }
    const selected = snapshot.selectedGraph;
    if (selected === null) {
      this.#writeCurrentState(null);
      this.#lastEvent.textContent = "—";
      this.#observes.textContent = String(view.observeCount);
      this.#writeGraphHooks(null);
      this.#graph.graph = null;
      return;
    }
    this.#writeCurrentState(selected.currentState);
    this.#lastEvent.textContent = selected.lastEventName;
    this.#observes.textContent = String(selected.observationCount);
    this.#writeGraphHooks(selected);
    this.#graph.graph = selected;
  }

  #writeCurrentState(currentState: string | null): void {
    const empty = currentState === null || currentState.length === 0;
    this.#pathBlock.dataset["empty"] = empty ? "true" : "false";
    this.#nowChip.hidden = empty;
    if (empty || currentState === null) {
      this.#path.textContent = "—";
      return;
    }
    this.#path.replaceChildren();
    const crumbs = pathCrumbs(currentState);
    if (crumbs.length === 0) {
      this.#path.textContent = currentState;
      return;
    }
    const crumbRow = document.createElement("span");
    crumbRow.className = "crumbs";
    crumbs.forEach((crumb, index) => {
      if (index > 0) {
        const sep = document.createElement("span");
        sep.className = "crumb-sep";
        sep.textContent = "/";
        crumbRow.append(sep);
      }
      const piece = document.createElement("span");
      piece.textContent = crumb;
      if (index === crumbs.length - 1) {
        piece.className = "crumb-now";
      }
      crumbRow.append(piece);
    });
    const full = document.createElement("span");
    full.className = "muted";
    full.textContent = currentState;
    this.#path.append(crumbRow, full);
  }

  #writeGraphHooks(selected: MachineGraph | null): void {
    this.#graph.setAttribute("data-current-state", selected?.currentState ?? "");
    this.#graph.setAttribute("data-node-count", selected === null ? "0" : String(selected.nodes.length));
  }

  readonly #onSource = (event: Event): void => {
    if (!(event instanceof CustomEvent) || this.#controller === null) {
      return;
    }
    if (!isSourceDetail(event.detail)) {
      return;
    }
    void this.#controller.dispatch("dashboard.source.selected", { source: event.detail.source });
  };

  readonly #onEdge = (event: Event): void => {
    if (!(event instanceof CustomEvent) || this.#controller === null || !isEdgeDetail(event.detail)) {
      return;
    }
    void this.#controller.dispatch("dashboard.command.prefill", { eventName: event.detail.eventName });
  };

  readonly #onZoom = (event: Event): void => {
    if (!(event instanceof CustomEvent) || !isZoomDetail(event.detail)) {
      return;
    }
    this.#zoom.textContent = `${Math.round(event.detail.zoom * 100)}%`;
  };

  readonly #onGesture = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof Element) || this.#controller === null) {
      return;
    }
    const control = target.closest("[data-event]");
    if (!(control instanceof HTMLElement)) {
      return;
    }
    const eventName = control.dataset["event"];
    if (eventName === undefined || !isDashboardEventName(eventName)) {
      return;
    }
    if (eventName === "dashboard.machine.selected") {
      const machineName =
        control instanceof HTMLSelectElement ? control.value : control.dataset["machineName"];
      if (machineName === undefined) {
        return;
      }
      void this.#controller.dispatch(eventName, { machineName });
      return;
    }
    if (eventName === "dashboard.command.send") {
      void this.#controller.dispatch(eventName, {
        eventName: this.#eventName.value,
        dataJson: this.#eventData.value,
      });
      return;
    }
    void this.#controller.dispatch(eventName);
  };
}

function isSourceDetail(value: unknown): value is OtelSourceDetail {
  if (typeof value === "object" && value !== null && "source" in value) {
    return isOtelSource(value.source);
  }
  return false;
}

function isZoomDetail(value: unknown): value is { zoom: number } {
  return typeof value === "object" && value !== null && "zoom" in value && typeof value.zoom === "number";
}

function isEdgeDetail(value: unknown): value is { eventName: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "eventName" in value &&
    typeof value.eventName === "string"
  );
}

export function registerBotDashboard(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) {
    customElements.define(ELEMENT_NAME, BotDashboard);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-dashboard": BotDashboard;
  }
}
