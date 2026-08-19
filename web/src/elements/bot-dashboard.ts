import {
  DashboardController,
  isDashboardEventName,
  type DashboardSnapshot,
} from "../dashboard-hsm.ts";
import { graphsForVisibility } from "../dashboard-graphs.ts";
import { type MachineGraph } from "../otel/machines.ts";
import { isOtelSource } from "../otel/source.ts";
import { BotMachineGraph } from "./bot-machine-graph.ts";
import { BotOtelSource, type OtelSourceDetail } from "./bot-otel-source.ts";
import { applyStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-dashboard";

const cssText = `
:host {
  display: flex;
  flex-direction: column;
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
  flex: 1 1 auto;
  min-height: 0;
  height: auto;
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
  height: auto;
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
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  width: 100%;
  margin: 0;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 0.4rem;
  background: #161922;
  color: inherit;
  font: inherit;
  text-align: left;
}
.machine-select {
  display: grid;
  gap: 0.1rem;
  min-width: 0;
  padding: 0.4rem 0.5rem;
  border: 0;
  border-radius: 0.4rem;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.machine-select[aria-current="true"] {
  border-color: color-mix(in srgb, var(--bot-accent, #2dd4bf) 55%, #2a3140);
  background: color-mix(in srgb, var(--bot-active-fill, #134e4a) 55%, #161922);
}
.machine-visibility {
  display: flex;
  align-items: center;
  padding: 0.4rem 0.5rem;
  color: var(--bot-muted, #8b93a7);
  cursor: pointer;
}
.machine-visibility input {
  width: 1rem;
  height: 1rem;
  margin: 0;
  accent-color: var(--bot-accent, #2dd4bf);
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
.brand {
  display: grid;
  gap: 0.1rem;
  min-width: 15rem;
  margin-right: auto;
}
.eyebrow {
  margin: 0;
  color: var(--bot-accent, #2dd4bf);
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.subtitle {
  margin: 0;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.7rem;
}
.topbar-tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
}
.machine-picker {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.7rem;
}
.studio {
  grid-template-columns: 250px minmax(0, 1fr) 290px;
}
.inspector {
  display: flex;
  flex-direction: column;
  align-content: initial;
  gap: 0.75rem;
  min-width: 0;
  padding: 0.75rem;
}
.members,
.event-rail,
.map-panel {
  min-width: 0;
  min-height: 0;
}
.members {
  display: grid;
  flex: 1 1 0;
  align-content: start;
  gap: 0.55rem;
  overflow: auto;
}
.rail-heading,
.map-heading,
.event-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
}
.rail-title,
.map-title,
.event-title {
  margin: 0;
  font-size: 0.78rem;
  font-weight: 750;
  letter-spacing: 0.02em;
}
.rail-count,
.map-stat,
.event-meta {
  color: var(--bot-muted, #8b93a7);
  font-size: 0.68rem;
}
.member-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
}
.member-action {
  padding: 0.2rem 0.38rem;
  border: 1px solid var(--bot-line, #2a3140);
  border-radius: 0.3rem;
  background: #161922;
  color: var(--bot-muted, #b6bfd2);
  font: inherit;
  font-size: 0.66rem;
  cursor: pointer;
}
.member-action:hover,
.member-action:focus-visible {
  border-color: var(--bot-accent, #2dd4bf);
  color: var(--bot-ink, #e8eaef);
}
.machine {
  position: relative;
  overflow: hidden;
  border-color: #242b38;
}
.machine[data-visible="false"] {
  opacity: 0.52;
}
.machine-select {
  gap: 0.16rem;
  padding: 0.48rem 0.5rem;
}
.machine-select[aria-current="true"] {
  outline: 1px solid color-mix(in srgb, var(--bot-accent, #2dd4bf) 55%, #2a3140);
  outline-offset: -1px;
}
.machine-observes,
.machine-state {
  color: var(--bot-muted, #8b93a7);
  font-size: 0.64rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.machine-state[data-empty="true"] {
  font-style: italic;
}
.machine-visibility {
  align-self: stretch;
  padding: 0.5rem;
}
.details {
  display: grid;
  gap: 0.6rem;
  padding-top: 0.7rem;
  border-top: 1px solid var(--bot-line, #2a3140);
}
.details-heading {
  margin: 0;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.map-panel {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  background: #0b0d12;
}
.map-heading {
  min-height: 2.65rem;
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid var(--bot-line, #2a3140);
  background: #10131a;
}
.map-heading-left,
.map-stats {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.45rem 0.7rem;
}
.map-stats {
  justify-content: flex-end;
}
.map-stat strong {
  color: var(--bot-ink, #e8eaef);
  font-weight: 700;
}
.map-tools {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}
.map-tools .tool {
  padding: 0.17rem 0.35rem;
  font-size: 0.68rem;
}
.event-rail {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  padding: 0.75rem;
  border-left: 1px solid var(--bot-line, #2a3140);
  background: var(--bot-panel, #12141a);
  overflow: hidden;
}
.event-list {
  display: grid;
  align-content: start;
  gap: 0.35rem;
  min-height: 0;
  overflow: auto;
}
.event-item {
  display: grid;
  gap: 0.25rem;
  padding: 0.55rem;
  border: 1px solid #242b38;
  border-radius: 0.42rem;
  background: #161922;
}
.event-item[data-current="true"] {
  border-color: color-mix(in srgb, var(--bot-accent, #2dd4bf) 45%, #2a3140);
}
.event-machine {
  color: var(--bot-ink, #e8eaef);
  font-size: 0.72rem;
  font-weight: 700;
}
.event-name {
  color: var(--bot-accent, #5eead4);
  font-size: 0.7rem;
  overflow-wrap: anywhere;
}
.event-state {
  color: var(--bot-muted, #8b93a7);
  font-size: 0.66rem;
  overflow-wrap: anywhere;
}
.event-empty {
  margin: 0;
  color: var(--bot-muted, #8b93a7);
  font-size: 0.7rem;
}
@media (max-width: 720px) {
  .studio {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 38vh) minmax(0, 1fr) auto;
  }
  .canvas {
    order: 1;
    height: auto;
    min-height: 0;
  }
  .inspector {
    order: 2;
    max-height: 38vh;
    border-right: 0;
    border-top: 1px solid var(--bot-line, #2a3140);
  }
}
@media (max-width: 1080px) {
  .studio {
    grid-template-columns: 230px minmax(0, 1fr) 245px;
  }
  .subtitle {
    display: none;
  }
}
@media (max-width: 820px) {
  .studio {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(0, 1fr) auto;
  }
  bot-machine-graph {
    height: 100%;
    min-height: 0;
  }
  .canvas {
    height: auto;
    min-height: 0;
  }
  .inspector {
    grid-column: 1;
    grid-row: 1;
    display: grid;
    grid-template-rows: minmax(7rem, 1fr) minmax(0, 1fr);
    align-content: stretch;
    max-height: none;
    min-height: 0;
    overflow: hidden;
    border-right: 0;
    border-bottom: 1px solid var(--bot-line, #2a3140);
  }
  .details {
    min-height: 0;
    overflow: auto;
  }
  .event-rail {
    grid-column: 1;
    grid-row: 3;
    max-height: 32vh;
    border-top: 1px solid var(--bot-line, #2a3140);
    border-left: 0;
  }
  .map-panel {
    display: grid;
    grid-column: 1;
    grid-row: 2;
  }
}
@media (max-width: 720px) {
  .topbar {
    align-items: flex-start;
  }
  .studio {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 38vh) minmax(0, 1fr) auto;
  }
  .inspector {
    grid-column: 1;
    grid-row: 1;
    max-height: none;
    min-height: 0;
    overflow: hidden;
    border-right: 0;
    border-bottom: 1px solid var(--bot-line, #2a3140);
  }
  .details {
    min-height: 0;
    overflow: auto;
  }
  .map-panel {
    display: grid;
    grid-column: 1;
    grid-row: 2;
  }
  .event-rail {
    grid-column: 1;
    grid-row: 3;
    max-height: 32vh;
    border-top: 1px solid var(--bot-line, #2a3140);
    border-left: 0;
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

type VisibilityAction = "show-all" | "hide-all" | "hide-unobserved";

function isVisibilityAction(value: string | undefined): value is VisibilityAction {
  return value === "show-all" || value === "hide-all" || value === "hide-unobserved";
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
  readonly #shownCount: HTMLSpanElement;
  readonly #hiddenCount: HTMLSpanElement;
  readonly #observedCount: HTMLSpanElement;
  readonly #memberCount: HTMLSpanElement;
  readonly #eventList: HTMLDivElement;
  #visibleMachines = new Map<string, boolean>();
  #controller: DashboardController | null = null;
  #abort: AbortController | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, cssText);

    const topbar = document.createElement("div");
    topbar.className = "topbar";
    const brand = document.createElement("div");
    brand.className = "brand";
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = "Environment workspace";
    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = "Environment";
    mark(title, "title");
    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = "A live map of loaded machines, ownership, and observed events.";
    brand.append(eyebrow, title, subtitle);
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
    const pickerLabel = document.createElement("label");
    pickerLabel.className = "machine-picker";
    pickerLabel.textContent = "Focus";
    pickerLabel.append(this.#picker);
    const topbarTools = document.createElement("div");
    topbarTools.className = "topbar-tools";
    topbarTools.append(this.#source, pickerLabel);
    topbar.append(brand, topbarTools);
    this.#root.addEventListener(BotOtelSource.eventName, this.#onSource);

    const studio = document.createElement("div");
    studio.className = "studio";
    const inspector = document.createElement("aside");
    inspector.className = "inspector";
    mark(inspector, "inspector");
    const members = document.createElement("section");
    members.className = "members";
    mark(members, "members");

    const statusBlock = document.createElement("div");
    const statusLabel = document.createElement("p");
    statusLabel.className = "section-label";
    statusLabel.textContent = "Status";
    this.#inspectorStatus = document.createElement("p");
    this.#inspectorStatus.className = "inspector-status";
    mark(this.#inspectorStatus, "inspector-status");
    statusBlock.append(statusLabel, this.#inspectorStatus);

    const machineBlock = document.createElement("div");
    const machineHeading = document.createElement("div");
    machineHeading.className = "rail-heading";
    const machineLabel = document.createElement("h2");
    machineLabel.className = "rail-title";
    machineLabel.textContent = "Members";
    this.#memberCount = document.createElement("span");
    this.#memberCount.className = "rail-count";
    machineHeading.append(machineLabel, this.#memberCount);
    this.#machines = document.createElement("div");
    this.#machines.className = "machine-list";
    mark(this.#machines, "machine-list");
    const memberActions = document.createElement("div");
    memberActions.className = "member-actions";
    memberActions.append(
      this.#makeVisibilityAction("Show all", "show-all"),
      this.#makeVisibilityAction("Hide all", "hide-all"),
      this.#makeVisibilityAction("Hide unobserved", "hide-unobserved"),
    );
    machineBlock.append(machineHeading, memberActions, this.#machines);
    members.append(machineBlock);

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

    const details = document.createElement("section");
    details.className = "details";
    const detailsHeading = document.createElement("h2");
    detailsHeading.className = "details-heading";
    detailsHeading.textContent = "Selected machine";
    details.append(
      detailsHeading,
      statusBlock,
      this.#pathBlock,
      eventBlock,
      observeBlock,
      commandBlock,
      this.#error,
    );
    inspector.append(members, details);

    const mapPanel = document.createElement("section");
    mapPanel.className = "map-panel";
    const mapHeading = document.createElement("header");
    mapHeading.className = "map-heading";
    mark(mapHeading, "map-header");
    const mapHeadingLeft = document.createElement("div");
    mapHeadingLeft.className = "map-heading-left";
    const mapTitle = document.createElement("h2");
    mapTitle.className = "map-title";
    mapTitle.textContent = "Environment map";
    const mapTools = document.createElement("div");
    mapTools.className = "map-tools";
    mapTools.append(fit, this.#zoom, reset);
    mapHeadingLeft.append(mapTitle, mapTools);
    const mapStats = document.createElement("div");
    mapStats.className = "map-stats";
    const shownStat = document.createElement("span");
    shownStat.className = "map-stat";
    this.#shownCount = document.createElement("span");
    const hiddenStat = document.createElement("span");
    hiddenStat.className = "map-stat";
    this.#hiddenCount = document.createElement("span");
    const observedStat = document.createElement("span");
    observedStat.className = "map-stat";
    this.#observedCount = document.createElement("span");
    shownStat.append(document.createTextNode("Shown "), this.#shownCount);
    hiddenStat.append(document.createTextNode("Hidden "), this.#hiddenCount);
    observedStat.append(document.createTextNode("Observed "), this.#observedCount);
    mapStats.append(shownStat, hiddenStat, observedStat);
    mapHeading.append(mapHeadingLeft, mapStats);

    const canvas = document.createElement("div");
    canvas.className = "canvas";
    this.#graph = document.createElement("bot-machine-graph");
    mark(this.#graph, "canvas");
    canvas.append(this.#graph);
    mapPanel.append(mapHeading, canvas);

    const eventRail = document.createElement("aside");
    eventRail.className = "event-rail";
    mark(eventRail, "event-rail");
    const eventHeading = document.createElement("div");
    eventHeading.className = "event-heading";
    const eventTitle = document.createElement("h2");
    eventTitle.className = "event-title";
    eventTitle.textContent = "Events";
    const eventMeta = document.createElement("span");
    eventMeta.className = "event-meta";
    eventMeta.textContent = "latest per machine";
    eventHeading.append(eventTitle, eventMeta);
    this.#eventList = document.createElement("div");
    this.#eventList.className = "event-list";
    mark(this.#eventList, "event-list");
    eventRail.append(eventHeading, this.#eventList);

    studio.append(inspector, mapPanel, eventRail);
    this.#root.append(topbar, studio);
  }

  #makeVisibilityAction(label: string, action: "show-all" | "hide-all" | "hide-unobserved"): HTMLButtonElement {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "member-action";
    button.textContent = label;
    button.dataset["dashboardAction"] = action;
    mark(button, action);
    return button;
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
    this.#eventList.replaceChildren();
    const view = snapshot.document;
    if (view === null || view.machines.length === 0) {
      this.#visibleMachines.clear();
      this.#memberCount.textContent = "0 loaded";
      this.#writeVisibilityStats(null);
      this.#writeEvents(null, null);
      this.#writeCurrentState(null);
      this.#lastEvent.textContent = "—";
      this.#observes.textContent = "0";
      this.#writeGraphHooks(null);
      this.#graph.graphs = [];
      return;
    }
    const machineNames = new Set(view.machines.map((machine) => machine.name));
    for (const machine of view.machines) {
      if (!this.#visibleMachines.has(machine.name)) {
        this.#visibleMachines.set(machine.name, true);
      }
    }
    for (const machineName of this.#visibleMachines.keys()) {
      if (!machineNames.has(machineName)) {
        this.#visibleMachines.delete(machineName);
      }
    }
    for (const machine of view.machines) {
      const option = document.createElement("option");
      option.value = machine.name;
      option.textContent = machine.name;
      option.selected = machine.name === view.selectedMachine;
      this.#picker.append(option);

      const item = document.createElement("div");
      item.className = "machine";
      item.dataset["machineName"] = machine.name;
      const select = document.createElement("button");
      select.type = "button";
      select.className = "machine-select";
      select.dataset["event"] = "dashboard.machine.selected";
      select.dataset["machineName"] = machine.name;
      select.setAttribute("aria-current", machine.name === view.selectedMachine ? "true" : "false");
      const name = document.createElement("span");
      name.className = "machine-name";
      name.textContent = machine.name;
      const component = document.createElement("span");
      component.className = "machine-component";
      component.textContent = machine.componentName;
      const observes = document.createElement("span");
      observes.className = "machine-observes";
      observes.textContent = `${String(machine.observationCount)} observations`;
      const state = document.createElement("span");
      state.className = "machine-state";
      state.dataset["empty"] = machine.currentState.length === 0 ? "true" : "false";
      state.textContent = machine.currentState.length === 0 ? "No current state" : machine.currentState;
      select.append(name, component, observes, state);
      item.dataset["visible"] = String(this.#visibleMachines.get(machine.name) === true);
      const visibilityLabel = document.createElement("label");
      visibilityLabel.className = "machine-visibility";
      visibilityLabel.title = `Show or hide ${machine.name} graph`;
      const visibility = document.createElement("input");
      visibility.type = "checkbox";
      visibility.checked = this.#visibleMachines.get(machine.name) === true;
      visibility.setAttribute("aria-label", `Show ${machine.name} graph`);
      visibility.dataset["machineVisibility"] = machine.name;
      visibility.setAttribute("data-testid", "machine-visibility");
      visibility.addEventListener("change", () => {
        this.#setMachineVisibility(machine.name, visibility.checked);
      });
      visibilityLabel.append(visibility);
      item.append(select, visibilityLabel);
      this.#machines.append(item);
    }
    this.#memberCount.textContent = `${String(view.machines.length)} loaded`;
    this.#writeVisibilityStats(view);
    this.#writeEvents(view, view.selectedMachine);
    const selected = snapshot.selectedGraph;
    if (selected === null) {
      this.#writeCurrentState(null);
      this.#lastEvent.textContent = "—";
      this.#observes.textContent = String(view.observeCount);
      this.#writeGraphHooks(null);
      this.#graph.graphs = [];
      return;
    }
    this.#writeCurrentState(selected.currentState);
    this.#lastEvent.textContent = selected.lastEventName;
    this.#observes.textContent = String(selected.observationCount);
    this.#writeGraphHooks(selected);
    this.#graph.graphs = graphsForVisibility(view.machines, this.#visibleMachines);
  }

  #writeVisibilityStats(view: DashboardSnapshot["document"]): void {
    const machines = view?.machines ?? [];
    const shown = machines.filter((machine) => this.#visibleMachines.get(machine.name) === true).length;
    const observed = machines.filter((machine) => machine.observationCount > 0).length;
    this.#shownCount.textContent = String(shown);
    this.#hiddenCount.textContent = String(machines.length - shown);
    this.#observedCount.textContent = String(observed);
  }

  #writeEvents(view: DashboardSnapshot["document"], selectedMachine: string | null): void {
    this.#eventList.replaceChildren();
    const machines = view === null ? [] : [...view.machines].sort((left, right) => {
      const observationOrder = right.observationCount - left.observationCount;
      return observationOrder === 0 ? left.name.localeCompare(right.name) : observationOrder;
    });
    if (machines.length === 0) {
      const empty = document.createElement("p");
      empty.className = "event-empty";
      empty.textContent = "No machine events yet.";
      this.#eventList.append(empty);
      return;
    }
    for (const machine of machines) {
      const item = document.createElement("article");
      item.className = "event-item";
      mark(item, "event-item");
      item.dataset["current"] = machine.name === selectedMachine ? "true" : "false";
      const machineName = document.createElement("span");
      machineName.className = "event-machine";
      machineName.textContent = machine.name;
      const eventName = document.createElement("span");
      eventName.className = "event-name";
      eventName.textContent = machine.lastEventName.length > 0 ? machine.lastEventName : "No event observed";
      const state = document.createElement("span");
      state.className = "event-state";
      state.textContent = machine.currentState.length > 0 ? machine.currentState : "No current state";
      const observations = document.createElement("span");
      observations.className = "event-meta";
      observations.textContent = `${String(machine.observationCount)} observations`;
      item.append(machineName, eventName, state, observations);
      this.#eventList.append(item);
    }
  }

  #syncMachineControls(): void {
    for (const visibility of this.#machines.querySelectorAll<HTMLInputElement>("input[data-machine-visibility]")) {
      const machineName = visibility.dataset["machineVisibility"];
      if (machineName === undefined) {
        continue;
      }
      const visible = this.#visibleMachines.get(machineName) === true;
      visibility.checked = visible;
      visibility.closest(".machine")?.setAttribute("data-visible", String(visible));
    }
  }

  #applyVisibilityAction(action: VisibilityAction): void {
    const view = this.#controller?.snapshot().document;
    if (view === null || view === undefined) {
      return;
    }
    for (const machine of view.machines) {
      const visible = action === "show-all" || (action === "hide-unobserved" && machine.observationCount > 0);
      this.#visibleMachines.set(machine.name, visible);
    }
    this.#syncMachineControls();
    this.#writeVisibilityStats(view);
    this.#graph.graphs = graphsForVisibility(view.machines, this.#visibleMachines);
  }

  #setMachineVisibility(machineName: string, visible: boolean): void {
    this.#visibleMachines.set(machineName, visible);
    const document = this.#controller?.snapshot().document;
    if (document === null || document === undefined) {
      this.#graph.graphs = [];
      return;
    }
    this.#syncMachineControls();
    this.#writeVisibilityStats(document);
    this.#graph.graphs = graphsForVisibility(document.machines, this.#visibleMachines);
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
    const actionControl = target.closest("[data-dashboard-action]");
    if (actionControl instanceof HTMLElement && isVisibilityAction(actionControl.dataset["dashboardAction"])) {
      this.#applyVisibilityAction(actionControl.dataset["dashboardAction"]);
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
