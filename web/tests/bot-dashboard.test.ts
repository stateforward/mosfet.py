import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";
import { BotDashboard, registerBotDashboard } from "../src/elements/bot-dashboard.ts";
import { BotMachineGraph, registerBotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { BotOtelSource, registerBotOtelSource } from "../src/elements/bot-otel-source.ts";
import { FlowGraph } from "../src/flow/index.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { streamSource } from "../src/otel/source.ts";
import { getAllByRole, getByRole } from "./by-role.ts";

registerFlowElements();
registerBotOtelSource();
registerBotMachineGraph();
registerBotDashboard();

const YIELD_MS = 0;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for bot-dashboard");
}

function publishedModel(name: string): Record<string, unknown> {
  return {
    name,
    owner: null,
    states: [{ qualified_name: name, parent: "/", initial: `${name}/.initial` }],
    transitions: [],
    initial: `${name}/.initial`,
  };
}

async function bootDashboard(): Promise<BotDashboard> {
  const host = document.createElement("bot-dashboard");
  assert.ok(host instanceof BotDashboard);
  host.connectStream = () => ({ close(): void { return; } });
  document.body.append(host);
  await waitFor(() => host.snapshot().statePath.includes("/connected"));
  await host.dispatch("dashboard.source.selected", {
    source: streamSource(),
    origin: "http://localhost",
  });
  return host;
}

const FRAME_PART = '[part="frame"]';

describe("bot-dashboard inspector focus", () => {
  test("constructor does not start nested hosts", () => {
    const host = document.createElement("bot-dashboard");
    assert.ok(host instanceof BotDashboard);
    assert.equal(host.state(), "");
    const source = host.shadowRoot?.querySelector("bot-otel-source");
    const graph = host.shadowRoot?.querySelector("bot-machine-graph");
    assert.ok(source instanceof BotOtelSource);
    assert.ok(graph instanceof BotMachineGraph);
    const flow = graph.shadowRoot?.querySelector(FRAME_PART);
    assert.ok(flow instanceof FlowGraph);
    assert.equal(source.state(), "");
    assert.equal(graph.state(), "");
    assert.equal(flow.state(), "");
  });

  test("Host.stop stops nested otel source, machine graph, and flow-graph", async () => {
    const host = document.createElement("bot-dashboard");
    assert.ok(host instanceof BotDashboard);
    host.connectStream = () => ({ close(): void { return; } });
    document.body.append(host);
    await waitFor(() => host.snapshot().statePath.includes("/connected"));
    const source = host.shadowRoot?.querySelector("bot-otel-source");
    const graph = host.shadowRoot?.querySelector("bot-machine-graph");
    assert.ok(source instanceof BotOtelSource);
    assert.ok(graph instanceof BotMachineGraph);
    const flow = graph.shadowRoot?.querySelector(FRAME_PART);
    assert.ok(flow instanceof FlowGraph);
    await waitFor(() => source.state() !== "" && graph.state() !== "" && flow.state() !== "");
    await host.stop();
    assert.equal(host.state(), "");
    assert.equal(source.state(), "");
    assert.equal(graph.state(), "");
    assert.equal(flow.state(), "");
    host.remove();
  });

  test("picker focus survives a snapshot rebuild", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const picker = host.querySelector('select[aria-label="Observed machine"]');
    assert.ok(picker instanceof HTMLElement);
    picker.focus();
    assert.equal(host.shadowRoot?.activeElement, picker);
    await host.dispatch("dashboard.model.published", publishedModel("/PhoneBot"));
    const next = host.querySelector('select[aria-label="Observed machine"]');
    assert.equal(host.shadowRoot?.activeElement, next);
    host.remove();
  });

  test("machine-select focus survives a snapshot rebuild", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const select = host.querySelector('button[data-machine-name="/Phone"]');
    assert.ok(select instanceof HTMLElement);
    select.focus();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const next = host.querySelector('button[data-machine-name="/Phone"]');
    assert.ok(next instanceof HTMLElement);
    assert.equal(host.shadowRoot?.activeElement, next);
    host.remove();
  });

  test("visibility checkbox focus survives CSS-special machine names", async () => {
    const host = await bootDashboard();
    const special = "/Phone.a[b]";
    await host.dispatch("dashboard.model.published", publishedModel(special));
    const box = host.querySelector(`input[data-machine-visibility="${CSS.escape(special)}"]`);
    assert.ok(box instanceof HTMLInputElement);
    box.focus();
    await host.dispatch("dashboard.model.published", publishedModel(special));
    const next = host.querySelector(`input[data-machine-visibility="${CSS.escape(special)}"]`);
    assert.ok(next instanceof HTMLInputElement);
    assert.equal(host.shadowRoot?.activeElement, next);
    host.remove();
  });

  test("data-node-count stays the held graph total after machine selection", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    await host.dispatch("dashboard.model.published", publishedModel("/PhoneBot"));
    const graph = host.querySelector("bot-machine-graph");
    assert.ok(graph instanceof BotMachineGraph);
    await waitFor(() => graph.graphs.length === 2);
    const heldTotal = graph.graphs.reduce((count, item) => count + item.nodes.length, 0);
    const phoneBot = graph.graphs.find((item) => item.name === "/PhoneBot");
    assert.ok(phoneBot !== undefined);
    assert.notEqual(phoneBot.nodes.length, heldTotal);
    await waitFor(() => graph.getAttribute("data-node-count") === String(heldTotal));
    await host.dispatch("dashboard.machine.selected", { machineName: "/PhoneBot" });
    assert.equal(graph.getAttribute("data-node-count"), String(heldTotal));
    assert.notEqual(graph.getAttribute("data-node-count"), String(phoneBot.nodes.length));
    host.remove();
  });

  test("empty control names are not restored", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const stray = document.createElement("button");
    stray.focus();
    await host.dispatch("dashboard.model.published", publishedModel("/PhoneBot"));
    const picker = host.querySelector('select[aria-label="Observed machine"]');
    assert.notEqual(host.shadowRoot?.activeElement, picker);
    host.remove();
  });

  test("inspector machine select stamps one event and focuses the graph", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const select = host.querySelector('button[data-machine-name="/Phone"]');
    assert.ok(select instanceof HTMLButtonElement);
    const selectedEvent = "dashboard.machine.selected";
    assert.equal(select.dataset["event"], selectedEvent);
    const graphFocusEvent = '[data-event="dashboard.graph.focus"]';
    assert.equal(host.querySelector(graphFocusEvent), null);
    const graph = host.querySelector("bot-machine-graph");
    assert.ok(graph instanceof BotMachineGraph);
    const names: string[] = [];
    const original = graph.dispatch.bind(graph);
    const spy = ((eventOrCtx: hsm.Event | hsm.Context, maybeEvent?: hsm.Event) => {
      const event = maybeEvent ?? (eventOrCtx instanceof hsm.Context ? undefined : eventOrCtx);
      if (event !== undefined && event.name === BotMachineGraph.focusEvent.name && hsm.isRecord(event.data)) {
        const machineName = event.data["machineName"];
        if (typeof machineName === "string") names.push(machineName);
      }
      return maybeEvent === undefined
        ? original(eventOrCtx as hsm.Event)
        : original(eventOrCtx as hsm.Context, maybeEvent);
    }) as BotMachineGraph["dispatch"];
    graph.dispatch = spy;
    const machineName = "/Phone";
    await host.dispatch(selectedEvent, { machineName });
    assert.deepEqual(names, [machineName]);
    graph.dispatch = original;
    host.remove();
  });

  test("machine-kind graph focus does not steal inspector control focus", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const select = host.querySelector('button[data-machine-name="/Phone"]');
    assert.ok(select instanceof HTMLElement);
    select.focus();
    const graph = host.querySelector("bot-machine-graph");
    assert.ok(graph instanceof BotMachineGraph);
    graph.focusMachine("/Phone");
    assert.equal(host.shadowRoot?.activeElement, select);
    host.remove();
  });

  test("empty graph focus names are not forwarded to the machine graph", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    const graph = host.querySelector("bot-machine-graph");
    assert.ok(graph instanceof BotMachineGraph);
    const names: string[] = [];
    const original = graph.dispatch.bind(graph);
    const spy = ((eventOrCtx: hsm.Event | hsm.Context, maybeEvent?: hsm.Event) => {
      const event = maybeEvent ?? (eventOrCtx instanceof hsm.Context ? undefined : eventOrCtx);
      if (event !== undefined && event.name === BotMachineGraph.focusEvent.name && hsm.isRecord(event.data)) {
        const machineName = event.data["machineName"];
        if (typeof machineName === "string") names.push(machineName);
      }
      return maybeEvent === undefined
        ? original(eventOrCtx as hsm.Event)
        : original(eventOrCtx as hsm.Context, maybeEvent);
    }) as BotMachineGraph["dispatch"];
    graph.dispatch = spy;
    const emptyName = "";
    await host.dispatch("dashboard.graph.focus", { machineName: emptyName });
    const noneFocused = 0;
    assert.equal(names.length, noneFocused);
    const machineName = "/Phone";
    await host.dispatch("dashboard.graph.focus", { machineName });
    assert.deepEqual(names, [machineName]);
    graph.dispatch = original;
    host.remove();
  });
});

describe("bot-dashboard agent accessibility", () => {
  test("operable controls and live statuses have stable role and name at construction", () => {
    const host = document.createElement("bot-dashboard");
    assert.ok(getByRole(host, "combobox", "Observed machine") instanceof HTMLElement);
    // The dashboard Fit button and the nested flow-controls fit button must
    // keep distinct stable names: each resolves to exactly one button, and no
    // case-insensitive "fit" name remains in the host tree.
    const fit = getByRole(host, "button", "Fit environment map");
    assert.ok(fit instanceof HTMLElement);
    const flowFit = getByRole(host, "button", "Fit flow view");
    assert.ok(flowFit instanceof HTMLElement);
    const bareFitMatches = getAllByRole(host, "button", "fit");
    const noBareFit = 0;
    assert.equal(bareFitMatches.length, noBareFit);
    assert.ok(getByRole(host, "button", "Reset") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Send") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Replay") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Previous replay event") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Next replay event") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Live") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "Play replay") instanceof HTMLElement);
    assert.ok(getByRole(host, "slider", "Replay position") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Viewport zoom") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Replay status") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Status") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Current state") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Last event") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Observes") instanceof HTMLElement);
    assert.ok(getByRole(host, "status", "Command result") instanceof HTMLElement);
    assert.ok(getByRole(host, "group", "Members") instanceof HTMLElement);
    assert.ok(getByRole(host, "textbox", "Event name") instanceof HTMLElement);
    assert.ok(getByRole(host, "textbox", "Event JSON data") instanceof HTMLElement);
    const picker = getByRole(host, "combobox", "Observed machine");
    assert.ok(picker instanceof HTMLElement);
    assert.equal(picker.getAttribute("id"), "observed-machine");
    const label = host.querySelector('label[for="observed-machine"]');
    assert.ok(label instanceof HTMLElement);
    assert.equal(label.getAttribute("for"), "observed-machine");
  });

  test("members and visibility are named for role queries after a snapshot", async () => {
    const host = await bootDashboard();
    await host.dispatch("dashboard.model.published", publishedModel("/Phone"));
    assert.ok(getByRole(host, "group", "Members") instanceof HTMLElement);
    assert.ok(getByRole(host, "button", "/Phone") instanceof HTMLButtonElement);
    assert.ok(getByRole(host, "checkbox", "Show /Phone graph") instanceof HTMLInputElement);
    host.remove();
  });
});
