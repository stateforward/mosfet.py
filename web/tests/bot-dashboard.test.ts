import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { BotDashboard, registerBotDashboard } from "../src/elements/bot-dashboard.ts";
import { registerBotMachineGraph } from "../src/elements/bot-machine-graph/index.ts";
import { registerBotOtelSource } from "../src/elements/bot-otel-source.ts";
import { registerFlowElements } from "../src/flow/register.ts";
import { streamSource } from "../src/otel/source.ts";

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

describe("bot-dashboard inspector focus", () => {
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
});
