import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { BotOtelSource, registerBotOtelSource } from "../src/elements/bot-otel-source.ts";
import { getByRole } from "./by-role.ts";

registerBotOtelSource();

const YIELD_MS = 0;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    if (predicate()) return;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, YIELD_MS);
    });
  }
  throw new Error("timed out waiting for bot-otel-source");
}

const publicEventBubbles = true;
const publicEventComposed = true;
const publicEventCancelable = false;
const atLeastOneReady = 1;

describe("bot-otel-source", () => {
  test("ready CustomEvent is non-cancelable and shares the host source handle", async () => {
    const host = document.createElement("bot-otel-source");
    assert.ok(host instanceof BotOtelSource);
    const ready: Array<{ cancelable: boolean; bubbles: boolean; composed: boolean; source: unknown }> = [];
    host.addEventListener("bot-otel-source", (event: Event) => {
      if (!(event instanceof CustomEvent)) return;
      ready.push({
        cancelable: event.cancelable,
        bubbles: event.bubbles,
        composed: event.composed,
        source: event.detail?.source,
      });
    });
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live");
    assert.ok(ready.length >= atLeastOneReady);
    for (const event of ready) {
      assert.equal(event.cancelable, publicEventCancelable);
      assert.equal(event.bubbles, publicEventBubbles);
      assert.equal(event.composed, publicEventComposed);
      assert.equal(event.source, host.snapshot().source);
    }
    host.remove();
  });

  test("accessible name tracks collector phase", async () => {
    const host = document.createElement("bot-otel-source");
    assert.ok(host instanceof BotOtelSource);
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live");
    assert.equal(host.snapshot().errorMessage, null);
    const live = getByRole(host, "button", "Collector status: live");
    assert.ok(live instanceof HTMLButtonElement);
    assert.ok(getByRole(host, "status", "Collector error") instanceof HTMLElement);
    await host.dispatch("source.connect.requested", { origin: "not-a-url" });
    await waitFor(() => host.snapshot().phase === "error");
    assert.ok((host.snapshot().errorMessage ?? "").length > 0);
    assert.ok(getByRole(host, "button", "Collector status: error") instanceof HTMLButtonElement);
    host.remove();
  });

  test("remove then append still dispatches connect", async () => {
    const host = document.createElement("bot-otel-source");
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live");
    host.remove();
    await waitFor(() => host.snapshot().statePath.includes("/disconnected"));
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live" || host.snapshot().phase === "connecting");
    await host.dispatch("source.connect.requested", { origin: "http://localhost" });
    await waitFor(() => host.snapshot().phase === "live" || host.snapshot().phase === "error");
    assert.notEqual(host.snapshot().phase, "idle");
    host.remove();
  });
});
