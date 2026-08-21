import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { BotOtelSource, registerBotOtelSource } from "../src/elements/bot-otel-source.ts";

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

function badgeOf(host: BotOtelSource): Element {
  const badge = host.shadowRoot?.querySelector("[data-testid=\"live-badge\"]");
  if (badge === null || badge === undefined) {
    throw new Error("live-badge missing");
  }
  return badge;
}

function errorOf(host: BotOtelSource): Element {
  const error = host.shadowRoot?.querySelector("p");
  if (error === null || error === undefined) {
    throw new Error("error status missing");
  }
  return error;
}

describe("bot-otel-source", () => {
  test("accessible name tracks collector phase", async () => {
    const host = document.createElement("bot-otel-source");
    assert.ok(host instanceof BotOtelSource);
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live");
    const badge = badgeOf(host);
    assert.equal(badge.getAttribute("aria-label"), `Collector status: ${host.snapshot().phase}`);
    await host.dispatch("source.connect.requested", { origin: "not-a-url" });
    await waitFor(() => host.snapshot().phase === "error");
    const error = errorOf(host);
    assert.equal(badge.getAttribute("aria-label"), `Collector status: ${host.snapshot().phase}`);
    assert.equal(error.getAttribute("role"), "status");
    assert.equal(badge.getAttribute("aria-describedby"), error.id);
    assert.ok((host.snapshot().errorMessage ?? "").length > 0);
    assert.equal(error.textContent, host.snapshot().errorMessage);
    host.remove();
  });
});
