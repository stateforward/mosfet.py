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

describe("bot-otel-source", () => {
  test("accessible name tracks collector phase", async () => {
    const host = document.createElement("bot-otel-source");
    assert.ok(host instanceof BotOtelSource);
    document.body.append(host);
    await waitFor(() => host.snapshot().phase === "live");
    assert.equal(host.snapshot().errorMessage, null);
    await host.dispatch("source.connect.requested", { origin: "not-a-url" });
    await waitFor(() => host.snapshot().phase === "error");
    assert.ok((host.snapshot().errorMessage ?? "").length > 0);
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
