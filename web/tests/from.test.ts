import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as library from "@stateforward/hsm.ts";
import * as hsm from "../src/hsm.ts";

describe("hsm.from(HTMLElement)", () => {
  test("starts, dispatches, and stops on a defined custom element", async () => {
    class Host extends hsm.from(HTMLElement) {
      static readonly pingEvent = { name: "ping", kind: hsm.Kinds.Event } as const;
      static readonly model = hsm.define(
        "Host",
        hsm.initial(hsm.target("idle")),
        hsm.state(
          "idle",
          hsm.transition(hsm.on(Host.pingEvent.name), hsm.target("../active")),
        ),
        hsm.state("active"),
      );
    }

    if (customElements.get("test-host") === undefined) {
      customElements.define("test-host", Host);
    }
    const host = document.createElement("test-host");
    assert.ok(host instanceof Host);
    assert.equal(host instanceof library.Instance, false);
    hsm.start(host, Host.model);
    assert.equal(typeof host.dispatch, "function");
    assert.equal(typeof host.context, "function");
    assert.match(host.state(), /\/idle$/);
    host.dispatch(hsm.typedEvent(Host.pingEvent));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
    assert.equal(host.state(), "");
    hsm.start(host, Host.model);
    assert.match(host.state(), /\/idle$/);
    host.dispatch(hsm.typedEvent(Host.pingEvent));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
  });
});
