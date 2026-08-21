import "./dom.ts";
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as hsm from "../src/hsm.ts";

describe("hsm.from(HTMLElement)", () => {
  test("starts, dispatches, and stops on a mixed host", async () => {
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

    const host = new Host();
    hsm.start(host, Host.model);
    assert.equal(typeof host.dispatch, "function");
    assert.equal(typeof host.context, "function");
    assert.match(host.state(), /\/idle$/);
    host.dispatch(hsm.namedEvent(Host.pingEvent.name));
    assert.match(host.state(), /\/active$/);
    await hsm.stop(host);
    assert.equal(host.state(), "");
  });
});
