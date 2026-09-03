# Environment workspace dashboard

Standalone TypeScript page that folds live OpenTelemetry spans into a composed
environment map. It does not reimplement the Python bot.

The Vite dev server is the collector. Bots export finished spans to OTLP gRPC
`TraceService.Export` on `127.0.0.1:4317`. The page subscribes to
`GET /v1/traces/stream` (SSE) and folds `bot.hsm.observe` spans into the graph.
JSONL is not a dashboard transport. Commands go back to the bot over a second
gRPC service on the same listener (`bot.control.v1.Control`); the browser posts
`POST /v1/commands` as an HTTP gateway.

The only graph source of truth is `bot.hsm.observe` spans. Low-cardinality
attributes come from `src/bot/telemetry/hsm.py` `observation_attributes`:

- `hsm.machine.name`
- `hsm.machine.state`
- `bot.component.name`
- `hsm.event.name`
- `hsm.event.kind`
- `hsm.observation.occurrence`
- `bot.outcome`

`flow-graph` custom elements represent slash-separated observed state paths.
Ownership is shown with nested machine shells, while edges connect consecutive
observations of the same `hsm.machine.name` when the state changes. The latest
observed state is highlighted. Clicking an edge prefills the send-event name.
Nothing here hard-codes a PhoneBot topology.

## Setup

```sh
cd web
npm install
npx playwright install chromium
npm run typecheck
npm test
npm run test:e2e
npm run build
```

## Live

Serve the page, the OTLP gRPC collector, and the command gateway:

```sh
cd web
npm run dev
```

Point a bot at the collector:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

Published models are persisted by the collector and reloaded when Vite starts.
The default store is `.data/models.json` (ignored by git); override it with
`BOT_MODEL_STORE_PATH` when the collector needs a different durable location.
The OTLP/control gRPC listener defaults to `127.0.0.1:4317`; override its port
with `BOT_GRPC_PORT` when an isolated collector is needed.
Writes use an atomic file replacement. A model POST returns an error if the
store cannot be updated, so the caller can retry without receiving a false
success. The collector is the single owner of this file: concurrent
cross-process writers are not supported. The file and its containing directory
are synchronized before a successful write is acknowledged.

Open the printed local URL. The dashboard auto-connects to
`/v1/traces/stream`. Live observe spans appear as the bot exports them.
Send a named event from the inspector; with a bot attached to an environment
the collector fans that command to `Control.Subscribe` and the bot
`hsm.dispatch_all`s it.

## Element machines

Studio elements are autonomous custom elements that `extends hsm.from(HTMLElement)`.
`hsm.from` copies `Instance.prototype` onto the host so
`hsm.start({ instance: this, model })` does not require `instanceof Instance`.
Start in `connectedCallback`, stop in `disconnectedCallback`.

The graph library lives in `src/flow/` and ports React Flow's public surface as
custom elements (`flow-graph`, `flow-node`, `flow-edge`, `flow-handle`,
`flow-node-resizer`, `flow-node-resize-control`, `flow-background`, `flow-controls`,
`flow-minimap`). Sibling HSMs own orthogonal behavior:

- `Renderer` (`clean` / `dirty` / `rendering`) coalesces paints
- `Panner` (`fixed` / `panning`) writes CSS `translate+scale` synchronously
- `Dragger` (`idle` / `dragging`) samples node drag per pointermove
- `Resizer` (`idle` / `resizing/pointer` / `resizing/keyboard`) samples node
  resize per pointermove or typed `resize_key_step`. Pointer and keyboard are
  exclusive nested sessions. `nodesResizable` defaults true. Selected nodes
  show eight named resize controls (`n`/`s`/`e`/`w`/`ne`/`nw`/`se`/`sw`).
  Observe-only events: `flow-node-resize-start`, `flow-node-resize`,
  `flow-node-resize-end`. Keyboard contract: each control's inner button is
  focusable and labeled `Resize <direction>`. Enter or Space on a control
  starts a resize in that direction (origin = the node's current bounds);
  ArrowUp/ArrowDown/ArrowLeft/ArrowRight step-resize by 1px in world units
  per keypress (the shared min/max/aspect clamp logic applies); Escape or a
  second Enter ends the resize. Keys arrive as typed `resize_key` events on
  `FlowGraph`; Pointer topology selects start/step/end. Removing the node,
  deselecting it, or turning `nodesResizable` off ends the session. A click
  originating from a control button never activates the node.
- `Focuser` (`unfocused` / `focused`)
- `Selection` (`none` / `picking` / `box`)
- `Connection` (`idle` / `connecting`)

`bot-machine-graph` composes `flow-graph` plus background and controls. It maps
`MachineGraph` layout onto flow nodes and edges. Pointer pan does not share a
dispatch tail with graph admission.

Machines:

- `bot-dashboard` → `Dashboard` (`idle` / `live` / `error`);
  the live activity opens `EventSource` on the OTLP stream;
  `dashboard.command.send` posts `/v1/commands`
- `bot-otel-source` → `OtelSource` (`idle` / `connecting` / `live` / `error`);
  the composed `bot-otel-source` event is an **effect** of entering `live`
- `bot-machine-graph` → `hsm.from(HTMLElement)` host plus `Graph` admission;
  `flow-graph` owns pan, paint, selection, and connection

## Library

Package exports (TypeScript source):

- `./hsm` — `@stateforward/hsm.ts` plus `from` / wrapped `start` (custom-element host protocol; not `instanceof Instance`)
- `./flow` — flow custom-element library
- `./elements` — studio element registration

## Stack

- TypeScript (strict, `noUncheckedIndexedAccess`)
- Autonomous custom elements: `bot-dashboard`, `bot-machine-graph`, `bot-otel-source`, `flow-*`
- CSS via constructable stylesheets plus `src/dashboard.css`
- `@stateforward/hsm.ts` 1.3.3 for every element machine
- Native HTML/SVG rendering through flow node and edge custom elements
- `@grpc/grpc-js` + `@grpc/proto-loader` for the in-process OTLP/control collector
- Vite for local serve/build
