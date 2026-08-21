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

Native HTML/SVG graph nodes represent slash-separated observed state paths.
Ownership is shown with nested machine shells, while SVG edges connect
consecutive observations of the same `hsm.machine.name` when the state changes.
The latest observed state is highlighted. Clicking an edge prefills the send-event
name. Nothing here hard-codes a PhoneBot topology.

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

Low-rate session and transport elements (`bot-dashboard`, `bot-otel-source`)
still use a Companion-style controller: the element is an `HTMLElement`, a
controller owns `hsm.start`, and clicks map to named events.

Interactive graph pieces use gogo's `From(HTMLElement)` split:

1. `From` is exported from `src/hsm-runtime.ts`. Any custom element can
   `extends From(HTMLElement)` and `startMachine(this, model)`.
2. `bot-machine-graph` is that host. It owns hit-testing and graph admission.
3. Sibling machines own the rest: `Renderer` (`clean` / `dirty` / `rendering`)
   coalesces paints; `Panner` (`fixed` / `panning`) writes CSS translate+scale
   synchronously; `Focuser` (`unfocused` / `focused`) is orthogonal to pan.
4. Graph empty/drawing is a separate `Graph` model. Pointer motion does not
   share a dispatch tail with `graph.set`.

Machines:

- `bot-dashboard` → `DashboardController` (`idle` / `live` / `error`);
  the live activity opens `EventSource` on the OTLP stream;
  `dashboard.command.send` posts `/v1/commands`
- `bot-otel-source` → `OtelSourceController` (`idle` / `connecting` / `live` / `error`);
  the composed `bot-otel-source` event is an **effect** of entering `live`
- `bot-machine-graph` → `From(HTMLElement)` host plus `Graph` / `Renderer` /
  `Panner` / `Focuser`; the native HTML/SVG painter draws from Renderer

## Stack

- TypeScript (strict, `noUncheckedIndexedAccess`)
- Autonomous custom elements: `bot-dashboard`, `bot-machine-graph`, `bot-otel-source`
- CSS via component stylesheets plus `src/dashboard.css`
- `@stateforward/hsm.ts` 1.3.3 for every element machine and companion controller
- Native HTML/SVG web-component rendering for nodes and edges
- `@grpc/grpc-js` + `@grpc/proto-loader` for the in-process OTLP/control collector
- Vite for local serve/build
