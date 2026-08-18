# bot HSM dashboard

Standalone TypeScript page that folds live OpenTelemetry spans into a hierarchical
state diagram. It does not reimplement the Python bot.

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

Compound cytoscape nodes are slash-separated observed state paths. Edges are
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
Writes use an atomic file replacement. A model POST returns an error if the
store cannot be updated, so the caller can retry without receiving a false
success.

Open the printed local URL. The dashboard auto-connects to
`/v1/traces/stream`. Live observe spans appear as the bot exports them.
Send a named event from the inspector; with a bot attached to an environment
the collector fans that command to `Control.Subscribe` and the bot
`hsm.dispatch_all`s it.

## Companion controller pattern

Every custom element follows the Companion `Controller` + `Runtime` split
(`companion/web/src/app/companion-app-state.ts`,
`companion/web/src/components/companion-app-element.ts`):

1. The element extends `HTMLElement`, never `hsm.Instance`.
2. A controller owns `hsm.start(new Runtime(), model)`.
3. `connectedCallback` constructs the controller and binds gestures.
4. Gestures use `data-event="…"` or an equivalent named event and call
   `controller.dispatch(eventName)`. They do not mutate fields and then maybe
   dispatch.
5. `#render(controller.snapshot())` projects `machine.takeSnapshot().state`
   plus documented view fields.
6. `disconnectedCallback` calls `controller.stop()` → `hsm.stop(machine)`.

Machines:

- `bot-dashboard` → `DashboardController` (`idle` / `live` / `error`);
  the live activity opens `EventSource` on the OTLP stream;
  `dashboard.command.send` posts `/v1/commands`
- `bot-otel-source` → `OtelSourceController` (`idle` / `connecting` / `live` / `error`);
  the composed `bot-otel-source` event is an **effect** of entering `live`
- `bot-machine-graph` → `MachineGraphController` (`empty` / `drawing`);
  cytoscape create/update/destroy runs from activities/effects

## Stack

- TypeScript (strict, `noUncheckedIndexedAccess`)
- Autonomous custom elements: `bot-dashboard`, `bot-machine-graph`, `bot-otel-source`
- CSS via component stylesheets plus `src/dashboard.css`
- `@stateforward/hsm.ts` 1.3.3 for every element controller
- cytoscape as the graph renderer only
- `@grpc/grpc-js` + `@grpc/proto-loader` for the in-process OTLP/control collector
- Vite for local serve/build
