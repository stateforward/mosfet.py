import type { IncomingMessage, ServerResponse } from "node:http";
import { mkdir, open, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

import {
  mergePublishedModel,
  parseLiveModel,
  parsePublishedModel,
  type PublishedModel,
} from "../src/otel/machines.ts";
import { parseExportTraceServiceRequest } from "../src/otel/otlp.ts";
import { type ObserveSpan } from "../src/otel/span.ts";

export const GRPC_HOST = "127.0.0.1";
const DEFAULT_GRPC_PORT = 4317;
const configuredGrpcPort = Number(process.env["BOT_GRPC_PORT"] ?? DEFAULT_GRPC_PORT);
if (!Number.isInteger(configuredGrpcPort) || configuredGrpcPort < 1 || configuredGrpcPort > 65_535) {
  throw new Error("BOT_GRPC_PORT must be an integer from 1 to 65535");
}
export const GRPC_PORT = configuredGrpcPort;
export const GRPC_ADDRESS = `${GRPC_HOST}:${String(GRPC_PORT)}`;

const RING_CAP = 5000;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const PROTO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../proto");
const MODEL_STORE_ENV = "BOT_MODEL_STORE_PATH";
const DEFAULT_MODEL_STORE_PATH = path.resolve(process.cwd(), ".data/models.json");

export type SpanBatch = {
  observeSpans: ObserveSpan[];
  skipped: number;
};

export type CommandPayload = {
  event_name: string;
  data_json: string;
};

export type CommandResult = {
  result: "accepted" | "no_subscriber" | "error";
  detail: string;
};

export class ObserveRing {
  #spans: ObserveSpan[] = [];
  #skipped = 0;

  snapshot(): SpanBatch {
    return { observeSpans: [...this.#spans], skipped: this.#skipped };
  }

  accept(spans: readonly ObserveSpan[], skipped: number): SpanBatch {
    this.#skipped += skipped;
    if (spans.length > 0) {
      this.#spans.push(...spans);
      if (this.#spans.length > RING_CAP) {
        this.#spans.splice(0, this.#spans.length - RING_CAP);
      }
    }
    return { observeSpans: [...spans], skipped };
  }
}

type CommandStream = {
  write(message: CommandPayload): void;
  on(event: "cancelled" | "error" | "close", listener: () => void): void;
};

export class ModelStore {
  #models = new Map<string, PublishedModel>();
  #writeTail: Promise<void> = Promise.resolve();
  readonly #filePath: string;

  constructor(filePath = process.env[MODEL_STORE_ENV] ?? DEFAULT_MODEL_STORE_PATH) {
    this.#filePath = filePath;
  }

  async load(): Promise<void> {
    let contents: string;
    try {
      contents = await readFile(this.#filePath, "utf8");
    } catch (error: unknown) {
      if (isNodeError(error) && error.code === "ENOENT") {
        return;
      }
      throw error instanceof Error ? error : new Error("model store could not be read");
    }
    let value: unknown;
    try {
      value = JSON.parse(contents) as unknown;
    } catch {
      console.warn("model store ignored corrupt persisted data");
      return;
    }
    const records = isRecord(value) ? value["models"] : value;
    if (!Array.isArray(records)) {
      console.warn("model store ignored persisted data with invalid shape");
      return;
    }
    let invalidRecords = 0;
    for (const record of records) {
      const model = parsePublishedModel(record);
      if (model === null) {
        invalidRecords += 1;
        continue;
      }
      this.put(durableModel(model));
    }
    if (invalidRecords > 0) {
      console.warn(`model store skipped ${String(invalidRecords)} invalid persisted record(s)`);
    }
  }

  persist(): Promise<void> {
    return this.#enqueue(async () => {
      await this.#write(JSON.stringify({ models: this.list() }));
    });
  }

  commit(model: PublishedModel): Promise<PublishedModel> {
    return this.#enqueue(async () => {
      const next = mergePublishedModel(this.#models.get(model.name), model);
      const nextModels = new Map(this.#models);
      nextModels.set(next.name, next);
      await this.#write(JSON.stringify({ models: [...nextModels.values()].sort((left, right) => left.name.localeCompare(right.name)) }));
      this.#models = nextModels;
      return next;
    });
  }

  commitLive(live: {
    name: string;
    component: string;
    state: string;
    live: boolean;
    owner?: string | null;
  }): Promise<PublishedModel> {
    return this.#enqueue(async () => {
      const existing = this.#models.get(live.name);
      const next = mergePublishedModel(existing, {
        name: live.name,
        states: existing?.states ?? [],
        transitions: existing?.transitions ?? [],
        initial: existing?.initial ?? "",
        live: live.live,
        state: live.state,
        component: live.component,
        ...(live.owner === undefined ? {} : { owner: live.owner }),
      });
      const nextModels = new Map(this.#models);
      nextModels.set(next.name, next);
      await this.#write(JSON.stringify({ models: [...nextModels.values()].sort((left, right) => left.name.localeCompare(right.name)) }));
      this.#models = nextModels;
      return next;
    });
  }

  #enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.#writeTail.then(operation, operation);
    this.#writeTail = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  async #write(contents: string): Promise<void> {
    const directory = path.dirname(this.#filePath);
    const temporaryPath = `${this.#filePath}.tmp-${String(process.pid)}-${randomUUID()}`;
    await mkdir(directory, { recursive: true });
    try {
      await writeFile(temporaryPath, `${contents}\n`, "utf8");
      const file = await open(temporaryPath, "r");
      try {
        await file.sync();
      } finally {
        await file.close();
      }
      await rename(temporaryPath, this.#filePath);
      const directoryHandle = await open(directory, "r");
      try {
        await directoryHandle.sync();
      } finally {
        await directoryHandle.close();
      }
    } finally {
      await unlink(temporaryPath).catch(() => undefined);
    }
  }

  put(model: PublishedModel): PublishedModel {
    const next = mergePublishedModel(this.#models.get(model.name), model);
    this.#models.set(next.name, next);
    return next;
  }

  applyLive(live: {
    name: string;
    component: string;
    state: string;
    live: boolean;
    owner?: string | null;
  }): PublishedModel {
    const existing = this.#models.get(live.name);
    return this.put({
      name: live.name,
      states: existing?.states ?? [],
      transitions: existing?.transitions ?? [],
      initial: existing?.initial ?? "",
      live: live.live,
      state: live.state,
      component: live.component,
      ...(live.owner === undefined ? {} : { owner: live.owner }),
    });
  }

  list(): PublishedModel[] {
    return [...this.#models.values()].sort((left, right) => left.name.localeCompare(right.name));
  }
}

function durableModel(model: PublishedModel): PublishedModel {
  const durable: PublishedModel = {
    name: model.name,
    states: model.states,
    transitions: model.transitions,
    initial: model.initial,
  };
  if (model.component !== undefined) {
    durable.component = model.component;
  }
  if (model.owner !== undefined) {
    durable.owner = model.owner;
  }
  return durable;
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}

export class CommandHub {
  #subscribers = new Set<CommandStream>();

  subscribe(stream: CommandStream): void {
    this.#subscribers.add(stream);
    const forget = (): void => {
      this.#subscribers.delete(stream);
    };
    stream.on("cancelled", forget);
    stream.on("error", forget);
    stream.on("close", forget);
  }

  dispatch(command: CommandPayload): CommandResult {
    const eventName = command.event_name.trim();
    if (eventName.length === 0) {
      return { result: "error", detail: "event_name is required" };
    }
    const payload: CommandPayload = { event_name: eventName, data_json: command.data_json };
    if (this.#subscribers.size === 0) {
      return { result: "no_subscriber", detail: "no subscriber" };
    }
    for (const subscriber of this.#subscribers) {
      subscriber.write(payload);
    }
    return { result: "accepted", detail: "accepted" };
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isContainer(value: unknown): value is Record<string, unknown> {
  return (typeof value === "object" || typeof value === "function") && value !== null && !Array.isArray(value);
}

function nested(root: unknown, ...keys: string[]): unknown {
  let current: unknown = root;
  for (const key of keys) {
    if (!isContainer(current)) {
      return undefined;
    }
    current = current[key];
  }
  return current;
}

function writeSse(response: ServerResponse, eventName: string, payload: unknown): void {
  if (response.writableEnded) {
    return;
  }
  response.write(`event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`);
}

function isJsonContentType(value: string | undefined): boolean {
  if (value === undefined) {
    return false;
  }
  const mediaType = value.split(";", 1)[0]?.trim().toLowerCase();
  return mediaType === "application/json";
}

function readJsonBody(request: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on("data", (chunk: Buffer | string) => {
      const buffer = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
      size += buffer.byteLength;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("request body too large"));
        request.destroy();
        return;
      }
      chunks.push(buffer);
    });
    request.on("end", () => {
      if (chunks.length === 0) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown);
      } catch (error: unknown) {
        reject(error instanceof Error ? error : new Error("invalid JSON"));
      }
    });
    request.on("error", reject);
  });
}

function protoJson(value: unknown): unknown {
  return JSON.parse(
    JSON.stringify(value, (_key, item: unknown) => {
      if (Buffer.isBuffer(item)) {
        return item.toString("hex");
      }
      return item;
    }),
  ) as unknown;
}

export function commandFromUnknown(value: unknown): CommandPayload | null {
  if (!isRecord(value) || typeof value["event_name"] !== "string") {
    return null;
  }
  const dataJson = value["data_json"];
  return {
    event_name: value["event_name"],
    data_json: typeof dataJson === "string" ? dataJson : "",
  };
}

type UnaryCallback = (error: grpc.ServiceError | null, response: unknown) => void;

export function loadPackage(): grpc.GrpcObject {
  const definition = protoLoader.loadSync(
    [
      path.join(PROTO_ROOT, "opentelemetry/proto/collector/trace/v1/trace_service.proto"),
      path.join(PROTO_ROOT, "bot/control/v1/control.proto"),
    ],
    {
      keepCase: false,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
      includeDirs: [PROTO_ROOT],
    },
  );
  return grpc.loadPackageDefinition(definition);
}

function serviceDefinition(root: unknown, ...keys: string[]): grpc.ServiceDefinition {
  const service = nested(root, ...keys, "service");
  if (service === undefined) {
    throw new Error(`missing proto service ${keys.join(".")}`);
  }
  return service as grpc.ServiceDefinition;
}

export function exportObserveSpans(
  request: unknown,
  ring: ObserveRing,
  subscribers: ReadonlySet<ServerResponse>,
): void {
  const parsed = parseExportTraceServiceRequest(protoJson(request));
  if (parsed === null) {
    return;
  }
  const accepted = ring.accept(parsed.spans, parsed.skipped);
  if (accepted.observeSpans.length === 0 && accepted.skipped === 0) {
    return;
  }
  for (const subscriber of subscribers) {
    writeSse(subscriber, "spans", accepted);
  }
}

export type CollectorHandles = {
  grpcServer: grpc.Server;
  middleware: (request: IncomingMessage, response: ServerResponse, next: () => void) => void;
};

export async function createCollector(): Promise<CollectorHandles> {
  const ring = new ObserveRing();
  const hub = new CommandHub();
  const models = new ModelStore();
  await models.load();
  const subscribers = new Set<ServerResponse>();
  const loaded = loadPackage();
  const grpcServer = new grpc.Server();
  grpcServer.addService(serviceDefinition(loaded, "opentelemetry", "proto", "collector", "trace", "v1", "TraceService"), {
    Export: (call: { request: unknown }, callback: UnaryCallback) => {
      exportObserveSpans(call.request, ring, subscribers);
      callback(null, {});
    },
  });
  grpcServer.addService(serviceDefinition(loaded, "bot", "control", "v1", "Control"), {
    Subscribe: (stream: CommandStream) => {
      hub.subscribe(stream);
    },
    Dispatch: (call: { request: unknown }, callback: UnaryCallback) => {
      const command = commandFromUnknown(call.request);
      if (command === null) {
        callback(null, { result: "error", detail: "event_name is required" });
        return;
      }
      callback(null, hub.dispatch(command));
    },
  });
  await new Promise<void>((resolve, reject) => {
    grpcServer.bindAsync(GRPC_ADDRESS, grpc.ServerCredentials.createInsecure(), (error) => {
      if (error !== null) {
        reject(error);
        return;
      }
      resolve();
    });
  });

  const writeJson = (response: ServerResponse, status: number, payload: unknown): void => {
    response.statusCode = status;
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify(payload));
  };

  const middleware = (request: IncomingMessage, response: ServerResponse, next: () => void): void => {
    const urlValue = request.url;
    if (urlValue === undefined) {
      next();
      return;
    }
    const url = new URL(urlValue, "http://127.0.0.1");
    if (request.method === "POST" && url.pathname === "/v1/models") {
      void (async () => {
        const contentType = request.headers["content-type"];
        const contentTypeValue = Array.isArray(contentType) ? contentType[0] : contentType;
        if (!isJsonContentType(contentTypeValue)) {
          writeJson(response, 400, { result: "error", detail: "Content-Type application/json required" });
          return;
        }
        let body: unknown;
        try {
          body = await readJsonBody(request);
        } catch {
          writeJson(response, 400, { result: "error", detail: "malformed JSON body" });
          return;
        }
        const model = parsePublishedModel(body);
        if (model === null) {
          writeJson(response, 400, { result: "error", detail: "name, states, transitions, initial required" });
          return;
        }
        let stored: PublishedModel;
        try {
          stored = await models.commit(model);
        } catch {
          writeJson(response, 500, { result: "error", detail: "model persistence failed" });
          return;
        }
        for (const subscriber of subscribers) {
          writeSse(subscriber, "model", stored);
        }
        writeJson(response, 200, { result: "accepted", detail: stored.name });
      })();
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/models/live") {
      void (async () => {
        const contentType = request.headers["content-type"];
        const contentTypeValue = Array.isArray(contentType) ? contentType[0] : contentType;
        if (!isJsonContentType(contentTypeValue)) {
          writeJson(response, 400, { result: "error", detail: "Content-Type application/json required" });
          return;
        }
        let body: unknown;
        try {
          body = await readJsonBody(request);
        } catch {
          writeJson(response, 400, { result: "error", detail: "malformed JSON body" });
          return;
        }
        const live = parseLiveModel(body);
        if (live === null) {
          writeJson(response, 400, { result: "error", detail: "name, component, state, live required" });
          return;
        }
        let stored: PublishedModel;
        try {
          stored = await models.commitLive(live);
        } catch {
          writeJson(response, 500, { result: "error", detail: "model persistence failed" });
          return;
        }
        for (const subscriber of subscribers) {
          writeSse(subscriber, "live", stored);
        }
        writeJson(response, 200, { result: "accepted", detail: stored.name });
      })();
      return;
    }
    if (request.method === "GET" && url.pathname === "/v1/models") {
      writeJson(response, 200, { models: models.list() });
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/commands") {
      void (async () => {
        const contentType = request.headers["content-type"];
        const contentTypeValue = Array.isArray(contentType) ? contentType[0] : contentType;
        if (!isJsonContentType(contentTypeValue)) {
          writeJson(response, 400, { result: "error", detail: "Content-Type application/json required" });
          return;
        }
        let body: unknown;
        try {
          body = await readJsonBody(request);
        } catch {
          writeJson(response, 400, { result: "error", detail: "malformed JSON body" });
          return;
        }
        const command = commandFromUnknown(body);
        if (command === null) {
          writeJson(response, 400, { result: "error", detail: "event_name is required" });
          return;
        }
        writeJson(response, 200, hub.dispatch(command));
      })();
      return;
    }
    if (request.method === "GET" && url.pathname === "/v1/traces/stream") {
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      writeSse(response, "snapshot", { ...ring.snapshot(), models: models.list() });
      subscribers.add(response);
      const forget = (): void => {
        subscribers.delete(response);
      };
      request.on("close", forget);
      request.on("error", forget);
      response.on("close", forget);
      return;
    }
    next();
  };
  return { grpcServer, middleware };
}

type ExportClient = {
  Export(req: unknown, callback: (error: grpc.ServiceError | null, response: unknown) => void): void;
  close(): void;
  waitForReady(deadline: number, callback: (error?: Error) => void): void;
};

export async function exportTraces(request: unknown): Promise<void> {
  const loaded = loadPackage();
  const ctor = nested(loaded, "opentelemetry", "proto", "collector", "trace", "v1", "TraceService");
  if (typeof ctor !== "function") {
    throw new Error("TraceService client missing");
  }
  const Client = ctor as unknown as new (address: string, creds: grpc.ChannelCredentials) => ExportClient;
  const client = new Client(GRPC_ADDRESS, grpc.credentials.createInsecure());
  try {
    await new Promise<void>((resolve, reject) => {
      client.waitForReady(Date.now() + 10_000, (error) => {
        if (error !== undefined) {
          reject(error);
          return;
        }
        resolve();
      });
    });
    await new Promise<void>((resolve, reject) => {
      client.Export(request, (error) => {
        if (error !== null) {
          reject(error);
          return;
        }
        resolve();
      });
    });
  } finally {
    client.close();
  }
}
