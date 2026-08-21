import { parseExportTraceServiceRequest } from "./otlp.ts";
import { type ObserveSpan } from "./span.ts";

export type { ObserveParseResult, ObserveSpan } from "./span.ts";
export { OBSERVE_SPAN_NAME, parseObserveSpan } from "./span.ts";
export { parseExportTraceServiceRequest, parseOtelSpanBatch, type OtelSpanBatch } from "./otlp.ts";

export type MachineStateNode = {
  path: string;
  parent: string | null;
  label: string;
};

export type MachineEdge = {
  source: string;
  target: string;
  eventName: string;
  count: number;
  lastFired: boolean;
};

export type MachineGraph = {
  name: string;
  owner?: string | null;
  componentName: string;
  currentState: string;
  lastEventName: string;
  nodes: MachineStateNode[];
  edges: MachineEdge[];
  observationCount: number;
};

export type OtelDocument = {
  machines: MachineGraph[];
  selectedMachine: string | null;
  observeCount: number;
  skippedCount: number;
};

export type PublishedState = {
  qualified_name: string;
  parent: string;
  initial: string;
};

export type PublishedTransition = {
  source: string;
  target: string;
  events: string[];
};

export type PublishedModel = {
  name: string;
  states: PublishedState[];
  transitions: PublishedTransition[];
  initial: string;
  live?: boolean;
  state?: string;
  component?: string;
  owner?: string | null;
};

export type LiveModel = {
  name: string;
  component: string;
  state: string;
  live: boolean;
  owner?: string | null;
};

const INVALID_OWNER = Symbol("invalid owner");

function parseOwner(value: Record<string, unknown>): string | null | undefined | typeof INVALID_OWNER {
  if (!Object.hasOwn(value, "owner")) {
    return undefined;
  }
  const owner = value["owner"];
  if (owner === null) {
    return null;
  }
  if (typeof owner === "string" && owner.length > 0) {
    return owner;
  }
  return INVALID_OWNER;
}

function stateSegments(path: string): string[] {
  return path.split("/").filter((part) => part.length > 0);
}

function ancestorPaths(path: string): string[] {
  const segments = stateSegments(path);
  const paths: string[] = [];
  for (let index = 0; index < segments.length; index += 1) {
    paths.push(`/${segments.slice(0, index + 1).join("/")}`);
  }
  return paths;
}

function nodeLabel(path: string): string {
  const segments = stateSegments(path);
  const last = segments[segments.length - 1];
  return last ?? path;
}

function parentPath(path: string): string | null {
  const segments = stateSegments(path);
  if (segments.length <= 1) {
    return null;
  }
  return `/${segments.slice(0, -1).join("/")}`;
}

function publishedParent(parent: string): string | null {
  if (parent === "" || parent === "/") {
    return null;
  }
  return parent;
}

function edgeKey(source: string, target: string, eventName: string): string {
  return `${source}\0${target}\0${eventName}`;
}

export function parsePublishedState(value: unknown): PublishedState | null {
  if (
    !isRecord(value) ||
    typeof value["qualified_name"] !== "string" ||
    typeof value["parent"] !== "string" ||
    typeof value["initial"] !== "string"
  ) {
    return null;
  }
  return {
    qualified_name: value["qualified_name"],
    parent: value["parent"],
    initial: value["initial"],
  };
}

export function parsePublishedTransition(value: unknown): PublishedTransition | null {
  if (
    !isRecord(value) ||
    typeof value["source"] !== "string" ||
    typeof value["target"] !== "string" ||
    !Array.isArray(value["events"])
  ) {
    return null;
  }
  const events: string[] = [];
  for (const item of value["events"]) {
    if (typeof item !== "string") {
      return null;
    }
    events.push(item);
  }
  return { source: value["source"], target: value["target"], events };
}

export function parsePublishedModel(value: unknown): PublishedModel | null {
  if (
    !isRecord(value) ||
    typeof value["name"] !== "string" ||
    value["name"].length === 0 ||
    typeof value["initial"] !== "string" ||
    !Array.isArray(value["states"]) ||
    !Array.isArray(value["transitions"])
  ) {
    return null;
  }
  const owner = parseOwner(value);
  if (owner === INVALID_OWNER) {
    return null;
  }
  const states: PublishedState[] = [];
  for (const item of value["states"]) {
    const state = parsePublishedState(item);
    if (state === null) {
      return null;
    }
    states.push(state);
  }
  const transitions: PublishedTransition[] = [];
  for (const item of value["transitions"]) {
    const transition = parsePublishedTransition(item);
    if (transition === null) {
      return null;
    }
    transitions.push(transition);
  }
  const model: PublishedModel = {
    name: value["name"],
    states,
    transitions,
    initial: value["initial"],
  };
  if (typeof value["live"] === "boolean") {
    model.live = value["live"];
  }
  if (typeof value["state"] === "string") {
    model.state = value["state"];
  }
  if (typeof value["component"] === "string") {
    model.component = value["component"];
  }
  if (owner !== undefined) {
    model.owner = owner;
  }
  return model;
}

export function parseLiveModel(value: unknown): LiveModel | null {
  if (
    !isRecord(value) ||
    typeof value["name"] !== "string" ||
    value["name"].length === 0 ||
    typeof value["component"] !== "string" ||
    typeof value["state"] !== "string" ||
    typeof value["live"] !== "boolean"
  ) {
    return null;
  }
  const owner = parseOwner(value);
  if (owner === INVALID_OWNER) {
    return null;
  }
  const model: LiveModel = {
    name: value["name"],
    component: value["component"],
    state: value["state"],
    live: value["live"],
  };
  if (owner !== undefined) {
    model.owner = owner;
  }
  return model;
}

export function mergePublishedModel(
  existing: PublishedModel | undefined,
  incoming: PublishedModel,
): PublishedModel {
  if (existing === undefined) {
    return incoming;
  }
  const incomingHasTopology =
    incoming.states.length > 0 || incoming.transitions.length > 0 || incoming.initial.length > 0;
  const merged: PublishedModel = {
    name: incoming.name,
    states: incomingHasTopology ? incoming.states : existing.states,
    transitions: incomingHasTopology ? incoming.transitions : existing.transitions,
    initial: incomingHasTopology && incoming.initial.length > 0 ? incoming.initial : existing.initial,
  };
  const live = incoming.live !== undefined ? incoming.live : existing.live;
  const state = incoming.state !== undefined && incoming.state.length > 0 ? incoming.state : existing.state;
  const component =
    incoming.component !== undefined && incoming.component.length > 0
      ? incoming.component
      : existing.component;
  const owner =
    incoming.owner !== undefined
      ? incoming.owner
      : incoming.live !== undefined
        ? undefined
        : existing.owner;
  if (live !== undefined) {
    merged.live = live;
  }
  if (state !== undefined) {
    merged.state = state;
  }
  if (component !== undefined) {
    merged.component = component;
  }
  if (owner !== undefined) {
    merged.owner = owner;
  }
  return merged;
}

export function parsePublishedModels(value: unknown): PublishedModel[] | null {
  if (Array.isArray(value)) {
    const models: PublishedModel[] = [];
    for (const item of value) {
      const model = parsePublishedModel(item);
      if (model === null) {
        return null;
      }
      models.push(model);
    }
    return models;
  }
  if (isRecord(value) && Array.isArray(value["models"])) {
    return parsePublishedModels(value["models"]);
  }
  const single = parsePublishedModel(value);
  return single === null ? null : [single];
}

export function graphFromPublishedModel(model: PublishedModel): MachineGraph {
  const nodes = model.states.map((state) => ({
    path: state.qualified_name,
    parent: publishedParent(state.parent),
    label: nodeLabel(state.qualified_name),
  }));
  const edges: MachineEdge[] = [];
  for (const transition of model.transitions) {
    const names = transition.events.length === 0 ? [""] : transition.events;
    for (const eventName of names) {
      edges.push({
        source: transition.source,
        target: transition.target,
        eventName,
        count: 0,
        lastFired: false,
      });
    }
  }
  return {
    name: model.name,
    ...(model.owner === undefined ? {} : { owner: model.owner }),
    componentName: model.component && model.component.length > 0 ? model.component : nodeLabel(model.name),
    currentState: model.live === true ? (model.state ?? "") : "",
    lastEventName: "",
    nodes,
    edges,
    observationCount: 0,
  };
}

export function overlayObserve(base: MachineGraph, spans: readonly ObserveSpan[]): MachineGraph {
  const folded = foldMachine(base.name, [...spans]);
  if (folded === null) {
    return base;
  }
  const nodes = new Map(base.nodes.map((node) => [node.path, node]));
  for (const node of folded.nodes) {
    if (!nodes.has(node.path)) {
      nodes.set(node.path, node);
    }
  }
  const edges = new Map(
    base.edges.map((edge) => [edgeKey(edge.source, edge.target, edge.eventName), { ...edge }]),
  );
  for (const edge of folded.edges) {
    const key = edgeKey(edge.source, edge.target, edge.eventName);
    const existing = edges.get(key);
    if (existing !== undefined) {
      existing.count = edge.count;
      existing.lastFired = edge.lastFired;
      continue;
    }
    edges.set(key, { ...edge });
  }
  return {
    name: base.name,
    ...(base.owner === undefined ? {} : { owner: base.owner }),
    componentName: folded.componentName || base.componentName,
    currentState: folded.currentState,
    lastEventName: folded.lastEventName,
    nodes: [...nodes.values()],
    edges: [...edges.values()],
    observationCount: folded.observationCount,
  };
}

function spanTime(span: ObserveSpan): string {
  return span.start_time ?? span.timestamp ?? "";
}

function compareSpans(left: ObserveSpan, right: ObserveSpan): number {
  const leftTime = spanTime(left);
  const rightTime = spanTime(right);
  if (leftTime < rightTime) {
    return -1;
  }
  if (leftTime > rightTime) {
    return 1;
  }
  return 0;
}

function foldMachine(name: string, spans: ObserveSpan[]): MachineGraph | null {
  if (spans.length === 0) {
    return null;
  }
  const ordered = [...spans].sort(compareSpans);
  const last = ordered[ordered.length - 1];
  if (last === undefined) {
    return null;
  }
  const nodes = new Map<string, MachineStateNode>();
  for (const span of ordered) {
    for (const path of ancestorPaths(span.attributes["hsm.machine.state"])) {
      if (!nodes.has(path)) {
        nodes.set(path, {
          path,
          parent: parentPath(path),
          label: nodeLabel(path),
        });
      }
    }
  }
  const edges = new Map<string, MachineEdge>();
  let lastEdgeKey: string | null = null;
  for (let index = 1; index < ordered.length; index += 1) {
    const previous = ordered[index - 1];
    const current = ordered[index];
    if (previous === undefined || current === undefined) {
      continue;
    }
    const source = previous.attributes["hsm.machine.state"];
    const target = current.attributes["hsm.machine.state"];
    if (source === target) {
      continue;
    }
    const eventName = current.attributes["hsm.event.name"];
    const key = `${source}\0${target}\0${eventName}`;
    lastEdgeKey = key;
    const existing = edges.get(key);
    if (existing !== undefined) {
      existing.count += 1;
      continue;
    }
    edges.set(key, { source, target, eventName, count: 1, lastFired: false });
  }
  if (lastEdgeKey !== null) {
    const lastEdge = edges.get(lastEdgeKey);
    if (lastEdge !== undefined) {
      lastEdge.lastFired = true;
    }
  }
  return {
    name,
    componentName: last.attributes["bot.component.name"],
    currentState: last.attributes["hsm.machine.state"],
    lastEventName: last.attributes["hsm.event.name"],
    nodes: [...nodes.values()],
    edges: [...edges.values()],
    observationCount: ordered.length,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStateNodeShape(value: unknown): boolean {
  if (!isRecord(value) || typeof value["path"] !== "string" || typeof value["label"] !== "string") {
    return false;
  }
  const parent = value["parent"];
  return parent === null || typeof parent === "string";
}

function parseStateNode(value: unknown): MachineStateNode | null {
  if (!isRecord(value) || typeof value["path"] !== "string" || typeof value["label"] !== "string") {
    return null;
  }
  const parent = value["parent"];
  if (parent !== null && typeof parent !== "string") {
    return null;
  }
  return { path: value["path"], parent, label: value["label"] };
}

function isEdgeShape(value: unknown): boolean {
  return isRecord(value)
    && typeof value["source"] === "string"
    && typeof value["target"] === "string"
    && typeof value["eventName"] === "string"
    && typeof value["count"] === "number"
    && typeof value["lastFired"] === "boolean";
}

function parseEdge(value: unknown): MachineEdge | null {
  if (
    !isRecord(value)
    || typeof value["source"] !== "string"
    || typeof value["target"] !== "string"
    || typeof value["eventName"] !== "string"
    || typeof value["count"] !== "number"
    || typeof value["lastFired"] !== "boolean"
  ) {
    return null;
  }
  return {
    source: value["source"],
    target: value["target"],
    eventName: value["eventName"],
    count: value["count"],
    lastFired: value["lastFired"],
  };
}

/**
 * Predicate: `value` has the `MachineGraph` field shape.
 *
 * Inputs: JSON-like candidate. Outputs: true when required fields and nested
 * node/edge items type-check. Never allocates node or edge arrays.
 * Ownership: caller owns `value`. Purity: no I/O. Failure modes: malformed
 * payload => false. Classification: runtime-safe.
 */
export function isMachineGraph(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if (
    typeof value["name"] !== "string"
    || typeof value["componentName"] !== "string"
    || typeof value["currentState"] !== "string"
    || typeof value["lastEventName"] !== "string"
    || typeof value["observationCount"] !== "number"
    || !Array.isArray(value["nodes"])
    || !Array.isArray(value["edges"])
  ) {
    return false;
  }
  if (parseOwner(value) === INVALID_OWNER) {
    return false;
  }
  for (const item of value["nodes"]) {
    if (!isStateNodeShape(item)) {
      return false;
    }
  }
  for (const item of value["edges"]) {
    if (!isEdgeShape(item)) {
      return false;
    }
  }
  return true;
}

export function parseMachineGraph(value: unknown): MachineGraph | null {
  if (!isMachineGraph(value) || !isRecord(value)) {
    return null;
  }
  const owner = parseOwner(value);
  if (owner === INVALID_OWNER) {
    return null;
  }
  const name = value["name"];
  const componentName = value["componentName"];
  const currentState = value["currentState"];
  const lastEventName = value["lastEventName"];
  const observationCount = value["observationCount"];
  const rawNodes = value["nodes"];
  const rawEdges = value["edges"];
  if (
    typeof name !== "string"
    || typeof componentName !== "string"
    || typeof currentState !== "string"
    || typeof lastEventName !== "string"
    || typeof observationCount !== "number"
    || !Array.isArray(rawNodes)
    || !Array.isArray(rawEdges)
  ) {
    return null;
  }
  const nodes: MachineStateNode[] = [];
  for (const item of rawNodes) {
    const node = parseStateNode(item);
    if (node === null) {
      return null;
    }
    nodes.push(node);
  }
  const edges: MachineEdge[] = [];
  for (const item of rawEdges) {
    const edge = parseEdge(item);
    if (edge === null) {
      return null;
    }
    edges.push(edge);
  }
  const graph: MachineGraph = {
    name,
    componentName,
    currentState,
    lastEventName,
    nodes,
    edges,
    observationCount,
  };
  if (owner !== undefined) {
    graph.owner = owner;
  }
  return graph;
}

export function foldMachines(spans: readonly ObserveSpan[]): MachineGraph[] {
  const grouped = new Map<string, ObserveSpan[]>();
  for (const span of spans) {
    const name = span.attributes["hsm.machine.name"];
    const bucket = grouped.get(name);
    if (bucket === undefined) {
      grouped.set(name, [span]);
      continue;
    }
    bucket.push(span);
  }
  const machines: MachineGraph[] = [];
  for (const [name, group] of grouped) {
    const graph = foldMachine(name, group);
    if (graph !== null) {
      machines.push(graph);
    }
  }
  machines.sort((left, right) => left.name.localeCompare(right.name));
  return machines;
}

export function documentFromOtlp(value: unknown, selectedMachine?: string | null): OtelDocument | null {
  const parsed = parseExportTraceServiceRequest(value);
  if (parsed === null) {
    return null;
  }
  return documentFromSpans(parsed.spans, parsed.skipped, selectedMachine);
}

export function documentFromSpans(
  spans: readonly ObserveSpan[],
  skippedCount: number,
  selectedMachine?: string | null,
  models: readonly PublishedModel[] = [],
): OtelDocument {
  const observed = foldMachines(spans);
  const published = new Map(models.map((model) => [model.name, graphFromPublishedModel(model)]));
  const names = new Set<string>([...published.keys(), ...observed.map((machine) => machine.name)]);
  const machines: MachineGraph[] = [];
  for (const name of names) {
    const base = published.get(name);
    const group = spans.filter((span) => span.attributes["hsm.machine.name"] === name);
    if (base !== undefined) {
      machines.push(overlayObserve(base, group));
      continue;
    }
    const folded = observed.find((machine) => machine.name === name);
    if (folded !== undefined) {
      machines.push(folded);
    }
  }
  machines.sort((left, right) => left.name.localeCompare(right.name));
  const requested = selectedMachine ?? null;
  const hasRequested = requested !== null && machines.some((machine) => machine.name === requested);
  const fallback = machines[0]?.name ?? null;
  return {
    machines,
    selectedMachine: hasRequested ? requested : fallback,
    observeCount: spans.length,
    skippedCount,
  };
}

export function machineByName(document: OtelDocument, name: string | null): MachineGraph | null {
  if (name === null) {
    return document.machines[0] ?? null;
  }
  return document.machines.find((machine) => machine.name === name) ?? null;
}
