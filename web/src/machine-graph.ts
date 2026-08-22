import * as hsm from "./hsm.ts";
import { isMachineGraph, parseMachineGraph, type MachineGraph } from "./otel/machines.ts";

export type MachineGraphPhase = "empty" | "drawing";

export type MachineGraphSnapshot = {
  readonly phase: MachineGraphPhase;
  readonly statePath: string;
  readonly graphs: readonly MachineGraph[];
};

export class Graph extends hsm.Instance {
  static readonly setEvent = { name: "graph.set", kind: hsm.Kinds.Event } as const;
  static readonly clearEvent = { name: "graph.clear", kind: hsm.Kinds.Event } as const;
  static readonly drawnEvent = { name: "graph.drawn", kind: hsm.Kinds.Event } as const;
  static readonly clearedEvent = { name: "graph.cleared", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Graph",
    hsm.initial(hsm.target("empty")),
    hsm.transition(hsm.on(Graph.setEvent.name), hsm.target("admit")),
    hsm.choice(
      "admit",
      hsm.transition(
        hsm.guard(Graph.hasGraphs),
        hsm.target("drawing"),
        hsm.effect(Graph.remember),
      ),
      hsm.transition(hsm.target("empty"), hsm.effect(Graph.clear)),
    ),
    hsm.state(
      "empty",
      hsm.entry(Graph.notifyCleared),
      hsm.transition(hsm.on(Graph.clearEvent.name), hsm.effect(Graph.clear)),
    ),
    hsm.state(
      "drawing",
      hsm.entry(Graph.notifyDrawn),
      hsm.transition(hsm.on(Graph.clearEvent.name), hsm.target("../empty"), hsm.effect(Graph.clear)),
    ),
  );

  #graphs: readonly MachineGraph[] = [];

  get graphs(): readonly MachineGraph[] {
    return copyGraphs(this.#graphs);
  }

  snapshot(): MachineGraphSnapshot {
    const statePath = this.state();
    return {
      phase: statePath.endsWith("/drawing") ? "drawing" : "empty",
      statePath,
      graphs: copyGraphs(this.#graphs),
    };
  }

  static hasGraphs(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    if (!hsm.isRecord(event.data) || !Array.isArray(event.data["graphs"])) return false;
    const graphs = event.data["graphs"];
    if (graphs.length === 0) return false;
    for (const item of graphs) {
      if (!isMachineGraph(item)) return false;
    }
    return true;
  }

  static remember(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    const graphs = graphsFromEvent(event);
    if (graphs === null || graphs.length === 0) {
      throw new TypeError("drawing entered without admitted graphs");
    }
    instance.#graphs = copyGraphs(graphs);
  }

  static clear(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    instance.#graphs = [];
  }

  static notifyDrawn(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    void hsm.notifyOwner({
      instance,
      event: hsm.typedEvent({ event: Graph.drawnEvent, data: { graphs: copyGraphs(instance.#graphs) } }),
    }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }

  static notifyCleared(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    void hsm.notifyOwner({ instance, event: hsm.typedEvent({ event: Graph.clearedEvent }) }).catch(hsm.catchFailure(hsm.ownerTarget(instance)));
  }
}

/**
 * Parse an unknown value as a list of machine graphs.
 *
 * Inputs: `value` — JSON-like payload expected to be an array of objects
 * accepted by `parseMachineGraph`.
 * Outputs: a new `MachineGraph[]`, or `null` when `value` is not an array or
 * any item fails `parseMachineGraph`. Never throws. An empty array is success
 * for `[]`; malformed payloads are `null`, not `[]`.
 * Ownership: caller owns `value` and the returned array; this function retains
 * neither. Purity: no I/O and no shared mutable state.
 * Concurrency: runtime-safe. Failure modes: malformed payload => `null`.
 * Units: none. Classification: runtime-safe.
 */
export function parseGraphs(value: unknown): MachineGraph[] | null {
  if (!Array.isArray(value)) return null;
  const graphs: MachineGraph[] = [];
  for (const item of value) {
    const graph = parseMachineGraph(item);
    if (graph === null) return null;
    graphs.push(graph);
  }
  return graphs;
}

/**
 * Read `event.data.graphs` as `MachineGraph[]`.
 *
 * Inputs: an HSM event whose `data` is a record with a `graphs` field.
 * Outputs: the parsed graphs, or `null` when `data` is not a record or
 * `parseGraphs` rejects the field. Never throws.
 * Ownership: caller owns the event; this function retains nothing.
 * Purity: no I/O. Concurrency: runtime-safe.
 * Failure modes: missing or malformed `graphs` => `null`.
 * Units: none. Classification: runtime-safe.
 */
export function graphsFromEvent(event: hsm.Event): MachineGraph[] | null {
  if (!hsm.isRecord(event.data)) return null;
  return parseGraphs(event.data["graphs"]);
}

function copyGraph(graph: MachineGraph): MachineGraph {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes.map((node) => ({ ...node })) : [];
  const edges = Array.isArray(graph.edges) ? graph.edges.map((edge) => ({ ...edge })) : [];
  return { ...graph, nodes, edges };
}

/**
 * Copy `graphs` so later mutation of the caller array or graph items cannot
 * change the returned list.
 *
 * Inputs: a list of machine graphs. Outputs: a new array whose items, node
 * arrays, and edge arrays are copies. Never throws.
 * Ownership: caller owns `graphs`; this function retains nothing. The caller
 * owns the returned array.
 * Purity: no I/O. Concurrency: runtime-safe.
 * Failure modes: missing `nodes`/`edges` arrays become empty copies.
 * Units: none. Classification: runtime-safe.
 */
export function copyGraphs(graphs: readonly MachineGraph[]): MachineGraph[] {
  return graphs.map(copyGraph);
}
