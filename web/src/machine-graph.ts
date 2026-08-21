import * as hsm from "./hsm.ts";
import { parseMachineGraph, type MachineGraph } from "./otel/machines.ts";

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
      hsm.transition(hsm.on(Graph.setEvent.name), hsm.target("../admit")),
    ),
    hsm.state(
      "drawing",
      hsm.entry(Graph.notifyDrawn),
      hsm.transition(hsm.on(Graph.setEvent.name), hsm.target("../admit")),
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
    const graphs = graphsFromEvent(event);
    return graphs !== null && graphs.length > 0;
  }

  static remember(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    const graphs = graphsFromEvent(event);
    if (graphs === null || graphs.length === 0) {
      instance.#graphs = [];
      return;
    }
    instance.#graphs = copyGraphs(graphs);
  }

  static clear(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    instance.#graphs = [];
  }

  static notifyDrawn(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph) || instance.#graphs.length === 0) return;
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

export function graphsFromEvent(event: hsm.Event): MachineGraph[] | null {
  if (!hsm.isRecord(event.data)) return null;
  return parseGraphs(event.data["graphs"]);
}

function copyGraph(graph: MachineGraph): MachineGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => ({ ...node })),
    edges: graph.edges.map((edge) => ({ ...edge })),
  };
}

export function copyGraphs(graphs: readonly MachineGraph[]): MachineGraph[] {
  return graphs.map(copyGraph);
}
