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

  graphs: readonly MachineGraph[] = [];

  snapshot(): MachineGraphSnapshot {
    const statePath = this.state();
    return {
      phase: statePath.endsWith("/drawing") ? "drawing" : "empty",
      statePath,
      graphs: this.graphs,
    };
  }

  admit(value: unknown): MachineGraphSnapshot {
    this.dispatch(hsm.typedEvent(Graph.setEvent, { graphs: value }));
    return this.snapshot();
  }

  static hasGraphs(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event): boolean {
    const graphs = graphsFromEvent(event);
    return graphs !== null && graphs.length > 0;
  }

  static remember(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    const graphs = graphsFromEvent(event);
    if (graphs === null || graphs.length === 0) {
      instance.graphs = [];
      return;
    }
    instance.graphs = graphs;
  }

  static clear(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    instance.graphs = [];
  }

  static notifyDrawn(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph) || instance.graphs.length === 0) return;
    hsm.notifyOwner({
      instance,
      event: hsm.typedEvent(Graph.drawnEvent, { graphs: instance.graphs }),
    });
  }

  static notifyCleared(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    hsm.notifyOwner({ instance, event: hsm.typedEvent(Graph.clearedEvent) });
  }
}

function parseGraphs(value: unknown): MachineGraph[] | null {
  if (!Array.isArray(value)) return null;
  const graphs: MachineGraph[] = [];
  for (const item of value) {
    const graph = parseMachineGraph(item);
    if (graph === null) return null;
    graphs.push(graph);
  }
  return graphs;
}

function graphsFromEvent(event: hsm.Event): MachineGraph[] | null {
  const data = event.data;
  if (typeof data !== "object" || data === null) return null;
  return parseGraphs((data as Record<string, unknown>)["graphs"]);
}
