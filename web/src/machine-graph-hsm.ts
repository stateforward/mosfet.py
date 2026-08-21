import * as hsm from "@stateforward/hsm.ts";

import { namedEvent, reportHsmFailure, startMachine, stopMachine } from "./hsm-runtime.ts";
import { parseMachineGraph, type MachineGraph } from "./otel/machines.ts";

export function reportMachineGraphFailure(error: unknown): void {
  reportHsmFailure(error);
}

export type MachineGraphPhase = "empty" | "drawing";

export type MachineGraphSnapshot = {
  readonly phase: MachineGraphPhase;
  readonly statePath: string;
  readonly graphs: readonly MachineGraph[];
};

export class Graph extends hsm.Instance {
  static readonly setEvent = { name: "graph.set", kind: hsm.Kinds.Event } as const;
  static readonly clearEvent = { name: "graph.clear", kind: hsm.Kinds.Event } as const;

  static readonly model = hsm.define(
    "Graph",
    hsm.initial(hsm.target("empty")),
    hsm.state(
      "empty",
      hsm.entry(Graph.destroyPaint),
      hsm.transition(hsm.on(Graph.clearEvent.name), hsm.target("."), hsm.effect(Graph.clear)),
      hsm.transition(hsm.on(Graph.setEvent.name), hsm.target("../drawing"), hsm.effect(Graph.remember)),
    ),
    hsm.state(
      "drawing",
      hsm.entry(Graph.requestDraw),
      hsm.exit(Graph.destroyPaint),
      hsm.transition(hsm.on(Graph.setEvent.name), hsm.effect(Graph.rememberAndDraw)),
      hsm.transition(hsm.on(Graph.clearEvent.name), hsm.target("../empty"), hsm.effect(Graph.clear)),
    ),
  );

  graphs: readonly MachineGraph[] = [];
  readonly onDraw: (graphs: readonly MachineGraph[]) => void;
  readonly onDestroy: () => void;

  constructor(hooks: { onDraw: (graphs: readonly MachineGraph[]) => void; onDestroy: () => void }) {
    super();
    this.onDraw = hooks.onDraw;
    this.onDestroy = hooks.onDestroy;
  }

  snapshot(): MachineGraphSnapshot {
    const statePath = this.state();
    return {
      phase: statePath.endsWith("/drawing") ? "drawing" : "empty",
      statePath,
      graphs: this.graphs,
    };
  }

  setGraphs(value: unknown): MachineGraphSnapshot {
    const graphs = parseGraphs(value);
    if (graphs === null || graphs.length === 0) {
      this.dispatch(namedEvent(Graph.clearEvent.name));
    } else {
      this.dispatch(namedEvent(Graph.setEvent.name, { graphs }));
    }
    return this.snapshot();
  }

  static remember(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    const graphs = graphsFromEvent(event);
    if (graphs === null) {
      instance.dispatch(namedEvent(Graph.clearEvent.name));
      return;
    }
    instance.graphs = graphs;
  }

  static rememberAndDraw(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
    Graph.remember(ctx, instance, event);
    if (instance instanceof Graph && instance.graphs.length > 0) instance.onDraw(instance.graphs);
  }

  static clear(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    instance.graphs = [];
  }

  static requestDraw(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph) || instance.graphs.length === 0) return;
    instance.onDraw(instance.graphs);
  }

  static destroyPaint(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
    if (!(instance instanceof Graph)) return;
    instance.onDestroy();
  }
}

export function startGraph(
  ctx: hsm.Context,
  hooks: { onDraw: (graphs: readonly MachineGraph[]) => void; onDestroy: () => void },
): Graph {
  return startMachine(ctx, new Graph(hooks), Graph.model);
}

export async function stopGraph(graph: Graph): Promise<void> {
  await stopMachine(graph);
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
