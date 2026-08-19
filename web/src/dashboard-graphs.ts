import { type MachineGraph } from "./otel/machines.ts";

function ownerShell(graph: MachineGraph): MachineGraph {
  const root = graph.nodes.find((node) => node.path === graph.name && node.parent === null);
  const rootNode = {
    path: graph.name,
    parent: null,
    label: root?.label ?? graph.componentName,
  };
  return {
    ...graph,
    currentState: "",
    lastEventName: "",
    nodes: [rootNode],
    edges: [],
    observationCount: 0,
  };
}

export function graphsForVisibility(
  machines: readonly MachineGraph[],
  visibleMachines: ReadonlyMap<string, boolean>,
): MachineGraph[] {
  const byName = new Map(machines.map((machine) => [machine.name, machine]));
  const included = new Set(
    machines.filter((machine) => visibleMachines.get(machine.name) === true).map((machine) => machine.name),
  );
  for (const machine of machines) {
    if (visibleMachines.get(machine.name) !== true) {
      continue;
    }
    const visitedOwners = new Set<string>();
    let ownerName = machine.owner;
    while (ownerName !== undefined && ownerName !== null && !visitedOwners.has(ownerName)) {
      visitedOwners.add(ownerName);
      const owner = byName.get(ownerName);
      if (owner === undefined) {
        break;
      }
      included.add(owner.name);
      ownerName = owner.owner;
    }
  }
  return machines
    .filter((machine) => included.has(machine.name))
    .map((machine) => (visibleMachines.get(machine.name) === true ? machine : ownerShell(machine)));
}
