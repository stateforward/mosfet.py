import { registerBotDashboard } from "./elements/bot-dashboard.ts";
import { registerBotMachineGraph } from "./elements/bot-machine-graph.ts";
import { registerBotOtelSource } from "./elements/bot-otel-source.ts";

export function registerBotElements(): void {
  registerBotOtelSource();
  registerBotMachineGraph();
  registerBotDashboard();
}
