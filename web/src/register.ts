import { registerFlowElements } from "./flow/register.ts";
import { registerBotDashboard } from "./elements/bot-dashboard.ts";
import { registerBotMachineGraph } from "./elements/bot-machine-graph/index.ts";
import { registerBotOtelSource } from "./elements/bot-otel-source.ts";

export function registerBotElements(): void {
  registerFlowElements();
  registerBotOtelSource();
  registerBotMachineGraph();
  registerBotDashboard();
}
