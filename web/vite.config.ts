import { defineConfig, type Plugin } from "vite";

import { createCollector } from "./collector/collector.ts";

function otelCollectorPlugin(): Plugin {
  return {
    name: "bot-otel-collector",
    async configureServer(server) {
      const collector = await createCollector();
      server.middlewares.use(collector.middleware);
      const shutdown = (): void => {
        collector.grpcServer.forceShutdown();
      };
      server.httpServer?.once("close", shutdown);
    },
  };
}

export default defineConfig({
  plugins: [otelCollectorPlugin()],
  server: {
    port: 5173,
  },
});
