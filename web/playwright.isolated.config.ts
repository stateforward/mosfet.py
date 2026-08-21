import { randomUUID } from "node:crypto";
import path from "node:path";
import baseConfig from "./playwright.config.ts";

const port = 5193;
const grpcPort = 43_325;

export default {
  ...baseConfig,
  use: {
    ...baseConfig.use,
    baseURL: `http://127.0.0.1:${String(port)}`,
  },
  webServer: {
    ...baseConfig.webServer,
    command: `npm run dev -- --host 127.0.0.1 --port ${String(port)} --strictPort`,
    env: {
      ...baseConfig.webServer?.env,
      BOT_GRPC_PORT: String(grpcPort),
      BOT_MODEL_STORE_PATH: path.join(
        "/tmp",
        `bot-hsm-dashboard-node-focus-${randomUUID()}.json`,
      ),
    },
    url: `http://127.0.0.1:${String(port)}`,
    reuseExistingServer: false,
  },
};
