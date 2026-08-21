import { defineConfig } from "@playwright/test";
import { randomUUID } from "node:crypto";
import os from "node:os";
import path from "node:path";
import baseConfig from "./playwright.config.ts";

/** Integration boundary: live browser + isolated webServer, not the unit suite. */
const ISOLATED_PORT = 5193;
const ISOLATED_GRPC_PORT = 43_325;
const TMP_DIR = os.tmpdir();
const STORE_FILE = `bot-hsm-dashboard-node-focus-${randomUUID()}.json`;
const webServer = baseConfig.webServer;
if (webServer === undefined || Array.isArray(webServer)) {
  throw new Error("isolated Playwright config requires a single webServer");
}

export default defineConfig({
  ...baseConfig,
  use: {
    ...baseConfig.use,
    baseURL: `http://127.0.0.1:${String(ISOLATED_PORT)}`,
  },
  webServer: {
    ...webServer,
    command: `npm run dev -- --host 127.0.0.1 --port ${String(ISOLATED_PORT)} --strictPort`,
    env: {
      ...webServer.env,
      BOT_GRPC_PORT: String(ISOLATED_GRPC_PORT),
      BOT_MODEL_STORE_PATH: path.join(TMP_DIR, STORE_FILE),
    },
    url: `http://127.0.0.1:${String(ISOLATED_PORT)}`,
    reuseExistingServer: false,
  },
});
