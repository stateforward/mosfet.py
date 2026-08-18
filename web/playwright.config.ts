import { defineConfig } from "@playwright/test";
import { randomUUID } from "node:crypto";
import os from "node:os";
import path from "node:path";

const E2E_GRPC_PORT = 43_318;
process.env["BOT_GRPC_PORT"] = String(E2E_GRPC_PORT);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  outputDir: "test-results",
  timeout: 30_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "off",
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173 --strictPort",
    env: {
      ...process.env,
      BOT_GRPC_PORT: String(E2E_GRPC_PORT),
      BOT_MODEL_STORE_PATH: path.join(os.tmpdir(), `bot-hsm-dashboard-e2e-${randomUUID()}.json`),
    },
    url: "http://127.0.0.1:5173",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
