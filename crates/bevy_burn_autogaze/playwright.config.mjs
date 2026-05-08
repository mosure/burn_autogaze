import { defineConfig, devices } from "@playwright/test";

const browserChannel = process.env.PLAYWRIGHT_BROWSER_CHANNEL;

export default defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  use: {
    ...devices["Desktop Chrome"],
    ...(browserChannel ? { channel: browserChannel } : {}),
    headless: true,
    baseURL: "http://127.0.0.1:8087",
    launchOptions: {
      args: ["--enable-unsafe-webgpu"],
    },
  },
  webServer: {
    command: "python3 -m http.server 8087 -d www",
    url: "http://127.0.0.1:8087/index.html",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
