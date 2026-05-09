import { test, expect } from "@playwright/test";

const wasmPanicNeedles = [
  "Creating a wgpu setup synchronously is unsupported on wasm",
  "Failed to read tensor data synchronously",
  "time not implemented on this platform",
  "std::time::Instant",
];

function combinedOutput(consoleLines, pageErrors) {
  return `${consoleLines.join("\n")}\n${pageErrors.join("\n")}`;
}

function expectNoKnownWasmPanic(consoleLines, pageErrors) {
  const output = combinedOutput(consoleLines, pageErrors);
  for (const needle of wasmPanicNeedles) {
    expect(output).not.toContain(needle);
  }
}

function latestTimingMetrics(consoleLines) {
  const timingPattern =
    /AutoGaze timing: ([0-9.]+) fps e2e \(([0-9.]+) ms\) clip=([0-9]+) ([0-9]+)x([0-9]+)/;
  for (let index = consoleLines.length - 1; index >= 0; index -= 1) {
    const line = consoleLines[index];
    const match = line.match(timingPattern);
    if (!match) {
      continue;
    }
    return {
      fps: Number.parseFloat(match[1]),
      totalMs: Number.parseFloat(match[2]),
      clipFrames: Number.parseInt(match[3], 10),
      width: Number.parseInt(match[4], 10),
      height: Number.parseInt(match[5], 10),
    };
  }
  return null;
}

function isDeviceLost(output) {
  return (
    output.includes("DeviceLost") ||
    output.includes("Destroyed Device was destroyed") ||
    output.includes("Quitting the application due to DeviceLost")
  );
}

test("boots bevy wasm with static frames and no webcam", async ({ page }) => {
  const consoleLines = [];
  const pageErrors = [];
  let getUserMediaCalls = 0;

  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          window.__autogazeGetUserMediaCalls =
            (window.__autogazeGetUserMediaCalls || 0) + 1;
          throw new Error("webcam should not be requested in static-source mode");
        },
      },
    });
  });

  await page.goto(
    "/?source=static&load-model=false&show-fps=false&show-gaze-ratio=true&show-psnr=true&mode=tile-224&visualization-mode=interframe&keyframe-duration=3&frames-per-clip=1&static-width=320&static-height=180&inference-width=640&inference-height=360&static-fps=5",
    { waitUntil: "domcontentloaded" },
  );

  const status = page.locator("#status");
  let state = "pending";
  await expect
    .poll(
      async () => {
        const text = (await status.textContent()) ?? "";
        if (text.includes("static source running")) {
          state = "running";
        } else if (text.includes("webgpu unavailable")) {
          state = "no-webgpu";
        } else if (text.includes("runtime error")) {
          state = "error";
        } else {
          state = "pending";
        }
        return state;
      },
      { timeout: 60_000 },
    )
    .not.toBe("pending");

  if (state === "no-webgpu") {
    expectNoKnownWasmPanic(consoleLines, pageErrors);
    return;
  }

  expect(state).toBe("running");
  await expect(page.locator(".loading")).toBeHidden();
  await expect(page.locator("#bevy")).toBeVisible();
  await page.waitForFunction(() => {
    const canvas = document.querySelector("#bevy");
    return canvas && canvas.width > 0 && canvas.height > 0;
  });
  await page.waitForTimeout(500);

  getUserMediaCalls = await page.evaluate(
    () => window.__autogazeGetUserMediaCalls || 0,
  );
  const frameStats = await page.evaluate(() => window.__autogazeFrameStats);
  expect(frameStats.count).toBeGreaterThanOrEqual(2);
  expect(frameStats.lastWidth).toBe(320);
  expect(frameStats.lastHeight).toBe(180);
  expect(frameStats.lastFrameMs).toBeGreaterThanOrEqual(
    frameStats.firstFrameMs,
  );
  expect(getUserMediaCalls).toBe(0);
  expect(pageErrors).toEqual([]);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});

test("starts wasm model load through async wgpu setup", async ({ page }) => {
  const consoleLines = [];
  const pageErrors = [];

  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });

  await page.goto(
    "/?source=static&show-fps=false&frames-per-clip=1&static-width=64&static-height=64&static-fps=2&config-url=data%3Aapplication%2Fjson%2C%7B%7D&weights-url=data%3Aapplication%2Foctet-stream%3Bbase64%2CAA%3D%3D",
    { waitUntil: "domcontentloaded" },
  );

  const status = page.locator("#status");
  await expect
    .poll(async () => (await status.textContent()) ?? "", { timeout: 60_000 })
    .toMatch(/static source running|webgpu unavailable|runtime error/);

  let state = "pending";
  await expect
    .poll(
      async () => {
        const output = `${consoleLines.join("\n")}\n${pageErrors.join("\n")}`;
        if (wasmPanicNeedles.some((needle) => output.includes(needle))) {
          state = "sync-panic";
        } else if (output.includes("failed to load AutoGaze model")) {
          state = "handled-model-error";
        } else if (
          output.toLowerCase().includes("webgpu") ||
          output.toLowerCase().includes("adapter")
        ) {
          state = "no-webgpu";
        } else {
          state = "pending";
        }
        return state;
      },
      { timeout: 60_000 },
    )
    .not.toBe("pending");

  expect(state).not.toBe("sync-panic");
});

test("runs optional real wasm inference smoke when model assets are available", async ({
  page,
}) => {
  test.skip(
    process.env.AUTOGAZE_WASM_MODEL_E2E !== "1",
    "set AUTOGAZE_WASM_MODEL_E2E=1 and provide www/config.json + www/model.safetensors",
  );

  const configResponse = await page.request.get("/config.json");
  const weightsResponse = await page.request.get("/model.safetensors");
  test.skip(
    !configResponse.ok() || !weightsResponse.ok(),
    "missing local wasm model assets",
  );

  const consoleLines = [];
  const pageErrors = [];
  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });

  await page.goto(
    "/?source=static&show-fps=true&show-gaze-ratio=true&show-psnr=false&mode=resize-224&visualization-mode=interframe&frames-per-clip=1&static-width=224&static-height=224&static-fps=1&top-k=1&max-gaze-tokens-each-frame=1&disable-task-loss-requirement=true&log-pipeline-timing=true&config-url=./config.json&weights-url=./model.safetensors",
    { waitUntil: "domcontentloaded" },
  );

  let state = "pending";
  await expect
    .poll(
      async () => {
        const output = combinedOutput(consoleLines, pageErrors);
        if (wasmPanicNeedles.some((needle) => output.includes(needle))) {
          state = "known-panic";
        } else if (latestTimingMetrics(consoleLines)) {
          state = "timing";
        } else if (isDeviceLost(output)) {
          state = "device-lost";
        } else if (output.includes("failed to load AutoGaze model")) {
          state = "model-error";
        } else {
          state = "pending";
        }
        return state;
      },
      { timeout: 180_000 },
    )
    .not.toBe("pending");

  expect(state).not.toBe("known-panic");
  if (state === "device-lost") {
    expectNoKnownWasmPanic(consoleLines, pageErrors);
    return;
  }

  expect(state).toBe("timing");

  const timing = latestTimingMetrics(consoleLines);
  expect(timing).not.toBeNull();
  expect(timing.fps).toBeGreaterThan(0);
  expect(timing.totalMs).toBeGreaterThan(0);
  expect(timing.clipFrames).toBe(1);
  expect(timing.width).toBe(224);
  expect(timing.height).toBe(224);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});
