import { test, expect } from "@playwright/test";

const wasmPanicNeedles = [
  "Creating a wgpu setup synchronously is unsupported on wasm",
  "Failed to read tensor data synchronously",
  "time not implemented on this platform",
  "std::time::Instant",
  "condvar wait not supported",
  "cannot recursively acquire mutex",
  "Buffer is already mapped",
  "used in submit while mapped",
  "Validation RenderError",
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
    /AutoGaze timing: ([0-9.]+) output fps \/ ([0-9.]+) model-frame fps \(([0-9.]+) ms\) clip=([0-9]+) model_frames=([0-9]+) points=([0-9]+) gaze=([0-9.]+)% ([0-9]+)x([0-9]+)/;
  for (let index = consoleLines.length - 1; index >= 0; index -= 1) {
    const line = consoleLines[index];
    const match = line.match(timingPattern);
    if (!match) {
      continue;
    }
    return {
      fps: Number.parseFloat(match[1]),
      modelFrameFps: Number.parseFloat(match[2]),
      totalMs: Number.parseFloat(match[3]),
      clipFrames: Number.parseInt(match[4], 10),
      modelFrames: Number.parseInt(match[5], 10),
      points: Number.parseInt(match[6], 10),
      gazeRatioPercent: Number.parseFloat(match[7]),
      width: Number.parseInt(match[8], 10),
      height: Number.parseInt(match[9], 10),
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

test("static wasm source defaults to live realtime dimensions", async ({ page }) => {
  const consoleLines = [];
  const pageErrors = [];

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
          throw new Error("webcam should not be requested in static-source mode");
        },
      },
    });
  });

  await page.goto("/?source=static&load-model=false&static-fps=1", {
    waitUntil: "domcontentloaded",
  });

  await expect
    .poll(
      async () => page.evaluate(() => window.__autogazeFrameStats?.count || 0),
      {
        timeout: 60_000,
      },
    )
    .toBeGreaterThanOrEqual(1);
  const frameStats = await page.evaluate(() => window.__autogazeFrameStats);
  expect(frameStats.lastWidth).toBe(640);
  expect(frameStats.lastHeight).toBe(360);
  expect(pageErrors).toEqual([]);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});

test("synthetic wasm source does not request webcam frames", async ({ page }) => {
  const consoleLines = [];
  const pageErrors = [];

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
          throw new Error("webcam should not be requested in synthetic-source mode");
        },
      },
    });
  });

  await page.goto(
    "/?source=synthetic-pan&load-model=false&show-fps=false&show-gaze-ratio=true&show-psnr=true&mode=realtime",
    { waitUntil: "domcontentloaded" },
  );

  const status = page.locator("#status");
  let state = "pending";
  await expect
    .poll(
      async () => {
        const text = (await status.textContent()) ?? "";
        if (text.includes("synthetic source running")) {
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
  const getUserMediaCalls = await page.evaluate(
    () => window.__autogazeGetUserMediaCalls || 0,
  );
  expect(getUserMediaCalls).toBe(0);
  if (state === "no-webgpu") {
    expectNoKnownWasmPanic(consoleLines, pageErrors);
    return;
  }
  expect(state).toBe("running");
  expect(pageErrors).toEqual([]);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});

test("patch-diff wasm gpu display does not reuse mapped readback buffers", async ({
  page,
}) => {
  const consoleLines = [];
  const pageErrors = [];

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
          throw new Error("webcam should not be requested in static-source mode");
        },
      },
    });
  });

  await page.goto(
    "/?source=static&load-model=false&mask-source=patch-diff&display-transfer=gpu&show-fps=false&show-gaze-ratio=false&show-psnr=false&mode=realtime&visualization-mode=interframe&frames-per-clip=2&static-width=64&static-height=64&inference-width=64&inference-height=64&static-fps=5&perf-summary-frames=2&patch-diff-grid=64&patch-diff-threshold=0.01",
    { waitUntil: "domcontentloaded" },
  );

  let state = "pending";
  await expect
    .poll(
      async () => {
        const output = combinedOutput(consoleLines, pageErrors);
        if (wasmPanicNeedles.some((needle) => output.includes(needle))) {
          state = "known-panic";
        } else if (await page.evaluate(() => Boolean(window.__autogazePerfSummary))) {
          state = "summary";
        } else if (isDeviceLost(output)) {
          state = "device-lost";
        } else {
          state = "pending";
        }
        return state;
      },
      { timeout: 90_000 },
    )
    .not.toBe("pending");

  expect(state).not.toBe("known-panic");
  if (state === "device-lost") {
    expectNoKnownWasmPanic(consoleLines, pageErrors);
    return;
  }

  expect(state).toBe("summary");
  const perfSummary = await page.evaluate(
    () => window.__autogazePerfSummary || null,
  );
  expect(perfSummary).not.toBeNull();
  expect(perfSummary.sparse_mask_source).toBe("patch-diff");
  expect(perfSummary.display_transfer).toBe("gpu");
  expect(perfSummary.latest_effective_display_transfer).toBe("gpu");
  expect(perfSummary.latest_width).toBe(64);
  expect(perfSummary.latest_height).toBe(64);
  expect(perfSummary.processed_frames).toBeGreaterThanOrEqual(2);
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
    "/?source=static&show-fps=true&show-gaze-ratio=true&show-psnr=false&mode=resize-224&visualization-mode=interframe&display-transfer=gpu&frames-per-clip=2&max-in-flight=2&streaming-cache=false&static-width=224&static-height=224&inference-width=224&inference-height=224&static-fps=1&top-k=1&max-gaze-tokens-each-frame=1&disable-task-loss-requirement=true&log-pipeline-timing=true&perf-summary-frames=2&config-url=./config.json&weights-url=./model.safetensors",
    { waitUntil: "domcontentloaded" },
  );

  let state = "pending";
  await expect
    .poll(
      async () => {
        const output = combinedOutput(consoleLines, pageErrors);
        if (wasmPanicNeedles.some((needle) => output.includes(needle))) {
          state = "known-panic";
        } else if (await page.evaluate(() => Boolean(window.__autogazePerfSummary))) {
          state = "summary";
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

  expect(state).toBe("summary");

  const timing = latestTimingMetrics(consoleLines);
  expect(timing).not.toBeNull();
  expect(timing.fps).toBeGreaterThan(0);
  expect(timing.totalMs).toBeGreaterThan(0);
  expect(timing.clipFrames).toBe(2);
  expect(timing.modelFrames).toBe(1);
  expect(timing.gazeRatioPercent).toBeGreaterThanOrEqual(0);
  expect(timing.gazeRatioPercent).toBeLessThanOrEqual(100);
  expect(timing.width).toBe(224);
  expect(timing.height).toBe(224);
  const perf = await page.evaluate(() => window.__autogazePerf || null);
  expect(perf).not.toBeNull();
  expect(perf.processed_frames).toBeGreaterThanOrEqual(2);
  expect(perf.latest_sequence).toBeGreaterThanOrEqual(2);
  expect(perf.latest_gaze_update_ratio).toBeGreaterThanOrEqual(0);
  expect(perf.latest_gaze_update_ratio).toBeLessThanOrEqual(1);
  expect(perf.latest_width).toBe(224);
  expect(perf.latest_height).toBe(224);
  expect(perf.mode).toBe("realtime");
  expect(perf.visualization_mode).toBe("interframe");
  expect(perf.display_transfer).toBe("gpu");
  expect(perf.latest_effective_display_transfer).toBe("gpu");
  expect(perf.streaming_cache).toBe(false);
  expect(perf.streaming_cache_effective).toBe(false);
  expect(perf.show_psnr).toBe(false);
  expect(perf.latest_psnr_db).toBeNull();
  expect(perf.latest_psnr_db_infinite).toBe(false);
  expect(perf.ema_psnr_db).toBeNull();
  expect(perf.ema_psnr_db_infinite).toBe(false);
  expect(perf.configured_max_in_flight).toBe(2);
  expect(perf.effective_max_in_flight).toBe(2);
  expect(perf.frames_per_clip).toBe(2);
  expect(perf.top_k).toBe(1);
  expect(perf.max_gaze_tokens_each_frame).toBe(1);
  expect(perf.tile_batch_size).toBeGreaterThan(0);
  expect(perf.inference_width).toBe(224);
  expect(perf.inference_height).toBe(224);
  expect(perf.tensor_sparse_update_max_rects).toBeGreaterThanOrEqual(0);
  expect(perf.tensor_sparse_update_max_ratio).toBeGreaterThanOrEqual(0);
  expect(perf.tensor_sparse_update_max_ratio).toBeLessThanOrEqual(1);
  expect(perf.tensor_full_frame_update_min_ratio).toBeGreaterThanOrEqual(0);
  expect(perf.tensor_full_frame_update_min_ratio).toBeLessThanOrEqual(1);
  expect(["sparse-rects", "dense-mask", "full-frame"]).toContain(
    perf.latest_tensor_interframe_path,
  );
  expect(typeof perf.render_adapter_name).toBe("string");
  expect(perf.render_adapter_name.length).toBeGreaterThan(0);
  expect(typeof perf.render_adapter_device_type).toBe("string");
  expect(typeof perf.render_adapter_backend).toBe("string");
  expect(perf.p95_total_ms).toBeGreaterThan(0);
  const perfSummary = await page.evaluate(
    () => window.__autogazePerfSummary || null,
  );
  expect(perfSummary).not.toBeNull();
  expect(perfSummary.processed_frames).toBeGreaterThanOrEqual(2);
  expect(["sparse-rects", "dense-mask", "full-frame"]).toContain(
    perfSummary.latest_tensor_interframe_path,
  );
  expect(perfSummary.render_adapter_name).toBe(perf.render_adapter_name);
  expect(perfSummary.render_adapter_backend).toBe(perf.render_adapter_backend);
  expect(perfSummary.mode).toBe(perf.mode);
  expect(perfSummary.visualization_mode).toBe(perf.visualization_mode);
  expect(perfSummary.display_transfer).toBe(perf.display_transfer);
  expect(perfSummary.latest_effective_display_transfer).toBe(
    perf.latest_effective_display_transfer,
  );
  expect(perfSummary.streaming_cache).toBe(perf.streaming_cache);
  expect(perfSummary.streaming_cache_effective).toBe(
    perf.streaming_cache_effective,
  );
  expect(perfSummary.show_psnr).toBe(perf.show_psnr);
  expect(perfSummary.latest_psnr_db).toBe(perf.latest_psnr_db);
  expect(perfSummary.latest_psnr_db_infinite).toBe(
    perf.latest_psnr_db_infinite,
  );
  expect(perfSummary.ema_psnr_db).toBe(perf.ema_psnr_db);
  expect(perfSummary.ema_psnr_db_infinite).toBe(perf.ema_psnr_db_infinite);
  expect(perfSummary.configured_max_in_flight).toBe(
    perf.configured_max_in_flight,
  );
  expect(perfSummary.effective_max_in_flight).toBe(
    perf.effective_max_in_flight,
  );
  expect(perfSummary.frames_per_clip).toBe(perf.frames_per_clip);
  expect(perfSummary.top_k).toBe(perf.top_k);
  expect(perfSummary.max_gaze_tokens_each_frame).toBe(
    perf.max_gaze_tokens_each_frame,
  );
  expect(perfSummary.tile_batch_size).toBe(perf.tile_batch_size);
  expect(perfSummary.tensor_sparse_update_max_rects).toBe(
    perf.tensor_sparse_update_max_rects,
  );
  expect(perfSummary.tensor_sparse_update_max_ratio).toBe(
    perf.tensor_sparse_update_max_ratio,
  );
  expect(perfSummary.tensor_full_frame_update_min_ratio).toBe(
    perf.tensor_full_frame_update_min_ratio,
  );
  expect(perfSummary.latest_width).toBe(224);
  expect(perfSummary.latest_height).toBe(224);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});
