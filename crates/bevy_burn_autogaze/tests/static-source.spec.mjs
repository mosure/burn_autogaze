import { test, expect } from "@playwright/test";

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
    "/?source=static&load-model=false&show-fps=false&show-gaze-ratio=true&mode=tile-224&visualization-mode=interframe&keyframe-duration=3&frames-per-clip=1&static-width=320&static-height=180&static-fps=5",
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
    expect(consoleLines.join("\n")).not.toContain(
      "Creating a wgpu setup synchronously is unsupported on wasm",
    );
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
  expect(getUserMediaCalls).toBe(0);
  expect(pageErrors).toEqual([]);
  expect(consoleLines.join("\n")).not.toContain(
    "Creating a wgpu setup synchronously is unsupported on wasm",
  );
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
        if (
          output.includes(
            "Creating a wgpu setup synchronously is unsupported on wasm",
          )
        ) {
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
