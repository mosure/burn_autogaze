import init, { WasmAutoGaze } from "./pkg/burn_autogaze.js";

const DEFAULT_CONFIG =
  "https://huggingface.co/nvidia/AutoGaze/resolve/main/config.json";
const DEFAULT_WEIGHTS =
  "https://huggingface.co/nvidia/AutoGaze/resolve/main/model.safetensors";

const configUrl = document.getElementById("config-url");
const weightsUrl = document.getElementById("weights-url");
const mode = document.getElementById("mode");
const visualizationMode = document.getElementById("visualization-mode");
const resolution = document.getElementById("resolution");
const clipFrames = document.getElementById("clip-frames");
const topK = document.getElementById("top-k");
const maxGazeTokens = document.getElementById("max-gaze-tokens");
const keyframeDuration = document.getElementById("keyframe-duration");
const loadModel = document.getElementById("load-model");
const startCamera = document.getElementById("start-camera");
const status = document.getElementById("status");
const stats = document.getElementById("stats");
const video = document.getElementById("video");
const capture = document.getElementById("capture");
const triptych = document.getElementById("triptych");

const captureCtx = capture.getContext("2d", { willReadFrequently: true });
const triptychCtx = triptych.getContext("2d");

let wasmReady = false;
let model = null;
let stream = null;
let running = false;
let processing = false;
let frameQueue = [];
let lastInferenceAt = 0;
let smoothedFps = 0;
let smoothedGazeRatio = 0;
let smoothedPsnr = 0;

configUrl.value = DEFAULT_CONFIG;
weightsUrl.value = DEFAULT_WEIGHTS;

loadModel.addEventListener("click", () => loadAutogazeModel());
startCamera.addEventListener("click", () => toggleCamera());
mode.addEventListener("change", applyModelOptions);
visualizationMode.addEventListener("change", applyModelOptions);
topK.addEventListener("change", applyModelOptions);
maxGazeTokens.addEventListener("change", applyModelOptions);
keyframeDuration.addEventListener("change", applyModelOptions);

function setStatus(message) {
  status.textContent = message;
}

function setBusy(value) {
  loadModel.disabled = value;
  startCamera.disabled = value;
}

async function loadAutogazeModel() {
  try {
    setBusy(true);
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }
    if (!wasmReady) {
      setStatus("loading wasm");
      await init();
      wasmReady = true;
    }

    setStatus("fetching config");
    const configText = await fetchText(configUrl.value.trim());
    setStatus("fetching weights");
    const weights = await fetchBytes(weightsUrl.value.trim(), (loaded, total) => {
      if (total) {
        setStatus(`fetching weights ${Math.round((loaded / total) * 100)}%`);
      } else {
        setStatus(`fetching weights ${formatBytes(loaded)}`);
      }
    });

    setStatus("loading model");
    model = await WasmAutoGaze.create(configText, weights);
    applyModelOptions();
    setStatus(`model ready (${WasmAutoGaze.version()})`);
  } catch (error) {
    console.error(error);
    setStatus(error.message || String(error));
  } finally {
    setBusy(false);
  }
}

async function toggleCamera() {
  if (running) {
    stopCamera();
    return;
  }
  if (!model) {
    await loadAutogazeModel();
  }
  if (!model) {
    return;
  }

  try {
    const constraints = cameraConstraints(resolution.value);
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    frameQueue = [];
    lastInferenceAt = 0;
    smoothedFps = 0;
    smoothedGazeRatio = 0;
    smoothedPsnr = 0;
    running = true;
    startCamera.textContent = "Stop";
    setStatus("camera running");
    requestAnimationFrame(captureLoop);
  } catch (error) {
    console.error(error);
    setStatus(error.message || String(error));
  }
}

function stopCamera() {
  running = false;
  startCamera.textContent = "Start";
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
  stream = null;
  video.srcObject = null;
  setStatus("stopped");
}

function captureLoop() {
  if (!running) {
    return;
  }

  if (video.videoWidth > 0 && video.videoHeight > 0) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (capture.width !== width || capture.height !== height) {
      capture.width = width;
      capture.height = height;
    }
    captureCtx.drawImage(video, 0, 0, width, height);
    const frame = captureCtx.getImageData(0, 0, width, height).data;
    frameQueue.push(new Uint8Array(frame));

    const requiredFrames = clampInteger(clipFrames.value, 1, 16);
    while (frameQueue.length > requiredFrames) {
      frameQueue.shift();
    }

    if (!processing && frameQueue.length === requiredFrames) {
      runInference(width, height, requiredFrames);
    }
  }

  requestAnimationFrame(captureLoop);
}

function runInference(width, height, frames) {
  processing = true;
  setTimeout(async () => {
    try {
      applyModelOptions();
      const frameBytes = width * height * 4;
      const clip = new Uint8Array(frameBytes * frames);
      for (let i = 0; i < frames; i += 1) {
        clip.set(frameQueue[i], i * frameBytes);
      }

      const started = performance.now();
      const output = await model.infer_rgba_clip(clip, width, height, frames);
      const elapsed = performance.now() - started;
      drawOutput(output);

      const now = performance.now();
      if (lastInferenceAt > 0) {
        const fps = 1000 / (now - lastInferenceAt);
        smoothedFps = smoothedFps ? smoothedFps * 0.85 + fps * 0.15 : fps;
      }
      const gazeRatio = output.update_ratio * 100;
      smoothedGazeRatio = smoothedGazeRatio
        ? smoothedGazeRatio * 0.85 + gazeRatio * 0.15
        : gazeRatio;
      const psnrText = output.psnr_text();
      if (Number.isFinite(output.psnr_db)) {
        smoothedPsnr = smoothedPsnr
          ? smoothedPsnr * 0.85 + output.psnr_db * 0.15
          : output.psnr_db;
      }
      lastInferenceAt = now;
      const psnrEma = smoothedPsnr ? smoothedPsnr.toFixed(1) : psnrText;
      stats.textContent = `${width}x${height} ${output.mode}, ${output.visualization_mode}, ${output.tile_count} tile(s), ${elapsed.toFixed(1)} ms, ${smoothedFps.toFixed(1)} fps, gaze ${gazeRatio.toFixed(1)}% ema ${smoothedGazeRatio.toFixed(1)}%, psnr ${psnrText} dB ema ${psnrEma}`;
      output.free();
    } catch (error) {
      console.error(error);
      setStatus(error.message || String(error));
    } finally {
      processing = false;
    }
  }, 0);
}

function drawOutput(output) {
  const pixels = output.side_by_side_rgba();
  triptych.style.aspectRatio = `${output.side_by_side_width} / ${output.height}`;
  const image = new ImageData(
    new Uint8ClampedArray(pixels),
    output.side_by_side_width,
    output.height,
  );
  if (triptych.width !== output.side_by_side_width || triptych.height !== output.height) {
    triptych.width = output.side_by_side_width;
    triptych.height = output.height;
  }
  triptychCtx.putImageData(image, 0, 0);
}

function applyModelOptions() {
  if (!model) {
    return;
  }
  model.set_top_k(clampInteger(topK.value, 1, 265));
  const maxTokens = clampInteger(maxGazeTokens.value, 0, 265);
  if (maxTokens > 0) {
    model.set_max_gaze_tokens_each_frame(maxTokens);
  } else {
    model.reset_max_gaze_tokens_each_frame();
  }
  model.set_keyframe_duration(clampInteger(keyframeDuration.value, 1, 300));
  model.set_visualization_mode(visualizationMode.value);
  if (mode.value === "tile") {
    if (typeof model.set_anyres_tiled_mode === 'function') {
      model.set_anyres_tiled_mode(224);
    } else {
      model.set_tiled_mode(224, 224);
    }
  } else {
    model.set_resize_mode();
  }
}

function cameraConstraints(value) {
  if (value === "1080") {
    return {
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    };
  }
  if (value === "native") {
    return {
      video: {
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    };
  }
  return {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 30 },
    },
    audio: false,
  };
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status}`);
  }
  return response.text();
}

async function fetchBytes(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status}`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function clampInteger(value, min, max) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) {
    return min;
  }
  return Math.min(max, Math.max(min, number));
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

if (!navigator.gpu) {
  setStatus("WebGPU is not available");
}
