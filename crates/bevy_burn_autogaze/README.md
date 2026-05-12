# bevy_burn_autogaze

Bevy viewer for `burn_autogaze`. Native and wasm builds use the same Bevy app
and UI layer; platform code only supplies frames and model bytes.

## Native

```sh
cargo run -p bevy_burn_autogaze -- \
  --model-dir /home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a
```

Use `--image-path path/to/frame.png` to run from a static image instead of the
native camera. The default path is a continuous realtime profile: `resize-224`,
640px aspect-preserving input, 16-frame rolling KV window, the model-configured
generation budget, GPU display transfer, PSNR overlay, interframe output, and no
periodic visualization keyframes. Common viewer/inference knobs include `--top-k`, `--frames-per-clip`,
`--max-in-flight`, `--max-gaze-tokens-each-frame`, `--inference-width`,
`--inference-height`, `--task-loss-requirement`, `--disable-task-loss-requirement`,
`--task-loss-requirement-db`, `--mask-cell-scale`, `--blend-alpha`, and `--show-fps`. `--show-gaze-ratio`
toggles the text overlay for per-frame and EMA output update ratio.
`--show-psnr=false` hides PSNR in dB between the current input and rendered
output; the pixel comparison is skipped when this overlay is disabled. GPU
display transfer remains active when PSNR is enabled.
`--task-loss-requirement-db 28` expresses the upstream L1 reconstruction-loss
threshold as `10^(-28 / 20)`, which is more PSNR-like but is not the same value
as the rendered output PSNR overlay.
`--help` lists the accepted values and aliases for mode-like options.
`--log-pipeline-timing` prints source capture, resize/prep, pack, input
upload/preprocess, model, visualization, and Bevy texture-update timing every
few seconds. In `tiled` mode, source frames are resized into a complete AnyRes
224px chunk grid and `--max-gaze-tokens-each-frame` controls the per-tile
generation cap. The output recovery stitches each tile-local scale grid into a
full-frame grid for that scale, matching upstream's mask recovery semantics.
The default mask panel uses `--mask-visualization scale-rows`, which mirrors the
upstream NVIDIA visualizer by drawing stable per-scale rows. Use
`--mask-visualization overlay` to inspect the combined sparse-update footprint.
Use `--perf-summary-frames N` with `--image-path` or another deterministic
source to process `N` inference outputs, print a JSON FPS/timing summary, and
exit. Add `--perf-summary-path target/autogaze-bevy-perf/run.json` to write the
same summary as a JSON artifact for hardware throughput reports.
The docs birds asset profile remains available explicitly:

```sh
cargo run -p bevy_burn_autogaze -- \
  --mode tiled --visualization-mode interframe \
  --max-gaze-tokens-each-frame 0 --frames-per-clip 16 \
  --tile-batch-size 4 --inference-width 1920 --inference-height 1080 \
  --blend-alpha 0.55 --mask-visualization scale-rows --keyframe-duration 0
```

Realtime mode uses `--streaming-cache=true` by default. The cache advances one
new frame at a time and evicts the oldest completed frame span from its rolling
KV window instead of cold-starting at the horizon. Pass
`--streaming-cache=false` for full-window comparison runs that reprocess the
whole clip each inference.
`--max-in-flight 1` is the default camera admission policy: if inference is busy,
new camera frames refresh the buffered input window but do not queue stale model
jobs. Realtime streaming-cache mode is always kept to one in-flight task when
enabled so KV state advances in order; values above `1` apply to tiled or
full-window non-streaming experiments. A max-gaze value of `0` uses the upstream
model default, which is `198` for the NVIDIA config and is also the realtime
default. The maximum frame budget is
`max-gaze-tokens-each-frame * tile-count`, before task-loss stopping and
confidence filtering. `--mode realtime` defaults to a 640px-wide
aspect-preserving source frame. `--mode tiled` defaults to a bounded 1280px-wide
aspect-preserving source frame, `--top-k 2`, 24 generated tokens per tile, and
a tile batch size of 64. Pass explicit `--top-k`, `--tile-batch-size`,
`--inference-width`, and `--inference-height` values for fixed full-resolution
inspection.
`--display-transfer gpu` exercises the shared Bevy/Burn WebGPU texture bridge;
it is the default display path for live runs.
For GPU interframe display, `--tensor-sparse-update-max-rects` and
`--tensor-sparse-update-max-ratio` choose when the tensor compositor uses sparse
rectangle copies instead of the dense mask path; use `0` rects to force dense
updates for apples-to-apples benchmarking.
Use `--require-hardware-adapter=true` for perf runs that should fail fast
instead of silently measuring a CPU/software render adapter.
Use `--load-model=false` to verify camera/preview rendering without waiting for
model load or inference.
From the repository root, run
`cargo run -p xtask -- bevy-perf-matrix --frames 120 --camera` on a real GPU
host to collect deterministic static-source and live camera throughput logs,
per-case JSON summaries, and an aggregate `summary.json` under
`target/autogaze-bevy-perf/`. The matrix runs the Bevy app with
`cargo run --release` by default; pass `--profile dev` only when debugging the
command path. Use `--case-timeout-seconds N` to override the default per-case
timeout for slow first-build or driver-tuning hosts.

`--visualization-mode full-blend` renders the current frame's alpha-blended
mask. The default `--blend-alpha 0.38` keeps live overlays readable; the docs
birds profile uses `0.55`. The center mask panel colors the decoded multi-scale
AutoGaze cells by scale and draws crisp per-scale rows by default.
`--visualization-mode
interframe --keyframe-duration 0` preserves the previous output outside masked
cells and updates masked cells to the current input without periodic full-frame
refreshes. Positive keyframe durations remain available for debugging. The
gaze-ratio overlay reports the percentage of output pixels updated on the
current frame plus an EMA across processed frames. The PSNR overlay reports
current-frame and EMA dB for the output column compared to the current input
frame.

In `full-blend` mode the update ratio reports selected effective mask coverage
as a percentage of the full source frame. In `interframe` mode scheduled
keyframes are excluded from the UI metric samples; intermediate frames report
masked-cell coverage as a percentage of the full source frame.
When the model is ready and inference is busy, camera frames continue to be
buffered for the next clip but the displayed texture is not replaced by raw live
preview frames; this keeps processed output monotonic and avoids apparent
frame-order reversals in slower wasm runs.

## Web

```sh
npm run build:wasm
npm run serve
```

Open `http://localhost:8080` in a WebGPU-capable browser. The web build fetches
NVIDIA AutoGaze `config.json` and `model.safetensors` from Hugging Face by
default and feeds browser camera frames through the exported `frame_input`
function.

The browser shell handles camera permission and frame upload only. The visible
UI is rendered by Bevy into the `#bevy` canvas, matching the native path. Pass
the same viewer/inference knobs as query parameters:

```text
http://localhost:8080/?mode=tiled&visualization-mode=interframe&mask-visualization=scale-rows&keyframe-duration=0&frames-per-clip=16&inference-width=1920&inference-height=1080&tile-batch-size=4&show-fps=true&show-gaze-ratio=true&show-psnr=true
```

Use `?source=static` for a generated static frame, or `?image-url=./frame.png`
to drive the Bevy UI from an image without requesting a webcam.
`inference-width` and `inference-height` resize any received frame before it is
queued for model inference and visualization; for generated static frames, those
same query values also control the generated source resolution unless
`static-width` or `static-height` are set. `load-model=false` keeps the viewer in
preview mode for browser smoke tests. `max-in-flight` controls the same
drop-if-busy inference admission policy as native.
`perf-summary-frames=N` exposes live app timing at `window.__autogazePerf` and a
final summary at `window.__autogazePerfSummary`, including dimensions, gaze
ratio, frame counts, tensor interframe path, render adapter metadata, and the
configured/effective realtime admission policy plus tensor sparse-update policy
for Playwright smoke/perf checks.

Use `config-url` and `weights-url` query parameters to point the wasm build at
alternate model assets. `mask-radius-scale` remains accepted as a compatibility
alias for `mask-cell-scale`.
