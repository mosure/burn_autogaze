# bevy_burn_autogaze

Bevy viewer for `burn_autogaze`. Native and wasm builds use the same Bevy app
and UI layer; platform code only supplies frames and model bytes.

## native

```sh
cargo run -p bevy_burn_autogaze
cargo run -p bevy_burn_autogaze -- --image-path path/to/frame.png
cargo run -p bevy_burn_autogaze -- --mode tiled --visualization-mode interframe
```

The no-arg default is the realtime camera profile:

- `resize-224` model input.
- 640px aspect-preserving source frames.
- 16-frame rolling streaming cache.
- Upstream model generation budget.
- Deduplicated mask geometry.
- Adaptive CPU/GPU display transfer.
- Interframe output with PSNR enabled.
- No periodic visualization keyframes.

Point at local model assets with:

```sh
cargo run -p bevy_burn_autogaze -- \
  --model-dir /path/to/AutoGaze
```

Use `--load-model=false` to check camera/preview rendering without loading the
model.

## common options

| option | default | notes |
|---|---|---|
| `--source` | `camera` | `camera`, `static`, or `synthetic-pan`; `--image-path` selects `static` automatically. |
| `--mode` | `realtime` | `realtime`, `resize-224`, `tiled`, or full-resolution tiled aliases. |
| `--frames-per-clip` | `16` | Number of frames in the model context window. |
| `--streaming-cache` | `true` in realtime | Advances one frame at a time and preserves KV/cache order. |
| `--max-in-flight` | `1` in realtime | Drops stale inference jobs instead of queueing old camera frames. |
| `--task-loss-requirement` | model/viewer default | Upstream L1 reconstruction-loss threshold. |
| `--task-loss-requirement-db` | unset | PSNR-like interface for the same task-loss threshold: `10^(-dB / 20)`. |
| `--max-gaze-tokens-each-frame` | model default in realtime | `0` means the NVIDIA config value, currently `198`. |
| `--top-k` | mode-specific | Number of gaze candidates considered per step. |
| `--inference-width`, `--inference-height` | mode-specific | Source resize before inference/visualization. |
| `--tile-batch-size` | mode-specific | Backend batch size for tiled modes. |
| `--display-transfer` | `auto` | `auto`, `cpu`, or `gpu`. |
| `--mask-geometry` | `deduplicated` | `native`, `deduplicated`, or `effective`. |
| `--mask-visualization` | `image-mask-only` | `image-mask-only`, `image-overlay`, `overlay`, or `scale-rows`. |
| `--visualization-mode` | `interframe` | `interframe` or `full-blend`. |
| `--show-fps`, `--show-gaze-ratio`, `--show-psnr` | enabled where useful | Text overlays; PSNR work is skipped when disabled. |

Run `cargo run -p bevy_burn_autogaze -- --help` for the complete list and
aliases.

## visualization

Mask geometry controls the cells that are drawn and applied:

- `deduplicated` preserves the native update union but removes cells fully
  covered by other selected cells. This is the interactive default because
  high-motion streams can otherwise repeat equivalent multi-scale work.
- `native` draws every decoded AutoGaze cell exactly. Use it for model/debug
  diagnostics and docs traces.
- `effective` projects selected tokens onto the finest active grid for compact
  sparse-token footprint views.

Mask visualization controls how those cells are displayed:

- `image-mask-only`: show only masked source pixels with transparent unmasked
  pixels.
- `image-overlay`: alpha-blend the colored mask over the full source frame.
- `overlay`: show only the combined sparse-update footprint.
- `scale-rows`: draw each active scale as a separate diagnostic row,
  letterboxed to the source aspect.

`--visualization-mode interframe --keyframe-duration 0` preserves prior output
outside masked cells and updates masked cells from the current input. Positive
keyframe durations are available for debugging. `full-blend` renders the current
frame with an alpha-blended mask and reports selected effective mask coverage.

The gaze-ratio overlay reports current and EMA output-pixel update ratio. The
PSNR overlay reports current and EMA dB between the output column and current
input.

## tiled/docs profile

The checked-in birds docs use full-resolution tiled inference and exact native
mask diagnostics:

```sh
cargo run -p bevy_burn_autogaze -- \
  --mode tiled --visualization-mode interframe \
  --max-gaze-tokens-each-frame 0 --frames-per-clip 16 \
  --tile-batch-size 4 --inference-width 1920 --inference-height 1080 \
  --blend-alpha 0.55 --mask-visualization scale-rows --mask-geometry native \
  --keyframe-duration 0
```

For ordinary interactive tiled runs, prefer the default deduplicated geometry
and bounded source dimensions unless you are explicitly inspecting the full
source-resolution trace.

## performance capture

For deterministic summaries:

```sh
cargo run -p bevy_burn_autogaze -- \
  --image-path tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png \
  --perf-summary-frames 120 \
  --perf-summary-path target/autogaze-bevy-perf/run.json
```

Use `--source synthetic-pan` for repeatable full-frame motion without a webcam.
This feeds generated moving RGBA frames through the same frame queue,
preprocessing, model, visualization, and display path as camera input.

Use `--log-pipeline-timing` to print source capture, resize/prep, pack,
preprocess/upload, model, visualization, and texture-update timing. Use
`--require-hardware-adapter=true` when a perf run should fail instead of
recording CPU/software adapter numbers.

The matrix runner is the preferred multi-case path on a GPU host:

```sh
cargo run -p xtask -- bevy-perf-matrix --frames 120 --camera
```

It runs `cargo run --release` by default and writes per-case JSON summaries plus
an aggregate `summary.json` under `target/autogaze-bevy-perf/`.

## web

```sh
npm run build:wasm
npm run serve
```

Open `http://localhost:8080` in a WebGPU-capable browser. The web build fetches
NVIDIA AutoGaze `config.json` and `model.safetensors` from Hugging Face by
default and feeds browser camera frames through the exported `frame_input`
function.

The browser shell handles camera permission and frame upload only. Bevy renders
the visible UI into the `#bevy` canvas, matching native. Pass viewer options as
query parameters:

```text
http://localhost:8080/?source=static&mode=realtime&mask-geometry=deduplicated&show-psnr=true
```

Useful browser parameters:

- `source=static`: generated static frame, no webcam prompt.
- `image-url=./frame.png`: image source, no webcam prompt.
- `load-model=false`: preview-only smoke test.
- `config-url` / `weights-url`: alternate model assets.
- `perf-summary-frames=N`: exposes `window.__autogazePerf` samples and
  `window.__autogazePerfSummary`.

`mask-radius-scale` remains accepted as a compatibility alias for
`mask-cell-scale`.
