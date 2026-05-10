# bevy_burn_autogaze

Bevy viewer for `burn_autogaze`. Native and wasm builds use the same Bevy app
and UI layer; platform code only supplies frames and model bytes.

## Native

```sh
cargo run -p bevy_burn_autogaze -- \
  --model-dir /home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a \
  --mode realtime
```

Use `--image-path path/to/frame.png` to run from a static image instead of the
native camera. `--mode tiled` runs the tiled full-resolution path. Common
viewer/inference knobs include `--top-k`, `--frames-per-clip`,
`--max-in-flight`, `--max-gaze-tokens-each-frame`, `--inference-width`,
`--inference-height`, `--task-loss-requirement`, `--disable-task-loss-requirement`,
`--mask-cell-scale`, `--blend-alpha`, and `--show-fps`. `--show-gaze-ratio`
toggles the text overlay for per-frame and EMA output update ratio.
`--show-psnr=true` toggles PSNR in dB between the current input and rendered
output; the pixel comparison is skipped when this overlay is disabled.
`--help` lists the accepted values and aliases for mode-like options.
`--log-pipeline-timing` prints source capture, resize/prep, pack, input
upload/preprocess, model, visualization, and Bevy texture-update timing every
few seconds. In `tiled` mode, source frames are resized into a complete AnyRes
224px chunk grid and `--max-gaze-tokens-each-frame` controls the per-tile
generation cap. The output recovery stitches each tile-local scale grid into a
full-frame grid for that scale, matching upstream's mask recovery semantics.
Use `--perf-summary-frames N` with `--image-path` or another deterministic
source to process `N` inference outputs, print a JSON FPS/timing summary, and
exit. Add `--perf-summary-path target/autogaze-bevy-perf/run.json` to write the
same summary as a JSON artifact for hardware throughput reports.
The viewer display default is `--top-k 10`,
`--max-gaze-tokens-each-frame 0`, and `--frames-per-clip 2` for realtime use.
In `tiled` mode the defaults are `--top-k 2`,
`--max-gaze-tokens-each-frame 24`, `--frames-per-clip 2`, and
`--tile-batch-size 64`. Leave `--streaming-cache=true` enabled in realtime mode
so the decoder advances one new frame through the KV cache per inference.
`--max-in-flight 1` is the default camera admission policy: if inference is busy,
new camera frames refresh the buffered input window but do not queue stale model
jobs. Realtime streaming-cache mode is always kept to one in-flight task so KV
state advances in order; values above `1` apply to tiled or
`--streaming-cache=false` experiments. Use `--streaming-cache=false` for
full-window comparison. A max-gaze value of `0` uses the model default, which is
`198` for the NVIDIA config. The realtime
default uses this upstream budget so the mask does not collapse to only coarse
multi-scale cells. The maximum frame budget is
`max-gaze-tokens-each-frame * tile-count`, before task-loss stopping and
confidence filtering. The native CLI defaults to a 640px-wide
aspect-preserving source frame in `realtime` mode and 1280px-wide source frames
in `tiled` mode. The native camera path requests a matching 16:9 stream when
height is omitted. Pass explicit `--top-k`, `--tile-batch-size`,
`--inference-width`, and `--inference-height` values for fixed full-resolution
inspection.
`--display-transfer gpu` exercises the shared Bevy/Burn WebGPU texture bridge;
the default CPU transfer is currently the fastest measured app path.
For GPU interframe display, `--tensor-sparse-update-max-rects` and
`--tensor-sparse-update-max-ratio` choose when the tensor compositor uses sparse
rectangle copies instead of the dense mask path; use `0` rects to force dense
updates for apples-to-apples benchmarking.
Use `--require-hardware-adapter=true` for perf runs that should fail fast
instead of silently measuring a CPU/software render adapter.
Use `--load-model=false` to verify camera/preview rendering without waiting for
model load or inference.
From the repository root, run `tools/run_bevy_perf_matrix.sh --frames 120
--camera` on a real GPU host to collect deterministic static-source and live
camera throughput logs, per-case JSON summaries, and an aggregate
`summary.json` under `target/autogaze-bevy-perf/`.

`--visualization-mode full-blend` renders the current frame's alpha-blended
mask. The default `--blend-alpha` is intentionally subtle so the output panel
keeps the input frame readable; raise it when you want a stronger white update
overlay. The center mask panel colors the decoded multi-scale AutoGaze cells by
scale and draws crisp cell bounds. `--visualization-mode interframe
--keyframe-duration 30` preserves the previous output outside masked cells,
updates masked cells to the current input, and redraws a full keyframe every 30
processed frames. The gaze-ratio overlay reports the percentage of output pixels
updated on the current frame plus an EMA across processed frames. The PSNR
overlay reports current-frame and EMA dB for the output column compared to the
current input frame.

In `full-blend` mode the update ratio reports selected effective mask coverage
as a percentage of the full source frame. In `interframe` mode keyframes are
`100%`; intermediate frames report masked-cell coverage as a percentage of the
full source frame.
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
http://localhost:8080/?mode=tiled&visualization-mode=interframe&keyframe-duration=12&frames-per-clip=2&inference-width=1920&inference-height=1080&task-loss-requirement=0.7&tile-batch-size=4&show-fps=true&show-gaze-ratio=true&show-psnr=true
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
