# burn_autogaze 🔥👁️🎯

[![test](https://github.com/mosure/burn_autogaze/workflows/test/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3Atest)
[![deploy github pages](https://github.com/mosure/burn_autogaze/workflows/deploy%20github%20pages/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3A%22deploy+github+pages%22)
[![crates.io](https://img.shields.io/crates/v/burn_autogaze.svg)](https://crates.io/crates/burn_autogaze)
[![docs.rs](https://docs.rs/burn_autogaze/badge.svg)](https://docs.rs/burn_autogaze)

burn-native [nvidia autogaze](https://huggingface.co/nvidia/AutoGaze) model
inference, fixation traces, crisp multi-scale token-cell mask visualization, and
bevy/webgpu demos.

| input | multi-scale mask | interframe output |
|---|---|---|
| <img src="./docs/autogaze_birds_input.gif" alt="birds input clip" title="https://www.vecteezy.com/free-videos Wildlife Stock Videos by Vecteezy"> | <img src="./docs/autogaze_birds_mask.gif" alt="scale-colored crisp autogaze token-cell mask"> | <img src="./docs/autogaze_birds_output.gif" alt="autogaze interframe output stream"> |

## features

### high-level

- practical burn-native gaze inference for video clips and RGBA frame buffers
- loads hugging face `config.json` + `model.safetensors`
- default fast path downsamples frames to the model's `224` input
- optional AnyRes tiled mode resizes source frames into complete 224px chunk
  grids, stitches each scale's tile-local feature grid, and maps cells back into
  source-frame coordinates
- tiled inference processes chunks in bounded tile batches by default so CUDA
  and WebGPU do not build one very large autoregressive graph for 1080p clips
- tiled mode decodes every non-padded generated token up to
  `max_gaze_tokens_each_frame`, matching upstream mask recovery instead of
  truncating the frame to a sparse global top-k list
- the NVIDIA fixed inference defaults are applied by default:
  `gazing_ratio=0.75` and `task_loss_requirement=0.7`
- task-loss stopping can also be configured as an L1 reconstruction-error dB
  target via app/API helpers, e.g. `28 dB` maps to a threshold of about `0.04`
- mask visualizations preserve the model's multi-scale `2x2`/`4x4`/`7x7`/`14x14`
  token cells with nearest sampling
- optional `interframe` visualization keeps stale output outside the mask and
  updates masked cells to the current input between configurable keyframes
- bevy overlay can show FPS, gaze/update ratio, and output PSNR with per-frame
  and EMA values
- runs on ndarray, webgpu, cuda, and wasm/webgpu
- ships a plain wasm-bindgen api plus a symmetric native/wasm bevy viewer

### cargo features

| feature | default | target | notes |
|---|---:|---|---|
| `ndarray` | yes | native | cpu reference backend |
| `webgpu` | yes | native/web | burn webgpu/wgsl backend |
| `wgpu` | no | native/web | burn wgpu backend without selecting the webgpu compiler feature |
| `cuda` | no | native | cuda backend |
| `wasm` | no | wasm32 | wasm-bindgen api over burn webgpu |
| `bevy-web-demo` | no | wasm32 | compatibility alias for demo builds |

## burn support

| burn_autogaze | burn | burn-store | status |
|---|---:|---:|---|
| `0.21.x` | `0.21.x` | `0.21.x` | current |
| `<0.21` | `<0.21` | `<0.21` | not supported in this repo |

## usage

```rust,no_run
use burn::backend::{wgpu, WebGpu};
use burn_autogaze::{
    AutoGazeClipShape, AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape,
};

let device = wgpu::WgpuDevice::default();
wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, Default::default());

let pipeline = AutoGazePipeline::<WebGpu>::from_hf_dir("/path/to/AutoGaze", &device)?;

// frames are [time, channels, height, width] and already AutoGaze-preprocessed
let shape = AutoGazeClipShape::new(16, 3, 720, 1280);
let frames = vec![0.0; shape.num_values()];

let trace = pipeline.trace_clip_from_frames_with_mode(
    &frames,
    shape,
    10,
    AutoGazeInferenceMode::ResizeToModelInput,
)?;

let tiled_trace = pipeline.trace_clip_from_frames_with_mode(
    &frames,
    shape,
    10,
    AutoGazeInferenceMode::tiled_model_input(224),
)?;

let rgba = vec![0_u8; shape.clip_len * shape.height * shape.width * 4];
let rgba_trace = pipeline.trace_rgba_clip_with_mode(
    &rgba,
    AutoGazeRgbaClipShape::new(shape.clip_len, shape.height, shape.width),
    10,
    AutoGazeInferenceMode::ResizeToModelInput,
    &device,
)?;

# Ok::<(), anyhow::Error>(())
```

`ResizeToModelInput` is the recommended realtime path. The AutoGaze gaze
decoder is trained around a fixed per-frame token vocabulary (`265` tokens for
the NVIDIA multi-scale config), so a larger source frame is not passed as
"more patches" to one decoder call. Upstream handles high-resolution clips by
chopping them into fixed-size chunks, or by resizing arbitrary-aspect clips into
a complete grid of fixed-size chunks. `TiledResizeToGrid` mirrors that AnyRes
path by resizing the source into a full `224px` tile canvas before chunking, so
all recovered per-scale grids have complete tile-local cells. `TiledFullResolution`
is still available for exact source-space edge padding. Tiled inference keeps
local high-res evidence, but it is much slower because every covered tile runs
through the model. In tiled mode, the maximum fixation budget for each frame is
`max_gaze_tokens_each_frame * tile_count`; task-loss stopping and confidence
filtering usually reduce the visible mask below that budget. The `top_k`
argument is retained as a compatibility lower bound for trace slots and does
not discard generated non-padded mask tokens.
The tile output recovery follows upstream's per-scale mask stitching: each
`2x2`, `4x4`, `7x7`, or `14x14` tile-local feature map is stitched into a
full-frame grid for that scale. For `1920x1080`, the AnyRes canvas is
`2016x1120`, yielding complete `18x10`, `36x20`, `63x35`, and `126x70`
stitched grids. `AutoGazePipeline::set_tile_batch_size` controls how many tiles
are generated in one backend batch; the default is `64`, which batches a full
45-tile 1080p AnyRes frame on capable backends. Lower it when a CUDA/WebGPU
backend favors smaller autoregressive graphs; the stitched tile layout remains
unchanged.

## tensor pipeline api

The core crate exposes small, composable pipeline nodes for downstream video and
codec integrations. `TensorClipInput` accepts already-normalized Burn video
tensors in `[batch, time, channel, height, width]` layout. `RgbaClipInput`
accepts packed RGBA clips and converts them through the same mode-aware RGBA
preparation as `trace_rgba_clip_with_mode`: realtime resize mode applies the
upstream-style shortest-edge processor resize before the model, while tiled modes
preserve the configured source tensor. Use `RgbaClipInput::with_inference_mode`
when the graph mode is not the default realtime path. Output nodes receive
`AutoGazePipelinePacket`, which carries the source tensor clip, mode, `top_k`,
and optionally decoded traces. Trace generation is disabled by default because it
requires model generation and backend-to-host reads; set `emit_traces` only for
consumers that need fixation points or masks. Downstream crates can plug in
`VecOutputNode`, `FnOutputNode`, or their own `AutoGazeOutputNode` for Bevy
rendering, file writing, transport, or codec-side reconstruction. Packets also
expose `frame_tensor`, `frame_mask`, and `frame_pyramid_tokens` helpers so output
nodes can stay tensor-native when traces are enabled.
For live RGBA sources, `AutoGazeRgbaFrameQueue` keeps a fixed-length frame
window and recycles clip buffers, so Bevy, file, and camera frontends can share
the same clip packing path instead of each carrying their own rolling buffer
logic.

```rust,no_run
use burn::backend::NdArray;
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazeRgbaClip, AutoGazeRgbaClipShape,
    AutoGazeTensorPipeline, AutoGazeTensorPipelineConfig, RgbaClipInput,
    VecOutputNode,
};

type B = NdArray<f32>;
let device = Default::default();
let pipeline = burn_autogaze::AutoGazePipeline::<B>::load("/path/to/AutoGaze", &device)?;
let shape = AutoGazeRgbaClipShape::new(2, 720, 1280);
let rgba = vec![0_u8; shape.clip_len * shape.height * shape.width * 4];
let input = RgbaClipInput::new().with_clip(AutoGazeRgbaClip::new(rgba, shape)?);
let output = VecOutputNode::new();
let config = AutoGazeTensorPipelineConfig::default()
    .with_mode(AutoGazeInferenceMode::ResizeToModelInput)
    .with_top_k(10)
    .with_emit_traces(true);
let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(config);
graph.run_next(&device)?;

# Ok::<(), anyhow::Error>(())
```

For ViT-like image-pyramid or codec work, `fixation_image_mask_tensor` converts
decoded AutoGaze points into a Burn mask tensor, `apply_image_mask` keeps or
fills image regions without leaving the backend, and
`tokenize_masked_image_pyramid` emits dense weighted tokens for requested
`ImagePyramidLevel`s. `sparsify_image_pyramid_tokens` then selects the highest
mask-density tokens with backend `topk`, avoiding synchronous host readback on
wasm/WebGPU.

When the downstream model already has its own patchifier, use the sparse
readout helpers instead of reinterpreting AutoGaze token ids directly.
`fixation_points_to_readout_rects` and `trace_to_frame_readout_rects` preserve
the decoded multi-scale cell geometry, while
`fixation_points_to_readout_tokens` and `trace_frame_readout_tokens` project
that geometry onto an arbitrary image-token grid with optional dilation and
per-frame token caps. `generated_to_frame_readout_tokens` provides the same
projection directly from `AutoGazeGenerateOutput` for hot paths that do not need
to allocate full decoded traces. `frame_readout_tokens_to_video_tokens` and
`frame_readout_rects_to_video_tokens` then map the per-frame image readout onto
generic `[temporal, row, col]` video-token grids for downstream sparse-video
models, while `generated_to_video_readout_tokens` and
`trace_to_video_readout_tokens` provide the same path as one-call adapters.
`video_readout_tokens_to_coords` and
`batched_video_readout_tokens_to_coords` then adapt those indices to
`[batch, temporal, row, col]` coordinate rows for downstream sparse patchifiers
such as `burn_flex_gmm`. `SparseVideoPatchGeometry` derives the video-token
grid from the downstream patchifier's frame, tubelet, and patch dimensions, so
consumers do not need to duplicate V-JEPA-style video-grid shape checks before
asking for coordinate tensors.
This keeps AutoGaze's model-specific scale layout in `burn_autogaze`,
leaving crates such as `burn_jepa` to wrap the resulting indices in their own
mask types and to use sparse patchification backends such as `burn_flex_gmm` for
the actual selected-token readout.
For tensor-node pipelines, set
`AutoGazeTensorPipelineConfig::emit_readout_points` when output nodes need
sparse readout without full trace allocation. Realtime resize mode decodes
generated output directly; tiled modes store remapped source-frame fixation
points so callers can still apply their own readout thresholds, dilation, and
token caps. `AutoGazePipelinePacket::video_readout_tokens` exposes the same
generic sparse-video projection directly to output nodes regardless of whether
the packet contains traces, readout points, or realtime generated output. Async
consumers can use
`AutoGazePipeline::readout_points_with_mode_async` for the same no-trace
readout path without falling back to full trace allocation on wasm/WebGPU.
See [docs/sparse-readout-integration.md](./docs/sparse-readout-integration.md)
for the ownership boundary between decoded AutoGaze geometry, V-JEPA tubelet
projection, and sparse patchify kernels.

```rust,no_run
use burn_autogaze::{
    AutoGazeConfig, AutoGazeGenerateOutput, SparseReadoutGrid, SparseReadoutOptions,
    SparseVideoPatchGeometry, SparseVideoReadoutOptions, SparseVideoReadoutProjection,
    generated_to_video_readout_tokens,
};

# let config = AutoGazeConfig::default();
# let generated = AutoGazeGenerateOutput {
#     gazing_pos: vec![Vec::new()],
#     num_gazing_each_frame: Vec::new(),
#     if_padded_gazing: vec![Vec::new()],
#     confidences: vec![Vec::new()],
# };
let grid = SparseReadoutGrid::square_from_token_count(config.num_vision_tokens_each_frame)?;
let options = SparseReadoutOptions::default()
    .with_max_fixations_per_frame(10)
    .with_dilation(1)
    .with_max_tokens_per_frame(32);
let _video_tokens = generated_to_video_readout_tokens(
    &generated,
    &config,
    0,
    grid,
    SparseVideoPatchGeometry::square_patch(2, 224, 224, 2, 16).readout_grid()?,
    options,
    SparseVideoReadoutOptions::default()
        .with_tubelet_size(2)
        .with_exact_tokens(32),
)?;
let _coord_projection = SparseVideoReadoutProjection::from_patch_geometry(
    grid,
    SparseVideoPatchGeometry::square_patch(2, 224, 224, 2, 16),
)?;

# Ok::<(), anyhow::Error>(())
```

For sparse video or renderer outputs, `fixation_sparse_update_plan` exposes the
same native multi-scale pixel rectangles used by the interframe visualization.
Use `copy_sparse_update_rgba` for CPU-side sparse accumulation,
`copy_sparse_update_tensor` for backend tensor-side accumulation, or pass the
plan's `FixationPixelRect`s to a renderer-specific GPU compositor. Use
`fixation_effective_sparse_update_plan` only when the consumer explicitly wants a
finest-active-grid footprint instead of the upstream-style native scale cells.
Runtime metric helpers such as `AutoGazeGazeRatioStats`, `AutoGazePsnrStats`,
`fps_from_millis`, and `format_psnr_db` live in the core crate so apps report the
same values without duplicating smoothing or formatting behavior.
`AutoGazeTensorVisualizationState::last_interframe_path` reports whether the
latest tensor-side interframe output used a keyframe, sparse rectangle update, or
dense mask update, which keeps app and benchmark diagnostics aligned.
`AutoGazeTensorVisualizationOptions::with_sparse_update_policy` and the
`DEFAULT_TENSOR_SPARSE_UPDATE_MAX_*` constants expose the sparse-rectangle
threshold used by tensor interframe composition so viewers can benchmark dense
versus sparse update tradeoffs without forking visualization logic.
`AutoGazeInferenceSequencer` provides the shared async stale-result gate used by
camera frontends to drop late model results instead of rendering frames out of
order.
`AutoGazeRealtimePolicy` captures the default one-in-flight frame admission and
preview behavior used by realtime wrappers; Bevy exposes this as
`--max-in-flight` / `max-in-flight`. Default realtime streaming-cache mode keeps
the effective policy to one in-flight task so KV state advances in order;
higher limits are for tiled or full-window non-streaming runs.
`should_use_streaming_cache` centralizes the rule that streaming KV cache is only
used for multi-frame resize-mode realtime inference.

## visualization

| mode | output behavior | update ratio |
|---|---|---|
| `full-blend` | redraws the current input with a white alpha-blended mask | selected effective mask pixels / full-frame pixels |
| `interframe` | keeps prior output outside the current mask, updates masked cells to the current input, and redraws a full keyframe every `keyframe-duration` frames | masked-cell pixels / full-frame pixels, or `100%` on keyframes |

AutoGaze emits multi-scale token positions. For the NVIDIA config, the Rust
trace decoder maps those tokens back to `2x2`, `4x4`, `7x7`, and `14x14`
cells. In tiled full-resolution mode those are recovered as per-scale
full-frame grids before rendering with nearest sampling so the cell structure
stays crisp without smoothing.
The upstream NVIDIA visualizer renders each scale in its own row rather than
forcing all scales into a single combined overlay. This matters because the
default `7x7` scale is not a quadtree subdivision of `2x2` or `4x4`. Use
`scale-rows` for the most upstream-faithful diagnostic mask view; it reserves a
stable row for each standard scale even when a frame has no active cells at
that scale. Use `overlay` when you specifically want the unioned sparse-update
footprint.

The gaze ratio metric reports how much of the output frame changed compared to
a full-frame redraw. The Bevy overlay shows the current frame ratio plus an EMA
across processed frames. When `show-psnr` is enabled, Bevy also computes PSNR in
dB between the current input frame and the rendered output frame; this RGB pixel
comparison is skipped when the PSNR overlay is disabled.
AutoGaze's task-loss head predicts the upstream reconstruction loss, which is
L1 for the NVIDIA VideoMAE reconstruction task, not the output-column PSNR
metric. The `task-loss-requirement-db` interface is therefore a convenience
conversion using `threshold = 10^(-dB / 20)`.

The README GIFs are generated from `/home/mosure/Videos/birds.mp4` at
`1920x1080` inference resolution with the NVIDIA AutoGaze weights and the same
Rust pipeline exposed by the crate. `trace_rgba_clip_with_mode(..., anyres-tile-224)`
resizes each 16-frame clip into a complete 45-tile AnyRes canvas before the
resulting stream is downsampled for README display, using the model default
`max_gaze_tokens_each_frame=198` and `task_loss_requirement=0.7`.
The maximum fixation budget is 8910 tokens per high-res frame before task-loss
stopping and confidence filtering; tiles are generated in batches of 4. The RGBA
convenience path applies the upstream
AutoGazeImageProcessor affine preprocessing (`image / 127.5 - 1`, then
ImageNet mean/std normalization). The mask GIF uses the upstream-style
per-scale row view with stable rows and colors from the decoded model grid
metadata. The per-run ratios, PSNR, and detected cell scale histogram are checked in at
[`docs/autogaze_birds_metrics.json`](./docs/autogaze_birds_metrics.json).

```sh
cargo run --example render_readme_assets --features webgpu --no-default-features -- \
  --input /home/mosure/Videos/birds.mp4 \
  --model-dir /path/to/AutoGaze \
  --inference-width 1920 --inference-height 1080 \
  --tile-batch-size 4 \
  --out-dir docs
```

## wasm

```sh
cd web
npm run build:wasm
npm run serve
```

`WasmAutoGaze.create(configJson, safetensors)` loads `config.json` plus
`model.safetensors` bytes through async WebGPU setup, accepts RGBA video clips,
and returns binary token-cell mask, visualization output, and `input | mask |
output` RGBA buffers (`output_rgba()` is the preferred accessor, with
`blend_rgba()` kept for compatibility). outputs also expose mask/update pixel
counts, ratios, and output PSNR in dB against the latest input frame. The
low-level wasm defaults remain resize-oriented:
`top_k=10`, model-config generation budget, `tile_batch_size=64`, and
`blend_alpha=0.38`. use `set_anyres_tiled_mode(224)`,
`set_visualization_mode("interframe")`, and
`set_keyframe_duration(n)` to enable stateful interframe output-stream updates.
this is the low-level wasm-bindgen api demo.

## bevy

```sh
cargo run -p bevy_burn_autogaze
cargo run -p bevy_burn_autogaze -- --mode realtime
cargo run -p bevy_burn_autogaze -- --mode tiled --visualization-mode interframe

cd crates/bevy_burn_autogaze
npm run build:wasm
npm run serve
```

`bevy_burn_autogaze` is the primary UI demo on both native and wasm. native and
browser builds render the same bevy app: the only platform split is camera/model
I/O (`nokhwa` or `--image-path` natively, browser camera plus `frame_input` on
wasm). both modes show the same bevy-rendered `input | mask | output`
visualization plus toggleable FPS, gaze/update-ratio, and output-PSNR overlays.

Set `--show-fps=false`, `--show-gaze-ratio=false`, or `--show-psnr=false` to
hide the default text overlays. Set
`--log-pipeline-timing` to print source capture, resize/prep, pack, input
upload/preprocess, model, visualization, display, and total timing. The native
CLI default is the live realtime profile: `--mode realtime`,
`--visualization-mode interframe`, `--top-k 10`,
`--max-gaze-tokens-each-frame 0` (the upstream model budget, 198 tokens for
NVIDIA AutoGaze), `--frames-per-clip 16`, 640px aspect-preserving input,
`--display-transfer gpu`, PSNR overlay, `--blend-alpha 0.38`,
`--keyframe-duration 0`, and `--streaming-cache=true` for a continuous rolling
KV window. Pass `--streaming-cache=false` for a full-window comparison that
reprocesses the whole clip each inference. Pass explicit
`--max-gaze-tokens-each-frame`,
`--top-k`, `--tile-batch-size`, `--inference-width`, and
`--inference-height` values for fixed full-resolution inspection. Native
`realtime` requests a 640x360 camera stream when height is omitted so camera
decode does not dominate the realtime path. `--mode tiled` defaults to a bounded
1280px-wide aspect-preserving input, `--top-k 2`, 24 generated tokens per tile,
and tile batch size 64.
`--display-transfer gpu` enables the Bevy/Burn shared-device texture bridge;
it is the default display path for live runs.
For GPU display-transfer interframe runs, `--tensor-sparse-update-max-rects`
and `--tensor-sparse-update-max-ratio` control when the tensor compositor uses a
sparse rectangle copy instead of the dense mask path; pass `0` rects to force
dense tensor updates.
Use `--require-hardware-adapter=true` for throughput runs that should fail fast
instead of reporting CPU/software render-adapter numbers.
Processed inference frames are not overwritten by raw camera previews while a model task is in
flight; this keeps wasm output monotonic and makes interframe accumulation
easier to inspect.
Use `--perf-summary-frames N` (or `perf-summary-frames=N` on wasm) for a
deterministic static-source perf run: native prints a JSON summary and exits,
while wasm exposes live samples on `window.__autogazePerf` and the final summary
on `window.__autogazePerfSummary`. Both include the latest frame dimensions,
FPS, gaze ratio, PSNR fields, frame counts, tensor interframe path,
render adapter metadata, and the configured/effective realtime admission
policy, including the tensor sparse-update policy. Native runs can also pass
`--perf-summary-path target/perf.json` to write the same summary directly as a
JSON artifact for hardware throughput reports.
Run `cargo run -p bevy_burn_autogaze -- --help` for accepted
values, aliases, and value ranges.

The native app accepts CLI flags; the wasm app accepts the same viewer/inference
knobs through query parameters:

```text
http://localhost:8080/?mode=tiled&visualization-mode=interframe&mask-visualization=scale-rows&keyframe-duration=0&frames-per-clip=16&inference-width=1920&inference-height=1080&tile-batch-size=4&show-fps=true&show-gaze-ratio=true&show-psnr=true
```

That query matches the full-resolution docs birds asset profile. The no-arg
native and web paths use the realtime default above.

For headless browsers or machines without a webcam, run the same Bevy UI from a
static source:

```text
http://localhost:8080/?source=static&frames-per-clip=1
http://localhost:8080/?source=static&mode=tiled&frames-per-clip=16&inference-width=1920&inference-height=1080
http://localhost:8080/?image-url=./frame.png&frames-per-clip=1
```

The web build fetches NVIDIA AutoGaze from Hugging Face by default. override
URLs with `config-url` and `weights-url` query parameters. the bevy crate pins
bevy to
`ae2fcc0353d95e887470f0f6fc8a7e434e5549ce` so burn and bevy resolve through
`wgpu` v29.

## benches

```sh
cargo bench --bench backend_pipeline --features webgpu
cargo bench --bench backend_pipeline --features cuda
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features cuda -- autogaze_real_trace_video
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features cuda -- autogaze_real_task_loss
AUTOGAZE_HF_DIR=/path/to/AutoGaze AUTOGAZE_VIDEO=/path/to/video.mp4 cargo bench --bench backend_pipeline --features cuda -- autogaze_real_video_file
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline
```

the benchmark suite covers full-resolution source clips (`1280x720` and
`1920x1080`), `resize-224`, `anyres-tile-224`, embedding, trace generation, and a
real-model group when the autogaze hugging face snapshot is available. synthetic
backend benches run the full matrix across `single-scale-224` and
`multiscale-32-64-112-224` model layouts, so tiled high-resolution runs are
measured with the same AnyRes tile layout and multi-scale gaze-token layout used
by the NVIDIA config. real-model tile and KV-cache groups include the realtime
640x360 default plus 720p/1080p two-frame cases. 16-frame long-context stress
cases are included for CUDA and can be enabled for WebGPU by setting
`AUTOGAZE_BENCH_LONG_CONTEXT=1`. the real task-loss group
compares model-default stopping, disabled stopping, and an explicit `0.7`
threshold for resize and tiled modes. the real video-file group decodes a short
RGBA clip with `ffmpeg` from `AUTOGAZE_VIDEO` (falling back to
`/home/mosure/Videos/birds.mp4` when present) so deterministic tensor benches
can be compared against a real source pipeline. the tile-batch group measures 1080p
multiscale tiled embedding and trace generation across tile batch sizes `1`,
`2`, `4`, `8`, `16`, `32`, and `64`, which is useful for finding the fastest
batch that still fits the target backend and model. the RGBA e2e group measures
source RGBA conversion, trace generation, and crisp visualization together. the
visualization group also
measures `full-blend`, `interframe-keyframe`, and `interframe-delta` output
paths for single-scale and multi-scale crisp masks. the tensor visualization
group measures Burn-side side-by-side and split-panel tensor compositing on each
enabled backend, with `model-fixations`, `tiny-sparse`, and `coarse-dense`
fixation cases to make sparse rectangle updates and dense mask updates visible
as separate measurements; this is the lane to use when checking whether the Bevy
GPU display-transfer path is dominated by tensor composition or texture handoff.
the Bevy viewer bench measures side-by-side and split-panel CPU
visualization plus persistent image asset updates at 720p and 1080p, also split
by `multiscale`, `tiny-sparse`, and `coarse-dense` fixation cases.

useful filters:

```sh
cargo bench --bench backend_pipeline -- autogaze_trace_video/webgpu/multiscale-32-64-112-224/anyres-tile-224
cargo bench --bench backend_pipeline -- autogaze_tile_batch_size/webgpu
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features webgpu -- autogaze_real_task_loss/webgpu
cargo bench --bench backend_pipeline -- autogaze_rgba_e2e_video/webgpu/multiscale-32-64-112-224/resize-224/720p
cargo bench --bench backend_pipeline -- autogaze_visualization/multiscale-32-64-112-224/interframe-delta
cargo bench --bench backend_pipeline --features webgpu -- autogaze_tensor_visualization/webgpu/multiscale-32-64-112-224/interframe-delta/tiny-sparse
cargo bench --bench backend_pipeline --features webgpu -- autogaze_tensor_visualization/webgpu/multiscale-32-64-112-224/interframe-delta/coarse-dense
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline -- bevy_autogaze_viewer_pipeline/interframe-delta-panels/coarse-dense/1080p
```

## validation

```sh
cargo run -p xtask -- release-readiness
cargo test
cargo test --features cuda --test backend_pipeline -- --nocapture
cargo clippy --all-targets --features cuda -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
cargo package --allow-dirty
cd crates/bevy_burn_autogaze && npm ci && npm run test:browser
cargo run -p xtask -- bevy-perf-matrix --dry-run --frames 2 --camera
```

`cargo run -p xtask -- release-readiness` runs the local non-hardware release gate:
root and Bevy tests, native/wasm checks, clippy with warnings denied, benchmark
compilation, package verification, and `git diff --check`. Pass `--browser` on
a host with a normal Node/Playwright setup to include the static-source browser
smoke, or `--real-model-browser` after staging local wasm model assets.
`cargo run -p xtask -- check-bevy-wasm-demo --browser` is the narrower Pages/demo gate used
by the deploy workflow; it checks the Bevy wasm target, installs the matching
`wasm-bindgen-cli`, installs npm dependencies, builds `www/out`, and runs the
static-source browser smoke. If the system `node`/`npm`/`npx` are Snap-provided
or otherwise unusable, pass `--node-bin-dir /path/to/node/bin` or set
`AUTOGAZE_NODE_BIN_DIR=/path/to/node/bin`. In sandboxed environments where
Playwright cannot use `sudo` for OS dependency installation, pass
`--no-browser-deps` and provide a browser cache such as
`PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright`.

cuda/webgpu backend tests and benches skip cleanly when the requested
accelerator is not available on the host.
See [docs/completion-audit.md](./docs/completion-audit.md) for the current
coverage checklist, browser-test blocker notes, and hardware FPS runbook. Run
`cargo run -p xtask -- bevy-perf-matrix --frames 120 --camera` on a real GPU host to
capture native Bevy throughput logs, per-case JSON summaries, and aggregate
`summary.json` with CPU-adapter failures guarded by
`--require-hardware-adapter=true`. The matrix uses `cargo run --release` by
default; pass `--profile dev` only for command-path debugging. Use
`--case-timeout-seconds N` to adjust the per-case timeout on slow first-build or
driver-tuning hosts.

Use `cargo run -p xtask -- upstream-fixture-matrix` to generate upstream
NVIDIA/Python parity fixtures from a manifest and immediately rerun the
fixture-only parity check:

```sh
cargo run -p xtask -- upstream-fixture-matrix \
  --manifest docs/upstream_fixture_matrix.example.json \
  --run-parity-test
```

The matrix wrapper is also dependency-light for `--help` and `--dry-run`; the
example manifest keeps outputs under `tests/fixtures`, where the existing Rust
fixture tests discover them automatically.
