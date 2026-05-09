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
are generated in one backend batch; the default is `8`, which keeps the 45-tile
1080p path away from large CUDA/WebGPU fusion graphs while preserving the same
tile layout.

## tensor pipeline api

The core crate exposes small, composable pipeline nodes for downstream video and
codec integrations. `TensorClipInput` accepts already-normalized Burn video
tensors in `[batch, time, channel, height, width]` layout. `RgbaClipInput`
accepts packed RGBA clips and converts them through the same AutoGaze processor
affine used by `trace_rgba_clip_with_mode`. Output nodes receive
`AutoGazePipelinePacket`, which carries the source tensor clip, decoded traces,
mode, and `top_k`; downstream crates can plug in `VecOutputNode`, `FnOutputNode`,
or their own `AutoGazeOutputNode` for Bevy rendering, file writing, transport, or
codec-side reconstruction. Packets also expose `frame_tensor`, `frame_mask`, and
`frame_pyramid_tokens` helpers so output nodes can stay tensor-native.

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
let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(
    AutoGazeTensorPipelineConfig {
        mode: AutoGazeInferenceMode::ResizeToModelInput,
        top_k: 10,
    },
);
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

## visualization

| mode | output behavior | update ratio |
|---|---|---|
| `full-blend` | redraws the current input with a white alpha-blended mask | `100%` |
| `interframe` | keeps prior output outside the current mask, updates masked cells to the current input, and redraws a full keyframe every `keyframe-duration` frames | masked-cell pixels / full-frame pixels, or `100%` on keyframes |

AutoGaze emits multi-scale token positions. For the NVIDIA config, the Rust
trace decoder maps those tokens back to `2x2`, `4x4`, `7x7`, and `14x14`
cells. In tiled full-resolution mode those are recovered as per-scale
full-frame grids before rendering with nearest sampling so the cell structure
stays crisp and scale-aligned.

The gaze ratio metric reports how much of the output frame changed compared to
a full-frame redraw. The Bevy overlay shows the current frame ratio plus an EMA
across processed frames. When `show-psnr` is enabled, Bevy also computes PSNR in
dB between the current input frame and the rendered output frame; this RGB pixel
comparison is skipped when the PSNR overlay is disabled.

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
ImageNet mean/std normalization). The mask GIF uses scale colors from the
decoded model grid metadata, drawing larger cells first and smaller cells on
top. The per-run ratios, PSNR, and detected cell scale histogram are checked in at
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
counts and ratios. use `set_visualization_mode("interframe")` and
`set_keyframe_duration(n)` to enable stateful interframe output-stream updates.
this is the low-level wasm-bindgen api demo.

## bevy

```sh
cargo run -p bevy_burn_autogaze -- --mode realtime --visualization-mode full-blend
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

Set `--show-fps=false` or `--show-gaze-ratio=false` to hide the default text
overlays, or `--show-psnr=true` to enable output PSNR. Set
`--log-pipeline-timing` to print source capture, resize/prep, pack, input
upload/preprocess, model, visualization, display, and total timing. The native CLI
defaults to the model generation budget (`--max-gaze-tokens-each-frame 0`) and a
640px-wide aspect-preserving source resize for realtime use. `--mode tiled`
defaults to a 1280px-wide source frame, `--top-k 2`,
`--max-gaze-tokens-each-frame 0`, `--frames-per-clip 2`, and
`--tile-batch-size 64`; set `--frames-per-clip 16` explicitly when you want the
upstream long-context clip length. The
viewer keeps prepared frame tensors rolling so multi-frame clips only
upload/preprocess the newest frame. Pass an explicit nonzero
`--max-gaze-tokens-each-frame` to cap generation for performance, or pass
explicit `--top-k`, `--tile-batch-size`, `--inference-width`, and
`--inference-height` values for fixed full-resolution inspection. Native
`realtime` requests a 640x360 camera stream when height is omitted so camera
decode does not dominate the realtime path.
The output panel uses a subtle `--blend-alpha` default, and processed inference
frames are not overwritten by raw camera previews while a model task is in
flight; this keeps wasm output monotonic and makes interframe accumulation
easier to inspect.
Use `--perf-summary-frames N` (or `perf-summary-frames=N` on wasm) for a
deterministic static-source perf run: native prints a JSON summary and exits,
while wasm exposes live samples on `window.__autogazePerf` and the final summary
on `window.__autogazePerfSummary`.
Run `cargo run -p bevy_burn_autogaze -- --help` for accepted
values, aliases, and value ranges.

The native app accepts CLI flags; the wasm app accepts the same viewer/inference
knobs through query parameters:

```text
http://localhost:8080/?mode=tiled&visualization-mode=interframe&keyframe-duration=12&frames-per-clip=2&inference-width=1920&inference-height=1080&task-loss-requirement=0.7&tile-batch-size=4&show-fps=true&show-gaze-ratio=true&show-psnr=true
```

For headless browsers or machines without a webcam, run the same Bevy UI from a
static source:

```text
http://localhost:8080/?source=static&frames-per-clip=1&inference-width=1920&inference-height=1080
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
by the NVIDIA config. real-model tile and KV-cache groups include 720p/1080p
two-frame cases plus 16-frame long-context cases. the real task-loss group
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
paths for single-scale and multi-scale crisp masks. the Bevy viewer bench
measures side-by-side visualization plus persistent image asset updates at 720p
and 1080p.

useful filters:

```sh
cargo bench --bench backend_pipeline -- autogaze_trace_video/webgpu/multiscale-32-64-112-224/anyres-tile-224
cargo bench --bench backend_pipeline -- autogaze_tile_batch_size/webgpu
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features webgpu -- autogaze_real_task_loss/webgpu
cargo bench --bench backend_pipeline -- autogaze_rgba_e2e_video/webgpu/multiscale-32-64-112-224/resize-224/720p
cargo bench --bench backend_pipeline -- autogaze_visualization/multiscale-32-64-112-224/interframe-delta
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline -- bevy_autogaze_viewer_pipeline/full-blend/1080p
```

## validation

```sh
cargo test
cargo test --features cuda --test backend_pipeline -- --nocapture
cargo clippy --all-targets --features cuda -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo package --allow-dirty
```

cuda/webgpu backend tests and benches skip cleanly when the requested
accelerator is not available on the host.
