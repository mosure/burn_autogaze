# burn_autogaze

[![test](https://github.com/mosure/burn_autogaze/workflows/test/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3Atest)
[![deploy github pages](https://github.com/mosure/burn_autogaze/workflows/deploy%20github%20pages/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3A%22deploy+github+pages%22)
[![crates.io](https://img.shields.io/crates/v/burn_autogaze.svg)](https://crates.io/crates/burn_autogaze)
[![docs.rs](https://docs.rs/burn_autogaze/badge.svg)](https://docs.rs/burn_autogaze)

Burn-native inference for the
[NVIDIA AutoGaze](https://huggingface.co/nvidia/AutoGaze) model, with
fixation traces, multi-scale token-cell masks, interframe reconstruction, and
native/wasm Bevy demos.

| input | mask | output |
|---|---|---|
| <img src="./docs/autogaze_birds_input.gif" alt="birds input clip" title="https://www.vecteezy.com/free-videos Wildlife Stock Videos by Vecteezy"> | <img src="./docs/autogaze_birds_mask.gif" alt="scale-colored autogaze token-cell mask"> | <img src="./docs/autogaze_birds_output.gif" alt="autogaze interframe output stream"> |

## features

### high-level

- Loads Hugging Face `config.json` and `model.safetensors` weights.
- Runs on ndarray, WebGPU, CUDA, and wasm/WebGPU backends.
- Supports the realtime resize path used for low-latency streams.
- Supports AnyRes-style tiled inference for high-resolution inspection.
- Decodes NVIDIA multi-scale masks as `2x2`, `4x4`, `7x7`, and `14x14` token
  cells, including tiled source-frame remapping.
- Provides interframe output accumulation, gaze/update ratio, and output PSNR
  helpers.
- Exposes tensor-native pipeline nodes for camera, file, RGBA, Burn tensor, and
  downstream sparse-token integrations.
- Ships a wasm-bindgen API and a symmetric native/wasm Bevy viewer.

### cargo features

| feature | default | target | notes |
|---|---:|---|---|
| `ndarray` | yes | native | CPU reference backend |
| `webgpu` | yes | native/web | Burn WebGPU backend |
| `wgpu` | no | native/web | Burn WGPU backend without selecting the WebGPU compiler feature |
| `cuda` | no | native | CUDA backend |
| `wasm` | no | wasm32 | wasm-bindgen API over Burn WebGPU |
| `bevy-web-demo` | no | wasm32 | compatibility alias for demo builds |

## burn support

| burn_autogaze | burn | burn-store | status |
|---|---:|---:|---|
| `0.21.x` | `0.21.x` | `0.21.x` | current |
| `<0.21` | `<0.21` | `<0.21` | not supported in this repo |

## quick start

```rust,no_run
use burn::backend::{wgpu, WebGpu};
use burn_autogaze::{
    AutoGazeClipShape, AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape,
};

let device = wgpu::WgpuDevice::default();
wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, Default::default());

let pipeline = AutoGazePipeline::<WebGpu>::from_hf_dir("/path/to/AutoGaze", &device)?;

let shape = AutoGazeClipShape::new(16, 3, 720, 1280);
let frames = vec![0.0; shape.num_values()];
let trace = pipeline.trace_clip_from_frames_with_mode(
    &frames,
    shape,
    10,
    AutoGazeInferenceMode::ResizeToModelInput,
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

`ResizeToModelInput` is the recommended realtime path. The model uses a fixed
per-frame token vocabulary (`265` tokens for the NVIDIA multi-scale config), so
high-resolution operation is handled by tiled chunking rather than by passing a
larger patch sequence into one decoder call. `TiledResizeToGrid` resizes frames
into complete `224px` tile grids before stitching each scale back into source
coordinates.

## bevy viewer

```sh
cargo run -p bevy_burn_autogaze
cargo run -p bevy_burn_autogaze -- --mode realtime
cargo run -p bevy_burn_autogaze -- --mode tiled --visualization-mode interframe
```

The no-arg native default is the realtime profile: resize mode, 16-frame rolling
KV window, model-config generation budget, adaptive display transfer, interframe
output, PSNR overlay, and no periodic visualization keyframes. The Bevy crate
selects native or wasm dependencies by target, so platform features are not
needed for normal runs.

See [crates/bevy_burn_autogaze/README.md](./crates/bevy_burn_autogaze/README.md)
for CLI/query parameters, static-source browser runs, performance summaries,
and Pages demo notes.

## wasm

```sh
cd web
npm run build:wasm
npm run serve
```

`WasmAutoGaze.create(configJson, safetensors)` performs async WebGPU setup,
loads model bytes, accepts RGBA clips, and returns mask/output buffers plus
ratio and PSNR metrics. See [web/README.md](./web/README.md) for the
wasm-bindgen API.

## docs

- [docs/api.md](./docs/api.md): core pipeline, visualization, readout, and
  sparse-token adapters.
- [docs/README.md](./docs/README.md): checked-in birds assets and regeneration
  command.
- [docs/benchmarking.md](./docs/benchmarking.md): benchmark lanes and useful
  filters.
- [docs/performance-goal-loop.md](./docs/performance-goal-loop.md):
  iterative performance, quality, metrics, and hot-path cleanup loop.
- [docs/validation.md](./docs/validation.md): release, wasm, browser, and
  upstream fixture gates.
- [docs/sparse-readout-integration.md](./docs/sparse-readout-integration.md):
  boundary between AutoGaze geometry and downstream sparse patchification.

## validation

```sh
cargo run -p xtask -- release-readiness
cargo test
cargo clippy --all-targets -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
```

Hardware-specific CUDA/WebGPU tests and browser perf checks are documented in
[docs/validation.md](./docs/validation.md).
