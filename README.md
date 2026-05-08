# burn_autogaze 🔥👁️🎯

[![test](https://github.com/mosure/burn_autogaze/workflows/test/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3Atest)
[![deploy github pages](https://github.com/mosure/burn_autogaze/workflows/deploy%20github%20pages/badge.svg)](https://github.com/mosure/burn_autogaze/actions?query=workflow%3A%22deploy+github+pages%22)
[![crates.io](https://img.shields.io/crates/v/burn_autogaze.svg)](https://crates.io/crates/burn_autogaze)
[![docs.rs](https://docs.rs/burn_autogaze/badge.svg)](https://docs.rs/burn_autogaze)

burn-native [nvidia autogaze](https://huggingface.co/nvidia/AutoGaze) model
inference, fixation traces, white-mask visualization, and bevy/webgpu demos.

| input | mask | output |
|---|---|---|
| <img src="./docs/autogaze_birds_input.gif" alt="birds input clip" title="https://www.vecteezy.com/free-videos Wildlife Stock Videos by Vecteezy"> | <img src="./docs/autogaze_birds_mask.gif" alt="autogaze white mask"> | <img src="./docs/autogaze_birds_output.gif" alt="autogaze alpha-blended output"> |

## vibes

- lowercase, small, practical burn crate for gaze inference
- loads hugging face `config.json` + `model.safetensors`
- default fast path downsamples frames to the model's `224` input
- optional tiled full-resolution mode remaps local 224px tile predictions back
  into source-frame coordinates
- runs on ndarray, webgpu, cuda, and wasm/webgpu
- ships a plain wasm-bindgen api plus a symmetric native/wasm bevy viewer

## burn support

| burn_autogaze | burn | burn-store | status |
|---|---:|---:|---|
| `0.21.x` | `0.21.x` | `0.21.x` | current |
| `<0.21` | `<0.21` | `<0.21` | not supported in this repo |

## features

| feature | default | target | notes |
|---|---:|---|---|
| `ndarray` | yes | native | cpu reference backend |
| `webgpu` | yes | native/web | burn webgpu/wgsl backend |
| `wgpu` | no | native/web | burn wgpu backend without selecting the webgpu compiler feature |
| `cuda` | no | native | cuda backend |
| `wasm` | no | wasm32 | wasm-bindgen api over burn webgpu |
| `bevy-web-demo` | no | wasm32 | compatibility alias for demo builds |

## usage

```rust,no_run
use burn::backend::{wgpu, WebGpu};
use burn_autogaze::{AutoGazeClipShape, AutoGazeInferenceMode, AutoGazePipeline};

let device = wgpu::WgpuDevice::default();
wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, Default::default());

let pipeline = AutoGazePipeline::<WebGpu>::from_hf_dir("/path/to/AutoGaze", &device)?
    .with_max_gaze_tokens_each_frame(8);

// frames are [time, channels, height, width], normalized as f32
let shape = AutoGazeClipShape::new(2, 3, 720, 1280);
let frames = vec![0.0; shape.num_values()];

let trace = pipeline.trace_clip_from_frames_with_mode(
    &frames,
    shape,
    4,
    AutoGazeInferenceMode::ResizeToModelInput,
)?;

let tiled_trace = pipeline.trace_clip_from_frames_with_mode(
    &frames,
    shape,
    4,
    AutoGazeInferenceMode::tiled_model_input(224),
)?;

# Ok::<(), anyhow::Error>(())
```

`ResizeToModelInput` is the recommended realtime path. `TiledFullResolution`
keeps local full-res evidence, but it is much slower because every covered tile
runs through the model.

## wasm

```sh
cd web
npm run build:wasm
npm run serve
```

`WasmAutoGaze` loads `config.json` plus `model.safetensors` bytes, accepts RGBA
video clips, and returns white mask, alpha-blended, and `input | mask | blend`
RGBA buffers. this is the low-level wasm-bindgen api demo.

## bevy

```sh
cargo run -p bevy_burn_autogaze --features native -- --mode resize-224

cd crates/bevy_burn_autogaze
npm run build:wasm
npm run serve
```

`bevy_burn_autogaze` is the primary UI demo on both native and wasm. native and
browser builds render the same bevy app: the only platform split is camera/model
I/O (`nokhwa` or `--image-path` natively, browser camera plus `frame_input` on
wasm). both modes show the same bevy-rendered `input | mask | blend`
visualization and FPS overlay.

The native app accepts CLI flags; the wasm app accepts the same viewer/inference
knobs through query parameters:

```text
http://localhost:8080/?mode=tile-224&top-k=2&frames-per-clip=2&show-fps=true
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
```

the benchmark suite covers full-resolution source clips (`1280x720` and
`1920x1080`), `resize-224`, `tile-224`, embedding, trace generation, and a
real-model group when the autogaze hugging face snapshot is available.

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
