# benchmarking

Core benchmarks live in `benches/backend_pipeline.rs`; Bevy viewer-path
benchmarks live in `crates/bevy_burn_autogaze/benches/viewer_pipeline.rs`.

```sh
cargo bench --bench backend_pipeline --features webgpu
cargo bench --bench backend_pipeline --features cuda
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features cuda -- autogaze_real_trace_video
AUTOGAZE_HF_DIR=/path/to/AutoGaze AUTOGAZE_VIDEO=/path/to/video.mp4 cargo bench --bench backend_pipeline --features cuda -- autogaze_real_video_file
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline
```

## coverage

- `1280x720` and `1920x1080` source clips.
- `resize-224`, AnyRes tiled, and full-resolution tiled paths.
- Single-scale and NVIDIA-style multi-scale model layouts.
- Real NVIDIA model lanes when `AUTOGAZE_HF_DIR` is set.
- Tile batch sizes `1`, `2`, `4`, `8`, `16`, `32`, and `64`.
- KV-cache versus full-window decoder cases.
- Task-loss stopping disabled, model-default, and explicit threshold cases.
- RGBA source conversion plus trace generation.
- CPU RGBA visualization and tensor-side visualization.
- Sparse rectangle and dense mask interframe update paths.

16-frame long-context stress cases are included for CUDA and can be enabled for
WebGPU with `AUTOGAZE_BENCH_LONG_CONTEXT=1`.

## useful filters

```sh
cargo bench --bench backend_pipeline -- autogaze_trace_video/webgpu/multiscale-32-64-112-224/anyres-tile-224
cargo bench --bench backend_pipeline -- autogaze_tile_batch_size/webgpu
AUTOGAZE_HF_DIR=/path/to/AutoGaze cargo bench --bench backend_pipeline --features webgpu -- autogaze_real_task_loss/webgpu
cargo bench --bench backend_pipeline -- autogaze_rgba_e2e_video/webgpu/multiscale-32-64-112-224/resize-224/720p
cargo bench --bench backend_pipeline --features webgpu -- autogaze_tensor_visualization/webgpu/multiscale-32-64-112-224/interframe-delta/tiny-sparse
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline -- bevy_autogaze_viewer_pipeline/interframe-delta-panels/coarse-dense/1080p
```

Use the Bevy perf matrix for end-to-end viewer numbers on a real GPU host:

```sh
cargo run -p xtask -- bevy-perf-matrix --frames 120 --camera
```

The matrix runs `cargo run --release` by default and writes per-case JSON plus
an aggregate `summary.json` under `target/autogaze-bevy-perf/`.
