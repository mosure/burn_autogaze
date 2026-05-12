# api notes

This crate keeps model-specific AutoGaze decoding in `burn_autogaze` and leaves
frontends or downstream codecs to choose how to render, store, or consume the
decoded geometry.

## inference modes

| mode | use case | behavior |
|---|---|---|
| `ResizeToModelInput` | realtime streams | Resize each frame to the model input size before generation. |
| `TiledResizeToGrid` | high-resolution inspection | Resize the source into a complete 224px tile canvas, run bounded tile batches, and stitch tile-local scale grids back into source space. |
| `TiledFullResolution` | exact source-space padding | Tile the original source dimensions directly and pad edge tiles. |

The NVIDIA config has a fixed multi-scale vocabulary of 265 tokens per frame.
The default realtime generation budget delegates to the model config, which is
198 tokens for the current NVIDIA weights. In tiled mode the budget applies per
tile before task-loss stopping and confidence filtering.

## tensor pipeline

The composable pipeline API is centered on `AutoGazeTensorPipeline`.

- `TensorClipInput` accepts normalized Burn tensors in
  `[batch, time, channel, height, width]` layout.
- `RgbaClipInput` accepts packed RGBA clips and applies the same mode-aware
  preparation as `trace_rgba_clip_with_mode`.
- `AutoGazePipelinePacket` carries the source tensor clip, inference mode,
  trace settings, optional decoded traces, and optional readout points.
- `VecOutputNode`, `FnOutputNode`, and custom `AutoGazeOutputNode`
  implementations can write to memory, renderers, disk, transport layers, or
  downstream codecs.
- `AutoGazeRgbaFrameQueue` owns the rolling RGBA frame window and recycles clip
  buffers so camera/file frontends do not duplicate packing logic.

```rust,no_run
use burn::backend::NdArray;
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazeRgbaClip, AutoGazeRgbaClipShape,
    AutoGazeTensorPipeline, AutoGazeTensorPipelineConfig, RgbaClipInput, VecOutputNode,
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

Set `emit_traces` only when the consumer needs decoded fixation points or mask
geometry; generation requires model work and backend-to-host readback. For
lower-overhead sparse consumers, prefer `emit_readout_points` or
`AutoGazePipeline::readout_points_with_mode_async`.

## visualization

AutoGaze emits multi-scale token positions. The Rust decoder maps NVIDIA tokens
back to `2x2`, `4x4`, `7x7`, and `14x14` cells and preserves those cells with
nearest sampling. `scale-rows` mirrors the upstream NVIDIA diagnostic view by
drawing one stable row per scale. `overlay` draws the unioned sparse-update
footprint in source space.

| mode | output behavior | update ratio |
|---|---|---|
| `full-blend` | current frame plus alpha-blended mask | selected effective mask pixels / full frame |
| `interframe` | previous output outside the mask, current input inside the mask | masked-cell pixels / full frame |

Use `AutoGazeVisualizationState` for CPU RGBA visualization and
`AutoGazeTensorVisualizationState` for tensor-side composition. Tensor
interframe composition reports whether the latest frame used a keyframe, sparse
rectangle update, or dense mask update through
`last_interframe_path`.

`task_loss_requirement_db` is a convenience interface for the upstream L1
reconstruction-loss head: `threshold = 10^(-dB / 20)`. It is not the same value
as output-column PSNR.

## sparse readout

Downstream image/video models should not reinterpret AutoGaze generated token
ids directly. Use the readout helpers to preserve the decoded scale layout and
then project into the downstream grid.

- `fixation_points_to_readout_rects`
- `trace_to_frame_readout_rects`
- `fixation_points_to_readout_tokens`
- `trace_frame_readout_tokens`
- `generated_to_frame_readout_tokens`
- `frame_readout_tokens_to_video_tokens`
- `trace_to_video_readout_tokens`
- `generated_to_video_readout_tokens`
- `video_readout_tokens_to_coords`
- `batched_video_readout_tokens_to_coords`

`SparseVideoPatchGeometry` derives video-grid dimensions from frame, tubelet,
and patch sizes. This keeps AutoGaze model geometry in this crate while letting
downstream crates such as `burn_jepa` wrap the resulting indices in their own
mask/readout types.

For ViT-like image pyramids, `fixation_image_mask_tensor` builds a Burn mask,
`apply_image_mask` keeps or fills selected image regions, and
`tokenize_masked_image_pyramid` emits weighted tokens for requested
`ImagePyramidLevel`s. `sparsify_image_pyramid_tokens` selects high-density
tokens with backend `topk`, avoiding synchronous host readback on WebGPU/wasm.
