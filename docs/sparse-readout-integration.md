# sparse readout integration

`burn_autogaze` owns AutoGaze-specific geometry. Downstream sparse image/video
models should consume decoded rectangles, projected image tokens, or projected
video-token coordinates from this crate instead of reinterpreting generated
AutoGaze token ids.

## boundary

| crate | owns |
|---|---|
| `burn_autogaze` | AutoGaze token decoding, multi-scale rectangle geometry, image-grid projection, generic video-token projection, and Burn coordinate tensors. |
| `burn_jepa` | V-JEPA mask types, target/context policy, predictor-plan cache, dense keyframe prediction, and model-specific sparse-stream logic. |
| `burn_flex_gmm` | Sparse patchification and sparse convolution kernels. |

The interframe RGBA path is a rectangle compositor problem, not a
`burn_flex_gmm` patchify problem. If CPU sparse copies become the bottleneck,
the next primitive should be a renderer/GPU rectangle-copy compositor. The
`burn_flex_gmm` handoff from this crate is coordinate production for downstream
sparse patchification.

## core helpers

Use rectangle helpers when the consumer needs source-space geometry:

- `fixation_points_to_readout_rects`
- `trace_to_frame_readout_rects`
- `frame_readout_rects_to_video_tokens`
- `frame_readout_rects_to_video_coords`
- `frame_readout_rects_to_video_coord_tensor`

Use token helpers when the consumer needs image/video-grid ids:

- `fixation_points_to_readout_tokens`
- `trace_frame_readout_tokens`
- `generated_frame_readout_tokens`
- `trace_to_frame_readout_tokens`
- `generated_to_frame_readout_tokens`
- `frame_readout_tokens_to_video_tokens`
- `video_readout_tokens_to_coords`
- `batched_video_readout_tokens_to_coords`
- `video_readout_tokens_to_coord_tensor`

Use one-call adapters when the consumer already has a full trace or generated
output:

- `trace_to_video_readout_tokens`
- `trace_to_video_readout_coords`
- `trace_to_video_readout_coord_tensor`
- `generated_to_video_readout_tokens`
- `generated_to_video_readout_coords`
- `generated_to_video_readout_coord_tensor`

Use packet helpers in graph-style pipelines:

- `AutoGazePipelinePacket::frame_readout_tokens`
- `AutoGazePipelinePacket::frame_readout_rects`
- `AutoGazePipelinePacket::video_readout_tokens`
- `AutoGazePipelinePacket::video_readout_coords`
- `AutoGazePipelinePacket::video_readout_coord_tensor`

`SparseVideoPatchGeometry` derives the video-token grid from frame size,
tubelet size, and patch size. `SparseVideoReadoutProjection` groups the image
grid, video grid, and readout options for one-call tensor adapters.

## recommended pipeline

1. Run AutoGaze through `AutoGazePipeline` or `AutoGazeTensorPipeline`.
2. Prefer `emit_readout_points` for no-trace packet output when the consumer
   only needs sparse readout.
3. Project readout points or generated output through the helpers above.
4. Pass `[batch, temporal, row, col]` coordinate tensors to the downstream
   sparse patchifier.
5. Keep downstream target/context mask policy in the downstream model crate.

`SparseReadoutOptions::with_max_fixations_per_frame` caps decoded AutoGaze
fixations before image-grid projection. `with_max_tokens_per_frame` caps
projected downstream tokens after dilation and deduplication.

## burn_jepa migration

The current `../burn_jepa` sparse E2E bench has local generated-token decoding
and image-to-video projection. That should be replaced with calls into this
crate:

- `generated_to_frame_readout_tokens`
- `generated_to_video_readout_tokens`
- `generated_to_video_readout_coords`
- `SparseVideoPatchGeometry`
- packet-level `video_readout_*_with_projection` helpers

A concrete downstream patch is checked in at
`docs/burn-jepa-sparse-readout-migration.patch`.

Audit an external checkout with:

```sh
cargo run -p xtask -- check-burn-jepa-sparse-readout-integration ../burn_jepa
```

The audit fails while downstream code still contains local
`generated_frame_tokens`, `context_mask_from_autogaze_generated`, or manual
generated-token frame-offset math.

## validation

Run the in-repo adapter coverage:

```sh
cargo run --example sparse_video_readout_adapter --features ndarray
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo test -p burn_autogaze --features ndarray visualization -- --nocapture
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture
```

Check the downstream patch without modifying the sibling checkout:

```sh
git -C ../burn_jepa apply --check ../burn_autogaze/docs/burn-jepa-sparse-readout-migration.patch
```
