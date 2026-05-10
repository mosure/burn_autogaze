# sparse readout integration

This crate owns AutoGaze-specific geometry. Downstream sparse image or video
models should not reinterpret AutoGaze generated token ids directly; they should
consume decoded rectangles or projected image-token ids from `burn_autogaze`.

## burn_flex_gmm fit

`burn_flex_gmm` provides sparse submanifold convolution kernels plus CPU and
WGPU sparse 3D patchification. The sparse patchify path accepts selected
`[batch, tubelet, row, col]` patch coordinates, gathers only those video
tubelets, and applies the patch projection on device. That is a good fit for
V-JEPA-style selected-token readout where masked-out patches should not be
patchified before the encoder.

It is not the right first primitive for AutoGaze interframe RGBA updates. The
interframe output path copies source pixels into a persistent output image over
decoded multi-scale rectangles. That is a rectangle copy or compositor problem,
not a patchify or submanifold convolution problem. A renderer-specific GPU
compositor or a small custom rectangle-copy kernel is the right target if the
CPU sparse copy becomes the bottleneck. The useful `burn_flex_gmm` handoff from
this crate is therefore coordinate production, not introducing a
`burn_flex_gmm` dependency into `burn_autogaze`.

## burn_jepa boundary

`burn_jepa` uses `burn_flex_gmm` behind its `sparse-patchify-wgpu` feature.
Its sparse stream projects per-frame image-token selections into V-JEPA
tubelet-token masks, caches sparse patchify and predictor plans, and can accept
precomputed context/target masks for stable-mask hot paths.

The shared adapter logic now belongs here:

- `fixation_points_to_readout_rects` preserves decoded AutoGaze multi-scale
  cell geometry.
- `trace_to_frame_readout_rects` returns per-frame rectangles for a complete
  decoded trace.
- `fixation_points_to_readout_tokens` projects decoded rectangles onto any
  image-token grid.
- `SparseReadoutGrid::square_from_token_count` builds the common square
  AutoGaze connector grid without downstream `sqrt` assertions.
- `trace_frame_readout_tokens` and `trace_to_frame_readout_tokens` provide
  frame-level and clip-level token projection with optional dilation and token
  caps.
- `generated_frame_readout_tokens` and `generated_to_frame_readout_tokens`
  provide the same projection directly from `AutoGazeGenerateOutput`, sharing
  the core multi-scale decoder without allocating full fixation traces.
- `frame_readout_tokens_to_video_tokens` and
  `frame_readout_rects_to_video_tokens` map decoded per-frame image readout onto
  generic `[temporal, row, col]` video-token grids with tubelet grouping,
  dilation, and exact/min/max token budgets. This mirrors the AutoGaze-to-V-JEPA
  adapter shape without depending on `burn_jepa`'s concrete `SparseTokenMask`
  type.
- `SparseVideoPatchGeometry` derives that video-token grid from
  `[frames, height, width, tubelet, patch_h, patch_w]`, matching the shape
  validation used by sparse 3D patchifiers such as `burn_flex_gmm`.
- `frame_readout_tokens_to_video_coords` and
  `frame_readout_rects_to_video_coords` return the corresponding
  `[batch, temporal, row, col]` sparse patchification coordinates directly.
- `frame_readout_tokens_to_video_coord_tensor` and
  `frame_readout_rects_to_video_coord_tensor` return those coordinates as Burn
  int tensors with shape `[rows, 4]`.
- `generated_to_video_readout_tokens` and `trace_to_video_readout_tokens`
  compose the image-grid and video-grid projection steps for consumers that
  want one adapter call from AutoGaze output to downstream sparse video-token
  indices.
- `generated_to_video_readout_coords` and `trace_to_video_readout_coords`
  compose the same path directly to sparse patchification coordinate rows.
- `generated_to_video_readout_coord_tensor` and
  `trace_to_video_readout_coord_tensor` compose the same path directly to Burn
  int coordinate tensors.
- `video_readout_tokens_to_coords` and
  `batched_video_readout_tokens_to_coords` turn those sparse video-token indices
  into `[batch, temporal, row, col]` coordinate rows, matching the coordinate
  shape used by `burn_flex_gmm` sparse 3D patchify plans without depending on
  `burn_flex_gmm`.
- `video_readout_coords_to_tensor`,
  `video_readout_tokens_to_coord_tensor`, and
  `batched_video_readout_tokens_to_coord_tensor` upload those rows as a Burn
  int tensor with shape `[rows, 4]`, which is the tensor shape accepted by the
  `burn_flex_gmm` sparse 3D patchify WGPU path.
- `SparseVideoReadoutProjection` groups the image grid, video grid, AutoGaze
  readout options, and sparse-video options for one-call tensor adapters.
- `SparseReadoutOptions::with_max_fixations_per_frame` caps decoded AutoGaze
  gaze points before image-grid projection, while
  `with_max_tokens_per_frame` caps the projected downstream tokens after
  dilation and deduplication. Downstream V-JEPA code can use the first cap to
  mirror an AutoGaze `top_k` budget and the second cap to enforce a patch-token
  density budget.
- `AutoGazePipelinePacket::frame_readout_tokens` and
  `AutoGazePipelinePacket::frame_readout_rects` use emitted traces when
  available, emitted readout points, or emitted generated output in realtime
  resize mode. `AutoGazePipelinePacket::video_readout_tokens` applies the
  generic sparse-video projection at packet level, and
  `AutoGazePipelinePacket::video_readout_coords` returns the flattened sparse
  patchification coordinate rows.
  `AutoGazePipelinePacket::video_readout_coord_tensor` returns those rows as a
  Burn int tensor. The corresponding `*_with_projection` helpers accept a
  `SparseVideoReadoutProjection` directly. Output nodes do not need to branch on
  whether a packet carries traces, readout points, or generated output.
  `AutoGazeTensorPipelineConfig::emit_readout_points` is the preferred no-trace
  path because tiled modes are remapped into source-frame fixation points before
  output adapters project them onto downstream token grids.
- `AutoGazePipeline::readout_points_with_mode` and
  `AutoGazePipeline::readout_points_with_mode_async` expose the same remapped
  no-trace readout path directly for native and wasm/WebGPU callers.

`burn_jepa` should keep V-JEPA-specific details: converting video-token indices
into its `SparseTokenMask`, choosing target masks, sparse patchify plan caching,
predictor-plan reuse, and `burn_flex_gmm` dispatch.
This prevents duplicate AutoGaze scale math from drifting across repos while
keeping the WGPU sparse patchify dependency out of `burn_autogaze`'s default
surface.

The current `burn_jepa` sparse E2E bench has local
`generated_frame_tokens` / `context_mask_from_autogaze_generated` glue. That
code should become a thin call into `generated_to_frame_readout_tokens`,
`generated_to_video_readout_tokens`, `generated_to_video_readout_coords`,
`SparseVideoPatchGeometry`, or the packet-level
`video_readout_*_with_projection` helpers now that `burn_autogaze` exposes the
adapter surface. `burn_jepa` can then keep only the final conversion from
generic video-token indices into its `SparseTokenMask` and plan cache.
`examples/sparse_video_readout_adapter.rs` is the compiled migration sketch for
that replacement: it builds synthetic multi-scale AutoGaze output, projects it
onto a downstream 3D patch grid, emits `[batch, temporal, row, col]` sparse
patchify coordinates, and wraps the returned video-token indices in a minimal
downstream mask type.
`docs/burn-jepa-sparse-readout-migration.patch` is a concrete patch for the
current `../burn_jepa/benches/autogaze_sparse_jepa_pipeline.rs` benchmark. It
was checked in a temporary copy of `../burn_jepa` with this crate patched in as
the local `burn_autogaze` dependency.
`tools/check_burn_jepa_sparse_readout_integration.sh PATH_TO_BURN_JEPA` is the
external audit for that migration. It fails while the downstream benchmark still
contains local `generated_frame_tokens` /
`context_mask_from_autogaze_generated` logic, and passes only after the bench
imports and calls the shared `burn_autogaze` readout helpers.

## validation commands

Run the core readout and visualization coverage from this repo:

```sh
cargo run --example sparse_video_readout_adapter --features ndarray
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo test -p burn_autogaze --features ndarray visualization -- --nocapture
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture
```

Run the downstream sparse patchify checks from `../burn_jepa`:

```sh
cargo test --test sparse_patchify_wgpu --no-default-features --features sparse-patchify-wgpu -- --nocapture
cargo test --test numerical_parity sparse_forward_hot_path_has_no_backend_readbacks --no-default-features --features ndarray -- --nocapture
git -C ../burn_jepa apply --check ../burn_autogaze/docs/burn-jepa-sparse-readout-migration.patch
tmp=$(mktemp -d /tmp/burn_jepa-readout.XXXXXX)
tar --exclude='./target' --exclude='./.git' -C ../burn_jepa -cf - . | tar -x -C "$tmp"
git -C "$tmp" apply "$PWD/docs/burn-jepa-sparse-readout-migration.patch"
perl -0pi -e 's#burn_autogaze = \{ version = "0\.21\.2", default-features = false, features = \["ndarray"\] \}#burn_autogaze = { version = "0.21.2", path = "'$PWD'", default-features = false, features = ["ndarray"] }#' "$tmp/Cargo.toml"
tools/check_burn_jepa_sparse_readout_integration.sh "$tmp"
cargo check --manifest-path "$tmp/Cargo.toml" --bench autogaze_sparse_jepa_pipeline --no-default-features --features ndarray,sparse-patchify-wgpu
find "$tmp" -depth -mindepth 1 -delete && rmdir "$tmp"
```
