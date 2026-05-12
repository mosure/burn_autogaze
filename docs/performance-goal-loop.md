# performance goal loop

The performance loop for this repo is iterative: measure the real hot path,
inspect correctness and edge conditions, form one falsifiable hypothesis, make a
small change, validate, record the result, and repeat. The goal is stable,
high-quality sparse video output where the model and GPU kernels are the
bottleneck, not duplicated plumbing, host transfers, allocation churn, or display
composition.

## goals

- Keep realtime defaults numerically sane for camera streams: low unnecessary
  gaze coverage, stable multi-scale masks, and useful PSNR without hiding
  composition bugs.
- Prevent large gaze coverage or many active mask cells from causing
  disproportionate FPS drops. Sparse update paths should switch to dense/full
  frame work when sparse work is no longer cheaper.
- Keep native, wasm, core library, and Bevy viewer paths aligned around the same
  core pipeline APIs.
- Make every hot-path segment measurable enough that bottlenecks can be found
  quickly.
- Keep the code organized around swappable stages with explicit data residency
  and minimal duplicate logic.

## loop

1. **Baseline**
   - Start from a clean tree and record the current git revision.
   - Run focused correctness tests before performance work so failures are not
     misread as speed regressions.
   - Capture at least one core benchmark lane and one Bevy e2e lane for the
     target configuration.

2. **Segment**
   - Break the active path into source, buffering, preprocessing, model/readout,
     mask planning, visualization/interframe, display transfer, texture upload,
     and metrics/UI.
   - Confirm the data residency for each handoff: CPU RGBA, Burn tensor,
     Bevy texture, or mixed fallback.
   - Record bytes moved and whether any stage synchronizes or reads tensor data
     back to host.

3. **Review correctness and edges**
   - Check channel order, alpha handling, resizing mode, aspect preservation,
     frame ordering, stale-frame rejection, and ring-buffer/cache behavior.
   - Exercise empty mask, tiny sparse mask, overlapping multi-scale cells,
     dense-grid mask, full-frame coverage, non-16:9 frames, 720p, and 1080p.
   - Compare mask and interframe output semantics against upstream/Python
     fixtures whenever model interpretation changes.

4. **Review metrics**
   - Inspect FPS, p50/p95 frame time, gaze ratio current/EMA, PSNR current/EMA,
     mask rects, row spans, updated pixels, output bytes, tensor path, display
     residency, input residency, dropped/stale frames, and in-flight depth.
   - Treat `inf` PSNR, `100%` gaze, or periodic metric pulses as state-machine
     symptoms until explained by a test.

5. **Hypothesize one change**
   - State the suspected bottleneck, expected direction, measurement to improve,
     and correctness risk before editing.
   - Prefer changes that remove duplicate work, reuse buffers, keep tensors
     device-resident, batch tiles/cells, or make sparse/dense dispatch adaptive.
   - Avoid model-quality tradeoffs unless the quality metric and expected
     gaze-ratio/PSNR movement are explicit.

6. **Implement narrowly**
   - Change one subsystem boundary at a time.
   - Add or update the module-level test that would have caught the bug or
     regression.
   - Keep Bevy as a wrapper around core inference, readout, visualization, and
     metrics APIs.

7. **Validate and record**
   - Re-run the smallest relevant test/bench first, then the broader gates.
   - Record before/after numbers and note whether the hypothesis held.
   - If the metric moved for the wrong reason, revert or revise before stacking
     more changes.

## hot-path coverage

Every performance pass should be able to isolate these stages:

| stage | required checks |
|---|---|
| source/camera/static input | frame order, format, channel order, aspect, stale-frame behavior |
| frame window/ring buffer | context length, cache continuity, drop policy, allocation reuse |
| preprocessing | resize mode, tile canvas, batch shape, CPU/GPU residence |
| model/readout | resize vs tiled mode, KV/cache path, task-loss stopping, tile batch size |
| mask planning | multi-scale cell geometry, overlap deduplication, sparse vs dense threshold |
| interframe output | update semantics, keyframe policy, PSNR/gaze-ratio accounting |
| display transfer | CPU RGBA vs tensor path, bevy_burn bridge use, uploaded bytes |
| UI/metrics | stable text, no image overlap, no hidden sync/readback |

## standard measurement lanes

Use both synthetic stress cases and real-model/video cases. Minimum useful set:

- `resize-224`, 16-frame realtime stream, default task-loss settings.
- AnyRes tiled 720p and 1080p.
- Tiny sparse, multi-scale, coarse dense, and dense-grid-64 masks.
- CPU RGBA panel transfer and GPU tensor panel transfer.
- Native WebGPU/Vulkan or CUDA where available.
- Wasm static-source smoke with no webcam requirement.
- Real video input, preferably the birds fixture and a high-motion camera-like
  stream.

Useful commands:

```sh
cargo test -p burn_autogaze --test source_hygiene
cargo test -p burn_autogaze --lib
cargo test -p bevy_burn_autogaze --lib
cargo clippy -p burn_autogaze -p bevy_burn_autogaze --all-targets -- -D warnings
cargo bench --bench backend_pipeline --features webgpu
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline
cargo run -p xtask -- bevy-perf-matrix --frames 120
```

For a focused high-coverage display check:

```sh
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline -- \
  bevy_autogaze_viewer_pipeline/interframe-delta-panels/dense-grid-64/1080p
```

## success criteria

- Large active-mask coverage does not tank FPS because the pipeline switches to
  the cheaper dense/full-frame path when appropriate.
- Interframe output copies source pixels only in selected regions and preserves
  unmasked history.
- Gaze ratio reflects updated pixels, not visible overlay color or duplicated
  overlapping cells.
- PSNR is enabled by default in the viewer path but computed only when requested
  by the UI/config and not by hidden sync-heavy readbacks.
- Bevy e2e metrics can explain the dominant cost without adding temporary
  instrumentation.
- New code paths have module-level correctness tests and benchmark coverage for
  the active hot-path segment.

## stop conditions

Stop an iteration when the current bottleneck is explained, the hypothesis is
validated or rejected, and the repo has passing focused gates. Continue with a
new iteration only after naming the next highest-impact bottleneck and the metric
that should move.
