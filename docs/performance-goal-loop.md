# performance goal loop

The performance loop for this repo is iterative: measure the real hot path,
inspect correctness and edge conditions, form one falsifiable hypothesis, make a
small change, validate, record the result, and repeat. The goal is stable,
high-quality sparse video output where the model and GPU kernels are the
bottleneck, not duplicated plumbing, host transfers, allocation churn, or display
composition.

This file is not a completion artifact. It is the active operating loop for
performance work. A docs-only update, a passing unit-test subset, or one
improved benchmark case is not enough to close the mission while high-motion
camera moves or full-frame motion still cause disproportionate Bevy FPS drops.

## goals

- Keep realtime defaults numerically sane for camera streams: low unnecessary
  gaze coverage, stable multi-scale masks, and useful PSNR without hiding
  composition bugs.
- Prevent large gaze coverage or many active mask cells from causing
  disproportionate FPS drops. Sparse update paths should switch to dense/full
  frame work when sparse work is no longer cheaper.
- Treat full-frame motion, camera pans, and high-overlap multi-scale masks as
  first-class acceptance cases, not optional stress tests.
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

Use this record for each loop iteration:

```text
date / revision:
target path:
configuration:
baseline:
hypothesis:
change:
correctness gate:
performance gate:
result:
follow-up if failed:
next bottleneck:
```

Keep the record small and evidence-based. The next iteration should be obvious
from the `next bottleneck` line without re-reading the whole investigation.
If the result still shows pathological FPS drops, metric flicker, incorrect
mask geometry, or unexplained quality loss, immediately start the next
iteration instead of treating the loop as done.

## current open bottleneck

2026-05-12 local iteration: the dense/full-frame CPU mask-panel compositor was
improved by using exact alpha-blend lookup tables and by taking a full-frame
interframe fallback before sparse rect dispatch. The focused redundant
multi-scale full-frame bench improved, but it is not completion evidence by
itself:

| case | current local mean |
|---|---:|
| 720p interframe panels, redundant full-frame | `5.23 ms` |
| 1080p interframe panels, redundant full-frame | `11.60 ms` |
| 1080p interframe side-by-side, redundant full-frame | `17.52 ms` |
| 1080p redundant multi-scale geometry, native | `7.13 ms` |
| 1080p redundant multi-scale geometry, deduplicated | `5.51 ms` |
| 1080p redundant multi-scale geometry, effective | `5.57 ms` |

Next iteration: capture a deterministic Bevy e2e high-motion source and split
the remaining time between source/preprocess, model, mask-panel generation,
display transfer, texture upload, and UI metrics. The mission remains open
until high-motion/full-frame Bevy runs have stable p50/p95 frame times and a
measured dominant bottleneck.

The Bevy app and xtask matrix now expose `--source synthetic-pan` for this next
iteration. It generates deterministic full-frame motion without a webcam while
still using the normal frame queue, preprocessing, model, visualization, display
transfer, texture upload, and metrics path.

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

## high-coverage mask stress loop

High-motion streams can activate many cells across multiple scales. This path
must be measured separately because duplicated multi-scale cells can make sparse
work more expensive than dense work even when the visual output is unchanged.

For this loop:

1. Compare `--mask-geometry native`, `deduplicated`, and `effective`.
2. Measure tiny sparse, overlapping multi-scale, dense-grid-64, and full-frame
   coverage at 720p and 1080p.
3. Include a camera-pan or synthetic full-frame-motion source, not only static
   fixture frames.
4. Record unique rects, row spans, updated pixels, copied bytes, CPU/GPU
   residency, texture uploads, and whether the path chose sparse or
   dense/full-frame dispatch.
5. Record p50/p95 frame time and stage timing for model, mask planning,
   interframe composition, display transfer, and texture update.
6. Verify gaze ratio is based on the union of updated pixels, not duplicated
   overlapping cells.
7. Check that mask visualization and interframe accumulation use the same
   geometry policy unless intentionally testing an exact native diagnostic.
8. Move the sparse/dense threshold only after the threshold benchmark shows the
   crossover point for the active backend and resolution.
9. Repeat after every threshold or geometry change until the high-motion case no
   longer tanks relative to the measured dense/full-frame fallback.

The default Bevy viewer path should prefer deduplicated geometry for stable
interactive performance. Native geometry is still useful for debugging decoded
model cells and docs traces, but it should not be the default high-motion
interactive mode if it repeats equivalent work.

## standard measurement lanes

Use both synthetic stress cases and real-model/video cases. Minimum useful set:

- `resize-224`, 16-frame realtime stream, default task-loss settings.
- AnyRes tiled 720p and 1080p.
- Tiny sparse, multi-scale, coarse dense, and dense-grid-64 masks.
- `native`, `deduplicated`, and `effective` mask geometry modes.
- CPU RGBA panel transfer and GPU tensor panel transfer.
- Native WebGPU/Vulkan or CUDA where available.
- Wasm static-source smoke with no webcam requirement.
- Real video input, preferably the birds fixture and a high-motion camera-like
  stream.
- At least one high-motion/full-frame source where most cells are active across
  scales.

Useful commands:

```sh
cargo test -p burn_autogaze --test source_hygiene
cargo test -p burn_autogaze --lib
cargo test -p bevy_burn_autogaze --lib
cargo clippy -p burn_autogaze -p bevy_burn_autogaze --all-targets -- -D warnings
cargo bench --bench backend_pipeline --features webgpu
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline
cargo run -p xtask -- bevy-perf-matrix --frames 120
cargo run -p xtask -- bevy-perf-matrix --dry-run --frames 2
```

For a focused high-coverage display check:

```sh
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline -- \
  bevy_autogaze_viewer_pipeline/interframe-delta-panels/dense-grid-64/1080p
```

## success criteria

- Large active-mask coverage does not tank FPS because the pipeline switches to
  the cheaper dense/full-frame path when appropriate.
- High-motion camera moves and full-frame motion have Bevy e2e perf summaries
  showing stable p50/p95 frame times, with the dominant cost identified.
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

## non-completion conditions

Do not call the mission complete if any of these are true:

- High-motion/full-frame Bevy runs still show disproportionate FPS drops.
- The current bottleneck is only guessed rather than isolated by per-stage
  timing or benchmark evidence.
- The only evidence is a core library benchmark that bypasses camera/source,
  interframe composition, display transfer, or texture upload.
- Sparse/dense fallback thresholds have not been tested near their crossover
  points at 720p and 1080p.
- Mask visualization, gaze ratio, or interframe output differ semantically
  between the tested path and the Bevy default path.
- Numerical quality regresses, PSNR/gaze-ratio behavior is unexplained, or
  high-overlap multi-scale masks are not covered by tests.
- Hardware/browser throughput was unavailable and no equivalent deterministic
  app-level perf artifact was captured.

## iteration stop conditions

Stop an iteration when the current bottleneck is explained, the hypothesis is
validated or rejected, and the repo has passing focused gates. Continue with a
new iteration only after naming the next highest-impact bottleneck and the metric
that should move.

Stopping an iteration is not the same as completing the mission. The next agent
should continue from the named bottleneck until the non-completion conditions no
longer apply.
