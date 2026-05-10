# completion audit

This file records the current implementation coverage and the remaining
environment-bound checks for `burn_autogaze`.

## prompt-to-artifact checklist

| requirement | concrete artifact or evidence | current status |
|---|---|---|
| Efficient core pipeline across realtime and tiled modes | `AutoGazePipelineOptions`, batched tile embedding/generation, benchmark build checks, Bevy perf summaries, `tools/run_bevy_perf_matrix.sh`, `--perf-summary-path`, and `--require-hardware-adapter=true` guard | Partially covered; hardware GPU FPS remains unverified on this host |
| Numerical parity with upstream NVIDIA/Python AutoGaze | embedding/generation parity tests, birds fixture, fixture-only multi-scale mask decode checks, fixture-only seeded 224px and 448px upstream mask decode checks, fixture-integrity checks for metadata/mask counts/token ranges, fixture-only birds visualization/interframe checks, multi-scale scale-token and AnyRes stitching unit tests | Covered for checked-in fixtures; broader upstream corpus coverage remains future work |
| Robust outputs across inputs/configs | RGBA processor tests, static-source browser tests, tiled/realtime Bevy config tests, Bevy config sanitization tests, 1080p AnyRes mask tests | Covered for fixture/static inputs; live camera hardware path remains environment-bound |
| Well-formed FPS, gaze-ratio, and PSNR metrics | `src/metrics.rs`, visualization PSNR/gaze-ratio tests, Bevy perf JSON tests, wasm Playwright assertions, perf-summary FPS/gaze/PSNR/config/admission-policy fields, and `tools/validate_bevy_perf_summary.py` per-case plus aggregate schema/range validation used by the native perf matrix | Covered by unit/browser/tool tests |
| Bevy app is a thin wrapper around core pipeline | Bevy tests assert core RGBA prep, core `AutoGazeRgbaFrameClip` buffer reuse, core runtime defaults, stale-result gate, metric delegation, and shared visualization output; source hygiene scans Bevy `lib.rs`, `main.rs`, and `platform.rs` for local generated-output decoding, and now also guards Bevy production visualization/metrics from reimplementing core mask, interframe, PSNR, gaze-ratio, or EMA logic | Covered at logic level |
| Native and wasm support with minimal platform split | wasm check/clippy, Playwright static/optional model tests, native Bevy tests, platform split limited to camera/model bytes, tensor readout sync/async, and no Bevy `native`/`web` feature toggles | Covered at compile/browser smoke level |
| Avoid duplicate AutoGaze scale/readout logic in downstream integrations | `src/readout.rs` exports trace and direct `AutoGazeGenerateOutput` readout helpers plus generic image/video-token projection helpers, `SparseVideoPatchGeometry`, one-call trace/generated-to-video adapters, sparse patchification coordinate adapters, and Burn coordinate tensor builders; `AutoGazePipeline::readout_points_with_mode` and `readout_points_with_mode_async` expose no-trace readout for native and wasm callers; `AutoGazePipelinePacket` can emit no-trace readout points for realtime/tiled modes and exposes packet-level sparse-video token, coordinate, and coordinate-tensor projection; `AutoGazeTensorPacketPlan` centralizes tensor-packet validation, generation budget, and packet assembly for sync and async runners; `docs/sparse-readout-integration.md` documents the `burn_jepa` boundary; `docs/burn-jepa-sparse-readout-migration.patch` is a concrete downstream benchmark migration patch | Covered in this repo; `../burn_jepa` still has benchmark-local generated-token decoding until that repo is updated |
| Async camera/inference pipeline drops stale frames instead of rendering out-of-order results | `AutoGazeInferenceSequencer`, `AutoGazeRealtimePolicy`, configurable Bevy `max_in_flight`, Bevy stale-result tests, latest-frame camera channel test | Covered by logic tests; live camera frame pacing remains hardware/browser dependent |
| KV cache and sane realtime defaults | `should_use_streaming_cache`, model streaming-cache parity tests, `AutoGazeTensorPipelineConfig::default`, Bevy default tests, and `realtime_policy_from_config` capping streaming-cache realtime to one in-flight task | Covered for resize realtime path |
| Publish readiness | clippy, wasm compile/clippy, package verification, package-checkout fixture-only upstream parity, diff whitespace | Covered locally with noted read-only cargo cache warning |

## concrete deliverables

- Core AutoGaze model/pipeline APIs load configs and weights, preprocess RGBA
  inputs through one shared path, run resize and tiled/AnyRes inference, decode
  multi-scale generated tokens, and expose both trace and no-trace readout.
- Numerical tests compare Burn outputs with checked-in NVIDIA/Python embedding,
  generation, birds, mask, visualization, and interframe fixtures.
- Metrics are centralized in core helpers for FPS, gaze ratio, and PSNR, and
  Bevy reports them through the same types rather than local duplicate math.
- Bevy native/wasm is a wrapper over the core pipeline: camera/source handling,
  frame buffering, model task scheduling, stale-result rejection, display
  transfer, and text overlays live in Bevy, while preprocessing, inference,
  masks, visualization, metrics, readout, and realtime policy live in
  `burn_autogaze`.
- Browser and wasm code use async tensor readout paths and avoid
  wasm-unsupported sync setup, sync tensor reads, and `std::time::Instant`.
- Sparse readout helpers in this repo own AutoGaze-specific scale geometry and
  produce generic image/video token ids and `[batch, temporal, row, col]`
  coordinate tensors; downstream crates such as `burn_jepa` own their concrete
  mask types, target-mask selection, plan caching, and `burn_flex_gmm` dispatch.
- Release/readiness scripts run the non-hardware gate, generated-package checks,
  package-checkout fixture-only upstream parity, and browser-demo build/test
  commands. They fail early with a clear diagnostic when local Snap Node is
  active, accept `--node-bin-dir`, and support `--no-browser-deps` for
  environments where Playwright browser installation cannot use `sudo`.

## remaining work and improvement plan

- Update `../burn_jepa` to call `burn_autogaze::generated_to_frame_readout_tokens`
  or `generated_to_frame_readout_rects`, then
  `frame_readout_tokens_to_video_tokens` or
  `frame_readout_rects_to_video_tokens`, instead of its benchmark-local
  generated-token decoder and image-to-video projection. That checkout is
  outside the current writable root, so the change is provided as
  `docs/burn-jepa-sparse-readout-migration.patch` and documented in
  `sparse-readout-integration.md` but not applied there. Run
  `tools/check_burn_jepa_sparse_readout_integration.sh ../burn_jepa` after the
  downstream patch; it currently fails against the unpatched checkout by
  detecting `generated_frame_tokens`, `context_mask_from_autogaze_generated`,
  and manual generated-token frame-offset math.
- Run the native Bevy app on a host with a real GPU render adapter and working
  camera device. This host selects llvmpipe CPU Vulkan, and the
  `--require-hardware-adapter=true` guard correctly exits before recording CPU
  adapter FPS.
- Expand upstream numerical parity beyond the checked-in official, birds,
  seeded 224px, and 448px upstream AnyRes-style fixtures when additional
  NVIDIA/Python fixture outputs are available.
- If downstream tensor-pipeline outputs need raw tiled generated tokens, add a
  representation that carries tile-local metadata. Sparse readout itself is now
  covered by `AutoGazeTensorPipelineConfig::emit_readout_points`, which stores
  remapped source-frame fixation points without a full trace.
- If Bevy GPU display transfer is still slower on a hardware adapter, profile
  the tensor-side visualization/compositor stage separately from Bevy texture
  handoff and replace the current generic tensor composition with a
  renderer-specific rectangle compositor only if that stage is the measured
  bottleneck.
- If browser frame pacing still stutters after real Playwright/GPU runs, collect
  `window.__autogazePerfSummary` plus browser trace timing and compare source
  frame sequence numbers against `AutoGazeInferenceSequencer` outputs before
  changing the scheduling policy.
- If experimenting with `--max-in-flight >1`, keep `--streaming-cache=false` or
  use tiled mode. Realtime streaming-cache mode intentionally keeps one
  in-flight task so decoder KV state advances in order instead of being cloned
  by concurrent tasks.

## coverage

| area | evidence | status |
|---|---|---|
| Upstream AutoGaze parity | `cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture` plus embedding parity fixtures; `upstream_generated_masks_decode_without_model_snapshot` decodes the checked-in official, birds, seeded 224px, and 448px upstream AnyRes-style `gazing_pos` fixtures into per-scale masks without requiring local model weights and rejects malformed fixture metadata, mask shapes/sums, non-padded token coverage, or out-of-frame token ids | Covered for checked-in fixtures |
| RGBA preprocessing and channel order | Core RGBA clip tests and birds Python fixture tests exercise RGBA packing, processor resize, and tensor layout | Covered |
| Multi-scale mask decoding | Unit coverage for per-scale token grids, 1080p AnyRes stitching, native mask drawing, effective update footprint, and high-res birds fixture visualization/interframe output against upstream scale masks | Covered |
| Interframe output | `AutoGazeVisualizationState` tests cover sparse accumulation, keyframe refresh, PSNR, and update ratios | Covered |
| Bevy wrapper thinness | Bevy tests cover delegation to core runtime defaults, core RGBA prep, stale-result rejection, perf JSON samples, and source hygiene against local generated-output decoding in Bevy `lib.rs`, `main.rs`, and `platform.rs`; source hygiene also asserts Bevy production visualization calls `AutoGazeVisualizationState` / `AutoGazeTensorVisualizationState` helpers and Bevy metrics wrap `AutoGazeGazeRatioStats` / `AutoGazePsnrStats` instead of carrying local mask, interframe, PSNR, gaze-ratio, or EMA math | Covered at logic level |
| Sparse readout adapters | `src/readout.rs`, `src/pipeline.rs`, and `src/nodes.rs` tests cover decoded rectangle projection, token-grid projection, same-grid boundary stability, image-to-video token projection, sparse patchifier grid derivation from frame/tubelet/patch geometry, one-call trace/generated-to-video projection, `burn_jepa` benchmark adapter replacement behavior, packet-level sparse-video readout for trace and generated packets, tubelet grouping, exact/min/max video-token budgets, dilation, fixation caps before projection, frame token caps after projection, full trace helpers, direct `AutoGazeGenerateOutput` readout helpers, packet-level generated readout without traces, tiled packet readout points without traces, and sync/async compile coverage for the same pipeline helpers | Covered |
| `burn_jepa` integration boundary | `docs/sparse-readout-integration.md` explains why AutoGaze geometry helpers live here and `burn_flex_gmm` patchification remains downstream; `docs/burn-jepa-sparse-readout-migration.patch` gives the downstream benchmark patch; `tools/check_burn_jepa_sparse_readout_integration.sh` audits an external burn_jepa checkout for the expected migration | Documented, externally checkable, and temp-validated |
| Tensor pipeline sync/async duplication | `AutoGazeTensorPacketPlan` keeps validation, generation-budget selection, generated-packet gating, ready readout projection, and packet assembly shared between `run_next` and `run_next_async` | Covered by nodes test and clippy |
| WGPU/wasm compile regressions | `cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm`, `cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings`, `cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown`, and `cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings` | Covered by compile and no-warning checks |
| Browser panic regressions | `.github/workflows/test.yml` and `.github/workflows/deploy-pages.yml` build the Bevy wasm demo and run `npm run test:browser`; the Playwright spec checks static-source startup, async model-load failure handling, optional real-model wasm inference, adapter metadata, perf summary fields, and known wasm panic strings; `tools/check_bevy_wasm_demo.sh` preflights `node`/`npm`/`npx`, accepts `--node-bin-dir`/`AUTOGAZE_NODE_BIN_DIR`, and can use `--no-browser-deps` when `sudo` is unavailable | Covered by CI command surface, local static-source browser smoke, and local real-model wasm smoke with staged model assets |
| Native/wasm target selection | `crates/bevy_burn_autogaze/Cargo.toml` has no empty `native` or `web` features; platform behavior is selected by `cfg(target_arch = "wasm32")` in code and target-specific dependencies in the manifest | Covered |
| Native camera/GPU FPS | Bevy static-source perf path emits JSON summaries with render adapter metadata; native `--perf-summary-path` writes summaries directly as JSON artifacts; `tools/run_bevy_perf_matrix.sh` extracts per-case summaries, validates them and aggregate `summary.json` with `tools/validate_bevy_perf_summary.py --require-hardware-adapter`, and fails on CPU adapters; this host's native smoke selected llvmpipe CPU Vulkan, so live camera and hardware WebGPU/CUDA need a real adapter run | Hardware-blocked locally |

## local blockers

- Local GPU probing found an NVIDIA PCI device, but `nvidia-smi` cannot
  communicate with the driver and no `/dev/dri` render node is exposed. Local
  Bevy smoke output used the llvmpipe CPU Vulkan adapter, so it is useful for
  deterministic logic and ratio checks but not for real FPS claims.
- A refreshed one-frame native perf harness check on 2026-05-10 again selected
  `llvmpipe (LLVM 20.1.2, 256 bits)` with `device_type: Cpu` and exited through
  the `--require-hardware-adapter=true` guard before accepting any FPS value:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
CARGO=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo \
tools/run_bevy_perf_matrix.sh --frames 1 --out target/autogaze-bevy-perf-audit
```

## current local validation

The latest local non-browser release gate passed on 2026-05-10 after adding the
sparse readout adapters, Node-toolchain preflight, checked-in upstream fixture
coverage through the 448px AnyRes-style case, and generated-package checkout
fixture-only upstream parity:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
tools/check_release_readiness.sh
```

That gate ran root tests, Bevy tests, native and wasm clippy/check lanes,
example compilation, benchmark binary compilation, `cargo package`, generated
package checkout source-hygiene/upstream-fixture-parity/metrics/readout/example
checks, and `git diff --check`. After the gate, source hygiene was expanded and
rerun locally to cover Bevy metric/visualization delegation:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
cargo test -p burn_autogaze --features ndarray --test source_hygiene -- --nocapture
```

That focused run passed 8 tests.

The Bevy perf-summary metric contract is now also a tool-level release gate.
`tools/validate_bevy_perf_summary.py` rejects missing or non-finite FPS/timing
fields, gaze/update ratios outside `0.0..=1.0`, invalid or missing PSNR fields,
invalid enum/config fields, invalid frame/dimension counts, p95 lower than p50,
malformed aggregate `summary.json` files, mismatched min/max FPS extrema, and
CPU adapters when hardware is required. PSNR is JSON-safe: finite dB values are
numeric, and perfect/infinite PSNR is represented by a boolean flag rather than
an invalid JSON number. The release gate runs its built-in self-test, Python
tooling bytecode checks, shell syntax checks, and the native perf-matrix
dry-run; the native perf matrix validates every extracted per-case summary and
its aggregate summary before accepting FPS:

```sh
python3 tools/validate_bevy_perf_summary.py --self-test
python3 -m py_compile tools/generate_upstream_fixture.py tools/validate_bevy_perf_summary.py
bash -n tools/common.sh tools/check_bevy_wasm_demo.sh tools/check_burn_jepa_sparse_readout_integration.sh tools/check_release_readiness.sh tools/run_bevy_perf_matrix.sh
tools/run_bevy_perf_matrix.sh --dry-run --frames 2 --camera
tools/check_release_readiness.sh --dry-run | rg -n "bash -n|py_compile|validate_bevy_perf_summary|run_bevy_perf_matrix"
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
cargo test -p bevy_burn_autogaze inference_timing_summary_json_reports_well_formed_metrics -- --nocapture
```

The browser-demo command surface and Node override were checked without running
Playwright:

```sh
bash -n tools/common.sh tools/check_bevy_wasm_demo.sh tools/check_release_readiness.sh
tools/check_bevy_wasm_demo.sh --browser --dry-run
tools/check_release_readiness.sh --browser --node-bin-dir /tmp/node/bin --dry-run
tools/check_release_readiness.sh --real-model-browser --node-bin-dir /tmp/node-v22.11.0-linux-x64/bin --no-browser-deps --dry-run
bash -lc 'source tools/common.sh; autogaze_require_node_toolchain'
```

The final command intentionally fails on this host with the Snap
`snap-confine has elevated permissions` error and prints the non-Snap Node
override guidance.

The static-source Bevy wasm browser smoke later passed locally by staging
Node.js v22.11.0 under `/tmp`, using `--node-bin-dir`, and installing Playwright
Chromium under `/tmp` without system dependency escalation:

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright \
tools/check_bevy_wasm_demo.sh \
  --browser \
  --node-bin-dir /tmp/node-v22.11.0-linux-x64/bin \
  --skip-check \
  --no-browser-deps
```

That run passed `boots bevy wasm with static frames and no webcam` and
`starts wasm model load through async wgpu setup`.

The optional real-model wasm inference smoke also passed locally by symlinking
the cached Hugging Face AutoGaze snapshot into `crates/bevy_burn_autogaze/www`
for the duration of the run and enabling `--real-model-browser`:

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright \
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:/home/mosure/.cargo/bin:/tmp/node-v22.11.0-linux-x64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
tools/check_bevy_wasm_demo.sh \
  --real-model-browser \
  --node-bin-dir /tmp/node-v22.11.0-linux-x64/bin \
  --skip-check \
  --no-browser-deps
```

That run passed all three Playwright cases, including `runs optional real wasm
inference smoke when model assets are available`. The staged model symlinks
were removed after the run and are not tracked.

`tools/check_bevy_wasm_demo.sh --real-model-browser` now performs that staging
itself when cached model assets are available. It creates temporary
`www/config.json` and `www/model.safetensors` symlinks from
`AUTOGAZE_WASM_MODEL_DIR` or the default local Hugging Face AutoGaze snapshot,
runs the real-model Playwright case, and removes only the symlinks it created.
This prevents the real-model browser lane from silently degrading to a skipped
test on machines that already have the NVIDIA AutoGaze snapshot cached.

The same Bevy wasm browser smoke and optional real-model wasm inference smoke
passed again on 2026-05-10 after the low-level model input-preparation
robustness change and after adding automatic real-model asset staging. This
reconfirmed static-source startup, async wasm model-load error handling,
real-model WebGPU inference, perf summary fields, known wasm panic-string
absence, and temporary model-asset cleanup:

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright \
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:/home/mosure/.cargo/bin:/tmp/node-v22.11.0-linux-x64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
tools/check_bevy_wasm_demo.sh \
  --real-model-browser \
  --node-bin-dir /tmp/node-v22.11.0-linux-x64/bin \
  --skip-check \
  --no-browser-deps
```

A full release/readiness run with `--real-model-browser` also passed after the
Bevy tensor sparse-update policy was exposed through CLI/query config and perf
JSON. The run included root tests, Bevy tests, native and wasm clippy/check
lanes, benchmark binary builds, `cargo package`, generated package checkout
tests, static-source browser smoke, real-model browser smoke with staged model
assets, and `git diff --check`:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:/home/mosure/.cargo/bin:/tmp/node-v22.11.0-linux-x64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright \
tools/check_release_readiness.sh \
  --real-model-browser \
  --node-bin-dir /tmp/node-v22.11.0-linux-x64/bin \
  --no-browser-deps
```

The fixture-only upstream mask decoder coverage was expanded with checked-in
seeded 224px and 448px AnyRes-style fixtures generated from NVIDIA/Python
AutoGaze. The 448px fixture uses target scales `64+128+224+448`; this catches
layout regressions where larger upstream generated masks would be decoded as
the default 224px `32+64+112+224` layout:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
cargo test -p burn_autogaze --features ndarray \
  --test native_autogaze_generate_parity \
  upstream_generated_masks_decode_without_model_snapshot -- --nocapture
```

That run passed and covered the official 224 fixture, the full-resolution birds
fixture, `tests/fixtures/autogaze_upstream_resize_224`, and
`tests/fixtures/autogaze_upstream_tile_448`.

The same fixture-only test now also validates fixture metadata, per-scale mask
shapes and sums, per-frame mask sums, non-padded token coverage, and per-frame
generated-token ranges. This makes a malformed upstream fixture fail before it
can be used as false evidence for numerical parity.

The release-readiness script now reruns that fixture-only upstream parity test
from the generated package checkout after `cargo package`, which verifies the
packaged crate still includes the seeded 224px and 448px upstream fixture
assets:

```sh
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
cargo test --manifest-path target/package/burn_autogaze-0.21.2/Cargo.toml \
  --features ndarray --test native_autogaze_generate_parity \
  upstream_generated_masks_decode_without_model_snapshot -- --nocapture
```

That package-checkout command passed after the script update.

After adding the seeded 224px fixture, `cargo package -p burn_autogaze
--allow-dirty`, `cargo clippy -p burn_autogaze --features ndarray --test
native_autogaze_generate_parity -- -D warnings`, `cargo fmt --check`, and
`git diff --check` also passed.

The downstream `burn_jepa` sparse readout audit was run against the current
external checkout and intentionally failed, proving it catches the remaining
unmigrated local generated-token decoder and projection glue:

```sh
tools/check_burn_jepa_sparse_readout_integration.sh ../burn_jepa
```

The migration patch in `docs/burn-jepa-sparse-readout-migration.patch` was also
checked in a temporary copy of `../burn_jepa` with this checkout patched in as
the local `burn_autogaze` dependency:

```sh
tmp=$(mktemp -d /tmp/burn_jepa-readout.XXXXXX)
tar --exclude='./target' --exclude='./.git' -C ../burn_jepa -cf - . | tar -x -C "$tmp"
git -C "$tmp" apply "$PWD/docs/burn-jepa-sparse-readout-migration.patch"
perl -0pi -e 's#burn_autogaze = \{ version = "0\.21\.2", default-features = false, features = \["ndarray"\] \}#burn_autogaze = { version = "0.21.2", path = "'$PWD'", default-features = false, features = ["ndarray"] }#' "$tmp/Cargo.toml"
tools/check_burn_jepa_sparse_readout_integration.sh "$tmp"
PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH \
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc \
cargo check --manifest-path "$tmp/Cargo.toml" \
  --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu
find "$tmp" -depth -mindepth 1 -delete && rmdir "$tmp"
```

The following checks passed on 2026-05-09 in this workspace:

```sh
cargo test -p burn_autogaze --features ndarray
cargo test -p burn_autogaze --features ndarray nodes -- --nocapture
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo test -p bevy_burn_autogaze
cargo clippy -p burn_autogaze --features ndarray -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
cargo bench --bench backend_pipeline --features ndarray --no-run
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline --no-run
cargo package -p burn_autogaze --allow-dirty
cd crates/bevy_burn_autogaze && npm ci && npm run test:browser
AUTOGAZE_WASM_MODEL_E2E=1 npm run test:browser
cargo run -p bevy_burn_autogaze -- --image-path tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png --mode realtime --display-transfer cpu --perf-summary-frames 4 --log-pipeline-timing --show-psnr=false
cargo run -p bevy_burn_autogaze -- --image-path tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png --mode realtime --perf-summary-frames 1 --require-hardware-adapter=true
tools/run_bevy_perf_matrix.sh --dry-run --frames 2 --camera
tools/run_bevy_perf_matrix.sh --help
git diff --check
```

Additional sparse-readout checks passed on 2026-05-10 after adding tiled
no-trace packet readout points:

```sh
cargo test -p burn_autogaze --features ndarray nodes -- --nocapture
cargo test -p burn_autogaze --features ndarray pipeline -- --nocapture
cargo clippy -p burn_autogaze --features ndarray -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc cargo package -p burn_autogaze --allow-dirty
tools/run_bevy_perf_matrix.sh --dry-run --frames 2 --out /tmp/autogaze-perf-dry-run
git diff --check
```

Additional async sparse-readout checks passed on 2026-05-10 after adding
`AutoGazePipeline::readout_points_with_mode_async`,
`AutoGazeTensorPipeline::run_next_async`, and shared
`AutoGazeTensorPacketPlan` packet assembly. The readout option coverage also
includes `SparseReadoutOptions::with_max_fixations_per_frame`, which caps
decoded AutoGaze gaze points before downstream image-token projection:

```sh
cargo test -p burn_autogaze --features ndarray tensor_pipeline_run_next_async_matches_sync_readout_packets -- --nocapture
cargo test -p burn_autogaze --features ndarray async_readout_points_match_sync_for_resize_and_tiled_modes -- --nocapture
cargo test -p burn_autogaze --features ndarray readout_points_match_trace_points_for_resize_and_tiled_modes -- --nocapture
cargo test -p burn_autogaze --features ndarray nodes -- --nocapture
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo clippy -p burn_autogaze --features ndarray -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc cargo package -p burn_autogaze --allow-dirty
git diff --check
```

Additional upstream output-interpretation coverage passed on 2026-05-10 after
adding a fixture-only generated-mask decode test. This keeps the strongest
multi-scale mask interpretation check active even when local HF model assets are
not present. The same pass removed obsolete empty `native`/`web` Bevy features
so native and wasm builds are selected by target architecture only:

```sh
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity upstream_generated_masks_decode_without_model_snapshot -- --nocapture
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo bench --bench backend_pipeline --features ndarray --no-run
cargo check -p bevy_burn_autogaze
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
python3 tools/generate_upstream_fixture.py --help
python3 -m py_compile tools/generate_upstream_fixture.py
```

The fixture-only generated-mask decode test was later made directory-driven:
every `tests/fixtures/*/fixture_outputs.safetensors` file containing
`gazing_pos`, `num_gazing_each_frame`, `if_padded_gazing`, and at least one
`gazing_mask_*` tensor is decoded and checked automatically. New NVIDIA/Python
fixture directories therefore expand upstream output-interpretation coverage
without adding a new Rust test case:

```sh
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
```

Additional downstream sparse-video adapter checks passed on 2026-05-10 after
adding `SparseVideoReadoutGrid`, `SparseVideoReadoutOptions`,
`SparseVideoReadoutProjection`,
`frame_readout_tokens_to_video_tokens`, `frame_readout_rects_to_video_tokens`,
`frame_readout_tokens_to_video_coords`, and
`frame_readout_rects_to_video_coords`,
`frame_readout_tokens_to_video_coord_tensor`, and
`frame_readout_rects_to_video_coord_tensor`, plus one-call
`trace_to_video_readout_tokens`, `trace_to_video_readout_coords`,
`trace_to_video_readout_coord_tensor`,
`generated_to_video_readout_tokens`, `generated_to_video_readout_coords`,
`generated_to_video_readout_coord_tensor`, `video_readout_tokens_to_coord_tensor`, and
`batched_video_readout_tokens_to_coord_tensor`.
These tests mirror the relevant `burn_jepa` image-token to tubelet-token and
`burn_flex_gmm` sparse-patchify coordinate projection cases without depending
on either crate's concrete mask or kernel types. Packet tests cover
`AutoGazePipelinePacket::video_readout_tokens`, `video_readout_coords`, and
`video_readout_coord_tensor` for both trace-backed and generated-output-backed
packets:

```sh
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity -- --nocapture
cargo clippy -p burn_autogaze --features ndarray -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc cargo package -p burn_autogaze --allow-dirty
```

The backend benchmark target now includes `autogaze_sparse_readout_adapter`,
which measures host readout-to-coordinate projection and backend coordinate
tensor construction for the downstream sparse patchify bridge:

```sh
cargo bench --bench backend_pipeline --features ndarray --no-run
```

Additional sparse-patchifier geometry checks passed on 2026-05-10 after adding
`SparseVideoPatchGeometry` and `SparseVideoReadoutProjection::from_patch_geometry`.
This pass also fixed same-grid rectangle projection so exact AutoGaze cell
boundaries do not leak into neighboring readout tokens through floating-point
roundoff. The root and Bevy gates were rerun against the current tree:

```sh
cargo test -p burn_autogaze --features ndarray
cargo test -p bevy_burn_autogaze
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo package -p burn_autogaze --allow-dirty
git diff --check
```

The full non-browser release gate passed again on 2026-05-10 after factoring
shared script behavior into `tools/common.sh` and wiring both the release and
Bevy wasm-demo scripts through it. This reran root tests, Bevy wrapper tests,
clippy with warnings denied, native and wasm checks, benchmark compilation,
package verification, and whitespace checks from the shared gate:

```sh
bash -n tools/common.sh tools/check_bevy_wasm_demo.sh tools/check_release_readiness.sh
tools/check_bevy_wasm_demo.sh --browser --dry-run
tools/check_release_readiness.sh --browser --dry-run
tools/check_release_readiness.sh
```

The same full gate passed again after adding the compiled sparse-video readout
adapter example to both workspace and generated-package checks:

```sh
tools/check_release_readiness.sh
```

The release gate now also enforces the generated-package checkout regression
tests after `cargo package`, so package-only source hygiene, metrics, and sparse
readout coverage cannot drift from the workspace-only checks. The newly added
package-checkout commands passed directly on 2026-05-10:

```sh
cd target/package/burn_autogaze-0.21.2
cargo test --features ndarray --test source_hygiene -- --nocapture
cargo test --features ndarray metrics -- --nocapture
cargo test --features ndarray readout -- --nocapture
```

The GitHub test workflow now delegates the Rust, wasm, benchmark-build,
package, and static-source browser smoke gates to `tools/check_release_readiness.sh --browser`,
so CI and the documented release gate share the same command list instead of
drifting separately. The script installs the matching `wasm-bindgen-cli`,
preflights `node`/`npm`/`npx`, runs `npm ci`, installs Playwright Chromium,
builds the Bevy wasm artifacts, and runs the browser smoke when `--browser` is
passed. The non-browser gate completed successfully in this workspace:

```sh
bash -n tools/check_release_readiness.sh
tools/check_release_readiness.sh --browser --dry-run
tools/check_release_readiness.sh
```

The Pages workflow now delegates its wasm target check, wasm-bindgen install,
npm toolchain preflight, npm dependency install, Bevy wasm build, and
static-source browser smoke to `tools/check_bevy_wasm_demo.sh --browser`. The
readiness gate calls that same script for browser coverage, so the deploy and
test workflows share the wasm demo build path:

```sh
bash -n tools/check_bevy_wasm_demo.sh
tools/check_bevy_wasm_demo.sh --browser --dry-run
```

Local browser smoke on this host now passes with a portable non-Snap Node
toolchain. Because `sudo` is blocked by the sandbox, use `--no-browser-deps` to
run Playwright's Chromium install without attempting OS dependency installation.

Additional Bevy timing and package-readiness checks passed on 2026-05-10 after
separating camera source latency from frame preparation latency in
`process_frames`, fixing aggregate `processed_model_frames` accounting, and
checking that Python bytecode generated by fixture-script validation is ignored
and no longer included in the crate package:

```sh
cargo test -p bevy_burn_autogaze inference_timing_summary_json_reports_well_formed_metrics -- --nocapture
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo package -p burn_autogaze --allow-dirty
git diff --check
```

Additional core/Bevy wrapper checks passed on 2026-05-10 after adding
`AutoGazePipeline::readout_prepared_run` and
`readout_prepared_run_async`, then switching `bevy_burn_autogaze` to request
prepared readout points for display instead of always allocating full
`FrameFixationTrace`s. The streaming-cache path still uses the same core
streaming trace implementation internally because that is where KV-cache decode
state is maintained, but non-streaming resize/tiled display now uses the
no-trace readout path:

```sh
cargo test -p burn_autogaze --features ndarray readout_prepared_run -- --nocapture
cargo test -p bevy_burn_autogaze inference_timing_summary_json_reports_well_formed_metrics -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo package -p burn_autogaze --allow-dirty
git diff --check
```

Additional API robustness checks passed on 2026-05-10 after replacing the
remaining production `expect` in generated sparse-readout decoding with an
explicit error and making invalid native `--image-path` static-source loading
log and fall back instead of panicking:

```sh
cargo test -p burn_autogaze --features ndarray readout -- --nocapture
cargo test -p bevy_burn_autogaze invalid_static_image_path_falls_back_without_panic -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
cargo package -p burn_autogaze --allow-dirty
```

Additional streaming-cache robustness checks passed on 2026-05-10 after
replacing the post-initialization cache-state `expect` in the realtime/KV-cache
generation paths with `Option::get_or_insert_with`, preserving behavior while
removing an unnecessary panic edge from the hot path:

```sh
cargo test -p burn_autogaze --features ndarray model::tests::streaming -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
tools/check_release_readiness.sh
```

Additional low-level input robustness checks passed on 2026-05-10 after making
`AutoGazeGazingModel::embed_video` normalize non-model-sized direct inputs
instead of panicking on non-square tensors. The stricter
`AutoGazePipeline::try_embed_model_input` path now requires the exact configured
square model-input size, so callers still get a fallible API when they explicitly
promise preprocessed model input:

```sh
cargo test -p burn_autogaze --features ndarray low_level_embed_video_prepares_non_model_sized_inputs_without_panic -- --nocapture
cargo test -p burn_autogaze --features ndarray try_embed_model_input_rejects_non_square_model_input_without_panic -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
tools/check_release_readiness.sh
```

An executable migration sketch was added for downstream crates that need to
replace benchmark-local AutoGaze token parsing with the shared readout helpers.
The example projects generated multi-scale AutoGaze output into downstream
video-token indices and `[batch, temporal, row, col]` coordinates suitable for
`burn_flex_gmm` sparse patchify plan construction, while leaving the final mask
type downstream-owned:

```sh
cargo check --example sparse_video_readout_adapter --features ndarray
cargo run --example sparse_video_readout_adapter --features ndarray
```

Additional metric robustness checks passed on 2026-05-10 after centralizing
gaze-ratio percent formatting in `burn_autogaze`, sanitizing non-finite gaze
ratio samples to a bounded `0.0..=1.0` value, and ignoring invalid PSNR samples
so they do not poison the EMA used by Bevy overlays:

```sh
cargo test -p burn_autogaze --features ndarray metrics -- --nocapture
cargo test -p bevy_burn_autogaze metric_resources_delegate_to_core_stats -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
git diff --check
```

Additional Bevy native/wasm organization checks passed on 2026-05-10 after
collapsing duplicated native and wasm visualization orchestration into one
`run_autogaze_visualization` path. The only target-specific split now sits at
tensor readout: native uses `readout_prepared_run`, while wasm uses
`readout_prepared_run_async` so WebGPU tensor data is not read synchronously:

```sh
cargo test -p bevy_burn_autogaze metric_resources_delegate_to_core_stats -- --nocapture
cargo test -p bevy_burn_autogaze gpu_display_transfer_matches_cpu_visualization_outputs -- --nocapture
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
```

Additional model sync/async de-duplication checks passed on 2026-05-10 after
factoring greedy multi-token selection into one shared device-side selection
helper. The native path still performs the final read with `into_data()`, while
the wasm-safe async path reads the same packed selection tensor with
`into_data_async()`:

```sh
cargo test -p burn_autogaze --features ndarray greedy_selection -- --nocapture
cargo test -p burn_autogaze --features ndarray cached_generation_matches_uncached_generation -- --nocapture
cargo test -p burn_autogaze --features ndarray streaming_cached_generation_matches_batched_cached_generation -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
```

Additional source-hygiene coverage passed on 2026-05-10 after adding
`tests/source_hygiene.rs`. The test fails if Bevy wasm readout stops using the
async prepared-readout path, if Bevy wasm timing drifts back to
`std::time::Instant`, if the public wasm API gains synchronous tensor readback,
if async greedy selection stops using `into_data_async()`, if Bevy hides
realtime admission behind a fixed local policy instead of the shared configured
policy, or if Bevy, examples, or benches reintroduce local generated-token
parsing instead of using the core generated-output readout helpers. The
Bevy-source checks skip cleanly when the root crate is tested from a published
package that does not contain the excluded `crates/bevy_burn_autogaze` tree:

```sh
cargo test -p burn_autogaze --features ndarray --test source_hygiene -- --nocapture
```

Additional realtime admission checks passed on 2026-05-10 after exposing
Bevy's max-in-flight policy through config, native CLI, and browser query
options. The effective Bevy policy intentionally caps realtime streaming-cache
mode to one in-flight task, while allowing higher limits for tiled mode and
full-window non-streaming runs:

```sh
cargo test -p burn_autogaze --features ndarray runtime -- --nocapture
cargo test -p burn_autogaze --features ndarray --test source_hygiene -- --nocapture
cargo test -p bevy_burn_autogaze -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
git diff --check
```

Additional Bevy perf-summary coverage passed on 2026-05-10 after adding run
configuration fields to native JSON summaries and wasm
`window.__autogazePerf*` samples. Perf artifacts now include mode,
visualization mode, display-transfer path, streaming-cache flags, configured and
effective max-in-flight values, frame-window size, top-k, generation budget,
tile batch size, configured inference dimensions, and tensor sparse-update
policy thresholds. The full non-browser release/readiness gate passed afterward,
including root tests, Bevy tests, native and wasm clippy/check lanes, benchmark
binary builds, package verification, package-checkout source-hygiene/metrics/readout
tests, package example compilation, and `git diff --check`:

```sh
cargo test -p bevy_burn_autogaze inference_timing_summary_json_reports_well_formed_metrics -- --nocapture
cargo test -p bevy_burn_autogaze -- --nocapture
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
tools/check_release_readiness.sh
```

The same source-hygiene guard also passed from the generated package checkout
after adding the duplicate-generated-parser check:

```sh
cd target/package/burn_autogaze-0.21.2
cargo test --features ndarray --test source_hygiene -- --nocapture
```

The same source-hygiene, metric, and sparse-readout checks also passed from the
generated package checkout on 2026-05-10. This verifies the published root crate
keeps package-local regression coverage without requiring the excluded
`crates/bevy_burn_autogaze` example tree:

```sh
cd target/package/burn_autogaze-0.21.2
cargo test --features ndarray --test source_hygiene -- --nocapture
cargo test --features ndarray metrics -- --nocapture
cargo test --features ndarray readout -- --nocapture
```

Additional high-resolution visualization parity coverage passed on 2026-05-10
after fixing the effective-grid update footprint to pixelize known source-grid
cells with integer grid math instead of converting exact AutoGaze cells back
through floating-point bounds. The new fixture-only test loads the committed
1080p birds upstream fixture, verifies decoded fixation points reproduce the
upstream native per-scale masks, verifies the visible mask panel keeps native
multi-scale cells, and verifies interframe output copies only the effective
masked cells from the second frame:

```sh
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity upstream_birds_visualization_matches_fixture_masks_without_model_snapshot -- --nocapture
cargo test -p burn_autogaze --features ndarray visualization -- --nocapture
cargo test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity upstream_generated_masks_decode_without_model_snapshot -- --nocapture
cargo test -p bevy_burn_autogaze bevy_mask_panel_matches_output_update_mask_cells -- --nocapture
cargo test -p bevy_burn_autogaze gpu_display_transfer_matches_cpu_visualization_outputs -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo package -p burn_autogaze --allow-dirty
git diff --check
```

The birds visualization test is strict in a full workspace checkout and skips in
the generated package checkout because `Cargo.toml` intentionally excludes the
large 1080p birds fixture from the published crate. The package checkout still
keeps the official generated-mask fixture active:

```sh
cd target/package/burn_autogaze-0.21.2
cargo test --features ndarray --test native_autogaze_generate_parity upstream_birds_visualization_matches_fixture_masks_without_model_snapshot -- --nocapture
cargo test --features ndarray --test native_autogaze_generate_parity upstream_generated_masks_decode_without_model_snapshot -- --nocapture
```

Additional trace-store robustness checks passed on 2026-05-10 after making the
safetensors loader validate decoded tensor lengths against declared shapes
before indexing into fixation, scale, confidence, stop-probability, and
visibility tensors:

```sh
cargo test -p burn_autogaze --features ndarray safetensors_io -- --nocapture
cargo clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
cargo package -p burn_autogaze --allow-dirty
git diff --check
```

Additional API-safety checks passed on 2026-05-10 after adding
`AutoGazePipeline::try_embed_model_input`. This gives downstream sparse-readout
and video-pipeline integrations a non-panicking way to reject accidental
full-resolution rectangular tensors before they reach the low-level gazing
model, which still expects square model-input frames. Existing `embed_video`
remains the preferred public helper for ordinary video input because it applies
the model resize first.

```sh
cargo test -p burn_autogaze --features ndarray try_embed_model_input -- --nocapture
```

`cargo package` emitted a non-fatal cargo registry cache warning because the
local cargo index cache is read-only. The default rustup `rustc` wrapper also
hit the local transient-scope DBus failure during some test/package/clippy
invocations, so those checks were rerun with the direct nightly toolchain
binaries, using `RUSTC=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc`
and, for clippy, `PATH=/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$PATH`.
The browser tests were run with temporary symlinks from
`crates/bevy_burn_autogaze/www/config.json` and
`crates/bevy_burn_autogaze/www/model.safetensors` to the local NVIDIA AutoGaze
Hugging Face snapshot, then the symlinks were removed.
The native static-source smoke completed and printed a well-formed JSON perf
summary, but selected `llvmpipe (LLVM 20.1.2, 256 bits)` with Bevy's software
rendering warning. The summary now includes `render_adapter_name`,
`render_adapter_device_type`, `render_adapter_backend`, `render_adapter_driver`,
and related driver fields; the local run reported
`render_adapter_device_type: "Cpu"` and `render_adapter_backend: "Vulkan"`. The
first reported timing was `0.2 output fps` and `5459.9 ms` total on a CPU Vulkan
adapter, so it is not a valid hardware GPU throughput result.
The `--require-hardware-adapter=true` smoke now exits with status 1 on this
llvmpipe host after logging the selected software adapter, so scripted perf runs
can fail instead of recording CPU-adapter numbers.

## release gate

Use this command before publishing or claiming current non-hardware readiness:

```sh
tools/check_release_readiness.sh
```

It runs the command list below with the direct nightly toolchain path when that
toolchain is available locally:

```sh
cargo test -p burn_autogaze --features ndarray
cargo test -p bevy_burn_autogaze
cargo clippy -p burn_autogaze --features ndarray -- -D warnings
cargo clippy -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
cargo check --example sparse_video_readout_adapter --features ndarray
cargo bench --bench backend_pipeline --features ndarray --no-run
cargo bench -p bevy_burn_autogaze --bench viewer_pipeline --no-run
cargo package -p burn_autogaze --allow-dirty
cd target/package/burn_autogaze-0.21.2 && cargo test --features ndarray --test source_hygiene -- --nocapture
cd target/package/burn_autogaze-0.21.2 && cargo test --features ndarray metrics -- --nocapture
cd target/package/burn_autogaze-0.21.2 && cargo test --features ndarray readout -- --nocapture
cd target/package/burn_autogaze-0.21.2 && cargo check --example sparse_video_readout_adapter --features ndarray
```

Pass `--browser` where a non-Snap Node toolchain and a browser are available.
If the usable Node tools are not first on `PATH`, provide their bin directory.
Use `--no-browser-deps` in sandboxed environments where Playwright cannot call
`sudo` to install OS packages:

```sh
tools/check_release_readiness.sh --browser
tools/check_release_readiness.sh --browser --node-bin-dir /path/to/node/bin
tools/check_release_readiness.sh --browser --node-bin-dir /path/to/node/bin --no-browser-deps
```

To run the optional real-model browser inference smoke, stage local model assets
outside git and set `AUTOGAZE_WASM_MODEL_E2E=1`:

```sh
ln -s /path/to/AutoGaze/config.json www/config.json
ln -s /path/to/AutoGaze/model.safetensors www/model.safetensors
tools/check_release_readiness.sh --real-model-browser
rm www/config.json www/model.safetensors
```

Run this on the target GPU host for real throughput evidence:

```sh
tools/run_bevy_perf_matrix.sh --frames 120 --camera
```

The script passes `--perf-summary-path` so each case writes a JSON summary
artifact directly under `target/autogaze-bevy-perf/`, keeps the matching logs,
includes deterministic static-source cases and optional live-camera cases,
writes aggregate `target/autogaze-bevy-perf/summary.json`, and keeps
`--require-hardware-adapter=true` enabled for every run.
