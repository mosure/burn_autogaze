# completion audit

This file is the current-state audit for `burn_autogaze`. It is intentionally
short: long command logs belong in CI artifacts, benchmark reports, or the
specific validation docs.

## status

| area | current evidence | remaining gap |
|---|---|---|
| Core inference | Shared `AutoGazePipeline` APIs load NVIDIA configs/weights, preprocess RGBA clips, run realtime resize and tiled modes, and decode generated multi-scale fixations. | Broader upstream fixture corpus can still be expanded. |
| Upstream numerical parity | Fixture tests cover checked-in NVIDIA/Python embedding, generation, mask decode, birds visualization, and interframe outputs. `xtask upstream-fixture-matrix` can add more fixture cases. | Full upstream environment is not always available locally. |
| Realtime/Bevy defaults | Bevy defaults use target-cfg platform selection, 16-frame realtime mode, streaming cache ordering, adaptive display transfer, interframe output, PSNR overlay, and deduplicated mask geometry. | High-motion/full-frame camera moves still need app-level bottleneck isolation and stable FPS evidence. Live camera FPS requires a real hardware adapter run. |
| Native/wasm support | Native and wasm use the same Bevy UI layer. Wasm paths use async WebGPU setup and async tensor readback. | Browser perf numbers are host/browser dependent. |
| Metrics | FPS, gaze ratio, PSNR, perf summaries, and JSON validation live in shared core/xtask helpers. | Hardware throughput summaries must come from a GPU host, not llvmpipe. |
| Sparse readout | AutoGaze-specific scale geometry and image/video sparse-token adapters live in this crate. Downstream crates own model-specific masks and `burn_flex_gmm` dispatch. | `../burn_jepa` still needs the checked-in migration patch applied. |
| Release readiness | `cargo run -p xtask -- release-readiness` is the non-hardware gate for tests, clippy, wasm checks, package checks, benchmark builds, and whitespace checks. | Browser and hardware lanes must be requested explicitly. |

## prompt-to-artifact checklist

| requirement | source of truth |
|---|---|
| Clear public API for loading, inference, readout, visualization, and tensor pipeline use | `README.md`, `docs/api.md`, `src/lib.rs` |
| Native/wasm Bevy viewer with symmetric UI | `crates/bevy_burn_autogaze/README.md`, Bevy crate tests, Pages workflow |
| Multi-scale mask and interframe semantics | `docs/api.md`, visualization tests, upstream fixture tests |
| High-coverage mask performance process | `docs/performance-goal-loop.md`, `docs/benchmarking.md`, Bevy viewer benchmarks; this remains active until high-motion/full-frame Bevy runs no longer tank |
| Upstream/Python parity path | `docs/validation.md`, `docs/upstream_fixture_matrix.example.json`, `xtask upstream-fixture-matrix` |
| Sparse-token/downstream adapter boundary | `docs/sparse-readout-integration.md`, readout tests, adapter example |
| Publish and CI readiness | `docs/validation.md`, `xtask release-readiness`, GitHub workflows |

## required gates

Use the shared gate before publishing or claiming current non-hardware
readiness:

```sh
cargo run -p xtask -- release-readiness
```

Useful focused checks:

```sh
cargo test -p burn_autogaze --test source_hygiene
cargo test -p burn_autogaze --lib
cargo test -p bevy_burn_autogaze --lib
cargo clippy -p burn_autogaze -p bevy_burn_autogaze --all-targets -- -D warnings
cargo check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
git diff --check
```

Browser/Pages lane:

```sh
cargo run -p xtask -- check-bevy-wasm-demo --browser
```

If local Node tools are Snap-provided or otherwise unusable, pass
`--node-bin-dir /path/to/node/bin`. In sandboxed environments where Playwright
cannot install OS dependencies, add `--no-browser-deps` and provide a browser
cache.

Hardware throughput lane:

```sh
cargo run -p xtask -- bevy-perf-matrix --frames 120 --camera
```

That command records per-case JSON summaries under
`target/autogaze-bevy-perf/` and rejects software adapters when hardware is
required.

## strict completion

Strict completion requires the ordinary release gate, high-motion Bevy
performance evidence, and the external lanes that cannot be proven from this
checkout alone:

```sh
cargo run -p xtask -- completion-audit --strict \
  --burn-jepa ../burn_jepa \
  --hardware-perf \
  --frames 120
```

The `burn_jepa` lane is expected to fail until that sibling checkout applies
`docs/burn-jepa-sparse-readout-migration.patch`. The hardware lane is expected
to fail on hosts that expose only a CPU/software render adapter.

Do not treat this audit as complete while the high-motion/full-frame Bevy path
has unexplained FPS collapse, metric flicker, or sparse/dense dispatch behavior
that has not been benchmarked near its crossover point.

## local caveats

- Local FPS claims are invalid if Bevy selects `llvmpipe` or another CPU
  adapter. Use `--require-hardware-adapter=true` for perf runs that must fail
  instead of accepting software-rendered numbers.
- CUDA/WebGPU tests skip when the requested accelerator is unavailable.
- The browser smoke can run static-source tests without webcam access; real
  model browser inference needs staged or cached NVIDIA AutoGaze model assets.
