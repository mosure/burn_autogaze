# validation

Use `xtask` for release and browser gates. Repo-level command entry points
belong in `xtask`, not ad hoc shell or Python scripts.

## local release gate

```sh
cargo run -p xtask -- release-readiness
```

The local gate runs root and Bevy tests, native/wasm checks, clippy with
warnings denied, benchmark compilation, package verification, and
`git diff --check`. Add `--browser` on a host with a normal Node/Playwright
setup. Add `--real-model-browser` after staging local wasm model assets.

## focused commands

```sh
cargo test
cargo test --features cuda --test backend_pipeline -- --nocapture
cargo clippy --all-targets --features cuda -- -D warnings
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo check -p bevy_burn_autogaze --target wasm32-unknown-unknown
cargo clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
cargo package --allow-dirty
cd crates/bevy_burn_autogaze && npm ci && npm run test:browser
```

CUDA/WebGPU backend tests and benches skip cleanly when the requested accelerator
is not available.

## pages and browser demo

```sh
cargo run -p xtask -- check-bevy-wasm-demo --browser
```

This is the narrow Pages/demo gate used by the deploy workflow. It checks the
Bevy wasm target, installs the matching `wasm-bindgen-cli`, installs npm
dependencies, builds `www/out`, and runs the static-source browser smoke.

If the system Node tools are unavailable or Snap-provided, pass
`--node-bin-dir /path/to/node/bin` or set `AUTOGAZE_NODE_BIN_DIR`. In sandboxed
environments where Playwright cannot install OS dependencies, pass
`--no-browser-deps` and provide a browser cache such as
`PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright`.

## upstream fixture matrix

Generate NVIDIA/Python parity fixtures and immediately rerun fixture-only
parity checks with:

```sh
cargo run -p xtask -- upstream-fixture-matrix \
  --manifest docs/upstream_fixture_matrix.example.json \
  --run-parity-test
```

The example manifest keeps outputs under `tests/fixtures`, where the Rust
fixture tests discover them automatically.
