#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

include_browser=0
include_real_model_browser=0
install_browser_deps=1

usage() {
  cat <<'EOF'
usage: tools/check_release_readiness.sh [options]

Runs the local non-hardware release/readiness gate for burn_autogaze. This
covers root tests, Bevy wrapper tests, native and wasm checks, clippy with
warnings denied, benchmark compilation, package verification, and whitespace
checks. It also runs root-crate regression tests from the generated package
checkout. Browser tests are optional because local Node installs are often
environment-specific.

options:
  --browser             also run the static-source Playwright browser tests
  --real-model-browser  also run the optional real-model wasm browser smoke
  --no-browser-deps     install Playwright Chromium without sudo/system deps
  --cargo PATH          cargo binary to use (default: $CARGO or cargo)
  --node-bin-dir PATH   prepend PATH before node/npm/npx browser checks
  --dry-run             print commands without executing them
  -h, --help            show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --browser)
      include_browser=1
      shift
      ;;
    --real-model-browser)
      include_browser=1
      include_real_model_browser=1
      shift
      ;;
    --no-browser-deps)
      install_browser_deps=0
      shift
      ;;
    --cargo)
      autogaze_cargo_bin="${2:?missing value for --cargo}"
      shift 2
      ;;
    --node-bin-dir)
      autogaze_node_bin_dir="${2:?missing value for --node-bin-dir}"
      shift 2
      ;;
    --dry-run)
      autogaze_dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

autogaze_setup_direct_toolchain

cd "$autogaze_root_dir"

autogaze_run "$autogaze_cargo_bin" test -p burn_autogaze --features ndarray
autogaze_run "$autogaze_cargo_bin" test -p bevy_burn_autogaze
autogaze_run "$autogaze_cargo_bin" clippy -p burn_autogaze --features ndarray --all-targets -- -D warnings
autogaze_run "$autogaze_cargo_bin" clippy -p bevy_burn_autogaze --all-targets -- -D warnings
autogaze_run "$autogaze_cargo_bin" check -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm
autogaze_run "$autogaze_cargo_bin" clippy -p burn_autogaze --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
autogaze_run "$autogaze_cargo_bin" check -p bevy_burn_autogaze --target wasm32-unknown-unknown
autogaze_run "$autogaze_cargo_bin" clippy -p bevy_burn_autogaze --target wasm32-unknown-unknown -- -D warnings
autogaze_run "$autogaze_cargo_bin" check --example sparse_video_readout_adapter --features ndarray
autogaze_run bash -n tools/common.sh tools/check_bevy_wasm_demo.sh tools/check_burn_jepa_sparse_readout_integration.sh tools/check_release_readiness.sh tools/run_bevy_perf_matrix.sh
autogaze_run python3 -m py_compile tools/generate_upstream_fixture.py tools/validate_bevy_perf_summary.py
autogaze_run python3 tools/validate_bevy_perf_summary.py --self-test
autogaze_run tools/run_bevy_perf_matrix.sh --dry-run --frames 2 --camera
autogaze_run "$autogaze_cargo_bin" bench --bench backend_pipeline --features ndarray --no-run
autogaze_run "$autogaze_cargo_bin" bench -p bevy_burn_autogaze --bench viewer_pipeline --no-run
autogaze_run "$autogaze_cargo_bin" package -p burn_autogaze --allow-dirty
package_dir="$(autogaze_package_dir)"
if [[ "$autogaze_dry_run" -eq 0 && ! -d "$package_dir" ]]; then
  echo "expected generated package checkout missing: $package_dir" >&2
  exit 1
fi
autogaze_run_in "$package_dir" "$autogaze_cargo_bin" test --features ndarray --test source_hygiene -- --nocapture
autogaze_run_in "$package_dir" "$autogaze_cargo_bin" test --features ndarray --test native_autogaze_generate_parity upstream_generated_masks_decode_without_model_snapshot -- --nocapture
autogaze_run_in "$package_dir" "$autogaze_cargo_bin" test --features ndarray metrics -- --nocapture
autogaze_run_in "$package_dir" "$autogaze_cargo_bin" test --features ndarray readout -- --nocapture
autogaze_run_in "$package_dir" "$autogaze_cargo_bin" check --example sparse_video_readout_adapter --features ndarray

if [[ "$include_browser" -eq 1 ]]; then
  browser_args=(--skip-check --browser --cargo "$autogaze_cargo_bin")
  if [[ -n "$autogaze_node_bin_dir" ]]; then
    browser_args+=(--node-bin-dir "$autogaze_node_bin_dir")
  fi
  if [[ "$autogaze_dry_run" -eq 1 ]]; then
    browser_args+=(--dry-run)
  fi
  if [[ "$include_real_model_browser" -eq 1 ]]; then
    browser_args+=(--real-model-browser)
  fi
  if [[ "$install_browser_deps" -eq 0 ]]; then
    browser_args+=(--no-browser-deps)
  fi
  autogaze_run "$autogaze_root_dir/tools/check_bevy_wasm_demo.sh" "${browser_args[@]}"
fi

autogaze_run git diff --check
