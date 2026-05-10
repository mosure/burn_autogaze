#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skip_check=0
include_browser=0
include_real_model_browser=0
install_browser_deps=1
default_wasm_model_dir="/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a"
staged_wasm_model_assets=()

usage() {
  cat <<'EOF'
usage: tools/check_bevy_wasm_demo.sh [options]

Builds and optionally browser-tests the Bevy wasm demo. The script installs the
wasm-bindgen-cli version pinned by Cargo.lock, installs npm dependencies, and
builds crates/bevy_burn_autogaze/www/out artifacts.
When `--real-model-browser` is enabled, the script stages `www/config.json`
and `www/model.safetensors` from `AUTOGAZE_WASM_MODEL_DIR` or the default local
Hugging Face AutoGaze snapshot when those files are not already present, then
removes only the assets it created.

options:
  --browser             also run the static-source Playwright browser smoke
  --real-model-browser  also run the optional real-model wasm browser smoke
  --skip-check          skip cargo check for the bevy wasm target
  --no-browser-deps     install Playwright Chromium without sudo/system deps
  --cargo PATH          cargo binary to use (default: $CARGO or cargo)
  --node-bin-dir PATH   prepend PATH before node/npm/npx checks
  --dry-run             print commands without executing them
  -h, --help            show this help
EOF
}

cleanup_staged_wasm_model_assets() {
  if [[ "$autogaze_dry_run" -eq 1 ]]; then
    return
  fi
  local path
  for path in "${staged_wasm_model_assets[@]}"; do
    rm -f "$path"
  done
}

stage_wasm_model_assets() {
  local www_dir="$autogaze_root_dir/crates/bevy_burn_autogaze/www"
  local model_dir="${AUTOGAZE_WASM_MODEL_DIR:-$default_wasm_model_dir}"
  local config_src="${AUTOGAZE_WASM_CONFIG:-$model_dir/config.json}"
  local weights_src="${AUTOGAZE_WASM_WEIGHTS:-$model_dir/model.safetensors}"
  local config_dst="$www_dir/config.json"
  local weights_dst="$www_dir/model.safetensors"

  if [[ -e "$config_dst" && -e "$weights_dst" ]]; then
    return
  fi
  if [[ ! -f "$config_src" || ! -f "$weights_src" ]]; then
    echo "real-model browser assets not staged; missing $config_src or $weights_src" >&2
    echo "Set AUTOGAZE_WASM_MODEL_DIR, AUTOGAZE_WASM_CONFIG, or AUTOGAZE_WASM_WEIGHTS to enable the real-model browser smoke." >&2
    return
  fi

  if [[ ! -e "$config_dst" && ! -L "$config_dst" ]]; then
    autogaze_run ln -s "$config_src" "$config_dst"
    staged_wasm_model_assets+=("$config_dst")
  fi
  if [[ ! -e "$weights_dst" && ! -L "$weights_dst" ]]; then
    autogaze_run ln -s "$weights_src" "$weights_dst"
    staged_wasm_model_assets+=("$weights_dst")
  fi
}

trap cleanup_staged_wasm_model_assets EXIT

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
    --skip-check)
      skip_check=1
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

if [[ "$skip_check" -eq 0 ]]; then
  autogaze_run "$autogaze_cargo_bin" check -p bevy_burn_autogaze --target wasm32-unknown-unknown
fi

autogaze_install_wasm_bindgen_cli
autogaze_require_node_toolchain
autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" npm ci
autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" npm run build:wasm

if [[ "$include_browser" -eq 1 ]]; then
  if [[ "$install_browser_deps" -eq 1 ]]; then
    autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" npx playwright install --with-deps chromium
  else
    autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" npx playwright install chromium
  fi
  autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" npm run test:browser
fi

if [[ "$include_real_model_browser" -eq 1 ]]; then
  stage_wasm_model_assets
  autogaze_run_in "$autogaze_root_dir/crates/bevy_burn_autogaze" env AUTOGAZE_WASM_MODEL_E2E=1 npm run test:browser
fi
