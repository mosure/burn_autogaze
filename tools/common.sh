# Shared shell helpers for burn_autogaze tool scripts.

autogaze_root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
autogaze_cargo_bin="${CARGO:-cargo}"
autogaze_dry_run=0
autogaze_node_bin_dir="${AUTOGAZE_NODE_BIN_DIR:-}"

autogaze_setup_direct_toolchain() {
  local nightly_bin="/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin"
  if [[ -d "$nightly_bin" ]]; then
    export PATH="$nightly_bin:$PATH"
    export RUSTC="${RUSTC:-$nightly_bin/rustc}"
  fi
}

autogaze_setup_node_toolchain() {
  if [[ -n "$autogaze_node_bin_dir" ]]; then
    export PATH="$autogaze_node_bin_dir:$PATH"
  fi
}

autogaze_run() {
  printf '\n'
  printf '+'
  printf ' %q' "$@"
  printf '\n'

  if [[ "$autogaze_dry_run" -eq 0 ]]; then
    "$@"
  fi
}

autogaze_run_in() {
  local dir="$1"
  shift
  printf '\n'
  printf '+ (cd %q &&' "$dir"
  printf ' %q' "$@"
  printf ')\n'

  if [[ "$autogaze_dry_run" -eq 0 ]]; then
    (cd "$dir" && "$@")
  fi
}

autogaze_require_node_toolchain() {
  autogaze_setup_node_toolchain

  if [[ "$autogaze_dry_run" -eq 1 ]]; then
    autogaze_run node --version
    autogaze_run npm --version
    autogaze_run npx --version
    return
  fi

  local tool
  for tool in node npm npx; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "missing required browser-test tool: $tool" >&2
      echo "install Node.js, prepend a working toolchain to PATH, set AUTOGAZE_NODE_BIN_DIR, or pass --node-bin-dir PATH" >&2
      exit 1
    fi
  done

  local output status=0
  output="$(
    {
      printf 'node: '
      node --version
      printf 'npm: '
      npm --version
      printf 'npx: '
      npx --version
    } 2>&1
  )" || status=$?

  if [[ "$status" -ne 0 ]]; then
    echo "node/npm/npx preflight failed:" >&2
    echo "$output" >&2
    if grep -qi 'snap-confine' <<<"$output"; then
      echo >&2
      echo "The active Node.js toolchain appears to be Snap-provided and cannot run in this environment." >&2
      echo "Use a non-Snap Node.js install, set AUTOGAZE_NODE_BIN_DIR, or pass --node-bin-dir PATH." >&2
    fi
    exit "$status"
  fi

  echo "node toolchain:"
  echo "$output"
}

autogaze_wasm_bindgen_version() {
  awk '
    $0 == "[[package]]" { in_package = 0 }
    $0 == "name = \"wasm-bindgen\"" { in_package = 1 }
    in_package && $1 == "version" {
      gsub("\"", "", $3)
      print $3
      exit
    }
  ' "$autogaze_root_dir/Cargo.lock"
}

autogaze_install_wasm_bindgen_cli() {
  local version
  version="$(autogaze_wasm_bindgen_version)"
  if [[ -z "$version" ]]; then
    echo "failed to find wasm-bindgen version in Cargo.lock" >&2
    exit 1
  fi

  if command -v wasm-bindgen >/dev/null 2>&1; then
    local installed
    installed="$(wasm-bindgen --version | awk '{print $2}')"
    if [[ "$installed" == "$version" ]]; then
      echo "wasm-bindgen-cli $version already installed"
      return
    fi
  fi

  autogaze_run "$autogaze_cargo_bin" install wasm-bindgen-cli --version "$version" --locked
}

autogaze_package_field() {
  local field="$1"
  awk -v field="$field" '
    $0 == "[package]" { in_package = 1; next }
    in_package && /^\[/ { exit }
    in_package && $1 == field {
      gsub("\"", "", $3)
      print $3
      exit
    }
  ' "$autogaze_root_dir/Cargo.toml"
}

autogaze_package_dir() {
  local name version
  name="$(autogaze_package_field name)"
  version="$(autogaze_package_field version)"
  if [[ -z "$name" || -z "$version" ]]; then
    echo "failed to read package name/version from Cargo.toml" >&2
    exit 1
  fi
  printf '%s/target/package/%s-%s\n' "$autogaze_root_dir" "$name" "$version"
}
