#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

burn_jepa_path=""
hardware_perf=0
frames=120
out_dir="target/autogaze-bevy-perf-audit"

usage() {
  cat <<'EOF'
usage: tools/check_completion_audit.sh [options]

Runs the focused completion-audit checks for burn_autogaze. The default checks
only in-repo evidence that is expected to pass on non-GPU CI/dev hosts:
source-hygiene, sparse readout coverage, upstream generated-mask fixture
decoding, and perf-summary schema validation.

options:
  --burn-jepa PATH      also require a sibling burn_jepa checkout to have
                       migrated onto burn_autogaze sparse-readout helpers
  --hardware-perf       also run the native Bevy perf matrix with hardware
                       adapter enforcement
  --frames N            frame count for --hardware-perf (default: 120)
  --out DIR             output directory for --hardware-perf artifacts
  --cargo PATH          cargo binary to use (default: $CARGO or cargo)
  --dry-run             print commands without executing them
  -h, --help            show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --burn-jepa)
      burn_jepa_path="${2:?missing value for --burn-jepa}"
      shift 2
      ;;
    --hardware-perf)
      hardware_perf=1
      shift
      ;;
    --frames)
      frames="${2:?missing value for --frames}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --cargo)
      autogaze_cargo_bin="${2:?missing value for --cargo}"
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

case "$frames" in
  ''|*[!0-9]*)
    echo "--frames must be a positive integer" >&2
    exit 2
    ;;
esac
if [[ "$frames" -eq 0 ]]; then
  echo "--frames must be greater than zero" >&2
  exit 2
fi

autogaze_setup_direct_toolchain

cd "$autogaze_root_dir"

autogaze_run "$autogaze_cargo_bin" test -p burn_autogaze --features ndarray --test source_hygiene -- --nocapture
autogaze_run "$autogaze_cargo_bin" test -p burn_autogaze --features ndarray readout -- --nocapture
autogaze_run "$autogaze_cargo_bin" test -p burn_autogaze --features ndarray --test native_autogaze_generate_parity upstream_generated_masks_decode_without_model_snapshot -- --nocapture
autogaze_run python3 tools/validate_bevy_perf_summary.py --self-test

if [[ -n "$burn_jepa_path" ]]; then
  autogaze_run tools/check_burn_jepa_sparse_readout_integration.sh "$burn_jepa_path"
else
  cat <<'EOF'

skipping burn_jepa migration audit; pass --burn-jepa PATH to enforce that the
sibling checkout no longer duplicates AutoGaze generated-token decoding.
EOF
fi

if [[ "$hardware_perf" -eq 1 ]]; then
  autogaze_run tools/run_bevy_perf_matrix.sh --frames "$frames" --out "$out_dir" --camera
else
  cat <<'EOF'

skipping native hardware Bevy perf audit; pass --hardware-perf on a host with a
real GPU render adapter and camera to enforce end-to-end throughput evidence.
EOF
fi

