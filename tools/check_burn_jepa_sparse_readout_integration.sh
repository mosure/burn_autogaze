#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/check_burn_jepa_sparse_readout_integration.sh PATH_TO_BURN_JEPA

Checks whether a burn_jepa checkout has migrated its AutoGaze sparse readout
adapter away from local generated-token decoding and onto burn_autogaze's
shared sparse readout helpers.

This is an external integration check. It is intentionally not part of the
local release gate because burn_jepa is a separate checkout.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

repo="$1"
bench="$repo/benches/autogaze_sparse_jepa_pipeline.rs"
manifest="$repo/Cargo.toml"

if [[ ! -f "$manifest" || ! -f "$bench" ]]; then
  echo "expected burn_jepa checkout with Cargo.toml and benches/autogaze_sparse_jepa_pipeline.rs: $repo" >&2
  exit 2
fi

status=0

require_contains() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if ! grep -q "$pattern" "$file"; then
    echo "missing: $message" >&2
    status=1
  fi
}

reject_contains() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if grep -q "$pattern" "$file"; then
    echo "still present: $message" >&2
    status=1
  fi
}

require_contains 'burn_autogaze = .*0\.21\.2' "$manifest" \
  "burn_jepa should depend on burn_autogaze >= 0.21.2 for sparse readout helpers"
require_contains 'generated_to_frame_readout_tokens' "$bench" \
  "bench should call burn_autogaze::generated_to_frame_readout_tokens for per-frame image readout"
require_contains 'generated_to_video_readout_tokens' "$bench" \
  "bench should call burn_autogaze::generated_to_video_readout_tokens for context mask projection"
require_contains 'SparseReadoutGrid' "$bench" \
  "bench should use burn_autogaze::SparseReadoutGrid for AutoGaze image-token geometry"
require_contains 'SparseVideoReadoutGrid' "$bench" \
  "bench should use burn_autogaze::SparseVideoReadoutGrid for downstream video-token geometry"
require_contains 'SparseVideoReadoutOptions' "$bench" \
  "bench should use burn_autogaze::SparseVideoReadoutOptions for tubelet/exact-token projection"

reject_contains 'fn generated_frame_tokens' "$bench" \
  "local generated_frame_tokens duplicates AutoGaze generated-output decoding"
reject_contains 'fn context_mask_from_autogaze_generated' "$bench" \
  "local context_mask_from_autogaze_generated duplicates AutoGaze image/video projection"
reject_contains 'raw_token - frame_offset' "$bench" \
  "bench should not manually subtract frame offsets from generated AutoGaze token ids"
reject_contains 'gazing_pos.first()' "$bench" \
  "bench should not manually index generated AutoGaze gazing_pos"

if [[ "$status" -eq 0 ]]; then
  echo "burn_jepa AutoGaze sparse readout integration looks migrated: $repo"
else
  cat >&2 <<'EOF'

Expected migration shape:
  - use burn_autogaze::generated_to_frame_readout_tokens for temporal stream frame tokens
  - use burn_autogaze::generated_to_video_readout_tokens for SparseTokenMask context indices
  - keep burn_jepa's SparseTokenMask, target-mask selection, plan caching, and burn_flex_gmm dispatch in burn_jepa
EOF
fi

exit "$status"
