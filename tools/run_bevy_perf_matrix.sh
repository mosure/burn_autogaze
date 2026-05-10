#!/usr/bin/env bash
set -euo pipefail

frames=120
image_path="tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png"
out_dir="target/autogaze-bevy-perf"
cargo_bin="${CARGO:-cargo}"
dry_run=0
include_camera=0
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
usage: tools/run_bevy_perf_matrix.sh [options]

Runs native bevy_burn_autogaze throughput checks on a target GPU host. Every
run uses --require-hardware-adapter=true so CPU/software render adapters fail
instead of producing misleading FPS numbers.

options:
  --frames N        processed inference outputs per case (default: 120)
  --image PATH     static RGBA source image for deterministic cases
  --out DIR        output directory for logs and extracted JSON summaries
  --camera         also run live camera cases
  --dry-run        print commands without executing them
  -h, --help       show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames)
      frames="${2:?missing value for --frames}"
      shift 2
      ;;
    --image)
      image_path="${2:?missing value for --image}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    --camera)
      include_camera=1
      shift
      ;;
    --dry-run)
      dry_run=1
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

run_case() {
  local name="$1"
  shift
  local log_path="$out_dir/$name.log"
  local json_path="$out_dir/$name.json"
  local -a command=(
    "$cargo_bin" run -p bevy_burn_autogaze --
    "$@"
    --perf-summary-frames "$frames"
    --perf-summary-path "$json_path"
    --log-pipeline-timing
    --require-hardware-adapter=true
  )

  printf '\n[%s]\n' "$name"
  printf '  '
  printf '%q ' "${command[@]}"
  printf '\n'

  if [[ "$dry_run" -eq 1 ]]; then
    return
  fi

  mkdir -p "$out_dir"
  set +e
  "${command[@]}" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "case $name failed with status $status; see $log_path" >&2
    return "$status"
  fi

  python3 - "$log_path" "$json_path" <<'PY'
import json
import sys

log_path, json_path = sys.argv[1:3]
try:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except FileNotFoundError:
    summary = None
    prefix = "AutoGaze perf summary:"
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if prefix in line:
                summary = line.split(prefix, 1)[1].strip()

    if summary is None:
        raise SystemExit(f"no AutoGaze perf summary found in {log_path}")
    data = json.loads(summary)

with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  python3 "$script_dir/validate_bevy_perf_summary.py" \
    --require-hardware-adapter \
    --print-summary \
    "$json_path"
}

write_summary() {
  python3 - "$out_dir" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    with path.open("r", encoding="utf-8") as handle:
        row = json.load(handle)
    row["case"] = path.stem
    rows.append(row)

if not rows:
    raise SystemExit(f"no per-case JSON summaries found in {out_dir}")

fps_values = [
    float(row.get("avg_output_fps", 0.0))
    for row in rows
    if isinstance(row.get("avg_output_fps"), (int, float))
]
summary = {
    "case_count": len(rows),
    "min_output_fps": min(fps_values) if fps_values else 0.0,
    "max_output_fps": max(fps_values) if fps_values else 0.0,
    "cases": rows,
}
summary_path = out_dir / "summary.json"
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(f"aggregate summary: {summary_path}")
for row in rows:
    fps = float(row.get("avg_output_fps", 0.0))
    total_ms = float(row.get("avg_total_ms", 0.0))
    model_ms = float(row.get("avg_model_ms", 0.0))
    adapter = row.get("render_adapter_name") or "unknown adapter"
    gaze_ratio = float(row.get("avg_gaze_update_ratio", 0.0))
    if row.get("latest_psnr_db_infinite"):
        psnr = "inf"
    elif row.get("latest_psnr_db") is None:
        psnr = "n/a"
    else:
        psnr = f"{float(row['latest_psnr_db']):.2f}"
    print(
        f"  {row['case']}: {fps:.2f} output fps, "
        f"total={total_ms:.2f} ms, model={model_ms:.2f} ms, "
        f"gaze={gaze_ratio * 100.0:.2f}%, psnr={psnr} dB, adapter={adapter}"
    )
PY
}

common_static=(--image-path "$image_path" --show-psnr=false)

run_case realtime-static-cpu \
  "${common_static[@]}" \
  --mode realtime \
  --display-transfer cpu \
  --visualization-mode full-blend

run_case realtime-static-gpu \
  "${common_static[@]}" \
  --mode realtime \
  --display-transfer gpu \
  --visualization-mode full-blend

run_case realtime-static-interframe \
  "${common_static[@]}" \
  --mode realtime \
  --display-transfer cpu \
  --visualization-mode interframe

run_case tiled-static-interframe \
  "${common_static[@]}" \
  --mode tiled \
  --display-transfer cpu \
  --visualization-mode interframe

if [[ "$include_camera" -eq 1 ]]; then
  run_case realtime-camera \
    --mode realtime \
    --display-transfer cpu \
    --visualization-mode full-blend

  run_case tiled-camera-interframe \
    --mode tiled \
    --display-transfer cpu \
    --visualization-mode interframe
fi

if [[ "$dry_run" -eq 0 ]]; then
  write_summary
  python3 "$script_dir/validate_bevy_perf_summary.py" \
    --require-hardware-adapter \
    --print-summary \
    "$out_dir/summary.json"
  echo
  echo "wrote logs and JSON summaries to $out_dir"
fi
