#!/usr/bin/env python3
"""Validate bevy_burn_autogaze perf summary JSON.

The Bevy viewer emits summary JSON from native runs and browser samples through
`window.__autogazePerfSummary`. This script keeps the metric contract explicit
for harnesses that need to reject malformed FPS, gaze-ratio, PSNR, adapter, or
configuration fields before treating a run as evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tempfile
from typing import Any


NUMERIC_FIELDS = [
    "avg_output_fps",
    "avg_model_frame_fps",
    "avg_input_fps",
    "p95_total_ms",
]

FULL_SUMMARY_NUMERIC_FIELDS = [
    "avg_total_ms",
    "p50_total_ms",
    "avg_model_ms",
    "avg_input_ms",
    "avg_pack_ms",
    "avg_visualize_ms",
    "avg_visualize_cpu_ms",
    "avg_tensor_ms",
    "avg_display_ms",
]

RATIO_FIELDS = [
    "avg_gaze_update_ratio",
    "latest_gaze_update_ratio",
]

POSITIVE_INT_FIELDS = [
    "processed_frames",
    "latest_clip_frames",
    "latest_model_frames",
    "latest_width",
    "latest_height",
]

NONNEGATIVE_INT_FIELDS = [
    "processed_model_frames",
    "latest_trace_points",
    "latest_sequence",
]

CONFIG_ENUMS = {
    "mode": {"resize-224", "tiled"},
    "visualization_mode": {"full-blend", "interframe"},
    "display_transfer": {"cpu", "gpu"},
    "latest_tensor_interframe_path": {None, "dense-tensor", "sparse-rects"},
}


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_object(value: Any, path: pathlib.Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: summary must be a JSON object")
    return value


def require_number(
    data: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    required: bool = True,
) -> float | None:
    if field not in data:
        if required:
            raise ValueError(f"missing required numeric field `{field}`")
        return None
    value = data[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"`{field}` must be numeric, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"`{field}` must be finite, got {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"`{field}` must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"`{field}` must be <= {maximum}, got {value}")
    return value


def require_int(
    data: dict[str, Any],
    field: str,
    *,
    minimum: int | None = None,
    required: bool = True,
) -> int | None:
    if field not in data:
        if required:
            raise ValueError(f"missing required integer field `{field}`")
        return None
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"`{field}` must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"`{field}` must be >= {minimum}, got {value}")
    return value


def require_bool(data: dict[str, Any], field: str) -> None:
    if field in data and not isinstance(data[field], bool):
        raise ValueError(f"`{field}` must be a boolean")


def require_nullable_number(
    data: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if field not in data:
        raise ValueError(f"missing required nullable numeric field `{field}`")
    if data[field] is None:
        return None
    return require_number(data, field, minimum=minimum)


def require_bool_field(data: dict[str, Any], field: str) -> bool:
    if field not in data:
        raise ValueError(f"missing required boolean field `{field}`")
    if not isinstance(data[field], bool):
        raise ValueError(f"`{field}` must be a boolean")
    return bool(data[field])


def validate_summary(
    data: dict[str, Any],
    *,
    require_hardware_adapter: bool = False,
) -> None:
    for field in NUMERIC_FIELDS:
        require_number(data, field, minimum=0.0)

    for field in FULL_SUMMARY_NUMERIC_FIELDS:
        require_number(data, field, minimum=0.0, required=False)

    for field in RATIO_FIELDS:
        require_number(data, field, minimum=0.0, maximum=1.0)

    for field in POSITIVE_INT_FIELDS:
        require_int(data, field, minimum=1)

    for field in NONNEGATIVE_INT_FIELDS:
        require_int(data, field, minimum=0)

    target_frames = require_int(data, "target_frames", minimum=1, required=False)
    if target_frames is not None and data["processed_frames"] > target_frames:
        raise ValueError(
            "`processed_frames` must not exceed `target_frames`: "
            f"{data['processed_frames']} > {target_frames}"
        )

    p50 = require_number(data, "p50_total_ms", minimum=0.0, required=False)
    p95 = require_number(data, "p95_total_ms", minimum=0.0)
    if p50 is not None and p95 < p50:
        raise ValueError(f"`p95_total_ms` must be >= `p50_total_ms`: {p95} < {p50}")

    for field, values in CONFIG_ENUMS.items():
        if field in data and data[field] not in values:
            allowed = ", ".join(str(value) for value in sorted(values, key=str))
            raise ValueError(f"`{field}` must be one of {{{allowed}}}, got {data[field]!r}")

    for field in ["streaming_cache", "streaming_cache_effective"]:
        require_bool(data, field)

    latest_psnr = require_nullable_number(data, "latest_psnr_db", minimum=0.0)
    latest_psnr_infinite = require_bool_field(data, "latest_psnr_db_infinite")
    ema_psnr = require_nullable_number(data, "ema_psnr_db", minimum=0.0)
    ema_psnr_infinite = require_bool_field(data, "ema_psnr_db_infinite")
    show_psnr = data.get("show_psnr", False)
    if "show_psnr" in data and not isinstance(show_psnr, bool):
        raise ValueError("`show_psnr` must be a boolean")
    if latest_psnr_infinite and latest_psnr is not None:
        raise ValueError("`latest_psnr_db` must be null when `latest_psnr_db_infinite` is true")
    if ema_psnr_infinite and ema_psnr is not None:
        raise ValueError("`ema_psnr_db` must be null when `ema_psnr_db_infinite` is true")
    if show_psnr and latest_psnr is None and not latest_psnr_infinite:
        raise ValueError("PSNR is enabled but latest PSNR is neither finite nor infinite")
    if show_psnr and ema_psnr is None and not ema_psnr_infinite:
        raise ValueError("PSNR is enabled but EMA PSNR is neither finite nor infinite")

    for field in [
        "configured_max_in_flight",
        "effective_max_in_flight",
        "frames_per_clip",
        "top_k",
        "tile_batch_size",
        "inference_width",
        "inference_height",
    ]:
        require_int(data, field, minimum=1, required=False)

    require_int(data, "max_gaze_tokens_each_frame", minimum=0, required=False)
    require_int(data, "tensor_sparse_update_max_rects", minimum=0, required=False)
    require_number(
        data,
        "tensor_sparse_update_max_ratio",
        minimum=0.0,
        maximum=1.0,
        required=False,
    )

    adapter_type = data.get("render_adapter_device_type")
    if require_hardware_adapter:
        if not isinstance(adapter_type, str) or not adapter_type:
            raise ValueError("hardware-adapter validation requires `render_adapter_device_type`")
        if adapter_type.lower() == "cpu":
            adapter_name = data.get("render_adapter_name") or "unknown adapter"
            raise ValueError(f"render adapter was CPU: {adapter_name}")


def validate_aggregate_summary(
    data: dict[str, Any],
    *,
    require_hardware_adapter: bool = False,
) -> None:
    case_count = require_int(data, "case_count", minimum=1)
    min_output_fps = require_number(data, "min_output_fps", minimum=0.0)
    max_output_fps = require_number(data, "max_output_fps", minimum=0.0)
    if min_output_fps is not None and max_output_fps is not None and max_output_fps < min_output_fps:
        raise ValueError(
            f"`max_output_fps` must be >= `min_output_fps`: {max_output_fps} < {min_output_fps}"
        )

    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("aggregate summary `cases` must be a list")
    if len(cases) != case_count:
        raise ValueError(f"`case_count` does not match cases length: {case_count} != {len(cases)}")

    fps_values = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"aggregate summary case {index} must be an object")
        if "case" not in case or not isinstance(case["case"], str) or not case["case"]:
            raise ValueError(f"aggregate summary case {index} must include a nonempty `case` label")
        try:
            validate_summary(case, require_hardware_adapter=require_hardware_adapter)
        except ValueError as err:
            raise ValueError(f"aggregate summary case `{case['case']}` invalid: {err}") from err
        fps_values.append(float(case["avg_output_fps"]))

    actual_min = min(fps_values)
    actual_max = max(fps_values)
    if not math.isclose(actual_min, float(min_output_fps), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"`min_output_fps` mismatch: {min_output_fps} != {actual_min}")
    if not math.isclose(actual_max, float(max_output_fps), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"`max_output_fps` mismatch: {max_output_fps} != {actual_max}")


def validate_document(
    data: dict[str, Any],
    *,
    require_hardware_adapter: bool = False,
) -> str:
    if "cases" in data or "case_count" in data:
        validate_aggregate_summary(data, require_hardware_adapter=require_hardware_adapter)
        return "aggregate"
    validate_summary(data, require_hardware_adapter=require_hardware_adapter)
    return "summary"


def summary_line(data: dict[str, Any], path: pathlib.Path) -> str:
    if "cases" in data or "case_count" in data:
        case_count = int(data["case_count"])
        min_fps = float(data["min_output_fps"])
        max_fps = float(data["max_output_fps"])
        return f"{path}: {case_count} cases, output fps range={min_fps:.2f}..{max_fps:.2f}"

    fps = float(data["avg_output_fps"])
    total_ms = float(data.get("avg_total_ms", 0.0))
    model_ms = float(data.get("avg_model_ms", 0.0))
    gaze_ratio = float(data["avg_gaze_update_ratio"])
    if data.get("latest_psnr_db_infinite"):
        psnr = "inf"
    elif data.get("latest_psnr_db") is None:
        psnr = "n/a"
    else:
        psnr = f"{float(data['latest_psnr_db']):.2f}"
    adapter = data.get("render_adapter_name") or "unknown adapter"
    return (
        f"{path}: {fps:.2f} output fps, total={total_ms:.2f} ms, "
        f"model={model_ms:.2f} ms, gaze={gaze_ratio * 100.0:.2f}%, "
        f"psnr={psnr} dB, adapter={adapter}"
    )


def run_self_test() -> None:
    valid = {
        "target_frames": 2,
        "processed_frames": 2,
        "processed_model_frames": 4,
        "avg_output_fps": 50.0,
        "avg_model_frame_fps": 100.0,
        "avg_input_fps": 100.0,
        "avg_total_ms": 20.0,
        "p50_total_ms": 19.0,
        "p95_total_ms": 21.0,
        "avg_model_ms": 8.0,
        "avg_input_ms": 1.0,
        "avg_pack_ms": 0.5,
        "avg_visualize_ms": 2.0,
        "avg_visualize_cpu_ms": 0.5,
        "avg_tensor_ms": 0.25,
        "avg_display_ms": 0.5,
        "avg_gaze_update_ratio": 0.25,
        "latest_clip_frames": 2,
        "latest_model_frames": 2,
        "latest_trace_points": 3,
        "latest_gaze_update_ratio": 0.3,
        "latest_sequence": 7,
        "latest_width": 640,
        "latest_height": 360,
        "render_adapter_name": "Discrete GPU",
        "render_adapter_device_type": "DiscreteGpu",
        "mode": "resize-224",
        "visualization_mode": "interframe",
        "display_transfer": "gpu",
        "streaming_cache": True,
        "streaming_cache_effective": True,
        "configured_max_in_flight": 1,
        "effective_max_in_flight": 1,
        "frames_per_clip": 2,
        "top_k": 8,
        "max_gaze_tokens_each_frame": 0,
        "tile_batch_size": 64,
        "inference_width": 640,
        "inference_height": 360,
        "tensor_sparse_update_max_rects": 4,
        "tensor_sparse_update_max_ratio": 0.02,
        "show_psnr": True,
        "latest_psnr_db": 42.0,
        "latest_psnr_db_infinite": False,
        "ema_psnr_db": 42.0,
        "ema_psnr_db_infinite": False,
        "latest_tensor_interframe_path": "sparse-rects",
    }
    validate_summary(valid, require_hardware_adapter=True)

    aggregate = {
        "case_count": 2,
        "min_output_fps": 50.0,
        "max_output_fps": 75.0,
        "cases": [
            dict(valid, case="realtime-static", avg_output_fps=50.0),
            dict(valid, case="tiled-static", avg_output_fps=75.0),
        ],
    }
    validate_aggregate_summary(aggregate, require_hardware_adapter=True)

    invalid = dict(valid)
    invalid["avg_gaze_update_ratio"] = 1.5
    try:
        validate_summary(invalid)
    except ValueError as err:
        if "avg_gaze_update_ratio" not in str(err):
            raise
    else:
        raise AssertionError("invalid gaze ratio should fail")

    invalid = dict(valid)
    invalid["render_adapter_device_type"] = "Cpu"
    try:
        validate_summary(invalid, require_hardware_adapter=True)
    except ValueError as err:
        if "render adapter was CPU" not in str(err):
            raise
    else:
        raise AssertionError("CPU adapter should fail when hardware is required")

    invalid = dict(valid)
    invalid["latest_psnr_db"] = None
    try:
        validate_summary(invalid)
    except ValueError as err:
        if "PSNR is enabled" not in str(err):
            raise
    else:
        raise AssertionError("enabled PSNR without a finite or infinite value should fail")

    valid_infinite_psnr = dict(valid)
    valid_infinite_psnr["latest_psnr_db"] = None
    valid_infinite_psnr["latest_psnr_db_infinite"] = True
    valid_infinite_psnr["ema_psnr_db"] = None
    valid_infinite_psnr["ema_psnr_db_infinite"] = True
    validate_summary(valid_infinite_psnr)

    invalid_aggregate = dict(aggregate)
    invalid_aggregate["max_output_fps"] = 80.0
    try:
        validate_aggregate_summary(invalid_aggregate)
    except ValueError as err:
        if "max_output_fps" not in str(err):
            raise
    else:
        raise AssertionError("aggregate fps extrema mismatch should fail")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "summary.json"
        path.write_text(json.dumps(valid), encoding="utf-8")
        loaded = require_object(load_json(path), path)
        validate_document(loaded, require_hardware_adapter=True)

        aggregate_path = pathlib.Path(tmpdir) / "aggregate.json"
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        loaded_aggregate = require_object(load_json(aggregate_path), aggregate_path)
        validate_document(loaded_aggregate, require_hardware_adapter=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "--require-hardware-adapter",
        action="store_true",
        help="Reject CPU/software render adapters.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print one concise metric line per valid summary.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in validator smoke tests.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    if not args.paths:
        if args.self_test:
            return
        parser.error("at least one summary path is required")

    for path in args.paths:
        data = require_object(load_json(path), path)
        validate_document(data, require_hardware_adapter=args.require_hardware_adapter)
        if args.print_summary:
            print(summary_line(data, path))


if __name__ == "__main__":
    main()
