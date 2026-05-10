#!/usr/bin/env python3
"""Generate a matrix of NVIDIA AutoGaze upstream parity fixtures.

The single-case generator keeps the upstream preprocessing and model calls in
one place. This wrapper makes broader parity coverage repeatable without
importing the heavy Python model stack for --help or --dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate multiple upstream AutoGaze safetensors fixtures"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest with defaults and cases",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="override the model_dir from the manifest and cases",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="case name to run; repeat to select multiple cases",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip cases whose fixture_outputs.safetensors already exists",
    )
    parser.add_argument(
        "--allow-outside-fixtures",
        action="store_true",
        help="allow out_dir outside tests/fixtures",
    )
    parser.add_argument(
        "--run-parity-test",
        action="store_true",
        help="run the fixture-only Rust parity test after generation",
    )
    parser.add_argument(
        "--cargo",
        default=os.environ.get("CARGO", "cargo"),
        help="cargo binary for --run-parity-test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing them or importing upstream deps",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except FileNotFoundError as err:
        raise SystemExit(f"manifest not found: {path}") from err
    except json.JSONDecodeError as err:
        raise SystemExit(f"invalid JSON manifest {path}: {err}") from err

    if not isinstance(manifest, dict):
        raise SystemExit("manifest root must be a JSON object")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("manifest must contain a non-empty cases array")
    return manifest


def resolve_repo_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if not str(path):
        raise SystemExit(f"{field} must not be empty")
    return REPO_ROOT / path


def ensure_fixture_path(path: Path, allow_outside: bool) -> None:
    if allow_outside:
        return
    resolved = path.resolve(strict=False)
    fixtures_root = FIXTURES_ROOT.resolve(strict=False)
    try:
        resolved.relative_to(fixtures_root)
    except ValueError as err:
        raise SystemExit(
            f"fixture out_dir must be under tests/fixtures unless "
            f"--allow-outside-fixtures is passed: {path}"
        ) from err


def optional_value(case: dict[str, Any], defaults: dict[str, Any], key: str) -> Any:
    return case[key] if key in case else defaults.get(key)


def validate_case(case: Any, index: int) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise SystemExit(f"case {index} must be a JSON object")
    for key in ("name", "video", "out_dir"):
        if not case.get(key):
            raise SystemExit(f"case {index} is missing required field {key!r}")
    return case


def generation_command(
    case: dict[str, Any],
    defaults: dict[str, Any],
    cli_model_dir: Path | None,
    allow_outside: bool,
    dry_run: bool,
) -> tuple[str, Path, list[str]]:
    name = str(case["name"])
    video = resolve_repo_path(str(case["video"]), field=f"{name}.video")
    out_dir = resolve_repo_path(str(case["out_dir"]), field=f"{name}.out_dir")
    ensure_fixture_path(out_dir, allow_outside)

    model_dir_value = cli_model_dir or optional_value(case, defaults, "model_dir")
    model_dir = (
        resolve_repo_path(model_dir_value, field=f"{name}.model_dir")
        if model_dir_value
        else None
    )

    if not dry_run:
        if not video.is_file():
            raise SystemExit(f"case {name}: video not found: {video}")
        if model_dir is not None and not model_dir.is_dir():
            raise SystemExit(f"case {name}: model_dir not found: {model_dir}")

    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "generate_upstream_fixture.py"),
        "--video",
        str(video),
        "--out-dir",
        str(out_dir),
    ]
    if model_dir is not None:
        command.extend(["--model-dir", str(model_dir)])

    for key, flag in (
        ("frames", "--frames"),
        ("gazing_ratio", "--gazing-ratio"),
        ("task_loss_requirement", "--task-loss-requirement"),
    ):
        value = optional_value(case, defaults, key)
        if value is not None:
            command.extend([flag, str(value)])

    extra_args = case.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list) or not all(
            isinstance(item, str) for item in extra_args
        ):
            raise SystemExit(f"case {name}: extra_args must be an array of strings")
        command.extend(extra_args)

    return name, out_dir, command


def run(command: list[str], dry_run: bool) -> None:
    printable = shlex.join(command)
    if dry_run:
        print(printable)
        return
    print(f"+ {printable}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parity_command(cargo: str) -> list[str]:
    return [
        cargo,
        "test",
        "-p",
        "burn_autogaze",
        "--features",
        "ndarray",
        "--test",
        "native_autogaze_generate_parity",
        "upstream_generated_masks_decode_without_model_snapshot",
        "--",
        "--nocapture",
    ]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise SystemExit("manifest defaults must be a JSON object when present")

    selected = set(args.case_names or [])
    ran_any = False
    seen: set[str] = set()
    for index, raw_case in enumerate(manifest["cases"]):
        case = validate_case(raw_case, index)
        name, out_dir, command = generation_command(
            case,
            defaults,
            args.model_dir,
            args.allow_outside_fixtures,
            args.dry_run,
        )
        if name in seen:
            raise SystemExit(f"duplicate case name: {name}")
        seen.add(name)
        if selected and name not in selected:
            continue
        if args.skip_existing and (out_dir / "fixture_outputs.safetensors").exists():
            print(f"skipping {name}: fixture_outputs.safetensors already exists")
            continue
        print(f"# {name}")
        run(command, args.dry_run)
        ran_any = True

    missing = selected - seen
    if missing:
        raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")
    if not ran_any:
        raise SystemExit("no fixture cases selected")

    if args.run_parity_test:
        print("# fixture-only parity test")
        run(parity_command(args.cargo), args.dry_run)


if __name__ == "__main__":
    main()
