#!/usr/bin/env python3
"""Generate NVIDIA AutoGaze parity fixtures from the upstream Python model.

This script intentionally keeps the upstream video preprocessing path intact so
the Rust tests can catch non-square, real-video input drift before it reaches
the Bevy viewer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def load_upstream_dependencies() -> None:
    """Load heavy optional dependencies after argparse has handled --help."""

    global AutoGaze, AutoGazeImageProcessor, F, cv2, np, save_file, torch

    try:
        import cv2 as cv2_module
        import numpy as np_module
        import torch as torch_module
        import torch.nn.functional as functional_module
        from autogaze.models.autogaze import (
            AutoGaze as AutoGazeType,
            AutoGazeImageProcessor as AutoGazeImageProcessorType,
        )
        from safetensors.torch import save_file as save_file_fn
    except ModuleNotFoundError as err:
        raise SystemExit(
            "missing Python dependency for upstream fixture generation: "
            f"{err.name}. Install the NVIDIA AutoGaze Python environment, "
            "OpenCV, Torch, and safetensors before generating fixtures."
        ) from err

    cv2 = cv2_module
    np = np_module
    torch = torch_module
    F = functional_module
    AutoGaze = AutoGazeType
    AutoGazeImageProcessor = AutoGazeImageProcessorType
    save_file = save_file_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate an upstream AutoGaze safetensors fixture"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/"
            "snapshots/5100fae739ec1bf3f875914fa1b703846a18943a"
        ),
        help="Hugging Face snapshot containing config.json, preprocessor_config.json, and model.safetensors",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("/home/mosure/Videos/birds.mp4"),
        help="source video decoded with OpenCV as BGR then converted to RGB",
    )
    parser.add_argument("--frames", type=int, default=2, help="number of initial frames")
    parser.add_argument(
        "--gazing-ratio",
        type=float,
        default=0.75,
        help="upstream gazing_ratio argument",
    )
    parser.add_argument(
        "--task-loss-requirement",
        type=float,
        default=0.7,
        help="upstream task_loss_requirement argument",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/autogaze_birds_python_generate"),
        help="fixture directory to write",
    )
    return parser.parse_args()


def read_rgb_frames(path: Path, frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    decoded: list[np.ndarray] = []
    for _ in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not decoded:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(decoded)


def rgb_to_rgba(raw: np.ndarray) -> np.ndarray:
    alpha = np.full((*raw.shape[:-1], 1), 255, dtype=raw.dtype)
    return np.concatenate([raw, alpha], axis=-1)


def write_rgba_frames(raw: np.ndarray, out_dir: Path) -> None:
    rgba = rgb_to_rgba(raw)
    for frame_idx, frame in enumerate(rgba):
        path = out_dir / f"raw_rgba_frame_{frame_idx:02}.png"
        bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        if not cv2.imwrite(str(path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise RuntimeError(f"failed to write {path}")


def preprocess_video(processor: AutoGazeImageProcessor, raw: np.ndarray) -> torch.Tensor:
    pixel_values = processor(list(raw), return_tensors=None).pixel_values
    if isinstance(pixel_values[0], list):
        array = np.stack(pixel_values[0])
    else:
        array = np.stack(pixel_values)
    return torch.tensor(array, dtype=torch.float32).unsqueeze(0).contiguous()


def resize_for_gazing_model(model: AutoGaze, video: torch.Tensor) -> torch.Tensor:
    batch, frames, channels, height, width = video.shape
    input_size = model.gazing_model.input_img_size
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        size=(input_size, input_size),
        mode="bicubic",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, input_size, input_size).contiguous()


def stack_vision_embeds(model: AutoGaze, video: torch.Tensor) -> torch.Tensor:
    embeds, *_ = model.gazing_model.embed(video=video, use_cache=False)
    return torch.stack(embeds, dim=1).contiguous()


def stack_streaming_vision_embeds(model: AutoGaze, video: torch.Tensor) -> torch.Tensor:
    frames = video.shape[1]
    past_conv_values = None
    stacked: list[torch.Tensor] = []
    for frame_idx in range(frames):
        embeds, _, __, ___, past_conv_values = model.gazing_model.embed(
            video=video[:, frame_idx : frame_idx + 1],
            use_cache=True,
            past_conv_values=past_conv_values,
        )
        stacked.append(embeds[0])
    return torch.stack(stacked, dim=1).contiguous()


def main() -> None:
    args = parse_args()
    load_upstream_dependencies()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Upstream AutoGaze currently targets transformers 4.x. This compatibility
    # shim keeps fixture generation working when a newer global transformers is
    # installed.
    AutoGaze.all_tied_weights_keys = {}

    processor = AutoGazeImageProcessor.from_pretrained(args.model_dir)
    model = AutoGaze.from_pretrained(args.model_dir).eval()
    raw = read_rgb_frames(args.video, args.frames)
    video = preprocess_video(processor, raw)

    with torch.inference_mode():
        gazing_model_video = resize_for_gazing_model(model, video)
        video_embeds = stack_vision_embeds(model, gazing_model_video)
        streaming_video_embeds = stack_streaming_vision_embeds(model, gazing_model_video)
        outputs = model(
            {"video": video},
            gazing_ratio=args.gazing_ratio,
            task_loss_requirement=args.task_loss_requirement,
            generate_only=True,
        )

    tensors: dict[str, torch.Tensor] = {
        "video": video.cpu(),
        "gazing_model_video": gazing_model_video.cpu(),
        "video_embeds": video_embeds.cpu(),
        "streaming_video_embeds": streaming_video_embeds.cpu(),
        "gazing_pos": outputs["gazing_pos"].cpu().to(torch.int64).contiguous(),
        "num_gazing_each_frame": outputs["num_gazing_each_frame"]
        .cpu()
        .to(torch.int64)
        .contiguous(),
        "if_padded_gazing": outputs["if_padded_gazing"]
        .cpu()
        .to(torch.int64)
        .contiguous(),
    }
    for index, mask in enumerate(outputs["gazing_mask"]):
        tensors[f"gazing_mask_{index}"] = mask.cpu().float().contiguous()

    write_rgba_frames(raw, args.out_dir)
    save_file(tensors, args.out_dir / "fixture_outputs.safetensors")
    metadata = {
        "source": str(args.video),
        "frames": int(raw.shape[0]),
        "raw_shape": list(raw.shape),
        "raw_rgba_frames": [
            f"raw_rgba_frame_{frame_idx:02}.png" for frame_idx in range(raw.shape[0])
        ],
        "processed_shape": list(video.shape),
        "gazing_model_shape": list(gazing_model_video.shape),
        "video_embeds_shape": list(video_embeds.shape),
        "streaming_video_embed_max_abs_diff": float(
            (streaming_video_embeds - video_embeds).abs().max().item()
        ),
        "gazing_ratio": args.gazing_ratio,
        "task_loss_requirement": args.task_loss_requirement,
        "valid_tokens": int((~outputs["if_padded_gazing"]).sum().item()),
        "num_gazing_each_frame": outputs["num_gazing_each_frame"].cpu().tolist(),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
