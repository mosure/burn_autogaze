# docs assets

| file | purpose |
|---|---|
| `autogaze_birds_input.gif` | source clip example; attribution is included in the root README image title |
| `autogaze_birds_mask.gif` | scale-colored crisp multi-scale token-cell mask from the tiled AutoGaze trace pipeline |
| `autogaze_birds_output.gif` | interframe output stream from `AutoGazeVisualizationState::Interframe` |
| `autogaze_birds_metrics.json` | renderer settings, model scale config, tile/fixation budget, gaze/update ratios, PSNR, and detected cell-grid histogram for the GIF run |

The gaze ratio metric is `updated output pixels / full-frame pixels`. Full-blend
frames and interframe keyframes count as `100%`; interframe delta frames count
only masked cells.

The checked-in render uses full `1920x1080` inference, one `16`-frame clip, a
complete `2016x1120` AnyRes tile canvas, `45` `224px` chunks per frame, and the
model default `max_gaze_tokens_each_frame=198` plus `task_loss_requirement=0.7`.
The maximum fixation budget is `8910` tokens per high-res frame before task-loss
stopping and confidence filtering. Tiles are generated in backend batches of
`4` to avoid one very large CUDA/WebGPU autoregressive graph. The GIFs are
downsampled only after the tiled model trace and interframe output stream are
produced.

Regenerate these assets with:

```sh
cargo run --example render_readme_assets --features webgpu --no-default-features -- \
  --input /home/mosure/Videos/birds.mp4 \
  --model-dir /path/to/AutoGaze \
  --inference-width 1920 --inference-height 1080 \
  --tile-batch-size 4 \
  --out-dir docs
```

See [`sparse-readout-integration.md`](./sparse-readout-integration.md) for the
AutoGaze geometry to downstream sparse-token boundary, and
[`completion-audit.md`](./completion-audit.md) for validation coverage and
remaining hardware/browser checks.
