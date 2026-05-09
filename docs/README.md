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

The checked-in render uses full `1920x1080` inference, one non-overlapping
`16`-frame clip, `45` padded `224px` chunks per frame, and the realtime viewer
cap `max_gaze_tokens_each_frame=10` plus `task_loss_requirement=0.7`. The
maximum fixation budget is `450` tokens per high-res frame before task-loss
stopping and padded-edge filtering. Tiles are generated in backend batches of
`8` to avoid one very large CUDA/WebGPU autoregressive graph. The GIFs are
downsampled only after the tiled model trace and interframe output stream are
produced.

Regenerate these assets with:

```sh
cargo run --example render_readme_assets --features webgpu --no-default-features -- \
  --input /home/mosure/Videos/birds.mp4 \
  --model-dir /path/to/AutoGaze \
  --inference-width 1920 --inference-height 1080 \
  --tile-batch-size 8 \
  --out-dir docs
```
