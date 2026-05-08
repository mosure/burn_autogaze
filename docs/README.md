# docs assets

| file | purpose |
|---|---|
| `autogaze_birds_input.gif` | source clip example; attribution is included in the root README image title |
| `autogaze_birds_mask.gif` | scale-colored crisp multi-scale token-cell mask from the tiled AutoGaze trace pipeline |
| `autogaze_birds_output.gif` | interframe output stream from `AutoGazeVisualizationState::Interframe` |
| `autogaze_birds_metrics.json` | renderer settings, model scale config, tile/fixation budget, gaze/update ratios, and detected cell-grid histogram for the GIF run |

The gaze ratio metric is `updated output pixels / full-frame pixels`. Full-blend
frames and interframe keyframes count as `100%`; interframe delta frames count
only masked cells.

The checked-in render uses full `1920x1080` inference, `45` non-overlapping or
edge-overlapping `224px` tiles, and `top_k=16` per tile (`720` fixation budget
per frame). The GIFs are downsampled only after the tiled model trace and
interframe output stream are produced.

Regenerate these assets with:

```sh
cargo run --example render_readme_assets --features cuda -- \
  --input /home/mosure/Videos/birds.mp4 \
  --model-dir /path/to/AutoGaze \
  --inference-width 1920 --inference-height 1080 \
  --out-dir docs
```
