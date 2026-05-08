# docs assets

| file | purpose |
|---|---|
| `autogaze_birds_input.gif` | source clip example; attribution is included in the root README image title |
| `autogaze_birds_mask.gif` | actual crisp white multi-scale token-cell mask from the AutoGaze trace pipeline |
| `autogaze_birds_output.gif` | actual interframe alpha-blended output from `AutoGazeVisualizationState::Interframe` |
| `autogaze_birds_metrics.json` | renderer settings, model scale config, gaze/update ratios, and detected cell-grid histogram for the GIF run |

The gaze ratio metric is `updated output pixels / full-frame pixels`. Full-blend
frames and interframe keyframes count as `100%`; interframe delta frames count
only masked cells.

Regenerate these assets with:

```sh
cargo run --example render_readme_assets --features cuda -- \
  --input /home/mosure/Videos/birds.mp4 \
  --model-dir /path/to/AutoGaze \
  --out-dir docs
```
