# docs assets

| file | purpose |
|---|---|
| `autogaze_birds_input.gif` | source clip example; attribution is included in the root README image title |
| `autogaze_birds_mask.gif` | crisp white multi-scale token-cell mask visualization |
| `autogaze_birds_output.gif` | alpha-blended input/mask output visualization |
| `autogaze_capabilities.svg` | static overview of multi-scale masks, interframe output updates, and gaze/update-ratio metrics |

The gaze ratio metric is `updated output pixels / full-frame pixels`. Full-blend
frames and interframe keyframes count as `100%`; interframe delta frames count
only masked cells.
