# bevy_burn_autogaze

Bevy viewer for `burn_autogaze`. Native and wasm builds use the same Bevy app
and UI layer; platform code only supplies frames and model bytes.

## Native

```sh
cargo run -p bevy_burn_autogaze --features native -- \
  --model-dir /home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a \
  --mode resize-224
```

Use `--image-path path/to/frame.png` to run from a static image instead of the
native camera. `--mode tile-224` runs the tiled full-resolution path. Common
viewer/inference knobs include `--top-k`, `--frames-per-clip`,
`--max-gaze-tokens-each-frame`, `--inference-width`, `--inference-height`,
`--task-loss-requirement`, `--disable-task-loss-requirement`,
`--mask-cell-scale`, `--blend-alpha`, and `--show-fps`. `--show-gaze-ratio`
toggles the text overlay for per-frame and EMA output update ratio.
`--show-psnr=true` toggles PSNR in dB between the current input and rendered
output; the pixel comparison is skipped when this overlay is disabled.
`--log-pipeline-timing` prints pack, trace, visualization, and Bevy
texture-update timing every few seconds. In `tile-224` mode, source frames are
padded to a non-overlapping 224px chunk grid and
`--max-gaze-tokens-each-frame` controls the per-tile generation cap. The output
recovery stitches each tile-local scale grid into a full-frame grid for that
scale before clipping padded edge cells, matching upstream's mask recovery
semantics. The viewer default is `10` for realtime use, matching one NVIDIA
multi-token decoder step while exposing more of the multi-scale mask than the
previous 4-token cap. A value of `0` uses the model default, which is `198` for
the NVIDIA config and is not a realtime setting. The maximum frame budget is
`max-gaze-tokens-each-frame * tile-count`, before task-loss stopping and
padded-edge filtering. The viewer also defaults source frames to 224px wide
while preserving aspect ratio, and the native camera path requests a 640x360
stream for `resize-224` so camera decode does not dominate the realtime path.
Pass explicit `--inference-width` and `--inference-height` values for
full-resolution inspection.
Use `--load-model=false` to verify camera/preview rendering without waiting for
model load or inference.

`--visualization-mode full-blend` renders the current frame's alpha-blended
mask. The center mask panel colors the decoded multi-scale AutoGaze cells by
scale and draws crisp cell bounds. `--visualization-mode interframe
--keyframe-duration 30` preserves the previous output outside masked cells,
updates masked cells to the current input, and redraws a full keyframe every 30
processed frames. The gaze-ratio overlay reports the percentage of output pixels
updated on the current frame plus an EMA across processed frames. The PSNR
overlay reports current-frame and EMA dB for the output column compared to the
current input frame.

In `full-blend` mode every processed frame is a full redraw, so the update ratio
is `100%`. In `interframe` mode keyframes are also `100%`; intermediate frames
report masked-cell coverage as a percentage of the full source frame.

## Web

```sh
npm run build:wasm
npm run serve
```

Open `http://localhost:8080` in a WebGPU-capable browser. The web build fetches
NVIDIA AutoGaze `config.json` and `model.safetensors` from Hugging Face by
default and feeds browser camera frames through the exported `frame_input`
function.

The browser shell handles camera permission and frame upload only. The visible
UI is rendered by Bevy into the `#bevy` canvas, matching the native path. Pass
the same viewer/inference knobs as query parameters:

```text
http://localhost:8080/?mode=tile-224&visualization-mode=interframe&keyframe-duration=12&frames-per-clip=16&inference-width=1920&inference-height=1080&task-loss-requirement=0.7&show-fps=true&show-gaze-ratio=true&show-psnr=true
```

Use `?source=static` for a generated static frame, or `?image-url=./frame.png`
to drive the Bevy UI from an image without requesting a webcam.
`inference-width` and `inference-height` resize any received frame before it is
queued for model inference and visualization; for generated static frames, those
same query values also control the generated source resolution unless
`static-width` or `static-height` are set. `load-model=false` keeps the viewer in
preview mode for browser smoke tests.

Use `config-url` and `weights-url` query parameters to point the wasm build at
alternate model assets. `mask-radius-scale` remains accepted as a compatibility
alias for `mask-cell-scale`.
