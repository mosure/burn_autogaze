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
`--mask-cell-scale`, `--blend-alpha`, and `--show-fps`. `--show-gaze-ratio`
toggles the text overlay for per-frame and EMA output update ratio.
Use `--load-model=false` to verify camera/preview rendering without waiting for
model load or inference.

`--visualization-mode full-blend` renders the current frame's alpha-blended
mask. `--visualization-mode interframe --keyframe-duration 30` preserves the
previous output outside masked cells, updates masked cells to the current input,
and redraws a full keyframe every 30 processed frames. The gaze-ratio overlay
reports the percentage of output pixels updated on the current frame plus an EMA
across processed frames.

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
http://localhost:8080/?mode=tile-224&visualization-mode=interframe&keyframe-duration=30&top-k=2&frames-per-clip=2&inference-width=1920&inference-height=1080&show-fps=true&show-gaze-ratio=true
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
