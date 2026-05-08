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
`--max-gaze-tokens-each-frame`, `--mask-radius-scale`, `--blend-alpha`, and
`--show-fps`.

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
http://localhost:8080/?mode=tile-224&top-k=2&frames-per-clip=2&show-fps=true
```

Use `config-url` and `weights-url` query parameters to point the wasm build at
alternate model assets.
