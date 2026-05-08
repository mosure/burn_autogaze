# bevy_burn_autogaze

Bevy viewer for `burn_autogaze`.

## Native

```sh
cargo run -p bevy_burn_autogaze --features native -- \
  --model-dir /home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a \
  --mode resize-224
```

Use `--image-path path/to/frame.png` to run from a static image instead of the
native camera. `--mode tile-224` runs the tiled full-resolution path.

## Web

```sh
npm run build:wasm
npm run serve
```

Open `http://localhost:8080` in a WebGPU-capable browser. The web build fetches
NVIDIA AutoGaze `config.json` and `model.safetensors` from Hugging Face by
default and feeds browser camera frames through the exported `frame_input`
function.
