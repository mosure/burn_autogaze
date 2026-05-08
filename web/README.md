# burn_autogaze web demo

Build the wasm-bindgen package:

```sh
npm run build:wasm
```

Serve the demo:

```sh
npm run serve
```

Open `http://localhost:8080` in a browser with WebGPU enabled. The default
model URLs point at the NVIDIA AutoGaze Hugging Face files; local `config.json`
and `model.safetensors` URLs work as long as the server can fetch them.

The demo exposes both inference input modes (`resize-224`, `tile-224`) and
output visualization modes (`full blend`, `interframe`). Interframe mode keeps
stale output outside the mask and updates only masked token cells to the current
input between configurable keyframes. The stats line includes the current output
update ratio plus an EMA. The same query knobs used by the Bevy app are
available, including `visualization-mode`, `keyframe-duration`, `show-fps`, and
`show-gaze-ratio`.

Check that the Bevy web dependency anchor resolves the requested git revision
and the same `wgpu` v29 tree as Burn:

```sh
npm run check:bevy-wgpu
```
