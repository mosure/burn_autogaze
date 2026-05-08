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

Check that the Bevy web dependency anchor resolves the requested git revision
and the same `wgpu` v29 tree as Burn:

```sh
npm run check:bevy-wgpu
```
