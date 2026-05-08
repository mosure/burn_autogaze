use crate::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazeLoadOptions, AutoGazePipeline,
    AutoGazeRgbaClipShape, NativeAutoGazeModel, rgba_clip_to_tensor, visualize_fixations_rgba,
};
use burn::tensor::backend::Backend;
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

type WasmBackend = burn::backend::WebGpu<f32, i32>;
type WasmDevice = burn::backend::wgpu::WgpuDevice;

static WEBGPU_DEVICE: OnceLock<WasmDevice> = OnceLock::new();

#[wasm_bindgen]
pub struct WasmAutoGaze {
    pipeline: AutoGazePipeline<WasmBackend>,
    device: WasmDevice,
    mode: AutoGazeInferenceMode,
    top_k: usize,
    mask_cell_scale: f32,
    blend_alpha: f32,
}

#[wasm_bindgen]
impl WasmAutoGaze {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, safetensors: &[u8]) -> Result<WasmAutoGaze, JsValue> {
        let _ = (config_json, safetensors);
        Err(js_error(
            "synchronous WebGPU setup is unsupported on wasm; use WasmAutoGaze.create(configJson, safetensors)",
        ))
    }

    #[wasm_bindgen(js_name = create)]
    pub async fn create(config_json: &str, safetensors: &[u8]) -> Result<WasmAutoGaze, JsValue> {
        console_error_panic_hook::set_once();
        let config: AutoGazeConfig = serde_json::from_str(config_json)
            .map_err(|err| js_error(format!("failed to parse AutoGaze config: {err}")))?;
        let device = webgpu_device().await;
        let model = NativeAutoGazeModel::<WasmBackend>::from_config_and_safetensors_bytes(
            &config,
            safetensors.to_vec(),
            &device,
            AutoGazeLoadOptions::strict(),
        )
        .map_err(|err| js_error(format!("failed to load AutoGaze weights: {err:#}")))?;
        let pipeline = AutoGazePipeline::new(model).with_max_gaze_tokens_each_frame(4);
        Ok(Self {
            pipeline,
            device,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 4,
            mask_cell_scale: 1.0,
            blend_alpha: 0.72,
        })
    }

    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    pub fn mode(&self) -> String {
        mode_label(self.mode)
    }

    pub fn set_resize_mode(&mut self) {
        self.mode = AutoGazeInferenceMode::ResizeToModelInput;
    }

    pub fn set_tiled_mode(&mut self, tile_size: usize, stride: usize) {
        self.mode = AutoGazeInferenceMode::TiledFullResolution { tile_size, stride };
    }

    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k.max(1);
    }

    pub fn set_max_gaze_tokens_each_frame(&mut self, max_tokens: usize) {
        self.pipeline.set_max_gaze_tokens_each_frame(max_tokens);
    }

    pub fn set_mask_radius_scale(&mut self, scale: f32) {
        self.set_mask_cell_scale(scale);
    }

    pub fn set_mask_cell_scale(&mut self, scale: f32) {
        self.mask_cell_scale = scale.clamp(0.25, 12.0);
    }

    pub fn set_blend_alpha(&mut self, alpha: f32) {
        self.blend_alpha = alpha.clamp(0.0, 1.0);
    }

    pub fn tile_count(&self, width: usize, height: usize) -> usize {
        match self.mode {
            AutoGazeInferenceMode::ResizeToModelInput => 1,
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                crate::AutoGazeTileLayout::tiled(height, width, tile_size, stride).tile_count()
            }
        }
    }

    pub fn infer_rgba_clip(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
        frames: usize,
    ) -> Result<WasmAutoGazeOutput, JsValue> {
        let width = width.max(1);
        let height = height.max(1);
        let frames = frames.max(1);
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| bytes.checked_mul(frames))
            .ok_or_else(|| js_error("clip dimensions overflow"))?;
        if rgba.len() != expected {
            return Err(js_error(format!(
                "expected {expected} RGBA bytes for {frames} frame(s) at {width}x{height}, got {}",
                rgba.len()
            )));
        }

        let shape = AutoGazeRgbaClipShape::new(frames, height, width);
        let video = rgba_clip_to_tensor::<WasmBackend>(rgba, shape, &self.device)
            .map_err(|err| js_error(format!("failed to build RGBA clip tensor: {err:#}")))?;
        let traces = self
            .pipeline
            .trace_video_with_mode(video, self.top_k, self.mode);
        WasmBackend::sync(&self.device)
            .map_err(|err| js_error(format!("failed to sync WebGPU backend: {err:?}")))?;

        let frame_index = frames.saturating_sub(1);
        let points = traces
            .first()
            .and_then(|trace| trace.frames.get(frame_index))
            .map(|set| set.points.clone())
            .unwrap_or_default();
        let points_json = serde_json::to_string(&points)
            .map_err(|err| js_error(format!("failed to serialize fixation points: {err}")))?;
        let last_frame = last_rgba_frame(rgba, width, height, frames);
        let visualization = visualize_fixations_rgba(
            last_frame,
            width,
            height,
            &points,
            self.mask_cell_scale,
            self.blend_alpha,
        )
        .map_err(|err| js_error(format!("failed to render AutoGaze visualization: {err:#}")))?;

        Ok(WasmAutoGazeOutput {
            width,
            height,
            side_by_side_width: visualization.side_by_side_width,
            mask_rgba: visualization.mask_rgba,
            blend_rgba: visualization.blend_rgba,
            side_by_side_rgba: visualization.side_by_side_rgba,
            points_json,
            mode: mode_label(self.mode),
            tile_count: self.tile_count(width, height),
        })
    }
}

#[wasm_bindgen]
pub struct WasmAutoGazeOutput {
    width: usize,
    height: usize,
    side_by_side_width: usize,
    mask_rgba: Vec<u8>,
    blend_rgba: Vec<u8>,
    side_by_side_rgba: Vec<u8>,
    points_json: String,
    mode: String,
    tile_count: usize,
}

#[wasm_bindgen]
impl WasmAutoGazeOutput {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn side_by_side_width(&self) -> usize {
        self.side_by_side_width
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.mode.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn tile_count(&self) -> usize {
        self.tile_count
    }

    pub fn mask_rgba(&self) -> Vec<u8> {
        self.mask_rgba.clone()
    }

    pub fn blend_rgba(&self) -> Vec<u8> {
        self.blend_rgba.clone()
    }

    pub fn side_by_side_rgba(&self) -> Vec<u8> {
        self.side_by_side_rgba.clone()
    }

    pub fn points_json(&self) -> String {
        self.points_json.clone()
    }
}

async fn webgpu_device() -> WasmDevice {
    if let Some(device) = WEBGPU_DEVICE.get() {
        return device.clone();
    }

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        &device,
        Default::default(),
    )
    .await;
    let _ = WEBGPU_DEVICE.set(device.clone());
    device
}

fn last_rgba_frame(rgba: &[u8], width: usize, height: usize, frames: usize) -> &[u8] {
    let frame_bytes = width * height * 4;
    let start = frames.saturating_sub(1) * frame_bytes;
    &rgba[start..start + frame_bytes]
}

fn mode_label(mode: AutoGazeInferenceMode) -> String {
    match mode {
        AutoGazeInferenceMode::ResizeToModelInput => "resize-224".to_string(),
        AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
            format!("tile-{tile_size}/{stride}")
        }
    }
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
