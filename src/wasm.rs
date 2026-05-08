use crate::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazeLoadOptions, AutoGazePipeline, FixationPoint,
    NativeAutoGazeModel,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

type WasmBackend = burn::backend::WebGpu<f32, i32>;
type WasmDevice = burn::backend::wgpu::WgpuDevice;

static WEBGPU_INIT: OnceLock<()> = OnceLock::new();

#[wasm_bindgen]
pub struct WasmAutoGaze {
    pipeline: AutoGazePipeline<WasmBackend>,
    device: WasmDevice,
    mode: AutoGazeInferenceMode,
    top_k: usize,
    mask_radius_scale: f32,
    blend_alpha: f32,
}

#[wasm_bindgen]
impl WasmAutoGaze {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, safetensors: &[u8]) -> Result<WasmAutoGaze, JsValue> {
        console_error_panic_hook::set_once();
        let config: AutoGazeConfig = serde_json::from_str(config_json)
            .map_err(|err| js_error(format!("failed to parse AutoGaze config: {err}")))?;
        let device = webgpu_device();
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
            mask_radius_scale: 1.0,
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
        self.mask_radius_scale = scale.clamp(0.25, 12.0);
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

        let video = rgba_clip_to_tensor::<WasmBackend>(rgba, width, height, frames, &self.device);
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
        let visualization = visualize_points(
            last_frame,
            width,
            height,
            &points,
            self.mask_radius_scale,
            self.blend_alpha,
        );

        Ok(WasmAutoGazeOutput {
            width,
            height,
            side_by_side_width: width * 3,
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

struct Visualization {
    mask_rgba: Vec<u8>,
    blend_rgba: Vec<u8>,
    side_by_side_rgba: Vec<u8>,
}

fn webgpu_device() -> WasmDevice {
    let device = burn::backend::wgpu::WgpuDevice::default();
    WEBGPU_INIT.get_or_init(|| {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
            &device,
            Default::default(),
        );
    });
    device
}

fn rgba_clip_to_tensor<B: Backend>(
    rgba: &[u8],
    width: usize,
    height: usize,
    frames: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    let pixels_per_frame = width * height;
    let mut values = Vec::with_capacity(frames * 3 * pixels_per_frame);
    for frame in 0..frames {
        let frame_offset = frame * pixels_per_frame * 4;
        for channel in 0..3 {
            for pixel in 0..pixels_per_frame {
                values.push(rgba[frame_offset + pixel * 4 + channel] as f32 / 255.0);
            }
        }
    }
    Tensor::from_data(
        TensorData::new(values, [1, frames, 3, height, width]),
        device,
    )
}

fn last_rgba_frame(rgba: &[u8], width: usize, height: usize, frames: usize) -> &[u8] {
    let frame_bytes = width * height * 4;
    let start = frames.saturating_sub(1) * frame_bytes;
    &rgba[start..start + frame_bytes]
}

fn visualize_points(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    radius_scale: f32,
    blend_alpha: f32,
) -> Visualization {
    let pixels = width * height;
    let mut alpha = vec![0.0f32; pixels];
    let frame_extent = width.max(height) as f32;

    for point in points {
        if point.confidence <= 0.0 {
            continue;
        }
        let cx = point.x * (width.saturating_sub(1) as f32);
        let cy = point.y * (height.saturating_sub(1) as f32);
        let radius = (point.scale * frame_extent * radius_scale).max(12.0);
        let sigma = (radius * 0.45).max(1.0);
        let search = (radius * 2.0).ceil() as isize;
        let min_x = ((cx as isize) - search).max(0) as usize;
        let max_x = ((cx as isize) + search).min(width.saturating_sub(1) as isize) as usize;
        let min_y = ((cy as isize) - search).max(0) as usize;
        let max_y = ((cy as isize) + search).min(height.saturating_sub(1) as isize) as usize;
        let denom = 2.0 * sigma * sigma;

        for y in min_y..=max_y {
            let dy = y as f32 - cy;
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                let weight = (-(dx * dx + dy * dy) / denom).exp() * point.confidence;
                let idx = y * width + x;
                alpha[idx] = alpha[idx].max(weight.clamp(0.0, 1.0));
            }
        }
    }

    normalize_alpha(&mut alpha);

    let mut mask_rgba = vec![0u8; pixels * 4];
    let mut blend_rgba = vec![0u8; pixels * 4];
    let mut side_by_side_rgba = vec![0u8; width * 3 * height * 4];

    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let src = pixel * 4;
            let a = alpha[pixel].clamp(0.0, 1.0);
            let mask = (a * 255.0).round() as u8;
            mask_rgba[src] = mask;
            mask_rgba[src + 1] = mask;
            mask_rgba[src + 2] = mask;
            mask_rgba[src + 3] = 255;

            let overlay = (a * blend_alpha).clamp(0.0, 1.0);
            for channel in 0..3 {
                let base = rgba[src + channel] as f32;
                blend_rgba[src + channel] =
                    (base * (1.0 - overlay) + 255.0 * overlay).round() as u8;
            }
            blend_rgba[src + 3] = rgba[src + 3];

            write_side_by_side(
                &mut side_by_side_rgba,
                width,
                height,
                0,
                x,
                y,
                &rgba[src..src + 4],
            );
            write_side_by_side(
                &mut side_by_side_rgba,
                width,
                height,
                1,
                x,
                y,
                &mask_rgba[src..src + 4],
            );
            write_side_by_side(
                &mut side_by_side_rgba,
                width,
                height,
                2,
                x,
                y,
                &blend_rgba[src..src + 4],
            );
        }
    }

    Visualization {
        mask_rgba,
        blend_rgba,
        side_by_side_rgba,
    }
}

fn write_side_by_side(
    out: &mut [u8],
    width: usize,
    height: usize,
    column: usize,
    x: usize,
    y: usize,
    rgba: &[u8],
) {
    let out_width = width * 3;
    let out_x = column * width + x;
    let dst = (y.min(height - 1) * out_width + out_x) * 4;
    out[dst..dst + 4].copy_from_slice(rgba);
}

fn normalize_alpha(alpha: &mut [f32]) {
    let max_alpha = alpha.iter().copied().fold(0.0, f32::max);
    if max_alpha <= 0.0 {
        return;
    }
    for value in alpha {
        *value = (*value / max_alpha).clamp(0.0, 1.0);
    }
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
