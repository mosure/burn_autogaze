use crate::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazeLoadOptions, AutoGazePipeline,
    AutoGazePipelineOptions, AutoGazeRgbaClipShape, AutoGazeVisualizationMode,
    AutoGazeVisualizationState, DEFAULT_BLEND_ALPHA, DEFAULT_KEYFRAME_DURATION,
    DEFAULT_REALTIME_TOP_K, DEFAULT_TILED_TILE_BATCH_SIZE, NativeAutoGazeModel, format_psnr_db,
    last_rgba_frame,
};
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
    visualization_mode: AutoGazeVisualizationMode,
    keyframe_duration: usize,
    visualization_state: AutoGazeVisualizationState,
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
        let pipeline = AutoGazePipeline::new(model).with_options(
            AutoGazePipelineOptions::default().with_tile_batch_size(DEFAULT_TILED_TILE_BATCH_SIZE),
        );
        Ok(Self {
            pipeline,
            device,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: DEFAULT_REALTIME_TOP_K,
            mask_cell_scale: 1.0,
            blend_alpha: DEFAULT_BLEND_ALPHA,
            visualization_mode: AutoGazeVisualizationMode::FullBlend,
            keyframe_duration: DEFAULT_KEYFRAME_DURATION,
            visualization_state: AutoGazeVisualizationState::new(
                AutoGazeVisualizationMode::FullBlend,
                DEFAULT_KEYFRAME_DURATION,
            ),
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

    pub fn set_anyres_tiled_mode(&mut self, tile_size: usize) {
        self.mode = AutoGazeInferenceMode::TiledResizeToGrid { tile_size };
    }

    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k.max(1);
    }

    pub fn set_max_gaze_tokens_each_frame(&mut self, max_tokens: usize) {
        self.pipeline.set_max_gaze_tokens_each_frame(max_tokens);
    }

    pub fn reset_max_gaze_tokens_each_frame(&mut self) {
        self.pipeline.reset_max_gaze_tokens_each_frame();
    }

    pub fn set_tile_batch_size(&mut self, tile_batch_size: usize) {
        self.pipeline.set_tile_batch_size(tile_batch_size);
    }

    pub fn set_task_loss_requirement(&mut self, task_loss_requirement: f32) {
        self.pipeline
            .set_task_loss_requirement(Some(task_loss_requirement));
    }

    pub fn disable_task_loss_requirement(&mut self) {
        self.pipeline.set_task_loss_requirement(None);
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

    pub fn visualization_mode(&self) -> String {
        self.visualization_mode.as_str().to_string()
    }

    pub fn set_visualization_mode(&mut self, mode: &str) -> Result<(), JsValue> {
        let mode = mode
            .parse()
            .map_err(|err| js_error(format!("failed to parse visualization mode: {err}")))?;
        self.visualization_mode = mode;
        self.visualization_state
            .configure(self.visualization_mode, self.keyframe_duration);
        Ok(())
    }

    pub fn set_full_blend_visualization_mode(&mut self) {
        self.visualization_mode = AutoGazeVisualizationMode::FullBlend;
        self.visualization_state
            .configure(self.visualization_mode, self.keyframe_duration);
    }

    pub fn set_interframe_visualization_mode(&mut self) {
        self.visualization_mode = AutoGazeVisualizationMode::Interframe;
        self.visualization_state
            .configure(self.visualization_mode, self.keyframe_duration);
    }

    pub fn set_keyframe_duration(&mut self, duration: usize) {
        self.keyframe_duration = duration.max(1);
        self.visualization_state
            .configure(self.visualization_mode, self.keyframe_duration);
    }

    pub fn reset_visualization_state(&mut self) {
        self.visualization_state.reset();
    }

    pub fn tile_count(&self, width: usize, height: usize) -> usize {
        match self.mode {
            AutoGazeInferenceMode::ResizeToModelInput => 1,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                crate::AutoGazeTileLayout::resized_grid(height, width, tile_size).tile_count()
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                crate::AutoGazeTileLayout::tiled(height, width, tile_size, stride).tile_count()
            }
        }
    }

    pub async fn infer_rgba_clip(
        &mut self,
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
        let traces = self
            .pipeline
            .trace_rgba_clip_with_mode_async(rgba, shape, self.top_k, self.mode, &self.device)
            .await
            .map_err(|err| js_error(format!("failed to read AutoGaze tensor data: {err:?}")))?;

        let frame_index = frames.saturating_sub(1);
        let points = traces
            .first()
            .and_then(|trace| trace.frames.get(frame_index))
            .map(|set| set.points.clone())
            .unwrap_or_default();
        let points_json = serde_json::to_string(&points)
            .map_err(|err| js_error(format!("failed to serialize fixation points: {err}")))?;
        let last_frame = last_rgba_frame(rgba, shape)
            .map_err(|err| js_error(format!("failed to select latest RGBA frame: {err:#}")))?;
        self.visualization_state
            .configure(self.visualization_mode, self.keyframe_duration);
        let visualization = self
            .visualization_state
            .visualize_rgba(
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
            mask_pixel_count: visualization.mask_pixel_count,
            updated_pixel_count: visualization.updated_pixel_count,
            mask_ratio: visualization.mask_ratio(),
            update_ratio: visualization.update_ratio(),
            psnr_db: visualization.output_psnr_db(last_frame).map_err(|err| {
                js_error(format!(
                    "failed to calculate AutoGaze output PSNR against input frame: {err:#}"
                ))
            })?,
            mask_rgba: visualization.mask_rgba,
            blend_rgba: visualization.blend_rgba,
            side_by_side_rgba: visualization.side_by_side_rgba,
            points_json,
            mode: mode_label(self.mode),
            visualization_mode: self.visualization_mode.as_str().to_string(),
            keyframe_duration: self.keyframe_duration,
            tile_count: self.tile_count(width, height),
        })
    }
}

#[wasm_bindgen]
pub struct WasmAutoGazeOutput {
    width: usize,
    height: usize,
    side_by_side_width: usize,
    mask_pixel_count: usize,
    updated_pixel_count: usize,
    mask_ratio: f64,
    update_ratio: f64,
    psnr_db: f64,
    mask_rgba: Vec<u8>,
    blend_rgba: Vec<u8>,
    side_by_side_rgba: Vec<u8>,
    points_json: String,
    mode: String,
    visualization_mode: String,
    keyframe_duration: usize,
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
    pub fn mask_pixel_count(&self) -> usize {
        self.mask_pixel_count
    }

    #[wasm_bindgen(getter)]
    pub fn updated_pixel_count(&self) -> usize {
        self.updated_pixel_count
    }

    #[wasm_bindgen(getter)]
    pub fn mask_ratio(&self) -> f64 {
        self.mask_ratio
    }

    #[wasm_bindgen(getter)]
    pub fn update_ratio(&self) -> f64 {
        self.update_ratio
    }

    #[wasm_bindgen(getter)]
    pub fn psnr_db(&self) -> f64 {
        self.psnr_db
    }

    pub fn psnr_text(&self) -> String {
        format_psnr_db(self.psnr_db)
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.mode.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn visualization_mode(&self) -> String {
        self.visualization_mode.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn keyframe_duration(&self) -> usize {
        self.keyframe_duration
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

    pub fn output_rgba(&self) -> Vec<u8> {
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

fn mode_label(mode: AutoGazeInferenceMode) -> String {
    match mode {
        AutoGazeInferenceMode::ResizeToModelInput => "resize-224".to_string(),
        AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
            format!("anyres-tile-{tile_size}")
        }
        AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
            format!("tile-{tile_size}/{stride}")
        }
    }
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
