#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{
        Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
        RegisterDiagnostic,
    },
    ecs::system::SystemParam,
    ecs::world::CommandQueue,
    image::ImageSampler,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::widget::ImageNode,
    window::PrimaryWindow,
};
use bevy_burn::{BevyBurnBridgePlugin, BurnDevice};
use burn::tensor::Tensor;
#[cfg(test)]
use burn_autogaze::fixation_scale_mask_rgba;
#[cfg(target_arch = "wasm32")]
use burn_autogaze::{AutoGazeConfig, AutoGazeLoadOptions, NativeAutoGazeModel};
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape, AutoGazeVisualizationMode,
    AutoGazeVisualizationState, FixationPoint, rgba_clip_to_tensor,
};
use image::{RgbaImage, imageops::FilterType};

pub mod platform;

pub type AutoGazeBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type AutoGazeBevyDevice = burn::backend::wgpu::WgpuDevice;

pub const DEFAULT_NATIVE_MODEL_DIR: &str = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a";
pub const DEFAULT_CONFIG_URL: &str =
    "https://huggingface.co/nvidia/AutoGaze/resolve/main/config.json";
pub const DEFAULT_WEIGHTS_URL: &str =
    "https://huggingface.co/nvidia/AutoGaze/resolve/main/model.safetensors";
const MODEL_INPUT_SIZE: usize = 224;
pub const DEFAULT_REALTIME_INFERENCE_WIDTH: u32 = 640;
pub const DEFAULT_REALTIME_TOP_K: usize = 24;
pub const DEFAULT_MODEL_GENERATION_BUDGET: usize = 0;
pub const DEFAULT_REALTIME_FRAMES_PER_CLIP: usize = 2;
pub const DEFAULT_TILED_INFERENCE_WIDTH: u32 = 1280;
pub const DEFAULT_TILED_TOP_K: usize = 2;
pub const DEFAULT_TILED_FRAMES_PER_CLIP: usize = 2;
pub const DEFAULT_TILED_TILE_BATCH_SIZE: usize = 64;
const MAX_IN_FLIGHT_TASKS: usize = 1;
const MAX_SPARE_CLIP_BUFFERS: usize = 2;
pub const DEFAULT_KEYFRAME_DURATION: usize = 30;
pub const DEFAULT_BLEND_ALPHA: f32 = 0.38;
const GAZE_RATIO_EMA_ALPHA: f64 = 0.15;
const PSNR_EMA_ALPHA: f64 = 0.15;
const TIMING_LOG_INTERVAL_MS: f64 = 5_000.0;
const METRIC_OVERLAY_BOTTOM: f32 = 8.0;
const METRIC_OVERLAY_STEP: f32 = 34.0;
const INFERENCE_FPS: DiagnosticPath = DiagnosticPath::const_new("autogaze_inference_fps");

#[cfg(not(target_arch = "wasm32"))]
type Timestamp = Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Default)]
struct Timestamp(f64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BevyAutoGazeMode {
    #[default]
    Resize224,
    Tile224,
}

impl BevyAutoGazeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resize224 => "realtime",
            Self::Tile224 => "tiled",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "realtime",
            "resize",
            "resize-224",
            "tiled",
            "tile-224",
            "full-res",
        ]
    }

    pub const fn inference_mode(self) -> AutoGazeInferenceMode {
        match self {
            Self::Resize224 => AutoGazeInferenceMode::ResizeToModelInput,
            Self::Tile224 => AutoGazeInferenceMode::TiledResizeToGrid {
                tile_size: MODEL_INPUT_SIZE,
            },
        }
    }
}

impl std::str::FromStr for BevyAutoGazeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "realtime" | "resize" | "resize-224" | "resize-to-model" | "fast" => {
                Ok(Self::Resize224)
            }
            "tile" | "tile-224" | "tiled" | "full-res" | "fullres" | "anyres" => Ok(Self::Tile224),
            other => Err(format!(
                "unsupported autogaze mode `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

impl std::fmt::Display for BevyAutoGazeMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Resource, Clone, Debug)]
pub struct BevyBurnAutoGazeConfig {
    pub press_esc_to_close: bool,
    pub show_fps: bool,
    pub show_gaze_ratio: bool,
    pub show_psnr: bool,
    pub model_dir: PathBuf,
    pub config_url: String,
    pub weights_url: String,
    pub load_model: bool,
    pub image_path: Option<PathBuf>,
    pub mode: BevyAutoGazeMode,
    pub top_k: usize,
    pub max_gaze_tokens_each_frame: usize,
    pub tile_batch_size: usize,
    pub task_loss_requirement: Option<f32>,
    pub disable_task_loss_requirement: bool,
    pub frames_per_clip: usize,
    pub inference_width: Option<u32>,
    pub inference_height: Option<u32>,
    pub mask_cell_scale: f32,
    pub blend_alpha: f32,
    pub visualization_mode: AutoGazeVisualizationMode,
    pub keyframe_duration: usize,
    pub log_pipeline_timing: bool,
    pub perf_summary_frames: Option<usize>,
}

impl Default for BevyBurnAutoGazeConfig {
    fn default() -> Self {
        Self {
            press_esc_to_close: true,
            show_fps: true,
            show_gaze_ratio: true,
            show_psnr: false,
            model_dir: PathBuf::from(DEFAULT_NATIVE_MODEL_DIR),
            config_url: DEFAULT_CONFIG_URL.to_string(),
            weights_url: DEFAULT_WEIGHTS_URL.to_string(),
            load_model: true,
            image_path: None,
            mode: BevyAutoGazeMode::Resize224,
            top_k: DEFAULT_REALTIME_TOP_K,
            max_gaze_tokens_each_frame: DEFAULT_MODEL_GENERATION_BUDGET,
            tile_batch_size: DEFAULT_TILED_TILE_BATCH_SIZE,
            task_loss_requirement: None,
            disable_task_loss_requirement: false,
            frames_per_clip: DEFAULT_REALTIME_FRAMES_PER_CLIP,
            inference_width: Some(DEFAULT_REALTIME_INFERENCE_WIDTH),
            inference_height: None,
            mask_cell_scale: 1.0,
            blend_alpha: DEFAULT_BLEND_ALPHA,
            visualization_mode: AutoGazeVisualizationMode::FullBlend,
            keyframe_duration: DEFAULT_KEYFRAME_DURATION,
            log_pipeline_timing: false,
            perf_summary_frames: None,
        }
    }
}

impl BevyBurnAutoGazeConfig {
    pub fn apply_option(&mut self, key: &str, value: &str) -> Result<(), String> {
        let key = normalized_option_key(key);
        match key.as_str() {
            "" => Ok(()),
            "press-esc-to-close" => {
                self.press_esc_to_close = parse_bool_option(&key, value)?;
                Ok(())
            }
            "show-fps" => {
                self.show_fps = parse_bool_option(&key, value)?;
                Ok(())
            }
            "show-gaze-ratio" | "show-gaze" | "show-update-ratio" => {
                self.show_gaze_ratio = parse_bool_option(&key, value)?;
                Ok(())
            }
            "show-psnr" | "show-output-psnr" => {
                self.show_psnr = parse_bool_option(&key, value)?;
                Ok(())
            }
            "model-dir" => {
                self.model_dir = PathBuf::from(value);
                Ok(())
            }
            "config-url" | "config" => {
                self.config_url = value.to_string();
                Ok(())
            }
            "weights-url" | "weights" | "model-url" => {
                self.weights_url = value.to_string();
                Ok(())
            }
            "load-model" => {
                self.load_model = parse_bool_option(&key, value)?;
                Ok(())
            }
            "image-path" => {
                self.image_path = (!value.is_empty()).then(|| PathBuf::from(value));
                Ok(())
            }
            "mode" => {
                self.mode = value.parse()?;
                Ok(())
            }
            "top-k" => {
                self.top_k = parse_usize_option(&key, value)?;
                Ok(())
            }
            "max-gaze-tokens-each-frame" => {
                self.max_gaze_tokens_each_frame = parse_usize_option(&key, value)?;
                Ok(())
            }
            "tile-batch-size" | "tile-batch" | "tiles-per-batch" => {
                self.tile_batch_size = parse_usize_option(&key, value)?.max(1);
                Ok(())
            }
            "task-loss-requirement" | "task-loss" => {
                match value.trim().to_ascii_lowercase().as_str() {
                    "" | "default" | "model" => {
                        self.task_loss_requirement = None;
                        self.disable_task_loss_requirement = false;
                    }
                    "none" | "off" | "false" | "disabled" => {
                        self.task_loss_requirement = None;
                        self.disable_task_loss_requirement = true;
                    }
                    _ => {
                        self.task_loss_requirement = Some(parse_f32_option(&key, value)?);
                        self.disable_task_loss_requirement = false;
                    }
                }
                Ok(())
            }
            "disable-task-loss-requirement" | "disable-task-loss" => {
                self.disable_task_loss_requirement = parse_bool_option(&key, value)?;
                if self.disable_task_loss_requirement {
                    self.task_loss_requirement = None;
                }
                Ok(())
            }
            "frames-per-clip" => {
                self.frames_per_clip = parse_usize_option(&key, value)?;
                Ok(())
            }
            "inference-width" | "input-width" | "source-width" | "frame-width" | "width" => {
                self.inference_width = parse_optional_u32_option(&key, value)?;
                Ok(())
            }
            "inference-height" | "input-height" | "source-height" | "frame-height" | "height" => {
                self.inference_height = parse_optional_u32_option(&key, value)?;
                Ok(())
            }
            "mask-cell-scale" | "mask-radius-scale" => {
                self.mask_cell_scale = parse_f32_option(&key, value)?;
                Ok(())
            }
            "blend-alpha" => {
                self.blend_alpha = parse_f32_option(&key, value)?;
                Ok(())
            }
            "visualization-mode" | "visualisation-mode" | "viz-mode" => {
                self.visualization_mode = value.parse()?;
                Ok(())
            }
            "keyframe-duration" | "keyframe-interval" => {
                self.keyframe_duration = parse_usize_option(&key, value)?.max(1);
                Ok(())
            }
            "log-pipeline-timing" | "log-timing" | "timing" => {
                self.log_pipeline_timing = parse_bool_option(&key, value)?;
                Ok(())
            }
            "perf-summary-frames" | "perf-frames" | "benchmark-frames" => {
                self.perf_summary_frames = parse_optional_usize_option(&key, value)?;
                Ok(())
            }
            other => Err(format!("unsupported bevy_burn_autogaze option `{other}`")),
        }
    }

    pub fn apply_query_string(&mut self, query: &str) -> Vec<String> {
        let query = query.strip_prefix('?').unwrap_or(query);
        let mut errors = Vec::new();
        let mut saw_top_k = false;
        let mut saw_max_gaze_tokens = false;
        let mut saw_tile_batch_size = false;
        let mut saw_inference_width = false;
        let mut saw_inference_height = false;
        let mut saw_mode = false;
        let mut saw_frames_per_clip = false;

        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, "true"));
            let key = decode_url_component(key);
            let value = decode_url_component(value);
            match normalized_option_key(&key).as_str() {
                "mode" => saw_mode = true,
                "top-k" => saw_top_k = true,
                "max-gaze-tokens-each-frame" => saw_max_gaze_tokens = true,
                "tile-batch-size" | "tile-batch" | "tiles-per-batch" => {
                    saw_tile_batch_size = true;
                }
                "frames-per-clip" => saw_frames_per_clip = true,
                "inference-width" | "input-width" | "source-width" | "frame-width" | "width" => {
                    saw_inference_width = true;
                }
                "inference-height" | "input-height" | "source-height" | "frame-height"
                | "height" => {
                    saw_inference_height = true;
                }
                _ => {}
            }
            if let Err(err) = self.apply_option(&key, &value) {
                errors.push(err);
            }
        }

        if saw_mode {
            self.apply_implicit_mode_defaults(ImplicitModeDefaults {
                top_k: !saw_top_k,
                max_gaze_tokens_each_frame: !saw_max_gaze_tokens,
                tile_batch_size: !saw_tile_batch_size,
                frames_per_clip: !saw_frames_per_clip,
                inference_dimensions: !saw_inference_width && !saw_inference_height,
            });
        }

        errors
    }

    pub fn apply_implicit_mode_defaults(&mut self, defaults: ImplicitModeDefaults) {
        if defaults.top_k {
            self.top_k = default_top_k(self.mode);
        }
        if defaults.max_gaze_tokens_each_frame {
            self.max_gaze_tokens_each_frame = default_max_gaze_tokens_each_frame(self.mode);
        }
        if defaults.tile_batch_size {
            self.tile_batch_size = default_tile_batch_size(self.mode);
        }
        if defaults.frames_per_clip {
            self.frames_per_clip = default_frames_per_clip(self.mode);
        }
        if defaults.inference_dimensions {
            let (width, height) = default_inference_dimensions(self.mode);
            self.inference_width = width;
            self.inference_height = height;
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_browser_query() -> Self {
        let mut config = Self::default();
        if let Some(window) = web_sys::window() {
            match window.location().search() {
                Ok(search) => {
                    for err in config.apply_query_string(&search) {
                        log(&format!("ignoring invalid URL option: {err}"));
                    }
                }
                Err(err) => log(&format!("failed to read URL query: {err:?}")),
            }
        }
        config
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImplicitModeDefaults {
    pub top_k: bool,
    pub max_gaze_tokens_each_frame: bool,
    pub tile_batch_size: bool,
    pub frames_per_clip: bool,
    pub inference_dimensions: bool,
}

pub const fn default_top_k(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_REALTIME_TOP_K,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_TOP_K,
    }
}

pub const fn default_max_gaze_tokens_each_frame(_mode: BevyAutoGazeMode) -> usize {
    DEFAULT_MODEL_GENERATION_BUDGET
}

pub const fn default_tile_batch_size(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_TILED_TILE_BATCH_SIZE,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_TILE_BATCH_SIZE,
    }
}

pub const fn default_frames_per_clip(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_REALTIME_FRAMES_PER_CLIP,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_FRAMES_PER_CLIP,
    }
}

pub const fn default_inference_dimensions(mode: BevyAutoGazeMode) -> (Option<u32>, Option<u32>) {
    match mode {
        BevyAutoGazeMode::Resize224 => (Some(DEFAULT_REALTIME_INFERENCE_WIDTH), None),
        BevyAutoGazeMode::Tile224 => (Some(DEFAULT_TILED_INFERENCE_WIDTH), None),
    }
}

fn normalized_option_key(key: &str) -> String {
    key.trim().replace('_', "-").to_ascii_lowercase()
}

fn parse_bool_option(key: &str, value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean for `{key}`: `{value}`")),
    }
}

fn parse_usize_option(key: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid usize for `{key}`: `{value}`"))
}

fn parse_optional_u32_option(key: &str, value: &str) -> Result<Option<u32>, String> {
    if value.trim().is_empty() || value.eq_ignore_ascii_case("native") {
        return Ok(None);
    }
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("invalid u32 for `{key}`: `{value}`"))?;
    if parsed == 0 {
        return Err(format!("invalid zero dimension for `{key}`"));
    }
    Ok(Some(parsed))
}

fn parse_optional_usize_option(key: &str, value: &str) -> Result<Option<usize>, String> {
    if value.trim().is_empty()
        || value.eq_ignore_ascii_case("none")
        || value.eq_ignore_ascii_case("off")
    {
        return Ok(None);
    }
    Ok(Some(parse_usize_option(key, value)?.max(1)))
}

fn parse_f32_option(key: &str, value: &str) -> Result<f32, String> {
    value
        .parse()
        .map_err(|_| format!("invalid f32 for `{key}`: `{value}`"))
}

fn decode_url_component(value: &str) -> String {
    let value = value.replace('+', " ");
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) =
                (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
        {
            decoded.push(high << 4 | low);
            index += 3;
            continue;
        }

        decoded.push(bytes[index]);
        index += 1;
    }

    String::from_utf8(decoded).unwrap_or(value)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[derive(Resource)]
struct AutoGazeModelState {
    config: BevyBurnAutoGazeConfig,
    pipeline: Option<Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>>,
    load_task: Option<Task<Result<AutoGazePipeline<AutoGazeBevyBackend>, String>>>,
}

#[derive(Resource)]
struct AutoGazeTexture {
    image: Handle<Image>,
    entity: Option<Entity>,
    width: u32,
    height: u32,
}

impl Default for AutoGazeTexture {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            entity: None,
            width: 3,
            height: 1,
        }
    }
}

#[derive(Resource, Default)]
struct FrameQueue {
    width: u32,
    height: u32,
    frames: VecDeque<Arc<RgbaImage>>,
    spare_clip_buffers: Vec<Vec<u8>>,
}

impl FrameQueue {
    fn push(&mut self, frame: Arc<RgbaImage>, max_len: usize) {
        let max_len = max_len.max(1);
        let (width, height) = frame.dimensions();
        if self.width != width || self.height != height {
            self.frames.clear();
            self.spare_clip_buffers.clear();
            self.width = width;
            self.height = height;
        }

        self.frames.push_back(frame);
        while self.frames.len() > max_len {
            self.frames.pop_front();
        }
    }

    fn latest(&self) -> Option<&RgbaImage> {
        self.frames.back().map(AsRef::as_ref)
    }

    fn build_clip(&mut self, max_len: usize) -> Result<Option<FrameClip>, String> {
        let max_len = max_len.max(1);
        if self.frames.len() != max_len {
            return Ok(None);
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let frame_bytes = frame_byte_len(width, height)?;
        let required_bytes = frame_bytes
            .checked_mul(max_len)
            .ok_or_else(|| "AutoGaze clip byte length overflow".to_string())?;
        let pack_start = timestamp_now();
        let mut rgba = self
            .spare_clip_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(required_bytes));
        rgba.clear();
        if rgba.capacity() < required_bytes {
            rgba.reserve_exact(required_bytes - rgba.capacity());
        }

        for frame in &self.frames {
            if frame.width() as usize != width || frame.height() as usize != height {
                return Err("AutoGaze clip frame dimensions changed".to_string());
            }
            if frame.as_raw().len() != frame_bytes {
                return Err(format!(
                    "expected {frame_bytes} RGBA bytes for {width}x{height}, got {}",
                    frame.as_raw().len()
                ));
            }
            rgba.extend_from_slice(frame.as_raw());
        }

        Ok(Some(FrameClip {
            width,
            height,
            clip_len: max_len,
            rgba,
            source_ms: 0.0,
            prepare_ms: 0.0,
            pack_ms: elapsed_ms(pack_start),
        }))
    }

    fn recycle_clip_buffer(&mut self, mut rgba: Vec<u8>) {
        rgba.clear();
        if self.spare_clip_buffers.len() < MAX_SPARE_CLIP_BUFFERS {
            self.spare_clip_buffers.push(rgba);
        }
    }
}

struct FrameClip {
    width: usize,
    height: usize,
    clip_len: usize,
    rgba: Vec<u8>,
    source_ms: f64,
    prepare_ms: f64,
    pack_ms: f64,
}

impl FrameClip {
    fn shape(&self) -> AutoGazeRgbaClipShape {
        AutoGazeRgbaClipShape::new(self.clip_len, self.height, self.width)
    }

    fn last_frame_rgba(&self) -> Result<&[u8], String> {
        let frame_bytes = frame_byte_len(self.width, self.height)?;
        let start = self
            .clip_len
            .saturating_sub(1)
            .checked_mul(frame_bytes)
            .ok_or_else(|| "AutoGaze last frame offset overflow".to_string())?;
        let end = start
            .checked_add(frame_bytes)
            .ok_or_else(|| "AutoGaze last frame end overflow".to_string())?;
        self.rgba
            .get(start..end)
            .ok_or_else(|| "AutoGaze clip is missing its last frame".to_string())
    }
}

struct PreparedFrame<T> {
    width: usize,
    height: usize,
    tensor: T,
}

struct RollingFrameQueue<T> {
    width: usize,
    height: usize,
    frames: VecDeque<T>,
}

impl<T> Default for RollingFrameQueue<T> {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            frames: VecDeque::new(),
        }
    }
}

impl<T: Clone> RollingFrameQueue<T> {
    fn previous_frames(&mut self, width: usize, height: usize, count: usize) -> Vec<T> {
        if count == 0 {
            return Vec::new();
        }
        if self.width != width || self.height != height {
            self.frames.clear();
            self.width = width;
            self.height = height;
            return Vec::new();
        }
        self.frames
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

impl<T> RollingFrameQueue<T> {
    fn push(&mut self, frame: PreparedFrame<T>, max_len: usize) {
        let max_len = max_len.max(1);
        if self.width != frame.width || self.height != frame.height {
            self.frames.clear();
            self.width = frame.width;
            self.height = frame.height;
        }
        self.frames.push_back(frame.tensor);
        while self.frames.len() > max_len {
            self.frames.pop_front();
        }
    }
}

type PreparedTensorFrame = PreparedFrame<Tensor<AutoGazeBevyBackend, 5>>;

#[derive(Resource, Default)]
struct PreparedTensorQueue(RollingFrameQueue<Tensor<AutoGazeBevyBackend, 5>>);

impl PreparedTensorQueue {
    fn previous_frames(
        &mut self,
        width: usize,
        height: usize,
        count: usize,
    ) -> Vec<Tensor<AutoGazeBevyBackend, 5>> {
        self.0.previous_frames(width, height, count)
    }

    fn push(&mut self, frame: PreparedTensorFrame, max_len: usize) {
        self.0.push(frame, max_len);
    }
}

fn frame_byte_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "AutoGaze frame byte length overflow".to_string())
}

#[derive(Resource, Default, Clone)]
struct StaticFrame(Option<Arc<RgbaImage>>);

#[derive(Resource, Default, Clone, Debug)]
struct InferenceSequencer {
    next_sequence: u64,
    latest_applied_sequence: u64,
}

impl InferenceSequencer {
    fn reserve(&mut self) -> u64 {
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.next_sequence
    }

    fn accept(&mut self, sequence: u64) -> bool {
        if sequence <= self.latest_applied_sequence {
            return false;
        }
        self.latest_applied_sequence = sequence;
        true
    }
}

#[derive(Resource, Clone)]
struct BevyVisualizationState(AutoGazeVisualizationState);

impl BevyVisualizationState {
    fn new(mode: AutoGazeVisualizationMode, keyframe_duration: usize) -> Self {
        Self(AutoGazeVisualizationState::new(mode, keyframe_duration))
    }

    fn configure(&mut self, mode: AutoGazeVisualizationMode, keyframe_duration: usize) {
        self.0.configure(mode, keyframe_duration);
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

#[derive(Resource, Clone, Debug)]
struct GazeRatioStats {
    current: f64,
    ema: f64,
    initialized: bool,
}

impl Default for GazeRatioStats {
    fn default() -> Self {
        Self {
            current: 0.0,
            ema: 0.0,
            initialized: false,
        }
    }
}

impl GazeRatioStats {
    fn record(&mut self, ratio: f64) {
        self.current = ratio.clamp(0.0, 1.0);
        self.ema = if self.initialized {
            self.ema * (1.0 - GAZE_RATIO_EMA_ALPHA) + self.current * GAZE_RATIO_EMA_ALPHA
        } else {
            self.initialized = true;
            self.current
        };
    }
}

#[derive(Resource, Clone, Debug)]
struct PsnrStats {
    current: f64,
    ema: f64,
    initialized: bool,
}

impl Default for PsnrStats {
    fn default() -> Self {
        Self {
            current: 0.0,
            ema: 0.0,
            initialized: false,
        }
    }
}

impl PsnrStats {
    fn record(&mut self, psnr_db: f64) {
        self.current = psnr_db;
        self.ema = if self.initialized {
            ema_metric(self.ema, self.current, PSNR_EMA_ALPHA)
        } else {
            self.initialized = true;
            self.current
        };
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InferenceTiming {
    sequence: u64,
    clip_frames: usize,
    width: usize,
    height: usize,
    source_ms: f64,
    prepare_ms: f64,
    pack_ms: f64,
    input_ms: f64,
    model_ms: f64,
    trace_ms: f64,
    sync_ms: f64,
    visualize_cpu_ms: f64,
    tensor_ms: f64,
    visualize_ms: f64,
    display_ms: f64,
    total_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
}

#[derive(Resource, Clone, Debug, Default)]
struct InferenceTimingStats {
    latest: Option<InferenceTiming>,
    last_log: Option<Timestamp>,
    total_ms: f64,
    model_ms: f64,
    display_ms: f64,
    samples: Vec<f64>,
    emitted_summary: bool,
}

impl InferenceTimingStats {
    fn record(&mut self, timing: InferenceTiming, should_log: bool) {
        self.total_ms += timing.total_ms;
        self.model_ms += timing.model_ms;
        self.display_ms += timing.display_ms;
        self.samples.push(timing.total_ms);
        self.latest = Some(timing);
        publish_wasm_perf_sample(self);
        if !should_log {
            return;
        }

        let now = timestamp_now();
        let should_emit = self
            .last_log
            .map(|last_log| elapsed_between_ms(last_log, now) >= TIMING_LOG_INTERVAL_MS)
            .unwrap_or(true);
        if !should_emit {
            return;
        }

        self.last_log = Some(now);
        log(&format!(
            "AutoGaze timing: {:.1} fps e2e ({:.1} ms) clip={} {}x{}, source={:.1} ms, prepare={:.1} ms, pack={:.1} ms, input={:.1} ms, model={:.1} ms, trace={:.1} ms, sync={:.1} ms, visualize_cpu={:.1} ms, tensor={:.1} ms, visualize={:.1} ms, display={:.1} ms, output={:.1} MiB rgba/{:.1} MiB f32",
            timing.e2e_fps(),
            timing.total_ms,
            timing.clip_frames,
            timing.width,
            timing.height,
            timing.source_ms,
            timing.prepare_ms,
            timing.pack_ms,
            timing.input_ms,
            timing.model_ms,
            timing.trace_ms,
            timing.sync_ms,
            timing.visualize_cpu_ms,
            timing.tensor_ms,
            timing.visualize_ms,
            timing.display_ms,
            timing.output_rgba_bytes as f64 / (1024.0 * 1024.0),
            timing.output_tensor_bytes as f64 / (1024.0 * 1024.0),
        ));
    }

    fn processed_frames(&self) -> usize {
        self.samples.len()
    }

    fn summary_json(&self, target_frames: usize) -> String {
        let processed_frames = self.processed_frames();
        let avg_total_ms = mean_or_zero(self.total_ms, processed_frames);
        let avg_model_ms = mean_or_zero(self.model_ms, processed_frames);
        let avg_display_ms = mean_or_zero(self.display_ms, processed_frames);
        let p50_total_ms = percentile_ms(&self.samples, 0.50);
        let p95_total_ms = percentile_ms(&self.samples, 0.95);
        serde_json::json!({
            "target_frames": target_frames,
            "processed_frames": processed_frames,
            "avg_fps": fps_from_ms(avg_total_ms),
            "avg_total_ms": avg_total_ms,
            "p50_total_ms": p50_total_ms,
            "p95_total_ms": p95_total_ms,
            "avg_model_ms": avg_model_ms,
            "avg_display_ms": avg_display_ms,
            "latest_sequence": self.latest.map(|timing| timing.sequence).unwrap_or_default(),
            "latest_width": self.latest.map(|timing| timing.width).unwrap_or_default(),
            "latest_height": self.latest.map(|timing| timing.height).unwrap_or_default(),
        })
        .to_string()
    }
}

impl InferenceTiming {
    fn e2e_fps(self) -> f64 {
        if self.total_ms > 0.0 {
            1000.0 / self.total_ms
        } else {
            0.0
        }
    }
}

fn mean_or_zero(total: f64, count: usize) -> f64 {
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn fps_from_ms(ms: f64) -> f64 {
    if ms > 0.0 { 1000.0 / ms } else { 0.0 }
}

fn percentile_ms(samples: &[f64], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index =
        ((sorted.len().saturating_sub(1)) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

#[cfg(target_arch = "wasm32")]
fn publish_wasm_perf_sample(stats: &InferenceTimingStats) {
    use wasm_bindgen::JsValue;

    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(latest) = stats.latest else {
        return;
    };
    let value = serde_json::json!({
        "processed_frames": stats.processed_frames(),
        "latest_sequence": latest.sequence,
        "latest_total_ms": latest.total_ms,
        "latest_model_ms": latest.model_ms,
        "latest_display_ms": latest.display_ms,
        "avg_fps": fps_from_ms(mean_or_zero(stats.total_ms, stats.processed_frames())),
        "p95_total_ms": percentile_ms(&stats.samples, 0.95),
    })
    .to_string();
    let value = js_sys::JSON::parse(&value).unwrap_or_else(|_| JsValue::from_str(&value));
    let _ = js_sys::Reflect::set(&window, &JsValue::from_str("__autogazePerf"), &value);
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_wasm_perf_sample(_stats: &InferenceTimingStats) {}

#[derive(SystemParam)]
struct FrameInputParams<'w> {
    config: Res<'w, BevyBurnAutoGazeConfig>,
    static_frame: Res<'w, StaticFrame>,
    frame_queue: ResMut<'w, FrameQueue>,
    inference_sequencer: ResMut<'w, InferenceSequencer>,
    visualization_state: ResMut<'w, BevyVisualizationState>,
    prepared_tensor_queue: ResMut<'w, PreparedTensorQueue>,
    gaze_ratio_stats: ResMut<'w, GazeRatioStats>,
    psnr_stats: ResMut<'w, PsnrStats>,
    timing_stats: Res<'w, InferenceTimingStats>,
}

#[derive(Component)]
struct ProcessAutoGaze(Task<CommandQueue>);

pub fn viewer_app(config: BevyBurnAutoGazeConfig) -> App {
    let mut app = App::new();
    let title = "bevy_burn_autogaze".to_string();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: title.clone(),
        present_mode: bevy::window::PresentMode::AutoVsync,
        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(1280, 720),
        title,
        present_mode: bevy::window::PresentMode::AutoVsync,
        ..default()
    });

    app.insert_resource(config.clone());
    app.insert_resource(ClearColor(Color::BLACK));
    app.insert_resource(AutoGazeTexture::default());
    app.insert_resource(FrameQueue::default());
    app.insert_resource(PreparedTensorQueue::default());
    app.insert_resource(BevyVisualizationState::new(
        config.visualization_mode,
        config.keyframe_duration,
    ));
    app.insert_resource(GazeRatioStats::default());
    app.insert_resource(PsnrStats::default());
    app.insert_resource(InferenceTimingStats::default());
    app.insert_resource(InferenceSequencer::default());
    app.insert_resource(AutoGazeModelState {
        config: config.clone(),
        pipeline: None,
        load_task: None,
    });
    app.insert_resource(load_static_frame(config.image_path.as_deref(), &config));

    app.add_plugins(
        DefaultPlugins
            .set(ImagePlugin::default_nearest())
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                    features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    ..Default::default()
                })),
                ..Default::default()
            })
            .set(WindowPlugin {
                primary_window,
                ..default()
            }),
    );
    app.add_plugins(BevyBurnBridgePlugin::<AutoGazeBevyBackend>::default());

    if config.press_esc_to_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.register_diagnostic(Diagnostic::new(INFERENCE_FPS));
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    if config.show_gaze_ratio {
        app.add_systems(Startup, gaze_ratio_display_setup);
        app.add_systems(Update, gaze_ratio_update_system);
    }

    if config.show_psnr {
        app.add_systems(Startup, psnr_display_setup);
        app.add_systems(Update, psnr_update_system);
    }

    app.add_systems(
        Update,
        (
            setup_ui,
            begin_model_load,
            finish_model_load,
            handle_tasks,
            preview_frames,
            process_frames,
            maybe_emit_perf_summary,
            fit_visualization_node,
        )
            .chain(),
    );

    app
}

pub fn run_app(config: BevyBurnAutoGazeConfig) {
    viewer_app(config).run();

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(sender) = platform::camera::APP_RUN_SENDER.get() {
        let _ = sender.send(());
    }
}

fn setup_ui(
    mut commands: Commands,
    mut texture: ResMut<AutoGazeTexture>,
    mut images: ResMut<Assets<Image>>,
    burn_device: Option<Res<BurnDevice>>,
) {
    if texture.entity.is_some() {
        return;
    }
    if burn_device
        .as_ref()
        .and_then(|device| device.device())
        .is_none()
    {
        return;
    }

    texture.image = images.add(visualization_image(
        texture.width.max(1),
        texture.height.max(1),
        vec![0; texture.width.max(1) as usize * texture.height.max(1) as usize * 4],
    ));

    let mut image_entity = None;
    commands
        .spawn(Node {
            display: Display::Grid,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_items: JustifyItems::Center,
            grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
            ..default()
        })
        .with_children(|builder| {
            let entity = builder
                .spawn((
                    ImageNode::new(texture.image.clone()).with_mode(NodeImageMode::Stretch),
                    Node {
                        width: Val::Px(texture.width.max(1) as f32),
                        height: Val::Px(texture.height.max(1) as f32),
                        ..default()
                    },
                ))
                .id();
            image_entity = Some(entity);

            builder
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        top: Val::Px(10.0),
                        left: Val::Px(0.0),
                        right: Val::Px(0.0),
                        display: Display::Grid,
                        grid_template_columns: RepeatedGridTrack::flex(3, 1.0),
                        ..default()
                    },
                    ZIndex(2),
                ))
                .with_children(|labels| {
                    for label in ["Input", "Mask", "Output"] {
                        labels.spawn((
                            Text(label.to_string()),
                            TextFont {
                                font_size: bevy::text::FontSize::Px(24.0),
                                ..default()
                            },
                            TextColor(Color::WHITE),
                            Node {
                                justify_self: JustifySelf::Center,
                                ..default()
                            },
                        ));
                    }
                });
        });

    texture.entity = image_entity;
    commands.spawn(Camera2d);
}

fn fit_visualization_node(
    texture: Res<AutoGazeTexture>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut nodes: Query<&mut Node>,
) {
    let Some(entity) = texture.entity else {
        return;
    };
    let Some(window) = windows.iter().next() else {
        return;
    };
    let Ok(mut node) = nodes.get_mut(entity) else {
        return;
    };

    let source_width = texture.width.max(1) as f32;
    let source_height = texture.height.max(1) as f32;
    let available_width = window.resolution.width().max(1.0);
    let available_height = window.resolution.height().max(1.0);
    let source_aspect = source_width / source_height;
    let window_aspect = available_width / available_height;
    let (display_width, display_height) = if window_aspect > source_aspect {
        let height = available_height;
        (height * source_aspect, height)
    } else {
        let width = available_width;
        (width, width / source_aspect)
    };

    node.width = Val::Px(display_width.max(1.0));
    node.height = Val::Px(display_height.max(1.0));
}

fn begin_model_load(mut state: ResMut<AutoGazeModelState>, burn_device: Option<Res<BurnDevice>>) {
    if !state.config.load_model {
        return;
    }
    if state.pipeline.is_some() || state.load_task.is_some() {
        return;
    }
    let Some(device) = burn_device
        .as_ref()
        .and_then(|device| device.device())
        .cloned()
    else {
        return;
    };

    log("loading AutoGaze model...");
    state.load_task = Some(spawn_model_load_task(state.config.clone(), device));
}

fn finish_model_load(
    mut state: ResMut<AutoGazeModelState>,
    mut visualization_state: ResMut<BevyVisualizationState>,
) {
    let Some(task) = state.load_task.as_mut() else {
        return;
    };

    if let Some(result) = block_on(future::poll_once(task)) {
        match result {
            Ok(pipeline) => {
                log("AutoGaze model ready");
                state.pipeline = Some(Arc::new(Mutex::new(pipeline)));
                visualization_state.reset();
            }
            Err(err) => {
                log(&format!("failed to load AutoGaze model: {err}"));
            }
        }
        state.load_task = None;
    }
}

fn process_frames(
    mut commands: Commands,
    model: Res<AutoGazeModelState>,
    texture: Res<AutoGazeTexture>,
    burn_device: Option<Res<BurnDevice>>,
    mut frame_input: FrameInputParams,
    active_tasks: Query<&ProcessAutoGaze>,
    mut logged_first_inference: Local<bool>,
) {
    let Some(pipeline) = model.pipeline.as_ref() else {
        return;
    };
    if texture.entity.is_none() {
        return;
    }
    let Some(device) = burn_device
        .as_ref()
        .and_then(|device| device.device())
        .cloned()
    else {
        return;
    };
    if active_tasks.iter().count() >= MAX_IN_FLIGHT_TASKS {
        return;
    }
    if frame_input
        .config
        .perf_summary_frames
        .is_some_and(|target| frame_input.timing_stats.processed_frames() >= target)
    {
        return;
    }

    let source_start = timestamp_now();
    let frame = if let Some(frame) = frame_input.static_frame.0.as_ref() {
        Some((Arc::clone(frame), 0.0))
    } else {
        receive_frame().map(|frame| {
            let prepare_start = timestamp_now();
            let frame = prepare_frame_for_inference(frame, &frame_input.config);
            (Arc::new(frame), elapsed_ms(prepare_start))
        })
    };
    let source_ms = elapsed_ms(source_start);

    let Some((frame, prepare_ms)) = frame else {
        return;
    };
    frame_input
        .frame_queue
        .push(frame, frame_input.config.frames_per_clip);
    let mut clip = match frame_input
        .frame_queue
        .build_clip(frame_input.config.frames_per_clip)
    {
        Ok(Some(clip)) => clip,
        Ok(None) => return,
        Err(err) => {
            log(&format!("failed to pack AutoGaze clip: {err}"));
            return;
        }
    };
    clip.source_ms = source_ms;
    clip.prepare_ms = prepare_ms;

    let task_entity = commands.spawn_empty().id();
    let pipeline = pipeline.clone();
    let mode = frame_input.config.mode.inference_mode();
    let top_k = frame_input.config.top_k.max(1);
    let log_pipeline_timing = frame_input.config.log_pipeline_timing;
    let visualization_options = VisualizationOptions::new(
        frame_input.config.mask_cell_scale,
        frame_input.config.blend_alpha,
        frame_input.config.show_psnr,
    );
    frame_input.visualization_state.configure(
        frame_input.config.visualization_mode,
        frame_input.config.keyframe_duration,
    );
    let visualization_state = frame_input.visualization_state.clone();
    if !*logged_first_inference {
        log("AutoGaze inference started; the first native run may spend time tuning GPU kernels");
        *logged_first_inference = true;
    }
    let sequence = frame_input.inference_sequencer.reserve();
    let previous_tensors = frame_input.prepared_tensor_queue.previous_frames(
        clip.width,
        clip.height,
        clip.clip_len.saturating_sub(1),
    );
    let frame_input_clip_len = frame_input.config.frames_per_clip;

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let job = AutoGazeRunContext {
            clip: &clip,
            sequence,
            previous_tensors,
            top_k,
            mode,
            visualization_options,
            visualization_state,
            device,
        };

        #[cfg(target_arch = "wasm32")]
        let result = run_autogaze_visualization(pipeline, job).await;

        #[cfg(not(target_arch = "wasm32"))]
        let result = run_autogaze_visualization(pipeline, job);
        let clip_rgba = clip.rgba;

        let mut queue = CommandQueue::default();
        queue.push(move |world: &mut World| {
            if let Some(mut frame_queue) = world.get_resource_mut::<FrameQueue>() {
                frame_queue.recycle_clip_buffer(clip_rgba);
            }
            if let Some(mut sequencer) = world.get_resource_mut::<InferenceSequencer>()
                && !sequencer.accept(sequence)
            {
                if let Ok(mut tracker) = world.get_entity_mut(task_entity) {
                    tracker.remove::<ProcessAutoGaze>();
                    tracker.despawn();
                }
                return;
            }

            match result {
                Ok((visualization, visualization_state)) => {
                    let Visualization {
                        width,
                        height,
                        rgba,
                        prepared_frame,
                        gaze_update_ratio,
                        psnr_db,
                        mut timing,
                        ..
                    } = visualization;
                    let display_start = timestamp_now();
                    apply_visualization_to_world(world, width, height, rgba);
                    if let Some(ref mut timing) = timing {
                        timing.display_ms = elapsed_ms(display_start);
                        timing.total_ms += timing.display_ms;
                    }

                    if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
                        texture.width = width;
                        texture.height = height;
                    }

                    if let Some(prepared_frame) = prepared_frame
                        && let Some(mut tensor_queue) =
                            world.get_resource_mut::<PreparedTensorQueue>()
                    {
                        tensor_queue.push(prepared_frame, frame_input_clip_len);
                    }

                    if let Some(mut state) = world.get_resource_mut::<BevyVisualizationState>() {
                        *state = visualization_state;
                    }

                    if let Some(mut stats) = world.get_resource_mut::<GazeRatioStats>() {
                        stats.record(gaze_update_ratio);
                    }

                    if let Some(psnr_db) = psnr_db
                        && let Some(mut stats) = world.get_resource_mut::<PsnrStats>()
                    {
                        stats.record(psnr_db);
                    }

                    if let Some(timing) = timing
                        && let Some(mut stats) = world.get_resource_mut::<InferenceTimingStats>()
                    {
                        stats.record(timing, log_pipeline_timing);
                    }
                }
                Err(err) => {
                    log(&format!("AutoGaze inference failed: {err}"));
                }
            }

            if let Ok(mut tracker) = world.get_entity_mut(task_entity) {
                tracker.remove::<ProcessAutoGaze>();
                tracker.despawn();
            }
        });

        queue
    });

    commands.entity(task_entity).insert(ProcessAutoGaze(task));
}

fn preview_frames(
    model: Res<AutoGazeModelState>,
    mut texture: ResMut<AutoGazeTexture>,
    mut frame_input: FrameInputParams,
    active_tasks: Query<&ProcessAutoGaze>,
    mut images: ResMut<Assets<Image>>,
) {
    let model_ready = model.pipeline.is_some();
    let inference_busy = active_tasks.iter().count() >= MAX_IN_FLIGHT_TASKS;
    if model_ready && !inference_busy {
        return;
    }

    if texture.entity.is_none() {
        return;
    }

    let frame = if let Some(frame) = frame_input.static_frame.0.as_ref() {
        Some(Arc::clone(frame))
    } else {
        receive_frame()
            .map(|frame| Arc::new(prepare_frame_for_inference(frame, &frame_input.config)))
    };

    let Some(frame) = frame else {
        return;
    };

    frame_input
        .frame_queue
        .push(frame, frame_input.config.frames_per_clip);
    let Some(frame) = frame_input.frame_queue.latest() else {
        return;
    };

    if !should_draw_live_preview(model_ready, inference_busy) {
        return;
    }

    let visualization = match live_preview_visualization(frame, frame_input.config.show_psnr) {
        Ok(visualization) => visualization,
        Err(err) => {
            log(&format!("failed to draw AutoGaze preview: {err}"));
            return;
        }
    };
    frame_input
        .gaze_ratio_stats
        .record(visualization.gaze_update_ratio);
    if let Some(psnr_db) = visualization.psnr_db {
        frame_input.psnr_stats.record(psnr_db);
    }
    apply_visualization_to_texture(visualization, &mut texture, &mut images);
}

fn handle_tasks(
    mut commands: Commands,
    mut diagnostics: Diagnostics,
    mut last_frame: Local<Option<Timestamp>>,
    mut active_tasks: Query<&mut ProcessAutoGaze>,
) {
    for mut task in &mut active_tasks {
        if let Some(mut queue) = block_on(future::poll_once(&mut task.0)) {
            let now = timestamp_now();
            if let Some(last_frame) = *last_frame {
                let delta_seconds = elapsed_between_ms(last_frame, now) / 1000.0;
                if delta_seconds.is_finite() && delta_seconds > 0.0 {
                    diagnostics.add_measurement(&INFERENCE_FPS, || 1.0 / delta_seconds);
                }
            }
            *last_frame = Some(now);
            commands.append(&mut queue);
        }
    }
}

fn spawn_model_load_task(
    config: BevyBurnAutoGazeConfig,
    device: AutoGazeBevyDevice,
) -> Task<Result<AutoGazePipeline<AutoGazeBevyBackend>, String>> {
    AsyncComputeTaskPool::get().spawn(async move { load_model(config, &device).await })
}

#[cfg(not(target_arch = "wasm32"))]
async fn load_model(
    config: BevyBurnAutoGazeConfig,
    device: &AutoGazeBevyDevice,
) -> Result<AutoGazePipeline<AutoGazeBevyBackend>, String> {
    let mut pipeline = AutoGazePipeline::from_hf_dir(&config.model_dir, device)
        .map_err(|err| format!("{err:#}"))?;
    apply_max_gaze_tokens_config(&mut pipeline, &config);
    apply_task_loss_requirement_config(&mut pipeline, &config);
    apply_tile_batch_config(&mut pipeline, &config);
    Ok(pipeline)
}

#[cfg(target_arch = "wasm32")]
async fn load_model(
    config: BevyBurnAutoGazeConfig,
    device: &AutoGazeBevyDevice,
) -> Result<AutoGazePipeline<AutoGazeBevyBackend>, String> {
    let config_json = fetch_text(&config.config_url).await?;
    let model_config: AutoGazeConfig =
        serde_json::from_str(&config_json).map_err(|err| format!("{err}"))?;
    let weights = fetch_bytes(&config.weights_url).await?;
    let model = NativeAutoGazeModel::<AutoGazeBevyBackend>::from_config_and_safetensors_bytes(
        &model_config,
        weights,
        device,
        AutoGazeLoadOptions::strict(),
    )
    .map_err(|err| format!("{err:#}"))?;
    let mut pipeline = AutoGazePipeline::new(model);
    apply_max_gaze_tokens_config(&mut pipeline, &config);
    apply_task_loss_requirement_config(&mut pipeline, &config);
    apply_tile_batch_config(&mut pipeline, &config);
    Ok(pipeline)
}

fn apply_max_gaze_tokens_config<B: burn::tensor::backend::Backend>(
    pipeline: &mut AutoGazePipeline<B>,
    config: &BevyBurnAutoGazeConfig,
) {
    if config.max_gaze_tokens_each_frame > 0 {
        pipeline.set_max_gaze_tokens_each_frame(config.max_gaze_tokens_each_frame);
    }
}

fn apply_task_loss_requirement_config<B: burn::tensor::backend::Backend>(
    pipeline: &mut AutoGazePipeline<B>,
    config: &BevyBurnAutoGazeConfig,
) {
    if config.disable_task_loss_requirement {
        pipeline.set_task_loss_requirement(None);
    } else if let Some(task_loss_requirement) = config.task_loss_requirement {
        pipeline.set_task_loss_requirement(Some(task_loss_requirement));
    }
}

fn apply_tile_batch_config<B: burn::tensor::backend::Backend>(
    pipeline: &mut AutoGazePipeline<B>,
    config: &BevyBurnAutoGazeConfig,
) {
    pipeline.set_tile_batch_size(config.tile_batch_size.max(1));
}

fn prepare_clip_video_tensor(
    clip: &FrameClip,
    previous_tensors: Vec<Tensor<AutoGazeBevyBackend, 5>>,
    device: &AutoGazeBevyDevice,
) -> Result<(Tensor<AutoGazeBevyBackend, 5>, PreparedTensorFrame, f64), String> {
    let input_start = timestamp_now();
    let current_shape = AutoGazeRgbaClipShape::new(1, clip.height, clip.width);
    let current_tensor =
        rgba_clip_to_tensor::<AutoGazeBevyBackend>(clip.last_frame_rgba()?, current_shape, device)
            .map_err(|err| format!("{err:#}"))?;
    let prepared_frame = PreparedTensorFrame {
        width: clip.width,
        height: clip.height,
        tensor: current_tensor.clone(),
    };

    let video = if previous_tensors.len() + 1 >= clip.clip_len {
        let mut frames = previous_tensors;
        frames.push(current_tensor);
        if frames.len() == 1 {
            frames
                .pop()
                .ok_or_else(|| "AutoGaze prepared frame queue was empty".to_string())?
        } else {
            Tensor::cat(frames, 1)
        }
    } else {
        rgba_clip_to_tensor::<AutoGazeBevyBackend>(&clip.rgba, clip.shape(), device)
            .map_err(|err| format!("{err:#}"))?
    };

    Ok((video, prepared_frame, elapsed_ms(input_start)))
}

struct AutoGazeRunContext<'a> {
    clip: &'a FrameClip,
    sequence: u64,
    previous_tensors: Vec<Tensor<AutoGazeBevyBackend, 5>>,
    top_k: usize,
    mode: AutoGazeInferenceMode,
    visualization_options: VisualizationOptions,
    visualization_state: BevyVisualizationState,
    device: AutoGazeBevyDevice,
}

#[cfg(not(target_arch = "wasm32"))]
fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    context: AutoGazeRunContext<'_>,
) -> Result<(Visualization, BevyVisualizationState), String> {
    let AutoGazeRunContext {
        clip,
        sequence,
        previous_tensors,
        top_k,
        mode,
        visualization_options,
        mut visualization_state,
        device,
    } = context;
    let total_start = timestamp_now();
    let width = clip.width;
    let height = clip.height;
    let (video, prepared_frame, input_ms) =
        prepare_clip_video_tensor(clip, previous_tensors, &device)?;
    let (traces, model_ms) = {
        let pipeline = pipeline
            .lock()
            .map_err(|_| "AutoGaze model lock was poisoned".to_string())?;
        let model_start = timestamp_now();
        let traces = pipeline.trace_video_with_mode(video, top_k, mode);
        (traces, elapsed_ms(model_start))
    };
    let trace_ms = input_ms + model_ms;
    let frame_index = clip.clip_len.saturating_sub(1);
    let points = traces
        .first()
        .and_then(|trace| trace.frames.get(frame_index))
        .map(|set| set.points.clone())
        .unwrap_or_default();
    let visualize_start = timestamp_now();
    let mut visualization = visualize_rgba_bytes(
        clip.last_frame_rgba()?,
        width,
        height,
        &points,
        visualization_options,
        &mut visualization_state,
    )?;
    visualization.prepared_frame = Some(prepared_frame);
    visualization.timing = Some(InferenceTiming {
        sequence,
        clip_frames: clip.clip_len,
        width,
        height,
        source_ms: clip.source_ms,
        prepare_ms: clip.prepare_ms,
        pack_ms: clip.pack_ms,
        input_ms,
        model_ms,
        trace_ms,
        sync_ms: 0.0,
        visualize_cpu_ms: visualization.visualize_cpu_ms,
        tensor_ms: visualization.tensor_ms,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start) + clip.source_ms + clip.prepare_ms + clip.pack_ms,
        output_rgba_bytes: visualization.output_rgba_bytes,
        output_tensor_bytes: visualization.output_tensor_bytes,
    });
    Ok((visualization, visualization_state))
}

#[cfg(target_arch = "wasm32")]
async fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    context: AutoGazeRunContext<'_>,
) -> Result<(Visualization, BevyVisualizationState), String> {
    let AutoGazeRunContext {
        clip,
        sequence,
        previous_tensors,
        top_k,
        mode,
        visualization_options,
        mut visualization_state,
        device,
    } = context;
    let total_start = timestamp_now();
    let width = clip.width;
    let height = clip.height;
    let (video, prepared_frame, input_ms) =
        prepare_clip_video_tensor(clip, previous_tensors, &device)?;
    let (traces, model_ms) = {
        let pipeline = pipeline
            .lock()
            .map_err(|_| "AutoGaze model lock was poisoned".to_string())?
            .clone();
        let model_start = timestamp_now();
        let traces = pipeline
            .trace_video_with_mode_async(video, top_k, mode)
            .await
            .map_err(|err| {
                format!("failed to read AutoGaze tensor data asynchronously: {err:?}")
            })?;
        (traces, elapsed_ms(model_start))
    };
    let trace_ms = input_ms + model_ms;

    let frame_index = clip.clip_len.saturating_sub(1);
    let points = traces
        .first()
        .and_then(|trace| trace.frames.get(frame_index))
        .map(|set| set.points.clone())
        .unwrap_or_default();
    let visualize_start = timestamp_now();
    let mut visualization = visualize_rgba_bytes(
        clip.last_frame_rgba()?,
        width,
        height,
        &points,
        visualization_options,
        &mut visualization_state,
    )?;
    visualization.prepared_frame = Some(prepared_frame);
    visualization.timing = Some(InferenceTiming {
        sequence,
        clip_frames: clip.clip_len,
        width,
        height,
        source_ms: clip.source_ms,
        prepare_ms: clip.prepare_ms,
        pack_ms: clip.pack_ms,
        input_ms,
        model_ms,
        trace_ms,
        sync_ms: 0.0,
        visualize_cpu_ms: visualization.visualize_cpu_ms,
        tensor_ms: visualization.tensor_ms,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start) + clip.source_ms + clip.prepare_ms + clip.pack_ms,
        output_rgba_bytes: visualization.output_rgba_bytes,
        output_tensor_bytes: visualization.output_tensor_bytes,
    });
    Ok((visualization, visualization_state))
}

#[derive(Clone, Copy)]
struct VisualizationOptions {
    cell_scale: f32,
    blend_alpha: f32,
    calculate_psnr: bool,
}

impl VisualizationOptions {
    fn new(cell_scale: f32, blend_alpha: f32, calculate_psnr: bool) -> Self {
        Self {
            cell_scale,
            blend_alpha,
            calculate_psnr,
        }
    }
}

struct Visualization {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    prepared_frame: Option<PreparedTensorFrame>,
    gaze_update_ratio: f64,
    psnr_db: Option<f64>,
    visualize_cpu_ms: f64,
    tensor_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    timing: Option<InferenceTiming>,
}

fn visualize_points(
    rgba: &RgbaImage,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
) -> Result<Visualization, String> {
    visualize_rgba_bytes(
        rgba.as_raw(),
        rgba.width() as usize,
        rgba.height() as usize,
        points,
        options,
        visualization_state,
    )
}

fn visualize_rgba_bytes(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
) -> Result<Visualization, String> {
    let width = width.max(1);
    let height = height.max(1);
    let visualize_cpu_start = timestamp_now();
    let visualization = visualization_state
        .0
        .visualize_rgba(
            rgba,
            width,
            height,
            points,
            options.cell_scale,
            options.blend_alpha,
        )
        .map_err(|err| format!("{err:#}"))?;
    let psnr_db = options
        .calculate_psnr
        .then(|| {
            visualization
                .output_psnr_db(rgba)
                .map_err(|err| format!("{err:#}"))
        })
        .transpose()?;
    let visualize_cpu_ms = elapsed_ms(visualize_cpu_start);
    let gaze_update_ratio = visualization.update_ratio();
    let output_rgba_bytes = visualization.side_by_side_rgba.len();
    Ok(Visualization {
        width: visualization.side_by_side_width as u32,
        height: visualization.height as u32,
        rgba: visualization.side_by_side_rgba,
        prepared_frame: None,
        gaze_update_ratio,
        psnr_db,
        visualize_cpu_ms,
        tensor_ms: 0.0,
        output_rgba_bytes,
        output_tensor_bytes: 0,
        timing: None,
    })
}

fn live_preview_visualization(
    rgba: &RgbaImage,
    calculate_psnr: bool,
) -> Result<Visualization, String> {
    let mut state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 1);
    let mut visualization = visualize_points(
        rgba,
        &[],
        VisualizationOptions::new(1.0, 0.0, false),
        &mut state,
    )?;
    visualization.gaze_update_ratio = 0.0;
    visualization.psnr_db = calculate_psnr.then_some(f64::INFINITY);
    Ok(visualization)
}

fn should_draw_live_preview(model_ready: bool, _inference_busy: bool) -> bool {
    !model_ready
}

fn apply_visualization_to_texture(
    visualization: Visualization,
    texture: &mut AutoGazeTexture,
    images: &mut Assets<Image>,
) {
    let width = visualization.width;
    let height = visualization.height;
    set_visualization_image(&texture.image, width, height, visualization.rgba, images);
    texture.width = width;
    texture.height = height;
}

fn apply_visualization_to_world(world: &mut World, width: u32, height: u32, rgba: Vec<u8>) {
    let Some(image_handle) = world
        .get_resource::<AutoGazeTexture>()
        .map(|texture| texture.image.clone())
    else {
        return;
    };

    if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
        set_visualization_image(&image_handle, width, height, rgba, &mut images);
    }
}

fn set_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    images: &mut Assets<Image>,
) {
    if let Some(mut image) = images.get_mut(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb
        && image
            .texture_descriptor
            .usage
            .contains(TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING)
    {
        image.data = Some(rgba);
        return;
    }

    let _ = images.insert(handle.id(), visualization_image(width, height, rgba));
}

fn visualization_image(width: u32, height: u32, mut rgba: Vec<u8>) -> Image {
    let width = width.max(1);
    let height = height.max(1);
    let expected_len = width as usize * height as usize * 4;
    if rgba.len() != expected_len {
        rgba.resize(expected_len, 0);
    }

    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

#[cfg(not(target_arch = "wasm32"))]
fn timestamp_now() -> Timestamp {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn timestamp_now() -> Timestamp {
    Timestamp(js_sys::Date::now())
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_between_ms(start: Timestamp, end: Timestamp) -> f64 {
    end.duration_since(start).as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
fn elapsed_between_ms(start: Timestamp, end: Timestamp) -> f64 {
    (end.0 - start.0).max(0.0)
}

fn elapsed_ms(start: Timestamp) -> f64 {
    elapsed_between_ms(start, timestamp_now())
}

fn receive_frame() -> Option<RgbaImage> {
    platform::camera::receive_image()
}

fn load_static_frame(path: Option<&Path>, config: &BevyBurnAutoGazeConfig) -> StaticFrame {
    let frame = path.map(|path| {
        let frame = image::open(path)
            .unwrap_or_else(|err| panic!("failed to load image `{}`: {err}", path.display()))
            .to_rgba8();
        Arc::new(prepare_frame_for_inference(frame, config))
    });
    StaticFrame(frame)
}

fn prepare_frame_for_inference(frame: RgbaImage, config: &BevyBurnAutoGazeConfig) -> RgbaImage {
    let (width, height) = frame.dimensions();
    let (target_width, target_height) = configured_inference_dimensions(
        width,
        height,
        config.inference_width,
        config.inference_height,
    );
    if target_width == width && target_height == height {
        return frame;
    }
    image::imageops::resize(&frame, target_width, target_height, FilterType::Triangle)
}

fn configured_inference_dimensions(
    width: u32,
    height: u32,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
) -> (u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    match (inference_width, inference_height) {
        (Some(target_width), Some(target_height)) => (target_width.max(1), target_height.max(1)),
        (Some(target_width), None) => {
            let target_width = target_width.max(1);
            let target_height =
                ((height as f64 * target_width as f64 / width as f64).round() as u32).max(1);
            (target_width, target_height)
        }
        (None, Some(target_height)) => {
            let target_height = target_height.max(1);
            let target_width =
                ((width as f64 * target_height as f64 / height as f64).round() as u32).max(1);
            (target_width, target_height)
        }
        (None, None) => (width, height),
    }
}

fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

fn maybe_emit_perf_summary(
    config: Res<BevyBurnAutoGazeConfig>,
    mut timing: ResMut<InferenceTimingStats>,
    mut exit: MessageWriter<AppExit>,
) {
    #[cfg(target_arch = "wasm32")]
    let _ = &mut exit;

    let Some(target_frames) = config.perf_summary_frames else {
        return;
    };
    if timing.emitted_summary || timing.processed_frames() < target_frames {
        return;
    }

    let summary = timing.summary_json(target_frames);
    log(&format!("AutoGaze perf summary: {summary}"));
    publish_wasm_perf_summary(&summary);
    timing.emitted_summary = true;

    #[cfg(not(target_arch = "wasm32"))]
    exit.write(AppExit::Success);
}

#[cfg(target_arch = "wasm32")]
fn publish_wasm_perf_summary(summary: &str) {
    use wasm_bindgen::JsValue;

    if let Some(window) = web_sys::window() {
        let value = js_sys::JSON::parse(summary).unwrap_or_else(|_| JsValue::from_str(summary));
        let _ = js_sys::Reflect::set(&window, &JsValue::from_str("__autogazePerfSummary"), &value);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_wasm_perf_summary(_summary: &str) {}

fn fps_display_setup(mut commands: Commands) {
    commands
        .spawn((
            Text("fps: ".to_string()),
            TextFont {
                font_size: bevy::text::FontSize::Px(28.0),
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(METRIC_OVERLAY_BOTTOM),
                left: Val::Px(12.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            FpsText,
            TextColor(Color::srgb(1.0, 0.84, 0.0)),
            TextFont {
                font_size: bevy::text::FontSize::Px(28.0),
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    timing: Res<InferenceTimingStats>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(timing) = timing.latest {
            **text = format!("{:.1}", timing.e2e_fps());
        } else if let Some(fps) = diagnostics.get(&INFERENCE_FPS)
            && let Some(value) = fps.smoothed()
        {
            **text = format!("{value:.1}");
        }
    }
}

fn gaze_ratio_display_setup(mut commands: Commands, config: Res<BevyBurnAutoGazeConfig>) {
    let bottom = metric_overlay_bottom(usize::from(config.show_fps));
    commands
        .spawn((
            Text("gaze: ".to_string()),
            TextFont {
                font_size: bevy::text::FontSize::Px(24.0),
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(bottom),
                left: Val::Px(12.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            GazeRatioText,
            TextColor(Color::srgb(0.55, 0.9, 1.0)),
            TextFont {
                font_size: bevy::text::FontSize::Px(24.0),
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct GazeRatioText;

fn gaze_ratio_update_system(
    stats: Res<GazeRatioStats>,
    mut query: Query<&mut TextSpan, With<GazeRatioText>>,
) {
    for mut text in &mut query {
        if stats.initialized {
            **text = format!(
                "{:.1}% ema {:.1}%",
                stats.current * 100.0,
                stats.ema * 100.0
            );
        } else {
            **text = "--.-% ema --.-%".to_string();
        }
    }
}

fn psnr_display_setup(mut commands: Commands, config: Res<BevyBurnAutoGazeConfig>) {
    let row = usize::from(config.show_fps) + usize::from(config.show_gaze_ratio);
    commands
        .spawn((
            Text("psnr: ".to_string()),
            TextFont {
                font_size: bevy::text::FontSize::Px(24.0),
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(metric_overlay_bottom(row)),
                left: Val::Px(12.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            PsnrText,
            TextColor(Color::srgb(0.7, 1.0, 0.6)),
            TextFont {
                font_size: bevy::text::FontSize::Px(24.0),
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct PsnrText;

fn psnr_update_system(stats: Res<PsnrStats>, mut query: Query<&mut TextSpan, With<PsnrText>>) {
    for mut text in &mut query {
        if stats.initialized {
            **text = format!(
                "{} dB ema {} dB",
                format_psnr_db(stats.current),
                format_psnr_db(stats.ema)
            );
        } else {
            **text = "--.- dB ema --.- dB".to_string();
        }
    }
}

fn metric_overlay_bottom(row: usize) -> f32 {
    METRIC_OVERLAY_BOTTOM + row as f32 * METRIC_OVERLAY_STEP
}

fn ema_metric(previous: f64, current: f64, alpha: f64) -> f64 {
    if previous.is_finite() && current.is_finite() {
        previous * (1.0 - alpha) + current * alpha
    } else {
        current
    }
}

fn format_psnr_db(value: f64) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        "inf".to_string()
    } else if value.is_finite() {
        format!("{value:.1}")
    } else {
        "--.-".to_string()
    }
}

pub fn log(message: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&message.into());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{message}");
    }
}

#[cfg(target_arch = "wasm32")]
async fn fetch_text(url: &str) -> Result<String, String> {
    let value = fetch_array_buffer(url).await?;
    String::from_utf8(value).map_err(|err| format!("{err}"))
}

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    fetch_array_buffer(url).await
}

#[cfg(target_arch = "wasm32")]
async fn fetch_array_buffer(url: &str) -> Result<Vec<u8>, String> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response, window};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(url, &opts).map_err(|err| format!("{err:?}"))?;

    let window = window().ok_or_else(|| "missing browser window".to_string())?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|err| format!("{err:?}"))?;
    let response: Response = response
        .dyn_into()
        .map_err(|_| "invalid fetch response".to_string())?;
    if !response.ok() {
        return Err(format!("GET {url} failed: {}", response.status()));
    }

    let buffer = JsFuture::from(response.array_buffer().map_err(|err| format!("{err:?}"))?)
        .await
        .map_err(|err| format!("{err:?}"))?;
    let bytes = Uint8Array::new(&buffer);
    let mut data = vec![0; bytes.length() as usize];
    bytes.copy_to(&mut data);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_url_query_to_viewer_config() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string(
            "?mode=full-res&top_k=2&frames-per-clip=3&inference-width=1920&inference-height=1080&show-fps=false&show-psnr=false&task-loss-requirement=0.65&tile-batch-size=4&config-url=%2Fconfig.json&weights-url=%2Fmodel.safetensors&load-model=false&mask-cell-scale=2.5&blend-alpha=0.5&visualization-mode=interframe&keyframe-duration=7&perf-summary-frames=5",
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.frames_per_clip, 3);
        assert_eq!(config.inference_width, Some(1920));
        assert_eq!(config.inference_height, Some(1080));
        assert!(!config.show_fps);
        assert!(config.show_gaze_ratio);
        assert!(!config.show_psnr);
        assert_eq!(config.task_loss_requirement, Some(0.65));
        assert!(!config.disable_task_loss_requirement);
        assert_eq!(config.tile_batch_size, 4);
        assert_eq!(config.config_url, "/config.json");
        assert_eq!(config.weights_url, "/model.safetensors");
        assert!(!config.load_model);
        assert_eq!(config.mask_cell_scale, 2.5);
        assert_eq!(config.blend_alpha, 0.5);
        assert_eq!(
            config.visualization_mode,
            AutoGazeVisualizationMode::Interframe
        );
        assert_eq!(config.keyframe_duration, 7);
        assert_eq!(config.perf_summary_frames, Some(5));

        let errors = config.apply_query_string("?show-gaze-ratio=false");
        assert!(errors.is_empty(), "{errors:?}");
        assert!(!config.show_gaze_ratio);

        let errors = config.apply_query_string("?show-psnr=true");
        assert!(errors.is_empty(), "{errors:?}");
        assert!(config.show_psnr);

        let errors = config.apply_query_string("?task-loss-requirement=none");
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.task_loss_requirement, None);
        assert!(config.disable_task_loss_requirement);
    }

    #[test]
    fn bevy_mode_parser_accepts_documented_aliases() {
        assert_eq!(
            "realtime".parse::<BevyAutoGazeMode>().unwrap(),
            BevyAutoGazeMode::Resize224
        );
        assert_eq!(
            "resize-224".parse::<BevyAutoGazeMode>().unwrap(),
            BevyAutoGazeMode::Resize224
        );
        assert_eq!(
            "full-res".parse::<BevyAutoGazeMode>().unwrap(),
            BevyAutoGazeMode::Tile224
        );
        assert_eq!(
            "anyres".parse::<BevyAutoGazeMode>().unwrap(),
            BevyAutoGazeMode::Tile224
        );

        let err = "bad-mode".parse::<BevyAutoGazeMode>().unwrap_err();
        assert!(err.contains("realtime"), "{err}");
        assert!(err.contains("full-res"), "{err}");
    }

    #[test]
    fn tiled_query_defaults_keep_model_generation_budget() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string("?mode=tiled&visualization-mode=interframe");

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, DEFAULT_TILED_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_MODEL_GENERATION_BUDGET
        );
        assert_eq!(config.tile_batch_size, DEFAULT_TILED_TILE_BATCH_SIZE);
        assert_eq!(config.frames_per_clip, DEFAULT_TILED_FRAMES_PER_CLIP);
        assert_eq!(config.inference_width, Some(DEFAULT_TILED_INFERENCE_WIDTH));
        assert_eq!(config.inference_height, None);
        assert_eq!(
            config.visualization_mode,
            AutoGazeVisualizationMode::Interframe
        );
    }

    #[test]
    fn tiled_query_defaults_preserve_explicit_performance_knobs() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string(
            "?mode=tiled&top-k=5&max-gaze-tokens-each-frame=7&tile-batch-size=4&width=1920",
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.top_k, 5);
        assert_eq!(config.max_gaze_tokens_each_frame, 7);
        assert_eq!(config.tile_batch_size, 4);
        assert_eq!(config.inference_width, Some(1920));
        assert_eq!(config.inference_height, None);
    }

    #[test]
    fn default_blend_alpha_keeps_output_mask_subtle() {
        let config = BevyBurnAutoGazeConfig::default();

        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
        assert!(
            config.blend_alpha <= 0.4,
            "default blend alpha should keep the output panel interpretable"
        );
    }

    #[test]
    fn inference_sequence_rejects_stale_results() {
        let mut sequencer = InferenceSequencer::default();
        let first = sequencer.reserve();
        let second = sequencer.reserve();

        assert!(sequencer.accept(second));
        assert!(!sequencer.accept(first));
        assert!(sequencer.accept(second + 1));
    }

    #[test]
    fn live_preview_only_draws_before_model_is_ready() {
        assert!(should_draw_live_preview(false, false));
        assert!(should_draw_live_preview(false, true));
        assert!(!should_draw_live_preview(true, false));
        assert!(!should_draw_live_preview(true, true));
    }

    #[test]
    fn frame_queue_packs_static_clip_buffer_and_reuses_it() {
        let mut queue = FrameQueue::default();
        let first = Arc::new(RgbaImage::from_pixel(2, 1, image::Rgba([1, 2, 3, 255])));
        let second = Arc::new(RgbaImage::from_pixel(2, 1, image::Rgba([4, 5, 6, 255])));
        let third = Arc::new(RgbaImage::from_pixel(2, 1, image::Rgba([7, 8, 9, 255])));

        queue.push(Arc::clone(&first), 2);
        assert_eq!(Arc::strong_count(&first), 2);
        assert!(queue.build_clip(2).unwrap().is_none());
        queue.push(Arc::clone(&second), 2);
        let clip = queue.build_clip(2).unwrap().unwrap();

        assert_eq!(clip.width, 2);
        assert_eq!(clip.height, 1);
        assert_eq!(clip.clip_len, 2);
        assert_eq!(&clip.rgba[..first.as_raw().len()], first.as_raw());
        assert_eq!(&clip.rgba[first.as_raw().len()..], second.as_raw());
        assert_eq!(clip.last_frame_rgba().unwrap(), second.as_raw());

        let capacity = clip.rgba.capacity();
        queue.recycle_clip_buffer(clip.rgba);
        assert_eq!(queue.spare_clip_buffers.len(), 1);

        queue.push(Arc::clone(&third), 2);
        assert_eq!(Arc::strong_count(&first), 1);
        let clip = queue.build_clip(2).unwrap().unwrap();
        assert_eq!(queue.spare_clip_buffers.len(), 0);
        assert_eq!(clip.rgba.capacity(), capacity);
        assert_eq!(&clip.rgba[..second.as_raw().len()], second.as_raw());
        assert_eq!(&clip.rgba[second.as_raw().len()..], third.as_raw());
    }

    #[test]
    fn rolling_frame_queue_reuses_recent_frames_and_resets_on_resize() {
        let mut queue = RollingFrameQueue::<u32>::default();

        queue.push(
            PreparedFrame {
                width: 4,
                height: 2,
                tensor: 1,
            },
            2,
        );
        queue.push(
            PreparedFrame {
                width: 4,
                height: 2,
                tensor: 2,
            },
            2,
        );
        queue.push(
            PreparedFrame {
                width: 4,
                height: 2,
                tensor: 3,
            },
            2,
        );

        assert_eq!(queue.previous_frames(4, 2, 2), vec![2, 3]);
        assert_eq!(queue.previous_frames(4, 2, 1), vec![3]);
        assert!(queue.previous_frames(8, 2, 1).is_empty());

        queue.push(
            PreparedFrame {
                width: 8,
                height: 2,
                tensor: 4,
            },
            2,
        );
        assert_eq!(queue.previous_frames(8, 2, 2), vec![4]);
    }

    #[test]
    fn inference_dimensions_preserve_aspect_when_one_axis_is_configured() {
        assert_eq!(
            configured_inference_dimensions(1280, 720, Some(1920), None),
            (1920, 1080)
        );
        assert_eq!(
            configured_inference_dimensions(1280, 720, None, Some(1080)),
            (1920, 1080)
        );
    }

    #[test]
    fn resizes_frame_to_configured_inference_resolution() {
        let frame = RgbaImage::from_pixel(4, 2, image::Rgba([10, 20, 30, 255]));
        let config = BevyBurnAutoGazeConfig {
            inference_width: Some(8),
            inference_height: Some(4),
            ..Default::default()
        };
        let resized = prepare_frame_for_inference(frame, &config);

        assert_eq!(resized.dimensions(), (8, 4));
    }

    #[test]
    fn bevy_visualization_mask_data_uses_crisp_multiscale_cell_bounds() {
        let point = FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 1.0);
        let mask = fixation_scale_mask_rgba(4, 4, &[point], 1.0);

        for y in 0..4 {
            for x in 0..4 {
                let src = (y * 4 + x) * 4;
                let expected = if x < 2 && y < 2 {
                    [255, 180, 0, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(&mask[src..src + 4], &expected, "mask {x},{y}");
            }
        }
    }
}
