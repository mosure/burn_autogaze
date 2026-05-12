use std::{fmt, path::PathBuf, str::FromStr};

use bevy::prelude::Resource;
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazeMaskVisualizationMode, AutoGazeRealtimePolicy,
    AutoGazeVisualizationMode, DEFAULT_BLEND_ALPHA, DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MODEL_GENERATION_BUDGET, DEFAULT_REALTIME_TOP_K,
    DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO, DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
    DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS, DEFAULT_TILED_FRAMES_PER_CLIP,
    DEFAULT_TILED_MAX_GAZE_TOKENS, DEFAULT_TILED_TILE_BATCH_SIZE, DEFAULT_TILED_TOP_K,
    should_use_streaming_cache, task_loss_requirement_from_l1_db,
};

pub const DEFAULT_NATIVE_MODEL_DIR: &str = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a";
pub const DEFAULT_CONFIG_URL: &str =
    "https://huggingface.co/nvidia/AutoGaze/resolve/main/config.json";
pub const DEFAULT_WEIGHTS_URL: &str =
    "https://huggingface.co/nvidia/AutoGaze/resolve/main/model.safetensors";
pub(crate) const MODEL_INPUT_SIZE: usize = 224;
pub const DEFAULT_REALTIME_INFERENCE_WIDTH: u32 = 640;
pub const DEFAULT_REALTIME_MAX_GAZE_TOKENS: usize = DEFAULT_MODEL_GENERATION_BUDGET;
pub const DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP: usize = 16;
pub const DEFAULT_BEVY_STREAMING_CACHE: bool = true;
pub const DEFAULT_BEVY_TILED_TOP_K: usize = DEFAULT_TILED_TOP_K;
pub const DEFAULT_BIRDS_INFERENCE_WIDTH: u32 = 1920;
pub const DEFAULT_BIRDS_INFERENCE_HEIGHT: u32 = 1080;
pub const DEFAULT_BIRDS_TOP_K: usize = DEFAULT_REALTIME_TOP_K;
pub const DEFAULT_BIRDS_MAX_GAZE_TOKENS: usize = DEFAULT_MODEL_GENERATION_BUDGET;
pub const DEFAULT_BIRDS_TILE_BATCH_SIZE: usize = 4;
pub const DEFAULT_BIRDS_FRAMES_PER_CLIP: usize = 16;
pub const DEFAULT_BIRDS_BLEND_ALPHA: f32 = 0.55;
pub const DEFAULT_BIRDS_KEYFRAME_DURATION: usize = 0;

/// Viewer config sentinel for using the AutoGaze model's configured inference budget.
///
/// The NVIDIA config uses a fixed inference gazing ratio of 0.75, which maps to
/// 198 tokens for its 265-token multi-scale vocabulary. This remains available
/// by passing `0`. The live Bevy realtime default also delegates to this model
/// budget so camera streams run in the same numerical regime as the upstream
/// resize-224 path.
pub const DEFAULT_TILED_INFERENCE_WIDTH: u32 = 1280;

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

impl FromStr for BevyAutoGazeMode {
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

impl fmt::Display for BevyAutoGazeMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub const DEFAULT_BEVY_MODE: BevyAutoGazeMode = BevyAutoGazeMode::Resize224;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BevyDisplayTransfer {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl BevyDisplayTransfer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

impl FromStr for BevyDisplayTransfer {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "adaptive" | "default" => Ok(Self::Auto),
            "gpu" | "device" | "burn-to-bevy" | "bevy-burn" | "interop" => Ok(Self::Gpu),
            "cpu" | "host" | "rgba" => Ok(Self::Cpu),
            other => Err(format!(
                "unsupported display transfer `{other}`; expected auto, gpu, or cpu"
            )),
        }
    }
}

impl fmt::Display for BevyDisplayTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub warmup_model: bool,
    pub image_path: Option<PathBuf>,
    pub mode: BevyAutoGazeMode,
    pub top_k: usize,
    pub max_gaze_tokens_each_frame: usize,
    pub tile_batch_size: usize,
    pub task_loss_requirement: Option<f32>,
    pub disable_task_loss_requirement: bool,
    pub frames_per_clip: usize,
    pub max_in_flight: usize,
    pub inference_width: Option<u32>,
    pub inference_height: Option<u32>,
    pub mask_cell_scale: f32,
    pub mask_visualization_mode: AutoGazeMaskVisualizationMode,
    pub blend_alpha: f32,
    pub visualization_mode: AutoGazeVisualizationMode,
    pub keyframe_duration: usize,
    pub display_transfer: BevyDisplayTransfer,
    pub tensor_sparse_update_max_rects: usize,
    pub tensor_sparse_update_max_ratio: f64,
    pub tensor_full_frame_update_min_ratio: f64,
    pub streaming_cache: bool,
    pub require_hardware_adapter: bool,
    pub log_pipeline_timing: bool,
    pub perf_summary_frames: Option<usize>,
    pub perf_summary_path: Option<PathBuf>,
}

impl Default for BevyBurnAutoGazeConfig {
    fn default() -> Self {
        let mode = DEFAULT_BEVY_MODE;
        let (inference_width, inference_height) = default_inference_dimensions(mode);
        Self {
            press_esc_to_close: true,
            show_fps: true,
            show_gaze_ratio: true,
            show_psnr: true,
            model_dir: PathBuf::from(DEFAULT_NATIVE_MODEL_DIR),
            config_url: DEFAULT_CONFIG_URL.to_string(),
            weights_url: DEFAULT_WEIGHTS_URL.to_string(),
            load_model: true,
            warmup_model: true,
            image_path: None,
            mode,
            top_k: default_top_k(mode),
            max_gaze_tokens_each_frame: default_max_gaze_tokens_each_frame(mode),
            tile_batch_size: default_tile_batch_size(mode),
            task_loss_requirement: None,
            disable_task_loss_requirement: false,
            frames_per_clip: default_frames_per_clip(mode),
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
            inference_width,
            inference_height,
            mask_cell_scale: 1.0,
            mask_visualization_mode: AutoGazeMaskVisualizationMode::Overlay,
            blend_alpha: DEFAULT_BLEND_ALPHA,
            visualization_mode: AutoGazeVisualizationMode::Interframe,
            keyframe_duration: DEFAULT_BIRDS_KEYFRAME_DURATION,
            display_transfer: BevyDisplayTransfer::Auto,
            tensor_sparse_update_max_rects: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            tensor_sparse_update_max_ratio: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
            tensor_full_frame_update_min_ratio: DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
            streaming_cache: DEFAULT_BEVY_STREAMING_CACHE,
            require_hardware_adapter: false,
            log_pipeline_timing: false,
            perf_summary_frames: None,
            perf_summary_path: None,
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
            "warmup-model" | "warm-model" | "model-warmup" => {
                self.warmup_model = parse_bool_option(&key, value)?;
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
            "task-loss-requirement-db"
            | "task-loss-db"
            | "task-psnr-db"
            | "task-psnr"
            | "task-psnr-requirement" => {
                self.task_loss_requirement = Some(task_loss_requirement_from_l1_db(f64::from(
                    parse_nonnegative_f32_option(&key, value)?,
                )));
                self.disable_task_loss_requirement = false;
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
            "max-in-flight" | "in-flight" | "max-inflight" => {
                self.max_in_flight = parse_usize_option(&key, value)?;
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
            "mask-visualization" | "mask-visualization-mode" | "mask-mode" | "mask-display" => {
                self.mask_visualization_mode = value.parse()?;
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
                self.keyframe_duration = parse_usize_option(&key, value)?;
                Ok(())
            }
            "display-transfer" | "transfer" | "texture-transfer" => {
                self.display_transfer = value.parse()?;
                Ok(())
            }
            "tensor-sparse-update-max-rects" | "sparse-update-max-rects" => {
                self.tensor_sparse_update_max_rects = parse_usize_option(&key, value)?;
                Ok(())
            }
            "tensor-sparse-update-max-ratio" | "sparse-update-max-ratio" => {
                self.tensor_sparse_update_max_ratio = parse_f64_option(&key, value)?;
                Ok(())
            }
            "tensor-full-frame-update-min-ratio"
            | "full-frame-update-min-ratio"
            | "dense-update-full-frame-min-ratio" => {
                self.tensor_full_frame_update_min_ratio = parse_f64_option(&key, value)?;
                Ok(())
            }
            "streaming-cache" | "kv-cache" | "stream-cache" => {
                self.streaming_cache = parse_bool_option(&key, value)?;
                Ok(())
            }
            "require-hardware-adapter" | "require-gpu" | "fail-on-cpu-adapter" => {
                self.require_hardware_adapter = parse_bool_option(&key, value)?;
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
            "perf-summary-path" | "perf-json" | "perf-output" => {
                self.perf_summary_path = (!value.trim().is_empty()).then(|| PathBuf::from(value));
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
        if saw_inference_width && !saw_inference_height {
            self.inference_height = None;
        } else if saw_inference_height && !saw_inference_width {
            self.inference_width = None;
        }
        self.sanitize();

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

    pub fn sanitize(&mut self) {
        if self.top_k == 0 {
            self.top_k = default_top_k(self.mode);
        }
        if self.tile_batch_size == 0 {
            self.tile_batch_size = default_tile_batch_size(self.mode);
        }
        if self.frames_per_clip == 0 {
            self.frames_per_clip = default_frames_per_clip(self.mode);
        }
        if self.max_in_flight == 0 {
            self.max_in_flight = DEFAULT_MAX_IN_FLIGHT;
        }
        // A value of 0 disables periodic interframe keyframes. The first frame
        // and dimension changes still prime/reset the interframe state.
        if self.inference_width == Some(0) || self.inference_height == Some(0) {
            let (width, height) = default_inference_dimensions(self.mode);
            if self.inference_width == Some(0) {
                self.inference_width = width;
            }
            if self.inference_height == Some(0) {
                self.inference_height = height;
            }
        }
        if !self.mask_cell_scale.is_finite() || self.mask_cell_scale <= 0.0 {
            self.mask_cell_scale = 1.0;
        }
        if !self.blend_alpha.is_finite() {
            self.blend_alpha = DEFAULT_BLEND_ALPHA;
        } else {
            self.blend_alpha = self.blend_alpha.clamp(0.0, 1.0);
        }
        if !self.tensor_sparse_update_max_ratio.is_finite()
            || self.tensor_sparse_update_max_ratio < 0.0
        {
            self.tensor_sparse_update_max_ratio = DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO;
        } else {
            self.tensor_sparse_update_max_ratio =
                self.tensor_sparse_update_max_ratio.clamp(0.0, 1.0);
        }
        if !self.tensor_full_frame_update_min_ratio.is_finite()
            || self.tensor_full_frame_update_min_ratio < 0.0
        {
            self.tensor_full_frame_update_min_ratio = DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO;
        } else {
            self.tensor_full_frame_update_min_ratio =
                self.tensor_full_frame_update_min_ratio.clamp(0.0, 1.0);
        }
        self.task_loss_requirement = self
            .task_loss_requirement
            .filter(|value| value.is_finite() && *value >= 0.0);
    }

    pub fn sanitized(mut self) -> Self {
        self.sanitize();
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_browser_query() -> Self {
        let mut config = Self::default();
        if let Some(window) = web_sys::window() {
            match window.location().search() {
                Ok(search) => {
                    for err in config.apply_query_string(&search) {
                        log_browser_config_message(&format!("ignoring invalid URL option: {err}"));
                    }
                }
                Err(err) => {
                    log_browser_config_message(&format!("failed to read URL query: {err:?}"));
                }
            }
        }
        config.sanitize();
        config
    }

    pub fn docs_birds() -> Self {
        Self {
            mode: BevyAutoGazeMode::Tile224,
            top_k: DEFAULT_BIRDS_TOP_K,
            max_gaze_tokens_each_frame: DEFAULT_BIRDS_MAX_GAZE_TOKENS,
            tile_batch_size: DEFAULT_BIRDS_TILE_BATCH_SIZE,
            frames_per_clip: DEFAULT_BIRDS_FRAMES_PER_CLIP,
            inference_width: Some(DEFAULT_BIRDS_INFERENCE_WIDTH),
            inference_height: Some(DEFAULT_BIRDS_INFERENCE_HEIGHT),
            mask_visualization_mode: AutoGazeMaskVisualizationMode::Overlay,
            blend_alpha: DEFAULT_BIRDS_BLEND_ALPHA,
            keyframe_duration: DEFAULT_BIRDS_KEYFRAME_DURATION,
            display_transfer: BevyDisplayTransfer::Auto,
            ..Self::default()
        }
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
        BevyAutoGazeMode::Tile224 => DEFAULT_BEVY_TILED_TOP_K,
    }
}

pub const fn default_max_gaze_tokens_each_frame(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_REALTIME_MAX_GAZE_TOKENS,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_MAX_GAZE_TOKENS,
    }
}

pub const fn default_tile_batch_size(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_TILED_TILE_BATCH_SIZE,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_TILE_BATCH_SIZE,
    }
}

pub const fn default_frames_per_clip(mode: BevyAutoGazeMode) -> usize {
    match mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_FRAMES_PER_CLIP,
    }
}

pub const fn default_inference_dimensions(mode: BevyAutoGazeMode) -> (Option<u32>, Option<u32>) {
    match mode {
        BevyAutoGazeMode::Resize224 => (Some(DEFAULT_REALTIME_INFERENCE_WIDTH), None),
        BevyAutoGazeMode::Tile224 => (Some(DEFAULT_TILED_INFERENCE_WIDTH), None),
    }
}

pub const fn realtime_policy_from_config(
    config: &BevyBurnAutoGazeConfig,
) -> AutoGazeRealtimePolicy {
    let max_in_flight = if should_use_streaming_cache(
        config.streaming_cache,
        config.frames_per_clip,
        config.mode.inference_mode(),
    ) {
        DEFAULT_MAX_IN_FLIGHT
    } else {
        config.max_in_flight
    };
    AutoGazeRealtimePolicy::new(max_in_flight)
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

fn parse_nonnegative_f32_option(key: &str, value: &str) -> Result<f32, String> {
    let parsed = parse_f32_option(key, value)?;
    if parsed.is_finite() && parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(format!("invalid non-negative f32 for `{key}`: `{value}`"))
    }
}

fn parse_f64_option(key: &str, value: &str) -> Result<f64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid f64 for `{key}`: `{value}`"))
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

#[cfg(target_arch = "wasm32")]
fn log_browser_config_message(message: &str) {
    web_sys::console::log_1(&message.into());
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use burn_autogaze::{
        AutoGazeInferenceMode, DEFAULT_BLEND_ALPHA, DEFAULT_MAX_IN_FLIGHT, DEFAULT_REALTIME_TOP_K,
        DEFAULT_TILED_FRAMES_PER_CLIP, DEFAULT_TILED_MAX_GAZE_TOKENS,
        DEFAULT_TILED_TILE_BATCH_SIZE, should_use_streaming_cache,
    };

    use super::*;

    #[test]
    fn applies_url_query_to_viewer_config() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string(
            "?mode=full-res&top_k=2&frames-per-clip=3&max-in-flight=2&inference-width=1920&inference-height=1080&show-fps=false&show-psnr=false&task-loss-requirement=0.65&tile-batch-size=4&config-url=%2Fconfig.json&weights-url=%2Fmodel.safetensors&load-model=false&mask-cell-scale=2.5&mask-visualization=overlay&blend-alpha=0.5&visualization-mode=interframe&keyframe-duration=7&display-transfer=cpu&tensor-sparse-update-max-rects=8&tensor-sparse-update-max-ratio=0.05&tensor-full-frame-update-min-ratio=0.45&streaming-cache=false&require-hardware-adapter=true&perf-summary-frames=5&perf-summary-path=target%2Fperf.json",
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.frames_per_clip, 3);
        assert_eq!(config.max_in_flight, 2);
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
        assert_eq!(
            config.mask_visualization_mode,
            AutoGazeMaskVisualizationMode::Overlay
        );
        assert_eq!(config.blend_alpha, 0.5);
        assert_eq!(
            config.visualization_mode,
            AutoGazeVisualizationMode::Interframe
        );
        assert_eq!(config.keyframe_duration, 7);
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Cpu);
        assert_eq!(config.tensor_sparse_update_max_rects, 8);
        assert_eq!(config.tensor_sparse_update_max_ratio, 0.05);
        assert_eq!(config.tensor_full_frame_update_min_ratio, 0.45);
        assert!(!config.streaming_cache);
        assert!(config.require_hardware_adapter);
        assert_eq!(config.perf_summary_frames, Some(5));
        assert_eq!(
            config.perf_summary_path,
            Some(PathBuf::from("target/perf.json"))
        );

        let errors = config.apply_query_string("?show-gaze-ratio=false");
        assert!(errors.is_empty(), "{errors:?}");
        assert!(!config.show_gaze_ratio);

        let errors = config.apply_query_string("?show-psnr=true");
        assert!(errors.is_empty(), "{errors:?}");
        assert!(config.show_psnr);

        let errors = config.apply_query_string("?warmup-model=false");
        assert!(errors.is_empty(), "{errors:?}");
        assert!(!config.warmup_model);

        let errors = config.apply_query_string("?task-loss-requirement-db=20");
        assert!(errors.is_empty(), "{errors:?}");
        assert!((config.task_loss_requirement.expect("threshold") - 0.1).abs() < 1.0e-6);
        assert!(!config.disable_task_loss_requirement);

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
    fn docs_birds_profile_matches_readme_asset_pipeline() {
        let config = BevyBurnAutoGazeConfig::docs_birds();

        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, DEFAULT_BIRDS_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_BIRDS_MAX_GAZE_TOKENS
        );
        assert_eq!(config.tile_batch_size, DEFAULT_BIRDS_TILE_BATCH_SIZE);
        assert_eq!(config.frames_per_clip, DEFAULT_BIRDS_FRAMES_PER_CLIP);
        assert_eq!(config.inference_width, Some(DEFAULT_BIRDS_INFERENCE_WIDTH));
        assert_eq!(
            config.inference_height,
            Some(DEFAULT_BIRDS_INFERENCE_HEIGHT)
        );
        assert_eq!(config.blend_alpha, DEFAULT_BIRDS_BLEND_ALPHA);
        assert_eq!(
            config.mask_visualization_mode,
            AutoGazeMaskVisualizationMode::Overlay
        );
        assert_eq!(config.keyframe_duration, DEFAULT_BIRDS_KEYFRAME_DURATION);
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Auto);
        assert!(!should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            config.mode.inference_mode()
        ));
    }

    #[test]
    fn realtime_query_defaults_keep_continuous_streaming_context() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string("?mode=realtime");

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.mode, BevyAutoGazeMode::Resize224);
        assert_eq!(config.top_k, DEFAULT_REALTIME_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_REALTIME_MAX_GAZE_TOKENS
        );
        assert_eq!(
            config.frames_per_clip,
            DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP
        );
        assert_eq!(
            config.inference_width,
            Some(DEFAULT_REALTIME_INFERENCE_WIDTH)
        );
        assert_eq!(config.inference_height, None);
        assert!(should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            config.mode.inference_mode()
        ));
    }

    #[test]
    fn tiled_query_defaults_use_bounded_generation_budget() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string("?mode=tiled&visualization-mode=interframe");

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, DEFAULT_BEVY_TILED_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_TILED_MAX_GAZE_TOKENS
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
    fn viewer_config_sanitizes_zero_and_nonfinite_url_values() {
        let mut config = BevyBurnAutoGazeConfig::default();
        let errors = config.apply_query_string(
            "?mode=tiled&top-k=0&frames-per-clip=0&max-in-flight=0&mask-cell-scale=NaN&blend-alpha=2&task-loss-requirement=NaN",
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(config.top_k, DEFAULT_BEVY_TILED_TOP_K);
        assert_eq!(config.frames_per_clip, DEFAULT_TILED_FRAMES_PER_CLIP);
        assert_eq!(config.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(config.mask_cell_scale, 1.0);
        assert_eq!(config.blend_alpha, 1.0);
        assert_eq!(config.task_loss_requirement, None);
        assert!(!config.disable_task_loss_requirement);

        let config = BevyBurnAutoGazeConfig {
            mode: BevyAutoGazeMode::Tile224,
            top_k: 0,
            tile_batch_size: 0,
            frames_per_clip: 0,
            max_in_flight: 0,
            keyframe_duration: 0,
            blend_alpha: f32::NAN,
            mask_cell_scale: -1.0,
            ..Default::default()
        }
        .sanitized();
        assert_eq!(config.top_k, DEFAULT_BEVY_TILED_TOP_K);
        assert_eq!(config.tile_batch_size, DEFAULT_TILED_TILE_BATCH_SIZE);
        assert_eq!(config.frames_per_clip, DEFAULT_TILED_FRAMES_PER_CLIP);
        assert_eq!(config.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(config.keyframe_duration, DEFAULT_BIRDS_KEYFRAME_DURATION);
        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
        assert_eq!(config.mask_cell_scale, 1.0);
    }

    #[test]
    fn viewer_uses_core_streaming_cache_policy() {
        let mut config = BevyBurnAutoGazeConfig {
            frames_per_clip: 16,
            streaming_cache: true,
            ..Default::default()
        };

        assert!(should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            AutoGazeInferenceMode::ResizeToModelInput
        ));
        assert!(!should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            AutoGazeInferenceMode::TiledResizeToGrid {
                tile_size: MODEL_INPUT_SIZE
            }
        ));

        config.frames_per_clip = 1;
        assert!(!should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            AutoGazeInferenceMode::ResizeToModelInput
        ));

        config.frames_per_clip = 16;
        config.streaming_cache = false;
        assert!(!should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            AutoGazeInferenceMode::ResizeToModelInput
        ));
    }

    #[test]
    fn viewer_realtime_policy_uses_core_defaults() {
        let policy = realtime_policy_from_config(&BevyBurnAutoGazeConfig::default());
        assert_eq!(policy.max_in_flight(), DEFAULT_MAX_IN_FLIGHT);
        assert!(policy.should_start_inference(0));
        assert!(!policy.should_start_inference(1));
        assert!(policy.should_draw_live_preview(false));
        assert!(!policy.should_draw_live_preview(true));

        let policy = realtime_policy_from_config(&BevyBurnAutoGazeConfig {
            max_in_flight: 2,
            streaming_cache: false,
            ..Default::default()
        });
        assert!(policy.should_start_inference(1));
        assert!(!policy.should_start_inference(2));
    }

    #[test]
    fn viewer_realtime_policy_keeps_streaming_cache_single_in_flight() {
        let config = BevyBurnAutoGazeConfig {
            mode: BevyAutoGazeMode::Resize224,
            frames_per_clip: 16,
            streaming_cache: true,
            max_in_flight: 4,
            ..Default::default()
        };

        let policy = realtime_policy_from_config(&config);
        assert_eq!(policy.max_in_flight(), DEFAULT_MAX_IN_FLIGHT);
        assert!(!policy.should_start_inference(1));

        let policy = realtime_policy_from_config(&BevyBurnAutoGazeConfig {
            streaming_cache: false,
            ..config.clone()
        });
        assert_eq!(policy.max_in_flight(), 4);
        assert!(policy.should_start_inference(3));
        assert!(!policy.should_start_inference(4));

        let policy = realtime_policy_from_config(&BevyBurnAutoGazeConfig {
            mode: BevyAutoGazeMode::Tile224,
            ..config
        });
        assert_eq!(policy.max_in_flight(), 4);
    }
}
