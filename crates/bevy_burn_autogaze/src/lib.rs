#![recursion_limit = "512"]

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use bevy::{
    diagnostic::{
        Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
        RegisterDiagnostic,
    },
    ecs::system::SystemParam,
    ecs::world::CommandQueue,
    prelude::*,
    render::{
        RenderPlugin,
        renderer::RenderAdapterInfo,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::{RelativeCursorPosition, widget::ImageNode},
    window::PrimaryWindow,
};
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BurnDevice};
use burn::tensor::Tensor;
use burn_autogaze::{
    AutoGazeConfig, AutoGazeDeviceMask, AutoGazeDeviceReadoutRunOutput, AutoGazeDeviceTokens,
    AutoGazeGazeRatioStats, AutoGazeInferenceMode, AutoGazeInferenceSequencer,
    AutoGazeMaskGeometryMode, AutoGazeMaskPlanStats, AutoGazeMaskVisualizationMode,
    AutoGazePatchDiffConfig, AutoGazePipeline, AutoGazePipelineOptions,
    AutoGazePreparedRun as CoreAutoGazePreparedRun, AutoGazePsnrStats, AutoGazeReadoutRunOutput,
    AutoGazeRealtimePolicy, AutoGazeRgbaClipShape, AutoGazeRgbaFrameClip, AutoGazeRgbaFrameQueue,
    AutoGazeRgbaVisualizationBuffers, AutoGazeRgbaVisualizationOptions, AutoGazeStreamingCache,
    AutoGazeTensorInterframePath, AutoGazeTensorVisualizationOptions,
    AutoGazeTensorVisualizationState, AutoGazeVisualizationMode, AutoGazeVisualizationState,
    FixationPoint, fixation_deduplicated_sparse_update_plan, fixation_effective_sparse_update_plan,
    fixation_sparse_update_plan, format_fps, format_gaze_ratio_percent, format_psnr_db,
    fps_from_millis, patch_diff_device_mask_async, patch_diff_readout_points_async,
    prepare_rgba_clip_for_trace, resize_rgba_frame_to_dimensions, rgba_clip_to_tensor,
    should_use_streaming_cache, video_frame_tensor,
};
#[cfg(target_arch = "wasm32")]
use burn_autogaze::{AutoGazeLoadOptions, NativeAutoGazeModel};
#[cfg(test)]
use burn_autogaze::{
    AutoGazeTaskLossOption, fixation_alpha_mask, fixation_scale_mask_rgba,
    rgba_clip_to_inference_tensor, rgba_clip_to_processor_tensor,
};
pub use burn_autogaze::{
    DEFAULT_BLEND_ALPHA, DEFAULT_KEYFRAME_DURATION, DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MODEL_GENERATION_BUDGET, DEFAULT_REALTIME_FRAMES_PER_CLIP, DEFAULT_REALTIME_TOP_K,
    DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO, DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
    DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS, DEFAULT_TILED_FRAMES_PER_CLIP,
    DEFAULT_TILED_MAX_GAZE_TOKENS, DEFAULT_TILED_TILE_BATCH_SIZE, DEFAULT_TILED_TOP_K,
};
use image::RgbaImage;

mod config;
mod display;
pub mod platform;
#[cfg(test)]
use config::MODEL_INPUT_SIZE;
pub use config::{
    BevyAutoGazeMode, BevyBurnAutoGazeConfig, BevyDisplayTransfer, BevyFrameSource,
    BevySparseMaskSource, DEFAULT_BEVY_DECODE_CHUNK_SIZE, DEFAULT_BEVY_DECODE_STRATEGY,
    DEFAULT_BEVY_LIMIT_GENERATION_BUDGET, DEFAULT_BEVY_MASK_GEOMETRY_MODE, DEFAULT_BEVY_MODE,
    DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP, DEFAULT_BEVY_SHOW_TASK_LOSS_SLIDER,
    DEFAULT_BEVY_STREAMING_CACHE, DEFAULT_BEVY_TASK_LOSS_REQUIREMENT, DEFAULT_BEVY_TILED_TOP_K,
    DEFAULT_BIRDS_BLEND_ALPHA, DEFAULT_BIRDS_FRAMES_PER_CLIP, DEFAULT_BIRDS_INFERENCE_HEIGHT,
    DEFAULT_BIRDS_INFERENCE_WIDTH, DEFAULT_BIRDS_KEYFRAME_DURATION, DEFAULT_BIRDS_MAX_GAZE_TOKENS,
    DEFAULT_BIRDS_TILE_BATCH_SIZE, DEFAULT_BIRDS_TOP_K, DEFAULT_CONFIG_URL,
    DEFAULT_NATIVE_MODEL_DIR, DEFAULT_REALTIME_INFERENCE_WIDTH, DEFAULT_REALTIME_MAX_GAZE_TOKENS,
    DEFAULT_TILED_INFERENCE_WIDTH, DEFAULT_WEIGHTS_URL, ImplicitModeDefaults,
    default_frames_per_clip, default_inference_dimensions, default_max_gaze_tokens_each_frame,
    default_max_gaze_tokens_for_limit, default_tile_batch_size, default_top_k,
    realtime_policy_from_config,
};
use display::{
    AutoGazeTexture, OneShotGpuUpload, TensorPanelVisualizationData, Visualization,
    VisualizationImageData, apply_visualization_to_preview_display, apply_visualization_to_world,
    visualization_image,
};
#[cfg(test)]
use display::{apply_visualization_to_texture, sync_texture_layout_nodes};

pub type AutoGazeBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type AutoGazeBevyDevice = burn::backend::wgpu::WgpuDevice;
const AUTO_GAZE_BEVY_BACKEND_NAME: &str = "webgpu";
const DEFAULT_REALTIME_MODEL_WARMUP_RUNS: usize = 3;
const DEFAULT_STREAMING_MODEL_WARMUP_EXTRA_RUNS: usize = 8;
const DEFAULT_TILED_MODEL_WARMUP_RUNS: usize = 1;
const SYNTHETIC_LOCAL_STRONG_FRAMES: u64 = 40;
const SYNTHETIC_LOCAL_SUBTLE_FRAMES: u64 = 40;
const SYNTHETIC_LOCAL_STILL_FRAMES: u64 = 32;
const SYNTHETIC_LOCAL_CYCLE_FRAMES: u64 =
    SYNTHETIC_LOCAL_STRONG_FRAMES + SYNTHETIC_LOCAL_SUBTLE_FRAMES + SYNTHETIC_LOCAL_STILL_FRAMES;
const MAX_SPARE_CLIP_BUFFERS: usize = 2;
const TIMING_LOG_INTERVAL_MS: f64 = 5_000.0;
const UI_MARGIN_PX: f32 = 12.0;
const METRIC_ROW_HEIGHT: f32 = 34.0;
const PANEL_LABEL_ROW_HEIGHT: f32 = 38.0;
const TASK_LOSS_SLIDER_MIN: f32 = 0.0;
const TASK_LOSS_SLIDER_MAX: f32 = 1.0;
const TASK_LOSS_SLIDER_STEP: f32 = 0.01;
const TASK_LOSS_SLIDER_WIDTH: f32 = 180.0;
const MODEL_FPS: DiagnosticPath = DiagnosticPath::const_new("autogaze_model_fps");
const AUTO_GPU_DISPLAY_MAX_PIXELS: usize = 224 * 224;

#[cfg(not(target_arch = "wasm32"))]
type Timestamp = Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Default)]
struct Timestamp(f64);

#[derive(Resource)]
struct AutoGazeModelState {
    config: BevyBurnAutoGazeConfig,
    pipeline: Option<Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>>,
    load_task: Option<Task<Result<AutoGazePipeline<AutoGazeBevyBackend>, String>>>,
}

#[derive(Resource)]
struct FrameQueue {
    inner: AutoGazeRgbaFrameQueue,
    latest_source_ms: f64,
    latest_prepare_ms: f64,
}

impl Default for FrameQueue {
    fn default() -> Self {
        Self {
            inner: AutoGazeRgbaFrameQueue::new(MAX_SPARE_CLIP_BUFFERS),
            latest_source_ms: 0.0,
            latest_prepare_ms: 0.0,
        }
    }
}

impl FrameQueue {
    #[cfg(test)]
    fn push(&mut self, frame: Arc<RgbaImage>, max_len: usize) {
        self.push_timed(frame, max_len, 0.0, 0.0);
    }

    fn push_timed(
        &mut self,
        frame: Arc<RgbaImage>,
        max_len: usize,
        source_ms: f64,
        prepare_ms: f64,
    ) {
        self.inner.push(frame, max_len);
        self.latest_source_ms = source_ms;
        self.latest_prepare_ms = prepare_ms;
    }

    fn latest(&self) -> Option<&RgbaImage> {
        self.inner.latest()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.latest_source_ms = 0.0;
        self.latest_prepare_ms = 0.0;
    }

    fn latest_timing(&self) -> (f64, f64) {
        (self.latest_source_ms, self.latest_prepare_ms)
    }

    fn build_clip(&mut self, max_len: usize) -> Result<Option<FrameClip>, String> {
        let pack_start = timestamp_now();
        self.inner
            .build_clip(max_len)
            .map_err(|err| format!("{err:#}"))
            .map(|clip| clip.map(|clip| FrameClip::from_core(clip, elapsed_ms(pack_start))))
    }

    fn build_latest_clip(&mut self) -> Result<Option<FrameClip>, String> {
        let pack_start = timestamp_now();
        self.inner
            .build_latest_clip()
            .map_err(|err| format!("{err:#}"))
            .map(|clip| clip.map(|clip| FrameClip::from_core(clip, elapsed_ms(pack_start))))
    }

    fn recycle_clip_buffer(&mut self, rgba: Vec<u8>) {
        self.inner.recycle_clip_buffer(rgba);
    }

    #[cfg(test)]
    fn spare_clip_buffer_count(&self) -> usize {
        self.inner.spare_clip_buffer_count()
    }
}

struct FrameClip {
    core: AutoGazeRgbaFrameClip,
    source_ms: f64,
    prepare_ms: f64,
    pack_ms: f64,
}

impl FrameClip {
    fn from_core(clip: AutoGazeRgbaFrameClip, pack_ms: f64) -> Self {
        Self {
            core: clip,
            source_ms: 0.0,
            prepare_ms: 0.0,
            pack_ms,
        }
    }

    fn width(&self) -> usize {
        self.core.width()
    }

    fn height(&self) -> usize {
        self.core.height()
    }

    #[cfg(test)]
    fn clip_len(&self) -> usize {
        self.core.clip_len()
    }

    fn rgba(&self) -> &[u8] {
        self.core.rgba()
    }

    #[cfg(test)]
    fn rgba_capacity(&self) -> usize {
        self.core.rgba_capacity()
    }

    fn shape(&self) -> AutoGazeRgbaClipShape {
        self.core.shape()
    }

    fn last_frame_rgba(&self) -> Result<&[u8], String> {
        self.core
            .last_frame_rgba()
            .map_err(|err| format!("{err:#}"))
    }

    fn into_rgba(self) -> Vec<u8> {
        self.core.into_rgba()
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct BevyStreamingGenerationState {
    enabled: bool,
    width: usize,
    height: usize,
    horizon_frames: usize,
    cache: Option<AutoGazeStreamingCache<AutoGazeBevyBackend>>,
}

impl BevyStreamingGenerationState {
    fn configure(&mut self, enabled: bool, width: usize, height: usize, horizon_frames: usize) {
        let horizon_frames = horizon_frames.max(1);
        if !enabled {
            self.reset();
            return;
        }
        let shape_changed =
            self.width != width || self.height != height || self.horizon_frames != horizon_frames;
        self.enabled = true;
        self.width = width;
        self.height = height;
        self.horizon_frames = horizon_frames;
        if shape_changed || self.cache.is_none() {
            self.cache = Some(AutoGazeStreamingCache::new(horizon_frames));
        }
    }

    fn cache_mut(&mut self) -> &mut AutoGazeStreamingCache<AutoGazeBevyBackend> {
        self.cache
            .get_or_insert_with(|| AutoGazeStreamingCache::new(self.horizon_frames.max(1)))
    }

    fn reset(&mut self) {
        self.enabled = false;
        self.width = 0;
        self.height = 0;
        self.horizon_frames = 0;
        self.cache = None;
    }
}

#[derive(Resource, Default, Clone)]
struct StaticFrame(Option<Arc<RgbaImage>>);

#[derive(Resource, Clone, Debug, Default)]
struct SyntheticFrameSource {
    frame_index: u64,
}

impl SyntheticFrameSource {
    fn next_frame(&mut self, config: &BevyBurnAutoGazeConfig) -> RgbaImage {
        let (width, height) = synthetic_source_dimensions(config);
        let frame = match config.source {
            BevyFrameSource::SyntheticPulse => {
                synthetic_pulse_frame(width, height, self.frame_index)
            }
            BevyFrameSource::SyntheticLocalMotion => {
                synthetic_local_motion_frame(width, height, self.frame_index)
            }
            _ => synthetic_pan_frame(width, height, self.frame_index),
        };
        self.frame_index = self.frame_index.wrapping_add(1);
        frame
    }
}

#[derive(Resource, Default, Clone, Debug)]
struct InferenceSequencer(AutoGazeInferenceSequencer);

impl InferenceSequencer {
    fn reserve(&mut self) -> u64 {
        self.0.reserve()
    }

    fn accept(&mut self, sequence: u64) -> bool {
        self.0.accept(sequence)
    }

    fn invalidate_pending(&mut self) {
        self.0.invalidate_pending();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompletedModelDisplayAction {
    DisplayVisualization,
    UpdateMaskOnly,
}

impl CompletedModelDisplayAction {
    const fn displays_visualization(self) -> bool {
        matches!(self, Self::DisplayVisualization)
    }
}

fn completed_model_display_action(
    policy: AutoGazeRealtimePolicy,
    model_ready: bool,
    active_task_count: usize,
) -> CompletedModelDisplayAction {
    if policy.should_draw_async_stream_preview(model_ready, active_task_count) {
        CompletedModelDisplayAction::UpdateMaskOnly
    } else {
        CompletedModelDisplayAction::DisplayVisualization
    }
}

fn completed_run_display_action(
    policy: AutoGazeRealtimePolicy,
    use_patch_diff: bool,
    active_task_count: usize,
) -> CompletedModelDisplayAction {
    if use_patch_diff {
        CompletedModelDisplayAction::DisplayVisualization
    } else {
        completed_model_display_action(policy, true, active_task_count)
    }
}

#[derive(Resource, Clone)]
struct BevyVisualizationState {
    cpu: AutoGazeVisualizationState,
    gpu: AutoGazeTensorVisualizationState<AutoGazeBevyBackend>,
}

impl BevyVisualizationState {
    fn new(mode: AutoGazeVisualizationMode, keyframe_duration: usize) -> Self {
        Self {
            cpu: AutoGazeVisualizationState::new(mode, keyframe_duration),
            gpu: AutoGazeTensorVisualizationState::new(mode, keyframe_duration),
        }
    }

    fn configure(&mut self, mode: AutoGazeVisualizationMode, keyframe_duration: usize) {
        self.cpu.configure(mode, keyframe_duration);
        self.gpu.configure(mode, keyframe_duration);
    }

    fn reset(&mut self) {
        self.cpu.reset();
        self.gpu.reset();
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct TaskLossSliderState {
    value: f32,
    pending_value: Option<f32>,
    dragging: bool,
}

impl TaskLossSliderState {
    fn new(config: &BevyBurnAutoGazeConfig) -> Self {
        let value = quality_slider_config_value(config);
        Self {
            value: quantize_task_loss_slider_value(value),
            pending_value: None,
            dragging: false,
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct LatestMaskPrediction {
    points: Vec<FixationPoint>,
}

impl LatestMaskPrediction {
    fn update(&mut self, points: Vec<FixationPoint>) {
        self.points = points;
    }

    fn clear(&mut self) {
        self.points.clear();
    }

    fn points(&self) -> &[FixationPoint] {
        &self.points
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct GazeRatioStats(AutoGazeGazeRatioStats);

impl GazeRatioStats {
    fn record(&mut self, ratio: f64) {
        self.0.record(ratio);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct PsnrStats(AutoGazePsnrStats);

impl PsnrStats {
    fn record(&mut self, psnr_db: f64) {
        self.0.record(psnr_db);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InferenceTiming {
    sequence: u64,
    clip_frames: usize,
    model_frames: usize,
    effective_generation_budget: usize,
    generated_tokens: usize,
    active_generated_tokens: usize,
    padded_generated_tokens: usize,
    trace_points: usize,
    active_trace_points: usize,
    width: usize,
    height: usize,
    source_ms: f64,
    prepare_ms: f64,
    pack_ms: f64,
    input_ms: f64,
    display_input_ms: f64,
    model_ms: f64,
    trace_ms: f64,
    sync_ms: f64,
    visualize_cpu_ms: f64,
    psnr_ms: f64,
    tensor_ms: f64,
    visualize_ms: f64,
    display_ms: f64,
    total_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    display_input_residency: DisplayInputResidency,
    effective_display_transfer: BevyDisplayTransfer,
    gaze_update_ratio: f64,
    gaze_update_ratio_sample: Option<f64>,
    output_update_ratio: f64,
    output_update_ratio_sample: Option<f64>,
    psnr_db: Option<f64>,
    tensor_interframe_path: Option<AutoGazeTensorInterframePath>,
    mask_plan_stats: AutoGazeMaskPlanStats,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DisplayInputResidency {
    #[default]
    None,
    HostRgbaUpload,
    ModelTensorReuse,
}

impl DisplayInputResidency {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::HostRgbaUpload => "host-rgba-upload",
            Self::ModelTensorReuse => "model-tensor-reuse",
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct InferenceTimingStats {
    latest: Option<InferenceTiming>,
    render_adapter: Option<RenderAdapterSummary>,
    run_config: Option<InferenceRunConfigSummary>,
    last_log: Option<Timestamp>,
    total_ms: f64,
    model_ms: f64,
    model_frames: usize,
    generated_tokens: usize,
    active_generated_tokens: usize,
    padded_generated_tokens: usize,
    trace_points: usize,
    active_trace_points: usize,
    input_ms: f64,
    display_input_ms: f64,
    pack_ms: f64,
    visualize_ms: f64,
    visualize_cpu_ms: f64,
    psnr_ms: f64,
    tensor_ms: f64,
    display_ms: f64,
    source_samples: Vec<f64>,
    prepare_samples: Vec<f64>,
    pack_samples: Vec<f64>,
    input_samples: Vec<f64>,
    display_input_samples: Vec<f64>,
    visualize_samples: Vec<f64>,
    visualize_cpu_samples: Vec<f64>,
    psnr_ms_samples: Vec<f64>,
    tensor_samples: Vec<f64>,
    display_samples: Vec<f64>,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    gaze_update_ratio: f64,
    gaze_update_samples: usize,
    latest_gaze_update_ratio: Option<f64>,
    output_update_ratio: f64,
    output_update_samples: usize,
    latest_output_update_ratio: Option<f64>,
    psnr_stats: AutoGazePsnrStats,
    psnr_samples: usize,
    mask_rects: usize,
    mask_row_spans: usize,
    mask_pixels: usize,
    samples: Vec<f64>,
    model_samples: Vec<f64>,
    stale_results: usize,
    skipped_warmup_frames: usize,
    latest_skipped_warmup_sequence: Option<u64>,
    emitted_summary: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InferenceRunConfigSummary {
    source: &'static str,
    sparse_mask_source: &'static str,
    mode: &'static str,
    visualization_mode: &'static str,
    mask_visualization_mode: &'static str,
    mask_geometry_mode: &'static str,
    display_transfer: &'static str,
    streaming_cache: bool,
    streaming_cache_effective: bool,
    configured_max_in_flight: usize,
    effective_max_in_flight: usize,
    frames_per_clip: usize,
    top_k: usize,
    max_gaze_tokens_each_frame: usize,
    patch_diff_grid_size: usize,
    patch_diff_threshold: f32,
    tile_batch_size: usize,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
    tensor_sparse_update_max_rects: usize,
    tensor_sparse_update_max_ratio: f64,
    tensor_full_frame_update_min_ratio: f64,
    show_psnr: bool,
    warmup_model: bool,
    perf_summary_warmup_frames: usize,
    burn_backend: &'static str,
}

impl From<&BevyBurnAutoGazeConfig> for InferenceRunConfigSummary {
    fn from(config: &BevyBurnAutoGazeConfig) -> Self {
        Self {
            source: config.source.as_str(),
            sparse_mask_source: config.sparse_mask_source.as_str(),
            mode: config.mode.as_str(),
            visualization_mode: config.visualization_mode.as_str(),
            mask_visualization_mode: config.mask_visualization_mode.as_str(),
            mask_geometry_mode: config.mask_geometry_mode.as_str(),
            display_transfer: config.display_transfer.as_str(),
            streaming_cache: config.streaming_cache,
            streaming_cache_effective: should_use_streaming_cache(
                config.streaming_cache,
                config.frames_per_clip,
                config.mode.inference_mode(),
            ),
            configured_max_in_flight: config.max_in_flight,
            effective_max_in_flight: realtime_policy_from_config(config).max_in_flight(),
            frames_per_clip: config.frames_per_clip,
            top_k: config.top_k,
            max_gaze_tokens_each_frame: config.max_gaze_tokens_each_frame,
            patch_diff_grid_size: config.patch_diff_grid_size,
            patch_diff_threshold: config.patch_diff_threshold,
            tile_batch_size: config.tile_batch_size,
            inference_width: config.inference_width,
            inference_height: config.inference_height,
            tensor_sparse_update_max_rects: config.tensor_sparse_update_max_rects,
            tensor_sparse_update_max_ratio: config.tensor_sparse_update_max_ratio,
            tensor_full_frame_update_min_ratio: config.tensor_full_frame_update_min_ratio,
            show_psnr: config.show_psnr,
            warmup_model: config.warmup_model,
            perf_summary_warmup_frames: config.perf_summary_warmup_frames,
            burn_backend: AUTO_GAZE_BEVY_BACKEND_NAME,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RenderAdapterSummary {
    name: String,
    vendor: u32,
    device_type: String,
    backend: String,
    driver: String,
    driver_info: String,
}

impl From<&RenderAdapterInfo> for RenderAdapterSummary {
    fn from(adapter: &RenderAdapterInfo) -> Self {
        Self {
            name: adapter.name.clone(),
            vendor: adapter.vendor,
            device_type: format!("{:?}", adapter.device_type),
            backend: format!("{:?}", adapter.backend),
            driver: adapter.driver.clone(),
            driver_info: adapter.driver_info.clone(),
        }
    }
}

fn is_software_render_adapter(adapter: &RenderAdapterInfo) -> bool {
    matches!(format!("{:?}", adapter.device_type).as_str(), "Cpu")
}

impl InferenceTimingStats {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn set_render_adapter(&mut self, render_adapter: RenderAdapterSummary) {
        self.render_adapter = Some(render_adapter);
    }

    fn set_run_config(&mut self, run_config: InferenceRunConfigSummary) {
        self.run_config = Some(run_config);
    }

    fn record(&mut self, timing: InferenceTiming, should_log: bool) {
        self.total_ms += timing.total_ms;
        self.model_ms += timing.model_ms;
        self.model_frames += timing.model_frames;
        self.generated_tokens = self
            .generated_tokens
            .saturating_add(timing.generated_tokens);
        self.active_generated_tokens = self
            .active_generated_tokens
            .saturating_add(timing.active_generated_tokens);
        self.padded_generated_tokens = self
            .padded_generated_tokens
            .saturating_add(timing.padded_generated_tokens);
        self.trace_points += timing.trace_points;
        self.active_trace_points += timing.active_trace_points;
        self.input_ms += timing.input_ms;
        self.display_input_ms += timing.display_input_ms;
        self.pack_ms += timing.pack_ms;
        self.visualize_ms += timing.visualize_ms;
        self.visualize_cpu_ms += timing.visualize_cpu_ms;
        self.psnr_ms += timing.psnr_ms;
        self.tensor_ms += timing.tensor_ms;
        self.display_ms += timing.display_ms;
        self.source_samples.push(timing.source_ms);
        self.prepare_samples.push(timing.prepare_ms);
        self.pack_samples.push(timing.pack_ms);
        self.input_samples.push(timing.input_ms);
        self.display_input_samples.push(timing.display_input_ms);
        self.visualize_samples.push(timing.visualize_ms);
        self.visualize_cpu_samples.push(timing.visualize_cpu_ms);
        self.psnr_ms_samples.push(timing.psnr_ms);
        self.tensor_samples.push(timing.tensor_ms);
        self.display_samples.push(timing.display_ms);
        self.output_rgba_bytes = self
            .output_rgba_bytes
            .saturating_add(timing.output_rgba_bytes);
        self.output_tensor_bytes = self
            .output_tensor_bytes
            .saturating_add(timing.output_tensor_bytes);
        if let Some(ratio) = timing.gaze_update_ratio_sample {
            self.gaze_update_ratio += ratio;
            self.gaze_update_samples = self.gaze_update_samples.saturating_add(1);
            self.latest_gaze_update_ratio = Some(ratio);
        }
        if let Some(ratio) = timing.output_update_ratio_sample {
            self.output_update_ratio += ratio;
            self.output_update_samples = self.output_update_samples.saturating_add(1);
            self.latest_output_update_ratio = Some(ratio);
        }
        if let Some(psnr_db) = timing.psnr_db {
            self.psnr_stats.record(psnr_db);
            self.psnr_samples = self.psnr_samples.saturating_add(1);
        }
        self.mask_rects = self
            .mask_rects
            .saturating_add(timing.mask_plan_stats.rect_count);
        self.mask_row_spans = self
            .mask_row_spans
            .saturating_add(timing.mask_plan_stats.row_span_count);
        self.mask_pixels = self
            .mask_pixels
            .saturating_add(timing.mask_plan_stats.pixel_count);
        self.samples.push(timing.total_ms);
        self.model_samples.push(timing.model_ms);
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
        let (source_label, frame_fps_label) = match self
            .run_config
            .as_ref()
            .map(|config| config.sparse_mask_source)
        {
            Some("patch-diff") => ("Patch-diff", "mask-frame fps"),
            _ => ("AutoGaze", "model-frame fps"),
        };
        log(&format!(
            "{source_label} timing: {:.1} output fps / {:.1} {frame_fps_label} ({:.1} ms) clip={} model_frames={} budget={} generated={}/{} padded={} points={}/{} rects={} spans={} mask={:.2}% output={:.2}% {}x{}, source={:.1} ms, prepare={:.1} ms, pack={:.1} ms, input={:.1} ms, display_input={:.1} ms ({}) model={:.1} ms, trace={:.1} ms, sync={:.1} ms, visualize_cpu={:.1} ms, psnr={:.1} ms, tensor={:.1} ms, display_transfer={}, tensor_path={}, visualize={:.1} ms, display={:.1} ms, output={:.1} MiB rgba/{:.1} MiB f32",
            timing.e2e_fps(),
            timing.model_frame_fps(),
            timing.total_ms,
            timing.clip_frames,
            timing.model_frames,
            timing.effective_generation_budget,
            timing.active_generated_tokens,
            timing.generated_tokens,
            timing.padded_generated_tokens,
            timing.active_trace_points,
            timing.trace_points,
            timing.mask_plan_stats.rect_count,
            timing.mask_plan_stats.row_span_count,
            timing.gaze_update_ratio * 100.0,
            timing.output_update_ratio * 100.0,
            timing.width,
            timing.height,
            timing.source_ms,
            timing.prepare_ms,
            timing.pack_ms,
            timing.input_ms,
            timing.display_input_ms,
            timing.display_input_residency.as_str(),
            timing.model_ms,
            timing.trace_ms,
            timing.sync_ms,
            timing.visualize_cpu_ms,
            timing.psnr_ms,
            timing.tensor_ms,
            timing.effective_display_transfer.as_str(),
            timing
                .tensor_interframe_path
                .map(|path| path.as_str())
                .unwrap_or("none"),
            timing.visualize_ms,
            timing.display_ms,
            timing.output_rgba_bytes as f64 / (1024.0 * 1024.0),
            timing.output_tensor_bytes as f64 / (1024.0 * 1024.0),
        ));
    }

    fn record_stale_result(&mut self) {
        self.stale_results = self.stale_results.saturating_add(1);
    }

    fn skip_warmup_frame(&mut self, timing: InferenceTiming) {
        self.skipped_warmup_frames = self.skipped_warmup_frames.saturating_add(1);
        self.latest_skipped_warmup_sequence = Some(timing.sequence);
    }

    fn processed_frames(&self) -> usize {
        self.samples.len()
    }

    fn summary_json(&self, target_frames: usize) -> String {
        let processed_frames = self.processed_frames();
        let avg_total_ms = mean_or_zero(self.total_ms, processed_frames);
        let avg_model_ms = mean_or_zero(self.model_ms, processed_frames);
        let avg_input_ms = mean_or_zero(self.input_ms, processed_frames);
        let avg_display_input_ms = mean_or_zero(self.display_input_ms, processed_frames);
        let avg_pack_ms = mean_or_zero(self.pack_ms, processed_frames);
        let avg_visualize_ms = mean_or_zero(self.visualize_ms, processed_frames);
        let avg_visualize_cpu_ms = mean_or_zero(self.visualize_cpu_ms, processed_frames);
        let avg_psnr_ms = mean_or_zero(self.psnr_ms, processed_frames);
        let avg_tensor_ms = mean_or_zero(self.tensor_ms, processed_frames);
        let avg_display_ms = mean_or_zero(self.display_ms, processed_frames);
        let avg_output_rgba_bytes = mean_or_zero(self.output_rgba_bytes as f64, processed_frames);
        let avg_output_tensor_bytes =
            mean_or_zero(self.output_tensor_bytes as f64, processed_frames);
        let avg_gaze_update_ratio = mean_or_zero(self.gaze_update_ratio, self.gaze_update_samples);
        let avg_output_update_ratio =
            mean_or_zero(self.output_update_ratio, self.output_update_samples);
        let avg_input_fps = fps_from_millis(avg_total_ms)
            * mean_or_zero(self.model_frames as f64, processed_frames);
        let avg_model_frame_fps = if self.model_ms > 0.0 {
            self.model_frames as f64 * 1_000.0 / self.model_ms
        } else {
            0.0
        };
        let avg_trace_points = mean_or_zero(self.trace_points as f64, processed_frames);
        let avg_active_trace_points =
            mean_or_zero(self.active_trace_points as f64, processed_frames);
        let avg_generated_tokens = mean_or_zero(self.generated_tokens as f64, processed_frames);
        let avg_active_generated_tokens =
            mean_or_zero(self.active_generated_tokens as f64, processed_frames);
        let avg_padded_generated_tokens =
            mean_or_zero(self.padded_generated_tokens as f64, processed_frames);
        let avg_mask_rects = mean_or_zero(self.mask_rects as f64, processed_frames);
        let avg_mask_row_spans = mean_or_zero(self.mask_row_spans as f64, processed_frames);
        let avg_mask_pixels = mean_or_zero(self.mask_pixels as f64, processed_frames);
        let p50_total_ms = percentile_ms(&self.samples, 0.50);
        let p95_total_ms = percentile_ms(&self.samples, 0.95);
        let p99_total_ms = percentile_ms(&self.samples, 0.99);
        let max_total_ms = max_ms(&self.samples);
        let p50_model_ms = percentile_ms(&self.model_samples, 0.50);
        let p95_model_ms = percentile_ms(&self.model_samples, 0.95);
        let p99_model_ms = percentile_ms(&self.model_samples, 0.99);
        let max_model_ms = max_ms(&self.model_samples);
        let p05_output_fps = fps_from_millis(p95_total_ms);
        let worst_output_fps = fps_from_millis(max_total_ms);
        let fps_stability_p05_to_avg = if avg_total_ms > 0.0 && p95_total_ms > 0.0 {
            avg_total_ms / p95_total_ms
        } else {
            0.0
        };
        let latest_clip_frames = self
            .latest
            .map(|timing| timing.clip_frames)
            .unwrap_or_default();
        let latest_model_frames = self
            .latest
            .map(|timing| timing.model_frames)
            .unwrap_or_default();
        let mut summary = serde_json::json!({
            "target_frames": target_frames,
            "skipped_warmup_frames": self.skipped_warmup_frames,
            "latest_skipped_warmup_sequence": self.latest_skipped_warmup_sequence,
            "processed_frames": processed_frames,
            "processed_model_frames": self.model_frames,
            "avg_output_fps": fps_from_millis(avg_total_ms),
            "avg_model_frame_fps": avg_model_frame_fps,
            "avg_input_fps": avg_input_fps,
            "avg_total_ms": avg_total_ms,
            "p50_total_ms": p50_total_ms,
            "p95_total_ms": p95_total_ms,
            "p99_total_ms": p99_total_ms,
            "max_total_ms": max_total_ms,
            "p05_output_fps": p05_output_fps,
            "worst_output_fps": worst_output_fps,
            "fps_stability_p05_to_avg": fps_stability_p05_to_avg,
            "avg_model_ms": avg_model_ms,
            "p50_model_ms": p50_model_ms,
            "p95_model_ms": p95_model_ms,
            "p99_model_ms": p99_model_ms,
            "max_model_ms": max_model_ms,
            "p95_source_ms": percentile_ms(&self.source_samples, 0.95),
            "p95_prepare_ms": percentile_ms(&self.prepare_samples, 0.95),
            "p95_pack_ms": percentile_ms(&self.pack_samples, 0.95),
            "p95_input_ms": percentile_ms(&self.input_samples, 0.95),
            "p95_display_input_ms": percentile_ms(&self.display_input_samples, 0.95),
            "p95_visualize_ms": percentile_ms(&self.visualize_samples, 0.95),
            "p95_visualize_cpu_ms": percentile_ms(&self.visualize_cpu_samples, 0.95),
            "p95_psnr_ms": percentile_ms(&self.psnr_ms_samples, 0.95),
            "p95_tensor_ms": percentile_ms(&self.tensor_samples, 0.95),
            "p95_display_ms": percentile_ms(&self.display_samples, 0.95),
            "avg_trace_points": avg_trace_points,
            "avg_active_trace_points": avg_active_trace_points,
            "avg_generated_tokens": avg_generated_tokens,
            "avg_active_generated_tokens": avg_active_generated_tokens,
            "avg_padded_generated_tokens": avg_padded_generated_tokens,
            "avg_mask_rects": avg_mask_rects,
            "avg_mask_row_spans": avg_mask_row_spans,
            "avg_mask_pixels": avg_mask_pixels,
            "avg_input_ms": avg_input_ms,
            "avg_display_input_ms": avg_display_input_ms,
            "avg_pack_ms": avg_pack_ms,
            "avg_visualize_ms": avg_visualize_ms,
            "avg_visualize_cpu_ms": avg_visualize_cpu_ms,
            "avg_psnr_ms": avg_psnr_ms,
            "avg_tensor_ms": avg_tensor_ms,
            "avg_display_ms": avg_display_ms,
            "avg_output_rgba_bytes": avg_output_rgba_bytes,
            "avg_output_tensor_bytes": avg_output_tensor_bytes,
            "avg_gaze_update_ratio": avg_gaze_update_ratio,
            "avg_mask_update_ratio": avg_gaze_update_ratio,
            "avg_output_update_ratio": avg_output_update_ratio,
            "psnr_samples": self.psnr_samples,
            "latest_psnr_db": psnr_metric_json_value(&self.psnr_stats, PsnrMetricKind::Current),
            "latest_psnr_db_infinite": psnr_metric_is_infinite(&self.psnr_stats, PsnrMetricKind::Current),
            "ema_psnr_db": psnr_metric_json_value(&self.psnr_stats, PsnrMetricKind::Ema),
            "ema_psnr_db_infinite": psnr_metric_is_infinite(&self.psnr_stats, PsnrMetricKind::Ema),
            "latest_output_rgba_bytes": self.latest.map(|timing| timing.output_rgba_bytes).unwrap_or_default(),
            "latest_output_tensor_bytes": self.latest.map(|timing| timing.output_tensor_bytes).unwrap_or_default(),
            "latest_effective_display_transfer": self.latest.map(|timing| timing.effective_display_transfer.as_str()).unwrap_or("none"),
            "display_residency": self.latest.map(display_residency).unwrap_or("none"),
            "display_input_residency": self.latest.map(|timing| timing.display_input_residency.as_str()).unwrap_or("none"),
            "latest_display_input_ms": self.latest.map(|timing| timing.display_input_ms).unwrap_or_default(),
            "latest_clip_frames": latest_clip_frames,
            "latest_model_frames": latest_model_frames,
            "latest_trace_points": self.latest.map(|timing| timing.trace_points).unwrap_or_default(),
            "latest_active_trace_points": self.latest.map(|timing| timing.active_trace_points).unwrap_or_default(),
            "latest_generated_tokens": self.latest.map(|timing| timing.generated_tokens).unwrap_or_default(),
            "latest_active_generated_tokens": self.latest.map(|timing| timing.active_generated_tokens).unwrap_or_default(),
            "latest_padded_generated_tokens": self.latest.map(|timing| timing.padded_generated_tokens).unwrap_or_default(),
            "latest_effective_generation_budget": self.latest.map(|timing| timing.effective_generation_budget).unwrap_or_default(),
            "latest_mask_rects": self.latest.map(|timing| timing.mask_plan_stats.rect_count).unwrap_or_default(),
            "latest_mask_row_spans": self.latest.map(|timing| timing.mask_plan_stats.row_span_count).unwrap_or_default(),
            "latest_mask_pixels": self.latest.map(|timing| timing.mask_plan_stats.pixel_count).unwrap_or_default(),
            "latest_gaze_update_ratio": self.latest_gaze_update_ratio.unwrap_or_default(),
            "latest_mask_update_ratio": self.latest_gaze_update_ratio.unwrap_or_default(),
            "latest_output_update_ratio": self.latest_output_update_ratio.unwrap_or_default(),
            "latest_tensor_interframe_path": self.latest.and_then(|timing| timing.tensor_interframe_path).map(|path| path.as_str()),
            "latest_sequence": self.latest.map(|timing| timing.sequence).unwrap_or_default(),
            "latest_width": self.latest.map(|timing| timing.width).unwrap_or_default(),
            "latest_height": self.latest.map(|timing| timing.height).unwrap_or_default(),
            "stale_results": self.stale_results,
            "render_adapter_name": self.render_adapter.as_ref().map(|adapter| adapter.name.as_str()),
            "render_adapter_vendor": self.render_adapter.as_ref().map(|adapter| adapter.vendor),
            "render_adapter_device_type": self.render_adapter.as_ref().map(|adapter| adapter.device_type.as_str()),
            "render_adapter_backend": self.render_adapter.as_ref().map(|adapter| adapter.backend.as_str()),
            "render_adapter_driver": self.render_adapter.as_ref().map(|adapter| adapter.driver.as_str()),
            "render_adapter_driver_info": self.render_adapter.as_ref().map(|adapter| adapter.driver_info.as_str()),
        });
        if let Some(fields) = summary.as_object_mut() {
            insert_run_config_json_fields(fields, self.run_config);
        }
        summary.to_string()
    }
}

impl InferenceTiming {
    fn e2e_fps(self) -> f64 {
        fps_from_millis(self.total_ms)
    }

    fn model_frame_fps(self) -> f64 {
        if self.model_frames > 0 {
            fps_from_millis(self.model_ms / self.model_frames as f64)
        } else {
            0.0
        }
    }
}

fn display_residency(timing: InferenceTiming) -> &'static str {
    if timing.output_tensor_bytes > 0 && timing.output_rgba_bytes == 0 {
        "gpu-tensor"
    } else if timing.output_rgba_bytes > 0 && timing.output_tensor_bytes == 0 {
        "cpu-rgba"
    } else if timing.output_rgba_bytes > 0 && timing.output_tensor_bytes > 0 {
        "mixed"
    } else {
        "none"
    }
}

fn mean_or_zero(total: f64, count: usize) -> f64 {
    if count > 0 { total / count as f64 } else { 0.0 }
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

fn max_ms(samples: &[f64]) -> f64 {
    samples.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn insert_run_config_json_fields(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    run_config: Option<InferenceRunConfigSummary>,
) {
    let Some(config) = run_config else {
        return;
    };
    fields.insert("mode".to_string(), config.mode.into());
    fields.insert("source".to_string(), config.source.into());
    fields.insert(
        "sparse_mask_source".to_string(),
        config.sparse_mask_source.into(),
    );
    fields.insert(
        "visualization_mode".to_string(),
        config.visualization_mode.into(),
    );
    fields.insert(
        "mask_visualization_mode".to_string(),
        config.mask_visualization_mode.into(),
    );
    fields.insert(
        "mask_geometry_mode".to_string(),
        config.mask_geometry_mode.into(),
    );
    fields.insert(
        "display_transfer".to_string(),
        config.display_transfer.into(),
    );
    fields.insert("streaming_cache".to_string(), config.streaming_cache.into());
    fields.insert(
        "streaming_cache_effective".to_string(),
        config.streaming_cache_effective.into(),
    );
    fields.insert(
        "configured_max_in_flight".to_string(),
        config.configured_max_in_flight.into(),
    );
    fields.insert(
        "effective_max_in_flight".to_string(),
        config.effective_max_in_flight.into(),
    );
    fields.insert("frames_per_clip".to_string(), config.frames_per_clip.into());
    fields.insert("top_k".to_string(), config.top_k.into());
    fields.insert(
        "max_gaze_tokens_each_frame".to_string(),
        config.max_gaze_tokens_each_frame.into(),
    );
    fields.insert(
        "patch_diff_grid_size".to_string(),
        config.patch_diff_grid_size.into(),
    );
    fields.insert(
        "patch_diff_threshold".to_string(),
        config.patch_diff_threshold.into(),
    );
    fields.insert("tile_batch_size".to_string(), config.tile_batch_size.into());
    fields.insert("inference_width".to_string(), config.inference_width.into());
    fields.insert(
        "inference_height".to_string(),
        config.inference_height.into(),
    );
    fields.insert(
        "tensor_sparse_update_max_rects".to_string(),
        config.tensor_sparse_update_max_rects.into(),
    );
    fields.insert(
        "tensor_sparse_update_max_ratio".to_string(),
        config.tensor_sparse_update_max_ratio.into(),
    );
    fields.insert(
        "tensor_full_frame_update_min_ratio".to_string(),
        config.tensor_full_frame_update_min_ratio.into(),
    );
    fields.insert("show_psnr".to_string(), config.show_psnr.into());
    fields.insert("warmup_model".to_string(), config.warmup_model.into());
    fields.insert(
        "perf_summary_warmup_frames".to_string(),
        config.perf_summary_warmup_frames.into(),
    );
    fields.insert("burn_backend".to_string(), config.burn_backend.into());
}

#[derive(Clone, Copy)]
enum PsnrMetricKind {
    Current,
    Ema,
}

fn psnr_metric_value(stats: &AutoGazePsnrStats, kind: PsnrMetricKind) -> Option<f64> {
    stats.is_initialized().then(|| match kind {
        PsnrMetricKind::Current => stats.current(),
        PsnrMetricKind::Ema => stats.ema(),
    })
}

fn psnr_metric_json_value(stats: &AutoGazePsnrStats, kind: PsnrMetricKind) -> serde_json::Value {
    psnr_metric_value(stats, kind)
        .and_then(serde_json::Number::from_f64)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn psnr_metric_is_infinite(stats: &AutoGazePsnrStats, kind: PsnrMetricKind) -> bool {
    psnr_metric_value(stats, kind)
        .map(|value| value.is_infinite() && value.is_sign_positive())
        .unwrap_or(false)
}

fn perf_sample_json(stats: &InferenceTimingStats) -> Option<String> {
    let latest = stats.latest?;
    let avg_output_fps = fps_from_millis(mean_or_zero(stats.total_ms, stats.processed_frames()));
    let avg_model_frames_per_output =
        mean_or_zero(stats.model_frames as f64, stats.processed_frames());
    let avg_input_fps = avg_output_fps * avg_model_frames_per_output;
    let avg_model_frame_fps = if stats.model_ms > 0.0 {
        stats.model_frames as f64 * 1_000.0 / stats.model_ms
    } else {
        0.0
    };
    let avg_trace_points = mean_or_zero(stats.trace_points as f64, stats.processed_frames());
    let avg_active_trace_points =
        mean_or_zero(stats.active_trace_points as f64, stats.processed_frames());
    let avg_generated_tokens =
        mean_or_zero(stats.generated_tokens as f64, stats.processed_frames());
    let avg_active_generated_tokens = mean_or_zero(
        stats.active_generated_tokens as f64,
        stats.processed_frames(),
    );
    let avg_padded_generated_tokens = mean_or_zero(
        stats.padded_generated_tokens as f64,
        stats.processed_frames(),
    );
    let avg_mask_rects = mean_or_zero(stats.mask_rects as f64, stats.processed_frames());
    let avg_mask_row_spans = mean_or_zero(stats.mask_row_spans as f64, stats.processed_frames());
    let avg_mask_pixels = mean_or_zero(stats.mask_pixels as f64, stats.processed_frames());
    let p95_total_ms = percentile_ms(&stats.samples, 0.95);
    let p99_total_ms = percentile_ms(&stats.samples, 0.99);
    let max_total_ms = max_ms(&stats.samples);
    let mut sample = serde_json::json!({
        "processed_frames": stats.processed_frames(),
        "skipped_warmup_frames": stats.skipped_warmup_frames,
        "latest_skipped_warmup_sequence": stats.latest_skipped_warmup_sequence,
        "processed_model_frames": stats.model_frames,
        "latest_sequence": latest.sequence,
        "latest_clip_frames": latest.clip_frames,
        "latest_model_frames": latest.model_frames,
        "latest_source_ms": latest.source_ms,
        "latest_prepare_ms": latest.prepare_ms,
        "latest_pack_ms": latest.pack_ms,
        "latest_input_ms": latest.input_ms,
        "latest_display_input_ms": latest.display_input_ms,
        "latest_total_ms": latest.total_ms,
        "latest_model_ms": latest.model_ms,
        "latest_trace_ms": latest.trace_ms,
        "latest_sync_ms": latest.sync_ms,
        "latest_visualize_cpu_ms": latest.visualize_cpu_ms,
        "latest_psnr_ms": latest.psnr_ms,
        "latest_tensor_ms": latest.tensor_ms,
        "latest_visualize_ms": latest.visualize_ms,
        "latest_display_ms": latest.display_ms,
        "latest_effective_generation_budget": latest.effective_generation_budget,
        "latest_generated_tokens": latest.generated_tokens,
        "latest_active_generated_tokens": latest.active_generated_tokens,
        "latest_padded_generated_tokens": latest.padded_generated_tokens,
        "latest_trace_points": latest.trace_points,
        "latest_active_trace_points": latest.active_trace_points,
        "latest_mask_rects": latest.mask_plan_stats.rect_count,
        "latest_mask_row_spans": latest.mask_plan_stats.row_span_count,
        "latest_mask_pixels": latest.mask_plan_stats.pixel_count,
        "latest_gaze_update_ratio": stats.latest_gaze_update_ratio.unwrap_or_default(),
        "latest_mask_update_ratio": stats.latest_gaze_update_ratio.unwrap_or_default(),
        "latest_output_update_ratio": stats.latest_output_update_ratio.unwrap_or_default(),
        "latest_tensor_interframe_path": latest.tensor_interframe_path.map(|path| path.as_str()),
        "latest_output_rgba_bytes": latest.output_rgba_bytes,
        "latest_output_tensor_bytes": latest.output_tensor_bytes,
        "latest_effective_display_transfer": latest.effective_display_transfer.as_str(),
        "display_residency": display_residency(latest),
        "display_input_residency": latest.display_input_residency.as_str(),
        "latest_width": latest.width,
        "latest_height": latest.height,
        "render_adapter_name": stats.render_adapter.as_ref().map(|adapter| adapter.name.as_str()),
        "render_adapter_vendor": stats.render_adapter.as_ref().map(|adapter| adapter.vendor),
        "render_adapter_device_type": stats.render_adapter.as_ref().map(|adapter| adapter.device_type.as_str()),
        "render_adapter_backend": stats.render_adapter.as_ref().map(|adapter| adapter.backend.as_str()),
        "render_adapter_driver": stats.render_adapter.as_ref().map(|adapter| adapter.driver.as_str()),
        "render_adapter_driver_info": stats.render_adapter.as_ref().map(|adapter| adapter.driver_info.as_str()),
        "avg_output_fps": avg_output_fps,
        "avg_model_frame_fps": avg_model_frame_fps,
        "avg_input_fps": avg_input_fps,
        "avg_trace_points": avg_trace_points,
        "avg_active_trace_points": avg_active_trace_points,
        "avg_generated_tokens": avg_generated_tokens,
        "avg_active_generated_tokens": avg_active_generated_tokens,
        "avg_padded_generated_tokens": avg_padded_generated_tokens,
        "avg_mask_rects": avg_mask_rects,
        "avg_mask_row_spans": avg_mask_row_spans,
        "avg_mask_pixels": avg_mask_pixels,
        "avg_display_input_ms": mean_or_zero(stats.display_input_ms, stats.processed_frames()),
        "avg_psnr_ms": mean_or_zero(stats.psnr_ms, stats.processed_frames()),
        "avg_output_rgba_bytes": mean_or_zero(stats.output_rgba_bytes as f64, stats.processed_frames()),
        "avg_output_tensor_bytes": mean_or_zero(stats.output_tensor_bytes as f64, stats.processed_frames()),
        "avg_gaze_update_ratio": mean_or_zero(stats.gaze_update_ratio, stats.gaze_update_samples),
        "avg_mask_update_ratio": mean_or_zero(stats.gaze_update_ratio, stats.gaze_update_samples),
        "avg_output_update_ratio": mean_or_zero(stats.output_update_ratio, stats.output_update_samples),
        "psnr_samples": stats.psnr_samples,
        "latest_psnr_db": psnr_metric_json_value(&stats.psnr_stats, PsnrMetricKind::Current),
        "latest_psnr_db_infinite": psnr_metric_is_infinite(&stats.psnr_stats, PsnrMetricKind::Current),
        "ema_psnr_db": psnr_metric_json_value(&stats.psnr_stats, PsnrMetricKind::Ema),
        "ema_psnr_db_infinite": psnr_metric_is_infinite(&stats.psnr_stats, PsnrMetricKind::Ema),
        "p50_total_ms": percentile_ms(&stats.samples, 0.50),
        "p95_total_ms": p95_total_ms,
        "p99_total_ms": p99_total_ms,
        "max_total_ms": max_total_ms,
        "p05_output_fps": fps_from_millis(p95_total_ms),
        "worst_output_fps": fps_from_millis(max_total_ms),
        "fps_stability_p05_to_avg": if p95_total_ms > 0.0 {
            mean_or_zero(stats.total_ms, stats.processed_frames()) / p95_total_ms
        } else {
            0.0
        },
        "p95_model_ms": percentile_ms(&stats.model_samples, 0.95),
        "p99_model_ms": percentile_ms(&stats.model_samples, 0.99),
        "max_model_ms": max_ms(&stats.model_samples),
        "p95_source_ms": percentile_ms(&stats.source_samples, 0.95),
        "p95_prepare_ms": percentile_ms(&stats.prepare_samples, 0.95),
        "p95_pack_ms": percentile_ms(&stats.pack_samples, 0.95),
        "p95_input_ms": percentile_ms(&stats.input_samples, 0.95),
        "p95_display_input_ms": percentile_ms(&stats.display_input_samples, 0.95),
        "p95_visualize_ms": percentile_ms(&stats.visualize_samples, 0.95),
        "p95_visualize_cpu_ms": percentile_ms(&stats.visualize_cpu_samples, 0.95),
        "p95_psnr_ms": percentile_ms(&stats.psnr_ms_samples, 0.95),
        "p95_tensor_ms": percentile_ms(&stats.tensor_samples, 0.95),
        "p95_display_ms": percentile_ms(&stats.display_samples, 0.95),
        "stale_results": stats.stale_results,
    });
    if let Some(fields) = sample.as_object_mut() {
        insert_run_config_json_fields(fields, stats.run_config);
    }
    Some(sample.to_string())
}

#[cfg(target_arch = "wasm32")]
fn publish_wasm_perf_sample(stats: &InferenceTimingStats) {
    use wasm_bindgen::JsValue;

    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(value) = perf_sample_json(stats) else {
        return;
    };
    let value = js_sys::JSON::parse(&value).unwrap_or_else(|_| JsValue::from_str(&value));
    let _ = js_sys::Reflect::set(&window, &JsValue::from_str("__autogazePerf"), &value);
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_wasm_perf_sample(_stats: &InferenceTimingStats) {}

#[derive(SystemParam)]
struct FrameInputParams<'w> {
    config: Res<'w, BevyBurnAutoGazeConfig>,
    static_frame: Res<'w, StaticFrame>,
    synthetic_source: ResMut<'w, SyntheticFrameSource>,
    frame_queue: ResMut<'w, FrameQueue>,
    inference_sequencer: ResMut<'w, InferenceSequencer>,
    visualization_state: ResMut<'w, BevyVisualizationState>,
    streaming_state: ResMut<'w, BevyStreamingGenerationState>,
    latest_mask: Res<'w, LatestMaskPrediction>,
    gaze_ratio_stats: ResMut<'w, GazeRatioStats>,
    psnr_stats: ResMut<'w, PsnrStats>,
    timing_stats: Res<'w, InferenceTimingStats>,
}

#[derive(Component)]
struct ProcessAutoGaze(Task<CommandQueue>);

pub fn viewer_app(mut config: BevyBurnAutoGazeConfig) -> App {
    config.sanitize();
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
    app.insert_resource(BevyVisualizationState::new(
        config.visualization_mode,
        config.keyframe_duration,
    ));
    app.insert_resource(BevyStreamingGenerationState::default());
    app.insert_resource(GazeRatioStats::default());
    app.insert_resource(PsnrStats::default());
    app.insert_resource(TaskLossSliderState::new(&config));
    app.insert_resource(LatestMaskPrediction::default());
    app.insert_resource(InferenceTimingStats::default());
    app.insert_resource(InferenceSequencer::default());
    app.insert_resource(AutoGazeModelState {
        config: config.clone(),
        pipeline: None,
        load_task: None,
    });
    app.insert_resource(load_static_frame(config.image_path.as_deref(), &config));
    app.insert_resource(SyntheticFrameSource::default());

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
    app.add_systems(First, clear_completed_gpu_uploads);

    if config.press_esc_to_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.register_diagnostic(Diagnostic::new(MODEL_FPS));
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

    if config.show_task_loss_slider {
        app.add_systems(Startup, task_loss_slider_display_setup);
        app.add_systems(Update, task_loss_slider_style_system);
    }

    app.add_systems(
        Update,
        (
            setup_ui,
            enforce_required_hardware_adapter,
            begin_model_load,
            finish_model_load,
            task_loss_slider_update_system,
            mask_source_toggle_system,
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

pub fn run_app(config: BevyBurnAutoGazeConfig) -> AppExit {
    let exit = viewer_app(config).run();

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(sender) = platform::camera::APP_RUN_SENDER.get() {
        let _ = sender.send(());
    }

    exit
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
    texture.input_image = images.add(visualization_image(1, 1, vec![0, 0, 0, 255]));
    texture.mask_image = images.add(visualization_image(1, 1, vec![0, 0, 0, 255]));
    texture.output_image = images.add(visualization_image(1, 1, vec![0, 0, 0, 255]));

    let mut root = commands.spawn(Node {
        position_type: PositionType::Absolute,
        display: Display::Grid,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        align_items: AlignItems::Center,
        justify_items: JustifyItems::Center,
        grid_template_columns: RepeatedGridTrack::flex(3, 1.0),
        grid_template_rows: vec![GridTrack::px(PANEL_LABEL_ROW_HEIGHT), GridTrack::flex(1.0)],
        ..default()
    });
    let root_entity = root.id();

    let mut side_by_side_entity = None;
    let mut input_entity = None;
    let mut mask_entity = None;
    let mut output_entity = None;
    root.with_children(|builder| {
        side_by_side_entity = Some(
            builder
                .spawn((
                    ImageNode::new(texture.image.clone()).with_mode(NodeImageMode::Stretch),
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        grid_column: GridPlacement::span(3),
                        grid_row: GridPlacement::start(2),
                        ..default()
                    },
                ))
                .id(),
        );

        input_entity = Some(spawn_panel_image(builder, texture.input_image.clone()));
        mask_entity = Some(spawn_panel_image(builder, texture.mask_image.clone()));
        output_entity = Some(spawn_panel_image(builder, texture.output_image.clone()));

        for label in ["Input", "Mask", "Output"] {
            builder.spawn((
                Text(label.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(24.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    grid_row: GridPlacement::start(1),
                    align_self: AlignSelf::Center,
                    justify_self: JustifySelf::Center,
                    padding: UiRect::horizontal(Val::Px(8.0)),
                    ..default()
                },
            ));
        }
    });

    texture.entity = Some(root_entity);
    texture.side_by_side_entity = side_by_side_entity;
    texture.input_entity = input_entity;
    texture.mask_entity = mask_entity;
    texture.output_entity = output_entity;
    commands.spawn(Camera2d);
}

fn spawn_panel_image(builder: &mut ChildSpawnerCommands<'_>, image: Handle<Image>) -> Entity {
    builder
        .spawn((
            ImageNode::new(image).with_mode(NodeImageMode::Stretch),
            Node {
                display: Display::None,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_row: GridPlacement::start(2),
                ..default()
            },
        ))
        .id()
}

fn fit_visualization_node(
    config: Res<BevyBurnAutoGazeConfig>,
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
    let reserved_top = metric_panel_top_reserved_height(&config);
    let available_height = (window.resolution.height().max(1.0) - reserved_top).max(1.0);
    let available_image_height = (available_height - PANEL_LABEL_ROW_HEIGHT).max(1.0);
    let source_aspect = source_width / source_height;
    let window_aspect = available_width / available_image_height;
    let (display_width, display_height) = if window_aspect > source_aspect {
        let height = available_image_height;
        (height * source_aspect, height)
    } else {
        let width = available_width;
        (width, width / source_aspect)
    };

    let total_height = display_height + PANEL_LABEL_ROW_HEIGHT;
    node.width = Val::Px(display_width.max(1.0));
    node.height = Val::Px(total_height.max(1.0));
    node.left = Val::Px(((available_width - display_width) * 0.5).max(0.0));
    node.top = Val::Px(reserved_top + ((available_height - total_height) * 0.5).max(0.0));
}

fn begin_model_load(mut state: ResMut<AutoGazeModelState>, burn_device: Option<Res<BurnDevice>>) {
    if !state.config.sparse_mask_source.requires_autogaze_model() {
        return;
    }
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
    mut streaming_state: ResMut<BevyStreamingGenerationState>,
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
                streaming_state.reset();
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
    let use_autogaze_model = frame_input
        .config
        .sparse_mask_source
        .requires_autogaze_model();
    let pipeline = if use_autogaze_model {
        let Some(pipeline) = model.pipeline.as_ref() else {
            return;
        };
        Some(pipeline.clone())
    } else {
        None
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
    let realtime_policy = realtime_policy_from_config(&frame_input.config);
    let active_task_count = active_tasks.iter().count();
    let log_pipeline_timing = frame_input.config.log_pipeline_timing;
    if !realtime_policy.should_start_inference(active_task_count) {
        return;
    }
    if frame_input
        .config
        .perf_summary_frames
        .is_some_and(|target| frame_input.timing_stats.processed_frames() >= target)
    {
        return;
    }

    let mode = frame_input.config.mode.inference_mode();
    let use_streaming_cache = should_use_streaming_cache(
        frame_input.config.streaming_cache,
        frame_input.config.frames_per_clip,
        mode,
    );
    let use_patch_diff = frame_input.config.sparse_mask_source == BevySparseMaskSource::PatchDiff;
    let clip_frames = if use_patch_diff {
        2
    } else {
        frame_input.config.frames_per_clip
    };
    if log_pipeline_timing {
        log(&format!(
            "starting {} inference: active_tasks={active_task_count} queue_len={} clip_frames={clip_frames}",
            if use_patch_diff {
                "patch-diff"
            } else {
                "autogaze"
            },
            frame_input.frame_queue.len()
        ));
    }
    let mut clip = match if use_patch_diff {
        frame_input.frame_queue.build_clip(clip_frames)
    } else if use_streaming_cache {
        frame_input.frame_queue.build_latest_clip()
    } else {
        frame_input.frame_queue.build_clip(clip_frames)
    } {
        Ok(Some(clip)) => clip,
        Ok(None) => {
            if log_pipeline_timing {
                log(&format!(
                    "{} inference waiting for clip: queue_len={} clip_frames={clip_frames}",
                    if use_patch_diff {
                        "patch-diff"
                    } else {
                        "autogaze"
                    },
                    frame_input.frame_queue.len()
                ));
            }
            return;
        }
        Err(err) => {
            log(&format!("failed to pack AutoGaze clip: {err}"));
            return;
        }
    };
    let (source_ms, prepare_ms) = frame_input.frame_queue.latest_timing();
    clip.source_ms = source_ms;
    clip.prepare_ms = prepare_ms;

    let task_entity = commands.spawn_empty().id();
    let top_k = frame_input.config.top_k.max(1);
    let perf_summary_warmup_frames = frame_input.config.perf_summary_warmup_frames;
    let perf_trace_path = frame_input.config.perf_trace_path.clone();
    let context_frames = clip_frames.max(1);
    let patch_diff_config = AutoGazePatchDiffConfig::new(
        frame_input.config.patch_diff_grid_size,
        frame_input.config.patch_diff_threshold,
    );
    let visualization_options = VisualizationOptions::new(
        frame_input.config.mask_cell_scale,
        frame_input.config.blend_alpha,
        frame_input.config.show_psnr,
        frame_input.config.display_transfer,
    )
    .with_sparse_update_policy(
        frame_input.config.tensor_sparse_update_max_rects,
        frame_input.config.tensor_sparse_update_max_ratio,
    )
    .with_full_frame_update_policy(frame_input.config.tensor_full_frame_update_min_ratio)
    .with_mask_visualization_mode(frame_input.config.mask_visualization_mode)
    .with_mask_geometry_mode(frame_input.config.mask_geometry_mode)
    .with_cpu_panels();
    let run_config = InferenceRunConfigSummary::from(frame_input.config.as_ref());
    let completed_display_action =
        completed_run_display_action(realtime_policy, use_patch_diff, active_task_count + 1);
    frame_input.visualization_state.configure(
        frame_input.config.visualization_mode,
        frame_input.config.keyframe_duration,
    );
    let visualization_state = frame_input.visualization_state.clone();
    if !*logged_first_inference {
        if use_patch_diff {
            log(
                "Patch-diff sparse mask inference started; the first native run may spend time tuning GPU kernels",
            );
        } else {
            log(
                "AutoGaze inference started; the first native run may spend time tuning GPU kernels",
            );
        }
        *logged_first_inference = true;
    }
    let sequence = frame_input.inference_sequencer.reserve();
    frame_input.streaming_state.configure(
        use_streaming_cache && !use_patch_diff,
        clip.width(),
        clip.height(),
        context_frames,
    );
    let streaming_state = frame_input.streaming_state.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let job = AutoGazeRunContext {
            clip: &clip,
            sequence,
            streaming_state,
            use_streaming_cache,
            context_frames,
            top_k,
            mode,
            visualization_options,
            visualization_state,
            device,
            log_pipeline_timing,
        };

        let result = if let Some(pipeline) = pipeline {
            run_autogaze_visualization(pipeline, job).await
        } else {
            run_patch_diff_visualization(job, patch_diff_config).await
        };
        let clip_rgba = clip.into_rgba();

        let mut queue = CommandQueue::default();
        queue.push(move |world: &mut World| {
            if let Some(mut frame_queue) = world.get_resource_mut::<FrameQueue>() {
                frame_queue.recycle_clip_buffer(clip_rgba);
            }
            if let Some(mut sequencer) = world.get_resource_mut::<InferenceSequencer>()
                && !sequencer.accept(sequence)
            {
                if let Some(mut stats) = world.get_resource_mut::<InferenceTimingStats>() {
                    stats.record_stale_result();
                }
                if let Ok(mut tracker) = world.get_entity_mut(task_entity) {
                    tracker.remove::<ProcessAutoGaze>();
                    tracker.despawn();
                }
                return;
            }

            match result {
                Ok((visualization, visualization_state, streaming_state, points)) => {
                    let Visualization {
                        width,
                        height,
                        image_data,
                        gaze_update_ratio,
                        output_update_ratio: _,
                        interframe_keyframe,
                        psnr_db,
                        mut timing,
                        ..
                    } = visualization;
                    let display_ms = apply_completed_model_visualization(
                        world,
                        width,
                        height,
                        image_data,
                        visualization_state,
                        completed_display_action,
                    );
                    if let Some(ref mut timing) = timing {
                        timing.display_ms = display_ms;
                        timing.total_ms += timing.display_ms;
                    }

                    if let Some(mut state) =
                        world.get_resource_mut::<BevyStreamingGenerationState>()
                    {
                        *state = streaming_state;
                    }

                    if let Some(mut latest_mask) = world.get_resource_mut::<LatestMaskPrediction>()
                    {
                        latest_mask.update(points);
                    }

                    if completed_display_action.displays_visualization()
                        && !interframe_keyframe
                        && let Some(mut stats) = world.get_resource_mut::<GazeRatioStats>()
                    {
                        stats.record(gaze_update_ratio);
                    }

                    if completed_display_action.displays_visualization()
                        && let Some(psnr_db) = psnr_db
                        && let Some(mut stats) = world.get_resource_mut::<PsnrStats>()
                    {
                        stats.record(psnr_db);
                    }

                    if let Some(timing) = timing {
                        let render_adapter = world
                            .get_resource::<RenderAdapterInfo>()
                            .map(RenderAdapterSummary::from);
                        if let Some(mut stats) = world.get_resource_mut::<InferenceTimingStats>() {
                            stats.set_run_config(run_config);
                            if let Some(render_adapter) = render_adapter {
                                stats.set_render_adapter(render_adapter);
                            }
                            if stats.skipped_warmup_frames < perf_summary_warmup_frames {
                                stats.skip_warmup_frame(timing);
                            } else {
                                stats.record(timing, log_pipeline_timing);
                                if perf_trace_path.is_some()
                                    && let Some(sample) = perf_sample_json(&stats)
                                {
                                    let truncate = stats.processed_frames() == 1;
                                    if let Err(err) = write_perf_trace_sample(
                                        perf_trace_path.as_deref(),
                                        &sample,
                                        truncate,
                                    ) {
                                        log(&format!("failed to write AutoGaze perf trace: {err}"));
                                    }
                                }
                            }
                        }
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

fn apply_completed_model_visualization(
    world: &mut World,
    width: u32,
    height: u32,
    image_data: VisualizationImageData,
    visualization_state: BevyVisualizationState,
    action: CompletedModelDisplayAction,
) -> f64 {
    if !action.displays_visualization() {
        return 0.0;
    }

    let display_start = timestamp_now();
    apply_visualization_to_world(world, width, height, image_data);

    if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
        texture.width = width;
        texture.height = height;
    }

    if let Some(mut state) = world.get_resource_mut::<BevyVisualizationState>() {
        *state = visualization_state;
    }

    elapsed_ms(display_start)
}

fn preview_frames(
    model: Res<AutoGazeModelState>,
    mut texture: ResMut<AutoGazeTexture>,
    mut frame_input: FrameInputParams,
    active_tasks: Query<&ProcessAutoGaze>,
    mut images: ResMut<Assets<Image>>,
    mut nodes: Query<&mut Node>,
) {
    let model_ready = model.pipeline.is_some()
        || !frame_input
            .config
            .sparse_mask_source
            .requires_autogaze_model();
    let realtime_policy = realtime_policy_from_config(&frame_input.config);
    let active_task_count = active_tasks.iter().count();
    if texture.entity.is_none() {
        return;
    }

    let frame = next_source_frame(
        &frame_input.config,
        &frame_input.static_frame,
        &mut frame_input.synthetic_source,
    );

    let Some((frame, source_ms, prepare_ms)) = frame else {
        return;
    };

    let frame_queue_len = frame_queue_len_for_config(&frame_input.config);
    frame_input
        .frame_queue
        .push_timed(frame, frame_queue_len, source_ms, prepare_ms);
    let Some(frame) = frame_input.frame_queue.latest() else {
        return;
    };

    if !realtime_policy.should_draw_async_stream_preview(model_ready, active_task_count) {
        return;
    }

    let latest_points = if model_ready {
        frame_input.latest_mask.points().to_vec()
    } else {
        Vec::new()
    };
    let visualization = if model_ready {
        async_mask_preview_visualization(
            frame,
            &latest_points,
            &frame_input.config,
            &mut frame_input.visualization_state,
        )
    } else {
        live_preview_visualization(frame, frame_input.config.show_psnr)
    };
    let visualization = match visualization {
        Ok(visualization) => visualization,
        Err(err) => {
            log(&format!("failed to draw AutoGaze preview: {err}"));
            return;
        }
    };
    if !visualization.interframe_keyframe {
        frame_input
            .gaze_ratio_stats
            .record(visualization.gaze_update_ratio);
    }
    if let Some(psnr_db) = visualization.psnr_db {
        frame_input.psnr_stats.record(psnr_db);
    }
    apply_visualization_to_preview_display(visualization, &mut texture, &mut images, &mut nodes);
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
                    diagnostics.add_measurement(&MODEL_FPS, || 1.0 / delta_seconds);
                }
            }
            *last_frame = Some(now);
            commands.append(&mut queue);
        }
    }
}

fn clear_completed_gpu_uploads(
    mut commands: Commands,
    mut query: Query<(Entity, &mut BevyBurnHandle<AutoGazeBevyBackend>), With<OneShotGpuUpload>>,
) {
    for (entity, mut handle) in &mut query {
        handle.upload = false;
        commands.entity(entity).remove::<OneShotGpuUpload>();
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
    pipeline.apply_options(pipeline_options_from_config(&config));
    warmup_pipeline_if_enabled(&config, &pipeline, device).await?;
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
    pipeline.apply_options(pipeline_options_from_config(&config));
    warmup_pipeline_if_enabled(&config, &pipeline, device).await?;
    Ok(pipeline)
}

async fn warmup_pipeline_if_enabled(
    config: &BevyBurnAutoGazeConfig,
    pipeline: &AutoGazePipeline<AutoGazeBevyBackend>,
    device: &AutoGazeBevyDevice,
) -> Result<(), String> {
    if !config.warmup_model {
        return Ok(());
    }

    let warmup_start = timestamp_now();
    let (width, height) = warmup_dimensions(config);
    let frames_per_clip = config.frames_per_clip.max(1);
    let use_streaming_cache = should_use_streaming_cache(
        config.streaming_cache,
        frames_per_clip,
        config.mode.inference_mode(),
    );
    let clip_len = if use_streaming_cache {
        1
    } else {
        frames_per_clip
    };
    let mut cache = AutoGazeStreamingCache::new(frames_per_clip);
    let warmup_pipeline = pipeline.clone();
    let warmup_runs = model_warmup_runs(config, use_streaming_cache);
    for run_idx in 0..warmup_runs {
        let rgba = warmup_rgba_clip(config.source, width, height, clip_len, run_idx, warmup_runs)?;
        let shape = AutoGazeRgbaClipShape::new(clip_len, height, width);
        let prepared = prepare_rgba_clip_for_trace::<AutoGazeBevyBackend>(
            &rgba,
            shape,
            config.mode.inference_mode(),
            use_streaming_cache,
            device,
        )
        .map_err(|err| format!("failed to prepare AutoGaze warmup input: {err:#}"))?;
        let cache = use_streaming_cache.then_some(&mut cache);
        warmup_pipeline
            .readout_prepared_run_async(prepared, config.top_k.max(1), cache)
            .await
            .map_err(|err| format!("failed to warm AutoGaze model: {err:?}"))?;
    }
    sync_warmup_backend(device)?;
    log(&format!(
        "AutoGaze model warmup complete in {:.1} ms ({warmup_runs} runs)",
        elapsed_ms(warmup_start),
    ));
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_warmup_backend(device: &AutoGazeBevyDevice) -> Result<(), String> {
    <AutoGazeBevyBackend as burn::tensor::backend::Backend>::sync(device)
        .map_err(|err| format!("failed to sync AutoGaze warmup: {err:?}"))
}

#[cfg(target_arch = "wasm32")]
fn sync_warmup_backend(_device: &AutoGazeBevyDevice) -> Result<(), String> {
    Ok(())
}

fn model_warmup_runs(config: &BevyBurnAutoGazeConfig, use_streaming_cache: bool) -> usize {
    let base_runs = match config.mode {
        BevyAutoGazeMode::Resize224 => DEFAULT_REALTIME_MODEL_WARMUP_RUNS,
        BevyAutoGazeMode::Tile224 => DEFAULT_TILED_MODEL_WARMUP_RUNS,
    };
    if use_streaming_cache {
        return base_runs.max(
            config
                .frames_per_clip
                .saturating_add(DEFAULT_STREAMING_MODEL_WARMUP_EXTRA_RUNS),
        );
    }

    base_runs
}

fn warmup_dimensions(config: &BevyBurnAutoGazeConfig) -> (usize, usize) {
    let (default_width, default_height) = default_inference_dimensions(config.mode);
    let width = config
        .inference_width
        .or(default_width)
        .unwrap_or(DEFAULT_REALTIME_INFERENCE_WIDTH)
        .max(1) as usize;
    let height = config
        .inference_height
        .or(default_height)
        .unwrap_or_else(|| ((width as f64 * 9.0 / 16.0).round() as u32).max(1))
        .max(1) as usize;
    (width, height)
}

fn warmup_rgba_clip(
    source: BevyFrameSource,
    width: usize,
    height: usize,
    clip_len: usize,
    seed: usize,
    warmup_runs: usize,
) -> Result<Vec<u8>, String> {
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "AutoGaze warmup byte length overflow".to_string())?;
    let mut rgba = Vec::with_capacity(frame_bytes.saturating_mul(clip_len));
    for frame_idx in 0..clip_len {
        let frame_seed = warmup_frame_seed(source, seed, frame_idx, clip_len, warmup_runs);
        let frame = warmup_frame_for_source(source, width, height, frame_seed)?;
        rgba.extend_from_slice(frame.as_raw());
    }
    Ok(rgba)
}

fn warmup_frame_seed(
    source: BevyFrameSource,
    run_idx: usize,
    frame_idx: usize,
    clip_len: usize,
    warmup_runs: usize,
) -> usize {
    let ordinal = run_idx.saturating_mul(clip_len).saturating_add(frame_idx);
    if source != BevyFrameSource::SyntheticLocalMotion {
        return ordinal;
    }

    let total = warmup_runs.saturating_mul(clip_len).max(1);
    ((ordinal as u128 * u128::from(SYNTHETIC_LOCAL_CYCLE_FRAMES)) / total as u128) as usize
}

fn warmup_frame_for_source(
    source: BevyFrameSource,
    width: usize,
    height: usize,
    seed: usize,
) -> Result<RgbaImage, String> {
    let width_u32 = u32::try_from(width).map_err(|_| "AutoGaze warmup width overflow")?;
    let height_u32 = u32::try_from(height).map_err(|_| "AutoGaze warmup height overflow")?;
    let frame_index = seed as u64;
    match source {
        BevyFrameSource::SyntheticPan => {
            Ok(synthetic_pan_frame(width_u32, height_u32, frame_index))
        }
        BevyFrameSource::SyntheticPulse => {
            Ok(synthetic_pulse_frame(width_u32, height_u32, frame_index))
        }
        BevyFrameSource::SyntheticLocalMotion => Ok(synthetic_local_motion_frame(
            width_u32,
            height_u32,
            frame_index,
        )),
        BevyFrameSource::Camera | BevyFrameSource::StaticImage => warmup_frame(width, height, seed),
    }
}

fn warmup_frame(width: usize, height: usize, seed: usize) -> Result<RgbaImage, String> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "AutoGaze warmup dimensions overflow".to_string())?;
    let bytes = pixels
        .checked_mul(4)
        .ok_or_else(|| "AutoGaze warmup byte length overflow".to_string())?;
    let mut rgba = vec![0_u8; bytes];
    for (idx, pixel) in rgba.chunks_exact_mut(4).enumerate() {
        let x = idx % width;
        let y = idx / width;
        pixel[0] = ((x * 13 + y * 7 + seed * 29) & 0xff) as u8;
        pixel[1] = ((x * 3 + y * 17 + seed * 31) & 0xff) as u8;
        pixel[2] = (((x / 8 + y / 8 + seed) & 1) * 255) as u8;
        pixel[3] = 255;
    }
    RgbaImage::from_raw(width as u32, height as u32, rgba)
        .ok_or_else(|| "failed to build AutoGaze warmup frame".to_string())
}

fn pipeline_options_from_config(config: &BevyBurnAutoGazeConfig) -> AutoGazePipelineOptions {
    let mut options =
        AutoGazePipelineOptions::default().with_tile_batch_size(config.tile_batch_size.max(1));
    if config.max_gaze_tokens_each_frame > 0 {
        options = options.with_max_gaze_tokens_each_frame(config.max_gaze_tokens_each_frame);
    }
    if config.disable_task_loss_requirement {
        options = options.without_task_loss_requirement();
    } else if let Some(task_loss_requirement) = config.task_loss_requirement {
        options = options.with_task_loss_requirement(task_loss_requirement);
    }
    if matches!(
        config.visualization_mode,
        AutoGazeVisualizationMode::Interframe
    ) {
        options =
            options.with_generation_coverage_stop_ratio(config.tensor_full_frame_update_min_ratio);
    }
    options = options.with_decode_strategy(config.decode_strategy);
    options
}

struct AutoGazeRunContext<'a> {
    clip: &'a FrameClip,
    sequence: u64,
    streaming_state: BevyStreamingGenerationState,
    use_streaming_cache: bool,
    context_frames: usize,
    top_k: usize,
    mode: AutoGazeInferenceMode,
    visualization_options: VisualizationOptions,
    visualization_state: BevyVisualizationState,
    device: AutoGazeBevyDevice,
    log_pipeline_timing: bool,
}

struct PreparedAutoGazeRun {
    trace_input: CoreAutoGazePreparedRun<AutoGazeBevyBackend>,
    visualization: PreparedVisualizationRun,
}

struct PreparedVisualizationRun {
    visualization_tensor: Option<Tensor<AutoGazeBevyBackend, 5>>,
    input_ms: f64,
    display_input_ms: f64,
    display_input_residency: DisplayInputResidency,
}

struct PreparedDisplayTensor {
    tensor: Option<Tensor<AutoGazeBevyBackend, 5>>,
    ms: f64,
    residency: DisplayInputResidency,
}

fn prepare_autogaze_run(
    clip: &FrameClip,
    use_streaming_cache: bool,
    mode: AutoGazeInferenceMode,
    visualization_options: VisualizationOptions,
    device: &AutoGazeBevyDevice,
) -> Result<PreparedAutoGazeRun, String> {
    let input_start = timestamp_now();
    let core = prepare_rgba_clip_for_trace::<AutoGazeBevyBackend>(
        clip.rgba(),
        clip.shape(),
        mode,
        use_streaming_cache,
        device,
    )
    .map_err(|err| format!("{err:#}"))?;
    let input_ms = elapsed_ms(input_start);
    let display = display_tensor_from_prepared_trace(clip, &core, visualization_options, device)?;

    Ok(PreparedAutoGazeRun {
        trace_input: core,
        visualization: PreparedVisualizationRun {
            visualization_tensor: display.tensor,
            input_ms,
            display_input_ms: display.ms,
            display_input_residency: display.residency,
        },
    })
}

struct FinishedReadout {
    points: Vec<Vec<Vec<FixationPoint>>>,
    device_tokens: Option<AutoGazeDeviceTokens<AutoGazeBevyBackend>>,
    device_mask: Option<AutoGazeDeviceMask<AutoGazeBevyBackend>>,
    model_config: Option<AutoGazeConfig>,
    frame_index: usize,
    model_frames: usize,
    model_ms: f64,
    effective_generation_budget: usize,
    generated_tokens: usize,
    active_generated_tokens: usize,
    padded_generated_tokens: usize,
}

fn finished_readout_from_run_output(
    output: AutoGazeReadoutRunOutput,
    model_ms: f64,
    effective_generation_budget: usize,
) -> FinishedReadout {
    FinishedReadout {
        points: output.points,
        device_tokens: None,
        device_mask: None,
        model_config: None,
        frame_index: output.frame_index,
        model_frames: output.model_frames,
        model_ms,
        effective_generation_budget,
        generated_tokens: output.stats.generated_tokens,
        active_generated_tokens: output.stats.active_generated_tokens,
        padded_generated_tokens: output.stats.padded_generated_tokens,
    }
}

fn finished_readout_from_device_run_output(
    output: AutoGazeDeviceReadoutRunOutput<AutoGazeBevyBackend>,
    model_ms: f64,
    effective_generation_budget: usize,
) -> FinishedReadout {
    FinishedReadout {
        points: output.points,
        device_tokens: output.device_tokens,
        device_mask: None,
        model_config: None,
        frame_index: output.frame_index,
        model_frames: output.model_frames,
        model_ms,
        effective_generation_budget,
        generated_tokens: output.stats.generated_tokens,
        active_generated_tokens: output.stats.active_generated_tokens,
        padded_generated_tokens: output.stats.padded_generated_tokens,
    }
}

async fn finish_autogaze_visualization(
    context: AutoGazeRunContext<'_>,
    prepared: PreparedVisualizationRun,
    finished: FinishedReadout,
    total_start: Timestamp,
) -> Result<
    (
        Visualization,
        BevyVisualizationState,
        BevyStreamingGenerationState,
        Vec<FixationPoint>,
    ),
    String,
> {
    let AutoGazeRunContext {
        clip,
        sequence,
        streaming_state,
        context_frames,
        visualization_options,
        mut visualization_state,
        device,
        ..
    } = context;
    let width = clip.width();
    let height = clip.height();
    let FinishedReadout {
        points: batch_points,
        device_tokens,
        device_mask,
        model_config,
        frame_index,
        model_frames,
        model_ms,
        effective_generation_budget,
        generated_tokens,
        active_generated_tokens,
        padded_generated_tokens,
    } = finished;
    let trace_ms = prepared.input_ms + model_ms;
    let points = batch_points
        .first()
        .and_then(|frames| frames.get(frame_index))
        .cloned()
        .unwrap_or_default();
    let active_trace_points = points.iter().filter(|point| point.confidence > 0.0).count();
    let calculate_psnr = visualization_options.calculate_psnr;
    let visualize_start = timestamp_now();
    let mut visualization = visualize_frame_rgba_with_device_tokens(
        FrameVisualInput {
            rgba: clip.last_frame_rgba()?,
            width,
            height,
            tensor: prepared.visualization_tensor,
        },
        &points,
        device_tokens.as_ref(),
        device_mask,
        model_config.as_ref(),
        visualization_options,
        &mut visualization_state,
        &device,
    )?;
    calculate_tensor_psnr_if_needed(&mut visualization, calculate_psnr).await?;
    visualization.timing = Some(InferenceTiming {
        sequence,
        clip_frames: context_frames,
        model_frames,
        effective_generation_budget,
        generated_tokens,
        active_generated_tokens,
        padded_generated_tokens,
        trace_points: points.len(),
        active_trace_points,
        width,
        height,
        source_ms: clip.source_ms,
        prepare_ms: clip.prepare_ms,
        pack_ms: clip.pack_ms,
        input_ms: prepared.input_ms,
        display_input_ms: prepared.display_input_ms,
        model_ms,
        trace_ms,
        sync_ms: 0.0,
        visualize_cpu_ms: visualization.visualize_cpu_ms,
        psnr_ms: visualization.psnr_ms,
        tensor_ms: visualization.tensor_ms,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start) + clip.source_ms + clip.prepare_ms + clip.pack_ms,
        output_rgba_bytes: visualization.output_rgba_bytes,
        output_tensor_bytes: visualization.output_tensor_bytes,
        display_input_residency: prepared.display_input_residency,
        effective_display_transfer: visualization.effective_display_transfer,
        gaze_update_ratio: visualization.gaze_update_ratio,
        gaze_update_ratio_sample: (!visualization.interframe_keyframe)
            .then_some(visualization.gaze_update_ratio),
        output_update_ratio: visualization.output_update_ratio,
        output_update_ratio_sample: (!visualization.interframe_keyframe)
            .then_some(visualization.output_update_ratio),
        psnr_db: visualization.psnr_db,
        tensor_interframe_path: visualization.tensor_interframe_path,
        mask_plan_stats: visualization.mask_plan_stats,
    });
    Ok((visualization, visualization_state, streaming_state, points))
}

async fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    mut context: AutoGazeRunContext<'_>,
) -> Result<
    (
        Visualization,
        BevyVisualizationState,
        BevyStreamingGenerationState,
        Vec<FixationPoint>,
    ),
    String,
> {
    let AutoGazeRunContext {
        clip,
        ref mut streaming_state,
        use_streaming_cache,
        top_k,
        mode,
        visualization_options,
        ref device,
        ..
    } = context;
    let total_start = timestamp_now();
    let prepared = prepare_autogaze_run(
        clip,
        use_streaming_cache,
        mode,
        visualization_options,
        device,
    )?;
    let PreparedAutoGazeRun {
        trace_input,
        visualization,
    } = prepared;
    let finished = run_autogaze_readout(
        pipeline,
        trace_input,
        top_k,
        use_streaming_cache,
        should_use_device_token_readout(
            visualization_options.display_transfer,
            clip.width(),
            clip.height(),
            use_streaming_cache,
        ),
        streaming_state,
    )
    .await?;
    finish_autogaze_visualization(context, visualization, finished, total_start).await
}

async fn run_patch_diff_visualization(
    context: AutoGazeRunContext<'_>,
    config: AutoGazePatchDiffConfig,
) -> Result<
    (
        Visualization,
        BevyVisualizationState,
        BevyStreamingGenerationState,
        Vec<FixationPoint>,
    ),
    String,
> {
    let total_start = timestamp_now();
    let AutoGazeRunContext {
        clip,
        visualization_options,
        ref device,
        log_pipeline_timing,
        ..
    } = context;
    let input_start = timestamp_now();
    let video = rgba_clip_to_tensor::<AutoGazeBevyBackend>(clip.rgba(), clip.shape(), device)
        .map_err(|err| format!("failed to prepare patch-diff input tensor: {err:#}"))?;
    let input_ms = elapsed_ms(input_start);
    if log_pipeline_timing {
        log(&format!("patch-diff input prepared in {input_ms:.3} ms"));
    }
    let display_start = timestamp_now();
    let use_tensor_display = uses_tensor_display_transfer(
        visualization_options.display_transfer,
        clip.width(),
        clip.height(),
    );
    let display_tensor = if use_tensor_display {
        Some(
            video_frame_tensor(video.clone(), clip.shape().clip_len.saturating_sub(1))
                .map_err(|err| format!("failed to reuse patch-diff display tensor: {err:#}"))?,
        )
    } else {
        None
    };
    let display_ms = elapsed_ms(display_start);
    let display_residency = if use_tensor_display {
        DisplayInputResidency::ModelTensorReuse
    } else {
        DisplayInputResidency::None
    };
    if log_pipeline_timing {
        log(&format!(
            "patch-diff display tensor residency {display_residency:?} in {display_ms:.3} ms"
        ));
    }
    let model_start = timestamp_now();
    let finished = if use_tensor_display {
        if log_pipeline_timing {
            log("patch-diff building device mask");
        }
        let output = patch_diff_device_mask_async(video, config, clip.height(), clip.width())
            .await
            .map_err(|err| format!("failed to build patch-diff device mask: {err:#}"))?;
        FinishedReadout {
            points: output.points,
            device_tokens: None,
            device_mask: Some(output.mask),
            model_config: None,
            frame_index: output.frame_index,
            model_frames: output.model_frames,
            model_ms: elapsed_ms(model_start),
            effective_generation_budget: output.grid_size.saturating_mul(output.grid_size),
            generated_tokens: output.stats.generated_tokens,
            active_generated_tokens: output.stats.active_generated_tokens,
            padded_generated_tokens: output.stats.padded_generated_tokens,
        }
    } else {
        if log_pipeline_timing {
            log("patch-diff reading compact sparse points");
        }
        let output = patch_diff_readout_points_async(video, config)
            .await
            .map_err(|err| format!("failed to read patch-diff sparse mask: {err:#}"))?;
        finished_readout_from_run_output(
            output,
            elapsed_ms(model_start),
            config.normalized().token_budget(),
        )
    };
    if log_pipeline_timing {
        log(&format!(
            "patch-diff readout finished in {:.3} ms",
            finished.model_ms
        ));
    }
    let prepared = PreparedVisualizationRun {
        visualization_tensor: display_tensor,
        input_ms,
        display_input_ms: display_ms,
        display_input_residency: display_residency,
    };
    finish_autogaze_visualization(context, prepared, finished, total_start).await
}

async fn run_autogaze_readout(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    trace_input: CoreAutoGazePreparedRun<AutoGazeBevyBackend>,
    top_k: usize,
    use_streaming_cache: bool,
    use_device_tokens: bool,
    streaming_state: &mut BevyStreamingGenerationState,
) -> Result<FinishedReadout, String> {
    let pipeline = pipeline
        .lock()
        .map_err(|_| "AutoGaze model lock was poisoned".to_string())?
        .clone();
    let effective_generation_budget = pipeline.effective_max_gaze_tokens_each_frame();
    let model_config = pipeline.config().clone();
    let model_start = timestamp_now();
    let mut finished = if use_streaming_cache && use_device_tokens {
        let run_output = pipeline
            .device_readout_prepared_run_async(
                trace_input,
                top_k,
                Some(streaming_state.cache_mut()),
            )
            .await
            .map_err(|err| {
                format!("failed to read AutoGaze tensor data asynchronously: {err:?}")
            })?;
        finished_readout_from_device_run_output(
            run_output,
            elapsed_ms(model_start),
            effective_generation_budget,
        )
    } else if use_streaming_cache {
        let run_output = pipeline
            .readout_prepared_run_async(trace_input, top_k, Some(streaming_state.cache_mut()))
            .await
            .map_err(|err| {
                format!("failed to read AutoGaze tensor data asynchronously: {err:?}")
            })?;
        finished_readout_from_run_output(
            run_output,
            elapsed_ms(model_start),
            effective_generation_budget,
        )
    } else {
        let run_output = pipeline
            .readout_prepared_run_async(trace_input, top_k, None)
            .await
            .map_err(|err| {
                format!("failed to read AutoGaze tensor data asynchronously: {err:?}")
            })?;
        finished_readout_from_run_output(
            run_output,
            elapsed_ms(model_start),
            effective_generation_budget,
        )
    };
    finished.model_config = Some(model_config);
    Ok(finished)
}

fn should_use_device_token_readout(
    display_transfer: BevyDisplayTransfer,
    width: usize,
    height: usize,
    use_streaming_cache: bool,
) -> bool {
    use_streaming_cache && uses_tensor_display_transfer(display_transfer, width, height)
}

#[derive(Clone, Copy)]
enum CpuVisualizationLayout {
    SideBySide,
    Panels,
}

#[derive(Clone, Copy)]
struct VisualizationOptions {
    cell_scale: f32,
    blend_alpha: f32,
    mask_visualization_mode: AutoGazeMaskVisualizationMode,
    mask_geometry_mode: AutoGazeMaskGeometryMode,
    calculate_psnr: bool,
    display_transfer: BevyDisplayTransfer,
    sparse_update_max_rects: usize,
    sparse_update_max_ratio: f64,
    full_frame_update_min_ratio: f64,
    cpu_layout: CpuVisualizationLayout,
}

impl VisualizationOptions {
    fn new(
        cell_scale: f32,
        blend_alpha: f32,
        calculate_psnr: bool,
        display_transfer: BevyDisplayTransfer,
    ) -> Self {
        Self {
            cell_scale,
            blend_alpha,
            mask_visualization_mode: AutoGazeMaskVisualizationMode::ImageMaskOnly,
            mask_geometry_mode: DEFAULT_BEVY_MASK_GEOMETRY_MODE,
            calculate_psnr,
            display_transfer,
            sparse_update_max_rects: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            sparse_update_max_ratio: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
            full_frame_update_min_ratio: DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
            cpu_layout: CpuVisualizationLayout::SideBySide,
        }
    }

    fn with_sparse_update_policy(mut self, max_rects: usize, max_update_ratio: f64) -> Self {
        self.sparse_update_max_rects = max_rects;
        self.sparse_update_max_ratio = max_update_ratio;
        self
    }

    fn with_full_frame_update_policy(mut self, min_update_ratio: f64) -> Self {
        self.full_frame_update_min_ratio = min_update_ratio;
        self
    }

    fn with_mask_visualization_mode(mut self, mode: AutoGazeMaskVisualizationMode) -> Self {
        self.mask_visualization_mode = mode;
        self
    }

    fn with_mask_geometry_mode(mut self, mode: AutoGazeMaskGeometryMode) -> Self {
        self.mask_geometry_mode = mode;
        self
    }

    fn with_cpu_panels(mut self) -> Self {
        self.cpu_layout = CpuVisualizationLayout::Panels;
        self
    }
}

struct FrameVisualInput<'a> {
    rgba: &'a [u8],
    width: usize,
    height: usize,
    tensor: Option<Tensor<AutoGazeBevyBackend, 5>>,
}

#[cfg(test)]
fn visualize_frame_rgba(
    input: FrameVisualInput<'_>,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    visualize_frame_rgba_with_device_tokens(
        input,
        points,
        None,
        None,
        None,
        options,
        visualization_state,
        device,
    )
}

fn visualize_frame_rgba_with_device_tokens(
    input: FrameVisualInput<'_>,
    points: &[FixationPoint],
    device_tokens: Option<&AutoGazeDeviceTokens<AutoGazeBevyBackend>>,
    device_mask: Option<AutoGazeDeviceMask<AutoGazeBevyBackend>>,
    model_config: Option<&AutoGazeConfig>,
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    if uses_tensor_display_transfer(options.display_transfer, input.width, input.height) {
        visualize_rgba_tensor(
            input,
            points,
            device_tokens,
            device_mask,
            model_config,
            options,
            visualization_state,
            device,
        )
    } else {
        visualize_rgba_bytes(
            input.rgba,
            input.width,
            input.height,
            points,
            options,
            visualization_state,
        )
    }
}

fn uses_tensor_display_transfer(
    display_transfer: BevyDisplayTransfer,
    width: usize,
    height: usize,
) -> bool {
    match display_transfer {
        BevyDisplayTransfer::Gpu => true,
        BevyDisplayTransfer::Cpu => false,
        BevyDisplayTransfer::Auto => width.saturating_mul(height) <= AUTO_GPU_DISPLAY_MAX_PIXELS,
    }
}

fn display_tensor_from_clip(
    clip: &FrameClip,
    options: VisualizationOptions,
    device: &AutoGazeBevyDevice,
) -> Result<Option<Tensor<AutoGazeBevyBackend, 5>>, String> {
    if !uses_tensor_display_transfer(options.display_transfer, clip.width(), clip.height()) {
        return Ok(None);
    }

    rgba_clip_to_tensor::<AutoGazeBevyBackend>(
        clip.last_frame_rgba()?,
        AutoGazeRgbaClipShape::new(1, clip.height(), clip.width()),
        device,
    )
    .map(Some)
    .map_err(|err| format!("failed to derive AutoGaze display tensor from source frame: {err:#}"))
}

fn display_tensor_from_prepared_trace(
    clip: &FrameClip,
    trace_input: &CoreAutoGazePreparedRun<AutoGazeBevyBackend>,
    options: VisualizationOptions,
    device: &AutoGazeBevyDevice,
) -> Result<PreparedDisplayTensor, String> {
    if !uses_tensor_display_transfer(options.display_transfer, clip.width(), clip.height()) {
        return Ok(PreparedDisplayTensor {
            tensor: None,
            ms: 0.0,
            residency: DisplayInputResidency::None,
        });
    }

    let display_start = timestamp_now();
    let dims = trace_input.video.shape().dims::<5>();
    if dims[0] == 1
        && dims[2] == 3
        && dims[3] == clip.height()
        && dims[4] == clip.width()
        && trace_input.frame_index < dims[1]
    {
        let tensor = video_frame_tensor(trace_input.video.clone(), trace_input.frame_index)
            .map_err(|err| {
                format!(
                    "failed to reuse prepared AutoGaze display tensor from model input: {err:#}"
                )
            })?;
        return Ok(PreparedDisplayTensor {
            tensor: Some(tensor),
            ms: elapsed_ms(display_start),
            residency: DisplayInputResidency::ModelTensorReuse,
        });
    }

    Ok(PreparedDisplayTensor {
        tensor: display_tensor_from_clip(clip, options, device)?,
        ms: elapsed_ms(display_start),
        residency: DisplayInputResidency::HostRgbaUpload,
    })
}

async fn calculate_tensor_psnr_if_needed(
    visualization: &mut Visualization,
    calculate_psnr: bool,
) -> Result<(), String> {
    if !calculate_psnr || visualization.psnr_db.is_some() || visualization.interframe_keyframe {
        return Ok(());
    }

    let VisualizationImageData::TensorPanels(panels) = &visualization.image_data else {
        return Ok(());
    };

    let psnr_start = timestamp_now();
    visualization.psnr_db =
        Some(tensor_psnr_db(panels.input_rgba.clone(), panels.output_rgba.clone()).await?);
    visualization.psnr_ms = elapsed_ms(psnr_start);
    Ok(())
}

async fn tensor_psnr_db(
    reference_rgba: Tensor<AutoGazeBevyBackend, 3>,
    candidate_rgba: Tensor<AutoGazeBevyBackend, 3>,
) -> Result<f64, String> {
    let reference_dims = reference_rgba.shape().dims::<3>();
    let candidate_dims = candidate_rgba.shape().dims::<3>();
    if reference_dims != candidate_dims || reference_dims[2] != 4 {
        return Err(format!(
            "expected matching RGBA tensors, got {reference_dims:?} and {candidate_dims:?}"
        ));
    }

    let height = reference_dims[0];
    let width = reference_dims[1];
    let reference_rgb = reference_rgba.slice([0..height, 0..width, 0..3]);
    let candidate_rgb = candidate_rgba.slice([0..height, 0..width, 0..3]);
    let mse = reference_rgb
        .sub(candidate_rgb)
        .powf_scalar(2.0)
        .mean()
        .into_scalar_async()
        .await
        .map_err(|err| format!("failed to read AutoGaze tensor PSNR scalar: {err:?}"))?;
    let mse = f64::from(mse);
    if mse <= 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(10.0 * (1.0 / mse).log10())
    }
}

fn visualize_rgba_tensor(
    input: FrameVisualInput<'_>,
    points: &[FixationPoint],
    device_tokens: Option<&AutoGazeDeviceTokens<AutoGazeBevyBackend>>,
    device_mask: Option<AutoGazeDeviceMask<AutoGazeBevyBackend>>,
    model_config: Option<&AutoGazeConfig>,
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    let width = input.width.max(1);
    let height = input.height.max(1);
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "AutoGaze visualization dimensions overflow".to_string())?;
    let expected_len = pixels
        .checked_mul(4)
        .ok_or_else(|| "AutoGaze visualization byte length overflow".to_string())?;
    if input.rgba.len() != expected_len {
        return Err(format!(
            "expected {expected_len} RGBA bytes for {width}x{height}, got {}",
            input.rgba.len()
        ));
    }

    let tensor_start = timestamp_now();
    let tensor = input
        .tensor
        .ok_or_else(|| "GPU display transfer requires a model input tensor".to_string())?;
    let tensor_options = AutoGazeTensorVisualizationOptions::new(
        width,
        height,
        options.cell_scale,
        options.blend_alpha,
    )
    .with_mask_visualization_mode(options.mask_visualization_mode)
    .with_mask_geometry_mode(options.mask_geometry_mode)
    .with_sparse_update_policy(
        options.sparse_update_max_rects,
        options.sparse_update_max_ratio,
    )
    .with_full_frame_update_policy(options.full_frame_update_min_ratio);
    let used_device_tokens = device_tokens.is_some() && model_config.is_some();
    let used_device_mask = device_mask.is_some();
    let tensor_panels = if let Some(device_mask) = device_mask {
        visualization_state
            .gpu
            .visualize_normalized_rgb_clip_device_mask_panels(
                tensor,
                device_mask,
                tensor_options,
                device,
            )
    } else if let (Some(device_tokens), Some(model_config)) = (device_tokens, model_config) {
        visualization_state
            .gpu
            .visualize_normalized_rgb_clip_device_tokens_panels(
                tensor,
                device_tokens,
                model_config,
                tensor_options,
                device,
            )
    } else {
        visualization_state
            .gpu
            .visualize_normalized_rgb_clip_panels(tensor, points, tensor_options, device)
    }
    .map_err(|err| format!("failed to visualize AutoGaze tensor output: {err:#}"))?;
    let tensor_ms = elapsed_ms(tensor_start);
    let tensor_interframe_path = visualization_state.gpu.last_interframe_path();
    let device_mask_plan_stats = used_device_tokens
        .then(|| mask_plan_stats_for_points(width, height, points, options))
        .transpose()?;
    let mask_plan_stats = device_mask_plan_stats
        .or_else(|| visualization_state.gpu.last_mask_plan_stats())
        .unwrap_or_default();
    let interframe_keyframe = matches!(
        tensor_interframe_path,
        Some(AutoGazeTensorInterframePath::Keyframe)
    );
    let gaze_update_ratio =
        gaze_ratio_from_mask_stats(tensor_panels.width, tensor_panels.height, mask_plan_stats);
    let output_update_ratio = if (used_device_tokens || used_device_mask) && !interframe_keyframe {
        gaze_update_ratio
    } else {
        tensor_panels.update_ratio()
    };
    let output_matches_input = is_interframe_full_output_match(
        visualization_state.gpu.mode(),
        interframe_keyframe,
        output_update_ratio,
    );
    let tensor_panel_count = if output_matches_input { 2 } else { 3 };
    let output_tensor_bytes = width * height * tensor_panel_count * 4 * std::mem::size_of::<f32>();

    #[cfg(test)]
    let test_side_by_side_rgba = tensor_panels_to_side_by_side_rgba(tensor_panels.clone());
    Ok(Visualization {
        width: (width * 3) as u32,
        height: height as u32,
        #[cfg(test)]
        rgba: Vec::new(),
        #[cfg(test)]
        tensor: Some(test_side_by_side_rgba),
        image_data: VisualizationImageData::TensorPanels(Box::new(TensorPanelVisualizationData {
            panel_width: width as u32,
            panel_height: height as u32,
            input_rgba: tensor_panels.input_rgba,
            mask_rgba: tensor_panels.mask_rgba,
            output_rgba: tensor_panels.output_rgba,
            output_matches_input,
        })),
        gaze_update_ratio,
        output_update_ratio,
        interframe_keyframe,
        psnr_db: None,
        visualize_cpu_ms: 0.0,
        psnr_ms: 0.0,
        tensor_ms,
        output_rgba_bytes: 0,
        output_tensor_bytes,
        tensor_interframe_path,
        effective_display_transfer: BevyDisplayTransfer::Gpu,
        mask_plan_stats,
        timing: None,
    })
}

#[cfg(test)]
fn tensor_panels_to_side_by_side_rgba(
    panels: burn_autogaze::AutoGazeTensorVisualizationPanels<AutoGazeBevyBackend>,
) -> Tensor<AutoGazeBevyBackend, 3> {
    panels.into_side_by_side().side_by_side_rgba
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
    let rgba_options = AutoGazeRgbaVisualizationOptions::new(
        width,
        height,
        options.cell_scale,
        options.blend_alpha,
    )
    .with_mask_visualization_mode(options.mask_visualization_mode)
    .with_mask_geometry_mode(options.mask_geometry_mode)
    .with_full_frame_update_policy(options.full_frame_update_min_ratio);
    match options.cpu_layout {
        CpuVisualizationLayout::SideBySide => {
            let visualization = visualization_state
                .cpu
                .visualize_rgba_with_options(rgba, points, rgba_options)
                .map_err(|err| format!("{err:#}"))?;
            let psnr_db = options
                .calculate_psnr
                .then(|| {
                    if visualization_state.cpu.last_frame_was_keyframe() {
                        Ok(None)
                    } else {
                        visualization
                            .output_psnr_db(rgba)
                            .map(Some)
                            .map_err(|err| format!("{err:#}"))
                    }
                })
                .transpose()?;
            let psnr_db = psnr_db.flatten();
            let interframe_keyframe = visualization_state.cpu.mode()
                == AutoGazeVisualizationMode::Interframe
                && visualization_state.cpu.last_frame_was_keyframe();
            let visualize_cpu_ms = elapsed_ms(visualize_cpu_start);
            let mask_plan_stats = visualization.mask_plan_stats;
            let output_update_ratio = visualization.update_ratio();
            let gaze_update_ratio = gaze_ratio_from_mask_stats(
                visualization.width,
                visualization.height,
                mask_plan_stats,
            );
            let side_by_side_rgba = visualization.side_by_side_rgba;
            let output_rgba_bytes = side_by_side_rgba.len();
            Ok(Visualization {
                width: visualization.side_by_side_width as u32,
                height: visualization.height as u32,
                #[cfg(test)]
                rgba: side_by_side_rgba.clone(),
                #[cfg(test)]
                tensor: None,
                image_data: VisualizationImageData::SideBySideRgba(side_by_side_rgba),
                gaze_update_ratio,
                output_update_ratio,
                interframe_keyframe,
                psnr_db,
                visualize_cpu_ms,
                psnr_ms: 0.0,
                tensor_ms: 0.0,
                output_rgba_bytes,
                output_tensor_bytes: 0,
                tensor_interframe_path: None,
                effective_display_transfer: BevyDisplayTransfer::Cpu,
                mask_plan_stats,
                timing: None,
            })
        }
        CpuVisualizationLayout::Panels => {
            let mut buffers = AutoGazeRgbaVisualizationBuffers::default();
            let panels = visualization_state
                .cpu
                .visualize_rgba_panels_with_options_into(rgba, points, rgba_options, &mut buffers)
                .map_err(|err| format!("{err:#}"))?;
            let psnr_db = options
                .calculate_psnr
                .then(|| {
                    if visualization_state.cpu.last_frame_was_keyframe() {
                        Ok(None)
                    } else {
                        panels
                            .output_psnr_db(rgba)
                            .map(Some)
                            .map_err(|err| format!("{err:#}"))
                    }
                })
                .transpose()?;
            let psnr_db = psnr_db.flatten();
            let interframe_keyframe = visualization_state.cpu.mode()
                == AutoGazeVisualizationMode::Interframe
                && visualization_state.cpu.last_frame_was_keyframe();
            let visualize_cpu_ms = elapsed_ms(visualize_cpu_start);
            let mask_plan_stats = panels.mask_plan_stats;
            let output_update_ratio = panels.update_ratio();
            let gaze_update_ratio =
                gaze_ratio_from_mask_stats(panels.width, panels.height, mask_plan_stats);
            let output_matches_input = is_interframe_full_output_match(
                visualization_state.cpu.mode(),
                interframe_keyframe,
                output_update_ratio,
            );
            let mask_rgba = std::mem::take(&mut buffers.mask_rgba);
            let output_rgba = if output_matches_input {
                Vec::new()
            } else {
                std::mem::take(&mut buffers.blend_rgba)
            };
            let input_rgba = rgba.to_vec();
            let output_rgba_bytes = input_rgba.len()
                + mask_rgba.len()
                + if output_matches_input {
                    0
                } else {
                    output_rgba.len()
                };
            Ok(Visualization {
                width: (width * 3) as u32,
                height: height as u32,
                #[cfg(test)]
                rgba: Vec::new(),
                #[cfg(test)]
                tensor: None,
                image_data: VisualizationImageData::PanelsRgba {
                    panel_width: width as u32,
                    panel_height: height as u32,
                    input_rgba,
                    mask_rgba,
                    output_rgba,
                    output_matches_input,
                },
                gaze_update_ratio,
                output_update_ratio,
                interframe_keyframe,
                psnr_db,
                visualize_cpu_ms,
                psnr_ms: 0.0,
                tensor_ms: 0.0,
                output_rgba_bytes,
                output_tensor_bytes: 0,
                tensor_interframe_path: None,
                effective_display_transfer: BevyDisplayTransfer::Cpu,
                mask_plan_stats,
                timing: None,
            })
        }
    }
}

fn live_preview_visualization(
    rgba: &RgbaImage,
    calculate_psnr: bool,
) -> Result<Visualization, String> {
    let mut state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 1);
    let visualization_options =
        VisualizationOptions::new(1.0, 0.0, calculate_psnr, BevyDisplayTransfer::Cpu)
            .with_cpu_panels();
    let mut visualization = visualize_points(rgba, &[], visualization_options, &mut state)?;
    visualization.gaze_update_ratio = 0.0;
    visualization.output_update_ratio = 0.0;
    Ok(visualization)
}

fn async_mask_preview_visualization(
    rgba: &RgbaImage,
    points: &[FixationPoint],
    config: &BevyBurnAutoGazeConfig,
    visualization_state: &mut BevyVisualizationState,
) -> Result<Visualization, String> {
    visualization_state.configure(config.visualization_mode, config.keyframe_duration);
    let visualization_options = VisualizationOptions::new(
        config.mask_cell_scale,
        config.blend_alpha,
        config.show_psnr,
        BevyDisplayTransfer::Cpu,
    )
    .with_full_frame_update_policy(config.tensor_full_frame_update_min_ratio)
    .with_mask_visualization_mode(config.mask_visualization_mode)
    .with_mask_geometry_mode(config.mask_geometry_mode)
    .with_cpu_panels();
    visualize_points(rgba, points, visualization_options, visualization_state)
}

fn gaze_ratio_from_mask_stats(
    width: usize,
    height: usize,
    mask_plan_stats: AutoGazeMaskPlanStats,
) -> f64 {
    mask_plan_stats.update_ratio(width, height)
}

fn mask_plan_stats_for_points(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    options: VisualizationOptions,
) -> Result<AutoGazeMaskPlanStats, String> {
    let plan = match options.mask_geometry_mode {
        AutoGazeMaskGeometryMode::Native => {
            fixation_sparse_update_plan(width, height, points, options.cell_scale)
        }
        AutoGazeMaskGeometryMode::Deduplicated => {
            fixation_deduplicated_sparse_update_plan(width, height, points, options.cell_scale)
        }
        AutoGazeMaskGeometryMode::Effective => {
            fixation_effective_sparse_update_plan(width, height, points, options.cell_scale)
        }
    }
    .map_err(|err| format!("failed to derive AutoGaze mask stats: {err:#}"))?;
    Ok(plan.stats())
}

fn is_interframe_full_output_match(
    mode: AutoGazeVisualizationMode,
    interframe_keyframe: bool,
    output_update_ratio: f64,
) -> bool {
    mode == AutoGazeVisualizationMode::Interframe
        && (interframe_keyframe || output_update_ratio >= 1.0)
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

fn next_source_frame(
    config: &BevyBurnAutoGazeConfig,
    static_frame: &StaticFrame,
    synthetic_source: &mut SyntheticFrameSource,
) -> Option<(Arc<RgbaImage>, f64, f64)> {
    match config.source {
        BevyFrameSource::StaticImage => static_frame
            .0
            .as_ref()
            .map(|frame| (Arc::clone(frame), 0.0, 0.0)),
        BevyFrameSource::SyntheticPan
        | BevyFrameSource::SyntheticPulse
        | BevyFrameSource::SyntheticLocalMotion => {
            let source_start = timestamp_now();
            let frame = synthetic_source.next_frame(config);
            let source_ms = elapsed_ms(source_start);
            let prepare_start = timestamp_now();
            let frame = prepare_frame_for_inference(frame, config);
            Some((Arc::new(frame), source_ms, elapsed_ms(prepare_start)))
        }
        BevyFrameSource::Camera => {
            let source_start = timestamp_now();
            receive_frame().map(|frame| {
                let source_ms = elapsed_ms(source_start);
                let prepare_start = timestamp_now();
                let frame = prepare_frame_for_inference(frame, config);
                (Arc::new(frame), source_ms, elapsed_ms(prepare_start))
            })
        }
    }
}

fn frame_queue_len_for_config(config: &BevyBurnAutoGazeConfig) -> usize {
    if config.sparse_mask_source == BevySparseMaskSource::PatchDiff {
        2
    } else {
        config.frames_per_clip.max(1)
    }
}

fn load_static_frame(path: Option<&Path>, config: &BevyBurnAutoGazeConfig) -> StaticFrame {
    let Some(path) = path else {
        return StaticFrame(None);
    };
    match image::open(path) {
        Ok(frame) => StaticFrame(Some(Arc::new(prepare_frame_for_inference(
            frame.to_rgba8(),
            config,
        )))),
        Err(err) => {
            log(&format!(
                "failed to load static AutoGaze image `{}`: {err}",
                path.display()
            ));
            StaticFrame(None)
        }
    }
}

fn prepare_frame_for_inference(frame: RgbaImage, config: &BevyBurnAutoGazeConfig) -> RgbaImage {
    resize_rgba_frame_to_dimensions(frame, config.inference_width, config.inference_height)
}

fn synthetic_source_dimensions(config: &BevyBurnAutoGazeConfig) -> (u32, u32) {
    match (config.inference_width, config.inference_height) {
        (Some(width), Some(height)) => (width.max(1), height.max(1)),
        (Some(width), None) => (
            width.max(1),
            ((width as f32 * 9.0 / 16.0).round() as u32).max(1),
        ),
        (None, Some(height)) => (
            ((height as f32 * 16.0 / 9.0).round() as u32).max(1),
            height.max(1),
        ),
        (None, None) => match config.mode {
            BevyAutoGazeMode::Resize224 => (DEFAULT_REALTIME_INFERENCE_WIDTH, 360),
            BevyAutoGazeMode::Tile224 => (DEFAULT_TILED_INFERENCE_WIDTH, 720),
        },
    }
}

fn synthetic_pan_frame(width: u32, height: u32, frame_index: u64) -> RgbaImage {
    let width = width.max(1);
    let height = height.max(1);
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    let phase_x = (frame_index as u32).wrapping_mul(9);
    let phase_y = (frame_index as u32).wrapping_mul(5);
    let square = (width.min(height) / 5).max(24);
    let square_x =
        ((frame_index as u32).wrapping_mul(13) % (width + square)) as i32 - square as i32;
    let square_y =
        ((frame_index as u32).wrapping_mul(7) % (height + square)) as i32 - square as i32;
    let circle_radius = (width.min(height) / 8).max(12) as i32;
    let circle_x = ((frame_index as u32).wrapping_mul(5) % width) as i32;
    let circle_y = ((height / 2) as i32)
        + (((frame_index as f32) * 0.17).sin() * height as f32 * 0.25).round() as i32;

    for y in 0..height {
        for x in 0..width {
            let xf = x as f32 / width.saturating_sub(1).max(1) as f32;
            let yf = y as f32 / height.saturating_sub(1).max(1) as f32;
            let wave =
                (((x + phase_x) as f32 * 0.035).sin() + ((y + phase_y) as f32 * 0.047).cos()) * 0.5
                    + 0.5;
            let inside_square = (x as i32) >= square_x
                && (x as i32) < square_x + square as i32
                && (y as i32) >= square_y
                && (y as i32) < square_y + square as i32;
            let dx = x as i32 - circle_x;
            let dy = y as i32 - circle_y;
            let inside_circle = dx * dx + dy * dy <= circle_radius * circle_radius;
            let (r, g, b) = if inside_square {
                (238, (96.0 + wave * 84.0) as u8, 56)
            } else if inside_circle {
                (42, (190.0 + wave * 45.0) as u8, 214)
            } else {
                (
                    (34.0 + xf * 92.0 + wave * 22.0) as u8,
                    (70.0 + yf * 112.0 + wave * 18.0) as u8,
                    (118.0 + (1.0 - xf) * 68.0 + wave * 20.0) as u8,
                )
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }

    RgbaImage::from_raw(width, height, rgba).unwrap_or_else(|| RgbaImage::new(width, height))
}

fn synthetic_pulse_frame(width: u32, height: u32, frame_index: u64) -> RgbaImage {
    const STATIC_FRAMES: u64 = 16;
    const MOTION_FRAMES: u64 = 24;
    const HOLD_FRAMES: u64 = 8;
    let cycle = STATIC_FRAMES + MOTION_FRAMES + HOLD_FRAMES;
    let phase = frame_index % cycle;
    let pan_index = if phase < STATIC_FRAMES {
        0
    } else if phase < STATIC_FRAMES + MOTION_FRAMES {
        phase - STATIC_FRAMES + 1
    } else {
        MOTION_FRAMES
    };
    synthetic_pan_frame(width, height, pan_index)
}

fn synthetic_local_motion_frame(width: u32, height: u32, frame_index: u64) -> RgbaImage {
    let width = width.max(1);
    let height = height.max(1);
    let phase = frame_index % SYNTHETIC_LOCAL_CYCLE_FRAMES;
    let strong_dx = (width as f32 / 150.0).max(1.0);
    let subtle_dx = (width as f32 / 1800.0).max(0.25);
    let local_progress = if phase < SYNTHETIC_LOCAL_STRONG_FRAMES {
        phase as f32 * strong_dx
    } else if phase < SYNTHETIC_LOCAL_STRONG_FRAMES + SYNTHETIC_LOCAL_SUBTLE_FRAMES {
        SYNTHETIC_LOCAL_STRONG_FRAMES as f32 * strong_dx
            + (phase - SYNTHETIC_LOCAL_STRONG_FRAMES) as f32 * subtle_dx
    } else {
        SYNTHETIC_LOCAL_STRONG_FRAMES as f32 * strong_dx
            + SYNTHETIC_LOCAL_SUBTLE_FRAMES as f32 * subtle_dx
    };
    let base_x = (width as f32 * 0.22 + local_progress).round() as i32;
    let base_y = (height as f32 * 0.42).round() as i32;
    let patch_w = (width / 8).clamp(24, width.max(24));
    let patch_h = (height / 7).clamp(20, height.max(20));
    let patch_w_i = patch_w as i32;
    let patch_h_i = patch_h as i32;
    let small_w = (patch_w_i / 4).max(8);
    let small_h = (patch_h_i / 3).max(6);
    let center_x = base_x + patch_w_i / 2;
    let center_y = base_y + patch_h_i / 2;
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);

    for y in 0..height {
        for x in 0..width {
            let xf = x as f32 / width.saturating_sub(1).max(1) as f32;
            let yf = y as f32 / height.saturating_sub(1).max(1) as f32;
            let checker = (((x / 18) + (y / 18)) & 1) as u8;
            let fine = (((x.wrapping_mul(13) ^ y.wrapping_mul(7)) + 31) & 0x1f) as u8;
            let mut r = (38.0 + xf * 68.0) as u8 + checker * 14 + fine / 4;
            let mut g = (58.0 + yf * 92.0) as u8 + checker * 10 + fine / 5;
            let mut b = (86.0 + (1.0 - xf) * 54.0) as u8 + checker * 8 + fine / 3;

            let xi = x as i32;
            let yi = y as i32;
            let in_patch =
                xi >= base_x && xi < base_x + patch_w_i && yi >= base_y && yi < base_y + patch_h_i;
            if in_patch {
                let lx = (xi - base_x).max(0) as u32;
                let ly = (yi - base_y).max(0) as u32;
                let stripe = (((lx / 5) + (ly / 7)) & 1) as u8;
                r = 210_u8.saturating_sub(stripe * 34).saturating_add(fine / 3);
                g = 122_u8.saturating_add(stripe * 28);
                b = 48_u8.saturating_add(((lx + ly) & 0x1f) as u8);
            }

            let small_x = base_x + patch_w_i / 2 - small_w / 2;
            let small_y = base_y - patch_h_i / 3;
            let in_small =
                xi >= small_x && xi < small_x + small_w && yi >= small_y && yi < small_y + small_h;
            let dx = xi - center_x;
            let dy = yi - center_y;
            let radius = (patch_h_i.min(patch_w_i) / 3).max(6);
            let in_circle = dx * dx + dy * dy <= radius * radius;
            if in_small || in_circle {
                r = 42;
                g = 202_u8.saturating_add((((x + y) / 3) & 0x1f) as u8);
                b = 220;
            }

            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }

    RgbaImage::from_raw(width, height, rgba).unwrap_or_else(|| RgbaImage::new(width, height))
}

fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

fn enforce_required_hardware_adapter(
    config: Res<BevyBurnAutoGazeConfig>,
    adapter: Option<Res<RenderAdapterInfo>>,
    mut exit: MessageWriter<AppExit>,
    mut emitted: Local<bool>,
) {
    if *emitted || !config.require_hardware_adapter {
        return;
    }
    let Some(adapter) = adapter else {
        return;
    };
    if !is_software_render_adapter(&adapter) {
        return;
    }

    *emitted = true;
    let summary = RenderAdapterSummary::from(adapter.as_ref());
    log(&format!(
        "AutoGaze hardware adapter required, but Bevy selected software adapter `{}` ({}, {}, driver={} {}).",
        summary.name, summary.device_type, summary.backend, summary.driver, summary.driver_info
    ));
    exit.write(AppExit::error());
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
    log(&format!("sparse mask perf summary: {summary}"));
    if let Err(err) = write_perf_summary(config.perf_summary_path.as_deref(), &summary) {
        log(&format!("failed to write sparse mask perf summary: {err}"));
        timing.emitted_summary = true;
        #[cfg(not(target_arch = "wasm32"))]
        exit.write(AppExit::error());
        return;
    }
    publish_wasm_perf_summary(&summary);
    timing.emitted_summary = true;

    #[cfg(not(target_arch = "wasm32"))]
    exit.write(AppExit::Success);
}

#[cfg(not(target_arch = "wasm32"))]
fn write_perf_summary(path: Option<&Path>, summary: &str) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create perf summary directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let value: serde_json::Value = serde_json::from_str(summary)
        .map_err(|err| format!("invalid perf summary JSON before write: {err}"))?;
    let pretty = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to format perf summary JSON: {err}"))?;
    std::fs::write(path, format!("{pretty}\n"))
        .map_err(|err| format!("failed to write `{}`: {err}", path.display()))
}

#[cfg(target_arch = "wasm32")]
fn write_perf_summary(_path: Option<&Path>, _summary: &str) -> Result<(), String> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn write_perf_trace_sample(
    path: Option<&Path>,
    sample: &str,
    truncate: bool,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create perf trace directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let value: serde_json::Value = serde_json::from_str(sample)
        .map_err(|err| format!("invalid perf trace JSON before write: {err}"))?;
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true);
    if truncate {
        options.truncate(true);
    } else {
        options.append(true);
    }
    let mut file = options
        .open(path)
        .map_err(|err| format!("failed to open `{}`: {err}", path.display()))?;
    serde_json::to_writer(&mut file, &value)
        .map_err(|err| format!("failed to encode perf trace JSON: {err}"))?;
    use std::io::Write as _;
    writeln!(file).map_err(|err| format!("failed to write `{}`: {err}", path.display()))
}

#[cfg(target_arch = "wasm32")]
fn write_perf_trace_sample(
    _path: Option<&Path>,
    _sample: &str,
    _truncate: bool,
) -> Result<(), String> {
    Ok(())
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
                top: Val::Px(metric_overlay_top(0)),
                left: Val::Px(UI_MARGIN_PX),
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
            TextSpan::new(stable_fps_text(None, None)),
        ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    timing: Res<InferenceTimingStats>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    let render_fps = diagnostic_fps(&diagnostics, &FrameTimeDiagnosticsPlugin::FPS);
    let model_fps = if timing.processed_frames() > 0 {
        timing
            .latest
            .map(|timing| timing.e2e_fps())
            .or_else(|| diagnostic_fps(&diagnostics, &MODEL_FPS))
    } else {
        None
    };
    for mut text in &mut query {
        **text = stable_fps_text(render_fps, model_fps);
    }
}

fn diagnostic_fps(diagnostics: &DiagnosticsStore, path: &DiagnosticPath) -> Option<f64> {
    diagnostics
        .get(path)
        .and_then(|diagnostic| diagnostic.smoothed().or_else(|| diagnostic.value()))
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn gaze_ratio_display_setup(mut commands: Commands, config: Res<BevyBurnAutoGazeConfig>) {
    let top = metric_overlay_top(usize::from(config.show_fps));
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
                top: Val::Px(top),
                left: Val::Px(UI_MARGIN_PX),
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
            TextSpan::new(stable_gaze_ratio_text(None, None)),
        ));
}

#[derive(Component)]
struct GazeRatioText;

fn gaze_ratio_update_system(
    stats: Res<GazeRatioStats>,
    mut query: Query<&mut TextSpan, With<GazeRatioText>>,
) {
    for mut text in &mut query {
        if stats.0.is_initialized() {
            **text = stable_gaze_ratio_text(Some(stats.0.current()), Some(stats.0.ema()));
        } else {
            **text = stable_gaze_ratio_text(None, None);
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
                top: Val::Px(metric_overlay_top(row)),
                left: Val::Px(UI_MARGIN_PX),
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
            TextSpan::new(stable_psnr_text(None, None)),
        ));
}

#[derive(Component)]
struct PsnrText;

fn psnr_update_system(stats: Res<PsnrStats>, mut query: Query<&mut TextSpan, With<PsnrText>>) {
    for mut text in &mut query {
        if stats.0.is_initialized() {
            **text = stable_psnr_text(Some(stats.0.current()), Some(stats.0.ema()));
        } else {
            **text = stable_psnr_text(None, None);
        }
    }
}

#[derive(Component)]
struct TaskLossSliderTrack;

#[derive(Component)]
struct TaskLossSliderFill;

#[derive(Component)]
struct TaskLossSliderThumb;

#[derive(Component)]
struct TaskLossSliderValueText;

#[derive(Component)]
struct MaskSourceToggle;

#[derive(Component)]
struct MaskSourceValueText;

fn task_loss_slider_display_setup(
    mut commands: Commands,
    config: Res<BevyBurnAutoGazeConfig>,
    slider: Res<TaskLossSliderState>,
) {
    let row = usize::from(config.show_fps)
        + usize::from(config.show_gaze_ratio)
        + usize::from(config.show_psnr);
    let top = metric_overlay_top(row);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(top),
                left: Val::Px(UI_MARGIN_PX),
                height: Val::Px(METRIC_ROW_HEIGHT),
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                padding: UiRect::right(Val::Px(12.0)),
                ..default()
            },
            ZIndex(2),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text("quality: ".to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(22.0),
                    ..Default::default()
                },
                TextColor(Color::WHITE),
            ));
            parent.spawn((
                TaskLossSliderValueText,
                TextFont {
                    font_size: bevy::text::FontSize::Px(22.0),
                    ..Default::default()
                },
                TextColor(Color::srgb(1.0, 0.84, 0.0)),
                Text(task_loss_slider_label(slider.value)),
            ));
            parent
                .spawn((
                    MaskSourceToggle,
                    Button,
                    Interaction::None,
                    BackgroundColor(Color::srgba(0.95, 0.95, 0.95, 0.16)),
                    Node {
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        height: Val::Px(28.0),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                ))
                .with_child((
                    MaskSourceValueText,
                    TextFont {
                        font_size: bevy::text::FontSize::Px(18.0),
                        ..Default::default()
                    },
                    TextColor(Color::srgb(0.74, 0.88, 1.0)),
                    Text(mask_source_label(config.sparse_mask_source)),
                ));
            parent
                .spawn((
                    TaskLossSliderTrack,
                    Button,
                    Interaction::None,
                    RelativeCursorPosition::default(),
                    BackgroundColor(Color::srgba(0.95, 0.95, 0.95, 0.22)),
                    Node {
                        position_type: PositionType::Relative,
                        width: Val::Px(TASK_LOSS_SLIDER_WIDTH),
                        height: Val::Px(12.0),
                        margin: UiRect::left(Val::Px(4.0)),
                        ..default()
                    },
                ))
                .with_children(|track| {
                    track.spawn((
                        TaskLossSliderFill,
                        BackgroundColor(Color::srgba(1.0, 0.84, 0.0, 0.82)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(0.0),
                            width: Val::Percent(task_loss_slider_percent(slider.value) * 100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                    ));
                    track.spawn((
                        TaskLossSliderThumb,
                        BackgroundColor(Color::srgb(1.0, 0.93, 0.45)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Percent(task_loss_slider_percent(slider.value) * 100.0),
                            top: Val::Px(-4.0),
                            width: Val::Px(10.0),
                            height: Val::Px(20.0),
                            margin: UiRect::left(Val::Px(-5.0)),
                            ..default()
                        },
                    ));
                });
        });
}

fn task_loss_slider_update_system(
    buttons: Res<ButtonInput<MouseButton>>,
    mut slider: ResMut<TaskLossSliderState>,
    mut config: ResMut<BevyBurnAutoGazeConfig>,
    mut model: ResMut<AutoGazeModelState>,
    tracks: Query<(&Interaction, &RelativeCursorPosition), With<TaskLossSliderTrack>>,
) {
    if !config.show_task_loss_slider {
        return;
    }
    if buttons.just_released(MouseButton::Left) {
        slider.dragging = false;
    }

    let mut next_value = None;
    for (interaction, cursor) in &tracks {
        if matches!(*interaction, Interaction::Pressed) {
            slider.dragging = true;
        }
        if slider.dragging
            && buttons.pressed(MouseButton::Left)
            && let Some(normalized) = cursor.normalized
        {
            next_value = Some(task_loss_slider_value_from_normalized_x(normalized.x));
        }
    }

    if let Some(value) = next_value
        && (value - slider.value).abs() >= TASK_LOSS_SLIDER_STEP * 0.5
    {
        apply_task_loss_slider_value(&mut slider, &mut config, &mut model.config, value);
    }

    if let Some(value) = slider.pending_value {
        let Some(pipeline) = model.pipeline.as_ref() else {
            return;
        };
        if let Ok(mut pipeline) = pipeline.try_lock() {
            pipeline.set_task_loss_requirement(Some(value));
            pipeline.reset_max_gaze_tokens_each_frame();
            slider.pending_value = None;
        }
    }
}

fn mask_source_toggle_system(
    mut commands: Commands,
    mut config: ResMut<BevyBurnAutoGazeConfig>,
    mut model: ResMut<AutoGazeModelState>,
    mut slider: ResMut<TaskLossSliderState>,
    mut latest_mask: ResMut<LatestMaskPrediction>,
    mut visualization_state: ResMut<BevyVisualizationState>,
    mut streaming_state: ResMut<BevyStreamingGenerationState>,
    mut sequencer: ResMut<InferenceSequencer>,
    mut frame_queue: ResMut<FrameQueue>,
    mut gaze_ratio_stats: ResMut<GazeRatioStats>,
    mut psnr_stats: ResMut<PsnrStats>,
    mut timing_stats: ResMut<InferenceTimingStats>,
    active_tasks: Query<Entity, With<ProcessAutoGaze>>,
    toggles: Query<&Interaction, (Changed<Interaction>, With<MaskSourceToggle>)>,
) {
    if !config.show_task_loss_slider {
        return;
    }
    let mut toggled = false;
    for interaction in &toggles {
        if matches!(*interaction, Interaction::Pressed) {
            toggled = true;
        }
    }
    if !toggled {
        return;
    }

    apply_mask_source_toggle(
        &mut config,
        &mut model.config,
        &mut slider,
        &mut latest_mask,
        &mut visualization_state,
        &mut streaming_state,
        &mut sequencer,
        &mut frame_queue,
        &mut gaze_ratio_stats,
        &mut psnr_stats,
        &mut timing_stats,
    );
    for entity in &active_tasks {
        commands.entity(entity).despawn();
    }
    if !config.sparse_mask_source.requires_autogaze_model() {
        model.load_task = None;
    }
}

fn apply_mask_source_toggle(
    config: &mut BevyBurnAutoGazeConfig,
    model_config: &mut BevyBurnAutoGazeConfig,
    slider: &mut TaskLossSliderState,
    latest_mask: &mut LatestMaskPrediction,
    visualization_state: &mut BevyVisualizationState,
    streaming_state: &mut BevyStreamingGenerationState,
    sequencer: &mut InferenceSequencer,
    frame_queue: &mut FrameQueue,
    gaze_ratio_stats: &mut GazeRatioStats,
    psnr_stats: &mut PsnrStats,
    timing_stats: &mut InferenceTimingStats,
) {
    config.sparse_mask_source = match config.sparse_mask_source {
        BevySparseMaskSource::AutoGaze => BevySparseMaskSource::PatchDiff,
        BevySparseMaskSource::PatchDiff => BevySparseMaskSource::AutoGaze,
    };
    model_config.sparse_mask_source = config.sparse_mask_source;
    let quality = quality_slider_config_value(&config);
    slider.value = quantize_task_loss_slider_value(quality);
    slider.pending_value = (config.sparse_mask_source == BevySparseMaskSource::AutoGaze)
        .then_some(quality_slider_threshold(slider.value));
    slider.dragging = false;
    latest_mask.clear();
    visualization_state.reset();
    streaming_state.reset();
    sequencer.invalidate_pending();
    frame_queue.reset();
    gaze_ratio_stats.reset();
    psnr_stats.reset();
    timing_stats.reset();
}

fn apply_task_loss_slider_value(
    slider: &mut TaskLossSliderState,
    config: &mut BevyBurnAutoGazeConfig,
    model_config: &mut BevyBurnAutoGazeConfig,
    value: f32,
) {
    let quality = quantize_task_loss_slider_value(value);
    let threshold = quality_slider_threshold(quality);
    slider.value = quality;
    match config.sparse_mask_source {
        BevySparseMaskSource::AutoGaze => {
            slider.pending_value = Some(threshold);
            config.task_loss_requirement = Some(threshold);
            config.disable_task_loss_requirement = false;
            config.limit_generation_budget = false;
            config.max_gaze_tokens_each_frame = 0;
            model_config.task_loss_requirement = Some(threshold);
            model_config.disable_task_loss_requirement = false;
            model_config.limit_generation_budget = false;
            model_config.max_gaze_tokens_each_frame = 0;
        }
        BevySparseMaskSource::PatchDiff => {
            slider.pending_value = None;
            config.patch_diff_threshold = threshold;
            model_config.patch_diff_threshold = threshold;
        }
    }
}

fn task_loss_slider_style_system(
    config: Res<BevyBurnAutoGazeConfig>,
    slider: Res<TaskLossSliderState>,
    mut labels: Query<&mut Text, With<TaskLossSliderValueText>>,
    mut source_labels: Query<
        &mut Text,
        (With<MaskSourceValueText>, Without<TaskLossSliderValueText>),
    >,
    mut fills: Query<&mut Node, (With<TaskLossSliderFill>, Without<TaskLossSliderThumb>)>,
    mut thumbs: Query<&mut Node, (With<TaskLossSliderThumb>, Without<TaskLossSliderFill>)>,
) {
    if !slider.is_changed() && !config.is_changed() {
        return;
    }
    let percent = task_loss_slider_percent(slider.value) * 100.0;
    for mut label in &mut labels {
        **label = task_loss_slider_label(slider.value);
    }
    for mut node in &mut fills {
        node.width = Val::Percent(percent);
    }
    for mut node in &mut thumbs {
        node.left = Val::Percent(percent);
    }
    for mut label in &mut source_labels {
        **label = mask_source_label(config.sparse_mask_source);
    }
}

fn quality_slider_config_value(config: &BevyBurnAutoGazeConfig) -> f32 {
    let threshold = match config.sparse_mask_source {
        BevySparseMaskSource::AutoGaze => config
            .task_loss_requirement
            .unwrap_or(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT),
        BevySparseMaskSource::PatchDiff => config.patch_diff_threshold,
    };
    quality_slider_quality_from_threshold(threshold)
}

fn mask_source_label(source: BevySparseMaskSource) -> String {
    match source {
        BevySparseMaskSource::AutoGaze => "autogaze".to_string(),
        BevySparseMaskSource::PatchDiff => "patch-diff".to_string(),
    }
}

fn quantize_task_loss_slider_value(value: f32) -> f32 {
    let value = value.clamp(TASK_LOSS_SLIDER_MIN, TASK_LOSS_SLIDER_MAX);
    (value / TASK_LOSS_SLIDER_STEP).round() * TASK_LOSS_SLIDER_STEP
}

fn task_loss_slider_percent(value: f32) -> f32 {
    ((value - TASK_LOSS_SLIDER_MIN) / (TASK_LOSS_SLIDER_MAX - TASK_LOSS_SLIDER_MIN)).clamp(0.0, 1.0)
}

fn task_loss_slider_value_from_normalized_x(normalized_x: f32) -> f32 {
    let quality = (normalized_x + 0.5).clamp(0.0, 1.0);
    quantize_task_loss_slider_value(
        TASK_LOSS_SLIDER_MIN + quality * (TASK_LOSS_SLIDER_MAX - TASK_LOSS_SLIDER_MIN),
    )
}

fn task_loss_slider_label(value: f32) -> String {
    format!("{:>5.1}%", task_loss_slider_percent(value) * 100.0)
}

fn quality_slider_threshold(quality: f32) -> f32 {
    (TASK_LOSS_SLIDER_MAX - quantize_task_loss_slider_value(quality)).clamp(0.0, 1.0)
}

fn quality_slider_quality_from_threshold(threshold: f32) -> f32 {
    quantize_task_loss_slider_value((TASK_LOSS_SLIDER_MAX - threshold).clamp(0.0, 1.0))
}

fn stable_fps_text(render_fps: Option<f64>, model_fps: Option<f64>) -> String {
    format!(
        "render {} infer {}",
        format_fps(render_fps.unwrap_or(f64::NAN)),
        format_fps(model_fps.unwrap_or(f64::NAN))
    )
}

fn stable_gaze_ratio_text(current: Option<f64>, ema: Option<f64>) -> String {
    format!(
        "{} ema {}",
        format_gaze_ratio_percent(current.unwrap_or(f64::NAN)),
        format_gaze_ratio_percent(ema.unwrap_or(f64::NAN))
    )
}

fn stable_psnr_text(current: Option<f64>, ema: Option<f64>) -> String {
    format!(
        "{} dB ema {} dB",
        format_psnr_db(current.unwrap_or(f64::NAN)),
        format_psnr_db(ema.unwrap_or(f64::NAN))
    )
}

fn metric_overlay_top(row: usize) -> f32 {
    UI_MARGIN_PX + row as f32 * METRIC_ROW_HEIGHT
}

fn metric_panel_top_reserved_height(config: &BevyBurnAutoGazeConfig) -> f32 {
    let rows = usize::from(config.show_fps)
        + usize::from(config.show_gaze_ratio)
        + usize::from(config.show_psnr)
        + usize::from(config.show_task_loss_slider);
    if rows == 0 {
        UI_MARGIN_PX
    } else {
        UI_MARGIN_PX * 2.0 + rows as f32 * METRIC_ROW_HEIGHT
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
    use burn_autogaze::resize_dimensions_preserving_aspect;

    #[test]
    fn viewer_defaults_use_continuous_streaming_realtime_profile() {
        let config = BevyBurnAutoGazeConfig::default();

        assert_eq!(config.mode, BevyAutoGazeMode::Resize224);
        assert_eq!(config.top_k, DEFAULT_REALTIME_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            default_max_gaze_tokens_each_frame(config.mode)
        );
        assert_eq!(
            config.task_loss_requirement,
            Some(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT)
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
        assert_eq!(
            config.mask_visualization_mode,
            AutoGazeMaskVisualizationMode::ImageMaskOnly
        );
        assert_eq!(config.mask_geometry_mode, DEFAULT_BEVY_MASK_GEOMETRY_MODE);
        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
        assert_eq!(config.keyframe_duration, DEFAULT_BIRDS_KEYFRAME_DURATION);
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Auto);
        assert!(config.show_psnr);
        assert!(config.show_task_loss_slider);
        assert_eq!(config.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert!(config.streaming_cache);
        assert_eq!(config.decode_strategy, DEFAULT_BEVY_DECODE_STRATEGY);
        assert!(should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            config.mode.inference_mode()
        ));
        assert_eq!(
            pipeline_options_from_config(&config).max_gaze_tokens_each_frame(),
            Some(DEFAULT_REALTIME_MAX_GAZE_TOKENS),
            "realtime defaults should match the bounded throughput-bench profile"
        );
        assert_eq!(
            pipeline_options_from_config(&config).task_loss_requirement(),
            AutoGazeTaskLossOption::Value(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT)
        );
        assert_eq!(
            pipeline_options_from_config(&config).decode_strategy(),
            config.decode_strategy
        );
    }

    #[test]
    fn prepared_run_uses_core_processor_tensor_and_raw_display_frame() {
        if skip_native_wgpu_test_on_github_actions(
            "prepared_run_uses_core_processor_tensor_and_raw_display_frame",
        ) {
            return;
        }

        let device = AutoGazeBevyDevice::default();
        let first = [10, 20, 30, 255, 40, 50, 60, 255];
        let second = [70, 80, 90, 255, 100, 110, 120, 255];
        let mut rgba = Vec::new();
        rgba.extend_from_slice(&first);
        rgba.extend_from_slice(&second);
        let clip = FrameClip::from_core(
            AutoGazeRgbaFrameClip::new(rgba, 2, 1, 2).expect("core frame clip"),
            0.0,
        );
        let options = VisualizationOptions::new(1.0, 0.0, false, BevyDisplayTransfer::Gpu);

        let prepared = prepare_autogaze_run(
            &clip,
            false,
            AutoGazeInferenceMode::ResizeToModelInput,
            options,
            &device,
        )
        .expect("prepare full clip");
        let expected_model = rgba_clip_to_processor_tensor::<AutoGazeBevyBackend>(
            clip.rgba(),
            clip.shape(),
            &device,
        )
        .expect("core processor tensor");
        assert_tensor_values_close(prepared.trace_input.video.clone(), expected_model);
        assert_eq!(prepared.trace_input.frame_index, 1);
        assert_eq!(prepared.trace_input.model_frames, 2);
        assert_eq!(
            prepared.visualization.display_input_residency,
            DisplayInputResidency::HostRgbaUpload
        );
        let expected_display = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            clip.last_frame_rgba().unwrap(),
            AutoGazeRgbaClipShape::new(1, clip.height(), clip.width()),
            &device,
        )
        .expect("core display tensor");
        assert_tensor_values_close(
            prepared
                .visualization
                .visualization_tensor
                .expect("display tensor should be derived from raw display input"),
            expected_display,
        );

        let prepared = prepare_autogaze_run(
            &clip,
            true,
            AutoGazeInferenceMode::ResizeToModelInput,
            options,
            &device,
        )
        .expect("prepare streaming frame");
        let expected_model = rgba_clip_to_processor_tensor::<AutoGazeBevyBackend>(
            clip.last_frame_rgba().unwrap(),
            AutoGazeRgbaClipShape::new(1, clip.height(), clip.width()),
            &device,
        )
        .expect("core current-frame processor tensor");
        assert_eq!(prepared.trace_input.frame_index, 0);
        assert_eq!(prepared.trace_input.model_frames, 1);
        assert_eq!(
            prepared.visualization.display_input_residency,
            DisplayInputResidency::HostRgbaUpload
        );
        assert_tensor_values_close(prepared.trace_input.video.clone(), expected_model);
        let expected_display = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            clip.last_frame_rgba().unwrap(),
            AutoGazeRgbaClipShape::new(1, clip.height(), clip.width()),
            &device,
        )
        .expect("core current-frame display tensor");
        assert_tensor_values_close(
            prepared
                .visualization
                .visualization_tensor
                .expect("streaming display tensor should be the current frame"),
            expected_display,
        );
    }

    #[test]
    fn prepared_birds_run_uses_same_core_rgba_path_as_python_fixture() {
        if skip_native_wgpu_test_on_github_actions(
            "prepared_birds_run_uses_same_core_rgba_path_as_python_fixture",
        ) {
            return;
        }

        let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/autogaze_birds_python_generate");
        let fixture_path = fixture_root.join("fixture_outputs.safetensors");
        if !fixture_path.exists() {
            eprintln!(
                "skipping Bevy birds input-path check: missing {}",
                fixture_path.display()
            );
            return;
        }

        let bytes = std::fs::read(&fixture_path).expect("read birds fixture");
        let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize fixture");
        let (raw_rgba, shape) =
            fixture_raw_rgba(&tensors, &fixture_root).expect("raw RGBA fixture frames");
        assert_eq!(shape, vec![2, 1080, 1920, 4]);

        let mut queue = FrameQueue::default();
        let config = BevyBurnAutoGazeConfig {
            frames_per_clip: shape[0],
            ..BevyBurnAutoGazeConfig::docs_birds()
        };
        let frame_bytes = shape[1] * shape[2] * shape[3];
        for frame_idx in 0..shape[0] {
            let start = frame_idx * frame_bytes;
            let end = start + frame_bytes;
            let frame = RgbaImage::from_raw(
                shape[2] as u32,
                shape[1] as u32,
                raw_rgba[start..end].to_vec(),
            )
            .expect("valid RGBA fixture frame");
            let frame = prepare_frame_for_inference(frame, &config);
            assert_eq!(
                frame.dimensions(),
                (
                    DEFAULT_BIRDS_INFERENCE_WIDTH,
                    DEFAULT_BIRDS_INFERENCE_HEIGHT
                )
            );
            queue.push(std::sync::Arc::new(frame), config.frames_per_clip);
        }
        let clip = queue
            .build_clip(config.frames_per_clip)
            .expect("build clip")
            .expect("complete clip");

        let device = AutoGazeBevyDevice::default();
        let options = VisualizationOptions::new(1.0, 0.0, false, BevyDisplayTransfer::Gpu);
        let prepared =
            prepare_autogaze_run(&clip, false, config.mode.inference_mode(), options, &device)
                .expect("prepare Bevy run from birds fixture");
        assert_eq!(
            prepared.trace_input.video.shape().dims::<5>(),
            [
                1,
                shape[0],
                3,
                DEFAULT_BIRDS_INFERENCE_HEIGHT as usize,
                DEFAULT_BIRDS_INFERENCE_WIDTH as usize
            ]
        );
        assert_eq!(prepared.trace_input.frame_index, 1);
        assert_eq!(prepared.trace_input.model_frames, 2);
        assert_eq!(
            prepared.visualization.display_input_residency,
            DisplayInputResidency::ModelTensorReuse
        );

        let expected_model = rgba_clip_to_inference_tensor::<AutoGazeBevyBackend>(
            clip.rgba(),
            clip.shape(),
            config.mode.inference_mode(),
            &device,
        )
        .expect("core inference tensor");
        assert_tensor_values_close(prepared.trace_input.video.clone(), expected_model);

        let expected_display = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            clip.last_frame_rgba().unwrap(),
            AutoGazeRgbaClipShape::new(1, clip.height(), clip.width()),
            &device,
        )
        .expect("core display tensor");
        assert_tensor_values_close(
            prepared
                .visualization
                .visualization_tensor
                .expect("display tensor should be raw latest frame"),
            expected_display,
        );
    }

    #[test]
    fn camera_stream_rgba_queue_uses_core_interleaved_layout() {
        if skip_native_wgpu_test_on_github_actions(
            "camera_stream_rgba_queue_uses_core_interleaved_layout",
        ) {
            return;
        }

        let first = vec![1, 2, 3, 91, 4, 5, 6, 92];
        let second = vec![7, 8, 9, 93, 10, 11, 12, 94];
        let mut queue = FrameQueue::default();
        queue.push(
            Arc::new(RgbaImage::from_raw(2, 1, first.clone()).expect("first RGBA frame")),
            2,
        );
        queue.push(
            Arc::new(RgbaImage::from_raw(2, 1, second.clone()).expect("second RGBA frame")),
            2,
        );
        let clip = queue
            .build_clip(2)
            .expect("pack camera frames")
            .expect("complete camera clip");

        let mut expected_rgba = first;
        expected_rgba.extend_from_slice(&second);
        assert_eq!(clip.rgba(), expected_rgba.as_slice());

        let device = AutoGazeBevyDevice::default();
        let options = VisualizationOptions::new(1.0, 0.0, false, BevyDisplayTransfer::Gpu);
        let prepared = prepare_autogaze_run(
            &clip,
            false,
            AutoGazeInferenceMode::TiledResizeToGrid {
                tile_size: MODEL_INPUT_SIZE,
            },
            options,
            &device,
        )
        .expect("prepare camera clip");
        assert_eq!(
            prepared.trace_input.video.shape().dims::<5>(),
            [1, 2, 3, 1, 2]
        );
        assert_eq!(
            prepared.visualization.display_input_residency,
            DisplayInputResidency::ModelTensorReuse
        );
        let expected_model =
            rgba_clip_to_tensor::<AutoGazeBevyBackend>(clip.rgba(), clip.shape(), &device)
                .expect("core tensor from packed RGBA camera clip");
        assert_tensor_values_close(prepared.trace_input.video.clone(), expected_model);
        let expected_display = video_frame_tensor(
            prepared.trace_input.video.clone(),
            prepared.trace_input.frame_index,
        )
        .expect("display frame from prepared tiled tensor");
        assert_tensor_values_close(
            prepared
                .visualization
                .visualization_tensor
                .expect("tiled display tensor should reuse model tensor"),
            expected_display,
        );
    }

    #[test]
    fn patch_diff_gpu_visualization_uses_device_mask_and_updates_host_preview_points() {
        if skip_native_wgpu_test_on_github_actions(
            "patch_diff_gpu_visualization_uses_device_mask_and_updates_host_preview_points",
        ) {
            return;
        }

        let device = AutoGazeBevyDevice::default();
        let width = 4;
        let height = 4;
        let previous = vec![0; width * height * 4];
        let mut current = previous.clone();
        for y in 0..2 {
            for x in 0..2 {
                let offset = (y * width + x) * 4;
                current[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let mut rgba = previous;
        rgba.extend_from_slice(&current);
        let clip = FrameClip::from_core(
            AutoGazeRgbaFrameClip::new(rgba, width, height, 2).expect("patch-diff frame clip"),
            0.0,
        );
        let context = AutoGazeRunContext {
            clip: &clip,
            sequence: 0,
            streaming_state: BevyStreamingGenerationState::default(),
            use_streaming_cache: false,
            context_frames: 2,
            top_k: 1,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            visualization_options: VisualizationOptions::new(
                1.0,
                0.0,
                false,
                BevyDisplayTransfer::Gpu,
            ),
            visualization_state: BevyVisualizationState::new(
                AutoGazeVisualizationMode::FullBlend,
                0,
            ),
            device,
            log_pipeline_timing: false,
        };

        let (visualization, _state, _streaming, points) = block_on(run_patch_diff_visualization(
            context,
            AutoGazePatchDiffConfig::new(2, 0.01),
        ))
        .expect("patch-diff GPU visualization");

        assert_eq!(
            points.len(),
            1,
            "GPU patch-diff should read back the compact score grid so async preview uses patch-diff points"
        );
        assert_eq!(
            visualization.effective_display_transfer,
            BevyDisplayTransfer::Gpu
        );
        assert_eq!(visualization.mask_plan_stats.pixel_count, 4);
        assert_eq!(visualization.mask_plan_stats.rect_count, 1);
        assert!((visualization.gaze_update_ratio - 0.25).abs() < 1.0e-6);
        let timing = visualization.timing.expect("patch-diff timing");
        assert_eq!(
            timing.display_input_residency,
            DisplayInputResidency::ModelTensorReuse
        );
        assert_eq!(timing.generated_tokens, 1);
        assert_eq!(timing.trace_points, 1);
    }

    fn fixture_raw_rgba(
        tensors: &safetensors::SafeTensors<'_>,
        fixture_root: &Path,
    ) -> Option<(Vec<u8>, Vec<usize>)> {
        if tensors.names().contains(&"raw_rgba") {
            let raw = tensors.tensor("raw_rgba").expect("raw_rgba fixture tensor");
            return Some((raw.data().to_vec(), raw.shape().to_vec()));
        }

        let mut frames = Vec::new();
        for frame_idx in 0.. {
            let path = fixture_root.join(format!("raw_rgba_frame_{frame_idx:02}.png"));
            if !path.exists() {
                break;
            }
            let frame = image::open(&path)
                .unwrap_or_else(|err| {
                    panic!("failed to read fixture frame {}: {err}", path.display())
                })
                .to_rgba8();
            frames.push(frame);
        }
        let first = frames.first()?;
        let width = first.width() as usize;
        let height = first.height() as usize;
        let mut rgba = Vec::with_capacity(frames.len() * width * height * 4);
        for frame in frames {
            assert_eq!(
                (frame.width() as usize, frame.height() as usize),
                (width, height),
                "fixture raw RGBA frames must share dimensions"
            );
            rgba.extend_from_slice(frame.as_raw());
        }
        let frame_count = rgba.len() / (width * height * 4);
        Some((rgba, vec![frame_count, height, width, 4]))
    }

    #[test]
    fn gpu_display_transfer_matches_cpu_visualization_outputs() {
        if skip_native_wgpu_test_on_github_actions(
            "gpu_display_transfer_matches_cpu_visualization_outputs",
        ) {
            return;
        }

        let device = AutoGazeBevyDevice::default();
        let width = 4;
        let height = 2;
        let previous = deterministic_test_rgba(width, height, 3);
        let current = deterministic_test_rgba(width, height, 17);
        let points = vec![
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.25, 0.25, 0.5, 1.0, 4),
        ];

        assert_gpu_cpu_visualization_match(
            &current,
            width,
            height,
            &points,
            AutoGazeVisualizationMode::FullBlend,
            1,
            &device,
        );

        let mut cpu_state = BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
        let mut gpu_state = BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
        let options_cpu = VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Cpu)
            .with_full_frame_update_policy(0.0);
        let options_gpu = VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Gpu)
            .with_full_frame_update_policy(0.0);

        let previous_tensor = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous display tensor");
        let current_tensor = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current display tensor");

        let _ = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &previous,
                width,
                height,
                tensor: Some(previous_tensor),
            },
            &points,
            options_gpu,
            &mut gpu_state,
            &device,
        )
        .expect("prime gpu interframe");
        let _ = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &previous,
                width,
                height,
                tensor: None,
            },
            &points,
            options_cpu,
            &mut cpu_state,
            &device,
        )
        .expect("prime cpu interframe");

        let gpu = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &current,
                width,
                height,
                tensor: Some(current_tensor),
            },
            &points,
            options_gpu,
            &mut gpu_state,
            &device,
        )
        .expect("gpu interframe visualization");
        assert_eq!(
            gpu.tensor_interframe_path,
            Some(AutoGazeTensorInterframePath::DenseMask)
        );
        let cpu = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &current,
                width,
                height,
                tensor: None,
            },
            &points,
            options_cpu,
            &mut cpu_state,
            &device,
        )
        .expect("cpu interframe visualization");
        assert_eq!(gpu.width, cpu.width);
        assert_eq!(gpu.height, cpu.height);
        assert_eq!(gpu.gaze_update_ratio, cpu.gaze_update_ratio);
        assert_eq!(
            tensor_visualization_to_rgba(gpu.tensor.expect("gpu tensor"), &device),
            cpu.rgba
        );
    }

    #[test]
    fn bevy_gaze_ratio_and_psnr_follow_mask_not_full_frame_policy() {
        let device = AutoGazeBevyDevice::default();
        let width = 8;
        let height = 4;
        let previous = deterministic_test_rgba(width, height, 41);
        let current = deterministic_test_rgba(width, height, 43);
        let point = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2);
        let options = VisualizationOptions::new(1.0, 0.38, true, BevyDisplayTransfer::Cpu)
            .with_full_frame_update_policy(0.45)
            .with_cpu_panels();
        let mut state = BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);

        visualize_frame_rgba(
            FrameVisualInput {
                rgba: &previous,
                width,
                height,
                tensor: None,
            },
            &[],
            options,
            &mut state,
            &device,
        )
        .expect("prime interframe state");
        let visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &current,
                width,
                height,
                tensor: None,
            },
            &[point],
            options,
            &mut state,
            &device,
        )
        .expect("masked interframe visualization");

        assert_eq!(visualization.gaze_update_ratio, 0.5);
        assert!(!visualization.psnr_db.expect("psnr").is_infinite());
        let VisualizationImageData::PanelsRgba { output_rgba, .. } = visualization.image_data
        else {
            panic!("expected split-panel CPU visualization");
        };
        assert_ne!(output_rgba, current);
        assert_eq!(
            &output_rgba[0..(width / 2 * 4)],
            &current[0..(width / 2 * 4)]
        );
        assert_eq!(
            &output_rgba[(width / 2 * 4)..(width * 4)],
            &previous[(width / 2 * 4)..(width * 4)]
        );
    }

    #[test]
    fn cpu_panel_transfer_uses_split_images_without_side_by_side_buffer() {
        let device = AutoGazeBevyDevice::default();
        let width = 4;
        let height = 2;
        let rgba = deterministic_test_rgba(width, height, 23);
        let points = vec![
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.25, 0.25, 0.5, 1.0, 4),
        ];
        let mut bevy_state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &rgba,
                width,
                height,
                tensor: None,
            },
            &points,
            VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Cpu).with_cpu_panels(),
            &mut bevy_state,
            &device,
        )
        .expect("panel visualization");

        assert!(visualization.rgba.is_empty());
        assert!(visualization.tensor.is_none());
        assert_eq!(visualization.width, (width * 3) as u32);
        assert_eq!(visualization.height, height as u32);
        assert_eq!(visualization.output_rgba_bytes, rgba.len() * 3);

        let mut expected_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let expected = expected_state
            .visualize_rgba_panels(&rgba, width, height, &points, 1.0, 0.38)
            .expect("core panels");
        match visualization.image_data {
            VisualizationImageData::PanelsRgba {
                panel_width,
                panel_height,
                input_rgba,
                mask_rgba,
                output_rgba,
                output_matches_input,
            } => {
                assert_eq!(panel_width, width as u32);
                assert_eq!(panel_height, height as u32);
                assert_eq!(input_rgba, rgba);
                assert_eq!(mask_rgba, expected.mask_rgba);
                assert_eq!(output_rgba, expected.blend_rgba);
                assert!(!output_matches_input);
            }
            _ => panic!("expected split panel visualization payload"),
        }
    }

    #[test]
    fn auto_display_transfer_uses_cpu_panels_for_full_resolution_frames() {
        let device = AutoGazeBevyDevice::default();
        let width = 640;
        let height = 360;
        let rgba = deterministic_test_rgba(width, height, 37);
        let points = vec![FixationPoint::with_grid_extent(
            0.5 / 64.0,
            0.5 / 64.0,
            1.0 / 64.0,
            1.0 / 64.0,
            1.0,
            64,
        )];
        let mut bevy_state = BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0);
        let visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &rgba,
                width,
                height,
                tensor: None,
            },
            &points,
            VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Auto)
                .with_cpu_panels(),
            &mut bevy_state,
            &device,
        )
        .expect("auto display visualization");

        assert_eq!(
            visualization.effective_display_transfer,
            BevyDisplayTransfer::Cpu
        );
        assert!(visualization.tensor.is_none());
        assert_eq!(visualization.output_tensor_bytes, 0);
        assert_eq!(visualization.output_rgba_bytes, rgba.len() * 2);
        match visualization.image_data {
            VisualizationImageData::PanelsRgba {
                output_matches_input,
                output_rgba,
                ..
            } => {
                assert!(output_matches_input);
                assert!(output_rgba.is_empty());
            }
            _ => panic!("expected split panel visualization payload"),
        }
    }

    #[test]
    fn preview_panel_texture_materializes_output_alias_data() {
        let width = 4;
        let height = 3;
        let input_rgba = deterministic_test_rgba(width, height, 41);
        let mask_rgba = vec![0; input_rgba.len()];
        let mut images = Assets::<Image>::default();
        let mut texture = AutoGazeTexture {
            image: images.add(visualization_image(1, 1, vec![0; 4])),
            input_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; input_rgba.len()],
            )),
            mask_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; input_rgba.len()],
            )),
            output_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![255; input_rgba.len()],
            )),
            ..AutoGazeTexture::default()
        };

        let visualization = Visualization {
            width: (width * 3) as u32,
            height: height as u32,
            rgba: Vec::new(),
            tensor: None,
            image_data: VisualizationImageData::PanelsRgba {
                panel_width: width as u32,
                panel_height: height as u32,
                input_rgba: input_rgba.clone(),
                mask_rgba,
                output_rgba: Vec::new(),
                output_matches_input: true,
            },
            gaze_update_ratio: 0.0,
            output_update_ratio: 0.0,
            interframe_keyframe: false,
            psnr_db: None,
            visualize_cpu_ms: 0.0,
            psnr_ms: 0.0,
            tensor_ms: 0.0,
            output_rgba_bytes: input_rgba.len() * 2,
            output_tensor_bytes: 0,
            tensor_interframe_path: None,
            effective_display_transfer: BevyDisplayTransfer::Cpu,
            mask_plan_stats: AutoGazeMaskPlanStats::default(),
            timing: None,
        };

        apply_visualization_to_texture(visualization, &mut texture, &mut images);

        let output = images
            .get(&texture.output_image)
            .expect("output panel image");
        let output_data = output.data.as_ref().expect("output panel data");
        assert_eq!(output_data.len(), input_rgba.len());
        assert_eq!(output_data, &input_rgba);
    }

    #[derive(Resource, Clone)]
    struct PreviewDisplayFixture {
        width: usize,
        height: usize,
        input_rgba: Vec<u8>,
        mask_rgba: Vec<u8>,
        output_rgba: Vec<u8>,
    }

    #[test]
    fn preview_display_update_writes_visible_nonblack_panels() {
        use crate::display::AutoGazeTextureLayout;

        let width = 4;
        let height = 3;
        let input_rgba = deterministic_test_rgba(width, height, 53);
        let mask_rgba = deterministic_test_rgba(width, height, 59);
        let output_rgba = deterministic_test_rgba(width, height, 61);
        assert_ne!(input_rgba, vec![0; input_rgba.len()]);

        let mut images = Assets::<Image>::default();
        let side_by_side_rgba = vec![0; 4];
        let texture = AutoGazeTexture {
            image: images.add(visualization_image(1, 1, side_by_side_rgba)),
            input_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; input_rgba.len()],
            )),
            mask_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; mask_rgba.len()],
            )),
            output_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; output_rgba.len()],
            )),
            layout: AutoGazeTextureLayout::SideBySide,
            ..AutoGazeTexture::default()
        };

        let mut app = App::new();
        let side_by_side_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::Flex,
                ..default()
            })
            .id();
        let input_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();
        let mask_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();
        let output_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();

        let mut texture = texture;
        let input_handle = texture.input_image.clone();
        let mask_handle = texture.mask_image.clone();
        let output_handle = texture.output_image.clone();
        texture.side_by_side_entity = Some(side_by_side_entity);
        texture.input_entity = Some(input_entity);
        texture.mask_entity = Some(mask_entity);
        texture.output_entity = Some(output_entity);
        app.insert_resource(images);
        app.insert_resource(texture);
        app.insert_resource(PreviewDisplayFixture {
            width,
            height,
            input_rgba: input_rgba.clone(),
            mask_rgba: mask_rgba.clone(),
            output_rgba: output_rgba.clone(),
        });
        app.add_systems(
            Update,
            |fixture: Res<PreviewDisplayFixture>,
             mut texture: ResMut<AutoGazeTexture>,
             mut images: ResMut<Assets<Image>>,
             mut nodes: Query<&mut Node>| {
                let visualization = Visualization {
                    width: (fixture.width * 3) as u32,
                    height: fixture.height as u32,
                    rgba: Vec::new(),
                    tensor: None,
                    image_data: VisualizationImageData::PanelsRgba {
                        panel_width: fixture.width as u32,
                        panel_height: fixture.height as u32,
                        input_rgba: fixture.input_rgba.clone(),
                        mask_rgba: fixture.mask_rgba.clone(),
                        output_rgba: fixture.output_rgba.clone(),
                        output_matches_input: false,
                    },
                    gaze_update_ratio: 0.25,
                    output_update_ratio: 0.25,
                    interframe_keyframe: false,
                    psnr_db: Some(42.0),
                    visualize_cpu_ms: 0.0,
                    psnr_ms: 0.0,
                    tensor_ms: 0.0,
                    output_rgba_bytes: fixture.input_rgba.len() * 3,
                    output_tensor_bytes: 0,
                    tensor_interframe_path: None,
                    effective_display_transfer: BevyDisplayTransfer::Cpu,
                    mask_plan_stats: AutoGazeMaskPlanStats::default(),
                    timing: None,
                };
                apply_visualization_to_preview_display(
                    visualization,
                    &mut texture,
                    &mut images,
                    &mut nodes,
                );
            },
        );
        app.update();

        let texture = app.world().resource::<AutoGazeTexture>();
        assert_eq!(texture.layout, AutoGazeTextureLayout::Panels);
        assert_eq!(texture.width, (width * 3) as u32);
        assert_eq!(texture.height, height as u32);
        assert_eq!(
            app.world()
                .get::<Node>(side_by_side_entity)
                .expect("side node")
                .display,
            Display::None
        );
        assert_eq!(
            app.world()
                .get::<Node>(input_entity)
                .expect("input node")
                .display,
            Display::Flex
        );
        assert_eq!(
            app.world()
                .get::<Node>(mask_entity)
                .expect("mask node")
                .display,
            Display::Flex
        );
        assert_eq!(
            app.world()
                .get::<Node>(output_entity)
                .expect("output node")
                .display,
            Display::Flex
        );

        let images = app.world().resource::<Assets<Image>>();
        assert_eq!(
            images
                .get(&input_handle)
                .expect("input image")
                .data
                .as_ref()
                .expect("input data"),
            &input_rgba
        );
        assert_eq!(
            images
                .get(&mask_handle)
                .expect("mask image")
                .data
                .as_ref()
                .expect("mask data"),
            &mask_rgba
        );
        assert_eq!(
            images
                .get(&output_handle)
                .expect("output image")
                .data
                .as_ref()
                .expect("output data"),
            &output_rgba
        );
    }

    #[test]
    fn preview_panel_texture_makes_panel_nodes_visible() {
        use crate::display::AutoGazeTextureLayout;

        let mut app = App::new();
        app.add_systems(
            Update,
            |texture: Res<AutoGazeTexture>, mut nodes: Query<&mut Node>| {
                sync_texture_layout_nodes(&texture, &mut nodes);
            },
        );

        let side_by_side_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::Flex,
                ..default()
            })
            .id();
        let input_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();
        let mask_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();
        let output_entity = app
            .world_mut()
            .spawn(Node {
                display: Display::None,
                ..default()
            })
            .id();
        app.insert_resource(AutoGazeTexture {
            side_by_side_entity: Some(side_by_side_entity),
            input_entity: Some(input_entity),
            mask_entity: Some(mask_entity),
            output_entity: Some(output_entity),
            layout: AutoGazeTextureLayout::Panels,
            ..AutoGazeTexture::default()
        });
        app.update();

        assert_eq!(
            app.world()
                .get::<Node>(side_by_side_entity)
                .expect("side node")
                .display,
            Display::None
        );
        assert_eq!(
            app.world()
                .get::<Node>(input_entity)
                .expect("input node")
                .display,
            Display::Flex
        );
        assert_eq!(
            app.world()
                .get::<Node>(mask_entity)
                .expect("mask node")
                .display,
            Display::Flex
        );
        assert_eq!(
            app.world()
                .get::<Node>(output_entity)
                .expect("output node")
                .display,
            Display::Flex
        );

        app.world_mut().resource_mut::<AutoGazeTexture>().layout =
            AutoGazeTextureLayout::SideBySide;
        app.update();

        assert_eq!(
            app.world()
                .get::<Node>(side_by_side_entity)
                .expect("side node")
                .display,
            Display::Flex
        );
        assert_eq!(
            app.world()
                .get::<Node>(input_entity)
                .expect("input node")
                .display,
            Display::None
        );
        assert_eq!(
            app.world()
                .get::<Node>(mask_entity)
                .expect("mask node")
                .display,
            Display::None
        );
        assert_eq!(
            app.world()
                .get::<Node>(output_entity)
                .expect("output node")
                .display,
            Display::None
        );
    }

    #[test]
    fn auto_display_transfer_keeps_tensor_path_for_model_sized_frames() {
        assert!(uses_tensor_display_transfer(
            BevyDisplayTransfer::Auto,
            224,
            224
        ));
        assert!(!uses_tensor_display_transfer(
            BevyDisplayTransfer::Auto,
            640,
            360
        ));
        assert!(uses_tensor_display_transfer(
            BevyDisplayTransfer::Gpu,
            1920,
            1080
        ));
        assert!(!uses_tensor_display_transfer(
            BevyDisplayTransfer::Cpu,
            224,
            224
        ));
        assert!(!should_use_device_token_readout(
            BevyDisplayTransfer::Auto,
            640,
            360,
            true
        ));
        assert!(should_use_device_token_readout(
            BevyDisplayTransfer::Gpu,
            640,
            360,
            true
        ));
        assert!(!should_use_device_token_readout(
            BevyDisplayTransfer::Gpu,
            640,
            360,
            false
        ));
    }

    #[test]
    fn gpu_display_transfer_uses_split_panel_tensors() {
        if skip_native_wgpu_test_on_github_actions("gpu_display_transfer_uses_split_panel_tensors")
        {
            return;
        }

        let device = AutoGazeBevyDevice::default();
        let width = 4;
        let height = 2;
        let rgba = deterministic_test_rgba(width, height, 29);
        let points = vec![
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.25, 0.25, 0.5, 1.0, 4),
        ];
        let tensor = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("display tensor");
        let mut bevy_state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &rgba,
                width,
                height,
                tensor: Some(tensor),
            },
            &points,
            VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Gpu),
            &mut bevy_state,
            &device,
        )
        .expect("gpu panel visualization");

        assert!(visualization.rgba.is_empty());
        assert!(visualization.tensor.is_some());
        assert_eq!(visualization.width, (width * 3) as u32);
        assert_eq!(visualization.height, height as u32);
        assert_eq!(visualization.tensor_interframe_path, None);

        let mut expected_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let expected = expected_state
            .visualize_rgba_panels(&rgba, width, height, &points, 1.0, 0.38)
            .expect("core panels");
        match visualization.image_data {
            VisualizationImageData::TensorPanels(panels) => {
                let TensorPanelVisualizationData {
                    panel_width,
                    panel_height,
                    input_rgba,
                    mask_rgba,
                    output_rgba,
                    output_matches_input,
                } = *panels;
                assert_eq!(panel_width, width as u32);
                assert_eq!(panel_height, height as u32);
                assert!(!output_matches_input);
                assert_eq!(tensor_visualization_to_rgba(input_rgba, &device), rgba);
                assert_eq!(
                    tensor_visualization_to_rgba(mask_rgba, &device),
                    expected.mask_rgba
                );
                assert_eq!(
                    tensor_visualization_to_rgba(output_rgba, &device),
                    expected.blend_rgba
                );
            }
            _ => panic!("expected split tensor panel visualization payload"),
        }
    }

    #[test]
    fn gpu_display_transfer_keeps_tensor_path_when_psnr_is_enabled() {
        if skip_native_wgpu_test_on_github_actions(
            "gpu_display_transfer_keeps_tensor_path_when_psnr_is_enabled",
        ) {
            return;
        }

        let device = AutoGazeBevyDevice::default();
        let width = 4;
        let height = 2;
        let rgba = deterministic_test_rgba(width, height, 31);
        let points = vec![FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2)];
        let tensor = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("display tensor");
        let mut bevy_state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let mut visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &rgba,
                width,
                height,
                tensor: Some(tensor),
            },
            &points,
            VisualizationOptions::new(1.0, 0.38, true, BevyDisplayTransfer::Gpu),
            &mut bevy_state,
            &device,
        )
        .expect("gpu psnr visualization");
        block_on(calculate_tensor_psnr_if_needed(&mut visualization, true)).expect("tensor psnr");

        let mut expected_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let expected = expected_state
            .visualize_rgba_panels(&rgba, width, height, &points, 1.0, 0.38)
            .expect("core panels");
        let expected_psnr = expected.output_psnr_db(&rgba).expect("expected psnr");

        assert!(matches!(
            visualization.image_data,
            VisualizationImageData::TensorPanels(_)
        ));
        assert!(visualization.tensor.is_some());
        assert_eq!(visualization.output_rgba_bytes, 0);
        assert!(visualization.output_tensor_bytes > 0);
        assert!((visualization.psnr_db.expect("psnr") - expected_psnr).abs() <= 0.2);
        assert_eq!(visualization.visualize_cpu_ms, 0.0);
        assert!(visualization.psnr_ms >= 0.0);
    }

    #[test]
    fn interframe_keyframe_psnr_samples_are_suppressed() {
        let width = 2;
        let height = 1;
        let points = [FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0)];
        let mut visualization_state =
            BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 2);
        let mut streaming_state = BevyStreamingGenerationState::default();
        streaming_state.configure(true, width, height, 16);
        let cache_horizon = streaming_state
            .cache
            .as_ref()
            .expect("streaming cache")
            .horizon_frames();

        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        let first_visualization = visualize_rgba_bytes(
            &first,
            width,
            height,
            &points,
            VisualizationOptions::new(1.0, 0.38, true, BevyDisplayTransfer::Cpu),
            &mut visualization_state,
        )
        .expect("first visualization");
        assert_eq!(first_visualization.psnr_db, None);
        assert!(first_visualization.interframe_keyframe);
        assert!(visualization_state.cpu.last_frame_was_keyframe());
        assert_eq!(
            streaming_state
                .cache
                .as_ref()
                .expect("streaming cache")
                .horizon_frames(),
            cache_horizon
        );

        let second = [30, 0, 0, 255, 40, 0, 0, 255];
        let second_visualization = visualize_rgba_bytes(
            &second,
            width,
            height,
            &points,
            VisualizationOptions::new(1.0, 0.38, true, BevyDisplayTransfer::Cpu),
            &mut visualization_state,
        )
        .expect("second visualization");
        assert!(second_visualization.psnr_db.is_some());
        assert!(!second_visualization.interframe_keyframe);
        assert!(!visualization_state.cpu.last_frame_was_keyframe());

        let third = [50, 0, 0, 255, 60, 0, 0, 255];
        let third_visualization = visualize_rgba_bytes(
            &third,
            width,
            height,
            &points,
            VisualizationOptions::new(1.0, 0.38, true, BevyDisplayTransfer::Cpu),
            &mut visualization_state,
        )
        .expect("third visualization");
        assert_eq!(third_visualization.psnr_db, None);
        assert!(third_visualization.interframe_keyframe);
        assert!(visualization_state.cpu.last_frame_was_keyframe());
        assert_eq!(
            streaming_state
                .cache
                .as_ref()
                .expect("streaming cache")
                .horizon_frames(),
            cache_horizon
        );
    }

    #[test]
    fn live_preview_uses_stable_panel_layout_and_reports_psnr() {
        let width = 3;
        let height = 2;
        let rgba = deterministic_test_rgba(width, height, 9);
        let frame = RgbaImage::from_raw(width as u32, height as u32, rgba.clone())
            .expect("test rgba frame");

        let visualization =
            live_preview_visualization(&frame, true).expect("live preview visualization");

        assert_eq!(visualization.width, (width * 3) as u32);
        assert_eq!(visualization.height, height as u32);
        assert_eq!(visualization.gaze_update_ratio, 0.0);
        assert_eq!(visualization.psnr_db, Some(f64::INFINITY));
        match visualization.image_data {
            VisualizationImageData::PanelsRgba {
                panel_width,
                panel_height,
                input_rgba,
                mask_rgba,
                output_rgba,
                output_matches_input,
            } => {
                assert_eq!(panel_width, width as u32);
                assert_eq!(panel_height, height as u32);
                assert_eq!(input_rgba, rgba);
                assert_eq!(output_rgba, rgba);
                assert!(!output_matches_input);
                assert!(mask_rgba.iter().all(|value| *value == 0));
            }
            _ => panic!("live preview should use the same split-panel texture layout as inference"),
        }
    }

    #[test]
    fn live_preview_skips_psnr_when_overlay_is_disabled() {
        let width = 2;
        let height = 2;
        let rgba = deterministic_test_rgba(width, height, 12);
        let frame =
            RgbaImage::from_raw(width as u32, height as u32, rgba).expect("test rgba frame");

        let visualization =
            live_preview_visualization(&frame, false).expect("live preview visualization");

        assert_eq!(visualization.psnr_db, None);
    }

    #[test]
    fn async_mask_preview_applies_latest_mask_to_new_frame() {
        let width = 4;
        let height = 4;
        let first = RgbaImage::from_pixel(width as u32, height as u32, image::Rgba([8, 8, 8, 255]));
        let second = RgbaImage::from_pixel(
            width as u32,
            height as u32,
            image::Rgba([220, 120, 40, 255]),
        );
        let points = vec![FixationPoint::with_extent(0.5, 0.5, 0.5, 0.5, 1.0)];
        let config = BevyBurnAutoGazeConfig {
            visualization_mode: AutoGazeVisualizationMode::Interframe,
            keyframe_duration: 0,
            show_psnr: true,
            ..Default::default()
        };
        let mut state =
            BevyVisualizationState::new(config.visualization_mode, config.keyframe_duration);

        async_mask_preview_visualization(&first, &[], &config, &mut state)
            .expect("initial preview keyframe");
        let visualization = async_mask_preview_visualization(&second, &points, &config, &mut state)
            .expect("masked preview");

        match visualization.image_data {
            VisualizationImageData::PanelsRgba {
                mask_rgba,
                output_rgba,
                ..
            } => {
                let masked_pixels = mask_rgba
                    .chunks_exact(4)
                    .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                    .count();
                let updated_pixels = output_rgba
                    .chunks_exact(4)
                    .filter(|pixel| pixel[0] == 220)
                    .count();

                assert!(masked_pixels > 0);
                assert_eq!(updated_pixels, masked_pixels);
                assert!(visualization.psnr_db.is_some());
            }
            _ => panic!("expected panel visualization payload"),
        }
    }

    fn assert_gpu_cpu_visualization_match(
        rgba: &[u8],
        width: usize,
        height: usize,
        points: &[FixationPoint],
        mode: AutoGazeVisualizationMode,
        keyframe_duration: usize,
        device: &AutoGazeBevyDevice,
    ) {
        let tensor = rgba_clip_to_tensor::<AutoGazeBevyBackend>(
            rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            device,
        )
        .expect("display tensor");
        let mut gpu_state = BevyVisualizationState::new(mode, keyframe_duration);
        let mut cpu_state = BevyVisualizationState::new(mode, keyframe_duration);
        let gpu = visualize_frame_rgba(
            FrameVisualInput {
                rgba,
                width,
                height,
                tensor: Some(tensor),
            },
            points,
            VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Gpu),
            &mut gpu_state,
            device,
        )
        .expect("gpu visualization");
        let cpu = visualize_frame_rgba(
            FrameVisualInput {
                rgba,
                width,
                height,
                tensor: None,
            },
            points,
            VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Cpu),
            &mut cpu_state,
            device,
        )
        .expect("cpu visualization");

        assert_eq!(gpu.width, cpu.width);
        assert_eq!(gpu.height, cpu.height);
        assert_eq!(gpu.gaze_update_ratio, cpu.gaze_update_ratio);
        assert_eq!(
            tensor_visualization_to_rgba(gpu.tensor.expect("gpu tensor"), device),
            cpu.rgba
        );
    }

    fn tensor_visualization_to_rgba(
        tensor: Tensor<AutoGazeBevyBackend, 3>,
        device: &AutoGazeBevyDevice,
    ) -> Vec<u8> {
        <AutoGazeBevyBackend as burn::tensor::backend::Backend>::sync(device)
            .expect("sync gpu visualization tensor");
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("gpu visualization tensor data")
            .into_iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    fn deterministic_test_rgba(width: usize, height: usize, seed: usize) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                rgba.push(((x * 31 + seed) % 256) as u8);
                rgba.push(((y * 47 + seed * 3) % 256) as u8);
                rgba.push(((x * 11 + y * 13 + seed * 5) % 256) as u8);
                rgba.push(255);
            }
        }
        rgba
    }

    fn panel_visualization_payload(
        width: usize,
        height: usize,
        rgba: Vec<u8>,
    ) -> VisualizationImageData {
        let len = width * height * 4;
        VisualizationImageData::PanelsRgba {
            panel_width: width as u32,
            panel_height: height as u32,
            input_rgba: rgba.clone(),
            mask_rgba: vec![0; len],
            output_rgba: rgba,
            output_matches_input: false,
        }
    }

    fn panel_image_data(world: &World, handle: &Handle<Image>) -> Vec<u8> {
        world
            .get_resource::<Assets<Image>>()
            .expect("image assets")
            .get(handle)
            .expect("panel image")
            .data
            .as_ref()
            .expect("panel image data")
            .clone()
    }

    fn assert_tensor_values_close(
        left: Tensor<AutoGazeBevyBackend, 5>,
        right: Tensor<AutoGazeBevyBackend, 5>,
    ) {
        let left_shape = left.shape().dims::<5>();
        let right_shape = right.shape().dims::<5>();
        assert_eq!(left_shape, right_shape);
        let left = left.into_data().to_vec::<f32>().expect("left tensor data");
        let right = right
            .into_data()
            .to_vec::<f32>()
            .expect("right tensor data");
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1.0e-6,
                "tensor value {index} diverged: {left} vs {right}"
            );
        }
    }

    fn skip_native_wgpu_test_on_github_actions(test_name: &str) -> bool {
        if std::env::var_os("GITHUB_ACTIONS").is_some() {
            eprintln!(
                "skipping {test_name}: GitHub Actions does not provide a stable native WGPU/CubeCL device; wasm WebGPU is covered by the Pages browser workflow"
            );
            return true;
        }

        false
    }

    #[test]
    fn bevy_output_mask_uses_native_multiscale_extents() {
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.25, 0.75, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.75, 0.5, 0.5, 1.0, 2),
        ];
        let alpha = fixation_alpha_mask(28, 28, &points, 1.0);

        assert_eq!(alpha.iter().filter(|&&value| value > 0).count(), 28 * 28);
    }

    #[test]
    fn bevy_mask_panel_matches_output_update_mask_cells() {
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(
                11.5 / 14.0,
                11.5 / 14.0,
                1.0 / 14.0,
                1.0 / 14.0,
                1.0,
                14,
            ),
        ];
        let alpha = fixation_alpha_mask(28, 28, &points, 1.0);
        let mask = fixation_scale_mask_rgba(28, 28, &points, 1.0);

        for (index, pixel) in mask.chunks_exact(4).enumerate() {
            let colored = pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0;
            assert_eq!(colored, alpha[index] > 0, "pixel {index}");
        }
    }

    #[test]
    fn default_blend_alpha_keeps_live_output_readable() {
        let config = BevyBurnAutoGazeConfig::default();

        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
    }

    #[test]
    fn metric_resources_delegate_to_core_stats() {
        let mut stats = GazeRatioStats::default();

        stats.record(2.0);
        assert!(stats.0.is_initialized());
        assert_eq!(stats.0.current(), 1.0);
        assert_eq!(stats.0.ema(), 1.0);
        assert_eq!(format_gaze_ratio_percent(stats.0.current()), "100.0%");
        stats.record(f64::NAN);
        assert_eq!(stats.0.current(), 0.0);

        let mut stats = PsnrStats::default();
        stats.record(f64::INFINITY);
        assert!(stats.0.is_initialized());
        assert!(stats.0.current().is_infinite());
        assert_eq!(format_psnr_db(stats.0.current()), "999.9");
        stats.record(f64::NAN);
        assert!(stats.0.current().is_infinite());
    }

    #[test]
    fn metric_overlay_strings_keep_constant_width() {
        assert_eq!(format_fps(f64::NAN), "---.-");
        assert_eq!(format_fps(0.0), "000.0");
        assert_eq!(format_fps(51.25), "051.2");
        assert_eq!(format_fps(1000.0), "999.9");

        let empty_fps = stable_fps_text(None, None);
        let mixed_fps = stable_fps_text(Some(60.0), Some(12.5));
        let saturated_fps = stable_fps_text(Some(1000.0), Some(1000.0));
        assert_eq!(empty_fps, "render ---.- infer ---.-");
        assert_eq!(mixed_fps, "render 060.0 infer 012.5");
        assert_eq!(saturated_fps, "render 999.9 infer 999.9");
        assert_eq!(empty_fps.len(), mixed_fps.len());
        assert_eq!(mixed_fps.len(), saturated_fps.len());

        let empty_gaze = stable_gaze_ratio_text(None, None);
        let low_gaze = stable_gaze_ratio_text(Some(0.0), Some(0.125));
        let high_gaze = stable_gaze_ratio_text(Some(1.0), Some(0.875));
        assert_eq!(empty_gaze, "---.-% ema ---.-%");
        assert_eq!(low_gaze, "000.0% ema 012.5%");
        assert_eq!(high_gaze, "100.0% ema 087.5%");
        assert_eq!(empty_gaze.len(), low_gaze.len());
        assert_eq!(low_gaze.len(), high_gaze.len());

        let empty_psnr = stable_psnr_text(None, None);
        let finite_psnr = stable_psnr_text(Some(34.25), Some(100.0));
        let infinite_psnr = stable_psnr_text(Some(f64::INFINITY), Some(f64::INFINITY));
        assert_eq!(empty_psnr, "---.- dB ema ---.- dB");
        assert_eq!(finite_psnr, "034.2 dB ema 100.0 dB");
        assert_eq!(infinite_psnr, "999.9 dB ema 999.9 dB");
        assert_eq!(empty_psnr.len(), finite_psnr.len());
        assert_eq!(finite_psnr.len(), infinite_psnr.len());
    }

    #[test]
    fn task_loss_slider_maps_right_to_higher_quality() {
        assert_eq!(task_loss_slider_percent(TASK_LOSS_SLIDER_MIN), 0.0);
        assert_eq!(task_loss_slider_percent(TASK_LOSS_SLIDER_MAX), 1.0);
        assert_eq!(
            task_loss_slider_value_from_normalized_x(-0.5),
            TASK_LOSS_SLIDER_MIN
        );
        assert_eq!(
            task_loss_slider_value_from_normalized_x(0.5),
            TASK_LOSS_SLIDER_MAX
        );
        assert!(
            task_loss_slider_value_from_normalized_x(0.25)
                > task_loss_slider_value_from_normalized_x(-0.25)
        );
        assert_eq!(quality_slider_threshold(1.0), 0.0);
        assert_eq!(quality_slider_threshold(0.0), 1.0);
        assert_eq!(quality_slider_quality_from_threshold(0.45), 0.55);
        assert_eq!(task_loss_slider_label(TASK_LOSS_SLIDER_MIN), "  0.0%");
        assert_eq!(task_loss_slider_label(TASK_LOSS_SLIDER_MAX), "100.0%");
    }

    #[test]
    fn task_loss_slider_uses_clean_quality_mapping_and_unlimits_autogaze_decode() {
        let mut config = BevyBurnAutoGazeConfig {
            sparse_mask_source: BevySparseMaskSource::AutoGaze,
            max_gaze_tokens_each_frame: 19,
            limit_generation_budget: true,
            task_loss_requirement: Some(0.56),
            ..Default::default()
        };
        let mut model_config = config.clone();
        let mut slider = TaskLossSliderState::new(&config);

        apply_task_loss_slider_value(&mut slider, &mut config, &mut model_config, 0.61);

        assert_eq!(slider.value, 0.61);
        assert_eq!(slider.pending_value, Some(0.39));
        assert_eq!(config.task_loss_requirement, Some(0.39));
        assert_eq!(model_config.task_loss_requirement, Some(0.39));
        assert_eq!(config.max_gaze_tokens_each_frame, 0);
        assert_eq!(model_config.max_gaze_tokens_each_frame, 0);
        assert!(!config.limit_generation_budget);
        assert!(!model_config.limit_generation_budget);
    }

    #[test]
    fn quality_slider_updates_patch_diff_threshold_when_patch_diff_is_active() {
        let mut config = BevyBurnAutoGazeConfig {
            sparse_mask_source: BevySparseMaskSource::PatchDiff,
            patch_diff_threshold: 0.45,
            task_loss_requirement: Some(0.45),
            ..Default::default()
        };
        let mut model_config = config.clone();
        let mut slider = TaskLossSliderState::new(&config);

        apply_task_loss_slider_value(&mut slider, &mut config, &mut model_config, 0.69);

        assert_eq!(slider.value, 0.69);
        assert_eq!(slider.pending_value, None);
        assert_eq!(config.patch_diff_threshold, 0.31);
        assert_eq!(model_config.patch_diff_threshold, 0.31);
        assert_eq!(config.task_loss_requirement, Some(0.45));
        assert_eq!(model_config.task_loss_requirement, Some(0.45));
    }

    #[test]
    fn metric_overlay_reserves_top_space_above_visualization() {
        let mut config = BevyBurnAutoGazeConfig {
            show_fps: false,
            show_gaze_ratio: false,
            show_psnr: false,
            show_task_loss_slider: false,
            ..Default::default()
        };
        assert_eq!(metric_overlay_top(0), UI_MARGIN_PX);
        assert_eq!(
            metric_overlay_top(2),
            UI_MARGIN_PX + METRIC_ROW_HEIGHT * 2.0
        );
        assert_eq!(metric_panel_top_reserved_height(&config), UI_MARGIN_PX);

        config.show_fps = true;
        assert_eq!(
            metric_panel_top_reserved_height(&config),
            UI_MARGIN_PX * 2.0 + METRIC_ROW_HEIGHT
        );

        config.show_gaze_ratio = true;
        config.show_psnr = true;
        config.show_task_loss_slider = true;
        assert_eq!(
            metric_panel_top_reserved_height(&config),
            UI_MARGIN_PX * 2.0 + METRIC_ROW_HEIGHT * 4.0
        );
    }

    #[test]
    fn inference_timing_summary_json_reports_well_formed_metrics() {
        let mut stats = InferenceTimingStats::default();
        stats.set_render_adapter(RenderAdapterSummary {
            name: "test adapter".to_string(),
            vendor: 1234,
            device_type: "DiscreteGpu".to_string(),
            backend: "Vulkan".to_string(),
            driver: "test-driver".to_string(),
            driver_info: "test-driver-info".to_string(),
        });
        stats.set_run_config(InferenceRunConfigSummary::from(&BevyBurnAutoGazeConfig {
            source: BevyFrameSource::SyntheticPan,
            mode: BevyAutoGazeMode::Tile224,
            visualization_mode: AutoGazeVisualizationMode::Interframe,
            display_transfer: BevyDisplayTransfer::Gpu,
            streaming_cache: true,
            max_in_flight: 3,
            frames_per_clip: 16,
            top_k: 5,
            max_gaze_tokens_each_frame: 7,
            tile_batch_size: 9,
            inference_width: Some(1280),
            inference_height: Some(720),
            tensor_sparse_update_max_rects: 8,
            tensor_sparse_update_max_ratio: 0.05,
            tensor_full_frame_update_min_ratio: 0.45,
            show_psnr: true,
            ..Default::default()
        }));
        stats.record(
            InferenceTiming {
                sequence: 7,
                clip_frames: 16,
                model_frames: 2,
                effective_generation_budget: 32,
                generated_tokens: 10,
                active_generated_tokens: 8,
                padded_generated_tokens: 2,
                trace_points: 42,
                active_trace_points: 9,
                width: 640,
                height: 360,
                total_ms: 20.0,
                model_ms: 8.0,
                input_ms: 2.0,
                display_input_ms: 0.25,
                pack_ms: 1.0,
                visualize_ms: 3.0,
                visualize_cpu_ms: 2.5,
                psnr_ms: 0.75,
                display_ms: 1.0,
                tensor_ms: 0.5,
                output_tensor_bytes: 640 * 360 * 3 * 4 * std::mem::size_of::<f32>(),
                display_input_residency: DisplayInputResidency::ModelTensorReuse,
                effective_display_transfer: BevyDisplayTransfer::Gpu,
                gaze_update_ratio: 0.25,
                gaze_update_ratio_sample: Some(0.25),
                output_update_ratio: 0.20,
                output_update_ratio_sample: Some(0.20),
                psnr_db: Some(42.0),
                tensor_interframe_path: Some(AutoGazeTensorInterframePath::SparseRects),
                mask_plan_stats: AutoGazeMaskPlanStats {
                    rect_count: 12,
                    row_span_count: 34,
                    pixel_count: 57_600,
                },
                ..Default::default()
            },
            false,
        );

        let summary: serde_json::Value =
            serde_json::from_str(&stats.summary_json(1)).expect("summary json");
        let sample: serde_json::Value =
            serde_json::from_str(&perf_sample_json(&stats).expect("perf sample"))
                .expect("perf sample json");

        assert_eq!(summary["target_frames"], 1);
        assert_eq!(summary["skipped_warmup_frames"], 0);
        assert_eq!(
            summary["latest_skipped_warmup_sequence"],
            serde_json::Value::Null
        );
        assert_eq!(summary["processed_frames"], 1);
        assert_eq!(summary["processed_model_frames"], 2);
        assert_eq!(summary["latest_sequence"], 7);
        assert_eq!(summary["latest_clip_frames"], 16);
        assert_eq!(summary["latest_model_frames"], 2);
        assert_eq!(summary["latest_effective_generation_budget"], 32);
        assert_eq!(summary["latest_generated_tokens"], 10);
        assert_eq!(summary["latest_active_generated_tokens"], 8);
        assert_eq!(summary["latest_padded_generated_tokens"], 2);
        assert_eq!(summary["latest_trace_points"], 42);
        assert_eq!(summary["latest_active_trace_points"], 9);
        assert_eq!(summary["latest_mask_rects"], 12);
        assert_eq!(summary["latest_mask_row_spans"], 34);
        assert_eq!(summary["latest_mask_pixels"], 57_600);
        assert_eq!(summary["latest_width"], 640);
        assert_eq!(summary["latest_height"], 360);
        assert_eq!(summary["latest_gaze_update_ratio"], 0.25);
        assert_eq!(summary["latest_mask_update_ratio"], 0.25);
        assert_eq!(summary["latest_output_update_ratio"], 0.20);
        assert_eq!(summary["avg_output_update_ratio"], 0.20);
        assert_eq!(summary["stale_results"], 0);
        assert_eq!(summary["latest_tensor_interframe_path"], "sparse-rects");
        assert_eq!(summary["latest_effective_display_transfer"], "gpu");
        assert_eq!(summary["display_residency"], "gpu-tensor");
        assert_eq!(summary["display_input_residency"], "model-tensor-reuse");
        assert_eq!(summary["latest_display_input_ms"], 0.25);
        assert_eq!(summary["latest_output_rgba_bytes"], 0);
        assert_eq!(
            summary["latest_output_tensor_bytes"],
            640 * 360 * 3 * 4 * std::mem::size_of::<f32>()
        );
        assert_eq!(summary["mode"], "tiled");
        assert_eq!(summary["visualization_mode"], "interframe");
        assert_eq!(summary["mask_visualization_mode"], "image-mask-only");
        assert_eq!(summary["mask_geometry_mode"], "deduplicated");
        assert_eq!(summary["display_transfer"], "gpu");
        assert_eq!(summary["streaming_cache"], true);
        assert_eq!(summary["streaming_cache_effective"], false);
        assert_eq!(summary["configured_max_in_flight"], 3);
        assert_eq!(summary["effective_max_in_flight"], 3);
        assert_eq!(summary["frames_per_clip"], 16);
        assert_eq!(summary["top_k"], 5);
        assert_eq!(summary["max_gaze_tokens_each_frame"], 7);
        assert_eq!(summary["tile_batch_size"], 9);
        assert_eq!(summary["inference_width"], 1280);
        assert_eq!(summary["inference_height"], 720);
        assert_eq!(summary["tensor_sparse_update_max_rects"], 8);
        assert_eq!(summary["tensor_sparse_update_max_ratio"], 0.05);
        assert_eq!(summary["tensor_full_frame_update_min_ratio"], 0.45);
        assert_eq!(summary["show_psnr"], true);
        assert_eq!(summary["warmup_model"], true);
        assert_eq!(summary["burn_backend"], "webgpu");
        assert_eq!(summary["render_adapter_name"], "test adapter");
        assert_eq!(summary["render_adapter_vendor"], 1234);
        assert_eq!(summary["render_adapter_device_type"], "DiscreteGpu");
        assert_eq!(summary["render_adapter_backend"], "Vulkan");
        assert_eq!(summary["render_adapter_driver"], "test-driver");
        assert_eq!(summary["render_adapter_driver_info"], "test-driver-info");
        assert_eq!(summary["avg_output_fps"], 50.0);
        assert_eq!(summary["avg_input_fps"], 100.0);
        assert_eq!(summary["avg_model_frame_fps"], 250.0);
        assert_eq!(summary["avg_model_ms"], 8.0);
        assert_eq!(summary["avg_generated_tokens"], 10.0);
        assert_eq!(summary["avg_active_generated_tokens"], 8.0);
        assert_eq!(summary["avg_padded_generated_tokens"], 2.0);
        assert_eq!(summary["avg_trace_points"], 42.0);
        assert_eq!(summary["avg_active_trace_points"], 9.0);
        assert_eq!(summary["avg_mask_rects"], 12.0);
        assert_eq!(summary["avg_mask_row_spans"], 34.0);
        assert_eq!(summary["avg_mask_pixels"], 57_600.0);
        assert_eq!(summary["avg_input_ms"], 2.0);
        assert_eq!(summary["avg_display_input_ms"], 0.25);
        assert_eq!(summary["avg_pack_ms"], 1.0);
        assert_eq!(summary["avg_visualize_ms"], 3.0);
        assert_eq!(summary["avg_psnr_ms"], 0.75);
        assert_eq!(summary["avg_display_ms"], 1.0);
        assert_eq!(summary["avg_output_rgba_bytes"], 0.0);
        assert_eq!(
            summary["avg_output_tensor_bytes"],
            (640 * 360 * 3 * 4 * std::mem::size_of::<f32>()) as f64
        );
        assert_eq!(summary["psnr_samples"], 1);
        assert_eq!(summary["latest_psnr_db"], 42.0);
        assert_eq!(summary["latest_psnr_db_infinite"], false);
        assert_eq!(summary["ema_psnr_db"], 42.0);
        assert_eq!(summary["ema_psnr_db_infinite"], false);
        assert_eq!(summary["source"], "synthetic-pan");
        assert_eq!(sample["processed_frames"], 1);
        assert_eq!(sample["processed_model_frames"], 2);
        assert_eq!(sample["latest_sequence"], 7);
        assert_eq!(sample["latest_clip_frames"], 16);
        assert_eq!(sample["latest_model_frames"], 2);
        assert_eq!(sample["latest_effective_generation_budget"], 32);
        assert_eq!(sample["latest_generated_tokens"], 10);
        assert_eq!(sample["latest_active_generated_tokens"], 8);
        assert_eq!(sample["latest_padded_generated_tokens"], 2);
        assert_eq!(sample["latest_trace_points"], 42);
        assert_eq!(sample["latest_active_trace_points"], 9);
        assert_eq!(sample["avg_generated_tokens"], 10.0);
        assert_eq!(sample["avg_active_generated_tokens"], 8.0);
        assert_eq!(sample["avg_padded_generated_tokens"], 2.0);
        assert_eq!(sample["avg_trace_points"], 42.0);
        assert_eq!(sample["avg_active_trace_points"], 9.0);
        assert_eq!(sample["latest_mask_rects"], 12);
        assert_eq!(sample["latest_mask_row_spans"], 34);
        assert_eq!(sample["latest_mask_pixels"], 57_600);
        assert_eq!(sample["avg_mask_rects"], 12.0);
        assert_eq!(sample["avg_mask_row_spans"], 34.0);
        assert_eq!(sample["avg_mask_pixels"], 57_600.0);
        assert_eq!(sample["latest_width"], 640);
        assert_eq!(sample["latest_height"], 360);
        assert_eq!(sample["latest_gaze_update_ratio"], 0.25);
        assert_eq!(sample["latest_tensor_interframe_path"], "sparse-rects");
        assert_eq!(sample["latest_effective_display_transfer"], "gpu");
        assert_eq!(sample["display_residency"], "gpu-tensor");
        assert_eq!(sample["display_input_residency"], "model-tensor-reuse");
        assert_eq!(sample["latest_display_input_ms"], 0.25);
        assert_eq!(sample["latest_output_rgba_bytes"], 0);
        assert_eq!(
            sample["latest_output_tensor_bytes"],
            640 * 360 * 3 * 4 * std::mem::size_of::<f32>()
        );
        assert_eq!(sample["source"], "synthetic-pan");
        assert_eq!(sample["mode"], "tiled");
        assert_eq!(sample["visualization_mode"], "interframe");
        assert_eq!(sample["mask_visualization_mode"], "image-mask-only");
        assert_eq!(sample["mask_geometry_mode"], "deduplicated");
        assert_eq!(sample["display_transfer"], "gpu");
        assert_eq!(sample["streaming_cache"], true);
        assert_eq!(sample["streaming_cache_effective"], false);
        assert_eq!(sample["configured_max_in_flight"], 3);
        assert_eq!(sample["effective_max_in_flight"], 3);
        assert_eq!(sample["frames_per_clip"], 16);
        assert_eq!(sample["top_k"], 5);
        assert_eq!(sample["max_gaze_tokens_each_frame"], 7);
        assert_eq!(sample["tile_batch_size"], 9);
        assert_eq!(sample["inference_width"], 1280);
        assert_eq!(sample["inference_height"], 720);
        assert_eq!(sample["tensor_sparse_update_max_rects"], 8);
        assert_eq!(sample["tensor_sparse_update_max_ratio"], 0.05);
        assert_eq!(sample["tensor_full_frame_update_min_ratio"], 0.45);
        assert_eq!(sample["show_psnr"], true);
        assert_eq!(sample["warmup_model"], true);
        assert_eq!(sample["burn_backend"], "webgpu");
        assert_eq!(sample["render_adapter_name"], "test adapter");
        assert_eq!(sample["render_adapter_vendor"], 1234);
        assert_eq!(sample["render_adapter_device_type"], "DiscreteGpu");
        assert_eq!(sample["render_adapter_backend"], "Vulkan");
        assert_eq!(sample["render_adapter_driver"], "test-driver");
        assert_eq!(sample["render_adapter_driver_info"], "test-driver-info");
        assert_eq!(sample["avg_output_fps"], 50.0);
        assert_eq!(sample["avg_input_fps"], 100.0);
        assert_eq!(sample["avg_psnr_ms"], 0.75);
        assert_eq!(sample["avg_model_frame_fps"], 250.0);
        assert_eq!(sample["avg_display_input_ms"], 0.25);
        assert_eq!(sample["avg_output_rgba_bytes"], 0.0);
        assert_eq!(
            sample["avg_output_tensor_bytes"],
            (640 * 360 * 3 * 4 * std::mem::size_of::<f32>()) as f64
        );
        assert_eq!(sample["psnr_samples"], 1);
        assert_eq!(sample["latest_psnr_db"], 42.0);
        assert_eq!(sample["latest_psnr_db_infinite"], false);
        assert_eq!(sample["ema_psnr_db"], 42.0);
        assert_eq!(sample["ema_psnr_db_infinite"], false);
        assert_eq!(sample["latest_display_ms"], 1.0);
    }

    #[test]
    fn inference_timing_ignores_keyframe_update_ratio_samples() {
        let mut stats = InferenceTimingStats::default();
        stats.record(
            InferenceTiming {
                total_ms: 10.0,
                gaze_update_ratio: 1.0,
                gaze_update_ratio_sample: None,
                ..Default::default()
            },
            false,
        );
        stats.record(
            InferenceTiming {
                total_ms: 10.0,
                gaze_update_ratio: 0.25,
                gaze_update_ratio_sample: Some(0.25),
                ..Default::default()
            },
            false,
        );

        let summary: serde_json::Value =
            serde_json::from_str(&stats.summary_json(2)).expect("summary json");
        assert_eq!(summary["avg_gaze_update_ratio"], 0.25);
        assert_eq!(summary["latest_gaze_update_ratio"], 0.25);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn perf_summary_writer_creates_pretty_json_artifact() {
        let path = std::env::temp_dir()
            .join(format!(
                "burn_autogaze_perf_summary_test_{}",
                std::process::id()
            ))
            .join("summary.json");
        let _ = std::fs::remove_file(&path);

        write_perf_summary(Some(&path), r#"{"b":1,"a":2}"#).expect("write summary");
        let content = std::fs::read_to_string(&path).expect("read summary");

        assert!(content.ends_with('\n'));
        assert!(content.contains("\"a\": 2"));
        assert!(content.contains("\"b\": 1"));
        let _ = std::fs::remove_file(&path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn perf_trace_writer_replaces_then_appends_jsonl_samples() {
        let path = std::env::temp_dir()
            .join(format!(
                "burn_autogaze_perf_trace_test_{}",
                std::process::id()
            ))
            .join("trace.jsonl");
        let _ = std::fs::remove_file(&path);

        write_perf_trace_sample(Some(&path), r#"{"frame":1}"#, true).expect("write trace");
        write_perf_trace_sample(Some(&path), r#"{"frame":2}"#, false).expect("append trace");
        let content = std::fs::read_to_string(&path).expect("read trace");

        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("\"frame\":1"));
        assert!(content.contains("\"frame\":2"));
        let _ = std::fs::remove_file(&path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
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
    fn mask_source_toggle_clears_stale_mask_state_and_invalidates_pending_results() {
        let mut config = BevyBurnAutoGazeConfig {
            sparse_mask_source: BevySparseMaskSource::AutoGaze,
            ..Default::default()
        };
        let mut model_config = config.clone();
        let mut slider = TaskLossSliderState::new(&config);
        let mut latest_mask = LatestMaskPrediction {
            points: vec![FixationPoint::with_extent(0.5, 0.5, 0.5, 0.5, 1.0)],
        };
        let mut visualization_state =
            BevyVisualizationState::new(config.visualization_mode, config.keyframe_duration);
        let mut streaming_state = BevyStreamingGenerationState::default();
        streaming_state.configure(true, 4, 4, 2);
        let mut sequencer = InferenceSequencer::default();
        let stale_sequence = sequencer.reserve();
        let mut frame_queue = FrameQueue::default();
        frame_queue.push(Arc::new(RgbaImage::new(2, 2)), config.frames_per_clip);
        let mut gaze_ratio_stats = GazeRatioStats::default();
        gaze_ratio_stats.record(0.5);
        let mut psnr_stats = PsnrStats::default();
        psnr_stats.record(30.0);
        let mut timing_stats = InferenceTimingStats {
            latest: Some(InferenceTiming::default()),
            samples: vec![1.0],
            ..Default::default()
        };

        apply_mask_source_toggle(
            &mut config,
            &mut model_config,
            &mut slider,
            &mut latest_mask,
            &mut visualization_state,
            &mut streaming_state,
            &mut sequencer,
            &mut frame_queue,
            &mut gaze_ratio_stats,
            &mut psnr_stats,
            &mut timing_stats,
        );

        assert_eq!(config.sparse_mask_source, BevySparseMaskSource::PatchDiff);
        assert_eq!(
            model_config.sparse_mask_source,
            BevySparseMaskSource::PatchDiff
        );
        assert!(
            latest_mask.points().is_empty(),
            "mode switches must not keep drawing the previous AutoGaze mask"
        );
        assert!(
            !sequencer.accept(stale_sequence),
            "in-flight AutoGaze completions from the old mode must be rejected"
        );
        assert_eq!(
            frame_queue.len(),
            0,
            "mode switches must rebuild the clip window for the target pipeline"
        );
        assert!(!gaze_ratio_stats.0.is_initialized());
        assert!(!psnr_stats.0.is_initialized());
        assert_eq!(timing_stats.processed_frames(), 0);
        assert!(timing_stats.latest.is_none());
        let fresh_sequence = sequencer.reserve();
        assert!(sequencer.accept(fresh_sequence));
    }

    #[test]
    fn async_preview_owns_default_model_completion_display() {
        let policy = AutoGazeRealtimePolicy::default();

        assert_eq!(
            completed_model_display_action(policy, true, 1),
            CompletedModelDisplayAction::UpdateMaskOnly
        );
        assert_eq!(
            completed_run_display_action(policy, false, 1),
            CompletedModelDisplayAction::UpdateMaskOnly
        );
        assert_eq!(
            completed_run_display_action(policy, true, 1),
            CompletedModelDisplayAction::DisplayVisualization
        );
    }

    #[test]
    fn mask_only_model_completion_does_not_rewind_preview_texture() {
        let width = 2;
        let height = 2;
        let latest_rgba = deterministic_test_rgba(width, height, 71);
        let old_rgba = deterministic_test_rgba(width, height, 17);
        let mut images = Assets::<Image>::default();
        let texture = AutoGazeTexture {
            image: images.add(visualization_image(1, 1, vec![0; 4])),
            input_image: images.add(visualization_image(
                width as u32,
                height as u32,
                latest_rgba.clone(),
            )),
            mask_image: images.add(visualization_image(
                width as u32,
                height as u32,
                vec![0; latest_rgba.len()],
            )),
            output_image: images.add(visualization_image(
                width as u32,
                height as u32,
                latest_rgba.clone(),
            )),
            ..AutoGazeTexture::default()
        };
        let input_handle = texture.input_image.clone();
        let mut world = World::new();
        world.insert_resource(images);
        world.insert_resource(texture);
        world.insert_resource(BevyVisualizationState::new(
            AutoGazeVisualizationMode::Interframe,
            0,
        ));

        let display_ms = apply_completed_model_visualization(
            &mut world,
            (width * 3) as u32,
            height as u32,
            panel_visualization_payload(width, height, old_rgba.clone()),
            BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0),
            CompletedModelDisplayAction::UpdateMaskOnly,
        );

        assert_eq!(display_ms, 0.0);
        assert_eq!(
            panel_image_data(&world, &input_handle),
            latest_rgba,
            "mask-only completions must not redraw older model input frames"
        );

        apply_completed_model_visualization(
            &mut world,
            (width * 3) as u32,
            height as u32,
            panel_visualization_payload(width, height, old_rgba.clone()),
            BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0),
            CompletedModelDisplayAction::DisplayVisualization,
        );

        assert_eq!(panel_image_data(&world, &input_handle), old_rgba);
    }

    #[test]
    fn viewer_pipeline_options_delegate_to_core_runtime_options() {
        let mut config = BevyBurnAutoGazeConfig {
            max_gaze_tokens_each_frame: 12,
            task_loss_requirement: Some(0.7),
            tile_batch_size: 9,
            ..Default::default()
        };

        let options = pipeline_options_from_config(&config);
        assert_eq!(options.max_gaze_tokens_each_frame(), Some(12));
        assert_eq!(options.decode_strategy(), config.decode_strategy);
        assert_eq!(
            options.task_loss_requirement(),
            burn_autogaze::AutoGazeTaskLossOption::Value(0.7)
        );
        assert_eq!(options.tile_batch_size(), Some(9));
        assert_eq!(
            options.generation_coverage_stop_ratio(),
            Some(config.tensor_full_frame_update_min_ratio)
        );

        config.disable_task_loss_requirement = true;
        let options = pipeline_options_from_config(&config);
        assert_eq!(
            options.task_loss_requirement(),
            burn_autogaze::AutoGazeTaskLossOption::Disabled
        );
    }

    #[test]
    fn streaming_model_warmup_reaches_cache_compaction_horizon() {
        let config = BevyBurnAutoGazeConfig {
            mode: BevyAutoGazeMode::Resize224,
            frames_per_clip: 16,
            ..Default::default()
        };

        assert_eq!(
            model_warmup_runs(&config, false),
            DEFAULT_REALTIME_MODEL_WARMUP_RUNS
        );
        assert_eq!(
            model_warmup_runs(&config, true),
            16 + DEFAULT_STREAMING_MODEL_WARMUP_EXTRA_RUNS
        );
        assert!(
            model_warmup_runs(&config, true) > config.frames_per_clip,
            "streaming warmup should run beyond the cache horizon so steady-state compaction is prewarmed"
        );
    }

    #[test]
    fn local_motion_warmup_spans_strong_subtle_and_still_phases() {
        let runs = 24;
        let seeds = (0..runs)
            .map(|run_idx| {
                warmup_frame_seed(BevyFrameSource::SyntheticLocalMotion, run_idx, 0, 1, runs) as u64
            })
            .collect::<Vec<_>>();

        assert!(
            seeds
                .iter()
                .any(|&seed| seed < SYNTHETIC_LOCAL_STRONG_FRAMES)
        );
        assert!(seeds.iter().any(|&seed| {
            (SYNTHETIC_LOCAL_STRONG_FRAMES
                ..SYNTHETIC_LOCAL_STRONG_FRAMES + SYNTHETIC_LOCAL_SUBTLE_FRAMES)
                .contains(&seed)
        }));
        assert!(seeds.iter().any(|&seed| {
            (SYNTHETIC_LOCAL_STRONG_FRAMES + SYNTHETIC_LOCAL_SUBTLE_FRAMES
                ..SYNTHETIC_LOCAL_CYCLE_FRAMES)
                .contains(&seed)
        }));
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

        assert_eq!(clip.width(), 2);
        assert_eq!(clip.height(), 1);
        assert_eq!(clip.clip_len(), 2);
        assert_eq!(&clip.rgba()[..first.as_raw().len()], first.as_raw());
        assert_eq!(&clip.rgba()[first.as_raw().len()..], second.as_raw());
        assert_eq!(clip.last_frame_rgba().unwrap(), second.as_raw());

        let capacity = clip.rgba_capacity();
        queue.recycle_clip_buffer(clip.into_rgba());
        assert_eq!(queue.spare_clip_buffer_count(), 1);

        queue.push(Arc::clone(&third), 2);
        assert_eq!(Arc::strong_count(&first), 1);
        let clip = queue.build_clip(2).unwrap().unwrap();
        assert_eq!(queue.spare_clip_buffer_count(), 0);
        assert_eq!(clip.rgba_capacity(), capacity);
        assert_eq!(&clip.rgba()[..second.as_raw().len()], second.as_raw());
        assert_eq!(&clip.rgba()[second.as_raw().len()..], third.as_raw());
    }

    #[test]
    fn patch_diff_preview_queue_keeps_two_frame_window() {
        let config = BevyBurnAutoGazeConfig {
            sparse_mask_source: BevySparseMaskSource::PatchDiff,
            frames_per_clip: 16,
            ..Default::default()
        };
        assert_eq!(frame_queue_len_for_config(&config), 2);

        let config = BevyBurnAutoGazeConfig {
            sparse_mask_source: BevySparseMaskSource::AutoGaze,
            frames_per_clip: 16,
            ..Default::default()
        };
        assert_eq!(frame_queue_len_for_config(&config), 16);
    }

    #[test]
    fn inference_dimensions_preserve_aspect_when_one_axis_is_configured() {
        assert_eq!(
            resize_dimensions_preserving_aspect(1280, 720, Some(1920), None),
            (1920, 1080)
        );
        assert_eq!(
            resize_dimensions_preserving_aspect(1280, 720, None, Some(1080)),
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
    fn invalid_static_image_path_falls_back_without_panic() {
        let path = std::env::temp_dir().join(format!(
            "burn_autogaze_missing_static_image_{}.png",
            std::process::id()
        ));
        let static_frame = load_static_frame(Some(&path), &BevyBurnAutoGazeConfig::default());

        assert!(static_frame.0.is_none());
    }

    #[test]
    fn synthetic_pan_source_generates_deterministic_motion_frames() {
        let config = BevyBurnAutoGazeConfig {
            source: BevyFrameSource::SyntheticPan,
            inference_width: Some(64),
            inference_height: Some(36),
            ..Default::default()
        };
        let mut source_a = SyntheticFrameSource::default();
        let mut source_b = SyntheticFrameSource::default();

        let first_a = source_a.next_frame(&config);
        let second_a = source_a.next_frame(&config);
        let first_b = source_b.next_frame(&config);

        assert_eq!(first_a.dimensions(), (64, 36));
        assert_eq!(first_a.as_raw(), first_b.as_raw());
        assert_ne!(first_a.as_raw(), second_a.as_raw());
    }

    #[test]
    fn synthetic_local_motion_has_motion_then_settles() {
        let strong_a = synthetic_local_motion_frame(96, 54, 4);
        let strong_b = synthetic_local_motion_frame(96, 54, 12);
        let subtle_a = synthetic_local_motion_frame(96, 54, 48);
        let subtle_b = synthetic_local_motion_frame(96, 54, 56);
        let still_a = synthetic_local_motion_frame(96, 54, 88);
        let still_b = synthetic_local_motion_frame(96, 54, 104);

        assert_eq!(strong_a.dimensions(), (96, 54));
        assert!(rgba_sum_abs_diff(strong_a.as_raw(), strong_b.as_raw()) > 0);
        assert!(rgba_sum_abs_diff(subtle_a.as_raw(), subtle_b.as_raw()) > 0);
        assert_eq!(still_a.as_raw(), still_b.as_raw());
        assert!(
            rgba_sum_abs_diff(strong_a.as_raw(), strong_b.as_raw())
                > rgba_sum_abs_diff(subtle_a.as_raw(), subtle_b.as_raw())
        );
    }

    fn rgba_sum_abs_diff(left: &[u8], right: &[u8]) -> u64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left.abs_diff(*right) as u64)
            .sum()
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

    #[test]
    fn bevy_interframe_output_matches_visible_multiscale_mask_cells() {
        let width = 126;
        let height = 70;
        let previous = vec![10u8; width * height * 4];
        let mut current = vec![200u8; width * height * 4];
        for pixel in current.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        let points = [
            FixationPoint::with_grid_extent(0.5 / 18.0, 0.5 / 10.0, 1.0 / 18.0, 1.0 / 10.0, 1.0, 2),
            FixationPoint::with_grid_extent(
                125.5 / 126.0,
                69.5 / 70.0,
                1.0 / 126.0,
                1.0 / 70.0,
                1.0,
                14,
            ),
        ];
        let options =
            VisualizationOptions::new(1.0, DEFAULT_BLEND_ALPHA, false, BevyDisplayTransfer::Cpu)
                .with_cpu_panels();
        let device = AutoGazeBevyDevice::default();
        let mut state = BevyVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);

        visualize_frame_rgba(
            FrameVisualInput {
                rgba: &previous,
                width,
                height,
                tensor: None,
            },
            &[],
            options,
            &mut state,
            &device,
        )
        .expect("initial keyframe");
        let visualization = visualize_frame_rgba(
            FrameVisualInput {
                rgba: &current,
                width,
                height,
                tensor: None,
            },
            &points,
            options,
            &mut state,
            &device,
        )
        .expect("interframe update");

        match visualization.image_data {
            VisualizationImageData::PanelsRgba {
                mask_rgba,
                output_rgba,
                ..
            } => {
                let colored_mask_pixels = mask_rgba
                    .chunks_exact(4)
                    .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                    .count();
                let updated_output_pixels = output_rgba
                    .chunks_exact(4)
                    .filter(|pixel| pixel[0] == 200)
                    .count();

                assert!(
                    colored_mask_pixels > 40,
                    "native mask should still show the coarse AnyRes cell"
                );
                assert_eq!(
                    updated_output_pixels, colored_mask_pixels,
                    "interframe output should update exactly the visible multi-scale mask cells"
                );
                assert_eq!(
                    visualization.gaze_update_ratio,
                    colored_mask_pixels as f64 / (width * height) as f64
                );
            }
            _ => panic!("expected panel visualization payload"),
        }
    }
}
