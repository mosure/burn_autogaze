#![recursion_limit = "256"]

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    path::Path,
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
        renderer::RenderAdapterInfo,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::widget::ImageNode,
    window::PrimaryWindow,
};
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind};
use burn::tensor::Tensor;
#[cfg(target_arch = "wasm32")]
use burn_autogaze::{AutoGazeConfig, AutoGazeLoadOptions, NativeAutoGazeModel};
use burn_autogaze::{
    AutoGazeGazeRatioStats, AutoGazeInferenceMode, AutoGazeInferenceSequencer,
    AutoGazeMaskVisualizationMode, AutoGazePipeline, AutoGazePipelineOptions,
    AutoGazePreparedRun as CoreAutoGazePreparedRun, AutoGazePsnrStats, AutoGazeReadoutRunOutput,
    AutoGazeRgbaClipShape, AutoGazeRgbaFrameClip, AutoGazeRgbaFrameQueue,
    AutoGazeRgbaVisualizationOptions, AutoGazeStreamingCache, AutoGazeTensorInterframePath,
    AutoGazeTensorVisualizationOptions, AutoGazeTensorVisualizationState,
    AutoGazeVisualizationMode, AutoGazeVisualizationState, FixationPoint, format_fps,
    format_gaze_ratio_percent, format_psnr_db, fps_from_millis, prepare_rgba_clip_for_trace,
    resize_rgba_frame_to_dimensions, rgba_clip_to_tensor, should_use_streaming_cache,
    video_frame_tensor,
};
pub use burn_autogaze::{
    DEFAULT_BLEND_ALPHA, DEFAULT_KEYFRAME_DURATION, DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MODEL_GENERATION_BUDGET, DEFAULT_REALTIME_FRAMES_PER_CLIP, DEFAULT_REALTIME_TOP_K,
    DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO, DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
    DEFAULT_TILED_FRAMES_PER_CLIP, DEFAULT_TILED_MAX_GAZE_TOKENS, DEFAULT_TILED_TILE_BATCH_SIZE,
    DEFAULT_TILED_TOP_K,
};
#[cfg(test)]
use burn_autogaze::{
    fixation_alpha_mask, fixation_scale_mask_rgba, rgba_clip_to_inference_tensor,
    rgba_clip_to_processor_tensor,
};
use image::RgbaImage;

mod config;
pub mod platform;
#[cfg(test)]
use config::MODEL_INPUT_SIZE;
pub use config::{
    BevyAutoGazeMode, BevyBurnAutoGazeConfig, BevyDisplayTransfer, DEFAULT_BEVY_MODE,
    DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP, DEFAULT_BEVY_STREAMING_CACHE, DEFAULT_BEVY_TILED_TOP_K,
    DEFAULT_BIRDS_BLEND_ALPHA, DEFAULT_BIRDS_FRAMES_PER_CLIP, DEFAULT_BIRDS_INFERENCE_HEIGHT,
    DEFAULT_BIRDS_INFERENCE_WIDTH, DEFAULT_BIRDS_KEYFRAME_DURATION, DEFAULT_BIRDS_MAX_GAZE_TOKENS,
    DEFAULT_BIRDS_TILE_BATCH_SIZE, DEFAULT_BIRDS_TOP_K, DEFAULT_CONFIG_URL,
    DEFAULT_NATIVE_MODEL_DIR, DEFAULT_REALTIME_INFERENCE_WIDTH, DEFAULT_REALTIME_MAX_GAZE_TOKENS,
    DEFAULT_TILED_INFERENCE_WIDTH, DEFAULT_WEIGHTS_URL, ImplicitModeDefaults,
    default_frames_per_clip, default_inference_dimensions, default_max_gaze_tokens_each_frame,
    default_tile_batch_size, default_top_k, realtime_policy_from_config,
};

pub type AutoGazeBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type AutoGazeBevyDevice = burn::backend::wgpu::WgpuDevice;
const AUTO_GAZE_BEVY_BACKEND_NAME: &str = "webgpu";
const MAX_SPARE_CLIP_BUFFERS: usize = 2;
const TIMING_LOG_INTERVAL_MS: f64 = 5_000.0;
const UI_MARGIN_PX: f32 = 12.0;
const METRIC_ROW_HEIGHT: f32 = 34.0;
const PANEL_LABEL_ROW_HEIGHT: f32 = 38.0;
const INFERENCE_FPS: DiagnosticPath = DiagnosticPath::const_new("autogaze_inference_fps");

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum AutoGazeTextureLayout {
    #[default]
    SideBySide,
    Panels,
}

#[derive(Resource, Clone)]
struct AutoGazeTexture {
    image: Handle<Image>,
    input_image: Handle<Image>,
    mask_image: Handle<Image>,
    output_image: Handle<Image>,
    entity: Option<Entity>,
    side_by_side_entity: Option<Entity>,
    input_entity: Option<Entity>,
    mask_entity: Option<Entity>,
    output_entity: Option<Entity>,
    width: u32,
    height: u32,
    layout: AutoGazeTextureLayout,
}

impl Default for AutoGazeTexture {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            input_image: Handle::default(),
            mask_image: Handle::default(),
            output_image: Handle::default(),
            entity: None,
            side_by_side_entity: None,
            input_entity: None,
            mask_entity: None,
            output_entity: None,
            width: 3,
            height: 1,
            layout: AutoGazeTextureLayout::default(),
        }
    }
}

#[derive(Resource)]
struct FrameQueue {
    inner: AutoGazeRgbaFrameQueue,
}

impl Default for FrameQueue {
    fn default() -> Self {
        Self {
            inner: AutoGazeRgbaFrameQueue::new(MAX_SPARE_CLIP_BUFFERS),
        }
    }
}

impl FrameQueue {
    fn push(&mut self, frame: Arc<RgbaImage>, max_len: usize) {
        self.inner.push(frame, max_len);
    }

    fn latest(&self) -> Option<&RgbaImage> {
        self.inner.latest()
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

#[derive(Resource, Default, Clone, Debug)]
struct InferenceSequencer(AutoGazeInferenceSequencer);

impl InferenceSequencer {
    fn reserve(&mut self) -> u64 {
        self.0.reserve()
    }

    fn accept(&mut self, sequence: u64) -> bool {
        self.0.accept(sequence)
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

#[derive(Resource, Clone, Debug, Default)]
struct GazeRatioStats(AutoGazeGazeRatioStats);

impl GazeRatioStats {
    fn record(&mut self, ratio: f64) {
        self.0.record(ratio);
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct PsnrStats(AutoGazePsnrStats);

impl PsnrStats {
    fn record(&mut self, psnr_db: f64) {
        self.0.record(psnr_db);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InferenceTiming {
    sequence: u64,
    clip_frames: usize,
    model_frames: usize,
    trace_points: usize,
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
    tensor_ms: f64,
    visualize_ms: f64,
    display_ms: f64,
    total_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    display_input_residency: DisplayInputResidency,
    gaze_update_ratio: f64,
    gaze_update_ratio_sample: Option<f64>,
    psnr_db: Option<f64>,
    tensor_interframe_path: Option<AutoGazeTensorInterframePath>,
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
    trace_points: usize,
    input_ms: f64,
    display_input_ms: f64,
    pack_ms: f64,
    visualize_ms: f64,
    visualize_cpu_ms: f64,
    tensor_ms: f64,
    display_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    gaze_update_ratio: f64,
    gaze_update_samples: usize,
    latest_gaze_update_ratio: Option<f64>,
    psnr_stats: AutoGazePsnrStats,
    psnr_samples: usize,
    samples: Vec<f64>,
    model_samples: Vec<f64>,
    emitted_summary: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InferenceRunConfigSummary {
    mode: &'static str,
    visualization_mode: &'static str,
    mask_visualization_mode: &'static str,
    display_transfer: &'static str,
    streaming_cache: bool,
    streaming_cache_effective: bool,
    configured_max_in_flight: usize,
    effective_max_in_flight: usize,
    frames_per_clip: usize,
    top_k: usize,
    max_gaze_tokens_each_frame: usize,
    tile_batch_size: usize,
    inference_width: Option<u32>,
    inference_height: Option<u32>,
    tensor_sparse_update_max_rects: usize,
    tensor_sparse_update_max_ratio: f64,
    show_psnr: bool,
    burn_backend: &'static str,
}

impl From<&BevyBurnAutoGazeConfig> for InferenceRunConfigSummary {
    fn from(config: &BevyBurnAutoGazeConfig) -> Self {
        Self {
            mode: config.mode.as_str(),
            visualization_mode: config.visualization_mode.as_str(),
            mask_visualization_mode: config.mask_visualization_mode.as_str(),
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
            tile_batch_size: config.tile_batch_size,
            inference_width: config.inference_width,
            inference_height: config.inference_height,
            tensor_sparse_update_max_rects: config.tensor_sparse_update_max_rects,
            tensor_sparse_update_max_ratio: config.tensor_sparse_update_max_ratio,
            show_psnr: config.show_psnr,
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
        self.trace_points += timing.trace_points;
        self.input_ms += timing.input_ms;
        self.display_input_ms += timing.display_input_ms;
        self.pack_ms += timing.pack_ms;
        self.visualize_ms += timing.visualize_ms;
        self.visualize_cpu_ms += timing.visualize_cpu_ms;
        self.tensor_ms += timing.tensor_ms;
        self.display_ms += timing.display_ms;
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
        if let Some(psnr_db) = timing.psnr_db {
            self.psnr_stats.record(psnr_db);
            self.psnr_samples = self.psnr_samples.saturating_add(1);
        }
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
        log(&format!(
            "AutoGaze timing: {:.1} output fps / {:.1} model-frame fps ({:.1} ms) clip={} model_frames={} points={} gaze={:.2}% {}x{}, source={:.1} ms, prepare={:.1} ms, pack={:.1} ms, input={:.1} ms, display_input={:.1} ms ({}) model={:.1} ms, trace={:.1} ms, sync={:.1} ms, visualize_cpu={:.1} ms, tensor={:.1} ms, tensor_path={}, visualize={:.1} ms, display={:.1} ms, output={:.1} MiB rgba/{:.1} MiB f32",
            timing.e2e_fps(),
            timing.model_frame_fps(),
            timing.total_ms,
            timing.clip_frames,
            timing.model_frames,
            timing.trace_points,
            timing.gaze_update_ratio * 100.0,
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
            timing.tensor_ms,
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
        let avg_tensor_ms = mean_or_zero(self.tensor_ms, processed_frames);
        let avg_display_ms = mean_or_zero(self.display_ms, processed_frames);
        let avg_output_rgba_bytes = mean_or_zero(self.output_rgba_bytes as f64, processed_frames);
        let avg_output_tensor_bytes =
            mean_or_zero(self.output_tensor_bytes as f64, processed_frames);
        let avg_gaze_update_ratio = mean_or_zero(self.gaze_update_ratio, self.gaze_update_samples);
        let avg_input_fps = fps_from_millis(avg_total_ms)
            * mean_or_zero(self.model_frames as f64, processed_frames);
        let avg_model_frame_fps = if self.model_ms > 0.0 {
            self.model_frames as f64 * 1_000.0 / self.model_ms
        } else {
            0.0
        };
        let avg_trace_points = mean_or_zero(self.trace_points as f64, processed_frames);
        let p50_total_ms = percentile_ms(&self.samples, 0.50);
        let p95_total_ms = percentile_ms(&self.samples, 0.95);
        let p50_model_ms = percentile_ms(&self.model_samples, 0.50);
        let p95_model_ms = percentile_ms(&self.model_samples, 0.95);
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
            "processed_frames": processed_frames,
            "processed_model_frames": self.model_frames,
            "avg_output_fps": fps_from_millis(avg_total_ms),
            "avg_model_frame_fps": avg_model_frame_fps,
            "avg_input_fps": avg_input_fps,
            "avg_total_ms": avg_total_ms,
            "p50_total_ms": p50_total_ms,
            "p95_total_ms": p95_total_ms,
            "avg_model_ms": avg_model_ms,
            "p50_model_ms": p50_model_ms,
            "p95_model_ms": p95_model_ms,
            "avg_trace_points": avg_trace_points,
            "avg_input_ms": avg_input_ms,
            "avg_display_input_ms": avg_display_input_ms,
            "avg_pack_ms": avg_pack_ms,
            "avg_visualize_ms": avg_visualize_ms,
            "avg_visualize_cpu_ms": avg_visualize_cpu_ms,
            "avg_tensor_ms": avg_tensor_ms,
            "avg_display_ms": avg_display_ms,
            "avg_output_rgba_bytes": avg_output_rgba_bytes,
            "avg_output_tensor_bytes": avg_output_tensor_bytes,
            "avg_gaze_update_ratio": avg_gaze_update_ratio,
            "psnr_samples": self.psnr_samples,
            "latest_psnr_db": psnr_metric_json_value(&self.psnr_stats, PsnrMetricKind::Current),
            "latest_psnr_db_infinite": psnr_metric_is_infinite(&self.psnr_stats, PsnrMetricKind::Current),
            "ema_psnr_db": psnr_metric_json_value(&self.psnr_stats, PsnrMetricKind::Ema),
            "ema_psnr_db_infinite": psnr_metric_is_infinite(&self.psnr_stats, PsnrMetricKind::Ema),
            "latest_output_rgba_bytes": self.latest.map(|timing| timing.output_rgba_bytes).unwrap_or_default(),
            "latest_output_tensor_bytes": self.latest.map(|timing| timing.output_tensor_bytes).unwrap_or_default(),
            "display_residency": self.latest.map(display_residency).unwrap_or("none"),
            "display_input_residency": self.latest.map(|timing| timing.display_input_residency.as_str()).unwrap_or("none"),
            "latest_display_input_ms": self.latest.map(|timing| timing.display_input_ms).unwrap_or_default(),
            "latest_clip_frames": latest_clip_frames,
            "latest_model_frames": latest_model_frames,
            "latest_trace_points": self.latest.map(|timing| timing.trace_points).unwrap_or_default(),
            "latest_gaze_update_ratio": self.latest_gaze_update_ratio.unwrap_or_default(),
            "latest_tensor_interframe_path": self.latest.and_then(|timing| timing.tensor_interframe_path).map(|path| path.as_str()),
            "latest_sequence": self.latest.map(|timing| timing.sequence).unwrap_or_default(),
            "latest_width": self.latest.map(|timing| timing.width).unwrap_or_default(),
            "latest_height": self.latest.map(|timing| timing.height).unwrap_or_default(),
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

fn insert_run_config_json_fields(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    run_config: Option<InferenceRunConfigSummary>,
) {
    let Some(config) = run_config else {
        return;
    };
    fields.insert("mode".to_string(), config.mode.into());
    fields.insert(
        "visualization_mode".to_string(),
        config.visualization_mode.into(),
    );
    fields.insert(
        "mask_visualization_mode".to_string(),
        config.mask_visualization_mode.into(),
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
    fields.insert("show_psnr".to_string(), config.show_psnr.into());
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

#[cfg(any(target_arch = "wasm32", test))]
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
    let mut sample = serde_json::json!({
        "processed_frames": stats.processed_frames(),
        "processed_model_frames": stats.model_frames,
        "latest_sequence": latest.sequence,
        "latest_clip_frames": latest.clip_frames,
        "latest_model_frames": latest.model_frames,
        "latest_total_ms": latest.total_ms,
        "latest_model_ms": latest.model_ms,
        "latest_display_ms": latest.display_ms,
        "latest_display_input_ms": latest.display_input_ms,
        "latest_trace_points": latest.trace_points,
        "latest_gaze_update_ratio": stats.latest_gaze_update_ratio.unwrap_or_default(),
        "latest_tensor_interframe_path": latest.tensor_interframe_path.map(|path| path.as_str()),
        "latest_output_rgba_bytes": latest.output_rgba_bytes,
        "latest_output_tensor_bytes": latest.output_tensor_bytes,
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
        "avg_display_input_ms": mean_or_zero(stats.display_input_ms, stats.processed_frames()),
        "avg_output_rgba_bytes": mean_or_zero(stats.output_rgba_bytes as f64, stats.processed_frames()),
        "avg_output_tensor_bytes": mean_or_zero(stats.output_tensor_bytes as f64, stats.processed_frames()),
        "avg_gaze_update_ratio": mean_or_zero(stats.gaze_update_ratio, stats.gaze_update_samples),
        "psnr_samples": stats.psnr_samples,
        "latest_psnr_db": psnr_metric_json_value(&stats.psnr_stats, PsnrMetricKind::Current),
        "latest_psnr_db_infinite": psnr_metric_is_infinite(&stats.psnr_stats, PsnrMetricKind::Current),
        "ema_psnr_db": psnr_metric_json_value(&stats.psnr_stats, PsnrMetricKind::Ema),
        "ema_psnr_db_infinite": psnr_metric_is_infinite(&stats.psnr_stats, PsnrMetricKind::Ema),
        "p95_total_ms": percentile_ms(&stats.samples, 0.95),
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
    frame_queue: ResMut<'w, FrameQueue>,
    inference_sequencer: ResMut<'w, InferenceSequencer>,
    visualization_state: ResMut<'w, BevyVisualizationState>,
    streaming_state: ResMut<'w, BevyStreamingGenerationState>,
    gaze_ratio_stats: ResMut<'w, GazeRatioStats>,
    psnr_stats: ResMut<'w, PsnrStats>,
    timing_stats: Res<'w, InferenceTimingStats>,
}

#[derive(Component)]
struct ProcessAutoGaze(Task<CommandQueue>);

#[derive(Component)]
struct OneShotGpuUpload;

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
    app.add_systems(First, clear_completed_gpu_uploads);

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
            enforce_required_hardware_adapter,
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
    let realtime_policy = realtime_policy_from_config(&frame_input.config);
    if !realtime_policy.should_start_inference(active_tasks.iter().count()) {
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
        Some((Arc::clone(frame), 0.0, 0.0))
    } else {
        let frame = receive_frame();
        let source_ms = elapsed_ms(source_start);
        frame.map(|frame| {
            let prepare_start = timestamp_now();
            let frame = prepare_frame_for_inference(frame, &frame_input.config);
            (Arc::new(frame), source_ms, elapsed_ms(prepare_start))
        })
    };

    let Some((frame, source_ms, prepare_ms)) = frame else {
        return;
    };
    frame_input
        .frame_queue
        .push(frame, frame_input.config.frames_per_clip);
    let mode = frame_input.config.mode.inference_mode();
    let use_streaming_cache = should_use_streaming_cache(
        frame_input.config.streaming_cache,
        frame_input.config.frames_per_clip,
        mode,
    );
    let mut clip = match if use_streaming_cache {
        frame_input.frame_queue.build_latest_clip()
    } else {
        frame_input
            .frame_queue
            .build_clip(frame_input.config.frames_per_clip)
    } {
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
    let top_k = frame_input.config.top_k.max(1);
    let log_pipeline_timing = frame_input.config.log_pipeline_timing;
    let context_frames = frame_input.config.frames_per_clip.max(1);
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
    .with_mask_visualization_mode(frame_input.config.mask_visualization_mode)
    .with_cpu_panels();
    let run_config = InferenceRunConfigSummary::from(frame_input.config.as_ref());
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
    frame_input.streaming_state.configure(
        use_streaming_cache,
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
        };

        let result = run_autogaze_visualization(pipeline, job).await;
        let clip_rgba = clip.into_rgba();

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
                Ok((visualization, visualization_state, streaming_state)) => {
                    let Visualization {
                        width,
                        height,
                        image_data,
                        gaze_update_ratio,
                        interframe_keyframe,
                        psnr_db,
                        mut timing,
                        ..
                    } = visualization;
                    let display_start = timestamp_now();
                    apply_visualization_to_world(world, width, height, image_data);
                    if let Some(ref mut timing) = timing {
                        timing.display_ms = elapsed_ms(display_start);
                        timing.total_ms += timing.display_ms;
                    }

                    if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
                        texture.width = width;
                        texture.height = height;
                    }

                    if let Some(mut state) = world.get_resource_mut::<BevyVisualizationState>() {
                        *state = visualization_state;
                    }

                    if let Some(mut state) =
                        world.get_resource_mut::<BevyStreamingGenerationState>()
                    {
                        *state = streaming_state;
                    }

                    if !interframe_keyframe
                        && let Some(mut stats) = world.get_resource_mut::<GazeRatioStats>()
                    {
                        stats.record(gaze_update_ratio);
                    }

                    if let Some(psnr_db) = psnr_db
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
                            stats.record(timing, log_pipeline_timing);
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

fn preview_frames(
    model: Res<AutoGazeModelState>,
    mut texture: ResMut<AutoGazeTexture>,
    mut frame_input: FrameInputParams,
    active_tasks: Query<&ProcessAutoGaze>,
    mut images: ResMut<Assets<Image>>,
) {
    let model_ready = model.pipeline.is_some();
    let realtime_policy = realtime_policy_from_config(&frame_input.config);
    let inference_busy = realtime_policy.inference_busy(active_tasks.iter().count());
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

    if !realtime_policy.should_draw_live_preview(model_ready) {
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
    Ok(pipeline)
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
    frame_index: usize,
    model_frames: usize,
    model_ms: f64,
}

fn finished_readout_from_run_output(
    output: AutoGazeReadoutRunOutput,
    model_ms: f64,
) -> FinishedReadout {
    FinishedReadout {
        points: output.points,
        frame_index: output.frame_index,
        model_frames: output.model_frames,
        model_ms,
    }
}

fn finish_autogaze_visualization(
    context: AutoGazeRunContext<'_>,
    prepared: PreparedVisualizationRun,
    finished: FinishedReadout,
    total_start: Timestamp,
) -> Result<
    (
        Visualization,
        BevyVisualizationState,
        BevyStreamingGenerationState,
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
        frame_index,
        model_frames,
        model_ms,
    } = finished;
    let trace_ms = prepared.input_ms + model_ms;
    let points = batch_points
        .first()
        .and_then(|frames| frames.get(frame_index))
        .cloned()
        .unwrap_or_default();
    let visualize_start = timestamp_now();
    let mut visualization = visualize_frame_rgba(
        FrameVisualInput {
            rgba: clip.last_frame_rgba()?,
            width,
            height,
            tensor: prepared.visualization_tensor,
        },
        &points,
        visualization_options,
        &mut visualization_state,
        &device,
    )?;
    visualization.timing = Some(InferenceTiming {
        sequence,
        clip_frames: context_frames,
        model_frames,
        trace_points: points.len(),
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
        tensor_ms: visualization.tensor_ms,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start) + clip.source_ms + clip.prepare_ms + clip.pack_ms,
        output_rgba_bytes: visualization.output_rgba_bytes,
        output_tensor_bytes: visualization.output_tensor_bytes,
        display_input_residency: prepared.display_input_residency,
        gaze_update_ratio: visualization.gaze_update_ratio,
        gaze_update_ratio_sample: (!visualization.interframe_keyframe)
            .then_some(visualization.gaze_update_ratio),
        psnr_db: visualization.psnr_db,
        tensor_interframe_path: visualization.tensor_interframe_path,
    });
    Ok((visualization, visualization_state, streaming_state))
}

async fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    mut context: AutoGazeRunContext<'_>,
) -> Result<
    (
        Visualization,
        BevyVisualizationState,
        BevyStreamingGenerationState,
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
        streaming_state,
    )
    .await?;
    finish_autogaze_visualization(context, visualization, finished, total_start)
}

async fn run_autogaze_readout(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    trace_input: CoreAutoGazePreparedRun<AutoGazeBevyBackend>,
    top_k: usize,
    use_streaming_cache: bool,
    streaming_state: &mut BevyStreamingGenerationState,
) -> Result<FinishedReadout, String> {
    let pipeline = pipeline
        .lock()
        .map_err(|_| "AutoGaze model lock was poisoned".to_string())?
        .clone();
    let model_start = timestamp_now();
    let run_output = if use_streaming_cache {
        pipeline
            .readout_prepared_run_async(trace_input, top_k, Some(streaming_state.cache_mut()))
            .await
    } else {
        pipeline
            .readout_prepared_run_async(trace_input, top_k, None)
            .await
    }
    .map_err(|err| format!("failed to read AutoGaze tensor data asynchronously: {err:?}"))?;
    Ok(finished_readout_from_run_output(
        run_output,
        elapsed_ms(model_start),
    ))
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
    calculate_psnr: bool,
    display_transfer: BevyDisplayTransfer,
    sparse_update_max_rects: usize,
    sparse_update_max_ratio: f64,
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
            mask_visualization_mode: AutoGazeMaskVisualizationMode::Overlay,
            calculate_psnr,
            display_transfer,
            sparse_update_max_rects: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            sparse_update_max_ratio: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
            cpu_layout: CpuVisualizationLayout::SideBySide,
        }
    }

    fn with_sparse_update_policy(mut self, max_rects: usize, max_update_ratio: f64) -> Self {
        self.sparse_update_max_rects = max_rects;
        self.sparse_update_max_ratio = max_update_ratio;
        self
    }

    fn with_mask_visualization_mode(mut self, mode: AutoGazeMaskVisualizationMode) -> Self {
        self.mask_visualization_mode = mode;
        self
    }

    fn with_cpu_panels(mut self) -> Self {
        self.cpu_layout = CpuVisualizationLayout::Panels;
        self
    }
}

enum VisualizationImageData {
    SideBySideRgba(Vec<u8>),
    PanelsRgba {
        panel_width: u32,
        panel_height: u32,
        input_rgba: Vec<u8>,
        mask_rgba: Vec<u8>,
        output_rgba: Vec<u8>,
    },
    TensorPanels(Box<TensorPanelVisualizationData>),
}

struct TensorPanelVisualizationData {
    panel_width: u32,
    panel_height: u32,
    input_rgba: Tensor<AutoGazeBevyBackend, 3>,
    mask_rgba: Tensor<AutoGazeBevyBackend, 3>,
    output_rgba: Tensor<AutoGazeBevyBackend, 3>,
}

struct Visualization {
    width: u32,
    height: u32,
    #[cfg(test)]
    rgba: Vec<u8>,
    #[cfg(test)]
    tensor: Option<Tensor<AutoGazeBevyBackend, 3>>,
    image_data: VisualizationImageData,
    gaze_update_ratio: f64,
    interframe_keyframe: bool,
    psnr_db: Option<f64>,
    visualize_cpu_ms: f64,
    tensor_ms: f64,
    output_rgba_bytes: usize,
    output_tensor_bytes: usize,
    tensor_interframe_path: Option<AutoGazeTensorInterframePath>,
    timing: Option<InferenceTiming>,
}

struct FrameVisualInput<'a> {
    rgba: &'a [u8],
    width: usize,
    height: usize,
    tensor: Option<Tensor<AutoGazeBevyBackend, 5>>,
}

fn visualize_frame_rgba(
    input: FrameVisualInput<'_>,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    if options.display_transfer == BevyDisplayTransfer::Gpu {
        visualize_rgba_tensor(input, points, options, visualization_state, device)
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

fn display_tensor_from_clip(
    clip: &FrameClip,
    options: VisualizationOptions,
    device: &AutoGazeBevyDevice,
) -> Result<Option<Tensor<AutoGazeBevyBackend, 5>>, String> {
    if options.display_transfer != BevyDisplayTransfer::Gpu {
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
    if options.display_transfer != BevyDisplayTransfer::Gpu {
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

fn visualize_rgba_tensor(
    input: FrameVisualInput<'_>,
    points: &[FixationPoint],
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
    let tensor_panels = visualization_state
        .gpu
        .visualize_normalized_rgb_clip_panels(
            tensor,
            points,
            AutoGazeTensorVisualizationOptions::new(
                width,
                height,
                options.cell_scale,
                options.blend_alpha,
            )
            .with_mask_visualization_mode(options.mask_visualization_mode)
            .with_sparse_update_policy(
                options.sparse_update_max_rects,
                options.sparse_update_max_ratio,
            ),
            device,
        )
        .map_err(|err| format!("failed to visualize AutoGaze tensor output: {err:#}"))?;
    let tensor_ms = elapsed_ms(tensor_start);
    let gaze_update_ratio = tensor_panels.update_ratio();
    let output_tensor_bytes = width * height * 3 * 4 * std::mem::size_of::<f32>();
    let tensor_interframe_path = visualization_state.gpu.last_interframe_path();
    let interframe_keyframe = matches!(
        tensor_interframe_path,
        Some(AutoGazeTensorInterframePath::Keyframe)
    );
    let cpu_psnr_start = timestamp_now();
    let psnr_db = if options.calculate_psnr {
        let rgba_options = AutoGazeRgbaVisualizationOptions::new(
            width,
            height,
            options.cell_scale,
            options.blend_alpha,
        )
        .with_mask_visualization_mode(options.mask_visualization_mode);
        let cpu_panels = visualization_state
            .cpu
            .visualize_rgba_panels_with_options(input.rgba, points, rgba_options)
            .map_err(|err| format!("failed to mirror AutoGaze CPU PSNR output: {err:#}"))?;
        debug_assert_eq!(
            visualization_state.cpu.last_frame_was_keyframe(),
            interframe_keyframe
        );
        if interframe_keyframe {
            None
        } else {
            Some(
                cpu_panels
                    .output_psnr_db(input.rgba)
                    .map_err(|err| format!("{err:#}"))?,
            )
        }
    } else {
        None
    };
    let visualize_cpu_ms = if options.calculate_psnr {
        elapsed_ms(cpu_psnr_start)
    } else {
        0.0
    };

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
        })),
        gaze_update_ratio,
        interframe_keyframe,
        psnr_db,
        visualize_cpu_ms,
        tensor_ms,
        output_rgba_bytes: 0,
        output_tensor_bytes,
        tensor_interframe_path,
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
    .with_mask_visualization_mode(options.mask_visualization_mode);
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
            let gaze_update_ratio = visualization.update_ratio();
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
                interframe_keyframe,
                psnr_db,
                visualize_cpu_ms,
                tensor_ms: 0.0,
                output_rgba_bytes,
                output_tensor_bytes: 0,
                tensor_interframe_path: None,
                timing: None,
            })
        }
        CpuVisualizationLayout::Panels => {
            let panels = visualization_state
                .cpu
                .visualize_rgba_panels_with_options(rgba, points, rgba_options)
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
            let gaze_update_ratio = panels.update_ratio();
            let mask_rgba = panels.mask_rgba;
            let output_rgba = panels.blend_rgba;
            let input_rgba = rgba.to_vec();
            let output_rgba_bytes = input_rgba.len() + mask_rgba.len() + output_rgba.len();
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
                },
                gaze_update_ratio,
                interframe_keyframe,
                psnr_db,
                visualize_cpu_ms,
                tensor_ms: 0.0,
                output_rgba_bytes,
                output_tensor_bytes: 0,
                tensor_interframe_path: None,
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
    let mut visualization = visualize_points(
        rgba,
        &[],
        VisualizationOptions::new(1.0, 0.0, false, BevyDisplayTransfer::Cpu),
        &mut state,
    )?;
    visualization.gaze_update_ratio = 0.0;
    let _ = calculate_psnr;
    visualization.psnr_db = None;
    Ok(visualization)
}

fn apply_visualization_to_texture(
    visualization: Visualization,
    texture: &mut AutoGazeTexture,
    images: &mut Assets<Image>,
) {
    let width = visualization.width;
    let height = visualization.height;
    match visualization.image_data {
        VisualizationImageData::SideBySideRgba(rgba) => {
            set_visualization_image(&texture.image, width, height, rgba, images);
            texture.layout = AutoGazeTextureLayout::SideBySide;
        }
        VisualizationImageData::PanelsRgba {
            panel_width,
            panel_height,
            input_rgba,
            mask_rgba,
            output_rgba,
        } => {
            set_panel_visualization_images(
                texture,
                images,
                PanelVisualizationImages {
                    width: panel_width,
                    height: panel_height,
                    input_rgba,
                    mask_rgba,
                    output_rgba,
                },
            );
            texture.layout = AutoGazeTextureLayout::Panels;
        }
        VisualizationImageData::TensorPanels(_) => {}
    }
    texture.width = width;
    texture.height = height;
}

fn apply_visualization_to_world(
    world: &mut World,
    width: u32,
    height: u32,
    image_data: VisualizationImageData,
) {
    let Some(texture) = world.get_resource::<AutoGazeTexture>().cloned() else {
        return;
    };

    match image_data {
        VisualizationImageData::TensorPanels(panels) => {
            let TensorPanelVisualizationData {
                panel_width,
                panel_height,
                input_rgba,
                mask_rgba,
                output_rgba,
            } = *panels;
            set_texture_layout(world, &texture, AutoGazeTextureLayout::Panels);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_gpu_visualization_image(
                    &texture.input_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
                set_gpu_visualization_image(
                    &texture.mask_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
                set_gpu_visualization_image(
                    &texture.output_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
            }
            set_gpu_panel_upload_handle(
                world,
                texture.input_entity,
                texture.input_image.clone(),
                input_rgba,
            );
            set_gpu_panel_upload_handle(
                world,
                texture.mask_entity,
                texture.mask_image.clone(),
                mask_rgba,
            );
            set_gpu_panel_upload_handle(
                world,
                texture.output_entity,
                texture.output_image.clone(),
                output_rgba,
            );
        }
        VisualizationImageData::SideBySideRgba(rgba) => {
            set_texture_layout(world, &texture, AutoGazeTextureLayout::SideBySide);
            remove_panel_gpu_visualization_handles(world, &texture);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_visualization_image(&texture.image, width, height, rgba, &mut images);
            }
        }
        VisualizationImageData::PanelsRgba {
            panel_width,
            panel_height,
            input_rgba,
            mask_rgba,
            output_rgba,
        } => {
            set_texture_layout(world, &texture, AutoGazeTextureLayout::Panels);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            remove_panel_gpu_visualization_handles(world, &texture);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_panel_visualization_images(
                    &texture,
                    &mut images,
                    PanelVisualizationImages {
                        width: panel_width,
                        height: panel_height,
                        input_rgba,
                        mask_rgba,
                        output_rgba,
                    },
                );
            }
        }
    }
}

fn set_texture_layout(world: &mut World, texture: &AutoGazeTexture, layout: AutoGazeTextureLayout) {
    if let Some(entity) = texture.side_by_side_entity {
        set_node_display(
            world,
            entity,
            if layout == AutoGazeTextureLayout::SideBySide {
                Display::Flex
            } else {
                Display::None
            },
        );
    }

    for entity in [
        texture.input_entity,
        texture.mask_entity,
        texture.output_entity,
    ]
    .into_iter()
    .flatten()
    {
        set_node_display(
            world,
            entity,
            if layout == AutoGazeTextureLayout::Panels {
                Display::Flex
            } else {
                Display::None
            },
        );
    }

    if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
        texture.layout = layout;
    }
}

fn set_node_display(world: &mut World, entity: Entity, display: Display) {
    if let Ok(mut entity) = world.get_entity_mut(entity)
        && let Some(mut node) = entity.get_mut::<Node>()
    {
        node.display = display;
    }
}

fn remove_gpu_visualization_handle(world: &mut World, entity: Option<Entity>) {
    if let Some(entity) = entity
        && let Ok(mut entity) = world.get_entity_mut(entity)
    {
        entity.remove::<BevyBurnHandle<AutoGazeBevyBackend>>();
    }
}

fn remove_panel_gpu_visualization_handles(world: &mut World, texture: &AutoGazeTexture) {
    remove_gpu_visualization_handle(world, texture.input_entity);
    remove_gpu_visualization_handle(world, texture.mask_entity);
    remove_gpu_visualization_handle(world, texture.output_entity);
}

fn set_gpu_panel_upload_handle(
    world: &mut World,
    entity: Option<Entity>,
    image: Handle<Image>,
    tensor: Tensor<AutoGazeBevyBackend, 3>,
) {
    let Some(entity) = entity else {
        return;
    };
    let Ok(mut entity) = world.get_entity_mut(entity) else {
        return;
    };
    if let Some(mut handle) = entity.get_mut::<BevyBurnHandle<AutoGazeBevyBackend>>() {
        handle.bevy_image = image;
        handle.tensor = tensor;
        handle.direction = BindingDirection::BurnToBevy;
        handle.xfer = TransferKind::Gpu;
        handle.upload = true;
    } else {
        entity.insert(BevyBurnHandle::<AutoGazeBevyBackend> {
            bevy_image: image,
            tensor,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Gpu,
        });
    }
    entity.insert(OneShotGpuUpload);
}

struct PanelVisualizationImages {
    width: u32,
    height: u32,
    input_rgba: Vec<u8>,
    mask_rgba: Vec<u8>,
    output_rgba: Vec<u8>,
}

fn set_panel_visualization_images(
    texture: &AutoGazeTexture,
    images: &mut Assets<Image>,
    panels: PanelVisualizationImages,
) {
    let PanelVisualizationImages {
        width,
        height,
        input_rgba,
        mask_rgba,
        output_rgba,
    } = panels;
    set_visualization_image(&texture.input_image, width, height, input_rgba, images);
    set_visualization_image(&texture.mask_image, width, height, mask_rgba, images);
    set_visualization_image(&texture.output_image, width, height, output_rgba, images);
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

fn set_gpu_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    images: &mut Assets<Image>,
) {
    let width = width.max(1);
    let height = height.max(1);
    if let Some(image) = images.get(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba32Float
        && image.texture_descriptor.usage.contains(
            TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
        )
    {
        return;
    }

    let _ = images.insert(handle.id(), gpu_visualization_image(width, height));
}

fn gpu_visualization_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
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
    log(&format!("AutoGaze perf summary: {summary}"));
    if let Err(err) = write_perf_summary(config.perf_summary_path.as_deref(), &summary) {
        log(&format!("failed to write AutoGaze perf summary: {err}"));
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
            TextSpan::new(format_fps(f64::NAN)),
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
            **text = format_fps(timing.e2e_fps());
        } else if let Some(fps) = diagnostics.get(&INFERENCE_FPS)
            && let Some(value) = fps.smoothed()
        {
            **text = format_fps(value);
        }
    }
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
        + usize::from(config.show_psnr);
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
        assert_eq!(
            config.mask_visualization_mode,
            AutoGazeMaskVisualizationMode::ScaleRows
        );
        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
        assert_eq!(config.keyframe_duration, DEFAULT_BIRDS_KEYFRAME_DURATION);
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Gpu);
        assert!(config.show_psnr);
        assert_eq!(config.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert!(config.streaming_cache);
        assert!(should_use_streaming_cache(
            config.streaming_cache,
            config.frames_per_clip,
            config.mode.inference_mode()
        ));
        assert_eq!(
            pipeline_options_from_config(&config).max_gaze_tokens_each_frame(),
            None,
            "realtime defaults must delegate to the model-configured inference budget"
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
        let options_cpu = VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Cpu);
        let options_gpu = VisualizationOptions::new(1.0, 0.38, false, BevyDisplayTransfer::Gpu);

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
            } => {
                assert_eq!(panel_width, width as u32);
                assert_eq!(panel_height, height as u32);
                assert_eq!(input_rgba, rgba);
                assert_eq!(mask_rgba, expected.mask_rgba);
                assert_eq!(output_rgba, expected.blend_rgba);
            }
            _ => panic!("expected split panel visualization payload"),
        }
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
                } = *panels;
                assert_eq!(panel_width, width as u32);
                assert_eq!(panel_height, height as u32);
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
        let visualization = visualize_frame_rgba(
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
        assert!((visualization.psnr_db.expect("psnr") - expected_psnr).abs() <= f64::EPSILON);
        assert!(visualization.visualize_cpu_ms >= 0.0);
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
    fn metric_overlay_reserves_top_space_above_visualization() {
        let mut config = BevyBurnAutoGazeConfig {
            show_fps: false,
            show_gaze_ratio: false,
            show_psnr: false,
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
        assert_eq!(
            metric_panel_top_reserved_height(&config),
            UI_MARGIN_PX * 2.0 + METRIC_ROW_HEIGHT * 3.0
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
            show_psnr: true,
            ..Default::default()
        }));
        stats.record(
            InferenceTiming {
                sequence: 7,
                clip_frames: 16,
                model_frames: 2,
                trace_points: 42,
                width: 640,
                height: 360,
                total_ms: 20.0,
                model_ms: 8.0,
                input_ms: 2.0,
                display_input_ms: 0.25,
                pack_ms: 1.0,
                visualize_ms: 3.0,
                visualize_cpu_ms: 2.5,
                display_ms: 1.0,
                tensor_ms: 0.5,
                output_tensor_bytes: 640 * 360 * 3 * 4 * std::mem::size_of::<f32>(),
                display_input_residency: DisplayInputResidency::ModelTensorReuse,
                gaze_update_ratio: 0.25,
                gaze_update_ratio_sample: Some(0.25),
                psnr_db: Some(42.0),
                tensor_interframe_path: Some(AutoGazeTensorInterframePath::SparseRects),
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
        assert_eq!(summary["processed_frames"], 1);
        assert_eq!(summary["processed_model_frames"], 2);
        assert_eq!(summary["latest_sequence"], 7);
        assert_eq!(summary["latest_clip_frames"], 16);
        assert_eq!(summary["latest_model_frames"], 2);
        assert_eq!(summary["latest_trace_points"], 42);
        assert_eq!(summary["latest_width"], 640);
        assert_eq!(summary["latest_height"], 360);
        assert_eq!(summary["latest_gaze_update_ratio"], 0.25);
        assert_eq!(summary["latest_tensor_interframe_path"], "sparse-rects");
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
        assert_eq!(summary["mask_visualization_mode"], "scale-rows");
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
        assert_eq!(summary["show_psnr"], true);
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
        assert_eq!(summary["avg_trace_points"], 42.0);
        assert_eq!(summary["avg_input_ms"], 2.0);
        assert_eq!(summary["avg_display_input_ms"], 0.25);
        assert_eq!(summary["avg_pack_ms"], 1.0);
        assert_eq!(summary["avg_visualize_ms"], 3.0);
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
        assert_eq!(sample["processed_frames"], 1);
        assert_eq!(sample["processed_model_frames"], 2);
        assert_eq!(sample["latest_sequence"], 7);
        assert_eq!(sample["latest_clip_frames"], 16);
        assert_eq!(sample["latest_model_frames"], 2);
        assert_eq!(sample["latest_trace_points"], 42);
        assert_eq!(sample["avg_trace_points"], 42.0);
        assert_eq!(sample["latest_width"], 640);
        assert_eq!(sample["latest_height"], 360);
        assert_eq!(sample["latest_gaze_update_ratio"], 0.25);
        assert_eq!(sample["latest_tensor_interframe_path"], "sparse-rects");
        assert_eq!(sample["display_residency"], "gpu-tensor");
        assert_eq!(sample["display_input_residency"], "model-tensor-reuse");
        assert_eq!(sample["latest_display_input_ms"], 0.25);
        assert_eq!(sample["latest_output_rgba_bytes"], 0);
        assert_eq!(
            sample["latest_output_tensor_bytes"],
            640 * 360 * 3 * 4 * std::mem::size_of::<f32>()
        );
        assert_eq!(sample["mode"], "tiled");
        assert_eq!(sample["visualization_mode"], "interframe");
        assert_eq!(sample["mask_visualization_mode"], "scale-rows");
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
        assert_eq!(sample["show_psnr"], true);
        assert_eq!(sample["burn_backend"], "webgpu");
        assert_eq!(sample["render_adapter_name"], "test adapter");
        assert_eq!(sample["render_adapter_vendor"], 1234);
        assert_eq!(sample["render_adapter_device_type"], "DiscreteGpu");
        assert_eq!(sample["render_adapter_backend"], "Vulkan");
        assert_eq!(sample["render_adapter_driver"], "test-driver");
        assert_eq!(sample["render_adapter_driver_info"], "test-driver-info");
        assert_eq!(sample["avg_output_fps"], 50.0);
        assert_eq!(sample["avg_input_fps"], 100.0);
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
    fn viewer_pipeline_options_delegate_to_core_runtime_options() {
        let mut config = BevyBurnAutoGazeConfig {
            max_gaze_tokens_each_frame: 12,
            task_loss_requirement: Some(0.7),
            tile_batch_size: 9,
            ..Default::default()
        };

        let options = pipeline_options_from_config(&config);
        assert_eq!(options.max_gaze_tokens_each_frame(), Some(12));
        assert_eq!(
            options.task_loss_requirement(),
            burn_autogaze::AutoGazeTaskLossOption::Value(0.7)
        );
        assert_eq!(options.tile_batch_size(), Some(9));

        config.disable_task_loss_requirement = true;
        let options = pipeline_options_from_config(&config);
        assert_eq!(
            options.task_loss_requirement(),
            burn_autogaze::AutoGazeTaskLossOption::Disabled
        );
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
