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
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind};
use burn::tensor::{Int, Tensor};
#[cfg(test)]
use burn_autogaze::fixation_scale_mask_rgba;
#[cfg(target_arch = "wasm32")]
use burn_autogaze::{AutoGazeConfig, AutoGazeLoadOptions, NativeAutoGazeModel};
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape, AutoGazeVisualizationMode,
    AutoGazeVisualizationState, FixationPoint,
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
pub const DEFAULT_REALTIME_INFERENCE_WIDTH: u32 = MODEL_INPUT_SIZE as u32;
pub const DEFAULT_REALTIME_TOP_K: usize = 10;
const MAX_IN_FLIGHT_TASKS: usize = 1;
pub const DEFAULT_KEYFRAME_DURATION: usize = 30;
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
    pub const fn inference_mode(self) -> AutoGazeInferenceMode {
        match self {
            Self::Resize224 => AutoGazeInferenceMode::ResizeToModelInput,
            Self::Tile224 => AutoGazeInferenceMode::TiledFullResolution {
                tile_size: MODEL_INPUT_SIZE,
                stride: MODEL_INPUT_SIZE,
            },
        }
    }
}

impl std::str::FromStr for BevyAutoGazeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "resize" | "resize-224" => Ok(Self::Resize224),
            "tile" | "tile-224" | "tiled" | "full-res" | "fullres" => Ok(Self::Tile224),
            other => Err(format!("unsupported autogaze mode `{other}`")),
        }
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
            max_gaze_tokens_each_frame: DEFAULT_REALTIME_TOP_K,
            tile_batch_size: 8,
            task_loss_requirement: None,
            disable_task_loss_requirement: false,
            frames_per_clip: 2,
            inference_width: Some(DEFAULT_REALTIME_INFERENCE_WIDTH),
            inference_height: None,
            mask_cell_scale: 1.0,
            blend_alpha: 0.72,
            visualization_mode: AutoGazeVisualizationMode::FullBlend,
            keyframe_duration: DEFAULT_KEYFRAME_DURATION,
            log_pipeline_timing: false,
        }
    }
}

impl BevyBurnAutoGazeConfig {
    pub fn apply_option(&mut self, key: &str, value: &str) -> Result<(), String> {
        let key = key.trim().replace('_', "-").to_ascii_lowercase();
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
            other => Err(format!("unsupported bevy_burn_autogaze option `{other}`")),
        }
    }

    pub fn apply_query_string(&mut self, query: &str) -> Vec<String> {
        let query = query.strip_prefix('?').unwrap_or(query);
        let mut errors = Vec::new();

        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, "true"));
            let key = decode_url_component(key);
            let value = decode_url_component(value);
            if let Err(err) = self.apply_option(&key, &value) {
                errors.push(err);
            }
        }

        errors
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
    frames: VecDeque<RgbaImage>,
}

impl FrameQueue {
    fn push(&mut self, frame: RgbaImage, max_len: usize) -> Option<Vec<RgbaImage>> {
        let max_len = max_len.max(1);
        let (width, height) = frame.dimensions();
        if self.width != width || self.height != height {
            self.frames.clear();
            self.width = width;
            self.height = height;
        }

        self.frames.push_back(frame);
        while self.frames.len() > max_len {
            self.frames.pop_front();
        }

        (self.frames.len() == max_len).then(|| self.frames.iter().cloned().collect())
    }
}

#[derive(Resource, Default, Clone)]
struct StaticFrame(Option<Arc<RgbaImage>>);

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
    clip_frames: usize,
    width: usize,
    height: usize,
    pack_ms: f64,
    trace_ms: f64,
    sync_ms: f64,
    visualize_ms: f64,
    display_ms: f64,
    total_ms: f64,
}

#[derive(Resource, Clone, Debug, Default)]
struct InferenceTimingStats {
    latest: Option<InferenceTiming>,
    last_log: Option<Timestamp>,
}

impl InferenceTimingStats {
    fn record(&mut self, timing: InferenceTiming, should_log: bool) {
        self.latest = Some(timing);
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
            "AutoGaze timing: {:.1} fps e2e ({:.1} ms) clip={} {}x{}, pack={:.1} ms, trace={:.1} ms, sync={:.1} ms, visualize={:.1} ms, display={:.1} ms",
            timing.e2e_fps(),
            timing.total_ms,
            timing.clip_frames,
            timing.width,
            timing.height,
            timing.pack_ms,
            timing.trace_ms,
            timing.sync_ms,
            timing.visualize_ms,
            timing.display_ms,
        ));
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

#[derive(SystemParam)]
struct FrameInputParams<'w> {
    config: Res<'w, BevyBurnAutoGazeConfig>,
    static_frame: Res<'w, StaticFrame>,
    frame_queue: ResMut<'w, FrameQueue>,
    visualization_state: ResMut<'w, BevyVisualizationState>,
    gaze_ratio_stats: ResMut<'w, GazeRatioStats>,
    psnr_stats: ResMut<'w, PsnrStats>,
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
    app.insert_resource(BevyVisualizationState::new(
        config.visualization_mode,
        config.keyframe_duration,
    ));
    app.insert_resource(GazeRatioStats::default());
    app.insert_resource(PsnrStats::default());
    app.insert_resource(InferenceTimingStats::default());
    app.insert_resource(AutoGazeModelState {
        config: config.clone(),
        pipeline: None,
        load_task: None,
    });
    app.insert_resource(load_static_frame(config.image_path.as_deref()));

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
            fit_visualization_node,
        )
            .chain(),
    );

    app
}

pub fn run_app(config: BevyBurnAutoGazeConfig) {
    viewer_app(config).run();

    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
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
    let Some(device) = burn_device.as_ref().and_then(|device| device.device()) else {
        return;
    };

    let size = Extent3d {
        width: texture.width.max(1),
        height: texture.height.max(1),
        depth_or_array_layers: 1,
    };
    texture.image = images.add(visualization_image(
        texture.width.max(1),
        texture.height.max(1),
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
                    BevyBurnHandle::<AutoGazeBevyBackend> {
                        bevy_image: texture.image.clone(),
                        tensor: Tensor::<AutoGazeBevyBackend, 3>::zeros(
                            [size.height as usize, size.width as usize, 4],
                            device,
                        ),
                        upload: true,
                        direction: BindingDirection::BurnToBevy,
                        xfer: TransferKind::Gpu,
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
    let Some(image_entity) = texture.entity else {
        return;
    };
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

    let frame = if let Some(frame) = frame_input.static_frame.0.as_ref() {
        Some((**frame).clone())
    } else {
        receive_frame()
    };

    let Some(frame) = frame else {
        return;
    };
    let frame = prepare_frame_for_inference(frame, &frame_input.config);
    let Some(clip) = frame_input
        .frame_queue
        .push(frame, frame_input.config.frames_per_clip)
    else {
        return;
    };

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

    let task = AsyncComputeTaskPool::get().spawn(async move {
        #[cfg(target_arch = "wasm32")]
        let result = run_autogaze_visualization(
            pipeline,
            clip,
            top_k,
            mode,
            visualization_options,
            visualization_state,
            device,
        )
        .await;

        #[cfg(not(target_arch = "wasm32"))]
        let result = run_autogaze_visualization(
            pipeline,
            clip,
            top_k,
            mode,
            visualization_options,
            visualization_state,
            device,
        );

        let mut queue = CommandQueue::default();
        queue.push(move |world: &mut World| {
            match result {
                Ok((visualization, visualization_state)) => {
                    let Visualization {
                        width,
                        height,
                        tensor,
                        gaze_update_ratio,
                        psnr_db,
                        mut timing,
                    } = visualization;
                    let display_start = timestamp_now();
                    apply_visualization_to_world(world, image_entity, width, height, tensor);
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
    burn_device: Option<Res<BurnDevice>>,
    mut frame_input: FrameInputParams,
    mut burn_handles: Query<&mut BevyBurnHandle<AutoGazeBevyBackend>>,
    active_tasks: Query<&ProcessAutoGaze>,
    mut images: ResMut<Assets<Image>>,
) {
    let model_ready = model.pipeline.is_some();
    let inference_busy = active_tasks.iter().count() >= MAX_IN_FLIGHT_TASKS;
    if model_ready && !inference_busy {
        return;
    }

    let Some(image_entity) = texture.entity else {
        return;
    };
    let Some(device) = burn_device.as_ref().and_then(|device| device.device()) else {
        return;
    };

    let frame = if let Some(frame) = frame_input.static_frame.0.as_ref() {
        Some((**frame).clone())
    } else {
        receive_frame()
    };

    let Some(frame) = frame else {
        return;
    };
    let frame = prepare_frame_for_inference(frame, &frame_input.config);

    frame_input
        .frame_queue
        .push(frame, frame_input.config.frames_per_clip);
    let Some(frame) = frame_input.frame_queue.frames.back() else {
        return;
    };

    if model_ready {
        let visualization =
            match live_preview_visualization(frame, frame_input.config.show_psnr, device) {
                Ok(visualization) => visualization,
                Err(err) => {
                    log(&format!("failed to draw AutoGaze live preview: {err}"));
                    return;
                }
            };
        apply_visualization_to_texture(
            image_entity,
            visualization,
            &mut texture,
            &mut burn_handles,
            &mut images,
        );
        return;
    }

    frame_input.visualization_state.configure(
        frame_input.config.visualization_mode,
        frame_input.config.keyframe_duration,
    );
    let visualization_options = VisualizationOptions::new(
        frame_input.config.mask_cell_scale,
        frame_input.config.blend_alpha,
        false,
    );
    let visualization = match visualize_points(
        frame,
        &[],
        visualization_options,
        &mut frame_input.visualization_state,
        device,
    ) {
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
    apply_visualization_to_texture(
        image_entity,
        visualization,
        &mut texture,
        &mut burn_handles,
        &mut images,
    );
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

#[cfg(not(target_arch = "wasm32"))]
fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    clip: Vec<RgbaImage>,
    top_k: usize,
    mode: AutoGazeInferenceMode,
    visualization_options: VisualizationOptions,
    mut visualization_state: BevyVisualizationState,
    device: AutoGazeBevyDevice,
) -> Result<(Visualization, BevyVisualizationState), String> {
    let total_start = timestamp_now();
    let first_frame = clip
        .first()
        .ok_or_else(|| "AutoGaze clip must contain at least one frame".to_string())?;
    let width = first_frame.width() as usize;
    let height = first_frame.height() as usize;
    let pack_start = timestamp_now();
    let mut rgba = Vec::with_capacity(clip.len() * width * height * 4);
    for frame in &clip {
        if frame.width() as usize != width || frame.height() as usize != height {
            return Err("AutoGaze clip frame dimensions changed".to_string());
        }
        rgba.extend_from_slice(frame.as_raw());
    }
    let pack_ms = elapsed_ms(pack_start);
    let traces = {
        let pipeline = pipeline
            .lock()
            .map_err(|_| "AutoGaze model lock was poisoned".to_string())?;
        let shape = AutoGazeRgbaClipShape::new(clip.len(), height, width);
        let trace_start = timestamp_now();
        let traces = pipeline
            .trace_rgba_clip_with_mode(&rgba, shape, top_k, mode, &device)
            .map_err(|err| format!("{err:#}"))?;
        (traces, elapsed_ms(trace_start))
    };
    let frame_index = clip.len().saturating_sub(1);
    let points = traces
        .0
        .first()
        .and_then(|trace| trace.frames.get(frame_index))
        .map(|set| set.points.clone())
        .unwrap_or_default();
    let visualize_start = timestamp_now();
    let mut visualization = visualize_points(
        clip.last()
            .ok_or_else(|| "AutoGaze clip must contain at least one frame".to_string())?,
        &points,
        visualization_options,
        &mut visualization_state,
        &device,
    )?;
    visualization.timing = Some(InferenceTiming {
        clip_frames: clip.len(),
        width,
        height,
        pack_ms,
        trace_ms: traces.1,
        sync_ms: 0.0,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start),
    });
    Ok((visualization, visualization_state))
}

#[cfg(target_arch = "wasm32")]
async fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    clip: Vec<RgbaImage>,
    top_k: usize,
    mode: AutoGazeInferenceMode,
    visualization_options: VisualizationOptions,
    mut visualization_state: BevyVisualizationState,
    device: AutoGazeBevyDevice,
) -> Result<(Visualization, BevyVisualizationState), String> {
    let total_start = timestamp_now();
    let first_frame = clip
        .first()
        .ok_or_else(|| "AutoGaze clip must contain at least one frame".to_string())?;
    let width = first_frame.width() as usize;
    let height = first_frame.height() as usize;
    let pack_start = timestamp_now();
    let mut rgba = Vec::with_capacity(clip.len() * width * height * 4);
    for frame in &clip {
        if frame.width() as usize != width || frame.height() as usize != height {
            return Err("AutoGaze clip frame dimensions changed".to_string());
        }
        rgba.extend_from_slice(frame.as_raw());
    }
    let pack_ms = elapsed_ms(pack_start);
    let traces = {
        let pipeline = pipeline
            .lock()
            .map_err(|_| "AutoGaze model lock was poisoned".to_string())?
            .clone();
        let shape = AutoGazeRgbaClipShape::new(clip.len(), height, width);
        let trace_start = timestamp_now();
        let traces = pipeline
            .trace_rgba_clip_with_mode_async(&rgba, shape, top_k, mode, &device)
            .await
            .map_err(|err| format!("{err:#}"))?;
        (traces, elapsed_ms(trace_start))
    };

    let frame_index = clip.len().saturating_sub(1);
    let points = traces
        .0
        .first()
        .and_then(|trace| trace.frames.get(frame_index))
        .map(|set| set.points.clone())
        .unwrap_or_default();
    let visualize_start = timestamp_now();
    let mut visualization = visualize_points(
        clip.last()
            .ok_or_else(|| "AutoGaze clip must contain at least one frame".to_string())?,
        &points,
        visualization_options,
        &mut visualization_state,
        &device,
    )?;
    visualization.timing = Some(InferenceTiming {
        clip_frames: clip.len(),
        width,
        height,
        pack_ms,
        trace_ms: traces.1,
        sync_ms: 0.0,
        visualize_ms: elapsed_ms(visualize_start),
        display_ms: 0.0,
        total_ms: elapsed_ms(total_start),
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
    tensor: Tensor<AutoGazeBevyBackend, 3>,
    gaze_update_ratio: f64,
    psnr_db: Option<f64>,
    timing: Option<InferenceTiming>,
}

fn visualize_points(
    rgba: &RgbaImage,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    visualize_points_tensor(rgba, points, options, visualization_state, device)
}

fn visualize_points_tensor(
    rgba: &RgbaImage,
    points: &[FixationPoint],
    options: VisualizationOptions,
    visualization_state: &mut BevyVisualizationState,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    let width = rgba.width().max(1) as usize;
    let height = rgba.height().max(1) as usize;
    let input = rgba.as_raw();
    let visualization = visualization_state
        .0
        .visualize_rgba(
            input,
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
                .output_psnr_db(input)
                .map_err(|err| format!("{err:#}"))
        })
        .transpose()?;
    let tensor = rgba_tensor(
        &visualization.side_by_side_rgba,
        visualization.side_by_side_width,
        visualization.height,
        device,
    );
    Ok(Visualization {
        width: visualization.side_by_side_width as u32,
        height: visualization.height as u32,
        tensor,
        gaze_update_ratio: visualization.update_ratio(),
        psnr_db,
        timing: None,
    })
}

fn live_preview_visualization(
    rgba: &RgbaImage,
    calculate_psnr: bool,
    device: &AutoGazeBevyDevice,
) -> Result<Visualization, String> {
    let mut state = BevyVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 1);
    let mut visualization = visualize_points(
        rgba,
        &[],
        VisualizationOptions::new(1.0, 0.0, false),
        &mut state,
        device,
    )?;
    visualization.gaze_update_ratio = 0.0;
    visualization.psnr_db = calculate_psnr.then_some(f64::INFINITY);
    Ok(visualization)
}

fn rgba_tensor(
    rgba: &[u8],
    width: usize,
    height: usize,
    device: &AutoGazeBevyDevice,
) -> Tensor<AutoGazeBevyBackend, 3> {
    Tensor::<AutoGazeBevyBackend, 1, Int>::from_data(rgba, device)
        .float()
        .div_scalar(255.0)
        .reshape([height, width, 4])
}

fn apply_visualization_to_texture(
    image_entity: Entity,
    visualization: Visualization,
    texture: &mut AutoGazeTexture,
    burn_handles: &mut Query<&mut BevyBurnHandle<AutoGazeBevyBackend>>,
    images: &mut Assets<Image>,
) {
    let Ok(mut handle) = burn_handles.get_mut(image_entity) else {
        return;
    };

    let width = visualization.width;
    let height = visualization.height;
    ensure_visualization_image(&handle.bevy_image, width, height, images);
    handle.tensor = visualization.tensor;
    handle.upload = true;
    handle.direction = BindingDirection::BurnToBevy;
    handle.xfer = TransferKind::Gpu;
    texture.width = width;
    texture.height = height;
}

fn apply_visualization_to_world(
    world: &mut World,
    image_entity: Entity,
    width: u32,
    height: u32,
    tensor: Tensor<AutoGazeBevyBackend, 3>,
) {
    let image_handle = {
        let Ok(mut entity) = world.get_entity_mut(image_entity) else {
            return;
        };
        let Some(mut handle) = entity.get_mut::<BevyBurnHandle<AutoGazeBevyBackend>>() else {
            return;
        };
        handle.tensor = tensor;
        handle.upload = true;
        handle.direction = BindingDirection::BurnToBevy;
        handle.xfer = TransferKind::Gpu;
        handle.bevy_image.clone()
    };

    if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
        ensure_visualization_image(&image_handle, width, height, &mut images);
    }
}

fn ensure_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    images: &mut Assets<Image>,
) {
    if let Some(image) = images.get(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba32Float
        && image
            .texture_descriptor
            .usage
            .contains(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
    {
        return;
    }

    let _ = images.insert(handle.id(), visualization_image(width, height));
}

fn visualization_image(width: u32, height: u32) -> Image {
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
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    {
        platform::camera::receive_image()
    }

    #[cfg(all(feature = "web", target_arch = "wasm32"))]
    {
        platform::camera::receive_image()
    }

    #[cfg(not(any(
        all(feature = "native", not(target_arch = "wasm32")),
        all(feature = "web", target_arch = "wasm32")
    )))]
    {
        None
    }
}

fn load_static_frame(path: Option<&Path>) -> StaticFrame {
    let frame = path.map(|path| {
        Arc::new(
            image::open(path)
                .unwrap_or_else(|err| panic!("failed to load image `{}`: {err}", path.display()))
                .to_rgba8(),
        )
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
            "?mode=full-res&top_k=2&frames-per-clip=3&inference-width=1920&inference-height=1080&show-fps=false&show-psnr=false&task-loss-requirement=0.65&tile-batch-size=4&config-url=%2Fconfig.json&weights-url=%2Fmodel.safetensors&load-model=false&mask-cell-scale=2.5&blend-alpha=0.5&visualization-mode=interframe&keyframe-duration=7",
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
