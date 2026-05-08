use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{
        Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
        RegisterDiagnostic,
    },
    ecs::world::CommandQueue,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::widget::ImageNode,
};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
#[cfg(target_arch = "wasm32")]
use burn_autogaze::{AutoGazeConfig, AutoGazeLoadOptions, NativeAutoGazeModel};
use burn_autogaze::{AutoGazeInferenceMode, AutoGazePipeline, FixationPoint};
use image::RgbaImage;

pub mod platform;

pub type AutoGazeBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type AutoGazeBevyDevice = burn::backend::wgpu::WgpuDevice;

pub const DEFAULT_NATIVE_MODEL_DIR: &str = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a";
const DEFAULT_CONFIG_URL: &str = "https://huggingface.co/nvidia/AutoGaze/resolve/main/config.json";
const DEFAULT_WEIGHTS_URL: &str =
    "https://huggingface.co/nvidia/AutoGaze/resolve/main/model.safetensors";
const MODEL_INPUT_SIZE: usize = 224;
const MAX_IN_FLIGHT_TASKS: usize = 1;
const INFERENCE_FPS: DiagnosticPath = DiagnosticPath::const_new("autogaze_inference_fps");

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
            "tile" | "tile-224" => Ok(Self::Tile224),
            other => Err(format!("unsupported autogaze mode `{other}`")),
        }
    }
}

#[derive(Resource, Clone, Debug)]
pub struct BevyBurnAutoGazeConfig {
    pub press_esc_to_close: bool,
    pub show_fps: bool,
    pub model_dir: PathBuf,
    pub config_url: String,
    pub weights_url: String,
    pub image_path: Option<PathBuf>,
    pub mode: BevyAutoGazeMode,
    pub top_k: usize,
    pub max_gaze_tokens_each_frame: usize,
    pub frames_per_clip: usize,
    pub mask_radius_scale: f32,
    pub blend_alpha: f32,
}

impl Default for BevyBurnAutoGazeConfig {
    fn default() -> Self {
        Self {
            press_esc_to_close: true,
            show_fps: true,
            model_dir: PathBuf::from(DEFAULT_NATIVE_MODEL_DIR),
            config_url: DEFAULT_CONFIG_URL.to_string(),
            weights_url: DEFAULT_WEIGHTS_URL.to_string(),
            image_path: None,
            mode: BevyAutoGazeMode::Resize224,
            top_k: 4,
            max_gaze_tokens_each_frame: 4,
            frames_per_clip: 2,
            mask_radius_scale: 3.0,
            blend_alpha: 0.72,
        }
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

    if config.press_esc_to_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.register_diagnostic(Diagnostic::new(INFERENCE_FPS));
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    app.add_systems(
        Update,
        (
            setup_ui,
            begin_model_load,
            finish_model_load,
            handle_tasks,
            process_frames,
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
) {
    if texture.entity.is_some() {
        return;
    }

    let size = Extent3d {
        width: texture.width.max(1),
        height: texture.height.max(1),
        depth_or_array_layers: 1,
    };
    texture.image = images.add(Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));

    let mut image_entity = None;
    commands
        .spawn(Node {
            display: Display::Grid,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
            ..default()
        })
        .with_children(|builder| {
            let entity = builder
                .spawn((
                    ImageNode::new(texture.image.clone()).with_mode(NodeImageMode::Stretch),
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
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
                    for label in ["Input", "Mask", "Blend"] {
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

fn begin_model_load(mut state: ResMut<AutoGazeModelState>) {
    if state.pipeline.is_some() || state.load_task.is_some() {
        return;
    }

    log("loading AutoGaze model...");
    state.load_task = Some(spawn_model_load_task(state.config.clone()));
}

fn finish_model_load(mut state: ResMut<AutoGazeModelState>) {
    let Some(task) = state.load_task.as_mut() else {
        return;
    };

    if let Some(result) = block_on(future::poll_once(task)) {
        match result {
            Ok(pipeline) => {
                log("AutoGaze model ready");
                state.pipeline = Some(Arc::new(Mutex::new(pipeline)));
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
    config: Res<BevyBurnAutoGazeConfig>,
    texture: Res<AutoGazeTexture>,
    static_frame: Res<StaticFrame>,
    mut frame_queue: ResMut<FrameQueue>,
    active_tasks: Query<&ProcessAutoGaze>,
) {
    let Some(pipeline) = model.pipeline.as_ref() else {
        return;
    };
    let Some(image_entity) = texture.entity else {
        return;
    };
    if active_tasks.iter().count() >= MAX_IN_FLIGHT_TASKS {
        return;
    }

    let frame = if let Some(frame) = static_frame.0.as_ref() {
        Some((**frame).clone())
    } else {
        receive_frame()
    };

    let Some(frame) = frame else {
        return;
    };
    let Some(clip) = frame_queue.push(frame, config.frames_per_clip) else {
        return;
    };

    let task_entity = commands.spawn_empty().id();
    let pipeline = pipeline.clone();
    let mode = config.mode.inference_mode();
    let top_k = config.top_k.max(1);
    let radius_scale = config.mask_radius_scale;
    let blend_alpha = config.blend_alpha;

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let visualization =
            run_autogaze_visualization(pipeline, clip, top_k, mode, radius_scale, blend_alpha);

        let mut queue = CommandQueue::default();
        queue.push(move |world: &mut World| {
            if let Ok(entity) = world.get_entity_mut(image_entity)
                && let Some(image_node) = entity.get::<ImageNode>()
            {
                let handle = image_node.image.clone();
                if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                    let image = Image::new(
                        Extent3d {
                            width: visualization.width,
                            height: visualization.height,
                            depth_or_array_layers: 1,
                        },
                        TextureDimension::D2,
                        visualization.rgba,
                        TextureFormat::Rgba8UnormSrgb,
                        RenderAssetUsages::default(),
                    );
                    let _ = images.insert(handle.id(), image);
                }
            }

            if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
                texture.width = visualization.width;
                texture.height = visualization.height;
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

fn handle_tasks(
    mut commands: Commands,
    mut diagnostics: Diagnostics,
    mut last_frame: Local<Time<Real>>,
    mut active_tasks: Query<&mut ProcessAutoGaze>,
) {
    for mut task in &mut active_tasks {
        if let Some(mut queue) = block_on(future::poll_once(&mut task.0)) {
            if let Some(last_instant) = last_frame.last_update() {
                let delta_seconds = last_instant.elapsed().as_secs_f64();
                if delta_seconds > 0.0 {
                    diagnostics.add_measurement(&INFERENCE_FPS, || 1.0 / delta_seconds);
                }
            }
            last_frame.update();
            commands.append(&mut queue);
        }
    }
}

fn spawn_model_load_task(
    config: BevyBurnAutoGazeConfig,
) -> Task<Result<AutoGazePipeline<AutoGazeBevyBackend>, String>> {
    AsyncComputeTaskPool::get().spawn(async move {
        let device = burn_device();
        load_model(config, &device).await
    })
}

#[cfg(not(target_arch = "wasm32"))]
async fn load_model(
    config: BevyBurnAutoGazeConfig,
    device: &AutoGazeBevyDevice,
) -> Result<AutoGazePipeline<AutoGazeBevyBackend>, String> {
    let mut pipeline = AutoGazePipeline::from_hf_dir(&config.model_dir, device)
        .map_err(|err| format!("{err:#}"))?;
    pipeline.set_max_gaze_tokens_each_frame(config.max_gaze_tokens_each_frame);
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
    pipeline.set_max_gaze_tokens_each_frame(config.max_gaze_tokens_each_frame);
    Ok(pipeline)
}

fn run_autogaze_visualization(
    pipeline: Arc<Mutex<AutoGazePipeline<AutoGazeBevyBackend>>>,
    clip: Vec<RgbaImage>,
    top_k: usize,
    mode: AutoGazeInferenceMode,
    radius_scale: f32,
    blend_alpha: f32,
) -> Visualization {
    let device = burn_device();
    let width = clip[0].width() as usize;
    let height = clip[0].height() as usize;
    let video = rgba_clip_to_tensor(&clip, &device);
    let traces = {
        let pipeline = pipeline.lock().expect("AutoGaze model poisoned");
        pipeline.trace_video_with_mode(video, top_k, mode)
    };
    AutoGazeBevyBackend::sync(&device).expect("failed to sync Burn WebGPU backend");

    let frame_index = clip.len().saturating_sub(1);
    let points = traces
        .first()
        .and_then(|trace| trace.frames.get(frame_index))
        .map(|set| set.points.clone())
        .unwrap_or_default();
    visualize_points(
        clip.last().expect("nonempty clip"),
        width,
        height,
        &points,
        radius_scale,
        blend_alpha,
    )
}

fn rgba_clip_to_tensor(
    clip: &[RgbaImage],
    device: &AutoGazeBevyDevice,
) -> Tensor<AutoGazeBevyBackend, 5> {
    let frames = clip.len();
    let width = clip[0].width() as usize;
    let height = clip[0].height() as usize;
    let pixels_per_frame = width * height;
    let mut values = Vec::with_capacity(frames * 3 * pixels_per_frame);

    for frame in clip {
        let rgba = frame.as_raw();
        for channel in 0..3 {
            for pixel in 0..pixels_per_frame {
                values.push(rgba[pixel * 4 + channel] as f32 / 255.0);
            }
        }
    }

    Tensor::from_data(
        TensorData::new(values, [1, frames, 3, height, width]),
        device,
    )
}

struct Visualization {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn visualize_points(
    rgba: &RgbaImage,
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
        let cx = point.x * width.saturating_sub(1) as f32;
        let cy = point.y * height.saturating_sub(1) as f32;
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

    let input = rgba.as_raw();
    let out_width = width * 3;
    let mut out = vec![0u8; out_width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let src = pixel * 4;
            let a = alpha[pixel].clamp(0.0, 1.0);
            let mask = (a * 255.0).round() as u8;
            let overlay = (a * blend_alpha).clamp(0.0, 1.0);
            let mut blended = [0u8; 4];
            for channel in 0..3 {
                let base = input[src + channel] as f32;
                blended[channel] = (base * (1.0 - overlay) + 255.0 * overlay).round() as u8;
            }
            blended[3] = input[src + 3];

            write_pixel(&mut out, out_width, 0, x, y, &input[src..src + 4]);
            write_pixel(&mut out, out_width, width, x, y, &[mask, mask, mask, 255]);
            write_pixel(&mut out, out_width, width * 2, x, y, &blended);
        }
    }

    Visualization {
        width: out_width as u32,
        height: height as u32,
        rgba: out,
    }
}

fn write_pixel(out: &mut [u8], out_width: usize, x_offset: usize, x: usize, y: usize, rgba: &[u8]) {
    let dst = (y * out_width + x_offset + x) * 4;
    out[dst..dst + 4].copy_from_slice(rgba);
}

fn burn_device() -> AutoGazeBevyDevice {
    static DEVICE: OnceLock<AutoGazeBevyDevice> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let device = AutoGazeBevyDevice::default();
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                &device,
                Default::default(),
            );
            device
        })
        .clone()
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
                bottom: Val::Px(8.0),
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
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&INFERENCE_FPS)
            && let Some(value) = fps.smoothed()
        {
            **text = format!("{value:.1}");
        }
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
