use bevy::app::AppExit;
#[cfg(not(target_arch = "wasm32"))]
use bevy_burn_autogaze::{
    BevyAutoGazeMode, BevyDisplayTransfer, BevyFrameSource, BevySparseMaskSource,
    DEFAULT_BEVY_DECODE_CHUNK_SIZE, DEFAULT_BEVY_LIMIT_GENERATION_BUDGET,
    DEFAULT_BEVY_SHOW_TASK_LOSS_SLIDER, DEFAULT_BEVY_STREAMING_CACHE,
    DEFAULT_BEVY_TASK_LOSS_REQUIREMENT, DEFAULT_BIRDS_KEYFRAME_DURATION, DEFAULT_BLEND_ALPHA,
    DEFAULT_TILED_INFERENCE_WIDTH, default_frames_per_clip, default_inference_dimensions,
    default_max_gaze_tokens_for_limit, default_tile_batch_size, default_top_k,
};
use bevy_burn_autogaze::{BevyBurnAutoGazeConfig, run_app};
#[cfg(not(target_arch = "wasm32"))]
use burn_autogaze::{
    AutoGazeDecodeStrategy, AutoGazeMaskGeometryMode, AutoGazeMaskVisualizationMode,
    AutoGazeVisualizationMode, DEFAULT_PATCH_DIFF_GRID_SIZE, DEFAULT_PATCH_DIFF_THRESHOLD,
    task_loss_requirement_from_l1_db,
};

#[cfg(not(target_arch = "wasm32"))]
use clap::{ArgAction, Parser, ValueEnum};
#[cfg(not(target_arch = "wasm32"))]
use std::{fmt, path::PathBuf, str::FromStr};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeInferenceMode {
    #[value(
        name = "realtime",
        alias = "resize",
        alias = "resize-224",
        alias = "resize-to-model",
        alias = "fast",
        help = "Resize frames before the model pass; fastest."
    )]
    Realtime,
    #[value(
        name = "tiled",
        alias = "tile",
        alias = "tile-224",
        alias = "full-res",
        alias = "fullres",
        alias = "anyres",
        help = "Run 224px tiled AnyRes-style inference over the configured frame size."
    )]
    Tiled,
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for NativeInferenceMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Realtime => "realtime",
            Self::Tiled => "tiled",
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeInferenceMode> for BevyAutoGazeMode {
    fn from(mode: NativeInferenceMode) -> Self {
        match mode {
            NativeInferenceMode::Realtime => Self::Resize224,
            NativeInferenceMode::Tiled => Self::Tile224,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeDecodeStrategy {
    #[value(name = "host", alias = "cpu", help = "CPU greedy selection baseline.")]
    Host,
    #[value(
        name = "device",
        alias = "gpu",
        alias = "chunked",
        help = "Device-side greedy selection with chunk-boundary compact stopping readbacks."
    )]
    Device,
    #[value(
        name = "terminal",
        alias = "device-terminal",
        alias = "terminal-device",
        help = "Device-side greedy selection with one terminal compact readback after the full decode budget."
    )]
    Terminal,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeDecodeStrategy> for AutoGazeDecodeStrategy {
    fn from(value: NativeDecodeStrategy) -> Self {
        match value {
            NativeDecodeStrategy::Host => AutoGazeDecodeStrategy::HostGreedy,
            NativeDecodeStrategy::Device => AutoGazeDecodeStrategy::DeviceGreedy {
                chunk_size: DEFAULT_BEVY_DECODE_CHUNK_SIZE,
            },
            NativeDecodeStrategy::Terminal => AutoGazeDecodeStrategy::DeviceTerminalGreedy {
                chunk_size: DEFAULT_BEVY_DECODE_CHUNK_SIZE,
            },
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeVisualizationMode {
    #[value(
        name = "full-blend",
        alias = "blend",
        alias = "alpha-blend",
        help = "Draw the current frame with the current alpha-blended mask."
    )]
    FullBlend,
    #[value(
        name = "interframe",
        alias = "delta",
        alias = "video",
        help = "Update only gaze-selected regions between keyframes."
    )]
    Interframe,
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for NativeVisualizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FullBlend => "full-blend",
            Self::Interframe => "interframe",
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeVisualizationMode> for AutoGazeVisualizationMode {
    fn from(mode: NativeVisualizationMode) -> Self {
        match mode {
            NativeVisualizationMode::FullBlend => Self::FullBlend,
            NativeVisualizationMode::Interframe => Self::Interframe,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeMaskVisualizationMode {
    #[value(
        name = "image-overlay",
        alias = "image",
        alias = "input-overlay",
        alias = "source-overlay",
        alias = "alpha-overlay",
        help = "Render the input image with the colored multi-scale mask alpha-blended on top."
    )]
    ImageOverlay,
    #[value(
        name = "image-mask-only",
        alias = "mask-only",
        alias = "image-mask",
        alias = "masked-image",
        help = "Render only masked pixels as input image plus alpha-blended colored mask; unmasked pixels are transparent."
    )]
    ImageMaskOnly,
    #[value(
        name = "scale-rows",
        alias = "rows",
        alias = "per-scale",
        alias = "upstream",
        help = "Render one aspect-preserved diagnostic mask row per AutoGaze scale."
    )]
    ScaleRows,
    #[value(
        name = "overlay",
        alias = "combined",
        alias = "union",
        help = "Render all selected scale cells in one full-frame overlay."
    )]
    Overlay,
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for NativeMaskVisualizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ImageOverlay => "image-overlay",
            Self::ImageMaskOnly => "image-mask-only",
            Self::ScaleRows => "scale-rows",
            Self::Overlay => "overlay",
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeMaskVisualizationMode> for AutoGazeMaskVisualizationMode {
    fn from(mode: NativeMaskVisualizationMode) -> Self {
        match mode {
            NativeMaskVisualizationMode::ImageOverlay => Self::ImageOverlay,
            NativeMaskVisualizationMode::ImageMaskOnly => Self::ImageMaskOnly,
            NativeMaskVisualizationMode::ScaleRows => Self::ScaleRows,
            NativeMaskVisualizationMode::Overlay => Self::Overlay,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeMaskGeometryMode {
    #[value(
        name = "deduplicated",
        alias = "dedup",
        alias = "union",
        alias = "union-dedup",
        help = "Drop selected cells fully covered by larger native-scale cells; preserves the native update union while reducing redundant high-motion draw work."
    )]
    Deduplicated,
    #[value(
        name = "native",
        alias = "raw",
        alias = "multiscale",
        help = "Draw and update every native AutoGaze scale cell exactly as decoded."
    )]
    Native,
    #[value(
        name = "effective",
        alias = "projected",
        alias = "finest-grid",
        help = "Project selected tokens to the finest active grid for a compact sparse-token footprint."
    )]
    Effective,
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for NativeMaskGeometryMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Deduplicated => "deduplicated",
            Self::Native => "native",
            Self::Effective => "effective",
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeMaskGeometryMode> for AutoGazeMaskGeometryMode {
    fn from(mode: NativeMaskGeometryMode) -> Self {
        match mode {
            NativeMaskGeometryMode::Deduplicated => Self::Deduplicated,
            NativeMaskGeometryMode::Native => Self::Native,
            NativeMaskGeometryMode::Effective => Self::Effective,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<AutoGazeMaskGeometryMode> for NativeMaskGeometryMode {
    fn from(mode: AutoGazeMaskGeometryMode) -> Self {
        match mode {
            AutoGazeMaskGeometryMode::Native => Self::Native,
            AutoGazeMaskGeometryMode::Deduplicated => Self::Deduplicated,
            AutoGazeMaskGeometryMode::Effective => Self::Effective,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_NATIVE_MASK_GEOMETRY_MODE: NativeMaskGeometryMode =
    NativeMaskGeometryMode::Deduplicated;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeFrameSource {
    #[value(
        name = "camera",
        alias = "webcam",
        alias = "live",
        help = "Use the native camera stream."
    )]
    Camera,
    #[value(
        name = "static",
        alias = "image",
        alias = "file",
        help = "Use --image-path as a repeated source frame."
    )]
    StaticImage,
    #[value(
        name = "synthetic-pan",
        alias = "synthetic",
        alias = "pan",
        alias = "camera-pan",
        alias = "full-frame-motion",
        help = "Generate deterministic full-frame motion for repeatable high-motion perf runs."
    )]
    SyntheticPan,
    #[value(
        name = "synthetic-pulse",
        alias = "pulse",
        alias = "motion-pulse",
        alias = "burst-motion",
        help = "Generate deterministic static-to-motion pulses for repeatable FPS stability runs."
    )]
    SyntheticPulse,
    #[value(
        name = "synthetic-local-motion",
        alias = "local-motion",
        alias = "local",
        alias = "subtle-motion",
        help = "Generate deterministic local motion that decays from movement to subtle movement to stillness."
    )]
    SyntheticLocalMotion,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeFrameSource> for BevyFrameSource {
    fn from(source: NativeFrameSource) -> Self {
        match source {
            NativeFrameSource::Camera => Self::Camera,
            NativeFrameSource::StaticImage => Self::StaticImage,
            NativeFrameSource::SyntheticPan => Self::SyntheticPan,
            NativeFrameSource::SyntheticPulse => Self::SyntheticPulse,
            NativeFrameSource::SyntheticLocalMotion => Self::SyntheticLocalMotion,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeSparseMaskSource {
    #[value(
        name = "autogaze",
        alias = "model",
        alias = "nvidia",
        help = "Use NVIDIA AutoGaze autoregressive model readout."
    )]
    AutoGaze,
    #[value(
        name = "patch-diff",
        alias = "patchdiff",
        alias = "diff",
        alias = "frame-diff",
        help = "Use a single-scale tensor patch-difference mask instead of the AutoGaze model."
    )]
    PatchDiff,
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for NativeSparseMaskSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AutoGaze => "autogaze",
            Self::PatchDiff => "patch-diff",
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeSparseMaskSource> for BevySparseMaskSource {
    fn from(source: NativeSparseMaskSource) -> Self {
        match source {
            NativeSparseMaskSource::AutoGaze => Self::AutoGaze,
            NativeSparseMaskSource::PatchDiff => Self::PatchDiff,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeDisplayTransfer {
    #[value(
        name = "auto",
        alias = "adaptive",
        help = "Use the fastest measured display path for the current frame size; this avoids full-resolution f32 tensor-panel upload when CPU u8 image panels are faster."
    )]
    Auto,
    #[value(
        name = "gpu",
        alias = "device",
        alias = "interop",
        help = "Render Burn tensor output directly into the Bevy texture with GPU interop."
    )]
    Gpu,
    #[value(
        name = "cpu",
        alias = "host",
        alias = "rgba",
        help = "Read visualization output through CPU RGBA and upload it as a Bevy image."
    )]
    Cpu,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeDisplayTransfer> for BevyDisplayTransfer {
    fn from(transfer: NativeDisplayTransfer) -> Self {
        match transfer {
            NativeDisplayTransfer::Auto => Self::Auto,
            NativeDisplayTransfer::Gpu => Self::Gpu,
            NativeDisplayTransfer::Cpu => Self::Cpu,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum TaskLossRequirementArg {
    ModelDefault,
    Disabled,
    Value(f32),
}

#[cfg(not(target_arch = "wasm32"))]
impl FromStr for TaskLossRequirementArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" | "model" => Ok(Self::ModelDefault),
            "none" | "off" | "false" | "disabled" => Ok(Self::Disabled),
            _ => parse_nonnegative_f32(value).map(Self::Value),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(
    about = "native Bevy viewer for burn_autogaze",
    version,
    long_about = "Runs the burn_autogaze video pipeline with camera or static-image input and renders Input | Mask | Output through Bevy. The default path is a continuous realtime streaming configuration: 640px source resize, 16-frame rolling KV window, bounded realtime generated-token budget, deduplicated native mask geometry, adaptive display transfer, PSNR overlay, interframe output, a live quality slider, and no periodic visualization keyframes. Camera preview frames continue with the latest accepted mask while the next model decode is in flight. Use --max-gaze-tokens-each-frame 0 or --limit-generation-budget=false for the NVIDIA model's full configured budget, --mask-geometry native for exact decoded-cell diagnostics, --display-transfer gpu to force Bevy/Burn tensor interop, --streaming-cache=false for full-window comparison, or --mode tiled plus explicit 1080p/docs settings for full-resolution inspection."
)]
struct NativeArgs {
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Allow Escape to close the native Bevy window."
    )]
    press_esc_to_close: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Show render FPS and accepted model-output FPS overlay."
    )]
    show_fps: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Show current and EMA gaze update ratio overlay."
    )]
    show_gaze_ratio: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Show PSNR between the input frame and rendered output."
    )]
    show_psnr: bool,

    #[arg(
        long,
        default_value_t = DEFAULT_BEVY_SHOW_TASK_LOSS_SLIDER,
        action = ArgAction::Set,
        help = "Show a Bevy UI slider that updates the task-loss quality threshold live."
    )]
    show_task_loss_slider: bool,

    #[arg(
        long,
        value_name = "DIR",
        default_value = bevy_burn_autogaze::DEFAULT_NATIVE_MODEL_DIR,
        help = "Directory containing NVIDIA AutoGaze config.json and model.safetensors."
    )]
    model_dir: PathBuf,

    #[arg(
        long,
        value_name = "IMAGE",
        help = "Use a static PNG/JPEG frame instead of the native camera."
    )]
    image_path: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        help = "Input source. Defaults to camera, or static when --image-path is supplied. synthetic-local-motion is the most useful deterministic source for fine-cell FPS stability repros."
    )]
    source: Option<NativeFrameSource>,

    #[arg(
        long = "mask-source",
        alias = "sparse-mask-source",
        alias = "mask-driver",
        value_enum,
        default_value_t = NativeSparseMaskSource::AutoGaze,
        help = "Sparse mask source. autogaze runs the NVIDIA model; patch-diff uses tensor patch differences on the latest two frames."
    )]
    sparse_mask_source: NativeSparseMaskSource,

    #[arg(
        long = "patch-diff-grid",
        alias = "patch-grid",
        value_name = "PATCHES",
        value_parser = parse_nonzero_usize,
        default_value_t = DEFAULT_PATCH_DIFF_GRID_SIZE,
        help = "Square patch-diff grid size. 14 means a 14x14 single-scale sparse mask."
    )]
    patch_diff_grid_size: usize,

    #[arg(
        long = "patch-diff-threshold",
        alias = "patch-threshold",
        alias = "diff-threshold",
        value_name = "FLOAT",
        value_parser = parse_nonnegative_f32,
        default_value_t = DEFAULT_PATCH_DIFF_THRESHOLD,
        help = "Patch-diff score threshold. Lower values select more patches; the Bevy quality slider edits this value while patch-diff is active."
    )]
    patch_diff_threshold: f32,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Load and run the AutoGaze model. Set false to preview input plumbing only."
    )]
    load_model: bool,

    #[arg(
        long = "no-load-model",
        action = ArgAction::SetTrue,
        help = "Shortcut for --load-model=false."
    )]
    no_load_model: bool,

    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Run one synthetic inference during model load so WebGPU autotune does not stall the first displayed frame."
    )]
    warmup_model: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = NativeInferenceMode::Realtime,
        help = "Inference path. Default is realtime for live throughput. Aliases: resize-224, fast, tile-224, full-res, anyres."
    )]
    mode: NativeInferenceMode,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_usize,
        help = "Trace-slot lower bound. The recovered mask keeps all generated non-padded tokens; pass a larger value to force more trace slots."
    )]
    top_k: Option<usize>,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_usize,
        help = "Model-side generated-token cap. Default is 16 in realtime and 24 in tiled mode. Pass 0 to use the NVIDIA model config budget."
    )]
    max_gaze_tokens_each_frame: Option<usize>,

    #[arg(
        long,
        default_value_t = DEFAULT_BEVY_LIMIT_GENERATION_BUDGET,
        action = ArgAction::Set,
        help = "Use bounded generated-token caps: 16 in realtime and 24 in tiled mode unless --max-gaze-tokens-each-frame is set. Disable for full-budget quality inspection."
    )]
    limit_generation_budget: bool,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        help = "Number of 224px tiles traced together in tiled mode. Defaults to 64 so 720p fits in one model batch."
    )]
    tile_batch_size: Option<usize>,

    #[arg(
        long,
        value_name = "FLOAT|none",
        help = "Override viewer task-loss threshold. Default is 0.45; lower values such as 0.3 ask for more reconstruction quality without implicitly capping model output. Use none/off to disable, default/model for model config."
    )]
    task_loss_requirement: Option<TaskLossRequirementArg>,

    #[arg(
        long = "task-loss-requirement-db",
        alias = "task-loss-db",
        alias = "task-psnr-db",
        alias = "task-psnr",
        value_name = "DB",
        value_parser = parse_nonnegative_f32,
        conflicts_with = "task_loss_requirement",
        help = "Override model task-loss threshold using L1 reconstruction-error dB: threshold = 10^(-dB / 20)."
    )]
    task_loss_requirement_db: Option<f32>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Disable model task-loss filtering."
    )]
    disable_task_loss_requirement: bool,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        help = "Decoder context horizon in frames. Defaults to 16 in realtime and 2 in tiled mode. Realtime mode advances this as a streaming KV cache by default; larger values increase WebGPU attention memory."
    )]
    frames_per_clip: Option<usize>,

    #[arg(
        long,
        value_enum,
        default_value_t = NativeDecodeStrategy::Terminal,
        help = "Autoregressive token selection path. terminal is the Bevy default for GPU-resident mask rendering; host is the CPU-readback baseline."
    )]
    decode_strategy: NativeDecodeStrategy,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        default_value_t = DEFAULT_BEVY_DECODE_CHUNK_SIZE,
        help = "Autoregressive decode steps per chunk for device and terminal decode strategies."
    )]
    decode_chunk_size: usize,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        default_value_t = bevy_burn_autogaze::DEFAULT_MAX_IN_FLIGHT,
        help = "Maximum concurrent inference tasks. Defaults to 1 so busy camera frames are dropped instead of queued."
    )]
    max_in_flight: usize,

    #[arg(
        long,
        alias = "width",
        value_name = "PX",
        value_parser = parse_positive_u32,
        help = "Frame width before inference. Defaults to 1280 in tiled mode and 640 in realtime. If only height is set, width preserves input aspect."
    )]
    inference_width: Option<u32>,

    #[arg(
        long,
        alias = "height",
        value_name = "PX",
        value_parser = parse_positive_u32,
        help = "Frame height before inference. If omitted, input aspect is preserved."
    )]
    inference_height: Option<u32>,

    #[arg(
        long = "mask-cell-scale",
        alias = "mask-radius-scale",
        value_name = "SCALE",
        value_parser = parse_positive_f32,
        help = "Scale factor for crisp multi-scale mask cell extents. Defaults to 1.0."
    )]
    mask_cell_scale: Option<f32>,

    #[arg(
        long = "mask-visualization",
        alias = "mask-visualization-mode",
        alias = "mask-mode",
        value_enum,
        default_value_t = NativeMaskVisualizationMode::ImageMaskOnly,
        help = "Mask panel display. overlay draws colored mask cells; image-overlay alpha-blends colored mask cells over the input image; image-mask-only alpha-blends only masked input pixels and leaves unmasked pixels transparent; scale-rows draws aspect-preserved diagnostic rows."
    )]
    mask_visualization_mode: NativeMaskVisualizationMode,

    #[arg(
        long = "mask-geometry",
        alias = "mask-geometry-mode",
        alias = "mask-update-mode",
        alias = "mask-scale-policy",
        value_enum,
        default_value_t = DEFAULT_NATIVE_MASK_GEOMETRY_MODE,
        help = "Mask cell geometry policy. deduplicated preserves the native update union while removing fully covered overlapping cells; native draws every decoded scale cell; effective projects to the finest active grid."
    )]
    mask_geometry_mode: NativeMaskGeometryMode,

    #[arg(
        long,
        value_name = "0..1",
        value_parser = parse_alpha,
        default_value_t = DEFAULT_BLEND_ALPHA,
        help = "Alpha used when blending gaze-selected regions with the input."
    )]
    blend_alpha: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = NativeVisualizationMode::Interframe,
        help = "Output visualization. Aliases: blend, alpha-blend, delta, video."
    )]
    visualization_mode: NativeVisualizationMode,

    #[arg(
        long,
        value_name = "FRAMES",
        value_parser = parse_usize,
        default_value_t = DEFAULT_BIRDS_KEYFRAME_DURATION,
        help = "Interframe mode periodic keyframe interval. 0 disables periodic keyframes; the first frame and dimension changes still reset state."
    )]
    keyframe_duration: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = NativeDisplayTransfer::Auto,
        help = "Display transfer path. auto uses the fastest measured path for the frame size; gpu forces Bevy/Burn shared-device texture interop; cpu writes u8 RGBA Bevy images."
    )]
    display_transfer: NativeDisplayTransfer,

    #[arg(
        long,
        alias = "sparse-update-max-rects",
        value_name = "COUNT",
        value_parser = parse_usize,
        default_value_t = bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
        help = "Maximum rect count for tensor-side sparse interframe updates. Use 0 to force dense tensor updates."
    )]
    tensor_sparse_update_max_rects: usize,

    #[arg(
        long,
        alias = "sparse-update-max-ratio",
        value_name = "0..1",
        value_parser = parse_ratio_f64,
        default_value_t = bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        help = "Maximum selected-pixel ratio eligible for tensor-side sparse interframe updates."
    )]
    tensor_sparse_update_max_ratio: f64,

    #[arg(
        long,
        alias = "full-frame-update-min-ratio",
        value_name = "0..1",
        value_parser = parse_ratio_f64,
        default_value_t = bevy_burn_autogaze::DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
        help = "Selected-pixel ratio where tensor interframe output switches to a full-frame update. Use 0 to disable."
    )]
    tensor_full_frame_update_min_ratio: f64,

    #[arg(
        long,
        default_value_t = DEFAULT_BEVY_STREAMING_CACHE,
        action = ArgAction::Set,
        help = "Use the continuous rolling decoder KV cache in realtime mode so each inference advances only the newest frame. Disable for full-window comparison."
    )]
    streaming_cache: bool,

    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Exit with an error if Bevy selects a CPU/software render adapter. Use this for trustworthy native perf runs."
    )]
    require_hardware_adapter: bool,

    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Log source capture, resize/prep, pack, input upload/preprocess, model, visualization, display, and total timing periodically."
    )]
    log_pipeline_timing: bool,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        help = "Process COUNT inference outputs, print a JSON perf summary, then exit."
    )]
    perf_summary_frames: Option<usize>,

    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 0,
        help = "Ignore the first COUNT inference outputs in perf summaries/traces. Useful for excluding GPU autotune and cache-fill startup."
    )]
    perf_summary_warmup_frames: usize,

    #[arg(
        long,
        value_name = "PATH",
        requires = "perf_summary_frames",
        help = "Write the perf summary JSON to PATH in addition to logging it."
    )]
    perf_summary_path: Option<std::path::PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        requires = "perf_summary_frames",
        help = "Write one JSON object per processed inference output to PATH for frame pacing and stage timing analysis."
    )]
    perf_trace_path: Option<std::path::PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<NativeArgs> for BevyBurnAutoGazeConfig {
    fn from(args: NativeArgs) -> Self {
        let mode = BevyAutoGazeMode::from(args.mode);
        let visualization_mode = AutoGazeVisualizationMode::from(args.visualization_mode);
        let defaults = BevyBurnAutoGazeConfig::default();
        let (inference_width, inference_height) =
            inference_dimensions_for_args(mode, args.inference_width, args.inference_height);
        let (task_loss_requirement, disable_task_loss_requirement) = task_loss_config(
            args.task_loss_requirement,
            args.task_loss_requirement_db,
            args.disable_task_loss_requirement,
        );
        let top_k = args.top_k.unwrap_or_else(|| default_top_k(mode));
        let max_gaze_tokens_each_frame = args.max_gaze_tokens_each_frame.unwrap_or_else(|| {
            default_max_gaze_tokens_for_limit(mode, args.limit_generation_budget)
        });
        let mask_cell_scale = args.mask_cell_scale.unwrap_or(1.0);
        let tile_batch_size = args
            .tile_batch_size
            .unwrap_or_else(|| default_tile_batch_size(mode));
        let frames_per_clip = args
            .frames_per_clip
            .unwrap_or_else(|| default_frames_per_clip(mode));
        let decode_strategy = match AutoGazeDecodeStrategy::from(args.decode_strategy) {
            AutoGazeDecodeStrategy::HostGreedy => AutoGazeDecodeStrategy::HostGreedy,
            AutoGazeDecodeStrategy::DeviceGreedy { .. } => AutoGazeDecodeStrategy::DeviceGreedy {
                chunk_size: args.decode_chunk_size,
            },
            AutoGazeDecodeStrategy::DeviceTerminalGreedy { .. } => {
                AutoGazeDecodeStrategy::DeviceTerminalGreedy {
                    chunk_size: args.decode_chunk_size,
                }
            }
        };
        BevyBurnAutoGazeConfig {
            press_esc_to_close: args.press_esc_to_close,
            show_fps: args.show_fps,
            show_gaze_ratio: args.show_gaze_ratio,
            show_psnr: args.show_psnr,
            show_task_loss_slider: args.show_task_loss_slider,
            model_dir: args.model_dir,
            source: args.source.map(BevyFrameSource::from).unwrap_or(
                if args.image_path.is_some() {
                    BevyFrameSource::StaticImage
                } else {
                    BevyFrameSource::Camera
                },
            ),
            image_path: args.image_path,
            sparse_mask_source: args.sparse_mask_source.into(),
            patch_diff_grid_size: args.patch_diff_grid_size,
            patch_diff_threshold: args.patch_diff_threshold,
            load_model: args.load_model && !args.no_load_model,
            warmup_model: args.warmup_model,
            mode,
            top_k,
            max_gaze_tokens_each_frame,
            limit_generation_budget: args.limit_generation_budget,
            tile_batch_size,
            task_loss_requirement,
            disable_task_loss_requirement,
            frames_per_clip,
            decode_strategy,
            max_in_flight: args.max_in_flight,
            inference_width,
            inference_height: inference_height.or(defaults.inference_height),
            mask_cell_scale,
            mask_visualization_mode: args.mask_visualization_mode.into(),
            mask_geometry_mode: args.mask_geometry_mode.into(),
            blend_alpha: args.blend_alpha,
            visualization_mode,
            keyframe_duration: args.keyframe_duration,
            display_transfer: args.display_transfer.into(),
            tensor_sparse_update_max_rects: args.tensor_sparse_update_max_rects,
            tensor_sparse_update_max_ratio: args.tensor_sparse_update_max_ratio,
            tensor_full_frame_update_min_ratio: args.tensor_full_frame_update_min_ratio,
            streaming_cache: args.streaming_cache,
            require_hardware_adapter: args.require_hardware_adapter,
            log_pipeline_timing: args.log_pipeline_timing,
            perf_summary_warmup_frames: args.perf_summary_warmup_frames,
            perf_summary_frames: args.perf_summary_frames,
            perf_summary_path: args.perf_summary_path,
            perf_trace_path: args.perf_trace_path,
            ..defaults
        }
        .sanitized()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn inference_dimensions_for_args(
    mode: BevyAutoGazeMode,
    width: Option<u32>,
    height: Option<u32>,
) -> (Option<u32>, Option<u32>) {
    match (width, height) {
        (None, None) => default_inference_dimensions(mode),
        configured => configured,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn task_loss_config(
    arg: Option<TaskLossRequirementArg>,
    db: Option<f32>,
    disable: bool,
) -> (Option<f32>, bool) {
    if disable {
        return (None, true);
    }
    if let Some(db) = db {
        return (Some(task_loss_requirement_from_l1_db(f64::from(db))), false);
    }
    match arg {
        None => (Some(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT), false),
        Some(TaskLossRequirementArg::ModelDefault) => (None, false),
        Some(TaskLossRequirementArg::Disabled) => (None, true),
        Some(TaskLossRequirementArg::Value(value)) => (Some(value), false),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("expected an unsigned integer, got `{value}`"))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_nonzero_usize(value: &str) -> Result<usize, String> {
    let parsed = parse_usize(value)?;
    if parsed == 0 {
        return Err("expected a value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("expected a positive pixel count, got `{value}`"))?;
    if parsed == 0 {
        return Err("expected a value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a positive number, got `{value}`"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err("expected a finite value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_nonnegative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a non-negative number or none/off, got `{value}`"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err("expected a finite value greater than or equal to zero".to_string());
    }
    Ok(parsed)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_alpha(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a value in 0..1, got `{value}`"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("expected a finite value in 0..1".to_string());
    }
    Ok(parsed)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_ratio_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("expected a value in 0..1, got `{value}`"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("expected a finite value in 0..1".to_string());
    }
    Ok(parsed)
}

fn main() -> AppExit {
    let config = runtime_config();

    #[cfg(not(target_arch = "wasm32"))]
    {
        if config.source == BevyFrameSource::Camera {
            let request = camera_request_for_config(&config);
            std::thread::spawn(move || {
                bevy_burn_autogaze::platform::camera::native_camera_thread_with_request(request);
            });
        }
    }

    run_app(config)
}

#[cfg(not(target_arch = "wasm32"))]
fn runtime_config() -> BevyBurnAutoGazeConfig {
    NativeArgs::parse().into()
}

#[cfg(not(target_arch = "wasm32"))]
fn camera_request_for_config(
    config: &BevyBurnAutoGazeConfig,
) -> bevy_burn_autogaze::platform::camera::CameraRequest {
    match config.mode {
        BevyAutoGazeMode::Resize224 => camera_request_with_fallback(config, 640, 360),
        BevyAutoGazeMode::Tile224 => {
            camera_request_with_fallback(config, DEFAULT_TILED_INFERENCE_WIDTH, 720)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn camera_request_with_fallback(
    config: &BevyBurnAutoGazeConfig,
    fallback_width: u32,
    fallback_height: u32,
) -> bevy_burn_autogaze::platform::camera::CameraRequest {
    const DEFAULT_CAMERA_FPS: u32 = 30;
    let fallback_width = fallback_width.max(1);
    let fallback_height = fallback_height.max(1);
    let (width, height) = match (config.inference_width, config.inference_height) {
        (Some(width), Some(height)) => (width.max(1), height.max(1)),
        (Some(width), None) => {
            let width = width.max(1);
            let height = ((fallback_height as f64 * width as f64 / fallback_width as f64).round()
                as u32)
                .max(1);
            (width, height)
        }
        (None, Some(height)) => {
            let height = height.max(1);
            let width = ((fallback_width as f64 * height as f64 / fallback_height as f64).round()
                as u32)
                .max(1);
            (width, height)
        }
        (None, None) => (fallback_width, fallback_height),
    };
    bevy_burn_autogaze::platform::camera::CameraRequest::new(width, height, DEFAULT_CAMERA_FPS)
}

#[cfg(target_arch = "wasm32")]
fn runtime_config() -> BevyBurnAutoGazeConfig {
    console_error_panic_hook::set_once();

    BevyBurnAutoGazeConfig::from_browser_query()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use bevy_burn_autogaze::{
        DEFAULT_BEVY_REALTIME_FRAMES_PER_CLIP, DEFAULT_BEVY_SHOW_TASK_LOSS_SLIDER,
        DEFAULT_BEVY_TASK_LOSS_REQUIREMENT, DEFAULT_BEVY_TILED_TOP_K,
        DEFAULT_REALTIME_INFERENCE_WIDTH, DEFAULT_REALTIME_MAX_GAZE_TOKENS, DEFAULT_REALTIME_TOP_K,
        DEFAULT_TILED_FRAMES_PER_CLIP, DEFAULT_TILED_INFERENCE_WIDTH,
        DEFAULT_TILED_MAX_GAZE_TOKENS, DEFAULT_TILED_TILE_BATCH_SIZE,
    };
    use clap::CommandFactory;

    #[test]
    fn native_cli_definition_is_valid() {
        NativeArgs::command().debug_assert();
    }

    #[test]
    fn native_cli_defaults_to_realtime_640_width_without_forcing_height() {
        assert_eq!(
            inference_dimensions_for_args(BevyAutoGazeMode::Resize224, None, None),
            (Some(DEFAULT_REALTIME_INFERENCE_WIDTH), None)
        );
    }

    #[test]
    fn native_cli_uses_higher_fidelity_default_for_tiled_mode() {
        assert_eq!(
            inference_dimensions_for_args(BevyAutoGazeMode::Tile224, None, None),
            (Some(DEFAULT_TILED_INFERENCE_WIDTH), None)
        );
    }

    #[test]
    fn native_cli_defaults_use_upstream_realtime_adaptive_display_profile() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.mode, BevyAutoGazeMode::Resize224);
        assert_eq!(config.source, BevyFrameSource::Camera);
        assert_eq!(config.top_k, DEFAULT_REALTIME_TOP_K);
        assert_eq!(
            config.task_loss_requirement,
            Some(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT)
        );
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            bevy_burn_autogaze::default_max_gaze_tokens_each_frame(config.mode)
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
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Auto);
        assert_eq!(
            config.mask_geometry_mode,
            AutoGazeMaskGeometryMode::Deduplicated
        );
        assert!(config.show_psnr);
        assert!(config.show_task_loss_slider);
        assert!(config.warmup_model);
        assert_eq!(config.streaming_cache, DEFAULT_BEVY_STREAMING_CACHE);
    }

    #[test]
    fn native_cli_task_loss_default_keeps_bounded_budget_unless_full_budget_requested() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--task-loss-requirement", "0.3"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_REALTIME_MAX_GAZE_TOKENS
        );
        assert_eq!(config.mask_cell_scale, 1.0);

        let args =
            NativeArgs::parse_from(["bevy_burn_autogaze", "--task-loss-requirement", "0.45"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_REALTIME_MAX_GAZE_TOKENS
        );

        let args = NativeArgs::parse_from([
            "bevy_burn_autogaze",
            "--task-loss-requirement",
            "0.3",
            "--limit-generation-budget=false",
        ]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.max_gaze_tokens_each_frame, 0);

        let args = NativeArgs::parse_from([
            "bevy_burn_autogaze",
            "--task-loss-requirement",
            "0.3",
            "--max-gaze-tokens-each-frame",
            "12",
            "--mask-cell-scale",
            "1.25",
        ]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.max_gaze_tokens_each_frame, 12);
        assert_eq!(config.mask_cell_scale, 1.25);
    }

    #[test]
    fn native_cli_tiled_defaults_use_bounded_generation_budget() {
        let args = NativeArgs {
            press_esc_to_close: true,
            show_fps: true,
            show_gaze_ratio: true,
            show_psnr: false,
            show_task_loss_slider: DEFAULT_BEVY_SHOW_TASK_LOSS_SLIDER,
            model_dir: bevy_burn_autogaze::DEFAULT_NATIVE_MODEL_DIR.into(),
            image_path: None,
            source: None,
            sparse_mask_source: NativeSparseMaskSource::AutoGaze,
            patch_diff_grid_size: DEFAULT_PATCH_DIFF_GRID_SIZE,
            patch_diff_threshold: DEFAULT_PATCH_DIFF_THRESHOLD,
            load_model: true,
            no_load_model: false,
            warmup_model: true,
            mode: NativeInferenceMode::Tiled,
            top_k: None,
            max_gaze_tokens_each_frame: None,
            limit_generation_budget: DEFAULT_BEVY_LIMIT_GENERATION_BUDGET,
            tile_batch_size: None,
            task_loss_requirement: None,
            task_loss_requirement_db: None,
            disable_task_loss_requirement: false,
            frames_per_clip: None,
            max_in_flight: bevy_burn_autogaze::DEFAULT_MAX_IN_FLIGHT,
            inference_width: None,
            inference_height: None,
            mask_cell_scale: Some(1.0),
            mask_visualization_mode: NativeMaskVisualizationMode::ImageMaskOnly,
            mask_geometry_mode: DEFAULT_NATIVE_MASK_GEOMETRY_MODE,
            blend_alpha: DEFAULT_BLEND_ALPHA,
            visualization_mode: NativeVisualizationMode::Interframe,
            keyframe_duration: DEFAULT_BIRDS_KEYFRAME_DURATION,
            display_transfer: NativeDisplayTransfer::Auto,
            tensor_sparse_update_max_rects:
                bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            tensor_sparse_update_max_ratio:
                bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
            tensor_full_frame_update_min_ratio:
                bevy_burn_autogaze::DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
            streaming_cache: DEFAULT_BEVY_STREAMING_CACHE,
            decode_strategy: NativeDecodeStrategy::Terminal,
            decode_chunk_size: DEFAULT_BEVY_DECODE_CHUNK_SIZE,
            require_hardware_adapter: false,
            log_pipeline_timing: false,
            perf_summary_warmup_frames: 0,
            perf_summary_frames: None,
            perf_summary_path: None,
            perf_trace_path: None,
        };
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.mode, BevyAutoGazeMode::Tile224);
        assert_eq!(config.top_k, DEFAULT_BEVY_TILED_TOP_K);
        assert_eq!(
            config.max_gaze_tokens_each_frame,
            DEFAULT_TILED_MAX_GAZE_TOKENS
        );
        assert_eq!(config.tile_batch_size, DEFAULT_TILED_TILE_BATCH_SIZE);
        assert_eq!(config.frames_per_clip, DEFAULT_TILED_FRAMES_PER_CLIP);
        assert_eq!(
            config.max_in_flight,
            bevy_burn_autogaze::DEFAULT_MAX_IN_FLIGHT
        );
        assert_eq!(config.inference_width, Some(DEFAULT_TILED_INFERENCE_WIDTH));
        assert_eq!(config.inference_height, None);
        assert_eq!(config.blend_alpha, DEFAULT_BLEND_ALPHA);
        assert_eq!(config.keyframe_duration, DEFAULT_BIRDS_KEYFRAME_DURATION);
        assert_eq!(
            config.decode_strategy,
            bevy_burn_autogaze::DEFAULT_BEVY_DECODE_STRATEGY
        );
        assert_eq!(config.display_transfer, BevyDisplayTransfer::Auto);
        assert_eq!(
            config.tensor_sparse_update_max_rects,
            bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS
        );
        assert_eq!(
            config.tensor_sparse_update_max_ratio,
            bevy_burn_autogaze::DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO
        );
        assert_eq!(
            config.tensor_full_frame_update_min_ratio,
            bevy_burn_autogaze::DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO
        );

        let config = BevyBurnAutoGazeConfig::from(NativeArgs::parse_from([
            "bevy_burn_autogaze",
            "--mode",
            "tiled",
            "--limit-generation-budget=false",
        ]));
        assert_eq!(config.max_gaze_tokens_each_frame, 0);
    }

    #[test]
    fn native_cli_image_path_defaults_to_static_source() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--image-path", "frame.png"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.source, BevyFrameSource::StaticImage);
    }

    #[test]
    fn native_cli_accepts_synthetic_pan_source() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--source", "synthetic-pan"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.source, BevyFrameSource::SyntheticPan);

        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--source", "synthetic-pulse"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.source, BevyFrameSource::SyntheticPulse);

        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--source", "local-motion"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.source, BevyFrameSource::SyntheticLocalMotion);
    }

    #[test]
    fn native_cli_preserves_explicit_single_axis_resolution() {
        assert_eq!(
            inference_dimensions_for_args(BevyAutoGazeMode::Resize224, None, Some(720)),
            (None, Some(720))
        );
        assert_eq!(
            inference_dimensions_for_args(BevyAutoGazeMode::Tile224, Some(1920), None),
            (Some(1920), None)
        );
    }

    #[test]
    fn native_cli_accepts_task_loss_disable_value() {
        assert_eq!(
            task_loss_config(Some(TaskLossRequirementArg::Disabled), None, false),
            (None, true)
        );
        assert_eq!(
            task_loss_config(Some(TaskLossRequirementArg::Value(0.7)), None, false),
            (Some(0.7), false)
        );
        assert_eq!(
            task_loss_config(None, None, false),
            (Some(DEFAULT_BEVY_TASK_LOSS_REQUIREMENT), false)
        );
        assert_eq!(
            task_loss_config(Some(TaskLossRequirementArg::ModelDefault), None, false),
            (None, false)
        );
        let (threshold, disabled) = task_loss_config(None, Some(20.0), false);
        assert!(!disabled);
        assert!((threshold.expect("threshold") - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn native_cli_accepts_hardware_adapter_requirement_flag() {
        let args =
            NativeArgs::parse_from(["bevy_burn_autogaze", "--require-hardware-adapter=true"]);
        assert!(args.require_hardware_adapter);
        let config = BevyBurnAutoGazeConfig::from(args);
        assert!(config.require_hardware_adapter);
    }

    #[test]
    fn native_cli_exposes_in_flight_admission_limit() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--max-in-flight", "2"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.max_in_flight, 2);
    }

    #[test]
    fn native_cli_exposes_tensor_sparse_update_policy() {
        let args = NativeArgs::parse_from([
            "bevy_burn_autogaze",
            "--tensor-sparse-update-max-rects",
            "8",
            "--tensor-sparse-update-max-ratio",
            "0.05",
            "--tensor-full-frame-update-min-ratio",
            "0.45",
        ]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.tensor_sparse_update_max_rects, 8);
        assert_eq!(config.tensor_sparse_update_max_ratio, 0.05);
        assert_eq!(config.tensor_full_frame_update_min_ratio, 0.45);
    }

    #[test]
    fn native_cli_exposes_mask_geometry_policy() {
        let args = NativeArgs::parse_from(["bevy_burn_autogaze", "--mask-geometry", "native"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(config.mask_geometry_mode, AutoGazeMaskGeometryMode::Native);

        let args =
            NativeArgs::parse_from(["bevy_burn_autogaze", "--mask-update-mode", "effective"]);
        let config = BevyBurnAutoGazeConfig::from(args);

        assert_eq!(
            config.mask_geometry_mode,
            AutoGazeMaskGeometryMode::Effective
        );
    }
}
