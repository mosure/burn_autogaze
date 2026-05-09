#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use bevy_burn_autogaze::{
    BevyAutoGazeMode, DEFAULT_REALTIME_INFERENCE_WIDTH, DEFAULT_REALTIME_TOP_K,
};
use bevy_burn_autogaze::{BevyBurnAutoGazeConfig, run_app};
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use burn_autogaze::AutoGazeVisualizationMode;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use clap::{ArgAction, Parser, ValueEnum};
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use std::{fmt, path::PathBuf, str::FromStr};

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NativeInferenceMode {
    #[value(
        name = "realtime",
        alias = "resize",
        alias = "resize-224",
        alias = "resize-to-model",
        alias = "fast",
        help = "Resize frames before the model pass; fastest and default."
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

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl fmt::Display for NativeInferenceMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Realtime => "realtime",
            Self::Tiled => "tiled",
        })
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl From<NativeInferenceMode> for BevyAutoGazeMode {
    fn from(mode: NativeInferenceMode) -> Self {
        match mode {
            NativeInferenceMode::Realtime => Self::Resize224,
            NativeInferenceMode::Tiled => Self::Tile224,
        }
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
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

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl fmt::Display for NativeVisualizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FullBlend => "full-blend",
            Self::Interframe => "interframe",
        })
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl From<NativeVisualizationMode> for AutoGazeVisualizationMode {
    fn from(mode: NativeVisualizationMode) -> Self {
        match mode {
            NativeVisualizationMode::FullBlend => Self::FullBlend,
            NativeVisualizationMode::Interframe => Self::Interframe,
        }
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum TaskLossRequirementArg {
    ModelDefault,
    Disabled,
    Value(f32),
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
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

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[derive(Debug, Parser)]
#[command(
    about = "native Bevy viewer for burn_autogaze",
    version,
    long_about = "Runs the burn_autogaze video pipeline with camera or static-image input and renders Input | Mask | Output through Bevy. The default realtime mode requests a 640px-wide stream before the model's internal 224px pass; tiled mode is higher fidelity and slower."
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
        help = "Show inference FPS overlay."
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
        default_value_t = false,
        action = ArgAction::Set,
        help = "Show PSNR between the input frame and rendered output."
    )]
    show_psnr: bool,

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
        value_enum,
        default_value_t = NativeInferenceMode::Realtime,
        help = "Inference path. Aliases: resize-224, fast, tile-224, full-res, anyres."
    )]
    mode: NativeInferenceMode,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        default_value_t = DEFAULT_REALTIME_TOP_K,
        help = "Number of highest-scoring gaze tokens to visualize per trace."
    )]
    top_k: usize,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_usize,
        default_value_t = DEFAULT_REALTIME_TOP_K,
        help = "Model-side generated-token cap. Use 0 for the model default."
    )]
    max_gaze_tokens_each_frame: usize,

    #[arg(
        long,
        value_name = "COUNT",
        value_parser = parse_nonzero_usize,
        default_value_t = 8,
        help = "Number of 224px tiles traced together in tiled mode."
    )]
    tile_batch_size: usize,

    #[arg(
        long,
        value_name = "FLOAT|none",
        help = "Override model task-loss threshold; use none/off to disable, default/model for model config."
    )]
    task_loss_requirement: Option<TaskLossRequirementArg>,

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
        default_value_t = 2,
        help = "Number of input frames buffered per AutoGaze clip."
    )]
    frames_per_clip: usize,

    #[arg(
        long,
        alias = "width",
        value_name = "PX",
        value_parser = parse_positive_u32,
        help = "Frame width before inference. Defaults to 640 in realtime and 1280 in tiled mode. If only height is set, width preserves input aspect."
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
        default_value_t = 1.0,
        help = "Scale factor for crisp multi-scale mask cell extents."
    )]
    mask_cell_scale: f32,

    #[arg(
        long,
        value_name = "0..1",
        value_parser = parse_alpha,
        default_value_t = bevy_burn_autogaze::DEFAULT_BLEND_ALPHA,
        help = "Alpha used when blending gaze-selected regions with the input."
    )]
    blend_alpha: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = NativeVisualizationMode::FullBlend,
        help = "Output visualization. Aliases: blend, alpha-blend, delta, video."
    )]
    visualization_mode: NativeVisualizationMode,

    #[arg(
        long,
        value_name = "FRAMES",
        value_parser = parse_nonzero_usize,
        default_value_t = bevy_burn_autogaze::DEFAULT_KEYFRAME_DURATION,
        help = "Interframe mode keyframe interval."
    )]
    keyframe_duration: usize,

    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Log pack, trace, visualization, display, and total timing periodically."
    )]
    log_pipeline_timing: bool,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl From<NativeArgs> for BevyBurnAutoGazeConfig {
    fn from(args: NativeArgs) -> Self {
        let mode = BevyAutoGazeMode::from(args.mode);
        let visualization_mode = AutoGazeVisualizationMode::from(args.visualization_mode);
        let defaults = BevyBurnAutoGazeConfig::default();
        let (inference_width, inference_height) =
            inference_dimensions_for_args(mode, args.inference_width, args.inference_height);
        let (task_loss_requirement, disable_task_loss_requirement) = task_loss_config(
            args.task_loss_requirement,
            args.disable_task_loss_requirement,
        );
        Self {
            press_esc_to_close: args.press_esc_to_close,
            show_fps: args.show_fps,
            show_gaze_ratio: args.show_gaze_ratio,
            show_psnr: args.show_psnr,
            model_dir: args.model_dir,
            image_path: args.image_path,
            load_model: args.load_model && !args.no_load_model,
            mode,
            top_k: args.top_k,
            max_gaze_tokens_each_frame: args.max_gaze_tokens_each_frame,
            tile_batch_size: args.tile_batch_size,
            task_loss_requirement,
            disable_task_loss_requirement,
            frames_per_clip: args.frames_per_clip,
            inference_width,
            inference_height: inference_height.or(defaults.inference_height),
            mask_cell_scale: args.mask_cell_scale,
            blend_alpha: args.blend_alpha,
            visualization_mode,
            keyframe_duration: args.keyframe_duration,
            log_pipeline_timing: args.log_pipeline_timing,
            ..defaults
        }
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn inference_dimensions_for_args(
    mode: BevyAutoGazeMode,
    width: Option<u32>,
    height: Option<u32>,
) -> (Option<u32>, Option<u32>) {
    match (width, height) {
        (None, None) => match mode {
            BevyAutoGazeMode::Resize224 => (Some(DEFAULT_REALTIME_INFERENCE_WIDTH), None),
            BevyAutoGazeMode::Tile224 => (Some(1280), None),
        },
        configured => configured,
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn task_loss_config(arg: Option<TaskLossRequirementArg>, disable: bool) -> (Option<f32>, bool) {
    if disable {
        return (None, true);
    }
    match arg {
        Some(TaskLossRequirementArg::ModelDefault) | None => (None, false),
        Some(TaskLossRequirementArg::Disabled) => (None, true),
        Some(TaskLossRequirementArg::Value(value)) => (Some(value), false),
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("expected an unsigned integer, got `{value}`"))
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_nonzero_usize(value: &str) -> Result<usize, String> {
    let parsed = parse_usize(value)?;
    if parsed == 0 {
        return Err("expected a value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("expected a positive pixel count, got `{value}`"))?;
    if parsed == 0 {
        return Err("expected a value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a positive number, got `{value}`"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err("expected a finite value greater than zero".to_string());
    }
    Ok(parsed)
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_nonnegative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a non-negative number or none/off, got `{value}`"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err("expected a finite value greater than or equal to zero".to_string());
    }
    Ok(parsed)
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn parse_alpha(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("expected a value in 0..1, got `{value}`"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("expected a finite value in 0..1".to_string());
    }
    Ok(parsed)
}

fn main() {
    let config = runtime_config();

    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    {
        if config.image_path.is_none() {
            let request = camera_request_for_config(&config);
            std::thread::spawn(move || {
                bevy_burn_autogaze::platform::camera::native_camera_thread_with_request(request);
            });
        }
    }

    run_app(config);
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn runtime_config() -> BevyBurnAutoGazeConfig {
    NativeArgs::parse().into()
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn camera_request_for_config(
    config: &BevyBurnAutoGazeConfig,
) -> bevy_burn_autogaze::platform::camera::CameraRequest {
    match config.mode {
        BevyAutoGazeMode::Resize224 => camera_request_with_fallback(config, 640, 360),
        BevyAutoGazeMode::Tile224 => camera_request_with_fallback(config, 1280, 720),
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
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
    #[cfg(feature = "web")]
    console_error_panic_hook::set_once();

    BevyBurnAutoGazeConfig::from_browser_query()
}

#[cfg(all(test, feature = "native", not(target_arch = "wasm32")))]
mod tests {
    use super::*;

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
            (Some(1280), None)
        );
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
            task_loss_config(Some(TaskLossRequirementArg::Disabled), false),
            (None, true)
        );
        assert_eq!(
            task_loss_config(Some(TaskLossRequirementArg::Value(0.7)), false),
            (Some(0.7), false)
        );
    }
}
