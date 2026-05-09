use bevy_burn_autogaze::{BevyBurnAutoGazeConfig, run_app};

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use clap::{ArgAction, Parser};

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[derive(Debug, Parser)]
#[command(about = "bevy_burn_autogaze", version, long_about = None)]
struct NativeArgs {
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    press_esc_to_close: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_fps: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_gaze_ratio: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_psnr: bool,

    #[arg(long, default_value = bevy_burn_autogaze::DEFAULT_NATIVE_MODEL_DIR)]
    model_dir: std::path::PathBuf,

    #[arg(long)]
    image_path: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    load_model: bool,

    #[arg(long, default_value = "resize-224")]
    mode: String,

    #[arg(long, default_value_t = 4)]
    top_k: usize,

    #[arg(long, default_value_t = 0)]
    max_gaze_tokens_each_frame: usize,

    #[arg(long, default_value_t = 8)]
    tile_batch_size: usize,

    #[arg(long)]
    task_loss_requirement: Option<f32>,

    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    disable_task_loss_requirement: bool,

    #[arg(long, default_value_t = 2)]
    frames_per_clip: usize,

    #[arg(long, alias = "width")]
    inference_width: Option<u32>,

    #[arg(long, alias = "height")]
    inference_height: Option<u32>,

    #[arg(
        long = "mask-cell-scale",
        alias = "mask-radius-scale",
        default_value_t = 1.0
    )]
    mask_cell_scale: f32,

    #[arg(long, default_value_t = 0.72)]
    blend_alpha: f32,

    #[arg(long, default_value = "full-blend")]
    visualization_mode: String,

    #[arg(long, default_value_t = bevy_burn_autogaze::DEFAULT_KEYFRAME_DURATION)]
    keyframe_duration: usize,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl From<NativeArgs> for BevyBurnAutoGazeConfig {
    fn from(args: NativeArgs) -> Self {
        let mode = args.mode.parse().unwrap_or_else(|err| panic!("{err}"));
        let visualization_mode = args
            .visualization_mode
            .parse()
            .unwrap_or_else(|err| panic!("{err}"));
        Self {
            press_esc_to_close: args.press_esc_to_close,
            show_fps: args.show_fps,
            show_gaze_ratio: args.show_gaze_ratio,
            show_psnr: args.show_psnr,
            model_dir: args.model_dir,
            image_path: args.image_path,
            load_model: args.load_model,
            mode,
            top_k: args.top_k,
            max_gaze_tokens_each_frame: args.max_gaze_tokens_each_frame,
            tile_batch_size: args.tile_batch_size.max(1),
            task_loss_requirement: args.task_loss_requirement,
            disable_task_loss_requirement: args.disable_task_loss_requirement,
            frames_per_clip: args.frames_per_clip,
            inference_width: args.inference_width,
            inference_height: args.inference_height,
            mask_cell_scale: args.mask_cell_scale,
            blend_alpha: args.blend_alpha,
            visualization_mode,
            keyframe_duration: args.keyframe_duration.max(1),
            ..Default::default()
        }
    }
}

fn main() {
    let config = runtime_config();

    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    {
        if config.image_path.is_none() {
            std::thread::spawn(bevy_burn_autogaze::platform::camera::native_camera_thread);
        }
    }

    run_app(config);
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
fn runtime_config() -> BevyBurnAutoGazeConfig {
    NativeArgs::parse().into()
}

#[cfg(target_arch = "wasm32")]
fn runtime_config() -> BevyBurnAutoGazeConfig {
    #[cfg(feature = "web")]
    console_error_panic_hook::set_once();

    BevyBurnAutoGazeConfig::from_browser_query()
}
