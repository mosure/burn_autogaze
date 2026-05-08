use bevy_burn_autogaze::{BevyBurnAutoGazeConfig, run_app};

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
use clap::Parser;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[derive(Debug, Parser)]
#[command(about = "bevy_burn_autogaze", version, long_about = None)]
struct NativeArgs {
    #[arg(long, default_value = "true")]
    press_esc_to_close: bool,

    #[arg(long, default_value = "true")]
    show_fps: bool,

    #[arg(long, default_value = bevy_burn_autogaze::DEFAULT_NATIVE_MODEL_DIR)]
    model_dir: std::path::PathBuf,

    #[arg(long)]
    image_path: Option<std::path::PathBuf>,

    #[arg(long, default_value = "resize-224")]
    mode: String,

    #[arg(long, default_value_t = 4)]
    top_k: usize,

    #[arg(long, default_value_t = 4)]
    max_gaze_tokens_each_frame: usize,

    #[arg(long, default_value_t = 2)]
    frames_per_clip: usize,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl From<NativeArgs> for BevyBurnAutoGazeConfig {
    fn from(args: NativeArgs) -> Self {
        let mode = args.mode.parse().unwrap_or_else(|err| panic!("{err}"));
        Self {
            press_esc_to_close: args.press_esc_to_close,
            show_fps: args.show_fps,
            model_dir: args.model_dir,
            image_path: args.image_path,
            mode,
            top_k: args.top_k,
            max_gaze_tokens_each_frame: args.max_gaze_tokens_each_frame,
            frames_per_clip: args.frames_per_clip,
            ..Default::default()
        }
    }
}

fn main() {
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    {
        let args = NativeArgs::parse();
        if args.image_path.is_none() {
            std::thread::spawn(bevy_burn_autogaze::platform::camera::native_camera_thread);
        }
        run_app(args.into());
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(feature = "web")]
        console_error_panic_hook::set_once();

        let config = BevyBurnAutoGazeConfig {
            mode: bevy_burn_autogaze::BevyAutoGazeMode::Resize224,
            ..Default::default()
        };
        run_app(config);
    }
}
