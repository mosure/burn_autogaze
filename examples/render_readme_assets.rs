use anyhow::{Context, Result, bail, ensure};
use burn::tensor::backend::Backend;
use burn_autogaze::{
    AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape, AutoGazeTileLayout,
    AutoGazeVisualizationMode, AutoGazeVisualizationState, FixationPoint,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const DEFAULT_SOURCE_VIDEO: &str = "/home/mosure/Videos/birds.mp4";
const DEFAULT_MODEL_DIR: &str = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a";
const DEFAULT_OUTPUT_DIR: &str = "docs";

#[cfg(feature = "cuda")]
type DocBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
type DocBackend = burn::backend::WebGpu<f32, i32>;
#[cfg(all(not(feature = "cuda"), not(feature = "webgpu"), feature = "ndarray"))]
type DocBackend = burn::backend::NdArray<f32>;

#[cfg(not(any(feature = "cuda", feature = "webgpu", feature = "ndarray")))]
compile_error!("render_readme_assets needs one of the cuda, webgpu, or ndarray features");

fn main() -> Result<()> {
    let args = Args::parse(env::args().skip(1))?;
    let device = doc_device();
    run::<DocBackend>(args, device)
}

#[cfg(feature = "cuda")]
fn doc_device() -> burn::backend::cuda::CudaDevice {
    burn::backend::cuda::CudaDevice::default()
}

#[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
fn doc_device() -> burn::backend::wgpu::WgpuDevice {
    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
        &device,
        Default::default(),
    );
    device
}

#[cfg(all(not(feature = "cuda"), not(feature = "webgpu"), feature = "ndarray"))]
fn doc_device() -> burn::backend::ndarray::NdArrayDevice {
    Default::default()
}

#[derive(Clone, Debug)]
struct Args {
    source_video: PathBuf,
    model_dir: PathBuf,
    output_dir: PathBuf,
    inference_width: usize,
    inference_height: usize,
    gif_width: usize,
    gif_height: usize,
    fps: usize,
    frames: usize,
    clip_len: usize,
    top_k: usize,
    max_gaze_tokens_each_frame: usize,
    tile_size: usize,
    stride: usize,
    mask_cell_scale: f32,
    blend_alpha: f32,
    keyframe_duration: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            source_video: PathBuf::from(DEFAULT_SOURCE_VIDEO),
            model_dir: PathBuf::from(DEFAULT_MODEL_DIR),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            inference_width: 1920,
            inference_height: 1080,
            gif_width: 384,
            gif_height: 216,
            fps: 8,
            frames: 16,
            clip_len: 2,
            top_k: 16,
            max_gaze_tokens_each_frame: 16,
            tile_size: 224,
            stride: 224,
            mask_cell_scale: 1.0,
            blend_alpha: 0.55,
            keyframe_duration: 12,
        }
    }
}

impl Args {
    fn parse(values: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = Self::default();
        let mut values = values.peekable();
        while let Some(key) = values.next() {
            let Some(value) = values.next() else {
                bail!("missing value for {key}");
            };
            match key.as_str() {
                "--input" | "--source-video" => args.source_video = PathBuf::from(value),
                "--model-dir" => args.model_dir = PathBuf::from(value),
                "--out-dir" | "--output-dir" => args.output_dir = PathBuf::from(value),
                "--width" | "--inference-width" => {
                    args.inference_width = parse_usize(&key, &value)?;
                }
                "--height" | "--inference-height" => {
                    args.inference_height = parse_usize(&key, &value)?;
                }
                "--gif-width" | "--display-width" => args.gif_width = parse_usize(&key, &value)?,
                "--gif-height" | "--display-height" => {
                    args.gif_height = parse_usize(&key, &value)?;
                }
                "--fps" => args.fps = parse_usize(&key, &value)?,
                "--frames" => args.frames = parse_usize(&key, &value)?,
                "--clip-len" | "--frames-per-clip" => args.clip_len = parse_usize(&key, &value)?,
                "--top-k" => args.top_k = parse_usize(&key, &value)?,
                "--max-gaze-tokens-each-frame" => {
                    args.max_gaze_tokens_each_frame = parse_usize(&key, &value)?;
                }
                "--tile-size" => args.tile_size = parse_usize(&key, &value)?,
                "--stride" => args.stride = parse_usize(&key, &value)?,
                "--mask-cell-scale" => args.mask_cell_scale = parse_f32(&key, &value)?,
                "--blend-alpha" => args.blend_alpha = parse_f32(&key, &value)?,
                "--keyframe-duration" => args.keyframe_duration = parse_usize(&key, &value)?,
                other => bail!("unsupported option {other}"),
            }
        }
        ensure!(
            args.inference_width > 0 && args.inference_height > 0,
            "inference dimensions must be nonzero"
        );
        ensure!(
            args.gif_width > 0 && args.gif_height > 0,
            "gif dimensions must be nonzero"
        );
        ensure!(args.fps > 0, "fps must be nonzero");
        ensure!(args.frames > 0, "frames must be nonzero");
        ensure!(args.clip_len > 0, "clip length must be nonzero");
        ensure!(args.top_k > 0, "top-k must be nonzero");
        Ok(args)
    }
}

#[derive(Debug, Serialize)]
struct RenderMetrics {
    source_video: String,
    model_dir: String,
    backend: &'static str,
    inference_mode: String,
    visualization_mode: &'static str,
    inference_width: usize,
    inference_height: usize,
    gif_width: usize,
    gif_height: usize,
    fps: usize,
    frames: usize,
    clip_len: usize,
    top_k: usize,
    max_gaze_tokens_each_frame: usize,
    tile_count: usize,
    fixation_budget_each_frame: usize,
    scales: String,
    num_vision_tokens_each_frame: usize,
    mask_cell_scale: f32,
    blend_alpha: f32,
    keyframe_duration: usize,
    average_mask_ratio: f64,
    average_update_ratio: f64,
    final_update_ratio_ema: f64,
    min_update_ratio: f64,
    max_update_ratio: f64,
    positive_fixations: usize,
    cell_grid_histogram: BTreeMap<String, usize>,
    mask_palette: BTreeMap<String, [u8; 3]>,
    output_files: BTreeMap<&'static str, String>,
}

#[derive(Default)]
struct RatioStats {
    mask_ratios: Vec<f64>,
    update_ratios: Vec<f64>,
    update_ratio_ema: Option<f64>,
    positive_fixations: usize,
    cell_grid_histogram: BTreeMap<String, usize>,
}

fn run<B>(args: Args, device: B::Device) -> Result<()>
where
    B: Backend,
{
    ensure!(
        args.source_video.exists(),
        "source video does not exist: {}",
        args.source_video.display()
    );
    ensure!(
        args.model_dir.exists(),
        "model directory does not exist: {}",
        args.model_dir.display()
    );
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    let mode = AutoGazeInferenceMode::tiled_full_resolution(args.tile_size, args.stride);
    let mut pipeline = AutoGazePipeline::<B>::from_hf_dir(&args.model_dir, &device)
        .with_context(|| format!("load AutoGaze model from {}", args.model_dir.display()))?;
    pipeline.set_max_gaze_tokens_each_frame(args.max_gaze_tokens_each_frame);
    let config = pipeline.model().config.clone();

    let input_rgba = decode_source_video(&args)?;
    let frame_bytes = args.inference_width * args.inference_height * 4;
    ensure!(
        input_rgba.len() == frame_bytes * args.frames,
        "decoded {} bytes, expected {}",
        input_rgba.len(),
        frame_bytes * args.frames
    );

    let mut mask_rgba = Vec::with_capacity(input_rgba.len());
    let mut output_rgba = Vec::with_capacity(input_rgba.len());
    let mut state = AutoGazeVisualizationState::new(
        AutoGazeVisualizationMode::Interframe,
        args.keyframe_duration,
    );
    let mut stats = RatioStats::default();

    for frame_idx in 0..args.frames {
        let current = frame_slice(&input_rgba, frame_idx, frame_bytes);
        let clip = clip_for_frame(&input_rgba, frame_idx, args.clip_len, frame_bytes);
        let traces = pipeline.trace_rgba_clip_with_mode(
            &clip,
            AutoGazeRgbaClipShape::new(args.clip_len, args.inference_height, args.inference_width),
            args.top_k,
            mode,
            &device,
        )?;
        B::sync(&device).context("sync backend after AutoGaze trace")?;

        let points = traces
            .first()
            .and_then(|trace| trace.frames.last())
            .map(|set| set.points.clone())
            .unwrap_or_default();
        record_points(&points, &mut stats, &args);
        let visualization = state.visualize_rgba(
            current,
            args.inference_width,
            args.inference_height,
            &points,
            args.mask_cell_scale,
            args.blend_alpha,
        )?;
        let scale_mask = scale_colored_mask_rgba(
            args.inference_width,
            args.inference_height,
            &points,
            args.mask_cell_scale,
            args.tile_size,
        );
        stats.record_ratios(visualization.mask_ratio(), visualization.update_ratio());
        mask_rgba.extend_from_slice(&scale_mask);
        output_rgba.extend_from_slice(visualization.output_rgba());
        println!(
            "frame {:02}/{:02}: points={} mask={:.2}% update={:.2}%",
            frame_idx + 1,
            args.frames,
            points.iter().filter(|point| point.confidence > 0.0).count(),
            visualization.mask_ratio() * 100.0,
            visualization.update_ratio() * 100.0
        );
    }

    let target_dir = PathBuf::from("target/readme_birds");
    fs::create_dir_all(&target_dir).with_context(|| format!("create {}", target_dir.display()))?;
    let input_raw_path = target_dir.join("input.rgba");
    let mask_raw_path = target_dir.join("mask.rgba");
    let output_raw_path = target_dir.join("output.rgba");
    fs::write(&input_raw_path, &input_rgba)
        .with_context(|| format!("write {}", input_raw_path.display()))?;
    fs::write(&mask_raw_path, &mask_rgba)
        .with_context(|| format!("write {}", mask_raw_path.display()))?;
    fs::write(&output_raw_path, &output_rgba)
        .with_context(|| format!("write {}", output_raw_path.display()))?;

    let input_gif = args.output_dir.join("autogaze_birds_input.gif");
    let mask_gif = args.output_dir.join("autogaze_birds_mask.gif");
    let output_gif = args.output_dir.join("autogaze_birds_output.gif");
    encode_gif(&input_raw_path, &input_gif, &args, ScaleFilter::Lanczos)?;
    encode_gif(&mask_raw_path, &mask_gif, &args, ScaleFilter::Nearest)?;
    encode_gif(&output_raw_path, &output_gif, &args, ScaleFilter::Lanczos)?;

    let output_files = BTreeMap::from([
        ("input", display_path(&input_gif)),
        ("mask", display_path(&mask_gif)),
        ("output", display_path(&output_gif)),
    ]);
    let metrics = RenderMetrics {
        source_video: display_path(&args.source_video),
        model_dir: display_path(&args.model_dir),
        backend: backend_name(),
        inference_mode: format!("tile-{}/{}", args.tile_size, args.stride),
        visualization_mode: AutoGazeVisualizationMode::Interframe.as_str(),
        inference_width: args.inference_width,
        inference_height: args.inference_height,
        gif_width: args.gif_width,
        gif_height: args.gif_height,
        fps: args.fps,
        frames: args.frames,
        clip_len: args.clip_len,
        top_k: args.top_k,
        max_gaze_tokens_each_frame: args.max_gaze_tokens_each_frame,
        tile_count: AutoGazeTileLayout::tiled(
            args.inference_height,
            args.inference_width,
            args.tile_size,
            args.stride,
        )
        .tile_count(),
        fixation_budget_each_frame: mode.fixation_budget(
            args.top_k,
            args.inference_height,
            args.inference_width,
        ),
        scales: config.scales,
        num_vision_tokens_each_frame: config.num_vision_tokens_each_frame,
        mask_cell_scale: args.mask_cell_scale,
        blend_alpha: args.blend_alpha,
        keyframe_duration: args.keyframe_duration,
        average_mask_ratio: average(&stats.mask_ratios),
        average_update_ratio: average(&stats.update_ratios),
        final_update_ratio_ema: stats.update_ratio_ema.unwrap_or(0.0),
        min_update_ratio: stats
            .update_ratios
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(0.0),
        max_update_ratio: stats
            .update_ratios
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0),
        positive_fixations: stats.positive_fixations,
        cell_grid_histogram: stats.cell_grid_histogram,
        mask_palette: mask_palette(),
        output_files,
    };
    let metrics_path = args.output_dir.join("autogaze_birds_metrics.json");
    let mut metrics_json = serde_json::to_string_pretty(&metrics)?;
    metrics_json.push('\n');
    fs::write(&metrics_path, metrics_json)
        .with_context(|| format!("write {}", metrics_path.display()))?;
    println!("wrote {}", input_gif.display());
    println!("wrote {}", mask_gif.display());
    println!("wrote {}", output_gif.display());
    println!("wrote {}", metrics_path.display());
    Ok(())
}

impl RatioStats {
    fn record_ratios(&mut self, mask_ratio: f64, update_ratio: f64) {
        self.mask_ratios.push(mask_ratio);
        self.update_ratios.push(update_ratio);
        self.update_ratio_ema = Some(match self.update_ratio_ema {
            Some(previous) => previous * 0.85 + update_ratio * 0.15,
            None => update_ratio,
        });
    }
}

fn decode_source_video(args: &Args) -> Result<Vec<u8>> {
    let filter = format!(
        "fps={},scale={}x{}:flags=lanczos",
        args.fps, args.inference_width, args.inference_height
    );
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &display_path(&args.source_video),
            "-vf",
            &filter,
            "-frames:v",
            &args.frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .output()
        .context("run ffmpeg video decode")?;
    ensure!(
        output.status.success(),
        "ffmpeg decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output.stdout)
}

#[derive(Clone, Copy, Debug)]
enum ScaleFilter {
    Lanczos,
    Nearest,
}

impl ScaleFilter {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Lanczos => "lanczos",
            Self::Nearest => "neighbor",
        }
    }
}

fn encode_gif(
    raw_path: &Path,
    gif_path: &Path,
    args: &Args,
    scale_filter: ScaleFilter,
) -> Result<()> {
    let size = format!("{}x{}", args.inference_width, args.inference_height);
    let filter = format!(
        "scale={}x{}:flags={},split[s0][s1];[s0]palettegen=max_colors=128:reserve_transparent=0[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        args.gif_width,
        args.gif_height,
        scale_filter.as_str()
    );
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s:v",
            &size,
            "-r",
            &args.fps.to_string(),
            "-i",
            &display_path(raw_path),
            "-vf",
            &filter,
            "-loop",
            "0",
            &display_path(gif_path),
        ])
        .status()
        .with_context(|| format!("run ffmpeg GIF encode for {}", gif_path.display()))?;
    ensure!(
        status.success(),
        "ffmpeg GIF encode failed for {}",
        gif_path.display()
    );
    Ok(())
}

fn clip_for_frame(frames: &[u8], frame_idx: usize, clip_len: usize, frame_bytes: usize) -> Vec<u8> {
    let mut clip = Vec::with_capacity(clip_len * frame_bytes);
    let missing = clip_len.saturating_sub(frame_idx + 1);
    let start = (frame_idx + 1).saturating_sub(clip_len);
    for clip_idx in 0..clip_len {
        let source = if clip_idx < missing {
            0
        } else {
            start + clip_idx - missing
        };
        clip.extend_from_slice(frame_slice(frames, source, frame_bytes));
    }
    clip
}

fn frame_slice(frames: &[u8], frame_idx: usize, frame_bytes: usize) -> &[u8] {
    let start = frame_idx * frame_bytes;
    &frames[start..start + frame_bytes]
}

fn record_points(points: &[FixationPoint], stats: &mut RatioStats, args: &Args) {
    for point in points.iter().filter(|point| point.confidence > 0.0) {
        stats.positive_fixations += 1;
        let grid = point_grid(
            *point,
            args.inference_width,
            args.inference_height,
            args.tile_size,
        );
        *stats
            .cell_grid_histogram
            .entry(format!("{grid}x{grid}"))
            .or_default() += 1;
    }
}

fn scale_colored_mask_rgba(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    tile_size: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 4];
    for pixel in out.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let mut ordered = points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .cell_width()
            .total_cmp(&left.cell_width())
            .then_with(|| right.cell_height().total_cmp(&left.cell_height()))
    });

    for point in ordered {
        let color = color_for_point(point, width, height, tile_size);
        let alpha = (0.45 + 0.40 * point.confidence).clamp(0.0, 0.88);
        let bounds = point.scaled_bounds(cell_scale);
        let (x0, x1) = pixel_range(bounds.x_min, bounds.x_max, width);
        let (y0, y1) = pixel_range(bounds.y_min, bounds.y_max, height);
        for y in y0..y1 {
            for x in x0..x1 {
                let offset = (y * width + x) * 4;
                for channel in 0..3 {
                    let base = out[offset + channel] as f32;
                    out[offset + channel] =
                        (base * (1.0 - alpha) + color[channel] as f32 * alpha).round() as u8;
                }
            }
        }
    }
    out
}

fn color_for_point(point: FixationPoint, width: usize, height: usize, tile_size: usize) -> [u8; 3] {
    match point_grid(point, width, height, tile_size) {
        0..=2 => [255, 180, 0],
        3..=4 => [60, 220, 120],
        5..=7 => [0, 185, 255],
        _ => [230, 110, 255],
    }
}

fn mask_palette() -> BTreeMap<String, [u8; 3]> {
    BTreeMap::from([
        ("2x2".to_string(), [255, 180, 0]),
        ("4x4".to_string(), [60, 220, 120]),
        ("7x7".to_string(), [0, 185, 255]),
        ("14x14".to_string(), [230, 110, 255]),
    ])
}

fn point_grid(point: FixationPoint, width: usize, height: usize, tile_size: usize) -> usize {
    point
        .cell_grid()
        .unwrap_or_else(|| grid_for_point_extent(point, width, height, tile_size))
}

fn grid_for_point_extent(
    point: FixationPoint,
    width: usize,
    height: usize,
    tile_size: usize,
) -> usize {
    let cell_width_px = point.cell_width() * width.max(1) as f32;
    let cell_height_px = point.cell_height() * height.max(1) as f32;
    let cell_px = cell_width_px.max(cell_height_px).max(1.0);
    let recovered = tile_size.max(1) as f32 / cell_px;
    nearest_grid(recovered)
}

fn nearest_grid(value: f32) -> usize {
    [2usize, 4, 7, 14]
        .into_iter()
        .min_by(|left, right| {
            ((*left as f32 - value).abs()).total_cmp(&(*right as f32 - value).abs())
        })
        .unwrap_or(14)
}

fn pixel_range(min: f32, max: f32, extent: usize) -> (usize, usize) {
    let extent_f = extent as f32;
    let mut start = (min.clamp(0.0, 1.0) * extent_f).floor() as usize;
    let mut end = (max.clamp(0.0, 1.0) * extent_f).ceil() as usize;
    start = start.min(extent.saturating_sub(1));
    end = end.min(extent);
    if end <= start {
        end = (start + 1).min(extent);
    }
    (start, end)
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn parse_usize(key: &str, value: &str) -> Result<usize> {
    value
        .parse()
        .with_context(|| format!("parse {key} value `{value}` as usize"))
}

fn parse_f32(key: &str, value: &str) -> Result<f32> {
    value
        .parse()
        .with_context(|| format!("parse {key} value `{value}` as f32"))
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(feature = "cuda")]
const fn backend_name() -> &'static str {
    "cuda"
}

#[cfg(all(not(feature = "cuda"), feature = "webgpu"))]
const fn backend_name() -> &'static str {
    "webgpu"
}

#[cfg(all(not(feature = "cuda"), not(feature = "webgpu"), feature = "ndarray"))]
const fn backend_name() -> &'static str {
    "ndarray"
}
