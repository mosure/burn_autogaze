use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape,
    AutoGazeVisualizationMode, AutoGazeVisualizationState, ConnectorConfig, FixationPoint,
    GazeDecoderConfig, GazeModelConfig, NativeAutoGazeModel, VisionModelConfig,
};
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "webgpu")]
use std::sync::OnceLock;
use std::{
    env,
    hint::black_box,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Copy, Debug)]
struct VideoCase {
    name: &'static str,
    width: usize,
    height: usize,
}

impl VideoCase {
    const fn frames_per_batch(&self) -> u64 {
        (BATCH * FRAMES) as u64
    }
}

#[derive(Clone, Copy, Debug)]
struct ModeCase {
    name: &'static str,
    mode: AutoGazeInferenceMode,
}

#[derive(Clone, Copy, Debug)]
struct ModelCase {
    name: &'static str,
    scales: &'static str,
    num_vision_tokens_each_frame: usize,
}

#[derive(Clone, Copy, Debug)]
struct VisualizationCase {
    name: &'static str,
    mode: AutoGazeVisualizationMode,
    force_delta_frame: bool,
}

const VIDEO_CASES: &[VideoCase] = &[
    VideoCase {
        name: "720p",
        width: 1280,
        height: 720,
    },
    VideoCase {
        name: "1080p",
        width: 1920,
        height: 1080,
    },
];
const MODE_CASES: &[ModeCase] = &[
    ModeCase {
        name: "resize-224",
        mode: AutoGazeInferenceMode::ResizeToModelInput,
    },
    ModeCase {
        name: "tile-224",
        mode: AutoGazeInferenceMode::TiledFullResolution {
            tile_size: MODEL_INPUT_SIZE,
            stride: MODEL_INPUT_SIZE,
        },
    },
];
const MODEL_CASES: &[ModelCase] = &[
    ModelCase {
        name: "single-scale-224",
        scales: "224",
        num_vision_tokens_each_frame: CONNECTOR_TOKENS,
    },
    ModelCase {
        name: "multiscale-32-64-112-224",
        scales: "32+64+112+224",
        num_vision_tokens_each_frame: 265,
    },
];
const VISUALIZATION_CASES: &[VisualizationCase] = &[
    VisualizationCase {
        name: "full-blend",
        mode: AutoGazeVisualizationMode::FullBlend,
        force_delta_frame: false,
    },
    VisualizationCase {
        name: "interframe-keyframe",
        mode: AutoGazeVisualizationMode::Interframe,
        force_delta_frame: false,
    },
    VisualizationCase {
        name: "interframe-delta",
        mode: AutoGazeVisualizationMode::Interframe,
        force_delta_frame: true,
    },
];
const MODEL_INPUT_SIZE: usize = 224;
const PATCH_SIZE: usize = 16;
const MODEL_GRID: usize = MODEL_INPUT_SIZE / PATCH_SIZE;
const CONNECTOR_TOKENS: usize = MODEL_GRID * MODEL_GRID;
const BATCH: usize = 1;
const FRAMES: usize = 2;
const CHANNELS: usize = 3;
const REAL_TOP_K: usize = 4;
const BLEND_ALPHA: f32 = 0.55;
const KEYFRAME_DURATION: usize = 30;

fn bench_embed_video(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_embed_video");
    group.sample_size(10);
    register_ndarray_embed(&mut group);

    #[cfg(feature = "webgpu")]
    register_webgpu_embed(&mut group);

    #[cfg(feature = "cuda")]
    register_cuda_embed(&mut group);

    group.finish();
}

fn bench_trace_video(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_trace_video");
    group.sample_size(10);
    register_ndarray_trace(&mut group);

    #[cfg(feature = "webgpu")]
    register_webgpu_trace(&mut group);

    #[cfg(feature = "cuda")]
    register_cuda_trace(&mut group);

    group.finish();
}

fn bench_real_trace_video(c: &mut Criterion) {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!(
            "skipping real AutoGaze benchmarks: set AUTOGAZE_HF_DIR to a Hugging Face snapshot"
        );
        return;
    };

    let mut group = c.benchmark_group("autogaze_real_trace_video");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    #[cfg(feature = "webgpu")]
    register_webgpu_real_trace(&mut group, &hf_dir);

    #[cfg(feature = "cuda")]
    register_cuda_real_trace(&mut group, &hf_dir);

    group.finish();
}

fn bench_rgba_e2e_video(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_rgba_e2e_video");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    register_ndarray_rgba_e2e(&mut group);

    #[cfg(feature = "webgpu")]
    register_webgpu_rgba_e2e(&mut group);

    #[cfg(feature = "cuda")]
    register_cuda_rgba_e2e(&mut group);

    group.finish();
}

fn bench_visualization(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_visualization");
    group.sample_size(10);

    for &case in VIDEO_CASES {
        group.throughput(Throughput::Elements((case.width * case.height) as u64));
        for &model in MODEL_CASES {
            for &visualization in VISUALIZATION_CASES {
                bench_visualization_case(&mut group, case, model, visualization);
            }
        }
    }

    group.finish();
}

#[cfg(feature = "ndarray")]
fn register_ndarray_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                let device = Default::default();
                bench_embed_case::<burn::backend::NdArray<f32>>(
                    group, "ndarray", case, model, mode, device,
                );
            }
        }
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_embed(_group: &mut BenchmarkGroup<'_, WallTime>) {}

#[cfg(feature = "ndarray")]
fn register_ndarray_trace(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                let device = Default::default();
                bench_trace_case::<burn::backend::NdArray<f32>>(
                    group, "ndarray", case, model, mode, device,
                );
            }
        }
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_trace(_group: &mut BenchmarkGroup<'_, WallTime>) {}

#[cfg(feature = "ndarray")]
fn register_ndarray_rgba_e2e(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                let device = Default::default();
                bench_rgba_e2e_case::<burn::backend::NdArray<f32>>(
                    group, "ndarray", case, model, mode, device,
                );
            }
        }
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_rgba_e2e(_group: &mut BenchmarkGroup<'_, WallTime>) {}

#[cfg(feature = "webgpu")]
fn register_webgpu_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let Some(device) = webgpu_device() else {
        return;
    };
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::WebGpu<f32, i32>>(
                    "webgpu",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_embed_case::<burn::backend::WebGpu<f32, i32>>(
                        group, "webgpu", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "webgpu")]
fn register_webgpu_trace(group: &mut BenchmarkGroup<'_, WallTime>) {
    let Some(device) = webgpu_device() else {
        return;
    };
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::WebGpu<f32, i32>>(
                    "webgpu",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_trace_case::<burn::backend::WebGpu<f32, i32>>(
                        group, "webgpu", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "webgpu")]
fn register_webgpu_rgba_e2e(group: &mut BenchmarkGroup<'_, WallTime>) {
    let Some(device) = webgpu_device() else {
        return;
    };
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::WebGpu<f32, i32>>(
                    "webgpu",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_rgba_e2e_case::<burn::backend::WebGpu<f32, i32>>(
                        group, "webgpu", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn register_cuda_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::Cuda<f32, i32>>(
                    "cuda",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_embed_case::<burn::backend::Cuda<f32, i32>>(
                        group, "cuda", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn register_cuda_trace(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::Cuda<f32, i32>>(
                    "cuda",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_trace_case::<burn::backend::Cuda<f32, i32>>(
                        group, "cuda", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn register_cuda_rgba_e2e(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        for &model in MODEL_CASES {
            for &mode in MODE_CASES {
                if let Some(device) = warm_backend::<burn::backend::Cuda<f32, i32>>(
                    "cuda",
                    case,
                    model,
                    mode,
                    device.clone(),
                ) {
                    bench_rgba_e2e_case::<burn::backend::Cuda<f32, i32>>(
                        group, "cuda", case, model, mode, device,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "webgpu")]
fn register_webgpu_real_trace(group: &mut BenchmarkGroup<'_, WallTime>, hf_dir: &Path) {
    let Some(device) = webgpu_device() else {
        return;
    };
    register_real_trace::<burn::backend::WebGpu<f32, i32>>(group, "webgpu", hf_dir, device);
}

#[cfg(feature = "cuda")]
fn register_cuda_real_trace(group: &mut BenchmarkGroup<'_, WallTime>, hf_dir: &Path) {
    let device = burn::backend::cuda::CudaDevice::default();
    register_real_trace::<burn::backend::Cuda<f32, i32>>(group, "cuda", hf_dir, device);
}

fn register_real_trace<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    hf_dir: &Path,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let Some(pipeline) = run_optional(backend, || {
        AutoGazePipeline::<B>::from_hf_dir(hf_dir, &device)
            .expect("load real AutoGaze model")
            .with_max_gaze_tokens_each_frame(REAL_TOP_K)
    }) else {
        return;
    };

    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            bench_real_trace_case::<B>(
                group,
                backend,
                case,
                mode,
                pipeline.clone(),
                device.clone(),
            );
        }
    }
}

fn bench_embed_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: ModeCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline = deterministic_pipeline::<B>(model, &device);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}/{}", model.name, mode.name), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    let output = pipeline.embed_video_with_mode(video, mode.mode);
                    black_box(output.layout.tile_count());
                    black_box(output.embeddings.into_data());
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_trace_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: ModeCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline = deterministic_pipeline::<B>(model, &device);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}/{}", model.name, mode.name), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    black_box(pipeline.trace_video_with_mode(video, 2, mode.mode));
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_visualization_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: VideoCase,
    model: ModelCase,
    visualization: VisualizationCase,
) {
    let rgba = deterministic_rgba(FRAMES, case.height, case.width);
    let previous_rgba = deterministic_rgba_frame(case.height, case.width, 17);
    let current_rgba = &rgba[rgba.len() - case.height * case.width * 4..];
    let points = deterministic_fixations(model);
    group.bench_with_input(
        BenchmarkId::new(format!("{}/{}", model.name, visualization.name), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state = AutoGazeVisualizationState::new(
                        visualization.mode,
                        if visualization.force_delta_frame {
                            usize::MAX
                        } else {
                            KEYFRAME_DURATION
                        },
                    );
                    if visualization.force_delta_frame {
                        state
                            .visualize_rgba(
                                &previous_rgba,
                                case.width,
                                case.height,
                                &points,
                                1.0,
                                BLEND_ALPHA,
                            )
                            .expect("prime interframe visualization state");
                    }
                    state
                },
                |mut state| {
                    let output = state
                        .visualize_rgba(
                            current_rgba,
                            case.width,
                            case.height,
                            &points,
                            1.0,
                            BLEND_ALPHA,
                        )
                        .expect("visualize autogaze mask");
                    black_box(output.mask_pixel_count);
                    black_box(output.updated_pixel_count);
                    black_box(output.update_ratio());
                    black_box(output.side_by_side_rgba.len());
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_rgba_e2e_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: ModeCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline = deterministic_pipeline::<B>(model, &device);
    let rgba = deterministic_rgba(FRAMES, case.height, case.width);
    let current_frame_start = (FRAMES - 1) * case.height * case.width * 4;
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}/{}", model.name, mode.name), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 1),
                |mut state| {
                    let traces = pipeline
                        .trace_rgba_clip_with_mode(
                            &rgba,
                            AutoGazeRgbaClipShape::new(FRAMES, case.height, case.width),
                            2,
                            mode.mode,
                            &device,
                        )
                        .expect("trace RGBA clip")
                        .first()
                        .and_then(|trace| trace.frames.last())
                        .map(|set| set.points.clone())
                        .unwrap_or_default();
                    let output = state
                        .visualize_rgba(
                            &rgba[current_frame_start..],
                            case.width,
                            case.height,
                            &traces,
                            1.0,
                            BLEND_ALPHA,
                        )
                        .expect("visualize e2e autogaze output");
                    black_box(output.update_ratio());
                    black_box(output.side_by_side_rgba.len());
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_real_trace_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    mode: ModeCase,
    pipeline: AutoGazePipeline<B>,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}", mode.name), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    black_box(pipeline.trace_video_with_mode(video, REAL_TOP_K, mode.mode));
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn warm_backend<B: Backend>(
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: ModeCase,
    device: B::Device,
) -> Option<B::Device>
where
    B::Device: Clone,
{
    run_optional(backend, || {
        let pipeline = deterministic_pipeline::<B>(model, &device);
        let video =
            deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
        let output = pipeline.embed_video_with_mode(video, mode.mode);
        black_box(output.embeddings.into_data());
        B::sync(&device).expect("backend sync");
        device
    })
}

#[cfg(feature = "webgpu")]
fn webgpu_device() -> Option<burn::backend::wgpu::WgpuDevice> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    let device = burn::backend::wgpu::WgpuDevice::default();
    match INIT.get_or_init(|| {
        match panic::catch_unwind(AssertUnwindSafe(|| {
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                &device,
                Default::default(),
            );
        })) {
            Ok(()) => Ok(()),
            Err(payload) => {
                let reason = panic_payload_to_string(payload);
                if reason.to_ascii_lowercase().contains("already initialized") {
                    Ok(())
                } else {
                    Err(reason)
                }
            }
        }
    }) {
        Ok(()) => Some(device),
        Err(reason) if is_unavailable_backend_reason(reason) => {
            eprintln!("skipping webgpu benchmark: {reason}");
            None
        }
        Err(reason) => panic!("webgpu benchmark initialization failed: {reason}"),
    }
}

fn deterministic_pipeline<B: Backend>(model: ModelCase, device: &B::Device) -> AutoGazePipeline<B> {
    let config = tiny_config(model);
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::new(&config, device).map(&mut mapper);
    AutoGazePipeline::new(model).with_max_gaze_tokens_each_frame(2)
}

fn tiny_config(model: ModelCase) -> AutoGazeConfig {
    let hidden = 8;
    let heads = 2;
    let vocab_size = model.num_vision_tokens_each_frame + 1;
    AutoGazeConfig {
        scales: model.scales.to_string(),
        max_num_frames: FRAMES,
        num_vision_tokens_each_frame: model.num_vision_tokens_each_frame,
        gaze_model_config: GazeModelConfig {
            input_img_size: MODEL_INPUT_SIZE,
            num_vision_tokens_each_frame: model.num_vision_tokens_each_frame,
            attn_mode: "sdpa".to_string(),
            vision_model_config: VisionModelConfig {
                hidden_dim: hidden,
                out_dim: hidden,
                depth: 1,
                kernel_size: PATCH_SIZE,
                temporal_patch_size: 1,
                trunk_temporal_kernel_size: 3,
                trunk_spatial_kernel_size: 1,
            },
            connector_config: ConnectorConfig {
                hidden_dim: hidden,
                num_tokens: CONNECTOR_TOKENS,
            },
            gaze_decoder_config: GazeDecoderConfig {
                vocab_size,
                hidden_size: hidden,
                intermediate_size: hidden * 2,
                num_hidden_layers: 1,
                num_attention_heads: heads,
                num_key_value_heads: heads,
                max_position_embeddings: 512,
                bos_token_id: 0,
                eos_token_id: model.num_vision_tokens_each_frame as i64,
                head_dim: hidden / heads,
                num_multi_token_pred: 2,
                ..GazeDecoderConfig::default()
            },
        },
        ..AutoGazeConfig::default()
    }
}

fn deterministic_fixations(model: ModelCase) -> Vec<FixationPoint> {
    if model.scales.contains('+') {
        vec![
            FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 0.98),
            FixationPoint::with_extent(0.625, 0.125, 0.25, 0.25, 0.91),
            FixationPoint::with_extent(3.5 / 7.0, 5.5 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 0.84),
            FixationPoint::with_extent(11.5 / 14.0, 8.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.77),
        ]
    } else {
        vec![
            FixationPoint::with_extent(3.5 / 14.0, 4.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.95),
            FixationPoint::with_extent(10.5 / 14.0, 7.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.82),
            FixationPoint::with_extent(6.5 / 14.0, 11.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.71),
        ]
    }
}

fn deterministic_video<B: Backend>(
    batch: usize,
    frames: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    let len = batch * frames * channels * height * width;
    let values = (0..len)
        .map(|idx| ((idx % 251) as f32 / 125.0) - 1.0)
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [batch, frames, channels, height, width]),
        device,
    )
}

fn deterministic_rgba(frames: usize, height: usize, width: usize) -> Vec<u8> {
    (0..frames)
        .flat_map(|frame| deterministic_rgba_frame(height, width, frame))
        .collect()
}

fn deterministic_rgba_frame(height: usize, width: usize, frame: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(height * width * 4);
    for y in 0..height {
        for x in 0..width {
            rgba.push(((x + frame * 13) % 256) as u8);
            rgba.push(((y + frame * 29) % 256) as u8);
            rgba.push(((x + y + frame * 7) % 256) as u8);
            rgba.push(255);
        }
    }
    rgba
}

struct DeterministicParamMapper {
    cursor: usize,
}

impl<B: Backend> ModuleMapper<B> for DeterministicParamMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let tensor = param.val();
        let shape = tensor.shape().dims::<D>();
        let device = tensor.device();
        let len = shape.iter().product::<usize>();
        let start = self.cursor;
        self.cursor += len;
        let values = (0..len)
            .map(|idx| (((start + idx) % 97) as f32 - 48.0) * 0.002)
            .collect::<Vec<_>>();
        Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), &device))
    }
}

fn run_optional<T>(name: &str, test: impl FnOnce() -> T) -> Option<T> {
    match panic::catch_unwind(AssertUnwindSafe(test)) {
        Ok(value) => Some(value),
        Err(payload) => {
            let reason = panic_payload_to_string(payload);
            if is_unavailable_backend_reason(&reason) {
                eprintln!("skipping {name} benchmark: {reason}");
                None
            } else {
                panic!("{name} benchmark setup failed: {reason}");
            }
        }
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).into(),
            Err(_) => "panic without string payload".into(),
        },
    }
}

fn is_unavailable_backend_reason(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    [
        "no adapter",
        "no possible adapter",
        "no suitable adapter",
        "adapter not found",
        "backend is not available",
        "backend unavailable",
        "cuda driver",
        "driver version is insufficient",
        "failed to initialize cuda",
        "could not initialize cuda",
        "libcuda",
        "not supported on this system",
        "webgpu",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn real_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("AUTOGAZE_HF_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "AUTOGAZE_HF_DIR does not exist, skipping real AutoGaze benchmarks: {}",
            path.display()
        );
        return None;
    }

    let default = PathBuf::from(
        "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a",
    );
    default.exists().then_some(default)
}

criterion_group!(
    benches,
    bench_embed_video,
    bench_trace_video,
    bench_real_trace_video,
    bench_rgba_e2e_video,
    bench_visualization
);
criterion_main!(benches);
