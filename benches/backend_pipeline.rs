use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazePipeline, ConnectorConfig, GazeDecoderConfig,
    GazeModelConfig, NativeAutoGazeModel, VisionModelConfig,
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
const MODEL_INPUT_SIZE: usize = 224;
const BATCH: usize = 1;
const FRAMES: usize = 2;
const CHANNELS: usize = 3;
const REAL_TOP_K: usize = 4;

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

#[cfg(feature = "ndarray")]
fn register_ndarray_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            let device = Default::default();
            bench_embed_case::<burn::backend::NdArray<f32>>(group, "ndarray", case, mode, device);
        }
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_embed(_group: &mut BenchmarkGroup<'_, WallTime>) {}

#[cfg(feature = "ndarray")]
fn register_ndarray_trace(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            let device = Default::default();
            bench_trace_case::<burn::backend::NdArray<f32>>(group, "ndarray", case, mode, device);
        }
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_trace(_group: &mut BenchmarkGroup<'_, WallTime>) {}

#[cfg(feature = "webgpu")]
fn register_webgpu_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let Some(device) = webgpu_device() else {
        return;
    };
    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            if let Some(device) = warm_backend::<burn::backend::WebGpu<f32, i32>>(
                "webgpu",
                case,
                mode,
                device.clone(),
            ) {
                bench_embed_case::<burn::backend::WebGpu<f32, i32>>(
                    group, "webgpu", case, mode, device,
                );
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
        for &mode in MODE_CASES {
            if let Some(device) = warm_backend::<burn::backend::WebGpu<f32, i32>>(
                "webgpu",
                case,
                mode,
                device.clone(),
            ) {
                bench_trace_case::<burn::backend::WebGpu<f32, i32>>(
                    group, "webgpu", case, mode, device,
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn register_cuda_embed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            if let Some(device) =
                warm_backend::<burn::backend::Cuda<f32, i32>>("cuda", case, mode, device.clone())
            {
                bench_embed_case::<burn::backend::Cuda<f32, i32>>(
                    group, "cuda", case, mode, device,
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn register_cuda_trace(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            if let Some(device) =
                warm_backend::<burn::backend::Cuda<f32, i32>>("cuda", case, mode, device.clone())
            {
                bench_trace_case::<burn::backend::Cuda<f32, i32>>(
                    group, "cuda", case, mode, device,
                );
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
    mode: ModeCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline = deterministic_pipeline::<B>(&device);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}", mode.name), case.name),
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
    mode: ModeCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline = deterministic_pipeline::<B>(&device);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}/{}", mode.name), case.name),
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
    mode: ModeCase,
    device: B::Device,
) -> Option<B::Device>
where
    B::Device: Clone,
{
    run_optional(backend, || {
        let pipeline = deterministic_pipeline::<B>(&device);
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

fn deterministic_pipeline<B: Backend>(device: &B::Device) -> AutoGazePipeline<B> {
    let config = tiny_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::new(&config, device).map(&mut mapper);
    AutoGazePipeline::new(model).with_max_gaze_tokens_each_frame(2)
}

fn tiny_config() -> AutoGazeConfig {
    let kernel_size = 16;
    let grid = MODEL_INPUT_SIZE / kernel_size;
    let tokens = grid * grid;
    let hidden = 8;
    let heads = 2;
    AutoGazeConfig {
        scales: MODEL_INPUT_SIZE.to_string(),
        max_num_frames: FRAMES,
        num_vision_tokens_each_frame: tokens,
        gaze_model_config: GazeModelConfig {
            input_img_size: MODEL_INPUT_SIZE,
            num_vision_tokens_each_frame: tokens,
            attn_mode: "sdpa".to_string(),
            vision_model_config: VisionModelConfig {
                hidden_dim: hidden,
                out_dim: hidden,
                depth: 1,
                kernel_size,
                temporal_patch_size: 1,
                trunk_temporal_kernel_size: 3,
                trunk_spatial_kernel_size: 1,
            },
            connector_config: ConnectorConfig {
                hidden_dim: hidden,
                num_tokens: tokens,
            },
            gaze_decoder_config: GazeDecoderConfig {
                vocab_size: 128,
                hidden_size: hidden,
                intermediate_size: hidden * 2,
                num_hidden_layers: 1,
                num_attention_heads: heads,
                num_key_value_heads: heads,
                max_position_embeddings: 512,
                eos_token_id: 127,
                head_dim: hidden / heads,
                num_multi_token_pred: 2,
                ..GazeDecoderConfig::default()
            },
        },
        ..AutoGazeConfig::default()
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
    bench_real_trace_video
);
criterion_main!(benches);
