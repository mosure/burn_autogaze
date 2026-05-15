use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeDecodeStrategy, AutoGazeDeviceTokens, AutoGazeInferenceMode,
    AutoGazePatchDiffConfig, AutoGazePipeline, AutoGazeRgbaClipShape, AutoGazeStreamingCache,
    AutoGazeTensorVisualizationOptions, AutoGazeTensorVisualizationState,
    AutoGazeVisualizationMode, AutoGazeVisualizationState, ConnectorConfig, FixationPoint,
    GazeDecoderConfig, GazeModelConfig, NativeAutoGazeModel, SparseReadoutGrid,
    SparseReadoutOptions, SparseVideoReadoutGrid, SparseVideoReadoutOptions, VisionModelConfig,
    fixation_points_to_readout_tokens, frame_readout_tokens_to_video_coords,
    patch_diff_device_mask_async, patch_diff_readout_points, scale_token_layouts,
    video_readout_coords_to_tensor,
};
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use futures_lite::future::block_on;
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "webgpu")]
use std::sync::OnceLock;
use std::{
    env,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
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

#[derive(Clone, Copy, Debug)]
enum TensorVisualizationLayout {
    SideBySide,
    Panels,
}

impl TensorVisualizationLayout {
    const fn name(self) -> &'static str {
        match self {
            Self::SideBySide => "side-by-side",
            Self::Panels => "panels",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorUpdatePolicyCase {
    AutoFullFrame,
    NoFullFrame,
}

impl TensorUpdatePolicyCase {
    const fn name(self) -> &'static str {
        match self {
            Self::AutoFullFrame => "auto-full-frame",
            Self::NoFullFrame => "no-full-frame",
        }
    }

    fn options(
        self,
        width: usize,
        height: usize,
        blend_alpha: f32,
    ) -> AutoGazeTensorVisualizationOptions {
        let options = AutoGazeTensorVisualizationOptions::new(width, height, 1.0, blend_alpha);
        match self {
            Self::AutoFullFrame => options,
            Self::NoFullFrame => options.with_full_frame_update_policy(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorFixationCase {
    ModelDefault,
    TinySparse,
    CoarseDense,
    DenseGrid64,
    DenseGrid128,
}

impl TensorFixationCase {
    const fn name(self) -> &'static str {
        match self {
            Self::ModelDefault => "model-fixations",
            Self::TinySparse => "tiny-sparse",
            Self::CoarseDense => "coarse-dense",
            Self::DenseGrid64 => "dense-grid-64",
            Self::DenseGrid128 => "dense-grid-128",
        }
    }

    fn points(self, model: ModelCase) -> Vec<FixationPoint> {
        match self {
            Self::ModelDefault => deterministic_fixations(model),
            Self::TinySparse => vec![FixationPoint::with_grid_extent(
                0.5 / 64.0,
                0.5 / 64.0,
                1.0 / 64.0,
                1.0 / 64.0,
                1.0,
                64,
            )],
            Self::CoarseDense => vec![FixationPoint::with_grid_extent(
                0.25, 0.25, 0.5, 0.5, 1.0, 2,
            )],
            Self::DenseGrid64 => dense_grid_fixations(64),
            Self::DenseGrid128 => dense_grid_fixations(128),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorTokenRenderPath {
    PointPanels,
    DeviceTokenPanels,
}

impl TensorTokenRenderPath {
    const fn name(self) -> &'static str {
        match self {
            Self::PointPanels => "point-panels",
            Self::DeviceTokenPanels => "device-token-panels",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TensorDeviceTokenCase {
    ModelDefault,
    AllModelTokens,
}

impl TensorDeviceTokenCase {
    const fn name(self) -> &'static str {
        match self {
            Self::ModelDefault => "model-tokens",
            Self::AllModelTokens => "all-model-tokens",
        }
    }

    fn points(self, model: ModelCase) -> Vec<FixationPoint> {
        match self {
            Self::ModelDefault => deterministic_fixations(model),
            Self::AllModelTokens => all_model_token_fixations(model),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CacheCase {
    name: &'static str,
    width: usize,
    height: usize,
    frames: usize,
    max_tokens: usize,
}

impl CacheCase {
    const fn frames_per_batch(&self) -> u64 {
        (BATCH * self.frames) as u64
    }
}

#[derive(Clone, Copy, Debug)]
struct TileVideoCase {
    name: &'static str,
    width: usize,
    height: usize,
    frames: usize,
}

impl TileVideoCase {
    const fn frames_per_batch(&self) -> u64 {
        (BATCH * self.frames) as u64
    }
}

#[derive(Clone, Copy, Debug)]
enum TaskLossBenchSetting {
    ModelDefault,
    Disabled,
    Threshold { name: &'static str, value: f32 },
}

impl TaskLossBenchSetting {
    const fn name(self) -> &'static str {
        match self {
            Self::ModelDefault => "model-default",
            Self::Disabled => "disabled",
            Self::Threshold { name, .. } => name,
        }
    }

    fn requirement<B: Backend>(self, model: &NativeAutoGazeModel<B>) -> Option<f32> {
        match self {
            Self::ModelDefault => model.default_task_loss_requirement(),
            Self::Disabled => None,
            Self::Threshold { value, .. } => Some(value),
        }
    }
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
        name: "anyres-tile-224",
        mode: AutoGazeInferenceMode::TiledResizeToGrid {
            tile_size: MODEL_INPUT_SIZE,
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
const TENSOR_VISUALIZATION_LAYOUTS: &[TensorVisualizationLayout] = &[
    TensorVisualizationLayout::SideBySide,
    TensorVisualizationLayout::Panels,
];
const TENSOR_UPDATE_POLICY_DEFAULT: &[TensorUpdatePolicyCase] =
    &[TensorUpdatePolicyCase::AutoFullFrame];
const TENSOR_UPDATE_POLICY_DELTA: &[TensorUpdatePolicyCase] = &[
    TensorUpdatePolicyCase::AutoFullFrame,
    TensorUpdatePolicyCase::NoFullFrame,
];
const TENSOR_FIXATION_CASES: &[TensorFixationCase] = &[
    TensorFixationCase::ModelDefault,
    TensorFixationCase::TinySparse,
    TensorFixationCase::CoarseDense,
    TensorFixationCase::DenseGrid64,
    TensorFixationCase::DenseGrid128,
];
const TENSOR_TOKEN_RENDER_PATHS: &[TensorTokenRenderPath] = &[
    TensorTokenRenderPath::PointPanels,
    TensorTokenRenderPath::DeviceTokenPanels,
];
const TENSOR_DEVICE_TOKEN_CASES: &[TensorDeviceTokenCase] = &[
    TensorDeviceTokenCase::ModelDefault,
    TensorDeviceTokenCase::AllModelTokens,
];
const MODEL_INPUT_SIZE: usize = 224;
const PATCH_SIZE: usize = 16;
const MODEL_GRID: usize = MODEL_INPUT_SIZE / PATCH_SIZE;
const CONNECTOR_TOKENS: usize = MODEL_GRID * MODEL_GRID;
const BATCH: usize = 1;
const FRAMES: usize = 2;
const CHANNELS: usize = 3;
const REAL_TOP_K: usize = 10;
const REAL_DECODE_CHUNK_SIZE: usize = 4;
const REAL_TILE_BATCH_CASES: &[(usize, usize)] = &[(2, 64), (10, 8), (10, 64), (24, 8), (24, 64)];
const REAL_TILE_VIDEO_CASES: &[TileVideoCase] = &[
    TileVideoCase {
        name: "720p-2f",
        width: 1280,
        height: 720,
        frames: 2,
    },
    TileVideoCase {
        name: "1080p-2f",
        width: 1920,
        height: 1080,
        frames: 2,
    },
    TileVideoCase {
        name: "720p-16f",
        width: 1280,
        height: 720,
        frames: 16,
    },
    TileVideoCase {
        name: "1080p-16f",
        width: 1920,
        height: 1080,
        frames: 16,
    },
];
const REAL_CACHE_CASES: &[CacheCase] = &[
    CacheCase {
        name: "realtime-640x360-2f-max10",
        width: 640,
        height: 360,
        frames: 2,
        max_tokens: 10,
    },
    CacheCase {
        name: "realtime-640x360-16f-max2",
        width: 640,
        height: 360,
        frames: 16,
        max_tokens: 2,
    },
    CacheCase {
        name: "realtime-640x360-16f-max10",
        width: 640,
        height: 360,
        frames: 16,
        max_tokens: 10,
    },
    CacheCase {
        name: "realtime-640x360-16f-max12",
        width: 640,
        height: 360,
        frames: 16,
        max_tokens: 12,
    },
    CacheCase {
        name: "720p-2f-max2",
        width: 1280,
        height: 720,
        frames: 2,
        max_tokens: 2,
    },
    CacheCase {
        name: "1080p-2f-max2",
        width: 1920,
        height: 1080,
        frames: 2,
        max_tokens: 2,
    },
    CacheCase {
        name: "720p-2f-max10",
        width: 1280,
        height: 720,
        frames: 2,
        max_tokens: 10,
    },
    CacheCase {
        name: "1080p-2f-max10",
        width: 1920,
        height: 1080,
        frames: 2,
        max_tokens: 10,
    },
    CacheCase {
        name: "720p-2f-max24",
        width: 1280,
        height: 720,
        frames: 2,
        max_tokens: 24,
    },
    CacheCase {
        name: "1080p-2f-max24",
        width: 1920,
        height: 1080,
        frames: 2,
        max_tokens: 24,
    },
    CacheCase {
        name: "720p-16f-max2",
        width: 1280,
        height: 720,
        frames: 16,
        max_tokens: 2,
    },
    CacheCase {
        name: "1080p-16f-max2",
        width: 1920,
        height: 1080,
        frames: 16,
        max_tokens: 2,
    },
    CacheCase {
        name: "720p-16f-max10",
        width: 1280,
        height: 720,
        frames: 16,
        max_tokens: 10,
    },
    CacheCase {
        name: "1080p-16f-max10",
        width: 1920,
        height: 1080,
        frames: 16,
        max_tokens: 10,
    },
];
const REAL_TASK_LOSS_CASES: &[TaskLossBenchSetting] = &[
    TaskLossBenchSetting::ModelDefault,
    TaskLossBenchSetting::Disabled,
    TaskLossBenchSetting::Threshold {
        name: "threshold-0.7",
        value: 0.7,
    },
];
const REAL_KV_QUALITY_SWEEP_CASE: CacheCase = CacheCase {
    name: "realtime-640x360-16f-full-budget",
    width: 640,
    height: 360,
    frames: 16,
    max_tokens: 0,
};
const REAL_KV_QUALITY_CASES: &[TaskLossBenchSetting] = &[
    TaskLossBenchSetting::ModelDefault,
    TaskLossBenchSetting::Disabled,
    TaskLossBenchSetting::Threshold {
        name: "threshold-0.7",
        value: 0.7,
    },
    TaskLossBenchSetting::Threshold {
        name: "threshold-0.45",
        value: 0.45,
    },
    TaskLossBenchSetting::Threshold {
        name: "threshold-0.3",
        value: 0.3,
    },
];
const BLEND_ALPHA: f32 = 0.55;
const KEYFRAME_DURATION: usize = 30;
const TILE_BATCH_CASES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

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

fn bench_real_task_loss(c: &mut Criterion) {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!(
            "skipping real AutoGaze task-loss benchmarks: set AUTOGAZE_HF_DIR to a Hugging Face snapshot"
        );
        return;
    };

    let mut group = c.benchmark_group("autogaze_real_task_loss");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_real_task_loss::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", &hf_dir, device,
        );
    }

    #[cfg(feature = "cuda")]
    register_real_task_loss::<burn::backend::Cuda<f32, i32>>(
        &mut group,
        "cuda",
        &hf_dir,
        burn::backend::cuda::CudaDevice::default(),
    );

    group.finish();
}

fn bench_real_video_file(c: &mut Criterion) {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!(
            "skipping real AutoGaze video-file benchmarks: set AUTOGAZE_HF_DIR to a Hugging Face snapshot"
        );
        return;
    };
    let Some(video_path) = real_video_path() else {
        eprintln!(
            "skipping real AutoGaze video-file benchmarks: set AUTOGAZE_VIDEO or provide /home/mosure/Videos/birds.mp4"
        );
        return;
    };

    let mut group = c.benchmark_group("autogaze_real_video_file");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_real_video_file::<burn::backend::WebGpu<f32, i32>>(
            &mut group,
            "webgpu",
            &hf_dir,
            &video_path,
            device,
        );
    }

    #[cfg(feature = "cuda")]
    register_real_video_file::<burn::backend::Cuda<f32, i32>>(
        &mut group,
        "cuda",
        &hf_dir,
        &video_path,
        burn::backend::cuda::CudaDevice::default(),
    );

    group.finish();
}

fn bench_real_tile_batch_size(c: &mut Criterion) {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!(
            "skipping real AutoGaze tile-batch benchmarks: set AUTOGAZE_HF_DIR to a Hugging Face snapshot"
        );
        return;
    };

    let mut group = c.benchmark_group("autogaze_real_tile_batch_size");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_real_tile_batch_size::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", &hf_dir, device,
        );
    }

    #[cfg(feature = "cuda")]
    register_real_tile_batch_size::<burn::backend::Cuda<f32, i32>>(
        &mut group,
        "cuda",
        &hf_dir,
        burn::backend::cuda::CudaDevice::default(),
    );

    group.finish();
}

fn bench_real_kv_cache(c: &mut Criterion) {
    let Some(hf_dir) = real_model_dir() else {
        eprintln!(
            "skipping real AutoGaze KV-cache benchmarks: set AUTOGAZE_HF_DIR to a Hugging Face snapshot"
        );
        return;
    };

    let mut group = c.benchmark_group("autogaze_real_kv_cache");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_real_kv_cache::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", &hf_dir, device,
        );
    }

    #[cfg(feature = "cuda")]
    register_real_kv_cache::<burn::backend::Cuda<f32, i32>>(
        &mut group,
        "cuda",
        &hf_dir,
        burn::backend::cuda::CudaDevice::default(),
    );

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

fn bench_patch_diff_sparse_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_patch_diff_sparse_mask");
    group.sample_size(10);
    register_ndarray_patch_diff(&mut group);

    #[cfg(feature = "webgpu")]
    register_webgpu_patch_diff(&mut group);

    #[cfg(feature = "cuda")]
    register_cuda_patch_diff(&mut group);

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

fn bench_tensor_visualization(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_tensor_visualization");
    group.sample_size(10);

    #[cfg(feature = "ndarray")]
    {
        register_tensor_visualization::<burn::backend::NdArray<f32>>(
            &mut group,
            "ndarray",
            Default::default(),
        );
    }

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_tensor_visualization::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", device,
        );
    }

    #[cfg(feature = "cuda")]
    {
        register_tensor_visualization::<burn::backend::Cuda<f32, i32>>(
            &mut group,
            "cuda",
            burn::backend::cuda::CudaDevice::default(),
        );
    }

    group.finish();
}

fn bench_tensor_device_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_tensor_device_tokens");
    group.sample_size(10);

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_tensor_device_tokens::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", device,
        );
    }

    #[cfg(feature = "cuda")]
    {
        register_tensor_device_tokens::<burn::backend::Cuda<f32, i32>>(
            &mut group,
            "cuda",
            burn::backend::cuda::CudaDevice::default(),
        );
    }

    group.finish();
}

fn bench_sparse_readout_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_sparse_readout_adapter");
    group.sample_size(50);

    for &model in MODEL_CASES {
        bench_sparse_readout_host_case(&mut group, model);
    }

    #[cfg(feature = "ndarray")]
    {
        register_sparse_readout_coord_tensor::<burn::backend::NdArray<f32>>(
            &mut group,
            "ndarray",
            Default::default(),
        );
    }

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_sparse_readout_coord_tensor::<burn::backend::WebGpu<f32, i32>>(
            &mut group, "webgpu", device,
        );
    }

    #[cfg(feature = "cuda")]
    {
        register_sparse_readout_coord_tensor::<burn::backend::Cuda<f32, i32>>(
            &mut group,
            "cuda",
            burn::backend::cuda::CudaDevice::default(),
        );
    }

    group.finish();
}

fn bench_tile_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("autogaze_tile_batch_size");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    #[cfg(feature = "ndarray")]
    {
        let device = Default::default();
        register_tile_batch_size::<burn::backend::NdArray<f32>>(&mut group, "ndarray", device);
    }

    #[cfg(feature = "webgpu")]
    if let Some(device) = webgpu_device() {
        register_tile_batch_size::<burn::backend::WebGpu<f32, i32>>(&mut group, "webgpu", device);
    }

    #[cfg(feature = "cuda")]
    {
        let device = burn::backend::cuda::CudaDevice::default();
        register_tile_batch_size::<burn::backend::Cuda<f32, i32>>(&mut group, "cuda", device);
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

#[cfg(feature = "ndarray")]
fn register_ndarray_patch_diff(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &case in VIDEO_CASES {
        let device = Default::default();
        bench_patch_diff_case::<burn::backend::NdArray<f32>>(group, "ndarray", case, device);
    }
}

#[cfg(not(feature = "ndarray"))]
fn register_ndarray_patch_diff(_group: &mut BenchmarkGroup<'_, WallTime>) {}

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

#[cfg(feature = "webgpu")]
fn register_webgpu_patch_diff(group: &mut BenchmarkGroup<'_, WallTime>) {
    let Some(device) = webgpu_device() else {
        return;
    };
    for &case in VIDEO_CASES {
        bench_patch_diff_case::<burn::backend::WebGpu<f32, i32>>(
            group,
            "webgpu",
            case,
            device.clone(),
        );
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

#[cfg(feature = "cuda")]
fn register_cuda_patch_diff(group: &mut BenchmarkGroup<'_, WallTime>) {
    let device = burn::backend::cuda::CudaDevice::default();
    for &case in VIDEO_CASES {
        bench_patch_diff_case::<burn::backend::Cuda<f32, i32>>(group, "cuda", case, device.clone());
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

fn register_real_task_loss<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    hf_dir: &Path,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let Some(base_pipeline) = run_optional(backend, || {
        AutoGazePipeline::<B>::from_hf_dir(hf_dir, &device)
            .expect("load real AutoGaze model")
            .with_max_gaze_tokens_each_frame(REAL_TOP_K)
    }) else {
        return;
    };

    for &case in VIDEO_CASES {
        for &mode in MODE_CASES {
            for &setting in REAL_TASK_LOSS_CASES {
                let mut pipeline = base_pipeline.clone();
                match setting {
                    TaskLossBenchSetting::ModelDefault => {}
                    TaskLossBenchSetting::Disabled => pipeline.set_task_loss_requirement(None),
                    TaskLossBenchSetting::Threshold { value, .. } => {
                        pipeline.set_task_loss_requirement(Some(value));
                    }
                }
                let video = deterministic_video::<B>(
                    BATCH,
                    FRAMES,
                    CHANNELS,
                    case.height,
                    case.width,
                    &device,
                );
                group.throughput(Throughput::Elements(case.frames_per_batch()));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{backend}/{}/{}", mode.name, setting.name()),
                        case.name,
                    ),
                    &(case, mode, setting),
                    |b, _| {
                        b.iter_batched(
                            || video.clone(),
                            |video| {
                                black_box(
                                    pipeline.trace_video_with_mode(video, REAL_TOP_K, mode.mode),
                                );
                                B::sync(&device).expect("backend sync");
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
}

fn register_real_video_file<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    hf_dir: &Path,
    video_path: &Path,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let Some(base_pipeline) = run_optional(backend, || {
        AutoGazePipeline::<B>::from_hf_dir(hf_dir, &device)
            .expect("load real AutoGaze model")
            .with_max_gaze_tokens_each_frame(REAL_TOP_K)
    }) else {
        return;
    };

    for &case in VIDEO_CASES {
        let Some(rgba) = decode_video_rgba(video_path, case.width, case.height, FRAMES) else {
            continue;
        };
        for &mode in MODE_CASES {
            let pipeline = base_pipeline.clone();
            group.throughput(Throughput::Elements(case.frames_per_batch()));
            group.bench_with_input(
                BenchmarkId::new(format!("{backend}/{}", mode.name), case.name),
                &(case, mode),
                |b, _| {
                    b.iter_batched(
                        || rgba.clone(),
                        |rgba| {
                            black_box(
                                pipeline
                                    .trace_rgba_clip_with_mode(
                                        &rgba,
                                        AutoGazeRgbaClipShape::new(FRAMES, case.height, case.width),
                                        REAL_TOP_K,
                                        mode.mode,
                                        &device,
                                    )
                                    .expect("trace real video RGBA clip"),
                            );
                            B::sync(&device).expect("backend sync");
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

fn register_real_tile_batch_size<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    hf_dir: &Path,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let Some(model) = run_optional(backend, || {
        AutoGazePipeline::<B>::from_hf_dir(hf_dir, &device).expect("load real AutoGaze model")
    }) else {
        return;
    };
    let mode = AutoGazeInferenceMode::TiledResizeToGrid {
        tile_size: MODEL_INPUT_SIZE,
    };

    for &case in REAL_TILE_VIDEO_CASES {
        if should_skip_long_context_case(backend, case.frames) {
            eprintln!(
                "skipping {backend} real tile stress case {}; set AUTOGAZE_BENCH_LONG_CONTEXT=1 to include it",
                case.name
            );
            continue;
        }
        for &(top_k, tile_batch_size) in REAL_TILE_BATCH_CASES {
            let pipeline = model
                .clone()
                .with_max_gaze_tokens_each_frame(top_k)
                .with_tile_batch_size(tile_batch_size);
            let video = deterministic_video::<B>(
                BATCH,
                case.frames,
                CHANNELS,
                case.height,
                case.width,
                &device,
            );
            group.throughput(Throughput::Elements(case.frames_per_batch()));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{backend}/top-k-{top_k}/tile-batch-{tile_batch_size}"),
                    case.name,
                ),
                &case,
                |b, _| {
                    b.iter_batched(
                        || video.clone(),
                        |video| {
                            black_box(pipeline.trace_video_with_mode(video, top_k, mode));
                            B::sync(&device).expect("backend sync");
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

fn register_real_kv_cache<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    hf_dir: &Path,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let Some(model) = run_optional(backend, || {
        NativeAutoGazeModel::<B>::from_hf_dir(hf_dir, &device).expect("load real AutoGaze model")
    }) else {
        return;
    };

    for &case in REAL_CACHE_CASES {
        if should_skip_long_context_case(backend, case.frames) {
            eprintln!(
                "skipping {backend} real KV stress case {}; set AUTOGAZE_BENCH_LONG_CONTEXT=1 to include it",
                case.name
            );
            continue;
        }
        let video = deterministic_video::<B>(
            BATCH,
            case.frames,
            CHANNELS,
            case.height,
            case.width,
            &device,
        );
        let stream_frames = (0..case.frames)
            .map(|frame_idx| video.clone().slice_dim(1, frame_idx..(frame_idx + 1)))
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(case.frames_per_batch()));

        group.bench_with_input(
            BenchmarkId::new(
                format!("{backend}/streaming-cache/max-{}", case.max_tokens),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || {
                        (
                            stream_frames.clone(),
                            AutoGazeStreamingCache::new(case.frames),
                        )
                    },
                    |(stream_frames, mut cache)| {
                        for frame in stream_frames {
                            black_box(model.gazing_model.generate_streaming_cached(
                                frame,
                                &mut cache,
                                case.max_tokens,
                                None,
                            ));
                        }
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!("{backend}/streaming-host-async/max-{}", case.max_tokens),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || {
                        (
                            stream_frames.clone(),
                            AutoGazeStreamingCache::new(case.frames),
                        )
                    },
                    |(stream_frames, mut cache)| {
                        for frame in stream_frames {
                            black_box(
                                block_on(model.generate_streaming_with_decode_strategy_async(
                                    frame,
                                    &mut cache,
                                    case.max_tokens,
                                    None,
                                    None,
                                    AutoGazeDecodeStrategy::HostGreedy,
                                ))
                                .expect("host async streaming generation"),
                            );
                        }
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "{backend}/streaming-device-chunk-{REAL_DECODE_CHUNK_SIZE}/max-{}",
                    case.max_tokens
                ),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || {
                        (
                            stream_frames.clone(),
                            AutoGazeStreamingCache::new(case.frames),
                        )
                    },
                    |(stream_frames, mut cache)| {
                        for frame in stream_frames {
                            black_box(
                                block_on(model.generate_streaming_with_decode_strategy_async(
                                    frame,
                                    &mut cache,
                                    case.max_tokens,
                                    None,
                                    None,
                                    AutoGazeDecodeStrategy::DeviceGreedy {
                                        chunk_size: REAL_DECODE_CHUNK_SIZE,
                                    },
                                ))
                                .expect("device streaming generation"),
                            );
                        }
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "{backend}/streaming-device-terminal-readback-chunk-{REAL_DECODE_CHUNK_SIZE}/max-{}",
                    case.max_tokens
                ),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || {
                        (
                            stream_frames.clone(),
                            AutoGazeStreamingCache::new(case.frames),
                        )
                    },
                    |(stream_frames, mut cache)| {
                        for frame in stream_frames {
                            black_box(
                                block_on(model.generate_streaming_with_decode_strategy_async(
                                    frame,
                                    &mut cache,
                                    case.max_tokens,
                                    None,
                                    None,
                                    AutoGazeDecodeStrategy::DeviceTerminalGreedy {
                                        chunk_size: REAL_DECODE_CHUNK_SIZE,
                                    },
                                ))
                                .expect("terminal device streaming generation"),
                            );
                        }
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!("{backend}/kv-cache/max-{}", case.max_tokens),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || video.clone(),
                    |video| {
                        black_box(
                            model
                                .gazing_model
                                .generate_cached(video, case.max_tokens, None),
                        );
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!("{backend}/full-seq/max-{}", case.max_tokens),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || video.clone(),
                    |video| {
                        black_box(model.gazing_model.generate_uncached(
                            video,
                            case.max_tokens,
                            None,
                        ));
                        B::sync(&device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    register_real_kv_quality_sweep(group, backend, &model, &device);
}

fn register_real_kv_quality_sweep<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    model: &NativeAutoGazeModel<B>,
    device: &B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let case = REAL_KV_QUALITY_SWEEP_CASE;
    if should_skip_long_context_case(backend, case.frames) {
        eprintln!(
            "skipping {backend} real KV quality sweep {}; set AUTOGAZE_BENCH_LONG_CONTEXT=1 to include it",
            case.name
        );
        return;
    }

    let max_tokens = model.default_max_gaze_tokens_each_frame();
    let video = deterministic_video::<B>(
        BATCH,
        case.frames,
        CHANNELS,
        case.height,
        case.width,
        device,
    );
    let stream_frames = (0..case.frames)
        .map(|frame_idx| video.clone().slice_dim(1, frame_idx..(frame_idx + 1)))
        .collect::<Vec<_>>();
    group.throughput(Throughput::Elements(case.frames_per_batch()));

    for &setting in REAL_KV_QUALITY_CASES {
        let task_loss_requirement = setting.requirement(model);
        group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "{backend}/kv-quality-{}/model-budget-max-{max_tokens}/terminal-chunk-{REAL_DECODE_CHUNK_SIZE}",
                    setting.name(),
                ),
                case.name,
            ),
            &case,
            |b, _| {
                b.iter_batched(
                    || {
                        (
                            stream_frames.clone(),
                            AutoGazeStreamingCache::new(case.frames),
                        )
                    },
                    |(stream_frames, mut cache)| {
                        for frame in stream_frames {
                            black_box(
                                block_on(model.generate_streaming_with_decode_strategy_async(
                                    frame,
                                    &mut cache,
                                    max_tokens,
                                    task_loss_requirement,
                                    None,
                                    AutoGazeDecodeStrategy::DeviceTerminalGreedy {
                                        chunk_size: REAL_DECODE_CHUNK_SIZE,
                                    },
                                ))
                                .expect("terminal device streaming quality-sweep generation"),
                            );
                        }
                        B::sync(device).expect("backend sync");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
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

fn bench_patch_diff_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    device: B::Device,
) where
    B: Backend,
    f64: From<<B as burn::tensor::backend::BackendTypes>::FloatElem>,
{
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    let config = AutoGazePatchDiffConfig::new(14, 0.45);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(BenchmarkId::new(backend, case.name), &case, |b, _| {
        b.iter_batched(
            || video.clone(),
            |video| {
                black_box(patch_diff_readout_points(video, config).expect("patch-diff readout"));
                B::sync(&device).expect("backend sync");
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}-device-mask"), case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    black_box(
                        futures_lite::future::block_on(patch_diff_device_mask_async(
                            video,
                            config,
                            case.height,
                            case.width,
                        ))
                        .expect("patch-diff device mask"),
                    );
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn register_tile_batch_size<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let case = VideoCase {
        name: "1080p",
        width: 1920,
        height: 1080,
    };
    let model = ModelCase {
        name: "multiscale-32-64-112-224",
        scales: "32+64+112+224",
        num_vision_tokens_each_frame: 265,
    };
    let mode = AutoGazeInferenceMode::TiledResizeToGrid {
        tile_size: MODEL_INPUT_SIZE,
    };

    if run_optional(backend, || {
        let pipeline = deterministic_pipeline::<B>(model, &device);
        let video =
            deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
        let output = pipeline.embed_video_with_mode(video, mode);
        black_box(output.embeddings.into_data());
        B::sync(&device).expect("backend sync");
    })
    .is_none()
    {
        return;
    }

    for &tile_batch_size in TILE_BATCH_CASES {
        bench_tile_batch_embed_case::<B>(
            group,
            backend,
            case,
            model,
            mode,
            tile_batch_size,
            device.clone(),
        );
        bench_tile_batch_trace_case::<B>(
            group,
            backend,
            case,
            model,
            mode,
            tile_batch_size,
            device.clone(),
        );
    }
}

fn bench_tile_batch_embed_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: AutoGazeInferenceMode,
    tile_batch_size: usize,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline =
        deterministic_pipeline::<B>(model, &device).with_tile_batch_size(tile_batch_size);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(
            format!("{backend}/embed/tile-batch-{tile_batch_size}"),
            case.name,
        ),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    let output = pipeline.embed_video_with_mode(video, mode);
                    black_box(output.layout.tile_count());
                    black_box(output.embeddings.into_data());
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_tile_batch_trace_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: VideoCase,
    model: ModelCase,
    mode: AutoGazeInferenceMode,
    tile_batch_size: usize,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let pipeline =
        deterministic_pipeline::<B>(model, &device).with_tile_batch_size(tile_batch_size);
    let video = deterministic_video::<B>(BATCH, FRAMES, CHANNELS, case.height, case.width, &device);
    group.throughput(Throughput::Elements(case.frames_per_batch()));
    group.bench_with_input(
        BenchmarkId::new(
            format!("{backend}/trace/tile-batch-{tile_batch_size}"),
            case.name,
        ),
        &case,
        |b, _| {
            b.iter_batched(
                || video.clone(),
                |video| {
                    black_box(pipeline.trace_video_with_mode(video, 2, mode));
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

fn bench_sparse_readout_host_case(group: &mut BenchmarkGroup<'_, WallTime>, model: ModelCase) {
    let points = deterministic_fixations(model);
    let image_grid = SparseReadoutGrid::new(64, 64);
    let video_grid = SparseVideoReadoutGrid::new(FRAMES, 64, 64);
    let readout_options = SparseReadoutOptions::default().with_max_tokens_per_frame(512);
    let video_options = SparseVideoReadoutOptions::default()
        .with_tubelet_size(1)
        .with_exact_tokens(1024);
    group.throughput(Throughput::Elements(
        video_options.max_tokens.unwrap_or(0) as u64
    ));
    group.bench_function(
        BenchmarkId::new("host/readout-to-coords", model.name),
        |b| {
            b.iter(|| {
                let frame_tokens = (0..FRAMES)
                    .map(|_| {
                        fixation_points_to_readout_tokens(
                            black_box(&points),
                            image_grid,
                            readout_options,
                        )
                        .expect("frame readout tokens")
                    })
                    .collect::<Vec<_>>();
                let coords = frame_readout_tokens_to_video_coords(
                    black_box(&frame_tokens),
                    image_grid,
                    video_grid,
                    video_options,
                    0,
                )
                .expect("video readout coords");
                black_box(coords);
            });
        },
    );
}

fn register_sparse_readout_coord_tensor<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    for &model in MODEL_CASES {
        let points = deterministic_fixations(model);
        let image_grid = SparseReadoutGrid::new(64, 64);
        let video_grid = SparseVideoReadoutGrid::new(FRAMES, 64, 64);
        let frame_tokens = (0..FRAMES)
            .map(|_| {
                fixation_points_to_readout_tokens(
                    &points,
                    image_grid,
                    SparseReadoutOptions::default().with_max_tokens_per_frame(512),
                )
                .expect("frame readout tokens")
            })
            .collect::<Vec<_>>();
        let coords = frame_readout_tokens_to_video_coords(
            &frame_tokens,
            image_grid,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(1)
                .with_exact_tokens(1024),
            0,
        )
        .expect("video readout coords");
        group.throughput(Throughput::Elements(coords.len() as u64));
        group.bench_function(
            BenchmarkId::new(format!("{backend}/coord-tensor"), model.name),
            |b| {
                b.iter(|| {
                    let tensor = video_readout_coords_to_tensor::<B>(black_box(&coords), &device);
                    black_box(tensor);
                    B::sync(&device).expect("backend sync");
                });
            },
        );
    }
}

fn register_tensor_visualization<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    for &case in VIDEO_CASES {
        group.throughput(Throughput::Elements((case.width * case.height) as u64));
        for &model in MODEL_CASES {
            for &visualization in VISUALIZATION_CASES {
                for &fixations in TENSOR_FIXATION_CASES {
                    for &layout in TENSOR_VISUALIZATION_LAYOUTS {
                        let update_policies = if visualization.force_delta_frame {
                            TENSOR_UPDATE_POLICY_DELTA
                        } else {
                            TENSOR_UPDATE_POLICY_DEFAULT
                        };
                        for &update_policy in update_policies {
                            bench_tensor_visualization_case::<B>(
                                group,
                                backend,
                                TensorVisualizationBenchCase {
                                    video: case,
                                    model,
                                    visualization,
                                    fixations,
                                    layout,
                                    update_policy,
                                },
                                device.clone(),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TensorVisualizationBenchCase {
    video: VideoCase,
    model: ModelCase,
    visualization: VisualizationCase,
    fixations: TensorFixationCase,
    layout: TensorVisualizationLayout,
    update_policy: TensorUpdatePolicyCase,
}

fn bench_tensor_visualization_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: TensorVisualizationBenchCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let current =
        deterministic_video::<B>(1, 1, CHANNELS, case.video.height, case.video.width, &device);
    let previous =
        deterministic_video::<B>(1, 1, CHANNELS, case.video.height, case.video.width, &device);
    let points = case.fixations.points(case.model);
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "{backend}/{}/{}/{}/{}/{}",
                case.model.name,
                case.visualization.name,
                case.fixations.name(),
                case.layout.name(),
                case.update_policy.name()
            ),
            case.video.name,
        ),
        &case.video,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state = AutoGazeTensorVisualizationState::<B>::new(
                        case.visualization.mode,
                        if case.visualization.force_delta_frame {
                            usize::MAX
                        } else {
                            KEYFRAME_DURATION
                        },
                    );
                    if case.visualization.force_delta_frame {
                        let _ = state
                            .visualize_normalized_rgb_clip(
                                previous.clone(),
                                &points,
                                AutoGazeTensorVisualizationOptions::new(
                                    case.video.width,
                                    case.video.height,
                                    1.0,
                                    BLEND_ALPHA,
                                )
                                .with_full_frame_update_policy(0.0),
                                &device,
                            )
                            .expect("prime tensor interframe visualization state");
                    }
                    state
                },
                |mut state| {
                    match case.layout {
                        TensorVisualizationLayout::SideBySide => {
                            let output = state
                                .visualize_normalized_rgb_clip(
                                    current.clone(),
                                    &points,
                                    case.update_policy.options(
                                        case.video.width,
                                        case.video.height,
                                        BLEND_ALPHA,
                                    ),
                                    &device,
                                )
                                .expect("visualize tensor autogaze mask");
                            black_box(output.mask_ratio());
                            black_box(output.update_ratio());
                            black_box(state.last_interframe_path().map(|path| path.as_str()));
                            black_box(output.side_by_side_rgba.shape());
                        }
                        TensorVisualizationLayout::Panels => {
                            let output = state
                                .visualize_normalized_rgb_clip_panels(
                                    current.clone(),
                                    &points,
                                    case.update_policy.options(
                                        case.video.width,
                                        case.video.height,
                                        BLEND_ALPHA,
                                    ),
                                    &device,
                                )
                                .expect("visualize tensor autogaze panel mask");
                            black_box(output.mask_ratio());
                            black_box(output.update_ratio());
                            black_box(state.last_interframe_path().map(|path| path.as_str()));
                            black_box(output.output_rgba.shape());
                        }
                    }
                    B::sync(&device).expect("backend sync");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn register_tensor_device_tokens<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    for &case in VIDEO_CASES {
        group.throughput(Throughput::Elements((case.width * case.height) as u64));
        for &model in MODEL_CASES {
            for &visualization in VISUALIZATION_CASES {
                if matches!(visualization.mode, AutoGazeVisualizationMode::Interframe)
                    && !visualization.force_delta_frame
                {
                    continue;
                }
                for &token_case in TENSOR_DEVICE_TOKEN_CASES {
                    for &path in TENSOR_TOKEN_RENDER_PATHS {
                        bench_tensor_device_token_case::<B>(
                            group,
                            backend,
                            TensorDeviceTokenBenchCase {
                                video: case,
                                model,
                                visualization,
                                token_case,
                                path,
                            },
                            device.clone(),
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TensorDeviceTokenBenchCase {
    video: VideoCase,
    model: ModelCase,
    visualization: VisualizationCase,
    token_case: TensorDeviceTokenCase,
    path: TensorTokenRenderPath,
}

fn bench_tensor_device_token_case<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    case: TensorDeviceTokenBenchCase,
    device: B::Device,
) where
    B: Backend,
    B::Device: Clone,
{
    let config = tiny_config(case.model);
    let current =
        deterministic_video::<B>(1, 1, CHANNELS, case.video.height, case.video.width, &device);
    let previous =
        deterministic_video::<B>(1, 1, CHANNELS, case.video.height, case.video.width, &device);
    let points = case.token_case.points(case.model);
    let device_tokens = device_tokens_from_points::<B>(&points, &config, &device);
    let options = AutoGazeTensorVisualizationOptions::new(
        case.video.width,
        case.video.height,
        1.0,
        BLEND_ALPHA,
    )
    .with_full_frame_update_policy(0.0);
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "{backend}/{}/{}/{}/{}",
                case.model.name,
                case.visualization.name,
                case.token_case.name(),
                case.path.name(),
            ),
            case.video.name,
        ),
        &case.video,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state = AutoGazeTensorVisualizationState::<B>::new(
                        case.visualization.mode,
                        if case.visualization.force_delta_frame {
                            usize::MAX
                        } else {
                            KEYFRAME_DURATION
                        },
                    );
                    if case.visualization.force_delta_frame {
                        match case.path {
                            TensorTokenRenderPath::PointPanels => {
                                let _ = state
                                    .visualize_normalized_rgb_clip_panels(
                                        previous.clone(),
                                        &points,
                                        options,
                                        &device,
                                    )
                                    .expect("prime point tensor visualization state");
                            }
                            TensorTokenRenderPath::DeviceTokenPanels => {
                                let _ = state
                                    .visualize_normalized_rgb_clip_device_tokens_panels(
                                        previous.clone(),
                                        &device_tokens,
                                        &config,
                                        options,
                                        &device,
                                    )
                                    .expect("prime device-token tensor visualization state");
                            }
                        }
                    }
                    state
                },
                |mut state| {
                    match case.path {
                        TensorTokenRenderPath::PointPanels => {
                            let output = state
                                .visualize_normalized_rgb_clip_panels(
                                    current.clone(),
                                    &points,
                                    options,
                                    &device,
                                )
                                .expect("visualize point tensor panels");
                            black_box(output.mask_ratio());
                            black_box(output.update_ratio());
                            black_box(output.output_rgba.shape());
                        }
                        TensorTokenRenderPath::DeviceTokenPanels => {
                            let output = state
                                .visualize_normalized_rgb_clip_device_tokens_panels(
                                    current.clone(),
                                    &device_tokens,
                                    &config,
                                    options,
                                    &device,
                                )
                                .expect("visualize device-token tensor panels");
                            black_box(output.mask_ratio());
                            black_box(output.update_ratio());
                            black_box(output.output_rgba.shape());
                        }
                    }
                    black_box(state.last_interframe_path().map(|path| path.as_str()));
                    B::sync(&device).expect("backend sync");
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

fn all_model_token_fixations(model: ModelCase) -> Vec<FixationPoint> {
    let config = tiny_config(model);
    let mut points = Vec::new();
    for layout in scale_token_layouts(&config) {
        let grid = layout.grid.max(1);
        let extent = 1.0 / grid as f32;
        for row in 0..grid {
            for col in 0..grid {
                points.push(FixationPoint::with_grid_extent(
                    (col as f32 + 0.5) * extent,
                    (row as f32 + 0.5) * extent,
                    extent,
                    extent,
                    1.0,
                    grid,
                ));
            }
        }
    }
    points
}

fn dense_grid_fixations(grid: usize) -> Vec<FixationPoint> {
    let extent = 1.0 / grid as f32;
    (0..grid)
        .flat_map(|row| {
            (0..grid).map(move |col| {
                FixationPoint::with_grid_extent(
                    (col as f32 + 0.5) * extent,
                    (row as f32 + 0.5) * extent,
                    extent,
                    extent,
                    1.0,
                    grid,
                )
            })
        })
        .collect()
}

fn device_tokens_from_points<B: Backend>(
    points: &[FixationPoint],
    config: &AutoGazeConfig,
    device: &B::Device,
) -> AutoGazeDeviceTokens<B> {
    let layouts = scale_token_layouts(config);
    let mut offset = 0usize;
    let offsets = layouts
        .iter()
        .map(|layout| {
            let start = offset;
            offset = offset.saturating_add(layout.token_count);
            (layout.grid, start)
        })
        .collect::<Vec<_>>();
    let mut tokens = Vec::new();
    for point in points {
        let grid = point.cell_grid().unwrap_or_else(|| {
            (1.0 / point.cell_width().max(point.cell_height()))
                .round()
                .max(1.0) as usize
        });
        let Some((_, start)) = offsets.iter().find(|(candidate, _)| *candidate == grid) else {
            continue;
        };
        let col = ((point.x * grid as f32).floor() as usize).min(grid.saturating_sub(1));
        let row = ((point.y * grid as f32).floor() as usize).min(grid.saturating_sub(1));
        tokens.push((*start + row * grid + col) as i64);
    }

    let slots = tokens.len().max(1);
    let mut valid = vec![true; tokens.len()];
    if tokens.is_empty() {
        tokens.push(config.num_vision_tokens_each_frame as i64);
        valid.push(false);
    }

    AutoGazeDeviceTokens {
        tokens: Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [1, slots]), device),
        valid: Tensor::<B, 2, Bool>::from_bool(TensorData::new(valid, [1, slots]), device),
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

fn real_video_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("AUTOGAZE_VIDEO") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "AUTOGAZE_VIDEO does not exist, skipping real video benchmarks: {}",
            path.display()
        );
        return None;
    }

    let default = PathBuf::from("/home/mosure/Videos/birds.mp4");
    default.exists().then_some(default)
}

fn should_skip_long_context_case(backend: &str, frames: usize) -> bool {
    backend == "webgpu" && frames > 2 && env::var_os("AUTOGAZE_BENCH_LONG_CONTEXT").is_none()
}

fn decode_video_rgba(path: &Path, width: usize, height: usize, frames: usize) -> Option<Vec<u8>> {
    let expected = frames
        .checked_mul(width)?
        .checked_mul(height)?
        .checked_mul(4)?;
    let scale = format!("scale={width}:{height}:flags=bicubic");
    let frames_arg = frames.to_string();
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str()?,
            "-vf",
            &scale,
            "-frames:v",
            &frames_arg,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .output();
    let output = match output {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            eprintln!(
                "skipping real video case {width}x{height}: ffmpeg failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return None;
        }
        Err(err) => {
            eprintln!("skipping real video case {width}x{height}: ffmpeg unavailable: {err}");
            return None;
        }
    };
    if output.stdout.len() < expected {
        eprintln!(
            "skipping real video case {width}x{height}: decoded {} bytes, expected {expected}",
            output.stdout.len()
        );
        return None;
    }
    Some(output.stdout[..expected].to_vec())
}

criterion_group!(
    benches,
    bench_embed_video,
    bench_trace_video,
    bench_real_trace_video,
    bench_real_task_loss,
    bench_real_video_file,
    bench_rgba_e2e_video,
    bench_patch_diff_sparse_mask,
    bench_visualization,
    bench_tensor_visualization,
    bench_tensor_device_tokens,
    bench_sparse_readout_adapter,
    bench_tile_batch_size,
    bench_real_tile_batch_size,
    bench_real_kv_cache
);
criterion_main!(benches);
