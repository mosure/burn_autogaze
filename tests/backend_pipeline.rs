#![cfg(all(feature = "ndarray", any(feature = "webgpu", feature = "cuda")))]

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazePipeline, ConnectorConfig, GazeDecoderConfig,
    GazeModelConfig, NativeAutoGazeModel, VisionModelConfig,
};
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "webgpu")]
use std::sync::OnceLock;

type CpuBackend = burn::backend::NdArray<f32>;

#[derive(Clone, Copy, Debug)]
struct ParityCase {
    name: &'static str,
    model_input: usize,
    height: usize,
    width: usize,
    mode: AutoGazeInferenceMode,
}

#[test]
fn accelerator_embeddings_match_ndarray_reference() {
    #[cfg(feature = "webgpu")]
    let webgpu_device = webgpu_device();

    let cases = [
        ParityCase {
            name: "resize",
            model_input: 16,
            height: 32,
            width: 48,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
        },
        ParityCase {
            name: "tile",
            model_input: 16,
            height: 32,
            width: 48,
            mode: AutoGazeInferenceMode::TiledFullResolution {
                tile_size: 16,
                stride: 16,
            },
        },
    ];

    for case in cases {
        let cpu_device = Default::default();
        let expected = embedding_output::<CpuBackend>(case, &cpu_device);

        #[cfg(feature = "webgpu")]
        if let Some(device) = webgpu_device.as_ref() {
            run_or_skip("webgpu", || {
                assert_backend_matches::<burn::backend::WebGpu<f32, i32>>(
                    "webgpu", case, device, &expected, 2.0e-2,
                );
            });
        }

        #[cfg(feature = "cuda")]
        run_or_skip("cuda", || {
            let device = burn::backend::cuda::CudaDevice::default();
            assert_backend_matches::<burn::backend::Cuda<f32, i32>>(
                "cuda", case, &device, &expected, 2.0e-2,
            );
        });
    }
}

#[test]
fn unavailable_backend_reason_matches_ci_adapter_failure() {
    assert!(is_unavailable_backend_reason(
        "No possible adapter available for backend"
    ));
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
            eprintln!("skipping webgpu backend parity: {reason}");
            None
        }
        Err(reason) => panic!("webgpu backend initialization failed: {reason}"),
    }
}

fn assert_backend_matches<B: Backend>(
    backend: &str,
    case: ParityCase,
    device: &B::Device,
    expected: &[f32],
    tolerance: f32,
) {
    let actual = embedding_output::<B>(case, device);
    let diff = max_abs_diff(&actual, expected);
    assert!(
        diff <= tolerance,
        "{backend} {} embedding drift: max_abs_diff={diff}, tolerance={tolerance}",
        case.name
    );

    let traces = inference_output::<B>(case, device);
    assert_eq!(traces.len(), 1, "{backend} should preserve batch size");
    assert_eq!(
        traces[0].len(),
        2,
        "{backend} should emit one trace set per frame"
    );
    assert_trace_confidences(backend, &traces);
    B::sync(device).expect("backend sync");
}

fn embedding_output<B: Backend>(case: ParityCase, device: &B::Device) -> Vec<f32> {
    let pipeline = deterministic_pipeline::<B>(case.model_input, device);
    let video = deterministic_video::<B>(1, 2, 3, case.height, case.width, device);
    let output = pipeline.embed_video_with_mode(video, case.mode);
    match case.mode {
        AutoGazeInferenceMode::ResizeToModelInput => assert_eq!(output.layout.tile_count(), 1),
        AutoGazeInferenceMode::TiledFullResolution { .. } => {
            assert!(
                output.layout.tile_count() > 1,
                "tiled case should exercise multiple tiles"
            );
        }
    }
    output
        .embeddings
        .into_data()
        .to_vec::<f32>()
        .expect("embedding vec")
}

fn inference_output<B: Backend>(
    case: ParityCase,
    device: &B::Device,
) -> Vec<burn_autogaze::FrameFixationTrace> {
    let pipeline = deterministic_pipeline::<B>(case.model_input, device);
    let video = deterministic_video::<B>(1, 2, 3, case.height, case.width, device);
    pipeline.trace_video_with_mode(video, 2, case.mode)
}

fn deterministic_pipeline<B: Backend>(
    model_input: usize,
    device: &B::Device,
) -> AutoGazePipeline<B> {
    let config = tiny_config(model_input);
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::new(&config, device).map(&mut mapper);
    AutoGazePipeline::new(model).with_max_gaze_tokens_each_frame(2)
}

fn tiny_config(resolution: usize) -> AutoGazeConfig {
    let kernel_size = 8;
    let grid = resolution / kernel_size;
    let tokens = grid * grid;
    let hidden = 8;
    let heads = 2;
    AutoGazeConfig {
        scales: resolution.to_string(),
        max_num_frames: 2,
        num_vision_tokens_each_frame: tokens,
        gaze_model_config: GazeModelConfig {
            input_img_size: resolution,
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

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn assert_trace_confidences(backend: &str, traces: &[burn_autogaze::FrameFixationTrace]) {
    assert!(
        traces
            .iter()
            .flat_map(|trace| trace.frames.iter())
            .flat_map(|set| set.points.iter())
            .all(|point| point.confidence.is_finite() && (0.0..=1.0).contains(&point.confidence)),
        "{backend} generated invalid trace confidences"
    );
}

fn run_or_skip(name: &str, test: impl FnOnce()) {
    match panic::catch_unwind(AssertUnwindSafe(test)) {
        Ok(()) => {}
        Err(payload) => {
            let reason = panic_payload_to_string(payload);
            if is_unavailable_backend_reason(&reason) {
                eprintln!("skipping {name} backend parity: {reason}");
            } else {
                panic!("{name} backend parity failed: {reason}");
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
