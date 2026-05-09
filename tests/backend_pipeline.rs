#![cfg(all(feature = "ndarray", any(feature = "webgpu", feature = "cuda")))]

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazePipeline, ConnectorConfig, FixationPoint,
    FrameFixationTrace, GazeDecoderConfig, GazeModelConfig, NativeAutoGazeModel, VisionModelConfig,
};
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "webgpu")]
use std::sync::OnceLock;

type CpuBackend = burn::backend::NdArray<f32>;

#[derive(Clone, Copy, Debug)]
struct ParityCase {
    name: &'static str,
    model_input: usize,
    scales: &'static str,
    connector_tokens: usize,
    num_vision_tokens_each_frame: usize,
    height: usize,
    width: usize,
    mode: AutoGazeInferenceMode,
}

#[derive(Clone, Debug)]
struct PipelineOutput {
    embeddings: Vec<f32>,
    traces: Vec<FrameFixationTrace>,
}

#[test]
fn accelerator_embeddings_match_ndarray_reference() {
    #[cfg(feature = "webgpu")]
    let webgpu_device = webgpu_device();

    let cases = [
        ParityCase {
            name: "single-scale-resize",
            model_input: 16,
            scales: "16",
            connector_tokens: 4,
            num_vision_tokens_each_frame: 4,
            height: 32,
            width: 48,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
        },
        ParityCase {
            name: "single-scale-tile",
            model_input: 16,
            scales: "16",
            connector_tokens: 4,
            num_vision_tokens_each_frame: 4,
            height: 32,
            width: 48,
            mode: AutoGazeInferenceMode::TiledFullResolution {
                tile_size: 16,
                stride: 16,
            },
        },
        ParityCase {
            name: "single-scale-anyres-tile",
            model_input: 16,
            scales: "16",
            connector_tokens: 4,
            num_vision_tokens_each_frame: 4,
            height: 31,
            width: 47,
            mode: AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
        },
        ParityCase {
            name: "multiscale-resize",
            model_input: 16,
            scales: "8+16",
            connector_tokens: 4,
            num_vision_tokens_each_frame: 5,
            height: 32,
            width: 48,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
        },
        ParityCase {
            name: "multiscale-tile",
            model_input: 16,
            scales: "8+16",
            connector_tokens: 4,
            num_vision_tokens_each_frame: 5,
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
        let expected = pipeline_output::<CpuBackend>(case, &cpu_device);
        assert_trace_shapes("ndarray", case, &expected.traces);
        assert_trace_confidences("ndarray", &expected.traces);

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
fn tiled_embedding_batching_preserves_batch_order() {
    for case in [batched_tiled_case(), batched_anyres_tiled_case()] {
        assert_tiled_embedding_batching_preserves_batch_order(case);
    }
}

fn assert_tiled_embedding_batching_preserves_batch_order(case: ParityCase) {
    let device = Default::default();
    let single_tile_batches =
        deterministic_pipeline::<CpuBackend>(case, &device).with_tile_batch_size(1);
    let multi_tile_batches =
        deterministic_pipeline::<CpuBackend>(case, &device).with_tile_batch_size(4);
    let video = deterministic_video::<CpuBackend>(2, 2, 3, case.height, case.width, &device);

    let expected = single_tile_batches.embed_video_with_mode(video.clone(), case.mode);
    let actual = multi_tile_batches.embed_video_with_mode(video, case.mode);
    let expected_shape = expected.embeddings.shape().dims::<4>();
    let actual_shape = actual.embeddings.shape().dims::<4>();
    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual_shape[0], 2);
    assert_eq!(actual.layout.tile_count(), expected.layout.tile_count());

    let expected = expected
        .embeddings
        .into_data()
        .to_vec::<f32>()
        .expect("expected embeddings");
    let actual = actual
        .embeddings
        .into_data()
        .to_vec::<f32>()
        .expect("actual embeddings");
    let diff = max_abs_diff(&actual, &expected);
    assert!(
        diff <= 1.0e-6,
        "batched tiled embedding drift: max_abs_diff={diff}"
    );
}

#[test]
fn tiled_trace_batching_preserves_batch_order() {
    for case in [batched_tiled_case(), batched_anyres_tiled_case()] {
        assert_tiled_trace_batching_preserves_batch_order(case);
    }
}

fn assert_tiled_trace_batching_preserves_batch_order(case: ParityCase) {
    let device = Default::default();
    let single_tile_batches =
        deterministic_pipeline::<CpuBackend>(case, &device).with_tile_batch_size(1);
    let multi_tile_batches =
        deterministic_pipeline::<CpuBackend>(case, &device).with_tile_batch_size(4);
    let video = deterministic_video::<CpuBackend>(2, 2, 3, case.height, case.width, &device);

    let expected = single_tile_batches.trace_video_with_mode(video.clone(), 2, case.mode);
    let actual = multi_tile_batches.trace_video_with_mode(video, 2, case.mode);

    assert_eq!(actual.len(), 2);
    assert_traces_match("ndarray", case, &actual, &expected, 1.0e-6);
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
    expected: &PipelineOutput,
    tolerance: f32,
) {
    let actual = pipeline_output::<B>(case, device);
    let diff = max_abs_diff(&actual.embeddings, &expected.embeddings);
    assert!(
        diff <= tolerance,
        "{backend} {} embedding drift: max_abs_diff={diff}, tolerance={tolerance}",
        case.name
    );

    assert_trace_shapes(backend, case, &actual.traces);
    assert_trace_confidences(backend, &actual.traces);
    assert_traces_match(backend, case, &actual.traces, &expected.traces, tolerance);
    B::sync(device).expect("backend sync");
}

fn pipeline_output<B: Backend>(case: ParityCase, device: &B::Device) -> PipelineOutput {
    let pipeline = deterministic_pipeline::<B>(case, device);
    let embeddings_video = deterministic_video::<B>(1, 2, 3, case.height, case.width, device);
    let output = pipeline.embed_video_with_mode(embeddings_video, case.mode);
    match case.mode {
        AutoGazeInferenceMode::ResizeToModelInput => assert_eq!(output.layout.tile_count(), 1),
        AutoGazeInferenceMode::TiledResizeToGrid { .. }
        | AutoGazeInferenceMode::TiledFullResolution { .. } => {
            assert!(
                output.layout.tile_count() > 1,
                "tiled case should exercise multiple tiles"
            );
        }
    }
    let embeddings = output
        .embeddings
        .into_data()
        .to_vec::<f32>()
        .expect("embedding vec");
    let video = deterministic_video::<B>(1, 2, 3, case.height, case.width, device);
    let traces = pipeline.trace_video_with_mode(video, 2, case.mode);
    PipelineOutput { embeddings, traces }
}

fn deterministic_pipeline<B: Backend>(case: ParityCase, device: &B::Device) -> AutoGazePipeline<B> {
    let config = tiny_config(case);
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::new(&config, device).map(&mut mapper);
    AutoGazePipeline::new(model).with_max_gaze_tokens_each_frame(2)
}

fn tiny_config(case: ParityCase) -> AutoGazeConfig {
    let kernel_size = 8;
    let hidden = 8;
    let heads = 2;
    AutoGazeConfig {
        scales: case.scales.to_string(),
        max_num_frames: 2,
        num_vision_tokens_each_frame: case.num_vision_tokens_each_frame,
        gaze_model_config: GazeModelConfig {
            input_img_size: case.model_input,
            num_vision_tokens_each_frame: case.num_vision_tokens_each_frame,
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
                num_tokens: case.connector_tokens,
            },
            gaze_decoder_config: GazeDecoderConfig {
                vocab_size: case.num_vision_tokens_each_frame + 1,
                hidden_size: hidden,
                intermediate_size: hidden * 2,
                num_hidden_layers: 1,
                num_attention_heads: heads,
                num_key_value_heads: heads,
                max_position_embeddings: 512,
                bos_token_id: 0,
                eos_token_id: case.num_vision_tokens_each_frame as i64,
                head_dim: hidden / heads,
                num_multi_token_pred: 2,
                ..GazeDecoderConfig::default()
            },
        },
        ..AutoGazeConfig::default()
    }
}

fn batched_tiled_case() -> ParityCase {
    ParityCase {
        name: "multiscale-tile-batched",
        model_input: 16,
        scales: "8+16",
        connector_tokens: 4,
        num_vision_tokens_each_frame: 5,
        height: 32,
        width: 48,
        mode: AutoGazeInferenceMode::TiledFullResolution {
            tile_size: 16,
            stride: 16,
        },
    }
}

fn batched_anyres_tiled_case() -> ParityCase {
    ParityCase {
        name: "multiscale-anyres-tile-batched",
        model_input: 16,
        scales: "8+16",
        connector_tokens: 4,
        num_vision_tokens_each_frame: 5,
        height: 31,
        width: 47,
        mode: AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
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

fn assert_trace_shapes(backend: &str, case: ParityCase, traces: &[FrameFixationTrace]) {
    assert_eq!(traces.len(), 1, "{backend} should preserve batch size");
    assert_eq!(
        traces[0].len(),
        2,
        "{backend} should emit one trace set per frame"
    );
    let expected_budget = case.mode.fixation_budget(2, case.height, case.width);
    for (frame_idx, set) in traces[0].frames.iter().enumerate() {
        assert_eq!(
            set.points.len(),
            expected_budget,
            "{backend} {} frame {frame_idx} should keep the configured fixation budget",
            case.name
        );
        for point in &set.points {
            assert_valid_point(backend, case, *point);
        }
    }
}

fn assert_valid_point(backend: &str, case: ParityCase, point: FixationPoint) {
    assert!(
        point.x.is_finite() && (0.0..=1.0).contains(&point.x),
        "{backend} {} emitted invalid x: {}",
        case.name,
        point.x
    );
    assert!(
        point.y.is_finite() && (0.0..=1.0).contains(&point.y),
        "{backend} {} emitted invalid y: {}",
        case.name,
        point.y
    );
    assert!(
        point.cell_width().is_finite() && point.cell_width() > 0.0 && point.cell_width() <= 1.0,
        "{backend} {} emitted invalid cell width: {}",
        case.name,
        point.cell_width()
    );
    assert!(
        point.cell_height().is_finite() && point.cell_height() > 0.0 && point.cell_height() <= 1.0,
        "{backend} {} emitted invalid cell height: {}",
        case.name,
        point.cell_height()
    );
    if point.confidence > 0.0 && case.scales.contains('+') {
        assert!(
            point.cell_grid().is_some(),
            "{backend} {} lost multi-scale grid metadata for positive point",
            case.name
        );
    }
}

fn assert_traces_match(
    backend: &str,
    case: ParityCase,
    actual: &[FrameFixationTrace],
    expected: &[FrameFixationTrace],
    tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{backend} trace batch drift");
    for (trace_idx, (actual_trace, expected_trace)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual_trace.frames.len(),
            expected_trace.frames.len(),
            "{backend} {} trace {trace_idx} frame count drift",
            case.name
        );
        for (frame_idx, (actual_set, expected_set)) in actual_trace
            .frames
            .iter()
            .zip(&expected_trace.frames)
            .enumerate()
        {
            assert!(
                (actual_set.stop_probability - expected_set.stop_probability).abs() <= tolerance,
                "{backend} {} frame {frame_idx} stop probability drift: actual={} expected={}",
                case.name,
                actual_set.stop_probability,
                expected_set.stop_probability
            );
            assert_eq!(
                actual_set.points.len(),
                expected_set.points.len(),
                "{backend} {} frame {frame_idx} point count drift",
                case.name
            );
            for (point_idx, (actual_point, expected_point)) in actual_set
                .points
                .iter()
                .zip(&expected_set.points)
                .enumerate()
            {
                assert_point_matches(
                    backend,
                    case,
                    frame_idx,
                    point_idx,
                    *actual_point,
                    *expected_point,
                    tolerance,
                );
            }
        }
    }
}

fn assert_point_matches(
    backend: &str,
    case: ParityCase,
    frame_idx: usize,
    point_idx: usize,
    actual: FixationPoint,
    expected: FixationPoint,
    tolerance: f32,
) {
    let context = PointAssertContext {
        backend,
        case,
        frame_idx,
        point_idx,
    };
    assert_close(actual.x, expected.x, 1.0e-6, context, "x");
    assert_close(actual.y, expected.y, 1.0e-6, context, "y");
    assert_close(
        actual.cell_width(),
        expected.cell_width(),
        1.0e-6,
        context,
        "cell_width",
    );
    assert_close(
        actual.cell_height(),
        expected.cell_height(),
        1.0e-6,
        context,
        "cell_height",
    );
    assert_close(
        actual.confidence,
        expected.confidence,
        tolerance,
        context,
        "confidence",
    );
    assert_eq!(
        actual.cell_grid(),
        expected.cell_grid(),
        "{backend} {} frame {frame_idx} point {point_idx} grid drift",
        case.name
    );
}

#[derive(Clone, Copy)]
struct PointAssertContext<'a> {
    backend: &'a str,
    case: ParityCase,
    frame_idx: usize,
    point_idx: usize,
}

fn assert_close(
    actual: f32,
    expected: f32,
    tolerance: f32,
    context: PointAssertContext<'_>,
    field: &str,
) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{} {} frame {} point {} {field} drift: actual={actual} expected={expected} tolerance={tolerance}",
        context.backend,
        context.case.name,
        context.frame_idx,
        context.point_idx
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
