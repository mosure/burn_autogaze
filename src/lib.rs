mod config;
mod metrics;
mod model;
mod nodes;
mod pipeline;
mod pyramid;
mod readout;
mod runtime;
mod safetensors_io;
mod teacher;
mod trace;
mod visualization;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use config::{
    AutoGazeConfig, ConnectorConfig, GazeDecoderConfig, GazeModelConfig, VisionModelConfig,
};
pub use metrics::{
    AutoGazeEmaMetric, AutoGazeGazeRatioStats, AutoGazePsnrStats, DEFAULT_METRIC_EMA_ALPHA,
    ema_metric, format_fps, format_gaze_ratio_percent, format_psnr_db, fps_from_millis,
    sanitize_gaze_ratio,
};
pub use model::{
    AutoGazeCausalLmOutput, AutoGazeGazingModel, AutoGazeGenerateOutput, AutoGazeLoadOptions,
    AutoGazeScaleTokenMask, AutoGazeStreamingCache, Connector, Conv3dBlockForStreaming,
    NativeAutoGazeModel, ShallowVideoConvNet,
};
pub use nodes::{
    AutoGazeInputNode, AutoGazeOutputNode, AutoGazePipelinePacket, AutoGazeRgbaClip,
    AutoGazeRgbaFrameClip, AutoGazeRgbaFrameQueue, AutoGazeTensorClip, AutoGazeTensorClipShape,
    AutoGazeTensorPipeline, AutoGazeTensorPipelineConfig, FnOutputNode, RgbaClipInput,
    TensorClipInput, VecOutputNode,
};
pub use pipeline::{
    AUTO_GAZE_IMAGE_MEAN, AUTO_GAZE_IMAGE_STD, AUTO_GAZE_PROCESSOR_SHORT_EDGE,
    AUTO_GAZE_RESCALE_FACTOR, AutoGazeClipShape, AutoGazeEmbedOutput, AutoGazeInferenceMode,
    AutoGazePipeline, AutoGazePipelineOptions, AutoGazePreparedRun, AutoGazeReadoutRunOutput,
    AutoGazeRgbaClipShape, AutoGazeTaskLossOption, AutoGazeTile, AutoGazeTileLayout,
    AutoGazeTraceRunOutput, last_rgba_frame, prepare_rgba_clip_for_trace,
    resize_dimensions_preserving_aspect, resize_rgba_frame_to_dimensions,
    resize_video_shortest_edge, rgba_clip_to_inference_tensor, rgba_clip_to_processor_tensor,
    rgba_clip_to_tensor, video_frame_tensor,
};
pub use pyramid::{
    ImagePyramidLevel, ImagePyramidMask, ImagePyramidMaskOptions, ImagePyramidTokens,
    SparseImagePyramidTokens, apply_image_mask, fixation_image_mask_tensor,
    frame_fixation_masks_tensor, image_pyramid_masks, sparsify_image_pyramid_tokens,
    tokenize_masked_image_pyramid,
};
pub use readout::{
    SparseReadoutGrid, SparseReadoutOptions, SparseReadoutRect, SparseVideoPatchGeometry,
    SparseVideoReadoutGrid, SparseVideoReadoutOptions, SparseVideoReadoutProjection,
    batched_video_readout_tokens_to_coord_tensor, batched_video_readout_tokens_to_coords,
    fixation_points_to_readout_rects, fixation_points_to_readout_tokens,
    frame_readout_rects_to_video_coord_tensor, frame_readout_rects_to_video_coords,
    frame_readout_rects_to_video_tokens, frame_readout_tokens_to_video_coord_tensor,
    frame_readout_tokens_to_video_coords, frame_readout_tokens_to_video_tokens,
    generated_frame_readout_rects, generated_frame_readout_tokens,
    generated_to_frame_readout_rects, generated_to_frame_readout_tokens,
    generated_to_video_readout_coord_tensor, generated_to_video_readout_coords,
    generated_to_video_readout_tokens, readout_rects_to_tokens, trace_frame_readout_rects,
    trace_frame_readout_tokens, trace_to_frame_readout_rects, trace_to_frame_readout_tokens,
    trace_to_video_readout_coord_tensor, trace_to_video_readout_coords,
    trace_to_video_readout_tokens, video_readout_coords_to_tensor,
    video_readout_tokens_to_coord_tensor, video_readout_tokens_to_coords,
};
pub use runtime::{
    AutoGazeInferenceSequencer, AutoGazeRealtimePolicy, DEFAULT_BLEND_ALPHA,
    DEFAULT_KEYFRAME_DURATION, DEFAULT_MAX_IN_FLIGHT, DEFAULT_MODEL_GENERATION_BUDGET,
    DEFAULT_REALTIME_FRAMES_PER_CLIP, DEFAULT_REALTIME_TOP_K, DEFAULT_TILED_FRAMES_PER_CLIP,
    DEFAULT_TILED_MAX_GAZE_TOKENS, DEFAULT_TILED_TILE_BATCH_SIZE, DEFAULT_TILED_TOP_K,
    should_use_streaming_cache,
};
pub use safetensors_io::AutoGazeTraceStore;
pub use teacher::AutoGazeTeacher;
pub use trace::{FixationBounds, FixationPoint, FixationSet, FrameFixationTrace};
pub use visualization::{
    AutoGazeSparseUpdatePlan, AutoGazeTensorInterframePath, AutoGazeTensorVisualization,
    AutoGazeTensorVisualizationOptions, AutoGazeTensorVisualizationPanels,
    AutoGazeTensorVisualizationState, AutoGazeVisualization, AutoGazeVisualizationMode,
    AutoGazeVisualizationPanels, AutoGazeVisualizationState,
    DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO, DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
    FixationPixelRect, copy_sparse_update_rgba, copy_sparse_update_tensor, fixation_alpha_mask,
    fixation_cell_rects, fixation_effective_alpha_mask, fixation_effective_cell_rects,
    fixation_effective_scale_mask_rgba, fixation_effective_sparse_update_plan,
    fixation_rect_union_pixel_count, fixation_scale_mask_rgba, fixation_sparse_update_plan,
    normalized_rgb_clip_to_unit_rgba_tensor, visualize_fixations_rgba,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm::*;

#[cfg(feature = "ndarray")]
pub type NdArrayAutoGazeModel = NativeAutoGazeModel<burn::backend::NdArray<f32>>;

#[cfg(feature = "ndarray")]
pub type NdArrayAutoGazePipeline = AutoGazePipeline<burn::backend::NdArray<f32>>;

#[cfg(feature = "cuda")]
pub type CudaAutoGazeModel = NativeAutoGazeModel<burn::backend::Cuda<f32, i32>>;

#[cfg(feature = "cuda")]
pub type CudaAutoGazePipeline = AutoGazePipeline<burn::backend::Cuda<f32, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuAutoGazeModel = NativeAutoGazeModel<burn::backend::Wgpu<f32, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuAutoGazePipeline = AutoGazePipeline<burn::backend::Wgpu<f32, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuAutoGazeModel = NativeAutoGazeModel<burn::backend::WebGpu<f32, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuAutoGazePipeline = AutoGazePipeline<burn::backend::WebGpu<f32, i32>>;
