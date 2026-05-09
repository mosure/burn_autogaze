mod config;
mod model;
mod nodes;
mod pipeline;
mod pyramid;
mod safetensors_io;
mod teacher;
mod trace;
mod visualization;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use config::{
    AutoGazeConfig, ConnectorConfig, GazeDecoderConfig, GazeModelConfig, VisionModelConfig,
};
pub use model::{
    AutoGazeCausalLmOutput, AutoGazeGazingModel, AutoGazeGenerateOutput, AutoGazeLoadOptions,
    Connector, Conv3dBlockForStreaming, NativeAutoGazeModel, ShallowVideoConvNet,
};
pub use nodes::{
    AutoGazeInputNode, AutoGazeOutputNode, AutoGazePipelinePacket, AutoGazeRgbaClip,
    AutoGazeTensorClip, AutoGazeTensorClipShape, AutoGazeTensorPipeline,
    AutoGazeTensorPipelineConfig, FnOutputNode, RgbaClipInput, TensorClipInput, VecOutputNode,
};
pub use pipeline::{
    AUTO_GAZE_IMAGE_MEAN, AUTO_GAZE_IMAGE_STD, AUTO_GAZE_RESCALE_FACTOR, AutoGazeClipShape,
    AutoGazeEmbedOutput, AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape,
    AutoGazeTile, AutoGazeTileLayout, rgba_clip_to_tensor,
};
pub use pyramid::{
    ImagePyramidLevel, ImagePyramidMask, ImagePyramidMaskOptions, ImagePyramidTokens,
    SparseImagePyramidTokens, apply_image_mask, fixation_image_mask_tensor,
    frame_fixation_masks_tensor, image_pyramid_masks, sparsify_image_pyramid_tokens,
    tokenize_masked_image_pyramid,
};
pub use safetensors_io::AutoGazeTraceStore;
pub use teacher::AutoGazeTeacher;
pub use trace::{FixationBounds, FixationPoint, FixationSet, FrameFixationTrace};
pub use visualization::{
    AutoGazeVisualization, AutoGazeVisualizationMode, AutoGazeVisualizationState,
    fixation_alpha_mask, fixation_scale_mask_rgba, visualize_fixations_rgba,
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
