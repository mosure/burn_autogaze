mod config;
mod model;
mod pipeline;
mod safetensors_io;
mod teacher;
mod trace;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use config::{
    AutoGazeConfig, ConnectorConfig, GazeDecoderConfig, GazeModelConfig, VisionModelConfig,
};
pub use model::{
    AutoGazeCausalLmOutput, AutoGazeGazingModel, AutoGazeGenerateOutput, AutoGazeLoadOptions,
    Connector, Conv3dBlockForStreaming, NativeAutoGazeModel, ShallowVideoConvNet,
};
pub use pipeline::{
    AutoGazeClipShape, AutoGazeEmbedOutput, AutoGazeInferenceMode, AutoGazePipeline, AutoGazeTile,
    AutoGazeTileLayout,
};
pub use safetensors_io::AutoGazeTraceStore;
pub use teacher::AutoGazeTeacher;
pub use trace::{FixationPoint, FixationSet, FrameFixationTrace};
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
