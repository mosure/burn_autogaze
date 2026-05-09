use crate::{
    AutoGazeInferenceMode, AutoGazePipeline, AutoGazeRgbaClipShape, FrameFixationTrace,
    ImagePyramidLevel, ImagePyramidMaskOptions, ImagePyramidTokens, frame_fixation_masks_tensor,
    rgba_clip_to_tensor, tokenize_masked_image_pyramid,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::VecDeque;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeTensorClipShape {
    pub batch: usize,
    pub clip_len: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl AutoGazeTensorClipShape {
    pub const fn new(
        batch: usize,
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        Self {
            batch,
            clip_len,
            channels,
            height,
            width,
        }
    }

    pub fn from_tensor<B: Backend>(tensor: &Tensor<B, 5>) -> Self {
        let [batch, clip_len, channels, height, width] = tensor.shape().dims::<5>();
        Self::new(batch, clip_len, channels, height, width)
    }

    pub const fn num_values(&self) -> usize {
        self.batch * self.clip_len * self.channels * self.height * self.width
    }

    pub const fn is_nonzero(&self) -> bool {
        self.batch > 0
            && self.clip_len > 0
            && self.channels > 0
            && self.height > 0
            && self.width > 0
    }
}

pub struct AutoGazeTensorClip<B: Backend> {
    tensor: Tensor<B, 5>,
    shape: AutoGazeTensorClipShape,
}

impl<B: Backend> AutoGazeTensorClip<B> {
    pub fn new(tensor: Tensor<B, 5>) -> Result<Self> {
        let shape = AutoGazeTensorClipShape::from_tensor(&tensor);
        ensure!(
            shape.is_nonzero(),
            "AutoGaze tensor clip dimensions must be nonzero"
        );
        Ok(Self { tensor, shape })
    }

    pub const fn shape(&self) -> AutoGazeTensorClipShape {
        self.shape
    }

    pub const fn tensor(&self) -> &Tensor<B, 5> {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor<B, 5> {
        self.tensor
    }
}

pub struct AutoGazeRgbaClip {
    rgba: Vec<u8>,
    shape: AutoGazeRgbaClipShape,
}

impl AutoGazeRgbaClip {
    pub fn new(rgba: Vec<u8>, shape: AutoGazeRgbaClipShape) -> Result<Self> {
        ensure!(
            shape.clip_len > 0 && shape.height > 0 && shape.width > 0,
            "RGBA clip dimensions must be nonzero"
        );
        let expected_len = shape
            .clip_len
            .checked_mul(shape.height)
            .and_then(|values| values.checked_mul(shape.width))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("RGBA clip byte length overflow"))?;
        ensure!(
            rgba.len() == expected_len,
            "expected {expected_len} RGBA bytes, got {}",
            rgba.len()
        );
        Ok(Self { rgba, shape })
    }

    pub const fn shape(&self) -> AutoGazeRgbaClipShape {
        self.shape
    }

    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }

    pub fn into_tensor_clip<B: Backend>(self, device: &B::Device) -> Result<AutoGazeTensorClip<B>> {
        AutoGazeTensorClip::new(rgba_clip_to_tensor::<B>(&self.rgba, self.shape, device)?)
    }
}

pub trait AutoGazeInputNode<B: Backend> {
    fn next_clip(&mut self, device: &B::Device) -> Result<Option<AutoGazeTensorClip<B>>>;
}

pub trait AutoGazeOutputNode<B: Backend> {
    fn write(&mut self, packet: AutoGazePipelinePacket<B>) -> Result<()>;
}

pub struct AutoGazePipelinePacket<B: Backend> {
    pub clip: AutoGazeTensorClip<B>,
    pub traces: Vec<FrameFixationTrace>,
    pub mode: AutoGazeInferenceMode,
    pub top_k: usize,
}

impl<B: Backend> AutoGazePipelinePacket<B> {
    pub fn frame_tensor(&self, frame_index: usize) -> Result<Tensor<B, 4>> {
        let shape = self.clip.shape();
        ensure!(
            frame_index < shape.clip_len,
            "frame index {frame_index} is outside clip length {}",
            shape.clip_len
        );
        Ok(self
            .clip
            .tensor()
            .clone()
            .slice_dim(1, frame_index..frame_index + 1)
            .squeeze_dim::<4>(1))
    }

    pub fn frame_mask(&self, frame_index: usize, device: &B::Device) -> Result<Tensor<B, 4>> {
        let shape = self.clip.shape();
        frame_fixation_masks_tensor(&self.traces, frame_index, shape.height, shape.width, device)
    }

    pub fn frame_pyramid_tokens(
        &self,
        frame_index: usize,
        levels: &[ImagePyramidLevel],
        options: ImagePyramidMaskOptions,
        device: &B::Device,
    ) -> Result<ImagePyramidTokens<B>> {
        tokenize_masked_image_pyramid(
            self.frame_tensor(frame_index)?,
            self.frame_mask(frame_index, device)?,
            levels,
            options,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeTensorPipelineConfig {
    pub mode: AutoGazeInferenceMode,
    pub top_k: usize,
}

impl Default for AutoGazeTensorPipelineConfig {
    fn default() -> Self {
        Self {
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 1,
        }
    }
}

pub struct AutoGazeTensorPipeline<B: Backend, I, O> {
    pub pipeline: AutoGazePipeline<B>,
    pub input: I,
    pub output: O,
    pub config: AutoGazeTensorPipelineConfig,
}

impl<B, I, O> AutoGazeTensorPipeline<B, I, O>
where
    B: Backend,
    I: AutoGazeInputNode<B>,
    O: AutoGazeOutputNode<B>,
{
    pub fn new(pipeline: AutoGazePipeline<B>, input: I, output: O) -> Self {
        Self {
            pipeline,
            input,
            output,
            config: AutoGazeTensorPipelineConfig::default(),
        }
    }

    pub fn with_config(mut self, config: AutoGazeTensorPipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn run_next(&mut self, device: &B::Device) -> Result<bool> {
        let Some(clip) = self.input.next_clip(device)? else {
            return Ok(false);
        };
        let traces = self.pipeline.trace_video_with_mode(
            clip.tensor().clone(),
            self.config.top_k,
            self.config.mode,
        );
        self.output.write(AutoGazePipelinePacket {
            clip,
            traces,
            mode: self.config.mode,
            top_k: self.config.top_k,
        })?;
        Ok(true)
    }
}

#[derive(Default)]
pub struct TensorClipInput<B: Backend> {
    clips: VecDeque<AutoGazeTensorClip<B>>,
}

impl<B: Backend> TensorClipInput<B> {
    pub fn new() -> Self {
        Self {
            clips: VecDeque::new(),
        }
    }

    pub fn with_clip(mut self, clip: AutoGazeTensorClip<B>) -> Self {
        self.push_clip(clip);
        self
    }

    pub fn push_clip(&mut self, clip: AutoGazeTensorClip<B>) {
        self.clips.push_back(clip);
    }

    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }
}

impl<B: Backend> AutoGazeInputNode<B> for TensorClipInput<B> {
    fn next_clip(&mut self, _device: &B::Device) -> Result<Option<AutoGazeTensorClip<B>>> {
        Ok(self.clips.pop_front())
    }
}

#[derive(Default)]
pub struct RgbaClipInput {
    clips: VecDeque<AutoGazeRgbaClip>,
}

impl RgbaClipInput {
    pub fn new() -> Self {
        Self {
            clips: VecDeque::new(),
        }
    }

    pub fn with_clip(mut self, clip: AutoGazeRgbaClip) -> Self {
        self.push_clip(clip);
        self
    }

    pub fn push_clip(&mut self, clip: AutoGazeRgbaClip) {
        self.clips.push_back(clip);
    }

    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }
}

impl<B: Backend> AutoGazeInputNode<B> for RgbaClipInput {
    fn next_clip(&mut self, device: &B::Device) -> Result<Option<AutoGazeTensorClip<B>>> {
        self.clips
            .pop_front()
            .map(|clip| clip.into_tensor_clip(device))
            .transpose()
    }
}

#[derive(Default)]
pub struct VecOutputNode<B: Backend> {
    packets: Vec<AutoGazePipelinePacket<B>>,
}

impl<B: Backend> VecOutputNode<B> {
    pub fn new() -> Self {
        Self {
            packets: Vec::new(),
        }
    }

    pub fn packets(&self) -> &[AutoGazePipelinePacket<B>] {
        &self.packets
    }

    pub fn into_packets(self) -> Vec<AutoGazePipelinePacket<B>> {
        self.packets
    }
}

impl<B: Backend> AutoGazeOutputNode<B> for VecOutputNode<B> {
    fn write(&mut self, packet: AutoGazePipelinePacket<B>) -> Result<()> {
        self.packets.push(packet);
        Ok(())
    }
}

pub struct FnOutputNode<B: Backend, F> {
    f: F,
    _backend: PhantomData<B>,
}

impl<B: Backend, F> FnOutputNode<B, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _backend: PhantomData,
        }
    }
}

impl<B, F> AutoGazeOutputNode<B> for FnOutputNode<B, F>
where
    B: Backend,
    F: FnMut(AutoGazePipelinePacket<B>) -> Result<()>,
{
    fn write(&mut self, packet: AutoGazePipelinePacket<B>) -> Result<()> {
        (self.f)(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn tensor_clip_input_preserves_burn_tensor_shape() {
        let device = Default::default();
        let tensor =
            Tensor::<B, 5>::from_data(TensorData::new(vec![1.0; 6], [1, 1, 1, 2, 3]), &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let shape = clip.shape();
        let mut input = TensorClipInput::<B>::new().with_clip(clip);

        assert_eq!(shape, AutoGazeTensorClipShape::new(1, 1, 1, 2, 3));
        assert_eq!(input.len(), 1);
        assert_eq!(input.next_clip(&device).unwrap().unwrap().shape(), shape);
        assert!(input.next_clip(&device).unwrap().is_none());
    }

    #[test]
    fn rgba_clip_input_converts_to_autogaze_tensor_node() {
        let device = Default::default();
        let rgba = vec![10, 20, 30, 255, 40, 50, 60, 0];
        let shape = AutoGazeRgbaClipShape::new(1, 1, 2);
        let clip = AutoGazeRgbaClip::new(rgba, shape).expect("rgba clip");
        let mut input = RgbaClipInput::new().with_clip(clip);

        let tensor_clip: AutoGazeTensorClip<B> =
            input.next_clip(&device).unwrap().expect("tensor clip");

        assert_eq!(
            tensor_clip.shape(),
            AutoGazeTensorClipShape::new(1, 1, 3, 1, 2)
        );
    }

    #[test]
    fn vec_output_node_collects_pipeline_packets() {
        let device = Default::default();
        let tensor =
            Tensor::<B, 5>::from_data(TensorData::new(vec![1.0], [1, 1, 1, 1, 1]), &device);
        let packet = AutoGazePipelinePacket {
            clip: AutoGazeTensorClip::new(tensor).expect("clip"),
            traces: Vec::new(),
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 1,
        };
        let mut output = VecOutputNode::<B>::new();

        output.write(packet).expect("write");

        assert_eq!(output.packets().len(), 1);
    }

    #[test]
    fn packet_exposes_frame_mask_and_pyramid_tokens_for_output_nodes() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 1, 1, 2]),
            &device,
        );
        let point = crate::FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let packet = AutoGazePipelinePacket {
            clip: AutoGazeTensorClip::new(tensor).expect("clip"),
            traces: vec![FrameFixationTrace::new(vec![
                crate::FixationSet::new(vec![], 0.0, 1),
                crate::FixationSet::new(vec![point], 1.0, 1),
            ])],
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 1,
        };

        let frame = packet.frame_tensor(1).expect("frame");
        let mask = packet.frame_mask(1, &device).expect("mask");
        let tokens = packet
            .frame_pyramid_tokens(
                1,
                &[ImagePyramidLevel::new(1, 2)],
                Default::default(),
                &device,
            )
            .expect("tokens");

        assert_eq!(frame.into_data().to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
        assert_eq!(mask.into_data().to_vec::<f32>().unwrap(), vec![1.0, 0.0]);
        assert_eq!(
            tokens.weights.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0]
        );
    }
}
