use crate::model::generated_to_frame_points;
use crate::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeInferenceMode, AutoGazePipeline,
    AutoGazeRgbaClipShape, AutoGazeSparseMaskSource, DEFAULT_REALTIME_TOP_K, FixationPoint,
    FrameFixationTrace, ImagePyramidLevel, ImagePyramidMaskOptions, ImagePyramidTokens,
    SparseVideoReadoutGrid, SparseVideoReadoutOptions, SparseVideoReadoutProjection,
    batched_video_readout_tokens_to_coords, fixation_points_to_readout_rects,
    fixation_points_to_readout_tokens, frame_fixation_masks_tensor,
    frame_readout_tokens_to_video_tokens, generated_frame_readout_rects,
    generated_frame_readout_tokens, generated_to_video_readout_tokens, patch_diff_points_to_traces,
    patch_diff_readout_points, patch_diff_readout_points_async, rgba_clip_to_inference_tensor,
    rgba_clip_to_tensor, tokenize_masked_image_pyramid, video_readout_coords_to_tensor,
};
use crate::{
    SparseReadoutGrid, SparseReadoutOptions, SparseReadoutRect, trace_frame_readout_rects,
    trace_frame_readout_tokens, trace_to_video_readout_tokens,
};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use image::RgbaImage;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

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

    pub fn into_inference_tensor_clip<B: Backend>(
        self,
        mode: AutoGazeInferenceMode,
        device: &B::Device,
    ) -> Result<AutoGazeTensorClip<B>> {
        AutoGazeTensorClip::new(rgba_clip_to_inference_tensor::<B>(
            &self.rgba, self.shape, mode, device,
        )?)
    }
}

pub struct AutoGazeRgbaFrameClip {
    width: usize,
    height: usize,
    clip_len: usize,
    rgba: Vec<u8>,
}

impl AutoGazeRgbaFrameClip {
    pub fn new(rgba: Vec<u8>, width: usize, height: usize, clip_len: usize) -> Result<Self> {
        let shape = AutoGazeRgbaClipShape::new(clip_len, height, width);
        let rgba = AutoGazeRgbaClip::new(rgba, shape)?.into_rgba();
        Ok(Self {
            width,
            height,
            clip_len,
            rgba,
        })
    }

    fn from_validated(rgba: Vec<u8>, width: usize, height: usize, clip_len: usize) -> Self {
        Self {
            width,
            height,
            clip_len,
            rgba,
        }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn clip_len(&self) -> usize {
        self.clip_len
    }

    pub const fn shape(&self) -> AutoGazeRgbaClipShape {
        AutoGazeRgbaClipShape::new(self.clip_len, self.height, self.width)
    }

    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    pub fn rgba_capacity(&self) -> usize {
        self.rgba.capacity()
    }

    pub fn last_frame_rgba(&self) -> Result<&[u8]> {
        crate::last_rgba_frame(&self.rgba, self.shape())
    }

    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }

    pub fn into_rgba_clip(self) -> Result<AutoGazeRgbaClip> {
        let shape = self.shape();
        AutoGazeRgbaClip::new(self.rgba, shape)
    }

    pub fn into_parts(self) -> (Vec<u8>, usize, usize, usize) {
        (self.rgba, self.width, self.height, self.clip_len)
    }
}

#[derive(Default)]
pub struct AutoGazeRgbaFrameQueue {
    width: u32,
    height: u32,
    frames: VecDeque<Arc<RgbaImage>>,
    spare_clip_buffers: Vec<Vec<u8>>,
    max_spare_clip_buffers: usize,
}

impl AutoGazeRgbaFrameQueue {
    pub fn new(max_spare_clip_buffers: usize) -> Self {
        Self {
            max_spare_clip_buffers,
            ..Self::default()
        }
    }

    pub fn push(&mut self, frame: Arc<RgbaImage>, max_len: usize) {
        let max_len = max_len.max(1);
        let (width, height) = frame.dimensions();
        if self.width != width || self.height != height {
            self.reset();
            self.width = width;
            self.height = height;
        }

        self.frames.push_back(frame);
        while self.frames.len() > max_len {
            self.frames.pop_front();
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
        self.spare_clip_buffers.clear();
        self.width = 0;
        self.height = 0;
    }

    pub fn latest(&self) -> Option<&RgbaImage> {
        self.frames.back().map(AsRef::as_ref)
    }

    pub fn build_clip(&mut self, max_len: usize) -> Result<Option<AutoGazeRgbaFrameClip>> {
        let max_len = max_len.max(1);
        if self.frames.len() != max_len {
            return Ok(None);
        }
        let frames = self.frames.iter().cloned().collect::<Vec<_>>();
        self.build_clip_from_frames(max_len, frames.iter())
    }

    pub fn build_latest_clip(&mut self) -> Result<Option<AutoGazeRgbaFrameClip>> {
        let Some(frame) = self.frames.back().cloned() else {
            return Ok(None);
        };
        self.build_clip_from_frames(1, std::iter::once(&frame))
    }

    pub fn recycle_clip_buffer(&mut self, mut rgba: Vec<u8>) {
        rgba.clear();
        if self.spare_clip_buffers.len() < self.max_spare_clip_buffers {
            self.spare_clip_buffers.push(rgba);
        }
    }

    pub fn spare_clip_buffer_count(&self) -> usize {
        self.spare_clip_buffers.len()
    }

    fn build_clip_from_frames<'a>(
        &mut self,
        clip_len: usize,
        frames: impl Iterator<Item = &'a Arc<RgbaImage>>,
    ) -> Result<Option<AutoGazeRgbaFrameClip>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let frame_bytes = rgba_frame_byte_len(width, height)?;
        let required_bytes = frame_bytes
            .checked_mul(clip_len)
            .ok_or_else(|| anyhow::anyhow!("AutoGaze clip byte length overflow"))?;
        let mut rgba = self
            .spare_clip_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(required_bytes));
        rgba.clear();
        if rgba.capacity() < required_bytes {
            rgba.reserve_exact(required_bytes - rgba.capacity());
        }

        let mut copied = 0usize;
        for frame in frames {
            ensure!(
                frame.width() as usize == width && frame.height() as usize == height,
                "AutoGaze clip frame dimensions changed"
            );
            ensure!(
                frame.as_raw().len() == frame_bytes,
                "expected {frame_bytes} RGBA bytes for {width}x{height}, got {}",
                frame.as_raw().len()
            );
            rgba.extend_from_slice(frame.as_raw());
            copied += 1;
        }
        ensure!(
            copied == clip_len,
            "expected {clip_len} frames for AutoGaze clip, got {copied}"
        );

        Ok(Some(AutoGazeRgbaFrameClip::from_validated(
            rgba, width, height, clip_len,
        )))
    }
}

fn rgba_frame_byte_len(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("AutoGaze frame byte length overflow"))
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
    pub generated: Option<AutoGazeGenerateOutput>,
    pub readout_points: Option<Vec<Vec<Vec<FixationPoint>>>>,
    pub model_config: AutoGazeConfig,
    pub mode: AutoGazeInferenceMode,
    pub top_k: usize,
}

impl<B: Backend> AutoGazePipelinePacket<B> {
    pub fn has_traces(&self) -> bool {
        !self.traces.is_empty()
    }

    pub const fn has_generated(&self) -> bool {
        self.generated.is_some()
    }

    pub const fn has_readout_points(&self) -> bool {
        self.readout_points.is_some()
    }

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
        ensure!(
            self.has_traces(),
            "AutoGaze traces are disabled for this packet; set AutoGazeTensorPipelineConfig::emit_traces to true"
        );
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

    pub fn frame_readout_tokens(
        &self,
        frame_index: usize,
        grid: SparseReadoutGrid,
        options: SparseReadoutOptions,
    ) -> Result<Vec<Vec<usize>>> {
        if self.has_traces() {
            return self
                .traces
                .iter()
                .map(|trace| trace_frame_readout_tokens(trace, frame_index, grid, options))
                .collect();
        }

        if let Some(readout_points) = &self.readout_points {
            return readout_points
                .iter()
                .map(|batch_frames| {
                    batch_frames
                        .get(frame_index)
                        .map(|points| fixation_points_to_readout_tokens(points, grid, options))
                        .unwrap_or_else(|| Ok(Vec::new()))
                })
                .collect();
        }

        let Some(generated) = &self.generated else {
            anyhow::bail!(
                "AutoGaze readout is disabled for this packet; set AutoGazeTensorPipelineConfig::emit_traces, emit_readout_points, or emit_generated to true"
            );
        };
        generated
            .gazing_pos
            .iter()
            .enumerate()
            .map(|(batch_index, _)| {
                generated_frame_readout_tokens(
                    generated,
                    &self.model_config,
                    batch_index,
                    frame_index,
                    grid,
                    options,
                )
            })
            .collect()
    }

    pub fn frame_readout_rects(
        &self,
        frame_index: usize,
        options: SparseReadoutOptions,
    ) -> Result<Vec<Vec<SparseReadoutRect>>> {
        if self.has_traces() {
            return Ok(self
                .traces
                .iter()
                .map(|trace| trace_frame_readout_rects(trace, frame_index, options))
                .collect());
        }

        if let Some(readout_points) = &self.readout_points {
            return Ok(readout_points
                .iter()
                .map(|batch_frames| {
                    batch_frames
                        .get(frame_index)
                        .map(|points| fixation_points_to_readout_rects(points, options))
                        .unwrap_or_default()
                })
                .collect());
        }

        let Some(generated) = &self.generated else {
            anyhow::bail!(
                "AutoGaze readout is disabled for this packet; set AutoGazeTensorPipelineConfig::emit_traces, emit_readout_points, or emit_generated to true"
            );
        };
        generated
            .gazing_pos
            .iter()
            .enumerate()
            .map(|(batch_index, _)| {
                generated_frame_readout_rects(
                    generated,
                    &self.model_config,
                    batch_index,
                    frame_index,
                    options,
                )
            })
            .collect()
    }

    pub fn video_readout_tokens(
        &self,
        image_grid: SparseReadoutGrid,
        video_grid: SparseVideoReadoutGrid,
        readout_options: SparseReadoutOptions,
        video_options: SparseVideoReadoutOptions,
    ) -> Result<Vec<Vec<usize>>> {
        if self.has_traces() {
            return self
                .traces
                .iter()
                .map(|trace| {
                    trace_to_video_readout_tokens(
                        trace,
                        image_grid,
                        video_grid,
                        readout_options,
                        video_options,
                    )
                })
                .collect();
        }

        if let Some(readout_points) = &self.readout_points {
            return readout_points
                .iter()
                .map(|batch_frames| {
                    let frame_tokens = batch_frames
                        .iter()
                        .map(|points| {
                            fixation_points_to_readout_tokens(points, image_grid, readout_options)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    frame_readout_tokens_to_video_tokens(
                        &frame_tokens,
                        image_grid,
                        video_grid,
                        video_options,
                    )
                })
                .collect();
        }

        let Some(generated) = &self.generated else {
            anyhow::bail!(
                "AutoGaze readout is disabled for this packet; set AutoGazeTensorPipelineConfig::emit_traces, emit_readout_points, or emit_generated to true"
            );
        };
        generated
            .gazing_pos
            .iter()
            .enumerate()
            .map(|(batch_index, _)| {
                generated_to_video_readout_tokens(
                    generated,
                    &self.model_config,
                    batch_index,
                    image_grid,
                    video_grid,
                    readout_options,
                    video_options,
                )
            })
            .collect()
    }

    pub fn video_readout_tokens_with_projection(
        &self,
        projection: SparseVideoReadoutProjection,
    ) -> Result<Vec<Vec<usize>>> {
        self.video_readout_tokens(
            projection.image_grid,
            projection.video_grid,
            projection.readout_options,
            projection.video_options,
        )
    }

    /// Return flattened `[batch, temporal, row, col]` sparse-video coordinates
    /// matching `burn_flex_gmm` sparse patchify convention.
    pub fn video_readout_coords(
        &self,
        image_grid: SparseReadoutGrid,
        video_grid: SparseVideoReadoutGrid,
        readout_options: SparseReadoutOptions,
        video_options: SparseVideoReadoutOptions,
    ) -> Result<Vec<[u32; 4]>> {
        let tokens =
            self.video_readout_tokens(image_grid, video_grid, readout_options, video_options)?;
        batched_video_readout_tokens_to_coords(&tokens, video_grid)
    }

    /// Return flattened sparse-video coordinate rows using grouped projection
    /// settings.
    pub fn video_readout_coords_with_projection(
        &self,
        projection: SparseVideoReadoutProjection,
    ) -> Result<Vec<[u32; 4]>> {
        self.video_readout_coords(
            projection.image_grid,
            projection.video_grid,
            projection.readout_options,
            projection.video_options,
        )
    }

    /// Return flattened `[batch, temporal, row, col]` sparse-video coordinates
    /// as a Burn int tensor with shape `[rows, 4]`.
    pub fn video_readout_coord_tensor(
        &self,
        image_grid: SparseReadoutGrid,
        video_grid: SparseVideoReadoutGrid,
        readout_options: SparseReadoutOptions,
        video_options: SparseVideoReadoutOptions,
        device: &B::Device,
    ) -> Result<Tensor<B, 2, Int>> {
        let coords =
            self.video_readout_coords(image_grid, video_grid, readout_options, video_options)?;
        Ok(video_readout_coords_to_tensor(&coords, device))
    }

    /// Return sparse-video coordinate rows as a Burn int tensor using grouped
    /// projection settings.
    pub fn video_readout_coord_tensor_with_projection(
        &self,
        projection: SparseVideoReadoutProjection,
        device: &B::Device,
    ) -> Result<Tensor<B, 2, Int>> {
        self.video_readout_coord_tensor(
            projection.image_grid,
            projection.video_grid,
            projection.readout_options,
            projection.video_options,
            device,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazeTensorPipelineConfig {
    /// Sparse mask source used by the graph.
    pub sparse_mask_source: AutoGazeSparseMaskSource,
    /// Inference geometry used by the graph.
    pub mode: AutoGazeInferenceMode,
    /// Display/readout lower bound for generated fixation points.
    pub top_k: usize,
    /// Emit decoded fixation traces in output packets.
    pub emit_traces: bool,
    /// Emit raw generated output in realtime resize-mode packets.
    ///
    /// Tiled modes should use `emit_readout_points` or `emit_traces` because
    /// tile-local token ids need global source-frame remapping before they are
    /// useful to output adapters.
    pub emit_generated: bool,
    /// Emit decoded fixation points for sparse readout without a full trace.
    ///
    /// This is cheaper than traces for output nodes that only need sparse
    /// readout tokens or rectangles. Tiled modes are remapped into source-frame
    /// coordinates before points are stored.
    pub emit_readout_points: bool,
}

impl Default for AutoGazeTensorPipelineConfig {
    fn default() -> Self {
        Self {
            sparse_mask_source: AutoGazeSparseMaskSource::AutoGaze,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: DEFAULT_REALTIME_TOP_K,
            emit_traces: false,
            emit_generated: false,
            emit_readout_points: false,
        }
    }
}

impl AutoGazeTensorPipelineConfig {
    pub const fn with_sparse_mask_source(mut self, source: AutoGazeSparseMaskSource) -> Self {
        self.sparse_mask_source = source;
        self
    }

    pub const fn with_mode(mut self, mode: AutoGazeInferenceMode) -> Self {
        self.mode = mode;
        self
    }

    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub const fn with_emit_traces(mut self, emit_traces: bool) -> Self {
        self.emit_traces = emit_traces;
        self
    }

    pub const fn with_emit_generated(mut self, emit_generated: bool) -> Self {
        self.emit_generated = emit_generated;
        self
    }

    pub const fn with_emit_readout_points(mut self, emit_readout_points: bool) -> Self {
        self.emit_readout_points = emit_readout_points;
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct AutoGazeTensorPacketPlan {
    config: AutoGazeTensorPipelineConfig,
    normalized_mode: AutoGazeInferenceMode,
    generation_budget: usize,
}

impl AutoGazeTensorPacketPlan {
    fn new(config: AutoGazeTensorPipelineConfig, model_generation_budget: usize) -> Result<Self> {
        let normalized_mode = config.mode.normalized();
        ensure!(
            !config.emit_generated
                || (config.sparse_mask_source == AutoGazeSparseMaskSource::AutoGaze
                    && normalized_mode == AutoGazeInferenceMode::ResizeToModelInput),
            "AutoGaze generated packets are only available in resize-to-model-input mode; enable emit_readout_points or emit_traces for tiled global readout"
        );
        Ok(Self {
            config,
            normalized_mode,
            generation_budget: model_generation_budget.max(config.top_k.max(1)),
        })
    }

    fn should_generate(&self) -> bool {
        if self.config.sparse_mask_source != AutoGazeSparseMaskSource::AutoGaze {
            return false;
        }
        self.normalized_mode == AutoGazeInferenceMode::ResizeToModelInput
            && (self.config.emit_generated
                || self.config.emit_readout_points
                || self.config.emit_traces)
    }

    fn direct_readout_required(&self, generated: Option<&AutoGazeGenerateOutput>) -> bool {
        self.config.sparse_mask_source == AutoGazeSparseMaskSource::AutoGaze
            && self.config.emit_readout_points
            && !self.config.emit_traces
            && generated.is_none()
    }

    fn should_run_patch_diff(&self) -> bool {
        self.config.sparse_mask_source.is_patch_diff()
            && (self.config.emit_readout_points || self.config.emit_traces)
    }

    fn ready_readout_points(
        &self,
        traces: &[FrameFixationTrace],
        generated: Option<&AutoGazeGenerateOutput>,
        model_config: &AutoGazeConfig,
    ) -> Option<Vec<Vec<Vec<FixationPoint>>>> {
        if !self.config.emit_readout_points {
            return None;
        }
        if self.config.emit_traces {
            return Some(traces_to_readout_points(traces));
        }
        generated.map(|generated| generated_to_frame_points(generated, model_config))
    }

    fn packet<B: Backend>(
        &self,
        clip: AutoGazeTensorClip<B>,
        model_config: AutoGazeConfig,
        generated: Option<AutoGazeGenerateOutput>,
        traces: Vec<FrameFixationTrace>,
        readout_points: Option<Vec<Vec<Vec<FixationPoint>>>>,
    ) -> AutoGazePipelinePacket<B> {
        AutoGazePipelinePacket {
            clip,
            traces,
            generated: if self.config.emit_generated {
                generated
            } else {
                None
            },
            readout_points,
            model_config,
            mode: self.config.mode,
            top_k: self.config.top_k,
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

    fn packet_plan(&self) -> Result<AutoGazeTensorPacketPlan> {
        AutoGazeTensorPacketPlan::new(self.config, self.pipeline.max_gaze_tokens_each_frame())
    }

    pub fn run_next(&mut self, device: &B::Device) -> Result<bool> {
        let Some(clip) = self.input.next_clip(device)? else {
            return Ok(false);
        };
        let plan = self.packet_plan()?;
        let model_config = self.pipeline.model().config.clone();
        if plan.should_run_patch_diff()
            && let AutoGazeSparseMaskSource::PatchDiff(config) = plan.config.sparse_mask_source
        {
            let readout = patch_diff_readout_points(clip.tensor().clone(), config)?;
            let traces = if plan.config.emit_traces {
                patch_diff_points_to_traces(&readout.points, plan.config.top_k)
            } else {
                Vec::new()
            };
            let readout_points = plan.config.emit_readout_points.then_some(readout.points);
            self.output
                .write(plan.packet(clip, model_config, None, traces, readout_points))?;
            return Ok(true);
        }
        let generated = if plan.should_generate() {
            Some(
                self.pipeline
                    .generate_with_limit(clip.tensor().clone(), plan.generation_budget),
            )
        } else {
            None
        };
        let traces = if plan.config.emit_traces {
            if let Some(generated) = &generated {
                generated.traces(&model_config, plan.generation_budget)
            } else {
                self.pipeline.trace_video_with_mode(
                    clip.tensor().clone(),
                    plan.config.top_k,
                    plan.config.mode,
                )
            }
        } else {
            Vec::new()
        };
        let mut readout_points =
            plan.ready_readout_points(&traces, generated.as_ref(), &model_config);
        if plan.direct_readout_required(generated.as_ref()) {
            readout_points = Some(self.pipeline.readout_points_with_mode(
                clip.tensor().clone(),
                plan.config.top_k,
                plan.config.mode,
            ));
        };
        self.output
            .write(plan.packet(clip, model_config, generated, traces, readout_points))?;
        Ok(true)
    }

    pub async fn run_next_async(&mut self, device: &B::Device) -> Result<bool> {
        let Some(clip) = self.input.next_clip(device)? else {
            return Ok(false);
        };
        let plan = self.packet_plan()?;
        let model_config = self.pipeline.model().config.clone();
        if plan.should_run_patch_diff()
            && let AutoGazeSparseMaskSource::PatchDiff(config) = plan.config.sparse_mask_source
        {
            let readout = patch_diff_readout_points_async(clip.tensor().clone(), config).await?;
            let traces = if plan.config.emit_traces {
                patch_diff_points_to_traces(&readout.points, plan.config.top_k)
            } else {
                Vec::new()
            };
            let readout_points = plan.config.emit_readout_points.then_some(readout.points);
            self.output
                .write(plan.packet(clip, model_config, None, traces, readout_points))?;
            return Ok(true);
        }
        let generated = if plan.should_generate() {
            Some(
                self.pipeline
                    .generate_with_limit_async(clip.tensor().clone(), plan.generation_budget)
                    .await
                    .map_err(|err| {
                        anyhow::anyhow!(
                            "failed to read AutoGaze generated output asynchronously: {err:?}"
                        )
                    })?,
            )
        } else {
            None
        };
        let traces = if plan.config.emit_traces {
            if let Some(generated) = &generated {
                generated.traces(&model_config, plan.generation_budget)
            } else {
                self.pipeline
                    .trace_video_with_mode_async(
                        clip.tensor().clone(),
                        plan.config.top_k,
                        plan.config.mode,
                    )
                    .await
                    .map_err(|err| {
                        anyhow::anyhow!(
                            "failed to read AutoGaze trace output asynchronously: {err:?}"
                        )
                    })?
            }
        } else {
            Vec::new()
        };
        let mut readout_points =
            plan.ready_readout_points(&traces, generated.as_ref(), &model_config);
        if plan.direct_readout_required(generated.as_ref()) {
            readout_points = Some(
                self.pipeline
                    .readout_points_with_mode_async(
                        clip.tensor().clone(),
                        plan.config.top_k,
                        plan.config.mode,
                    )
                    .await
                    .map_err(|err| {
                        anyhow::anyhow!(
                            "failed to read AutoGaze readout output asynchronously: {err:?}"
                        )
                    })?,
            );
        };
        self.output
            .write(plan.packet(clip, model_config, generated, traces, readout_points))?;
        Ok(true)
    }
}

fn traces_to_readout_points(traces: &[FrameFixationTrace]) -> Vec<Vec<Vec<FixationPoint>>> {
    traces
        .iter()
        .map(|trace| {
            trace
                .frames
                .iter()
                .map(|frame| frame.points.clone())
                .collect()
        })
        .collect()
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
    mode: AutoGazeInferenceMode,
}

impl RgbaClipInput {
    pub fn new() -> Self {
        Self {
            clips: VecDeque::new(),
            mode: AutoGazeInferenceMode::ResizeToModelInput,
        }
    }

    pub fn with_clip(mut self, clip: AutoGazeRgbaClip) -> Self {
        self.push_clip(clip);
        self
    }

    pub fn with_inference_mode(mut self, mode: AutoGazeInferenceMode) -> Self {
        self.mode = mode;
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

    pub const fn inference_mode(&self) -> AutoGazeInferenceMode {
        self.mode
    }
}

impl<B: Backend> AutoGazeInputNode<B> for RgbaClipInput {
    fn next_clip(&mut self, device: &B::Device) -> Result<Option<AutoGazeTensorClip<B>>> {
        self.clips
            .pop_front()
            .map(|clip| clip.into_inference_tensor_clip(self.mode, device))
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

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use std::future::Future;
    use std::task::{Context, Poll};

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
    fn rgba_clip_input_uses_core_inference_preprocessing_by_default() {
        let device = Default::default();
        let rgba = vec![10, 20, 30, 255, 40, 50, 60, 0];
        let shape = AutoGazeRgbaClipShape::new(1, 1, 2);
        let clip = AutoGazeRgbaClip::new(rgba, shape).expect("rgba clip");
        let mut input = RgbaClipInput::new().with_clip(clip);

        let tensor_clip: AutoGazeTensorClip<B> =
            input.next_clip(&device).unwrap().expect("tensor clip");

        assert_eq!(
            tensor_clip.shape(),
            AutoGazeTensorClipShape::new(1, 1, 3, 224, 448)
        );
    }

    #[test]
    fn rgba_clip_input_can_preserve_tiled_source_tensor_shape() {
        let device = Default::default();
        let rgba = vec![10, 20, 30, 255, 40, 50, 60, 0];
        let shape = AutoGazeRgbaClipShape::new(1, 1, 2);
        let clip = AutoGazeRgbaClip::new(rgba, shape).expect("rgba clip");
        let mut input = RgbaClipInput::new()
            .with_inference_mode(AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 224 })
            .with_clip(clip);

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
            generated: None,
            readout_points: None,
            model_config: AutoGazeConfig::default(),
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 1,
        };
        let mut output = VecOutputNode::<B>::new();

        output.write(packet).expect("write");

        assert_eq!(output.packets().len(), 1);
    }

    #[test]
    fn tensor_pipeline_traces_are_opt_in() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::zeros([1, 1, 3, 16, 16], &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            &device,
        ));
        let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output);

        assert_eq!(graph.config.top_k, DEFAULT_REALTIME_TOP_K);

        assert!(graph.run_next(&device).expect("run packet"));

        let packet = graph.output.packets().first().expect("packet");
        assert!(!packet.has_traces());
        assert!(!packet.has_generated());
        assert!(!packet.has_readout_points());
        assert!(packet.traces.is_empty());
        let err = packet
            .frame_mask(0, &device)
            .expect_err("mask requires traces");
        assert!(
            err.to_string().contains("emit_traces"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn tensor_pipeline_can_emit_generated_readout_without_traces() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::zeros([1, 1, 3, 16, 16], &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            &device,
        ));
        let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(
            AutoGazeTensorPipelineConfig {
                top_k: 1,
                emit_generated: true,
                ..Default::default()
            },
        );

        assert!(graph.run_next(&device).expect("run packet"));

        let packet = graph.output.packets().first().expect("packet");
        assert!(!packet.has_traces());
        assert!(packet.has_generated());
        packet
            .frame_readout_tokens(0, SparseReadoutGrid::new(1, 1), Default::default())
            .expect("generated readout");
        let projection = SparseVideoReadoutProjection::new(
            SparseReadoutGrid::new(1, 1),
            SparseVideoReadoutGrid::new(1, 1, 1),
        )
        .with_video_options(SparseVideoReadoutOptions::default().with_exact_tokens(1));
        let coords = packet
            .video_readout_coords_with_projection(projection)
            .expect("projected video readout coords");
        assert_eq!(
            coords,
            packet
                .video_readout_coords(
                    projection.image_grid,
                    projection.video_grid,
                    projection.readout_options,
                    projection.video_options,
                )
                .expect("legacy projected video readout coords")
        );
        assert_eq!(
            packet
                .video_readout_coord_tensor_with_projection(projection, &device)
                .expect("projected video readout tensor")
                .shape()
                .dims::<2>(),
            [coords.len(), 4]
        );
    }

    #[test]
    fn tensor_pipeline_can_emit_tiled_readout_points_without_traces() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::zeros([1, 1, 3, 16, 16], &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            &device,
        ));
        let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(
            AutoGazeTensorPipelineConfig::default()
                .with_mode(AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 })
                .with_top_k(1)
                .with_emit_readout_points(true),
        );

        assert!(graph.run_next(&device).expect("run packet"));

        let packet = graph.output.packets().first().expect("packet");
        assert!(!packet.has_traces());
        assert!(!packet.has_generated());
        assert!(packet.has_readout_points());
        packet
            .frame_readout_tokens(0, SparseReadoutGrid::new(1, 1), Default::default())
            .expect("tiled readout");
    }

    #[test]
    fn tensor_pipeline_can_emit_patch_diff_readout_points_without_model_decode() {
        let device = Default::default();
        let mut values = vec![0.0; 2 * 3 * 28 * 28];
        let frame_stride = 3 * 28 * 28;
        for channel in 0..3 {
            let channel_offset = frame_stride + channel * 28 * 28;
            for y in 14..28 {
                for x in 14..28 {
                    values[channel_offset + y * 28 + x] = 1.0;
                }
            }
        }
        let tensor = Tensor::<B, 5>::from_data(TensorData::new(values, [1, 2, 3, 28, 28]), &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            &device,
        ));
        let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(
            AutoGazeTensorPipelineConfig::default()
                .with_sparse_mask_source(AutoGazeSparseMaskSource::patch_diff(2, 0.25))
                .with_emit_readout_points(true),
        );

        assert!(graph.run_next(&device).expect("run packet"));

        let packet = graph.output.packets().first().expect("packet");
        assert!(!packet.has_generated());
        assert!(packet.has_readout_points());
        let points = packet.readout_points.as_ref().expect("readout points");
        assert_eq!(points[0][1].len(), 1);
        assert!((points[0][1][0].x - 0.75).abs() < 1.0e-6);
        assert!((points[0][1][0].y - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn tensor_pipeline_run_next_async_matches_sync_readout_packets() {
        let device = Default::default();
        let configs = [
            AutoGazeTensorPipelineConfig::default()
                .with_top_k(1)
                .with_emit_generated(true),
            AutoGazeTensorPipelineConfig::default()
                .with_mode(AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 })
                .with_top_k(1)
                .with_emit_readout_points(true),
        ];

        for config in configs {
            let mut sync_graph = tensor_pipeline_with_zero_clip(&device, config);
            let mut async_graph = tensor_pipeline_with_zero_clip(&device, config);

            assert!(sync_graph.run_next(&device).expect("sync packet"));
            assert!(block_on_ready(async_graph.run_next_async(&device)).expect("async packet"));

            let sync_packet = sync_graph.output.packets().first().expect("sync packet");
            let async_packet = async_graph.output.packets().first().expect("async packet");
            assert_eq!(async_packet.has_traces(), sync_packet.has_traces());
            assert_eq!(async_packet.has_generated(), sync_packet.has_generated());
            assert_eq!(
                async_packet.has_readout_points(),
                sync_packet.has_readout_points()
            );
            assert_eq!(
                async_packet
                    .frame_readout_tokens(0, SparseReadoutGrid::new(1, 1), Default::default())
                    .expect("async readout"),
                sync_packet
                    .frame_readout_tokens(0, SparseReadoutGrid::new(1, 1), Default::default())
                    .expect("sync readout")
            );
        }
    }

    #[test]
    fn tensor_pipeline_rejects_generated_packets_for_tiled_modes() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::zeros([1, 1, 3, 16, 16], &device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            &device,
        ));
        let mut graph = AutoGazeTensorPipeline::new(pipeline, input, output).with_config(
            AutoGazeTensorPipelineConfig::default()
                .with_mode(AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 })
                .with_top_k(1)
                .with_emit_traces(true)
                .with_emit_generated(true),
        );

        let err = graph
            .run_next(&device)
            .expect_err("tiled generated packets should be rejected");
        assert!(
            err.to_string().contains("emit_readout_points"),
            "unexpected error: {err:#}"
        );
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
            generated: None,
            readout_points: None,
            model_config: AutoGazeConfig::default(),
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
        let readout = packet
            .frame_readout_tokens(1, SparseReadoutGrid::new(1, 2), Default::default())
            .expect("readout");
        let rects = packet
            .frame_readout_rects(1, Default::default())
            .expect("readout rects");
        let video_readout = packet
            .video_readout_tokens(
                SparseReadoutGrid::new(1, 2),
                SparseVideoReadoutGrid::new(1, 1, 2),
                Default::default(),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(1),
            )
            .expect("video readout");
        let video_coords = packet
            .video_readout_coords(
                SparseReadoutGrid::new(1, 2),
                SparseVideoReadoutGrid::new(1, 1, 2),
                Default::default(),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(1),
            )
            .expect("video readout coords");
        let video_coord_tensor = packet
            .video_readout_coord_tensor(
                SparseReadoutGrid::new(1, 2),
                SparseVideoReadoutGrid::new(1, 1, 2),
                Default::default(),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(1),
                &device,
            )
            .expect("video readout coord tensor");

        assert_eq!(frame.into_data().to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
        assert_eq!(mask.into_data().to_vec::<f32>().unwrap(), vec![1.0, 0.0]);
        assert_eq!(
            tokens.weights.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0]
        );
        assert_eq!(readout, vec![vec![0]]);
        assert_eq!(
            rects,
            vec![vec![SparseReadoutRect::new(0.0, 0.0, 0.5, 1.0)]]
        );
        assert_eq!(video_readout, vec![vec![0]]);
        assert_eq!(video_coords, vec![[0, 0, 0, 0]]);
        assert_eq!(
            video_coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![0, 0, 0, 0]
        );
    }

    #[test]
    fn packet_readout_can_decode_generated_output_without_traces() {
        let device = Default::default();
        let tensor = Tensor::<B, 5>::zeros([1, 2, 3, 16, 16], &device);
        let mut model_config = AutoGazeConfig {
            scales: "32+64".to_string(),
            num_vision_tokens_each_frame: 20,
            ..AutoGazeConfig::default()
        };
        model_config.gaze_model_config.num_vision_tokens_each_frame = 20;
        model_config
            .gaze_model_config
            .vision_model_config
            .kernel_size = 16;
        let packet = AutoGazePipelinePacket {
            clip: AutoGazeTensorClip::new(tensor).expect("clip"),
            traces: Vec::new(),
            generated: Some(AutoGazeGenerateOutput {
                gazing_pos: vec![vec![0, 4, 39]],
                num_gazing_each_frame: vec![2, 1],
                if_padded_gazing: vec![vec![false, false, false]],
                confidences: vec![vec![1.0, 0.8, 0.9]],
            }),
            readout_points: None,
            model_config,
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            top_k: 2,
        };

        assert!(!packet.has_traces());
        assert!(packet.has_generated());
        let tokens = packet
            .frame_readout_tokens(1, SparseReadoutGrid::new(4, 4), Default::default())
            .expect("tokens");
        let rects = packet
            .frame_readout_rects(1, Default::default())
            .expect("rects");
        let video_tokens = packet
            .video_readout_tokens(
                SparseReadoutGrid::new(4, 4),
                SparseVideoReadoutGrid::new(1, 4, 4),
                SparseReadoutOptions::default().with_max_tokens_per_frame(4),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(4),
            )
            .expect("video tokens");
        let video_coords = packet
            .video_readout_coords(
                SparseReadoutGrid::new(4, 4),
                SparseVideoReadoutGrid::new(1, 4, 4),
                SparseReadoutOptions::default().with_max_tokens_per_frame(4),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(4),
            )
            .expect("video coords");
        let video_coord_tensor = packet
            .video_readout_coord_tensor(
                SparseReadoutGrid::new(4, 4),
                SparseVideoReadoutGrid::new(1, 4, 4),
                SparseReadoutOptions::default().with_max_tokens_per_frame(4),
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(4),
                &device,
            )
            .expect("video coord tensor");

        assert_eq!(tokens, vec![vec![15]]);
        assert_eq!(
            rects,
            vec![vec![SparseReadoutRect::new(0.75, 0.75, 1.0, 1.0)]]
        );
        assert_eq!(video_tokens, vec![vec![0, 1, 4, 5]]);
        assert_eq!(
            video_coords,
            vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
        );
        assert_eq!(
            video_coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn rgba_frame_queue_packs_latest_and_fixed_length_clips() {
        let frame_a =
            Arc::new(RgbaImage::from_raw(2, 1, vec![1, 2, 3, 255, 4, 5, 6, 255]).expect("frame a"));
        let frame_b = Arc::new(
            RgbaImage::from_raw(2, 1, vec![7, 8, 9, 255, 10, 11, 12, 255]).expect("frame b"),
        );
        let mut queue = AutoGazeRgbaFrameQueue::new(1);

        queue.push(Arc::clone(&frame_a), 2);
        assert!(queue.build_clip(2).expect("not enough frames").is_none());
        queue.push(Arc::clone(&frame_b), 2);

        let clip = queue
            .build_clip(2)
            .expect("build clip")
            .expect("complete clip");
        assert_eq!(clip.width(), 2);
        assert_eq!(clip.height(), 1);
        assert_eq!(clip.clip_len(), 2);
        assert_eq!(
            clip.rgba(),
            &[1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]
        );
        assert_eq!(
            clip.last_frame_rgba().expect("last frame"),
            frame_b.as_raw()
        );
        queue.recycle_clip_buffer(clip.into_rgba());

        let latest = queue
            .build_latest_clip()
            .expect("build latest")
            .expect("latest clip");
        assert_eq!(latest.clip_len(), 1);
        assert_eq!(latest.rgba(), frame_b.as_raw());
    }

    fn tensor_pipeline_with_zero_clip(
        device: &<burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device,
        config: AutoGazeTensorPipelineConfig,
    ) -> AutoGazeTensorPipeline<B, TensorClipInput<B>, VecOutputNode<B>> {
        let tensor = Tensor::<B, 5>::zeros([1, 1, 3, 16, 16], device);
        let clip = AutoGazeTensorClip::new(tensor).expect("clip");
        let input = TensorClipInput::<B>::new().with_clip(clip);
        let output = VecOutputNode::<B>::new();
        let pipeline = AutoGazePipeline::new(crate::NativeAutoGazeModel::new(
            &tiny_pipeline_config(),
            device,
        ));
        AutoGazeTensorPipeline::new(pipeline, input, output).with_config(config)
    }

    fn block_on_ready<F: Future>(future: F) -> F::Output {
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = std::pin::pin!(future);
        for _ in 0..1024 {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
            std::thread::yield_now();
        }
        panic!("future did not complete without an external executor");
    }

    fn tiny_pipeline_config() -> crate::AutoGazeConfig {
        let hidden = 4;
        crate::AutoGazeConfig {
            scales: "16".to_string(),
            max_num_frames: 1,
            num_vision_tokens_each_frame: 1,
            gaze_model_config: crate::GazeModelConfig {
                input_img_size: 16,
                num_vision_tokens_each_frame: 1,
                vision_model_config: crate::VisionModelConfig {
                    hidden_dim: hidden,
                    out_dim: hidden,
                    depth: 1,
                    kernel_size: 16,
                    temporal_patch_size: 1,
                    trunk_temporal_kernel_size: 1,
                    trunk_spatial_kernel_size: 1,
                },
                connector_config: crate::ConnectorConfig {
                    hidden_dim: hidden,
                    num_tokens: 1,
                },
                gaze_decoder_config: crate::GazeDecoderConfig {
                    vocab_size: 2,
                    hidden_size: hidden,
                    intermediate_size: hidden * 2,
                    num_hidden_layers: 1,
                    num_attention_heads: 1,
                    num_key_value_heads: 1,
                    max_position_embeddings: 8,
                    bos_token_id: 0,
                    eos_token_id: 1,
                    head_dim: hidden,
                    num_multi_token_pred: 1,
                    ..crate::GazeDecoderConfig::default()
                },
                ..crate::GazeModelConfig::default()
            },
            ..crate::AutoGazeConfig::default()
        }
    }
}
