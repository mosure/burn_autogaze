use crate::model::generated_to_frame_points;
use crate::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeLoadOptions, AutoGazeStreamingCache,
    DEFAULT_TILED_TILE_BATCH_SIZE, FixationPoint, FixationSet, FrameFixationTrace,
    NativeAutoGazeModel,
};
use anyhow::{Result, anyhow, ensure};
use burn::tensor::backend::{Backend, ExecutionError};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Tensor, TensorData};
use image::{RgbaImage, imageops::FilterType};
use std::path::Path;

pub const AUTO_GAZE_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const AUTO_GAZE_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const AUTO_GAZE_RESCALE_FACTOR: f32 = 1.0 / 127.5;
pub const AUTO_GAZE_PROCESSOR_SHORT_EDGE: usize = 224;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeInferenceMode {
    #[default]
    ResizeToModelInput,
    TiledResizeToGrid {
        tile_size: usize,
    },
    TiledFullResolution {
        tile_size: usize,
        stride: usize,
    },
}

impl AutoGazeInferenceMode {
    pub const fn resize_to_model_input() -> Self {
        Self::ResizeToModelInput
    }

    pub const fn tiled_full_resolution(tile_size: usize, stride: usize) -> Self {
        Self::TiledFullResolution { tile_size, stride }
    }

    pub const fn tiled_resize_to_grid(tile_size: usize) -> Self {
        Self::TiledResizeToGrid { tile_size }
    }

    pub fn tiled_model_input(model_input_size: usize) -> Self {
        let tile_size = model_input_size.max(1);
        Self::TiledResizeToGrid { tile_size }
    }

    pub fn normalized(self) -> Self {
        match self {
            Self::ResizeToModelInput => Self::ResizeToModelInput,
            Self::TiledResizeToGrid { tile_size } => Self::TiledResizeToGrid {
                tile_size: tile_size.max(1),
            },
            Self::TiledFullResolution { tile_size, stride } => {
                let tile_size = tile_size.max(1);
                let stride = stride.max(1).min(tile_size);
                Self::TiledFullResolution { tile_size, stride }
            }
        }
    }

    pub fn fixation_budget(self, k: usize, source_height: usize, source_width: usize) -> usize {
        let k = k.max(1);
        match self.normalized() {
            Self::ResizeToModelInput => k,
            Self::TiledResizeToGrid { tile_size } => {
                let tile_count =
                    AutoGazeTileLayout::resized_grid(source_height, source_width, tile_size)
                        .tile_count()
                        .max(1);
                k.saturating_mul(tile_count)
            }
            Self::TiledFullResolution { tile_size, stride } => {
                let tile_count =
                    AutoGazeTileLayout::tiled(source_height, source_width, tile_size, stride)
                        .tile_count()
                        .max(1);
                k.saturating_mul(tile_count)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeClipShape {
    pub clip_len: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl AutoGazeClipShape {
    pub const fn new(clip_len: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            clip_len,
            channels,
            height,
            width,
        }
    }

    pub const fn num_values(&self) -> usize {
        self.clip_len * self.channels * self.height * self.width
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeRgbaClipShape {
    pub clip_len: usize,
    pub height: usize,
    pub width: usize,
}

impl AutoGazeRgbaClipShape {
    pub const fn new(clip_len: usize, height: usize, width: usize) -> Self {
        Self {
            clip_len,
            height,
            width,
        }
    }
}

pub struct AutoGazePreparedRun<B: Backend> {
    pub video: Tensor<B, 5>,
    pub mode: AutoGazeInferenceMode,
    pub frame_index: usize,
    pub model_frames: usize,
}

pub struct AutoGazeTraceRunOutput {
    pub traces: Vec<FrameFixationTrace>,
    pub frame_index: usize,
    pub model_frames: usize,
}

/// Selected fixation points from a prepared AutoGaze run.
///
/// This is the lower-allocation counterpart to `AutoGazeTraceRunOutput` for
/// realtime consumers that only need the currently selected points. The shape is
/// `[batch][frame][point]`.
pub struct AutoGazeReadoutRunOutput {
    pub points: Vec<Vec<Vec<FixationPoint>>>,
    pub frame_index: usize,
    pub model_frames: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeTile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl AutoGazeTile {
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoGazeTileLayout {
    pub source_width: usize,
    pub source_height: usize,
    pub tile_size: usize,
    pub stride: usize,
    pub tiles: Vec<AutoGazeTile>,
}

impl AutoGazeTileLayout {
    pub fn full_frame(source_height: usize, source_width: usize) -> Self {
        Self {
            source_width,
            source_height,
            tile_size: source_height.max(source_width).max(1),
            stride: source_height.max(source_width).max(1),
            tiles: vec![AutoGazeTile::new(
                0,
                0,
                source_width.max(1),
                source_height.max(1),
            )],
        }
    }

    pub fn tiled(
        source_height: usize,
        source_width: usize,
        tile_size: usize,
        stride: usize,
    ) -> Self {
        let source_height = source_height.max(1);
        let source_width = source_width.max(1);
        let tile_size = tile_size.max(1);
        let stride = stride.max(1).min(tile_size);
        let y_origins = tile_origins(source_height, stride);
        let x_origins = tile_origins(source_width, stride);
        let mut tiles = Vec::with_capacity(y_origins.len() * x_origins.len());
        for y in y_origins {
            for &x in x_origins.iter() {
                tiles.push(AutoGazeTile::new(x, y, tile_size, tile_size));
            }
        }
        Self {
            source_width,
            source_height,
            tile_size,
            stride,
            tiles,
        }
    }

    pub fn resized_grid(source_height: usize, source_width: usize, tile_size: usize) -> Self {
        let source_height = source_height.max(1);
        let source_width = source_width.max(1);
        let tile_size = tile_size.max(1);
        let rows = source_height.div_ceil(tile_size).max(1);
        let cols = source_width.div_ceil(tile_size).max(1);
        let target_height = rows.saturating_mul(tile_size).max(1);
        let target_width = cols.saturating_mul(tile_size).max(1);
        let mut tiles = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                tiles.push(AutoGazeTile::new(
                    col * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                ));
            }
        }
        Self {
            source_width: target_width,
            source_height: target_height,
            tile_size,
            stride: tile_size,
            tiles,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

pub struct AutoGazeEmbedOutput<B: Backend> {
    pub embeddings: Tensor<B, 4>,
    pub past_conv_values: Vec<Tensor<B, 5>>,
    pub layout: AutoGazeTileLayout,
}

pub fn rgba_clip_to_tensor<B: Backend>(
    rgba: &[u8],
    shape: AutoGazeRgbaClipShape,
    device: &B::Device,
) -> Result<Tensor<B, 5>> {
    ensure!(
        shape.width > 0 && shape.height > 0 && shape.clip_len > 0,
        "RGBA clip dimensions must be nonzero"
    );
    let pixels_per_frame = shape
        .width
        .checked_mul(shape.height)
        .ok_or_else(|| anyhow::anyhow!("RGBA clip dimensions overflow"))?;
    let expected_len = pixels_per_frame
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_mul(shape.clip_len))
        .ok_or_else(|| anyhow::anyhow!("RGBA clip byte length overflow"))?;
    ensure!(
        rgba.len() == expected_len,
        "expected {expected_len} RGBA bytes for {} frame(s) at {}x{}, got {}",
        shape.clip_len,
        shape.width,
        shape.height,
        rgba.len()
    );

    let mut values = vec![0.0; shape.clip_len * 3 * pixels_per_frame];
    // The upstream processor emits RGB video as [batch, time, channel, height, width].
    // Alpha is intentionally ignored.
    for frame in 0..shape.clip_len {
        let frame_offset = frame * pixels_per_frame * 4;
        let output_offset = frame * 3 * pixels_per_frame;
        let red_offset = output_offset;
        let green_offset = output_offset + pixels_per_frame;
        let blue_offset = output_offset + 2 * pixels_per_frame;
        let frame_rgba = &rgba[frame_offset..frame_offset + pixels_per_frame * 4];
        for (pixel, channels) in frame_rgba.chunks_exact(4).enumerate() {
            values[red_offset + pixel] = autogaze_processor_value(channels[0], 0);
            values[green_offset + pixel] = autogaze_processor_value(channels[1], 1);
            values[blue_offset + pixel] = autogaze_processor_value(channels[2], 2);
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(values, [1, shape.clip_len, 3, shape.height, shape.width]),
        device,
    ))
}

pub fn rgba_clip_to_processor_tensor<B: Backend>(
    rgba: &[u8],
    shape: AutoGazeRgbaClipShape,
    device: &B::Device,
) -> Result<Tensor<B, 5>> {
    let (target_height, target_width) =
        processor_resize_dimensions(shape.height, shape.width, AUTO_GAZE_PROCESSOR_SHORT_EDGE);
    if target_height == shape.height && target_width == shape.width {
        return rgba_clip_to_tensor::<B>(rgba, shape, device);
    }

    let pixels_per_frame = shape
        .width
        .checked_mul(shape.height)
        .ok_or_else(|| anyhow!("RGBA clip dimensions overflow"))?;
    let bytes_per_frame = pixels_per_frame
        .checked_mul(4)
        .ok_or_else(|| anyhow!("RGBA frame byte length overflow"))?;
    ensure!(
        rgba.len() == bytes_per_frame * shape.clip_len,
        "expected {} RGBA bytes for {} frame(s) at {}x{}, got {}",
        bytes_per_frame * shape.clip_len,
        shape.clip_len,
        shape.width,
        shape.height,
        rgba.len()
    );

    let target_bytes_per_frame = target_width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| anyhow!("resized RGBA frame byte length overflow"))?;
    let mut resized_rgba = Vec::with_capacity(target_bytes_per_frame * shape.clip_len);
    for frame in 0..shape.clip_len {
        let start = frame * bytes_per_frame;
        let end = start + bytes_per_frame;
        let image = RgbaImage::from_raw(
            shape.width as u32,
            shape.height as u32,
            rgba[start..end].to_vec(),
        )
        .ok_or_else(|| anyhow!("failed to build RGBA frame for AutoGaze preprocessing"))?;
        let resized = image::imageops::resize(
            &image,
            target_width as u32,
            target_height as u32,
            FilterType::Triangle,
        );
        resized_rgba.extend_from_slice(resized.as_raw());
    }

    rgba_clip_to_tensor::<B>(
        &resized_rgba,
        AutoGazeRgbaClipShape::new(shape.clip_len, target_height, target_width),
        device,
    )
}

pub fn rgba_clip_to_inference_tensor<B: Backend>(
    rgba: &[u8],
    shape: AutoGazeRgbaClipShape,
    mode: AutoGazeInferenceMode,
    device: &B::Device,
) -> Result<Tensor<B, 5>> {
    match mode.normalized() {
        AutoGazeInferenceMode::ResizeToModelInput => {
            rgba_clip_to_processor_tensor::<B>(rgba, shape, device)
        }
        AutoGazeInferenceMode::TiledResizeToGrid { .. }
        | AutoGazeInferenceMode::TiledFullResolution { .. } => {
            rgba_clip_to_tensor::<B>(rgba, shape, device)
        }
    }
}

pub fn prepare_rgba_clip_for_trace<B: Backend>(
    rgba: &[u8],
    shape: AutoGazeRgbaClipShape,
    mode: AutoGazeInferenceMode,
    streaming_cache: bool,
    device: &B::Device,
) -> Result<AutoGazePreparedRun<B>> {
    if streaming_cache {
        let frame = last_rgba_frame(rgba, shape)?;
        let shape = AutoGazeRgbaClipShape::new(1, shape.height, shape.width);
        let video = rgba_clip_to_inference_tensor::<B>(frame, shape, mode, device)?;
        return Ok(AutoGazePreparedRun {
            video,
            mode,
            frame_index: 0,
            model_frames: 1,
        });
    }

    let video = rgba_clip_to_inference_tensor::<B>(rgba, shape, mode, device)?;
    Ok(AutoGazePreparedRun {
        video,
        mode,
        frame_index: shape.clip_len.saturating_sub(1),
        model_frames: shape.clip_len,
    })
}

pub fn last_rgba_frame(rgba: &[u8], shape: AutoGazeRgbaClipShape) -> Result<&[u8]> {
    ensure!(
        shape.width > 0 && shape.height > 0 && shape.clip_len > 0,
        "RGBA clip dimensions must be nonzero"
    );
    let frame_bytes = shape
        .width
        .checked_mul(shape.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| anyhow!("RGBA frame byte length overflow"))?;
    let expected_len = frame_bytes
        .checked_mul(shape.clip_len)
        .ok_or_else(|| anyhow!("RGBA clip byte length overflow"))?;
    ensure!(
        rgba.len() == expected_len,
        "expected {expected_len} RGBA bytes for {} frame(s) at {}x{}, got {}",
        shape.clip_len,
        shape.width,
        shape.height,
        rgba.len()
    );
    let start = frame_bytes
        .checked_mul(shape.clip_len.saturating_sub(1))
        .ok_or_else(|| anyhow!("RGBA last frame offset overflow"))?;
    let end = start
        .checked_add(frame_bytes)
        .ok_or_else(|| anyhow!("RGBA last frame end overflow"))?;
    Ok(&rgba[start..end])
}

pub fn resize_rgba_frame_to_dimensions(
    frame: RgbaImage,
    target_width: Option<u32>,
    target_height: Option<u32>,
) -> RgbaImage {
    let (width, height) = frame.dimensions();
    let (target_width, target_height) =
        resize_dimensions_preserving_aspect(width, height, target_width, target_height);
    if target_width == width && target_height == height {
        return frame;
    }

    image::imageops::resize(&frame, target_width, target_height, FilterType::Triangle)
}

pub fn resize_dimensions_preserving_aspect(
    width: u32,
    height: u32,
    target_width: Option<u32>,
    target_height: Option<u32>,
) -> (u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    match (target_width, target_height) {
        (Some(target_width), Some(target_height)) => (target_width.max(1), target_height.max(1)),
        (Some(target_width), None) => {
            let target_width = target_width.max(1);
            let target_height =
                ((height as f64 * target_width as f64 / width as f64).round() as u32).max(1);
            (target_width, target_height)
        }
        (None, Some(target_height)) => {
            let target_height = target_height.max(1);
            let target_width =
                ((width as f64 * target_height as f64 / height as f64).round() as u32).max(1);
            (target_width, target_height)
        }
        (None, None) => (width, height),
    }
}

pub fn resize_video_shortest_edge<B: Backend>(
    video: Tensor<B, 5>,
    shortest_edge: usize,
) -> Tensor<B, 5> {
    let [batch, time, channels, height, width] = video.shape().dims::<5>();
    let (target_height, target_width) = processor_resize_dimensions(height, width, shortest_edge);
    if target_height == height && target_width == width {
        return video;
    }

    let video = video.reshape([batch * time, channels, height, width]);
    let video = interpolate(
        video,
        [target_height, target_width],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    video.reshape([batch, time, channels, target_height, target_width])
}

fn processor_resize_dimensions(
    height: usize,
    width: usize,
    shortest_edge: usize,
) -> (usize, usize) {
    let height = height.max(1);
    let width = width.max(1);
    let shortest_edge = shortest_edge.max(1);
    let min_edge = height.min(width);
    if min_edge == shortest_edge {
        return (height, width);
    }

    let scale = shortest_edge as f64 / min_edge as f64;
    (
        ((height as f64 * scale).round() as usize).max(1),
        ((width as f64 * scale).round() as usize).max(1),
    )
}

pub fn video_frame_tensor<B: Backend>(
    video: Tensor<B, 5>,
    frame_index: usize,
) -> Result<Tensor<B, 5>> {
    let [batch, time, channels, height, width] = video.shape().dims::<5>();
    ensure!(
        frame_index < time,
        "frame index {frame_index} is out of bounds for {time} frame(s)"
    );

    Ok(video.slice([
        0..batch,
        frame_index..frame_index + 1,
        0..channels,
        0..height,
        0..width,
    ]))
}

fn autogaze_processor_value(value: u8, channel: usize) -> f32 {
    let rescaled = value as f32 * AUTO_GAZE_RESCALE_FACTOR - 1.0;
    (rescaled - AUTO_GAZE_IMAGE_MEAN[channel]) / AUTO_GAZE_IMAGE_STD[channel]
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutoGazePipelineOptions {
    max_gaze_tokens_each_frame: Option<usize>,
    task_loss_requirement: AutoGazeTaskLossOption,
    tile_batch_size: Option<usize>,
}

impl AutoGazePipelineOptions {
    pub const fn new() -> Self {
        Self {
            max_gaze_tokens_each_frame: None,
            task_loss_requirement: AutoGazeTaskLossOption::ModelDefault,
            tile_batch_size: None,
        }
    }

    pub const fn max_gaze_tokens_each_frame(&self) -> Option<usize> {
        self.max_gaze_tokens_each_frame
    }

    pub const fn task_loss_requirement(&self) -> AutoGazeTaskLossOption {
        self.task_loss_requirement
    }

    pub const fn tile_batch_size(&self) -> Option<usize> {
        self.tile_batch_size
    }

    pub const fn with_max_gaze_tokens_each_frame(
        mut self,
        max_gaze_tokens_each_frame: usize,
    ) -> Self {
        self.max_gaze_tokens_each_frame = Some(max_gaze_tokens_each_frame);
        self
    }

    pub const fn with_model_default_gaze_tokens(mut self) -> Self {
        self.max_gaze_tokens_each_frame = None;
        self
    }

    pub const fn with_task_loss_requirement(mut self, task_loss_requirement: f32) -> Self {
        self.task_loss_requirement = AutoGazeTaskLossOption::Value(task_loss_requirement);
        self
    }

    pub const fn without_task_loss_requirement(mut self) -> Self {
        self.task_loss_requirement = AutoGazeTaskLossOption::Disabled;
        self
    }

    pub const fn with_model_default_task_loss_requirement(mut self) -> Self {
        self.task_loss_requirement = AutoGazeTaskLossOption::ModelDefault;
        self
    }

    pub const fn with_tile_batch_size(mut self, tile_batch_size: usize) -> Self {
        self.tile_batch_size = Some(tile_batch_size);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AutoGazeTaskLossOption {
    #[default]
    ModelDefault,
    Disabled,
    Value(f32),
}

#[derive(Clone, Debug)]
pub struct AutoGazePipeline<B: Backend> {
    model: NativeAutoGazeModel<B>,
    max_gaze_tokens_each_frame: usize,
    task_loss_requirement: Option<f32>,
    tile_batch_size: usize,
}

struct TileTraceAccumulator<'a> {
    batch: usize,
    time: usize,
    layout: &'a AutoGazeTileLayout,
    frame_points: &'a mut [Vec<Vec<FixationPoint>>],
    stop_probabilities: &'a mut [Vec<f32>],
}

impl<B: Backend> AutoGazePipeline<B> {
    pub fn new(model: NativeAutoGazeModel<B>) -> Self {
        let max_gaze_tokens_each_frame = model.default_max_gaze_tokens_each_frame();
        let task_loss_requirement = model.default_task_loss_requirement();
        Self {
            model,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            tile_batch_size: DEFAULT_TILED_TILE_BATCH_SIZE,
        }
    }

    pub fn from_config(config: &AutoGazeConfig, device: &B::Device) -> Self {
        Self::new(NativeAutoGazeModel::new(config, device))
    }

    pub fn load(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        Self::from_hf_dir(dir, device)
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        Ok(Self::new(NativeAutoGazeModel::from_hf_dir(dir, device)?))
    }

    pub fn from_hf_dir_with_options(
        dir: impl AsRef<Path>,
        device: &B::Device,
        options: AutoGazeLoadOptions,
    ) -> Result<Self> {
        Ok(Self::new(NativeAutoGazeModel::from_hf_dir_with_options(
            dir, device, options,
        )?))
    }

    pub const fn max_gaze_tokens_each_frame(&self) -> usize {
        self.max_gaze_tokens_each_frame
    }

    pub const fn task_loss_requirement(&self) -> Option<f32> {
        self.task_loss_requirement
    }

    pub const fn tile_batch_size(&self) -> usize {
        self.tile_batch_size
    }

    pub fn with_options(mut self, options: AutoGazePipelineOptions) -> Self {
        self.apply_options(options);
        self
    }

    pub fn apply_options(&mut self, options: AutoGazePipelineOptions) {
        if let Some(max_gaze_tokens_each_frame) = options.max_gaze_tokens_each_frame {
            self.set_max_gaze_tokens_each_frame(max_gaze_tokens_each_frame);
        } else {
            self.reset_max_gaze_tokens_each_frame();
        }

        match options.task_loss_requirement {
            AutoGazeTaskLossOption::ModelDefault => self.reset_task_loss_requirement(),
            AutoGazeTaskLossOption::Disabled => self.set_task_loss_requirement(None),
            AutoGazeTaskLossOption::Value(value) => self.set_task_loss_requirement(Some(value)),
        }

        if let Some(tile_batch_size) = options.tile_batch_size {
            self.set_tile_batch_size(tile_batch_size);
        }
    }

    pub fn with_max_gaze_tokens_each_frame(mut self, max_gaze_tokens_each_frame: usize) -> Self {
        self.max_gaze_tokens_each_frame = max_gaze_tokens_each_frame.max(1);
        self
    }

    pub fn with_task_loss_requirement(mut self, task_loss_requirement: Option<f32>) -> Self {
        self.task_loss_requirement = task_loss_requirement.map(|value| value.max(0.0));
        self
    }

    pub fn with_tile_batch_size(mut self, tile_batch_size: usize) -> Self {
        self.tile_batch_size = tile_batch_size.max(1);
        self
    }

    pub fn set_max_gaze_tokens_each_frame(&mut self, max_gaze_tokens_each_frame: usize) {
        self.max_gaze_tokens_each_frame = max_gaze_tokens_each_frame.max(1);
    }

    pub fn reset_max_gaze_tokens_each_frame(&mut self) {
        self.max_gaze_tokens_each_frame = self.model.default_max_gaze_tokens_each_frame();
    }

    pub fn set_task_loss_requirement(&mut self, task_loss_requirement: Option<f32>) {
        self.task_loss_requirement = task_loss_requirement.map(|value| value.max(0.0));
    }

    pub fn reset_task_loss_requirement(&mut self) {
        self.task_loss_requirement = self.model.default_task_loss_requirement();
    }

    pub fn set_tile_batch_size(&mut self, tile_batch_size: usize) {
        self.tile_batch_size = tile_batch_size.max(1);
    }

    pub const fn model(&self) -> &NativeAutoGazeModel<B> {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut NativeAutoGazeModel<B> {
        &mut self.model
    }

    pub fn into_model(self) -> NativeAutoGazeModel<B> {
        self.model
    }

    pub fn prepare_video(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        self.model.gazing_model.prepare_video(video)
    }

    pub fn embed_video(&self, video: Tensor<B, 5>) -> (Tensor<B, 4>, Vec<Tensor<B, 5>>) {
        self.embed_video_resize(video)
    }

    pub fn embed_video_with_mode(
        &self,
        video: Tensor<B, 5>,
        mode: AutoGazeInferenceMode,
    ) -> AutoGazeEmbedOutput<B> {
        let [batch, _time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => {
                let (embeddings, past_conv_values) = self.embed_video_resize(video);
                AutoGazeEmbedOutput {
                    embeddings,
                    past_conv_values,
                    layout: AutoGazeTileLayout::full_frame(height, width),
                }
            }
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                let layout = AutoGazeTileLayout::resized_grid(height, width, tile_size);
                let video = resize_video_to_layout_grid(video, &layout);
                let embeddings = self.embed_tiled_video(video, batch, &layout);
                AutoGazeEmbedOutput {
                    embeddings,
                    past_conv_values: Vec::new(),
                    layout,
                }
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                let embeddings = self.embed_tiled_video(video, batch, &layout);
                AutoGazeEmbedOutput {
                    embeddings,
                    past_conv_values: Vec::new(),
                    layout,
                }
            }
        }
    }

    fn embed_video_resize(&self, video: Tensor<B, 5>) -> (Tensor<B, 4>, Vec<Tensor<B, 5>>) {
        let video = self.prepare_video(video);
        self.embed_model_input(video)
    }

    pub fn try_embed_model_input(
        &self,
        video: Tensor<B, 5>,
    ) -> Result<(Tensor<B, 4>, Vec<Tensor<B, 5>>)> {
        let [_batch, _time, _channels, height, width] = video.shape().dims::<5>();
        let input_img_size = self.model.config.gaze_model_config.input_img_size.max(1);
        ensure!(
            height == input_img_size && width == input_img_size,
            "AutoGaze model input frames must be square {input_img_size}x{input_img_size}, got {height}x{width}; \
             call prepare_video/embed_video or use a tiled inference mode for full-resolution input",
        );
        Ok(self.model.gazing_model.embed_video(video, false, None))
    }

    pub fn embed_model_input(&self, video: Tensor<B, 5>) -> (Tensor<B, 4>, Vec<Tensor<B, 5>>) {
        self.model.gazing_model.embed_video(video, false, None)
    }

    pub fn generate(&self, video: Tensor<B, 5>) -> AutoGazeGenerateOutput {
        self.model.generate_with_task_loss_requirement(
            video,
            self.max_gaze_tokens_each_frame,
            self.task_loss_requirement,
        )
    }

    pub fn generate_with_limit(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
    ) -> AutoGazeGenerateOutput {
        self.model.generate_with_task_loss_requirement(
            video,
            max_gaze_tokens_each_frame.max(1),
            self.task_loss_requirement,
        )
    }

    pub async fn generate_async(
        &self,
        video: Tensor<B, 5>,
    ) -> std::result::Result<AutoGazeGenerateOutput, ExecutionError> {
        self.model
            .generate_with_task_loss_requirement_async(
                video,
                self.max_gaze_tokens_each_frame,
                self.task_loss_requirement,
            )
            .await
    }

    pub async fn generate_with_limit_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
    ) -> std::result::Result<AutoGazeGenerateOutput, ExecutionError> {
        self.model
            .generate_with_task_loss_requirement_async(
                video,
                max_gaze_tokens_each_frame.max(1),
                self.task_loss_requirement,
            )
            .await
    }

    pub fn infer(&self, video: Tensor<B, 5>, k: usize) -> Vec<FrameFixationTrace> {
        self.trace_video(video, k)
    }

    pub fn infer_with_mode(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Vec<FrameFixationTrace> {
        self.trace_video_with_mode(video, k, mode)
    }

    pub fn trace_video(&self, video: Tensor<B, 5>, k: usize) -> Vec<FrameFixationTrace> {
        self.trace_video_resize(video, k)
    }

    pub async fn infer_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        self.trace_video_async(video, k).await
    }

    pub async fn infer_with_mode_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        self.trace_video_with_mode_async(video, k, mode).await
    }

    pub async fn trace_video_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        self.trace_video_resize_async(video, k).await
    }

    pub fn trace_video_with_mode(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Vec<FrameFixationTrace> {
        let [batch, time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => self.trace_video_resize(video, k),
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                let layout = AutoGazeTileLayout::resized_grid(height, width, tile_size);
                let video = resize_video_to_layout_grid(video, &layout);
                self.trace_tiled_video(video, k, batch, time, &layout)
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                self.trace_tiled_video(video, k, batch, time, &layout)
            }
        }
    }

    pub async fn trace_video_with_mode_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        let [batch, time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => {
                self.trace_video_resize_async(video, k).await
            }
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                let layout = AutoGazeTileLayout::resized_grid(height, width, tile_size);
                let video = resize_video_to_layout_grid(video, &layout);
                self.trace_tiled_video_async(video, k, batch, time, &layout)
                    .await
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                self.trace_tiled_video_async(video, k, batch, time, &layout)
                    .await
            }
        }
    }

    pub fn readout_points_with_mode(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Vec<Vec<Vec<FixationPoint>>> {
        let [batch, time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => {
                let generation_budget = self.max_gaze_tokens_each_frame.max(k.max(1));
                let generated = self.generate_with_limit(video, generation_budget);
                generated_to_frame_points(&generated, &self.model.config)
            }
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                let layout = AutoGazeTileLayout::resized_grid(height, width, tile_size);
                let video = resize_video_to_layout_grid(video, &layout);
                self.tiled_readout_points(video, k, batch, time, &layout)
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                self.tiled_readout_points(video, k, batch, time, &layout)
            }
        }
    }

    pub async fn readout_points_with_mode_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> std::result::Result<Vec<Vec<Vec<FixationPoint>>>, ExecutionError> {
        let [batch, time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => {
                let generation_budget = self.max_gaze_tokens_each_frame.max(k.max(1));
                let generated = self
                    .generate_with_limit_async(video, generation_budget)
                    .await?;
                Ok(generated_to_frame_points(&generated, &self.model.config))
            }
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size } => {
                let layout = AutoGazeTileLayout::resized_grid(height, width, tile_size);
                let video = resize_video_to_layout_grid(video, &layout);
                self.tiled_readout_points_async(video, k, batch, time, &layout)
                    .await
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                self.tiled_readout_points_async(video, k, batch, time, &layout)
                    .await
            }
        }
    }

    pub fn trace_video_streaming(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        cache: &mut AutoGazeStreamingCache<B>,
    ) -> Vec<FrameFixationTrace> {
        self.model.trace_streaming_with_task_loss_requirement(
            video,
            cache,
            k,
            self.max_gaze_tokens_each_frame.max(k.max(1)),
            self.task_loss_requirement,
        )
    }

    pub async fn trace_video_streaming_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        cache: &mut AutoGazeStreamingCache<B>,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        self.model
            .trace_streaming_with_task_loss_requirement_async(
                video,
                cache,
                k,
                self.max_gaze_tokens_each_frame.max(k.max(1)),
                self.task_loss_requirement,
            )
            .await
    }

    pub fn trace_prepared_run(
        &self,
        prepared: AutoGazePreparedRun<B>,
        k: usize,
        cache: Option<&mut AutoGazeStreamingCache<B>>,
    ) -> AutoGazeTraceRunOutput {
        let AutoGazePreparedRun {
            video,
            mode,
            frame_index,
            model_frames,
        } = prepared;
        let traces = if let Some(cache) = cache {
            self.trace_video_streaming(video, k, cache)
        } else {
            self.trace_video_with_mode(video, k, mode)
        };
        AutoGazeTraceRunOutput {
            traces,
            frame_index,
            model_frames,
        }
    }

    /// Run a prepared clip and return decoded fixation points without building
    /// full traces when the selected mode allows it.
    ///
    /// When `cache` is present, this still routes through the streaming trace
    /// implementation because KV-cache state is maintained there; the returned
    /// value is then reduced to points for callers that do not need trace
    /// metadata.
    pub fn readout_prepared_run(
        &self,
        prepared: AutoGazePreparedRun<B>,
        k: usize,
        cache: Option<&mut AutoGazeStreamingCache<B>>,
    ) -> AutoGazeReadoutRunOutput {
        let AutoGazePreparedRun {
            video,
            mode,
            frame_index,
            model_frames,
        } = prepared;
        let points = if let Some(cache) = cache {
            traces_to_frame_points(self.trace_video_streaming(video, k, cache))
        } else {
            self.readout_points_with_mode(video, k, mode)
        };
        AutoGazeReadoutRunOutput {
            points,
            frame_index,
            model_frames,
        }
    }

    pub async fn trace_prepared_run_async(
        &self,
        prepared: AutoGazePreparedRun<B>,
        k: usize,
        cache: Option<&mut AutoGazeStreamingCache<B>>,
    ) -> std::result::Result<AutoGazeTraceRunOutput, ExecutionError> {
        let AutoGazePreparedRun {
            video,
            mode,
            frame_index,
            model_frames,
        } = prepared;
        let traces = if let Some(cache) = cache {
            self.trace_video_streaming_async(video, k, cache).await?
        } else {
            self.trace_video_with_mode_async(video, k, mode).await?
        };
        Ok(AutoGazeTraceRunOutput {
            traces,
            frame_index,
            model_frames,
        })
    }

    /// Async version of `readout_prepared_run` for wasm/WebGPU callers that
    /// cannot block on tensor data readback.
    pub async fn readout_prepared_run_async(
        &self,
        prepared: AutoGazePreparedRun<B>,
        k: usize,
        cache: Option<&mut AutoGazeStreamingCache<B>>,
    ) -> std::result::Result<AutoGazeReadoutRunOutput, ExecutionError> {
        let AutoGazePreparedRun {
            video,
            mode,
            frame_index,
            model_frames,
        } = prepared;
        let points = if let Some(cache) = cache {
            traces_to_frame_points(self.trace_video_streaming_async(video, k, cache).await?)
        } else {
            self.readout_points_with_mode_async(video, k, mode).await?
        };
        Ok(AutoGazeReadoutRunOutput {
            points,
            frame_index,
            model_frames,
        })
    }

    fn embed_tiled_video(
        &self,
        video: Tensor<B, 5>,
        batch: usize,
        layout: &AutoGazeTileLayout,
    ) -> Tensor<B, 4> {
        if dense_grid_layout(layout) {
            let tiled_video = dense_grid_video_tiles(video, layout);
            return self.embed_batched_tile_video(tiled_video, batch, layout);
        }

        let mut tile_embedding_chunks =
            Vec::with_capacity(layout.tile_count().div_ceil(self.tile_batch_size));
        for tiles in layout.tiles.chunks(self.tile_batch_size) {
            let crops = tiles
                .iter()
                .copied()
                .map(|tile| crop_video_tile(video.clone(), tile))
                .collect::<Vec<_>>();
            let (embeddings, _) = self.embed_video_resize(Tensor::cat(crops, 0));
            tile_embedding_chunks.push(reassemble_tile_embeddings(embeddings, tiles.len(), batch));
        }
        Tensor::cat(tile_embedding_chunks, 2)
    }

    fn embed_batched_tile_video(
        &self,
        tile_video: Tensor<B, 5>,
        batch: usize,
        layout: &AutoGazeTileLayout,
    ) -> Tensor<B, 4> {
        let mut tile_embedding_chunks =
            Vec::with_capacity(layout.tile_count().div_ceil(self.tile_batch_size));
        for (chunk_index, tiles) in layout.tiles.chunks(self.tile_batch_size).enumerate() {
            let start_tile = chunk_index.saturating_mul(self.tile_batch_size);
            let tile_video = tile_batch_slice(tile_video.clone(), batch, start_tile, tiles.len());
            let (embeddings, _) = self.embed_video_resize(tile_video);
            tile_embedding_chunks.push(reassemble_tile_embeddings(embeddings, tiles.len(), batch));
        }
        Tensor::cat(tile_embedding_chunks, 2)
    }

    fn trace_tiled_video(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        batch: usize,
        time: usize,
        layout: &AutoGazeTileLayout,
    ) -> Vec<FrameFixationTrace> {
        let frame_budget = self
            .max_gaze_tokens_each_frame
            .max(k.max(1))
            .saturating_mul(layout.tile_count().max(1));
        let mut frame_points = empty_batch_frame_points(batch, time);
        let mut stop_probabilities = vec![vec![0.0f32; time]; batch];
        self.collect_tiled_trace_points(
            video,
            k,
            layout,
            &mut frame_points,
            &mut stop_probabilities,
        );
        build_tiled_traces(frame_points, stop_probabilities, frame_budget)
    }

    async fn trace_tiled_video_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        batch: usize,
        time: usize,
        layout: &AutoGazeTileLayout,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        let frame_budget = self
            .max_gaze_tokens_each_frame
            .max(k.max(1))
            .saturating_mul(layout.tile_count().max(1));
        let mut frame_points = empty_batch_frame_points(batch, time);
        let mut stop_probabilities = vec![vec![0.0f32; time]; batch];
        self.collect_tiled_trace_points_async(
            video,
            k,
            layout,
            &mut frame_points,
            &mut stop_probabilities,
        )
        .await?;
        Ok(build_tiled_traces(
            frame_points,
            stop_probabilities,
            frame_budget,
        ))
    }

    fn trace_video_resize(&self, video: Tensor<B, 5>, k: usize) -> Vec<FrameFixationTrace> {
        self.model.trace_video_with_task_loss_requirement(
            video,
            k,
            self.max_gaze_tokens_each_frame.max(k.max(1)),
            self.task_loss_requirement,
        )
    }

    async fn trace_video_resize_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
    ) -> std::result::Result<Vec<FrameFixationTrace>, ExecutionError> {
        self.model
            .trace_video_with_task_loss_requirement_async(
                video,
                k,
                self.max_gaze_tokens_each_frame.max(k.max(1)),
                self.task_loss_requirement,
            )
            .await
    }

    fn collect_tiled_trace_points(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        layout: &AutoGazeTileLayout,
        frame_points: &mut [Vec<Vec<FixationPoint>>],
        stop_probabilities: &mut [Vec<f32>],
    ) {
        let batch = frame_points.len();
        let time = frame_points.first().map_or(0, Vec::len);
        let tile_trace_k = per_tile_trace_k(self.max_gaze_tokens_each_frame, k);
        if dense_grid_layout(layout) {
            let tile_video = dense_grid_video_tiles(video, layout);
            let accumulator = TileTraceAccumulator {
                batch,
                time,
                layout,
                frame_points,
                stop_probabilities,
            };
            self.collect_batched_tile_trace_points(tile_video, tile_trace_k, accumulator);
            return;
        }

        for tiles in layout.tiles.chunks(self.tile_batch_size) {
            let crops = tiles
                .iter()
                .copied()
                .map(|tile| crop_video_tile(video.clone(), tile))
                .collect::<Vec<_>>();
            let tile_traces = self.trace_video_resize(Tensor::cat(crops, 0), tile_trace_k);
            collect_tile_trace_points(
                &tile_traces,
                tiles,
                batch,
                time,
                layout,
                frame_points,
                stop_probabilities,
            );
        }
    }

    async fn collect_tiled_trace_points_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        layout: &AutoGazeTileLayout,
        frame_points: &mut [Vec<Vec<FixationPoint>>],
        stop_probabilities: &mut [Vec<f32>],
    ) -> std::result::Result<(), ExecutionError> {
        let batch = frame_points.len();
        let time = frame_points.first().map_or(0, Vec::len);
        let tile_trace_k = per_tile_trace_k(self.max_gaze_tokens_each_frame, k);
        if dense_grid_layout(layout) {
            let tile_video = dense_grid_video_tiles(video, layout);
            let accumulator = TileTraceAccumulator {
                batch,
                time,
                layout,
                frame_points,
                stop_probabilities,
            };
            self.collect_batched_tile_trace_points_async(tile_video, tile_trace_k, accumulator)
                .await?;
            return Ok(());
        }

        for tiles in layout.tiles.chunks(self.tile_batch_size) {
            let crops = tiles
                .iter()
                .copied()
                .map(|tile| crop_video_tile(video.clone(), tile))
                .collect::<Vec<_>>();
            let tile_traces = self
                .trace_video_resize_async(Tensor::cat(crops, 0), tile_trace_k)
                .await?;
            collect_tile_trace_points(
                &tile_traces,
                tiles,
                batch,
                time,
                layout,
                frame_points,
                stop_probabilities,
            );
        }
        Ok(())
    }

    fn collect_batched_tile_trace_points(
        &self,
        tile_video: Tensor<B, 5>,
        tile_trace_k: usize,
        accumulator: TileTraceAccumulator<'_>,
    ) {
        for (chunk_index, tiles) in accumulator
            .layout
            .tiles
            .chunks(self.tile_batch_size)
            .enumerate()
        {
            let start_tile = chunk_index.saturating_mul(self.tile_batch_size);
            let tile_video = tile_batch_slice(
                tile_video.clone(),
                accumulator.batch,
                start_tile,
                tiles.len(),
            );
            let tile_traces = self.trace_video_resize(tile_video, tile_trace_k);
            collect_tile_trace_points(
                &tile_traces,
                tiles,
                accumulator.batch,
                accumulator.time,
                accumulator.layout,
                &mut *accumulator.frame_points,
                &mut *accumulator.stop_probabilities,
            );
        }
    }

    async fn collect_batched_tile_trace_points_async(
        &self,
        tile_video: Tensor<B, 5>,
        tile_trace_k: usize,
        accumulator: TileTraceAccumulator<'_>,
    ) -> std::result::Result<(), ExecutionError> {
        for (chunk_index, tiles) in accumulator
            .layout
            .tiles
            .chunks(self.tile_batch_size)
            .enumerate()
        {
            let start_tile = chunk_index.saturating_mul(self.tile_batch_size);
            let tile_video = tile_batch_slice(
                tile_video.clone(),
                accumulator.batch,
                start_tile,
                tiles.len(),
            );
            let tile_traces = self
                .trace_video_resize_async(tile_video, tile_trace_k)
                .await?;
            collect_tile_trace_points(
                &tile_traces,
                tiles,
                accumulator.batch,
                accumulator.time,
                accumulator.layout,
                &mut *accumulator.frame_points,
                &mut *accumulator.stop_probabilities,
            );
        }
        Ok(())
    }

    fn tiled_readout_points(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        batch: usize,
        time: usize,
        layout: &AutoGazeTileLayout,
    ) -> Vec<Vec<Vec<FixationPoint>>> {
        let mut frame_points = empty_batch_frame_points(batch, time);
        let mut stop_probabilities = vec![vec![0.0f32; time]; batch];
        self.collect_tiled_trace_points(
            video,
            k,
            layout,
            &mut frame_points,
            &mut stop_probabilities,
        );
        sort_batch_frame_points_by_confidence(&mut frame_points);
        frame_points
    }

    async fn tiled_readout_points_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        batch: usize,
        time: usize,
        layout: &AutoGazeTileLayout,
    ) -> std::result::Result<Vec<Vec<Vec<FixationPoint>>>, ExecutionError> {
        let mut frame_points = empty_batch_frame_points(batch, time);
        let mut stop_probabilities = vec![vec![0.0f32; time]; batch];
        self.collect_tiled_trace_points_async(
            video,
            k,
            layout,
            &mut frame_points,
            &mut stop_probabilities,
        )
        .await?;
        sort_batch_frame_points_by_confidence(&mut frame_points);
        Ok(frame_points)
    }

    pub fn trace_clip_from_frames(
        &self,
        frames: &[f32],
        shape: AutoGazeClipShape,
        k: usize,
    ) -> Result<FrameFixationTrace> {
        ensure!(
            frames.len() == shape.num_values(),
            "expected {} frame values for clip shape {:?}, got {}",
            shape.num_values(),
            shape,
            frames.len()
        );
        Ok(self.model.trace_clip_from_frames(
            frames,
            shape.clip_len,
            shape.channels,
            shape.height,
            shape.width,
            k,
        ))
    }

    pub fn trace_clip_from_frames_with_mode(
        &self,
        frames: &[f32],
        shape: AutoGazeClipShape,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Result<FrameFixationTrace> {
        ensure!(
            frames.len() == shape.num_values(),
            "expected {} frame values for clip shape {:?}, got {}",
            shape.num_values(),
            shape,
            frames.len()
        );
        let device = self.model.gazing_model.connector.pos_embed.val().device();
        let clip = Tensor::<B, 5>::from_data(
            TensorData::new(
                frames.to_vec(),
                [
                    1,
                    shape.clip_len.max(1),
                    shape.channels.max(1),
                    shape.height.max(1),
                    shape.width.max(1),
                ],
            ),
            &device,
        );
        Ok(self
            .trace_video_with_mode(clip, k, mode)
            .into_iter()
            .next()
            .unwrap_or_else(|| FrameFixationTrace::new(vec![])))
    }

    pub async fn trace_clip_from_frames_with_mode_async(
        &self,
        frames: &[f32],
        shape: AutoGazeClipShape,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Result<FrameFixationTrace> {
        ensure!(
            frames.len() == shape.num_values(),
            "expected {} frame values for clip shape {:?}, got {}",
            shape.num_values(),
            shape,
            frames.len()
        );
        let device = self.model.gazing_model.connector.pos_embed.val().device();
        let clip = Tensor::<B, 5>::from_data(
            TensorData::new(
                frames.to_vec(),
                [
                    1,
                    shape.clip_len.max(1),
                    shape.channels.max(1),
                    shape.height.max(1),
                    shape.width.max(1),
                ],
            ),
            &device,
        );
        Ok(self
            .trace_video_with_mode_async(clip, k, mode)
            .await
            .map_err(|err| anyhow!("failed to read AutoGaze tensor data asynchronously: {err:?}"))?
            .into_iter()
            .next()
            .unwrap_or_else(|| FrameFixationTrace::new(vec![])))
    }

    pub fn trace_rgba_clip(
        &self,
        rgba: &[u8],
        shape: AutoGazeRgbaClipShape,
        k: usize,
        device: &B::Device,
    ) -> Result<Vec<FrameFixationTrace>> {
        self.trace_rgba_clip_with_mode(
            rgba,
            shape,
            k,
            AutoGazeInferenceMode::ResizeToModelInput,
            device,
        )
    }

    pub fn trace_rgba_clip_with_mode(
        &self,
        rgba: &[u8],
        shape: AutoGazeRgbaClipShape,
        k: usize,
        mode: AutoGazeInferenceMode,
        device: &B::Device,
    ) -> Result<Vec<FrameFixationTrace>> {
        let video = rgba_clip_to_inference_tensor::<B>(rgba, shape, mode, device)?;
        Ok(self.trace_video_with_mode(video, k, mode))
    }

    pub async fn trace_rgba_clip_with_mode_async(
        &self,
        rgba: &[u8],
        shape: AutoGazeRgbaClipShape,
        k: usize,
        mode: AutoGazeInferenceMode,
        device: &B::Device,
    ) -> Result<Vec<FrameFixationTrace>> {
        let video = rgba_clip_to_inference_tensor::<B>(rgba, shape, mode, device)?;
        self.trace_video_with_mode_async(video, k, mode)
            .await
            .map_err(|err| anyhow!("failed to read AutoGaze tensor data asynchronously: {err:?}"))
    }
}

fn tile_origins(length: usize, stride: usize) -> Vec<usize> {
    let length = length.max(1);
    let stride = stride.max(1);
    let mut origins = Vec::new();
    let mut origin = 0usize;
    while origin < length {
        origins.push(origin);
        origin = origin.saturating_add(stride);
    }
    origins
}

fn resize_video_to_layout_grid<B: Backend>(
    video: Tensor<B, 5>,
    layout: &AutoGazeTileLayout,
) -> Tensor<B, 5> {
    let [batch, time, channels, height, width] = video.shape().dims::<5>();
    if height == layout.source_height && width == layout.source_width {
        return video;
    }

    let video = video.reshape([batch * time, channels, height, width]);
    let video = interpolate(
        video,
        [layout.source_height, layout.source_width],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    video.reshape([
        batch,
        time,
        channels,
        layout.source_height,
        layout.source_width,
    ])
}

fn dense_grid_layout(layout: &AutoGazeTileLayout) -> bool {
    let tile_size = layout.tile_size.max(1);
    if layout.stride != tile_size
        || !layout.source_height.is_multiple_of(tile_size)
        || !layout.source_width.is_multiple_of(tile_size)
    {
        return false;
    }

    let rows = layout.source_height / tile_size;
    let cols = layout.source_width / tile_size;
    if rows == 0 || cols == 0 || layout.tiles.len() != rows * cols {
        return false;
    }

    layout.tiles.iter().enumerate().all(|(idx, tile)| {
        let row = idx / cols;
        let col = idx % cols;
        tile.x == col * tile_size
            && tile.y == row * tile_size
            && tile.width == tile_size
            && tile.height == tile_size
    })
}

fn dense_grid_video_tiles<B: Backend>(
    video: Tensor<B, 5>,
    layout: &AutoGazeTileLayout,
) -> Tensor<B, 5> {
    debug_assert!(dense_grid_layout(layout));
    let tile_size = layout.tile_size.max(1);
    let [batch, time, channels, height, width] = video.shape().dims::<5>();
    debug_assert_eq!(height, layout.source_height);
    debug_assert_eq!(width, layout.source_width);
    let rows = height / tile_size;
    let cols = width / tile_size;
    video
        .reshape([batch * time, channels, rows, tile_size, cols, tile_size])
        .permute([2, 4, 0, 1, 3, 5])
        .reshape([rows * cols * batch, time, channels, tile_size, tile_size])
}

fn tile_batch_slice<B: Backend>(
    tile_video: Tensor<B, 5>,
    batch: usize,
    start_tile: usize,
    tile_count: usize,
) -> Tensor<B, 5> {
    let start = start_tile.saturating_mul(batch.max(1));
    let end = start.saturating_add(tile_count.saturating_mul(batch.max(1)));
    tile_video.slice_dim(0, start..end)
}

fn crop_video_tile<B: Backend>(video: Tensor<B, 5>, tile: AutoGazeTile) -> Tensor<B, 5> {
    let [_batch, _time, _channels, source_height, source_width] = video.shape().dims::<5>();
    let y_end = tile.y.saturating_add(tile.height).min(source_height);
    let x_end = tile.x.saturating_add(tile.width).min(source_width);
    let crop_height = y_end.saturating_sub(tile.y).max(1);
    let crop_width = x_end.saturating_sub(tile.x).max(1);
    let crop = video
        .slice_dim(3, tile.y..(tile.y + crop_height))
        .slice_dim(4, tile.x..(tile.x + crop_width));
    let pad_height = tile.height.saturating_sub(crop_height);
    let pad_width = tile.width.saturating_sub(crop_width);
    if pad_height == 0 && pad_width == 0 {
        crop
    } else {
        crop.pad(
            [(0, 0), (0, 0), (0, 0), (0, pad_height), (0, pad_width)],
            PadMode::Constant(0.0),
        )
    }
}

fn reassemble_tile_embeddings<B: Backend>(
    embeddings: Tensor<B, 4>,
    tile_count: usize,
    batch: usize,
) -> Tensor<B, 4> {
    let tile_count = tile_count.max(1);
    let batch = batch.max(1);
    let [batched_tiles, time, tokens, dim] = embeddings.shape().dims::<4>();
    debug_assert_eq!(batched_tiles, tile_count * batch);
    embeddings
        .reshape([tile_count, batch, time, tokens, dim])
        .permute([1, 2, 0, 3, 4])
        .reshape([batch, time, tile_count * tokens, dim])
}

fn per_tile_trace_k(max_gaze_tokens_each_frame: usize, display_k: usize) -> usize {
    max_gaze_tokens_each_frame.max(display_k.max(1))
}

fn build_tiled_traces(
    frame_points: Vec<Vec<Vec<FixationPoint>>>,
    stop_probabilities: Vec<Vec<f32>>,
    frame_budget: usize,
) -> Vec<FrameFixationTrace> {
    frame_points
        .into_iter()
        .zip(stop_probabilities)
        .map(|(batch_frames, batch_stop_probabilities)| {
            let frames = batch_frames
                .into_iter()
                .zip(batch_stop_probabilities)
                .map(|(mut points, stop_probability)| {
                    sort_points_by_confidence(&mut points);
                    FixationSet::new(points, stop_probability, frame_budget)
                })
                .collect();
            FrameFixationTrace::new(frames)
        })
        .collect()
}

fn traces_to_frame_points(traces: Vec<FrameFixationTrace>) -> Vec<Vec<Vec<FixationPoint>>> {
    traces
        .into_iter()
        .map(|trace| trace.frames.into_iter().map(|frame| frame.points).collect())
        .collect()
}

fn empty_batch_frame_points(batch: usize, time: usize) -> Vec<Vec<Vec<FixationPoint>>> {
    (0..batch)
        .map(|_| (0..time).map(|_| Vec::<FixationPoint>::new()).collect())
        .collect()
}

fn sort_batch_frame_points_by_confidence(frame_points: &mut [Vec<Vec<FixationPoint>>]) {
    for batch_frames in frame_points {
        for points in batch_frames {
            sort_points_by_confidence(points);
        }
    }
}

fn sort_points_by_confidence(points: &mut [FixationPoint]) {
    points.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
}

fn collect_tile_trace_points(
    tile_traces: &[FrameFixationTrace],
    tiles: &[AutoGazeTile],
    batch: usize,
    time: usize,
    layout: &AutoGazeTileLayout,
    frame_points: &mut [Vec<Vec<FixationPoint>>],
    stop_probabilities: &mut [Vec<f32>],
) {
    for (local_tile_idx, tile) in tiles.iter().copied().enumerate() {
        for batch_idx in 0..batch {
            let trace_idx = local_tile_idx * batch + batch_idx;
            let Some(tile_trace) = tile_traces.get(trace_idx) else {
                continue;
            };
            for (frame_idx, fixation_set) in tile_trace.frames.iter().enumerate().take(time) {
                stop_probabilities[batch_idx][frame_idx] =
                    stop_probabilities[batch_idx][frame_idx].max(fixation_set.stop_probability);
                frame_points[batch_idx][frame_idx].extend(
                    fixation_set
                        .points
                        .iter()
                        .copied()
                        .filter(|point| point.confidence > 0.0)
                        .filter_map(|point| remap_tile_point(point, tile, layout)),
                );
            }
        }
    }
}

fn remap_tile_point(
    point: FixationPoint,
    tile: AutoGazeTile,
    layout: &AutoGazeTileLayout,
) -> Option<FixationPoint> {
    if layout.stride == layout.tile_size {
        return remap_non_overlapping_tile_point(point, tile, layout);
    }

    let source_width = layout.source_width.max(1) as f32;
    let source_height = layout.source_height.max(1) as f32;
    let center_x = tile.x as f32 + point.x * tile.width as f32;
    let center_y = tile.y as f32 + point.y * tile.height as f32;
    if center_x >= source_width || center_y >= source_height {
        return None;
    }
    Some(FixationPoint::with_grid_extent(
        center_x / source_width,
        center_y / source_height,
        point.cell_width() * tile.width.max(1) as f32 / source_width,
        point.cell_height() * tile.height.max(1) as f32 / source_height,
        point.confidence,
        point.cell_grid().unwrap_or(0),
    ))
}

fn remap_non_overlapping_tile_point(
    point: FixationPoint,
    tile: AutoGazeTile,
    layout: &AutoGazeTileLayout,
) -> Option<FixationPoint> {
    let grid = point_scale_grid(point)?;
    let tile_size = layout.tile_size.max(1);
    let grid_width = recovered_scale_grid(layout.source_width, tile_size, grid);
    let grid_height = recovered_scale_grid(layout.source_height, tile_size, grid);
    let tile_col = tile.x / tile_size;
    let tile_row = tile.y / tile_size;
    let local_col = local_cell_index(point.x, grid);
    let local_row = local_cell_index(point.y, grid);
    let global_col = tile_col.saturating_mul(grid).saturating_add(local_col);
    let global_row = tile_row.saturating_mul(grid).saturating_add(local_row);
    if global_col >= grid_width || global_row >= grid_height {
        return None;
    }

    Some(FixationPoint::with_grid_extent(
        (global_col as f32 + 0.5) / grid_width as f32,
        (global_row as f32 + 0.5) / grid_height as f32,
        1.0 / grid_width as f32,
        1.0 / grid_height as f32,
        point.confidence,
        grid,
    ))
}

fn point_scale_grid(point: FixationPoint) -> Option<usize> {
    point.cell_grid().or_else(|| {
        let width_grid = (1.0 / point.cell_width()).round();
        let height_grid = (1.0 / point.cell_height()).round();
        if width_grid.is_finite()
            && height_grid.is_finite()
            && (width_grid - height_grid).abs() < 0.5
        {
            Some(width_grid.max(1.0) as usize)
        } else {
            None
        }
    })
}

fn recovered_scale_grid(source_extent: usize, tile_size: usize, tile_grid: usize) -> usize {
    source_extent
        .saturating_mul(tile_grid.max(1))
        .checked_div(tile_size.max(1))
        .unwrap_or(0)
        .max(1)
}

fn local_cell_index(position: f32, grid: usize) -> usize {
    ((position.clamp(0.0, 1.0) * grid.max(1) as f32).floor() as usize).min(grid.max(1) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Poll};

    #[cfg(feature = "ndarray")]
    #[test]
    fn rgba_clip_to_tensor_applies_autogaze_processor_affine() {
        let device = Default::default();
        let rgba = [10, 20, 30, 255, 40, 50, 60, 255];
        let shape = AutoGazeRgbaClipShape::new(1, 1, 2);
        let tensor = rgba_clip_to_tensor::<burn::backend::NdArray<f32>>(&rgba, shape, &device)
            .expect("rgba tensor");
        let values = tensor.into_data().to_vec::<f32>().expect("f32 tensor");

        let expected = [
            autogaze_processor_value(10, 0),
            autogaze_processor_value(40, 0),
            autogaze_processor_value(20, 1),
            autogaze_processor_value(50, 1),
            autogaze_processor_value(30, 2),
            autogaze_processor_value(60, 2),
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn rgba_clip_to_processor_tensor_preserves_aspect_before_model_square_resize() {
        let device = Default::default();
        let rgba = vec![128u8; 2 * 4 * 4];
        let shape = AutoGazeRgbaClipShape::new(1, 2, 4);
        let tensor =
            rgba_clip_to_processor_tensor::<burn::backend::NdArray<f32>>(&rgba, shape, &device)
                .expect("processor tensor");

        assert_eq!(
            tensor.shape().dims::<5>(),
            [1, 1, 3, AUTO_GAZE_PROCESSOR_SHORT_EDGE, 448]
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn rgba_clip_to_tensor_preserves_clip_channel_order_and_ignores_alpha() {
        let device = Default::default();
        let rgba = [1, 2, 3, 4, 10, 20, 30, 250];
        let shape = AutoGazeRgbaClipShape::new(2, 1, 1);
        let tensor = rgba_clip_to_tensor::<burn::backend::NdArray<f32>>(&rgba, shape, &device)
            .expect("rgba tensor");
        let values = tensor.into_data().to_vec::<f32>().expect("f32 tensor");

        let expected = [
            autogaze_processor_value(1, 0),
            autogaze_processor_value(2, 1),
            autogaze_processor_value(3, 2),
            autogaze_processor_value(10, 0),
            autogaze_processor_value(20, 1),
            autogaze_processor_value(30, 2),
        ];
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn video_frame_tensor_slices_requested_frame_without_reordering() {
        let device = Default::default();
        let video = Tensor::<burn::backend::NdArray<f32>, 5>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], [1, 2, 3, 1, 1]),
            &device,
        );

        let frame = video_frame_tensor(video, 1)
            .expect("frame tensor")
            .into_data()
            .to_vec::<f32>()
            .expect("frame values");

        assert_eq!(frame, vec![10.0, 20.0, 30.0]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn rgba_clip_display_frame_comes_from_same_model_input_tensor() {
        let device = Default::default();
        let rgba = [1, 2, 3, 255, 10, 20, 30, 128];
        let shape = AutoGazeRgbaClipShape::new(2, 1, 1);
        let video = rgba_clip_to_tensor::<burn::backend::NdArray<f32>>(&rgba, shape, &device)
            .expect("rgba tensor");
        let frame = video_frame_tensor(video, 1)
            .expect("frame tensor")
            .into_data()
            .to_vec::<f32>()
            .expect("frame values");

        let expected = [
            autogaze_processor_value(10, 0),
            autogaze_processor_value(20, 1),
            autogaze_processor_value(30, 2),
        ];
        assert_eq!(frame.len(), expected.len());
        for (actual, expected) in frame.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn prepared_rgba_trace_run_uses_full_clip_without_streaming_cache() {
        let device = Default::default();
        let rgba = [1, 2, 3, 255, 10, 20, 30, 128];
        let shape = AutoGazeRgbaClipShape::new(2, 1, 1);

        let prepared = prepare_rgba_clip_for_trace::<burn::backend::NdArray<f32>>(
            &rgba,
            shape,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 224 },
            false,
            &device,
        )
        .expect("prepared run");

        assert_eq!(prepared.frame_index, 1);
        assert_eq!(prepared.model_frames, 2);
        assert_eq!(prepared.video.shape().dims::<5>(), [1, 2, 3, 1, 1]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn prepared_rgba_trace_run_uses_latest_frame_with_streaming_cache() {
        let device = Default::default();
        let rgba = [1, 2, 3, 255, 10, 20, 30, 128];
        let shape = AutoGazeRgbaClipShape::new(2, 1, 1);

        let prepared = prepare_rgba_clip_for_trace::<burn::backend::NdArray<f32>>(
            &rgba,
            shape,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 224 },
            true,
            &device,
        )
        .expect("prepared run");
        let values = prepared
            .video
            .into_data()
            .to_vec::<f32>()
            .expect("prepared tensor values");

        let expected = [
            autogaze_processor_value(10, 0),
            autogaze_processor_value(20, 1),
            autogaze_processor_value(30, 2),
        ];
        assert_eq!(prepared.frame_index, 0);
        assert_eq!(prepared.model_frames, 1);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn last_rgba_frame_rejects_mismatched_clip_lengths() {
        let rgba = [1, 2, 3, 255];
        let shape = AutoGazeRgbaClipShape::new(2, 1, 1);

        let err = last_rgba_frame(&rgba, shape).expect_err("expected length mismatch");

        assert!(
            err.to_string().contains("expected 8 RGBA bytes"),
            "unexpected error: {err:#}"
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn trace_prepared_run_preserves_metadata() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(1);
        let prepared = AutoGazePreparedRun {
            video: Tensor::zeros([1, 1, 3, 16, 16], &device),
            mode: AutoGazeInferenceMode::ResizeToModelInput,
            frame_index: 0,
            model_frames: 1,
        };

        let output = pipeline.trace_prepared_run(prepared, 1, None);

        assert_eq!(output.frame_index, 0);
        assert_eq!(output.model_frames, 1);
        assert_eq!(output.traces.len(), 1);
        assert_eq!(output.traces[0].frames.len(), 1);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn try_embed_model_input_rejects_non_square_model_input_without_panic() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        );
        let video = Tensor::zeros([1, 1, 3, 16, 8], &device);

        let err = match pipeline.try_embed_model_input(video) {
            Ok(_) => panic!("expected non-square model input to be rejected"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("must be square"),
            "unexpected error: {err:#}"
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn embed_model_input_prepares_non_model_sized_inputs_without_panic() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        );
        let video = Tensor::zeros([1, 1, 1, 16, 8], &device);

        let (embeddings, _past) = pipeline.embed_model_input(video);
        let [batch, frames, tokens, dim] = embeddings.shape().dims::<4>();

        assert_eq!(batch, 1);
        assert_eq!(frames, 1);
        assert_eq!(tokens, 1);
        assert_eq!(dim, 4);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn readout_prepared_run_preserves_metadata_and_matches_trace_points() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(1);
        let modes = [
            AutoGazeInferenceMode::ResizeToModelInput,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
        ];

        for mode in modes {
            let video = Tensor::zeros([1, 1, 3, 16, 16], &device);
            let trace_output = pipeline.trace_prepared_run(
                AutoGazePreparedRun {
                    video: video.clone(),
                    mode,
                    frame_index: 0,
                    model_frames: 1,
                },
                1,
                None,
            );
            let readout_output = pipeline.readout_prepared_run(
                AutoGazePreparedRun {
                    video,
                    mode,
                    frame_index: 0,
                    model_frames: 1,
                },
                1,
                None,
            );

            assert_eq!(readout_output.frame_index, trace_output.frame_index);
            assert_eq!(readout_output.model_frames, trace_output.model_frames);
            assert_eq!(
                readout_output.points,
                positive_trace_points(&trace_output.traces),
                "{mode:?}"
            );
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn async_readout_prepared_run_matches_sync() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(1);
        let video = Tensor::zeros([1, 1, 3, 16, 16], &device);
        let expected = pipeline.readout_prepared_run(
            AutoGazePreparedRun {
                video: video.clone(),
                mode: AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
                frame_index: 0,
                model_frames: 1,
            },
            1,
            None,
        );
        let actual = block_on_ready(pipeline.readout_prepared_run_async(
            AutoGazePreparedRun {
                video,
                mode: AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
                frame_index: 0,
                model_frames: 1,
            },
            1,
            None,
        ))
        .expect("async readout prepared run");

        assert_eq!(actual.frame_index, expected.frame_index);
        assert_eq!(actual.model_frames, expected.model_frames);
        assert_eq!(actual.points, expected.points);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn readout_points_match_trace_points_for_resize_and_tiled_modes() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(1);
        let modes = [
            AutoGazeInferenceMode::ResizeToModelInput,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
        ];

        for mode in modes {
            let video = Tensor::zeros([1, 1, 3, 16, 16], &device);
            let traces = pipeline.trace_video_with_mode(video.clone(), 1, mode);
            let readout_points = pipeline.readout_points_with_mode(video, 1, mode);

            assert_eq!(readout_points, positive_trace_points(&traces), "{mode:?}");
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn async_readout_points_match_sync_for_resize_and_tiled_modes() {
        let device = Default::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(1);
        let modes = [
            AutoGazeInferenceMode::ResizeToModelInput,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 16 },
        ];

        for mode in modes {
            let video = Tensor::zeros([1, 1, 3, 16, 16], &device);
            let expected = pipeline.readout_points_with_mode(video.clone(), 1, mode);
            let actual = block_on_ready(pipeline.readout_points_with_mode_async(video, 1, mode))
                .expect("async readout");

            assert_eq!(actual, expected, "{mode:?}");
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn dense_grid_video_tiles_matches_row_major_crop_cat_order() {
        let device = Default::default();
        let batch = 2;
        let time = 3;
        let channels = 1;
        let height = 4;
        let width = 6;
        let values = (0..batch * time * channels * height * width)
            .map(|idx| idx as f32)
            .collect::<Vec<_>>();
        let video = Tensor::<burn::backend::NdArray<f32>, 5>::from_data(
            TensorData::new(values, [batch, time, channels, height, width]),
            &device,
        );
        let layout = AutoGazeTileLayout::resized_grid(height, width, 2);
        assert!(dense_grid_layout(&layout));

        let expected = Tensor::cat(
            layout
                .tiles
                .iter()
                .copied()
                .map(|tile| crop_video_tile(video.clone(), tile))
                .collect::<Vec<_>>(),
            0,
        )
        .into_data()
        .to_vec::<f32>()
        .expect("expected dense grid crop values");
        let actual = dense_grid_video_tiles(video, &layout)
            .into_data()
            .to_vec::<f32>()
            .expect("actual dense grid values");

        assert_eq!(actual, expected);
    }

    #[test]
    fn remap_tile_point_recovers_upstream_scale_grid() {
        let point = FixationPoint::with_grid_extent(0.5, 0.5, 1.0 / 14.0, 1.0 / 14.0, 1.0, 14);
        let tile = AutoGazeTile::new(224, 224, 224, 224);
        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);

        let remapped = remap_tile_point(point, tile, &layout).expect("valid tile point");

        assert!((remapped.x - 21.5 / 120.0).abs() < 1.0e-6);
        assert!((remapped.y - 21.5 / 67.0).abs() < 1.0e-6);
        assert!((remapped.cell_width() - 1.0 / 120.0).abs() < 1.0e-6);
        assert!((remapped.cell_height() - 1.0 / 67.0).abs() < 1.0e-6);
        assert_eq!(remapped.cell_grid(), Some(14));
    }

    #[test]
    fn remap_tile_point_recovers_all_anyres_scale_grids() {
        let tile = AutoGazeTile::new(224, 224, 224, 224);
        let layout = AutoGazeTileLayout::tiled(448, 448, 224, 224);

        for (grid, expected_grid) in [(2, 4), (4, 8), (7, 14), (14, 28)] {
            let point = FixationPoint::with_grid_extent(
                (grid as f32 - 0.5) / grid as f32,
                (grid as f32 - 0.5) / grid as f32,
                1.0 / grid as f32,
                1.0 / grid as f32,
                1.0,
                grid,
            );
            let remapped = remap_tile_point(point, tile, &layout).expect("valid tile point");

            assert_eq!(remapped.cell_grid(), Some(grid));
            assert!(
                (remapped.x - (expected_grid as f32 - 0.5) / expected_grid as f32).abs() < 1.0e-6
            );
            assert!(
                (remapped.y - (expected_grid as f32 - 0.5) / expected_grid as f32).abs() < 1.0e-6
            );
            assert!((remapped.cell_width() - 1.0 / expected_grid as f32).abs() < 1.0e-6);
            assert!((remapped.cell_height() - 1.0 / expected_grid as f32).abs() < 1.0e-6);
        }
    }

    #[test]
    fn remap_tile_point_stitches_full_1080p_scale_grids_without_holes() {
        use std::collections::HashSet;

        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);

        for grid in [2, 4, 7, 14] {
            let grid_width = recovered_scale_grid(layout.source_width, layout.tile_size, grid);
            let grid_height = recovered_scale_grid(layout.source_height, layout.tile_size, grid);
            let mut seen = HashSet::with_capacity(grid_width * grid_height);

            for tile in layout.tiles.iter().copied() {
                for row in 0..grid {
                    for col in 0..grid {
                        let point = FixationPoint::with_grid_extent(
                            (col as f32 + 0.5) / grid as f32,
                            (row as f32 + 0.5) / grid as f32,
                            1.0 / grid as f32,
                            1.0 / grid as f32,
                            1.0,
                            grid,
                        );
                        let Some(remapped) = remap_tile_point(point, tile, &layout) else {
                            continue;
                        };

                        assert_eq!(remapped.cell_grid(), Some(grid));
                        assert!((remapped.cell_width() - 1.0 / grid_width as f32).abs() < 1.0e-6);
                        assert!((remapped.cell_height() - 1.0 / grid_height as f32).abs() < 1.0e-6);

                        let global_col =
                            ((remapped.x * grid_width as f32).floor() as usize).min(grid_width - 1);
                        let global_row = ((remapped.y * grid_height as f32).floor() as usize)
                            .min(grid_height - 1);
                        assert!(
                            seen.insert((global_col, global_row)),
                            "duplicate {grid}x{grid} remap cell {global_col},{global_row}"
                        );
                    }
                }
            }

            assert_eq!(
                seen.len(),
                grid_width * grid_height,
                "{grid}x{grid} remap left holes in {grid_width}x{grid_height} full-frame grid"
            );
        }
    }

    #[test]
    fn tiled_layout_pads_edges_instead_of_overlapping_them() {
        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);

        assert_eq!(layout.tile_count(), 45);
        assert_eq!(layout.tiles[8], AutoGazeTile::new(1792, 0, 224, 224));
        assert_eq!(layout.tiles[44], AutoGazeTile::new(1792, 896, 224, 224));
    }

    #[test]
    fn resized_grid_layout_uses_complete_anyres_tiles() {
        let layout = AutoGazeTileLayout::resized_grid(1080, 1920, 224);

        assert_eq!(layout.tile_count(), 45);
        assert_eq!(layout.source_width, 2016);
        assert_eq!(layout.source_height, 1120);
        assert_eq!(layout.tiles[8], AutoGazeTile::new(1792, 0, 224, 224));
        assert_eq!(layout.tiles[44], AutoGazeTile::new(1792, 896, 224, 224));
    }

    #[test]
    fn resized_grid_remap_preserves_complete_scale_grids() {
        let layout = AutoGazeTileLayout::resized_grid(1080, 1920, 224);

        for (grid, expected_width, expected_height) in
            [(2, 18, 10), (4, 36, 20), (7, 63, 35), (14, 126, 70)]
        {
            assert_eq!(
                recovered_scale_grid(layout.source_width, layout.tile_size, grid),
                expected_width
            );
            assert_eq!(
                recovered_scale_grid(layout.source_height, layout.tile_size, grid),
                expected_height
            );

            let point = FixationPoint::with_grid_extent(
                (grid as f32 - 0.5) / grid as f32,
                (grid as f32 - 0.5) / grid as f32,
                1.0 / grid as f32,
                1.0 / grid as f32,
                1.0,
                grid,
            );
            let bottom_right = layout.tiles[44];
            let remapped =
                remap_tile_point(point, bottom_right, &layout).expect("complete tile point");

            assert_eq!(remapped.cell_grid(), Some(grid));
            assert!(
                (remapped.x - (expected_width as f32 - 0.5) / expected_width as f32).abs() < 1.0e-6
            );
            assert!(
                (remapped.y - (expected_height as f32 - 0.5) / expected_height as f32).abs()
                    < 1.0e-6
            );
            assert!((remapped.cell_width() - 1.0 / expected_width as f32).abs() < 1.0e-6);
            assert!((remapped.cell_height() - 1.0 / expected_height as f32).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn crop_video_tile_pads_edge_chunks_to_model_tile_size() {
        type B = burn::backend::NdArray<f32>;
        let device = Default::default();
        let video = Tensor::<B, 5>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 2, 2]),
            &device,
        );

        let crop = crop_video_tile(video, AutoGazeTile::new(1, 1, 2, 2));
        let values = crop.into_data().to_vec::<f32>().expect("f32 tensor");

        assert_eq!(values, vec![4.0, 0.0, 0.0, 0.0]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn resized_grid_mode_resizes_video_to_complete_tile_canvas() {
        type B = burn::backend::NdArray<f32>;
        let device = Default::default();
        let video = Tensor::<B, 5>::from_data(
            TensorData::new((0..6).map(|value| value as f32).collect(), [1, 1, 1, 2, 3]),
            &device,
        );
        let layout = AutoGazeTileLayout::resized_grid(2, 3, 2);

        let resized = resize_video_to_layout_grid(video, &layout);

        assert_eq!(resized.shape().dims::<5>(), [1, 1, 1, 2, 4]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn resized_grid_mode_uses_bilinear_resampling_not_nearest() {
        type B = burn::backend::NdArray<f32>;
        let device = Default::default();
        let video = Tensor::<B, 5>::from_data(
            TensorData::new(vec![0.0, 10.0, 20.0, 30.0], [1, 1, 1, 2, 2]),
            &device,
        );
        let layout = AutoGazeTileLayout::resized_grid(4, 4, 4);

        let resized = resize_video_to_layout_grid(video, &layout)
            .into_data()
            .to_vec::<f32>()
            .expect("resized values");

        assert_eq!(resized.len(), 16);
        assert!(
            resized
                .iter()
                .any(|value| *value > 0.0 && *value < 10.0 && value.fract() != 0.0),
            "resized grid did not contain fractional interpolated values: {resized:?}"
        );
    }

    #[test]
    fn tiled_trace_uses_generation_budget_not_display_top_k_per_tile() {
        assert_eq!(per_tile_trace_k(198, 2), 198);
        assert_eq!(per_tile_trace_k(10, 24), 24);
        assert_eq!(per_tile_trace_k(0, 0), 1);
    }

    #[test]
    fn remap_tile_point_discards_padded_edge_cells() {
        let layout = AutoGazeTileLayout::tiled(3, 3, 2, 2);
        let tile = AutoGazeTile::new(2, 2, 2, 2);
        let padded = FixationPoint::with_grid_extent(0.75, 0.75, 0.5, 0.5, 1.0, 2);

        assert!(remap_tile_point(padded, tile, &layout).is_none());
    }

    #[test]
    fn remap_tile_point_discards_scale_grid_rows_beyond_source_extent() {
        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);
        let bottom_tile = AutoGazeTile::new(0, 896, 224, 224);
        let padded_row =
            FixationPoint::with_grid_extent(0.5, 5.5 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0, 7);
        let last_valid_row =
            FixationPoint::with_grid_extent(0.5, 4.5 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0, 7);

        assert!(remap_tile_point(padded_row, bottom_tile, &layout).is_none());
        let remapped =
            remap_tile_point(last_valid_row, bottom_tile, &layout).expect("last valid row");
        assert!((remapped.y - 32.5 / 33.0).abs() < 1.0e-6);
        assert!((remapped.cell_height() - 1.0 / 33.0).abs() < 1.0e-6);
    }

    #[test]
    fn tiled_fixation_budget_scales_with_tile_count() {
        let mode = AutoGazeInferenceMode::tiled_full_resolution(16, 16);
        let layout = AutoGazeTileLayout::tiled(32, 48, 16, 16);

        assert_eq!(layout.tile_count(), 6);
        assert_eq!(mode.fixation_budget(2, 32, 48), 12);
        assert_eq!(
            AutoGazeInferenceMode::resize_to_model_input().fixation_budget(2, 32, 48),
            2
        );
    }

    #[test]
    fn resized_grid_fixation_budget_scales_with_anyres_tile_count() {
        let mode = AutoGazeInferenceMode::tiled_resize_to_grid(16);
        let layout = AutoGazeTileLayout::resized_grid(31, 47, 16);

        assert_eq!(layout.tile_count(), 6);
        assert_eq!(mode.fixation_budget(2, 31, 47), 12);
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn pipeline_default_tile_batch_size_uses_shared_realtime_default() {
        let device =
            <burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        );

        assert_eq!(pipeline.tile_batch_size(), DEFAULT_TILED_TILE_BATCH_SIZE);
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn pipeline_options_apply_wrapper_runtime_overrides() {
        let device =
            <burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
        let pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_options(
            AutoGazePipelineOptions::default()
                .with_max_gaze_tokens_each_frame(7)
                .without_task_loss_requirement()
                .with_tile_batch_size(3),
        );

        assert_eq!(pipeline.max_gaze_tokens_each_frame(), 7);
        assert_eq!(pipeline.task_loss_requirement(), None);
        assert_eq!(pipeline.tile_batch_size(), 3);
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn pipeline_options_reset_to_model_defaults() {
        let device =
            <burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
        let mut pipeline = AutoGazePipeline::<burn::backend::NdArray<f32>>::from_config(
            &tiny_pipeline_config(),
            &device,
        )
        .with_max_gaze_tokens_each_frame(7)
        .with_task_loss_requirement(None)
        .with_tile_batch_size(3);

        pipeline.apply_options(AutoGazePipelineOptions::default());

        assert_eq!(
            pipeline.max_gaze_tokens_each_frame(),
            pipeline.model().default_max_gaze_tokens_each_frame()
        );
        assert_eq!(
            pipeline.task_loss_requirement(),
            pipeline.model().default_task_loss_requirement()
        );
        assert_eq!(pipeline.tile_batch_size(), 3);
    }

    #[cfg(feature = "ndarray")]
    fn tiny_pipeline_config() -> AutoGazeConfig {
        let hidden = 4;
        AutoGazeConfig {
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
            ..AutoGazeConfig::default()
        }
    }

    #[cfg(feature = "ndarray")]
    fn positive_trace_points(traces: &[FrameFixationTrace]) -> Vec<Vec<Vec<FixationPoint>>> {
        traces
            .iter()
            .map(|trace| {
                trace
                    .frames
                    .iter()
                    .map(|frame| {
                        frame
                            .points
                            .iter()
                            .copied()
                            .filter(|point| point.confidence > 0.0)
                            .collect()
                    })
                    .collect()
            })
            .collect()
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
}
