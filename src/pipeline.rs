use crate::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeLoadOptions, FixationPoint, FixationSet,
    FrameFixationTrace, NativeAutoGazeModel,
};
use anyhow::{Result, anyhow, ensure};
use burn::tensor::backend::{Backend, ExecutionError};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Tensor, TensorData};
use std::path::Path;

pub const AUTO_GAZE_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const AUTO_GAZE_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const AUTO_GAZE_RESCALE_FACTOR: f32 = 1.0 / 127.5;

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

    fn normalized(self) -> Self {
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

    let mut values = Vec::with_capacity(shape.clip_len * 3 * pixels_per_frame);
    // The upstream processor emits RGB video as [batch, time, channel, height, width].
    // Alpha is intentionally ignored.
    for frame in 0..shape.clip_len {
        let frame_offset = frame * pixels_per_frame * 4;
        for channel in 0..3 {
            for pixel in 0..pixels_per_frame {
                values.push(autogaze_processor_value(
                    rgba[frame_offset + pixel * 4 + channel],
                    channel,
                ));
            }
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(values, [1, shape.clip_len, 3, shape.height, shape.width]),
        device,
    ))
}

fn autogaze_processor_value(value: u8, channel: usize) -> f32 {
    let rescaled = value as f32 * AUTO_GAZE_RESCALE_FACTOR - 1.0;
    (rescaled - AUTO_GAZE_IMAGE_MEAN[channel]) / AUTO_GAZE_IMAGE_STD[channel]
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
            tile_batch_size: 8,
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
        let mut frame_points = (0..batch)
            .map(|_| (0..time).map(|_| Vec::<FixationPoint>::new()).collect())
            .collect::<Vec<Vec<Vec<FixationPoint>>>>();
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
        let mut frame_points = (0..batch)
            .map(|_| (0..time).map(|_| Vec::<FixationPoint>::new()).collect())
            .collect::<Vec<Vec<Vec<FixationPoint>>>>();
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
        let video = rgba_clip_to_tensor::<B>(rgba, shape, device)?;
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
        let video = rgba_clip_to_tensor::<B>(rgba, shape, device)?;
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
                    points.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
                    FixationSet::new(points, stop_probability, frame_budget)
                })
                .collect();
            FrameFixationTrace::new(frames)
        })
        .collect()
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
}
