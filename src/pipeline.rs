use crate::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeLoadOptions, FixationPoint, FixationSet,
    FrameFixationTrace, NativeAutoGazeModel,
};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor, TensorData};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeInferenceMode {
    #[default]
    ResizeToModelInput,
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

    pub fn tiled_model_input(model_input_size: usize) -> Self {
        let tile_size = model_input_size.max(1);
        Self::TiledFullResolution {
            tile_size,
            stride: tile_size,
        }
    }

    fn normalized(self) -> Self {
        match self {
            Self::ResizeToModelInput => Self::ResizeToModelInput,
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
    for frame in 0..shape.clip_len {
        let frame_offset = frame * pixels_per_frame * 4;
        for channel in 0..3 {
            for pixel in 0..pixels_per_frame {
                values.push(rgba[frame_offset + pixel * 4 + channel] as f32 / 255.0);
            }
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(values, [1, shape.clip_len, 3, shape.height, shape.width]),
        device,
    ))
}

#[derive(Clone, Debug)]
pub struct AutoGazePipeline<B: Backend> {
    model: NativeAutoGazeModel<B>,
    max_gaze_tokens_each_frame: usize,
    task_loss_requirement: Option<f32>,
}

impl<B: Backend> AutoGazePipeline<B> {
    pub fn new(model: NativeAutoGazeModel<B>) -> Self {
        let max_gaze_tokens_each_frame = model.default_max_gaze_tokens_each_frame();
        let task_loss_requirement = model.default_task_loss_requirement();
        Self {
            model,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
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

    pub fn with_max_gaze_tokens_each_frame(mut self, max_gaze_tokens_each_frame: usize) -> Self {
        self.max_gaze_tokens_each_frame = max_gaze_tokens_each_frame.max(1);
        self
    }

    pub fn with_task_loss_requirement(mut self, task_loss_requirement: Option<f32>) -> Self {
        self.task_loss_requirement = task_loss_requirement.map(|value| value.max(0.0));
        self
    }

    pub fn set_max_gaze_tokens_each_frame(&mut self, max_gaze_tokens_each_frame: usize) {
        self.max_gaze_tokens_each_frame = max_gaze_tokens_each_frame.max(1);
    }

    pub fn set_task_loss_requirement(&mut self, task_loss_requirement: Option<f32>) {
        self.task_loss_requirement = task_loss_requirement.map(|value| value.max(0.0));
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
        let [_batch, _time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => {
                let (embeddings, past_conv_values) = self.embed_video_resize(video);
                AutoGazeEmbedOutput {
                    embeddings,
                    past_conv_values,
                    layout: AutoGazeTileLayout::full_frame(height, width),
                }
            }
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                let mut tile_embeddings = Vec::with_capacity(layout.tile_count());
                for tile in layout.tiles.iter().copied() {
                    let crop = crop_video_tile(video.clone(), tile);
                    let (embeddings, _) = self.embed_video_resize(crop);
                    tile_embeddings.push(embeddings);
                }
                let embeddings = Tensor::cat(tile_embeddings, 2);
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

    pub fn trace_video_with_mode(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        mode: AutoGazeInferenceMode,
    ) -> Vec<FrameFixationTrace> {
        let [batch, time, _channels, height, width] = video.shape().dims::<5>();
        match mode.normalized() {
            AutoGazeInferenceMode::ResizeToModelInput => self.trace_video_resize(video, k),
            AutoGazeInferenceMode::TiledFullResolution { tile_size, stride } => {
                let layout = AutoGazeTileLayout::tiled(height, width, tile_size, stride);
                let frame_budget = k.max(1).saturating_mul(layout.tile_count().max(1));
                let mut frame_points = (0..batch)
                    .map(|_| (0..time).map(|_| Vec::<FixationPoint>::new()).collect())
                    .collect::<Vec<Vec<Vec<FixationPoint>>>>();
                let mut stop_probabilities = vec![vec![0.0f32; time]; batch];
                for tile in layout.tiles.iter().copied() {
                    let crop = crop_video_tile(video.clone(), tile);
                    let tile_traces = self.trace_video_resize(crop, k);
                    for batch_idx in 0..batch.min(tile_traces.len()) {
                        for (frame_idx, fixation_set) in
                            tile_traces[batch_idx].frames.iter().enumerate().take(time)
                        {
                            stop_probabilities[batch_idx][frame_idx] = stop_probabilities
                                [batch_idx][frame_idx]
                                .max(fixation_set.stop_probability);
                            frame_points[batch_idx][frame_idx].extend(
                                fixation_set
                                    .points
                                    .iter()
                                    .copied()
                                    .filter(|point| point.confidence > 0.0)
                                    .filter_map(|point| remap_tile_point(point, tile, &layout)),
                            );
                        }
                    }
                }
                frame_points
                    .into_iter()
                    .zip(stop_probabilities)
                    .map(|(batch_frames, batch_stop_probabilities)| {
                        let frames = batch_frames
                            .into_iter()
                            .zip(batch_stop_probabilities)
                            .map(|(mut points, stop_probability)| {
                                points.sort_by(|left, right| {
                                    right.confidence.total_cmp(&left.confidence)
                                });
                                FixationSet::new(points, stop_probability, frame_budget)
                            })
                            .collect();
                        FrameFixationTrace::new(frames)
                    })
                    .collect()
            }
        }
    }

    fn trace_video_resize(&self, video: Tensor<B, 5>, k: usize) -> Vec<FrameFixationTrace> {
        self.model.trace_video_with_task_loss_requirement(
            video,
            k,
            self.max_gaze_tokens_each_frame.max(k.max(1)),
            self.task_loss_requirement,
        )
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

fn remap_tile_point(
    point: FixationPoint,
    tile: AutoGazeTile,
    layout: &AutoGazeTileLayout,
) -> Option<FixationPoint> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn rgba_clip_to_tensor_converts_rgba_to_channel_first_rgb() {
        let device = Default::default();
        let rgba = [10, 20, 30, 255, 40, 50, 60, 255];
        let shape = AutoGazeRgbaClipShape::new(1, 1, 2);
        let tensor = rgba_clip_to_tensor::<burn::backend::NdArray<f32>>(&rgba, shape, &device)
            .expect("rgba tensor");
        let values = tensor.into_data().to_vec::<f32>().expect("f32 tensor");

        assert_eq!(
            values,
            vec![
                10.0 / 255.0,
                40.0 / 255.0,
                20.0 / 255.0,
                50.0 / 255.0,
                30.0 / 255.0,
                60.0 / 255.0,
            ]
        );
    }

    #[test]
    fn remap_tile_point_preserves_source_space_cell_extent() {
        let point = FixationPoint::with_grid_extent(0.5, 0.5, 1.0 / 14.0, 1.0 / 14.0, 1.0, 14);
        let tile = AutoGazeTile::new(224, 112, 224, 224);
        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);

        let remapped = remap_tile_point(point, tile, &layout).expect("valid tile point");

        assert!((remapped.x - 336.0 / 1920.0).abs() < 1.0e-6);
        assert!((remapped.y - 224.0 / 1080.0).abs() < 1.0e-6);
        assert!((remapped.cell_width() - (224.0 / 14.0) / 1920.0).abs() < 1.0e-6);
        assert!((remapped.cell_height() - (224.0 / 14.0) / 1080.0).abs() < 1.0e-6);
        assert_eq!(remapped.cell_grid(), Some(14));
    }

    #[test]
    fn tiled_layout_pads_edges_instead_of_overlapping_them() {
        let layout = AutoGazeTileLayout::tiled(1080, 1920, 224, 224);

        assert_eq!(layout.tile_count(), 45);
        assert_eq!(layout.tiles[8], AutoGazeTile::new(1792, 0, 224, 224));
        assert_eq!(layout.tiles[44], AutoGazeTile::new(1792, 896, 224, 224));
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

    #[test]
    fn remap_tile_point_discards_padded_edge_cells() {
        let layout = AutoGazeTileLayout::tiled(3, 3, 2, 2);
        let tile = AutoGazeTile::new(2, 2, 2, 2);
        let padded = FixationPoint::with_grid_extent(0.75, 0.75, 0.5, 0.5, 1.0, 2);

        assert!(remap_tile_point(padded, tile, &layout).is_none());
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
}
