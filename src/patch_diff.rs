use crate::{
    AutoGazeDeviceMask, AutoGazeMaskPlanStats, AutoGazeReadoutRunOutput, AutoGazeReadoutStats,
    FixationPoint, FixationSet, FrameFixationTrace,
};
use anyhow::{Result, anyhow, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use std::cmp::Ordering;

pub const DEFAULT_PATCH_DIFF_GRID_SIZE: usize = 14;
pub const DEFAULT_PATCH_DIFF_THRESHOLD: f32 = 0.45;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazePatchDiffConfig {
    pub grid_size: usize,
    pub threshold: f32,
}

impl Default for AutoGazePatchDiffConfig {
    fn default() -> Self {
        Self {
            grid_size: DEFAULT_PATCH_DIFF_GRID_SIZE,
            threshold: DEFAULT_PATCH_DIFF_THRESHOLD,
        }
    }
}

impl AutoGazePatchDiffConfig {
    pub fn new(grid_size: usize, threshold: f32) -> Self {
        Self {
            grid_size: grid_size.max(1),
            threshold,
        }
    }

    pub fn normalized(self) -> Self {
        Self {
            grid_size: self.grid_size.max(1),
            threshold: if self.threshold.is_finite() {
                self.threshold.max(0.0)
            } else {
                DEFAULT_PATCH_DIFF_THRESHOLD
            },
        }
    }

    pub fn token_budget(self) -> usize {
        let config = self.normalized();
        config.grid_size.saturating_mul(config.grid_size)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AutoGazeSparseMaskSource {
    #[default]
    AutoGaze,
    PatchDiff(AutoGazePatchDiffConfig),
}

impl AutoGazeSparseMaskSource {
    pub const fn autogaze() -> Self {
        Self::AutoGaze
    }

    pub fn patch_diff(grid_size: usize, threshold: f32) -> Self {
        Self::PatchDiff(AutoGazePatchDiffConfig::new(grid_size, threshold))
    }

    pub const fn requires_autogaze_model(self) -> bool {
        matches!(self, Self::AutoGaze)
    }

    pub const fn is_patch_diff(self) -> bool {
        matches!(self, Self::PatchDiff(_))
    }
}

#[derive(Clone)]
pub struct AutoGazePatchDiffDeviceMask<B: Backend> {
    pub mask: AutoGazeDeviceMask<B>,
    pub grid_size: usize,
    pub active_cell_count: usize,
    pub frame_index: usize,
    pub model_frames: usize,
    pub stats: AutoGazeReadoutStats,
}

pub fn patch_diff_scores<B: Backend>(
    video: Tensor<B, 5>,
    config: AutoGazePatchDiffConfig,
) -> Result<Tensor<B, 2>> {
    let config = config.normalized();
    ensure!(
        config.threshold.is_finite() && config.threshold >= 0.0,
        "patch-diff threshold must be finite and non-negative"
    );
    let [batch, time, channels, height, width] = video.shape().dims::<5>();
    ensure!(batch > 0, "patch-diff video batch must be nonzero");
    ensure!(time > 0, "patch-diff video clip length must be nonzero");
    ensure!(channels > 0, "patch-diff video channels must be nonzero");
    ensure!(
        height > 0 && width > 0,
        "patch-diff video spatial dimensions must be nonzero"
    );

    let device = video.device();
    let grid_size = config.grid_size;
    if time < 2 {
        return Ok(Tensor::<B, 2>::zeros(
            [batch, grid_size.saturating_mul(grid_size)],
            &device,
        ));
    }

    let prev = video
        .clone()
        .slice_dim(1, time - 2..time - 1)
        .reshape([batch, channels, height, width]);
    let next = video
        .slice_dim(1, time - 1..time)
        .reshape([batch, channels, height, width]);
    let diff = (next - prev).abs();
    if height.is_multiple_of(grid_size) && width.is_multiple_of(grid_size) {
        let patch_height = height / grid_size;
        let patch_width = width / grid_size;
        let grid = diff
            .reshape([
                batch,
                channels,
                grid_size,
                patch_height,
                grid_size,
                patch_width,
            ])
            .mean_dim(5)
            .mean_dim(3)
            .mean_dim(1);
        return Ok(grid.reshape([batch, grid_size.saturating_mul(grid_size)]));
    }

    let channel_mean = if channels == 1 {
        diff
    } else {
        diff.mean_dim(1)
    };
    let grid = interpolate(
        channel_mean,
        [grid_size, grid_size],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    Ok(grid.reshape([batch, grid_size.saturating_mul(grid_size)]))
}

pub async fn patch_diff_device_mask_async<B: Backend>(
    video: Tensor<B, 5>,
    config: AutoGazePatchDiffConfig,
    height: usize,
    width: usize,
) -> Result<AutoGazePatchDiffDeviceMask<B>>
where
    f64: From<<B as burn::tensor::backend::BackendTypes>::FloatElem>,
{
    let config = config.normalized();
    ensure!(
        height > 0 && width > 0,
        "patch-diff mask dimensions must be nonzero"
    );
    let [_video_batch, time, _channels, _video_height, _video_width] = video.shape().dims::<5>();
    let scores = patch_diff_scores(video, config)?;
    let [batch, cells] = scores.shape().dims::<2>();
    ensure!(
        batch == 1,
        "patch-diff device mask currently supports one visualization batch"
    );
    ensure!(
        cells == config.token_budget(),
        "patch-diff score grid must match patch-diff config"
    );

    let grid_size = config.grid_size;
    let active_grid = scores
        .greater_elem(config.threshold)
        .float()
        .reshape([batch, 1, grid_size, grid_size]);
    let active_cell_count = active_grid
        .clone()
        .sum()
        .into_scalar_async()
        .await
        .map_err(|err| anyhow!("failed to read patch-diff active-cell count: {err:?}"))?;
    let alpha_grid = if height.is_multiple_of(grid_size) && width.is_multiple_of(grid_size) {
        active_grid
            .reshape([batch, 1, grid_size, 1, grid_size, 1])
            .repeat_dim(3, height / grid_size)
            .repeat_dim(5, width / grid_size)
            .reshape([batch, 1, height, width])
    } else {
        interpolate(
            active_grid,
            [height, width],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    };
    let alpha = alpha_grid
        .slice([0..1, 0..1, 0..height, 0..width])
        .reshape([height, width, 1]);
    let pixel_count = alpha
        .clone()
        .sum()
        .into_scalar_async()
        .await
        .map_err(|err| anyhow!("failed to read patch-diff mask pixel count: {err:?}"))?;
    let active_cell_count = scalar_count_to_usize(
        f64::from(active_cell_count),
        grid_size.saturating_mul(grid_size),
    );
    let pixel_count = scalar_count_to_usize(f64::from(pixel_count), height.saturating_mul(width));
    let stats = AutoGazeReadoutStats {
        generated_tokens: active_cell_count,
        active_generated_tokens: active_cell_count,
        padded_generated_tokens: 0,
    };

    Ok(AutoGazePatchDiffDeviceMask {
        mask: AutoGazeDeviceMask {
            alpha,
            mask_plan_stats: AutoGazeMaskPlanStats {
                rect_count: active_cell_count,
                row_span_count: height,
                pixel_count,
            },
        },
        grid_size,
        active_cell_count,
        frame_index: time.saturating_sub(1),
        model_frames: time,
        stats,
    })
}

pub fn patch_diff_readout_points<B: Backend>(
    video: Tensor<B, 5>,
    config: AutoGazePatchDiffConfig,
) -> Result<AutoGazeReadoutRunOutput> {
    let [batch, time, _channels, _height, _width] = video.shape().dims::<5>();
    let scores = patch_diff_scores(video, config)?;
    let score_values = scores
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow!("failed to read patch-diff score tensor: {err}"))?;
    Ok(patch_diff_points_from_scores(
        score_values,
        batch,
        time,
        config,
    )?)
}

pub async fn patch_diff_readout_points_async<B: Backend>(
    video: Tensor<B, 5>,
    config: AutoGazePatchDiffConfig,
) -> Result<AutoGazeReadoutRunOutput> {
    let [batch, time, _channels, _height, _width] = video.shape().dims::<5>();
    let scores = patch_diff_scores(video, config)?;
    let score_values = scores
        .into_data_async()
        .await
        .map_err(|err| anyhow!("failed to read patch-diff score tensor asynchronously: {err:?}"))?
        .to_vec::<f32>()
        .map_err(|err| anyhow!("failed to decode patch-diff score tensor: {err}"))?;
    patch_diff_points_from_scores(score_values, batch, time, config)
}

pub fn patch_diff_points_to_traces(
    points: &[Vec<Vec<FixationPoint>>],
    min_points: usize,
) -> Vec<FrameFixationTrace> {
    let min_points = min_points.max(1);
    points
        .iter()
        .map(|frames| {
            FrameFixationTrace::new(
                frames
                    .iter()
                    .map(|points| FixationSet::with_min_len(points.clone(), 1.0, min_points))
                    .collect(),
            )
        })
        .collect()
}

fn patch_diff_points_from_scores(
    score_values: Vec<f32>,
    batch: usize,
    time: usize,
    config: AutoGazePatchDiffConfig,
) -> Result<AutoGazeReadoutRunOutput> {
    let config = config.normalized();
    let grid_size = config.grid_size;
    let frame_index = time.saturating_sub(1);
    let scores_per_batch = grid_size.saturating_mul(grid_size);
    ensure!(
        score_values.len() == batch.saturating_mul(scores_per_batch),
        "patch-diff score tensor length must match batch and grid size"
    );

    let mut points = vec![vec![Vec::new(); time.max(1)]; batch];
    for batch_index in 0..batch {
        let start = batch_index * scores_per_batch;
        let end = start + scores_per_batch;
        let mut selected = score_values[start..end]
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| score.is_finite() && *score > config.threshold)
            .collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        points[batch_index][frame_index] = selected
            .into_iter()
            .map(|(index, score)| patch_diff_point(index, score, config))
            .collect();
    }

    let stats = patch_diff_readout_stats(&points, frame_index);
    Ok(AutoGazeReadoutRunOutput {
        points,
        frame_index,
        model_frames: time,
        stats,
    })
}

fn patch_diff_point(index: usize, score: f32, config: AutoGazePatchDiffConfig) -> FixationPoint {
    let grid_size = config.grid_size.max(1);
    let row = index / grid_size;
    let col = index % grid_size;
    let cell = 1.0 / grid_size as f32;
    let x = (col as f32 + 0.5) * cell;
    let y = (row as f32 + 0.5) * cell;
    let confidence = if config.threshold > 0.0 {
        (score / config.threshold).clamp(0.0, 1.0)
    } else {
        score.clamp(0.0, 1.0)
    };
    FixationPoint::with_grid_extent(x, y, cell, cell, confidence.max(f32::EPSILON), grid_size)
}

fn patch_diff_readout_stats(
    points: &[Vec<Vec<FixationPoint>>],
    frame_index: usize,
) -> AutoGazeReadoutStats {
    let active_generated_tokens = points
        .iter()
        .filter_map(|frames| frames.get(frame_index))
        .flat_map(|frame| frame.iter())
        .filter(|point| point.confidence > 0.0)
        .count();
    AutoGazeReadoutStats {
        generated_tokens: active_generated_tokens,
        active_generated_tokens,
        padded_generated_tokens: 0,
    }
}

fn scalar_count_to_usize(value: f64, max: usize) -> usize {
    if value.is_finite() {
        value.round().clamp(0.0, max as f64) as usize
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;

    #[test]
    fn patch_diff_selects_changed_patch_on_single_scale_grid() {
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
        let video = Tensor::<TestBackend, 5>::from_data(
            TensorData::new(values, [1, 2, 3, 28, 28]),
            &device,
        );

        let output =
            patch_diff_readout_points(video, AutoGazePatchDiffConfig::new(2, 0.25)).unwrap();
        let points = &output.points[0][output.frame_index];

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].grid, 2);
        assert!((points[0].x - 0.75).abs() < 1.0e-6);
        assert!((points[0].y - 0.75).abs() < 1.0e-6);
        assert!((points[0].cell_width() - 0.5).abs() < 1.0e-6);
        assert!((points[0].cell_height() - 0.5).abs() < 1.0e-6);
        assert_eq!(output.stats.active_generated_tokens, 1);
    }

    #[test]
    fn patch_diff_handles_fourteen_by_fourteen_grid_geometry() {
        let device = Default::default();
        let mut values = vec![0.0; 2 * 3 * 28 * 28];
        let frame_stride = 3 * 28 * 28;
        for channel in 0..3 {
            let channel_offset = frame_stride + channel * 28 * 28;
            for y in 10..12 {
                for x in 8..10 {
                    values[channel_offset + y * 28 + x] = 1.0;
                }
            }
        }
        let video = Tensor::<TestBackend, 5>::from_data(
            TensorData::new(values, [1, 2, 3, 28, 28]),
            &device,
        );

        let output =
            patch_diff_readout_points(video, AutoGazePatchDiffConfig::new(14, 0.25)).unwrap();
        let points = &output.points[0][output.frame_index];

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].grid, 14);
        assert!((points[0].cell_width() - (1.0 / 14.0)).abs() < 1.0e-6);
        assert!((points[0].cell_height() - (1.0 / 14.0)).abs() < 1.0e-6);
    }

    #[test]
    fn patch_diff_scores_use_exact_patch_average_when_divisible() {
        let device = Default::default();
        let mut values = vec![0.0; 2 * 3 * 4 * 4];
        let frame_stride = 3 * 4 * 4;
        for channel in 0..3 {
            let channel_offset = frame_stride + channel * 4 * 4;
            values[channel_offset] = 1.0;
            values[channel_offset + 1] = 1.0;
            values[channel_offset + 4] = 1.0;
            values[channel_offset + 5] = 0.0;
        }
        let video =
            Tensor::<TestBackend, 5>::from_data(TensorData::new(values, [1, 2, 3, 4, 4]), &device);

        let scores = patch_diff_scores(video, AutoGazePatchDiffConfig::new(2, 0.0))
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(scores.len(), 4);
        assert!((scores[0] - 0.75).abs() < 1.0e-6);
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 0.0);
        assert_eq!(scores[3], 0.0);
    }

    #[test]
    fn patch_diff_device_mask_keeps_mask_tensor_on_device() {
        let device = Default::default();
        let mut values = vec![0.0; 2 * 3 * 4 * 4];
        let frame_stride = 3 * 4 * 4;
        for channel in 0..3 {
            let channel_offset = frame_stride + channel * 4 * 4;
            for y in 0..2 {
                for x in 0..2 {
                    values[channel_offset + y * 4 + x] = 1.0;
                }
            }
        }
        let video =
            Tensor::<TestBackend, 5>::from_data(TensorData::new(values, [1, 2, 3, 4, 4]), &device);

        let output = futures_lite::future::block_on(patch_diff_device_mask_async(
            video,
            AutoGazePatchDiffConfig::new(2, 0.25),
            4,
            4,
        ))
        .unwrap();
        let alpha = output.mask.alpha.into_data().to_vec::<f32>().unwrap();

        assert_eq!(output.active_cell_count, 1);
        assert_eq!(output.stats.active_generated_tokens, 1);
        assert_eq!(output.mask.mask_plan_stats.pixel_count, 4);
        assert_eq!(alpha.iter().filter(|value| **value > 0.0).count(), 4);
    }

    #[test]
    fn patch_diff_single_frame_clip_emits_no_points() {
        let device = Default::default();
        let video = Tensor::<TestBackend, 5>::zeros([1, 1, 3, 28, 28], &device);

        let output =
            patch_diff_readout_points(video, AutoGazePatchDiffConfig::new(14, 0.0)).unwrap();

        assert!(output.points[0][output.frame_index].is_empty());
        assert_eq!(output.stats.active_generated_tokens, 0);
    }
}
