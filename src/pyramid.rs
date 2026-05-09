use crate::{FixationPoint, fixation_alpha_mask};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::module::{adaptive_avg_pool2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Int, Tensor, TensorData};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePyramidLevel {
    pub height: usize,
    pub width: usize,
}

impl ImagePyramidLevel {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub const fn token_count(&self) -> usize {
        self.height * self.width
    }

    fn normalized(self) -> Self {
        Self {
            height: self.height.max(1),
            width: self.width.max(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePyramidMaskOptions {
    pub threshold: f32,
}

impl Default for ImagePyramidMaskOptions {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

pub struct ImagePyramidMask<B: Backend> {
    pub level: ImagePyramidLevel,
    pub density: Tensor<B, 4>,
    pub active: Tensor<B, 4>,
}

pub struct ImagePyramidTokens<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub weights: Tensor<B, 3>,
    pub levels: Vec<ImagePyramidLevel>,
    pub level_token_ranges: Vec<Range<usize>>,
}

pub struct SparseImagePyramidTokens<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub weights: Tensor<B, 3>,
    pub indices: Tensor<B, 2, Int>,
}

pub fn fixation_image_mask_tensor<B: Backend>(
    batch: usize,
    height: usize,
    width: usize,
    points: &[FixationPoint],
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    ensure!(batch > 0, "mask batch must be nonzero");
    ensure!(height > 0 && width > 0, "mask dimensions must be nonzero");
    let alpha = fixation_alpha_mask(width, height, points, 1.0);
    let values = alpha
        .into_iter()
        .map(|value| if value > 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let mask = Tensor::<B, 4>::from_data(TensorData::new(values, [1, 1, height, width]), device);
    Ok(if batch == 1 {
        mask
    } else {
        mask.repeat_dim(0, batch)
    })
}

pub fn frame_fixation_masks_tensor<B: Backend>(
    traces: &[crate::FrameFixationTrace],
    frame_index: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    ensure!(
        !traces.is_empty(),
        "at least one fixation trace is required"
    );
    ensure!(height > 0 && width > 0, "mask dimensions must be nonzero");
    let mut values = Vec::with_capacity(traces.len() * height * width);
    for trace in traces {
        let points = trace
            .frames
            .get(frame_index)
            .map(|set| set.points.as_slice())
            .unwrap_or(&[]);
        values.extend(
            fixation_alpha_mask(width, height, points, 1.0)
                .into_iter()
                .map(|value| if value > 0 { 1.0 } else { 0.0 }),
        );
    }
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(values, [traces.len(), 1, height, width]),
        device,
    ))
}

pub fn apply_image_mask<B: Backend>(
    image: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    fill_value: f32,
) -> Result<Tensor<B, 4>> {
    let [batch, channels, height, width] = image.shape().dims::<4>();
    ensure_image_shape(batch, channels, height, width)?;
    let mask = image_channel_mask(mask, batch, channels, height, width)?;
    let fill = image.zeros_like().add_scalar(fill_value);
    let inverse = mask.clone().mul_scalar(-1.0).add_scalar(1.0);
    Ok(image * mask + fill * inverse)
}

pub fn image_pyramid_masks<B: Backend>(
    mask: Tensor<B, 4>,
    image_height: usize,
    image_width: usize,
    levels: &[ImagePyramidLevel],
    options: ImagePyramidMaskOptions,
) -> Result<Vec<ImagePyramidMask<B>>> {
    ensure!(
        image_height > 0 && image_width > 0,
        "image dimensions must be nonzero"
    );
    let [batch, _channels, _height, _width] = mask.shape().dims::<4>();
    ensure!(batch > 0, "mask batch must be nonzero");
    let mask = single_channel_mask(mask, batch, image_height, image_width)?;
    Ok(levels
        .iter()
        .copied()
        .map(ImagePyramidLevel::normalized)
        .map(|level| {
            let density =
                adaptive_avg_pool2d(mask.clone(), [level.height, level.width]).clamp(0.0, 1.0);
            let active = density.clone().greater_elem(options.threshold).float();
            ImagePyramidMask {
                level,
                density,
                active,
            }
        })
        .collect())
}

pub fn tokenize_masked_image_pyramid<B: Backend>(
    image: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    levels: &[ImagePyramidLevel],
    options: ImagePyramidMaskOptions,
) -> Result<ImagePyramidTokens<B>> {
    ensure!(
        !levels.is_empty(),
        "image pyramid must contain at least one level"
    );
    let [batch, channels, height, width] = image.shape().dims::<4>();
    ensure_image_shape(batch, channels, height, width)?;
    let mask = single_channel_mask(mask, batch, height, width)?;
    let mut level_token_ranges = Vec::with_capacity(levels.len());
    let mut normalized_levels = Vec::with_capacity(levels.len());
    let mut token_tensors = Vec::with_capacity(levels.len());
    let mut weight_tensors = Vec::with_capacity(levels.len());
    let mut offset = 0usize;

    for level in levels.iter().copied().map(ImagePyramidLevel::normalized) {
        let density =
            adaptive_avg_pool2d(mask.clone(), [level.height, level.width]).clamp(0.0, 1.0);
        let active = density.clone().greater_elem(options.threshold).float();
        let weights = density * active;
        let pooled = adaptive_avg_pool2d(image.clone(), [level.height, level.width]);
        let masked = pooled * weights.clone().repeat_dim(1, channels);
        let token_count = level.token_count();

        token_tensors.push(
            masked
                .reshape([batch, channels, token_count])
                .swap_dims(1, 2),
        );
        weight_tensors.push(weights.reshape([batch, 1, token_count]).swap_dims(1, 2));
        level_token_ranges.push(offset..offset + token_count);
        offset += token_count;
        normalized_levels.push(level);
    }

    Ok(ImagePyramidTokens {
        tokens: Tensor::cat(token_tensors, 1),
        weights: Tensor::cat(weight_tensors, 1),
        levels: normalized_levels,
        level_token_ranges,
    })
}

pub fn sparsify_image_pyramid_tokens<B: Backend>(
    tokens: ImagePyramidTokens<B>,
    max_tokens: usize,
) -> Result<SparseImagePyramidTokens<B>> {
    let [_batch, total_tokens, channels] = tokens.tokens.shape().dims::<3>();
    ensure!(total_tokens > 0, "image pyramid tokens must be nonempty");
    ensure!(channels > 0, "image pyramid token channels must be nonzero");
    let k = max_tokens.max(1).min(total_tokens);
    let scores = tokens.weights.clone().squeeze_dim::<2>(2);
    let (_values, indices) = scores.topk_with_indices(k, 1);
    let token_indices = indices
        .clone()
        .unsqueeze_dim::<3>(2)
        .repeat_dim(2, channels);
    let weight_indices = indices.clone().unsqueeze_dim::<3>(2);
    Ok(SparseImagePyramidTokens {
        tokens: tokens.tokens.gather(1, token_indices),
        weights: tokens.weights.gather(1, weight_indices),
        indices,
    })
}

fn ensure_image_shape(batch: usize, channels: usize, height: usize, width: usize) -> Result<()> {
    ensure!(batch > 0, "image batch must be nonzero");
    ensure!(channels > 0, "image channels must be nonzero");
    ensure!(height > 0 && width > 0, "image dimensions must be nonzero");
    Ok(())
}

fn single_channel_mask<B: Backend>(
    mask: Tensor<B, 4>,
    batch: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<B, 4>> {
    let [mask_batch, mask_channels, mask_height, mask_width] = mask.shape().dims::<4>();
    ensure!(
        mask_batch == 1 || mask_batch == batch,
        "mask batch must be 1 or match image batch"
    );
    ensure!(mask_channels > 0, "mask channels must be nonzero");
    let mask = if mask_channels == 1 {
        mask
    } else {
        mask.mean_dim(1)
    };
    let mask = if mask_height == height && mask_width == width {
        mask
    } else {
        interpolate(
            mask,
            [height, width],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    };
    let mask = if mask_batch == 1 && batch > 1 {
        mask.repeat_dim(0, batch)
    } else {
        mask
    };
    Ok(mask.clamp(0.0, 1.0))
}

fn image_channel_mask<B: Backend>(
    mask: Tensor<B, 4>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<B, 4>> {
    let mask = single_channel_mask(mask, batch, height, width)?;
    Ok(if channels == 1 {
        mask
    } else {
        mask.repeat_dim(1, channels)
    })
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn fixation_mask_tensor_matches_crisp_cell_bounds() {
        let device = Default::default();
        let point = FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 1.0);

        let mask = fixation_image_mask_tensor::<B>(1, 4, 4, &[point], &device).expect("mask");
        let values = mask.into_data().to_vec::<f32>().expect("f32 mask");

        for y in 0..4 {
            for x in 0..4 {
                let expected = if x < 2 && y < 2 { 1.0 } else { 0.0 };
                assert_eq!(values[y * 4 + x], expected, "{x},{y}");
            }
        }
    }

    #[test]
    fn frame_fixation_masks_tensor_keeps_batches_separate() {
        let device = Default::default();
        let left = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let right = FixationPoint::with_extent(0.75, 0.5, 0.5, 1.0, 1.0);
        let traces = vec![
            crate::FrameFixationTrace::new(vec![crate::FixationSet::new(vec![left], 1.0, 1)]),
            crate::FrameFixationTrace::new(vec![crate::FixationSet::new(vec![right], 1.0, 1)]),
        ];

        let mask = frame_fixation_masks_tensor::<B>(&traces, 0, 1, 2, &device).expect("mask");
        let values = mask.into_data().to_vec::<f32>().expect("mask values");

        assert_eq!(values, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn apply_image_mask_preserves_selected_pixels_and_fills_the_rest() {
        let device = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let mask = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]),
            &device,
        );

        let masked = apply_image_mask(image, mask, -1.0).expect("masked image");
        let values = masked.into_data().to_vec::<f32>().expect("f32 image");

        assert_eq!(values, vec![1.0, -1.0, -1.0, 4.0]);
    }

    #[test]
    fn tokenizes_masked_image_pyramid_with_density_weights() {
        let device = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let mask = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 2, 2]),
            &device,
        );

        let tokens = tokenize_masked_image_pyramid(
            image,
            mask,
            &[ImagePyramidLevel::new(1, 1), ImagePyramidLevel::new(2, 2)],
            ImagePyramidMaskOptions::default(),
        )
        .expect("tokens");
        let token_values = tokens
            .tokens
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("token values");
        let weight_values = tokens
            .weights
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("weight values");

        assert_eq!(tokens.level_token_ranges, vec![0..1, 1..5]);
        assert_eq!(weight_values, vec![0.25, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(token_values, vec![0.625, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn sparsifies_image_pyramid_tokens_by_highest_mask_density() {
        let device = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let mask = Tensor::<B, 4>::from_data(
            TensorData::new(vec![0.1, 0.9, 0.2, 0.7], [1, 1, 2, 2]),
            &device,
        );
        let tokens = tokenize_masked_image_pyramid(
            image,
            mask,
            &[ImagePyramidLevel::new(2, 2)],
            ImagePyramidMaskOptions::default(),
        )
        .expect("tokens");

        let sparse = sparsify_image_pyramid_tokens(tokens, 2).expect("sparse tokens");
        let weights = sparse.weights.into_data().to_vec::<f32>().expect("weights");

        assert_eq!(weights, vec![0.9, 0.7]);
    }
}
