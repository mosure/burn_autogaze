use crate::model::{generated_frame_fixations, generated_to_frame_fixations};
use crate::{
    AutoGazeConfig, AutoGazeGenerateOutput, FixationBounds, FixationPoint, FrameFixationTrace,
};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

const GRID_BOUNDARY_EPSILON: f32 = 1.0e-6;

/// Downstream image-token grid used to project AutoGaze regions into sparse readout tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseReadoutGrid {
    pub height: usize,
    pub width: usize,
}

impl SparseReadoutGrid {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub fn square_from_token_count(token_count: usize) -> Result<Self> {
        ensure!(
            token_count > 0,
            "sparse readout token count must be nonzero"
        );
        let width = (token_count as f64).sqrt() as usize;
        ensure!(
            width * width == token_count,
            "sparse readout token count must form a square grid"
        );
        Ok(Self::new(width, width))
    }

    pub const fn token_count(&self) -> usize {
        self.height * self.width
    }

    pub const fn is_empty(&self) -> bool {
        self.height == 0 || self.width == 0
    }

    pub fn token_index(&self, row: usize, col: usize) -> Option<usize> {
        (row < self.height && col < self.width).then_some(row * self.width + col)
    }

    pub fn token_rect(&self, token: usize) -> Option<SparseReadoutRect> {
        if self.is_empty() || token >= self.token_count() {
            return None;
        }
        let row = token / self.width;
        let col = token % self.width;
        let width = self.width as f32;
        let height = self.height as f32;
        Some(SparseReadoutRect {
            x0: col as f32 / width,
            y0: row as f32 / height,
            x1: (col + 1) as f32 / width,
            y1: (row + 1) as f32 / height,
        })
    }
}

/// Sparse video-token grid used to project per-frame image readout into
/// downstream tubelet or temporal-token layouts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseVideoReadoutGrid {
    pub temporal_bins: usize,
    pub height: usize,
    pub width: usize,
}

impl SparseVideoReadoutGrid {
    pub const fn new(temporal_bins: usize, height: usize, width: usize) -> Self {
        Self {
            temporal_bins,
            height,
            width,
        }
    }

    pub const fn token_count(&self) -> usize {
        self.temporal_bins * self.height * self.width
    }

    pub const fn tokens_per_temporal_bin(&self) -> usize {
        self.height * self.width
    }

    pub const fn is_empty(&self) -> bool {
        self.temporal_bins == 0 || self.height == 0 || self.width == 0
    }

    pub fn token_index(&self, temporal_bin: usize, row: usize, col: usize) -> Option<usize> {
        (temporal_bin < self.temporal_bins && row < self.height && col < self.width)
            .then_some((temporal_bin * self.height + row) * self.width + col)
    }

    pub fn token_coords(&self, token: usize) -> Option<(usize, usize, usize)> {
        if self.is_empty() || token >= self.token_count() {
            return None;
        }
        let tokens_per_bin = self.tokens_per_temporal_bin();
        let temporal_bin = token / tokens_per_bin;
        let spatial = token % tokens_per_bin;
        let row = spatial / self.width;
        let col = spatial % self.width;
        Some((temporal_bin, row, col))
    }
}

/// Video/tubelet/patch geometry for downstream sparse video patchifiers.
///
/// This is intentionally backend-agnostic. Crates such as `burn_jepa` can use
/// the derived [`SparseVideoReadoutGrid`] and coordinate tensors with
/// `burn_flex_gmm` without making `burn_autogaze` depend on that kernel crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseVideoPatchGeometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub tubelet_size: usize,
    pub patch_h: usize,
    pub patch_w: usize,
}

impl SparseVideoPatchGeometry {
    pub const fn new(
        frames: usize,
        height: usize,
        width: usize,
        tubelet_size: usize,
        patch_h: usize,
        patch_w: usize,
    ) -> Self {
        Self {
            frames,
            height,
            width,
            tubelet_size,
            patch_h,
            patch_w,
        }
    }

    pub const fn square_patch(
        frames: usize,
        height: usize,
        width: usize,
        tubelet_size: usize,
        patch_size: usize,
    ) -> Self {
        Self::new(frames, height, width, tubelet_size, patch_size, patch_size)
    }

    pub fn readout_grid(self) -> Result<SparseVideoReadoutGrid> {
        ensure!(self.frames > 0, "sparse video frame count must be nonzero");
        ensure!(
            self.height > 0 && self.width > 0,
            "sparse video dimensions must be nonzero"
        );
        ensure!(
            self.tubelet_size > 0,
            "sparse video tubelet size must be nonzero"
        );
        ensure!(
            self.patch_h > 0 && self.patch_w > 0,
            "sparse video patch dimensions must be nonzero"
        );
        ensure!(
            self.frames.is_multiple_of(self.tubelet_size),
            "video frames must be divisible by sparse video tubelet size"
        );
        ensure!(
            self.height.is_multiple_of(self.patch_h),
            "video height must be divisible by sparse video patch height"
        );
        ensure!(
            self.width.is_multiple_of(self.patch_w),
            "video width must be divisible by sparse video patch width"
        );
        Ok(SparseVideoReadoutGrid::new(
            self.frames / self.tubelet_size,
            self.height / self.patch_h,
            self.width / self.patch_w,
        ))
    }
}

/// Normalized image-space rectangle selected for sparse downstream readout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseReadoutRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl SparseReadoutRect {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        let x_min = x0.min(x1).clamp(0.0, 1.0);
        let x_max = x0.max(x1).clamp(0.0, 1.0);
        let y_min = y0.min(y1).clamp(0.0, 1.0);
        let y_max = y0.max(y1).clamp(0.0, 1.0);
        Self {
            x0: x_min,
            y0: y_min,
            x1: x_max,
            y1: y_max,
        }
    }

    pub fn from_bounds(bounds: FixationBounds) -> Self {
        Self::new(bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)
    }

    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

/// Projection options for converting AutoGaze fixations into sparse image readout tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseReadoutOptions {
    /// Discard fixation points with confidence less than or equal to this threshold.
    pub confidence_threshold: f32,
    /// Optional cap applied to decoded AutoGaze fixations before projecting
    /// multi-scale cells onto the downstream readout grid.
    pub max_fixations_per_frame: Option<usize>,
    /// Multiplier applied to each fixation cell before projecting onto the readout grid.
    pub fixation_scale: f32,
    /// Token-space dilation applied after projecting each region onto the readout grid.
    pub dilation: usize,
    /// Optional cap applied after deduplicating readout tokens in trace order.
    pub max_tokens_per_frame: Option<usize>,
}

/// Projection options for converting per-frame image readout into sparse video
/// token indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseVideoReadoutOptions {
    /// Number of source frames represented by one downstream temporal token.
    pub tubelet_size: usize,
    /// Token-space dilation applied after projecting each region onto the
    /// downstream video-token grid.
    pub dilation: usize,
    /// Optional lower bound filled with deterministic fallback tokens after
    /// projected tokens are deduplicated.
    pub min_tokens: usize,
    /// Optional upper bound applied after deduplicating projected tokens.
    pub max_tokens: Option<usize>,
}

impl Default for SparseVideoReadoutOptions {
    fn default() -> Self {
        Self {
            tubelet_size: 1,
            dilation: 0,
            min_tokens: 0,
            max_tokens: None,
        }
    }
}

impl SparseVideoReadoutOptions {
    pub const fn with_tubelet_size(mut self, tubelet_size: usize) -> Self {
        self.tubelet_size = tubelet_size;
        self
    }

    pub const fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub const fn with_min_tokens(mut self, min_tokens: usize) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_exact_tokens(mut self, tokens: usize) -> Self {
        self.min_tokens = tokens;
        self.max_tokens = Some(tokens);
        self
    }
}

/// Complete sparse-video projection settings for adapting AutoGaze image
/// readout into a downstream video-token grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseVideoReadoutProjection {
    pub image_grid: SparseReadoutGrid,
    pub video_grid: SparseVideoReadoutGrid,
    pub readout_options: SparseReadoutOptions,
    pub video_options: SparseVideoReadoutOptions,
}

impl SparseVideoReadoutProjection {
    pub fn new(image_grid: SparseReadoutGrid, video_grid: SparseVideoReadoutGrid) -> Self {
        Self {
            image_grid,
            video_grid,
            readout_options: SparseReadoutOptions::default(),
            video_options: SparseVideoReadoutOptions::default(),
        }
    }

    pub fn from_patch_geometry(
        image_grid: SparseReadoutGrid,
        patch_geometry: SparseVideoPatchGeometry,
    ) -> Result<Self> {
        Ok(Self::new(image_grid, patch_geometry.readout_grid()?))
    }

    pub const fn with_readout_options(mut self, readout_options: SparseReadoutOptions) -> Self {
        self.readout_options = readout_options;
        self
    }

    pub const fn with_video_options(mut self, video_options: SparseVideoReadoutOptions) -> Self {
        self.video_options = video_options;
        self
    }
}

impl Default for SparseReadoutOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.0,
            max_fixations_per_frame: None,
            fixation_scale: 1.0,
            dilation: 0,
            max_tokens_per_frame: None,
        }
    }
}

impl SparseReadoutOptions {
    pub fn with_confidence_threshold(mut self, confidence_threshold: f32) -> Self {
        self.confidence_threshold = confidence_threshold;
        self
    }

    pub const fn with_max_fixations_per_frame(mut self, max_fixations_per_frame: usize) -> Self {
        self.max_fixations_per_frame = Some(max_fixations_per_frame);
        self
    }

    pub fn with_fixation_scale(mut self, fixation_scale: f32) -> Self {
        self.fixation_scale = fixation_scale;
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub const fn with_max_tokens_per_frame(mut self, max_tokens_per_frame: usize) -> Self {
        self.max_tokens_per_frame = Some(max_tokens_per_frame);
        self
    }
}

/// Convert AutoGaze fixation points into normalized readout rectangles.
pub fn fixation_points_to_readout_rects(
    points: &[FixationPoint],
    options: SparseReadoutOptions,
) -> Vec<SparseReadoutRect> {
    if options.max_fixations_per_frame == Some(0) {
        return Vec::new();
    }
    let confidence_threshold = options.confidence_threshold.clamp(0.0, 1.0);
    let fixation_scale = options.fixation_scale.max(f32::EPSILON);
    let mut rects = Vec::new();
    for point in points
        .iter()
        .copied()
        .filter(|point| point.confidence > confidence_threshold)
    {
        if options
            .max_fixations_per_frame
            .is_some_and(|max_fixations| rects.len() >= max_fixations)
        {
            break;
        }
        let rect = SparseReadoutRect::from_bounds(point.scaled_bounds(fixation_scale));
        if !rect.is_empty() {
            rects.push(rect);
        }
    }
    rects
}

/// Project normalized readout rectangles onto a sparse image-token grid.
pub fn readout_rects_to_tokens(
    rects: &[SparseReadoutRect],
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<usize>> {
    ensure!(
        !grid.is_empty(),
        "sparse readout grid dimensions must be nonzero"
    );
    let Some(max_tokens) = options.max_tokens_per_frame else {
        return Ok(project_rects(rects, grid, options.dilation, None));
    };
    Ok(project_rects(
        rects,
        grid,
        options.dilation,
        Some(max_tokens),
    ))
}

/// Convert AutoGaze fixation points directly into sparse image readout token indices.
pub fn fixation_points_to_readout_tokens(
    points: &[FixationPoint],
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<usize>> {
    let rects = fixation_points_to_readout_rects(points, options);
    readout_rects_to_tokens(&rects, grid, options)
}

/// Convert one frame of an AutoGaze trace into normalized readout rectangles.
pub fn trace_frame_readout_rects(
    trace: &FrameFixationTrace,
    frame_index: usize,
    options: SparseReadoutOptions,
) -> Vec<SparseReadoutRect> {
    trace
        .frames
        .get(frame_index)
        .map(|frame| fixation_points_to_readout_rects(&frame.points, options))
        .unwrap_or_default()
}

/// Convert every frame in an AutoGaze trace into normalized readout rectangles.
pub fn trace_to_frame_readout_rects(
    trace: &FrameFixationTrace,
    options: SparseReadoutOptions,
) -> Vec<Vec<SparseReadoutRect>> {
    (0..trace.frames.len())
        .map(|frame_index| trace_frame_readout_rects(trace, frame_index, options))
        .collect()
}

/// Convert one frame of an AutoGaze trace into sparse image readout token indices.
pub fn trace_frame_readout_tokens(
    trace: &FrameFixationTrace,
    frame_index: usize,
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<usize>> {
    let rects = trace_frame_readout_rects(trace, frame_index, options);
    readout_rects_to_tokens(&rects, grid, options)
}

/// Convert every frame in an AutoGaze trace into sparse image readout token indices.
pub fn trace_to_frame_readout_tokens(
    trace: &FrameFixationTrace,
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<Vec<usize>>> {
    (0..trace.frames.len())
        .map(|frame_index| trace_frame_readout_tokens(trace, frame_index, grid, options))
        .collect()
}

/// Convert one frame from generated AutoGaze output into normalized readout rectangles.
///
/// This path uses the same multi-scale token decoding as `FrameFixationTrace`
/// generation, but avoids constructing a full trace when downstream code only
/// needs sparse image-token readout.
pub fn generated_frame_readout_rects(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    frame_index: usize,
    options: SparseReadoutOptions,
) -> Result<Vec<SparseReadoutRect>> {
    ensure!(
        batch_index < generated.gazing_pos.len(),
        "generated AutoGaze batch index out of range"
    );
    ensure!(
        frame_index < generated.num_gazing_each_frame.len(),
        "generated AutoGaze frame index out of range"
    );
    let Some(frame) = generated_frame_fixations(generated, config, batch_index, frame_index) else {
        anyhow::bail!("generated AutoGaze frame could not be decoded");
    };
    Ok(fixation_points_to_readout_rects(&frame.points, options))
}

/// Convert every frame from generated AutoGaze output into normalized readout rectangles.
pub fn generated_to_frame_readout_rects(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    options: SparseReadoutOptions,
) -> Result<Vec<Vec<SparseReadoutRect>>> {
    ensure!(
        batch_index < generated.gazing_pos.len(),
        "generated AutoGaze batch index out of range"
    );
    Ok(generated_to_frame_fixations(generated, config, batch_index)
        .into_iter()
        .map(|frame| fixation_points_to_readout_rects(&frame.points, options))
        .collect())
}

/// Convert one frame from generated AutoGaze output into sparse image readout token indices.
pub fn generated_frame_readout_tokens(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    frame_index: usize,
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<usize>> {
    let rects =
        generated_frame_readout_rects(generated, config, batch_index, frame_index, options)?;
    readout_rects_to_tokens(&rects, grid, options)
}

/// Convert every frame from generated AutoGaze output into sparse image readout token indices.
pub fn generated_to_frame_readout_tokens(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    grid: SparseReadoutGrid,
    options: SparseReadoutOptions,
) -> Result<Vec<Vec<usize>>> {
    generated_to_frame_readout_rects(generated, config, batch_index, options)?
        .iter()
        .map(|rects| readout_rects_to_tokens(rects, grid, options))
        .collect()
}

/// Project per-frame image readout token indices into a sparse video-token grid.
///
/// This is useful for downstream video models such as V-JEPA: first decode
/// AutoGaze output into image-token ids with `generated_to_frame_readout_tokens`,
/// then project those ids into the downstream `[temporal, row, col]` token grid.
pub fn frame_readout_tokens_to_video_tokens(
    frame_tokens: &[Vec<usize>],
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
) -> Result<Vec<usize>> {
    ensure!(
        !image_grid.is_empty(),
        "sparse image readout grid dimensions must be nonzero"
    );
    ensure!(
        !video_grid.is_empty(),
        "sparse video readout grid dimensions must be nonzero"
    );
    ensure!(
        options.tubelet_size > 0,
        "sparse video readout tubelet size must be nonzero"
    );
    let max_tokens = bounded_video_max_tokens(video_grid, options);
    if max_tokens == 0 {
        return Ok(Vec::new());
    }

    let mut tokens = Vec::new();
    let mut seen = vec![false; video_grid.token_count()];
    for temporal_bin in 0..video_grid.temporal_bins {
        let start = temporal_bin * options.tubelet_size;
        if start >= frame_tokens.len() {
            break;
        }
        let end = ((temporal_bin + 1) * options.tubelet_size).min(frame_tokens.len());
        for frame in &frame_tokens[start..end] {
            for &token in frame {
                let Some(rect) = image_grid.token_rect(token) else {
                    continue;
                };
                push_video_rect_tokens_limited(
                    rect,
                    temporal_bin,
                    video_grid,
                    options.dilation,
                    max_tokens,
                    &mut seen,
                    &mut tokens,
                );
                if tokens.len() >= max_tokens {
                    return Ok(tokens);
                }
            }
        }
    }

    fill_video_tokens(
        &mut tokens,
        &mut seen,
        video_grid,
        options.min_tokens,
        max_tokens,
    );
    Ok(tokens)
}

/// Project per-frame image readout token indices into sparse video-token
/// coordinates for downstream sparse patchification kernels.
pub fn frame_readout_tokens_to_video_coords(
    frame_tokens: &[Vec<usize>],
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
    batch_index: usize,
) -> Result<Vec<[u32; 4]>> {
    let tokens =
        frame_readout_tokens_to_video_tokens(frame_tokens, image_grid, video_grid, options)?;
    video_readout_tokens_to_coords(&tokens, video_grid, batch_index)
}

/// Project per-frame image readout token indices directly into a Burn sparse
/// video-coordinate tensor with shape `[rows, 4]`.
pub fn frame_readout_tokens_to_video_coord_tensor<B: Backend>(
    frame_tokens: &[Vec<usize>],
    projection: SparseVideoReadoutProjection,
    batch_index: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords = frame_readout_tokens_to_video_coords(
        frame_tokens,
        projection.image_grid,
        projection.video_grid,
        projection.video_options,
        batch_index,
    )?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

/// Project per-frame normalized readout rectangles into a sparse video-token grid.
pub fn frame_readout_rects_to_video_tokens(
    frame_rects: &[Vec<SparseReadoutRect>],
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
) -> Result<Vec<usize>> {
    ensure!(
        !video_grid.is_empty(),
        "sparse video readout grid dimensions must be nonzero"
    );
    ensure!(
        options.tubelet_size > 0,
        "sparse video readout tubelet size must be nonzero"
    );
    let max_tokens = bounded_video_max_tokens(video_grid, options);
    if max_tokens == 0 {
        return Ok(Vec::new());
    }

    let mut tokens = Vec::new();
    let mut seen = vec![false; video_grid.token_count()];
    for temporal_bin in 0..video_grid.temporal_bins {
        let start = temporal_bin * options.tubelet_size;
        if start >= frame_rects.len() {
            break;
        }
        let end = ((temporal_bin + 1) * options.tubelet_size).min(frame_rects.len());
        for rects in &frame_rects[start..end] {
            for &rect in rects {
                push_video_rect_tokens_limited(
                    rect,
                    temporal_bin,
                    video_grid,
                    options.dilation,
                    max_tokens,
                    &mut seen,
                    &mut tokens,
                );
                if tokens.len() >= max_tokens {
                    return Ok(tokens);
                }
            }
        }
    }

    fill_video_tokens(
        &mut tokens,
        &mut seen,
        video_grid,
        options.min_tokens,
        max_tokens,
    );
    Ok(tokens)
}

/// Project per-frame normalized readout rectangles into sparse video-token
/// coordinates for downstream sparse patchification kernels.
pub fn frame_readout_rects_to_video_coords(
    frame_rects: &[Vec<SparseReadoutRect>],
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
    batch_index: usize,
) -> Result<Vec<[u32; 4]>> {
    let tokens = frame_readout_rects_to_video_tokens(frame_rects, video_grid, options)?;
    video_readout_tokens_to_coords(&tokens, video_grid, batch_index)
}

/// Project per-frame normalized readout rectangles directly into a Burn sparse
/// video-coordinate tensor with shape `[rows, 4]`.
pub fn frame_readout_rects_to_video_coord_tensor<B: Backend>(
    frame_rects: &[Vec<SparseReadoutRect>],
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
    batch_index: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords =
        frame_readout_rects_to_video_coords(frame_rects, video_grid, options, batch_index)?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

/// Convert an AutoGaze trace directly into downstream sparse video-token
/// indices.
///
/// This is the convenience path for consumers that want an AutoGaze trace to
/// become a sparse-video mask without owning any AutoGaze scale-token logic.
pub fn trace_to_video_readout_tokens(
    trace: &FrameFixationTrace,
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    readout_options: SparseReadoutOptions,
    video_options: SparseVideoReadoutOptions,
) -> Result<Vec<usize>> {
    let frame_tokens = trace_to_frame_readout_tokens(trace, image_grid, readout_options)?;
    frame_readout_tokens_to_video_tokens(&frame_tokens, image_grid, video_grid, video_options)
}

/// Convert an AutoGaze trace directly into downstream sparse-video coordinates.
pub fn trace_to_video_readout_coords(
    trace: &FrameFixationTrace,
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    readout_options: SparseReadoutOptions,
    video_options: SparseVideoReadoutOptions,
    batch_index: usize,
) -> Result<Vec<[u32; 4]>> {
    let tokens = trace_to_video_readout_tokens(
        trace,
        image_grid,
        video_grid,
        readout_options,
        video_options,
    )?;
    video_readout_tokens_to_coords(&tokens, video_grid, batch_index)
}

/// Convert an AutoGaze trace directly into a Burn sparse-video coordinate
/// tensor with shape `[rows, 4]`.
pub fn trace_to_video_readout_coord_tensor<B: Backend>(
    trace: &FrameFixationTrace,
    projection: SparseVideoReadoutProjection,
    batch_index: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords = trace_to_video_readout_coords(
        trace,
        projection.image_grid,
        projection.video_grid,
        projection.readout_options,
        projection.video_options,
        batch_index,
    )?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

/// Convert generated AutoGaze output directly into downstream sparse
/// video-token indices.
///
/// This shares the same multi-scale generated-token decoder used by trace
/// creation, then applies the same image-token and video-token projection as
/// `trace_to_video_readout_tokens`.
pub fn generated_to_video_readout_tokens(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    readout_options: SparseReadoutOptions,
    video_options: SparseVideoReadoutOptions,
) -> Result<Vec<usize>> {
    let frame_tokens = generated_to_frame_readout_tokens(
        generated,
        config,
        batch_index,
        image_grid,
        readout_options,
    )?;
    frame_readout_tokens_to_video_tokens(&frame_tokens, image_grid, video_grid, video_options)
}

/// Convert generated AutoGaze output directly into downstream sparse-video
/// coordinates.
pub fn generated_to_video_readout_coords(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    image_grid: SparseReadoutGrid,
    video_grid: SparseVideoReadoutGrid,
    readout_options: SparseReadoutOptions,
    video_options: SparseVideoReadoutOptions,
) -> Result<Vec<[u32; 4]>> {
    let tokens = generated_to_video_readout_tokens(
        generated,
        config,
        batch_index,
        image_grid,
        video_grid,
        readout_options,
        video_options,
    )?;
    video_readout_tokens_to_coords(&tokens, video_grid, batch_index)
}

/// Convert generated AutoGaze output directly into a Burn sparse-video
/// coordinate tensor with shape `[rows, 4]`.
pub fn generated_to_video_readout_coord_tensor<B: Backend>(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_index: usize,
    projection: SparseVideoReadoutProjection,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords = generated_to_video_readout_coords(
        generated,
        config,
        batch_index,
        projection.image_grid,
        projection.video_grid,
        projection.readout_options,
        projection.video_options,
    )?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

/// Convert sparse video-token indices into `[batch, temporal, row, col]`
/// coordinates for downstream sparse patchification kernels.
///
/// This mirrors the coordinate rows expected by `burn_flex_gmm` sparse 3D
/// patchify without making `burn_autogaze` depend on that crate.
pub fn video_readout_tokens_to_coords(
    tokens: &[usize],
    grid: SparseVideoReadoutGrid,
    batch_index: usize,
) -> Result<Vec<[u32; 4]>> {
    ensure!(
        !grid.is_empty(),
        "sparse video readout grid dimensions must be nonzero"
    );
    let batch_index = u32_coord(batch_index, "batch index")?;
    tokens
        .iter()
        .copied()
        .map(|token| {
            let Some((temporal_bin, row, col)) = grid.token_coords(token) else {
                anyhow::bail!("sparse video readout token index outside grid");
            };
            Ok([
                batch_index,
                u32_coord(temporal_bin, "temporal bin")?,
                u32_coord(row, "row")?,
                u32_coord(col, "col")?,
            ])
        })
        .collect()
}

/// Convert sparse patchification coordinate rows into a Burn int tensor with
/// shape `[rows, 4]`.
pub fn video_readout_coords_to_tensor<B: Backend>(
    coords: &[[u32; 4]],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut flat = Vec::with_capacity(coords.len() * 4);
    for coord in coords {
        flat.extend(coord.iter().map(|&value| i64::from(value)));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [coords.len(), 4]), device)
}

/// Convert sparse video-token indices directly into a Burn coordinate tensor
/// with shape `[rows, 4]`.
pub fn video_readout_tokens_to_coord_tensor<B: Backend>(
    tokens: &[usize],
    grid: SparseVideoReadoutGrid,
    batch_index: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords = video_readout_tokens_to_coords(tokens, grid, batch_index)?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

/// Convert per-batch sparse video-token indices into flattened sparse
/// patchification coordinates.
pub fn batched_video_readout_tokens_to_coords(
    batch_tokens: &[Vec<usize>],
    grid: SparseVideoReadoutGrid,
) -> Result<Vec<[u32; 4]>> {
    let mut coords = Vec::new();
    for (batch_index, tokens) in batch_tokens.iter().enumerate() {
        coords.extend(video_readout_tokens_to_coords(tokens, grid, batch_index)?);
    }
    Ok(coords)
}

/// Convert per-batch sparse video-token indices directly into a flattened Burn
/// coordinate tensor with shape `[rows, 4]`.
pub fn batched_video_readout_tokens_to_coord_tensor<B: Backend>(
    batch_tokens: &[Vec<usize>],
    grid: SparseVideoReadoutGrid,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let coords = batched_video_readout_tokens_to_coords(batch_tokens, grid)?;
    Ok(video_readout_coords_to_tensor(&coords, device))
}

fn project_rects(
    rects: &[SparseReadoutRect],
    grid: SparseReadoutGrid,
    dilation: usize,
    max_tokens: Option<usize>,
) -> Vec<usize> {
    if max_tokens == Some(0) {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let mut seen = vec![false; grid.token_count()];
    for rect in rects.iter().copied().filter(|rect| !rect.is_empty()) {
        let Some((row_start, row_end, col_start, col_end)) = rect_grid_bounds(rect, grid, dilation)
        else {
            continue;
        };
        for row in row_start..row_end {
            for col in col_start..col_end {
                let token = row * grid.width + col;
                if seen[token] {
                    continue;
                }
                seen[token] = true;
                tokens.push(token);
                if max_tokens.is_some_and(|max_tokens| tokens.len() >= max_tokens) {
                    return tokens;
                }
            }
        }
    }
    tokens
}

fn bounded_video_max_tokens(
    video_grid: SparseVideoReadoutGrid,
    options: SparseVideoReadoutOptions,
) -> usize {
    options
        .max_tokens
        .unwrap_or_else(|| video_grid.token_count())
        .min(video_grid.token_count())
}

fn push_video_rect_tokens_limited(
    rect: SparseReadoutRect,
    temporal_bin: usize,
    video_grid: SparseVideoReadoutGrid,
    dilation: usize,
    max_tokens: usize,
    seen: &mut [bool],
    tokens: &mut Vec<usize>,
) {
    if tokens.len() >= max_tokens {
        return;
    }
    let spatial_grid = SparseReadoutGrid::new(video_grid.height, video_grid.width);
    let Some((row_start, row_end, col_start, col_end)) =
        rect_grid_bounds(rect, spatial_grid, dilation)
    else {
        return;
    };
    for row in row_start..row_end {
        for col in col_start..col_end {
            let Some(token) = video_grid.token_index(temporal_bin, row, col) else {
                continue;
            };
            if seen[token] {
                continue;
            }
            seen[token] = true;
            tokens.push(token);
            if tokens.len() >= max_tokens {
                return;
            }
        }
    }
}

fn fill_video_tokens(
    tokens: &mut Vec<usize>,
    seen: &mut [bool],
    video_grid: SparseVideoReadoutGrid,
    min_tokens: usize,
    max_tokens: usize,
) {
    let target = min_tokens.min(max_tokens).min(video_grid.token_count());
    if tokens.len() >= target {
        return;
    }
    for token in evenly_spaced_indices(video_grid.token_count(), target) {
        push_video_index_limited(token, target, seen, tokens);
        if tokens.len() >= target {
            return;
        }
    }
    for token in 0..video_grid.token_count() {
        push_video_index_limited(token, target, seen, tokens);
        if tokens.len() >= target {
            return;
        }
    }
}

fn push_video_index_limited(
    token: usize,
    target: usize,
    seen: &mut [bool],
    tokens: &mut Vec<usize>,
) {
    if tokens.len() >= target || token >= seen.len() || seen[token] {
        return;
    }
    seen[token] = true;
    tokens.push(token);
}

fn evenly_spaced_indices(dense_len: usize, keep: usize) -> Vec<usize> {
    let keep = keep.max(1).min(dense_len.max(1));
    if keep == dense_len {
        return (0..dense_len).collect();
    }
    let last = dense_len.saturating_sub(1);
    (0..keep)
        .map(|index| ((index * last) + (keep / 2)) / keep.max(1))
        .collect()
}

fn u32_coord(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("sparse video readout {name} exceeds u32"))
}

fn rect_grid_bounds(
    rect: SparseReadoutRect,
    grid: SparseReadoutGrid,
    dilation: usize,
) -> Option<(usize, usize, usize, usize)> {
    if rect.is_empty() || grid.is_empty() {
        return None;
    }
    let dilation = dilation as isize;
    let height = grid.height as f32;
    let width = grid.width as f32;
    let row_start = ((rect.y0 * height + GRID_BOUNDARY_EPSILON).floor() as isize - dilation)
        .clamp(0, grid.height as isize);
    let row_end = ((rect.y1 * height - GRID_BOUNDARY_EPSILON).ceil() as isize + dilation)
        .clamp(0, grid.height as isize);
    let col_start = ((rect.x0 * width + GRID_BOUNDARY_EPSILON).floor() as isize - dilation)
        .clamp(0, grid.width as isize);
    let col_end = ((rect.x1 * width - GRID_BOUNDARY_EPSILON).ceil() as isize + dilation)
        .clamp(0, grid.width as isize);
    (row_start < row_end && col_start < col_end).then_some((
        row_start as usize,
        row_end as usize,
        col_start as usize,
        col_end as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AutoGazeConfig, AutoGazeGenerateOutput, FixationSet, FrameFixationTrace};

    #[test]
    fn grid_token_rects_are_normalized() {
        let grid = SparseReadoutGrid::new(2, 4);
        assert_eq!(grid.token_index(1, 2), Some(6));
        assert_eq!(
            grid.token_rect(6),
            Some(SparseReadoutRect {
                x0: 0.5,
                y0: 0.5,
                x1: 0.75,
                y1: 1.0
            })
        );
        assert_eq!(grid.token_rect(8), None);
    }

    #[test]
    fn square_grid_can_be_built_from_token_count() {
        assert_eq!(
            SparseReadoutGrid::square_from_token_count(196).unwrap(),
            SparseReadoutGrid::new(14, 14)
        );
        let err = SparseReadoutGrid::square_from_token_count(198).unwrap_err();
        assert!(err.to_string().contains("square grid"));
    }

    #[test]
    fn video_grid_indices_are_temporal_row_major() {
        let grid = SparseVideoReadoutGrid::new(2, 3, 4);
        assert_eq!(grid.tokens_per_temporal_bin(), 12);
        assert_eq!(grid.token_index(1, 2, 3), Some(23));
        assert_eq!(grid.token_index(2, 0, 0), None);
        assert_eq!(grid.token_coords(0), Some((0, 0, 0)));
        assert_eq!(grid.token_coords(7), Some((0, 1, 3)));
        assert_eq!(grid.token_coords(23), Some((1, 2, 3)));
        assert_eq!(grid.token_coords(24), None);
    }

    #[test]
    fn patch_geometry_derives_sparse_patchifier_grid() {
        let geometry = SparseVideoPatchGeometry::square_patch(4, 64, 96, 2, 16);
        let grid = geometry.readout_grid().unwrap();
        let projection = SparseVideoReadoutProjection::from_patch_geometry(
            SparseReadoutGrid::new(2, 2),
            geometry,
        )
        .unwrap()
        .with_video_options(
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
        );

        assert_eq!(grid, SparseVideoReadoutGrid::new(2, 4, 6));
        assert_eq!(projection.video_grid, grid);

        let frame_tokens = vec![vec![], vec![1], vec![2], vec![]];
        let tokens = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            projection.image_grid,
            projection.video_grid,
            projection.video_options,
        )
        .unwrap();
        let coords = video_readout_tokens_to_coords(&tokens, projection.video_grid, 0).unwrap();

        assert_eq!(tokens, vec![3, 4, 5, 9]);
        assert_eq!(
            coords,
            vec![[0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5], [0, 0, 1, 3]]
        );
    }

    #[test]
    fn patch_geometry_rejects_non_patch_aligned_video_shapes() {
        let err = SparseVideoPatchGeometry::square_patch(5, 64, 96, 2, 16)
            .readout_grid()
            .unwrap_err();
        assert!(err.to_string().contains("frames"));

        let err = SparseVideoPatchGeometry::square_patch(4, 63, 96, 2, 16)
            .readout_grid()
            .unwrap_err();
        assert!(err.to_string().contains("height"));

        let err = SparseVideoPatchGeometry::new(4, 64, 95, 2, 16, 16)
            .readout_grid()
            .unwrap_err();
        assert!(err.to_string().contains("width"));
    }

    #[test]
    fn multiscale_fixation_expands_to_intersecting_readout_tokens() {
        let grid = SparseReadoutGrid::new(4, 4);
        let points = [FixationPoint::with_grid_extent(
            0.25, 0.25, 0.5, 0.5, 1.0, 2,
        )];
        let tokens =
            fixation_points_to_readout_tokens(&points, grid, SparseReadoutOptions::default())
                .unwrap();
        assert_eq!(tokens, vec![0, 1, 4, 5]);
    }

    #[test]
    fn fine_fixation_maps_to_single_readout_token() {
        let grid = SparseReadoutGrid::new(4, 4);
        let points = [FixationPoint::with_grid_extent(
            0.875, 0.875, 0.25, 0.25, 1.0, 4,
        )];
        let tokens =
            fixation_points_to_readout_tokens(&points, grid, SparseReadoutOptions::default())
                .unwrap();
        assert_eq!(tokens, vec![15]);
    }

    #[test]
    fn confidence_threshold_filters_padding_points() {
        let grid = SparseReadoutGrid::new(4, 4);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 0.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.875, 0.25, 0.25, 0.8, 4),
        ];
        let tokens = fixation_points_to_readout_tokens(
            &points,
            grid,
            SparseReadoutOptions::default().with_confidence_threshold(0.1),
        )
        .unwrap();
        assert_eq!(tokens, vec![15]);
    }

    #[test]
    fn max_fixations_caps_decoded_gaze_points_before_projection() {
        let grid = SparseReadoutGrid::new(4, 4);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.875, 0.25, 0.25, 1.0, 4),
        ];

        let tokens = fixation_points_to_readout_tokens(
            &points,
            grid,
            SparseReadoutOptions::default().with_max_fixations_per_frame(1),
        )
        .unwrap();
        let zero_tokens = fixation_points_to_readout_tokens(
            &points,
            grid,
            SparseReadoutOptions::default().with_max_fixations_per_frame(0),
        )
        .unwrap();

        assert_eq!(tokens, vec![0, 1, 4, 5]);
        assert!(zero_tokens.is_empty());
    }

    #[test]
    fn dilation_and_max_tokens_are_applied_after_deduplication() {
        let grid = SparseReadoutGrid::new(4, 4);
        let rects = [SparseReadoutRect::new(0.5, 0.5, 0.75, 0.75)];
        let tokens = readout_rects_to_tokens(
            &rects,
            grid,
            SparseReadoutOptions::default()
                .with_dilation(1)
                .with_max_tokens_per_frame(5),
        )
        .unwrap();
        assert_eq!(tokens, vec![5, 6, 7, 9, 10]);
    }

    #[test]
    fn zero_max_tokens_returns_empty_readout() {
        let grid = SparseReadoutGrid::new(4, 4);
        let rects = [SparseReadoutRect::new(0.0, 0.0, 1.0, 1.0)];
        let tokens = readout_rects_to_tokens(
            &rects,
            grid,
            SparseReadoutOptions::default().with_max_tokens_per_frame(0),
        )
        .unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn trace_projection_preserves_frame_boundaries() {
        let trace = FrameFixationTrace::new(vec![
            FixationSet::new(
                vec![FixationPoint::with_grid_extent(
                    0.25, 0.25, 0.5, 0.5, 1.0, 2,
                )],
                0.0,
                1,
            ),
            FixationSet::new(
                vec![FixationPoint::with_grid_extent(
                    0.875, 0.875, 0.25, 0.25, 1.0, 4,
                )],
                0.0,
                1,
            ),
        ]);
        let tokens = trace_to_frame_readout_tokens(
            &trace,
            SparseReadoutGrid::new(4, 4),
            SparseReadoutOptions::default(),
        )
        .unwrap();
        let rects = trace_to_frame_readout_rects(&trace, SparseReadoutOptions::default());
        assert_eq!(tokens, vec![vec![0, 1, 4, 5], vec![15]]);
        assert_eq!(rects.len(), 2);
        assert_eq!(rects[0], vec![SparseReadoutRect::new(0.0, 0.0, 0.5, 0.5)]);
        assert_eq!(rects[1], vec![SparseReadoutRect::new(0.75, 0.75, 1.0, 1.0)]);
    }

    #[test]
    fn generated_projection_matches_trace_multiscale_decoding_without_trace_allocation() {
        let mut config = AutoGazeConfig {
            scales: "32+64".to_string(),
            num_vision_tokens_each_frame: 20,
            ..AutoGazeConfig::default()
        };
        config.gaze_model_config.num_vision_tokens_each_frame = 20;
        config.gaze_model_config.vision_model_config.kernel_size = 16;
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 4, 39]],
            num_gazing_each_frame: vec![2, 1],
            if_padded_gazing: vec![vec![false, false, false]],
            confidences: vec![vec![1.0, 0.8, 0.9]],
        };
        let grid = SparseReadoutGrid::new(4, 4);
        let options = SparseReadoutOptions::default();

        let tokens =
            generated_to_frame_readout_tokens(&generated, &config, 0, grid, options).unwrap();
        let rects = generated_to_frame_readout_rects(&generated, &config, 0, options).unwrap();

        assert_eq!(tokens, vec![vec![0, 1, 4, 5], vec![15]]);
        assert_eq!(
            rects,
            vec![
                vec![
                    SparseReadoutRect::new(0.0, 0.0, 0.5, 0.5),
                    SparseReadoutRect::new(0.0, 0.0, 0.25, 0.25),
                ],
                vec![SparseReadoutRect::new(0.75, 0.75, 1.0, 1.0)],
            ]
        );
    }

    #[test]
    fn generated_projection_reports_invalid_batch_or_frame() {
        let config = AutoGazeConfig::default();
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0]],
            num_gazing_each_frame: vec![1],
            if_padded_gazing: vec![vec![false]],
            confidences: vec![vec![1.0]],
        };
        let grid = SparseReadoutGrid::new(1, 1);
        let options = SparseReadoutOptions::default();

        let err =
            generated_to_frame_readout_tokens(&generated, &config, 1, grid, options).unwrap_err();
        assert!(err.to_string().contains("batch index"));
        let err =
            generated_frame_readout_tokens(&generated, &config, 0, 1, grid, options).unwrap_err();
        assert!(err.to_string().contains("frame index"));
    }

    #[test]
    fn frame_readout_tokens_project_to_video_tubelet_tokens() {
        let image_grid = SparseReadoutGrid::new(2, 2);
        let video_grid = SparseVideoReadoutGrid::new(2, 4, 4);
        let frame_tokens = vec![vec![], vec![1], vec![2], vec![]];
        let tokens = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            image_grid,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
        )
        .unwrap();
        let coords = frame_readout_tokens_to_video_coords(
            &frame_tokens,
            image_grid,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
            3,
        )
        .unwrap();
        let device = Default::default();
        let coord_tensor =
            frame_readout_tokens_to_video_coord_tensor::<burn::backend::NdArray<f32>>(
                &frame_tokens,
                SparseVideoReadoutProjection::new(image_grid, video_grid).with_video_options(
                    SparseVideoReadoutOptions::default()
                        .with_tubelet_size(2)
                        .with_exact_tokens(4),
                ),
                3,
                &device,
            )
            .unwrap();

        assert_eq!(tokens, vec![2, 3, 6, 7]);
        assert_eq!(
            coords,
            vec![[3, 0, 0, 2], [3, 0, 0, 3], [3, 0, 1, 2], [3, 0, 1, 3]]
        );
        assert_eq!(
            coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![3, 0, 0, 2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 0, 1, 3]
        );
    }

    #[test]
    fn frame_readout_tokens_fill_to_minimum_video_tokens() {
        let image_grid = SparseReadoutGrid::new(2, 2);
        let video_grid = SparseVideoReadoutGrid::new(1, 4, 4);
        let frame_tokens = vec![vec![0]];
        let tokens = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            image_grid,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(1)
                .with_exact_tokens(6),
        )
        .unwrap();

        assert_eq!(tokens.len(), 6);
        for token in [0, 1, 4, 5] {
            assert!(tokens.contains(&token));
        }
    }

    #[test]
    fn frame_readout_tokens_allow_partial_video_window() {
        let image_grid = SparseReadoutGrid::new(1, 1);
        let video_grid = SparseVideoReadoutGrid::new(3, 2, 2);
        let frame_tokens = vec![vec![0]];
        let tokens = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            image_grid,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(5),
        )
        .unwrap();

        assert_eq!(tokens.len(), 5);
        for token in [0, 1, 2, 3] {
            assert!(tokens.contains(&token));
        }
    }

    #[test]
    fn frame_readout_rects_project_to_video_tubelet_tokens() {
        let video_grid = SparseVideoReadoutGrid::new(2, 4, 4);
        let mut frame_rects = vec![Vec::new(); 4];
        frame_rects[1].push(SparseReadoutRect::new(0.5, 0.0, 0.75, 0.25));
        let tokens = frame_readout_rects_to_video_tokens(
            &frame_rects,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_min_tokens(1),
        )
        .unwrap();
        let coords = frame_readout_rects_to_video_coords(
            &frame_rects,
            video_grid,
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_min_tokens(1),
            1,
        )
        .unwrap();
        let device = Default::default();
        let coord_tensor =
            frame_readout_rects_to_video_coord_tensor::<burn::backend::NdArray<f32>>(
                &frame_rects,
                video_grid,
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_min_tokens(1),
                1,
                &device,
            )
            .unwrap();

        assert_eq!(tokens, vec![2]);
        assert_eq!(coords, vec![[1, 0, 0, 2]]);
        assert_eq!(
            coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![1, 0, 0, 2]
        );
    }

    #[test]
    fn video_readout_tokens_convert_to_sparse_patchify_coords() {
        let grid = SparseVideoReadoutGrid::new(2, 3, 4);
        let coords =
            batched_video_readout_tokens_to_coords(&[vec![0, 7, 23], vec![0, 7, 23]], grid)
                .unwrap();
        let device = Default::default();
        let coord_tensor = batched_video_readout_tokens_to_coord_tensor::<
            burn::backend::NdArray<f32>,
        >(&[vec![0, 7, 23]], grid, &device)
        .unwrap();
        let coord_values = coord_tensor.into_data().to_vec::<i64>().unwrap();

        assert_eq!(
            coords,
            vec![
                [0, 0, 0, 0],
                [0, 0, 1, 3],
                [0, 1, 2, 3],
                [1, 0, 0, 0],
                [1, 0, 1, 3],
                [1, 1, 2, 3],
            ]
        );
        assert_eq!(coord_values, vec![0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 2, 3]);

        let err = video_readout_tokens_to_coords(&[24], grid, 0).unwrap_err();
        assert!(err.to_string().contains("outside grid"));
    }

    #[test]
    fn trace_projects_directly_to_video_readout_tokens() {
        let trace = FrameFixationTrace::new(vec![
            FixationSet::new(Vec::new(), 0.0, 1),
            FixationSet::new(
                vec![FixationPoint::with_grid_extent(
                    0.75, 0.25, 0.5, 0.5, 1.0, 2,
                )],
                0.0,
                1,
            ),
        ]);
        let tokens = trace_to_video_readout_tokens(
            &trace,
            SparseReadoutGrid::new(2, 2),
            SparseVideoReadoutGrid::new(1, 4, 4),
            SparseReadoutOptions::default(),
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
        )
        .unwrap();
        let coords = trace_to_video_readout_coords(
            &trace,
            SparseReadoutGrid::new(2, 2),
            SparseVideoReadoutGrid::new(1, 4, 4),
            SparseReadoutOptions::default(),
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
            2,
        )
        .unwrap();
        let device = Default::default();
        let coord_tensor = trace_to_video_readout_coord_tensor::<burn::backend::NdArray<f32>>(
            &trace,
            SparseVideoReadoutProjection::new(
                SparseReadoutGrid::new(2, 2),
                SparseVideoReadoutGrid::new(1, 4, 4),
            )
            .with_video_options(
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(4),
            ),
            2,
            &device,
        )
        .unwrap();

        assert_eq!(tokens, vec![2, 3, 6, 7]);
        assert_eq!(
            coords,
            vec![[2, 0, 0, 2], [2, 0, 0, 3], [2, 0, 1, 2], [2, 0, 1, 3]]
        );
        assert_eq!(
            coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![2, 0, 0, 2, 2, 0, 0, 3, 2, 0, 1, 2, 2, 0, 1, 3]
        );
    }

    #[test]
    fn generated_output_projects_directly_to_video_readout_tokens() {
        let mut config = AutoGazeConfig {
            scales: "32+64".to_string(),
            num_vision_tokens_each_frame: 20,
            ..AutoGazeConfig::default()
        };
        config.gaze_model_config.num_vision_tokens_each_frame = 20;
        config.gaze_model_config.vision_model_config.kernel_size = 16;
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 4, 39]],
            num_gazing_each_frame: vec![2, 1],
            if_padded_gazing: vec![vec![false, false, false]],
            confidences: vec![vec![1.0, 0.8, 0.9]],
        };
        let tokens = generated_to_video_readout_tokens(
            &generated,
            &config,
            0,
            SparseReadoutGrid::new(4, 4),
            SparseVideoReadoutGrid::new(1, 4, 4),
            SparseReadoutOptions::default().with_max_tokens_per_frame(4),
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
        )
        .unwrap();
        let coords = generated_to_video_readout_coords(
            &generated,
            &config,
            0,
            SparseReadoutGrid::new(4, 4),
            SparseVideoReadoutGrid::new(1, 4, 4),
            SparseReadoutOptions::default().with_max_tokens_per_frame(4),
            SparseVideoReadoutOptions::default()
                .with_tubelet_size(2)
                .with_exact_tokens(4),
        )
        .unwrap();
        let device = Default::default();
        let coord_tensor = generated_to_video_readout_coord_tensor::<burn::backend::NdArray<f32>>(
            &generated,
            &config,
            0,
            SparseVideoReadoutProjection::new(
                SparseReadoutGrid::new(4, 4),
                SparseVideoReadoutGrid::new(1, 4, 4),
            )
            .with_readout_options(SparseReadoutOptions::default().with_max_tokens_per_frame(4))
            .with_video_options(
                SparseVideoReadoutOptions::default()
                    .with_tubelet_size(2)
                    .with_exact_tokens(4),
            ),
            &device,
        )
        .unwrap();

        assert_eq!(tokens, vec![0, 1, 4, 5]);
        assert_eq!(
            coords,
            vec![[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
        );
        assert_eq!(
            coord_tensor.into_data().to_vec::<i64>().unwrap(),
            vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn generated_video_readout_matches_burn_jepa_benchmark_adapter_shape() {
        const FRAMES: usize = 4;
        const CONNECTOR_TOKENS: usize = 196;
        const PATCH_SIZE: usize = 16;
        const TUBELET_SIZE: usize = 2;
        const TOP_K: usize = 2;
        const KEEP_TOKENS: usize = 8;

        let mut config = AutoGazeConfig {
            scales: "224".to_string(),
            max_num_frames: FRAMES,
            num_vision_tokens_each_frame: CONNECTOR_TOKENS,
            ..AutoGazeConfig::default()
        };
        config.gaze_model_config.num_vision_tokens_each_frame = CONNECTOR_TOKENS;
        config.gaze_model_config.vision_model_config.kernel_size = PATCH_SIZE;

        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![
                0,
                10,
                12,
                250,
                196 + 30,
                196 + 31,
                196 + 44,
                392 + 130,
                392 + 131,
                588 - 1,
                588 + 195,
            ]],
            num_gazing_each_frame: vec![4, 3, 2, 2],
            if_padded_gazing: vec![vec![
                false, false, false, false, false, false, false, true, false, false, false,
            ]],
            confidences: vec![vec![1.0; 11]],
        };
        let image_grid = SparseReadoutGrid::square_from_token_count(CONNECTOR_TOKENS).unwrap();
        let video_grid =
            SparseVideoPatchGeometry::square_patch(FRAMES, 64, 64, TUBELET_SIZE, PATCH_SIZE)
                .readout_grid()
                .unwrap();
        let readout_options = SparseReadoutOptions::default().with_max_fixations_per_frame(TOP_K);
        let video_options = SparseVideoReadoutOptions::default()
            .with_tubelet_size(TUBELET_SIZE)
            .with_exact_tokens(KEEP_TOKENS);

        let legacy_frame_tokens =
            legacy_burn_jepa_generated_frame_tokens(&generated, FRAMES, TOP_K, CONNECTOR_TOKENS);
        let actual_frame_tokens =
            generated_to_frame_readout_tokens(&generated, &config, 0, image_grid, readout_options)
                .unwrap();
        let expected_video_tokens = legacy_burn_jepa_project_video_tokens(
            &legacy_frame_tokens,
            image_grid,
            video_grid,
            TUBELET_SIZE,
            KEEP_TOKENS,
        );
        let actual_video_tokens = generated_to_video_readout_tokens(
            &generated,
            &config,
            0,
            image_grid,
            video_grid,
            readout_options,
            video_options,
        )
        .unwrap();
        let actual_coords = generated_to_video_readout_coords(
            &generated,
            &config,
            0,
            image_grid,
            video_grid,
            readout_options,
            video_options,
        )
        .unwrap();

        assert_eq!(
            legacy_frame_tokens,
            vec![vec![0, 10], vec![30, 31], vec![131], vec![195]]
        );
        assert_eq!(actual_frame_tokens, legacy_frame_tokens);
        assert_eq!(actual_video_tokens, expected_video_tokens);
        assert_eq!(
            actual_coords,
            video_readout_tokens_to_coords(&expected_video_tokens, video_grid, 0).unwrap()
        );
    }

    #[test]
    fn video_projection_rejects_invalid_grids_and_tubelets() {
        let frame_tokens = vec![vec![0]];
        let err = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            SparseReadoutGrid::new(0, 1),
            SparseVideoReadoutGrid::new(1, 1, 1),
            SparseVideoReadoutOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("image readout grid"));

        let err = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            SparseReadoutGrid::new(1, 1),
            SparseVideoReadoutGrid::new(0, 1, 1),
            SparseVideoReadoutOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("video readout grid"));

        let err = frame_readout_tokens_to_video_tokens(
            &frame_tokens,
            SparseReadoutGrid::new(1, 1),
            SparseVideoReadoutGrid::new(1, 1, 1),
            SparseVideoReadoutOptions::default().with_tubelet_size(0),
        )
        .unwrap_err();
        assert!(err.to_string().contains("tubelet size"));
    }

    fn legacy_burn_jepa_generated_frame_tokens(
        generated: &AutoGazeGenerateOutput,
        frames: usize,
        top_k: usize,
        connector_tokens: usize,
    ) -> Vec<Vec<usize>> {
        let tokens = generated.gazing_pos.first();
        let padded = generated.if_padded_gazing.first();
        let mut cursor = 0usize;
        (0..frames)
            .map(|frame_idx| {
                let frame_len = generated
                    .num_gazing_each_frame
                    .get(frame_idx)
                    .copied()
                    .unwrap_or(0);
                let frame_offset = (frame_idx * connector_tokens) as i64;
                let mut frame_tokens = Vec::with_capacity(top_k.min(frame_len));
                for local_idx in 0..frame_len {
                    if frame_tokens.len() >= top_k {
                        break;
                    }
                    let token_index = cursor + local_idx;
                    if padded
                        .and_then(|flags| flags.get(token_index))
                        .copied()
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    let Some(raw_token) =
                        tokens.and_then(|tokens| tokens.get(token_index)).copied()
                    else {
                        continue;
                    };
                    let token = raw_token - frame_offset;
                    if token < 0 {
                        continue;
                    }
                    let token = token as usize;
                    if token < connector_tokens {
                        frame_tokens.push(token);
                    }
                }
                cursor += frame_len;
                frame_tokens
            })
            .collect()
    }

    fn legacy_burn_jepa_project_video_tokens(
        frame_tokens: &[Vec<usize>],
        image_grid: SparseReadoutGrid,
        video_grid: SparseVideoReadoutGrid,
        tubelet_size: usize,
        keep_tokens: usize,
    ) -> Vec<usize> {
        let target = keep_tokens.max(1).min(video_grid.token_count());
        let mut selected = Vec::with_capacity(target);
        let mut keep = vec![false; video_grid.token_count()];
        for tubelet in 0..video_grid.temporal_bins {
            let start = tubelet * tubelet_size;
            if start >= frame_tokens.len() {
                break;
            }
            let end = ((tubelet + 1) * tubelet_size).min(frame_tokens.len());
            for frame in &frame_tokens[start..end] {
                for &token in frame {
                    let Some(rect) = image_grid.token_rect(token) else {
                        continue;
                    };
                    legacy_burn_jepa_push_rect_tokens(
                        rect,
                        tubelet,
                        video_grid,
                        target,
                        &mut keep,
                        &mut selected,
                    );
                    if selected.len() >= target {
                        return selected;
                    }
                }
            }
        }

        for index in legacy_evenly_spaced(video_grid.token_count(), target) {
            legacy_push_sparse_index(index, target, &mut keep, &mut selected);
            if selected.len() >= target {
                return selected;
            }
        }
        for index in 0..video_grid.token_count() {
            legacy_push_sparse_index(index, target, &mut keep, &mut selected);
            if selected.len() >= target {
                break;
            }
        }
        selected
    }

    fn legacy_burn_jepa_push_rect_tokens(
        rect: SparseReadoutRect,
        tubelet: usize,
        video_grid: SparseVideoReadoutGrid,
        target: usize,
        keep: &mut [bool],
        selected: &mut Vec<usize>,
    ) {
        let Some((row_start, row_end, col_start, col_end)) =
            legacy_rect_patch_bounds(rect, video_grid)
        else {
            return;
        };
        for row in row_start..=row_end {
            for col in col_start..=col_end {
                let Some(index) = video_grid.token_index(tubelet, row, col) else {
                    continue;
                };
                legacy_push_sparse_index(index, target, keep, selected);
                if selected.len() >= target {
                    return;
                }
            }
        }
    }

    fn legacy_rect_patch_bounds(
        rect: SparseReadoutRect,
        grid: SparseVideoReadoutGrid,
    ) -> Option<(usize, usize, usize, usize)> {
        if rect.is_empty() || grid.height == 0 || grid.width == 0 {
            return None;
        }
        let col_start = ((rect.x0 * grid.width as f32).floor() as usize).min(grid.width - 1);
        let row_start = ((rect.y0 * grid.height as f32).floor() as usize).min(grid.height - 1);
        let col_end = ((rect.x1 * grid.width as f32).ceil() as usize)
            .saturating_sub(1)
            .min(grid.width - 1);
        let row_end = ((rect.y1 * grid.height as f32).ceil() as usize)
            .saturating_sub(1)
            .min(grid.height - 1);
        Some((row_start, row_end, col_start, col_end))
    }

    fn legacy_push_sparse_index(
        index: usize,
        target: usize,
        keep: &mut [bool],
        selected: &mut Vec<usize>,
    ) {
        if selected.len() >= target || index >= keep.len() || keep[index] {
            return;
        }
        keep[index] = true;
        selected.push(index);
    }

    fn legacy_evenly_spaced(dense_len: usize, keep: usize) -> Vec<usize> {
        let keep = keep.max(1).min(dense_len.max(1));
        if keep == dense_len {
            return (0..dense_len).collect();
        }
        let last = dense_len.saturating_sub(1);
        (0..keep)
            .map(|index| ((index * last) + (keep / 2)) / keep.max(1))
            .collect()
    }
}
