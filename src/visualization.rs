use crate::{
    AutoGazeConfig, AutoGazeDeviceTokens, DEFAULT_KEYFRAME_DURATION, FixationPoint,
    pipeline::{AUTO_GAZE_IMAGE_MEAN, AUTO_GAZE_IMAGE_STD},
    scale_token_layouts,
};
use anyhow::{Result, ensure};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use std::{collections::BTreeMap, fmt, str::FromStr};

const DEFAULT_AUTOGAZE_SCALE_GRIDS: [usize; 4] = [2, 4, 7, 14];
const BLEND_LOOKUP_LEN: usize = 256 * 256;

#[derive(Clone, Debug, PartialEq)]
pub struct AutoGazeVisualization {
    pub width: usize,
    pub height: usize,
    pub side_by_side_width: usize,
    pub mask_rgba: Vec<u8>,
    pub blend_rgba: Vec<u8>,
    pub side_by_side_rgba: Vec<u8>,
    pub mask_pixel_count: usize,
    pub updated_pixel_count: usize,
    pub mask_plan_stats: AutoGazeMaskPlanStats,
}

impl AutoGazeVisualization {
    pub fn output_rgba(&self) -> &[u8] {
        &self.blend_rgba
    }

    pub fn mask_ratio(&self) -> f64 {
        ratio(self.mask_pixel_count, self.width * self.height)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.updated_pixel_count, self.width * self.height)
    }

    pub fn output_psnr_db(&self, input_rgba: &[u8]) -> Result<f64> {
        rgba_psnr_db(input_rgba, &self.blend_rgba)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoGazeVisualizationPanels {
    pub width: usize,
    pub height: usize,
    pub mask_rgba: Vec<u8>,
    pub blend_rgba: Vec<u8>,
    pub mask_pixel_count: usize,
    pub updated_pixel_count: usize,
    pub mask_plan_stats: AutoGazeMaskPlanStats,
}

impl AutoGazeVisualizationPanels {
    pub fn output_rgba(&self) -> &[u8] {
        &self.blend_rgba
    }

    pub fn mask_ratio(&self) -> f64 {
        ratio(self.mask_pixel_count, self.width * self.height)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.updated_pixel_count, self.width * self.height)
    }

    pub fn output_psnr_db(&self, input_rgba: &[u8]) -> Result<f64> {
        rgba_psnr_db(input_rgba, &self.blend_rgba)
    }

    pub fn into_side_by_side(self, input_rgba: &[u8]) -> Result<AutoGazeVisualization> {
        build_visualization(input_rgba, self)
    }
}

/// Reusable RGBA work buffers for allocation-stable CPU visualization.
///
/// The default visualization methods return owned `Vec<u8>` values for simple
/// downstream use. Hot paths can pass this buffer set to the `*_into` methods
/// to keep allocations stable across frames while still using the same core
/// mask, PSNR, and interframe logic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AutoGazeRgbaVisualizationBuffers {
    pub mask_rgba: Vec<u8>,
    pub blend_rgba: Vec<u8>,
    pub side_by_side_rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct InterframeMaskUpdate<'a> {
    plan: &'a AutoGazeSparseUpdatePlan,
    row_spans: &'a [(usize, usize, usize)],
    mask_rgba: &'a [u8],
    stats: AutoGazeMaskPlanStats,
}

impl AutoGazeRgbaVisualizationBuffers {
    pub fn clear(&mut self) {
        self.mask_rgba.clear();
        self.blend_rgba.clear();
        self.side_by_side_rgba.clear();
    }

    pub fn reserve_exact_panels(&mut self, width: usize, height: usize) -> Result<()> {
        let bytes = visualization_rgba_len(width, height)?;
        self.mask_rgba
            .reserve(bytes.saturating_sub(self.mask_rgba.capacity()));
        self.blend_rgba
            .reserve(bytes.saturating_sub(self.blend_rgba.capacity()));
        Ok(())
    }

    pub fn reserve_exact_side_by_side(&mut self, width: usize, height: usize) -> Result<()> {
        let bytes = visualization_rgba_len(
            width
                .checked_mul(3)
                .ok_or_else(|| anyhow::anyhow!("side-by-side visualization width overflow"))?,
            height,
        )?;
        self.side_by_side_rgba
            .reserve(bytes.saturating_sub(self.side_by_side_rgba.capacity()));
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazeVisualizationPanelsView<'a> {
    pub width: usize,
    pub height: usize,
    pub mask_rgba: &'a [u8],
    pub blend_rgba: &'a [u8],
    pub mask_pixel_count: usize,
    pub updated_pixel_count: usize,
    pub mask_plan_stats: AutoGazeMaskPlanStats,
}

impl<'a> AutoGazeVisualizationPanelsView<'a> {
    pub fn output_rgba(&self) -> &'a [u8] {
        self.blend_rgba
    }

    pub fn mask_ratio(&self) -> f64 {
        ratio(self.mask_pixel_count, self.width * self.height)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.updated_pixel_count, self.width * self.height)
    }

    pub fn output_psnr_db(&self, input_rgba: &[u8]) -> Result<f64> {
        rgba_psnr_db(input_rgba, self.blend_rgba)
    }

    pub fn to_owned(self) -> AutoGazeVisualizationPanels {
        AutoGazeVisualizationPanels {
            width: self.width,
            height: self.height,
            mask_rgba: self.mask_rgba.to_vec(),
            blend_rgba: self.blend_rgba.to_vec(),
            mask_pixel_count: self.mask_pixel_count,
            updated_pixel_count: self.updated_pixel_count,
            mask_plan_stats: self.mask_plan_stats,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeVisualizationMode {
    #[default]
    FullBlend,
    Interframe,
}

impl AutoGazeVisualizationMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullBlend => "full-blend",
            Self::Interframe => "interframe",
        }
    }
}

impl fmt::Display for AutoGazeVisualizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AutoGazeVisualizationMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "blend" | "full" | "full-blend" | "alphablend" | "alpha-blend" | "alpha" => {
                Ok(Self::FullBlend)
            }
            "interframe" | "inter-frame" | "video" | "video-encoding" | "delta" => {
                Ok(Self::Interframe)
            }
            other => Err(format!("unsupported AutoGaze visualization mode `{other}`")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeMaskVisualizationMode {
    Overlay,
    ImageOverlay,
    #[default]
    ImageMaskOnly,
    ScaleRows,
}

impl AutoGazeMaskVisualizationMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overlay => "overlay",
            Self::ImageOverlay => "image-overlay",
            Self::ImageMaskOnly => "image-mask-only",
            Self::ScaleRows => "scale-rows",
        }
    }
}

impl fmt::Display for AutoGazeMaskVisualizationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AutoGazeMaskVisualizationMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "overlay" | "combined" | "union" | "native" | "native-overlay" => Ok(Self::Overlay),
            "image-overlay"
            | "image"
            | "input-overlay"
            | "source-overlay"
            | "alpha-overlay"
            | "alpha-blend-overlay" => Ok(Self::ImageOverlay),
            "image-mask-only" | "mask-only" | "image-mask" | "masked-image" | "input-mask-only"
            | "source-mask-only" => Ok(Self::ImageMaskOnly),
            "scale-rows" | "rows" | "per-scale" | "upstream" | "nvidia" => Ok(Self::ScaleRows),
            other => Err(format!(
                "unsupported AutoGaze mask visualization mode `{other}`"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeMaskGeometryMode {
    #[default]
    Native,
    Deduplicated,
    Effective,
}

impl AutoGazeMaskGeometryMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Deduplicated => "deduplicated",
            Self::Effective => "effective",
        }
    }
}

impl fmt::Display for AutoGazeMaskGeometryMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AutoGazeMaskGeometryMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" | "raw" | "upstream" | "nvidia" | "multiscale" | "multi-scale" => {
                Ok(Self::Native)
            }
            "dedup" | "dedupe" | "deduplicated" | "union" | "union-dedup"
            | "union-deduplicated" | "non-overlap" | "non-overlapping" => Ok(Self::Deduplicated),
            "effective" | "projected" | "finest" | "finest-grid" | "sparse-footprint"
            | "token-footprint" => Ok(Self::Effective),
            other => Err(format!("unsupported AutoGaze mask geometry mode `{other}`")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazeRgbaVisualizationOptions {
    pub width: usize,
    pub height: usize,
    pub cell_scale: f32,
    pub blend_alpha: f32,
    pub mask_mode: AutoGazeMaskVisualizationMode,
    pub mask_geometry_mode: AutoGazeMaskGeometryMode,
    pub full_frame_update_min_ratio: f64,
}

impl AutoGazeRgbaVisualizationOptions {
    pub const fn new(width: usize, height: usize, cell_scale: f32, blend_alpha: f32) -> Self {
        Self {
            width,
            height,
            cell_scale,
            blend_alpha,
            mask_mode: AutoGazeMaskVisualizationMode::ImageMaskOnly,
            mask_geometry_mode: AutoGazeMaskGeometryMode::Native,
            full_frame_update_min_ratio: 0.0,
        }
    }

    pub const fn with_mask_visualization_mode(
        mut self,
        mode: AutoGazeMaskVisualizationMode,
    ) -> Self {
        self.mask_mode = mode;
        self
    }

    pub const fn with_mask_geometry_mode(mut self, mode: AutoGazeMaskGeometryMode) -> Self {
        self.mask_geometry_mode = mode;
        self
    }

    pub const fn with_full_frame_update_policy(mut self, min_update_ratio: f64) -> Self {
        self.full_frame_update_min_ratio = min_update_ratio;
        self
    }
}

#[derive(Clone, Debug)]
pub struct AutoGazeVisualizationState {
    mode: AutoGazeVisualizationMode,
    keyframe_duration: usize,
    frame_index: usize,
    interframe_output_rgba: Vec<u8>,
    interframe_width: usize,
    interframe_height: usize,
    last_keyframe: bool,
}

impl Default for AutoGazeVisualizationState {
    fn default() -> Self {
        Self::new(
            AutoGazeVisualizationMode::FullBlend,
            DEFAULT_KEYFRAME_DURATION,
        )
    }
}

impl AutoGazeVisualizationState {
    pub fn new(mode: AutoGazeVisualizationMode, keyframe_duration: usize) -> Self {
        Self {
            mode,
            keyframe_duration,
            frame_index: 0,
            interframe_output_rgba: Vec::new(),
            interframe_width: 0,
            interframe_height: 0,
            last_keyframe: false,
        }
    }

    pub fn mode(&self) -> AutoGazeVisualizationMode {
        self.mode
    }

    pub fn keyframe_duration(&self) -> usize {
        self.keyframe_duration
    }

    pub fn last_frame_was_keyframe(&self) -> bool {
        self.last_keyframe
    }

    pub fn configure(&mut self, mode: AutoGazeVisualizationMode, keyframe_duration: usize) {
        if self.mode != mode {
            self.reset();
        }
        self.mode = mode;
        self.keyframe_duration = keyframe_duration;
    }

    pub fn reset(&mut self) {
        self.frame_index = 0;
        self.interframe_output_rgba.clear();
        self.interframe_width = 0;
        self.interframe_height = 0;
        self.last_keyframe = false;
    }

    pub fn visualize_rgba(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        points: &[FixationPoint],
        cell_scale: f32,
        blend_alpha: f32,
    ) -> Result<AutoGazeVisualization> {
        self.visualize_rgba_with_options(
            rgba,
            points,
            AutoGazeRgbaVisualizationOptions::new(width, height, cell_scale, blend_alpha),
        )
    }

    pub fn visualize_rgba_with_options(
        &mut self,
        rgba: &[u8],
        points: &[FixationPoint],
        options: AutoGazeRgbaVisualizationOptions,
    ) -> Result<AutoGazeVisualization> {
        let mut buffers = AutoGazeRgbaVisualizationBuffers::default();
        let view =
            self.visualize_rgba_panels_with_options_into(rgba, points, options, &mut buffers)?;
        let mask_pixel_count = view.mask_pixel_count;
        let updated_pixel_count = view.updated_pixel_count;
        let mask_plan_stats = view.mask_plan_stats;
        build_visualization_from_buffers(
            rgba,
            options.width,
            options.height,
            &mut buffers,
            mask_pixel_count,
            updated_pixel_count,
            mask_plan_stats,
        )
    }

    pub fn visualize_rgba_panels(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        points: &[FixationPoint],
        cell_scale: f32,
        blend_alpha: f32,
    ) -> Result<AutoGazeVisualizationPanels> {
        self.visualize_rgba_panels_with_options(
            rgba,
            points,
            AutoGazeRgbaVisualizationOptions::new(width, height, cell_scale, blend_alpha),
        )
    }

    pub fn visualize_rgba_panels_with_options(
        &mut self,
        rgba: &[u8],
        points: &[FixationPoint],
        options: AutoGazeRgbaVisualizationOptions,
    ) -> Result<AutoGazeVisualizationPanels> {
        let mut buffers = AutoGazeRgbaVisualizationBuffers::default();
        let view =
            self.visualize_rgba_panels_with_options_into(rgba, points, options, &mut buffers)?;
        let width = view.width;
        let height = view.height;
        let mask_pixel_count = view.mask_pixel_count;
        let updated_pixel_count = view.updated_pixel_count;
        let mask_plan_stats = view.mask_plan_stats;
        Ok(AutoGazeVisualizationPanels {
            width,
            height,
            mask_rgba: std::mem::take(&mut buffers.mask_rgba),
            blend_rgba: std::mem::take(&mut buffers.blend_rgba),
            mask_pixel_count,
            updated_pixel_count,
            mask_plan_stats,
        })
    }

    pub fn visualize_rgba_panels_with_options_into<'a>(
        &mut self,
        rgba: &[u8],
        points: &[FixationPoint],
        options: AutoGazeRgbaVisualizationOptions,
        buffers: &'a mut AutoGazeRgbaVisualizationBuffers,
    ) -> Result<AutoGazeVisualizationPanelsView<'a>> {
        let width = options.width;
        let height = options.height;
        let _ = validate_rgba_dimensions(rgba, width, height)?;
        let (mask_pixel_count, updated_pixel_count, mask_plan_stats) = match self.mode {
            AutoGazeVisualizationMode::FullBlend => {
                self.last_keyframe = false;
                let mask = mask_rgba_and_rects_into(rgba, points, options, &mut buffers.mask_rgba)?;
                blend_masked_row_spans_rgba_into(
                    rgba,
                    width,
                    height,
                    &mask.row_spans,
                    options.blend_alpha,
                    &mut buffers.blend_rgba,
                )?;
                (mask.plan.pixel_count, mask.plan.pixel_count, mask.stats)
            }
            AutoGazeVisualizationMode::Interframe => {
                let mask = mask_rgba_and_rects_into(rgba, points, options, &mut buffers.mask_rgba)?;
                let mask_update = InterframeMaskUpdate {
                    plan: &mask.plan,
                    row_spans: &mask.row_spans,
                    mask_rgba: &buffers.mask_rgba,
                    stats: mask.stats,
                };
                let updated_pixel_count =
                    self.interframe_rgba_into(rgba, mask_update, options, &mut buffers.blend_rgba)?;
                (mask.plan.pixel_count, updated_pixel_count, mask.stats)
            }
        };
        self.frame_index = self.frame_index.saturating_add(1);
        Ok(AutoGazeVisualizationPanelsView {
            width,
            height,
            mask_rgba: &buffers.mask_rgba,
            blend_rgba: &buffers.blend_rgba,
            mask_pixel_count,
            updated_pixel_count,
            mask_plan_stats,
        })
    }

    fn interframe_rgba_into(
        &mut self,
        rgba: &[u8],
        mask_update: InterframeMaskUpdate<'_>,
        options: AutoGazeRgbaVisualizationOptions,
        output_rgba: &mut Vec<u8>,
    ) -> Result<usize> {
        let plan = mask_update.plan;
        let width = plan.width;
        let height = plan.height;
        let pixels = validate_rgba_dimensions(rgba, width, height)?;
        let dimensions_changed = self.interframe_width != width || self.interframe_height != height;
        let keyframe = dimensions_changed
            || self.interframe_output_rgba.len() != pixels * 4
            || self.frame_index == 0
            || (self.keyframe_duration > 0
                && self.frame_index.is_multiple_of(self.keyframe_duration));
        self.last_keyframe = keyframe;
        let updated_pixel_count = if keyframe { pixels } else { plan.pixel_count };

        if keyframe {
            self.interframe_output_rgba.clear();
            self.interframe_output_rgba.extend_from_slice(rgba);
            self.interframe_width = width;
            self.interframe_height = height;
        }

        if !keyframe {
            if mask_update.stats.pixel_count >= pixels
                && should_use_full_frame_update(
                    width,
                    height,
                    mask_update.stats,
                    options.full_frame_update_min_ratio,
                )
            {
                self.interframe_output_rgba.clear();
                self.interframe_output_rgba.extend_from_slice(rgba);
            } else if should_use_dense_mask_rgba_update(
                width,
                height,
                mask_update.stats,
                mask_update.row_spans.len(),
                options.mask_mode,
                options.full_frame_update_min_ratio,
            ) {
                copy_masked_pixels_rgba(
                    rgba,
                    mask_update.mask_rgba,
                    &mut self.interframe_output_rgba,
                )?;
            } else {
                copy_row_spans_rgba(
                    rgba,
                    width,
                    height,
                    mask_update.row_spans,
                    &mut self.interframe_output_rgba,
                );
            }
        }

        output_rgba.clear();
        output_rgba.extend_from_slice(&self.interframe_output_rgba);
        Ok(updated_pixel_count)
    }
}

pub struct AutoGazeTensorVisualization<B: Backend> {
    pub width: usize,
    pub height: usize,
    pub side_by_side_width: usize,
    pub side_by_side_rgba: Tensor<B, 3>,
    pub output_rgba: Tensor<B, 3>,
    pub mask_pixel_count: usize,
    pub updated_pixel_count: usize,
}

impl<B: Backend> AutoGazeTensorVisualization<B> {
    pub fn mask_ratio(&self) -> f64 {
        ratio(self.mask_pixel_count, self.width * self.height)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.updated_pixel_count, self.width * self.height)
    }
}

#[derive(Clone)]
pub struct AutoGazeTensorVisualizationPanels<B: Backend> {
    pub width: usize,
    pub height: usize,
    pub input_rgba: Tensor<B, 3>,
    pub mask_rgba: Tensor<B, 3>,
    pub output_rgba: Tensor<B, 3>,
    pub mask_pixel_count: usize,
    pub updated_pixel_count: usize,
}

impl<B: Backend> AutoGazeTensorVisualizationPanels<B> {
    pub fn mask_ratio(&self) -> f64 {
        ratio(self.mask_pixel_count, self.width * self.height)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.updated_pixel_count, self.width * self.height)
    }

    pub fn into_side_by_side(self) -> AutoGazeTensorVisualization<B> {
        let Self {
            width,
            height,
            input_rgba,
            mask_rgba,
            output_rgba,
            mask_pixel_count,
            updated_pixel_count,
        } = self;
        let side_by_side_rgba = Tensor::cat(vec![input_rgba, mask_rgba, output_rgba.clone()], 1);
        AutoGazeTensorVisualization {
            width,
            height,
            side_by_side_width: width * 3,
            side_by_side_rgba,
            output_rgba,
            mask_pixel_count,
            updated_pixel_count,
        }
    }
}

pub const DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS: usize = 4;
pub const DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO: f64 = 0.02;
pub const DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO: f64 = 0.45;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AutoGazeMaskPlanStats {
    pub rect_count: usize,
    pub row_span_count: usize,
    pub pixel_count: usize,
}

impl AutoGazeMaskPlanStats {
    pub fn update_ratio(self, width: usize, height: usize) -> f64 {
        ratio(self.pixel_count, width.max(1) * height.max(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazeTensorVisualizationOptions {
    pub width: usize,
    pub height: usize,
    pub cell_scale: f32,
    pub blend_alpha: f32,
    pub mask_mode: AutoGazeMaskVisualizationMode,
    pub mask_geometry_mode: AutoGazeMaskGeometryMode,
    pub sparse_update_max_rects: usize,
    pub sparse_update_max_ratio: f64,
    pub full_frame_update_min_ratio: f64,
}

impl AutoGazeTensorVisualizationOptions {
    pub const fn new(width: usize, height: usize, cell_scale: f32, blend_alpha: f32) -> Self {
        Self {
            width,
            height,
            cell_scale,
            blend_alpha,
            mask_mode: AutoGazeMaskVisualizationMode::ImageMaskOnly,
            mask_geometry_mode: AutoGazeMaskGeometryMode::Native,
            sparse_update_max_rects: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            sparse_update_max_ratio: DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
            full_frame_update_min_ratio: DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
        }
    }

    pub const fn with_mask_visualization_mode(
        mut self,
        mode: AutoGazeMaskVisualizationMode,
    ) -> Self {
        self.mask_mode = mode;
        self
    }

    pub const fn with_mask_geometry_mode(mut self, mode: AutoGazeMaskGeometryMode) -> Self {
        self.mask_geometry_mode = mode;
        self
    }

    pub const fn with_sparse_update_policy(
        mut self,
        max_rects: usize,
        max_update_ratio: f64,
    ) -> Self {
        self.sparse_update_max_rects = max_rects;
        self.sparse_update_max_ratio = max_update_ratio;
        self
    }

    pub const fn with_full_frame_update_policy(mut self, min_update_ratio: f64) -> Self {
        self.full_frame_update_min_ratio = min_update_ratio;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutoGazeTensorInterframePath {
    Keyframe,
    SparseRects,
    DenseMask,
    /// Full-frame masked dispatch. This scans the frame densely, but still
    /// preserves previous output wherever the AutoGaze mask is inactive.
    FullFrame,
}

impl AutoGazeTensorInterframePath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Keyframe => "keyframe",
            Self::SparseRects => "sparse-rects",
            Self::DenseMask => "dense-mask",
            Self::FullFrame => "full-frame",
        }
    }
}

#[derive(Clone)]
pub struct AutoGazeTensorVisualizationState<B: Backend> {
    mode: AutoGazeVisualizationMode,
    keyframe_duration: usize,
    frame_index: usize,
    width: usize,
    height: usize,
    interframe_output_rgba: Option<Tensor<B, 3>>,
    last_interframe_path: Option<AutoGazeTensorInterframePath>,
    last_mask_plan_stats: Option<AutoGazeMaskPlanStats>,
}

impl<B: Backend> AutoGazeTensorVisualizationState<B> {
    pub fn new(mode: AutoGazeVisualizationMode, keyframe_duration: usize) -> Self {
        Self {
            mode,
            keyframe_duration,
            frame_index: 0,
            width: 0,
            height: 0,
            interframe_output_rgba: None,
            last_interframe_path: None,
            last_mask_plan_stats: None,
        }
    }

    pub fn mode(&self) -> AutoGazeVisualizationMode {
        self.mode
    }

    pub fn keyframe_duration(&self) -> usize {
        self.keyframe_duration
    }

    pub fn last_interframe_path(&self) -> Option<AutoGazeTensorInterframePath> {
        self.last_interframe_path
    }

    pub fn last_mask_plan_stats(&self) -> Option<AutoGazeMaskPlanStats> {
        self.last_mask_plan_stats
    }

    pub fn configure(&mut self, mode: AutoGazeVisualizationMode, keyframe_duration: usize) {
        if self.mode != mode {
            self.reset();
        }
        self.mode = mode;
        self.keyframe_duration = keyframe_duration;
    }

    pub fn reset(&mut self) {
        self.frame_index = 0;
        self.width = 0;
        self.height = 0;
        self.interframe_output_rgba = None;
        self.last_interframe_path = None;
        self.last_mask_plan_stats = None;
    }

    pub fn visualize_normalized_rgb_clip(
        &mut self,
        tensor: Tensor<B, 5>,
        points: &[FixationPoint],
        options: AutoGazeTensorVisualizationOptions,
        device: &B::Device,
    ) -> Result<AutoGazeTensorVisualization<B>> {
        self.visualize_normalized_rgb_clip_panels(tensor, points, options, device)
            .map(AutoGazeTensorVisualizationPanels::into_side_by_side)
    }

    pub fn visualize_normalized_rgb_clip_panels(
        &mut self,
        tensor: Tensor<B, 5>,
        points: &[FixationPoint],
        options: AutoGazeTensorVisualizationOptions,
        device: &B::Device,
    ) -> Result<AutoGazeTensorVisualizationPanels<B>> {
        let width = options.width;
        let height = options.height;
        let pixels = validate_dimensions(width, height)?;
        let (plan, mask_stats) = fixation_sparse_update_plan_with_stats_for_geometry(
            width,
            height,
            points,
            options.cell_scale,
            options.mask_geometry_mode,
        )?;
        let input = normalized_rgb_clip_to_unit_rgba_tensor(tensor, width, height, device)?;
        let interframe_keyframe = matches!(self.mode, AutoGazeVisualizationMode::Interframe)
            && self.is_keyframe(width, height);
        let interframe_full_frame = matches!(self.mode, AutoGazeVisualizationMode::Interframe)
            && !interframe_keyframe
            && should_use_full_frame_update(
                width,
                height,
                mask_stats,
                options.full_frame_update_min_ratio,
            );
        let mask = if interframe_full_frame
            && mask_stats.pixel_count >= pixels
            && supports_full_frame_mask_fast_path(options.mask_mode)
        {
            full_frame_mask_panel_tensor(
                input.clone(),
                width,
                height,
                options.blend_alpha,
                options.mask_mode,
                device,
            )
        } else {
            let mut mask_rgba = Vec::new();
            fixation_mask_rgba_with_geometry_into(
                width,
                height,
                points,
                options.cell_scale,
                options.mask_mode,
                options.mask_geometry_mode,
                &mut mask_rgba,
            );
            mask_panel_tensor_from_rgba(
                input.clone(),
                &mask_rgba,
                width,
                height,
                options.blend_alpha,
                options.mask_mode,
                device,
            )?
        };
        let mut interframe_path = None;
        let (output, mask_pixel_count, updated_pixel_count) = match self.mode {
            AutoGazeVisualizationMode::FullBlend => {
                let alpha = alpha_mask_from_rects(width, height, &plan.rects);
                let mask_pixel_count = plan.pixel_count;
                let alpha = alpha_u8_to_unit_tensor(&alpha, width, height, device)?;
                let blend = alpha_blend_tensor(alpha, width, height, options.blend_alpha, device);
                let inverse = Tensor::<B, 3>::ones([height, width, 4], device).sub(blend.clone());
                (
                    input.clone().mul(inverse).add(blend),
                    mask_pixel_count,
                    mask_pixel_count,
                )
            }
            AutoGazeVisualizationMode::Interframe => {
                let keyframe = interframe_keyframe;
                let use_sparse = !keyframe
                    && !interframe_full_frame
                    && should_use_sparse_tensor_update_rects(
                        width,
                        height,
                        &plan.rects,
                        options.sparse_update_max_rects,
                        options.sparse_update_max_ratio,
                    );
                if interframe_full_frame {
                    interframe_path = Some(AutoGazeTensorInterframePath::FullFrame);
                    if mask_stats.pixel_count >= pixels {
                        (input.clone(), plan.pixel_count, pixels)
                    } else {
                        let previous = self
                            .interframe_output_rgba
                            .clone()
                            .unwrap_or_else(|| input.clone());
                        let alpha = alpha_mask_from_rects(width, height, &plan.rects);
                        let alpha = alpha_u8_to_unit_tensor(&alpha, width, height, device)?;
                        (
                            dense_interframe_update_tensor(input.clone(), previous, alpha),
                            plan.pixel_count,
                            plan.pixel_count,
                        )
                    }
                } else if use_sparse {
                    interframe_path = Some(AutoGazeTensorInterframePath::SparseRects);
                    let previous = self
                        .interframe_output_rgba
                        .clone()
                        .unwrap_or_else(|| input.clone());
                    let output = copy_sparse_update_tensor(input.clone(), previous, &plan)?;
                    (output, plan.pixel_count, plan.pixel_count)
                } else {
                    interframe_path = Some(if keyframe {
                        AutoGazeTensorInterframePath::Keyframe
                    } else {
                        AutoGazeTensorInterframePath::DenseMask
                    });
                    let alpha = alpha_mask_from_rects(width, height, &plan.rects);
                    let mask_pixel_count = plan.pixel_count;
                    let updated_pixel_count = if keyframe { pixels } else { mask_pixel_count };
                    let output = if keyframe {
                        input.clone()
                    } else {
                        let previous = self
                            .interframe_output_rgba
                            .clone()
                            .unwrap_or_else(|| input.clone());
                        let alpha = alpha_u8_to_unit_tensor(&alpha, width, height, device)?;
                        dense_interframe_update_tensor(input.clone(), previous, alpha)
                    };
                    (output, mask_pixel_count, updated_pixel_count)
                }
            }
        };

        self.advance(width, height, output.clone());
        self.last_interframe_path = interframe_path;
        self.last_mask_plan_stats = Some(mask_stats);
        Ok(AutoGazeTensorVisualizationPanels {
            width,
            height,
            input_rgba: input,
            mask_rgba: mask,
            output_rgba: output,
            mask_pixel_count,
            updated_pixel_count,
        })
    }

    pub fn visualize_normalized_rgb_clip_device_tokens_panels(
        &mut self,
        tensor: Tensor<B, 5>,
        tokens: &AutoGazeDeviceTokens<B>,
        config: &AutoGazeConfig,
        options: AutoGazeTensorVisualizationOptions,
        device: &B::Device,
    ) -> Result<AutoGazeTensorVisualizationPanels<B>> {
        let width = options.width;
        let height = options.height;
        let pixels = validate_dimensions(width, height)?;
        let input = normalized_rgb_clip_to_unit_rgba_tensor(tensor, width, height, device)?;
        let alpha = token_alpha_mask_tensor(tokens, config, width, height, device)?;
        let mask_pixel_count = pixels;
        let mask = token_mask_panel_tensor(
            input.clone(),
            alpha.clone(),
            width,
            height,
            options.blend_alpha,
            options.mask_mode,
            device,
        );
        let interframe_keyframe = matches!(self.mode, AutoGazeVisualizationMode::Interframe)
            && self.is_keyframe(width, height);
        let mut interframe_path = None;
        let output = match self.mode {
            AutoGazeVisualizationMode::FullBlend => {
                let blend = alpha_blend_tensor(alpha, width, height, options.blend_alpha, device);
                let inverse = Tensor::<B, 3>::ones([height, width, 4], device).sub(blend.clone());
                input.clone().mul(inverse).add(blend)
            }
            AutoGazeVisualizationMode::Interframe => {
                if interframe_keyframe {
                    interframe_path = Some(AutoGazeTensorInterframePath::Keyframe);
                    input.clone()
                } else {
                    interframe_path = Some(AutoGazeTensorInterframePath::FullFrame);
                    let previous = self
                        .interframe_output_rgba
                        .clone()
                        .unwrap_or_else(|| input.clone());
                    dense_interframe_update_tensor(input.clone(), previous, alpha)
                }
            }
        };

        self.advance(width, height, output.clone());
        self.last_interframe_path = interframe_path;
        self.last_mask_plan_stats = Some(AutoGazeMaskPlanStats {
            rect_count: tokens.tokens.shape().dims::<2>()[1],
            row_span_count: height,
            pixel_count: pixels,
        });
        Ok(AutoGazeTensorVisualizationPanels {
            width,
            height,
            input_rgba: input,
            mask_rgba: mask,
            output_rgba: output,
            mask_pixel_count,
            updated_pixel_count: pixels,
        })
    }

    fn is_keyframe(&self, width: usize, height: usize) -> bool {
        self.width != width
            || self.height != height
            || self.interframe_output_rgba.is_none()
            || self.frame_index == 0
            || (self.keyframe_duration > 0
                && self.frame_index.is_multiple_of(self.keyframe_duration))
    }

    fn advance(&mut self, width: usize, height: usize, output: Tensor<B, 3>) {
        self.width = width;
        self.height = height;
        self.interframe_output_rgba = Some(output);
        self.frame_index = self.frame_index.saturating_add(1);
    }
}

pub fn copy_sparse_update_tensor<B: Backend>(
    source_rgba: Tensor<B, 3>,
    target_rgba: Tensor<B, 3>,
    plan: &AutoGazeSparseUpdatePlan,
) -> Result<Tensor<B, 3>> {
    validate_unit_rgba_tensor_shape(&source_rgba, plan.width, plan.height)?;
    validate_unit_rgba_tensor_shape(&target_rgba, plan.width, plan.height)?;

    let mut output = target_rgba;
    for rect in &plan.rects {
        let rect = rect.clamped(plan.width, plan.height);
        if rect.is_empty() {
            continue;
        }
        let y = rect.y0..rect.y1;
        let x = rect.x0..rect.x1;
        let channels = 0..4;
        let source_patch = source_rgba
            .clone()
            .slice([y.clone(), x.clone(), channels.clone()]);
        output = output.slice_assign([y, x, channels], source_patch);
    }

    Ok(output)
}

fn validate_unit_rgba_tensor_shape<B: Backend>(
    tensor: &Tensor<B, 3>,
    width: usize,
    height: usize,
) -> Result<()> {
    let dims = tensor.shape().dims::<3>();
    ensure!(
        dims == [height, width, 4],
        "expected unit RGBA tensor shape [{height},{width},4], got {dims:?}"
    );
    Ok(())
}

fn dense_interframe_update_tensor<B: Backend>(
    input: Tensor<B, 3>,
    previous: Tensor<B, 3>,
    alpha: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let update = alpha.repeat_dim(2, 4);
    previous
        .clone()
        .add(input.clone().sub(previous).mul(update))
}

fn should_use_sparse_tensor_update_rects(
    width: usize,
    height: usize,
    rects: &[FixationPixelRect],
    max_rects: usize,
    max_update_ratio: f64,
) -> bool {
    if rects.is_empty()
        || max_rects == 0
        || rects.len() > max_rects
        || !max_update_ratio.is_finite()
        || max_update_ratio <= 0.0
    {
        return false;
    }

    let pixels = width.max(1) * height.max(1);
    let pixel_count_upper_bound = rects
        .iter()
        .map(|rect| rect.pixel_count(width, height))
        .sum::<usize>();
    ratio(pixel_count_upper_bound, pixels) <= max_update_ratio.clamp(0.0, 1.0)
}

fn should_use_full_frame_update(
    width: usize,
    height: usize,
    stats: AutoGazeMaskPlanStats,
    min_update_ratio: f64,
) -> bool {
    if stats.pixel_count == 0 || !min_update_ratio.is_finite() || min_update_ratio <= 0.0 {
        return false;
    }

    stats.update_ratio(width, height) >= min_update_ratio.clamp(0.0, 1.0)
}

pub fn normalized_rgb_clip_to_unit_rgba_tensor<B: Backend>(
    tensor: Tensor<B, 5>,
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let _ = validate_dimensions(width, height)?;
    let dims = tensor.shape().dims::<5>();
    ensure!(
        dims == [1, 1, 3, height, width],
        "expected normalized RGB clip tensor shape [1,1,3,{height},{width}], got {dims:?}"
    );
    let rgb = tensor
        .reshape([3, height, width])
        .permute([1, 2, 0])
        .mul(channel_vector_tensor(AUTO_GAZE_IMAGE_STD, device))
        .add(channel_vector_tensor(AUTO_GAZE_IMAGE_MEAN, device))
        .add_scalar(1.0)
        .div_scalar(2.0)
        .clamp(0.0, 1.0);
    let alpha = Tensor::<B, 3>::ones([height, width, 1], device);
    Ok(Tensor::cat(vec![rgb, alpha], 2))
}

pub fn fixation_alpha_mask(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let rects = fixation_cell_rects(width, height, points, cell_scale);
    alpha_mask_from_rects(width, height, &rects)
}

fn alpha_mask_from_rects(width: usize, height: usize, rects: &[FixationPixelRect]) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let mut alpha = vec![0u8; width * height];

    for (y, x0, x1) in merged_rect_row_spans(width, height, rects) {
        let start = y * width + x0;
        let end = y * width + x1;
        alpha[start..end].fill(255);
    }

    alpha
}

/// Build a projected sparse footprint from multi-scale gaze tokens.
///
/// AutoGaze predicts tokens from a multi-scale image pyramid. A coarse 2x2
/// token is drawn in the default visualization at its native pyramid scale so
/// the mask panel and interframe output agree. Downstream codec experiments can
/// use this helper to project selected tokens onto the finest active display
/// grid when they want selected token cells rather than native pyramid cells.
pub fn fixation_effective_alpha_mask(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let rects = compact_pixel_rects(
        width,
        height,
        fixation_effective_cell_rects(width, height, points, cell_scale),
    );
    alpha_mask_from_rects(width, height, &rects)
}

pub fn fixation_scale_mask_rgba(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let mut rgba = Vec::new();
    fixation_scale_mask_rgba_into(width, height, points, cell_scale, &mut rgba);
    rgba
}

pub fn fixation_scale_mask_rgba_into(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    rgba: &mut Vec<u8>,
) {
    let width = width.max(1);
    let height = height.max(1);
    reset_transparent_black_rgba(rgba, width, height);
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let mut ordered = points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .cell_width()
            .total_cmp(&left.cell_width())
            .then_with(|| right.cell_height().total_cmp(&left.cell_height()))
    });

    for point in ordered {
        let color = scale_color_for_point(point);
        let bounds = point.scaled_bounds(cell_scale);
        let (x0, x1) = pixel_range(bounds.x_min, bounds.x_max, width);
        let (y0, y1) = pixel_range(bounds.y_min, bounds.y_max, height);
        let rect = FixationPixelRect { x0, x1, y0, y1 };
        fill_cell(rgba, width, rect, color, 0.42);
        stroke_cell(rgba, width, rect, color);
    }
}

pub fn fixation_scale_rows_mask_rgba(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let mut rgba = Vec::new();
    fixation_scale_rows_mask_rgba_into(width, height, points, cell_scale, &mut rgba);
    rgba
}

pub fn fixation_scale_rows_mask_rgba_into(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    rgba: &mut Vec<u8>,
) {
    let width = width.max(1);
    let height = height.max(1);
    reset_transparent_black_rgba(rgba, width, height);
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let mut by_grid = BTreeMap::<usize, Vec<FixationPoint>>::new();
    for point in points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
    {
        by_grid
            .entry(scale_grid_for_point(point))
            .or_default()
            .push(point);
    }
    let row_grids = scale_row_grids(&by_grid);
    let rows = row_grids.len().max(1);

    for (row_idx, grid) in row_grids.into_iter().enumerate() {
        let Some(mut row_points) = by_grid.remove(&grid) else {
            continue;
        };
        let (row_y0, row_y1) = partition_range(row_idx, rows, height);
        let row_height = row_y1.saturating_sub(row_y0);
        if row_height == 0 {
            continue;
        }
        let viewport = aspect_preserving_row_viewport(width, height, row_y0, row_y1);
        row_points.sort_by(|left, right| {
            right
                .cell_width()
                .total_cmp(&left.cell_width())
                .then_with(|| right.cell_height().total_cmp(&left.cell_height()))
        });

        for point in row_points {
            let color = scale_color_for_point(point);
            let bounds = point.scaled_bounds(cell_scale);
            let (local_x0, local_x1) = pixel_range(bounds.x_min, bounds.x_max, viewport.width);
            let (local_y0, local_y1) = pixel_range(bounds.y_min, bounds.y_max, viewport.height);
            let rect = FixationPixelRect {
                x0: viewport.x0 + local_x0,
                x1: viewport.x0 + local_x1,
                y0: viewport.y0 + local_y0,
                y1: viewport.y0 + local_y1,
            };
            fill_cell(rgba, width, rect, color, 0.42);
            stroke_cell(rgba, width, rect, color);
        }
    }
}

pub fn fixation_mask_rgba(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    mode: AutoGazeMaskVisualizationMode,
) -> Vec<u8> {
    let mut rgba = Vec::new();
    fixation_mask_rgba_into(width, height, points, cell_scale, mode, &mut rgba);
    rgba
}

pub fn fixation_mask_rgba_into(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    mode: AutoGazeMaskVisualizationMode,
    rgba: &mut Vec<u8>,
) {
    match mode {
        AutoGazeMaskVisualizationMode::Overlay
        | AutoGazeMaskVisualizationMode::ImageOverlay
        | AutoGazeMaskVisualizationMode::ImageMaskOnly => {
            fixation_scale_mask_rgba_into(width, height, points, cell_scale, rgba);
        }
        AutoGazeMaskVisualizationMode::ScaleRows => {
            fixation_scale_rows_mask_rgba_into(width, height, points, cell_scale, rgba);
        }
    }
}

fn fixation_mask_rgba_with_geometry_into(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    mode: AutoGazeMaskVisualizationMode,
    geometry_mode: AutoGazeMaskGeometryMode,
    rgba: &mut Vec<u8>,
) {
    match (mode, geometry_mode) {
        (AutoGazeMaskVisualizationMode::ScaleRows, AutoGazeMaskGeometryMode::Native) => {
            fixation_scale_rows_mask_rgba_into(width, height, points, cell_scale, rgba);
        }
        (AutoGazeMaskVisualizationMode::ScaleRows, AutoGazeMaskGeometryMode::Deduplicated) => {
            let points = fixation_deduplicated_points(width, height, points, cell_scale);
            fixation_scale_rows_mask_rgba_into(width, height, &points, cell_scale, rgba);
        }
        (_, AutoGazeMaskGeometryMode::Native) => {
            fixation_scale_mask_rgba_into(width, height, points, cell_scale, rgba);
        }
        (_, AutoGazeMaskGeometryMode::Deduplicated) => {
            let points = fixation_deduplicated_points(width, height, points, cell_scale);
            fixation_scale_mask_rgba_into(width, height, &points, cell_scale, rgba);
        }
        (_, AutoGazeMaskGeometryMode::Effective) => {
            fixation_effective_scale_mask_rgba_into(width, height, points, cell_scale, rgba);
        }
    }
}

pub fn fixation_image_overlay_mask_rgba(
    source_rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<Vec<u8>> {
    let mut rgba = Vec::new();
    fixation_image_overlay_mask_rgba_into(
        source_rgba,
        width,
        height,
        points,
        cell_scale,
        blend_alpha,
        &mut rgba,
    )?;
    Ok(rgba)
}

pub fn fixation_image_overlay_mask_rgba_into(
    source_rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
    rgba: &mut Vec<u8>,
) -> Result<()> {
    let _ = validate_rgba_dimensions(source_rgba, width, height)?;
    fixation_scale_mask_rgba_into(width, height, points, cell_scale, rgba);
    alpha_blend_colored_mask_with_source_into(source_rgba, blend_alpha, true, rgba)
}

pub fn fixation_image_mask_only_rgba(
    source_rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<Vec<u8>> {
    let mut rgba = Vec::new();
    fixation_image_mask_only_rgba_into(
        source_rgba,
        width,
        height,
        points,
        cell_scale,
        blend_alpha,
        &mut rgba,
    )?;
    Ok(rgba)
}

pub fn fixation_image_mask_only_rgba_into(
    source_rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
    rgba: &mut Vec<u8>,
) -> Result<()> {
    let _ = validate_rgba_dimensions(source_rgba, width, height)?;
    fixation_scale_mask_rgba_into(width, height, points, cell_scale, rgba);
    alpha_blend_colored_mask_with_source_into(source_rgba, blend_alpha, false, rgba)
}

/// Colorize the same projected cells used by [`fixation_effective_alpha_mask`].
///
/// This is useful when a downstream codec wants a sparse finest-grid token
/// footprint. The default visualization uses [`fixation_scale_mask_rgba`] so
/// the displayed cells retain their native pyramid scale, matching upstream.
pub fn fixation_effective_scale_mask_rgba(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let mut rgba = Vec::new();
    fixation_effective_scale_mask_rgba_into(width, height, points, cell_scale, &mut rgba);
    rgba
}

pub fn fixation_effective_scale_mask_rgba_into(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    rgba: &mut Vec<u8>,
) {
    let width = width.max(1);
    let height = height.max(1);
    reset_transparent_black_rgba(rgba, width, height);
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let mut ordered = points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .cell_width()
            .total_cmp(&left.cell_width())
            .then_with(|| right.cell_height().total_cmp(&left.cell_height()))
    });

    let target_grid = effective_display_grid(&ordered);
    let mut unique_cells = BTreeMap::<FixationPixelRect, [u8; 3]>::new();
    for point in ordered {
        let color = scale_color_for_point(point);
        let rect = effective_point_pixel_rect(width, height, point, target_grid, cell_scale);
        unique_cells.insert(rect, color);
    }

    for (rect, color) in unique_cells {
        fill_cell(rgba, width, rect, color, 0.42);
        stroke_cell(rgba, width, rect, color);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaskViewport {
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
}

fn aspect_preserving_row_viewport(
    canvas_width: usize,
    canvas_height: usize,
    row_y0: usize,
    row_y1: usize,
) -> MaskViewport {
    let canvas_width = canvas_width.max(1);
    let canvas_height = canvas_height.max(1);
    let row_y0 = row_y0.min(canvas_height.saturating_sub(1));
    let row_y1 = row_y1.min(canvas_height).max(row_y0 + 1);
    let row_height = row_y1 - row_y0;
    let viewport_width = ((row_height as f64 * canvas_width as f64 / canvas_height as f64).round()
        as usize)
        .clamp(1, canvas_width);
    let x0 = (canvas_width - viewport_width) / 2;
    MaskViewport {
        x0,
        y0: row_y0,
        width: viewport_width,
        height: row_height,
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FixationPixelRect {
    pub x0: usize,
    pub x1: usize,
    pub y0: usize,
    pub y1: usize,
}

impl FixationPixelRect {
    pub const fn new(x0: usize, x1: usize, y0: usize, y1: usize) -> Self {
        Self { x0, x1, y0, y1 }
    }

    pub const fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    pub fn clamped(&self, width: usize, height: usize) -> Self {
        Self {
            x0: self.x0.min(width),
            x1: self.x1.min(width),
            y0: self.y0.min(height),
            y1: self.y1.min(height),
        }
    }

    pub fn pixel_count(&self, width: usize, height: usize) -> usize {
        let rect = self.clamped(width, height);
        rect.x1.saturating_sub(rect.x0) * rect.y1.saturating_sub(rect.y0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoGazeSparseUpdatePlan {
    pub width: usize,
    pub height: usize,
    pub rects: Vec<FixationPixelRect>,
    pub pixel_count: usize,
}

impl AutoGazeSparseUpdatePlan {
    pub fn new(width: usize, height: usize, rects: Vec<FixationPixelRect>) -> Result<Self> {
        sparse_update_plan_from_rects(width, height, rects).map(|(plan, _)| plan)
    }

    pub fn update_ratio(&self) -> f64 {
        ratio(self.pixel_count, self.width * self.height)
    }

    pub fn stats(&self) -> AutoGazeMaskPlanStats {
        let row_span_count = merged_rect_row_spans(self.width, self.height, &self.rects).len();
        AutoGazeMaskPlanStats {
            rect_count: self.rects.len(),
            row_span_count,
            pixel_count: self.pixel_count,
        }
    }
}

struct AutoGazeSparseUpdatePlanParts {
    plan: AutoGazeSparseUpdatePlan,
    stats: AutoGazeMaskPlanStats,
    row_spans: Vec<(usize, usize, usize)>,
}

fn sparse_update_plan_from_rects(
    width: usize,
    height: usize,
    rects: Vec<FixationPixelRect>,
) -> Result<(AutoGazeSparseUpdatePlan, AutoGazeMaskPlanStats)> {
    let parts = sparse_update_plan_parts_from_rects(width, height, rects)?;
    Ok((parts.plan, parts.stats))
}

fn sparse_update_plan_parts_from_rects(
    width: usize,
    height: usize,
    rects: Vec<FixationPixelRect>,
) -> Result<AutoGazeSparseUpdatePlanParts> {
    let _ = validate_dimensions(width, height)?;
    let rects = compact_pixel_rects(width, height, rects);
    let row_spans = row_spans_from_compacted_rects(&rects);
    let pixel_count = row_spans
        .iter()
        .map(|(_, x0, x1)| x1.saturating_sub(*x0))
        .sum();
    let stats = AutoGazeMaskPlanStats {
        rect_count: rects.len(),
        row_span_count: row_spans.len(),
        pixel_count,
    };
    Ok(AutoGazeSparseUpdatePlanParts {
        plan: AutoGazeSparseUpdatePlan {
            width,
            height,
            rects,
            pixel_count,
        },
        stats,
        row_spans,
    })
}

/// Build the native-scale sparse update rectangles used by the default AutoGaze visualization.
pub fn fixation_sparse_update_plan(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Result<AutoGazeSparseUpdatePlan> {
    fixation_sparse_update_plan_with_stats(width, height, points, cell_scale).map(|(plan, _)| plan)
}

fn fixation_sparse_update_plan_with_stats(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Result<(AutoGazeSparseUpdatePlan, AutoGazeMaskPlanStats)> {
    sparse_update_plan_from_rects(
        width,
        height,
        fixation_cell_rects(width, height, points, cell_scale),
    )
}

fn fixation_sparse_update_plan_with_stats_for_geometry(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    geometry_mode: AutoGazeMaskGeometryMode,
) -> Result<(AutoGazeSparseUpdatePlan, AutoGazeMaskPlanStats)> {
    match geometry_mode {
        AutoGazeMaskGeometryMode::Native => {
            fixation_sparse_update_plan_with_stats(width, height, points, cell_scale)
        }
        AutoGazeMaskGeometryMode::Deduplicated => sparse_update_plan_from_rects(
            width,
            height,
            fixation_deduplicated_cell_rects(width, height, points, cell_scale),
        ),
        AutoGazeMaskGeometryMode::Effective => sparse_update_plan_from_rects(
            width,
            height,
            fixation_effective_cell_rects(width, height, points, cell_scale),
        ),
    }
}

fn fixation_sparse_update_plan_parts_for_geometry(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    geometry_mode: AutoGazeMaskGeometryMode,
) -> Result<AutoGazeSparseUpdatePlanParts> {
    match geometry_mode {
        AutoGazeMaskGeometryMode::Native => sparse_update_plan_parts_from_rects(
            width,
            height,
            fixation_cell_rects(width, height, points, cell_scale),
        ),
        AutoGazeMaskGeometryMode::Deduplicated => sparse_update_plan_parts_from_rects(
            width,
            height,
            fixation_deduplicated_cell_rects(width, height, points, cell_scale),
        ),
        AutoGazeMaskGeometryMode::Effective => sparse_update_plan_parts_from_rects(
            width,
            height,
            fixation_effective_cell_rects(width, height, points, cell_scale),
        ),
    }
}

/// Build native-scale sparse update rectangles after dropping fully covered cells.
///
/// The resulting union matches [`fixation_sparse_update_plan`], but redundant
/// cells from finer scales that are already covered by larger selected cells are
/// removed. This is useful for realtime viewers that need native-scale output
/// coverage without repeatedly drawing or copying overlapping mask cells.
pub fn fixation_deduplicated_sparse_update_plan(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Result<AutoGazeSparseUpdatePlan> {
    sparse_update_plan_from_rects(
        width,
        height,
        fixation_deduplicated_cell_rects(width, height, points, cell_scale),
    )
    .map(|(plan, _)| plan)
}

/// Build sparse update rectangles projected onto the finest active gaze grid.
pub fn fixation_effective_sparse_update_plan(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Result<AutoGazeSparseUpdatePlan> {
    let rects = compact_pixel_rects(
        width.max(1),
        height.max(1),
        fixation_effective_cell_rects(width, height, points, cell_scale),
    );
    AutoGazeSparseUpdatePlan::new(width, height, rects)
}

/// Copy source RGBA pixels for sparse update rectangles into a persistent output frame.
pub fn copy_sparse_update_rgba(
    source_rgba: &[u8],
    target_rgba: &mut [u8],
    plan: &AutoGazeSparseUpdatePlan,
) -> Result<usize> {
    let _ = validate_rgba_dimensions(source_rgba, plan.width, plan.height)?;
    ensure!(
        target_rgba.len() == source_rgba.len(),
        "target RGBA byte length must match source frame"
    );
    let row_spans = merged_rect_row_spans(plan.width, plan.height, &plan.rects);
    copy_row_spans_rgba(
        source_rgba,
        plan.width,
        plan.height,
        &row_spans,
        target_rgba,
    );
    Ok(plan.pixel_count)
}

fn copy_row_spans_rgba(
    source_rgba: &[u8],
    width: usize,
    height: usize,
    row_spans: &[(usize, usize, usize)],
    target_rgba: &mut [u8],
) {
    debug_assert_eq!(source_rgba.len(), width * height * 4);
    debug_assert_eq!(target_rgba.len(), source_rgba.len());
    let row_bytes = width * 4;
    for &(y, x0, x1) in row_spans {
        if y >= height || x0 >= x1 {
            continue;
        }
        let x0 = x0.min(width);
        let x1 = x1.min(width);
        if x0 >= x1 {
            continue;
        }
        let row = y * row_bytes;
        let start = row + x0 * 4;
        let end = row + x1 * 4;
        target_rgba[start..end].copy_from_slice(&source_rgba[start..end]);
    }
}

pub fn fixation_cell_rects(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<FixationPixelRect> {
    let width = width.max(1);
    let height = height.max(1);
    points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .map(|point| point_pixel_rect(width, height, point, cell_scale))
        .collect()
}

pub fn fixation_deduplicated_cell_rects(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<FixationPixelRect> {
    let width = width.max(1);
    let height = height.max(1);
    fixation_deduplicated_points(width, height, points, cell_scale)
        .into_iter()
        .map(|point| point_pixel_rect(width, height, point, cell_scale))
        .collect()
}

pub fn fixation_effective_cell_rects(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<FixationPixelRect> {
    let width = width.max(1);
    let height = height.max(1);
    let target_grid = effective_display_grid(points);
    points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .map(|point| effective_point_pixel_rect(width, height, point, target_grid, cell_scale))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EffectiveDisplayGrid {
    cols: usize,
    rows: usize,
}

impl EffectiveDisplayGrid {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols: cols.max(1),
            rows: rows.max(1),
        }
    }
}

fn effective_display_grid(points: &[FixationPoint]) -> EffectiveDisplayGrid {
    let (cols, rows) = points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .map(|point| {
            let cols = (1.0 / point.cell_width().max(1.0e-6)).round();
            let rows = (1.0 / point.cell_height().max(1.0e-6)).round();
            (
                finite_grid_extent(cols).unwrap_or_else(|| nearest_scale_grid(point)),
                finite_grid_extent(rows).unwrap_or_else(|| nearest_scale_grid(point)),
            )
        })
        .fold((14usize, 14usize), |(max_cols, max_rows), (cols, rows)| {
            (max_cols.max(cols), max_rows.max(rows))
        });
    EffectiveDisplayGrid::new(cols, rows)
}

fn finite_grid_extent(value: f32) -> Option<usize> {
    value.is_finite().then_some(value.max(1.0) as usize)
}

fn effective_point_pixel_rect(
    width: usize,
    height: usize,
    point: FixationPoint,
    target_grid: EffectiveDisplayGrid,
    cell_scale: f32,
) -> FixationPixelRect {
    let (row, col) = project_point_to_grid_cell(point, target_grid);
    if (cell_scale - 1.0).abs() <= f32::EPSILON {
        let (x0, x1) = grid_cell_pixel_range(col, target_grid.cols, width);
        let (y0, y1) = grid_cell_pixel_range(row, target_grid.rows, height);
        return FixationPixelRect { x0, x1, y0, y1 };
    }

    let cols = target_grid.cols.max(1) as f64;
    let rows = target_grid.rows.max(1) as f64;
    let extent_x = (cell_scale.max(1.0e-6) as f64 / cols).clamp(1.0e-6, 1.0);
    let extent_y = (cell_scale.max(1.0e-6) as f64 / rows).clamp(1.0e-6, 1.0);
    let center_x = (col as f64 + 0.5) / cols;
    let center_y = (row as f64 + 0.5) / rows;
    let half_x = extent_x * 0.5;
    let half_y = extent_y * 0.5;
    let (x0, x1) = pixel_range_f64(
        (center_x - half_x).clamp(0.0, 1.0),
        (center_x + half_x).clamp(0.0, 1.0),
        width,
    );
    let (y0, y1) = pixel_range_f64(
        (center_y - half_y).clamp(0.0, 1.0),
        (center_y + half_y).clamp(0.0, 1.0),
        height,
    );
    FixationPixelRect { x0, x1, y0, y1 }
}

fn point_pixel_rect(
    width: usize,
    height: usize,
    point: FixationPoint,
    cell_scale: f32,
) -> FixationPixelRect {
    let bounds = point.scaled_bounds(cell_scale);
    let (x0, x1) = pixel_range(bounds.x_min, bounds.x_max, width);
    let (y0, y1) = pixel_range(bounds.y_min, bounds.y_max, height);
    FixationPixelRect { x0, x1, y0, y1 }
}

fn fixation_deduplicated_points(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<FixationPoint> {
    let width = width.max(1);
    let height = height.max(1);
    let mut ordered = points
        .iter()
        .copied()
        .filter(|point| point.confidence > 0.0)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .cell_width()
            .total_cmp(&left.cell_width())
            .then_with(|| right.cell_height().total_cmp(&left.cell_height()))
            .then_with(|| scale_grid_for_point(*left).cmp(&scale_grid_for_point(*right)))
    });

    let mut covered = vec![Vec::<(usize, usize)>::new(); height];
    let mut deduplicated = Vec::with_capacity(ordered.len());
    for point in ordered {
        let rect = point_pixel_rect(width, height, point, cell_scale).clamped(width, height);
        if rect.is_empty() || rect_fully_covered(&covered, rect) {
            continue;
        }
        insert_covered_rect(&mut covered, rect);
        deduplicated.push(point);
    }
    deduplicated
}

fn rect_fully_covered(covered: &[Vec<(usize, usize)>], rect: FixationPixelRect) -> bool {
    for row in covered.iter().take(rect.y1).skip(rect.y0) {
        let mut cursor = rect.x0;
        for &(x0, x1) in row {
            if x1 <= cursor {
                continue;
            }
            if x0 > cursor {
                return false;
            }
            cursor = cursor.max(x1);
            if cursor >= rect.x1 {
                break;
            }
        }
        if cursor < rect.x1 {
            return false;
        }
    }
    true
}

fn insert_covered_rect(covered: &mut [Vec<(usize, usize)>], rect: FixationPixelRect) {
    for row in &mut covered[rect.y0..rect.y1] {
        insert_covered_span(row, rect.x0, rect.x1);
    }
}

fn insert_covered_span(row: &mut Vec<(usize, usize)>, x0: usize, x1: usize) {
    if x0 >= x1 {
        return;
    }
    row.push((x0, x1));
    row.sort_unstable_by_key(|span| span.0);

    let mut write = 0usize;
    for read in 0..row.len() {
        let (start, end) = row[read];
        if write == 0 || start > row[write - 1].1 {
            row[write] = (start, end);
            write += 1;
        } else {
            row[write - 1].1 = row[write - 1].1.max(end);
        }
    }
    row.truncate(write);
}

fn compact_pixel_rects<I>(width: usize, height: usize, rects: I) -> Vec<FixationPixelRect>
where
    I: IntoIterator<Item = FixationPixelRect>,
{
    let mut rects = rects
        .into_iter()
        .map(|rect| rect.clamped(width, height))
        .filter(|rect| !rect.is_empty())
        .collect::<Vec<_>>();
    rects.sort_unstable_by_key(|rect| (rect.y0, rect.y1, rect.x0, rect.x1));

    let mut compacted: Vec<FixationPixelRect> = Vec::with_capacity(rects.len());
    for rect in rects {
        if let Some(last) = compacted.last_mut()
            && last.y0 == rect.y0
            && last.y1 == rect.y1
            && rect.x0 <= last.x1
        {
            last.x1 = last.x1.max(rect.x1);
            continue;
        }
        compacted.push(rect);
    }
    compacted
}

fn merged_rect_row_spans(
    width: usize,
    height: usize,
    rects: &[FixationPixelRect],
) -> Vec<(usize, usize, usize)> {
    if rects.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    let rects = compact_pixel_rects(width, height, rects.iter().copied());
    row_spans_from_compacted_rects(&rects)
}

fn row_spans_from_compacted_rects(rects: &[FixationPixelRect]) -> Vec<(usize, usize, usize)> {
    if rects.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for rect in rects.iter().copied() {
        for y in rect.y0..rect.y1 {
            spans.push((y, rect.x0, rect.x1));
        }
    }
    if spans.len() <= 1 {
        return spans;
    }

    spans.sort_unstable();
    let mut merged = Vec::with_capacity(spans.len());
    let mut current_y = spans[0].0;
    let mut current_x0 = spans[0].1;
    let mut current_x1 = spans[0].2;
    for (y, x0, x1) in spans.into_iter().skip(1) {
        if y == current_y && x0 <= current_x1 {
            current_x1 = current_x1.max(x1);
        } else {
            merged.push((current_y, current_x0, current_x1));
            current_y = y;
            current_x0 = x0;
            current_x1 = x1;
        }
    }
    merged.push((current_y, current_x0, current_x1));
    merged
}

fn project_point_to_grid_cell(
    point: FixationPoint,
    target_grid: EffectiveDisplayGrid,
) -> (usize, usize) {
    let col = (point.x.clamp(0.0, 1.0 - f32::EPSILON) * target_grid.cols as f32).floor() as usize;
    let row = (point.y.clamp(0.0, 1.0 - f32::EPSILON) * target_grid.rows as f32).floor() as usize;
    (row.min(target_grid.rows - 1), col.min(target_grid.cols - 1))
}

fn grid_cell_pixel_range(index: usize, grid: usize, extent: usize) -> (usize, usize) {
    let grid = grid.max(1);
    let extent = extent.max(1);
    let index = index.min(grid - 1);
    let start = index.saturating_mul(extent) / grid;
    let end = (index + 1).saturating_mul(extent).saturating_add(grid - 1) / grid;
    (
        start.min(extent.saturating_sub(1)),
        end.min(extent).max(start + 1),
    )
}

fn fill_cell(rgba: &mut [u8], width: usize, rect: FixationPixelRect, color: [u8; 3], opacity: f32) {
    let opacity = opacity.clamp(0.0, 1.0);
    if rect.is_empty() {
        return;
    }

    let pixel = [
        (color[0] as f32 * opacity).round() as u8,
        (color[1] as f32 * opacity).round() as u8,
        (color[2] as f32 * opacity).round() as u8,
        255,
    ];
    let span = rect.x1 - rect.x0;

    for y in rect.y0..rect.y1 {
        let start = (y * width + rect.x0) * 4;
        let end = start + span * 4;
        fill_rgba_span(&mut rgba[start..end], pixel);
    }
}

fn fill_rgba_span(span: &mut [u8], pixel: [u8; 4]) {
    if span.len() < 4 {
        return;
    }

    span[..4].copy_from_slice(&pixel);
    let mut filled = 4;
    while filled < span.len() {
        let copy_len = filled.min(span.len() - filled);
        let (source, target) = span.split_at_mut(filled);
        target[..copy_len].copy_from_slice(&source[..copy_len]);
        filled += copy_len;
    }
}

fn stroke_cell(rgba: &mut [u8], width: usize, rect: FixationPixelRect, color: [u8; 3]) {
    if rect.x0 >= rect.x1 || rect.y0 >= rect.y1 {
        return;
    }

    for x in rect.x0..rect.x1 {
        write_mask_pixel(rgba, width, x, rect.y0, color);
        write_mask_pixel(rgba, width, x, rect.y1 - 1, color);
    }
    for y in rect.y0..rect.y1 {
        write_mask_pixel(rgba, width, rect.x0, y, color);
        write_mask_pixel(rgba, width, rect.x1 - 1, y, color);
    }
}

fn write_mask_pixel(rgba: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 3]) {
    let offset = (y * width + x) * 4;
    if offset + 3 <= rgba.len() {
        rgba[offset..offset + 3].copy_from_slice(&color);
    }
}

pub fn visualize_fixations_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<AutoGazeVisualization> {
    let mask_blend = mask_and_blend_rgba(rgba, width, height, points, cell_scale, blend_alpha)?;
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    build_visualization(
        rgba,
        AutoGazeVisualizationPanels {
            width,
            height,
            mask_rgba: mask_blend.mask_rgba,
            blend_rgba: mask_blend.blend_rgba,
            mask_pixel_count: mask_blend.mask_pixel_count,
            updated_pixel_count: mask_blend.mask_pixel_count,
            mask_plan_stats: mask_blend.mask_plan_stats,
        },
    )
}

pub fn rgba_psnr_db(reference_rgba: &[u8], candidate_rgba: &[u8]) -> Result<f64> {
    ensure!(
        reference_rgba.len() == candidate_rgba.len(),
        "PSNR inputs must have the same byte length"
    );
    ensure!(
        reference_rgba.len().is_multiple_of(4),
        "PSNR inputs must be RGBA buffers"
    );
    ensure!(!reference_rgba.is_empty(), "PSNR inputs must be nonempty");

    let mut squared_error = 0.0f64;
    let mut samples = 0usize;
    for (reference, candidate) in reference_rgba
        .chunks_exact(4)
        .zip(candidate_rgba.chunks_exact(4))
    {
        for channel in 0..3 {
            let diff = reference[channel] as f64 - candidate[channel] as f64;
            squared_error += diff * diff;
            samples += 1;
        }
    }

    if squared_error == 0.0 {
        return Ok(f64::INFINITY);
    }

    let mse = squared_error / samples.max(1) as f64;
    Ok(10.0 * ((255.0 * 255.0) / mse).log10())
}

fn validate_rgba_dimensions(rgba: &[u8], width: usize, height: usize) -> Result<usize> {
    let pixels = validate_dimensions(width, height)?;
    let expected_len = visualization_rgba_len(width, height)?;
    ensure!(
        rgba.len() == expected_len,
        "expected {expected_len} RGBA bytes for {width}x{height}, got {}",
        rgba.len()
    );
    Ok(pixels)
}

fn validate_dimensions(width: usize, height: usize) -> Result<usize> {
    ensure!(
        width > 0 && height > 0,
        "visualization dimensions must be nonzero"
    );
    width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("visualization dimensions overflow"))
}

fn visualization_rgba_len(width: usize, height: usize) -> Result<usize> {
    validate_dimensions(width, height)?
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("visualization byte length overflow"))
}

fn reset_transparent_black_rgba(rgba: &mut Vec<u8>, width: usize, height: usize) {
    let len = width.saturating_mul(height).saturating_mul(4);
    rgba.clear();
    rgba.resize(len, 0);
}

fn channel_vector_tensor<B: Backend>(values: [f32; 3], device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(values.to_vec(), [1, 1, 3]), device)
}

fn rgba_u8_to_unit_tensor<B: Backend>(
    rgba: &[u8],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let pixels = validate_rgba_dimensions(rgba, width, height)?;
    Ok(Tensor::<B, 1, Int>::from_data(rgba, device)
        .float()
        .div_scalar(255.0)
        .reshape([pixels, 4])
        .reshape([height, width, 4]))
}

fn alpha_u8_to_unit_tensor<B: Backend>(
    alpha: &[u8],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let pixels = validate_dimensions(width, height)?;
    ensure!(
        alpha.len() == pixels,
        "expected {pixels} alpha bytes for {width}x{height}, got {}",
        alpha.len()
    );
    Ok(Tensor::<B, 1, Int>::from_data(alpha, device)
        .float()
        .div_scalar(255.0)
        .reshape([height, width, 1]))
}

fn mask_panel_tensor_from_rgba<B: Backend>(
    input_rgba: Tensor<B, 3>,
    mask_rgba: &[u8],
    width: usize,
    height: usize,
    blend_alpha: f32,
    mode: AutoGazeMaskVisualizationMode,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let mask = rgba_u8_to_unit_tensor(mask_rgba, width, height, device)?;
    if mode != AutoGazeMaskVisualizationMode::ImageOverlay
        && mode != AutoGazeMaskVisualizationMode::ImageMaskOnly
    {
        return Ok(mask);
    }

    let alpha = colored_mask_alpha_u8(mask_rgba)?;
    let alpha = alpha_u8_to_unit_tensor(&alpha, width, height, device)?;
    let visible = alpha.clone().repeat_dim(2, 4);
    let blend = alpha_blend_tensor(alpha, width, height, blend_alpha, device);
    let inverse = Tensor::<B, 3>::ones([height, width, 4], device).sub(blend.clone());
    let overlay = input_rgba.mul(inverse).add(mask.mul(blend));
    if mode == AutoGazeMaskVisualizationMode::ImageMaskOnly {
        Ok(overlay.mul(visible))
    } else {
        Ok(overlay)
    }
}

fn supports_full_frame_mask_fast_path(mode: AutoGazeMaskVisualizationMode) -> bool {
    matches!(
        mode,
        AutoGazeMaskVisualizationMode::Overlay
            | AutoGazeMaskVisualizationMode::ImageOverlay
            | AutoGazeMaskVisualizationMode::ImageMaskOnly
    )
}

fn full_frame_mask_panel_tensor<B: Backend>(
    input_rgba: Tensor<B, 3>,
    width: usize,
    height: usize,
    blend_alpha: f32,
    mode: AutoGazeMaskVisualizationMode,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mask = Tensor::<B, 3>::ones([height, width, 4], device);
    if mode != AutoGazeMaskVisualizationMode::ImageOverlay
        && mode != AutoGazeMaskVisualizationMode::ImageMaskOnly
    {
        return mask;
    }

    let blend = Tensor::cat(
        vec![
            Tensor::<B, 3>::ones([height, width, 3], device)
                .mul_scalar(blend_alpha.clamp(0.0, 1.0)),
            Tensor::<B, 3>::zeros([height, width, 1], device),
        ],
        2,
    );
    let inverse = Tensor::<B, 3>::ones([height, width, 4], device).sub(blend.clone());
    input_rgba.mul(inverse).add(mask.mul(blend))
}

fn token_alpha_mask_tensor<B: Backend>(
    tokens: &AutoGazeDeviceTokens<B>,
    config: &AutoGazeConfig,
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let [batch, slots] = tokens.tokens.shape().dims::<2>();
    let layouts = scale_token_layouts(config);
    let vocab = layouts
        .iter()
        .map(|layout| layout.token_count)
        .sum::<usize>();
    ensure!(vocab > 0, "AutoGaze token mask vocab must not be empty");
    let grid = layouts
        .iter()
        .map(|layout| layout.grid.max(1))
        .max()
        .unwrap_or(1);
    let basis = token_to_finest_grid_basis::<B>(&layouts, grid, device);
    let vocab_range = Tensor::<B, 1, Int>::arange(0..vocab as i64, device)
        .reshape([1, 1, vocab])
        .repeat_dim(0, batch)
        .repeat_dim(1, slots);
    let selected = tokens
        .tokens
        .clone()
        .reshape([batch, slots, 1])
        .repeat_dim(2, vocab)
        .equal(vocab_range)
        .bool_and(
            tokens
                .valid
                .clone()
                .reshape([batch, slots, 1])
                .repeat_dim(2, vocab),
        )
        .float()
        .sum_dim(1)
        .reshape([batch, vocab])
        .clamp(0.0, 1.0);
    let grid_mask = selected
        .matmul(basis)
        .clamp(0.0, 1.0)
        .reshape([batch, 1, grid, grid]);
    let options = InterpolateOptions::new(InterpolateMode::Nearest);
    let alpha = interpolate(grid_mask, [height, width], options);
    Ok(alpha
        .slice([0..1, 0..1, 0..height, 0..width])
        .reshape([height, width, 1]))
}

fn token_to_finest_grid_basis<B: Backend>(
    layouts: &[crate::AutoGazeScaleTokenLayout],
    target_grid: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let target_grid = target_grid.max(1);
    let pixels = target_grid * target_grid;
    let vocab = layouts
        .iter()
        .map(|layout| layout.token_count)
        .sum::<usize>();
    let mut values = vec![0.0f32; vocab * pixels];
    let mut token_offset = 0usize;
    for layout in layouts {
        let grid = layout.grid.max(1);
        for local in 0..layout.token_count {
            let row = local / grid;
            let col = local % grid;
            let y0 = row.saturating_mul(target_grid) / grid;
            let x0 = col.saturating_mul(target_grid) / grid;
            let y1 = (row + 1).saturating_mul(target_grid).div_ceil(grid);
            let x1 = (col + 1).saturating_mul(target_grid).div_ceil(grid);
            let base = (token_offset + local) * pixels;
            for y in y0.min(target_grid)..y1.min(target_grid) {
                let row_offset = base + y * target_grid;
                for x in x0.min(target_grid)..x1.min(target_grid) {
                    values[row_offset + x] = 1.0;
                }
            }
        }
        token_offset += layout.token_count;
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [vocab, pixels]), device)
}

fn token_mask_panel_tensor<B: Backend>(
    input_rgba: Tensor<B, 3>,
    alpha: Tensor<B, 3>,
    width: usize,
    height: usize,
    blend_alpha: f32,
    mode: AutoGazeMaskVisualizationMode,
    device: &B::Device,
) -> Tensor<B, 3> {
    if mode != AutoGazeMaskVisualizationMode::ImageOverlay
        && mode != AutoGazeMaskVisualizationMode::ImageMaskOnly
    {
        return alpha.repeat_dim(2, 4);
    }

    let visible = alpha.clone().repeat_dim(2, 4);
    let blend = alpha_blend_tensor(alpha, width, height, blend_alpha, device);
    let inverse = Tensor::<B, 3>::ones([height, width, 4], device).sub(blend.clone());
    let overlay = input_rgba
        .mul(inverse)
        .add(Tensor::<B, 3>::ones([height, width, 4], device).mul(blend));
    if mode == AutoGazeMaskVisualizationMode::ImageMaskOnly {
        overlay.mul(visible)
    } else {
        overlay
    }
}

fn alpha_blend_tensor<B: Backend>(
    alpha: Tensor<B, 3>,
    width: usize,
    height: usize,
    blend_alpha: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    let rgb = alpha
        .repeat_dim(2, 3)
        .mul_scalar(blend_alpha.clamp(0.0, 1.0));
    let alpha = Tensor::<B, 3>::zeros([height, width, 1], device);
    Tensor::cat(vec![rgb, alpha], 2)
}

fn colored_mask_alpha_u8(mask_rgba: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        mask_rgba.len().is_multiple_of(4),
        "colored mask must be an RGBA buffer"
    );
    Ok(mask_rgba
        .chunks_exact(4)
        .map(|pixel| {
            if pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 {
                255
            } else {
                0
            }
        })
        .collect())
}

fn build_blend_lookup(blend_alpha: f32) -> [u8; BLEND_LOOKUP_LEN] {
    let blend_alpha = blend_alpha.clamp(0.0, 1.0);
    let inverse = 1.0 - blend_alpha;
    let mut lookup = [0u8; BLEND_LOOKUP_LEN];
    for base in 0..=u8::MAX {
        for overlay in 0..=u8::MAX {
            lookup[blend_lookup_index(base, overlay)] =
                (base as f32 * inverse + overlay as f32 * blend_alpha).round() as u8;
        }
    }
    lookup
}

#[inline]
fn blend_lookup_index(base: u8, overlay: u8) -> usize {
    ((base as usize) << 8) | overlay as usize
}

fn alpha_blend_colored_mask_with_source_into(
    source_rgba: &[u8],
    blend_alpha: f32,
    preserve_unmasked_source: bool,
    mask_rgba: &mut [u8],
) -> Result<()> {
    ensure!(
        source_rgba.len() == mask_rgba.len(),
        "mask overlay source and mask buffers must have the same byte length"
    );
    ensure!(
        source_rgba.len().is_multiple_of(4),
        "mask overlay source must be an RGBA buffer"
    );
    let lookup = build_blend_lookup(blend_alpha);
    for (source, mask) in source_rgba
        .chunks_exact(4)
        .zip(mask_rgba.chunks_exact_mut(4))
    {
        let has_mask = mask[0] != 0 || mask[1] != 0 || mask[2] != 0;
        if has_mask {
            for channel in 0..3 {
                mask[channel] = lookup[blend_lookup_index(source[channel], mask[channel])];
            }
            mask[3] = source[3];
        } else if preserve_unmasked_source {
            mask.copy_from_slice(source);
        } else {
            mask.fill(0);
        }
    }
    Ok(())
}

fn alpha_blend_full_colored_mask_with_source_into(
    source_rgba: &[u8],
    blend_alpha: f32,
    mask_rgba: &mut [u8],
) -> Result<()> {
    ensure!(
        source_rgba.len() == mask_rgba.len(),
        "mask overlay source and mask buffers must have the same byte length"
    );
    ensure!(
        source_rgba.len().is_multiple_of(4),
        "mask overlay source must be an RGBA buffer"
    );
    let lookup = build_blend_lookup(blend_alpha);
    for (source, mask) in source_rgba
        .chunks_exact(4)
        .zip(mask_rgba.chunks_exact_mut(4))
    {
        for channel in 0..3 {
            mask[channel] = lookup[blend_lookup_index(source[channel], mask[channel])];
        }
        mask[3] = source[3];
    }
    Ok(())
}

struct MaskRgbaAndRects {
    mask_rgba: Vec<u8>,
    plan: AutoGazeSparseUpdatePlan,
}

fn mask_rgba_and_rects(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
    mask_mode: AutoGazeMaskVisualizationMode,
) -> Result<MaskRgbaAndRects> {
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    let plan = fixation_sparse_update_plan(width, height, points, cell_scale)?;
    let mask_rgba = match mask_mode {
        AutoGazeMaskVisualizationMode::ImageOverlay => {
            fixation_image_overlay_mask_rgba(rgba, width, height, points, cell_scale, blend_alpha)?
        }
        AutoGazeMaskVisualizationMode::ImageMaskOnly => {
            fixation_image_mask_only_rgba(rgba, width, height, points, cell_scale, blend_alpha)?
        }
        AutoGazeMaskVisualizationMode::Overlay | AutoGazeMaskVisualizationMode::ScaleRows => {
            fixation_mask_rgba(width, height, points, cell_scale, mask_mode)
        }
    };

    Ok(MaskRgbaAndRects { mask_rgba, plan })
}

struct MaskRects {
    plan: AutoGazeSparseUpdatePlan,
    stats: AutoGazeMaskPlanStats,
    row_spans: Vec<(usize, usize, usize)>,
}

fn mask_rgba_and_rects_into(
    rgba: &[u8],
    points: &[FixationPoint],
    options: AutoGazeRgbaVisualizationOptions,
    mask_rgba: &mut Vec<u8>,
) -> Result<MaskRects> {
    let width = options.width;
    let height = options.height;
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    let parts = fixation_sparse_update_plan_parts_for_geometry(
        width,
        height,
        points,
        options.cell_scale,
        options.mask_geometry_mode,
    )?;
    let full_mask = parts.plan.pixel_count == width.saturating_mul(height);
    if options.mask_mode == AutoGazeMaskVisualizationMode::ImageOverlay {
        fixation_mask_rgba_with_geometry_into(
            width,
            height,
            points,
            options.cell_scale,
            AutoGazeMaskVisualizationMode::Overlay,
            options.mask_geometry_mode,
            mask_rgba,
        );
        if full_mask {
            alpha_blend_full_colored_mask_with_source_into(rgba, options.blend_alpha, mask_rgba)?;
        } else {
            alpha_blend_colored_mask_with_source_into(rgba, options.blend_alpha, true, mask_rgba)?;
        }
    } else if options.mask_mode == AutoGazeMaskVisualizationMode::ImageMaskOnly {
        fixation_mask_rgba_with_geometry_into(
            width,
            height,
            points,
            options.cell_scale,
            AutoGazeMaskVisualizationMode::Overlay,
            options.mask_geometry_mode,
            mask_rgba,
        );
        if full_mask {
            alpha_blend_full_colored_mask_with_source_into(rgba, options.blend_alpha, mask_rgba)?;
        } else {
            alpha_blend_colored_mask_with_source_into(rgba, options.blend_alpha, false, mask_rgba)?;
        }
    } else {
        fixation_mask_rgba_with_geometry_into(
            width,
            height,
            points,
            options.cell_scale,
            options.mask_mode,
            options.mask_geometry_mode,
            mask_rgba,
        );
    }
    Ok(MaskRects {
        plan: parts.plan,
        stats: parts.stats,
        row_spans: parts.row_spans,
    })
}

struct MaskBlendRgba {
    mask_rgba: Vec<u8>,
    blend_rgba: Vec<u8>,
    mask_pixel_count: usize,
    mask_plan_stats: AutoGazeMaskPlanStats,
}

fn mask_and_blend_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<MaskBlendRgba> {
    let mask = mask_rgba_and_rects(
        rgba,
        width,
        height,
        points,
        cell_scale,
        blend_alpha,
        AutoGazeMaskVisualizationMode::Overlay,
    )?;
    let blend_rgba = blend_masked_rects_rgba(rgba, width, height, &mask.plan.rects, blend_alpha)?;

    Ok(MaskBlendRgba {
        mask_rgba: mask.mask_rgba,
        blend_rgba,
        mask_pixel_count: mask.plan.pixel_count,
        mask_plan_stats: mask.plan.stats(),
    })
}

fn blend_masked_rects_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    rects: &[FixationPixelRect],
    blend_alpha: f32,
) -> Result<Vec<u8>> {
    let mut blend_rgba = Vec::new();
    blend_masked_rects_rgba_into(rgba, width, height, rects, blend_alpha, &mut blend_rgba)?;
    Ok(blend_rgba)
}

fn blend_masked_rects_rgba_into(
    rgba: &[u8],
    width: usize,
    height: usize,
    rects: &[FixationPixelRect],
    blend_alpha: f32,
    blend_rgba: &mut Vec<u8>,
) -> Result<()> {
    let row_spans = merged_rect_row_spans(width, height, rects);
    blend_masked_row_spans_rgba_into(rgba, width, height, &row_spans, blend_alpha, blend_rgba)
}

fn blend_masked_row_spans_rgba_into(
    rgba: &[u8],
    width: usize,
    height: usize,
    row_spans: &[(usize, usize, usize)],
    blend_alpha: f32,
    blend_rgba: &mut Vec<u8>,
) -> Result<()> {
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    blend_rgba.clear();
    blend_rgba.extend_from_slice(rgba);
    let blend_alpha = blend_alpha.clamp(0.0, 1.0);
    if blend_alpha <= 0.0 || row_spans.is_empty() {
        return Ok(());
    }

    for &(y, x0, x1) in row_spans {
        if y >= height || x0 >= x1 {
            continue;
        }
        let x0 = x0.min(width);
        let x1 = x1.min(width);
        if x0 >= x1 {
            continue;
        }
        for x in x0..x1 {
            let offset = (y * width + x) * 4;
            for channel in 0..3 {
                let base = rgba[offset + channel] as f32;
                blend_rgba[offset + channel] =
                    (base * (1.0 - blend_alpha) + 255.0 * blend_alpha).round() as u8;
            }
        }
    }

    Ok(())
}

fn copy_masked_pixels_rgba(
    source_rgba: &[u8],
    mask_rgba: &[u8],
    target_rgba: &mut [u8],
) -> Result<()> {
    ensure!(
        source_rgba.len() == mask_rgba.len() && source_rgba.len() == target_rgba.len(),
        "source, mask, and target RGBA byte lengths must match"
    );
    ensure!(
        source_rgba.len().is_multiple_of(4),
        "dense masked update inputs must be RGBA buffers"
    );
    for ((source, mask), target) in source_rgba
        .chunks_exact(4)
        .zip(mask_rgba.chunks_exact(4))
        .zip(target_rgba.chunks_exact_mut(4))
    {
        if mask[0] != 0 || mask[1] != 0 || mask[2] != 0 {
            target.copy_from_slice(source);
        }
    }
    Ok(())
}

fn should_use_dense_mask_rgba_update(
    width: usize,
    height: usize,
    stats: AutoGazeMaskPlanStats,
    row_span_count: usize,
    mode: AutoGazeMaskVisualizationMode,
    min_update_ratio: f64,
) -> bool {
    if !matches!(
        mode,
        AutoGazeMaskVisualizationMode::Overlay
            | AutoGazeMaskVisualizationMode::ImageMaskOnly
            | AutoGazeMaskVisualizationMode::ScaleRows
    ) {
        return false;
    }
    should_use_full_frame_update(width, height, stats, min_update_ratio)
        && stats.pixel_count < width.saturating_mul(height)
        && row_span_count > height.max(1).saturating_mul(2)
}

pub fn fixation_rect_union_pixel_count(
    rects: &[FixationPixelRect],
    width: usize,
    height: usize,
) -> usize {
    if rects.is_empty() || width == 0 || height == 0 {
        return 0;
    }

    merged_rect_row_spans(width, height, rects)
        .into_iter()
        .map(|(_, x0, x1)| x1.saturating_sub(x0))
        .sum()
}

fn build_visualization(
    rgba: &[u8],
    panels: AutoGazeVisualizationPanels,
) -> Result<AutoGazeVisualization> {
    let AutoGazeVisualizationPanels {
        width,
        height,
        mask_rgba,
        blend_rgba,
        mask_pixel_count,
        updated_pixel_count,
        mask_plan_stats,
    } = panels;
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    ensure!(
        mask_rgba.len() == rgba.len(),
        "mask RGBA byte length must match input frame"
    );
    ensure!(
        blend_rgba.len() == rgba.len(),
        "blend RGBA byte length must match input frame"
    );

    let mut side_by_side_rgba = Vec::new();
    let side_by_side_width = build_side_by_side_rgba_into(
        rgba,
        width,
        height,
        &mask_rgba,
        &blend_rgba,
        &mut side_by_side_rgba,
    )?;

    Ok(AutoGazeVisualization {
        width,
        height,
        side_by_side_width,
        mask_rgba,
        blend_rgba,
        side_by_side_rgba,
        mask_pixel_count,
        updated_pixel_count,
        mask_plan_stats,
    })
}

fn build_visualization_from_buffers(
    rgba: &[u8],
    width: usize,
    height: usize,
    buffers: &mut AutoGazeRgbaVisualizationBuffers,
    mask_pixel_count: usize,
    updated_pixel_count: usize,
    mask_plan_stats: AutoGazeMaskPlanStats,
) -> Result<AutoGazeVisualization> {
    let side_by_side_width = build_side_by_side_rgba_into(
        rgba,
        width,
        height,
        &buffers.mask_rgba,
        &buffers.blend_rgba,
        &mut buffers.side_by_side_rgba,
    )?;
    let mask_rgba = std::mem::take(&mut buffers.mask_rgba);
    let blend_rgba = std::mem::take(&mut buffers.blend_rgba);
    let side_by_side_rgba = std::mem::take(&mut buffers.side_by_side_rgba);

    Ok(AutoGazeVisualization {
        width,
        height,
        side_by_side_width,
        mask_rgba,
        blend_rgba,
        side_by_side_rgba,
        mask_pixel_count,
        updated_pixel_count,
        mask_plan_stats,
    })
}

fn build_side_by_side_rgba_into(
    rgba: &[u8],
    width: usize,
    height: usize,
    mask_rgba: &[u8],
    blend_rgba: &[u8],
    side_by_side_rgba: &mut Vec<u8>,
) -> Result<usize> {
    let _ = validate_rgba_dimensions(rgba, width, height)?;
    ensure!(
        mask_rgba.len() == rgba.len(),
        "mask RGBA byte length must match input frame"
    );
    ensure!(
        blend_rgba.len() == rgba.len(),
        "blend RGBA byte length must match input frame"
    );
    let side_by_side_width = width
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("side-by-side visualization width overflow"))?;
    let side_by_side_bytes = visualization_rgba_len(side_by_side_width, height)?;
    side_by_side_rgba.clear();
    side_by_side_rgba.resize(side_by_side_bytes, 0);

    let row_bytes = width * 4;
    let out_row_bytes = side_by_side_width * 4;
    for y in 0..height {
        let src = y * row_bytes;
        let dst = y * out_row_bytes;
        side_by_side_rgba[dst..dst + row_bytes].copy_from_slice(&rgba[src..src + row_bytes]);
        side_by_side_rgba[dst + row_bytes..dst + 2 * row_bytes]
            .copy_from_slice(&mask_rgba[src..src + row_bytes]);
        side_by_side_rgba[dst + 2 * row_bytes..dst + 3 * row_bytes]
            .copy_from_slice(&blend_rgba[src..src + row_bytes]);
    }
    Ok(side_by_side_width)
}

fn ratio(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

fn scale_color_for_point(point: FixationPoint) -> [u8; 3] {
    match scale_grid_for_point(point) {
        0..=2 => [255, 180, 0],
        3..=4 => [60, 220, 120],
        5..=7 => [0, 185, 255],
        _ => [230, 110, 255],
    }
}

fn scale_grid_for_point(point: FixationPoint) -> usize {
    point
        .cell_grid()
        .unwrap_or_else(|| nearest_scale_grid(point))
}

fn nearest_scale_grid(point: FixationPoint) -> usize {
    let recovered = 1.0 / point.cell_width().max(point.cell_height()).max(1.0e-6);
    DEFAULT_AUTOGAZE_SCALE_GRIDS
        .into_iter()
        .min_by(|left, right| {
            ((*left as f32 - recovered).abs()).total_cmp(&(*right as f32 - recovered).abs())
        })
        .unwrap_or(14)
}

fn scale_row_grids(by_grid: &BTreeMap<usize, Vec<FixationPoint>>) -> Vec<usize> {
    by_grid.keys().copied().collect()
}

fn partition_range(index: usize, parts: usize, extent: usize) -> (usize, usize) {
    let parts = parts.max(1);
    let extent = extent.max(1);
    let index = index.min(parts - 1);
    let start = index.saturating_mul(extent) / parts;
    let end = (index + 1).saturating_mul(extent).saturating_add(parts - 1) / parts;
    (start.min(extent), end.min(extent).max(start))
}

fn pixel_range(min: f32, max: f32, extent: usize) -> (usize, usize) {
    let extent_f = extent as f32;
    let mut start = (min.clamp(0.0, 1.0) * extent_f).floor() as usize;
    let mut end = (max.clamp(0.0, 1.0) * extent_f).ceil() as usize;
    start = start.min(extent.saturating_sub(1));
    end = end.min(extent);
    if end <= start {
        end = (start + 1).min(extent);
    }
    (start, end)
}

fn pixel_range_f64(min: f64, max: f64, extent: usize) -> (usize, usize) {
    let extent_f = extent as f64;
    let mut start = (min.clamp(0.0, 1.0) * extent_f).floor() as usize;
    let mut end = (max.clamp(0.0, 1.0) * extent_f).ceil() as usize;
    start = start.min(extent.saturating_sub(1));
    end = end.min(extent);
    if end <= start {
        end = (start + 1).min(extent);
    }
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{AutoGazeRgbaClipShape, rgba_clip_to_tensor};
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn draws_crisp_binary_cell_mask() {
        let point = FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 0.9);
        let alpha = fixation_alpha_mask(8, 8, &[point], 1.0);

        for y in 0..8 {
            for x in 0..8 {
                let expected = if x < 4 && y < 4 { 255 } else { 0 };
                assert_eq!(alpha[y * 8 + x], expected, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn effective_alpha_mask_does_not_turn_all_coarse_tokens_into_full_frame_update() {
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.25, 0.75, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.75, 0.5, 0.5, 1.0, 2),
        ];
        let naive = fixation_alpha_mask(28, 28, &points, 1.0);
        let effective = fixation_effective_alpha_mask(28, 28, &points, 1.0);

        assert_eq!(naive.iter().filter(|&&value| value > 0).count(), 28 * 28);
        assert_eq!(effective.iter().filter(|&&value| value > 0).count(), 16);
    }

    #[test]
    fn deduplicated_sparse_update_plan_preserves_native_union_with_fewer_rects() {
        let width = 28;
        let height = 28;
        let coarse = FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2);
        let fine_inside = FixationPoint::with_grid_extent(
            1.5 / 14.0,
            1.5 / 14.0,
            1.0 / 14.0,
            1.0 / 14.0,
            1.0,
            14,
        );
        let points = [fine_inside, coarse];
        let native = fixation_sparse_update_plan(width, height, &points, 1.0).expect("native plan");
        let deduplicated = fixation_deduplicated_sparse_update_plan(width, height, &points, 1.0)
            .expect("deduplicated plan");

        assert_eq!(native.pixel_count, deduplicated.pixel_count);
        assert_eq!(native.update_ratio(), deduplicated.update_ratio());
        assert_eq!(native.rects.len(), 2);
        assert_eq!(deduplicated.rects.len(), 1);
        assert_eq!(
            deduplicated.rects[0],
            point_pixel_rect(width, height, coarse, 1.0)
        );
    }

    #[test]
    fn deduplicated_visualization_keeps_output_union_but_skips_covered_cell_draw() {
        let width = 4;
        let height = 4;
        let rgba = [20, 40, 60, 255].repeat(width * height);
        let coarse = FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2);
        let fine_inside = FixationPoint::with_grid_extent(0.125, 0.125, 0.25, 0.25, 1.0, 4);
        let points = [coarse, fine_inside];
        let mut native = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 0);
        let native = native
            .visualize_rgba_panels_with_options(
                &rgba,
                &points,
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.5)
                    .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::Overlay),
            )
            .expect("native panels");
        let mut dedup = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 0);
        let dedup = dedup
            .visualize_rgba_panels_with_options(
                &rgba,
                &points,
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.5)
                    .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::Overlay)
                    .with_mask_geometry_mode(AutoGazeMaskGeometryMode::Deduplicated),
            )
            .expect("deduplicated panels");

        assert_eq!(native.mask_pixel_count, dedup.mask_pixel_count);
        assert_eq!(native.blend_rgba, dedup.blend_rgba);
        assert!(dedup.mask_rgba.chunks_exact(4).any(|pixel| pixel[0] == 255));
        assert!(
            dedup
                .mask_rgba
                .chunks_exact(4)
                .all(|pixel| pixel[0..3] != [60, 220, 120]),
            "covered fine cell should be dropped from the mask drawing"
        );
    }

    #[test]
    fn visualization_uses_native_multiscale_cells_for_mask_and_output() {
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.25, 0.75, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.75, 0.5, 0.5, 1.0, 2),
        ];
        let rgba = [10, 20, 30, 255].repeat(28 * 28);
        let visualization =
            visualize_fixations_rgba(&rgba, 28, 28, &points, 1.0, 1.0).expect("visualize");
        let white_pixels = visualization
            .blend_rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
            .count();
        let colored_mask_pixels = visualization
            .mask_rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
            .count();

        assert_eq!(visualization.mask_pixel_count, 28 * 28);
        assert_eq!(white_pixels, 28 * 28);
        assert_eq!(colored_mask_pixels, 28 * 28);
        assert_eq!(visualization.mask_ratio(), 1.0);
    }

    #[test]
    fn effective_mask_color_and_alpha_use_identical_projected_cells() {
        let coarse = FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2);
        let fine = FixationPoint::with_grid_extent(
            11.5 / 14.0,
            11.5 / 14.0,
            1.0 / 14.0,
            1.0 / 14.0,
            1.0,
            14,
        );
        let points = [coarse, fine];
        let alpha = fixation_effective_alpha_mask(28, 28, &points, 1.0);
        let mask = fixation_effective_scale_mask_rgba(28, 28, &points, 1.0);

        for (index, pixel) in mask.chunks_exact(4).enumerate() {
            let colored = pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0;
            assert_eq!(colored, alpha[index] > 0, "pixel {index}");
        }
    }

    #[test]
    fn draws_crisp_scale_colored_cells_with_fine_cells_on_top() {
        let coarse = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 0.9, 2);
        let fine = FixationPoint::with_grid_extent(0.625, 0.625, 0.25, 0.25, 0.9, 4);
        let rgba = fixation_scale_mask_rgba(8, 8, &[fine, coarse], 1.0);

        assert_eq!(&rgba[0..4], &[255, 180, 0, 255]);
        let fine_offset = (5 * 8 + 5) * 4;
        assert_eq!(&rgba[fine_offset..fine_offset + 4], &[60, 220, 120, 255]);
    }

    #[test]
    fn scale_colored_cell_fill_uses_topmost_scale_without_color_mixing() {
        let coarse = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 0.9, 2);
        let mid = FixationPoint::with_grid_extent(0.5, 0.5, 0.5, 0.5, 0.9, 4);
        let fine = FixationPoint::with_grid_extent(0.5, 0.5, 0.25, 0.25, 0.9, 7);
        let rgba = fixation_scale_mask_rgba(16, 16, &[coarse, mid, fine], 1.0);

        let fine_interior = (8 * 16 + 8) * 4;
        assert_eq!(&rgba[fine_interior..fine_interior + 4], &[0, 78, 107, 255]);

        let mid_only_interior = (5 * 16 + 5) * 4;
        assert_eq!(
            &rgba[mid_only_interior..mid_only_interior + 4],
            &[25, 92, 50, 255]
        );
    }

    #[test]
    fn image_overlay_mask_visualization_blends_mask_over_source() {
        let source = vec![
            10, 20, 30, 255, //
            100, 110, 120, 255,
        ];
        let point = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2);
        let colored = fixation_scale_mask_rgba(2, 1, &[point], 1.0);
        let overlay = fixation_image_overlay_mask_rgba(&source, 2, 1, &[point], 1.0, 0.5).unwrap();

        assert_eq!(
            &overlay[0..4],
            &[
                ((source[0] as f32 + colored[0] as f32) * 0.5).round() as u8,
                ((source[1] as f32 + colored[1] as f32) * 0.5).round() as u8,
                ((source[2] as f32 + colored[2] as f32) * 0.5).round() as u8,
                255,
            ]
        );
        assert_eq!(&overlay[4..8], &source[4..8]);
    }

    #[test]
    fn image_mask_only_visualization_blanks_unmasked_pixels() {
        let source = vec![
            10, 20, 30, 255, //
            100, 110, 120, 255,
        ];
        let point = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2);
        let colored = fixation_scale_mask_rgba(2, 1, &[point], 1.0);
        let mask_only = fixation_image_mask_only_rgba(&source, 2, 1, &[point], 1.0, 0.5).unwrap();

        assert_eq!(
            &mask_only[0..4],
            &[
                ((source[0] as f32 + colored[0] as f32) * 0.5).round() as u8,
                ((source[1] as f32 + colored[1] as f32) * 0.5).round() as u8,
                ((source[2] as f32 + colored[2] as f32) * 0.5).round() as u8,
                255,
            ]
        );
        assert_eq!(&mask_only[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn full_coverage_image_mask_only_blends_every_pixel() {
        let width = 2;
        let height = 2;
        let source = vec![
            10, 20, 30, 255, //
            40, 50, 60, 255, //
            70, 80, 90, 255, //
            100, 110, 120, 255,
        ];
        let point = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 0);
        let panels = state
            .visualize_rgba_panels_with_options(
                &source,
                &[point],
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.5)
                    .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageMaskOnly),
            )
            .expect("panels");
        let colored = fixation_scale_mask_rgba(width, height, &[point], 1.0);

        for pixel in 0..(width * height) {
            let offset = pixel * 4;
            for channel in 0..3 {
                assert_eq!(
                    panels.mask_rgba[offset + channel],
                    ((source[offset + channel] as f32 + colored[offset + channel] as f32) * 0.5)
                        .round() as u8,
                    "pixel {pixel} channel {channel}"
                );
            }
            assert_eq!(panels.mask_rgba[offset + 3], 255);
        }
    }

    #[test]
    fn scale_rows_mask_visualization_separates_non_nested_scale_grids() {
        let coarse = FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2);
        let fine = FixationPoint::with_grid_extent(0.875, 0.875, 0.25, 0.25, 1.0, 4);
        let rgba = fixation_scale_rows_mask_rgba(8, 8, &[fine, coarse], 1.0);

        let coarse_offset = 3 * 4;
        assert_eq!(&rgba[coarse_offset..coarse_offset + 4], &[255, 180, 0, 255]);
        let fine_offset = (7 * 8 + 5) * 4;
        assert_eq!(&rgba[fine_offset..fine_offset + 4], &[60, 220, 120, 255]);

        let unused_scale_position = (7 * 8 + 7) * 4;
        assert_eq!(
            &rgba[unused_scale_position..unused_scale_position + 4],
            &[0, 0, 0, 255]
        );
    }

    #[test]
    fn scale_rows_mask_visualization_preserves_source_aspect_inside_rows() {
        let coarse = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2);
        let fine = FixationPoint::with_grid_extent(0.875, 0.875, 0.25, 0.25, 1.0, 4);
        let rgba = fixation_scale_rows_mask_rgba(16, 8, &[fine, coarse], 1.0);

        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[(4 * 4)..(4 * 4) + 4], &[255, 180, 0, 255]);

        let right_margin = (7 * 16 + 15) * 4;
        assert_eq!(&rgba[right_margin..right_margin + 4], &[0, 0, 0, 255]);
        let fine_offset = (7 * 16 + 11) * 4;
        assert_eq!(&rgba[fine_offset..fine_offset + 4], &[60, 220, 120, 255]);
    }

    #[test]
    fn scale_rows_mask_visualization_does_not_reserve_empty_default_rows() {
        let coarse = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2);
        let rgba = fixation_scale_rows_mask_rgba(16, 8, &[coarse], 1.0);

        assert_eq!(&rgba[0..4], &[255, 180, 0, 255]);
        assert_eq!(
            &rgba[((7 * 16 + 15) * 4)..((7 * 16 + 15) * 4) + 4],
            &[255, 180, 0, 255]
        );
    }

    #[test]
    fn blends_selected_cells_with_white() {
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let visualization =
            visualize_fixations_rgba(&rgba, 2, 1, &[point], 1.0, 0.5).expect("visualize");

        assert_eq!(&visualization.mask_rgba[0..4], &[255, 180, 0, 255]);
        assert_eq!(&visualization.mask_rgba[4..8], &[0, 0, 0, 255]);
        assert_eq!(&visualization.blend_rgba[0..4], &[178, 153, 128, 255]);
        assert_eq!(&visualization.blend_rgba[4..8], &[10, 20, 30, 255]);
        assert_eq!(visualization.mask_pixel_count, 1);
        assert_eq!(visualization.updated_pixel_count, 1);
        assert_eq!(visualization.mask_ratio(), 0.5);
        assert_eq!(visualization.update_ratio(), 0.5);
    }

    #[test]
    fn side_by_side_layout_uses_input_mask_output_columns() {
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let visualization =
            visualize_fixations_rgba(&rgba, 2, 1, &[point], 1.0, 0.5).expect("visualize");

        assert_eq!(visualization.side_by_side_width, 6);
        assert_eq!(&visualization.side_by_side_rgba[0..8], &rgba);
        assert_eq!(
            &visualization.side_by_side_rgba[8..16],
            &visualization.mask_rgba
        );
        assert_eq!(
            &visualization.side_by_side_rgba[16..24],
            &visualization.blend_rgba
        );
    }

    #[test]
    fn panel_visualization_matches_side_by_side_visualization() {
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let mut panel_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let panels = panel_state
            .visualize_rgba_panels(&rgba, 2, 1, &[point], 1.0, 0.5)
            .expect("panel visualization");
        let side_by_side_from_panels = panels
            .clone()
            .into_side_by_side(&rgba)
            .expect("side-by-side from panels");

        let mut regular_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let regular = regular_state
            .visualize_rgba(&rgba, 2, 1, &[point], 1.0, 0.5)
            .expect("regular visualization");

        assert_eq!(panels.mask_rgba, regular.mask_rgba);
        assert_eq!(panels.blend_rgba, regular.blend_rgba);
        assert_eq!(panels.mask_ratio(), regular.mask_ratio());
        assert_eq!(panels.update_ratio(), regular.update_ratio());
        assert_eq!(
            side_by_side_from_panels.side_by_side_rgba,
            regular.side_by_side_rgba
        );
    }

    #[test]
    fn reusable_panel_buffers_match_owned_visualization() {
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let options = AutoGazeRgbaVisualizationOptions::new(2, 1, 1.0, 0.5);
        let mut owned_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let owned = owned_state
            .visualize_rgba_panels_with_options(&rgba, &[point], options)
            .expect("owned panels");

        let mut buffers = AutoGazeRgbaVisualizationBuffers::default();
        let mut borrowed_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let borrowed = borrowed_state
            .visualize_rgba_panels_with_options_into(&rgba, &[point], options, &mut buffers)
            .expect("borrowed panels");

        assert_eq!(borrowed.to_owned(), owned);
        assert!(buffers.mask_rgba.capacity() >= rgba.len());
        assert!(buffers.blend_rgba.capacity() >= rgba.len());
    }

    #[test]
    fn reusable_interframe_buffers_keep_capacity_stable() {
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        let second = [30, 0, 0, 255, 40, 0, 0, 255];
        let options = AutoGazeRgbaVisualizationOptions::new(2, 1, 1.0, 1.0);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0);
        let mut buffers = AutoGazeRgbaVisualizationBuffers::default();

        state
            .visualize_rgba_panels_with_options_into(&first, &[point], options, &mut buffers)
            .expect("first frame");
        let mask_capacity = buffers.mask_rgba.capacity();
        let blend_capacity = buffers.blend_rgba.capacity();
        {
            let second_view = state
                .visualize_rgba_panels_with_options_into(&second, &[point], options, &mut buffers)
                .expect("second frame");
            assert_eq!(&second_view.blend_rgba[0..4], &[30, 0, 0, 255]);
            assert_eq!(&second_view.blend_rgba[4..8], &[20, 0, 0, 255]);
        }
        assert_eq!(buffers.mask_rgba.capacity(), mask_capacity);
        assert_eq!(buffers.blend_rgba.capacity(), blend_capacity);
    }

    #[test]
    fn tensor_panel_visualization_matches_byte_panel_visualization() {
        let device = Default::default();
        let width = 2;
        let height = 1;
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let mut byte_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let byte = byte_state
            .visualize_rgba_panels(&rgba, width, height, &[point], 1.0, 0.5)
            .expect("byte panels");
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("normalized tensor");
        let mut tensor_state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::FullBlend,
            30,
        );
        let tensor = tensor_state
            .visualize_normalized_rgb_clip_panels(
                tensor,
                &[point],
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.5),
                &device,
            )
            .expect("tensor panels");

        assert_eq!(tensor.update_ratio(), byte.update_ratio());
        assert_eq!(tensor_to_rgba_bytes(tensor.input_rgba), rgba);
        assert_eq!(tensor_to_rgba_bytes(tensor.mask_rgba), byte.mask_rgba);
        assert_eq!(tensor_to_rgba_bytes(tensor.output_rgba), byte.blend_rgba);
    }

    #[test]
    fn side_by_side_buffer_has_exact_three_column_size_for_tall_frames() {
        let rgba = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 1, 2, 3, 255,
        ];
        let visualization =
            visualize_fixations_rgba(&rgba, 2, 2, &[], 1.0, 0.5).expect("visualize tall frame");

        assert_eq!(visualization.side_by_side_width, 6);
        assert_eq!(visualization.side_by_side_rgba.len(), 2 * 2 * 3 * 4);
    }

    #[test]
    fn interframe_mode_preserves_unmasked_regions_until_keyframe() {
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 3);

        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        let first_visualization = state
            .visualize_rgba(&first, 2, 1, &[point], 1.0, 1.0)
            .expect("first visualization");
        assert!(state.last_frame_was_keyframe());
        assert_eq!(&first_visualization.blend_rgba[0..4], &[10, 0, 0, 255]);
        assert_eq!(&first_visualization.blend_rgba[4..8], &[20, 0, 0, 255]);
        assert_eq!(first_visualization.mask_ratio(), 0.5);
        assert_eq!(first_visualization.update_ratio(), 1.0);

        let second = [30, 0, 0, 255, 40, 0, 0, 255];
        let second_visualization = state
            .visualize_rgba(&second, 2, 1, &[point], 1.0, 1.0)
            .expect("second visualization");
        assert!(!state.last_frame_was_keyframe());
        assert_eq!(
            &second_visualization.blend_rgba[0..8],
            &[30, 0, 0, 255, 20, 0, 0, 255]
        );
        assert_eq!(second_visualization.mask_ratio(), 0.5);
        assert_eq!(second_visualization.update_ratio(), 0.5);

        let third = [50, 0, 0, 255, 60, 0, 0, 255];
        let third_visualization = state
            .visualize_rgba(&third, 2, 1, &[], 1.0, 1.0)
            .expect("third visualization");
        assert!(!state.last_frame_was_keyframe());
        assert_eq!(
            &third_visualization.blend_rgba[0..8],
            &[30, 0, 0, 255, 20, 0, 0, 255]
        );
        assert_eq!(third_visualization.mask_ratio(), 0.0);
        assert_eq!(third_visualization.update_ratio(), 0.0);

        let fourth = [70, 0, 0, 255, 80, 0, 0, 255];
        let fourth_visualization = state
            .visualize_rgba(&fourth, 2, 1, &[], 1.0, 1.0)
            .expect("fourth visualization");
        assert!(state.last_frame_was_keyframe());
        assert_eq!(
            &fourth_visualization.blend_rgba[0..8],
            &[70, 0, 0, 255, 80, 0, 0, 255]
        );
        assert_eq!(fourth_visualization.mask_ratio(), 0.0);
        assert_eq!(fourth_visualization.update_ratio(), 1.0);
    }

    #[test]
    fn zero_keyframe_duration_disables_periodic_interframe_keyframes() {
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0);

        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        state
            .visualize_rgba(&first, 2, 1, &[point], 1.0, 1.0)
            .expect("first visualization");
        assert!(state.last_frame_was_keyframe());

        for frame_idx in 1..8 {
            let value = (30 + frame_idx) as u8;
            let frame = [value, 0, 0, 255, value + 1, 0, 0, 255];
            state
                .visualize_rgba(&frame, 2, 1, &[point], 1.0, 1.0)
                .expect("interframe visualization");
            assert!(!state.last_frame_was_keyframe());
        }
    }

    #[test]
    fn interframe_updates_are_driven_by_alpha_not_visible_mask_color() {
        let blue_scale = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 7);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 10);

        let first = [10, 0, 0, 255];
        let first_visualization = state
            .visualize_rgba(&first, 1, 1, &[blue_scale], 1.0, 1.0)
            .expect("first visualization");
        assert_eq!(&first_visualization.mask_rgba[0..4], &[0, 185, 255, 255]);

        let second = [50, 0, 0, 255];
        let second_visualization = state
            .visualize_rgba(&second, 1, 1, &[blue_scale], 1.0, 1.0)
            .expect("second visualization");

        assert_eq!(&second_visualization.blend_rgba[0..4], &[50, 0, 0, 255]);
        assert_eq!(second_visualization.update_ratio(), 1.0);
    }

    #[test]
    fn rect_union_pixel_count_deduplicates_overlapping_multiscale_cells() {
        let rects = [
            FixationPixelRect {
                x0: 0,
                x1: 3,
                y0: 0,
                y1: 2,
            },
            FixationPixelRect {
                x0: 2,
                x1: 4,
                y0: 1,
                y1: 3,
            },
        ];

        assert_eq!(fixation_rect_union_pixel_count(&rects, 4, 4), 9);
    }

    #[test]
    fn effective_sparse_update_plan_merges_dense_grid_cells_into_row_bands() {
        let width = 640;
        let height = 360;
        let grid = 64;
        let points = dense_grid_points(grid);

        let raw_rects = fixation_effective_cell_rects(width, height, &points, 1.0);
        let plan =
            fixation_effective_sparse_update_plan(width, height, &points, 1.0).expect("plan");

        assert_eq!(raw_rects.len(), grid * grid);
        assert_eq!(plan.rects.len(), grid);
        assert_eq!(plan.pixel_count, width * height);
    }

    #[test]
    fn tensor_visualization_reports_compacted_dense_grid_mask_stats() {
        let device = Default::default();
        let width = 64;
        let height = 36;
        let grid = 64;
        let rgba = deterministic_rgba(width, height, 17);
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("tensor");
        let points = dense_grid_points(grid);
        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );

        let panels = state
            .visualize_normalized_rgb_clip_panels(
                tensor,
                &points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("dense visualization");
        let stats = state.last_mask_plan_stats().expect("mask plan stats");

        assert_eq!(panels.mask_pixel_count, width * height);
        assert_eq!(stats.pixel_count, width * height);
        assert_eq!(stats.rect_count, grid);
        assert_eq!(stats.row_span_count, height);
    }

    #[test]
    fn full_blend_overlapping_cells_are_blended_once() {
        let coarse = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2);
        let duplicate = FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2);
        let rgba = [100, 0, 0, 255];
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 10);

        let visualization = state
            .visualize_rgba_panels(&rgba, 1, 1, &[coarse, duplicate], 1.0, 0.5)
            .expect("visualize");

        assert_eq!(&visualization.blend_rgba[0..4], &[178, 128, 128, 255]);
        assert_eq!(visualization.mask_pixel_count, 1);
    }

    #[test]
    fn sparse_update_plan_matches_interframe_rect_copy() {
        let width = 4;
        let height = 2;
        let source_a = vec![10u8; width * height * 4];
        let mut source_b = vec![20u8; width * height * 4];
        for pixel in source_b.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        let point = FixationPoint::with_grid_extent(0.125, 0.25, 0.25, 0.5, 1.0, 4);
        let plan =
            fixation_sparse_update_plan(width, height, &[point], 1.0).expect("sparse update plan");
        assert_eq!(
            plan.rects,
            vec![FixationPixelRect {
                x0: 0,
                x1: 1,
                y0: 0,
                y1: 1
            }]
        );
        assert_eq!(plan.pixel_count, 1);

        let mut target = source_a.clone();
        let copied =
            copy_sparse_update_rgba(&source_b, &mut target, &plan).expect("sparse rgba update");
        assert_eq!(copied, 1);
        assert_eq!(&target[0..4], &source_b[0..4]);
        assert_eq!(&target[4..], &source_a[4..]);
    }

    #[test]
    fn sparse_update_tensor_matches_rect_copy() {
        let device = Default::default();
        let width = 4;
        let height = 2;
        let source_a = deterministic_rgba(width, height, 3);
        let source_b = deterministic_rgba(width, height, 11);
        let point = FixationPoint::with_grid_extent(0.125, 0.25, 0.25, 0.5, 1.0, 4);
        let plan =
            fixation_sparse_update_plan(width, height, &[point], 1.0).expect("sparse update plan");

        let mut expected = source_a.clone();
        copy_sparse_update_rgba(&source_b, &mut expected, &plan).expect("sparse rgba update");
        let source_tensor =
            rgba_u8_to_unit_tensor::<TestBackend>(&source_b, width, height, &device)
                .expect("source tensor");
        let target_tensor =
            rgba_u8_to_unit_tensor::<TestBackend>(&source_a, width, height, &device)
                .expect("target tensor");
        let actual = copy_sparse_update_tensor(source_tensor, target_tensor, &plan)
            .expect("sparse tensor update");

        assert_eq!(tensor_to_rgba_bytes(actual), expected);
    }

    #[test]
    fn sparse_tensor_update_heuristic_only_uses_tiny_regions() {
        let tiny = AutoGazeSparseUpdatePlan::new(
            100,
            100,
            vec![FixationPixelRect {
                x0: 0,
                x1: 4,
                y0: 0,
                y1: 4,
            }],
        )
        .expect("tiny plan");
        assert!(should_use_sparse_tensor_update_rects(
            tiny.width,
            tiny.height,
            &tiny.rects,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        ));

        let coarse = AutoGazeSparseUpdatePlan::new(
            100,
            100,
            vec![FixationPixelRect {
                x0: 0,
                x1: 50,
                y0: 0,
                y1: 50,
            }],
        )
        .expect("coarse plan");
        assert!(!should_use_sparse_tensor_update_rects(
            coarse.width,
            coarse.height,
            &coarse.rects,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        ));

        let many = AutoGazeSparseUpdatePlan::new(
            100,
            100,
            (0..5)
                .map(|index| FixationPixelRect {
                    x0: index * 2,
                    x1: index * 2 + 1,
                    y0: 0,
                    y1: 1,
                })
                .collect(),
        )
        .expect("many plan");
        assert!(!should_use_sparse_tensor_update_rects(
            many.width,
            many.height,
            &many.rects,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        ));
        assert!(should_use_sparse_tensor_update_rects(
            many.width,
            many.height,
            &many.rects,
            8,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        ));
        assert!(!should_use_sparse_tensor_update_rects(
            tiny.width,
            tiny.height,
            &tiny.rects,
            0,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RATIO,
        ));
        assert!(!should_use_sparse_tensor_update_rects(
            tiny.width,
            tiny.height,
            &tiny.rects,
            DEFAULT_TENSOR_SPARSE_UPDATE_MAX_RECTS,
            f64::NAN,
        ));
    }

    #[test]
    fn tensor_interframe_path_reports_keyframe_sparse_and_dense_modes() {
        let device = Default::default();
        let width = 100;
        let height = 100;
        let previous = deterministic_rgba(width, height, 5);
        let current = deterministic_rgba(width, height, 13);
        let tiny = [FixationPoint::with_grid_extent(
            0.02, 0.02, 0.04, 0.04, 1.0, 100,
        )];
        let many = [
            FixationPoint::with_grid_extent(0.02, 0.02, 0.04, 0.04, 1.0, 100),
            FixationPoint::with_grid_extent(0.08, 0.14, 0.04, 0.04, 1.0, 100),
            FixationPoint::with_grid_extent(0.14, 0.26, 0.04, 0.04, 1.0, 100),
            FixationPoint::with_grid_extent(0.20, 0.38, 0.04, 0.04, 1.0, 100),
            FixationPoint::with_grid_extent(0.26, 0.50, 0.04, 0.04, 1.0, 100),
        ];
        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );

        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                previous_tensor,
                &tiny,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("keyframe");
        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::Keyframe)
        );

        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                current_tensor.clone(),
                &tiny,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("tiny sparse update");
        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::SparseRects)
        );

        state
            .visualize_normalized_rgb_clip_panels(
                current_tensor,
                &many,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.0),
                &device,
            )
            .expect("coarse dense update");
        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::DenseMask)
        );

        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );
        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                previous_tensor,
                &tiny,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_sparse_update_policy(8, 0.05),
                &device,
            )
            .expect("custom policy keyframe");
        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                current_tensor,
                &many,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_sparse_update_policy(8, 0.05),
                &device,
            )
            .expect("custom policy sparse update");
        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::SparseRects)
        );
    }

    #[test]
    fn tensor_interframe_switches_to_full_frame_for_dense_updates() {
        let device = Default::default();
        let width = 8;
        let height = 4;
        let previous = deterministic_rgba(width, height, 11);
        let current = deterministic_rgba(width, height, 23);
        let points = [FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2)];
        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );

        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                previous_tensor,
                &[],
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("keyframe");

        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        let output = state
            .visualize_normalized_rgb_clip_panels(
                current_tensor,
                &points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
                &device,
            )
            .expect("full-frame update");

        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::FullFrame)
        );
        assert_eq!(output.mask_ratio(), 1.0);
        assert_eq!(output.update_ratio(), 1.0);
        assert_eq!(tensor_to_rgba_bytes(output.output_rgba), current);
    }

    #[test]
    fn tensor_interframe_full_frame_policy_wins_over_permissive_sparse_policy() {
        let device = Default::default();
        let width = 16;
        let height = 16;
        let previous = deterministic_rgba(width, height, 31);
        let current = deterministic_rgba(width, height, 37);
        let points = [FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2)];
        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );

        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                previous_tensor,
                &[],
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("keyframe");

        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        let output = state
            .visualize_normalized_rgb_clip_panels(
                current_tensor,
                &points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_sparse_update_policy(8, 1.0)
                    .with_full_frame_update_policy(0.45),
                &device,
            )
            .expect("full-frame update");

        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::FullFrame),
            "dense coverage should not take the sparse-rect path just because sparse thresholds are permissive"
        );
        assert_eq!(output.update_ratio(), 1.0);
        assert_eq!(tensor_to_rgba_bytes(output.output_rgba), current);
    }

    #[test]
    fn tensor_full_frame_policy_still_applies_masked_interframe_updates() {
        let device = Default::default();
        let width = 8;
        let height = 4;
        let previous = deterministic_rgba(width, height, 41);
        let current = deterministic_rgba(width, height, 43);
        let point = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2);
        let mut state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );

        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        state
            .visualize_normalized_rgb_clip_panels(
                previous_tensor,
                &[],
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38),
                &device,
            )
            .expect("keyframe");

        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        let output = state
            .visualize_normalized_rgb_clip_panels(
                current_tensor,
                &[point],
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
                &device,
            )
            .expect("masked full-frame update");

        assert_eq!(
            state.last_interframe_path(),
            Some(AutoGazeTensorInterframePath::FullFrame)
        );
        assert_eq!(output.mask_ratio(), 0.5);
        assert_eq!(output.update_ratio(), 0.5);
        let output = tensor_to_rgba_bytes(output.output_rgba);
        assert_eq!(&output[0..(width / 2 * 4)], &current[0..(width / 2 * 4)]);
        assert_eq!(
            &output[(width / 2 * 4)..(width * 4)],
            &previous[(width / 2 * 4)..(width * 4)]
        );
    }

    #[test]
    fn rgba_interframe_uses_full_frame_update_policy_for_dense_motion() {
        let width = 8;
        let height = 4;
        let previous = deterministic_rgba(width, height, 41);
        let current = deterministic_rgba(width, height, 43);
        let points = [FixationPoint::with_grid_extent(0.5, 0.5, 1.0, 1.0, 1.0, 2)];
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);

        state
            .visualize_rgba_panels_with_options(
                &previous,
                &[],
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
            )
            .expect("keyframe");
        let output = state
            .visualize_rgba_panels_with_options(
                &current,
                &points,
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
            )
            .expect("full-frame update");

        assert!(!state.last_frame_was_keyframe());
        assert_eq!(output.mask_pixel_count, width * height);
        assert_eq!(output.updated_pixel_count, width * height);
        assert_eq!(output.blend_rgba, current);
    }

    #[test]
    fn rgba_full_frame_policy_still_applies_masked_interframe_updates() {
        let width = 8;
        let height = 4;
        let previous = deterministic_rgba(width, height, 47);
        let current = deterministic_rgba(width, height, 53);
        let point = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);

        state
            .visualize_rgba_panels_with_options(
                &previous,
                &[],
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
            )
            .expect("keyframe");
        let output = state
            .visualize_rgba_panels_with_options(
                &current,
                &[point],
                AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.45),
            )
            .expect("masked full-frame update");

        assert!(!state.last_frame_was_keyframe());
        assert_eq!(output.mask_ratio(), 0.5);
        assert_eq!(output.update_ratio(), 0.5);
        assert_eq!(
            &output.blend_rgba[0..(width / 2 * 4)],
            &current[0..(width / 2 * 4)]
        );
        assert_eq!(
            &output.blend_rgba[(width / 2 * 4)..(width * 4)],
            &previous[(width / 2 * 4)..(width * 4)]
        );
    }

    #[test]
    fn effective_sparse_update_plan_projects_coarse_tokens_to_finest_active_grid() {
        let coarse = FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2);
        let fine = FixationPoint::with_grid_extent(0.875, 0.875, 0.25, 0.25, 1.0, 4);
        let native =
            fixation_sparse_update_plan(16, 16, &[coarse, fine], 1.0).expect("native update plan");
        let effective = fixation_effective_sparse_update_plan(16, 16, &[coarse, fine], 1.0)
            .expect("effective update plan");

        assert!(
            native.pixel_count > effective.pixel_count,
            "native coarse cells should cover more pixels than finest-grid projected cells"
        );
        assert_eq!(effective.rects.len(), 2);
    }

    #[test]
    fn effective_sparse_update_plan_uses_global_anyres_grid_extents() {
        let width = 1920;
        let height = 1080;
        let coarse =
            FixationPoint::with_grid_extent(0.5 / 18.0, 0.5 / 10.0, 1.0 / 18.0, 1.0 / 10.0, 1.0, 2);
        let fine = FixationPoint::with_grid_extent(
            125.5 / 126.0,
            69.5 / 70.0,
            1.0 / 126.0,
            1.0 / 70.0,
            1.0,
            14,
        );

        let plan = fixation_effective_sparse_update_plan(width, height, &[coarse, fine], 1.0)
            .expect("effective AnyRes update plan");

        assert_eq!(plan.rects.len(), 2);
        for rect in &plan.rects {
            assert!(
                rect.x1 - rect.x0 <= 16,
                "AnyRes effective update rect should use the global 126-column grid, got {rect:?}"
            );
            assert!(
                rect.y1 - rect.y0 <= 16,
                "AnyRes effective update rect should use the global 70-row grid, got {rect:?}"
            );
        }
        assert!(
            plan.pixel_count <= 512,
            "AnyRes effective output footprint should not fall back to the local 14x14 grid"
        );
    }

    #[test]
    fn interframe_update_ratio_deduplicates_overlapping_multiscale_cells() {
        let coarse = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let fine = FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 4);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 10);

        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        state
            .visualize_rgba(&first, 2, 1, &[], 1.0, 1.0)
            .expect("initial keyframe");

        let second = [30, 0, 0, 255, 40, 0, 0, 255];
        let visualization = state
            .visualize_rgba(&second, 2, 1, &[coarse, fine], 1.0, 1.0)
            .expect("interframe update");

        assert_eq!(
            &visualization.blend_rgba[0..8],
            &[30, 0, 0, 255, 20, 0, 0, 255]
        );
        assert_eq!(visualization.mask_pixel_count, 1);
        assert_eq!(visualization.updated_pixel_count, 1);
        assert_eq!(visualization.update_ratio(), 0.5);
    }

    #[test]
    fn full_blend_opacity_controls_white_overlay_strength() {
        let point = FixationPoint::with_extent(0.5, 0.5, 1.0, 1.0, 1.0);
        let rgba = [100, 50, 0, 200];
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 10);

        let transparent = state
            .visualize_rgba(&rgba, 1, 1, &[point], 1.0, 0.0)
            .expect("transparent visualization");
        assert_eq!(&transparent.blend_rgba[0..4], &rgba);

        let subtle = state
            .visualize_rgba(&rgba, 1, 1, &[point], 1.0, 0.25)
            .expect("subtle visualization");
        assert_eq!(&subtle.blend_rgba[0..4], &[139, 101, 64, 200]);
    }

    #[test]
    fn interframe_updates_copy_source_pixels_without_alpha_overlay() {
        let point = FixationPoint::with_extent(0.5, 0.5, 1.0, 1.0, 1.0);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 10);

        let first = [10, 10, 10, 255];
        state
            .visualize_rgba(&first, 1, 1, &[], 1.0, 0.0)
            .expect("initial keyframe");

        let second = [100, 50, 0, 255];
        let visualization = state
            .visualize_rgba(&second, 1, 1, &[point], 1.0, 0.25)
            .expect("interframe update");

        assert_eq!(&visualization.blend_rgba[0..4], &second);
        assert_eq!(visualization.update_ratio(), 1.0);
    }

    #[test]
    fn tensor_visualization_matches_byte_visualization() {
        let device = Default::default();
        let width = 6;
        let height = 4;
        let previous = deterministic_rgba(width, height, 7);
        let current = deterministic_rgba(width, height, 19);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.75, 0.25, 0.25, 0.5, 1.0, 4),
        ];

        assert_tensor_visualization_matches_bytes(
            &current,
            width,
            height,
            &points,
            AutoGazeVisualizationMode::FullBlend,
            30,
            &device,
        );

        let mut byte_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
        let mut tensor_state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::Interframe,
            30,
        );
        let _ = byte_state
            .visualize_rgba(&previous, width, height, &points, 1.0, 0.38)
            .expect("prime byte interframe");
        let previous_tensor = rgba_clip_to_tensor::<TestBackend>(
            &previous,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("previous tensor");
        let _ = tensor_state
            .visualize_normalized_rgb_clip(
                previous_tensor,
                &points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.0),
                &device,
            )
            .expect("prime tensor interframe");
        let byte = byte_state
            .visualize_rgba(&current, width, height, &points, 1.0, 0.38)
            .expect("byte interframe");
        let current_tensor = rgba_clip_to_tensor::<TestBackend>(
            &current,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("current tensor");
        let tensor = tensor_state
            .visualize_normalized_rgb_clip(
                current_tensor,
                &points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.0),
                &device,
            )
            .expect("tensor interframe");
        assert_eq!(tensor.update_ratio(), byte.update_ratio());
        assert_eq!(
            tensor_to_rgba_bytes(tensor.side_by_side_rgba),
            byte.side_by_side_rgba
        );
    }

    #[test]
    fn tensor_image_overlay_mask_visualization_matches_byte_visualization() {
        let device = Default::default();
        let width = 4;
        let height = 2;
        let rgba = deterministic_rgba(width, height, 31);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.25, 0.25, 0.5, 1.0, 4),
        ];
        let options = AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageOverlay);
        let mut byte_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let byte = byte_state
            .visualize_rgba_panels_with_options(&rgba, &points, options)
            .expect("byte panels");
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("normalized tensor");
        let tensor_options = AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageOverlay);
        let mut tensor_state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::FullBlend,
            30,
        );
        let tensor = tensor_state
            .visualize_normalized_rgb_clip_panels(tensor, &points, tensor_options, &device)
            .expect("tensor panels");

        assert_eq!(tensor_to_rgba_bytes(tensor.mask_rgba), byte.mask_rgba);
        assert_eq!(tensor_to_rgba_bytes(tensor.output_rgba), byte.blend_rgba);
    }

    #[test]
    fn tensor_image_mask_only_visualization_matches_byte_visualization() {
        let device = Default::default();
        let width = 4;
        let height = 2;
        let rgba = deterministic_rgba(width, height, 37);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.5, 0.5, 1.0, 1.0, 2),
            FixationPoint::with_grid_extent(0.875, 0.25, 0.25, 0.5, 1.0, 4),
        ];
        let options = AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageMaskOnly);
        let mut byte_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let byte = byte_state
            .visualize_rgba_panels_with_options(&rgba, &points, options)
            .expect("byte panels");
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("normalized tensor");
        let tensor_options = AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageMaskOnly);
        let mut tensor_state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::FullBlend,
            30,
        );
        let tensor = tensor_state
            .visualize_normalized_rgb_clip_panels(tensor, &points, tensor_options, &device)
            .expect("tensor panels");

        assert_eq!(tensor_to_rgba_bytes(tensor.mask_rgba), byte.mask_rgba);
        assert_eq!(tensor_to_rgba_bytes(tensor.output_rgba), byte.blend_rgba);
    }

    #[test]
    fn tensor_deduplicated_mask_geometry_matches_byte_visualization() {
        let device = Default::default();
        let width = 8;
        let height = 8;
        let rgba = deterministic_rgba(width, height, 41);
        let points = [
            FixationPoint::with_grid_extent(0.25, 0.25, 0.5, 0.5, 1.0, 2),
            FixationPoint::with_grid_extent(0.125, 0.125, 0.25, 0.25, 1.0, 4),
        ];
        let options = AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_geometry_mode(AutoGazeMaskGeometryMode::Deduplicated);
        let mut byte_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
        let byte = byte_state
            .visualize_rgba_panels_with_options(&rgba, &points, options)
            .expect("byte panels");
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            &rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            &device,
        )
        .expect("normalized tensor");
        let tensor_options = AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.4)
            .with_mask_geometry_mode(AutoGazeMaskGeometryMode::Deduplicated);
        let mut tensor_state = AutoGazeTensorVisualizationState::<TestBackend>::new(
            AutoGazeVisualizationMode::FullBlend,
            30,
        );
        let tensor = tensor_state
            .visualize_normalized_rgb_clip_panels(tensor, &points, tensor_options, &device)
            .expect("tensor panels");

        assert_eq!(tensor_to_rgba_bytes(tensor.mask_rgba), byte.mask_rgba);
        assert_eq!(tensor_to_rgba_bytes(tensor.output_rgba), byte.blend_rgba);
        assert_eq!(tensor.mask_pixel_count, byte.mask_pixel_count);
    }

    #[test]
    fn dense_mask_update_policy_matches_sparse_updates_for_fragmented_coverage() {
        let width = 64;
        let height = 64;
        let previous = deterministic_rgba(width, height, 59);
        let current = deterministic_rgba(width, height, 61);
        let points = checkerboard_grid_points(16);
        let parts = fixation_sparse_update_plan_parts_for_geometry(
            width,
            height,
            &points,
            1.0,
            AutoGazeMaskGeometryMode::Native,
        )
        .expect("fragmented plan");
        assert!(parts.stats.update_ratio(width, height) >= 0.45);
        assert!(parts.stats.row_span_count > height * 2);
        assert!(should_use_dense_mask_rgba_update(
            width,
            height,
            parts.stats,
            parts.stats.row_span_count,
            AutoGazeMaskVisualizationMode::ImageMaskOnly,
            0.45,
        ));

        let options = AutoGazeRgbaVisualizationOptions::new(width, height, 1.0, 0.38)
            .with_mask_visualization_mode(AutoGazeMaskVisualizationMode::ImageMaskOnly);
        let mut sparse_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
        sparse_state
            .visualize_rgba_panels_with_options(
                &previous,
                &[],
                options.with_full_frame_update_policy(0.0),
            )
            .expect("prime sparse state");
        let sparse = sparse_state
            .visualize_rgba_panels_with_options(
                &current,
                &points,
                options.with_full_frame_update_policy(0.0),
            )
            .expect("sparse interframe output");

        let mut dense_state =
            AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
        dense_state
            .visualize_rgba_panels_with_options(
                &previous,
                &[],
                options.with_full_frame_update_policy(0.45),
            )
            .expect("prime dense state");
        let dense = dense_state
            .visualize_rgba_panels_with_options(
                &current,
                &points,
                options.with_full_frame_update_policy(0.45),
            )
            .expect("dense interframe output");

        assert_eq!(dense.mask_pixel_count, sparse.mask_pixel_count);
        assert_eq!(dense.updated_pixel_count, sparse.updated_pixel_count);
        assert_eq!(dense.blend_rgba, sparse.blend_rgba);
        assert_eq!(dense.mask_rgba, sparse.mask_rgba);
    }

    #[test]
    fn psnr_is_infinite_for_identical_rgba_buffers() {
        let rgba = [10, 20, 30, 255, 40, 50, 60, 255];

        assert!(rgba_psnr_db(&rgba, &rgba).expect("psnr").is_infinite());
    }

    #[test]
    fn psnr_uses_rgb_channels() {
        let reference = [10, 20, 30, 0];
        let candidate = [20, 20, 30, 255];
        let psnr = rgba_psnr_db(&reference, &candidate).expect("psnr");
        let expected = 10.0f64 * ((255.0f64 * 255.0f64) / (100.0f64 / 3.0f64)).log10();

        assert!((psnr - expected).abs() < 1.0e-12);
    }

    fn assert_tensor_visualization_matches_bytes(
        rgba: &[u8],
        width: usize,
        height: usize,
        points: &[FixationPoint],
        mode: AutoGazeVisualizationMode,
        keyframe_duration: usize,
        device: &burn::backend::ndarray::NdArrayDevice,
    ) {
        let mut byte_state = AutoGazeVisualizationState::new(mode, keyframe_duration);
        let byte = byte_state
            .visualize_rgba(rgba, width, height, points, 1.0, 0.38)
            .expect("byte visualization");
        let tensor = rgba_clip_to_tensor::<TestBackend>(
            rgba,
            AutoGazeRgbaClipShape::new(1, height, width),
            device,
        )
        .expect("normalized tensor");
        let mut tensor_state =
            AutoGazeTensorVisualizationState::<TestBackend>::new(mode, keyframe_duration);
        let tensor = tensor_state
            .visualize_normalized_rgb_clip(
                tensor,
                points,
                AutoGazeTensorVisualizationOptions::new(width, height, 1.0, 0.38)
                    .with_full_frame_update_policy(0.0),
                device,
            )
            .expect("tensor visualization");
        assert_eq!(tensor.update_ratio(), byte.update_ratio());
        assert_eq!(
            tensor_to_rgba_bytes(tensor.side_by_side_rgba),
            byte.side_by_side_rgba
        );
    }

    fn tensor_to_rgba_bytes(tensor: Tensor<TestBackend, 3>) -> Vec<u8> {
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("tensor data")
            .into_iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    fn deterministic_rgba(width: usize, height: usize, seed: usize) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                rgba.push(((x * 31 + seed) % 256) as u8);
                rgba.push(((y * 47 + seed * 3) % 256) as u8);
                rgba.push(((x * 11 + y * 13 + seed * 5) % 256) as u8);
                rgba.push(255);
            }
        }
        rgba
    }

    fn dense_grid_points(grid: usize) -> Vec<FixationPoint> {
        let extent = 1.0 / grid as f32;
        (0..grid)
            .flat_map(|row| {
                (0..grid).map(move |col| {
                    FixationPoint::with_grid_extent(
                        (col as f32 + 0.5) * extent,
                        (row as f32 + 0.5) * extent,
                        extent,
                        extent,
                        1.0,
                        grid,
                    )
                })
            })
            .collect()
    }

    fn checkerboard_grid_points(grid: usize) -> Vec<FixationPoint> {
        let extent = 1.0 / grid as f32;
        (0..grid)
            .flat_map(|row| {
                (0..grid)
                    .filter(move |col| (row + col) % 2 == 0)
                    .map(move |col| {
                        FixationPoint::with_grid_extent(
                            (col as f32 + 0.5) * extent,
                            (row as f32 + 0.5) * extent,
                            extent,
                            extent,
                            1.0,
                            grid,
                        )
                    })
            })
            .collect()
    }
}
