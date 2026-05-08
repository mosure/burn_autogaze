use crate::FixationPoint;
use anyhow::{Result, ensure};
use std::{fmt, str::FromStr};

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

#[derive(Clone, Debug)]
pub struct AutoGazeVisualizationState {
    mode: AutoGazeVisualizationMode,
    keyframe_duration: usize,
    frame_index: usize,
    interframe_output_rgba: Vec<u8>,
    interframe_width: usize,
    interframe_height: usize,
}

impl Default for AutoGazeVisualizationState {
    fn default() -> Self {
        Self::new(AutoGazeVisualizationMode::FullBlend, 30)
    }
}

impl AutoGazeVisualizationState {
    pub fn new(mode: AutoGazeVisualizationMode, keyframe_duration: usize) -> Self {
        Self {
            mode,
            keyframe_duration: keyframe_duration.max(1),
            frame_index: 0,
            interframe_output_rgba: Vec::new(),
            interframe_width: 0,
            interframe_height: 0,
        }
    }

    pub fn mode(&self) -> AutoGazeVisualizationMode {
        self.mode
    }

    pub fn keyframe_duration(&self) -> usize {
        self.keyframe_duration
    }

    pub fn configure(&mut self, mode: AutoGazeVisualizationMode, keyframe_duration: usize) {
        if self.mode != mode {
            self.reset();
        }
        self.mode = mode;
        self.keyframe_duration = keyframe_duration.max(1);
    }

    pub fn reset(&mut self) {
        self.frame_index = 0;
        self.interframe_output_rgba.clear();
        self.interframe_width = 0;
        self.interframe_height = 0;
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
        let (mask_rgba, full_blend_rgba, mask_pixel_count) =
            mask_and_blend_rgba(rgba, width, height, points, cell_scale, blend_alpha)?;
        let pixels = validate_rgba_dimensions(rgba, width, height)?;
        let (output_rgba, updated_pixel_count) = match self.mode {
            AutoGazeVisualizationMode::FullBlend => (full_blend_rgba.clone(), pixels),
            AutoGazeVisualizationMode::Interframe => {
                self.interframe_rgba(rgba, width, height, &mask_rgba, &full_blend_rgba)?
            }
        };
        self.frame_index = self.frame_index.saturating_add(1);
        build_visualization(
            rgba,
            width,
            height,
            mask_rgba,
            output_rgba,
            mask_pixel_count,
            updated_pixel_count,
        )
    }

    fn interframe_rgba(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        mask_rgba: &[u8],
        full_blend_rgba: &[u8],
    ) -> Result<(Vec<u8>, usize)> {
        let pixels = validate_rgba_dimensions(rgba, width, height)?;
        let dimensions_changed = self.interframe_width != width || self.interframe_height != height;
        let keyframe = dimensions_changed
            || self.interframe_output_rgba.len() != pixels * 4
            || self.frame_index == 0
            || self.frame_index.is_multiple_of(self.keyframe_duration);
        let mut updated_pixel_count = if keyframe { pixels } else { 0 };

        if keyframe {
            self.interframe_output_rgba.clear();
            self.interframe_output_rgba.extend_from_slice(rgba);
            self.interframe_width = width;
            self.interframe_height = height;
        }

        for pixel in 0..pixels {
            let offset = pixel * 4;
            if mask_rgba[offset] > 0 {
                self.interframe_output_rgba[offset..offset + 4]
                    .copy_from_slice(&full_blend_rgba[offset..offset + 4]);
                if !keyframe {
                    updated_pixel_count += 1;
                }
            }
        }

        Ok((self.interframe_output_rgba.clone(), updated_pixel_count))
    }
}

pub fn fixation_alpha_mask(
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let mut alpha = vec![0u8; width * height];

    for point in points {
        if point.confidence <= 0.0 {
            continue;
        }

        let bounds = point.scaled_bounds(cell_scale);
        let (x0, x1) = pixel_range(bounds.x_min, bounds.x_max, width);
        let (y0, y1) = pixel_range(bounds.y_min, bounds.y_max, height);
        for y in y0..y1 {
            let row = y * width;
            for x in x0..x1 {
                alpha[row + x] = 255;
            }
        }
    }

    alpha
}

pub fn visualize_fixations_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<AutoGazeVisualization> {
    let (mask_rgba, blend_rgba, mask_pixel_count) =
        mask_and_blend_rgba(rgba, width, height, points, cell_scale, blend_alpha)?;
    let pixels = validate_rgba_dimensions(rgba, width, height)?;
    build_visualization(
        rgba,
        width,
        height,
        mask_rgba,
        blend_rgba,
        mask_pixel_count,
        pixels,
    )
}

fn validate_rgba_dimensions(rgba: &[u8], width: usize, height: usize) -> Result<usize> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("visualization dimensions overflow"))?;
    ensure!(
        width > 0 && height > 0,
        "visualization dimensions must be nonzero"
    );
    ensure!(
        rgba.len() == pixels * 4,
        "expected {} RGBA bytes for {}x{}, got {}",
        pixels * 4,
        width,
        height,
        rgba.len()
    );
    Ok(pixels)
}

fn mask_and_blend_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    points: &[FixationPoint],
    cell_scale: f32,
    blend_alpha: f32,
) -> Result<(Vec<u8>, Vec<u8>, usize)> {
    let pixels = validate_rgba_dimensions(rgba, width, height)?;
    let alpha = fixation_alpha_mask(width, height, points, cell_scale);
    let mut mask_rgba = vec![0u8; pixels * 4];
    let mut blend_rgba = vec![0u8; pixels * 4];
    let blend_alpha = blend_alpha.clamp(0.0, 1.0);
    let mut mask_pixel_count = 0usize;

    for (pixel, mask) in alpha.iter().copied().enumerate() {
        let src = pixel * 4;
        if mask > 0 {
            mask_pixel_count += 1;
        }
        mask_rgba[src] = mask;
        mask_rgba[src + 1] = mask;
        mask_rgba[src + 2] = mask;
        mask_rgba[src + 3] = 255;

        let overlay = if mask > 0 { blend_alpha } else { 0.0 };
        for channel in 0..3 {
            let base = rgba[src + channel] as f32;
            blend_rgba[src + channel] = (base * (1.0 - overlay) + 255.0 * overlay).round() as u8;
        }
        blend_rgba[src + 3] = rgba[src + 3];
    }

    Ok((mask_rgba, blend_rgba, mask_pixel_count))
}

fn build_visualization(
    rgba: &[u8],
    width: usize,
    height: usize,
    mask_rgba: Vec<u8>,
    blend_rgba: Vec<u8>,
    mask_pixel_count: usize,
    updated_pixel_count: usize,
) -> Result<AutoGazeVisualization> {
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
    let side_by_side_bytes = side_by_side_width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("side-by-side visualization byte length overflow"))?;
    let mut side_by_side_rgba = vec![0u8; side_by_side_bytes];

    for y in 0..height {
        for x in 0..width {
            let src = (y * width + x) * 4;
            write_side_by_side(&mut side_by_side_rgba, width, 0, x, y, &rgba[src..src + 4]);
            write_side_by_side(
                &mut side_by_side_rgba,
                width,
                1,
                x,
                y,
                &mask_rgba[src..src + 4],
            );
            write_side_by_side(
                &mut side_by_side_rgba,
                width,
                2,
                x,
                y,
                &blend_rgba[src..src + 4],
            );
        }
    }

    Ok(AutoGazeVisualization {
        width,
        height,
        side_by_side_width,
        mask_rgba,
        blend_rgba,
        side_by_side_rgba,
        mask_pixel_count,
        updated_pixel_count,
    })
}

fn ratio(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
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

fn write_side_by_side(
    out: &mut [u8],
    width: usize,
    column: usize,
    x: usize,
    y: usize,
    rgba: &[u8],
) {
    let out_width = width * 3;
    let out_x = column * width + x;
    let dst = (y * out_width + out_x) * 4;
    out[dst..dst + 4].copy_from_slice(rgba);
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn blends_selected_cells_with_white() {
        let rgba = [100, 50, 0, 255, 10, 20, 30, 255];
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let visualization =
            visualize_fixations_rgba(&rgba, 2, 1, &[point], 1.0, 0.5).expect("visualize");

        assert_eq!(&visualization.mask_rgba[0..4], &[255, 255, 255, 255]);
        assert_eq!(&visualization.mask_rgba[4..8], &[0, 0, 0, 255]);
        assert_eq!(&visualization.blend_rgba[0..4], &[178, 153, 128, 255]);
        assert_eq!(&visualization.blend_rgba[4..8], &[10, 20, 30, 255]);
        assert_eq!(visualization.mask_pixel_count, 1);
        assert_eq!(visualization.updated_pixel_count, 2);
        assert_eq!(visualization.mask_ratio(), 0.5);
        assert_eq!(visualization.update_ratio(), 1.0);
    }

    #[test]
    fn interframe_mode_preserves_unmasked_regions_until_keyframe() {
        let point = FixationPoint::with_extent(0.25, 0.5, 0.5, 1.0, 1.0);
        let mut state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 3);

        let first = [10, 0, 0, 255, 20, 0, 0, 255];
        let first_visualization = state
            .visualize_rgba(&first, 2, 1, &[point], 1.0, 1.0)
            .expect("first visualization");
        assert_eq!(&first_visualization.blend_rgba[0..4], &[255, 255, 255, 255]);
        assert_eq!(&first_visualization.blend_rgba[4..8], &[20, 0, 0, 255]);
        assert_eq!(first_visualization.mask_ratio(), 0.5);
        assert_eq!(first_visualization.update_ratio(), 1.0);

        let second = [30, 0, 0, 255, 40, 0, 0, 255];
        let second_visualization = state
            .visualize_rgba(&second, 2, 1, &[point], 1.0, 1.0)
            .expect("second visualization");
        assert_eq!(
            &second_visualization.blend_rgba[0..8],
            &[255, 255, 255, 255, 20, 0, 0, 255]
        );
        assert_eq!(second_visualization.mask_ratio(), 0.5);
        assert_eq!(second_visualization.update_ratio(), 0.5);

        let third = [50, 0, 0, 255, 60, 0, 0, 255];
        let third_visualization = state
            .visualize_rgba(&third, 2, 1, &[], 1.0, 1.0)
            .expect("third visualization");
        assert_eq!(
            &third_visualization.blend_rgba[0..8],
            &[255, 255, 255, 255, 20, 0, 0, 255]
        );
        assert_eq!(third_visualization.mask_ratio(), 0.0);
        assert_eq!(third_visualization.update_ratio(), 0.0);

        let fourth = [70, 0, 0, 255, 80, 0, 0, 255];
        let fourth_visualization = state
            .visualize_rgba(&fourth, 2, 1, &[], 1.0, 1.0)
            .expect("fourth visualization");
        assert_eq!(
            &fourth_visualization.blend_rgba[0..8],
            &[70, 0, 0, 255, 80, 0, 0, 255]
        );
        assert_eq!(fourth_visualization.mask_ratio(), 0.0);
        assert_eq!(fourth_visualization.update_ratio(), 1.0);
    }
}
