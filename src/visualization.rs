use crate::FixationPoint;
use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct AutoGazeVisualization {
    pub width: usize,
    pub height: usize,
    pub side_by_side_width: usize,
    pub mask_rgba: Vec<u8>,
    pub blend_rgba: Vec<u8>,
    pub side_by_side_rgba: Vec<u8>,
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

    let alpha = fixation_alpha_mask(width, height, points, cell_scale);
    let mut mask_rgba = vec![0u8; pixels * 4];
    let mut blend_rgba = vec![0u8; pixels * 4];
    let side_by_side_width = width * 3;
    let mut side_by_side_rgba = vec![0u8; side_by_side_width * height * 4];
    let blend_alpha = blend_alpha.clamp(0.0, 1.0);

    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let src = pixel * 4;
            let mask = alpha[pixel];
            mask_rgba[src] = mask;
            mask_rgba[src + 1] = mask;
            mask_rgba[src + 2] = mask;
            mask_rgba[src + 3] = 255;

            let overlay = if mask > 0 { blend_alpha } else { 0.0 };
            for channel in 0..3 {
                let base = rgba[src + channel] as f32;
                blend_rgba[src + channel] =
                    (base * (1.0 - overlay) + 255.0 * overlay).round() as u8;
            }
            blend_rgba[src + 3] = rgba[src + 3];

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
    })
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
    }
}
