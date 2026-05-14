use crate::FixationPoint;
use crate::config::AutoGazeConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeScaleTokenLayout {
    pub token_count: usize,
    pub grid: usize,
}

#[derive(Clone, Debug)]
pub(super) struct GenerationCoverageTracker {
    scale_layouts: Vec<AutoGazeScaleTokenLayout>,
    grid: usize,
    covered: Vec<bool>,
    pub(super) covered_count: usize,
    stop_cells: usize,
}

impl GenerationCoverageTracker {
    pub(super) fn new(scale_layouts: &[AutoGazeScaleTokenLayout], stop_ratio: f64) -> Option<Self> {
        if !stop_ratio.is_finite() || stop_ratio <= 0.0 {
            return None;
        }

        let scale_layouts = normalize_scale_layouts(scale_layouts.to_vec(), 1);
        let grid = coverage_grid_for_layouts(&scale_layouts);
        let cells = grid.checked_mul(grid)?;
        let stop_cells =
            ((stop_ratio.clamp(0.0, 1.0) * cells as f64).ceil() as usize).clamp(1, cells);
        Some(Self {
            scale_layouts,
            grid,
            covered: vec![false; cells],
            covered_count: 0,
            stop_cells,
        })
    }

    pub(super) fn observe_token(&mut self, token: i64) -> bool {
        if token < 0 || self.covered_count >= self.stop_cells {
            return self.covered_count >= self.stop_cells;
        }

        let Some((scale_idx, local)) = scale_token_index(token as usize, &self.scale_layouts)
        else {
            return false;
        };
        let source_grid = self.scale_layouts[scale_idx].grid.max(1);
        let row = local / source_grid;
        let col = local % source_grid;
        let y0 = row.saturating_mul(self.grid) / source_grid;
        let x0 = col.saturating_mul(self.grid) / source_grid;
        let y1 = (row + 1).saturating_mul(self.grid).div_ceil(source_grid);
        let x1 = (col + 1).saturating_mul(self.grid).div_ceil(source_grid);

        for y in y0.min(self.grid)..y1.min(self.grid) {
            let row_offset = y * self.grid;
            for x in x0.min(self.grid)..x1.min(self.grid) {
                let idx = row_offset + x;
                if !self.covered[idx] {
                    self.covered[idx] = true;
                    self.covered_count += 1;
                }
            }
        }
        self.covered_count >= self.stop_cells
    }
}

pub(super) fn generation_coverage_trackers(
    batch: usize,
    stop_ratio: Option<f64>,
    scale_layouts: &[AutoGazeScaleTokenLayout],
) -> Option<Vec<GenerationCoverageTracker>> {
    let tracker = GenerationCoverageTracker::new(scale_layouts, stop_ratio?)?;
    Some(vec![tracker; batch])
}

pub(super) fn observe_generation_coverage(
    trackers: &mut Option<Vec<GenerationCoverageTracker>>,
    batch_idx: usize,
    token: i64,
) -> bool {
    trackers
        .as_mut()
        .and_then(|trackers| trackers.get_mut(batch_idx))
        .map(|tracker| tracker.observe_token(token))
        .unwrap_or(false)
}

pub(super) fn effective_generation_max_tokens(
    configured_max_tokens: usize,
    coverage_stop_ratio: Option<f64>,
    scale_layouts: &[AutoGazeScaleTokenLayout],
    num_multi_token_pred: usize,
) -> usize {
    let configured_max_tokens = configured_max_tokens.max(1);
    let Some(stop_ratio) = coverage_stop_ratio else {
        return configured_max_tokens;
    };
    if !stop_ratio.is_finite() || stop_ratio <= 0.0 || stop_ratio >= 1.0 {
        return configured_max_tokens;
    }

    let Some(finest_grid) = scale_layouts
        .iter()
        .filter(|layout| layout.token_count > 0)
        .map(|layout| layout.grid.max(1))
        .max()
    else {
        return configured_max_tokens;
    };
    let finest_cells = finest_grid.saturating_mul(finest_grid).max(1);
    let required_tokens = (stop_ratio.clamp(0.0, 1.0) * finest_cells as f64).ceil() as usize;
    let chunk = num_multi_token_pred.max(1);
    let chunk_aligned = required_tokens.max(1).div_ceil(chunk).saturating_mul(chunk);
    configured_max_tokens.min(chunk_aligned.max(1))
}

pub(super) fn normalize_scale_layouts(
    mut layouts: Vec<AutoGazeScaleTokenLayout>,
    fallback_token_count: usize,
) -> Vec<AutoGazeScaleTokenLayout> {
    layouts.retain(|layout| layout.token_count > 0);
    if layouts.is_empty() {
        let token_count = fallback_token_count.max(1);
        return vec![AutoGazeScaleTokenLayout {
            token_count,
            grid: square_grid(token_count),
        }];
    }

    for layout in &mut layouts {
        layout.grid = layout.grid.max(1);
    }
    layouts
}

fn coverage_grid_for_layouts(layouts: &[AutoGazeScaleTokenLayout]) -> usize {
    const MAX_COVERAGE_GRID: usize = 256;
    let max_grid = layouts.iter().map(|layout| layout.grid).max().unwrap_or(1);
    let mut grid = 1usize;
    for layout in layouts {
        let Some(next) = bounded_lcm(grid, layout.grid.max(1), MAX_COVERAGE_GRID) else {
            return max_grid.clamp(1, MAX_COVERAGE_GRID);
        };
        grid = next;
    }
    grid.max(max_grid.max(1)).min(MAX_COVERAGE_GRID)
}

fn bounded_lcm(left: usize, right: usize, max_value: usize) -> Option<usize> {
    let gcd = gcd(left.max(1), right.max(1));
    left.checked_div(gcd)?
        .checked_mul(right.max(1))
        .filter(|value| *value <= max_value)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

pub(super) fn scale_token_index(
    token: usize,
    scale_layouts: &[AutoGazeScaleTokenLayout],
) -> Option<(usize, usize)> {
    let mut offset = 0usize;
    for (scale_idx, layout) in scale_layouts.iter().enumerate() {
        if token < offset + layout.token_count {
            return Some((scale_idx, token - offset));
        }
        offset += layout.token_count;
    }
    None
}

pub(super) fn token_to_fixation_point(
    token: usize,
    scale_layouts: &[AutoGazeScaleTokenLayout],
    confidence: f32,
) -> Option<FixationPoint> {
    let (scale_idx, local) = scale_token_index(token, scale_layouts)?;
    let grid = scale_layouts[scale_idx].grid.max(1);
    let row = local / grid;
    let col = local % grid;
    let x = (col as f32 + 0.5) / grid as f32;
    let y = (row as f32 + 0.5) / grid as f32;
    let cell = (1.0 / grid as f32).clamp(1.0e-6, 1.0);
    Some(FixationPoint::with_grid_extent(
        x, y, cell, cell, confidence, grid,
    ))
}

pub fn scale_token_layouts(config: &AutoGazeConfig) -> Vec<AutoGazeScaleTokenLayout> {
    let scales = config.scale_values();
    if scales.is_empty() {
        let token_count = config.num_vision_tokens_each_frame.max(1);
        return vec![AutoGazeScaleTokenLayout {
            token_count,
            grid: square_grid(token_count),
        }];
    }

    let patch_size = config
        .gaze_model_config
        .vision_model_config
        .kernel_size
        .max(1);
    let direct_layouts = scales
        .iter()
        .map(|scale| {
            let grid = (scale / patch_size).max(1);
            AutoGazeScaleTokenLayout {
                token_count: grid * grid,
                grid,
            }
        })
        .collect::<Vec<_>>();
    let direct_tokens = direct_layouts
        .iter()
        .map(|layout| layout.token_count)
        .sum::<usize>();
    if direct_tokens == config.num_vision_tokens_each_frame {
        return direct_layouts;
    }

    let sum_sq: usize = scales.iter().map(|scale| scale * scale).sum();
    let mut counts = Vec::with_capacity(scales.len());
    let mut assigned = 0usize;
    for (index, scale) in scales.iter().copied().enumerate() {
        if index + 1 == scales.len() {
            counts.push(config.num_vision_tokens_each_frame.saturating_sub(assigned));
        } else {
            let count = ((scale * scale) as f64 / sum_sq.max(1) as f64
                * config.num_vision_tokens_each_frame as f64)
                .floor() as usize;
            counts.push(count);
            assigned += count;
        }
    }
    counts
        .into_iter()
        .map(|token_count| AutoGazeScaleTokenLayout {
            token_count,
            grid: square_grid(token_count),
        })
        .collect()
}

pub(super) fn square_grid(token_count: usize) -> usize {
    (token_count.max(1) as f64).sqrt().round().max(1.0) as usize
}
