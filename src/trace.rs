use serde::{Deserialize, Serialize};

const MIN_CELL_EXTENT: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixationPoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub confidence: f32,
    #[serde(default)]
    pub width: f32,
    #[serde(default)]
    pub height: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub grid: usize,
}

impl FixationPoint {
    pub fn new(x: f32, y: f32, scale: f32, confidence: f32) -> Self {
        Self::with_extent(x, y, scale, scale, confidence)
    }

    pub fn with_extent(x: f32, y: f32, width: f32, height: f32, confidence: f32) -> Self {
        let width = width.clamp(MIN_CELL_EXTENT, 1.0);
        let height = height.clamp(MIN_CELL_EXTENT, 1.0);
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            scale: width.max(height),
            confidence: confidence.clamp(0.0, 1.0),
            width,
            height,
            grid: 0,
        }
    }

    pub fn with_grid_extent(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        confidence: f32,
        grid: usize,
    ) -> Self {
        let mut point = Self::with_extent(x, y, width, height, confidence);
        point.grid = grid;
        point
    }

    pub fn cell_grid(&self) -> Option<usize> {
        (self.grid > 0).then_some(self.grid)
    }

    pub fn cell_width(&self) -> f32 {
        if self.width > 0.0 {
            self.width.clamp(MIN_CELL_EXTENT, 1.0)
        } else {
            self.scale.clamp(MIN_CELL_EXTENT, 1.0)
        }
    }

    pub fn cell_height(&self) -> f32 {
        if self.height > 0.0 {
            self.height.clamp(MIN_CELL_EXTENT, 1.0)
        } else {
            self.scale.clamp(MIN_CELL_EXTENT, 1.0)
        }
    }

    pub fn bounds(&self) -> FixationBounds {
        self.scaled_bounds(1.0)
    }

    pub fn scaled_bounds(&self, scale: f32) -> FixationBounds {
        let scale = scale.max(MIN_CELL_EXTENT);
        let half_width = (self.cell_width() * scale * 0.5).clamp(0.0, 0.5);
        let half_height = (self.cell_height() * scale * 0.5).clamp(0.0, 0.5);
        FixationBounds {
            x_min: (self.x - half_width).clamp(0.0, 1.0),
            y_min: (self.y - half_height).clamp(0.0, 1.0),
            x_max: (self.x + half_width).clamp(0.0, 1.0),
            y_max: (self.y + half_height).clamp(0.0, 1.0),
        }
    }
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixationBounds {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixationSet {
    pub points: Vec<FixationPoint>,
    pub stop_probability: f32,
}

impl FixationSet {
    pub fn new(mut points: Vec<FixationPoint>, stop_probability: f32, k: usize) -> Self {
        points.truncate(k.max(1));
        while points.len() < k.max(1) {
            points.push(FixationPoint::new(0.5, 0.5, 0.25, 0.0));
        }
        Self {
            points,
            stop_probability: stop_probability.clamp(0.0, 1.0),
        }
    }

    pub fn with_min_len(
        mut points: Vec<FixationPoint>,
        stop_probability: f32,
        min_len: usize,
    ) -> Self {
        while points.len() < min_len.max(1) {
            points.push(FixationPoint::new(0.5, 0.5, 0.25, 0.0));
        }
        Self {
            points,
            stop_probability: stop_probability.clamp(0.0, 1.0),
        }
    }

    pub fn top_point(&self) -> FixationPoint {
        self.points
            .first()
            .copied()
            .unwrap_or_else(|| FixationPoint::new(0.5, 0.5, 0.25, 0.0))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameFixationTrace {
    pub frames: Vec<FixationSet>,
}

impl FrameFixationTrace {
    pub fn new(frames: Vec<FixationSet>) -> Self {
        Self { frames }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
