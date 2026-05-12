/// Default exponential moving average coefficient used by AutoGaze runtime metrics.
pub const DEFAULT_METRIC_EMA_ALPHA: f64 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoGazeEmaMetric {
    current: f64,
    ema: f64,
    initialized: bool,
    alpha: f64,
}

impl Default for AutoGazeEmaMetric {
    fn default() -> Self {
        Self::new(DEFAULT_METRIC_EMA_ALPHA)
    }
}

impl AutoGazeEmaMetric {
    pub const fn new(alpha: f64) -> Self {
        Self {
            current: 0.0,
            ema: 0.0,
            initialized: false,
            alpha,
        }
    }

    pub const fn current(&self) -> f64 {
        self.current
    }

    pub const fn ema(&self) -> f64 {
        self.ema
    }

    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn record(&mut self, current: f64) {
        self.current = current;
        self.ema = if self.initialized {
            ema_metric(self.ema, self.current, self.alpha)
        } else {
            self.initialized = true;
            self.current
        };
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutoGazeGazeRatioStats {
    metric: AutoGazeEmaMetric,
}

impl AutoGazeGazeRatioStats {
    pub fn record(&mut self, ratio: f64) {
        self.metric.record(sanitize_gaze_ratio(ratio));
    }

    pub const fn current(&self) -> f64 {
        self.metric.current()
    }

    pub const fn ema(&self) -> f64 {
        self.metric.ema()
    }

    pub const fn is_initialized(&self) -> bool {
        self.metric.is_initialized()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutoGazePsnrStats {
    metric: AutoGazeEmaMetric,
}

impl AutoGazePsnrStats {
    pub fn record(&mut self, psnr_db: f64) {
        if !psnr_db.is_finite() && !(psnr_db.is_infinite() && psnr_db.is_sign_positive()) {
            return;
        }
        self.metric.record(psnr_db);
    }

    pub const fn current(&self) -> f64 {
        self.metric.current()
    }

    pub const fn ema(&self) -> f64 {
        self.metric.ema()
    }

    pub const fn is_initialized(&self) -> bool {
        self.metric.is_initialized()
    }
}

pub fn ema_metric(previous: f64, current: f64, alpha: f64) -> f64 {
    if previous.is_finite() && current.is_finite() {
        previous * (1.0 - alpha.clamp(0.0, 1.0)) + current * alpha.clamp(0.0, 1.0)
    } else {
        current
    }
}

pub fn fps_from_millis(ms: f64) -> f64 {
    if ms > 0.0 { 1_000.0 / ms } else { 0.0 }
}

pub fn format_fps(value: f64) -> String {
    format_fixed_one_decimal(value, 999.9, "---.-")
}

pub fn sanitize_gaze_ratio(ratio: f64) -> f64 {
    if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn format_gaze_ratio_percent(value: f64) -> String {
    if value.is_finite() {
        format!("{:05.1}%", sanitize_gaze_ratio(value) * 100.0)
    } else {
        "---.-%".to_string()
    }
}

pub fn format_psnr_db(value: f64) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        "999.9".to_string()
    } else if value.is_finite() {
        format_fixed_one_decimal(value, 999.9, "---.-")
    } else {
        "---.-".to_string()
    }
}

fn format_fixed_one_decimal(value: f64, max: f64, invalid: &str) -> String {
    if value.is_finite() {
        format!("{:05.1}", value.clamp(0.0, max))
    } else {
        invalid.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaze_ratio_stats_clamp_samples_and_track_ema() {
        let mut stats = AutoGazeGazeRatioStats::default();

        stats.record(2.0);
        assert!(stats.is_initialized());
        assert_eq!(stats.current(), 1.0);
        assert_eq!(stats.ema(), 1.0);

        stats.record(-1.0);
        assert_eq!(stats.current(), 0.0);
        assert!((stats.ema() - (1.0 - DEFAULT_METRIC_EMA_ALPHA)).abs() < 1.0e-12);

        stats.record(f64::NAN);
        assert_eq!(stats.current(), 0.0);
        assert_eq!(format_gaze_ratio_percent(stats.current()), "000.0%");
        assert_eq!(format_gaze_ratio_percent(f64::NAN), "---.-%");
    }

    #[test]
    fn psnr_stats_and_format_handle_infinite_and_invalid_values_without_poisoning_ema() {
        let mut stats = AutoGazePsnrStats::default();

        stats.record(f64::INFINITY);
        assert!(stats.is_initialized());
        assert!(stats.current().is_infinite());
        assert_eq!(format_psnr_db(stats.current()), "999.9");

        stats.record(42.25);
        assert_eq!(stats.current(), 42.25);
        assert_eq!(stats.ema(), 42.25);
        assert_eq!(format_psnr_db(stats.current()), "042.2");

        stats.record(f64::NAN);
        assert_eq!(stats.current(), 42.25);
        assert_eq!(stats.ema(), 42.25);
        assert_eq!(format_psnr_db(f64::NAN), "---.-");
    }

    #[test]
    fn fps_from_millis_handles_zero_and_positive_values() {
        assert_eq!(fps_from_millis(0.0), 0.0);
        assert_eq!(fps_from_millis(20.0), 50.0);
    }

    #[test]
    fn display_metric_formatters_keep_fixed_width() {
        for value in [0.0, 7.5, 51.25, 999.9, 1000.0, f64::NAN] {
            assert_eq!(format_fps(value).len(), 5);
            assert_eq!(format_gaze_ratio_percent(value / 100.0).len(), 6);
        }
        for value in [0.0, 7.5, 42.25, 100.0, 1000.0, f64::INFINITY, f64::NAN] {
            assert_eq!(format_psnr_db(value).len(), 5);
        }
    }
}
