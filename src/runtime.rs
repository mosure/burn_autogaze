use crate::pipeline::AutoGazeInferenceMode;

/// Wrapper sentinel for keeping the model's configured generation budget.
pub const DEFAULT_MODEL_GENERATION_BUDGET: usize = 0;
/// Default number of displayed trace slots for realtime resize-mode streams.
pub const DEFAULT_REALTIME_TOP_K: usize = 10;
/// Default displayed trace slots per tile for tiled inspection modes.
pub const DEFAULT_TILED_TOP_K: usize = 2;
/// Default per-frame generated-token cap for tiled inspection modes.
pub const DEFAULT_TILED_MAX_GAZE_TOKENS: usize = 24;
/// Default number of frames kept in realtime input windows.
pub const DEFAULT_REALTIME_FRAMES_PER_CLIP: usize = 2;
/// Default maximum number of realtime AutoGaze inference tasks in flight.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 1;
/// Default number of frames kept in tiled input windows.
pub const DEFAULT_TILED_FRAMES_PER_CLIP: usize = 2;
/// Default tile batch size for tiled AutoGaze embedding and tracing.
pub const DEFAULT_TILED_TILE_BATCH_SIZE: usize = 64;
/// Default keyframe interval for interframe visualization.
pub const DEFAULT_KEYFRAME_DURATION: usize = 30;
/// Default alpha for readable white-mask overlays.
pub const DEFAULT_BLEND_ALPHA: f32 = 0.38;

/// Monotonic sequence gate for asynchronous AutoGaze inference results.
///
/// Frontends should call `reserve` before launching a model task and call
/// `accept` when that task completes. Older completions are rejected so camera
/// streams can drop stale frames instead of displaying results out of order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AutoGazeInferenceSequencer {
    next_sequence: u64,
    latest_applied_sequence: u64,
}

impl AutoGazeInferenceSequencer {
    pub fn reserve(&mut self) -> u64 {
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.next_sequence
    }

    pub fn accept(&mut self, sequence: u64) -> bool {
        if self.is_stale(sequence) {
            return false;
        }
        self.latest_applied_sequence = sequence;
        true
    }

    pub const fn is_stale(&self, sequence: u64) -> bool {
        sequence <= self.latest_applied_sequence
    }

    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub const fn latest_applied_sequence(&self) -> u64 {
        self.latest_applied_sequence
    }
}

/// Frontend admission policy for realtime AutoGaze streams.
///
/// The default keeps one model task in flight. New camera frames can still be
/// buffered and displayed while the model is busy, but processed mask results
/// should be sequence-gated so an older decode cannot overwrite a newer mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeRealtimePolicy {
    max_in_flight: usize,
}

impl Default for AutoGazeRealtimePolicy {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_IN_FLIGHT)
    }
}

impl AutoGazeRealtimePolicy {
    pub const fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight: if max_in_flight == 0 {
                DEFAULT_MAX_IN_FLIGHT
            } else {
                max_in_flight
            },
        }
    }

    pub const fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    pub const fn inference_busy(&self, in_flight: usize) -> bool {
        in_flight >= self.max_in_flight
    }

    pub const fn should_start_inference(&self, in_flight: usize) -> bool {
        !self.inference_busy(in_flight)
    }

    pub const fn should_draw_live_preview(&self, model_ready: bool) -> bool {
        !model_ready
    }

    pub const fn should_draw_async_stream_preview(
        &self,
        _model_ready: bool,
        _in_flight: usize,
    ) -> bool {
        true
    }
}

pub const fn should_use_streaming_cache(
    enabled: bool,
    frames_per_clip: usize,
    mode: AutoGazeInferenceMode,
) -> bool {
    enabled && frames_per_clip > 1 && matches!(mode, AutoGazeInferenceMode::ResizeToModelInput)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_sequencer_rejects_stale_results() {
        let mut sequencer = AutoGazeInferenceSequencer::default();
        let first = sequencer.reserve();
        let second = sequencer.reserve();

        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert!(sequencer.accept(second));
        assert!(!sequencer.accept(first));
        assert!(sequencer.is_stale(second));
        assert!(sequencer.accept(second + 1));
    }

    #[test]
    fn realtime_policy_caps_in_flight_work_and_preserves_processed_output() {
        let policy = AutoGazeRealtimePolicy::new(0);
        assert_eq!(policy.max_in_flight(), 1);
        assert!(policy.should_start_inference(0));
        assert!(!policy.should_start_inference(1));
        assert!(policy.inference_busy(1));

        assert!(policy.should_draw_live_preview(false));
        assert!(!policy.should_draw_live_preview(true));
        assert!(policy.should_draw_async_stream_preview(false, 0));
        assert!(policy.should_draw_async_stream_preview(true, 0));
        assert!(policy.should_draw_async_stream_preview(true, 1));
    }

    #[test]
    fn streaming_cache_is_realtime_only_and_requires_context() {
        assert!(should_use_streaming_cache(
            true,
            16,
            AutoGazeInferenceMode::ResizeToModelInput
        ));
        assert!(!should_use_streaming_cache(
            true,
            16,
            AutoGazeInferenceMode::TiledResizeToGrid { tile_size: 224 }
        ));
        assert!(!should_use_streaming_cache(
            true,
            1,
            AutoGazeInferenceMode::ResizeToModelInput
        ));
        assert!(!should_use_streaming_cache(
            false,
            16,
            AutoGazeInferenceMode::ResizeToModelInput
        ));
    }

    #[test]
    fn shared_runtime_defaults_are_sane_for_realtime_wrappers() {
        assert_eq!(DEFAULT_MODEL_GENERATION_BUDGET, 0);
        assert_eq!(DEFAULT_REALTIME_TOP_K, 10);
        assert_eq!(DEFAULT_TILED_TOP_K, 2);
        assert_eq!(DEFAULT_TILED_MAX_GAZE_TOKENS, 24);
        assert_eq!(DEFAULT_REALTIME_FRAMES_PER_CLIP, 2);
        assert_eq!(DEFAULT_MAX_IN_FLIGHT, 1);
        assert_eq!(DEFAULT_TILED_FRAMES_PER_CLIP, 2);
        assert_eq!(DEFAULT_TILED_TILE_BATCH_SIZE, 64);
        assert_eq!(DEFAULT_KEYFRAME_DURATION, 30);
        assert!((DEFAULT_BLEND_ALPHA - 0.38).abs() < f32::EPSILON);
    }
}
