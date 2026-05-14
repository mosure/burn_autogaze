use super::layout::{
    GenerationCoverageTracker, effective_generation_max_tokens, token_to_fixation_point,
};
use super::selection::{
    GreedySelectionContext, device_greedy_chunk_from_packed_data,
    greedy_select_multi_tokens_from_data, greedy_select_multi_tokens_from_packed_data,
};
use super::*;

#[cfg(feature = "ndarray")]
use burn::module::ModuleMapper;

#[test]
fn token_to_fixation_point_preserves_multiscale_cells() {
    let mut config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    config.gaze_model_config.num_vision_tokens_each_frame = 265;

    let scale_layouts = scale_token_layouts(&config);
    assert_eq!(
        scale_layouts,
        vec![
            AutoGazeScaleTokenLayout {
                token_count: 4,
                grid: 2
            },
            AutoGazeScaleTokenLayout {
                token_count: 16,
                grid: 4
            },
            AutoGazeScaleTokenLayout {
                token_count: 49,
                grid: 7
            },
            AutoGazeScaleTokenLayout {
                token_count: 196,
                grid: 14
            }
        ]
    );

    let coarse = token_to_fixation_point(0, &scale_layouts, 1.0).expect("coarse token");
    assert_eq!(coarse.x, 0.25);
    assert_eq!(coarse.y, 0.25);
    assert_eq!(coarse.cell_width(), 0.5);
    assert_eq!(coarse.cell_height(), 0.5);
    assert_eq!(coarse.cell_grid(), Some(2));

    let mid = token_to_fixation_point(4, &scale_layouts, 1.0).expect("second-scale token");
    assert_eq!(mid.x, 0.125);
    assert_eq!(mid.y, 0.125);
    assert_eq!(mid.cell_width(), 0.25);
    assert_eq!(mid.cell_height(), 0.25);
    assert_eq!(mid.cell_grid(), Some(4));

    let fine_offset = scale_layouts[..3]
        .iter()
        .map(|layout| layout.token_count)
        .sum::<usize>();
    let fine = token_to_fixation_point(fine_offset + 13, &scale_layouts, 1.0).expect("fine token");
    assert!((fine.x - 13.5 / 14.0).abs() < 1.0e-6);
    assert!((fine.y - 0.5 / 14.0).abs() < 1.0e-6);
    assert!((fine.cell_width() - 1.0 / 14.0).abs() < 1.0e-6);
    assert!((fine.cell_height() - 1.0 / 14.0).abs() < 1.0e-6);
    assert_eq!(fine.cell_grid(), Some(14));
}

#[test]
fn generation_coverage_tracker_stops_after_redundant_full_frame_cells() {
    let config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    let layouts = scale_token_layouts(&config);
    let mut tracker = GenerationCoverageTracker::new(&layouts, 1.0).expect("coverage tracker");

    assert!(!tracker.observe_token(0));
    assert!(!tracker.observe_token(1));
    assert!(!tracker.observe_token(2));
    assert!(
        tracker.observe_token(3),
        "the four 2x2 coarse tokens cover the full normalized frame"
    );
    let covered_after_full_frame = tracker.covered_count;
    assert!(tracker.observe_token(4));
    assert_eq!(
        tracker.covered_count, covered_after_full_frame,
        "finer tokens inside already-covered coarse cells should not add work"
    );
}

#[test]
fn coverage_stop_bounds_effective_generation_budget_to_finest_grid_need() {
    let config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    let layouts = scale_token_layouts(&config);

    assert_eq!(
        effective_generation_max_tokens(198, Some(0.45), &layouts, 10),
        90,
        "45% coverage cannot require more than ninety 14x14 fine cells, rounded to the ten-token decoder chunk"
    );
    assert_eq!(
        effective_generation_max_tokens(198, None, &layouts, 10),
        198
    );
    assert_eq!(
        effective_generation_max_tokens(24, Some(0.45), &layouts, 10),
        24,
        "explicit user caps stay authoritative"
    );
}

#[cfg(feature = "ndarray")]
#[test]
fn generation_tail_positions_repeat_prefix_tail_for_generated_chunks() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let mut position_rows = vec![vec![0, 1, 1, 2, 3]];
    let tail = generation_tail_positions(&position_rows, 2);
    position_rows[0].extend([tail[0][0], tail[0][1], tail[0][0]]);
    let position_ids = position_ids_tensor::<B>(&position_rows, 8, &device)
        .into_data()
        .to_vec::<i64>()
        .expect("position ids");

    assert_eq!(position_ids, vec![0, 1, 1, 2, 3, 2, 3, 2]);
}

#[cfg(feature = "ndarray")]
#[test]
fn optimized_position_ids_preserve_contiguous_and_shared_rows() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();

    let contiguous = vec![vec![4, 5, 6], vec![4, 5, 6]];
    let values = position_ids_slice_tensor_optimized::<B>(&contiguous, 0, 3, &device)
        .into_data()
        .to_vec::<i64>()
        .expect("position ids");
    assert_eq!(values, vec![4, 5, 6, 4, 5, 6]);

    let shared_non_contiguous = vec![vec![8, 13, 8], vec![8, 13, 8]];
    let values = position_ids_slice_tensor_optimized::<B>(&shared_non_contiguous, 0, 3, &device)
        .into_data()
        .to_vec::<i64>()
        .expect("position ids");
    assert_eq!(values, vec![8, 13, 8, 8, 13, 8]);

    let per_batch = vec![vec![8, 13, 8], vec![9, 14, 9]];
    let values = position_ids_slice_tensor_optimized::<B>(&per_batch, 0, 3, &device)
        .into_data()
        .to_vec::<i64>()
        .expect("position ids");
    assert_eq!(values, vec![8, 13, 8, 9, 14, 9]);
}

#[cfg(feature = "ndarray")]
#[test]
fn attention_mask_upload_is_skipped_when_all_keys_are_valid() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();

    assert!(attention_mask_tensor_or_none::<B>(&[vec![1, 1, 1]], 3, &device).is_none());
    assert!(attention_mask_tensor_or_none::<B>(&[vec![1, 0, 1]], 3, &device).is_some());
}

#[test]
fn commit_pending_position_ids_uses_attention_cumsum() {
    let masks = vec![vec![1, 1, 0, 1, 1, 0, 1]];
    let mut positions = vec![vec![0, 1, 1, 2, 2, 2, 2]];
    let pending = vec![vec![4, 5, 6]];

    commit_pending_position_ids(&masks, &mut positions, &pending);

    assert_eq!(positions[0], vec![0, 1, 1, 2, 3, 3, 4]);
}

#[test]
fn streaming_position_commit_preserves_rolling_cache_offsets() {
    let masks = vec![vec![1, 1, 1, 1, 0]];
    let mut positions = vec![vec![42, 43, 44, 44, 44]];
    let pending = vec![vec![3, 4]];
    let offsets = vec![42];

    commit_pending_position_ids_with_offsets(&masks, &mut positions, &pending, &offsets);

    assert_eq!(positions[0], vec![42, 43, 44, 45, 45]);
    assert_eq!(next_valid_position_with_offset(&masks[0], offsets[0]), 46);
}

#[cfg(feature = "ndarray")]
#[test]
fn cached_generation_matches_uncached_generation() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = tiny_cache_test_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::<B>::new(&config, &device).map(&mut mapper);
    let values = (0..(2 * 3 * 16 * 16))
        .map(|idx| ((idx % 251) as f32 / 125.0) - 1.0)
        .collect::<Vec<_>>();
    let video = Tensor::<B, 5>::from_data(TensorData::new(values, [1, 2, 3, 16, 16]), &device);

    let uncached = model.gazing_model.generate_uncached(video.clone(), 4, None);
    let cached = model.gazing_model.generate_cached(video, 4, None);

    assert_eq!(cached.gazing_pos, uncached.gazing_pos);
    assert_eq!(cached.num_gazing_each_frame, uncached.num_gazing_each_frame);
    assert_eq!(cached.if_padded_gazing, uncached.if_padded_gazing);
    assert_eq!(cached.confidences[0].len(), uncached.confidences[0].len());
    for (left, right) in cached.confidences[0].iter().zip(&uncached.confidences[0]) {
        assert!((left - right).abs() < 1.0e-5);
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn streaming_cached_generation_matches_batched_cached_generation() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = tiny_cache_test_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::<B>::new(&config, &device).map(&mut mapper);
    let values = (0..(2 * 3 * 16 * 16))
        .map(|idx| ((idx % 251) as f32 / 125.0) - 1.0)
        .collect::<Vec<_>>();
    let video = Tensor::<B, 5>::from_data(TensorData::new(values, [1, 2, 3, 16, 16]), &device);

    let cached = model.gazing_model.generate_cached(video.clone(), 4, None);
    let mut cache = AutoGazeStreamingCache::new(2);
    let streaming =
        model
            .gazing_model
            .generate_streaming_cached(video.clone(), &mut cache, 4, None);

    assert_eq!(streaming.gazing_pos, cached.gazing_pos);
    assert_eq!(
        streaming.num_gazing_each_frame,
        cached.num_gazing_each_frame
    );
    assert_eq!(streaming.if_padded_gazing, cached.if_padded_gazing);
    assert_eq!(streaming.confidences[0].len(), cached.confidences[0].len());
    for (left, right) in streaming.confidences[0].iter().zip(&cached.confidences[0]) {
        assert!((left - right).abs() < 1.0e-5);
    }

    let cached_traces = generated_to_traces(&cached, &config, 4);
    let mut cache = AutoGazeStreamingCache::new(2);
    for frame_idx in 0..2 {
        let frame = video.clone().slice_dim(1, frame_idx..(frame_idx + 1));
        let next = model
            .gazing_model
            .generate_streaming_cached(frame, &mut cache, 4, None);
        let next_traces = generated_to_traces(&next, &config, 4);
        assert_fixation_points_close(
            &next_traces[0].frames[0].points,
            &cached_traces[0].frames[frame_idx].points,
            1.0e-5,
        );
        assert_eq!(
            next_traces[0].frames[0].stop_probability,
            cached_traces[0].frames[frame_idx].stop_probability
        );
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn streaming_device_chunked_decode_matches_host_async_generation() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = tiny_cache_test_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::<B>::new(&config, &device).map(&mut mapper);
    let frames = 2;
    let values = (0..(frames * 3 * 16 * 16))
        .map(|idx| ((idx % 251) as f32 / 125.0) - 1.0)
        .collect::<Vec<_>>();
    let video = Tensor::<B, 5>::from_data(TensorData::new(values, [1, frames, 3, 16, 16]), &device);

    let mut host_cache = AutoGazeStreamingCache::new(frames);
    let host = futures_lite::future::block_on(model.generate_streaming_with_decode_strategy_async(
        video.clone(),
        &mut host_cache,
        4,
        None,
        None,
        AutoGazeDecodeStrategy::HostGreedy,
    ))
    .expect("host async generation");
    for strategy in [
        AutoGazeDecodeStrategy::DeviceGreedy { chunk_size: 2 },
        AutoGazeDecodeStrategy::DeviceTerminalGreedy { chunk_size: 2 },
    ] {
        let mut device_cache = AutoGazeStreamingCache::new(frames);
        let device =
            futures_lite::future::block_on(model.generate_streaming_with_decode_strategy_async(
                video.clone(),
                &mut device_cache,
                4,
                None,
                None,
                strategy,
            ))
            .expect("device async generation");

        assert_eq!(device.gazing_pos, host.gazing_pos, "{strategy:?}");
        assert_eq!(
            device.num_gazing_each_frame, host.num_gazing_each_frame,
            "{strategy:?}"
        );
        assert_eq!(
            device.if_padded_gazing, host.if_padded_gazing,
            "{strategy:?}"
        );
        assert_eq!(device.confidences[0].len(), host.confidences[0].len());
        for ((confidence, padded), host_confidence) in device.confidences[0]
            .iter()
            .zip(&device.if_padded_gazing[0])
            .zip(&host.confidences[0])
        {
            if *padded {
                assert_eq!(*confidence, 0.0, "{strategy:?}");
            } else {
                assert!(*confidence > 0.0, "{strategy:?}");
                assert!(
                    *host_confidence > 0.0,
                    "host should report positive confidence for valid token"
                );
            }
        }
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn streaming_cache_rolls_past_horizon_without_resetting() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = tiny_cache_test_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::<B>::new(&config, &device).map(&mut mapper);
    let frames = 5;
    let values = (0..(frames * 3 * 16 * 16))
        .map(|idx| ((idx % 251) as f32 / 125.0) - 1.0)
        .collect::<Vec<_>>();
    let video = Tensor::<B, 5>::from_data(TensorData::new(values, [1, frames, 3, 16, 16]), &device);
    let mut cache = AutoGazeStreamingCache::new(2);

    for frame_idx in 0..frames {
        let frame = video.clone().slice_dim(1, frame_idx..(frame_idx + 1));
        let output = model
            .gazing_model
            .generate_streaming_cached(frame, &mut cache, 4, None);

        assert_eq!(output.num_gazing_each_frame.len(), 1);
        assert_eq!(cache.processed_frames(), frame_idx + 1);
        assert!(
            cache.active_frames() <= cache.horizon_frames(),
            "active frame window {} exceeded horizon {}",
            cache.active_frames(),
            cache.horizon_frames()
        );
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn low_level_embed_video_prepares_non_model_sized_inputs_without_panic() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = tiny_cache_test_config();
    let model = AutoGazeGazingModel::<B>::new(&config.gaze_model_config, &device);
    let video = Tensor::<B, 5>::zeros([1, 1, 1, 16, 8], &device);

    let (embeddings, _past) = model.embed_video(video, false, None);
    let [batch, frames, tokens, dim] = embeddings.shape().dims::<4>();

    assert_eq!(batch, 1);
    assert_eq!(frames, 1);
    assert_eq!(tokens, config.gaze_model_config.connector_config.num_tokens);
    assert_eq!(dim, config.gaze_model_config.connector_config.hidden_dim);
}

#[cfg(feature = "ndarray")]
fn assert_fixation_points_close(left: &[FixationPoint], right: &[FixationPoint], epsilon: f32) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!((left.x - right.x).abs() <= epsilon);
        assert!((left.y - right.y).abs() <= epsilon);
        assert!((left.scale - right.scale).abs() <= epsilon);
        assert!((left.confidence - right.confidence).abs() <= epsilon);
        assert!((left.width - right.width).abs() <= epsilon);
        assert!((left.height - right.height).abs() <= epsilon);
        assert_eq!(left.grid, right.grid);
    }
}

#[cfg(feature = "ndarray")]
fn tiny_cache_test_config() -> AutoGazeConfig {
    let hidden = 8;
    let heads = 2;
    AutoGazeConfig {
        scales: "8+16".to_string(),
        max_num_frames: 2,
        num_vision_tokens_each_frame: 5,
        gaze_model_config: GazeModelConfig {
            input_img_size: 16,
            num_vision_tokens_each_frame: 5,
            attn_mode: "sdpa".to_string(),
            vision_model_config: VisionModelConfig {
                hidden_dim: hidden,
                out_dim: hidden,
                depth: 1,
                kernel_size: 8,
                temporal_patch_size: 1,
                trunk_temporal_kernel_size: 3,
                trunk_spatial_kernel_size: 1,
            },
            connector_config: ConnectorConfig {
                hidden_dim: hidden,
                num_tokens: 4,
            },
            gaze_decoder_config: crate::config::GazeDecoderConfig {
                vocab_size: 6,
                hidden_size: hidden,
                intermediate_size: hidden * 2,
                num_hidden_layers: 1,
                num_attention_heads: heads,
                num_key_value_heads: heads,
                max_position_embeddings: 512,
                bos_token_id: 0,
                eos_token_id: 5,
                head_dim: hidden / heads,
                num_multi_token_pred: 2,
                ..crate::config::GazeDecoderConfig::default()
            },
        },
        ..AutoGazeConfig::default()
    }
}

#[cfg(feature = "ndarray")]
struct DeterministicParamMapper {
    cursor: usize,
}

#[cfg(feature = "ndarray")]
impl<B: Backend> ModuleMapper<B> for DeterministicParamMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let tensor = param.val();
        let shape = tensor.shape().dims::<D>();
        let device = tensor.device();
        let len = shape.iter().product::<usize>();
        let start = self.cursor;
        self.cursor += len;
        let values = (0..len)
            .map(|idx| (((start + idx) % 97) as f32 - 48.0) * 0.002)
            .collect::<Vec<_>>();
        Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), &device))
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn greedy_selection_applies_task_loss_requirement_after_first_token() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let scores = vec![
        10.0, 1.0, 0.0, -1.0, //
        0.0, 9.0, 1.0, -1.0,
    ];
    let task_losses = vec![0.1, 0.2];
    let logits = Tensor::<B, 3>::from_data(TensorData::new(scores.clone(), [1, 2, 4]), &device);
    let task_loss =
        Tensor::<B, 2>::from_data(TensorData::new(task_losses.clone(), [1, 2]), &device);

    let (tokens, valid, confidences) = greedy_select_multi_tokens(
        logits,
        task_loss,
        &[Vec::new()],
        &[false],
        3,
        2,
        TaskLossStop {
            requirement: Some(0.5),
            is_first_token: true,
        },
    );

    assert_eq!(tokens, vec![vec![0, 3]]);
    assert_eq!(valid, vec![vec![true, false]]);
    assert!(confidences[0][0] > 0.0);
    assert_eq!(confidences[0][1], 0.0);

    let reference = greedy_select_multi_tokens_from_data(
        scores,
        task_losses,
        1,
        2,
        4,
        GreedySelectionContext {
            prior_tokens: &[Vec::new()],
            finished: &[false],
            eos_token_id: 3,
            max_tokens: 2,
            task_loss_stop: TaskLossStop {
                requirement: Some(0.5),
                is_first_token: true,
            },
        },
    );
    assert_eq!((tokens, valid), (reference.0, reference.1));
    for (left, right) in confidences[0].iter().zip(&reference.2[0]) {
        assert!((left - right).abs() < 1.0e-6);
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn greedy_selection_continues_multi_token_block_after_task_loss_stop() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let scores = vec![
        8.0, 1.0, 0.0, -1.0, -2.0, //
        0.0, 9.0, 2.0, 1.0, -2.0, //
        0.0, 9.0, 8.0, 1.0, -2.0,
    ];
    let task_losses = vec![1.0, 0.1, 1.0];
    let logits = Tensor::<B, 3>::from_data(TensorData::new(scores.clone(), [1, 3, 5]), &device);
    let task_loss =
        Tensor::<B, 2>::from_data(TensorData::new(task_losses.clone(), [1, 3]), &device);

    let selected = greedy_select_multi_tokens(
        logits,
        task_loss,
        &[Vec::new()],
        &[false],
        4,
        3,
        TaskLossStop {
            requirement: Some(0.5),
            is_first_token: true,
        },
    );
    let reference = greedy_select_multi_tokens_from_data(
        scores,
        task_losses,
        1,
        3,
        5,
        GreedySelectionContext {
            prior_tokens: &[Vec::new()],
            finished: &[false],
            eos_token_id: 4,
            max_tokens: 3,
            task_loss_stop: TaskLossStop {
                requirement: Some(0.5),
                is_first_token: true,
            },
        },
    );

    assert_eq!(selected.0, vec![vec![0, 4, 2]]);
    assert_eq!(selected.1, vec![vec![true, false, true]]);
    assert_eq!(selected.0, reference.0);
    assert_eq!(selected.1, reference.1);
}

#[cfg(feature = "ndarray")]
#[test]
fn device_greedy_selection_matches_host_selection_for_one_step() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let scores = vec![
        8.0, 1.0, 0.0, -1.0, -2.0, //
        0.0, 9.0, 2.0, 1.0, -2.0, //
        0.0, 9.0, 8.0, 1.0, -2.0,
    ];
    let task_losses = vec![1.0, 0.1, 1.0];
    let logits = Tensor::<B, 3>::from_data(TensorData::new(scores.clone(), [1, 3, 5]), &device);
    let task_loss =
        Tensor::<B, 2>::from_data(TensorData::new(task_losses.clone(), [1, 3]), &device);
    let context = GreedySelectionContext {
        prior_tokens: &[Vec::new()],
        finished: &[false],
        eos_token_id: 4,
        max_tokens: 3,
        task_loss_stop: TaskLossStop {
            requirement: Some(0.5),
            is_first_token: true,
        },
    };
    let reference = greedy_select_multi_tokens_from_data(scores, task_losses, 1, 3, 5, context);

    let selected = device_select_multi_tokens(
        logits,
        task_loss,
        DeviceGreedySelectionState {
            disallowed: disallowed_token_mask::<B>(
                context.prior_tokens,
                context.eos_token_id,
                5,
                &device,
            ),
            finished: finished_token_mask::<B>(context.finished, &device),
            vocab_range: vocab_range_tensor::<B>(1, 5, &device),
            current_len: 0,
        },
        DeviceGreedySelectionConfig {
            eos_token_id: context.eos_token_id,
            max_tokens: context.max_tokens,
            task_loss_stop: context.task_loss_stop,
        },
    );
    let packed = pack_device_greedy_chunk(vec![selected.tokens], vec![selected.valid]);
    let actual = device_greedy_chunk_from_packed_data(packed.into_data(), 1, 3, 4)
        .expect("device greedy chunk");

    assert_eq!(actual.0, reference.0);
    assert_eq!(actual.1, reference.1);
    for (confidence, valid) in actual.2[0].iter().zip(&actual.1[0]) {
        if *valid {
            assert_eq!(*confidence, 1.0);
        } else {
            assert_eq!(*confidence, 0.0);
        }
    }
}

#[test]
fn greedy_selection_rejects_malformed_packed_data() {
    let truncated = TensorData::new(vec![0.0_f32, 1.0, 0.5], [1, 3]);
    assert!(
        greedy_select_multi_tokens_from_packed_data(
            truncated,
            1,
            1,
            3,
            GreedySelectionContext {
                prior_tokens: &[Vec::new()],
                finished: &[false],
                eos_token_id: 3,
                max_tokens: 1,
                task_loss_stop: TaskLossStop {
                    requirement: None,
                    is_first_token: true,
                },
            },
        )
        .is_none()
    );

    let wrong_dtype = TensorData::new(vec![0_i64, 1, 2, 3], [1, 4]);
    assert!(
        greedy_select_multi_tokens_from_packed_data(
            wrong_dtype,
            1,
            1,
            3,
            GreedySelectionContext {
                prior_tokens: &[Vec::new()],
                finished: &[false],
                eos_token_id: 3,
                max_tokens: 1,
                task_loss_stop: TaskLossStop {
                    requirement: None,
                    is_first_token: true,
                },
            },
        )
        .is_none()
    );
}

#[test]
fn scale_layout_falls_back_to_proportional_counts_for_mismatched_totals() {
    let mut config = AutoGazeConfig {
        scales: "32+64+224".to_string(),
        num_vision_tokens_each_frame: 10,
        ..Default::default()
    };
    config.gaze_model_config.vision_model_config.kernel_size = 16;

    let layouts = scale_token_layouts(&config);
    assert_eq!(
        layouts
            .iter()
            .map(|layout| layout.token_count)
            .sum::<usize>(),
        10
    );
    assert_eq!(
        layouts.iter().map(|layout| layout.grid).collect::<Vec<_>>(),
        vec![1, 1, 3]
    );
}

#[test]
fn generated_to_traces_preserves_all_non_padded_multiscale_tokens() {
    let mut config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    config.gaze_model_config.num_vision_tokens_each_frame = 265;
    let generated = AutoGazeGenerateOutput {
        gazing_pos: vec![vec![0, 4, 20, 69]],
        num_gazing_each_frame: vec![4],
        if_padded_gazing: vec![vec![false, false, false, false]],
        confidences: vec![vec![0.9, 0.8, 0.7, 0.6]],
    };

    let traces = generated_to_traces(&generated, &config, 4);

    let grids = traces[0].frames[0]
        .points
        .iter()
        .filter(|point| point.confidence > 0.0)
        .map(|point| point.cell_grid())
        .collect::<Vec<_>>();
    assert_eq!(grids, vec![Some(2), Some(4), Some(7), Some(14)]);

    let traces = generated_to_traces(&generated, &config, 1);
    let grids = traces[0].frames[0]
        .points
        .iter()
        .filter(|point| point.confidence > 0.0)
        .map(|point| point.cell_grid())
        .collect::<Vec<_>>();
    assert_eq!(
        grids,
        vec![Some(2), Some(4), Some(7), Some(14)],
        "generated output conversion must preserve all real tokens even when the display lower-bound is smaller"
    );
}

#[test]
fn generated_scale_token_masks_match_upstream_per_scale_layout() {
    let mut config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    config.gaze_model_config.num_vision_tokens_each_frame = 265;
    let generated = AutoGazeGenerateOutput {
        gazing_pos: vec![vec![0, 4, 20, 69, 1 + 265, 5 + 265]],
        num_gazing_each_frame: vec![4, 2],
        if_padded_gazing: vec![vec![false, false, false, false, false, true]],
        confidences: vec![vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.0]],
    };

    let masks = generated.scale_token_masks(&config);

    assert_eq!(masks.len(), 1);
    assert_eq!(
        masks[0]
            .iter()
            .map(|mask| (mask.grid, mask.token_count))
            .collect::<Vec<_>>(),
        vec![(2, 4), (4, 16), (7, 49), (14, 196)]
    );
    assert!(masks[0][0].frames[0][0]);
    assert!(masks[0][1].frames[0][0]);
    assert!(masks[0][2].frames[0][0]);
    assert!(masks[0][3].frames[0][0]);
    assert!(masks[0][0].frames[1][1]);
    assert!(
        !masks[0][1].frames[1][1],
        "padded gaze tokens must not set scale masks"
    );
}
