use super::layout::{
    AutoGazeScaleTokenLayout, scale_token_index, scale_token_layouts, token_to_fixation_point,
};
use super::{AutoGazeGenerateOutput, AutoGazeScaleTokenMask, GeneratedFrameFixations};
use crate::config::AutoGazeConfig;
use crate::{FixationPoint, FixationSet, FrameFixationTrace};

pub(super) fn generated_to_traces(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    k: usize,
) -> Vec<FrameFixationTrace> {
    let mut traces = Vec::with_capacity(generated.gazing_pos.len());
    for batch_idx in 0..generated.gazing_pos.len() {
        let frames = generated_to_frame_fixations(generated, config, batch_idx)
            .into_iter()
            .map(|frame| FixationSet::with_min_len(frame.points, frame.stop_probability, k))
            .collect();
        traces.push(FrameFixationTrace::new(frames));
    }
    traces
}

pub(crate) fn generated_to_frame_fixations(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_idx: usize,
) -> Vec<GeneratedFrameFixations> {
    let scale_layouts = scale_token_layouts(config);
    let mut cursor = 0usize;
    let mut frames = Vec::with_capacity(generated.num_gazing_each_frame.len());
    for (frame_idx, frame_len) in generated.num_gazing_each_frame.iter().copied().enumerate() {
        let frame = generated_frame_fixations_from_layouts(
            generated,
            config,
            batch_idx,
            frame_idx,
            cursor,
            frame_len,
            &scale_layouts,
        );
        cursor += frame_len;
        frames.push(frame);
    }
    frames
}

pub(crate) fn generated_frame_fixations(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_idx: usize,
    frame_idx: usize,
) -> Option<GeneratedFrameFixations> {
    if batch_idx >= generated.gazing_pos.len() || frame_idx >= generated.num_gazing_each_frame.len()
    {
        return None;
    }
    let cursor = generated
        .num_gazing_each_frame
        .iter()
        .take(frame_idx)
        .sum::<usize>();
    let frame_len = generated.num_gazing_each_frame[frame_idx];
    let scale_layouts = scale_token_layouts(config);
    Some(generated_frame_fixations_from_layouts(
        generated,
        config,
        batch_idx,
        frame_idx,
        cursor,
        frame_len,
        &scale_layouts,
    ))
}

fn generated_frame_fixations_from_layouts(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    batch_idx: usize,
    frame_idx: usize,
    cursor: usize,
    frame_len: usize,
    scale_layouts: &[AutoGazeScaleTokenLayout],
) -> GeneratedFrameFixations {
    let tokens = generated.gazing_pos.get(batch_idx);
    let padded = generated.if_padded_gazing.get(batch_idx);
    let confidences = generated.confidences.get(batch_idx);
    let mut points = Vec::new();
    let mut stop_probability = 0.0f32;
    for local_idx in 0..frame_len {
        let global_idx = cursor + local_idx;
        let Some(&raw_token) = tokens.and_then(|tokens| tokens.get(global_idx)) else {
            continue;
        };
        let is_padded = padded
            .and_then(|flags| flags.get(global_idx))
            .copied()
            .unwrap_or(true);
        if is_padded {
            stop_probability = 1.0;
            continue;
        }
        let frame_offset = (frame_idx * config.num_vision_tokens_each_frame) as i64;
        let token = raw_token - frame_offset;
        if token < 0 {
            continue;
        }
        let confidence = confidences
            .and_then(|confidences| confidences.get(global_idx))
            .copied()
            .unwrap_or(1.0);
        if let Some(point) = token_to_fixation_point(token as usize, scale_layouts, confidence) {
            points.push(point);
        }
    }
    GeneratedFrameFixations {
        points,
        stop_probability,
    }
}

pub(crate) fn generated_to_frame_points(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
) -> Vec<Vec<Vec<FixationPoint>>> {
    (0..generated.gazing_pos.len())
        .map(|batch_idx| {
            generated_to_frame_fixations(generated, config, batch_idx)
                .into_iter()
                .map(|frame| frame.points)
                .collect()
        })
        .collect()
}

pub(super) fn generated_scale_token_masks(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
) -> Vec<Vec<AutoGazeScaleTokenMask>> {
    let scale_layouts = scale_token_layouts(config);
    let frames = generated.num_gazing_each_frame.len();
    let mut batches = Vec::with_capacity(generated.gazing_pos.len());

    for batch_idx in 0..generated.gazing_pos.len() {
        let mut masks = scale_layouts
            .iter()
            .map(|layout| AutoGazeScaleTokenMask {
                grid: layout.grid,
                token_count: layout.token_count,
                frames: vec![vec![false; layout.token_count]; frames],
            })
            .collect::<Vec<_>>();

        let mut cursor = 0usize;
        for (frame_idx, frame_len) in generated.num_gazing_each_frame.iter().copied().enumerate() {
            for local_idx in 0..frame_len {
                let global_idx = cursor + local_idx;
                let Some(&raw_token) = generated
                    .gazing_pos
                    .get(batch_idx)
                    .and_then(|tokens| tokens.get(global_idx))
                else {
                    continue;
                };
                let padded = generated
                    .if_padded_gazing
                    .get(batch_idx)
                    .and_then(|flags| flags.get(global_idx))
                    .copied()
                    .unwrap_or(true);
                if padded {
                    continue;
                }
                let frame_offset = (frame_idx * config.num_vision_tokens_each_frame) as i64;
                let token = raw_token - frame_offset;
                if token < 0 {
                    continue;
                }
                if let Some((scale_idx, local_token)) =
                    scale_token_index(token as usize, &scale_layouts)
                {
                    masks[scale_idx].frames[frame_idx][local_token] = true;
                }
            }
            cursor += frame_len;
        }

        batches.push(masks);
    }

    batches
}
