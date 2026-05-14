use super::AutoGazePastKeyValues;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

pub(super) fn adapt_video_channels<B: Backend>(
    video: Tensor<B, 5>,
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    match channels {
        3 => video,
        1 => video.repeat_dim(2, 3),
        other => {
            let [batch, time, _, height, width] = video.shape().dims::<5>();
            let data = vec![0.0f32; batch * time * 3 * height * width];
            let mut out = Tensor::<B, 5>::from_data(
                TensorData::new(data, [batch, time, 3, height, width]),
                device,
            );
            let keep = other.min(3);
            out = out.slice_assign(
                [0..batch, 0..time, 0..keep, 0..height, 0..width],
                video.slice_dim(2, 0..keep),
            );
            out
        }
    }
}

pub(super) fn split_heads<B: Backend>(
    tokens: Tensor<B, 3>,
    heads: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, seq, _] = tokens.shape().dims::<3>();
    tokens
        .reshape([batch, seq, heads.max(1), head_dim.max(1)])
        .swap_dims(1, 2)
}

pub(super) fn merge_heads<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = tokens.shape().dims::<4>();
    tokens
        .swap_dims(1, 2)
        .reshape([batch, seq, heads * head_dim])
}

pub(super) fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dim = x.shape().dims::<4>()[3];
    let half = dim / 2;
    let x1 = x.clone().slice_dim(3, 0..half);
    let x2 = x.slice_dim(3, half..dim);
    Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3)
}

#[derive(Clone, Copy)]
pub(super) struct CausalAttentionShape {
    pub(super) batch: usize,
    pub(super) heads: usize,
    pub(super) query_len: usize,
    pub(super) key_len: usize,
    pub(super) past_len: usize,
    pub(super) head_dim: usize,
}

pub(super) fn causal_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    shape: CausalAttentionShape,
) -> Tensor<B, 4> {
    let _ = shape.heads;
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .div_scalar((shape.head_dim as f32).sqrt().max(1.0));
    let bias = causal_attention_bias_for_query(
        shape.batch,
        shape.query_len,
        shape.key_len,
        shape.past_len,
        attention_mask,
        &scores.device(),
    );
    let attn = activation::softmax(scores + bias, 3);
    attn.matmul(v)
}

pub(super) fn causal_attention_bias_for_query<B: Backend>(
    batch: usize,
    query_len: usize,
    key_len: usize,
    past_len: usize,
    attention_mask: Option<Tensor<B, 2, Int>>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let q_pos = Tensor::<B, 1, Int>::arange(past_len as i64..(past_len + query_len) as i64, device)
        .reshape([1, 1, query_len, 1]);
    let k_pos = Tensor::<B, 1, Int>::arange(0..key_len as i64, device).reshape([1, 1, 1, key_len]);
    let causal = k_pos.lower_equal(q_pos).float();
    let mut bias = causal.sub_scalar(1.0).abs().mul_scalar(-1.0e9);
    if let Some(mask) = attention_mask {
        let key_valid = mask.float().reshape([batch.max(1), 1, 1, key_len]);
        let key_bias = key_valid.sub_scalar(1.0).abs().mul_scalar(-1.0e9);
        bias = bias + key_bias;
    }
    bias
}

pub(super) fn llama3_inv_freq<B: Backend>(
    config: &crate::config::GazeDecoderConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let head_dim = config.head_dim.max(1);
    let half = (head_dim / 2).max(1);
    let base = config.rope_theta.max(1.0);
    let mut inv_freq = Vec::with_capacity(half);
    for idx in 0..half {
        let dim_index = (idx * 2) as f32;
        inv_freq.push(1.0 / base.powf(dim_index / head_dim as f32));
    }

    if let Some(rope_scaling) = config.rope_scaling.as_ref() {
        let rope_type = rope_scaling
            .get("rope_type")
            .and_then(|value| value.as_str())
            .unwrap_or("default");
        if rope_type == "llama3" {
            let factor = rope_scaling
                .get("factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32;
            let low_freq_factor = rope_scaling
                .get("low_freq_factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32;
            let high_freq_factor = rope_scaling
                .get("high_freq_factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(4.0) as f32;
            let original_max_position_embeddings = rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|value| value.as_f64())
                .unwrap_or(config.max_position_embeddings as f64)
                as f32;

            let low_freq_wavelen = original_max_position_embeddings / low_freq_factor.max(1.0);
            let high_freq_wavelen =
                original_max_position_embeddings / high_freq_factor.max(low_freq_factor + 1.0e-6);
            for value in inv_freq.iter_mut() {
                let wavelen = 2.0 * std::f32::consts::PI / (*value).max(1.0e-12);
                let scaled = if wavelen > low_freq_wavelen {
                    *value / factor.max(1.0)
                } else {
                    *value
                };
                if wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen {
                    let smooth_factor = (original_max_position_embeddings / wavelen
                        - low_freq_factor)
                        / (high_freq_factor - low_freq_factor).max(1.0e-6);
                    *value =
                        (1.0 - smooth_factor) * scaled / factor.max(1.0) + smooth_factor * scaled;
                } else {
                    *value = scaled;
                }
            }
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device)
}

pub(super) fn attention_mask_tensor<B: Backend>(
    mask_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = mask_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in mask_rows {
        values.extend(row.iter().copied().take(seq_len));
        values.resize(
            values.len() + seq_len.saturating_sub(row.len().min(seq_len)),
            0,
        );
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, seq_len]), device)
}

pub(super) fn attention_mask_tensor_or_none<B: Backend>(
    mask_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2, Int>> {
    if attention_mask_rows_are_all_valid(mask_rows, seq_len) {
        None
    } else {
        Some(attention_mask_tensor(mask_rows, seq_len, device))
    }
}

pub(super) fn attention_mask_rows_are_all_valid(mask_rows: &[Vec<i64>], seq_len: usize) -> bool {
    !mask_rows.is_empty()
        && mask_rows
            .iter()
            .all(|row| row.len() >= seq_len && row.iter().take(seq_len).all(|mask| *mask != 0))
}

#[cfg(test)]
pub(super) fn position_ids_tensor<B: Backend>(
    position_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = position_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in position_rows {
        values.extend(row.iter().copied().take(seq_len));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, seq_len]), device)
}

pub(super) fn position_ids_tensor_optimized<B: Backend>(
    position_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    position_ids_slice_tensor_optimized(position_rows, 0, seq_len, device)
}

pub(super) fn position_ids_slice_tensor<B: Backend>(
    position_rows: &[Vec<i64>],
    start: usize,
    len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = position_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * len);
    for row in position_rows {
        values.extend(row.iter().copied().skip(start).take(len));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, len]), device)
}

pub(super) fn position_ids_slice_tensor_optimized<B: Backend>(
    position_rows: &[Vec<i64>],
    start: usize,
    len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = position_rows.len().max(1);
    if let Some(first_value) = contiguous_position_start(position_rows, start, len) {
        return Tensor::<B, 1, Int>::arange(first_value..(first_value + len as i64), device)
            .reshape([1, len])
            .repeat_dim(0, batch);
    }

    if let Some(row) = identical_position_slice(position_rows, start, len) {
        return Tensor::<B, 1, Int>::from_data(TensorData::new(row, [len]), device)
            .reshape([1, len])
            .repeat_dim(0, batch);
    }

    position_ids_slice_tensor(position_rows, start, len, device)
}

pub(super) fn contiguous_position_start(
    position_rows: &[Vec<i64>],
    start: usize,
    len: usize,
) -> Option<i64> {
    let first = position_rows.first()?.get(start).copied().or(Some(0))?;
    if position_rows.iter().all(|row| {
        row.len() >= start + len
            && (0..len).all(|offset| row[start + offset] == first + offset as i64)
    }) {
        Some(first)
    } else {
        None
    }
}

pub(super) fn identical_position_slice(
    position_rows: &[Vec<i64>],
    start: usize,
    len: usize,
) -> Option<Vec<i64>> {
    let first = position_rows.first()?;
    if first.len() < start + len {
        return None;
    }
    let row = first[start..start + len].to_vec();
    if position_rows
        .iter()
        .all(|candidate| candidate.len() >= start + len && candidate[start..start + len] == row)
    {
        Some(row)
    } else {
        None
    }
}

pub(super) fn cached_sequence_len<B: Backend>(
    past_key_values: &Option<AutoGazePastKeyValues<B>>,
) -> usize {
    past_key_values
        .as_ref()
        .and_then(|past| past.first())
        .map(|past| past.len)
        .unwrap_or(0)
}

pub(super) fn compact_past_key_values<B: Backend>(
    past_key_values: &mut Option<AutoGazePastKeyValues<B>>,
    drop: usize,
) {
    let Some(values) = past_key_values.as_mut() else {
        return;
    };
    if drop == 0 {
        return;
    }

    for past in values {
        let drop = drop.min(past.len);
        if drop == 0 {
            continue;
        }
        let keep = past.len - drop;
        if keep > 0 {
            let [batch, heads, _capacity, head_dim] = past.key.shape().dims::<4>();
            let retained_key = past.key.clone().slice_dim(2, drop..past.len);
            let retained_value = past.value.clone().slice_dim(2, drop..past.len);
            past.key = past
                .key
                .clone()
                .slice_assign([0..batch, 0..heads, 0..keep, 0..head_dim], retained_key);
            past.value = past
                .value
                .clone()
                .slice_assign([0..batch, 0..heads, 0..keep, 0..head_dim], retained_value);
        }
        past.len = keep;
    }
}

pub(super) fn generation_tail_positions(
    position_rows: &[Vec<i64>],
    num_multi_token_pred: usize,
) -> Vec<Vec<i64>> {
    let chunk = num_multi_token_pred.max(1);
    position_rows
        .iter()
        .map(|row| {
            if row.is_empty() {
                vec![0]
            } else {
                row[row.len().saturating_sub(chunk)..].to_vec()
            }
        })
        .collect()
}

pub(super) fn commit_pending_position_ids(
    mask_rows: &[Vec<i64>],
    position_rows: &mut [Vec<i64>],
    pending_rows: &[Vec<usize>],
) {
    for ((mask_row, position_row), pending) in mask_rows
        .iter()
        .zip(position_rows.iter_mut())
        .zip(pending_rows)
    {
        if pending.is_empty() {
            continue;
        }

        let mut valid_count = 0i64;
        for (idx, mask) in mask_row.iter().copied().enumerate() {
            if mask != 0 {
                valid_count += 1;
            }
            if pending.contains(&idx) && idx < position_row.len() {
                position_row[idx] = valid_count.saturating_sub(1);
            }
        }
    }
}

pub(super) fn commit_pending_position_ids_with_offsets(
    mask_rows: &[Vec<i64>],
    position_rows: &mut [Vec<i64>],
    pending_rows: &[Vec<usize>],
    position_offsets: &[i64],
) {
    for (((mask_row, position_row), pending), offset) in mask_rows
        .iter()
        .zip(position_rows.iter_mut())
        .zip(pending_rows)
        .zip(position_offsets.iter().copied())
    {
        if pending.is_empty() {
            continue;
        }

        let mut valid_count = offset;
        for (idx, mask) in mask_row.iter().copied().enumerate() {
            if mask != 0 {
                valid_count = valid_count.saturating_add(1);
            }
            if pending.contains(&idx) && idx < position_row.len() {
                position_row[idx] = valid_count.saturating_sub(1);
            }
        }
    }
}

pub(super) fn append_generated_position_slots(
    mask_rows: &mut [Vec<i64>],
    position_rows: &mut [Vec<i64>],
    generation_tail_positions: &[Vec<i64>],
    new_tokens: usize,
) -> Vec<Vec<usize>> {
    let mut generated_indices = vec![Vec::new(); mask_rows.len()];
    for batch_idx in 0..mask_rows.len() {
        let tail = generation_tail_positions
            .get(batch_idx)
            .filter(|tail| !tail.is_empty());
        for local_idx in 0..new_tokens {
            generated_indices[batch_idx].push(mask_rows[batch_idx].len());
            mask_rows[batch_idx].push(1);
            let position = tail
                .map(|tail| tail[local_idx % tail.len()])
                .unwrap_or(local_idx as i64);
            position_rows[batch_idx].push(position);
        }
    }
    generated_indices
}

pub(super) fn truncate_generation_rows(
    mask_rows: &mut [Vec<i64>],
    position_rows: &mut [Vec<i64>],
    pending_rows: &mut [Vec<usize>],
    drop: usize,
) {
    if drop == 0 {
        return;
    }
    for ((mask_row, position_row), pending) in mask_rows
        .iter_mut()
        .zip(position_rows.iter_mut())
        .zip(pending_rows.iter_mut())
    {
        let new_len = mask_row.len().saturating_sub(drop);
        mask_row.truncate(new_len);
        position_row.truncate(new_len);
        pending.retain(|idx| *idx < new_len);
    }
}

pub(super) fn truncate_past_key_values_tail<B: Backend>(
    past_key_values: &mut Option<AutoGazePastKeyValues<B>>,
    drop: usize,
) {
    if drop == 0 {
        return;
    }
    let Some(values) = past_key_values.as_mut() else {
        return;
    };
    for past in values {
        past.len = past.len.saturating_sub(drop);
    }
}

pub(super) fn next_valid_position_with_offset(mask_row: &[i64], position_offset: i64) -> i64 {
    let valid_count = mask_row.iter().filter(|mask| **mask != 0).count() as i64;
    position_offset.saturating_add(valid_count)
}
