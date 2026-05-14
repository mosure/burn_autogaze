use super::{
    DeviceGreedySelection, DeviceGreedySelectionConfig, DeviceGreedySelectionState,
    GreedyTokenSelection, TaskLossStop,
};
use burn::tensor::backend::{Backend, ExecutionError};
use burn::tensor::{Bool, Int, Tensor, TensorData};

pub(super) fn greedy_select_multi_tokens<B: Backend>(
    logits: Tensor<B, 3>,
    task_loss: Tensor<B, 2>,
    prior_tokens: &[Vec<i64>],
    finished: &[bool],
    eos_token_id: i64,
    max_tokens: usize,
    task_loss_stop: TaskLossStop,
) -> GreedyTokenSelection {
    let [batch, num_multi, vocab] = logits.shape().dims::<3>();
    let context = GreedySelectionContext {
        prior_tokens,
        finished,
        eos_token_id,
        max_tokens,
        task_loss_stop,
    };
    greedy_select_multi_tokens_from_packed_data(
        pack_greedy_logits_task(logits, task_loss).into_data(),
        batch,
        num_multi,
        vocab,
        context,
    )
    .unwrap_or_else(|| GreedySelectionBuilder::new(batch).finish(eos_token_id))
}

pub(super) async fn greedy_select_multi_tokens_async<B: Backend>(
    logits: Tensor<B, 3>,
    task_loss: Tensor<B, 2>,
    prior_tokens: &[Vec<i64>],
    finished: &[bool],
    eos_token_id: i64,
    max_tokens: usize,
    task_loss_stop: TaskLossStop,
) -> Result<GreedyTokenSelection, ExecutionError> {
    let [batch, num_multi, vocab] = logits.shape().dims::<3>();
    let context = GreedySelectionContext {
        prior_tokens,
        finished,
        eos_token_id,
        max_tokens,
        task_loss_stop,
    };
    let data = pack_greedy_logits_task(logits, task_loss)
        .into_data_async()
        .await?;
    Ok(
        greedy_select_multi_tokens_from_packed_data(data, batch, num_multi, vocab, context)
            .unwrap_or_else(|| GreedySelectionBuilder::new(batch).finish(eos_token_id)),
    )
}

pub(super) fn device_select_multi_tokens<B: Backend>(
    logits: Tensor<B, 3>,
    task_loss: Tensor<B, 2>,
    state: DeviceGreedySelectionState<B>,
    config: DeviceGreedySelectionConfig,
) -> DeviceGreedySelection<B> {
    let [batch, num_multi, vocab] = logits.shape().dims::<3>();
    let device = logits.device();
    let eos_tokens = Tensor::<B, 2, Int>::full([batch, 1], config.eos_token_id, &device);
    let false_column = bool_tensor_from_values::<B, 2>(vec![false; batch], [batch, 1], &device);
    let mut disallowed = state.disallowed;
    let mut finished = state.finished;
    let mut token_columns = Vec::with_capacity(num_multi);
    let mut valid_columns = Vec::with_capacity(num_multi);

    for multi_idx in 0..num_multi {
        let capacity_active = state.current_len.saturating_add(multi_idx) < config.max_tokens;
        let active =
            bool_tensor_from_values::<B, 2>(vec![capacity_active; batch], [batch, 1], &device)
                .bool_and(finished.clone().bool_not());
        let logits_step = logits
            .clone()
            .slice_dim(1, multi_idx..(multi_idx + 1))
            .reshape([batch, vocab]);
        let inactive_or_disallowed = disallowed
            .clone()
            .bool_or(active.clone().bool_not().repeat_dim(1, vocab));
        let masked_logits = logits_step.mask_fill(inactive_or_disallowed, -1.0e30_f32);
        let (best_scores, best_indices) = masked_logits.clone().max_dim_with_indices(1);
        let has_score = best_scores.clone().greater_elem(-1.0e29_f32);
        let task_stop = if let Some(threshold) = config.task_loss_stop.requirement {
            if config.task_loss_stop.is_first_token && multi_idx == 0 {
                false_column.clone()
            } else {
                task_loss
                    .clone()
                    .slice_dim(1, multi_idx..(multi_idx + 1))
                    .lower_equal_elem(threshold)
                    .bool_and(active.clone())
                    .bool_and(has_score.clone())
            }
        } else {
            false_column.clone()
        };
        let valid = active
            .clone()
            .bool_and(has_score.clone())
            .bool_and(task_stop.clone().bool_not());
        let tokens = best_indices
            .clone()
            .mask_where(valid.clone().bool_not(), eos_tokens.clone());
        let selected_for_disallow = best_indices
            .repeat_dim(1, vocab)
            .equal(state.vocab_range.clone());
        disallowed = disallowed.bool_or(
            selected_for_disallow.bool_and(
                active
                    .clone()
                    .bool_and(has_score.clone())
                    .repeat_dim(1, vocab),
            ),
        );
        finished = finished.bool_or(has_score.clone().bool_not().bool_and(active));

        token_columns.push(tokens);
        valid_columns.push(valid);
    }

    DeviceGreedySelection {
        tokens: Tensor::cat(token_columns, 1),
        valid: Tensor::cat(valid_columns, 1),
        finished,
        disallowed,
    }
}

pub(super) fn vocab_range_tensor<B: Backend>(
    batch: usize,
    vocab: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 1, Int>::arange(0..vocab as i64, device)
        .reshape([1, vocab])
        .repeat_dim(0, batch)
}

pub(super) fn device_finished_after_selection_block<B: Backend>(
    finished: Tensor<B, 2, Bool>,
    valid: Tensor<B, 2, Bool>,
) -> Tensor<B, 2, Bool> {
    let invalid_selected = valid.bool_not().float().sum_dim(1).greater_elem(0.0_f32);
    finished.bool_or(invalid_selected)
}

pub(super) fn pack_device_greedy_chunk<B: Backend>(
    token_steps: Vec<Tensor<B, 2, Int>>,
    valid_steps: Vec<Tensor<B, 2, Bool>>,
) -> Tensor<B, 2> {
    let tokens = Tensor::cat(token_steps, 1).float();
    let valid = Tensor::cat(valid_steps, 1).float();
    Tensor::cat(vec![tokens, valid], 1)
}

pub(super) async fn read_device_greedy_chunk_async<B: Backend>(
    packed: Tensor<B, 2>,
    slots: usize,
    eos_token_id: i64,
) -> Result<GreedyTokenSelection, ExecutionError> {
    let batch = packed.shape().dims::<2>()[0];
    let data = packed.into_data_async().await?;
    Ok(
        device_greedy_chunk_from_packed_data(data, batch, slots, eos_token_id)
            .unwrap_or_else(|| GreedySelectionBuilder::new(batch).finish(eos_token_id)),
    )
}

pub(super) fn device_greedy_chunk_from_packed_data(
    data: TensorData,
    batch: usize,
    slots: usize,
    eos_token_id: i64,
) -> Option<GreedyTokenSelection> {
    let values = data.to_vec::<f32>().ok()?;
    let row_stride = slots.checked_mul(2)?;
    if values.len() < batch.saturating_mul(row_stride) {
        return None;
    }

    let mut tokens = vec![vec![eos_token_id; slots]; batch];
    let mut valid = vec![vec![false; slots]; batch];
    let mut confidences = vec![vec![0.0; slots]; batch];
    for batch_idx in 0..batch {
        let row = batch_idx * row_stride;
        for slot in 0..slots {
            tokens[batch_idx][slot] = values[row + slot].round() as i64;
            valid[batch_idx][slot] = values[row + slots + slot] > 0.5;
            confidences[batch_idx][slot] = if valid[batch_idx][slot] { 1.0 } else { 0.0 };
        }
    }
    Some((tokens, valid, confidences))
}

pub(super) fn disallowed_token_mask<B: Backend>(
    prior_tokens: &[Vec<i64>],
    eos_token_id: i64,
    vocab: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let batch = prior_tokens.len().max(1);
    let mut values = vec![false; batch * vocab];
    for (batch_idx, row) in prior_tokens.iter().enumerate() {
        for token in row.iter().copied().chain(std::iter::once(eos_token_id)) {
            if token >= 0 {
                let token = token as usize;
                if token < vocab {
                    values[batch_idx * vocab + token] = true;
                }
            }
        }
    }
    bool_tensor_from_values(values, [batch, vocab], device)
}

pub(super) fn finished_token_mask<B: Backend>(
    finished: &[bool],
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    bool_tensor_from_values(finished.to_vec(), [finished.len().max(1), 1], device)
}

fn bool_tensor_from_values<B: Backend, const D: usize>(
    values: Vec<bool>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Bool> {
    Tensor::<B, D, Bool>::from_bool(TensorData::new(values, shape), device)
}

fn pack_greedy_logits_task<B: Backend>(
    logits: Tensor<B, 3>,
    task_loss: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch, num_multi, vocab] = logits.shape().dims::<3>();
    Tensor::cat(
        vec![logits.reshape([batch, num_multi * vocab]), task_loss],
        1,
    )
}

pub(super) fn greedy_select_multi_tokens_from_packed_data(
    data: TensorData,
    batch: usize,
    num_multi: usize,
    vocab: usize,
    context: GreedySelectionContext<'_>,
) -> Option<GreedyTokenSelection> {
    let values = data.to_vec::<f32>().ok()?;
    let row_stride = num_multi.checked_mul(vocab)?.checked_add(num_multi)?;
    if values.len() < batch.saturating_mul(row_stride) {
        return None;
    }

    let packed = GreedyPackedLogits {
        values: &values,
        batch,
        num_multi,
        vocab,
        row_stride,
    };
    let mut builder = GreedySelectionBuilder::new(batch);
    let mut selected = GreedyStepValues::new(batch);
    for multi_idx in 0..num_multi {
        if !builder.has_active_rows(context) {
            break;
        }

        packed.step_values(multi_idx, &builder, context, &mut selected);
        builder.push_step(
            multi_idx,
            &selected.tokens,
            &selected.scores,
            &selected.confidences,
            &selected.task_losses,
            context,
        );
    }

    Some(builder.finish(context.eos_token_id))
}

struct GreedyPackedLogits<'a> {
    values: &'a [f32],
    batch: usize,
    num_multi: usize,
    vocab: usize,
    row_stride: usize,
}

struct GreedyStepValues {
    tokens: Vec<i64>,
    scores: Vec<f32>,
    confidences: Vec<f32>,
    task_losses: Vec<f32>,
}

impl GreedyStepValues {
    fn new(batch: usize) -> Self {
        Self {
            tokens: vec![0_i64; batch],
            scores: vec![f32::NEG_INFINITY; batch],
            confidences: vec![0.0_f32; batch],
            task_losses: vec![f32::INFINITY; batch],
        }
    }

    fn reset(&mut self, batch: usize) {
        self.tokens.resize(batch, 0);
        self.scores.resize(batch, f32::NEG_INFINITY);
        self.confidences.resize(batch, 0.0);
        self.task_losses.resize(batch, f32::INFINITY);
        self.tokens.fill(0);
        self.scores.fill(f32::NEG_INFINITY);
        self.confidences.fill(0.0);
        self.task_losses.fill(f32::INFINITY);
    }
}

impl GreedyPackedLogits<'_> {
    fn step_values(
        &self,
        multi_idx: usize,
        builder: &GreedySelectionBuilder,
        context: GreedySelectionContext<'_>,
        values: &mut GreedyStepValues,
    ) {
        values.reset(self.batch);

        for batch_idx in 0..self.batch {
            values.task_losses[batch_idx] =
                self.values[batch_idx * self.row_stride + self.num_multi * self.vocab + multi_idx];
            if context.finished.get(batch_idx).copied().unwrap_or(false)
                || context.prior_tokens[batch_idx].len() + builder.per_batch_tokens[batch_idx].len()
                    >= context.max_tokens
            {
                continue;
            }

            let row_base = batch_idx * self.row_stride + multi_idx * self.vocab;
            let mut best_token = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut exp_sum = 0.0_f32;
            for vocab_idx in 0..self.vocab {
                if builder.disallows_token(batch_idx, vocab_idx, context) {
                    continue;
                }
                let score = self.values[row_base + vocab_idx];
                if !score.is_finite() {
                    continue;
                }
                exp_sum += score.exp();
                if best_token.is_none() || score > best_score {
                    best_token = Some(vocab_idx as i64);
                    best_score = score;
                }
            }

            if let Some(token) = best_token {
                values.tokens[batch_idx] = token;
                values.scores[batch_idx] = best_score;
                values.confidences[batch_idx] = if exp_sum > 0.0 {
                    best_score.exp() / exp_sum
                } else {
                    1.0
                };
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct GreedySelectionContext<'a> {
    pub(super) prior_tokens: &'a [Vec<i64>],
    pub(super) finished: &'a [bool],
    pub(super) eos_token_id: i64,
    pub(super) max_tokens: usize,
    pub(super) task_loss_stop: TaskLossStop,
}

#[derive(Debug)]
struct GreedySelectionBuilder {
    per_batch_tokens: Vec<Vec<i64>>,
    per_batch_disallowed: Vec<Vec<i64>>,
    per_batch_valid: Vec<Vec<bool>>,
    per_batch_confidences: Vec<Vec<f32>>,
}

impl GreedySelectionBuilder {
    fn new(batch: usize) -> Self {
        Self {
            per_batch_tokens: vec![Vec::new(); batch],
            per_batch_disallowed: vec![Vec::new(); batch],
            per_batch_valid: vec![Vec::new(); batch],
            per_batch_confidences: vec![Vec::new(); batch],
        }
    }

    fn has_active_rows(&self, context: GreedySelectionContext<'_>) -> bool {
        self.per_batch_tokens
            .iter()
            .enumerate()
            .any(|(batch_idx, selected)| {
                !context.finished.get(batch_idx).copied().unwrap_or(false)
                    && context.prior_tokens[batch_idx].len() + selected.len() < context.max_tokens
            })
    }

    fn disallows_token(
        &self,
        batch_idx: usize,
        token_idx: usize,
        context: GreedySelectionContext<'_>,
    ) -> bool {
        if context.eos_token_id >= 0 && token_idx == context.eos_token_id as usize {
            return true;
        }
        context.prior_tokens[batch_idx]
            .iter()
            .chain(&self.per_batch_disallowed[batch_idx])
            .any(|&token| token >= 0 && token as usize == token_idx)
    }

    fn push_step(
        &mut self,
        multi_idx: usize,
        best_tokens: &[i64],
        best_scores: &[f32],
        confidences: &[f32],
        task_losses: &[f32],
        context: GreedySelectionContext<'_>,
    ) {
        for batch_idx in 0..self.per_batch_tokens.len() {
            if context.finished.get(batch_idx).copied().unwrap_or(false)
                || context.prior_tokens[batch_idx].len() + self.per_batch_tokens[batch_idx].len()
                    >= context.max_tokens
            {
                continue;
            }

            let Some((&token, &best_score)) =
                best_tokens.get(batch_idx).zip(best_scores.get(batch_idx))
            else {
                continue;
            };
            if !best_score.is_finite() {
                continue;
            }

            let task_loss = task_losses.get(batch_idx).copied().unwrap_or(f32::INFINITY);
            let meets_task_loss_requirement =
                context.task_loss_stop.requirement.is_some_and(|threshold| {
                    !(context.task_loss_stop.is_first_token && multi_idx == 0)
                        && task_loss <= threshold
                });
            self.per_batch_disallowed[batch_idx].push(token);
            self.per_batch_tokens[batch_idx].push(if meets_task_loss_requirement {
                context.eos_token_id
            } else {
                token
            });
            self.per_batch_valid[batch_idx].push(!meets_task_loss_requirement);
            self.per_batch_confidences[batch_idx].push(if meets_task_loss_requirement {
                0.0
            } else {
                let confidence = confidences.get(batch_idx).copied().unwrap_or(1.0);
                if confidence.is_finite() && confidence > 0.0 {
                    confidence
                } else {
                    1.0
                }
            });
        }
    }

    fn finish(self, eos_token_id: i64) -> GreedyTokenSelection {
        let padded_len = self
            .per_batch_tokens
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        let mut out_tokens = Vec::with_capacity(self.per_batch_tokens.len());
        let mut out_valid = Vec::with_capacity(self.per_batch_tokens.len());
        let mut out_confidences = Vec::with_capacity(self.per_batch_tokens.len());
        for batch_idx in 0..self.per_batch_tokens.len() {
            let mut tokens = self.per_batch_tokens[batch_idx].clone();
            let mut valid = self.per_batch_valid[batch_idx].clone();
            let mut confidences = self.per_batch_confidences[batch_idx].clone();
            while tokens.len() < padded_len {
                tokens.push(eos_token_id);
                valid.push(false);
                confidences.push(0.0);
            }
            out_tokens.push(tokens);
            out_valid.push(valid);
            out_confidences.push(confidences);
        }

        (out_tokens, out_valid, out_confidences)
    }
}

#[cfg(test)]
pub(super) fn greedy_select_multi_tokens_from_data(
    scores: Vec<f32>,
    task_losses: Vec<f32>,
    batch: usize,
    num_multi: usize,
    vocab: usize,
    context: GreedySelectionContext<'_>,
) -> GreedyTokenSelection {
    let GreedySelectionContext {
        prior_tokens,
        finished,
        eos_token_id,
        max_tokens,
        task_loss_stop,
    } = context;
    let mut per_batch_tokens = vec![Vec::new(); batch];
    let mut per_batch_valid = vec![Vec::new(); batch];
    let mut per_batch_confidences = vec![Vec::new(); batch];

    for batch_idx in 0..batch {
        let mut disallowed = prior_tokens[batch_idx].clone();
        for multi_idx in 0..num_multi {
            if finished.get(batch_idx).copied().unwrap_or(false)
                || prior_tokens[batch_idx].len() + per_batch_tokens[batch_idx].len() >= max_tokens
            {
                break;
            }

            let base = (batch_idx * num_multi + multi_idx) * vocab;
            let mut best_index = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut exp_sum = 0.0f32;
            for vocab_idx in 0..vocab {
                if vocab_idx as i64 == eos_token_id || disallowed.contains(&(vocab_idx as i64)) {
                    continue;
                }
                let score = scores[base + vocab_idx];
                if score.is_finite() {
                    exp_sum += score.exp();
                    if score > best_score {
                        best_score = score;
                        best_index = Some(vocab_idx as i64);
                    }
                }
            }

            if let Some(token) = best_index {
                disallowed.push(token);
                let meets_task_loss_requirement =
                    task_loss_stop.requirement.is_some_and(|threshold| {
                        !(task_loss_stop.is_first_token && multi_idx == 0)
                            && task_losses[batch_idx * num_multi + multi_idx] <= threshold
                    });
                per_batch_tokens[batch_idx].push(if meets_task_loss_requirement {
                    eos_token_id
                } else {
                    token
                });
                per_batch_valid[batch_idx].push(!meets_task_loss_requirement);
                per_batch_confidences[batch_idx].push(if meets_task_loss_requirement {
                    0.0
                } else if exp_sum > 0.0 {
                    best_score.exp() / exp_sum
                } else {
                    1.0
                });
            } else {
                break;
            }
        }
    }

    let padded_len = per_batch_tokens.iter().map(Vec::len).max().unwrap_or(0);
    let mut out_tokens = Vec::with_capacity(batch);
    let mut out_valid = Vec::with_capacity(batch);
    let mut out_confidences = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let mut tokens = per_batch_tokens[batch_idx].clone();
        let mut valid = per_batch_valid[batch_idx].clone();
        let mut confidences = per_batch_confidences[batch_idx].clone();
        while tokens.len() < padded_len {
            tokens.push(eos_token_id);
            valid.push(false);
            confidences.push(0.0);
        }
        out_tokens.push(tokens);
        out_valid.push(valid);
        out_confidences.push(confidences);
    }

    (out_tokens, out_valid, out_confidences)
}
