use crate::config::{AutoGazeConfig, ConnectorConfig, GazeModelConfig, VisionModelConfig};
use crate::{FixationPoint, FrameFixationTrace};
use anyhow::{Context, Result, bail};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig3d,
};
use burn::tensor::Bool;
use burn::tensor::activation;
use burn::tensor::backend::{Backend, ExecutionError};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::collections::VecDeque;
use std::path::Path;

mod layout;
mod selection;
mod tensor_utils;
#[cfg(test)]
mod tests;
mod trace;

pub use self::layout::{AutoGazeScaleTokenLayout, scale_token_layouts};
use self::layout::{
    effective_generation_max_tokens, generation_coverage_trackers, normalize_scale_layouts,
    observe_generation_coverage, square_grid,
};
use self::selection::{
    device_finished_after_selection_block, device_select_multi_tokens, disallowed_token_mask,
    finished_token_mask, greedy_select_multi_tokens, greedy_select_multi_tokens_async,
    pack_device_greedy_chunk, read_device_greedy_chunk_async, vocab_range_tensor,
};
use self::tensor_utils::*;
pub(crate) use self::trace::{
    generated_frame_fixations, generated_to_frame_fixations, generated_to_frame_points,
};
use self::trace::{generated_scale_token_masks, generated_to_traces};

#[derive(Module, Debug)]
pub struct Conv3dBlockForStreaming<B: Backend> {
    pub conv3d: Conv3d<B>,
    #[module(skip)]
    temporal_patch_size: usize,
}

impl<B: Backend> Conv3dBlockForStreaming<B> {
    pub fn new(
        hidden_dim: usize,
        temporal_patch_size: usize,
        spatial_kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv3d: Conv3dConfig::new(
                [hidden_dim.max(1), hidden_dim.max(1)],
                [
                    temporal_patch_size.max(1),
                    spatial_kernel_size.max(1),
                    spatial_kernel_size.max(1),
                ],
            )
            .with_padding(PaddingConfig3d::Explicit(
                0,
                spatial_kernel_size.saturating_sub(1) / 2,
                spatial_kernel_size.saturating_sub(1) / 2,
            ))
            .init(device),
            temporal_patch_size: temporal_patch_size.max(1),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Tensor<B, 5>>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let x = if use_cache {
            if let Some(past) = past_conv_values {
                Tensor::cat(vec![past, x], 2)
            } else {
                x.pad(
                    [
                        (0, 0),
                        (0, 0),
                        (self.temporal_patch_size.saturating_sub(1), 0),
                        (0, 0),
                        (0, 0),
                    ],
                    PadMode::Constant(0.0),
                )
            }
        } else {
            x.pad(
                [
                    (0, 0),
                    (0, 0),
                    (self.temporal_patch_size.saturating_sub(1), 0),
                    (0, 0),
                    (0, 0),
                ],
                PadMode::Constant(0.0),
            )
        };
        let time = x.shape().dims::<5>()[2];
        let keep = self.temporal_patch_size.saturating_sub(1);
        let new_past_conv_values = if keep == 0 {
            Tensor::<B, 5>::zeros(
                [
                    x.shape().dims::<5>()[0],
                    x.shape().dims::<5>()[1],
                    0,
                    x.shape().dims::<5>()[3],
                    x.shape().dims::<5>()[4],
                ],
                &x.device(),
            )
        } else {
            x.clone().slice_dim(2, time.saturating_sub(keep)..time)
        };

        let x = self.conv3d.forward(x);
        (activation::relu(x), new_past_conv_values)
    }
}

#[derive(Module, Debug)]
pub struct ShallowVideoConvNet<B: Backend> {
    pub temporal_conv: Conv3d<B>,
    pub norm: LayerNorm<B>,
    pub blocks: Vec<Conv3dBlockForStreaming<B>>,
    pub out_proj: Conv3d<B>,
    #[module(skip)]
    temporal_patch_size: usize,
}

impl<B: Backend> ShallowVideoConvNet<B> {
    pub fn new(config: &VisionModelConfig, device: &B::Device) -> Self {
        let hidden_dim = config.hidden_dim.max(1);
        let out_dim = config.out_dim.max(1);
        let temporal_patch_size = config.temporal_patch_size.max(1);
        let temporal_conv = Conv3dConfig::new(
            [3, hidden_dim],
            [
                temporal_patch_size,
                config.kernel_size.max(1),
                config.kernel_size.max(1),
            ],
        )
        .with_stride([
            temporal_patch_size,
            config.kernel_size.max(1),
            config.kernel_size.max(1),
        ])
        .init(device);
        let norm = LayerNormConfig::new(hidden_dim).init(device);
        let blocks = (0..config.depth.max(1))
            .map(|_| {
                Conv3dBlockForStreaming::new(
                    hidden_dim,
                    config.trunk_temporal_kernel_size.max(1),
                    config.trunk_spatial_kernel_size.max(1),
                    device,
                )
            })
            .collect();
        let out_proj = Conv3dConfig::new([hidden_dim, out_dim], [1, 1, 1]).init(device);
        Self {
            temporal_conv,
            norm,
            blocks,
            out_proj,
            temporal_patch_size,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Vec<Tensor<B, 5>>>,
    ) -> (Tensor<B, 5>, Vec<Tensor<B, 5>>) {
        let mut x = x.permute([0, 2, 1, 3, 4]);
        x = self.temporal_conv.forward(x);
        let [batch, channels, time, height, width] = x.shape().dims::<5>();
        let x_flat = x
            .permute([0, 2, 1, 3, 4])
            .reshape([batch * time, channels, height * width])
            .swap_dims(1, 2);
        let x_flat = self.norm.forward(x_flat);
        let mut x = x_flat
            .swap_dims(1, 2)
            .reshape([batch, time, channels, height, width])
            .permute([0, 2, 1, 3, 4]);

        let mut new_past = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let past = past_conv_values
                .as_ref()
                .and_then(|values| values.get(index))
                .cloned();
            let (next_x, next_past) = block.forward(x, use_cache, past);
            x = next_x;
            new_past.push(next_past);
        }
        let x = self.out_proj.forward(x);
        (x, new_past)
    }
}

#[derive(Module, Debug)]
pub struct Connector<B: Backend> {
    pub pos_embed: Param<Tensor<B, 2>>,
}

impl<B: Backend> Connector<B> {
    pub fn new(config: &ConnectorConfig, device: &B::Device) -> Self {
        Self {
            pos_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [config.num_tokens.max(1), config.hidden_dim.max(1)],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, time, tokens, dim] = x.shape().dims::<4>();
        let pos = self.pos_embed.val().reshape([1, 1, tokens, dim]);
        x + pos.repeat_dim(0, batch).repeat_dim(1, time)
    }
}

#[derive(Clone, Debug)]
pub struct AutoGazeGenerateOutput {
    pub gazing_pos: Vec<Vec<i64>>,
    pub num_gazing_each_frame: Vec<usize>,
    pub if_padded_gazing: Vec<Vec<bool>>,
    pub confidences: Vec<Vec<f32>>,
}

pub struct AutoGazeDeviceTokens<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub valid: Tensor<B, 2, Bool>,
}

pub struct AutoGazeDeviceGenerateOutput<B: Backend> {
    pub generated: AutoGazeGenerateOutput,
    pub device_tokens: Option<AutoGazeDeviceTokens<B>>,
}

impl AutoGazeGenerateOutput {
    /// Decode generated AutoGaze token ids into upstream-style per-scale token masks.
    ///
    /// The original NVIDIA implementation keeps one boolean mask per pyramid scale
    /// instead of immediately unioning tokens into a full-resolution pixel mask.
    /// Keeping that representation available makes visualization and downstream
    /// sparse video/image pipelines less error-prone.
    pub fn scale_token_masks(&self, config: &AutoGazeConfig) -> Vec<Vec<AutoGazeScaleTokenMask>> {
        generated_scale_token_masks(self, config)
    }

    /// Decode generated AutoGaze token ids into fixation traces.
    ///
    /// This is the same conversion used by the high-level tracing APIs. Use the
    /// direct sparse readout helpers when downstream code only needs image-token
    /// selections and does not need a full trace allocation.
    pub fn traces(
        &self,
        config: &AutoGazeConfig,
        min_points_per_frame: usize,
    ) -> Vec<FrameFixationTrace> {
        generated_to_traces(self, config, min_points_per_frame)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedFrameFixations {
    pub points: Vec<FixationPoint>,
    pub stop_probability: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoGazeScaleTokenMask {
    pub grid: usize,
    pub token_count: usize,
    pub frames: Vec<Vec<bool>>,
}

/// Greedy decoder readout strategy for streaming AutoGaze generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoGazeDecodeStrategy {
    /// CPU-side greedy selection with one compact logits/task-loss readback per decode step.
    #[default]
    HostGreedy,
    /// Device-side greedy selection with compact token readbacks at chunk boundaries for stopping.
    ///
    /// Tokens and validity match host greedy selection. Confidence values are binary validity
    /// markers so render/readout paths do not pay for a full vocab softmax on every decode step.
    DeviceGreedy {
        /// Maximum autoregressive decode steps before each compact token readback.
        chunk_size: usize,
    },
    /// Device-side greedy selection with one compact readback after the full frame decode budget.
    ///
    /// Tokens and validity match host greedy selection. Confidence values are binary validity
    /// markers so render/readout paths do not pay for a full vocab softmax on every decode step.
    DeviceTerminalGreedy {
        /// Maximum autoregressive decode steps per internal scheduling chunk.
        chunk_size: usize,
    },
}

impl AutoGazeDecodeStrategy {
    pub const fn device_greedy(chunk_size: usize) -> Self {
        Self::DeviceGreedy { chunk_size }
    }

    pub const fn device_terminal_greedy(chunk_size: usize) -> Self {
        Self::DeviceTerminalGreedy { chunk_size }
    }

    pub const fn chunk_size(self) -> usize {
        match self {
            Self::HostGreedy => 1,
            Self::DeviceGreedy { chunk_size } | Self::DeviceTerminalGreedy { chunk_size } => {
                if chunk_size == 0 {
                    1
                } else {
                    chunk_size
                }
            }
        }
    }

    pub const fn normalized(self) -> Self {
        match self {
            Self::HostGreedy => Self::HostGreedy,
            Self::DeviceGreedy { chunk_size } => Self::DeviceGreedy {
                chunk_size: if chunk_size == 0 { 1 } else { chunk_size },
            },
            Self::DeviceTerminalGreedy { chunk_size } => Self::DeviceTerminalGreedy {
                chunk_size: if chunk_size == 0 { 1 } else { chunk_size },
            },
        }
    }
}

/// Stateful generation cache for advancing an AutoGaze video stream one frame at a time.
///
/// This keeps decoder KV tensors, pending generated-token embeddings, and RoPE position state
/// across frames. It is intended for realtime pipelines that already have a rolling frame source
/// and want to avoid re-decoding older frames for every sliding window.
#[derive(Clone, Debug)]
pub struct AutoGazeStreamingCache<B: Backend> {
    horizon_frames: usize,
    state: Option<AutoGazeStreamingCacheState<B>>,
}

impl<B: Backend> AutoGazeStreamingCache<B> {
    pub fn new(horizon_frames: usize) -> Self {
        Self {
            horizon_frames: horizon_frames.max(1),
            state: None,
        }
    }

    pub fn horizon_frames(&self) -> usize {
        self.horizon_frames
    }

    pub fn processed_frames(&self) -> usize {
        self.state
            .as_ref()
            .map(|state| state.processed_frames)
            .unwrap_or(0)
    }

    pub fn active_frames(&self) -> usize {
        self.state
            .as_ref()
            .map(|state| state.frame_token_lengths.len())
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

#[derive(Clone, Debug)]
struct AutoGazeStreamingCacheState<B: Backend> {
    past_conv_values: Option<Vec<Tensor<B, 5>>>,
    past_key_values: Option<AutoGazePastKeyValues<B>>,
    pending_query_embeds: Option<Tensor<B, 3>>,
    prefix_attention_mask: Vec<Vec<i64>>,
    prefix_position_ids: Vec<Vec<i64>>,
    pending_position_indices: Vec<Vec<usize>>,
    position_offsets: Vec<i64>,
    next_position_ids: Vec<i64>,
    processed_frames: usize,
    batch: usize,
    vision_tokens: usize,
    dim: usize,
    max_tokens: usize,
    cache_capacity: usize,
    frame_token_lengths: VecDeque<usize>,
}

impl<B: Backend> AutoGazeStreamingCacheState<B> {
    fn new(
        batch: usize,
        vision_tokens: usize,
        dim: usize,
        max_tokens: usize,
        horizon_frames: usize,
    ) -> Self {
        Self {
            past_conv_values: None,
            past_key_values: None,
            pending_query_embeds: None,
            prefix_attention_mask: vec![vec![]; batch],
            prefix_position_ids: vec![vec![]; batch],
            pending_position_indices: vec![Vec::<usize>::new(); batch],
            position_offsets: vec![0; batch],
            next_position_ids: vec![0; batch],
            processed_frames: 0,
            batch,
            vision_tokens,
            dim,
            max_tokens,
            cache_capacity: horizon_frames.max(1) * (vision_tokens + max_tokens.max(1)),
            frame_token_lengths: VecDeque::new(),
        }
    }

    fn matches(
        &self,
        batch: usize,
        vision_tokens: usize,
        dim: usize,
        max_tokens: usize,
        horizon_frames: usize,
    ) -> bool {
        self.batch == batch
            && self.vision_tokens == vision_tokens
            && self.dim == dim
            && self.max_tokens == max_tokens
            && self.cache_capacity == horizon_frames.max(1) * (vision_tokens + max_tokens.max(1))
    }

    fn matches_runtime(
        &self,
        batch: usize,
        max_tokens: usize,
        _horizon_frames: usize,
        incoming_frames: usize,
    ) -> bool {
        self.batch == batch && self.max_tokens == max_tokens && incoming_frames > 0
    }

    fn active_sequence_len(&self) -> usize {
        self.prefix_attention_mask
            .first()
            .map(Vec::len)
            .unwrap_or(0)
    }

    fn compact_for_next_frame(&mut self, required_frame_tokens: usize, horizon_frames: usize) {
        let horizon_frames = horizon_frames.max(1);
        while self.frame_token_lengths.len() >= horizon_frames {
            let Some(drop_tokens) = self.frame_token_lengths.pop_front() else {
                break;
            };
            self.drop_oldest_tokens(drop_tokens);
        }

        let required_frame_tokens = required_frame_tokens.min(self.cache_capacity);
        while self
            .active_sequence_len()
            .saturating_add(required_frame_tokens)
            > self.cache_capacity
        {
            let Some(drop_tokens) = self.frame_token_lengths.pop_front() else {
                break;
            };
            self.drop_oldest_tokens(drop_tokens);
        }
    }

    fn record_completed_frame(&mut self, token_count: usize) {
        self.frame_token_lengths.push_back(token_count);
    }

    fn commit_pending_position_ids(&mut self) {
        commit_pending_position_ids_with_offsets(
            &self.prefix_attention_mask,
            &mut self.prefix_position_ids,
            &self.pending_position_indices,
            &self.position_offsets,
        );
        for (batch_idx, next_position_id) in self.next_position_ids.iter_mut().enumerate() {
            *next_position_id = next_valid_position_with_offset(
                &self.prefix_attention_mask[batch_idx],
                self.position_offsets[batch_idx],
            );
        }
    }

    fn clear_pending_position_indices(&mut self) {
        self.pending_position_indices
            .iter_mut()
            .for_each(Vec::clear);
    }

    fn append_frame_positions(&mut self, vision_tokens: usize) {
        for batch_idx in 0..self.batch {
            for _ in 0..vision_tokens {
                self.prefix_attention_mask[batch_idx].push(1);
                self.prefix_position_ids[batch_idx].push(self.next_position_ids[batch_idx]);
                self.next_position_ids[batch_idx] =
                    self.next_position_ids[batch_idx].saturating_add(1);
            }
        }
    }

    fn drop_oldest_tokens(&mut self, token_count: usize) {
        let cached_len = cached_sequence_len(&self.past_key_values);
        let drop = token_count.min(cached_len).min(self.active_sequence_len());
        if drop == 0 {
            return;
        }

        for (batch_idx, row) in self.prefix_attention_mask.iter_mut().enumerate() {
            let dropped_valid = row.iter().take(drop).filter(|mask| **mask != 0).count() as i64;
            self.position_offsets[batch_idx] =
                self.position_offsets[batch_idx].saturating_add(dropped_valid);
            row.drain(0..drop);
        }
        for row in &mut self.prefix_position_ids {
            row.drain(0..drop);
        }
        for pending in &mut self.pending_position_indices {
            pending.retain_mut(|idx| {
                if *idx < drop {
                    false
                } else {
                    *idx -= drop;
                    true
                }
            });
        }
        compact_past_key_values(&mut self.past_key_values, drop);
    }
}

type GreedyTokenSelection = (Vec<Vec<i64>>, Vec<Vec<bool>>, Vec<Vec<f32>>);

#[derive(Clone, Copy, Debug)]
struct TaskLossStop {
    requirement: Option<f32>,
    is_first_token: bool,
}

struct DeviceGreedySelectionState<B: Backend> {
    disallowed: Tensor<B, 2, Bool>,
    finished: Tensor<B, 2, Bool>,
    vocab_range: Tensor<B, 2, Int>,
    current_len: usize,
}

#[derive(Clone, Copy)]
struct DeviceGreedySelectionConfig {
    eos_token_id: i64,
    max_tokens: usize,
    task_loss_stop: TaskLossStop,
}

struct DeviceGreedySelection<B: Backend> {
    tokens: Tensor<B, 2, Int>,
    valid: Tensor<B, 2, Bool>,
    finished: Tensor<B, 2, Bool>,
    disallowed: Tensor<B, 2, Bool>,
}

struct DeviceGreedyChunkStep<B: Backend> {
    token_embeds: Tensor<B, 3>,
    generated_indices: Vec<Vec<usize>>,
}

struct DeviceGreedyChunkTensors<B: Backend> {
    tokens: Tensor<B, 2, Int>,
    valid: Tensor<B, 2, Bool>,
}

#[derive(Clone, Copy, Debug)]
struct DeviceChunkDecodeOptions {
    max_gaze_tokens_each_frame: usize,
    task_loss_requirement: Option<f32>,
    coverage_stop_ratio: Option<f64>,
    decode_chunk_size: usize,
    collect_device_tokens: bool,
}

#[derive(Debug)]
pub struct AutoGazeCausalLmOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub task_loss_prediction: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
struct AutoGazePastKeyValue<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    len: usize,
}

type AutoGazePastKeyValues<B> = Vec<AutoGazePastKeyValue<B>>;

#[derive(Debug)]
struct AutoGazeCachedCausalLmOutput<B: Backend> {
    logits: Tensor<B, 3>,
    task_loss_prediction: Tensor<B, 3>,
    past_key_values: AutoGazePastKeyValues<B>,
}

#[derive(Module, Debug)]
pub struct LlamaRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    eps: f32,
}

impl<B: Backend> LlamaRmsNorm<B> {
    pub fn new(width: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::ones([width.max(1)], device)),
            eps: eps.max(1.0e-8),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, width] = x.shape().dims::<3>();
        let (var, mean) = x.clone().var_mean_bias(2);
        let rms = var.add(mean.powf_scalar(2.0)).add_scalar(self.eps).sqrt();
        let weight = self.weight.val().reshape([1, 1, width]);
        x.div(rms).mul(weight)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_key_value_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    inv_freq: Tensor<B, 1>,
}

impl<B: Backend> AutoGazeLlamaAttention<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        let num_heads = config.num_attention_heads.max(1);
        let num_key_value_heads = config.num_key_value_heads.max(1);
        let head_dim = config.head_dim.max(1);
        let inv_freq = llama3_inv_freq(config, device);
        Self {
            q_proj: LinearConfig::new(config.hidden_size.max(1), num_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size.max(1), num_key_value_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size.max(1), num_key_value_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            o_proj: LinearConfig::new(num_heads * head_dim, config.hidden_size.max(1))
                .with_bias(config.attention_bias)
                .init(device),
            num_heads,
            num_key_value_heads,
            head_dim,
            inv_freq,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden_states.shape().dims::<3>();
        let q = split_heads(
            self.q_proj.forward(hidden_states.clone()),
            self.num_heads,
            self.head_dim,
        );
        let k = split_heads(
            self.k_proj.forward(hidden_states.clone()),
            self.num_key_value_heads,
            self.head_dim,
        );
        let v = split_heads(
            self.v_proj.forward(hidden_states),
            self.num_key_value_heads,
            self.head_dim,
        );

        let q = self.apply_rope(q, position_ids.clone());
        let mut k = self.apply_rope(k, position_ids);
        let mut v = v;
        if self.num_key_value_heads != self.num_heads {
            let repeat = (self.num_heads / self.num_key_value_heads.max(1)).max(1);
            k = k.repeat_dim(1, repeat);
            v = v.repeat_dim(1, repeat);
        }

        let out = causal_attention(
            q,
            k,
            v,
            attention_mask,
            CausalAttentionShape {
                batch,
                heads: self.num_heads,
                query_len: seq,
                key_len: seq,
                past_len: 0,
                head_dim: self.head_dim,
            },
        );
        self.o_proj.forward(merge_heads(out))
    }

    fn forward_cached(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
        past: Option<AutoGazePastKeyValue<B>>,
        cache_capacity: usize,
    ) -> (Tensor<B, 3>, AutoGazePastKeyValue<B>) {
        let [batch, query_len, _] = hidden_states.shape().dims::<3>();
        let q = split_heads(
            self.q_proj.forward(hidden_states.clone()),
            self.num_heads,
            self.head_dim,
        );
        let k = split_heads(
            self.k_proj.forward(hidden_states.clone()),
            self.num_key_value_heads,
            self.head_dim,
        );
        let v = split_heads(
            self.v_proj.forward(hidden_states),
            self.num_key_value_heads,
            self.head_dim,
        );

        let q = self.apply_rope(q, position_ids.clone());
        let next_k = self.apply_rope(k, position_ids);
        let next_v = v;
        let cache_device = next_k.device();
        let past_len = past.as_ref().map(|past| past.len).unwrap_or(0);
        let present_len = past_len + query_len;
        let capacity = cache_capacity.max(present_len).max(1);
        let (key_cache, value_cache, mut k, mut v) = if let Some(past) = past {
            let key_cache = past.key.slice_assign(
                [
                    0..batch,
                    0..self.num_key_value_heads,
                    past_len..present_len,
                    0..self.head_dim,
                ],
                next_k,
            );
            let value_cache = past.value.slice_assign(
                [
                    0..batch,
                    0..self.num_key_value_heads,
                    past_len..present_len,
                    0..self.head_dim,
                ],
                next_v,
            );
            (
                key_cache.clone(),
                value_cache.clone(),
                key_cache.slice_dim(2, 0..present_len),
                value_cache.slice_dim(2, 0..present_len),
            )
        } else {
            let key_cache = Tensor::<B, 4>::empty(
                [batch, self.num_key_value_heads, capacity, self.head_dim],
                &cache_device,
            )
            .slice_assign(
                [
                    0..batch,
                    0..self.num_key_value_heads,
                    0..present_len,
                    0..self.head_dim,
                ],
                next_k.clone(),
            );
            let value_cache = Tensor::<B, 4>::empty(
                [batch, self.num_key_value_heads, capacity, self.head_dim],
                &cache_device,
            )
            .slice_assign(
                [
                    0..batch,
                    0..self.num_key_value_heads,
                    0..present_len,
                    0..self.head_dim,
                ],
                next_v.clone(),
            );
            (key_cache, value_cache, next_k, next_v)
        };
        let present = AutoGazePastKeyValue {
            key: key_cache,
            value: value_cache,
            len: present_len,
        };
        let key_len = present_len;

        if self.num_key_value_heads != self.num_heads {
            let repeat = (self.num_heads / self.num_key_value_heads.max(1)).max(1);
            k = k.repeat_dim(1, repeat);
            v = v.repeat_dim(1, repeat);
        }

        let out = causal_attention(
            q,
            k,
            v,
            attention_mask,
            CausalAttentionShape {
                batch,
                heads: self.num_heads,
                query_len,
                key_len,
                past_len,
                head_dim: self.head_dim,
            },
        );
        (self.o_proj.forward(merge_heads(out)), present)
    }

    fn apply_rope(&self, x: Tensor<B, 4>, position_ids: Tensor<B, 2, Int>) -> Tensor<B, 4> {
        let [batch, heads, seq, dim] = x.shape().dims::<4>();
        let half = dim / 2;
        let pos = position_ids.float().reshape([batch, 1, seq, 1]);
        let inv = self.inv_freq.clone().reshape([1, 1, 1, half]);
        let freqs = pos.mul(inv);
        let phases = Tensor::cat(vec![freqs.clone(), freqs], 3);
        let cos = phases.clone().cos().repeat_dim(1, heads);
        let sin = phases.sin().repeat_dim(1, heads);
        x.clone().mul(cos) + rotate_half(x).mul(sin)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> AutoGazeLlamaMlp<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(
                config.hidden_size.max(1),
                config.intermediate_size.max(1),
            )
            .with_bias(config.mlp_bias)
            .init(device),
            up_proj: LinearConfig::new(config.hidden_size.max(1), config.intermediate_size.max(1))
                .with_bias(config.mlp_bias)
                .init(device),
            down_proj: LinearConfig::new(
                config.intermediate_size.max(1),
                config.hidden_size.max(1),
            )
            .with_bias(config.mlp_bias)
            .init(device),
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(hidden_states.clone()));
        let up = self.up_proj.forward(hidden_states);
        self.down_proj.forward(gate.mul(up))
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaDecoderLayer<B: Backend> {
    pub self_attn: AutoGazeLlamaAttention<B>,
    pub mlp: AutoGazeLlamaMlp<B>,
    pub input_layernorm: LlamaRmsNorm<B>,
    pub post_attention_layernorm: LlamaRmsNorm<B>,
}

impl<B: Backend> AutoGazeLlamaDecoderLayer<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            self_attn: AutoGazeLlamaAttention::new(config, device),
            mlp: AutoGazeLlamaMlp::new(config, device),
            input_layernorm: LlamaRmsNorm::new(
                config.hidden_size.max(1),
                config.rms_norm_eps,
                device,
            ),
            post_attention_layernorm: LlamaRmsNorm::new(
                config.hidden_size.max(1),
                config.rms_norm_eps,
                device,
            ),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let attn = self.self_attn.forward(
            self.input_layernorm.forward(hidden_states.clone()),
            attention_mask.clone(),
            position_ids.clone(),
        );
        let hidden_states = hidden_states + attn;
        let mlp = self
            .mlp
            .forward(self.post_attention_layernorm.forward(hidden_states.clone()));
        hidden_states + mlp
    }

    fn forward_cached(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
        past: Option<AutoGazePastKeyValue<B>>,
        cache_capacity: usize,
    ) -> (Tensor<B, 3>, AutoGazePastKeyValue<B>) {
        let (attn, present) = self.self_attn.forward_cached(
            self.input_layernorm.forward(hidden_states.clone()),
            attention_mask,
            position_ids,
            past,
            cache_capacity,
        );
        let hidden_states = hidden_states + attn;
        let mlp = self
            .mlp
            .forward(self.post_attention_layernorm.forward(hidden_states.clone()));
        (hidden_states + mlp, present)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaModel<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<AutoGazeLlamaDecoderLayer<B>>,
    pub norm: LlamaRmsNorm<B>,
}

impl<B: Backend> AutoGazeLlamaModel<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        let layers = (0..config.num_hidden_layers.max(1))
            .map(|_| AutoGazeLlamaDecoderLayer::new(config, device))
            .collect();
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size.max(1), config.hidden_size.max(1))
                .init(device),
            layers,
            norm: LlamaRmsNorm::new(config.hidden_size.max(1), config.rms_norm_eps, device),
        }
    }

    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = inputs_embeds;
        for layer in self.layers.iter() {
            hidden_states =
                layer.forward(hidden_states, attention_mask.clone(), position_ids.clone());
        }
        self.norm.forward(hidden_states)
    }

    fn forward_cached(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
        past_key_values: Option<AutoGazePastKeyValues<B>>,
        cache_capacity: usize,
    ) -> (Tensor<B, 3>, AutoGazePastKeyValues<B>) {
        let mut hidden_states = inputs_embeds;
        let mut next_past = Vec::with_capacity(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            let past = past_key_values
                .as_ref()
                .and_then(|past_values| past_values.get(idx))
                .cloned();
            let (next_hidden_states, present) = layer.forward_cached(
                hidden_states,
                attention_mask.clone(),
                position_ids.clone(),
                past,
                cache_capacity,
            );
            hidden_states = next_hidden_states;
            next_past.push(present);
        }
        (self.norm.forward(hidden_states), next_past)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaForCausalLmMultiTokenPred<B: Backend> {
    pub model: AutoGazeLlamaModel<B>,
    pub lm_head: Linear<B>,
    pub task_loss_prediction_head: Linear<B>,
    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    num_multi_token_pred: usize,
}

impl<B: Backend> AutoGazeLlamaForCausalLmMultiTokenPred<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            model: AutoGazeLlamaModel::new(config, device),
            lm_head: LinearConfig::new(
                config.hidden_size.max(1),
                config.vocab_size.max(1) * config.num_multi_token_pred.max(1),
            )
            .with_bias(false)
            .init(device),
            task_loss_prediction_head: LinearConfig::new(
                config.hidden_size.max(1),
                config.num_multi_token_pred.max(1),
            )
            .with_bias(false)
            .init(device),
            vocab_size: config.vocab_size.max(1),
            num_multi_token_pred: config.num_multi_token_pred.max(1),
        }
    }

    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> AutoGazeCausalLmOutput<B> {
        let hidden_states = self
            .model
            .forward(inputs_embeds, attention_mask, position_ids);
        let logits = self.lm_head.forward(hidden_states.clone());
        let task_loss_prediction = self
            .task_loss_prediction_head
            .forward(hidden_states.clone());
        AutoGazeCausalLmOutput {
            logits,
            task_loss_prediction,
            hidden_states,
        }
    }

    fn forward_cached(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
        past_key_values: Option<AutoGazePastKeyValues<B>>,
        cache_capacity: usize,
    ) -> AutoGazeCachedCausalLmOutput<B> {
        let (hidden_states, past_key_values) = self.model.forward_cached(
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            cache_capacity,
        );
        let logits = self.lm_head.forward(hidden_states.clone());
        let task_loss_prediction = self
            .task_loss_prediction_head
            .forward(hidden_states.clone());
        AutoGazeCachedCausalLmOutput {
            logits,
            task_loss_prediction,
            past_key_values,
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeGazingModel<B: Backend> {
    pub vision_model: ShallowVideoConvNet<B>,
    pub connector: Connector<B>,
    pub gaze_decoder: AutoGazeLlamaForCausalLmMultiTokenPred<B>,
    #[module(skip)]
    scale_layouts: Vec<AutoGazeScaleTokenLayout>,
    #[module(skip)]
    input_img_size: usize,
    #[module(skip)]
    num_vision_tokens_each_frame: usize,
    #[module(skip)]
    frame_sampling_rate: usize,
    #[module(skip)]
    num_multi_token_pred: usize,
    #[module(skip)]
    eos_token_id: i64,
}

impl<B: Backend> AutoGazeGazingModel<B> {
    pub fn new(config: &GazeModelConfig, device: &B::Device) -> Self {
        let token_count = config.num_vision_tokens_each_frame.max(1);
        Self::new_with_scale_layouts(
            config,
            vec![AutoGazeScaleTokenLayout {
                token_count,
                grid: square_grid(token_count),
            }],
            device,
        )
    }

    pub(crate) fn new_with_scale_layouts(
        config: &GazeModelConfig,
        scale_layouts: Vec<AutoGazeScaleTokenLayout>,
        device: &B::Device,
    ) -> Self {
        Self {
            vision_model: ShallowVideoConvNet::new(&config.vision_model_config, device),
            connector: Connector::new(&config.connector_config, device),
            gaze_decoder: AutoGazeLlamaForCausalLmMultiTokenPred::new(
                &config.gaze_decoder_config,
                device,
            ),
            scale_layouts: normalize_scale_layouts(
                scale_layouts,
                config.num_vision_tokens_each_frame.max(1),
            ),
            input_img_size: config.input_img_size.max(1),
            num_vision_tokens_each_frame: config.num_vision_tokens_each_frame.max(1),
            frame_sampling_rate: config.vision_model_config.temporal_patch_size.max(1),
            num_multi_token_pred: config.gaze_decoder_config.num_multi_token_pred.max(1),
            eos_token_id: config.gaze_decoder_config.eos_token_id,
        }
    }

    pub fn embed_video(
        &self,
        video: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Vec<Tensor<B, 5>>>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 5>>) {
        let [_batch, _time, channels, height, width] = video.shape().dims::<5>();
        let video =
            if channels != 3 || height != self.input_img_size || width != self.input_img_size {
                self.resize_video(video)
            } else {
                video
            };
        let (vision_features, new_past) =
            self.vision_model
                .forward(video, use_cache, past_conv_values);
        let vision_features = vision_features.swap_dims(1, 2).permute([0, 1, 3, 4, 2]);
        let [batch, time, height, width, dim] = vision_features.shape().dims::<5>();
        let vision_features = vision_features.reshape([batch, time, height * width, dim]);
        (self.connector.forward(vision_features), new_past)
    }

    pub fn prepare_video(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        self.resize_video(video)
    }

    pub fn generate(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub(crate) fn generate_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        let frames = video.shape().dims::<5>()[1];
        if frames > 1 {
            self.generate_cached_with_coverage_stop(
                video,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
        } else {
            self.generate_uncached_with_coverage_stop(
                video,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
        }
    }

    pub fn generate_cached(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_cached_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub(crate) fn generate_cached_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        let video = self.resize_video(video);
        let (video_embeds, _) = self.embed_video(video, false, None);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        let mut past_key_values: Option<AutoGazePastKeyValues<B>> = None;
        let mut pending_query_embeds: Option<Tensor<B, 3>> = None;
        let mut prefix_attention_mask = vec![vec![]; batch];
        let mut prefix_position_ids = vec![vec![]; batch];
        let mut pending_position_indices = vec![Vec::<usize>::new(); batch];
        let max_tokens =
            self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
        let cache_capacity = frames * (vision_tokens + max_tokens);

        for frame_idx in 0..frames {
            commit_pending_position_ids(
                &prefix_attention_mask,
                &mut prefix_position_ids,
                &pending_position_indices,
            );
            pending_position_indices.iter_mut().for_each(Vec::clear);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, frame_idx..(frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            for batch_idx in 0..batch {
                let valid_start = prefix_attention_mask[batch_idx]
                    .iter()
                    .filter(|&&mask| mask != 0)
                    .count() as i64;
                for valid_count in (valid_start..).take(vision_tokens) {
                    prefix_attention_mask[batch_idx].push(1);
                    prefix_position_ids[batch_idx].push(valid_count);
                }
            }
            let initial_query_embeds = if let Some(pending) = pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = prefix_attention_mask.first().map(Vec::len).unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let Some(query_embeds) = next_query_embeds.take() else {
                    break;
                };
                let query_len = query_embeds.shape().dims::<3>()[1];
                let query_start = cached_sequence_len(&past_key_values);
                let key_len = query_start + query_len;
                let attention_mask =
                    attention_mask_tensor_or_none::<B>(&prefix_attention_mask, key_len, &device);
                let position_ids = position_ids_slice_tensor_optimized::<B>(
                    &prefix_position_ids,
                    query_start,
                    query_len,
                    &device,
                );
                let outputs = self.gaze_decoder.forward_cached(
                    query_embeds,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    cache_capacity,
                );
                past_key_values = Some(outputs.past_key_values);
                let last_logits = outputs
                    .logits
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                );

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                next_query_embeds = Some(token_embeds);
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] = 0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            pending_position_indices = last_generated_indices;
            pending_query_embeds = next_query_embeds;
        }

        AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        }
    }

    pub fn generate_streaming_cached(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_streaming_cached_with_coverage_stop(
            video,
            cache,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub(crate) fn generate_streaming_cached_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        let video = self.resize_video(video);
        let [batch, frames, _channels, _height, _width] = video.shape().dims::<5>();
        let max_tokens =
            self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
        let horizon_frames = cache.horizon_frames.max(1);
        if cache
            .state
            .as_ref()
            .map(|state| !state.matches_runtime(batch, max_tokens, horizon_frames, frames))
            .unwrap_or(false)
        {
            cache.state = None;
        }

        let past_conv_values = cache
            .state
            .as_mut()
            .and_then(|state| state.past_conv_values.take());
        let (video_embeds, new_past_conv_values) = self.embed_video(video, true, past_conv_values);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);
        for output_frame_idx in 0..frames {
            let should_reset = cache
                .state
                .as_ref()
                .map(|state| !state.matches(batch, vision_tokens, dim, max_tokens, horizon_frames))
                .unwrap_or(true);
            if should_reset {
                cache.state = None;
            }
            let state = cache.state.get_or_insert_with(|| {
                AutoGazeStreamingCacheState::new(
                    batch,
                    vision_tokens,
                    dim,
                    max_tokens,
                    horizon_frames,
                )
            });

            state.commit_pending_position_ids();
            state.clear_pending_position_indices();
            state.compact_for_next_frame(vision_tokens + max_tokens, horizon_frames);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, output_frame_idx..(output_frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            state.append_frame_positions(vision_tokens);
            let initial_query_embeds = if let Some(pending) = state.pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = state
                .prefix_attention_mask
                .first()
                .map(Vec::len)
                .unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&state.prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let Some(query_embeds) = next_query_embeds.take() else {
                    break;
                };
                let query_len = query_embeds.shape().dims::<3>()[1];
                let query_start = cached_sequence_len(&state.past_key_values);
                let key_len = query_start + query_len;
                let attention_mask = attention_mask_tensor_or_none::<B>(
                    &state.prefix_attention_mask,
                    key_len,
                    &device,
                );
                let position_ids = position_ids_slice_tensor_optimized::<B>(
                    &state.prefix_position_ids,
                    query_start,
                    query_len,
                    &device,
                );
                let outputs = self.gaze_decoder.forward_cached(
                    query_embeds,
                    attention_mask,
                    position_ids,
                    state.past_key_values.take(),
                    state.cache_capacity,
                );
                state.past_key_values = Some(outputs.past_key_values);
                let last_logits = outputs
                    .logits
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                );

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(state.prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        state.prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        state.prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                next_query_embeds = Some(token_embeds);
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (output_frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        state.prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] =
                            0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            state.pending_position_indices = last_generated_indices;
            state.pending_query_embeds = next_query_embeds;
            state.record_completed_frame(vision_tokens + frame_count);
            state.processed_frames = state.processed_frames.saturating_add(1);
        }

        if let Some(state) = cache.state.as_mut() {
            state.past_conv_values = Some(new_past_conv_values);
        }

        AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        }
    }

    pub fn generate_uncached(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_uncached_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub(crate) fn generate_uncached_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        let video = self.resize_video(video);
        let (video_embeds, _) = self.embed_video(video, false, None);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        let mut prefix_embeds: Option<Tensor<B, 3>> = None;
        let mut prefix_attention_mask = vec![vec![]; batch];
        // Mirror upstream Transformers cached generation while using full-sequence Burn forwards.
        // Completed chunks keep cached RoPE positions; the final chunk is committed on the next frame.
        let mut prefix_position_ids = vec![vec![]; batch];
        let mut pending_position_indices = vec![Vec::<usize>::new(); batch];

        for frame_idx in 0..frames {
            commit_pending_position_ids(
                &prefix_attention_mask,
                &mut prefix_position_ids,
                &pending_position_indices,
            );
            pending_position_indices.iter_mut().for_each(Vec::clear);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, frame_idx..(frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            let mut sequence_embeds = if let Some(prefix) = prefix_embeds.take() {
                Tensor::cat(vec![prefix, frame_embed.clone()], 1)
            } else {
                frame_embed.clone()
            };
            for batch_idx in 0..batch {
                let valid_start = prefix_attention_mask[batch_idx]
                    .iter()
                    .filter(|&&mask| mask != 0)
                    .count() as i64;
                for valid_count in (valid_start..).take(vision_tokens) {
                    prefix_attention_mask[batch_idx].push(1);
                    prefix_position_ids[batch_idx].push(valid_count);
                }
            }

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let max_tokens =
                self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
            let generation_prefix_len = sequence_embeds.shape().dims::<3>()[1];
            let generation_tail_positions =
                generation_tail_positions(&prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let seq_len = sequence_embeds.shape().dims::<3>()[1];
                let attention_mask =
                    attention_mask_tensor_or_none::<B>(&prefix_attention_mask, seq_len, &device);
                let position_ids =
                    position_ids_tensor_optimized::<B>(&prefix_position_ids, seq_len, &device);
                let outputs = self.gaze_decoder.forward(
                    sequence_embeds.clone(),
                    attention_mask,
                    position_ids,
                );
                let last_logits = outputs
                    .logits
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                );

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);
                sequence_embeds = Tensor::cat(vec![sequence_embeds, token_embeds], 1);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] = 0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            pending_position_indices = last_generated_indices;
            prefix_embeds = Some(sequence_embeds);
        }

        AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        }
    }

    pub async fn generate_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_async_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub(crate) async fn generate_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        let frames = video.shape().dims::<5>()[1];
        if frames > 1 {
            self.generate_cached_async_with_coverage_stop(
                video,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
            .await
        } else {
            self.generate_uncached_async_with_coverage_stop(
                video,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
            .await
        }
    }

    pub async fn generate_cached_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_cached_async_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub(crate) async fn generate_cached_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        let video = self.resize_video(video);
        let (video_embeds, _) = self.embed_video(video, false, None);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        let mut past_key_values: Option<AutoGazePastKeyValues<B>> = None;
        let mut pending_query_embeds: Option<Tensor<B, 3>> = None;
        let mut prefix_attention_mask = vec![vec![]; batch];
        let mut prefix_position_ids = vec![vec![]; batch];
        let mut pending_position_indices = vec![Vec::<usize>::new(); batch];
        let max_tokens =
            self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
        let cache_capacity = frames * (vision_tokens + max_tokens);

        for frame_idx in 0..frames {
            commit_pending_position_ids(
                &prefix_attention_mask,
                &mut prefix_position_ids,
                &pending_position_indices,
            );
            pending_position_indices.iter_mut().for_each(Vec::clear);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, frame_idx..(frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            for batch_idx in 0..batch {
                let valid_start = prefix_attention_mask[batch_idx]
                    .iter()
                    .filter(|&&mask| mask != 0)
                    .count() as i64;
                for valid_count in (valid_start..).take(vision_tokens) {
                    prefix_attention_mask[batch_idx].push(1);
                    prefix_position_ids[batch_idx].push(valid_count);
                }
            }
            let initial_query_embeds = if let Some(pending) = pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = prefix_attention_mask.first().map(Vec::len).unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let Some(query_embeds) = next_query_embeds.take() else {
                    break;
                };
                let query_len = query_embeds.shape().dims::<3>()[1];
                let query_start = cached_sequence_len(&past_key_values);
                let key_len = query_start + query_len;
                let attention_mask =
                    attention_mask_tensor_or_none::<B>(&prefix_attention_mask, key_len, &device);
                let position_ids = position_ids_slice_tensor_optimized::<B>(
                    &prefix_position_ids,
                    query_start,
                    query_len,
                    &device,
                );
                let outputs = self.gaze_decoder.forward_cached(
                    query_embeds,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    cache_capacity,
                );
                past_key_values = Some(outputs.past_key_values);
                let last_logits = outputs
                    .logits
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens_async(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                )
                .await?;

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                next_query_embeds = Some(token_embeds);
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] = 0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            pending_position_indices = last_generated_indices;
            pending_query_embeds = next_query_embeds;
        }

        Ok(AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        })
    }

    pub async fn generate_streaming_cached_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_streaming_cached_async_with_coverage_stop(
            video,
            cache,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub(crate) async fn generate_streaming_cached_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        let video = self.resize_video(video);
        let [batch, frames, _channels, _height, _width] = video.shape().dims::<5>();
        let max_tokens =
            self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
        let horizon_frames = cache.horizon_frames.max(1);
        if cache
            .state
            .as_ref()
            .map(|state| !state.matches_runtime(batch, max_tokens, horizon_frames, frames))
            .unwrap_or(false)
        {
            cache.state = None;
        }

        let past_conv_values = cache
            .state
            .as_mut()
            .and_then(|state| state.past_conv_values.take());
        let (video_embeds, new_past_conv_values) = self.embed_video(video, true, past_conv_values);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        for output_frame_idx in 0..frames {
            let should_reset = cache
                .state
                .as_ref()
                .map(|state| !state.matches(batch, vision_tokens, dim, max_tokens, horizon_frames))
                .unwrap_or(true);
            if should_reset {
                cache.state = None;
            }
            let state = cache.state.get_or_insert_with(|| {
                AutoGazeStreamingCacheState::new(
                    batch,
                    vision_tokens,
                    dim,
                    max_tokens,
                    horizon_frames,
                )
            });

            state.commit_pending_position_ids();
            state.clear_pending_position_indices();
            state.compact_for_next_frame(vision_tokens + max_tokens, horizon_frames);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, output_frame_idx..(output_frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            state.append_frame_positions(vision_tokens);
            let initial_query_embeds = if let Some(pending) = state.pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = state
                .prefix_attention_mask
                .first()
                .map(Vec::len)
                .unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&state.prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let Some(query_embeds) = next_query_embeds.take() else {
                    break;
                };
                let query_len = query_embeds.shape().dims::<3>()[1];
                let query_start = cached_sequence_len(&state.past_key_values);
                let key_len = query_start + query_len;
                let attention_mask = attention_mask_tensor_or_none::<B>(
                    &state.prefix_attention_mask,
                    key_len,
                    &device,
                );
                let position_ids = position_ids_slice_tensor_optimized::<B>(
                    &state.prefix_position_ids,
                    query_start,
                    query_len,
                    &device,
                );
                let outputs = self.gaze_decoder.forward_cached(
                    query_embeds,
                    attention_mask,
                    position_ids,
                    state.past_key_values.take(),
                    state.cache_capacity,
                );
                state.past_key_values = Some(outputs.past_key_values);
                let last_logits = outputs
                    .logits
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, query_len.saturating_sub(1)..query_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens_async(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                )
                .await?;

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(state.prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        state.prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        state.prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                next_query_embeds = Some(token_embeds);
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (output_frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        state.prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] =
                            0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            state.pending_position_indices = last_generated_indices;
            state.pending_query_embeds = next_query_embeds;
            state.record_completed_frame(vision_tokens + frame_count);
            state.processed_frames = state.processed_frames.saturating_add(1);
        }

        if let Some(state) = cache.state.as_mut() {
            state.past_conv_values = Some(new_past_conv_values);
        }

        Ok(AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        })
    }

    pub(crate) async fn generate_streaming_cached_async_with_decode_strategy(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_strategy: AutoGazeDecodeStrategy,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        match decode_strategy.normalized() {
            AutoGazeDecodeStrategy::HostGreedy => {
                self.generate_streaming_cached_async_with_coverage_stop(
                    video,
                    cache,
                    max_gaze_tokens_each_frame,
                    task_loss_requirement,
                    coverage_stop_ratio,
                )
                .await
            }
            AutoGazeDecodeStrategy::DeviceGreedy { chunk_size } => {
                self.generate_streaming_cached_device_chunked_async_with_coverage_stop(
                    video,
                    cache,
                    max_gaze_tokens_each_frame,
                    task_loss_requirement,
                    coverage_stop_ratio,
                    chunk_size,
                )
                .await
            }
            AutoGazeDecodeStrategy::DeviceTerminalGreedy { chunk_size } => {
                self.generate_streaming_cached_device_terminal_async_with_coverage_stop(
                    video,
                    cache,
                    max_gaze_tokens_each_frame,
                    task_loss_requirement,
                    coverage_stop_ratio,
                    chunk_size,
                )
                .await
            }
        }
    }

    pub(crate) async fn generate_streaming_cached_device_output_async_with_decode_strategy(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_strategy: AutoGazeDecodeStrategy,
    ) -> Result<AutoGazeDeviceGenerateOutput<B>, ExecutionError> {
        match decode_strategy.normalized() {
            AutoGazeDecodeStrategy::HostGreedy => {
                let generated = self
                    .generate_streaming_cached_async_with_coverage_stop(
                        video,
                        cache,
                        max_gaze_tokens_each_frame,
                        task_loss_requirement,
                        coverage_stop_ratio,
                    )
                    .await?;
                Ok(AutoGazeDeviceGenerateOutput {
                    generated,
                    device_tokens: None,
                })
            }
            AutoGazeDecodeStrategy::DeviceGreedy { chunk_size } => {
                self.generate_streaming_cached_device_chunked_output_async_with_coverage_stop(
                    video,
                    cache,
                    DeviceChunkDecodeOptions {
                        max_gaze_tokens_each_frame,
                        task_loss_requirement,
                        coverage_stop_ratio,
                        decode_chunk_size: chunk_size,
                        collect_device_tokens: true,
                    },
                )
                .await
            }
            AutoGazeDecodeStrategy::DeviceTerminalGreedy { chunk_size } => {
                self.generate_streaming_cached_device_terminal_output_async_with_coverage_stop(
                    video,
                    cache,
                    DeviceChunkDecodeOptions {
                        max_gaze_tokens_each_frame,
                        task_loss_requirement,
                        coverage_stop_ratio,
                        decode_chunk_size: chunk_size,
                        collect_device_tokens: true,
                    },
                )
                .await
            }
        }
    }

    async fn generate_streaming_cached_device_chunked_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_chunk_size: usize,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_streaming_cached_device_chunked_output_async_with_coverage_stop(
            video,
            cache,
            DeviceChunkDecodeOptions {
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
                decode_chunk_size,
                collect_device_tokens: false,
            },
        )
        .await
        .map(|output| output.generated)
    }

    async fn generate_streaming_cached_device_chunked_output_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        options: DeviceChunkDecodeOptions,
    ) -> Result<AutoGazeDeviceGenerateOutput<B>, ExecutionError> {
        let video = self.resize_video(video);
        let [batch, frames, _channels, _height, _width] = video.shape().dims::<5>();
        let max_tokens = self.effective_max_gaze_tokens(
            options.max_gaze_tokens_each_frame,
            options.coverage_stop_ratio,
        );
        let horizon_frames = cache.horizon_frames.max(1);
        if cache
            .state
            .as_ref()
            .map(|state| !state.matches_runtime(batch, max_tokens, horizon_frames, frames))
            .unwrap_or(false)
        {
            cache.state = None;
        }

        let past_conv_values = cache
            .state
            .as_mut()
            .and_then(|state| state.past_conv_values.take());
        let (video_embeds, new_past_conv_values) = self.embed_video(video, true, past_conv_values);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();
        let decode_chunk_size = options.decode_chunk_size.max(1);

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);
        let mut device_token_chunks = options.collect_device_tokens.then(Vec::new);
        let mut device_valid_chunks = options.collect_device_tokens.then(Vec::new);
        let coverage_stop_ratio = options.coverage_stop_ratio;

        for output_frame_idx in 0..frames {
            let should_reset = cache
                .state
                .as_ref()
                .map(|state| !state.matches(batch, vision_tokens, dim, max_tokens, horizon_frames))
                .unwrap_or(true);
            if should_reset {
                cache.state = None;
            }
            let state = cache.state.get_or_insert_with(|| {
                AutoGazeStreamingCacheState::new(
                    batch,
                    vision_tokens,
                    dim,
                    max_tokens,
                    horizon_frames,
                )
            });

            state.commit_pending_position_ids();
            state.clear_pending_position_indices();
            state.compact_for_next_frame(vision_tokens + max_tokens, horizon_frames);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, output_frame_idx..(output_frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            state.append_frame_positions(vision_tokens);
            let initial_query_embeds = if let Some(pending) = state.pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = state
                .prefix_attention_mask
                .first()
                .map(Vec::len)
                .unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&state.prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);
            let mut disallowed = disallowed_token_mask::<B>(
                &frame_tokens,
                self.eos_token_id,
                self.gaze_decoder.vocab_size,
                &device,
            );
            let mut device_finished = finished_token_mask::<B>(&finished, &device);
            let vocab_range = vocab_range_tensor::<B>(batch, self.gaze_decoder.vocab_size, &device);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let current_len = frame_tokens.first().map(Vec::len).unwrap_or(0);
                let remaining_slots = max_tokens.saturating_sub(current_len);
                if remaining_slots == 0 {
                    break;
                }
                let requested_steps = decode_chunk_size
                    .min(remaining_slots.div_ceil(self.num_multi_token_pred.max(1)))
                    .max(1);
                let mut query_embeds = next_query_embeds.take();
                let mut token_steps = Vec::with_capacity(requested_steps);
                let mut valid_steps = Vec::with_capacity(requested_steps);
                let mut chunk_steps = Vec::with_capacity(requested_steps);

                for step_idx in 0..requested_steps {
                    let Some(query) = query_embeds.take() else {
                        break;
                    };
                    let query_len = query.shape().dims::<3>()[1];
                    let query_start = cached_sequence_len(&state.past_key_values);
                    let key_len = query_start + query_len;
                    let attention_mask = attention_mask_tensor_or_none::<B>(
                        &state.prefix_attention_mask,
                        key_len,
                        &device,
                    );
                    let position_ids = position_ids_slice_tensor_optimized::<B>(
                        &state.prefix_position_ids,
                        query_start,
                        query_len,
                        &device,
                    );
                    let outputs = self.gaze_decoder.forward_cached(
                        query,
                        attention_mask,
                        position_ids,
                        state.past_key_values.take(),
                        state.cache_capacity,
                    );
                    state.past_key_values = Some(outputs.past_key_values);
                    let last_logits = outputs
                        .logits
                        .slice_dim(1, query_len.saturating_sub(1)..query_len)
                        .reshape([
                            batch,
                            self.num_multi_token_pred,
                            self.gaze_decoder.vocab_size,
                        ]);
                    let last_task = outputs
                        .task_loss_prediction
                        .slice_dim(1, query_len.saturating_sub(1)..query_len)
                        .reshape([batch, self.num_multi_token_pred]);
                    let selected = device_select_multi_tokens(
                        last_logits,
                        last_task,
                        DeviceGreedySelectionState {
                            disallowed,
                            finished: device_finished,
                            vocab_range: vocab_range.clone(),
                            current_len: current_len + step_idx * self.num_multi_token_pred,
                        },
                        DeviceGreedySelectionConfig {
                            eos_token_id: self.eos_token_id,
                            max_tokens,
                            task_loss_stop: TaskLossStop {
                                requirement: options.task_loss_requirement,
                                is_first_token,
                            },
                        },
                    );
                    let next_device_finished = device_finished_after_selection_block(
                        selected.finished,
                        selected.valid.clone(),
                    );
                    let token_embeds = self
                        .gaze_decoder
                        .model
                        .embed_tokens
                        .forward(selected.tokens.clone());
                    let generated_indices = append_generated_position_slots(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &generation_tail_positions,
                        self.num_multi_token_pred,
                    );
                    token_steps.push(selected.tokens);
                    valid_steps.push(selected.valid);
                    disallowed = selected.disallowed;
                    device_finished = next_device_finished;
                    query_embeds = Some(token_embeds.clone());
                    chunk_steps.push(DeviceGreedyChunkStep {
                        token_embeds,
                        generated_indices,
                    });
                    is_first_token = false;
                }

                if chunk_steps.is_empty() {
                    break;
                }
                next_query_embeds = query_embeds;

                let generated_slots = chunk_steps.len() * self.num_multi_token_pred;
                let device_sequence =
                    options
                        .collect_device_tokens
                        .then(|| DeviceGreedyChunkTensors {
                            tokens: Tensor::cat(token_steps.clone(), 1),
                            valid: Tensor::cat(valid_steps.clone(), 1),
                        });
                let packed = pack_device_greedy_chunk(token_steps, valid_steps);
                let (selected_tokens, selected_valid, selected_confidences) =
                    read_device_greedy_chunk_async(packed, generated_slots, self.eos_token_id)
                        .await?;

                let mut actual_slots = 0usize;
                let mut actual_steps = 0usize;
                let mut last_actual_new_tokens = 0usize;
                for (step_idx, decode_step) in chunk_steps.iter().enumerate() {
                    if frame_tokens.first().map(Vec::len).unwrap_or(0) >= max_tokens
                        || finished.iter().all(|done| *done)
                    {
                        break;
                    }
                    let new_tokens = self
                        .num_multi_token_pred
                        .min(max_tokens - frame_tokens.first().map(Vec::len).unwrap_or(0));
                    let slot_offset = step_idx * self.num_multi_token_pred;
                    last_generated_indices = decode_step
                        .generated_indices
                        .iter()
                        .map(|indices| indices.iter().copied().take(new_tokens).collect())
                        .collect();

                    let mut finish_after_step = vec![false; batch];
                    for batch_idx in 0..batch {
                        let was_finished = finished[batch_idx];
                        for local_idx in 0..new_tokens {
                            let slot = slot_offset + local_idx;
                            let (token, valid, confidence) = if was_finished {
                                (self.eos_token_id, false, 0.0)
                            } else {
                                (
                                    selected_tokens[batch_idx][slot],
                                    selected_valid[batch_idx][slot],
                                    selected_confidences[batch_idx][slot],
                                )
                            };
                            frame_tokens[batch_idx].push(token);
                            frame_padded[batch_idx].push(!valid);
                            frame_confidences[batch_idx].push(confidence);
                            let coverage_stop = valid
                                && observe_generation_coverage(
                                    &mut coverage_trackers,
                                    batch_idx,
                                    token,
                                );
                            if !valid || coverage_stop {
                                finish_after_step[batch_idx] = true;
                            }
                        }
                    }
                    for (done, finish_after_step) in finished.iter_mut().zip(finish_after_step) {
                        *done |= finish_after_step;
                    }

                    actual_slots += new_tokens;
                    actual_steps = step_idx + 1;
                    last_actual_new_tokens = new_tokens;
                }

                if actual_slots == 0 {
                    truncate_generation_rows(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &mut last_generated_indices,
                        generated_slots,
                    );
                    truncate_past_key_values_tail(&mut state.past_key_values, generated_slots);
                } else if generated_slots > actual_slots {
                    truncate_generation_rows(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &mut last_generated_indices,
                        generated_slots - actual_slots,
                    );
                    truncate_past_key_values_tail(
                        &mut state.past_key_values,
                        generated_slots - actual_slots,
                    );
                }

                if let Some(device_sequence) = device_sequence
                    && let (Some(tokens), Some(valid)) =
                        (device_token_chunks.as_mut(), device_valid_chunks.as_mut())
                {
                    tokens.push(device_sequence.tokens.slice_dim(1, 0..actual_slots));
                    valid.push(device_sequence.valid.slice_dim(1, 0..actual_slots));
                }

                if actual_steps > 0 {
                    let last_step = &chunk_steps[actual_steps - 1];
                    next_query_embeds = Some(
                        last_step
                            .token_embeds
                            .clone()
                            .slice_dim(1, 0..last_actual_new_tokens),
                    );
                }
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (output_frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        state.prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] =
                            0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            state.pending_position_indices = last_generated_indices;
            state.pending_query_embeds = next_query_embeds;
            state.record_completed_frame(vision_tokens + frame_count);
            state.processed_frames = state.processed_frames.saturating_add(1);
        }

        if let Some(state) = cache.state.as_mut() {
            state.past_conv_values = Some(new_past_conv_values);
        }

        let generated = AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        };
        let device_tokens = match (device_token_chunks, device_valid_chunks) {
            (Some(tokens), Some(valid)) if !tokens.is_empty() => Some(AutoGazeDeviceTokens {
                tokens: Tensor::cat(tokens, 1),
                valid: Tensor::cat(valid, 1),
            }),
            _ => None,
        };

        Ok(AutoGazeDeviceGenerateOutput {
            generated,
            device_tokens,
        })
    }

    async fn generate_streaming_cached_device_terminal_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_chunk_size: usize,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_streaming_cached_device_terminal_output_async_with_coverage_stop(
            video,
            cache,
            DeviceChunkDecodeOptions {
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
                decode_chunk_size,
                collect_device_tokens: false,
            },
        )
        .await
        .map(|output| output.generated)
    }

    async fn generate_streaming_cached_device_terminal_output_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        options: DeviceChunkDecodeOptions,
    ) -> Result<AutoGazeDeviceGenerateOutput<B>, ExecutionError> {
        let video = self.resize_video(video);
        let [batch, frames, _channels, _height, _width] = video.shape().dims::<5>();
        let max_tokens = self.effective_max_gaze_tokens(
            options.max_gaze_tokens_each_frame,
            options.coverage_stop_ratio,
        );
        let horizon_frames = cache.horizon_frames.max(1);
        if cache
            .state
            .as_ref()
            .map(|state| !state.matches_runtime(batch, max_tokens, horizon_frames, frames))
            .unwrap_or(false)
        {
            cache.state = None;
        }

        let past_conv_values = cache
            .state
            .as_mut()
            .and_then(|state| state.past_conv_values.take());
        let (video_embeds, new_past_conv_values) = self.embed_video(video, true, past_conv_values);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();
        let decode_chunk_size = options.decode_chunk_size.max(1);

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);
        let mut device_token_chunks = options.collect_device_tokens.then(Vec::new);
        let mut device_valid_chunks = options.collect_device_tokens.then(Vec::new);
        let coverage_stop_ratio = options.coverage_stop_ratio;

        for output_frame_idx in 0..frames {
            let should_reset = cache
                .state
                .as_ref()
                .map(|state| !state.matches(batch, vision_tokens, dim, max_tokens, horizon_frames))
                .unwrap_or(true);
            if should_reset {
                cache.state = None;
            }
            let state = cache.state.get_or_insert_with(|| {
                AutoGazeStreamingCacheState::new(
                    batch,
                    vision_tokens,
                    dim,
                    max_tokens,
                    horizon_frames,
                )
            });

            state.commit_pending_position_ids();
            state.clear_pending_position_indices();
            state.compact_for_next_frame(vision_tokens + max_tokens, horizon_frames);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, output_frame_idx..(output_frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            state.append_frame_positions(vision_tokens);
            let initial_query_embeds = if let Some(pending) = state.pending_query_embeds.take() {
                Tensor::cat(vec![pending, frame_embed], 1)
            } else {
                frame_embed
            };

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let generation_prefix_len = state
                .prefix_attention_mask
                .first()
                .map(Vec::len)
                .unwrap_or(0);
            let generation_tail_positions =
                generation_tail_positions(&state.prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];
            let mut next_query_embeds = Some(initial_query_embeds);

            let mut disallowed = disallowed_token_mask::<B>(
                &frame_tokens,
                self.eos_token_id,
                self.gaze_decoder.vocab_size,
                &device,
            );
            let mut device_finished = finished_token_mask::<B>(&finished, &device);
            let vocab_range = vocab_range_tensor::<B>(batch, self.gaze_decoder.vocab_size, &device);
            let mut token_steps = Vec::new();
            let mut valid_steps = Vec::new();
            let mut decode_steps = Vec::new();

            while token_steps.len() * self.num_multi_token_pred < max_tokens {
                let current_len = token_steps.len() * self.num_multi_token_pred;
                let remaining_slots = max_tokens.saturating_sub(current_len);
                let requested_steps = decode_chunk_size
                    .min(remaining_slots.div_ceil(self.num_multi_token_pred.max(1)))
                    .max(1);
                let mut query_embeds = next_query_embeds.take();
                let steps_before_chunk = decode_steps.len();

                for step_idx in 0..requested_steps {
                    let Some(query) = query_embeds.take() else {
                        break;
                    };
                    let query_len = query.shape().dims::<3>()[1];
                    let query_start = cached_sequence_len(&state.past_key_values);
                    let key_len = query_start + query_len;
                    let attention_mask = attention_mask_tensor_or_none::<B>(
                        &state.prefix_attention_mask,
                        key_len,
                        &device,
                    );
                    let position_ids = position_ids_slice_tensor_optimized::<B>(
                        &state.prefix_position_ids,
                        query_start,
                        query_len,
                        &device,
                    );
                    let outputs = self.gaze_decoder.forward_cached(
                        query,
                        attention_mask,
                        position_ids,
                        state.past_key_values.take(),
                        state.cache_capacity,
                    );
                    state.past_key_values = Some(outputs.past_key_values);
                    let last_logits = outputs
                        .logits
                        .slice_dim(1, query_len.saturating_sub(1)..query_len)
                        .reshape([
                            batch,
                            self.num_multi_token_pred,
                            self.gaze_decoder.vocab_size,
                        ]);
                    let last_task = outputs
                        .task_loss_prediction
                        .slice_dim(1, query_len.saturating_sub(1)..query_len)
                        .reshape([batch, self.num_multi_token_pred]);
                    let selected = device_select_multi_tokens(
                        last_logits,
                        last_task,
                        DeviceGreedySelectionState {
                            disallowed,
                            finished: device_finished,
                            vocab_range: vocab_range.clone(),
                            current_len: current_len + step_idx * self.num_multi_token_pred,
                        },
                        DeviceGreedySelectionConfig {
                            eos_token_id: self.eos_token_id,
                            max_tokens,
                            task_loss_stop: TaskLossStop {
                                requirement: options.task_loss_requirement,
                                is_first_token,
                            },
                        },
                    );
                    let next_device_finished = device_finished_after_selection_block(
                        selected.finished,
                        selected.valid.clone(),
                    );
                    let token_embeds = self
                        .gaze_decoder
                        .model
                        .embed_tokens
                        .forward(selected.tokens.clone());
                    let generated_indices = append_generated_position_slots(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &generation_tail_positions,
                        self.num_multi_token_pred,
                    );
                    token_steps.push(selected.tokens);
                    valid_steps.push(selected.valid);
                    disallowed = selected.disallowed;
                    device_finished = next_device_finished;
                    query_embeds = Some(token_embeds.clone());
                    decode_steps.push(DeviceGreedyChunkStep {
                        token_embeds,
                        generated_indices,
                    });
                    is_first_token = false;
                }

                if decode_steps.len() == steps_before_chunk {
                    break;
                }
                next_query_embeds = query_embeds;
            }

            let generated_slots = decode_steps.len() * self.num_multi_token_pred;
            let device_sequence = options
                .collect_device_tokens
                .then(|| DeviceGreedyChunkTensors {
                    tokens: Tensor::cat(token_steps.clone(), 1),
                    valid: Tensor::cat(valid_steps.clone(), 1),
                });
            if generated_slots > 0 {
                let packed = pack_device_greedy_chunk(token_steps, valid_steps);
                let (selected_tokens, selected_valid, selected_confidences) =
                    read_device_greedy_chunk_async(packed, generated_slots, self.eos_token_id)
                        .await?;

                let mut actual_slots = 0usize;
                let mut actual_steps = 0usize;
                let mut last_actual_new_tokens = 0usize;
                for (step_idx, decode_step) in decode_steps.iter().enumerate() {
                    if frame_tokens.first().map(Vec::len).unwrap_or(0) >= max_tokens
                        || finished.iter().all(|done| *done)
                    {
                        break;
                    }
                    let new_tokens = self
                        .num_multi_token_pred
                        .min(max_tokens - frame_tokens.first().map(Vec::len).unwrap_or(0));
                    let slot_offset = step_idx * self.num_multi_token_pred;
                    last_generated_indices = decode_step
                        .generated_indices
                        .iter()
                        .map(|indices| indices.iter().copied().take(new_tokens).collect())
                        .collect();

                    let mut finish_after_step = vec![false; batch];
                    for batch_idx in 0..batch {
                        let was_finished = finished[batch_idx];
                        for local_idx in 0..new_tokens {
                            let slot = slot_offset + local_idx;
                            let (token, valid, confidence) = if was_finished {
                                (self.eos_token_id, false, 0.0)
                            } else {
                                (
                                    selected_tokens[batch_idx][slot],
                                    selected_valid[batch_idx][slot],
                                    selected_confidences[batch_idx][slot],
                                )
                            };
                            frame_tokens[batch_idx].push(token);
                            frame_padded[batch_idx].push(!valid);
                            frame_confidences[batch_idx].push(confidence);
                            let coverage_stop = valid
                                && observe_generation_coverage(
                                    &mut coverage_trackers,
                                    batch_idx,
                                    token,
                                );
                            if !valid || coverage_stop {
                                finish_after_step[batch_idx] = true;
                            }
                        }
                    }
                    for (done, finish_after_step) in finished.iter_mut().zip(finish_after_step) {
                        *done |= finish_after_step;
                    }

                    actual_slots += new_tokens;
                    actual_steps = step_idx + 1;
                    last_actual_new_tokens = new_tokens;
                }

                if actual_slots == 0 {
                    truncate_generation_rows(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &mut last_generated_indices,
                        generated_slots,
                    );
                    truncate_past_key_values_tail(&mut state.past_key_values, generated_slots);
                } else if generated_slots > actual_slots {
                    truncate_generation_rows(
                        &mut state.prefix_attention_mask,
                        &mut state.prefix_position_ids,
                        &mut last_generated_indices,
                        generated_slots - actual_slots,
                    );
                    truncate_past_key_values_tail(
                        &mut state.past_key_values,
                        generated_slots - actual_slots,
                    );
                }

                if let Some(device_sequence) = device_sequence
                    && let (Some(tokens), Some(valid)) =
                        (device_token_chunks.as_mut(), device_valid_chunks.as_mut())
                {
                    tokens.push(device_sequence.tokens.slice_dim(1, 0..actual_slots));
                    valid.push(device_sequence.valid.slice_dim(1, 0..actual_slots));
                }

                if actual_steps > 0 {
                    let last_step = &decode_steps[actual_steps - 1];
                    next_query_embeds = Some(
                        last_step
                            .token_embeds
                            .clone()
                            .slice_dim(1, 0..last_actual_new_tokens),
                    );
                }
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (output_frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        state.prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] =
                            0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            state.pending_position_indices = last_generated_indices;
            state.pending_query_embeds = next_query_embeds;
            state.record_completed_frame(vision_tokens + frame_count);
            state.processed_frames = state.processed_frames.saturating_add(1);
        }

        if let Some(state) = cache.state.as_mut() {
            state.past_conv_values = Some(new_past_conv_values);
        }

        let generated = AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        };
        let device_tokens = match (device_token_chunks, device_valid_chunks) {
            (Some(tokens), Some(valid)) if !tokens.is_empty() => Some(AutoGazeDeviceTokens {
                tokens: Tensor::cat(tokens, 1),
                valid: Tensor::cat(valid, 1),
            }),
            _ => None,
        };

        Ok(AutoGazeDeviceGenerateOutput {
            generated,
            device_tokens,
        })
    }

    pub async fn generate_uncached_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_uncached_async_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub(crate) async fn generate_uncached_async_with_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        let video = self.resize_video(video);
        let (video_embeds, _) = self.embed_video(video, false, None);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        let mut prefix_embeds: Option<Tensor<B, 3>> = None;
        let mut prefix_attention_mask = vec![vec![]; batch];
        // Keep the async path position-compatible with the sync upstream-generation emulation.
        let mut prefix_position_ids = vec![vec![]; batch];
        let mut pending_position_indices = vec![Vec::<usize>::new(); batch];

        for frame_idx in 0..frames {
            commit_pending_position_ids(
                &prefix_attention_mask,
                &mut prefix_position_ids,
                &pending_position_indices,
            );
            pending_position_indices.iter_mut().for_each(Vec::clear);

            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, frame_idx..(frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            let mut sequence_embeds = if let Some(prefix) = prefix_embeds.take() {
                Tensor::cat(vec![prefix, frame_embed.clone()], 1)
            } else {
                frame_embed.clone()
            };
            for batch_idx in 0..batch {
                let valid_start = prefix_attention_mask[batch_idx]
                    .iter()
                    .filter(|&&mask| mask != 0)
                    .count() as i64;
                for valid_count in (valid_start..).take(vision_tokens) {
                    prefix_attention_mask[batch_idx].push(1);
                    prefix_position_ids[batch_idx].push(valid_count);
                }
            }

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let mut coverage_trackers =
                generation_coverage_trackers(batch, coverage_stop_ratio, &self.scale_layouts);
            let mut is_first_token = true;
            let max_tokens =
                self.effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio);
            let generation_prefix_len = sequence_embeds.shape().dims::<3>()[1];
            let generation_tail_positions =
                generation_tail_positions(&prefix_position_ids, self.num_multi_token_pred);
            let mut last_generated_indices = vec![Vec::<usize>::new(); batch];

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let seq_len = sequence_embeds.shape().dims::<3>()[1];
                let attention_mask =
                    attention_mask_tensor_or_none::<B>(&prefix_attention_mask, seq_len, &device);
                let position_ids =
                    position_ids_tensor_optimized::<B>(&prefix_position_ids, seq_len, &device);
                let outputs = self.gaze_decoder.forward(
                    sequence_embeds.clone(),
                    attention_mask,
                    position_ids,
                );
                let last_logits = outputs
                    .logits
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens_async(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                    TaskLossStop {
                        requirement: task_loss_requirement,
                        is_first_token,
                    },
                )
                .await?;

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);
                sequence_embeds = Tensor::cat(vec![sequence_embeds, token_embeds], 1);

                for batch_idx in 0..batch {
                    last_generated_indices[batch_idx].clear();
                    for local_idx in 0..new_tokens {
                        last_generated_indices[batch_idx]
                            .push(prefix_attention_mask[batch_idx].len());
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        prefix_attention_mask[batch_idx].push(1);
                        let tail = &generation_tail_positions[batch_idx];
                        prefix_position_ids[batch_idx].push(tail[local_idx % tail.len()]);
                        let coverage_stop = valid
                            && observe_generation_coverage(
                                &mut coverage_trackers,
                                batch_idx,
                                token,
                            );
                        if !valid || coverage_stop {
                            finished[batch_idx] = true;
                        }
                    }
                }
                is_first_token = false;
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                for (local_idx, padded) in frame_padded[batch_idx].iter().copied().enumerate() {
                    if padded {
                        prefix_attention_mask[batch_idx][generation_prefix_len + local_idx] = 0;
                    }
                }
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            pending_position_indices = last_generated_indices;
            prefix_embeds = Some(sequence_embeds);
        }

        Ok(AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        })
    }

    fn resize_video(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, time, channels, height, width] = video.shape().dims::<5>();
        let device = video.device();
        let video = adapt_video_channels(video, channels, &device);
        if height == self.input_img_size && width == self.input_img_size {
            return video;
        }
        let video = video.reshape([batch * time, 3, height, width]);
        let video = interpolate(
            video,
            [self.input_img_size, self.input_img_size],
            InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
        );
        video.reshape([batch, time, 3, self.input_img_size, self.input_img_size])
    }

    pub fn effective_max_gaze_tokens(
        &self,
        max_gaze_tokens_each_frame: usize,
        coverage_stop_ratio: Option<f64>,
    ) -> usize {
        effective_generation_max_tokens(
            max_gaze_tokens_each_frame.max(1),
            coverage_stop_ratio,
            &self.scale_layouts,
            self.num_multi_token_pred,
        )
    }
}

#[derive(Module, Debug)]
pub struct NativeAutoGazeModel<B: Backend> {
    pub gazing_model: AutoGazeGazingModel<B>,
    #[module(skip)]
    pub config: AutoGazeConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoGazeLoadOptions {
    pub allow_partial: bool,
    pub validate: bool,
}

impl AutoGazeLoadOptions {
    pub const fn strict() -> Self {
        Self {
            allow_partial: false,
            validate: true,
        }
    }

    pub const fn permissive() -> Self {
        Self {
            allow_partial: true,
            validate: false,
        }
    }

    pub const fn with_allow_partial(mut self, allow_partial: bool) -> Self {
        self.allow_partial = allow_partial;
        self
    }

    pub const fn with_validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

impl Default for AutoGazeLoadOptions {
    fn default() -> Self {
        Self::strict()
    }
}

impl<B: Backend> NativeAutoGazeModel<B> {
    pub fn new(config: &AutoGazeConfig, device: &B::Device) -> Self {
        Self {
            gazing_model: AutoGazeGazingModel::new_with_scale_layouts(
                &config.gaze_model_config,
                scale_token_layouts(config),
                device,
            ),
            config: config.clone(),
        }
    }

    pub fn load(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        Self::from_hf_dir(dir, device)
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        Self::from_hf_dir_with_options(dir, device, AutoGazeLoadOptions::strict())
    }

    pub fn from_hf_dir_with_options(
        dir: impl AsRef<Path>,
        device: &B::Device,
        options: AutoGazeLoadOptions,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let config = AutoGazeConfig::from_json_file(dir.join("config.json"))?;
        Self::from_config_and_safetensors_file(
            &config,
            dir.join("model.safetensors"),
            device,
            options,
        )
    }

    pub fn from_config_and_safetensors_file(
        config: &AutoGazeConfig,
        path: impl Into<std::path::PathBuf>,
        device: &B::Device,
        options: AutoGazeLoadOptions,
    ) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(options.allow_partial)
            .validate(options.validate);
        model.load_safetensors_store(&mut store, options)?;
        Ok(model)
    }

    pub fn from_config_and_safetensors_bytes(
        config: &AutoGazeConfig,
        bytes: Vec<u8>,
        device: &B::Device,
        options: AutoGazeLoadOptions,
    ) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = SafetensorsStore::from_bytes(Some(bytes))
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(options.allow_partial)
            .validate(options.validate);
        model.load_safetensors_store(&mut store, options)?;
        Ok(model)
    }

    fn load_safetensors_store(
        &mut self,
        store: &mut SafetensorsStore,
        options: AutoGazeLoadOptions,
    ) -> Result<()> {
        let result = self
            .load_from(store)
            .context("load AutoGaze safetensors weights")?;
        if !options.allow_partial && !result.errors.is_empty() {
            bail!("failed to apply AutoGaze weights: {:?}", result.errors);
        }
        Ok(())
    }

    pub fn into_pipeline(self) -> crate::AutoGazePipeline<B> {
        crate::AutoGazePipeline::new(self)
    }

    pub fn generate(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
    ) -> AutoGazeGenerateOutput {
        self.gazing_model
            .generate(video, max_gaze_tokens_each_frame, None)
    }

    pub fn generate_with_task_loss_requirement(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_with_task_loss_requirement_and_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub fn generate_with_task_loss_requirement_and_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        self.gazing_model.generate_with_coverage_stop(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            coverage_stop_ratio,
        )
    }

    pub async fn generate_with_task_loss_requirement_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_with_task_loss_requirement_and_coverage_stop_async(
            video,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub async fn generate_with_task_loss_requirement_and_coverage_stop_async(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.gazing_model
            .generate_async_with_coverage_stop(
                video,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
            .await
    }

    pub fn generate_streaming_with_task_loss_requirement(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> AutoGazeGenerateOutput {
        self.generate_streaming_with_task_loss_requirement_and_coverage_stop(
            video,
            cache,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
    }

    pub fn generate_streaming_with_task_loss_requirement_and_coverage_stop(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> AutoGazeGenerateOutput {
        self.gazing_model
            .generate_streaming_cached_with_coverage_stop(
                video,
                cache,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
    }

    pub async fn generate_streaming_with_task_loss_requirement_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_streaming_with_task_loss_requirement_and_coverage_stop_async(
            video,
            cache,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
            None,
        )
        .await
    }

    pub async fn generate_streaming_with_task_loss_requirement_and_coverage_stop_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.gazing_model
            .generate_streaming_cached_async_with_coverage_stop(
                video,
                cache,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
            )
            .await
    }

    pub async fn generate_streaming_with_decode_strategy_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_strategy: AutoGazeDecodeStrategy,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.gazing_model
            .generate_streaming_cached_async_with_decode_strategy(
                video,
                cache,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
                decode_strategy,
            )
            .await
    }

    pub async fn generate_streaming_device_output_with_decode_strategy_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
        coverage_stop_ratio: Option<f64>,
        decode_strategy: AutoGazeDecodeStrategy,
    ) -> Result<AutoGazeDeviceGenerateOutput<B>, ExecutionError> {
        self.gazing_model
            .generate_streaming_cached_device_output_async_with_decode_strategy(
                video,
                cache,
                max_gaze_tokens_each_frame,
                task_loss_requirement,
                coverage_stop_ratio,
                decode_strategy,
            )
            .await
    }

    pub fn default_max_gaze_tokens_each_frame(&self) -> usize {
        self.config
            .inference_gazing_ratio()
            .map(|ratio| {
                (ratio.clamp(0.0, 1.0) * self.config.num_vision_tokens_each_frame.max(1) as f32)
                    .floor()
                    .max(1.0) as usize
            })
            .unwrap_or_else(|| self.gazing_model.num_multi_token_pred.max(1))
    }

    pub fn default_task_loss_requirement(&self) -> Option<f32> {
        self.config.inference_task_loss_requirement()
    }

    pub fn effective_max_gaze_tokens_each_frame(
        &self,
        max_gaze_tokens_each_frame: usize,
        coverage_stop_ratio: Option<f64>,
    ) -> usize {
        self.gazing_model
            .effective_max_gaze_tokens(max_gaze_tokens_each_frame, coverage_stop_ratio)
    }

    pub fn infer(&self, video: Tensor<B, 5>) -> AutoGazeGenerateOutput {
        self.generate_with_task_loss_requirement(
            video,
            self.default_max_gaze_tokens_each_frame(),
            self.default_task_loss_requirement(),
        )
    }

    pub async fn infer_async(
        &self,
        video: Tensor<B, 5>,
    ) -> Result<AutoGazeGenerateOutput, ExecutionError> {
        self.generate_with_task_loss_requirement_async(
            video,
            self.default_max_gaze_tokens_each_frame(),
            self.default_task_loss_requirement(),
        )
        .await
    }

    pub fn trace_video(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
    ) -> Vec<FrameFixationTrace> {
        self.trace_video_with_task_loss_requirement(
            video,
            k,
            max_gaze_tokens_each_frame,
            self.default_task_loss_requirement(),
        )
    }

    pub fn trace_video_with_task_loss_requirement(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Vec<FrameFixationTrace> {
        let trace_budget = max_gaze_tokens_each_frame.max(k.max(1));
        let generated =
            self.generate_with_task_loss_requirement(video, trace_budget, task_loss_requirement);
        generated_to_traces(&generated, &self.config, trace_budget)
    }

    pub async fn trace_video_with_task_loss_requirement_async(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<Vec<FrameFixationTrace>, ExecutionError> {
        let trace_budget = max_gaze_tokens_each_frame.max(k.max(1));
        let generated = self
            .generate_with_task_loss_requirement_async(video, trace_budget, task_loss_requirement)
            .await?;
        Ok(generated_to_traces(&generated, &self.config, trace_budget))
    }

    pub fn trace_streaming_with_task_loss_requirement(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Vec<FrameFixationTrace> {
        let trace_budget = max_gaze_tokens_each_frame.max(k.max(1));
        let generated = self.generate_streaming_with_task_loss_requirement(
            video,
            cache,
            trace_budget,
            task_loss_requirement,
        );
        generated_to_traces(&generated, &self.config, trace_budget)
    }

    pub async fn trace_streaming_with_task_loss_requirement_async(
        &self,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
        task_loss_requirement: Option<f32>,
    ) -> Result<Vec<FrameFixationTrace>, ExecutionError> {
        let trace_budget = max_gaze_tokens_each_frame.max(k.max(1));
        let generated = self
            .generate_streaming_with_task_loss_requirement_async(
                video,
                cache,
                trace_budget,
                task_loss_requirement,
            )
            .await?;
        Ok(generated_to_traces(&generated, &self.config, trace_budget))
    }

    pub fn trace_clip_from_frames(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace {
        let device = self.gazing_model.connector.pos_embed.val().device();
        let clip = Tensor::<B, 5>::from_data(
            TensorData::new(
                frames.to_vec(),
                [
                    1,
                    clip_len.max(1),
                    channels.max(1),
                    height.max(1),
                    width.max(1),
                ],
            ),
            &device,
        );
        self.trace_video(
            clip,
            k,
            self.gazing_model.num_multi_token_pred.max(k.max(1)),
        )
        .into_iter()
        .next()
        .unwrap_or_else(|| FrameFixationTrace::new(vec![]))
    }
}

impl<B: Backend> crate::AutoGazeTeacher for NativeAutoGazeModel<B> {
    fn trace_clip(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace {
        self.trace_clip_from_frames(frames, clip_len, channels, height, width, k)
    }
}
