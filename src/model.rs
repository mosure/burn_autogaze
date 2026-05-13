use crate::config::{AutoGazeConfig, ConnectorConfig, GazeModelConfig, VisionModelConfig};
use crate::{FixationPoint, FixationSet, FrameFixationTrace};
use anyhow::{Context, Result, bail};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig3d,
};
#[cfg(not(feature = "cuda"))]
use burn::tensor::Bool;
use burn::tensor::activation;
use burn::tensor::backend::{Backend, ExecutionError};
#[cfg(not(feature = "cuda"))]
use burn::tensor::module::attention;
use burn::tensor::module::interpolate;
#[cfg(not(feature = "cuda"))]
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::collections::VecDeque;
use std::path::Path;

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
    scale_layouts: Vec<ScaleTokenLayout>,
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
            vec![ScaleTokenLayout {
                token_count,
                grid: square_grid(token_count),
            }],
            device,
        )
    }

    pub(crate) fn new_with_scale_layouts(
        config: &GazeModelConfig,
        scale_layouts: Vec<ScaleTokenLayout>,
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

fn adapt_video_channels<B: Backend>(
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

fn split_heads<B: Backend>(tokens: Tensor<B, 3>, heads: usize, head_dim: usize) -> Tensor<B, 4> {
    let [batch, seq, _] = tokens.shape().dims::<3>();
    tokens
        .reshape([batch, seq, heads.max(1), head_dim.max(1)])
        .swap_dims(1, 2)
}

fn merge_heads<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = tokens.shape().dims::<4>();
    tokens
        .swap_dims(1, 2)
        .reshape([batch, seq, heads * head_dim])
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dim = x.shape().dims::<4>()[3];
    let half = dim / 2;
    let x1 = x.clone().slice_dim(3, 0..half);
    let x2 = x.slice_dim(3, half..dim);
    Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3)
}

#[derive(Clone, Copy)]
struct CausalAttentionShape {
    batch: usize,
    heads: usize,
    query_len: usize,
    key_len: usize,
    past_len: usize,
    head_dim: usize,
}

#[cfg(feature = "cuda")]
fn causal_attention<B: Backend>(
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

#[cfg(not(feature = "cuda"))]
fn causal_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    attention_mask: Option<Tensor<B, 2, Int>>,
    shape: CausalAttentionShape,
) -> Tensor<B, 4> {
    let _ = shape.past_len;
    let mask = attention_mask.map(|mask| {
        key_padding_attention_mask(
            mask,
            shape.batch,
            shape.heads,
            shape.query_len,
            shape.key_len,
        )
    });
    attention(
        q,
        k,
        v,
        mask,
        None,
        causal_attention_options(shape.head_dim),
    )
}

#[cfg(not(feature = "cuda"))]
fn causal_attention_options(head_dim: usize) -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: Some(1.0 / (head_dim.max(1) as f64).sqrt()),
        softcap: None,
        is_causal: true,
    }
}

#[cfg(not(feature = "cuda"))]
fn key_padding_attention_mask<B: Backend>(
    mask: Tensor<B, 2, Int>,
    batch: usize,
    heads: usize,
    query_len: usize,
    key_len: usize,
) -> Tensor<B, 4, Bool> {
    mask.equal_elem(0)
        .reshape([batch.max(1), 1, 1, key_len])
        .repeat_dim(1, heads.max(1))
        .repeat_dim(2, query_len)
}

#[cfg(feature = "cuda")]
fn causal_attention_bias_for_query<B: Backend>(
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

fn llama3_inv_freq<B: Backend>(
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

fn attention_mask_tensor<B: Backend>(
    mask_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = mask_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in mask_rows {
        values.extend(row.iter().copied().take(seq_len));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, seq_len]), device)
}

fn attention_mask_tensor_or_none<B: Backend>(
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

fn attention_mask_rows_are_all_valid(mask_rows: &[Vec<i64>], seq_len: usize) -> bool {
    !mask_rows.is_empty()
        && mask_rows
            .iter()
            .all(|row| row.len() >= seq_len && row.iter().take(seq_len).all(|mask| *mask != 0))
}

#[cfg(test)]
fn position_ids_tensor<B: Backend>(
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

fn position_ids_tensor_optimized<B: Backend>(
    position_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    position_ids_slice_tensor_optimized(position_rows, 0, seq_len, device)
}

fn position_ids_slice_tensor<B: Backend>(
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

fn position_ids_slice_tensor_optimized<B: Backend>(
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

fn contiguous_position_start(position_rows: &[Vec<i64>], start: usize, len: usize) -> Option<i64> {
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

fn identical_position_slice(
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

fn cached_sequence_len<B: Backend>(past_key_values: &Option<AutoGazePastKeyValues<B>>) -> usize {
    past_key_values
        .as_ref()
        .and_then(|past| past.first())
        .map(|past| past.len)
        .unwrap_or(0)
}

fn compact_past_key_values<B: Backend>(
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

fn generation_tail_positions(
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

fn commit_pending_position_ids(
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

fn commit_pending_position_ids_with_offsets(
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

fn next_valid_position_with_offset(mask_row: &[i64], position_offset: i64) -> i64 {
    let valid_count = mask_row.iter().filter(|mask| **mask != 0).count() as i64;
    position_offset.saturating_add(valid_count)
}

fn greedy_select_multi_tokens<B: Backend>(
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

async fn greedy_select_multi_tokens_async<B: Backend>(
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

fn greedy_select_multi_tokens_from_packed_data(
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
struct GreedySelectionContext<'a> {
    prior_tokens: &'a [Vec<i64>],
    finished: &'a [bool],
    eos_token_id: i64,
    max_tokens: usize,
    task_loss_stop: TaskLossStop,
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
fn greedy_select_multi_tokens_from_data(
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

fn generated_to_traces(
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
    scale_layouts: &[ScaleTokenLayout],
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

fn generated_scale_token_masks(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScaleTokenLayout {
    token_count: usize,
    grid: usize,
}

#[derive(Clone, Debug)]
struct GenerationCoverageTracker {
    scale_layouts: Vec<ScaleTokenLayout>,
    grid: usize,
    covered: Vec<bool>,
    covered_count: usize,
    stop_cells: usize,
}

impl GenerationCoverageTracker {
    fn new(scale_layouts: &[ScaleTokenLayout], stop_ratio: f64) -> Option<Self> {
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

    fn observe_token(&mut self, token: i64) -> bool {
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

fn generation_coverage_trackers(
    batch: usize,
    stop_ratio: Option<f64>,
    scale_layouts: &[ScaleTokenLayout],
) -> Option<Vec<GenerationCoverageTracker>> {
    let tracker = GenerationCoverageTracker::new(scale_layouts, stop_ratio?)?;
    Some(vec![tracker; batch])
}

fn observe_generation_coverage(
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

fn effective_generation_max_tokens(
    configured_max_tokens: usize,
    coverage_stop_ratio: Option<f64>,
    scale_layouts: &[ScaleTokenLayout],
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

fn normalize_scale_layouts(
    mut layouts: Vec<ScaleTokenLayout>,
    fallback_token_count: usize,
) -> Vec<ScaleTokenLayout> {
    layouts.retain(|layout| layout.token_count > 0);
    if layouts.is_empty() {
        let token_count = fallback_token_count.max(1);
        return vec![ScaleTokenLayout {
            token_count,
            grid: square_grid(token_count),
        }];
    }

    for layout in &mut layouts {
        layout.grid = layout.grid.max(1);
    }
    layouts
}

fn coverage_grid_for_layouts(layouts: &[ScaleTokenLayout]) -> usize {
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

fn scale_token_index(token: usize, scale_layouts: &[ScaleTokenLayout]) -> Option<(usize, usize)> {
    let mut offset = 0usize;
    for (scale_idx, layout) in scale_layouts.iter().enumerate() {
        if token < offset + layout.token_count {
            return Some((scale_idx, token - offset));
        }
        offset += layout.token_count;
    }
    None
}

fn token_to_fixation_point(
    token: usize,
    scale_layouts: &[ScaleTokenLayout],
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

fn scale_token_layouts(config: &AutoGazeConfig) -> Vec<ScaleTokenLayout> {
    let scales = config.scale_values();
    if scales.is_empty() {
        let token_count = config.num_vision_tokens_each_frame.max(1);
        return vec![ScaleTokenLayout {
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
            ScaleTokenLayout {
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
        .map(|token_count| ScaleTokenLayout {
            token_count,
            grid: square_grid(token_count),
        })
        .collect()
}

fn square_grid(token_count: usize) -> usize {
    (token_count.max(1) as f64).sqrt().round().max(1.0) as usize
}

#[cfg(test)]
mod tests {
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
                ScaleTokenLayout {
                    token_count: 4,
                    grid: 2
                },
                ScaleTokenLayout {
                    token_count: 16,
                    grid: 4
                },
                ScaleTokenLayout {
                    token_count: 49,
                    grid: 7
                },
                ScaleTokenLayout {
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
        let fine =
            token_to_fixation_point(fine_offset + 13, &scale_layouts, 1.0).expect("fine token");
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
        let values =
            position_ids_slice_tensor_optimized::<B>(&shared_non_contiguous, 0, 3, &device)
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
        let video =
            Tensor::<B, 5>::from_data(TensorData::new(values, [1, frames, 3, 16, 16]), &device);
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
}
