"""Small, native PyTorch decoder models used by Trident's language trainer.

No Transformers dependency is required for training or generation. Parameter
names deliberately follow the Qwen3 dense/MoE layout for optional conversion.
"""
import math
from dataclasses import asdict, dataclass

from contextvars import ContextVar
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CausalLMConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 1
    router_aux_loss_coef: float = 0.0005
    rope_scaling: dict = None

    def __post_init__(self):
        for key in ("vocab_size", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "intermediate_size",
                    "max_position_embeddings"):
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("GQA requires divisible head counts and an even head_dim")
        if self.num_experts < 0 or (self.num_experts and not
                                  1 <= self.num_experts_per_tok <= self.num_experts):
            raise ValueError("invalid expert configuration")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        for token in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
            if token is not None and not 0 <= token < self.vocab_size:
                raise ValueError("special token IDs must be in the vocabulary")
        if self.rope_scaling is not None:
            if self.rope_scaling.get("rope_type", self.rope_scaling.get("type")) != "yarn":
                raise ValueError("only YaRN rope scaling is supported")
            if self.rope_scaling.get("factor", 0) < 1:
                raise ValueError("YaRN factor must be >= 1")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value):
        normed = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(value.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        frequencies = config.rope_theta ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        self.attention_factor = 1.0
        if config.rope_scaling:
            scale = config.rope_scaling
            factor = float(scale["factor"])
            original = scale.get("original_max_position_embeddings", config.max_position_embeddings)
            def dimension(rotations):
                return config.head_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(config.rope_theta))
            low = max(0, math.floor(dimension(scale.get("beta_fast", 32))))
            high = min(config.head_dim - 1, math.ceil(dimension(scale.get("beta_slow", 1))))
            ramp = ((torch.arange(config.head_dim // 2).float() - low) / max(high - low, 0.001)).clamp(0, 1)
            frequencies = frequencies * (1 - ramp + ramp / factor)
            self.attention_factor = scale.get("attention_factor", 1 + 0.1 * math.log(factor))
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def forward(self, positions, dtype):
        # Keep phase computation in float32 even under autocast.
        with torch.autocast(device_type=positions.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            return ((angles.cos() * self.attention_factor).to(dtype),
                    (angles.sin() * self.attention_factor).to(dtype))


def _rotate(value, cos, sin):
    left, right = value.chunk(2, dim=-1)
    return value * cos + torch.cat((-right, left), dim=-1) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config)

    def forward(self, value, positions, mask, past=None, use_cache=False):
        batch, length, _ = value.shape
        dim = self.config.head_dim
        q = self.q_proj(value).view(batch, length, -1, dim).transpose(1, 2)
        k = self.k_proj(value).view(batch, length, -1, dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, length, -1, dim).transpose(1, 2)
        cos, sin = self.rotary(positions, q.dtype)
        q, k = _rotate(self.q_norm(q), cos, sin), _rotate(self.k_norm(k), cos, sin)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        cache = (k, v) if use_cache else None
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        output = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1),
            attn_mask=mask, dropout_p=self.config.attention_dropout if self.training else 0.0)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1)), cache


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


# A training window may collect routing statistics without changing inference.
routing_statistics = ContextVar("trident_routing_statistics", default=None)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.num_experts)])

    def forward(self, value, valid):
        flat = value.reshape(-1, value.shape[-1])
        probabilities = self.gate(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.config.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        result = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)
            if rows.numel():
                result.index_add_(0, rows, expert(flat[rows]) * weights[rows, slots, None].to(flat.dtype))
            elif self.training and torch.is_grad_enabled():
                # Keep all expert parameters in the DDP autograd graph.
                result = result + sum(parameter.sum() * 0 for parameter in expert.parameters())
        valid = valid.reshape(-1).float()
        count = valid.sum()
        loads = (F.one_hot(indices, self.config.num_experts).float().sum(1) * valid[:, None]).sum(0)
        probability_sum = (probabilities * valid[:, None]).sum(0)
        statistics = routing_statistics.get()
        if statistics is not None and self in statistics:
            statistics[self].append((loads, probability_sum, count))
            return result.view_as(value), value.new_zeros(())
        count = count.clamp_min(1)
        loads = loads / (count * self.config.num_experts_per_tok)
        mean_prob = probability_sum / count
        auxiliary = self.config.num_experts * (loads * mean_prob).sum() * self.config.router_aux_loss_coef
        return result.view_as(value), auxiliary


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SparseMoE(config) if config.num_experts else SwiGLU(config)

    def forward(self, value, positions, mask, valid, past=None, use_cache=False):
        attention, cache = self.self_attn(self.input_layernorm(value), positions, mask, past, use_cache)
        value = value + attention
        feed = self.post_attention_layernorm(value)
        if isinstance(self.mlp, SparseMoE):
            feed, auxiliary = self.mlp(feed, valid)
        else:
            feed, auxiliary = self.mlp(feed), value.new_zeros(())
        return value + feed, auxiliary, cache


class DecoderBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class CausalLM(nn.Module):
    """Decoder-only network. Labels and loss shifting belong to the objective."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CausalLMConfig()
        self.model = DecoderBackbone(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.offload_checkpoint_inputs = False
        self.apply(self._initialize)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def gradient_checkpointing_enable(self, offload_to_cpu=False):
        self.gradient_checkpointing = True
        self.offload_checkpoint_inputs = offload_to_cpu

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, segment_ids=None,
                logits_to_keep=0, loss_labels=None, logits_chunk_size=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        batch, length = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.model.layers):
            raise ValueError("cache layer count does not match model")
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total = past_length + length
        if total > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask is None:
            attention_mask = torch.ones(batch, total, device=input_ids.device, dtype=torch.bool)
        if tuple(attention_mask.shape) != (batch, total):
            raise ValueError("attention_mask must cover cached and current tokens")
        valid = attention_mask.bool()
        if position_ids is None:
            position_ids = (valid.long().cumsum(-1) - 1).clamp_min(0)[:, -length:]
        query_positions = torch.arange(past_length, total, device=input_ids.device)
        key_positions = torch.arange(total, device=input_ids.device)
        mask = (key_positions[None, :] <= query_positions[:, None])[None, None]
        mask = mask & valid[:, None, None, :] & valid[:, None, -length:, None]
        if segment_ids is not None:
            if past_length or segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids require a full, uncached input sequence")
            mask = mask & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("gradient checkpointing and training KV cache are mutually exclusive")
        value = self.model.embed_tokens(input_ids)
        auxiliary = value.new_zeros(())
        caches = []
        for index, layer in enumerate(self.model.layers):
            past = None if past_key_values is None else past_key_values[index]
            if self.gradient_checkpointing and self.training:
                def run(hidden, block=layer):
                    result, aux, _ = block(hidden, position_ids, mask, valid[:, -length:])
                    return result, aux
                # The non-reentrant checkpoint saves its inputs through autograd hooks.
                storage = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                           if self.offload_checkpoint_inputs and value.is_cuda else nullcontext())
                with storage:
                    value, aux = checkpoint(run, value, use_reentrant=False)
                cache = None
            else:
                value, aux, cache = layer(value, position_ids, mask, valid[:, -length:], past, use_cache)
            auxiliary = auxiliary + aux.to(auxiliary.device)
            if use_cache:
                caches.append(cache)
        hidden = self.model.norm(value)
        output = {"hidden_states": hidden, "aux_loss": auxiliary,
                  "past_key_values": tuple(caches) if use_cache else None}
        if loss_labels is not None:
            if loss_labels.shape != input_ids.shape or not isinstance(logits_chunk_size, int) or logits_chunk_size < 1:
                raise ValueError("chunked loss requires aligned labels and positive logits_chunk_size")
            # Exclude ignored rows before projection. Checkpoint each projection AND
            # loss so backward never retains a sequence of full vocabulary tensors.
            targets = loss_labels[:, 1:]
            valid = targets != -100
            rows, targets = hidden[:, :-1][valid], targets[valid]
            def project_loss(block, labels):
                return F.cross_entropy(self.lm_head(block).float(), labels, reduction="sum")
            loss_sum = hidden.reshape(-1)[:0].sum().float()
            for block, labels in zip(rows.split(logits_chunk_size), targets.split(logits_chunk_size)):
                loss_sum = loss_sum + checkpoint(project_loss, block, labels, use_reentrant=False)
            output["nll_sum"] = loss_sum
        else:
            output["logits"] = self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        return output

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=32,
                 temperature=0.0, top_k=0, top_p=1.0, eos_token_id=None,
                 pad_token_id=None, generator=None):
        from trident.optims.language.generation import generate_tokens
        return generate_tokens(self, input_ids, attention_mask, max_new_tokens,
                               temperature, top_k, top_p, eos_token_id, pad_token_id, generator)

    def save_pretrained(self, directory, *, max_shard_size=None):
        from trident.optims.language.pretrained import save_pretrained
        save_pretrained(self, directory, max_shard_size)

    @classmethod
    def from_pretrained(cls, directory, map_location="cpu", *, dtype=None, device_map=None,
                        max_memory=None, offload_folder=None, quantization_config=None):
        from trident.optims.language.pretrained import load_pretrained
        return load_pretrained(cls, directory, map_location, dtype, device_map, max_memory,
                               offload_folder, quantization_config)
