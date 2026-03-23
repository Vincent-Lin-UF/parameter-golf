#!/usr/bin/env python3
"""
Frontier training script incorporating techniques from top parameter-golf submissions.
Runs on any platform with PyTorch (CPU or single CUDA GPU). No multi-GPU/NCCL/MLX.

Techniques included (all toggleable via env vars):
  - Value Residual (ResFormer): cache layer-0 V, blend into subsequent layers
  - Partial RoPE: rotate only 16/64 head dims
  - XSA: subtract self-value projection in last N layers
  - Star-ReLU MLP: relu^2 with learned scale+bias, wider hidden dim
  - BigramHash: hash token pairs into learned embeddings
  - SmearGate: blend each token with its predecessor
  - Int6 QAT: fake-quantize weights during training (late-enabled)
  - EMA: exponential moving average of weights
  - Sliding Window Eval: overlapping windows for better context
  - U-Net Skip Gates: learned sigmoid gates on skip connections

Usage:
    .venv\\Scripts\\python train_gpt_frontier.py

Quick smoke test:
    $env:ITERATIONS=10; $env:TRAIN_BATCH_TOKENS=8192; $env:VAL_BATCH_SIZE=8192
    $env:GRAD_ACCUM_STEPS=1; $env:WARMUP_STEPS=2
    .venv\\Scripts\\python train_gpt_frontier.py
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ==============================================================================
# DEVICE + DTYPE
# ==============================================================================

_device_name = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(_device_name)
COMPUTE_DTYPE = torch.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    # Data / tokenizer
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3000))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model — frontier defaults: 11 layers, 3.5x MLP, partial RoPE, etc.
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_hidden: int = int(os.environ.get("MLP_HIDDEN", 1792))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 16))  # Partial RoPE: 16 of 64
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Frontier technique toggles
    value_residual: bool = bool(int(os.environ.get("VALUE_RESIDUAL", "1")))
    xsa_layers: int = int(os.environ.get("XSA_LAYERS", 4))  # 0 to disable
    bigram_buckets: int = int(os.environ.get("BIGRAM_BUCKETS", 8192))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))
    smeargate: bool = bool(int(os.environ.get("SMEARGATE", "1")))
    skip_gates: bool = bool(int(os.environ.get("SKIP_GATES", "1")))

    # QAT
    qat_enabled: bool = bool(int(os.environ.get("QAT_ENABLED", "1")))
    qat_late_frac: float = float(os.environ.get("QAT_LATE_FRAC", 0.15))  # enable at last 15% of warmdown

    # EMA
    ema_enabled: bool = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))

    # Sliding window eval
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))  # 0 = standard eval

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.035))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.025))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_gate,vr_lambda,smear,bigram,star_scale,star_bias",
    ).split(",") if p
)

# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def zeropower_newtonschulz5(g: Tensor, steps: int, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


def build_rope_cache(seq_len: int, rope_dims: int, base: float, device: torch.device) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2, device=device, dtype=torch.float32) / rope_dims))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    # x: (B, H, T, D)  cos/sin: (T, rope_dims//2)
    orig_dtype = x.dtype
    T = x.shape[2]
    rd = rope_dims
    cos = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    if rd < x.shape[-1]:
        x_rope = x[..., :rd]
        x_pass = x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([x_rot, x_pass], dim=-1)
    half = rd // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# DATA LOADING
# ==============================================================================

class TokenStream:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return torch.from_numpy(x).to(DEVICE, dtype=torch.long), torch.from_numpy(y).to(DEVICE, dtype=torch.long)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Linear):
    """Linear layer with fp32 weights, bf16 compute, and optional int6 QAT."""
    _qat_enabled: bool = False

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__(in_dim, out_dim, bias=bias)
        self.weight.data = self.weight.data.float()
        self._zero_init = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = w.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w.to(x.dtype) + (w_q - w.to(x.dtype)).detach()  # STE
        else:
            w = w.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class RMSNormNoWeight(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x)


class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        bigram_hash = (prev_ids * 1009 + input_ids) % self.num_buckets
        return self.proj(self.embed(bigram_hash))


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - g) * x + g * x_prev


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_dims: int,
                 rope_base: float, qk_gain_init: float, max_seq_len: int,
                 value_residual: bool = False, use_xsa: bool = False, layer_idx: int = 0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = min(rope_dims, self.head_dim)
        self.use_xsa = use_xsa
        self.value_residual = value_residual

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * qk_gain_init)
        self.scale = self.head_dim ** -0.5

        # Value Residual: learnable blend with layer-0 V
        if value_residual and layer_idx > 0:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        else:
            self.vr_lambda = None

        # Pre-compute RoPE
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        cos, sin = build_rope_cache(max_seq_len, self.rope_dims, rope_base, torch.device("cpu"))
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(self, x: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Cache raw V for Value Residual before any blending
        raw_v = v if self.value_residual else None

        # Value Residual: blend with layer-0 V
        if self.vr_lambda is not None and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v

        # Partial RoPE
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        q = apply_rotary_emb(rms_norm(q).to(COMPUTE_DTYPE), cos, sin, self.rope_dims)
        k = apply_rotary_emb(rms_norm(k).to(COMPUTE_DTYPE), cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
            v_attn = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
        else:
            v_attn = v

        y = F.scaled_dot_product_attention(q, k, v_attn, scale=self.scale, is_causal=True)

        # XSA: subtract self-value projection
        if self.use_xsa:
            vn = F.normalize(v_attn, dim=-1)
            y = y - (y * vn).sum(dim=-1, keepdim=True) * vn

        y = y.permute(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y), raw_v


class MLP(nn.Module):
    """Star-ReLU MLP: relu^2 with learned per-channel scale and bias."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.star_scale = nn.Parameter(torch.ones(hidden, dtype=torch.float32))
        self.star_bias = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc(x)).pow(2)
        x = x * self.star_scale.to(dtype=x.dtype) + self.star_bias.to(dtype=x.dtype)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_hidden: int,
                 rope_dims: int, rope_base: float, qk_gain_init: float, max_seq_len: int,
                 value_residual: bool = False, use_xsa: bool = False, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_dims, rope_base,
            qk_gain_init, max_seq_len, value_residual, use_xsa, layer_idx,
        )
        self.mlp = MLP(dim, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]))

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x), v0=v0)
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x, raw_v


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        dim = args.model_dim
        num_layers = args.num_layers
        self.logit_softcap = args.logit_softcap
        self.value_residual = args.value_residual
        xsa_start = num_layers - args.xsa_layers if args.xsa_layers > 0 else num_layers

        self.tok_emb = nn.Embedding(args.vocab_size, dim)

        # BigramHash
        self.bigram = BigramHashEmbedding(args.bigram_buckets, args.bigram_dim, dim) if args.bigram_buckets > 0 else None

        # SmearGate
        self.smeargate = SmearGate(dim) if args.smeargate else None

        # Encoder/decoder split
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, dim, dtype=torch.float32))
        self.use_skip_gates = args.skip_gates
        if args.skip_gates:
            self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(dim, args.num_heads, args.num_kv_heads, args.mlp_hidden,
                  args.rope_dims, args.rope_base, args.qk_gain_init, args.train_seq_len,
                  value_residual=args.value_residual, use_xsa=(i >= xsa_start), layer_idx=i)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNormNoWeight()

        # Init: zero out proj weights, orthogonal init for main weights
        for block in self.blocks:
            for name, p in block.named_parameters():
                if hasattr(getattr(block.attn, 'proj', None), 'weight') and p is block.attn.proj.weight:
                    nn.init.zeros_(p)
                elif hasattr(getattr(block.mlp, 'proj', None), 'weight') and p is block.mlp.proj.weight:
                    nn.init.zeros_(p)
                elif p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                    nn.init.orthogonal_(p, gain=1.0)
        self.tok_emb.weight.data = (torch.randn(args.vocab_size, dim) * args.tied_embed_init_std).to(COMPUTE_DTYPE)

    def softcap(self, logits: Tensor) -> Tensor:
        c = self.logit_softcap
        return c * torch.tanh(logits / c)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids).to(COMPUTE_DTYPE)
        if self.bigram is not None:
            x = x + self.bigram(input_ids).to(COMPUTE_DTYPE)
        x = rms_norm(x)
        if self.smeargate is not None:
            x = self.smeargate(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x, raw_v = self.blocks[i](x, x0, v0=v0)
            if self.value_residual and v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                if self.use_skip_gates:
                    gate = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))
                    scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
                    x = gate[None, None, :] * x + (1.0 - gate[None, None, :]) * scaled_skip
                else:
                    x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skip
            x, _ = self.blocks[self.num_encoder_layers + i](x, x0, v0=v0)
        return self.final_norm(x)

    def loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = x @ self.tok_emb.weight.to(x.dtype).T
        logits = self.softcap(logits)
        return F.cross_entropy(logits.float(), y)


# ==============================================================================
# OPTIMIZERS
# ==============================================================================

class Muon:
    def __init__(self, params: list[tuple[str, nn.Parameter]], args: Hyperparameters):
        self.params = params
        self.args = args
        self.buffers = {name: torch.zeros_like(p.data) for name, p in params}

    @torch.no_grad()
    def step(self, grads: dict[str, Tensor], step: int, lr_mul: float) -> None:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        wd = self.args.weight_decay
        for name, p in self.params:
            g = grads.get(name)
            if g is None:
                continue
            # Decoupled weight decay
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)
            buf = momentum * self.buffers[name] + g
            self.buffers[name] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p.data -= lr * (g_ortho * scale).to(p.dtype)


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        self.embed_name = "tok_emb.weight"
        self.matrix_params = []
        self.scalar_params = []
        self.embed_param = None

        for name, p in model.named_parameters():
            if name == self.embed_name:
                self.embed_param = (name, p)
            elif p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                self.matrix_params.append((name, p))
            else:
                self.scalar_params.append((name, p))

        self.muon = Muon(self.matrix_params, args)
        self.adam_embed = torch.optim.AdamW(
            [self.embed_param[1]], lr=args.tied_embed_lr,
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay,
        )
        scalar_param_list = [p for _, p in self.scalar_params]
        self.adam_scalar = torch.optim.AdamW(
            scalar_param_list, lr=args.scalar_lr,
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay,
        ) if scalar_param_list else None

    def step(self, model: GPT, step: int, lr_mul: float) -> None:
        grads: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad
        self.muon.step(grads, step=step, lr_mul=lr_mul)
        for pg in self.adam_embed.param_groups:
            pg["lr"] = self.args.tied_embed_lr * lr_mul
        self.adam_embed.step()
        if self.adam_scalar is not None:
            for pg in self.adam_scalar.param_groups:
                pg["lr"] = self.args.scalar_lr * lr_mul
            self.adam_scalar.step()


# ==============================================================================
# QUANTIZATION (INT6 + ZLIB)
# ==============================================================================

INT6_CLIP = 31
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT6_PER_ROW_SCALE_DTYPE = np.float16
INT_CLIP_PERCENTILE = 99.99984
INT_CLIP_Q = INT_CLIP_PERCENTILE / 100.0


def _np_float32(t: Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def quantize_int6_per_row(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f32 = arr.astype(np.float32)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / float(INT6_CLIP), 1.0 / float(INT6_CLIP)).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -(INT6_CLIP + 1), INT6_CLIP).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT6_PER_ROW_SCALE_DTYPE))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / float(INT6_CLIP) if clip_abs > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -(INT6_CLIP + 1), INT6_CLIP).astype(np.int8)
    return np.ascontiguousarray(q), scale


def quantize_state_dict(state_dict: dict[str, Tensor]) -> tuple[dict[str, object], dict[str, int]]:
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig_dtypes = {}, {}
    qmeta: dict[str, dict] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        arr = _np_float32(tensor)
        stats["param_count"] += arr.size
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += arr.nbytes
        if not np.issubdtype(arr.dtype, np.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(arr)
            stats["int_payload_bytes"] += passthrough[name].nbytes
            continue
        if arr.size <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                kept = np.ascontiguousarray(arr.astype(np.float32))
            else:
                passthrough_orig_dtypes[name] = "float32"
                kept = np.ascontiguousarray(arr.astype(INT8_KEEP_FLOAT_STORE_DTYPE))
            passthrough[name] = kept
            stats["int_payload_bytes"] += kept.nbytes
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_int6_per_row(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = "float32"
        stats["int_payload_bytes"] += q.nbytes + s.nbytes
    obj: dict[str, object] = {
        "__quant_format__": "int6_per_row_v1", "quantized": quantized, "scales": scales,
        "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict(quant_obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = torch.from_numpy(out_arr)
    for name, arr in quant_obj["passthrough"].items():
        out[name] = torch.from_numpy(np.array(arr, copy=True).astype(np.float32))
    return out


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None) if tokenizer_name else None
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(f"More train shards than expected: found {actual_train_files}, manifest says {expected_train_files}")
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


# ==============================================================================
# EVALUATION (standard + sliding window)
# ==============================================================================

def _score_batch(model: GPT, x: Tensor, y: Tensor, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                 score_start: int = 0) -> tuple[float, float, float]:
    """Score tokens[score_start:] in a batch, return (loss_sum, token_count, byte_count)."""
    with torch.no_grad():
        # Full forward pass
        hidden = model(x).reshape(-1, model.tok_emb.weight.shape[1])
        logits = hidden @ model.tok_emb.weight.to(hidden.dtype).T
        logits = model.softcap(logits)
        # Per-token loss
        nll = F.cross_entropy(logits.float(), y.reshape(-1), reduction="none").reshape(y.shape)
    # Score only from score_start onward
    nll_scored = nll[:, score_start:]
    y_scored = y[:, score_start:]
    x_prev = x[:, score_start:] if score_start > 0 else x[:, :y_scored.shape[1]]

    loss_sum = float(nll_scored.sum().item())
    token_count = float(nll_scored.numel())

    tgt_ids = y_scored.cpu().numpy().reshape(-1)
    if score_start > 0:
        prev_ids = x[:, score_start - 1 : score_start - 1 + y_scored.shape[1]].cpu().numpy().reshape(-1)
    else:
        prev_ids = x[:, :y_scored.shape[1]].cpu().numpy().reshape(-1)
    bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16)
    byte_count = float(bytes_np.astype(np.float64).sum())
    return loss_sum, token_count, byte_count


@torch.no_grad()
def eval_val(args: Hyperparameters, model: GPT, val_tokens: np.ndarray,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             log_fn=None) -> tuple[float, float]:
    model.eval()
    seq_len = args.train_seq_len
    stride = args.eval_stride

    if stride > 0 and stride < seq_len:
        # Sliding window evaluation
        total_tokens_available = val_tokens.size - 1
        total_loss_sum = 0.0
        total_tokens = 0.0
        total_bytes = 0.0
        window_starts = list(range(0, total_tokens_available - seq_len + 1, stride))
        if not window_starts:
            window_starts = [0]
        total_windows = len(window_starts)
        for wi, ws in enumerate(window_starts):
            chunk = val_tokens[ws : ws + seq_len + 1]
            x_np = chunk[:-1].reshape(1, seq_len)
            y_np = chunk[1:].reshape(1, seq_len)
            x = torch.from_numpy(x_np).to(DEVICE, dtype=torch.long)
            y = torch.from_numpy(y_np).to(DEVICE, dtype=torch.long)
            # Score only the last `stride` tokens (or all for first window)
            score_start = 0 if ws == 0 else max(seq_len - stride, 0)
            ls, tc, bc = _score_batch(model, x, y, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, score_start)
            total_loss_sum += ls
            total_tokens += tc
            total_bytes += bc
            if log_fn and total_windows > 1 and (wi == 0 or wi == total_windows - 1 or (wi + 1) % 500 == 0):
                log_fn(f"val_slide_progress:{wi + 1}/{total_windows}")
    else:
        # Standard non-overlapping evaluation
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        val_batch_seqs = max(val_batch_tokens // seq_len, 1)
        total_seqs = (val_tokens.size - 1) // seq_len
        total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
        total_loss_sum = 0.0
        total_tokens = 0.0
        total_bytes = 0.0
        for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
            batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            chunk = val_tokens[raw_start:raw_end]
            x_np = chunk[:-1].reshape(-1, seq_len)
            y_np = chunk[1:].reshape(-1, seq_len)
            x = torch.from_numpy(x_np).to(DEVICE, dtype=torch.long)
            y = torch.from_numpy(y_np).to(DEVICE, dtype=torch.long)
            ls, tc, bc = _score_batch(model, x, y, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, 0)
            total_loss_sum += ls
            total_tokens += tc
            total_bytes += bc
            if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
                log_fn(f"val_progress:{batch_idx}/{total_batches}")

    model.train()
    val_loss = total_loss_sum / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb


# ==============================================================================
# TRAINING
# ==============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Python {sys.version}", console=False)
    log(f"PyTorch {torch.__version__} | Device: {DEVICE}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("Only tied embeddings supported")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    model = GPT(args).to(DEVICE)
    opt = SplitOptimizers(model, args)

    # EMA setup
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Print config
    n_params = sum(p.numel() for p in model.parameters())
    log(f"run_id:{args.run_id}")
    log(f"device:{DEVICE} | pytorch:{torch.__version__}")
    log(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads}")
    log(f"mlp_hidden:{args.mlp_hidden} rope_dims:{args.rope_dims} xsa_layers:{args.xsa_layers}")
    log(f"value_residual:{args.value_residual} bigram_buckets:{args.bigram_buckets} smeargate:{args.smeargate} skip_gates:{args.skip_gates}")
    log(f"qat:{args.qat_enabled}(late_frac={args.qat_late_frac}) ema:{args.ema_enabled}(decay={args.ema_decay})")
    log(f"eval_stride:{args.eval_stride} grad_clip:{args.grad_clip_norm} weight_decay:{args.weight_decay}")
    log(f"train_batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len} iterations:{args.iterations}")
    log(f"muon_lr:{args.matrix_lr} embed_lr:{args.tied_embed_lr} scalar_lr:{args.scalar_lr} muon_momentum:{args.muon_momentum}")
    log(f"warmdown_iters:{args.warmdown_iters} warmup_steps:{args.warmup_steps} max_wallclock:{args.max_wallclock_seconds}s")
    if expected_train_files and actual_train_files < expected_train_files:
        log(f"WARNING: train_shards:{actual_train_files}/{expected_train_files}")
    log(f"train_shards:{actual_train_files} val_tokens:{val_tokens.size - 1}")

    # Warmup
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            model.zero_grad()
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                x, y = train_loader.next_batch(args.microbatch_tokens, args.train_seq_len)
                loss = model.loss(x, y)
                (loss * grad_scale).backward()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    qat_active = False
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            if step % 25 == 0 or last_step:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms")
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))

        # Late QAT: enable when entering final fraction of warmdown
        if args.qat_enabled and not qat_active and lr_mul < args.qat_late_frac:
            CastedLinear._qat_enabled = True
            qat_active = True
            log(f"qat:enabled at step:{step} lr_mul:{lr_mul:.4f}")

        step_t0 = time.perf_counter()
        model.zero_grad()
        train_loss = torch.tensor(0.0, device=DEVICE)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.microbatch_tokens, args.train_seq_len)
            loss = model.loss(x, y)
            (loss * grad_scale).backward()
            train_loss = train_loss + loss.detach() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        train_loss_value = train_loss.item()
        opt.step(model, step=step, lr_mul=lr_mul)
        model.zero_grad()

        # EMA update
        if ema_state is not None:
            decay = args.ema_decay
            with torch.no_grad():
                for name, param in model.state_dict().items():
                    ema_state[name].mul_(decay).add_(param.cpu(), alpha=1.0 - decay)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # APPLY EMA WEIGHTS BEFORE EXPORT
    # ==============================================================================
    if ema_state is not None:
        log(f"ema:applying averaged weights (decay={args.ema_decay})")
        ema_sd = {k: v.to(dtype=model.state_dict()[k].dtype, device=DEVICE) for k, v in ema_state.items()}
        model.load_state_dict(ema_sd)

    # ==============================================================================
    # SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    out_path = out_dir / f"{args.run_id}_frontier_model.pt"
    torch.save(model.state_dict(), out_path)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    state_dict = {k: v for k, v in model.state_dict().items()}
    quant_obj, quant_stats = quantize_state_dict(state_dict)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_path = out_dir / f"{args.run_id}_frontier_model.int6.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int_payload_bytes"], 1)
    log(f"serialized_int6_zlib:{quant_file_bytes} bytes (payload:{quant_stats['int_payload_bytes']} ratio:{ratio:.2f}x)")

    # Load back and validate roundtrip
    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict(pickle.loads(zlib.decompress(quant_blob_disk)))
    current_sd = model.state_dict()
    for k, v in quant_flat.items():
        if k in current_sd:
            current_sd[k] = v.to(current_sd[k].dtype).to(DEVICE)
    model.load_state_dict(current_sd)

    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int6_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int6_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    log(f"artifact_size:{quant_file_bytes} bytes ({quant_file_bytes / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
