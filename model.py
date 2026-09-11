from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class C1Config:
    vocab_size: int
    hidden_size: int = 1024
    num_layers: int = 32
    num_q_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2730
    max_seq_len: int = 512
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        positions = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[-2]

        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        return q, k


class C1Attention(nn.Module):
    def __init__(self, config: C1Config) -> None:
        super().__init__()

        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_q_heads * config.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.num_q_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch,
            seq_len,
            self.num_q_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        repeat_factor = self.num_q_heads // self.num_kv_heads

        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / (self.head_dim ** 0.5)

        causal = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)

        out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.num_q_heads * self.head_dim,
        )

        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: C1Config) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: C1Config) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attention = C1Attention(config)

        self.ffn_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.attn_norm(x)
        )

        x = x + self.mlp(
            self.ffn_norm(x)
        )

        return x


class C1Micro(nn.Module):
    def __init__(self, config: C1Config) -> None:
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.num_layers)
        )

        self.final_norm = RMSNorm(
            config.hidden_size
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tied embeddings.
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
        }
