from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PaLMConfig:
    d_model: int = 4096
    n_layer: int = 32
    n_head: int = 16
    seq_len: int = 2048
    vocab_size: int = 256000


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: PaLMConfig) -> None:
        super(Attention, self).__init__()

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head

        self.wq = nn.Linear(config.d_model, config.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, self.d_head, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(B, T, self.n_head, self.d_head)
        k = k.view(B, T, 1, self.d_head)
        v = v.view(B, T, 1, self.d_head)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        k = torch.repeat_interleave(k, dim=2, repeats=self.n_head)
        v = torch.repeat_interleave(v, dim=2, repeats=self.n_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.wo(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: PaLMConfig) -> None:
        super(MLP, self).__init__()

        self.w1 = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.w2 = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.w3 = nn.Linear(4 * config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return x


class PaLMBlock(nn.Module):
    def __init__(self, config: PaLMConfig) -> None:
        super(PaLMBlock, self).__init__()

        self.attn = Attention(config)
        self.ffwd = MLP(config)
        self.attn_norm = nn.LayerNorm(config.d_model, bias=False)
        self.ffwd_norm = nn.LayerNorm(config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return (
            x + self.attn(self.attn_norm(x), freqs_cis) + self.ffwd(self.ffwd_norm(x))
        )


class PaLM(nn.Module):
    def __init__(self, config: PaLMConfig) -> None:
        super(PaLM, self).__init__()

        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([PaLMBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.d_model, bias=False)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.tok_embed.weight = self.head.weight

        self.freqs_cis = precompute_freqs_cis(
            config.d_model, config.seq_len * 2, 10000.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        freqs_cis = self.freqs_cis[:T]

        x = self.tok_embed(x)
        for block in self.blocks:
            x = block(x, freqs_cis)
        x = self.norm(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    config = PaLMConfig()
    model = PaLM(config)

    print(f"{sum(p.numel() for p in model.parameters()):,}")
