"""Anomaly Transformer for point & contextual sensor anomaly detection.

A compact implementation of the *Anomaly-Attention* mechanism (Xu et al.,
ICLR 2022). The key idea is the **association discrepancy** between:

* the **prior association** — a learnable Gaussian kernel over relative time
  positions (how a point *should* associate with its neighbours), and
* the **series association** — the data-driven self-attention map.

Normal points exhibit strong adjacent-time (series) association that matches
the prior; anomalies cannot, producing a measurable discrepancy that becomes
the anomaly score. Reconstruction error is combined with the discrepancy for
the final criterion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import AnomalyConfig


class AnomalyAttention(nn.Module):
    def __init__(self, win_size: int, d_model: int, n_heads: int):
        super().__init__()
        self.win_size = win_size
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, n_heads)
        self.out = nn.Linear(d_model, d_model)

        # Relative temporal distance matrix [T, T].
        idx = torch.arange(win_size)
        distances = (idx[None, :] - idx[:, None]).float().abs()
        self.register_buffer("distances", distances)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.query(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        sigma = self.sigma(x).transpose(1, 2)                       # [B, H, T]

        # Series association (data-driven self-attention).
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        series = torch.softmax(scores, dim=-1)                     # [B, H, T, T]

        # Prior association (learnable Gaussian over temporal distance).
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, T)             # [B, H, T, T]
        prior = (
            1.0
            / (math.sqrt(2 * math.pi) * sigma)
            * torch.exp(-(self.distances**2) / (2 * sigma**2))
        )
        prior = prior / prior.sum(dim=-1, keepdim=True)

        out = torch.matmul(series, v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out), series, prior


class AnomalyTransformerLayer(nn.Module):
    def __init__(self, win_size: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = AnomalyAttention(win_size, d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        attn_out, series, prior = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, series, prior


class AnomalyTransformer(nn.Module):
    """Encoder-decoder Anomaly Transformer with reconstruction head."""

    def __init__(self, win_size: int, config: AnomalyConfig | None = None):
        super().__init__()
        self.config = config or AnomalyConfig()
        c = self.config
        self.win_size = win_size
        self.embedding = nn.Linear(c.input_size, c.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, c.d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                AnomalyTransformerLayer(win_size, c.d_model, c.n_heads, c.dropout)
                for _ in range(c.e_layers)
            ]
        )
        self.projection = nn.Linear(c.d_model, c.input_size)

    def forward(self, x: torch.Tensor):
        """Return ``(reconstruction, series_list, prior_list)``."""
        h = self.embedding(x) + self.pos_emb
        series_list, prior_list = [], []
        for layer in self.layers:
            h, series, prior = layer(h)
            series_list.append(series)
            prior_list.append(prior)
        recon = self.projection(h)
        return recon, series_list, prior_list


def association_discrepancy(
    series_list: list[torch.Tensor], prior_list: list[torch.Tensor]
) -> torch.Tensor:
    """Symmetric KL between series and prior associations, averaged over layers.

    Returns a per-time-step discrepancy ``[B, T]`` used as the anomaly signal.
    """
    total = 0.0
    for series, prior in zip(series_list, prior_list, strict=False):
        series = series.clamp(min=1e-8)
        prior = prior.clamp(min=1e-8)
        kl1 = (series * (series.log() - prior.log())).sum(dim=-1)
        kl2 = (prior * (prior.log() - series.log())).sum(dim=-1)
        total = total + (kl1 + kl2).mean(dim=1)          # mean over heads -> [B, T]
    return total / len(series_list)


def anomaly_score(
    x: torch.Tensor,
    recon: torch.Tensor,
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    temperature: float = 50.0,
) -> torch.Tensor:
    """Combine reconstruction error and association discrepancy into a score.

    Returns per-time-step scores ``[B, T]``.
    """
    recon_err = ((x - recon) ** 2).mean(dim=-1)          # [B, T]
    discrepancy = association_discrepancy(series_list, prior_list)
    weight = torch.softmax(-discrepancy * temperature, dim=-1)
    return weight * recon_err
