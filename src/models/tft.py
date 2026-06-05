"""Temporal Fusion Transformer (TFT) for probabilistic RUL prediction.

A compact PyTorch implementation of the core TFT building blocks (Google
Research, Lim et al. 2021):

* **Variable Selection Network** — gated, instance-wise feature weighting that
  also exposes *which sensors drive the prediction*.
* **Gated Residual Network (GRN)** — the non-linear processing unit used
  throughout the architecture.
* **LSTM encoder + interpretable multi-head attention** — temporal locality
  plus long-range dependency modelling.
* **Quantile output head** — emits P10 / P50 / P90 RUL with a pinball
  (quantile) loss for calibrated confidence intervals.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import TFTConfig


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        a, b = x[..., : self.hidden_size], x[..., self.hidden_size :]
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: dense -> ELU -> dense -> GLU -> add&norm with optional context."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Instance-wise soft feature selection with interpretable weights."""

    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.flattened_grn = GatedResidualNetwork(
            n_features, hidden_size, n_features, dropout
        )
        self.per_feature_grn = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(n_features)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        weights = self.softmax(self.flattened_grn(x))            # [B, T, F]
        transformed = torch.stack(
            [grn(x[..., i : i + 1]) for i, grn in enumerate(self.per_feature_grn)],
            dim=-1,
        )                                                        # [B, T, H, F]
        combined = (transformed * weights.unsqueeze(-2)).sum(dim=-1)  # [B, T, H]
        return combined, weights


class TemporalFusionTransformer(nn.Module):
    """End-to-end TFT producing quantile RUL predictions."""

    def __init__(self, config: TFTConfig | None = None):
        super().__init__()
        self.config = config or TFTConfig()
        c = self.config
        self.quantiles = list(c.quantiles)

        self.vsn = VariableSelectionNetwork(c.input_size, c.hidden_size, c.dropout)
        self.lstm = nn.LSTM(
            c.hidden_size,
            c.hidden_size,
            num_layers=c.lstm_layers,
            batch_first=True,
            dropout=c.dropout if c.lstm_layers > 1 else 0.0,
        )
        self.gate = GatedLinearUnit(c.hidden_size, c.hidden_size)
        self.norm = nn.LayerNorm(c.hidden_size)
        self.attention = nn.MultiheadAttention(
            c.hidden_size, c.attention_heads, dropout=c.dropout, batch_first=True
        )
        self.post_attn_grn = GatedResidualNetwork(
            c.hidden_size, c.hidden_size, c.hidden_size, c.dropout
        )
        self.output_head = nn.Linear(c.hidden_size, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(quantile_preds [B, Q], variable_weights [B, F])``."""
        selected, var_weights = self.vsn(x)              # [B, T, H], [B, T, F]
        lstm_out, _ = self.lstm(selected)                # [B, T, H]
        gated = self.norm(self.gate(lstm_out) + selected)
        attn_out, _ = self.attention(gated, gated, gated)
        enriched = self.post_attn_grn(attn_out)
        last = enriched[:, -1, :]                        # [B, H]
        quantile_preds = self.output_head(last)          # [B, Q]
        avg_weights = var_weights.mean(dim=1)            # [B, F] (avg over time)
        return quantile_preds, avg_weights


class QuantileLoss(nn.Module):
    """Pinball loss across the configured quantiles."""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[..., i : i + 1]
            losses.append(torch.max(q * err, (q - 1) * err))
        return torch.mean(torch.cat(losses, dim=-1))
