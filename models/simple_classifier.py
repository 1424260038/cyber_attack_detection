"""Small MLP classifier used by demo checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Feed-forward classifier compatible with existing demo checkpoints."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() != 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
