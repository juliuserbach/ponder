"""Baselines that reuse the PONDER sampler."""

import torch
from torch import nn


class UniformPolicy(nn.Module):
    """Random order: zero logits make every Gumbel-max choice uniform over the
    legal positions. With ``R > 0`` this also removes uniformly at random."""

    def forward(self, observations: torch.Tensor, action_count: int) -> torch.Tensor:
        batch, positions = observations.shape[:2]
        return torch.zeros(batch, action_count, positions, device=observations.device)
