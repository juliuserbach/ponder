"""The ordering policy."""

import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
    """Bidirectional transformer that scores every position for each decision of a round.

    Its input per position is the frozen generator feature concatenated with
    the noise for that position's next fill (zeroed outside the legal set).
    It outputs one row of logits per action head. A round with ``K``
    decisions uses the first ``K`` heads.
    """

    def __init__(self, positions: int, input_width: int, width: int, depth: int,
                 heads: int, feedforward_width: int, action_heads: int):
        super().__init__()
        self.input_projection = (
            nn.Identity() if input_width == width else nn.Linear(input_width, width)
        )
        self.position_embedding = nn.Parameter(torch.randn(positions, width) * width**-0.5)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(width, heads, feedforward_width, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, action_heads)

    def forward(self, observations: torch.Tensor, action_count: int) -> torch.Tensor:
        """Observations [B,N,D_in] -> float32 logits [B,K,N]."""
        x = self.input_projection(observations)
        x = x + self.position_embedding.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        logits = F.linear(x, self.output.weight[:action_count], self.output.bias[:action_count])
        return logits.transpose(1, 2).float()


def load_policy(path, device="cuda") -> Policy:
    """Load a released policy file: {"policy": state_dict, "policy_config": kwargs}."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy = Policy(**state["policy_config"])
    policy.load_state_dict(state["policy"])
    return policy.to(device).eval()
