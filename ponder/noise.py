"""Generator noise and policy-action noise for one batch of rollouts."""

import torch

from ponder.rng import Stream, keyed_generator


def gumbel(shape, generator: torch.Generator, device) -> torch.Tensor:
    uniform = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
    limits = torch.finfo(torch.float32)
    uniform.clamp_(min=limits.tiny, max=1.0 - limits.eps)
    return -torch.log(-torch.log(uniform))


class NoiseBank:
    """Keyed noise for a batch made of contiguous GRPO groups.

    All members of a group share their generator noise, so reward differences
    within a group come from the policy's choices alone. The ``k``-th fill of
    position ``i`` always uses ``bank[k, i]``, whatever round it happens in: a
    position that is removed and refilled gets fresh noise. The policy sees
    exactly the noise that the generator will use for each position's next
    fill. Action noise for the Gumbel-max selection is drawn per member.

    ``capacity`` bounds how often a position can be refilled. The bank shape
    determines every draw, so evaluations keep it fixed across budgets.
    """

    def __init__(self, seed: int, group_ids: torch.Tensor, group_size: int,
                 positions: int, width: int, capacity: int):
        self.seed = seed
        self.capacity = capacity
        self.device = group_ids.device
        unique, self.rows = torch.unique(group_ids, sorted=True, return_inverse=True)
        self.bank = torch.stack([
            torch.randn(capacity + 1, positions, width, device=self.device,
                        generator=keyed_generator(seed, Stream.GENERATOR_NOISE, group,
                                                  device=self.device))
            for group in unique.tolist()
        ])
        members = torch.arange(len(group_ids)) % group_size
        self.action_keys = list(zip(group_ids.tolist(), members.tolist()))
        self.positions = positions

    def fill_noise(self, fills: torch.Tensor) -> torch.Tensor:
        """Noise for each position's next fill, given its fill count [B,N]."""
        index = fills.long().clamp_max(self.capacity)
        positions = torch.arange(self.positions, device=self.device)[None, :]
        return self.bank[self.rows[:, None], index, positions]

    def action_gumbel(self, round_index: int, action_count: int) -> torch.Tensor:
        """Gumbel noise [B,K,N] for the K decisions of one round."""
        return torch.stack([
            gumbel((action_count, self.positions),
                   keyed_generator(self.seed, Stream.POLICY_ACTION, group, member, round_index,
                                   device=self.device),
                   self.device)
            for group, member in self.action_keys
        ])
