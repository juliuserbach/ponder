import pytest
import torch

from ponder import grpo, schedules
from ponder.baselines import UniformPolicy
from ponder.noise import NoiseBank
from ponder.policy import Policy
from ponder.sampler import round_log_probs, sample


class ToyGenerator:
    """Features see the visible canvas, and a fill returns its noise."""

    def contextualize(self, canvas, visible, cond):
        features = torch.cat((canvas * visible[..., None], visible[..., None].float()), -1)
        return features, None

    def sample(self, context, positions, noise, masked_after):
        return noise

    def decode(self, canvas, cond):
        return canvas


def rollout(policy, schedule, given=0, batch=4, group_size=2, width=3, capacity=None):
    writable = torch.ones(batch, 81 if given else schedule.positions, dtype=torch.bool)
    for row in range(batch):
        writable[row, torch.randperm(writable.shape[1])[:given]] = False
    canvas = torch.randn(*writable.shape, width) * ~writable[..., None]
    group_ids = torch.arange(batch // group_size).repeat_interleave(group_size)
    noise = NoiseBank(0, group_ids, group_size, writable.shape[1], width,
                      schedule.removals if capacity is None else capacity)
    return sample(ToyGenerator(), policy, schedule, noise, None, canvas, writable, record=True), \
        canvas, writable, noise


def test_paper_schedules():
    assert len(schedules.mar_schedule(8).commits) == 64 and max(schedules.mar_schedule(8).commits) == 7
    assert max(schedules.mar_schedule(32).commits) == 7
    assert len(schedules.maskgen_schedule(4).commits) == 40
    empty = schedules.sudoku_schedule(0, 9)
    assert len(empty.commits) == 81 and sum(empty.commits) == 99


@pytest.mark.parametrize("given", [0, 30, 79, 80])
@pytest.mark.parametrize("removals", [0, 1, 9])
def test_rollouts_finish_on_budget(given, removals):
    torch.manual_seed(given + removals)
    schedule = schedules.sudoku_schedule(given, removals)
    result, canvas, writable, noise = rollout(UniformPolicy(), schedule, given=given)
    assert (result.removed.sum(1) == removals).all()
    assert torch.equal(result.canvas[~writable], canvas[~writable])  # clues untouched
    # Each cell holds the noise of its last fill: fill k uses bank entry k - 1, so
    # a removed cell is regenerated with fresh noise.
    fills = torch.zeros(writable.shape, dtype=torch.long)
    fills.scatter_add_(1, result.positions, (~result.removed).long())
    assert torch.equal(result.canvas[writable], noise.fill_noise(fills - 1)[writable])


def test_replay_matches_sampling():
    torch.manual_seed(0)
    schedule = schedules.sudoku_schedule(40, 9)
    policy = Policy(81, input_width=7, width=16, depth=1, heads=2, feedforward_width=32,
                    action_heads=2)
    result, _, writable, _ = rollout(policy, schedule, given=40)
    visible, removals, decided = ~writable, torch.zeros(4, dtype=torch.long), 0
    pre_final = sum(schedule.commits[:-1])
    for index, (count, observation) in enumerate(zip(schedule.commits, result.observations)):
        final = index == len(schedule.commits) - 1
        picks = result.positions[:, decided:decided + count]
        log_probs, entropy = round_log_probs(policy(observation, count), visible, writable, removals,
                                             9, 0 if final else pre_final - decided, final, picks)
        assert torch.isfinite(log_probs).all() and (entropy >= 0).all()
        removing = result.removed[:, decided:decided + count]
        visible = visible.scatter(1, picks, ~removing)
        removals, decided = removals + removing.sum(1), decided + count


def test_group_advantages():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 2.0, 4.0])
    advantages = grpo.group_advantages(rewards, group_size=3).view(2, 3)
    assert torch.equal(advantages[0], torch.zeros(3))
    assert torch.allclose(advantages[1].mean(), torch.tensor(0.0))
    assert torch.allclose(advantages[1].pow(2).mean(), torch.tensor(1.0))
