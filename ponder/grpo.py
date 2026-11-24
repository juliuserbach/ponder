"""Group-relative policy gradient: one plain update per batch of rollouts.

    L = -1/(B T) sum_j sum_d [ A_j log pi(a_jd | o_jd) + beta H(pi(. | o_jd)) ]

No reference policy, importance ratios, clipping or repeated epochs.
"""

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from ponder.sampler import Rollout, round_log_probs
from ponder.schedules import Schedule


def group_advantages(rewards: torch.Tensor, group_size: int, threshold: float = 1e-5):
    """(r - group mean) / group std (population), or 0 for groups without spread."""
    grouped = rewards.float().view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1, keepdim=True).sqrt()
    informative = std > threshold
    return (centered / torch.where(informative, std, torch.ones_like(std)) * informative).view(-1)


def linear_schedule(update: int, initial: float, final: float, hold: int, decay: int) -> float:
    """``initial`` for ``hold`` updates, then linear to ``final`` over ``decay`` updates."""
    if decay == 0 or update < hold:
        return initial
    return initial + min((update - hold + 1) / decay, 1.0) * (final - initial)


def entropy_coefficient(update: int, initial: float, final: float, decay: int) -> float:
    """Linear from ``initial`` at update 0 to ``final`` at update ``decay - 1``."""
    if decay <= 1:
        return final
    return initial + min(update, decay - 1) / (decay - 1) * (final - initial)


def backward(policy, rollout: Rollout, writable, schedule: Schedule, advantages,
             entropy_bonus: float, microbatch: int) -> torch.Tensor:
    """Replay the recorded observations and accumulate the loss gradient.

    Each (round, microbatch) is backpropagated immediately, so memory does not
    grow with the number of rounds. DDP synchronizes only on the last one.
    """
    batch = rollout.positions.shape[0]
    denominator = float(batch * rollout.positions.shape[1])
    visible = ~writable
    removals = torch.zeros(batch, dtype=torch.long, device=writable.device)
    pre_final = sum(schedule.commits[:-1])
    decided = 0
    total = torch.zeros((), device=writable.device)
    for round_index, observation in enumerate(rollout.observations):
        count = schedule.commits[round_index]
        final = round_index == len(schedule.commits) - 1
        picks = rollout.positions[:, decided:decided + count]
        for start in range(0, batch, microbatch):
            rows = slice(start, start + microbatch)
            last = final and start + microbatch >= batch
            sync = nullcontext() if last or not isinstance(policy, DistributedDataParallel) \
                else policy.no_sync()
            with sync:
                logits = policy(observation[rows].to(writable.device), count)
                log_probs, entropy = round_log_probs(
                    logits, visible[rows], writable[rows], removals[rows], schedule.removals,
                    0 if final else pre_final - decided, final, picks[rows])
                loss = -(log_probs * advantages[rows, None].expand_as(log_probs)).sum() / denominator
                loss = loss - entropy_bonus * entropy.sum() / denominator
                loss.backward()
            total = total + loss.detach()
        removing = rollout.removed[:, decided:decided + count]
        visible = visible.scatter(1, picks, ~removing)
        removals = removals + removing.sum(dim=1)
        decided += count
    return total


def update(policy, optimizer, rollout: Rollout, writable, schedule: Schedule, advantages,
           learning_rate: float, entropy_bonus: float, microbatch: int, clip: float = 1.0):
    """One optimizer step on a batch of recorded rollouts. Returns (loss, grad norm)."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    # Parameters changed since the rollout, so drop autocast's cached bf16 copies.
    torch.clear_autocast_cache()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = backward(policy, rollout, writable, schedule, advantages, entropy_bonus, microbatch)
    torch.clear_autocast_cache()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return loss, norm
