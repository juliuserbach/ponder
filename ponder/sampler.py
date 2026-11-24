"""Policy-guided masked generation with rethinking.

Each round evaluates the frozen backbone once on the current canvas. The
policy then picks the round's positions one at a time, without replacement.
Picking a masked position fills it. Picking a visible position removes it
(re-masks it), so it can be regenerated later with more context. Whether a
pick fills or removes is decided by its visibility at the start of the round.

Bookkeeping keeps every rollout on budget: removals are disabled once ``R``
are spent and in the final round, and they are forced once every remaining
non-final decision is needed to spend the rest of ``R``.

A generator adapter implements:

    contextualize(canvas, visible, cond) -> (features [B,N,F], context)
        One backbone pass. ``features`` are the policy's per-position inputs.
    sample(context, positions [B,K], noise [B,K,D], masked_after [B]) -> values [B,K,D]
        Fill the given positions from this round's context and noise.
        ``masked_after`` counts the positions still masked after the round.
    decode(canvas, cond) -> images
"""

from dataclasses import dataclass

import torch

from ponder.noise import NoiseBank
from ponder.schedules import Schedule


@dataclass
class Rollout:
    canvas: torch.Tensor  # [B,N,D] final tokens
    images: torch.Tensor  # adapter.decode(canvas)
    positions: torch.Tensor  # [B,T] position chosen by each decision
    removed: torch.Tensor  # [B,T] whether that decision removed a visible position
    observations: list[torch.Tensor] | None  # per-round policy inputs (CPU), for training


def concatenate(rollouts: list[Rollout]) -> Rollout:
    """Join microbatch rollouts that used the same schedule."""
    cat = lambda name: torch.cat([getattr(r, name) for r in rollouts])  # noqa: E731
    observations = None
    if rollouts[0].observations is not None:
        observations = [torch.cat(round_) for round_ in zip(*(r.observations for r in rollouts))]
    return Rollout(cat("canvas"), cat("images"), cat("positions"), cat("removed"), observations)


def legal_positions(visible, writable, selected, removals_left, decisions_left, final):
    """Positions the next decision may pick [B,N].

    ``decisions_left`` counts the round's remaining non-final decisions,
    including this one. When it equals ``removals_left``, only removals are legal.
    """
    masked = writable & ~visible & ~selected
    if final:
        return masked
    removable = writable & visible & ~selected
    force_removal = (removals_left == decisions_left)[:, None]
    return (masked & ~force_removal) | (removable & (removals_left > 0)[:, None])


def policy_input(features, noise, legal):
    """Generator features plus next-fill noise, zeroed outside the legal positions."""
    return torch.cat((features, noise.to(features.dtype) * legal[..., None]), dim=-1)


@torch.no_grad()
def sample(adapter, policy, schedule: Schedule, noise: NoiseBank, cond, canvas, writable,
           record: bool = False) -> Rollout:
    """Run one rollout. ``canvas`` [B,N,D] holds the given tokens (e.g. Sudoku
    clues). ``writable`` [B,N] marks the positions to generate."""
    batch = writable.shape[0]
    budget = schedule.removals
    canvas = canvas.clone()
    visible = ~writable
    fills = torch.zeros(writable.shape, dtype=torch.long, device=writable.device)
    removals = torch.zeros(batch, dtype=torch.long, device=writable.device)
    pre_final = sum(schedule.commits[:-1])
    decided = 0
    positions, removed, observations = [], [], []

    features, context = adapter.contextualize(canvas, visible, cond)
    for round_index, count in enumerate(schedule.commits):
        final = round_index == len(schedule.commits) - 1
        left = 0 if final else pre_final - decided
        nothing = torch.zeros_like(writable)
        fill_noise = noise.fill_noise(fills)
        legal = legal_positions(visible, writable, nothing, budget - removals, left, final)
        observation = policy_input(features, fill_noise, legal)
        if record:
            observations.append(observation.cpu())
        logits = policy(observation, count)
        gumbel = noise.action_gumbel(round_index, count)

        selected, round_removals = nothing, removals.clone()
        picks, removing = [], []
        for head in range(count):
            legal = legal_positions(visible, writable, selected, budget - round_removals,
                                    0 if final else left - head, final)
            if not legal.any(dim=1).all():
                raise RuntimeError("the schedule reached a state without legal positions")
            scores = logits[:, head].masked_fill(~legal, float("-inf"))
            pick = (scores + gumbel[:, head]).argmax(dim=1)
            removes = visible.gather(1, pick[:, None]).squeeze(1)
            selected = selected.scatter(1, pick[:, None], True)
            round_removals = round_removals + removes.long()
            picks.append(pick)
            removing.append(removes)
        picks, removing = torch.stack(picks, 1), torch.stack(removing, 1)

        # The generator samples every pick from the round-start context. For
        # removals the sample is discarded and the old value is kept.
        visible_after = visible.scatter(1, picks, ~removing)
        masked_after = (writable & ~visible_after).sum(dim=1)
        width = canvas.shape[-1]
        pick_noise = fill_noise.gather(1, picks[..., None].expand(-1, -1, width))
        values = adapter.sample(context, picks, pick_noise, masked_after)
        old = canvas.gather(1, picks[..., None].expand(-1, -1, width))
        values = torch.where(removing[..., None], old, values)
        canvas = canvas.scatter(1, picks[..., None].expand(-1, -1, width), values)
        fills = fills.scatter_add(1, picks, (~removing).long())
        visible = visible_after
        removals = round_removals
        positions.append(picks)
        removed.append(removing)
        decided += count
        if not final:
            features, context = adapter.contextualize(canvas, visible, cond)

    if not (visible.all() and (removals == budget).all()):
        raise RuntimeError("rollout did not finish on budget")
    return Rollout(canvas=canvas, images=adapter.decode(canvas, cond),
                   positions=torch.cat(positions, 1), removed=torch.cat(removed, 1),
                   observations=observations if record else None)


def round_log_probs(logits, visible, writable, removals, budget, left, final, picks):
    """Log-probabilities and entropies [B,K] of one round's recorded picks.

    Rebuilds every decision's legal set in parallel from the round-start
    state and the picks before it, exactly as :func:`sample` did sequentially.
    """
    batch, count, _ = logits.shape
    heads = torch.arange(count, device=logits.device)
    first_pick = torch.full(writable.shape, count, dtype=torch.long, device=logits.device)
    first_pick.scatter_reduce_(1, picks, heads.expand(batch, -1), reduce="amin")
    available = writable[:, None, :] & (first_pick[:, None, :] >= heads[None, :, None])
    removes = visible.gather(1, picks)
    if final:
        legal = available & ~visible[:, None, :]
    else:
        removals_left = budget - removals[:, None] - (removes.long().cumsum(1) - removes.long())
        force_removal = removals_left == (left - heads[None, :])
        legal = available & (
            (~visible[:, None, :] & ~force_removal[:, :, None])
            | (visible[:, None, :] & (removals_left > 0)[:, :, None])
        )
    log_probs = logits.float().masked_fill(~legal, float("-inf")).log_softmax(dim=2)
    picked = log_probs.gather(2, picks[:, :, None]).squeeze(2)
    entropy = -(log_probs.exp() * log_probs.masked_fill(~legal, 0.0)).sum(dim=2)
    return picked, entropy
