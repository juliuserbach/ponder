"""How many positions the policy picks per round.

With ``N`` writable positions and a rethinking budget ``R``, a rollout makes
``T = N + 2R`` decisions: ``N + R`` fills and ``R`` removals. A schedule splits
them into rounds. All decisions of one round share one backbone evaluation.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Schedule:
    positions: int  # N, writable positions
    removals: int  # R, rethinking budget
    commits: tuple[int, ...]  # decisions per round

    def __post_init__(self):
        object.__setattr__(self, "commits", tuple(self.commits))
        if sum(self.commits) != self.positions + 2 * self.removals:
            raise ValueError("a schedule must contain N + 2R decisions")
        if min(self.commits) < 1 or max(self.commits) > self.positions:
            raise ValueError("every round must make between 1 and N decisions")


def _cosine_commits(total: int, rounds: int) -> tuple[int, ...]:
    """MAR's cosine schedule: C_r = max(C_{r-1} + 1, ceil(T (1 - cos(pi r / 2L))))."""
    cumulative = []
    committed = 0
    for r in range(1, rounds + 1):
        target = math.ceil(total * (1.0 - math.cos(math.pi * r / (2.0 * rounds))) - 1.0e-12)
        committed = total if r == rounds else max(committed + 1, target)
        cumulative.append(committed)
    return tuple(b - a for a, b in zip([0] + cumulative, cumulative))


def mar_schedule(removals: int) -> Schedule:
    """MAR-B, 256 tokens. 64 rounds up to the trained budget R=8, then one more
    round per four extra decisions, so no round exceeds the 7 trained heads."""
    rounds = 64 + math.ceil(2 * max(0, removals - 8) / 4)
    return Schedule(256, removals, _cosine_commits(256 + 2 * removals, rounds))


def maskgen_schedule(removals: int) -> Schedule:
    """MaskGen-KL, 32 tokens, one decision per round."""
    return Schedule(32, removals, (1,) * (32 + 2 * removals))


def sudoku_round_bounds(given: int, removals: int) -> tuple[int, int]:
    """Fewest and most rounds for a board with ``given`` clues (1 or 2 decisions per round)."""
    writable = 81 - given
    decisions = writable + 2 * removals
    return (decisions if writable == 1 else decisions // 2 + 1), decisions


def sudoku_schedule(given: int, removals: int, rounds: int | None = None) -> Schedule:
    """Sudoku, 81 cells minus clues. Two decisions per round first, then one.

    By default, ``2R`` two-decision rounds (fewer on nearly full boards), so an
    empty board takes 81 backbone evaluations. Training draws ``rounds``
    uniformly from :func:`sudoku_round_bounds` instead.
    """
    writable = 81 - given
    decisions = writable + 2 * removals
    if rounds is None:
        doubles = 0 if writable == 1 else min(2 * removals, (decisions - 1) // 2)
    else:
        low, high = sudoku_round_bounds(given, removals)
        if not low <= rounds <= high:
            raise ValueError(f"rounds must be in [{low}, {high}]")
        doubles = decisions - rounds
    return Schedule(writable, removals, (2,) * doubles + (1,) * (decisions - 2 * doubles))
