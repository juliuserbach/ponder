"""Counter-based random keys.

Every random draw is addressed by ``(seed, stream, coordinates)``, for example
the generator noise of GRPO group ``g``. Samples therefore do not depend on
batch layout, DDP world size or resumption, and all members of a group can
share their generator noise.
"""

from enum import IntEnum

import torch

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class Stream(IntEnum):
    # The values are part of the sampling protocol: changing one changes
    # every sample drawn from that stream.
    IMAGENET_CLASS = 1
    MASKGEN_TRAIN_PROMPT = 2
    SUDOKU_BOARD = 4
    SUDOKU_GIVEN_COUNT = 5
    SUDOKU_CLUE_POSITIONS = 6
    SUDOKU_GLYPH = 7
    GENERATOR_NOISE = 8
    POLICY_ACTION = 10
    FIDGRAD_EMA_MEMBER = 14
    SUDOKU_TRAIN_BOARD = 15
    SUDOKU_TRAIN_GIVEN_COUNT = 16
    SUDOKU_TRAIN_CLUE_POSITIONS = 17
    SUDOKU_TRAIN_GLYPH = 18
    GENEVAL_TRAIN_PROMPT = 21
    SUDOKU_TRAIN_ROUND_COUNT = 28
    MASKGEN_TRAIN_REMOVAL_BUDGET = 30


def _splitmix64(value: int) -> int:
    value = (value + GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def keyed_u64(seed: int, stream: Stream, *coordinates: int) -> int:
    """Map ``(seed, stream, coordinates)`` to one stable 64-bit word."""
    value = _splitmix64(seed ^ int(stream) * GAMMA)
    for position, coordinate in enumerate(coordinates, start=1):
        value = _splitmix64(value ^ _splitmix64((coordinate + position * GAMMA) & MASK64))
    return value


def bounded_index(word: int, size: int) -> int:
    """Map a 64-bit word to ``[0, size)`` without floating point."""
    return (word * size) >> 64


def keyed_permutation(size: int, seed: int, stream: Stream, *coordinates: int) -> list[int]:
    """Fisher-Yates shuffle of ``range(size)`` with keyed swaps."""
    values = list(range(size))
    for end in range(size - 1, 0, -1):
        other = bounded_index(keyed_u64(seed, stream, *coordinates, end), end + 1)
        values[end], values[other] = values[other], values[end]
    return values


def keyed_generator(seed: int, stream: Stream, *coordinates: int, device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(keyed_u64(seed, stream, *coordinates))
    return generator


def group_ids_for_update(
    update: int, groups_per_update: int, group_size: int, rank: int, world_size: int, device
) -> torch.Tensor:
    """Global group ids owned by one DDP rank, each repeated for its members."""
    local = groups_per_update // world_size
    start = update * groups_per_update + rank * local
    ids = torch.arange(start, start + local, dtype=torch.long, device=device)
    return ids.repeat_interleave(group_size)
