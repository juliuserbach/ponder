"""Visual Sudoku: the frozen MNIST-Sudoku MAR, its puzzles and its reward.

A board is 81 cells, each a 32-dimensional VAE latent of a 28x28 MNIST
glyph. Clues are the latents of the solution's glyphs at the given cells.
The reward is 1 if the classified digits of the generated board satisfy every
row, column and box constraint and keep every clue, else 0.

Boards are the solved Sudokus in ``boards`` [N,9,9]: the last 10,000 are held
out for evaluation, the rest are for training. Glyphs [9,5000,28,28] hold
5,000 MNIST images per digit 1-9. Every board, clue set and glyph is drawn
by key (``ponder.rng``), so batches are reproducible.
"""

import numpy as np
import torch

from ponder.generators.sudoku_models import FlowHead, MNISTClassifier, MNISTVAE, SudokuMAR
from ponder.rng import Stream, bounded_index, keyed_permutation, keyed_u64
from ponder.schedules import sudoku_round_bounds, sudoku_schedule

HELD_OUT = 10_000
DIFFICULTIES = {"hard": (0, 26), "medium": (27, 53), "easy": (54, 80)}  # clue counts
LATENT_SCALE = 1.0 / 0.9621499180793762  # 1 / std of the VAE latents
EULER_STEPS = 12


def frozen(module, state, device):
    module.load_state_dict(state)
    return module.to(device).eval().requires_grad_(False)


def cells(images):
    """Board images [B,1,252,252] -> cell tiles [B*81,1,28,28], row-major."""
    return images.reshape(-1, 1, 9, 28, 9, 28).permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, 28, 28)


class SudokuAdapter:
    def __init__(self, generator, vae, classifier, device):
        load = lambda path: torch.load(path, map_location="cpu", weights_only=True)  # noqa: E731
        weights = load(generator)
        self.mar = frozen(SudokuMAR(), weights["backbone"], device)
        self.flow = frozen(FlowHead(), weights["velocity"], device)
        self.vae = frozen(MNISTVAE(), load(vae)["state_dict"], device)
        self.classifier = frozen(MNISTClassifier(), load(classifier), device)
        self.device = device

    @torch.no_grad()
    def contextualize(self, canvas, visible, cond):
        """The MAR's per-cell conditions are both the policy features and the
        flow head's context."""
        context = self.mar(canvas, visible)
        return context, context

    @torch.no_grad()
    def sample(self, context, positions, noise, masked_after):
        """Integrate the flow from noise (t=1) to a latent (t=0) with 12 Euler steps."""
        condition = context.gather(1, positions[..., None].expand(-1, -1, context.shape[-1]))
        condition = condition.reshape(-1, context.shape[-1])
        x = noise.reshape(-1, noise.shape[-1])
        times = torch.linspace(1.0, 0.0, EULER_STEPS + 1, device=x.device)
        for step in range(EULER_STEPS):
            velocity = self.flow(x, times[step].expand(x.shape[0]), condition)
            x = x + (times[step + 1] - times[step]) * velocity
        return x.reshape(noise.shape)

    @torch.no_grad()
    def decode(self, canvas, cond):
        """Latents [B,81,32] -> board images [B,1,252,252] in [0, 1]."""
        tiles = self.vae.decode(canvas.reshape(-1, canvas.shape[-1]) / LATENT_SCALE)
        tiles = tiles.reshape(-1, 9, 9, 1, 28, 28).permute(0, 3, 1, 4, 2, 5)
        return (tiles.reshape(-1, 1, 252, 252) + 1.0) / 2.0

    @torch.no_grad()
    def encode(self, tiles):
        """Glyph tiles [B,81,28,28] in [0, 1] -> latents [B,81,32] (bf16 VAE means)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            means = self.vae.encode(tiles.reshape(-1, 784) * 2.0 - 1.0) * LATENT_SCALE
        return means.float().reshape(*tiles.shape[:2], -1)

    @torch.no_grad()
    def classify(self, images):
        """Board images -> digits [B,81]. The classifier's class 0 (empty cell)
        is replaced by its second choice."""
        with torch.autocast("cuda", enabled=False):
            top = self.classifier(cells(images.float())).topk(2, dim=1).indices
        return torch.where(top[:, 0] == 0, top[:, 1], top[:, 0]).reshape(-1, 81)


def solved(digits, clues):
    """Whether boards of digits [B,81] obey the Sudoku rules and keep every clue
    (``clues`` [B,81] holds the given digits, 0 elsewhere)."""
    grid = digits.reshape(-1, 9, 9)
    boxes = grid.reshape(-1, 3, 3, 3, 3).transpose(2, 3).reshape(-1, 9, 9)
    units = torch.cat((grid, grid.transpose(1, 2), boxes), dim=1).sort(dim=2).values
    valid = (units == torch.arange(1, 10, device=digits.device)).all(dim=2).all(dim=1)
    return valid & ((clues == 0) | (digits == clues)).all(dim=1)


def puzzles(adapter, glyphs, boards, clues, ordinals, seed, glyph_stream):
    """Clue-conditioned starting canvases for solved ``boards`` [B,9,9].

    Each cell of board ``ordinals[b]`` is drawn with a keyed MNIST glyph of its
    digit. The cells listed in ``clues[b]`` keep their latents, all others are
    writable. Returns (canvas [B,81,32], writable [B,81], clue digits [B,81]).
    """
    device = adapter.device
    digits = boards.reshape(-1, 81).astype(np.int64)
    choice = [[bounded_index(keyed_u64(seed, glyph_stream, ordinal, cell), glyphs.shape[1])
               for cell in range(81)] for ordinal in ordinals]
    tiles = torch.from_numpy(glyphs[digits - 1, np.array(choice)]).to(device, torch.float32) / 255.0
    given = np.zeros(digits.shape, dtype=bool)
    for row, positions in enumerate(clues):
        given[row, positions] = True
    given = torch.from_numpy(given).to(device)
    canvas = adapter.encode(tiles).masked_fill(~given[..., None], 0.0)
    return canvas, ~given, torch.from_numpy(digits).to(device) * given


def triangular_given_count(word):
    """Clue count in [0, 80] with weights 81, 80, ..., 1."""
    draw = bounded_index(word, 81 * 82 // 2)
    cumulative = 0
    for given in range(81):
        cumulative += 81 - given
        if draw < cumulative:
            return given


class Task:
    """Policy training on the training boards with rethinking budget ``removals``.

    Every update draws one clue count (triangular, favouring hard boards) and
    one round count for all of its groups. The round count is uniform between
    packing two decisions per round as far as possible and one decision per
    round. Each group gets its own keyed board, clue positions and glyphs. Its
    members share them and the generator noise.
    """

    positions = 81
    token_width = 32

    def __init__(self, config, device, seed):
        self.adapter = SudokuAdapter(config["generator"], config["vae"], config["classifier"],
                                     device)
        self.boards = np.load(config["boards"], mmap_mode="r")[:-HELD_OUT]
        self.glyphs = np.load(config["glyphs"])
        self.removals = config["removals"]
        self.noise_capacity = self.removals
        self.seed = seed

    def batch(self, update, group_ids):
        seed = self.seed
        given = triangular_given_count(keyed_u64(seed, Stream.SUDOKU_TRAIN_GIVEN_COUNT, update, 0))
        low, high = sudoku_round_bounds(given, self.removals)
        rounds = low + bounded_index(keyed_u64(seed, Stream.SUDOKU_TRAIN_ROUND_COUNT, update),
                                     high - low + 1)
        schedule = sudoku_schedule(given, self.removals, rounds)

        groups = torch.unique_consecutive(group_ids).tolist()
        boards = self.boards[[bounded_index(keyed_u64(seed, Stream.SUDOKU_TRAIN_BOARD, group),
                                            len(self.boards)) for group in groups]]
        clues = [keyed_permutation(81, seed, Stream.SUDOKU_TRAIN_CLUE_POSITIONS, group)[:given]
                 for group in groups]
        canvas, writable, cond = puzzles(self.adapter, self.glyphs, boards, clues, groups, seed,
                                         Stream.SUDOKU_TRAIN_GLYPH)
        members = len(group_ids) // len(groups)
        expand = lambda x: x.repeat_interleave(members, dim=0)  # noqa: E731
        return schedule, expand(cond), expand(canvas), expand(writable)

    def reward(self, rollout, cond, group_ids):
        # Classified in chunks of the paper's 512-trajectory microbatches.
        digits = torch.cat([self.adapter.classify(images) for images in rollout.images.split(512)])
        return solved(digits, cond).float()

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def evaluation_batches(adapter, boards, glyphs, seed, boards_per_difficulty=HELD_OUT):
    """The held-out evaluation, in batches of at most 256 boards with one clue count.

    Each difficulty draws ``boards_per_difficulty`` of the held-out boards in
    keyed order and gives every clue count in its range an equal share of
    them, also in keyed order. An ordinal per board keys its clue positions,
    glyphs and noise. Yields ``(difficulty, given, ordinals, (canvas,
    writable, clue digits))``.
    """
    held_out = boards[-HELD_OUT:]
    for index, (difficulty, (low, high)) in enumerate(DIFFICULTIES.items()):
        chosen = keyed_permutation(boards_per_difficulty, seed, Stream.SUDOKU_BOARD, index)
        offsets = keyed_permutation(high - low + 1, seed, Stream.SUDOKU_GIVEN_COUNT, index)
        share, extra = divmod(boards_per_difficulty, high - low + 1)
        first = 0
        for i, offset in enumerate(offsets):
            given, count = low + offset, share + (i < extra)
            parts = -(-count // 256)
            for part in range(parts):
                rows = range(first + count * part // parts, first + count * (part + 1) // parts)
                ordinals = [index * boards_per_difficulty + row for row in rows]
                clues = [keyed_permutation(81, seed, Stream.SUDOKU_CLUE_POSITIONS, index,
                                           ordinal)[:given] for ordinal in ordinals]
                yield difficulty, given, ordinals, puzzles(
                    adapter, glyphs, held_out[[chosen[row] for row in rows]], clues, ordinals, seed,
                    Stream.SUDOKU_GLYPH)
            first += count
