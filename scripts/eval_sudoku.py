"""Sudoku accuracy over rethinking budgets (the paper's test-time scaling).

    python scripts/eval_sudoku.py
    python scripts/eval_sudoku.py --policy random --budgets 0 9

Every budget sees the same 30,000 puzzles (each of the 10,000 held-out boards
once per difficulty) with the same keyed noise. The noise capacity stays fixed
across budgets. A board counts as solved if its classified digits obey every
Sudoku rule and keep the clues. ``--boards-per-difficulty`` runs a smaller
version of the protocol for quick checks. Exact samples also depend on the GPU
and ``--microbatch``.
"""

import argparse
import json

import numpy as np
import torch

from ponder.baselines import UniformPolicy
from ponder.generators.sudoku import (DIFFICULTIES, HELD_OUT, SudokuAdapter, evaluation_batches,
                                     solved)
from ponder.noise import NoiseBank
from ponder.policy import load_policy
from ponder.sampler import sample
from ponder.schedules import sudoku_schedule


def evaluate(adapter, policy, boards, glyphs, removals, seed=42, noise_capacity=128,
             microbatch=64, boards_per_difficulty=HELD_OUT):
    """Fraction of solved boards per difficulty and overall, at budget ``removals``."""
    correct = dict.fromkeys(DIFFICULTIES, 0)
    batches = evaluation_batches(adapter, boards, glyphs, seed, boards_per_difficulty)
    for difficulty, given, ordinals, (canvas, writable, clues) in batches:
        schedule = sudoku_schedule(given, removals)
        images = []
        for start in range(0, len(ordinals), microbatch):
            rows = slice(start, start + microbatch)
            group_ids = torch.tensor(ordinals[rows], device=adapter.device)
            noise = NoiseBank(seed, group_ids, 1, 81, 32, noise_capacity)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rollout = sample(adapter, policy, schedule, noise, clues[rows], canvas[rows],
                                 writable[rows])
            images.append(rollout.images)
        correct[difficulty] += int(solved(adapter.classify(torch.cat(images)), clues).sum())
    accuracy = {name: count / boards_per_difficulty for name, count in correct.items()}
    accuracy["overall"] = sum(correct.values()) / (len(DIFFICULTIES) * boards_per_difficulty)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/sudoku.pt",
                        help="policy file, or 'random' for random order")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 9, 16, 32, 64, 128])
    parser.add_argument("--generator", default="checkpoints/sudoku_generator.pt")
    parser.add_argument("--vae", default="checkpoints/sudoku_vae.pt")
    parser.add_argument("--classifier", default="checkpoints/sudoku_classifier.pt")
    parser.add_argument("--boards", default="data/sudoku_boards.npy")
    parser.add_argument("--glyphs", default="data/sudoku_glyphs.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-capacity", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--boards-per-difficulty", type=int, default=HELD_OUT)
    args = parser.parse_args()
    if max(args.budgets) > args.noise_capacity:
        parser.error("the noise capacity must cover every budget")

    device = torch.device("cuda")
    adapter = SudokuAdapter(args.generator, args.vae, args.classifier, device)
    policy = UniformPolicy() if args.policy == "random" else load_policy(args.policy, device)
    boards, glyphs = np.load(args.boards, mmap_mode="r"), np.load(args.glyphs)
    for removals in args.budgets:
        accuracy = evaluate(adapter, policy, boards, glyphs, removals, args.seed,
                            args.noise_capacity, args.microbatch, args.boards_per_difficulty)
        print(json.dumps({"R": removals, **accuracy}), flush=True)


if __name__ == "__main__":
    main()
