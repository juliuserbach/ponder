"""MAR-B FID, Inception Score and FDr-5 over rethinking budgets (ImageNet 256).

    torchrun --nproc-per-node=8 scripts/eval_mar.py --budgets 0 8 32
    python scripts/eval_mar.py --policy random --budgets 0 --samples 1000

Sample ``i`` has class ``i % 1000`` (50 per class for 50,000 samples) and
generator noise keyed by ``(seed, i)``. The noise capacity stays fixed across
budgets. Metrics come from the FD-Loss evaluator (Jiawei-Yang/FD-Loss at
5c03b81). FDr-5 is FDr-6 without Inception, the space FIDgrad optimizes.
Exact samples also depend on the GPU, the world size and ``--batch``.
"""

import argparse
import contextlib
import importlib
import json
import math
import sys
from pathlib import Path

import torch

from ponder.baselines import UniformPolicy
from ponder.generators.mar import TOKEN_WIDTH, TOKENS, MARAdapter
from ponder.noise import NoiseBank
from ponder.policy import load_policy
from ponder.sampler import sample
from ponder.schedules import mar_schedule
from ponder.train import setup_distributed

FDR5 = ("convnext", "dinov2_cls", "mae_cls", "siglip_cls", "clip_cls")


def generate(adapter, policy, removals, ordinals, seed=42, noise_capacity=32):
    """Images [B,3,256,256] for the given sample ordinals at budget ``removals``."""
    noise = NoiseBank(seed, ordinals, 1, TOKENS, TOKEN_WIDTH, noise_capacity)
    canvas = torch.zeros(len(ordinals), TOKENS, TOKEN_WIDTH, device=ordinals.device)
    writable = torch.ones(len(ordinals), TOKENS, dtype=torch.bool, device=ordinals.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return sample(adapter, policy, mar_schedule(removals), noise, ordinals % 1000,
                      canvas, writable).images.float()


def load_fd_loss(root: Path):
    """FD-Loss's evaluator module and its six feature extractors."""
    backend = torch.backends
    flags = (backend.cuda.matmul.allow_tf32, backend.cudnn.allow_tf32,
             backend.cudnn.benchmark, backend.cudnn.deterministic)
    sys.path.insert(0, str(root))
    try:
        evaluator = importlib.import_module("eval_all_fds")
    finally:
        sys.path.remove(str(root))
        # The import enables TF32 and cuDNN autotuning. The paper's numbers use
        # PyTorch's defaults for both generation and feature extraction.
        (backend.cuda.matmul.allow_tf32, backend.cudnn.allow_tf32,
         backend.cudnn.benchmark, backend.cudnn.deterministic) = flags
    with contextlib.chdir(root):  # reference statistics live in data/fid_stats
        models = evaluator._load_repr_models(evaluator.DEFAULT_MODELS, evaluator.TARGET_SIZE,
                                             target_size_overrides=evaluator.DEFAULT_TARGET_SIZES)
    return evaluator, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/mar_fidgrad.pt",
                        help="policy file, or 'random' for random order")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 8, 32])
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=32, help="samples per rollout and GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-capacity", type=int, default=32)
    parser.add_argument("--mar-source", type=Path, default=Path("external/mar"))
    parser.add_argument("--fd-loss", type=Path, default=Path("external/FD-Loss"))
    parser.add_argument("--model", default="checkpoints/mar/checkpoint-last.pth")
    parser.add_argument("--vae", default="checkpoints/mar/kl16.ckpt")
    args = parser.parse_args()
    if max(args.budgets) > args.noise_capacity:
        parser.error("the noise capacity must cover every budget")
    rank, world_size, device = setup_distributed()
    if args.samples % world_size:
        parser.error("--samples must be a multiple of the number of GPUs")

    adapter = MARAdapter(args.mar_source, args.model, args.vae, device)
    policy = UniformPolicy() if args.policy == "random" else load_policy(args.policy, device)
    evaluator, models = load_fd_loss(args.fd_loss.resolve())
    local = args.samples // world_size
    for removals in args.budgets:
        accumulators, inception_index = evaluator._make_accumulators(models, device)
        logits = []
        for start in range(rank * local, (rank + 1) * local, args.batch):
            ordinals = torch.arange(start, min(start + args.batch, (rank + 1) * local), device=device)
            images = generate(adapter, policy, removals, ordinals, args.seed, args.noise_capacity)
            evaluator.accumulate_batch(images, models, accumulators, logits, {})
        distances, inception_score, _, _ = evaluator._reduce_and_compute(
            models, accumulators, logits, inception_index, {}, prc_ref_features={},
            prc_model_names=[], mmd_ref_features={}, mmd_model_names=[])
        fdr5 = math.fsum(distances[name] / evaluator.FDR_VALIDATION_FD[name] for name in FDR5) / 5
        if rank == 0:
            print(json.dumps({"policy": args.policy, "removals": removals,
                              "samples": local * world_size, "fid": distances["FID(ADM)"],
                              "inception_score": inception_score, "fdr5": fdr5}), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
