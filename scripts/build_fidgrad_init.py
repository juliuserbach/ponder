"""Build the FIDgrad moment initialization for MAR-B training.

    torchrun --nproc-per-node=8 scripts/build_fidgrad_init.py

Inception mean and raw second moment of 50,048 images generated in random
order (uniform policy) with the R=8 training schedule, uniformly sampled
ImageNet classes and seed 42. Rank ``r`` of ``W`` generates samples
``r, r + W, ...``. With 8 GPUs these are the batches of the paper's run.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ponder.baselines import UniformPolicy
from ponder.fidgrad import inception_features, load_inception
from ponder.generators.mar import TOKEN_WIDTH, TOKENS, MARAdapter, imagenet_classes
from ponder.noise import NoiseBank
from ponder.sampler import sample
from ponder.schedules import mar_schedule
from ponder.train import setup_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mar_fidgrad_init.npz"))
    parser.add_argument("--samples", type=int, default=50_048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mar-source", default="external/mar")
    parser.add_argument("--fd-loss", default="external/FD-Loss")
    parser.add_argument("--model", default="checkpoints/mar/checkpoint-last.pth")
    parser.add_argument("--vae", default="checkpoints/mar/kl16.ckpt")
    args = parser.parse_args()
    rank, world_size, device = setup_distributed()

    adapter = MARAdapter(args.mar_source, args.model, args.vae, device)
    inception = load_inception(args.fd_loss, device)
    schedule = mar_schedule(8)
    total = torch.zeros(2048, dtype=torch.float64)
    raw_total = torch.zeros(2048, 2048, dtype=torch.float64)
    for ordinals in torch.arange(rank, args.samples, world_size).split(args.batch):
        ordinals = ordinals.to(device)
        noise = NoiseBank(args.seed, ordinals, 1, TOKENS, TOKEN_WIDTH, schedule.removals)
        canvas = torch.zeros(len(ordinals), TOKENS, TOKEN_WIDTH, device=device)
        writable = torch.ones(len(ordinals), TOKENS, dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rollout = sample(adapter, UniformPolicy(), schedule, noise,
                             imagenet_classes(args.seed, ordinals), canvas, writable)
        features = inception_features(inception, rollout.images.float()).cpu().double()
        total += features.sum(dim=0)
        raw_total += features.T @ features
    if world_size > 1:
        sums = [total.to(device), raw_total.to(device)]
        for tensor in sums:
            dist.all_reduce(tensor)
        total, raw_total = (tensor.cpu() for tensor in sums)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw_second_moment = (raw_total / args.samples + raw_total.T / args.samples) * 0.5
        np.savez(args.output, generated_mean=(total / args.samples).numpy(),
                 generated_raw_second_moment=raw_second_moment.numpy())
        print(f"wrote {args.output} from {args.samples} samples")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
