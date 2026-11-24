"""Export MaskGen-KL-XL samples for the official GenEval benchmark (553 prompts,
4 images each) over rethinking budgets.

    python scripts/eval_maskgen_geneval.py --budgets 0 4 --output runs/geneval \
        --metadata external/geneval/prompts/evaluation_metadata.jsonl

Then score each budget directory (e.g. runs/geneval/r004) with official GenEval
(djghosh13/geneval at af4902f), for example in the geneval_server image. See
geneval_server/Dockerfile for the command.

Generation runs under bf16 autocast. Image ``j`` of prompt ``i`` uses generator
noise keyed by ``4i + j``. The noise capacity stays 32 for every budget. The
paper used ``--batch 64`` on one RTX 5090. Exact samples also depend on the GPU
and the batch.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from ponder.baselines import UniformPolicy
from ponder.generators.maskgen import MaskGenAdapter, generate
from ponder.policy import load_policy

IMAGES_PER_PROMPT = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/maskgen_geneval.pt",
                        help="policy file, or 'random' for random order")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="official GenEval prompts/evaluation_metadata.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--prompts", type=int, default=553, help="use the first N prompts")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-capacity", type=int, default=32)
    parser.add_argument("--maskgen-source", type=Path, default=Path("external/1d-tokenizer"))
    args = parser.parse_args()
    if max(args.budgets) > args.noise_capacity:
        parser.error("the noise capacity must cover every budget")
    device = torch.device("cuda")

    adapter = MaskGenAdapter(args.maskgen_source, device)
    policy = UniformPolicy() if args.policy == "random" else load_policy(args.policy, device)
    rows = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    rows = rows[:args.prompts]
    text = adapter.encode([row["prompt"] for row in rows], torch.bfloat16)
    samples = len(rows) * IMAGES_PER_PROMPT
    for removals in args.budgets:
        root = args.output / f"r{removals:03d}"
        for index, row in enumerate(rows):
            (root / f"{index:05d}" / "samples").mkdir(parents=True, exist_ok=True)
            (root / f"{index:05d}" / "metadata.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n")
        for start in range(0, samples, args.batch):
            ordinals = torch.arange(start, min(start + args.batch, samples), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                images = generate(adapter, policy, removals, text[ordinals // IMAGES_PER_PROMPT],
                                  ordinals, args.seed, args.noise_capacity)
            # MaskGen's 8-bit conversion truncates.
            arrays = images.mul(255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for ordinal, array in zip(ordinals.tolist(), arrays):
                prompt, image = divmod(ordinal, IMAGES_PER_PROMPT)
                Image.fromarray(array).save(root / f"{prompt:05d}" / "samples" / f"{image:04d}.png")
        print(json.dumps({"policy": args.policy, "removals": removals, "images": samples,
                          "output": str(root)}), flush=True)


if __name__ == "__main__":
    main()
