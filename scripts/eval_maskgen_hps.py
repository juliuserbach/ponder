"""MaskGen-KL-XL on the HPSv2 benchmark (3,200 prompts): HPSv2.1, PickScore and
ImageReward over rethinking budgets.

    python scripts/eval_maskgen_hps.py --budgets 0 1 2 4 8 16 32
    python scripts/eval_maskgen_hps.py --policy random --budgets 0 --metrics hpsv2

Generation and scoring run in float32. Prompt ``i`` uses generator noise keyed
by ``103,672 + i``, after the 103,672 HPDv2 training prompts, as in the paper.
The noise capacity stays 32 for every budget. The paper used ``--batch 64`` on
one RTX 5090. Exact samples also depend on the GPU and the batch.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

from ponder.baselines import UniformPolicy
from ponder.generators.maskgen import MaskGenAdapter, generate
from ponder.policy import load_policy
from ponder.rewards.hpsv2 import HPSv2, clip_pixels

BENCHMARK = ("zhwang/HPDv2", "b9517430f34b1080d4f741118d2a440155a165cd")
STYLES = ("anime", "concept-art", "paintings", "photo")
NOISE_OFFSET = 103_672


def benchmark_prompts():
    """The 3,200 HPSv2 benchmark prompts, 800 per style."""
    name, revision = BENCHMARK
    prompts = []
    for style in STYLES:
        path = hf_hub_download(name, f"benchmark/{style}.json", repo_type="dataset", revision=revision)
        prompts += json.loads(Path(path).read_text())
    return prompts


class PickScore:
    """yuvalkirstain/PickScore_v1: learned temperature times the image-prompt cosine."""

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.text = {}  # prompt -> normalized text embedding

    @torch.no_grad()
    def __call__(self, images, prompts):
        pooled = self.model.vision_model(pixel_values=clip_pixels(images)).pooler_output
        image = F.normalize(self.model.visual_projection(pooled).float(), dim=-1)
        missing = [prompt for prompt in dict.fromkeys(prompts) if prompt not in self.text]
        if missing:
            tokens = self.tokenizer(missing, padding=True, truncation=True, max_length=77,
                                    return_tensors="pt").to(images.device)
            text = self.model.text_projection(self.model.text_model(**tokens).pooler_output)
            self.text.update(zip(missing, F.normalize(text.float(), dim=-1)))
        text = torch.stack([self.text[prompt] for prompt in prompts])
        return self.model.logit_scale.exp().float() * (image * text).sum(dim=-1)


class ImageRewardScore:
    """THUDM ImageReward-v1.0 on 8-bit images."""

    def __init__(self, device):
        import ImageReward

        self.model = ImageReward.load("ImageReward-v1.0", device=str(device))

    @torch.no_grad()
    def __call__(self, images, prompts):
        from PIL import Image

        arrays = images.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        scores = [float(self.model.score(prompt, Image.fromarray(array)))
                  for prompt, array in zip(prompts, arrays)]
        return torch.tensor(scores, device=images.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/maskgen_hpsv2.pt",
                        help="policy file, or 'random' for random order")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--metrics", nargs="+", default=["hpsv2", "pickscore", "imagereward"])
    parser.add_argument("--prompts", type=int, default=3200, help="use the first N benchmark prompts")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-capacity", type=int, default=32)
    parser.add_argument("--maskgen-source", type=Path, default=Path("external/1d-tokenizer"))
    parser.add_argument("--hps-checkpoint", default="checkpoints/HPS_v2.1_compressed.pt")
    args = parser.parse_args()
    if max(args.budgets) > args.noise_capacity:
        parser.error("the noise capacity must cover every budget")
    device = torch.device("cuda")

    adapter = MaskGenAdapter(args.maskgen_source, device)
    policy = UniformPolicy() if args.policy == "random" else load_policy(args.policy, device)
    metrics = {"hpsv2": lambda: HPSv2(args.hps_checkpoint, device),
               "pickscore": lambda: PickScore(device), "imagereward": lambda: ImageRewardScore(device)}
    metrics = {name: metrics[name]() for name in args.metrics}
    prompts = benchmark_prompts()[:args.prompts]
    for removals in args.budgets:
        totals = dict.fromkeys(metrics, 0.0)
        for start in range(0, len(prompts), args.batch):
            batch = prompts[start:start + args.batch]
            group_ids = NOISE_OFFSET + torch.arange(start, start + len(batch), device=device)
            images = generate(adapter, policy, removals, adapter.encode(batch), group_ids,
                              args.seed, args.noise_capacity).float()
            for name, metric in metrics.items():
                totals[name] += metric(images, batch).double().sum().item()
        means = {name: total / len(prompts) for name, total in totals.items()}
        if "hpsv2" in means:
            means["hpsv2"] *= 100  # the benchmark convention
        print(json.dumps({"policy": args.policy, "removals": removals, "prompts": len(prompts),
                          **means}), flush=True)


if __name__ == "__main__":
    main()
