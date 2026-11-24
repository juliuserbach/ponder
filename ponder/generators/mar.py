"""MAR-B on ImageNet 256 (LTH14/mar at c6d53f7): CFG 1.0, DDIM 25, KL16 VAE.

The policy feature of a position is MAR's decoder output there, the same
vector that conditions its diffusion head. MAR's own ``forward_mae_encoder``
assumes every sample has the same number of visible tokens. With rethinking
that no longer holds, so the encoder pass is reimplemented with padding.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from ponder.fidgrad import FIDgrad, inception_features, load_inception
from ponder.rng import Stream, bounded_index, keyed_u64
from ponder.schedules import mar_schedule

TOKENS = 256
TOKEN_WIDTH = 16
WIDTH = 768
PREFIX = 64  # MAR's class-embedding buffer tokens
LATENT_SCALE = 0.2325


class MARAdapter:
    def __init__(self, source, model, vae, device):
        sys.path.insert(0, str(source))
        from models.mar import mar_base
        from models.vae import AutoencoderKL

        self.mar = mar_base(img_size=256, vae_stride=16, patch_size=1, vae_embed_dim=TOKEN_WIDTH,
                            buffer_size=PREFIX, diffloss_d=6, diffloss_w=1024,
                            num_sampling_steps="ddim25")
        self.mar.load_state_dict(torch.load(model, map_location="cpu", weights_only=True)["model_ema"])
        self.vae = AutoencoderKL(embed_dim=TOKEN_WIDTH, ch_mult=(1, 1, 2, 2, 4))
        self.vae.load_state_dict(torch.load(vae, map_location="cpu", weights_only=True)["model"])
        for module in (self.mar, self.vae):
            module.to(device).eval().requires_grad_(False)
        # Compiled as in the paper's runs: in bfloat16, eager kernels round
        # differently and give visibly different samples.
        self.denoiser = torch.compile(self.mar.diffloss.net.forward)

    @torch.no_grad()
    def contextualize(self, canvas, visible, cond):
        mar = self.mar
        batch = canvas.shape[0]
        x = mar.z_proj(canvas.masked_fill(~visible[..., None], 0))
        prefix = mar.class_emb(cond).to(x.dtype)[:, None].expand(-1, PREFIX, -1)
        x = torch.cat((prefix, x), dim=1)
        x = mar.z_proj_ln(x + mar.encoder_pos_embed_learned.to(x.dtype))

        # The encoder sees the prefix and the visible tokens, packed to the left
        # and padded to the longest sample.
        keep = torch.cat((visible.new_ones(batch, PREFIX), visible), dim=1)
        lengths = keep.sum(dim=1)
        length = int(lengths.max())
        slot = keep.cumsum(dim=1) - 1
        rows, columns = torch.nonzero(keep, as_tuple=True)
        packed = torch.zeros(batch, length, dtype=torch.long, device=canvas.device)
        packed[rows, slot[rows, columns]] = columns
        padding = torch.arange(length, device=canvas.device) >= lengths[:, None]  # [B,L]
        x = x.gather(1, packed[..., None].expand(-1, -1, WIDTH)).masked_fill(padding[..., None], 0)
        key_mask = torch.zeros(batch, 1, 1, length, device=canvas.device, dtype=x.dtype)
        key_mask = key_mask.masked_fill(padding[:, None, None], float("-inf"))
        for block in mar.encoder_blocks:
            x = block(x, attn_mask=key_mask).masked_fill(padding[..., None], 0)
        x = mar.encoder_norm(x).masked_fill(padding[..., None], 0)

        # The decoder sees every position, with mask tokens where nothing is visible.
        encoded = mar.decoder_embed(x)
        x = mar.mask_token.to(encoded.dtype).expand(batch, PREFIX + TOKENS, -1).clone()
        x[rows, columns] = encoded[rows, slot[rows, columns]]
        x = x + mar.decoder_pos_embed_learned.to(encoded.dtype)
        for block in mar.decoder_blocks:
            x = block(x)
        x = mar.decoder_norm(x)
        features = x[:, PREFIX:] + mar.diffusion_pos_embed_learned.to(x.dtype)
        return features, features

    @torch.no_grad()
    def sample(self, context, positions, noise, masked_after):
        condition = context.gather(1, positions[..., None].expand(-1, -1, WIDTH))
        noise = noise.reshape(-1, TOKEN_WIDTH)
        tokens = self.mar.diffloss.gen_diffusion.ddim_sample_loop(
            self.denoiser, tuple(noise.shape), noise=noise, clip_denoised=False,
            model_kwargs={"c": condition.reshape(-1, WIDTH)}, progress=False, eta=0.0)
        return tokens.reshape(*positions.shape, TOKEN_WIDTH)

    @torch.no_grad()
    def decode(self, canvas, cond):
        images = self.vae.decode(self.mar.unpatchify(canvas) / LATENT_SCALE)
        return ((images + 1) / 2).clamp(0, 1)


def imagenet_classes(seed: int, group_ids: torch.Tensor) -> torch.Tensor:
    """One uniform ImageNet class per group id."""
    classes = [bounded_index(keyed_u64(seed, Stream.IMAGENET_CLASS, group), 1000)
               for group in group_ids.tolist()]
    return torch.tensor(classes, dtype=torch.long, device=group_ids.device)


class Task:
    """FIDgrad training on uniformly sampled ImageNet classes."""

    positions = TOKENS
    token_width = TOKEN_WIDTH

    def __init__(self, config, device, seed):
        self.seed = seed
        self.device = device
        self.adapter = MARAdapter(config["source"], config["model"], config["vae"], device)
        self.schedule = mar_schedule(config["removals"])
        self.noise_capacity = config["noise_capacity"]
        self.inception = load_inception(config["fd_loss"], device)
        real = np.load(Path(config["fd_loss"]) / "data" / "fid_stats" / "guided_diffusion_stats.npz")
        init = np.load(config["fidgrad_init"])
        self.fidgrad = FIDgrad(real["mu"], real["sigma"], init["generated_mean"],
                               init["generated_raw_second_moment"], config["ema_decay"], device)

    def batch(self, update, group_ids):
        cond = imagenet_classes(self.seed, group_ids)
        canvas = torch.zeros(len(group_ids), TOKENS, TOKEN_WIDTH, device=self.device)
        writable = torch.ones(len(group_ids), TOKENS, dtype=torch.bool, device=self.device)
        return self.schedule, cond, canvas, writable

    def reward(self, rollout, cond, group_ids):
        features = inception_features(self.inception, rollout.images.float())
        return self.fidgrad.reward(features, group_ids, self.seed)

    def state_dict(self):
        return {"fidgrad": self.fidgrad.state_dict()}

    def load_state_dict(self, state):
        self.fidgrad.load_state_dict(state["fidgrad"])
