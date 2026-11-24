"""MaskGen-KL-XL text-to-image (bytedance/1d-tokenizer at 942a96f): 32 continuous
tokens, DDIM 25, MaskGen's linear CFG schedule from 1 to 3, TA-TiTok decoder.

The policy feature of a position is the generator's prompt-conditioned decoder
output there, the vector that conditions its diffusion head. Prompts are
encoded with CLIP ViT-L/14-336, as in MaskGen: the penultimate layer conditions
the generator, the final layer guides the TA-TiTok decoder.
"""

import json
import sys
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download

from ponder.noise import NoiseBank
from ponder.rewards.geneval import GenEval
from ponder.rewards.hpsv2 import HPSv2
from ponder.rng import Stream, bounded_index, keyed_u64
from ponder.sampler import sample
from ponder.schedules import maskgen_schedule

GENERATOR = ("turkeyju/generator_maskgen_kl_xl", "2952dcff424e18ffe4b8d6938cebeee56a92cbb5")
TOKENIZER = ("turkeyju/tokenizer_tatitok_bl32_vae", "af85e787f2a4afd447c1e10fd6bef97cdb6e033f")
HPDV2 = ("ymhao/HPDv2", "bb0f2563e451e11c3f376b2ba93615db00e042e1")
TOKENS = 32
TOKEN_WIDTH = 16
LATENT_SCALE = 0.7525
AESTHETIC_SCORE = 6.5
DDIM_STEPS = 25


@dataclass
class Text:
    """CLIP conditioning for a batch of prompts."""

    prompts: list[str]
    sequence: torch.Tensor  # [B,77,768] penultimate layer, conditions the generator
    pooled: torch.Tensor  # [B,1,768] projected end-of-text token
    guidance: torch.Tensor  # [B,77,768] final layer, guides the image decoder

    def __getitem__(self, rows):
        """A slice or an index tensor of rows."""
        if isinstance(rows, slice):
            prompts = self.prompts[rows]
        else:
            prompts = [self.prompts[i] for i in rows.tolist()]
        return Text(prompts, self.sequence[rows], self.pooled[rows], self.guidance[rows])


class MaskGenAdapter:
    def __init__(self, source, device):
        sys.path.insert(0, str(source))
        import open_clip
        from modeling.diffusion import create_diffusion
        from modeling.maskgen import MaskGen_KL
        from modeling.tatitok import TATiTok

        name, revision = GENERATOR
        self.model = MaskGen_KL.from_pretrained(name, revision=revision, strict=True)
        name, revision = TOKENIZER
        self.tokenizer = TATiTok.from_pretrained(name, revision=revision, strict=True)
        self.clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        del self.clip.visual
        for module in (self.model, self.tokenizer, self.clip):
            module.to(device).eval().requires_grad_(False)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
        self.diffusion = create_diffusion(
            timestep_respacing=f"ddim{DDIM_STEPS}", noise_schedule="cosine", use_kl=False,
            sigma_small=False, predict_xstart=False, learn_sigma=True,
            rescale_learned_sigmas=False, diffusion_steps=1000)
        self.null = self.encode([""])  # the unconditional branch of CFG

    @torch.no_grad()
    def encode(self, prompts, dtype=torch.float32) -> Text:
        """MaskGen's ``open_clip_text_encoding``, plus the final layer for the decoder."""
        clip = self.clip
        tokens = self.clip_tokenizer(prompts).to(clip.token_embedding.weight.device)
        x = clip.token_embedding(tokens) + clip.positional_embedding
        for block in clip.transformer.resblocks[:-1]:
            x = block(x, attn_mask=clip.attn_mask)
        sequence = x
        guidance = clip.ln_final(clip.transformer.resblocks[-1](x, attn_mask=clip.attn_mask))
        pooled = guidance[torch.arange(len(prompts), device=tokens.device), tokens.argmax(dim=-1)]
        pooled = pooled @ clip.text_projection
        return Text(list(prompts), sequence.to(dtype), pooled[:, None].to(dtype), guidance.to(dtype))

    @torch.no_grad()
    def contextualize(self, canvas, visible, cond: Text):
        """One generator pass for the prompt and for the empty prompt (CFG)."""
        model = self.model
        batch = canvas.shape[0]
        null_sequence = self.null.sequence.to(cond.sequence.dtype).expand(batch, -1, -1)
        null_pooled = self.null.pooled.to(cond.pooled.dtype).expand(batch, -1, -1)
        condition = model.text_emb(torch.cat((cond.sequence, null_sequence)))
        aesthetic = torch.full((2 * batch,), AESTHETIC_SCORE, device=canvas.device)
        micro = model.get_sinusoidal_encoding(aesthetic * 100.0, model.micro_condition_embed_dim)
        pooled = torch.cat((torch.cat((cond.pooled, null_pooled)).squeeze(1), micro), dim=1)
        pooled = model.text_pooled_emb(pooled.unsqueeze(1)).squeeze(1)

        x = model.z_proj_ln(model.z_proj(torch.cat((canvas, canvas)) * LATENT_SCALE))
        mask_token = model.mask_token.to(x).expand_as(x)
        x = torch.where(torch.cat((visible, visible))[..., None], x, mask_token)
        x = x + model.decoder_pos_embed_learned
        for block in model.decoder_blocks:
            condition, x = block(x, condition, pooled)
        x = model.decoder_norm(x, pooled) + model.diffusion_pos_embed_learned
        conditional, unconditional = x.chunk(2)
        return conditional, (conditional, unconditional)

    @torch.no_grad()
    def sample(self, context, positions, noise, masked_after):
        batch, count = positions.shape
        conditional, unconditional = context
        index = positions[..., None].expand(-1, -1, conditional.shape[-1])
        condition = torch.cat((conditional.gather(1, index).flatten(0, 1),
                               unconditional.gather(1, index).flatten(0, 1)))
        # MaskGen's linear CFG schedule. It reads the number of positions still
        # masked after this round, which upstream clamps to at least one.
        remaining = masked_after.clamp_min(1).float()
        scale = 1.0 + 2.0 * (TOKENS - remaining) / TOKENS
        scale = scale[:, None].expand(batch, count).reshape(-1, 1)
        x = noise.reshape(-1, TOKEN_WIDTH)
        x = torch.cat((x, x))
        for step in reversed(range(DDIM_STEPS)):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x = self.diffusion.ddim_sample(
                self._guided_denoiser, x, t, clip_denoised=False,
                model_kwargs={"c": condition, "cfg_scale": scale}, eta=0.0)["sample"]
        return x[:batch * count].reshape(batch, count, TOKEN_WIDTH) / LATENT_SCALE

    def _guided_denoiser(self, x, t, c, cfg_scale):
        """The first half of ``x`` is denoised under both conditions. The result fills both halves."""
        half = x.shape[0] // 2
        output = self.model.diffloss.net(torch.cat((x[:half], x[:half])), t, c)
        epsilon, variance = output.split(TOKEN_WIDTH, dim=1)
        conditional, unconditional = epsilon.chunk(2)
        guided = unconditional + cfg_scale * (conditional - unconditional)
        return torch.cat((torch.cat((guided, guided)), variance), dim=1)

    @torch.no_grad()
    def decode(self, canvas, cond: Text):
        latent = canvas.permute(0, 2, 1).unsqueeze(2).contiguous()
        return self.tokenizer.decode_tokens(latent, cond.guidance).clamp(0.0, 1.0)


def generate(adapter, policy, removals, text: Text, group_ids, seed=42, noise_capacity=32):
    """Images [B,3,256,256] at budget ``removals``. Generator noise is keyed by ``group_ids``."""
    noise = NoiseBank(seed, group_ids, 1, TOKENS, TOKEN_WIDTH, noise_capacity)
    canvas = torch.zeros(len(group_ids), TOKENS, TOKEN_WIDTH, device=group_ids.device)
    writable = torch.ones(len(group_ids), TOKENS, dtype=torch.bool, device=group_ids.device)
    return sample(adapter, policy, maskgen_schedule(removals), noise, text, canvas, writable).images


def hpdv2_prompts():
    """The 103,672 unique HPDv2 training prompts, in first-occurrence order."""
    name, revision = HPDV2
    path = hf_hub_download(name, "train.json", repo_type="dataset", revision=revision)
    with open(path) as file:
        prompts = (row["prompt"].strip() for row in json.load(file))
    return list(dict.fromkeys(prompt for prompt in prompts if prompt))


class Task:
    """GRPO on HPDv2 prompts with HPSv2, or on Flow-GRPO's GenEval prompts with
    the dense GenEval reward. Every update draws one rethinking budget R for all
    groups. All members of a group share its prompt."""

    positions = TOKENS
    token_width = TOKEN_WIDTH

    def __init__(self, config, device, seed):
        self.seed = seed
        self.adapter = MaskGenAdapter(config["source"], device)
        self.budgets = config["removal_budgets"]
        self.noise_capacity = max(self.budgets)
        self.warm_start = config.get("warm_start", {"updates": 0})
        if config["reward"] == "hpsv2":
            self.prompts = hpdv2_prompts()
            self.prompt_stream = Stream.MASKGEN_TRAIN_PROMPT
            self.score = HPSv2(config["hps_checkpoint"], device)
        else:
            with open(config["prompts"]) as file:  # Flow-GRPO train_metadata.jsonl
                rows = [json.loads(line) for line in file if line.strip()]
            self.prompts = [row["prompt"] for row in rows]
            metadata = {row["prompt"]: row for row in rows}
            geneval = GenEval(config["server"])
            self.prompt_stream = Stream.GENEVAL_TRAIN_PROMPT
            self.score = lambda images, prompts: geneval(images, [metadata[p] for p in prompts])

    def batch(self, update, group_ids):
        if update < self.warm_start["updates"]:
            # A fixed budget, with noise drawn for that budget only.
            removals = self.noise_capacity = self.warm_start["removals"]
        else:
            word = keyed_u64(self.seed, Stream.MASKGEN_TRAIN_REMOVAL_BUDGET, update)
            removals = self.budgets[bounded_index(word, len(self.budgets))]
            self.noise_capacity = max(self.budgets)
        groups, sizes = torch.unique_consecutive(group_ids, return_counts=True)
        prompts = [self.prompts[bounded_index(keyed_u64(self.seed, self.prompt_stream, group),
                                              len(self.prompts))] for group in groups.tolist()]
        index = torch.arange(len(prompts), device=group_ids.device).repeat_interleave(sizes)
        cond = self.adapter.encode(prompts, torch.bfloat16)[index]
        canvas = torch.zeros(len(group_ids), TOKENS, TOKEN_WIDTH, device=group_ids.device)
        writable = torch.ones(len(group_ids), TOKENS, dtype=torch.bool, device=group_ids.device)
        return maskgen_schedule(removals), cond, canvas, writable

    def reward(self, rollout, cond, group_ids):
        return self.score(rollout.images.float(), cond.prompts)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
