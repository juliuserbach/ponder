"""The frozen MNIST-Sudoku models.

An unconditional MAR (encoder-decoder transformer over the 81 cells) with a
flow-matching head generates 32-dimensional per-cell latents of an MNIST VAE.
A small CNN classifies the decoded cells for the reward.
"""

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F

CELLS = 81
BUFFER = 8  # MAR's prefix of learned buffer tokens
WIDTH = 256
LATENT = 32


class Attention(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x, mask):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(batch, length, WIDTH))


class Block(nn.Module):
    """Pre-norm transformer block (timm ViT layout)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(OrderedDict(
            fc1=nn.Linear(WIDTH, 4 * WIDTH), act=nn.GELU(), fc2=nn.Linear(4 * WIDTH, WIDTH)))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class SudokuMAR(nn.Module):
    """MAR backbone: 3 encoder and 3 decoder blocks, width 256."""

    def __init__(self):
        super().__init__()
        self.class_emb = nn.Embedding(1, WIDTH)  # unused, the model is unconditional
        self.fake_latent = nn.Parameter(torch.empty(1, WIDTH))
        self.z_proj = nn.Linear(LATENT, WIDTH)
        self.z_proj_ln = nn.LayerNorm(WIDTH, eps=1e-6)
        self.encoder_pos_embed_learned = nn.Parameter(torch.empty(1, BUFFER + CELLS, WIDTH))
        self.encoder_blocks = nn.ModuleList(Block() for _ in range(3))
        self.encoder_norm = nn.LayerNorm(WIDTH)
        self.decoder_embed = nn.Linear(WIDTH, WIDTH)
        self.mask_token = nn.Parameter(torch.empty(1, 1, WIDTH))
        self.decoder_pos_embed_learned = nn.Parameter(torch.empty(1, BUFFER + CELLS, WIDTH))
        self.decoder_blocks = nn.ModuleList(Block() for _ in range(3))
        self.decoder_norm = nn.LayerNorm(WIDTH)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.empty(1, CELLS, WIDTH))

    def forward(self, canvas, visible):
        """Latents [B,81,32] and visibility [B,81] -> per-cell conditions [B,81,256].

        As in MAR, the encoder sees only the buffer and the visible cells. Rows
        can have different numbers of visible cells, so they are packed to the
        left and padded. Padding is masked out of attention.
        """
        batch = canvas.shape[0]
        x = self.z_proj(canvas.masked_fill(~visible[..., None], 0))
        x = torch.cat((self.fake_latent.to(x).expand(batch, BUFFER, -1), x), dim=1)
        x = self.z_proj_ln(x + self.encoder_pos_embed_learned.to(x))

        keep = torch.cat((visible.new_ones(batch, BUFFER), visible), dim=1)
        lengths = keep.sum(dim=1)
        ranks = keep.cumsum(dim=1) - 1
        rows, columns = torch.nonzero(keep, as_tuple=True)
        packed = torch.zeros(batch, int(lengths.max()), dtype=torch.long, device=canvas.device)
        packed[rows, ranks[rows, columns]] = columns
        padding = torch.arange(packed.shape[1], device=canvas.device)[None] >= lengths[:, None]
        x = x.gather(1, packed[..., None].expand(-1, -1, WIDTH))
        mask = torch.zeros(batch, 1, 1, packed.shape[1], dtype=x.dtype, device=canvas.device)
        mask = mask.masked_fill(padding[:, None, None, :], float("-inf"))
        for block in self.encoder_blocks:
            x = block(x, mask)
        x = self.decoder_embed(self.encoder_norm(x))

        full = self.mask_token.to(x).expand(batch, BUFFER + CELLS, -1).clone()
        full[rows, columns] = x[rows, ranks[rows, columns]]
        x = full + self.decoder_pos_embed_learned.to(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return x[:, BUFFER:] + self.diffusion_pos_embed_learned.to(x)


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        frequencies = torch.exp(-math.log(10_000) * torch.arange(128, dtype=torch.float32) / 128)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))

    def forward(self, time):
        angles = time[:, None].float() * self.frequencies[None]
        return self.mlp(torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ln = nn.LayerNorm(WIDTH, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(WIDTH, 3 * WIDTH))

    def forward(self, x, condition):
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        return x + gate * self.mlp(modulate(self.in_ln(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = nn.LayerNorm(WIDTH, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(WIDTH, LATENT)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(WIDTH, 2 * WIDTH))

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class FlowHead(nn.Module):
    """Per-cell velocity v(x_t, t | condition): an AdaLN MLP with 3 residual blocks."""

    def __init__(self):
        super().__init__()
        self.time_embed = TimestepEmbedder()
        self.cond_embed = nn.Linear(WIDTH, WIDTH)
        self.input_proj = nn.Linear(LATENT, WIDTH)
        self.res_blocks = nn.ModuleList(ResidualBlock() for _ in range(3))
        self.final_layer = FinalLayer()

    def forward(self, x, time, condition):
        hidden = self.input_proj(x)
        condition = self.time_embed(time) + self.cond_embed(condition)
        for block in self.res_blocks:
            hidden = block(hidden, condition)
        return self.final_layer(hidden, condition)


class MNISTVAE(nn.Module):
    """Per-cell VAE: 28x28 tiles in [-1, 1] <-> 32-dimensional latents."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU())
        self.fc_mu = nn.Linear(256, LATENT)
        self.fc_logvar = nn.Linear(256, LATENT)  # unused, encoding takes the posterior mean
        self.decoder = nn.Sequential(nn.Linear(LATENT, 256), nn.SiLU(), nn.Linear(256, 256),
                                     nn.SiLU(), nn.Linear(256, 512), nn.SiLU(),
                                     nn.Linear(512, 784))

    def encode(self, tiles):
        """[T,784] -> posterior means [T,32]."""
        return self.fc_mu(self.encoder(tiles))

    def decode(self, latents):
        """[T,32] -> [T,784] in [-1, 1]."""
        return torch.sigmoid(self.decoder(latents)) * 2.0 - 1.0


class MNISTClassifier(nn.Module):
    """Digit classifier for 28x28 tiles in [0, 1]. Class 0 means an empty cell."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, tiles):
        x = self.pool(F.relu(self.conv1(tiles)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc2(F.relu(self.fc1(x.reshape(-1, 64 * 7 * 7))))
