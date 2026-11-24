"""FIDgrad: reward a sample by how much adding it would lower FID.

With generated feature moments (mu_g, Sigma_g), real moments (mu_r, Sigma_r)
and u = z - mu_g for a sample's Inception feature z, the paper's closed form
(Proposition 1) is

    r(z) = -2 (mu_g - mu_r)^T u - u^T (I - B) u + tr[(I - B) Sigma_g],
    B = Sigma_r^1/2 (Sigma_r^1/2 Sigma_g Sigma_r^1/2)^-1/2 Sigma_r^1/2,

the negative derivative of FID when an infinitesimal mass is moved onto z.

The generated moments follow the policy: an exponential moving average of the
mean and the raw second moment M, with Sigma_g = sym(M - mu_g mu_g^T). A batch
is always scored before it updates the moments. Everything is float64.

As a GRPO reward:

    inception = load_inception("external/FD-Loss", device)
    fidgrad = FIDgrad(real_mean, real_covariance, initial_mean, initial_raw_second_moment)
    rewards = fidgrad.reward(inception_features(inception, images), group_ids, seed)

The initial moments are the Inception mean and raw second moment of samples
from the frozen generator in random order (scripts/build_fidgrad_init.py for
MAR-B). Save
``fidgrad.state_dict()`` with the training state to resume.
"""

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist

from ponder.rng import Stream, bounded_index, keyed_u64

INCEPTION_BATCH = 32


def load_inception(fd_loss, device):
    """FD-Loss's TF-compatible Inception-v3 (Jiawei-Yang/FD-Loss at 5c03b81) for images in [0, 1]."""
    spec = importlib.util.spec_from_file_location(
        "fd_loss_perception_util", Path(fd_loss) / "utils" / "perception_util.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_inception(device=device, normalize=False)


@torch.no_grad()
def inception_features(inception, images: torch.Tensor) -> torch.Tensor:
    """2048-d pool features [B,2048] of float32 images in [0, 1]."""
    return torch.cat([inception(images[i:i + INCEPTION_BATCH].clamp(0, 1))[0].float()
                      for i in range(0, len(images), INCEPTION_BATCH)])


def _symmetric(matrix):
    return (matrix + matrix.T) * 0.5


def _eigh_psd(matrix):
    """Eigendecomposition of a PSD matrix, with numerically zero eigenvalues set to 0."""
    values, vectors = torch.linalg.eigh(_symmetric(matrix))
    tolerance = torch.finfo(values.dtype).eps * values.numel() * float(values.abs().max())
    return values.masked_fill(values <= tolerance, 0.0), vectors


class FIDgrad:
    def __init__(self, real_mean, real_covariance, generated_mean, generated_raw_second_moment,
                 ema_decay: float = 0.001, device="cuda"):
        as64 = lambda x: torch.as_tensor(x).to(device, torch.float64)  # noqa: E731
        self.real_mean = as64(real_mean)
        self.real_covariance = as64(real_covariance)
        self.mean = as64(generated_mean)
        self.raw_second_moment = as64(generated_raw_second_moment)
        self.ema_decay = ema_decay
        values, vectors = _eigh_psd(self.real_covariance)
        self.real_sqrt = (vectors * values.sqrt()) @ vectors.T
        self._linearize()

    def _linearize(self):
        """Precompute the terms of r(z) at the current generated moments."""
        covariance = _symmetric(self.raw_second_moment - torch.outer(self.mean, self.mean))
        values, vectors = _eigh_psd(self.real_sqrt @ covariance @ self.real_sqrt)
        # (Sigma_r^1/2 Sigma_g Sigma_r^1/2)^-1/2, pseudoinverse on singular directions.
        scaled = torch.where(values > 0, vectors / values.sqrt(), 0.0)
        identity = torch.eye(len(self.mean), dtype=torch.float64, device=self.mean.device)
        self.i_minus_b = _symmetric(identity - self.real_sqrt @ (scaled @ vectors.T) @ self.real_sqrt)
        self.mean_gradient = 2.0 * (self.mean - self.real_mean)
        self.trace_term = torch.sum(self.i_minus_b * covariance)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Rewards [B] (float32) for Inception features [B,F] at the current moments."""
        u = features.to(torch.float64) - self.mean[None]
        derivative = u @ self.mean_gradient + ((u @ self.i_minus_b) * u).sum(dim=1) - self.trace_term
        return (-derivative).float()

    def update(self, features: torch.Tensor):
        """EMA step towards the moments of ``features``, pooled over all DDP ranks."""
        features = features.to(torch.float64)
        count = torch.tensor(float(len(features)), dtype=torch.float64, device=features.device)
        total = features.sum(dim=0)
        raw_total = features.T @ features
        if dist.is_initialized():
            for tensor in (count, total, raw_total):
                dist.all_reduce(tensor)
        alpha = self.ema_decay
        self.mean = (1.0 - alpha) * self.mean + alpha * (total / count)
        self.raw_second_moment = (1.0 - alpha) * self.raw_second_moment + alpha * (raw_total / count)
        self._linearize()

    def reward(self, features: torch.Tensor, group_ids: torch.Tensor, seed: int) -> torch.Tensor:
        """Score a batch of GRPO groups (runs of equal ``group_ids``), then update
        the moments with one keyed member of every group."""
        rewards = self.score(features)
        groups, sizes = torch.unique_consecutive(group_ids, return_counts=True)
        size = int(sizes[0])
        members = [bounded_index(keyed_u64(seed, Stream.FIDGRAD_EMA_MEMBER, group), size)
                   for group in groups.tolist()]
        rows = torch.arange(len(groups), device=features.device) * size
        self.update(features[rows + torch.tensor(members, device=features.device)])
        return rewards

    def state_dict(self):
        return {"generated_mean": self.mean.cpu(),
                "generated_raw_second_moment": self.raw_second_moment.cpu()}

    def load_state_dict(self, state):
        self.mean = state["generated_mean"].to(self.mean)
        self.raw_second_moment = state["generated_raw_second_moment"].to(self.raw_second_moment)
        self._linearize()
