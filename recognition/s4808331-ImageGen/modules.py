"""Model components for WGAN-GP and the DCGAN baseline.

Three networks live here:

* :class:`Generator` -- shared by both models. Maps a 128-d vector to a
  single-channel 64x128 slice with a ``tanh`` output.
* :class:`Critic` -- the WGAN-GP critic. No batch normalisation, unbounded
  scalar output.
* :class:`Discriminator` -- the DCGAN baseline. Batch normalisation, single
  logit for binary cross-entropy.

The generator is deliberately identical for both models so that the comparison
between them isolates the effect of the loss function rather than mixing it
with pipeline differences.

Spatial sizes follow the data: slices are 64x128 (see dataset.py), so the
generator starts from a 4x8 feature map and doubles four times
(4x8 -> 8x16 -> 16x32 -> 32x64 -> 64x128) instead of the square 4x4 seed used in
most DCGAN references.

This module depends only on PyTorch, as required by the project specification --
no NumPy, and no import of dataset.py ot avoid any NumPy leaks.

Scaffolded, debugged, docstrings and some documenting by Claude Opus 5.5
"""

from __future__ import annotations

import torch
from torch import nn

# default latent dimensions
LATENT_DIM = 128

# Default output size as (height, width), mathces var in dataset.py.
IMAGE_SIZE = (64, 128)

# Channel width of the shallowest feature map. Deeper layers scale this by
# 2, 4 and 8. size was guestimated to fit on the hardware resources. 8 GB VRAM
BASE_CHANNELS = 64

# Number of 2x resolution changes between the latent seed and the image.
N_BLOCKS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_size(image_size: tuple[int, int] = IMAGE_SIZE, n_blocks: int = N_BLOCKS) -> tuple[int, int]:
    """Spatial size of the generator's first feature map.

    Each block doubles both axes, so the seed is the image size halved
    ``n_blocks`` times. Raises if that does not divide evenly, which would
    otherwise surface as a confusing shape mismatch deep in the network.
    """
    ...


def weights_init(module: nn.Module) -> None:
    """DCGAN weight initialisation (Radford et al., 2016, Section 4).

    Convolutions start from a normal distribution with standard deviation 0.02
    and normalisation layers from a mean of 1. GANs are sensitive to the
    starting point: default PyTorch initialisation tends to make the critic win
    too early, which starves the generator of useful gradients.

    Apply with ``model.apply(weights_init)``.
    """
    ...


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, for the resource table in the report."""
    ...


def _critic_norm(kind: str, channels: int) -> nn.Module:
    """Build the critic's normalisation layer.

    Batch normalisation is invalid in a WGAN-GP critic: the gradient penalty is
    defined per sample, but batch norm makes each output depend on the rest of
    the batch, so the penalty no longer constrains what it is meant to
    (Gulrajani et al., 2017, Section 4). Layer normalisation is the recommended
    substitute and is implemented here as ``GroupNorm`` with a single group,
    which normalises over channels and space without needing the spatial size
    declared up front.
    """
    ...


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """Maps a latent vector to a single-channel MRI slice.

    The latent vector is projected to a small, deep feature map and then
    repeatedly doubled in resolution while thinning in channels, until it
    reaches the image size. ``tanh`` puts the output in [-1, 1], matching the
    normalisation applied in ``dataset.py``.

    Scaffolded and debugged by Claude

    Args:
        latent_dim: Size of the input noise vector.
        image_size: Output size as (height, width).
        base_channels: Channel width of the final feature map.
        out_channels: Image channels; 1 for greyscale MR.
        n_blocks: Number of 2x upsampling stages.
        upsample: ``"transpose"`` for strided transposed convolutions (the
            DCGAN/WGAN-GP reference design), or ``"nearest"`` for nearest-
            neighbour upsampling followed by a plain convolution. Transposed
            convolutions can leave periodic checkerboard artifacts; the
            ``"nearest"`` variant is the standard remedy and is kept available
            so the failure autopsy can test that explanation rather than
            assert it.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        image_size: tuple[int, int] = IMAGE_SIZE,
        base_channels: int = BASE_CHANNELS,
        out_channels: int = 1,
        n_blocks: int = N_BLOCKS,
        upsample: str = "transpose",
    ) -> None:
        ...

    def _block(self, in_channels: int, out_channels: int) -> list[nn.Module]:
        """One 2x upsampling stage: normalise and activate after upsampling."""
        ...

    def _to_image(self, in_channels: int, out_channels: int) -> list[nn.Module]:
        """Final 2x stage. No normalisation here -- it would fight the tanh."""
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from latent vectors of shape (N, latent_dim)."""
        ...

    def sample_latent(self, n: int, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` latent vectors from the standard normal prior.

        Passing an explicit ``generator`` gives reproducible, fixed-seed sample
        grids, which the specification requires for uncurated evaluation.
        """
        ...

    @torch.no_grad()
    def generate(self, n: int, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
        """Convenience wrapper: sample latents and run them through the model."""
        ...


# ---------------------------------------------------------------------------
# WGAN-GP critic
# ---------------------------------------------------------------------------


class Critic(nn.Module):
    """Scores an image with an unbounded number.

    This is not a classifier. It estimates the Wasserstein distance between the
    real and generated distributions, so the output has no sigmoid and no fixed
    range -- only the *difference* between its scores on real and fake batches
    is meaningful. Keeping it unbounded is what gives the generator a usable
    gradient even when the two distributions do not overlap, which is the
    failure mode that makes normal GAN training collapse.

    Args:
        image_size: Input size as (height, width).
        in_channels: Image channels.
        base_channels: Channel width of the first feature map.
        n_blocks: Number of 2x downsampling stages.
        norm: ``"layer"`` (default), ``"instance"`` or ``"none"``. Batch
            normalisation is intentionally not an option -- see
            :func:`_critic_norm`.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_SIZE,
        in_channels: int = 1,
        base_channels: int = BASE_CHANNELS,
        n_blocks: int = N_BLOCKS,
        norm: str = "layer",
    ) -> None:
        ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one score per image, shape (N,)."""
        ...


# ---------------------------------------------------------------------------
# DCGAN baseline discriminator
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """Binary real/fake classifier for the DCGAN baseline.

    Structurally the same as :class:`Critic`, with two deliberate differences
    that define the baseline: batch normalisation between stages, and an output
    interpreted as a logit for binary cross-entropy.

    The logit is returned raw, without a sigmoid, so training can use
    ``BCEWithLogitsLoss``. That is numerically stabler than a sigmoid followed
    by ``BCELoss``, which saturates and produces no gradient exactly when the
    discriminator is winning.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_SIZE,
        in_channels: int = 1,
        base_channels: int = BASE_CHANNELS,
        n_blocks: int = N_BLOCKS,
    ) -> None:
        ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one logit per image, shape (N,)."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_models(
    model: str = "wgan-gp",
    latent_dim: int = LATENT_DIM,
    image_size: tuple[int, int] = IMAGE_SIZE,
    base_channels: int = BASE_CHANNELS,
    upsample: str = "transpose",
    critic_norm: str = "layer",
) -> tuple[Generator, nn.Module]:
    """Build an initialised (generator, discriminator-or-critic) pair.

    Args:
        model: ``"wgan-gp"`` for the critic, ``"dcgan"`` for the baseline
            discriminator. The generator is the same either way.

    Returns:
        Both networks, with DCGAN weight initialisation already applied.
    """
    ...


# ---------------------------------------------------------------------------
# Shape and sanity check
# ---------------------------------------------------------------------------
