"""Training, validation and checkpointing for WGAN-GP and the DCGAN baseline.

Both models are trained here on the same split, with the same generator
architecture and the same logging, so the only thing separating them is the
loss function and the optimiser settings each paper specifies.

Models and constants come from modules.py and data from dataset.py this file owns
everything that happens between them - the losses, the gradient penalty, the
critic/generator update schedule, checkpointing, and the loss and metric plots
the specification requires.

Two things worth noting before reading the loops:

* WGAN-GP trains the critic n_critic times per generator step, because the
  Wasserstein estimate is only meaningful when the critic is near optimal.
  DCGAN alternates 1:1, because a discriminator trained to optimality
  saturates and stops producing useful gradients.
* "Validation" means something different for a generative model than for a
  classifier - there is no held-out label to score. The validate split is used
  for distribution-level metrics instead, never for a training signal.

An "epoch" here is one pass over the training DataLoader, i.e. one head update
per batch. Under n_critic = 5 that is five critic updates for every generator
update, so WGAN-GP performs roughly a fifth as many generator steps per epoch
as the DCGAN baseline. Both generator and head step counts are logged every
epoch so the comparison in the report can be made on steps as well as epochs.

Scaffolded by Claude Opus 5.5
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

import dataset
import modules


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything that defines one training run.

    Saved alongside the checkpoint so a run can be reproduced from its own
    record rather than from memory.
    """

    # What to train
    model: str = "wgan-gp"                      # "wgan-gp" or "dcgan"

    data_root: str = "data"
    image_size: tuple[int, int] = modules.IMAGE_SIZE
    batch_size: int = 64
    num_workers: int = 2

    latent_dim: int = modules.LATENT_DIM
    base_channels: int = modules.BASE_CHANNELS
    upsample: str = "transpose"
    critic_norm: str = "layer"

    # Optimisation. Defaults below are the WGAN-GP paper's; the DCGAN baseline
    # overrides them in DCGAN_DEFAULTS.
    epochs: int = 100
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    n_critic: int = 5                           # critic steps per generator step
    gp_lambda: float = 10.0                     # gradient penalty weight

    # Reproducibility and output
    seed: int = 0
    sample_seed: int = 1234                     # fixed seed for grids
    out_dir: str = "outputs"
    sample_every: int = 1                       # epochs between sample grids
    checkpoint_every: int = 10                  # epochs between checkpoints


# Overrides that turn the config above into the DCGAN baseline. The generator
# is untouched: only the head, the loss and the optimiser settings change.
DCGAN_DEFAULTS: dict = {
    "model": "dcgan",
    "lr": 2e-4,
    "betas": (0.5, 0.999),
    "n_critic": 1,
    "gp_lambda": 0.0,
}


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Turn parsed arguments into a config, applying the DCGAN defaults first.

    Order matters: the baseline's optimiser settings are applied before the
    command line, so an explicit --lr still wins over DCGAN_DEFAULTS.
    """
    config = TrainConfig()
    if args.model == "dcgan":
        config = replace(config, **DCGAN_DEFAULTS)

    overrides = {
        key: value for key, value in vars(args).items()
        if value is not None and key != "model" and hasattr(config, key)
    }
    return replace(config, model=args.model, **overrides)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Start a training session with a specfic seed for exact reproducibility.
    off by default
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)            # no-op when CUDA is absent
    if deterministic:
        # benchmark=True lets cuDNN pick algorithms by timing them, which can
        # select a different (non-reproducible) kernel per run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    """Pick CUDA else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Gradient penalty enforcing the critic's Lipschitz constraint.
    """
    batch = real.size(0)
    # One mixing coefficient per image, broadcast over C/H/W so every image in
    # the batch is interpolated by a different amount.
    alpha = torch.rand(batch, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    scores = critic(interpolated)
    # create_graph=True keeps this gradient differentiable: the penalty is part
    # of the critic's loss, so backward() has to pass through it a second time.
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten to one vector per image, then one norm per image.
    norms = gradients.flatten(start_dim=1).norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean()


def critic_loss(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    gp_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wasserstein critic loss plus gradient penalty.

    Returns the loss to backpropagate and a dict of the components for logging:
    the raw Wasserstein estimate (real score minus fake score) is the number
    worth watching, since it should trend upward and then plateau.
    """
    real_score = critic(real).mean()
    fake_score = critic(fake).mean()
    penalty = gradient_penalty(critic, real, fake, device)

    # Minimising (fake - real) is the same as maximising the gap, which is what
    # the critic wants. The penalty is added, not subtracted: it is a cost.
    loss = fake_score - real_score + gp_lambda * penalty

    logs = {
        "critic_loss": loss.item(),
        "critic_real": real_score.item(),
        "critic_fake": fake_score.item(),
        "wasserstein": (real_score - fake_score).item(),
        "gradient_penalty": penalty.item(),
    }
    return loss, logs


def discriminator_loss(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary cross-entropy loss for the DCGAN baseline.

    Uses BCEWithLogitsLoss against targets. Logs the mean
    predicted probability on each, which is the standard early warning for a
    discriminator that has run away from the generator.
    """
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)

    loss_real = criterion(real_logits, torch.ones_like(real_logits))
    loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    loss = loss_real + loss_fake

    logs = {
        "disc_loss": loss.item(),
        "disc_loss_real": loss_real.item(),
        "disc_loss_fake": loss_fake.item(),
        # Sigmoid here only to log a readable probability; the loss itself
        # works on raw logits.
        "d_prob_real": torch.sigmoid(real_logits).mean().item(),
        "d_prob_fake": torch.sigmoid(fake_logits).mean().item(),
    }
    return loss, logs


def generator_loss(
    head: nn.Module,
    fake: torch.Tensor,
    model: str,
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    """Generator loss for whichever model is being trained.

    WGAN-GP maximises the critic's score on fakes. DCGAN uses the non-saturating
    form - the generator maximises log D(fake) rather than minimising
    log(1 - D(fake)) - because the latter has almost no gradient early in
    training, when the discriminator is winning easily.
    """
    scores = head(fake)
    if model == "wgan-gp":
        # Negated because optimisers minimise: the generator wants high scores.
        return -scores.mean()
    if model == "dcgan":
        if criterion is None:
            raise ValueError("A BCE criterion is required for the DCGAN generator loss.")
        # Target of 1 on the generator's own fakes: the non-saturating form.
        return criterion(scores, torch.ones_like(scores))
    raise ValueError(f"Unknown model {model!r}; expected 'wgan-gp' or 'dcgan'.")


# ---------------------------------------------------------------------------
# Logging and artefacts
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Per-epoch record of losses, metrics and cost.

    Written to JSON next to the checkpoints so the report's plots and resource
    table can be regenerated without retraining.
    """

    config: dict = field(default_factory=dict)
    epochs: list[dict] = field(default_factory=list)

    def record(self, epoch: int, **values: float) -> None:
        """Append one epoch's numbers."""
        ...

    def save(self, path: str | Path) -> Path:
        """Write the history to JSON."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        """Read a history back for plotting or comparison."""
        ...


def plot_history(history: TrainingHistory, path: str | Path, model: str) -> Path:
    """Plot losses and metrics against epoch.

    The specification requires loss and metric plots, so this writes the figure
    the README will embed: generator and critic/discriminator loss, the
    Wasserstein estimate for WGAN-GP, and any validation metrics recorded.
    """
    ...


def save_sample_grid(
    generator: modules.Generator,
    path: str | Path,
    n: int = 16,
    seed: int = 1234,
    device: torch.device | str = "cpu",
) -> Path:
    """Write an uncurated grid of generated slices from a fixed seed.

    The same seed is used at every epoch, so the grid shows one set of latent
    vectors evolving rather than a fresh draw each time. Fixed seeds and no
    cherry-picking are both required by the specification.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    # A generator seeded on the same device each epoch reproduces identical
    # latents without disturbing the global RNG that drives training.
    rng = torch.Generator(device=str(device)).manual_seed(seed)
    images = generator.generate(n, device=device, generator=rng).cpu()

    grid = make_grid(images, nrow=4, normalize=True, value_range=(-1, 1), padding=2)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"Uncurated samples, seed {seed}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_checkpoint(
    path: str | Path,
    generator: modules.Generator,
    head: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_h: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
) -> Path:
    """Save weights, optimiser state, epoch and config in one file.

    Optimiser state is included so an interrupted run can resume rather than
    restart -- Adam's moment estimates matter, especially for the critic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(config),
            "generator": generator.state_dict(),
            "head": head.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_h": opt_h.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    generator: modules.Generator,
    head: nn.Module | None = None,
    opt_g: torch.optim.Optimizer | None = None,
    opt_h: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Restore a checkpoint, optionally including optimiser state.

    ``predict.py`` calls this with the generator alone, since the critic and
    discriminator are training scaffolding and are not needed at inference.
    """
    # weights_only=True refuses to unpickle arbitrary objects; these checkpoints
    # hold only tensors and plain config values, so it costs nothing here.
    payload = torch.load(path, map_location=map_location, weights_only=True)
    generator.load_state_dict(payload["generator"])
    if head is not None:
        head.load_state_dict(payload["head"])
    if opt_g is not None:
        opt_g.load_state_dict(payload["opt_g"])
    if opt_h is not None:
        opt_h.load_state_dict(payload["opt_h"])
    return payload


# ---------------------------------------------------------------------------
# Validation and resource profiling
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    generator: modules.Generator,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Cheap per-epoch health metrics on the validate split.

    A GAN has no validation loss in the usual sense, so this reports signals
    that are cheap enough to run every epoch and that catch the common failure
    modes early: the head's mean score on held-out real data versus on fakes,
    and the pixel-wise spread across a batch of samples as a mode-collapse
    warning.

    Most evaluation will live in predict.py
    """
    was_training = generator.training
    generator.eval()
    head.eval()

    real_total, fake_total, batches = 0.0, 0.0, 0
    real_std, fake_std = 0.0, 0.0

    for real in loader:
        real = real.to(device, non_blocking=True)
        fake = generator(generator.sample_latent(real.size(0), device=device))

        real_total += head(real).mean().item()
        fake_total += head(fake).mean().item()

        # Spread across the batch at each pixel, averaged. A collapsed
        # generator produces near-identical images, driving this toward zero;
        # the real figure alongside it gives the scale to read it against.
        fake_std += fake.std(dim=0).mean().item()
        real_std += real.std(dim=0).mean().item()
        batches += 1

    if was_training:
        generator.train()
    head.train()

    if batches == 0:
        return {}
    return {
        "val_score_real": real_total / batches,
        "val_score_fake": fake_total / batches,
        "val_score_gap": (real_total - fake_total) / batches,
        "val_sample_std": fake_std / batches,
        "val_real_std": real_std / batches,
    }


def resource_profile(device: torch.device) -> dict[str, float]:
    """Peak VRAM and wall-clock figures for the report's resource table.

    Reads torch.cuda.max_memory_allocated rather than ``nvidia-smi`` so the
    number reflects this process alone and not the desktop session sharing the
    card.
    """
    if device.type != "cuda":
        return {}
    return {
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _build_loaders(config: TrainConfig) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Build the train loader and, when available, the validate loader.

    Both splits are cached as preprocessed tensors under ``out_dir`` so the
    gzip decode is paid once rather than every epoch.
    """
    cache_dir = Path(config.out_dir) / "cache"
    height, width = config.image_size

    def cache_for(split: str) -> Path:
        return cache_dir / f"{split}_{height}x{width}.pt"

    train_loader = dataset.build_dataloader(
        config.data_root,
        split="train",
        batch_size=config.batch_size,
        size=tuple(config.image_size),
        num_workers=config.num_workers,
        cache_path=cache_for("train"),
    )
    try:
        validate_loader = dataset.build_dataloader(
            config.data_root,
            split="validate",
            batch_size=config.batch_size,
            size=tuple(config.image_size),
            shuffle=False,
            num_workers=config.num_workers,
            cache_path=cache_for("validate"),
        )
    except FileNotFoundError:
        validate_loader = None
    return train_loader, validate_loader


def _setup(config: TrainConfig):
    """Device, data, models, optimisers and output paths for one run."""
    set_seed(config.seed)
    device = select_device()

    # Fail fast on a split that would invalidate every downstream metric.
    dataset.check_patient_leakage(config.data_root)

    train_loader, validate_loader = _build_loaders(config)

    generator, head = modules.build_models(
        model=config.model,
        latent_dim=config.latent_dim,
        image_size=tuple(config.image_size),
        base_channels=config.base_channels,
        upsample=config.upsample,
        critic_norm=config.critic_norm,
    )
    generator, head = generator.to(device), head.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=tuple(config.betas))
    opt_h = torch.optim.Adam(head.parameters(), lr=config.lr, betas=tuple(config.betas))

    run_dir = Path(config.out_dir) / config.model
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    history = TrainingHistory(config=asdict(config))
    history.config["device"] = str(device)
    history.config["generator_parameters"] = modules.count_parameters(generator)
    history.config["head_parameters"] = modules.count_parameters(head)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    return device, train_loader, validate_loader, generator, head, opt_g, opt_h, run_dir, history


def _end_of_epoch(
    epoch: int,
    config: TrainConfig,
    generator: modules.Generator,
    head: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_h: torch.optim.Optimizer,
    run_dir: Path,
    history: TrainingHistory,
    device: torch.device,
    validate_loader: torch.utils.data.DataLoader | None,
    epoch_logs: dict[str, float],
) -> None:
    """Validation, sample grid, checkpoint and history write for one epoch."""
    if validate_loader is not None:
        epoch_logs.update(validate(generator, head, validate_loader, config, device))
    epoch_logs.update(resource_profile(device))

    if config.sample_every and epoch % config.sample_every == 0:
        save_sample_grid(
            generator,
            run_dir / "samples" / f"epoch_{epoch:04d}.png",
            seed=config.sample_seed,
            device=device,
        )
    # Always write "last" so an interrupted run can resume from the most recent
    # epoch, not just the most recent multiple of checkpoint_every.
    save_checkpoint(run_dir / "checkpoints" / "last.pt", generator, head, opt_g, opt_h, epoch, config)
    if config.checkpoint_every and epoch % config.checkpoint_every == 0:
        save_checkpoint(
            run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
            generator, head, opt_g, opt_h, epoch, config,
        )


def _progress(iterable, description: str):
    """ Nice qui progress bar for the progress of the epochs
    Wrap a loader in tqdm when it is installed, otherwise pass it through.
    Method and rest of tqdm implementation written by Claude Opus 5.5
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, leave=False)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_wgan_gp(config: TrainConfig) -> TrainingHistory:
    """Train the WGAN-GP model.

    Each generator step is preceded by config.n_critic critic steps, each
    on a fresh batch. The critic maximises the gap between its scores on real
    and fake batches, regularised by the gradient penalty; the generator then
    tries to close that gap.

    There is no early stopping: the critic loss is not a quality metric and
    does not converge to anything meaningful to stop on.
    """
    device, train_loader, validate_loader, generator, head, opt_g, opt_h, run_dir, history = _setup(config)
    print(f"WGAN-GP on {device} | {len(train_loader)} batches/epoch | "
          f"~{len(train_loader) // config.n_critic} generator steps/epoch")

    gen_steps = head_steps = 0
    for epoch in range(1, config.epochs + 1):
        generator.train()
        head.train()
        started = time.perf_counter()
        critic_logs: list[dict[str, float]] = []
        gen_losses: list[float] = []

        for index, real in enumerate(_progress(train_loader, f"epoch {epoch}/{config.epochs}")):
            real = real.to(device, non_blocking=True)

            # --- critic step -------------------------------------------------
            # no_grad around generation: the generator is not being updated
            # here, so there is no reason to build a graph through it.
            with torch.no_grad():
                fake = generator(generator.sample_latent(real.size(0), device=device))

            loss_c, logs = critic_loss(head, real, fake, config.gp_lambda, device)
            opt_h.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_h.step()
            critic_logs.append(logs)
            head_steps += 1

            # --- generator step, every n_critic batches ----------------------
            if (index + 1) % config.n_critic == 0:
                fake = generator(generator.sample_latent(real.size(0), device=device))
                loss_g = generator_loss(head, fake, "wgan-gp")
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                gen_losses.append(loss_g.item())
                gen_steps += 1

        epoch_logs = {key: _mean([entry[key] for entry in critic_logs]) for key in critic_logs[0]}
        epoch_logs.update({
            "gen_loss": _mean(gen_losses),
            "epoch_seconds": time.perf_counter() - started,
            "generator_steps": gen_steps,
            "head_steps": head_steps,
        })
        _end_of_epoch(epoch, config, generator, head, opt_g, opt_h, run_dir,
                      history, device, validate_loader, epoch_logs)
        print(f"epoch {epoch:4d} | W {epoch_logs['wasserstein']:+8.3f} | "
              f"critic {epoch_logs['critic_loss']:+8.3f} | gen {epoch_logs['gen_loss']:+8.3f} | "
              f"{epoch_logs['epoch_seconds']:5.1f}s")

    # TODO plot history
    return history


def train_dcgan(config: TrainConfig) -> TrainingHistory:
    """Train the DCGAN baseline.

    The discriminator and generator alternate one step each on the same batch.
    This is the baseline the WGAN-GP results are measured against, so it gets
    the same data, generator, epoch budget and logging -- a weak baseline would
    make the comparison meaningless.
    """
    device, train_loader, validate_loader, generator, head, opt_g, opt_h, run_dir, history = _setup(config)
    criterion = nn.BCEWithLogitsLoss()
    print(f"DCGAN on {device} | {len(train_loader)} batches/epoch")

    gen_steps = head_steps = 0
    for epoch in range(1, config.epochs + 1):
        generator.train()
        head.train()
        started = time.perf_counter()
        disc_logs: list[dict[str, float]] = []
        gen_losses: list[float] = []

        for real in _progress(train_loader, f"epoch {epoch}/{config.epochs}"):
            real = real.to(device, non_blocking=True)
            fake = generator(generator.sample_latent(real.size(0), device=device))

            # --- discriminator step ------------------------------------------
            # detach() stops the discriminator's gradients flowing back into the
            # generator; the same fake tensor is reused below with its graph
            # intact for the generator's own step.
            loss_d, logs = discriminator_loss(head, real, fake.detach(), criterion)
            opt_h.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_h.step()
            disc_logs.append(logs)
            head_steps += 1

            # --- generator step ----------------------------------------------
            loss_g = generator_loss(head, fake, "dcgan", criterion)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            gen_losses.append(loss_g.item())
            gen_steps += 1

        epoch_logs = {key: _mean([entry[key] for entry in disc_logs]) for key in disc_logs[0]}
        epoch_logs.update({
            "gen_loss": _mean(gen_losses),
            "epoch_seconds": time.perf_counter() - started,
            "generator_steps": gen_steps,
            "head_steps": head_steps,
        })
        _end_of_epoch(epoch, config, generator, head, opt_g, opt_h, run_dir,
                      history, device, validate_loader, epoch_logs)
        print(f"epoch {epoch:4d} | disc {epoch_logs['disc_loss']:+8.3f} | "
              f"gen {epoch_logs['gen_loss']:+8.3f} | "
              f"D(real) {epoch_logs['d_prob_real']:.3f} D(fake) {epoch_logs['d_prob_fake']:.3f} | "
              f"{epoch_logs['epoch_seconds']:5.1f}s")

    #TODO plot history
    return history


def train(config: TrainConfig) -> TrainingHistory:
    """Dispatch to the right training loop and write out the artefacts."""
    if config.model == "wgan-gp":
        return train_wgan_gp(config)
    if config.model == "dcgan":
        return train_dcgan(config)
    raise ValueError(f"Unknown model {config.model!r}; expected 'wgan-gp' or 'dcgan'.")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line options into overrides for TrainConfig class"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["wgan-gp", "dcgan"], default="wgan-gp")
    parser.add_argument("--data-root", default=None, help="Folder holding keras_slices_* subfolders")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-critic", type=int, default=None)
    parser.add_argument("--gp-lambda", type=float, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--upsample", choices=["transpose", "nearest"], default=None)
    parser.add_argument("--critic-norm", choices=["layer", "instance", "none"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the config, train, and report outputs"""
    config = build_config(parse_args(argv))
    history = train(config)

    run_dir = Path(config.out_dir) / config.model
    print(f"\nFinished {config.model} after {config.epochs} epochs.")
    if history.epochs:
        final = history.epochs[-1]
        if "peak_vram_gib" in final:
            print(f"  peak VRAM        {final['peak_vram_gib']:.2f} GiB")
        print(f"  generator steps  {final.get('generator_steps', 0):,}")
        total_seconds = sum(e.get("epoch_seconds", 0.0) for e in history.epochs)
        print(f"  training time    {total_seconds / 60:.1f} min")
    print(f"  artefacts        {run_dir}")


if __name__ == "__main__":
    main()
