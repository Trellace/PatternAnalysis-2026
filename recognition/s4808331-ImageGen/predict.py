"""Evaluation and example usage for the trained WGAN-GP and DCGAN generators.

Loads saved generator weights, produces uncurated samples, and runs every
measurement the report needs. The critic and discriminator are not loaded:
they are training scaffolding and have no role at inference.

The four acceptance criteria from the feasibility review map onto the
functions below:

* AC1 fidelity     - mean SSIM to the closest real test slice, and FID below
                      the DCGAN baseline.
* AC2 diversity    - mean pairwise SSIM among generated slices no higher than
                      among real slices.
* AC3 memorisation - max SSIM to the most similar training slice at or below
                      0.95, read against the same statistic for held-out real
                      slices rather than against the threshold alone.
* AC4 resources    - parameter counts, peak VRAM and generation throughput.

Evaluation uses the test split, which no model has seen. The memorisation
audit is the exception: it compares against the training split by definition,
since the question is whether the generator reproduces what it was trained on.

Scaffolded and debugged by Claude Opus 5.5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                           # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as _ssim
from torchvision.utils import make_grid

import dataset
import modules
import train

# SSIM is computed on images in [-1, 1], so the dynamic range is 2.
DATA_RANGE = 2.0

# AC1 and AC3 thresholds from the accepted feasibility review, set by spec and project feasibility review
AC1_MIN_SSIM = 0.6
AC3_MAX_SSIM = 0.95


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Settings for one evaluation run."""

    # What to evaluate
    checkpoint: str = "outputs/wgan-gp/checkpoints/last.pt"
    model: str = "wgan-gp"

    data_root: str = "data"
    image_size: tuple[int, int] = modules.IMAGE_SIZE

    # Sample counts. n_samples drives SSIM, diversity and the audit; FID is
    # unstable below a couple of thousand images so it draws its own, larger
    # set.
    n_samples: int = 1000
    n_fid_samples: int = 2048
    batch_size: int = 128

    # Pairwise SSIM is quadratic, so diversity uses a subsample of each set.
    n_diversity: int = 400

    # Nearest-neighbour search. Comparing every generated slice against all
    # 11,460 training slices with SSIM directly is far too slow, so an L2
    # prefilter shortlists candidates and SSIM is computed only on those.
    prefilter_k: int = 50

    # Figures
    n_grid: int = 16                            # images per uncurated grid
    n_grid_seeds: int = 3                       # independent grids
    n_interpolation_steps: int = 9
    n_interpolation_pairs: int = 4
    n_autopsy: int = 5                          # flawed samples to dissect

    seed: int = 1234
    out_dir: str = "outputs/evaluation"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_generator(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[modules.Generator, dict]:
    """Rebuild a generator from a checkpoint and load its weights.

    The architecture is reconstructed from the config stored alongside the
    weights rather than from the current defaults, so an old checkpoint still
    loads after the defaults in ``modules.py`` change.

    Returns the generator in eval mode and the stored config.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    stored = payload.get("config", {})

    generator = modules.Generator(
        latent_dim=stored.get("latent_dim", modules.LATENT_DIM),
        image_size=tuple(stored.get("image_size", modules.IMAGE_SIZE)),
        base_channels=stored.get("base_channels", modules.BASE_CHANNELS),
        upsample=stored.get("upsample", "transpose"),
    )
    generator.load_state_dict(payload["generator"])
    generator.to(device).eval()
    stored["epoch"] = payload.get("epoch")
    return generator, stored


def load_real_images(
    config: EvalConfig,
    split: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load one preprocessed split as a single (N, 1, H, W) tensor.

    Reuses ``dataset.HipMRISliceDataset`` so evaluation applies exactly the
    same preprocessing the models were trained on; any mismatch here would
    make every metric below meaningless.
    """
    cache = Path("outputs") / "cache" / f"{split}_{config.image_size[0]}x{config.image_size[1]}.pt"
    images = dataset.HipMRISliceDataset(
        config.data_root,
        split=split,
        size=tuple(config.image_size),
        cache_path=cache if cache.parent.exists() else None,
        verbose=False,
    ).images
    return images.to(device)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    generator: modules.Generator,
    n: int,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    batch_size: int = 128,
) -> torch.Tensor:
    """Generate ``n`` images in batches, optionally from a fixed seed.

    Batched so large sample counts do not have to fit in VRAM at once.
    """
    generator.eval()
    rng = None
    if seed is not None:
        # A device-local generator keeps the draw reproducible without
        # disturbing the global RNG.
        rng = torch.Generator(device=str(device)).manual_seed(seed)

    chunks = []
    produced = 0
    while produced < n:
        size = min(batch_size, n - produced)
        z = torch.randn(size, generator.latent_dim, device=device, generator=rng)
        chunks.append(generator(z).cpu())
        produced += size
    return torch.cat(chunks)[:n]


def _save_grid(images: torch.Tensor, path: str | Path, title: str, nrow: int = 4) -> Path:
    """Render a batch of [-1, 1] images as one greyscale grid."""
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 2.5 * (len(images) / nrow)))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_sample_grids(
    generator: modules.Generator,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> list[Path]:
    """Write sample grids from several independent seeds.

    Uncurated means every image drawn is shown. The specification requires
    grids across diverse random seeds precisely so that cherry-picking is
    visible by its absence.
    """
    paths = []
    out = Path(config.out_dir) / config.model
    for index in range(config.n_grid_seeds):
        seed = config.seed + index
        images = generate(generator, config.n_grid, device=device, seed=seed)
        paths.append(_save_grid(
            images,
            out / f"samples_seed_{seed}.png",
            f"{config.model}: uncurated samples, seed {seed} (no selection applied)",
        ))
    return paths


def measure_generation_throughput(
    generator: modules.Generator,
    device: torch.device | str = "cpu",
    n: int = 1000,
    batch_size: int = 128,
) -> dict[str, float]:
    """Time the generation of n images for AC4.

    Includes a warm-up pass before timing, since the first CUDA call absorbs
    context setup and kernel autotuning that would otherwise be charged to the
    measurement.
    """
    generate(generator, batch_size, device=device)          # warm up
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    started = time.perf_counter()
    generate(generator, n, device=device, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "generation_count": n,
        "generation_seconds": elapsed,
        "images_per_second": n / elapsed if elapsed else float("nan"),
    }


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _ssim_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM for matched pairs: a[i] against b[i]. Returns one score per pair."""
    return _ssim(a, b, data_range=DATA_RANGE, reduction="none").flatten()


@torch.no_grad()
def ssim_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device | str = "cpu",
    chunk: int = 64,
) -> torch.Tensor:
    """SSIM between every image in a and every image in b.

    Returns an (len(a), len(b)) matrix. Computed on the GPU in chunks: the
    full matrix for the memorisation audit is 1,000 x 11,460 comparisons, which
    is impractical one pair at a time.

    Claude Opus 5.5 used extensively to explain the theory and debug torch syntax
    """
    a, b = a.to(device), b.to(device)
    out = torch.empty(len(a), len(b))
    rows = max(1, chunk // 4)                   # keep the expanded pair tensor small
    for i in range(0, len(a), rows):
        ai = a[i:i + rows]
        for j in range(0, len(b), chunk):
            bj = b[j:j + chunk]
            na, nb = len(ai), len(bj)
            # Every ai paired with every bj, flattened into one batch.
            x = ai.unsqueeze(1).expand(na, nb, *ai.shape[1:]).reshape(na * nb, *ai.shape[1:])
            y = bj.unsqueeze(0).expand(na, nb, *bj.shape[1:]).reshape(na * nb, *bj.shape[1:])
            out[i:i + na, j:j + nb] = _ssim_pairs(x, y).view(na, nb).cpu()
    return out


@torch.no_grad()
def nearest_by_l2(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    device: torch.device | str = "cpu",
    chunk: int = 128,
) -> torch.Tensor:
    """Shortlist the ``k`` most similar reference images per query by L2.

    A cheap prefilter for the SSIM searches. L2 and SSIM disagree in general,
    so ``k`` is kept generous; the shortlist only has to contain the true SSIM
    nearest neighbour, not rank it correctly.

    Claude Opus 5.5 was used extensively to explain the theory and write or debug torch syntax

    """
    flat_ref = reference.to(device).flatten(1)
    k = min(k, len(reference))
    indices = torch.empty(len(query), k, dtype=torch.long)
    for i in range(0, len(query), chunk):
        flat_q = query[i:i + chunk].to(device).flatten(1)
        distances = torch.cdist(flat_q, flat_ref)
        indices[i:i + len(flat_q)] = distances.topk(k, largest=False).indices.cpu()
    return indices


@torch.no_grad()
def nearest_neighbour_ssim(
    query: torch.Tensor,
    reference: torch.Tensor,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Best SSIM and matching reference index for each query image.

    Returns (scores, indices), both length len(query). Used for fidelity
    against the test split and for the memorisation audit against the train
    split.

    Claude Opus 5.5 used extensively to write and debug torch syntax and the rest of teh function

    """
    shortlist = nearest_by_l2(query, reference, config.prefilter_k, device=device)
    k = shortlist.shape[1]

    scores = np.empty(len(query), dtype=np.float32)
    best = np.empty(len(query), dtype=np.int64)

    rows = max(1, 512 // k)                     # cap the expanded pair batch
    for i in range(0, len(query), rows):
        q = query[i:i + rows].to(device)
        idx = shortlist[i:i + rows]
        candidates = reference[idx.flatten()].to(device)
        repeated = q.unsqueeze(1).expand(len(q), k, *q.shape[1:]).reshape(len(q) * k, *q.shape[1:])
        pair_scores = _ssim_pairs(repeated, candidates).view(len(q), k)

        top = pair_scores.max(dim=1)
        scores[i:i + len(q)] = top.values.cpu().numpy()
        # Map the position within the shortlist back to a reference index.
        best[i:i + len(q)] = idx.gather(1, top.indices.cpu().unsqueeze(1)).squeeze(1).numpy()
    return scores, best


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summary statistics reported for every SSIM distribution."""
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p5": float(np.percentile(values, 5)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def fidelity_ssim(
    fake: torch.Tensor,
    real_test: torch.Tensor,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """AC1 fidelity: SSIM between each generated slice and its closest real one.

    The acceptance criterion is a mean above 0.6. The distribution matters as
    well as the mean, so the percentiles are reported too - a good average
    hiding a tail of failures is a different result from uniform mediocrity.
    """
    scores, _ = nearest_neighbour_ssim(fake, real_test, config, device=device)
    result = _distribution(scores, "fidelity_ssim")
    result["ac1_ssim_pass"] = bool(result["fidelity_ssim_mean"] > AC1_MIN_SSIM)
    return result


@torch.no_grad()
def compute_fid(
    fake: torch.Tensor,
    real: torch.Tensor,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> float:
    """Frechet Inception Distance between generated and real images.

    Inception expects three channels at 299x299, so the single-channel slices
    are repeated across channels and resized. FID is biased at small sample
    sizes, which matters here: the test split holds only 540 slices, so the
    caller reports the reference set size alongside the score.

    Claude Opus 5.5 used extensively to explain the theory and context and write and debug torch syntax

    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def feed(images: torch.Tensor, is_real: bool) -> None:
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            # [-1, 1] -> [0, 1] for normalize=True, then grey -> RGB.
            batch = ((batch + 1.0) / 2.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
            metric.update(batch, real=is_real)

    feed(real, True)
    feed(fake, False)
    score = float(metric.compute())
    del metric
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


def _mean_pairwise_ssim(images: torch.Tensor, device: torch.device | str) -> float:
    """Mean SSIM over all distinct pairs within one set."""
    matrix = ssim_matrix(images, images, device=device)
    # Exclude the diagonal: an image compared with itself always scores 1.
    mask = ~torch.eye(len(images), dtype=torch.bool)
    return float(matrix[mask].mean())


def diversity_ssim(
    fake: torch.Tensor,
    real: torch.Tensor,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """AC2 diversity: mean pairwise SSIM within each set.

    Generated slices should be no more similar to each other than real slices
    are to each other. The real figure is the control: neighbouring slices from
    one patient are genuinely alike, so a raw number without that baseline
    cannot distinguish mode collapse from the anatomy simply being repetitive.
    """
    rng = np.random.default_rng(config.seed)

    def subsample(images: torch.Tensor) -> torch.Tensor:
        n = min(config.n_diversity, len(images))
        picks = rng.choice(len(images), size=n, replace=False)
        return images[torch.from_numpy(picks)]

    fake_score = _mean_pairwise_ssim(subsample(fake), device)
    real_score = _mean_pairwise_ssim(subsample(real), device)
    return {
        "diversity_ssim_generated": fake_score,
        "diversity_ssim_real": real_score,
        "diversity_margin": real_score - fake_score,
        "ac2_diversity_pass": bool(fake_score <= real_score),
    }


def memorisation_audit(
    fake: torch.Tensor,
    real_train: torch.Tensor,
    real_test: torch.Tensor,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """AC3 memorisation: how close generated slices sit to training slices.

    Reports the maximum and the distribution of each generated slice's best
    SSIM against the training split, and the same statistic for held-out real
    test slices.

    The test figure is the control that makes the result interpretable. Real
    slices from unseen patients still score highly against the training set
    because pelvic anatomy repeats; a generated score at or below that level is
    evidence of novelty, whereas the 0.95 threshold alone says little.
    """
    fake_scores, fake_indices = nearest_neighbour_ssim(fake, real_train, config, device=device)
    test_scores, _ = nearest_neighbour_ssim(real_test, real_train, config, device=device)

    result = _distribution(fake_scores, "memorisation_generated")
    result.update(_distribution(test_scores, "memorisation_real_test"))
    result["memorisation_over_threshold"] = int((fake_scores > AC3_MAX_SSIM).sum())
    result["memorisation_margin"] = float(test_scores.max() - fake_scores.max())
    result["ac3_memorisation_pass"] = bool(fake_scores.max() <= AC3_MAX_SSIM)
    # Kept out of the JSON by the caller; used to draw the nearest-neighbour figure.
    result["_scores"] = fake_scores
    result["_indices"] = fake_indices
    return result


# ---------------------------------------------------------------------------
# Qualitative evidence
# ---------------------------------------------------------------------------


@torch.no_grad()
def latent_interpolation(
    generator: modules.Generator,
    config: EvalConfig,
    device: torch.device | str = "cpu",
) -> Path:
    """Walk between pairs of latent vectors and render each step.

    A generator that has learned a continuous model of anatomy morphs smoothly
    between endpoints; one that has memorised a set of training images jumps
    between them with implausible frames in between. This is required evidence
    for the research question.

    Claude Opus 5.5 used extensively to explain the theory and torch syntax
    """
    rng = torch.Generator(device=str(device)).manual_seed(config.seed)
    steps = config.n_interpolation_steps
    rows = []
    for _ in range(config.n_interpolation_pairs):
        start = torch.randn(1, generator.latent_dim, device=device, generator=rng)
        end = torch.randn(1, generator.latent_dim, device=device, generator=rng)
        # Straight-line interpolation in latent space; the smoothness of the
        # resulting images is the thing being tested.
        weights = torch.linspace(0.0, 1.0, steps, device=device).view(steps, 1)
        latents = (1.0 - weights) * start + weights * end
        rows.append(generator(latents).cpu())

    path = Path(config.out_dir) / config.model / "latent_interpolation.png"
    return _save_grid(
        torch.cat(rows), path,
        f"{config.model}: latent interpolation, {steps} steps per row",
        nrow=steps,
    )


def nearest_neighbour_figure(
    fake: torch.Tensor,
    real_train: torch.Tensor,
    scores: np.ndarray,
    indices: np.ndarray,
    config: EvalConfig,
    path: str | Path,
) -> Path:
    """Show generated slices beside their closest training slice.

    The visual part to the memorisation audit: the pairs that scored
    highest are the ones a reviewer will want to see, since those are the
    strongest candidates for having been copied.
    """
    top = np.argsort(scores)[::-1][:config.n_autopsy]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(top), 2, figsize=(8, 2.0 * len(top)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(top):
        for column, (image, label) in enumerate((
            (fake[index], "generated"),
            (real_train[indices[index]], "nearest training slice"),
        )):
            axes[row, column].imshow(image.squeeze(0).numpy(), cmap="gray", vmin=-1, vmax=1)
            axes[row, column].axis("off")
            axes[row, column].set_title(f"{label} (SSIM {scores[index]:.3f})", fontsize=8)
    fig.suptitle(f"{config.model}: closest training matches (worst case for memorisation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def failure_autopsy(
    fake: torch.Tensor,
    scores: np.ndarray,
    config: EvalConfig,
    path: str | Path,
) -> tuple[Path, list[int]]:
    """Select and render the worst generations for the written autopsy.

    Picks the lowest-fidelity samples rather than hand-chosen ones so the
    selection is reproducible.
    """
    worst = np.argsort(scores)[:config.n_autopsy]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(worst), figsize=(3.2 * len(worst), 3.4))
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, worst):
        axis.imshow(fake[index].squeeze(0).numpy(), cmap="gray", vmin=-1, vmax=1)
        axis.axis("off")
        axis.set_title(f"#{index}  SSIM {scores[index]:.3f}", fontsize=8)
    fig.suptitle(f"{config.model}: lowest-fidelity samples, selected by score not by eye")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, [int(i) for i in worst]


def embedding_plot(
    fake: torch.Tensor,
    real: torch.Tensor,
    path: str | Path,
    method: str = "pca",
) -> Path:
    """Optional 2D embedding of real against generated images.

    Was curious to see how the generated and real datasets a positioned int he same space. 
    Helps visualise that outputs continually changed into neighbours.

    Claude Opus 5.5 used for writing this function
    """
    combined = torch.cat([real, fake]).flatten(1)
    labels = np.array([0] * len(real) + [1] * len(fake))

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            points = TSNE(n_components=2, init="pca", random_state=0).fit_transform(combined.numpy())
        except ImportError:
            method = "pca"
    if method != "tsne":
        centred = combined - combined.mean(dim=0, keepdim=True)
        # Two leading principal components via SVD; no extra dependency.
        _, _, components = torch.pca_lowrank(centred, q=2)
        points = (centred @ components[:, :2]).numpy()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    for value, name, colour in ((0, "real", "tab:blue"), (1, "generated", "tab:orange")):
        mask = labels == value
        plt.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.5, label=name, c=colour)
    plt.legend()
    plt.title(f"Real vs generated ({method.upper()})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Everything measured for one model, written to JSON."""

    model: str = ""
    checkpoint: str = ""
    metrics: dict = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        """Write the report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def summary(self) -> str:
        """One-screen summary with each acceptance criterion marked pass or fail."""
        m = self.metrics
        lines = [f"{self.model}  ({self.checkpoint})", "-" * 60]

        def mark(key: str) -> str:
            return "PASS" if m.get(key) else "FAIL"

        lines += [
            f"AC1 fidelity    mean SSIM {m.get('fidelity_ssim_mean', float('nan')):.4f} "
            f"(> {AC1_MIN_SSIM})  [{mark('ac1_ssim_pass')}]",
            f"                FID vs test  {m.get('fid_vs_test', float('nan')):.2f} "
            f"(n={m.get('fid_test_reference_n', 0)})",
            f"                FID vs train {m.get('fid_vs_train', float('nan')):.2f} "
            f"(n={m.get('fid_train_reference_n', 0)})",
            f"AC2 diversity   generated {m.get('diversity_ssim_generated', float('nan')):.4f} "
            f"vs real {m.get('diversity_ssim_real', float('nan')):.4f}  [{mark('ac2_diversity_pass')}]",
            f"AC3 memorise    max SSIM {m.get('memorisation_generated_max', float('nan')):.4f} "
            f"(<= {AC3_MAX_SSIM}); real test control "
            f"{m.get('memorisation_real_test_max', float('nan')):.4f}  [{mark('ac3_memorisation_pass')}]",
            f"AC4 resources   {m.get('generator_parameters', 0):,} params | "
            f"peak {m.get('peak_vram_gib', float('nan')):.2f} GiB | "
            f"{m.get('images_per_second', float('nan')):.0f} img/s",
        ]
        return "\n".join(lines)


def evaluate(config: EvalConfig) -> EvalReport:
    """Run the full evaluation for one trained model.
    
    Claude Opus 5.5 used for debugging and checking
    """
    device = train.select_device()
    train.set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator, stored = load_generator(config.checkpoint, device=device)
    print(f"Loaded {config.model} from epoch {stored.get('epoch')} on {device}")

    #load provided files
    real_test = load_real_images(config, "test", device="cpu")
    real_train = load_real_images(config, "train", device="cpu")

    fake = generate(generator, config.n_samples, device=device,
                    seed=config.seed, batch_size=config.batch_size)

    report = EvalReport(model=config.model, checkpoint=str(config.checkpoint))
    metrics: dict = {
        "epoch": stored.get("epoch"),
        "generator_parameters": modules.count_parameters(generator),
        "n_samples": config.n_samples,
    }

    print("Uncurated sample grids...")
    report.figures += [str(p) for p in save_sample_grids(generator, config, device=device)]

    print("AC1 fidelity (SSIM to nearest real test slice)...")
    metrics.update(fidelity_ssim(fake, real_test, config, device=device))

    print("AC1 fidelity (FID)...")
    fid_fake = generate(generator, config.n_fid_samples, device=device,
                        seed=config.seed + 99, batch_size=config.batch_size)
    metrics["fid_vs_test"] = compute_fid(fid_fake, real_test, device=device)
    metrics["fid_test_reference_n"] = len(real_test)
    # The test split is small enough that FID is noticeably biased, so a
    # second score against the much larger train split is reported alongside.
    metrics["fid_vs_train"] = compute_fid(fid_fake, real_train, device=device)
    metrics["fid_train_reference_n"] = len(real_train)
    del fid_fake

    print("AC2 diversity (pairwise SSIM)...")
    metrics.update(diversity_ssim(fake, real_test, config, device=device))

    print("AC3 memorisation audit...")
    audit = memorisation_audit(fake, real_train, real_test, config, device=device)
    nn_scores, nn_indices = audit.pop("_scores"), audit.pop("_indices")
    metrics.update(audit)

    print("AC4 generation throughput...")
    metrics.update(measure_generation_throughput(
        generator, device=device, n=config.n_samples, batch_size=config.batch_size))
    if device.type == "cuda":
        metrics["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3

    print("Figures...")
    out = Path(config.out_dir) / config.model
    report.figures.append(str(latent_interpolation(generator, config, device=device)))
    report.figures.append(str(nearest_neighbour_figure(
        fake, real_train, nn_scores, nn_indices, config, out / "nearest_neighbours.png")))

    fidelity_scores, _ = nearest_neighbour_ssim(fake, real_test, config, device=device)
    autopsy_path, worst = failure_autopsy(fake, fidelity_scores, config, out / "failure_autopsy.png")
    report.figures.append(str(autopsy_path))
    metrics["autopsy_indices"] = worst
    report.figures.append(str(embedding_plot(
        fake[:config.n_diversity], real_test[:config.n_diversity], out / "embedding.png")))

    report.metrics = metrics
    report.save(out / "report.json")
    return report


def compare(reports: list[EvalReport], path: str | Path) -> Path:
    """Generate a side-by-side WGAN-GP against DCGAN table.
    """
    rows = [
        ("Epoch", "epoch", "{:.0f}"),
        ("Generator parameters", "generator_parameters", "{:,.0f}"),
        ("AC1 mean SSIM to nearest real", "fidelity_ssim_mean", "{:.4f}"),
        ("AC1 FID vs test", "fid_vs_test", "{:.2f}"),
        ("AC1 FID vs train", "fid_vs_train", "{:.2f}"),
        ("AC2 pairwise SSIM (generated)", "diversity_ssim_generated", "{:.4f}"),
        ("AC2 pairwise SSIM (real)", "diversity_ssim_real", "{:.4f}"),
        ("AC3 max SSIM to training slice", "memorisation_generated_max", "{:.4f}"),
        ("AC3 same statistic, real test", "memorisation_real_test_max", "{:.4f}"),
        ("AC4 peak VRAM (GiB)", "peak_vram_gib", "{:.2f}"),
        ("AC4 images per second", "images_per_second", "{:.0f}"),
    ]

    header = "| Metric | " + " | ".join(r.model for r in reports) + " |"
    divider = "|---|" + "---|" * len(reports)
    lines = [header, divider]
    for label, key, fmt in rows:
        cells = []
        for report in reports:
            value = report.metrics.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line options into overrides for EvalConfig.
    Arg options suggested by Claude
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["wgan-gp", "dcgan", "both"], default="wgan-gp")
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to outputs/<model>/checkpoints/last.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-fid-samples", type=int, default=None)
    parser.add_argument("--n-diversity", type=int, default=None)
    parser.add_argument("--prefilter-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Turn parsed arguments into an EvalConfig."""
    config = EvalConfig()
    overrides = {
        key: value for key, value in vars(args).items()
        if value is not None and key != "model" and hasattr(config, key)
    }
    config = replace_config(config, **overrides)
    if args.model != "both":
        config = replace_config(config, model=args.model)
    if args.checkpoint is None:
        config = replace_config(
            config, checkpoint=str(Path("outputs") / config.model / "checkpoints" / "last.pt"))
    return config


def replace_config(config: EvalConfig, **changes) -> EvalConfig:
    """ Implemeneted by Claude as a helper
    
    dataclasses.replace, named locally to keep build_config readable."""
    from dataclasses import replace

    return replace(config, **changes)


def main(argv: list[str] | None = None) -> None:
    """Entry point to evaluate one or both models and write the artefacts."""
    args = parse_args(argv)
    base = build_config(args)

    models = ["wgan-gp", "dcgan"] if args.model == "both" else [base.model]
    reports = []
    for name in models:
        config = replace_config(base, model=name)
        if args.checkpoint is None or len(models) > 1:
            config = replace_config(
                config, checkpoint=str(Path("outputs") / name / "checkpoints" / "last.pt"))
        if not Path(config.checkpoint).exists():
            print(f"Skipping {name}: no checkpoint at {config.checkpoint}")
            continue
        report = evaluate(config)
        print()
        print(report.summary())
        print()
        reports.append(report)

    if len(reports) > 1:
        path = compare(reports, Path(base.out_dir) / "comparison.md")
        print(f"Comparison table: {path}")


if __name__ == "__main__":
    main()
