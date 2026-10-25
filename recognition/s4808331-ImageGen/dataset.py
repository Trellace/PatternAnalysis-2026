"""Loading, preprocessing and leakage-free splitting for the HipMRI 2D slice data.

The dataset ships as one gzipped Nifti file per 2D slice, already divided into
``keras_slices_train`` / ``keras_slices_validate`` / ``keras_slices_test``.
Filenames follow ``case_<patient>_week_<session>_slice_<index>.nii.gz``, so the
patient a slice belongs to can be recovered from its name. That is what makes the
leakage check below possible without any extra metadata.

Findings from the data audit that drive the choices here:

* Slices are 256x128 with isotropic 1x1 mm spacing, i.e. a true 2:1 landscape
  field of view. One scan session (``case_019_week_1``, 60 slices) is 256x144.
* Intensities are non-negative but the per-slice maximum varies widely
  (p5 277, median 391, p95 621, max 1082), so normalisation must be per slice.
* No patient appears in more than one split.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Split folder names as they appear in ``keras_slices_data``.
SPLIT_DIRS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test",
}

# Target size as (height, width).
#
# The native slice is 256x128, which displays as a 2:1 landscape
# image. Resizing to a square will distort or skew details, so not designed 
# for this project when outputs are meant to be used for training.
# 64x128 is an exact 2x downsample of the native grid, so it
# preserves geometry and costs half the pixels of a 128x128 square.
DEFAULT_IMAGE_SIZE = (64, 128)

# Native in-plane size, used to make the one odd-sized session consistent.
NATIVE_SIZE = (256, 128)

# configurable variable to ignore a certain percentile of pixels that are outlier brightness values
# set to 100.0 to ignore for now and allow ALL pixels to have their brightness considered
DEFAULT_CLIP_PERCENTILE = 100

_CASE_RE = re.compile(r"case_(?P<patient>\d+)_week_(?P<week>\d+)_slice_(?P<slice>\d+)\.nii\.gz$") # Created by Claude Opus 5.5


# ---------------------------------------------------------------------------
# Filename parsing and leakage checks
# ---------------------------------------------------------------------------


def parse_slice_name(path: str | Path) -> dict[str, str]:
    """Pull the patient, session and slice index out of a slice filename.

    Raises ``ValueError`` on anything that does not match the expected pattern,
    so an unexpected file cannot silently be treated as an unknown patient.
    """
    match = _CASE_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognised slice filename: {Path(path).name!r}")
    return match.groupdict()


def patient_id(path: str | Path) -> str:
    """Return the patient identifier for a slice path."""
    return parse_slice_name(path)["patient"]


def list_slice_files(root: str | Path, split: str) -> list[Path]:
    """List the MR slice files for one split, sorted for reproducibility.

    Only ``case_*`` files are returned.
    """
    if split not in SPLIT_DIRS:
        raise KeyError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_DIRS)}")
    split_dir = Path(root) / SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    files = sorted(split_dir.glob("case_*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No slice files found in {split_dir}")
    return files


def patients_by_split(root: str | Path) -> dict[str, set[str]]:
    """Map each split to the set of patient IDs it contains."""
    return {split: {patient_id(f) for f in list_slice_files(root, split)} for split in SPLIT_DIRS}


def check_patient_leakage(root: str | Path, raise_on_leak: bool = True) -> dict[tuple[str, str], set[str]]:
    """Flag if a patient appears in more than one split else flag.

    Slices from different patients are still anatomical near-duplicates of each other, so a patient
    spanning two splits would let the model be evaluated on anatomy it trained
    on and would make the memorisation audit meaningless. The supplied splits
    are already patient-disjoint; this function checks that rather than
    assuming it.

    Returns a mapping from split pair to the shared patient IDs (empty when
    clean). With ``raise_on_leak`` set, a non-empty overlap raises instead.
    """
    patients = patients_by_split(root)
    names = sorted(patients)
    overlaps: dict[tuple[str, str], set[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = patients[a] & patients[b]
            overlaps[(a, b)] = shared
            if shared and raise_on_leak:
                raise RuntimeError(
                    f"Patient leakage between {a!r} and {b!r}: {sorted(shared)}. "
                    "Splits must be patient-disjoint."
                )
    return overlaps


def sessions_by_patient(files: list[Path]) -> dict[str, set[str]]:
    """Group scan sessions (weeks) by patient.

    A patient can be scanned several times. Repeat sessions of one patient are
    highly similar, which matters when interpreting nearest-neighbour distances.
    """
    sessions: dict[str, set[str]] = defaultdict(set)
    for f in files:
        meta = parse_slice_name(f)
        sessions[meta["patient"]].add(meta["week"])
    return dict(sessions)


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------


def load_slice(path: str | Path) -> np.ndarray:
    """Read one Nifti slice as a 2D float32 array.

    Some HipMRI files carry a trailing singleton dimension (spec Appendix B), so
    a third axis is dropped when present.
    """
    image = nib.load(str(path)).get_fdata(caching="unchanged")
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {image.shape} for {path}")
    return image.astype(np.float32)


def normalise(image: np.ndarray, clip_percentile: float | None = DEFAULT_CLIP_PERCENTILE) -> np.ndarray:
    """Scale one slice to [-1, 1] to match the generator's tanh output.

    Normalisation is per slice because the audit showed the per-slice maximum
    varying by roughly 5x across the dataset; a single global scale would leave
    some slices washed out and others clipped. A blank slice maps to all -1.
    """
    image = image.astype(np.float32)
    low = float(image.min())
    high = float(np.percentile(image, clip_percentile)) if clip_percentile else float(image.max())
    if high <= low:  # constant slice, e.g. all zeros
        return np.full_like(image, -1.0)
    image = np.clip(image, low, high)
    return (image - low) / (high - low) * 2.0 - 1.0


def to_native_grid(image: np.ndarray, native_size: tuple[int, int] = NATIVE_SIZE) -> np.ndarray:
    """Centre crop or pad a slice onto the dominant native grid.

    Most tested data was the same size, but in case of outliers files are cropped to the common size of ``256x128``. Trimming the
    odd data to the common grid keeps the physical scale consistent, instead of applying any stretching and skewing the resulting generation.
    """
    out = image
    for axis, target in enumerate(native_size):
        current = out.shape[axis]
        if current > target:  # centre crop
            start = (current - target) // 2
            out = out.take(range(start, start + target), axis=axis)
        elif current < target:  # centre pad with the slice background
            before = (target - current) // 2
            after = target - current - before
            pad = [(0, 0), (0, 0)]
            pad[axis] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=float(image.min()))
    return out


def resize(image: torch.Tensor, size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Resize a (1, H, W) slice tensor to ``size`` with antialiasing.
    """
    resized = F.interpolate(
        image.unsqueeze(0), size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.squeeze(0)


def preprocess(
    path: str | Path,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    clip_percentile: float | None = DEFAULT_CLIP_PERCENTILE,
) -> torch.Tensor:
    """Full single-slice pipeline: load, standardise grid, normalise, resize.

    Returns a (1, H, W) float32 tensor in [-1, 1]. The array is transposed so
    the 256-voxel left-right axis becomes the image width, matching how the
    slices are viewed.
    """
    image = load_slice(path)
    image = to_native_grid(image)
    image = normalise(image, clip_percentile)
    tensor = torch.from_numpy(np.ascontiguousarray(image.T)).unsqueeze(0)
    return resize(tensor, size)


# ---------------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------------


class HipMRISliceDataset(Dataset):
    """2D HipMRI slices for one split, preprocessed and saved into memory.

    Avoids reprocessing the slices every epoch. Does eat a large chunk of memory.

    Class designed and implemented with help from Claude Opus 5.5.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        clip_percentile: float | None = DEFAULT_CLIP_PERCENTILE,
        cache_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.size = size
        self.files = list_slice_files(root, split)
        self.patients = [patient_id(f) for f in self.files]

        if cache_path is not None and Path(cache_path).exists():
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("size") != list(size) and tuple(payload.get("size", ())) != size:
                raise ValueError(
                    f"Cache at {cache_path} holds size {payload.get('size')}, expected {size}. "
                    "Delete it to rebuild."
                )
            self.images = payload["images"]
        else:
            if verbose:
                print(f"Preprocessing {len(self.files)} slices from {split!r} at {size}...")
            self.images = torch.stack(
                [preprocess(f, size=size, clip_percentile=clip_percentile) for f in self.files]
            )
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"images": self.images, "size": tuple(size)}, cache_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Unconditional generation: no labels, just the image.
        return self.images[index]

    def summary(self) -> str:
        sessions = sessions_by_patient(self.files)
        n_sessions = sum(len(v) for v in sessions.values())
        return (
            f"{self.split}: {len(self)} slices | {len(set(self.patients))} patients | "
            f"{n_sessions} sessions | shape {tuple(self.images.shape[1:])} | "
            f"range [{self.images.min():.2f}, {self.images.max():.2f}]"
        )


def build_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = 64,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 2,
    cache_path: str | Path | None = None,
    **dataset_kwargs,
) -> DataLoader:
    """Build a DataLoader over one split.

    ``drop_last`` is on for training because a short final batch destabilises
    the gradient penalty estimate.
    """
    dataset = HipMRISliceDataset(root, split=split, size=size, cache_path=cache_path, **dataset_kwargs)
    training = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def save_sample_grid(dataset: Dataset, path: str | Path, n: int = 16, seed: int = 0) -> Path:
    """Save a grid of real slices so preprocessing can be eyeballed. 
    
    Claude Opus 5.5 used to generate below plot configuration and creation code 
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n]
    batch = torch.stack([dataset[int(i)] for i in indices])
    grid = make_grid(batch, nrow=4, normalize=True, value_range=(-1, 1), padding=2)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"Preprocessed slices, {tuple(batch.shape[2:])}, seed {seed}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit and sanity-check the HipMRI slice pipeline.")
    parser.add_argument("--root", default="data", help="Folder containing keras_slices_* subfolders")
    parser.add_argument("--out", default="outputs/real_samples.png", help="Where to write the sample grid")
    args = parser.parse_args()

    print("Checking patient-level separation between splits...")
    for (a, b), shared in check_patient_leakage(args.root).items():
        print(f"  {a:9s} vs {b:9s}: {len(shared)} shared patients")

    for split in SPLIT_DIRS:
        dataset = HipMRISliceDataset(args.root, split=split, verbose=False)
        print(dataset.summary())
        if split == "train":
            print(f"Wrote {save_sample_grid(dataset, args.out)}")
