# WGAN-GP Medical Slice Imagery Generation

## Feasibility Review

### 1. User Need, Scope & Acceptance Criteria

#### Intended User & Purpose
- **User:** a medical imaging research team that needs more MRI training data than patient privacy laws, such as HIPPA, and limited scans allow.
- **Purpose:** generate synthetic 2D pelvic MRI slice imagery that can augment downstream model training and be shared between centres without exposing real patients. This has value because synthetic slices are only useful if they are realistic (so downstream models learn true anatomy), diverse (model doesn't collapse to same output each time) and novel (no memorised, and therefore anonymised and unlinked, patient scans).

#### Scope & Acceptance Criteria
- **In scope:** unconditional 2D slice generation from HipMRI data; WGAN-GP vs a DCGAN baseline
- **AC1 – Fidelity:** mean SSIM > 0.6 between generated slices and their closest real test slice, and FID lower than the DCGAN baseline.
- **AC2 – Diversity:** mean SSIM between pairs of generated slices is no higher than between real slices.
- **AC3 – No memorisation:** Max SSIM ≤ 0.95 between a generated slice and the most similar-appearing training data entry
- **AC4 – Resources:** training fits in ≤ 16 GB peak VRAM within ≤ 6 GPU-hours on Rangpur; 1,000 slices generate in < 10 s. Other benchmark is to generate and fit within 30min on an Apple M4 16gb machine  ADVISED BY CLAUDE ON THIS BENCHMARK, goal to get running on my own machine set by myself

### 2. Model Choice & Course Concepts

#### Dataset Configuration
- MR images only (no masks) from the provided train/validate/test folders; the case IDs in the filenames are checked so each patient sits in a single split.
- The GAN trains on the train split only; the test split is the reference set for SSIM/FID and the memorisation control.
- Slices are resized to a fixed size (128×128, to confirm in the data audit) and scaled to [-1, 1] to match the generator's tanh output.

#### Model Architecture & Difficulty
- **Model:** WGAN-GP [Normal difficulty]. Convolutional generator (128-d latent → upsampling blocks → squash with tanh) and critic (strided convs, layer norm, no batch norm, unbounded linear output).
- **Training:** 5 critic steps per generator step, gradient penalty λ = 10 SUGGESTED BY CLAUDE
- **Baseline:** DCGAN with the same backbone, using batch norm and BCE loss, so the comparison isolates the effect of the Wasserstein loss.

#### Supporting Concepts & Literature
- GANs and DCGAN (Radford et al., 2016); Wasserstein distance and the Lipschitz critic (Arjovsky et al., 2017); gradient penalty (Gulrajani et al., 2017).
- Evaluation: SSIM (Wang et al., 2004), FID (Heusel et al., 2017), and nearest-neighbour memorisation audits with latent interpolation.

### 3. Preliminary Feasibility Evidence
<!-- Initial data audit, baseline smoke test, or measured resource estimate. A fully trained model is not required. -->

#### Data Audit
- Slices per split: [train / validate / test]; patients per split: [ ]; native image size(s): [ ]; intensity range: [ ].
- Patient overlap between splits: [none found / details].

#### Baseline Smoke Test
- DCGAN trained on MNIST for [ ] epochs, then on a HipMRI subset for [ ] epochs: [losses stable? outputs show pelvic outline?].

#### Resource Estimate
- Parameters: generator, critic; peak VRAM at batch: x GB; time per epoch: x s on GPU and CPU.

### 4. Risks, Budget & Fallback
#### Key Risks
- Training instability or mode collapse (most likely in the DCGAN baseline).
- Memorisation: the dataset comes from a small number of patients, and neighbouring slices from one patient look alike.
- How SSIM against sample slices should be computed is ambiguous in the spec
- Rangpur queue delays.

#### Compute Budget
- ~10 GPU-hours on Rangpur in total (baseline + WGAN-GP + evaluation); development and debugging done on my machine in subsets.

#### Next Planned Experiment
- Train the DCGAN baseline on the full train split, then WGAN-GP with the same backbone, logging losses, SSIM and FID every few epochs.

#### Fallback Plan
- If 128×128 is unstable or too slow, drop to 64×64. If the SSIM target is missed I'll report the gap with the failure analysis.

## Dependencies
Developed on Omarchy (Arch), with python 3.14.7 installed

All versions are pinned in [`requirements.txt`](requirements.txt).

| Package | Version | Used for |
|---|---|---|
| Python | 3.14.7 | Interpreter (pinned in `.python-version`) |
| torch | 2.14.1 | Models and training |
| torchvision | 0.29.1 | Image transforms and sample grids |
| numpy | 2.5.4 | Data loading and visualisation (not used in `modules.py`) |
| nibabel | 5.4.2 | Reading HipMRI Nifti slices |
| scikit-image | 0.26.0 | SSIM |
| scipy | 1.18.1 | Required for FID |
| torchmetrics | 1.9.0 | FID |
| torch-fidelity | 0.4.0 | Inception feature extractor for FID |
| matplotlib | 3.11.2 | Loss plots and figures |
| tqdm | 4.70.1 | Progress bars |

### Hardware
- Training and evaluation: NVIDIA RTX 3060 Ti, Linux (Arch). The card is sold as 8 GB, but PyTorch reports **7.63 GiB** usable and the desktop session holds ~0.9 GiB of that, so the training budget is roughly **6.5 GiB**.
- NVIDIA driver 610.57.04, CUDA 13.0 (`torch 2.14.1+cu130`), compute capability 8.6.
- Development and later feasibility test: Apple M4 (16 GB), PyTorch MPS backend.

### Setup

The interpreter version is pinned in `.python-version`, so [`uv`](https://docs.astral.sh/uv/)
creates the environment without needing a matching system Python:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

If you already have Python 3.14 installed and would rather not use `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Above instructions generated by Claude Opus 5.5.

On Linux the PyPI `torch` wheel already includes CUDA, so only the NVIDIA driver is needed.

## Usage

The dataset is never committed. Download `keras_slices_data` and keep it outside
the repository, then point the scripts at it with `--data-root` argument. The folder must
contain `keras_slices_train`, `keras_slices_validate` and `keras_slices_test`.

All commands below assume the virtual environment is active and are run from
`recognition/s4808331-ImageGen/`.

```bash
source .venv/bin/activate
export DATA=~/Projects/Comp3710/keras_slices_data
```

### 1. Check the data pipeline

Verifies that no patient appears in more than one split, reports the slice and
patient counts for each split, and writes a grid of preprocessed real slices so
the preprocessing can be inspected by eye.

```bash
python dataset.py --root $DATA --out outputs/real_samples.png
```

Expected output:

```
train:    11460 slices | 31 patients | 191 sessions | shape (1, 64, 128) | range [-1.00, 1.00]
validate:   660 slices |  4 patients |  11 sessions | shape (1, 64, 128) | range [-1.00, 0.94]
test:       540 slices |  3 patients |   9 sessions | shape (1, 64, 128) | range [-1.00, 0.95]
```

### 2. Check the models

Builds both model pairs, runs a batch of noise through them and asserts the
output shapes, the `tanh` range and the absence of batch normalisation in the
critic. Prints the parameter counts used in the resource table.

```bash
python modules.py
```

### 3. Train

The two models are trained separately. Run them one at a time: training both at
once makes the VRAM and timing measurements meaningless.

```bash
# WGAN-GP
python train.py --model wgan-gp --epochs 100 --data-root $DATA

# DCGAN baseline
python train.py --model dcgan --epochs 100 --data-root $DATA
```

A short run with less epochs is worth doing first to confirm the outputs all land where expected:

```bash
python train.py --model wgan-gp --epochs 2 --data-root $DATA
```

The first run of either model preprocesses the dataset once (about 30 seconds)
and caches it to `outputs/cache/`; later runs load the cache directly.

### 4. Resume or extend a run

Feature implemented by Claude Opus 5.5

`--epochs` is the target epoch to stop at, not a number of extra epochs, so the
command below continues a finished 100-epoch run through to epoch 400. Weights,
both optimiser states, the epoch counter and the logged history are all
restored, and the curves span the whole run rather than restarting.

```bash
python train.py --model wgan-gp --epochs 400 --resume --data-root $DATA
```

### Outputs

Each run writes to `outputs/<model>/`:

| Path | Contents |
|---|---|
| `history.json` | every logged metric, per epoch, plus the run config |
| `training_curves.png` | losses, Wasserstein estimate (or discriminator confidence), validation |
| `samples/epoch_NNNN.png` | uncurated 16-image grid from a fixed seed, one per epoch |
| `checkpoints/last.pt` | rewritten every epoch, used by `--resume` |
| `checkpoints/epoch_NNNN.pt` | kept every `--checkpoint-every` epochs |

The per-epoch sample grids all use the same latent vectors, so viewing them in
order shows the generator improving rather than a fresh random draw each time.
`outputs/` is ignored by git; figures for this report are copied into `assets/`.

### Useful options

| Option | Default | Notes |
|---|---|---|
| `--model` | `wgan-gp` | `wgan-gp` or `dcgan`; selects the loss, optimiser settings and head |
| `--epochs` | 100 | target epoch to stop at |
| `--batch-size` | 64 | 128 also fits comfortably |
| `--n-critic` | 5 (1 for DCGAN) | critic steps per generator step |
| `--gp-lambda` | 10 | gradient penalty weight |
| `--upsample` | `transpose` | `nearest` avoids checkerboard artifacts |
| `--critic-norm` | `layer` | `layer`, `instance` or `none`; batch norm is deliberately not offered |
| `--resume` | off | continue from `checkpoints/last.pt` |
| `--out-dir` | `outputs` | use a separate directory to keep experiments apart |

Run `python train.py --help` for the full list.

### Measured cost


On an RTX 3060 Ti at the default batch size of 64, WGAN-GP runs at roughly
9.9 s/epoch with a peak allocation of 1.12 GiB, so a 100-epoch run takes about
17 minutes. DCGAN is faster per epoch, having no gradient penalty and one
generator forward pass per batch instead of two.

Note that an epoch is one pass over the training data, which is 179 batches.
Under `--n-critic 5` that gives WGAN-GP about 35 generator steps per epoch
against the baseline's 179, so the two models see the same data but receive
different numbers of generator updates at equal epoch counts. Both
`generator_steps` and `head_steps` are logged every epoch for this reason.



Usage section drafted by Claude Opus 5.5.
