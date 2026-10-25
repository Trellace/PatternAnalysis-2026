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
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Above instructions generated by Claude Opus 5.5.

On Linux the PyPI `torch` wheel already includes CUDA, so only the NVIDIA driver is needed.
