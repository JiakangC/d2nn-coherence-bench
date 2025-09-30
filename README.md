# D2NN Coherence Bench

**Diffractive Optical Neural Networks (D2NN) with partial spatial coherence, noise, and quantization ablations.**
This repo provides a differentiable Fourier-optics simulator (propagation + learnable phase mask), a torchvision MNIST pipeline, and ready-made sweeps to study how **coherence**, **photon shot noise**, and **bit-depth quantization** affect accuracy.

> Why this matters: real optical neural networks don’t run under ideal lasers/cameras. Coherence is finite, photons are limited, and ADCs have limited bits. We quantify the performance impact and provide practical “working regions” for experiments.

---

## Key ideas (short)

* **D2NN forward:** `Prop(z/2) → PhaseMask(learnable) → Prop(z/2) → |·|^2 → small MLP`.
* **Partial coherence:** emulate with **Gaussian-correlated random phase screens**; average intensity over `K` samples.
* **Noise & quantization:** shot noise (Poisson on intensity), readout noise (Gaussian), finite bit-depth (n-bit).

---

## Results (current snapshot)

* **Baseline (fully coherent, no noise/quant):** test acc **0.9800**.
* **Quantization (bits):** null **0.9779**, 8-bit **0.9511**, 6-bit **0.9554**, 4-bit **0.9515**.

  * Takeaway: ≥6-bit is a practical floor in this setup; 4-bit is marginal.
* **Shot noise (shot_scale):** 500 **0.9532**, 100 **0.9501**, 50 **0.9489**.

  * Takeaway: brighter (larger shot_scale) → better; next step is probe **1000/2000** to find the ≥0.97 threshold.
* **Partial coherence (so far):**

  * `K` sweep: 1 **0.9770**, 2 **0.9780**, 4 **0.9777**, 8 **0.9775** — little change in the tested range.
  * `corr_px` sweep: 2.0 **0.9765**, 3.0 **0.9757**, 4.0 **0.9756**, 6.0 **0.9781** — little change in this range; we will test more extreme values.

> Figures are produced by `tools/plot_*.py` and saved under `reports/figures/`. Add them below once generated:
>
> * `reports/figures/curve_accuracy.png`
> * `reports/figures/curve_loss.png`
> * `reports/figures/ablation_K.png`
> * `reports/figures/ablation_corr_px.png`
> * `reports/figures/ablation_shot.png`
> * `reports/figures/ablation_bits.png`
> * `runs/<experiment>/figures/phase_mask.png` (example phase mask)
> * `runs/<experiment>/figures/intensity_grid.png` (detector intensities)

---

## Repository layout

```
.
├─ src/
│  ├─ main.py                 # entry (wraps train.py CLI)
│  ├─ train.py                # torchvision MNIST training + --outdir support
│  ├─ models/d2nn_classifier.py
│  ├─ optics/
│  │  ├─ fourier.py           # angular-spectrum propagation + learnable phase mask
│  │  ├─ coherence.py         # partial coherence via K phase-screen samples
│  │  └─ noise.py             # shot/readout noise + quantization
│  └─ vis.py                  # save phase mask, intensity grids
├─ experiments/
│  └─ A_coherence_noise.yaml  # base config
├─ tools/
│  ├─ sweep.py                # runs K/corr_px/shot/bits sweeps
│  ├─ plot_curves.py          # accuracy/loss vs epoch from metrics.csv
│  └─ plot_ablation.py        # single-parameter ablation curves from CSV
├─ runs/                      # created by sweeps; per-run outputs (metrics/figures/checkpoints)
└─ reports/
   ├─ figures/                # plots land here
   └─ *.csv                   # ablation CSVs (created by sweep.py)
```

---

## Setup

```bash
# (Windows PowerShell examples; adapt paths if needed)
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
# CPU-only wheels (works everywhere)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# For NVIDIA CUDA (optional): replace the index-url with /cu124

pip install numpy matplotlib einops PyYAML tqdm
```

---

## Quick start

```bash
# Baseline training (writes into runs\baseline\...)
python -m src.main --config experiments\A_coherence_noise.yaml --outdir runs\baseline

# Plot training curves from metrics.csv
python tools\plot_curves.py
```

Per-run outputs include:

```
runs\baseline\
  metrics.csv
  checkpoints\*.pt
  figures\phase_mask.png
  figures\intensity_grid.png
```

---

## Sweeps (one command)

```bash
python tools\sweep.py
```

This runs four sweeps and writes:

```
reports\ablation_K.csv
reports\ablation_corr_px.csv
reports\ablation_shot.csv
reports\ablation_bits.csv
```

Make figures:

```bash
python tools\plot_ablation.py reports\ablation_K.csv reports\figures\ablation_K.png
python tools\plot_ablation.py reports\ablation_corr_px.csv reports\figures\ablation_corr_px.png
python tools\plot_ablation.py reports\ablation_shot.csv reports\figures\ablation_shot.png
python tools\plot_ablation.py reports\ablation_bits.csv reports\figures\ablation_bits.png
```

---

## Configuration (A_coherence_noise.yaml)

Key knobs:

```yaml
optics:
  dx: 8.0e-6      # pixel pitch (m)
  lam: 5.32e-7    # wavelength (m)
  z: 0.02         # propagation distance (m)
model:
  N: 32           # input size (images are resized to NxN)
  hidden: 128
train:
  batch_size: 128
  lr: 2.0e-3
  epochs: 8       # increase to 12–15 for more stable sweeps
coherence:
  K: 1            # >1 enables partial-coherence averaging
  corr_px: 3.0    # correlation length (pixels) of the phase screen
noise:
  shot_scale: 0.0 # brightness; set 50/100/500/1000/2000 for Poisson shot noise
  read_sigma: 0.0 # Gaussian readout noise
  quant_bits: null  # null or 8/6/4
```

---

## How to interpret the knobs

* **K** (samples): higher K = stronger averaging over random phase → more “incoherence”.
* **corr_px**: phase-screen correlation length (pixels). Smaller values ≈ faster phase changes (worse spatial coherence).
* **shot_scale**: photon count scale; larger = brighter (less Poisson noise).
* **quant_bits**: ADC bit depth; lower = more quantization error.

---

## What to report (recommended)

* **Baseline** accuracy (and learning curves).
* **Four ablation plots:** `K`, `corr_px`, `shot_scale`, `quant_bits` vs accuracy.
* **Phase mask** heatmap and **intensity** grids for a few samples.
* **“Working region” recommendation**, e.g.:

  * *For ≥97% accuracy, we require shot_scale ≥ **X**, quant_bits ≥ **Y** under K=**K***, corr_px ≥ **C***.*
  * *Coherence sensitivity is visible for **K ≥ …** and **corr_px ≤ …** in our model; otherwise the system is robust.*

---

## Troubleshooting

* **`FileNotFoundError: reports/figures/...`**
  We auto-create folders; if you use an older `vis.py`, create once: `mkdir reports\figures`.

* **Poisson error (`invalid Poisson rate`)**
  Ensure `apply_shot_noise` clamps negatives & NaNs (the repo version already does). You can also set `noise.shot_scale: 0.0` to disable.

* **Torch/Torchvision install issues**
  Use the official index URLs above (CPU or CUDA). On Windows, a fresh venv avoids Conda wheel conflicts.

---

## Citation & background

* **D2NN:** Lin et al., *Science* (2018).
* **Partial coherence in diffractive ONNs:** recent Optics/Photonics literature; we emulate with phase-screen averaging.
* **Training optics with noise/quantization:** we model measurement effects explicitly to approximate real hardware constraints.


---

## License

MIT 

## Acknowledgments

This work uses PyTorch and Torchvision for learning/inference and implements a simple Fourier-optics simulator for diffractive propagation and learnable phase masks.


