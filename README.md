# CT Motion Artifact Reduction

A deep learning pipeline that corrects motion artifacts in CT sinograms using a
residual attention U-Net, trained entirely on synthetically generated paired
data (clean / motion-corrupted sinograms).

**Result:** correcting motion artifacts improved sinogram-domain PSNR from
**19.53 dB to 32.16 dB** and SSIM from **0.6509 to 0.9702** on a held-out test set.

---

## 1. Project Overview

Patient motion during a CT scan corrupts the raw projection data (the
**sinogram**) before an image is ever reconstructed, producing streaking and
blurring artifacts in the final scan. This project simulates that problem
end-to-end — generating realistic synthetic motion-corrupted sinograms from a
standard CT phantom, then training a neural network to reverse the corruption
directly in the sinogram domain, before reconstruction.

The project intentionally works in the **sinogram domain** rather than the
reconstructed-image domain, since that's where motion artifacts physically
originate — and where correcting them is provably harder, because errors here
get amplified by the reconstruction step (filtered back-projection) that
follows.

---

## 2. Pipeline

```
Shepp-Logan phantom (randomly rotated/scaled/translated)
        │
        ▼
  Radon transform  →  clean sinogram (sino_true)
        │
        ▼
Simulated motion (per-view shifts + jumps) + Poisson/Gaussian noise
        │
        ▼
  Motion-corrupted sinogram (sino_art)
        │
        ▼
  Paired dataset saved to HDF5  (sino_true / sino_art)
        │
        ▼
  Residual Attention U-Net  (trained: sino_art → predicted clean sinogram)
        │
        ▼
  Evaluation: PSNR / SSIM (sinogram domain) + FBP reconstruction comparison
```

Each of the three stages above is its own script — see [Project Structure](#7-project-structure).

---

## 3. Synthetic Data Generation

`generate_synthetic_sinograms.py`

Real CT data with a perfectly clean "ground truth" paired to a
motion-corrupted version of the *same* scan doesn't exist — a patient can't
hold still and move at the same time. This project sidesteps that by
generating both from the same source:

1. **Phantom**: the standard Shepp-Logan phantom (512×512), randomly rotated
   (±15°), scaled (90–110%), and translated per sample, so the dataset isn't
   600 copies of the same exact image.
2. **Forward projection**: a Radon transform produces the clean sinogram
   (432 angles × 1048 detector positions).
3. **Motion simulation**: each of the 432 projection angles is shifted
   according to a smooth sinusoidal trajectory (simulating gradual patient
   movement) with occasional larger random jumps (5% of views, simulating
   sudden movement).
4. **Physical noise**: per-view intensity variation, Gaussian sensor noise,
   and Poisson (photon-counting) noise are layered on top.
5. **Output**: 600 paired samples saved to a single HDF5 file, under
   `sino_true/<index>` and `sino_art/<index>`, with motion-severity metadata
   attached to each sample.

---

## 4. Model Architecture

`train_motion_corrector.py`

A **residual attention U-Net**, fully convolutional (no Flatten/Dense layers,
so spatial structure is preserved end-to-end):

- **Encoder**: 3 downsampling stages (Conv2D + MaxPooling2D), bottleneck at
  1/8 resolution.
- **Decoder**: 3 upsampling stages using **Conv2DTranspose** (learned
  upsampling — chosen over fixed nearest-neighbor upsampling, which was found
  to contribute to output blur during development).
- **Attention gates**: at each decoder stage, an attention block gates the
  matching encoder skip-connection, helping the network focus on relevant
  spatial regions when merging encoder and decoder features.
- **Residual output**: the network predicts a *correction* which is added
  back to the original input, rather than predicting the clean sinogram from
  scratch — an easier learning target since most of the input is already
  correct.

**Combined loss function:**
- MAE (pixel-wise accuracy)
- SSIM (structural similarity)
- Angle-consistency term (penalizes inconsistency between adjacent projection
  angles, which filtered back-projection otherwise amplifies into streaking)
- Identity-preservation term (discourages changing the output when the input
  was already close to clean)

---

## 5. Training

- **Optimizer**: Adam, wrapped in a `LossScaleOptimizer` for safe
  mixed-precision (`mixed_float16`) training.
- **Split**: 80% train / 10% validation / 10% test, randomly assigned with a
  fixed seed for reproducibility.
- **Callbacks**: `EarlyStopping` (patience 10 on validation loss),
  `ReduceLROnPlateau` (halves the learning rate on plateau), `ModelCheckpoint`
  (saves the best weights mid-training).
- **Hardware**: trained on a single Kaggle T4 GPU.

| Hyperparameter | Value |
|---|---|
| Input shape | 432 × 1048 × 1 |
| Base filters | 24 |
| Batch size | 1 |
| Learning rate | 1e-4 (initial) |
| Epochs | up to 100 (early-stopped) |

---

## 6. Results

Evaluated on the held-out test set (never seen during training):

| Metric | Before correction | After correction |
|---|---|---|
| PSNR (sinogram domain) | 19.53 dB | **32.16 dB** |
| SSIM (sinogram domain) | 0.6509 | **0.9702** |

The model was also evaluated after reconstructing corrected sinograms via
filtered back-projection (FBP), to confirm the correction holds up in the
image domain a radiologist would actually view — not just in raw sinogram
pixel error. See `results/` for side-by-side before/after/ground-truth
comparison images, in both the sinogram and reconstructed-image domains.

---

## 7. Project Structure

```
ct-motion-artifact-correction/
├── README.md
├── generate_synthetic_sinograms.py
├── train_motion_corrector.py
├── evaluate_motion_corrector.py
├── CODE_WALKTHROUGH.md             # line-by-line explanation of every script
├── results/
│   ├── metrics.txt
│   └── sample_*.png               # before/after comparison images
└── .gitignore                     # excludes the .h5 dataset and model weights
```

The synthetic dataset (`.h5`, several GB) and trained model weights are not
committed to this repository. [Add a link here once uploaded — e.g. a GitHub
Release, Google Drive, or Hugging Face link.]

---

## 8. How to Run

```bash
# 1. Generate the synthetic dataset
python src/generate_synthetic_sinograms.py

# 2. Train the model
python src/train_motion_corrector.py

# 3. Evaluate on the held-out test set
python src/evaluate_motion_corrector.py
```

A GPU is strongly recommended — this was developed and trained on a Kaggle
T4 GPU. File paths in each script's `CONFIG` dict may need adjusting depending
on your environment (they currently default to `/kaggle/working/`).

---

## 9. Limitations

- **Residual output softness**: despite testing several fixes (an
  angle-consistency loss term, increased model capacity, learned upsampling
  via `Conv2DTranspose`), corrected sinograms remain visibly softer than
  ground truth, and reconstructed images retain faint residual streaking.
  This is a known, documented limitation of pixel-wise loss functions
  (MAE/SSIM) in image restoration — the "safest" prediction under a pixel-wise
  objective is often a smoothed blend of plausible answers, which is why
  adversarial (GAN-based) losses are the standard next step for this class of
  problem. A pix2pix-style GAN extension of this project is planned.
- **Simplified motion model**: motion is simulated as per-view translation
  (a smooth trajectory plus occasional jumps), which does not capture the
  full complexity of real patient motion (rotation, deformable tissue
  movement, out-of-plane motion).
- **Limited anatomical diversity**: all samples derive from one base phantom
  (randomly transformed), rather than a variety of distinct anatomies —
  sufficient to demonstrate the method, but a real deployment would need
  training data spanning much more anatomical variation.
- **Synthetic-only evaluation**: the model has not been validated against
  real clinical CT motion artifact data.

---

## 10. Documentation

See [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md) for a full,
line-by-line explanation of what every function in all three scripts does and
why — from phantom generation through to the final evaluation metrics.
