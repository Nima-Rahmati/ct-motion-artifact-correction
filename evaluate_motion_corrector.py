"""
Evaluation script — the piece the original project was missing.

Computes PSNR/SSIM on the held-out test split (sinogram domain), saves
before/after comparison images, and reports an FBP-reconstructed-domain
PSNR/SSIM if `skimage` is available. Use the outputs of this script
directly in your README results table.
"""

import os
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon

H5_PATH = "/kaggle/working/synthetic_dataset_motion_fixed.h5"
MODEL_PATH = "/kaggle/working/trained_motion_vf.keras"
TEST_INDICES_PATH = "/kaggle/working/test_indices.npy"
OUT_DIR = "/kaggle/working/results"
NUM_QUALITATIVE_SAMPLES = 6

os.makedirs(OUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

test_idx = np.load(TEST_INDICES_PATH)
f = h5py.File(H5_PATH, 'r')
true_keys = sorted(list(f['sino_true'].keys()), key=lambda x: int(x))
art_keys = sorted(list(f['sino_art'].keys()), key=lambda x: int(x))

psnr_before, psnr_after = [], []
ssim_before, ssim_after = [], []

for n, i in enumerate(test_idx):
    art = np.array(f['sino_art'][art_keys[i]], dtype=np.float32)
    tru = np.array(f['sino_true'][true_keys[i]], dtype=np.float32)
    # FIX: data is (H, W, 1) from the generator, not legacy (1, H, W) —
    # squeeze the channel axis (last), and only when it's actually size 1.
    if art.ndim == 3 and art.shape[-1] == 1:
        art = np.squeeze(art, axis=-1)
    elif art.ndim == 3 and art.shape[0] == 1:  # legacy (1, H, W) format
        art = np.squeeze(art, axis=0)
    if tru.ndim == 3 and tru.shape[-1] == 1:
        tru = np.squeeze(tru, axis=-1)
    elif tru.ndim == 3 and tru.shape[0] == 1:
        tru = np.squeeze(tru, axis=0)

    art_in = np.expand_dims(np.expand_dims(art, 0), -1)
    pred = model.predict(art_in, verbose=0)[0, ..., 0]

    data_range = tru.max() - tru.min()
    psnr_before.append(psnr(tru, art, data_range=data_range))
    psnr_after.append(psnr(tru, pred, data_range=data_range))
    ssim_before.append(ssim(tru, art, data_range=data_range))
    ssim_after.append(ssim(tru, pred, data_range=data_range))

    if n < NUM_QUALITATIVE_SAMPLES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(
            axes, [art, pred, tru], ["Input (motion)", "Model output", "Ground truth"]
        ):
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"sample_{n}_sinogram.png"), dpi=150)
        plt.close(fig)

        # Reconstructed-image-domain comparison via FBP
        try:
            num_angles = tru.shape[0]
            theta = np.linspace(0, 180, num_angles, endpoint=False)
            recon_art = iradon(art.T, theta=theta, circle=True)
            recon_pred = iradon(pred.T, theta=theta, circle=True)
            recon_true = iradon(tru.T, theta=theta, circle=True)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, img, title in zip(
                axes, [recon_art, recon_pred, recon_true],
                ["Reconstruction (motion)", "Reconstruction (corrected)", "Reconstruction (ground truth)"]
            ):
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"sample_{n}_reconstruction.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Skipping reconstruction comparison for sample {n}: {e}")

f.close()

print("\n=== Sinogram-domain results (test set) ===")
print(f"PSNR  before: {np.mean(psnr_before):.2f} dB   after: {np.mean(psnr_after):.2f} dB")
print(f"SSIM  before: {np.mean(ssim_before):.4f}       after: {np.mean(ssim_after):.4f}")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as fp:
    fp.write("Sinogram-domain results (test set)\n")
    fp.write(f"PSNR before: {np.mean(psnr_before):.2f} dB\n")
    fp.write(f"PSNR after:  {np.mean(psnr_after):.2f} dB\n")
    fp.write(f"SSIM before: {np.mean(ssim_before):.4f}\n")
    fp.write(f"SSIM after:  {np.mean(ssim_after):.4f}\n")

print(f"\nQualitative comparisons and metrics.txt saved to ./{OUT_DIR}/")
