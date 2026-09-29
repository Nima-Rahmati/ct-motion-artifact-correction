"""
synthetic_sino_generator.py

Generates paired (clean, motion-corrupted) sinograms for training the
motion-artifact-correction U-Net.

Fixes applied vs. the original draft:
  1. HDF5 layout now uses groups 'sino_true'/'sino_art' with per-sample
     datasets keyed by index (matches SinoDataLoader's expectations).
  2. Image-domain phantom size is decoupled from the target sinogram shape
     (H, W) — no more conflating a reconstruction-domain image with a
     projection-domain array.
  3. Removed the tiling-based "radon_to_target_width" step, which destroyed
     detector-position information. Detector-axis resizing is now a proper
     interpolation on the real 2D sinogram.
  4. "num_channels" (which was actually angle count, mislabeled) is replaced
     by a clear NUM_ANGLES_GEN / H distinction.
  5. Motion is now simulated as a per-row (per-view) detector-direction
     shift directly on the 2D sinogram, matching how real motion artifacts
     arise (view-to-view inconsistency), instead of shifting slices of a
     misshapen 3D array.
  6. Each sample now uses a randomly rotated/scaled/translated phantom for
     anatomical diversity, instead of one fixed Shepp-Logan phantom repeated
     with only pixel-level noise.
  7. Motion severity is parameterized relative to the output width.
  8. Per-sample metadata (motion severity, trajectory params) saved as
     HDF5 attributes for later analysis.
"""

import numpy as np
import h5py
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize, rotate
from scipy.ndimage import shift as nd_shift
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- Config -----------------
CONFIG = {
    "num_samples": 600,       # INCREASED from 300 — cheaper per-sample now that resolution is
                               # reduced, so more samples costs roughly the same total time.
    "phantom_size": 512,      # square image-domain phantom (reconstruction domain)
    "H": 432,                 # REDUCED from 864 (half) — for portfolio purposes, full clinical
    "W": 1048,                # REDUCED from 2096 (half) — resolution isn't needed. This is 1/4 the
                               # total pixels, so training/eval iterate roughly 4x faster, and it's
                               # still cleanly divisible by 8 for the model's 3 pooling levels
                               # (432/8=54, 1048/8=131). Must match model input_shape in training script.
    "motion_shift_max_frac": 0.02,   # max shift as a fraction of W (was a fixed 8px on 2096 -> too subtle)
    "motion_jump_prob": 0.05,
    "motion_noise_scale": 0.6,
    "noise_std_gauss": 0.005,
    "poisson_scale": 1e5,
    "intensity_variation": 0.07,
    "apply_circular_mask": True,
    "h5_filename": "synthetic_dataset_motion_fixed.h5",
    "compress": True,
    "random_seed": 42,
    "rotation_range_deg": 15,
    "scale_range": (0.9, 1.1),
    "translate_range_px": 10,
}

rng = np.random.default_rng(CONFIG["random_seed"])


# ----------------- Phantom (image domain) -----------------
def crop_or_pad_to(img, size):
    h, w = img.shape
    if h >= size:
        top = (h - size) // 2
        img = img[top:top + size, :]
    else:
        pad = size - h
        img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0)), mode="reflect")
    h, w = img.shape
    if w >= size:
        left = (w - size) // 2
        img = img[:, left:left + size]
    else:
        pad = size - w
        img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2)), mode="reflect")
    return img


def circular_mask(image):
    H_img, W_img = image.shape
    Y, X = np.ogrid[:H_img, :W_img]
    center = (H_img // 2, W_img // 2)
    radius = min(center)
    dist = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = dist <= radius
    masked = image.copy()
    masked[~mask] = 0.0
    return masked


def generate_phantom(size):
    """Base Shepp-Logan phantom with random rotation/scale/translation for
    per-sample anatomical diversity."""
    img = shepp_logan_phantom()
    img = resize(img, (size, size), mode="reflect", anti_aliasing=True)

    angle = rng.uniform(-CONFIG["rotation_range_deg"], CONFIG["rotation_range_deg"])
    img = rotate(img, angle, resize=False, mode="reflect")

    scale = rng.uniform(*CONFIG["scale_range"])
    new_size = max(8, int(size * scale))
    img = resize(img, (new_size, new_size), mode="reflect", anti_aliasing=True)
    img = crop_or_pad_to(img, size)

    dx = rng.integers(-CONFIG["translate_range_px"], CONFIG["translate_range_px"] + 1)
    dy = rng.integers(-CONFIG["translate_range_px"], CONFIG["translate_range_px"] + 1)
    img = nd_shift(img, shift=(dy, dx), mode="reflect")

    img = img + rng.normal(0, 0.005, size=img.shape)
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


# ----------------- Sinogram construction -----------------
def phantom_to_sinogram(phantom, H, W):
    """Forward-project directly at H angles, then resize only the detector
    axis to W (a proper 2D interpolation, not the old tiling hack)."""
    angles = np.linspace(0.0, 180.0, H, endpoint=False)
    sino = radon(phantom, theta=angles, circle=True).astype(np.float32)  # (detector_len, H)
    sino = sino.T  # -> (H, detector_len): rows = angles, cols = detector position

    if sino.shape[1] != W:
        sino = resize(sino, (H, W), mode="reflect", anti_aliasing=True).astype(np.float32)

    minv, maxv = sino.min(), sino.max()
    if maxv - minv > 1e-8:
        sino = (sino - minv) / (maxv - minv)
    return np.clip(sino, 0.0, 1.0).astype(np.float32)


def smooth_motion_trajectory(n_angles, max_shift, freq=1.5, noise_scale=0.6):
    x = np.linspace(0, 1, n_angles)
    base = max_shift * np.sin(2.0 * np.pi * freq * x)
    noise = np.convolve(rng.normal(0, noise_scale, size=n_angles), np.ones(5) / 5, mode="same")
    return base + noise


def add_motion_and_physical_noise(sino_HxW, traj_shifts, poisson_scale, gaussian_std,
                                   jump_prob, intensity_variation):
    """Apply per-view (per-row) detector-direction shifts plus physical noise
    directly on a real (H, W) sinogram."""
    H, W = sino_HxW.shape
    sino_art = np.zeros_like(sino_HxW, dtype=np.float32)

    jumps = rng.random(H) < jump_prob
    jump_mags = rng.uniform(-3.0, 3.0, size=H) * jumps

    for a in range(H):
        shift_val = traj_shifts[a] + jump_mags[a]
        row = sino_HxW[a, :]
        shifted_row = nd_shift(row, shift=shift_val, mode="nearest", order=1)

        intensity_factor = 1.0 + rng.uniform(-intensity_variation, intensity_variation)
        shifted_row = shifted_row * intensity_factor
        shifted_row = shifted_row + rng.normal(0, gaussian_std, size=shifted_row.shape)
        shifted_row = np.clip(shifted_row, 0.0, None)

        counts = rng.poisson(np.clip(shifted_row * poisson_scale, 0, None)).astype(np.float32)
        if poisson_scale > 0:
            shifted_row = counts / (poisson_scale + 1e-8)

        sino_art[a, :] = np.clip(shifted_row, 0.0, 1.0)

    return sino_art


# ----------------- Main generation loop -----------------
def main():
    H, W = CONFIG["H"], CONFIG["W"]
    motion_shift_max = CONFIG["motion_shift_max_frac"] * W

    mb_per_sample = H * W * 4 / (1024 ** 2) * 2  # true + art
    logging.info(f"~{mb_per_sample:.1f} MB per sample pair, "
                 f"~{mb_per_sample * CONFIG['num_samples'] / 1024:.2f} GB total (uncompressed)")

    with h5py.File(CONFIG["h5_filename"], "w") as f:
        grp_true = f.create_group("sino_true")
        grp_art = f.create_group("sino_art")

        for i in tqdm(range(CONFIG["num_samples"]), desc="Generating samples"):
            phantom = generate_phantom(CONFIG["phantom_size"])
            if CONFIG["apply_circular_mask"]:
                phantom = circular_mask(phantom)

            sino_true = phantom_to_sinogram(phantom, H, W)

            freq = rng.uniform(0.8, 2.5)
            traj = smooth_motion_trajectory(H, motion_shift_max, freq=freq,
                                             noise_scale=CONFIG["motion_noise_scale"])
            sino_art = add_motion_and_physical_noise(
                sino_true, traj,
                poisson_scale=CONFIG["poisson_scale"],
                gaussian_std=CONFIG["noise_std_gauss"],
                jump_prob=CONFIG["motion_jump_prob"],
                intensity_variation=CONFIG["intensity_variation"],
            )

            # add channel dim -> (H, W, 1), matching the model's expected input
            sino_true_out = sino_true[..., None]
            sino_art_out = sino_art[..., None]

            kwargs = {"compression": "gzip", "compression_opts": 4} if CONFIG["compress"] else {}
            ds_true = grp_true.create_dataset(str(i), data=sino_true_out, **kwargs)
            ds_art = grp_art.create_dataset(str(i), data=sino_art_out, **kwargs)

            # metadata for later analysis / your results writeup
            ds_art.attrs["motion_shift_max"] = motion_shift_max
            ds_art.attrs["motion_freq"] = freq
            ds_art.attrs["jump_prob"] = CONFIG["motion_jump_prob"]

            if (i + 1) % 25 == 0:
                logging.info(f"{i + 1}/{CONFIG['num_samples']} samples generated")

    logging.info(f"Done. Dataset written to {CONFIG['h5_filename']}")


if __name__ == "__main__":
    main()
