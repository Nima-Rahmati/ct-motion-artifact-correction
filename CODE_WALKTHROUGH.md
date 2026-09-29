# Code Walkthrough: CT Motion Artifact Correction Pipeline

Plain-language explanation of the three scripts, in the order they run:
`generate_synthetic_sinograms.py` → `train_motion_corrector.py` → `evaluate_motion_corrector.py`.

---

## PART 1: `generate_synthetic_sinograms.py`

**Goal of this whole file**: create fake (but realistic-ish) paired training data —
for each sample, a *clean* sinogram and a *motion-corrupted* version of the same
sinogram — since real motion-artifact CT data with a perfect "clean" reference
doesn't exist (you can't ask a real patient to hold perfectly still AND move at
the same time). Simulating both from the same source phantom is the trick.

### Imports (lines 30-36)
```python
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize, rotate
from scipy.ndimage import shift as nd_shift
```
- `shepp_logan_phantom()` — a built-in, standard fake "brain slice" image used
  everywhere in CT research as a stand-in for a real patient scan.
- `radon(...)` — the actual math that turns a 2D image into a sinogram. This is
  literally simulating what a CT scanner does physically: shining X-rays through
  the object from many angles and recording how much gets absorbed along each ray.
- `resize`, `rotate` — standard image resizing/rotating.
- `shift` (renamed `nd_shift`) — moves an array's content by a sub-pixel amount,
  used to simulate motion.

### CONFIG dict (lines 41-63)
All the tunable numbers in one place instead of scattered through the code.
Key ones:
- `"H": 432, "W": 1048` — the final sinogram's shape: 432 different X-ray angles
  (rows), 1048 detector positions per angle (columns).
- `"phantom_size": 512` — the *image*-domain phantom is a separate, square
  512×512 image — kept deliberately separate from the sinogram's H×W, which was
  the original bug (conflating "picture of a brain" size with "sinogram" size).
- `"motion_shift_max_frac": 0.02` — how far (as a fraction of image width) the
  simulated motion can shift a projection.
- `"random_seed": 42` — makes the random generation reproducible; same seed
  always produces the same dataset.

### `crop_or_pad_to()` (lines 69-84)
A small utility: after scaling a phantom up or down, its size no longer matches
`phantom_size`. This function trims (crops) or fills in (pads) the image back to
the exact target size, centered.

### `circular_mask()` (lines 87-96)
Zeroes out everything outside a circle inscribed in the image. This matters
because `radon()` (a few functions down) assumes the object being scanned fits
inside a circular field of view — anything square/rectangular sticking into the
corners would produce invalid projections.

### `generate_phantom()` (lines 99-119) — makes one "patient" per sample
1. Start from the standard Shepp-Logan phantom, resized to 512×512.
2. **Randomly rotate** it (±15°).
3. **Randomly scale** it (90%-110% size), then crop/pad back to 512×512.
4. **Randomly shift/translate** it by a few pixels.
5. Add a tiny bit of random pixel noise.
6. Clip pixel values to stay in [0, 1].

This is what makes each of your 600 samples anatomically different rather than
600 copies of the exact same picture — same base shape, but rotated/scaled/
shifted differently each time.

### `phantom_to_sinogram()` (lines 123-136) — the actual "CT scan" simulation
```python
angles = np.linspace(0.0, 180.0, H, endpoint=False)
sino = radon(phantom, theta=angles, circle=True)
sino = sino.T
```
- Creates 432 evenly-spaced angles between 0° and 180°.
- `radon()` shoots simulated X-rays through the phantom at every one of those
  angles and records the absorption pattern — this **is** the sinogram.
- `.T` (transpose) just flips rows/columns so the array ends up shaped
  `(angle, detector_position)` = `(432, ~512)` instead of the other way around.
```python
if sino.shape[1] != W:
    sino = resize(sino, (H, W), ...)
```
The detector axis from `radon()` naturally comes out close to `phantom_size`
(~512), not exactly `W` (1048), so this stretches it to the target width via
proper interpolation.
```python
sino = (sino - minv) / (maxv - minv)   # normalize to [0, 1]
```
Rescales pixel values so the darkest point is 0 and the brightest is 1 — keeps
every sample on a comparable numeric scale for training.

### `smooth_motion_trajectory()` (lines 139-143) — designs *how* the patient "moves"
```python
base = max_shift * np.sin(2.0 * np.pi * freq * x)
noise = np.convolve(rng.normal(...), np.ones(5)/5, mode="same")
return base + noise
```
Creates a smooth, wavy back-and-forth motion pattern (a sine wave) across all
432 angles, plus a bit of lightly-smoothed random jitter on top. This is a
simplified stand-in for a patient's breathing/small movements during a real scan
— smooth and continuous, not random jumps at every single angle.

### `add_motion_and_physical_noise()` (lines 146-172) — corrupts the clean sinogram
Loops over each of the 432 rows (each row = one projection angle) one at a time:
```python
shift_val = traj_shifts[a] + jump_mags[a]
shifted_row = nd_shift(row, shift=shift_val, ...)
```
Shifts that one row sideways by the trajectory amount (plus an occasional larger
random "jump," simulating a sudden patient movement 5% of the time).
```python
shifted_row = shifted_row * intensity_factor      # brightness jitter
shifted_row = shifted_row + rng.normal(...)        # sensor noise
counts = rng.poisson(shifted_row * poisson_scale)  # photon-counting noise
```
Then layers on three more realistic imperfections: slight per-view brightness
variation, ordinary sensor noise, and **Poisson noise** — the specific kind of
"grainy" statistical noise that comes from counting individual X-ray photons,
which is how real CT detectors actually behave (fewer photons = noisier reading).

### `main()` (lines 176-226) — ties it all together and saves to disk
For each of the 600 samples:
1. Build a random phantom.
2. Turn it into a clean sinogram (`sino_true`).
3. Corrupt a copy of it with motion + noise (`sino_art`).
4. Save both into the HDF5 file, under `sino_true/<index>` and `sino_art/<index>`
   — this specific group structure is what the training script's loader expects.
5. Attach a few metadata tags (how much motion was applied) to each sample, in
   case you want to analyze correction quality vs. motion severity later.

---

## PART 2: `train_motion_corrector.py`

**Goal**: train a neural network that takes a motion-corrupted sinogram as input
and outputs a corrected (hopefully clean-looking) one.

### CONFIG (lines 38-66)
Same pattern as the generator — every tunable number in one place. Notable ones,
given everything discovered during debugging:
- `"base_filters": 24` — controls how much "capacity" (how many feature channels)
  the network has at each layer. Started at 16, raised after blur persisted.
- `"tv_angle_weight": 0.005` — weight of a custom loss term explained below.
- `"identity_threshold"`, `"identity_weight"` — control a special loss term that
  tells the network "if the input was already nearly clean, don't change it."

### `attention_block()` (lines 79-84)
This is a small "gate" used in the decoder (explained below) that helps the
network decide *where* in the image to pay attention when combining information
from the encoder with information from the decoder. It computes a "relevance
mask" (`psi`, values between 0 and 1 via sigmoid) and multiplies it against the
skip-connection features (`x`), effectively dimming down irrelevant regions and
keeping important ones.

### `build_residual_attention_unet()` (lines 88-133) — the network architecture
This is a **U-Net**: an hourglass-shaped network that shrinks the image down
(encoder), processes it at a small/abstract scale (bottleneck), then grows it
back up to full size (decoder), while passing detailed information sideways
from encoder to decoder at each scale via "skip connections."

**Encoder** (lines 92-105) — shrinks the image, 3 times:
```python
c1 = Conv2D(...)(inputs)      # process at full resolution
p1 = MaxPooling2D(2)(c1)      # then shrink it in half
c2 = Conv2D(...)(p1)          # process at half resolution
p2 = MaxPooling2D(2)(c2)      # shrink in half again
c3 = Conv2D(...)(p2)          # process at quarter resolution
p3 = MaxPooling2D(2)(c3)      # shrink in half again
bn = Conv2D(...)(p3)          # "bottleneck" — process at 1/8 resolution
```
Each `Conv2D` layer is scanning small local patches of the image and learning to
detect useful patterns; each `MaxPooling2D` halves the image's height and width,
so information becomes more compressed/abstract each step down.

**Decoder** (lines 107-126) — grows it back up, 3 times, mirroring the encoder:
```python
u3 = Conv2DTranspose(...)(bn)          # learned upsampling: double the size
a3 = attention_block(c3, u3, ...)      # gate the matching encoder features
u3 = Concatenate()([u3, a3])           # combine upsampled + gated skip features
c4 = Conv2D(...)(u3)                   # process the combination
```
This repeats 3 times, each time doubling the resolution back up and pulling in
the matching encoder layer's detail via the attention-gated skip connection —
this is how the network "remembers" fine detail that pure downsampling would
otherwise lose. (`Conv2DTranspose` replaced a simpler `UpSampling2D` here after
testing showed the fixed/non-learned upsampling was contributing to blur.)

**Output** (lines 128-131):
```python
residual = Conv2D(...)(c6)                          # predict a "correction"
outputs = Add()([inputs, residual])                  # add it to the original input
```
Instead of asking the network to predict the entire clean sinogram from
scratch, it predicts a small *correction* (residual) to add on top of the
original input. This is usually easier to learn, since most of the input is
already correct — the network just needs to figure out what to fix.

### `motion_loss_v4()` (lines 137-158) — how "wrong" is measured
```python
mae = tf.reduce_mean(tf.abs(y_true - y_pred))
```
**MAE (mean absolute error)**: average, pixel-by-pixel, how far off the
prediction is from the true clean sinogram. The main driver of training.
```python
ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
```
**SSIM (structural similarity)**: a more perceptual measure than raw pixel
error — rewards the output for having the right local contrast/structure, not
just the right average brightness.
```python
tv_angle = tf.reduce_mean(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))
```
**Angle-consistency term**: compares each row (angle) to the next one and
penalizes big jumps between them. Added specifically because FBP reconstruction
(used later, in evaluation) amplifies exactly this kind of row-to-row
inconsistency into visible streaks.
```python
total_loss = (1 - alpha) * mae - alpha * ssim_weight * ssim_val + tv_angle_weight * tv_angle
```
All three combined into one number the network tries to minimize. `alpha` and
the two weights control how much each term matters relative to the others.

### `SinoDataLoader` (lines 162-211) — feeds data to the model in batches
A `keras.utils.Sequence` subclass — Keras's standard way of streaming data from
disk in small pieces instead of loading the whole dataset into memory at once
(important here, since the full dataset is several GB).
- `__getitem__` (line ~184): given a batch index, opens the HDF5 file, reads the
  corresponding motion-corrupted (`X`) and clean (`Y`) sinograms, makes sure
  they're shaped `(H, W, 1)`, and returns them as a batch.
- `on_epoch_end` (line 205): reshuffles the sample order after every epoch, so
  the network doesn't see samples in the exact same sequence every time.

### `make_splits()` (lines 214-227)
Randomly (but reproducibly, via the seed) divides all 600 samples into three
non-overlapping groups: 80% train, 10% validation, 10% test. Training only ever
uses the "train" group; "val" is checked periodically during training to catch
overfitting; "test" is held back completely until final evaluation.

### `MotionCorrector` class (lines 231-297) — wraps the model with custom training logic
This is a Keras `Model` subclass that wraps the U-Net (`base_model`) and adds
custom behavior on top — specifically, the "identity preservation" idea:
```python
def _identity_loss(self, x, y_pred, y):
    diff = tf.reduce_mean(tf.abs(x - y), axis=[1, 2, 3])
    healthy = tf.cast(diff < self.identity_threshold, tf.float32)
    return tf.reduce_mean(tf.abs(y_pred - x) * healthy)
```
If a sample's input and true-clean versions were already very close to each
other (meaning: not much motion actually happened), `healthy` becomes 1, and
the network gets an extra penalty for changing the output much from the input
— i.e., "if it's already fine, don't mess with it."

`train_step()` (lines 268-288) is the actual training logic run on every batch:
1. Run the input through the model, get a prediction.
2. Compute the combined loss (`motion_loss_v4` + identity loss).
3. Compute gradients (how to adjust every weight in the network to reduce the
   loss) via `tf.GradientTape`.
4. Apply those gradients to update the model's weights (`apply_gradients`).

`test_step()` (lines 290-297) does the same forward pass and loss calculation,
but skips the gradient/weight-update steps — this is what runs during
validation, where you want to *measure* performance without changing the model.

### Main training block (lines 301-351)
1. Split the dataset, save which samples ended up in "test" so evaluation later
   uses the exact same held-out set.
2. Build the data loaders and the model.
3. `model.compile(...)` — register the optimizer and loss function.
4. Set up three callbacks: `EarlyStopping` (stop automatically once validation
   loss stops improving), `ReduceLROnPlateau` (shrink the learning rate when
   progress stalls), `ModelCheckpoint` (save the best-so-far weights during
   training, in case the session crashes).
5. `model.fit(...)` — actually runs training, looping over the data for up to
   100 epochs (but usually stopping earlier via `EarlyStopping`).
6. Save the final trained model to disk.

---

## PART 3: `evaluate_motion_corrector.py`

**Goal**: measure how well the trained model actually performs on data it never
saw during training, and produce the images/numbers you'll use in your README.

### Setup (lines 20-33)
Load the trained model, load the saved list of test-set sample indices (so
you're evaluating on genuinely unseen data), open the dataset file.

### Main loop (lines 38-93) — for each test sample:
```python
art_in = np.expand_dims(np.expand_dims(art, 0), -1)
pred = model.predict(art_in, verbose=0)[0, ..., 0]
```
Feed the motion-corrupted sinogram through the trained model to get its
prediction (the "corrected" sinogram), then strip the extra batch/channel
dimensions back off for easy comparison.
```python
psnr_before.append(psnr(tru, art, data_range=data_range))
psnr_after.append(psnr(tru, pred, data_range=data_range))
ssim_before.append(ssim(tru, art, data_range=data_range))
ssim_after.append(ssim(tru, pred, data_range=data_range))
```
**PSNR** and **SSIM** are computed twice per sample: once comparing the
*original corrupted* sinogram to ground truth ("before"), once comparing the
*model's prediction* to ground truth ("after"). The gap between these two
numbers, averaged across the whole test set, is your headline result.

For the first 6 samples, it also saves images:
- A side-by-side sinogram comparison (input / model output / ground truth).
- A **reconstruction-domain** comparison: runs `iradon()` (the inverse of
  `radon()` — turns a sinogram back into a viewable image) on all three
  versions, so you can see what the actual CT slice would look like, not just
  the raw sinogram data. This is the step that revealed the reconstruction-only
  streaking artifacts that sinogram-domain metrics alone didn't catch.

### Final summary (lines 97-108)
Prints and saves the averaged before/after PSNR and SSIM numbers to
`metrics.txt` — this is the exact number pair (`19.53 → 31.55 dB`,
`0.6509 → 0.9681`) you'd quote in your resume/README.

---

## The big picture, in one paragraph

You generate paired clean/corrupted synthetic CT sinograms (`generate_*.py`),
train a U-Net to map corrupted → clean using a combination of pixel-accuracy,
structural-similarity, and row-consistency loss terms (`train_*.py`), then
measure how much better the model's output matches ground truth compared to
doing nothing, in both the raw sinogram domain and the reconstructed-image
domain (`evaluate_*.py`). The final numbers and images are your evidence that
the model works.
