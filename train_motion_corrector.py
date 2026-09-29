"""
CT Motion Artifact Correction — training script (revised)

Fixes applied vs. the original draft:
  1. Model now performs true residual learning (input + predicted residual).
  2. Optimizer wrapped in LossScaleOptimizer for safe mixed-precision training
     with a custom train_step.
  3. Data loader reshuffles training indices every epoch.
  4. Train/val split is randomized (seeded) instead of taking a fixed slice.
  5. SSIM max_val is computed per-batch instead of hardcoded to 1.0, so the
     loss stays meaningful even without upstream normalization.
  6. Removed the dead/shadowed ssim_weight variable; one config block now
     controls every hyperparameter.
  7. Added ModelCheckpoint so a crash mid-training doesn't lose progress.
  8. Added global seeding for reproducibility.
  9. A separate test split is now held out (train/val/test instead of train/val).
 10. Data loader handles (H, W, 1) sinograms directly (generator's actual output
     shape) instead of assuming a legacy (1, H, W) format.
 11. _identity_loss now casts x/y/y_pred to float32 before subtracting, fixing
     a mixed_float16 dtype mismatch (y_pred is float32 from the model's final
     layer, but x arrives as whatever dtype the loader produced).
 12. All output paths (h5, checkpoints, final model) point at /kaggle/working/
     so nothing gets written somewhere that disappears with the session.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer, set_global_policy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable
import h5py
import logging

# ---------------- Config ----------------
CONFIG = {
    "h5_path": "/kaggle/working/synthetic_dataset_motion_fixed.h5",
    "input_shape": (432, 1048, 1),  # REDUCED to match the generator's new half-resolution output
    "base_filters": 24,           # INCREASED from 16 — the blurry/low-contrast output persisted
                                   # across every tv_angle_weight tried (0, 0.05, 0.005), which means
                                   # TV wasn't the actual cause. Most likely explanation: the network
                                   # doesn't have enough capacity to represent the fine crossing-band
                                   # structure in an 864x2096 sinogram after 3 rounds of 2x downsampling.
                                   # If this OOMs on your GPU, drop back to 20 or 16.
    "batch_size": 1,
    "epochs": 100,
    "learning_rate": 1e-4,
    "identity_threshold": 0.01,   # NOTE: validate this against your actual data scale (see eval script)
    "identity_weight": 2.0,
    "alpha": 0.1,                 # weight of MAE vs. SSIM term in motion_loss
    "ssim_weight": 0.3,           # weight of the SSIM term within the combined loss
    "tv_angle_weight": 0.005,     # REDUCED from 0.05 — that value over-smoothed the output
                                   # (lost real structural detail, washed-out low-contrast result).
                                   # Penalizes row-to-row (angle-to-angle) inconsistency, which FBP's
                                   # ramp filter amplifies into streak artifacts. Tune from here —
                                   # if streaks return, raise slightly (e.g. 0.01); if still blurry,
                                   # lower further (e.g. 0.002).
    "seed": 42,
    "train_split": 0.8,
    "val_split": 0.1,             # remainder becomes test
    "checkpoint_path": "/kaggle/working/checkpoints/motion_corrector_best.weights.h5",
    "final_model_path": "/kaggle/working/trained_motion_vf.keras",
    "test_indices_path": "/kaggle/working/test_indices.npy",
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])

set_global_policy('mixed_float16')

os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)


# ---------------- Attention block ----------------
def attention_block(x, g, filters, kernel_size=(1, 1)):
    theta_x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    phi_g = layers.Conv2D(filters, kernel_size, padding='same')(g)
    add = layers.add([theta_x, phi_g])
    psi = layers.Conv2D(1, kernel_size, activation='sigmoid', padding='same')(add)
    return layers.multiply([x, psi])


# ---------------- Model ----------------
def build_residual_attention_unet(input_shape=(432, 1048, 1), base_filters=16, num_channels=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(base_filters * 2, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(base_filters * 2, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(base_filters * 4, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(base_filters * 4, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    bn = layers.Conv2D(base_filters * 8, 3, activation='relu', padding='same')(p3)
    bn = layers.Conv2D(base_filters * 8, 3, activation='relu', padding='same')(bn)

    # Decoder
    # CHANGED: UpSampling2D (fixed nearest-neighbor, no learned parameters) -> Conv2DTranspose
    # (learned upsampling). Nearest-neighbor upsampling has no way to "invent" detail — it can only
    # repeat existing pixels — which is a common cause of exactly the soft/blurry output seen in
    # earlier runs. Conv2DTranspose learns how to upsample, giving the decoder more ability to
    # reconstruct fine structure lost during pooling.
    u3 = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding='same', activation='relu')(bn)
    a3 = attention_block(c3, u3, base_filters * 4)
    u3 = layers.Concatenate()([u3, a3])
    c4 = layers.Conv2D(base_filters * 4, 3, activation='relu', padding='same')(u3)

    u2 = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding='same', activation='relu')(c4)
    a2 = attention_block(c2, u2, base_filters * 2)
    u2 = layers.Concatenate()([u2, a2])
    c5 = layers.Conv2D(base_filters * 2, 3, activation='relu', padding='same')(u2)

    u1 = layers.Conv2DTranspose(base_filters, 3, strides=2, padding='same', activation='relu')(c5)
    a1 = attention_block(c1, u1, base_filters)
    u1 = layers.Concatenate()([u1, a1])
    c6 = layers.Conv2D(base_filters, 3, activation='relu', padding='same')(u1)

    residual = layers.Conv2D(num_channels, 1, activation='linear', dtype='float32', name='residual')(c6)

    # Actually add the residual back to the input instead of returning it directly.
    outputs = layers.Add(name='corrected_output')([inputs, residual])

    return Model(inputs, outputs)


# ---------------- Loss ----------------
@register_keras_serializable(package='Custom', name='motion_loss_v4')
def motion_loss_v4(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Derive max_val from the batch instead of assuming [0,1]-normalized data.
    max_val = tf.maximum(tf.reduce_max(y_true), 1e-6)
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))

    # NEW: angle-consistency term. y_pred is (batch, angle, detector, channel);
    # this penalizes abrupt changes between adjacent projection angles (rows),
    # which is exactly what FBP's ramp filter amplifies into radial streaking
    # during reconstruction — even when sinogram-domain MAE/SSIM look fine.
    tv_angle = tf.reduce_mean(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))

    alpha = CONFIG["alpha"]
    ssim_weight = CONFIG["ssim_weight"]
    tv_angle_weight = CONFIG["tv_angle_weight"]
    total_loss = (1 - alpha) * mae - alpha * ssim_weight * ssim_val + tv_angle_weight * tv_angle
    return total_loss


# ---------------- Data loader ----------------
class SinoDataLoader(tf.keras.utils.Sequence):
    """
    Expects an HDF5 file with groups 'sino_art' (artifact/motion-corrupted)
    and 'sino_true' (clean target), each containing one dataset per sample,
    keyed by an integer-parseable string. Samples are stored as (H, W, 1).
    """

    def __init__(self, h5_file, indices, batch_size=1, shuffle=True):
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.f = h5py.File(h5_file, 'r')

        self.true_keys = sorted(list(self.f['sino_true'].keys()), key=lambda x: int(x))
        self.art_keys = sorted(list(self.f['sino_art'].keys()), key=lambda x: int(x))

        self.indices = np.array(indices)

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        bidx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for i in bidx:
            art = np.array(self.f['sino_art'][self.art_keys[i]], dtype=np.float32)
            tru = np.array(self.f['sino_true'][self.true_keys[i]], dtype=np.float32)

            # Data is already (H, W, 1) from the generator — only reshape if it isn't.
            if art.ndim == 2:
                art = np.expand_dims(art, -1)
            elif art.ndim == 3 and art.shape[0] == 1:  # legacy (1, H, W) format
                art = np.expand_dims(np.squeeze(art, axis=0), -1)

            if tru.ndim == 2:
                tru = np.expand_dims(tru, -1)
            elif tru.ndim == 3 and tru.shape[0] == 1:
                tru = np.expand_dims(np.squeeze(tru, axis=0), -1)

            X.append(art)
            Y.append(tru)
        return np.array(X, np.float32), np.array(Y, np.float32)

    def on_epoch_end(self):
        # Reshuffle every epoch instead of using a fixed sample order.
        if self.shuffle:
            np.random.shuffle(self.indices)

    def close(self):
        self.f.close()


def make_splits(h5_path, train_frac, val_frac, seed):
    with h5py.File(h5_path, 'r') as f:
        n = len(f['sino_true'].keys())
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)  # randomize before splitting instead of taking a fixed slice

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ---------------- Wrapper model with identity-preservation loss ----------------
class MotionCorrector(tf.keras.Model):
    def __init__(self, base_model, identity_weight=1.0, identity_threshold=0.01):
        super().__init__()
        self.base_model = base_model
        self.identity_weight = identity_weight
        self.identity_threshold = identity_threshold
        self.train_mae = tf.keras.metrics.Mean(name='train_mae')
        self.val_mae = tf.keras.metrics.Mean(name='val_mae')

    def compile(self, optimizer, loss_fn):
        # FIX: pass loss= to super().compile(). Even though train_step/test_step
        # are fully overridden below, Keras 3 needs a registered loss here or
        # validation triggers "No loss to compute. Provide a `loss` argument
        # in `compile()`." internally, mid-fit().
        super().compile(optimizer=optimizer, loss=loss_fn)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.train_mae, self.val_mae]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def _identity_loss(self, x, y_pred, y):
        # FIX: cast to float32 before doing math — under mixed_float16, x/y can
        # arrive as float16 while y_pred (forced float32 by the model's final
        # layer) is float32, and TF refuses to subtract mismatched dtypes.
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        diff = tf.reduce_mean(tf.abs(x - y), axis=[1, 2, 3])
        healthy = tf.cast(diff < self.identity_threshold, tf.float32)
        return tf.reduce_mean(tf.abs(y_pred - x) * healthy)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # FIX: call self(x, ...) instead of self.base_model(x, ...). Calling
            # base_model directly bypasses this wrapper's own __call__, so Keras
            # never marks the MotionCorrector instance itself as "built" — which
            # ModelCheckpoint(save_weights_only=True) requires. self() delegates
            # to base_model via call() below, but also builds the wrapper.
            y_pred = self(x, training=True)
            base_loss = self.loss_fn(y, y_pred)
            identity_loss = self._identity_loss(x, y_pred, y)
            total_loss = base_loss + self.identity_weight * identity_loss
            # FIX: Keras 3's LossScaleOptimizer uses scale_loss(), not the old
            # TF1/TF2 get_scaled_loss()/get_unscaled_gradients() API.
            # apply_gradients() unscales internally now, so no separate step needed.
            scaled_loss = self.optimizer.scale_loss(total_loss)

        grads = tape.gradient(scaled_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.train_mae.update_state(tf.reduce_mean(tf.abs(tf.cast(y, tf.float32) - tf.cast(y_pred, tf.float32))))
        return {"loss": total_loss, "mae": self.train_mae.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        base_loss = self.loss_fn(y, y_pred)
        identity_loss = self._identity_loss(x, y_pred, y)
        total_loss = base_loss + self.identity_weight * identity_loss
        self.val_mae.update_state(tf.reduce_mean(tf.abs(tf.cast(y, tf.float32) - tf.cast(y_pred, tf.float32))))
        return {"loss": total_loss, "mae": self.val_mae.result()}


# ---------------- Train ----------------
if __name__ == "__main__":
    train_idx, val_idx, test_idx = make_splits(
        CONFIG["h5_path"], CONFIG["train_split"], CONFIG["val_split"], CONFIG["seed"]
    )
    logging.info(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    np.save(CONFIG["test_indices_path"], test_idx)  # so the eval script uses the exact same held-out set

    train_loader = SinoDataLoader(CONFIG["h5_path"], train_idx, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = SinoDataLoader(CONFIG["h5_path"], val_idx, batch_size=CONFIG["batch_size"], shuffle=False)

    base_model = build_residual_attention_unet(
        input_shape=CONFIG["input_shape"], base_filters=CONFIG["base_filters"]
    )

    # Wrap optimizer for safe mixed-precision gradients in a custom train_step.
    optimizer = LossScaleOptimizer(Adam(CONFIG["learning_rate"]))

    model = MotionCorrector(
        base_model,
        identity_weight=CONFIG["identity_weight"],
        identity_threshold=CONFIG["identity_threshold"],
    )
    model.compile(optimizer=optimizer, loss_fn=motion_loss_v4)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7),
        ModelCheckpoint(  # Checkpoint during training, not just after fit() finishes.
            # FIX: save_weights_only=True. MotionCorrector is a hand-written
            # subclassed model with no get_config(), so Keras 3 can't serialize
            # the whole wrapper object to .keras. Weights-only avoids that —
            # base_model (a plain functional model) is still saved in full at
            # the end via base_model.save(), which doesn't have this problem.
            CONFIG["checkpoint_path"], monitor='val_loss', save_best_only=True,
            save_weights_only=True,
        ),
    ]

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    base_model.save(CONFIG["final_model_path"], include_optimizer=False)
    logging.info(f"Model saved to {CONFIG['final_model_path']}")

    train_loader.close()
    val_loader.close()
