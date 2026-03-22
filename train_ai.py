"""
train_ai.py — CNN training on CBIS-DDSM dataset
Architecture: EfficientNetB3 transfer-learning + custom head
Run: python train_ai.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import (
    MODEL_PATH, TRAIN_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, CLASSES
)

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. Data generators with aggressive augmentation
# ══════════════════════════════════════════════════════════════

def build_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2,
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
        classes=["benign", "malignant"],
    )
    val_ds = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
        classes=["benign", "malignant"],
    )
    test_ds = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        classes=["benign", "malignant"],
    )
    return train_ds, val_ds, test_ds


# ══════════════════════════════════════════════════════════════
# 2. Model: EfficientNetB3 + custom classification head
# ══════════════════════════════════════════════════════════════

def build_model():
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    # Freeze base initially
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base


# ══════════════════════════════════════════════════════════════
# 3. Training — two-phase (frozen → fine-tune)
# ══════════════════════════════════════════════════════════════

def train():
    print("\n[TRAIN] Loading dataset …")
    train_ds, val_ds, test_ds = build_generators()

    print(f"[TRAIN] Classes: {train_ds.class_indices}")
    print(f"[TRAIN] Train samples : {train_ds.samples}")
    print(f"[TRAIN] Val samples   : {val_ds.samples}")
    print(f"[TRAIN] Test samples  : {test_ds.samples}")

    # Handle class imbalance
    n_benign    = len(list(Path(TRAIN_DIR).glob("benign/*")))
    n_malignant = len(list(Path(TRAIN_DIR).glob("malignant/*")))
    total = n_benign + n_malignant
    w_benign    = (1 / n_benign)    * total / 2.0
    w_malignant = (1 / n_malignant) * total / 2.0
    class_weights = {0: w_benign, 1: w_malignant}
    print(f"[TRAIN] Class weights: benign={w_benign:.3f}, malignant={w_malignant:.3f}")

    model, base = build_model()

    # ── Phase 1: Train head only ──────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor="val_auc", mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                    patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_auc", mode="max"),
    ]

    print("\n[PHASE 1] Training head …")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=cb_list,
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────
    base.trainable = True
    # Unfreeze top 30 layers only
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    print("\n[PHASE 2] Fine-tuning …")
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=cb_list,
    )

    # ── Evaluation ────────────────────────────────────────────
    print("\n[EVAL] Evaluating on test set …")
    results = model.evaluate(test_ds)
    print(f"  Test Loss     : {results[0]:.4f}")
    print(f"  Test Accuracy : {results[1]:.4f}")
    print(f"  Test AUC      : {results[2]:.4f}")

    preds = (model.predict(test_ds) > 0.5).astype(int).flatten()
    print("\n" + classification_report(test_ds.classes, preds,
                                       target_names=["Benign", "Malignant"]))

    # ── Confusion matrix plot ──────────────────────────────────
    cm = confusion_matrix(test_ds.classes, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_title("Confusion Matrix — Test Set")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig("models/confusion_matrix.png", dpi=150)
    print("[TRAIN] Confusion matrix saved.")

    # ── Training curves ────────────────────────────────────────
    _plot_history(h1, h2)
    print(f"\n[TRAIN] Model saved to {MODEL_PATH}")


def _plot_history(h1, h2):
    acc  = h1.history["accuracy"]   + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    auc  = h1.history["auc"]        + h2.history["auc"]
    vauc = h1.history["val_auc"]    + h2.history["val_auc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(acc, label="Train Acc");  ax1.plot(vacc, label="Val Acc")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(auc, label="Train AUC"); ax2.plot(vauc, label="Val AUC")
    ax2.set_title("AUC"); ax2.legend()
    fig.tight_layout()
    fig.savefig("models/training_curves.png", dpi=150)


# ══════════════════════════════════════════════════════════════
# Utility: prepare CBIS-DDSM from CSV metadata
# ══════════════════════════════════════════════════════════════

def prepare_cbis_ddsm_from_csv(
    mass_train_csv="data/mass_case_description_train_set.csv",
    mass_test_csv="data/mass_case_description_test_set.csv",
    calc_train_csv="data/calc_case_description_train_set.csv",
    calc_test_csv="data/calc_case_description_test_set.csv",
    images_root="data/CBIS-DDSM",
):
    """
    Organizes CBIS-DDSM images into data/train/benign|malignant
    and data/test/benign|malignant based on pathology column.

    Run this ONCE before training if your dataset has the CSV+DICOM layout.
    """
    import pandas as pd
    import shutil
    import pydicom
    from PIL import Image

    def convert_dicom(src, dst):
        ds  = pydicom.dcmread(src)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst)

    for csv_path, split in [
        (mass_train_csv, "train"), (mass_test_csv, "test"),
        (calc_train_csv, "train"), (calc_test_csv, "test"),
    ]:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for _, row in df.iterrows():
            pathology = str(row.get("pathology", "")).upper()
            label = "malignant" if "MALIGNANT" in pathology else "benign"
            out_dir = os.path.join("data", split, label)
            os.makedirs(out_dir, exist_ok=True)

            # Locate DICOM file
            patient_id = str(row.get("patient_id", "")).replace("/", "_")
            dcm_col = "cropped_image_file_path" if "cropped_image_file_path" in row.index else "image_file_path"
            rel_path = str(row.get(dcm_col, ""))
            dcm_path = os.path.join(images_root, rel_path)

            if os.path.exists(dcm_path):
                fname = f"{patient_id}_{label}_{_}.png"
                dst   = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    try:
                        convert_dicom(dcm_path, dst)
                    except Exception as e:
                        print(f"  [WARN] Could not convert {dcm_path}: {e}")
    print("[PREP] Dataset prepared.")


if __name__ == "__main__":
    # Uncomment if you need to prepare CBIS-DDSM from CSV/DICOM:
    # prepare_cbis_ddsm_from_csv()

    train()