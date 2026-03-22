"""
predict.py — Inference + synthetic feature extraction from mammogram images
"""
import os
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf

from config import MODEL_PATH, IMG_SIZE, BIRADS_THRESHOLDS, CELL_FEATURES, CELL_FEATURES_WORST


# ── Load model once ────────────────────────────────────────────

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run: python train_ai.py"
            )
        # compile=False avoids optimizer header mismatch error
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


# ══════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════

def predict_image(image_path: str) -> dict:
    model = get_model()
    img   = _load_image(image_path)
    prob  = float(model.predict(img, verbose=0)[0][0])  # P(malignant)

    prediction     = "Malignant" if prob >= 0.5 else "Benign"
    benign_prob    = round(1 - prob, 4)
    malignant_prob = round(prob, 4)

    birads_cat, birads_desc = _get_birads(prob)
    features = _extract_features(image_path, prob)

    return {
        "prediction":      prediction,
        "benign_prob":     benign_prob,
        "malignant_prob":  malignant_prob,
        "birads_category": birads_cat,
        "birads_desc":     birads_desc,
        "features":        features,
    }


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _get_birads(prob: float):
    for threshold, cat, desc in BIRADS_THRESHOLDS:
        if prob < threshold:
            return cat, desc
    return "BI-RADS 6", "Known biopsy-proven malignancy"


# ══════════════════════════════════════════════════════════════
# Feature extraction via image processing
# ══════════════════════════════════════════════════════════════

def _extract_features(image_path: str, malignant_prob: float) -> dict:
    # Read as grayscale — already uint8 from file
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        # Fallback: read with PIL and convert
        try:
            pil_img  = Image.open(image_path).convert("L")
            img_gray = np.array(pil_img, dtype=np.uint8)
        except Exception:
            return _synthetic_features(malignant_prob)

    # Resize and ensure uint8 — this is the key fix
    img_u8 = cv2.resize(img_gray, (256, 256))
    img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

    # OTSU threshold now works correctly on uint8
    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _synthetic_features(malignant_prob)

    # Largest contour → mass candidate
    c         = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull      = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) + 1e-6

    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 128, 128

    dists       = [np.sqrt((p[0][0] - cx)**2 + (p[0][1] - cy)**2) for p in c]
    radius_mean = np.mean(dists)
    radius_std  = np.std(dists)
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
    solidity    = area / hull_area
    concavity   = 1.0 - solidity
    smoothness  = radius_std / (radius_mean + 1e-6)

    # Texture via Laplacian variance (on uint8 image)
    lap_var = cv2.Laplacian(img_u8, cv2.CV_64F).var()

    # Symmetry
    x, y, w, h = cv2.boundingRect(c)
    roi = img_u8[y:y + h, x:x + w]
    if roi.shape[1] > 1:
        left  = roi[:, :roi.shape[1] // 2]
        right = cv2.flip(roi[:, roi.shape[1] // 2:], 1)
        min_h = min(left.shape[0], right.shape[0])
        min_w = min(left.shape[1], right.shape[1])
        sym_diff = np.mean(np.abs(
            left[:min_h, :min_w].astype(float) - right[:min_h, :min_w].astype(float)
        ))
        symmetry = 1.0 - (sym_diff / 255.0)
    else:
        symmetry = 0.5

    # Fractal dimension proxy
    edges       = cv2.Canny(img_u8, 50, 150)
    fractal_dim = edges.sum() / (256 * 256 * 255.0)

    def clamp(v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo + 1e-8), 0, 1))

    features = {
        "radius_mean":            clamp(radius_mean,  5,   100),
        "texture_mean":           clamp(lap_var,       0,  5000),
        "perimeter_mean":         clamp(perimeter,    50,  1000),
        "area_mean":              clamp(area,         100, 50000),
        "smoothness_mean":        clamp(smoothness,    0,   0.5),
        "compactness_mean":       clamp(compactness,   1,   5),
        "concavity_mean":         clamp(concavity,     0,   1),
        "concave_points_mean":    clamp(len(c) / 100,  0,   5),
        "symmetry_mean":          symmetry,
        "fractal_dimension_mean": clamp(fractal_dim,   0,   0.3),
    }

    # Worst = mean amplified by malignancy probability
    for k in list(features.keys()):
        features[k.replace("_mean", "_worst")] = float(
            np.clip(features[k] * (1 + malignant_prob * 0.5), 0, 1)
        )

    # SE = small noise around mean
    for k in list(features.keys()):
        if "_mean" in k:
            features[k.replace("_mean", "_se")] = float(
                np.clip(features[k] * 0.15 + np.random.normal(0, 0.02), 0, 1)
            )

    return features


def _synthetic_features(malignant_prob: float) -> dict:
    """Fallback when OpenCV cannot process the image."""
    rng   = np.random.default_rng(seed=int(malignant_prob * 1000))
    noise = lambda: float(rng.normal(0, 0.05))

    base = {
        "radius_mean":            np.clip(0.3  + malignant_prob * 0.50 + noise(), 0, 1),
        "texture_mean":           np.clip(0.25 + malignant_prob * 0.45 + noise(), 0, 1),
        "perimeter_mean":         np.clip(0.3  + malignant_prob * 0.50 + noise(), 0, 1),
        "area_mean":              np.clip(0.2  + malignant_prob * 0.55 + noise(), 0, 1),
        "smoothness_mean":        np.clip(0.4  - malignant_prob * 0.20 + noise(), 0, 1),
        "compactness_mean":       np.clip(0.2  + malignant_prob * 0.50 + noise(), 0, 1),
        "concavity_mean":         np.clip(0.1  + malignant_prob * 0.60 + noise(), 0, 1),
        "concave_points_mean":    np.clip(0.1  + malignant_prob * 0.65 + noise(), 0, 1),
        "symmetry_mean":          np.clip(0.6  - malignant_prob * 0.30 + noise(), 0, 1),
        "fractal_dimension_mean": np.clip(0.3  + malignant_prob * 0.30 + noise(), 0, 1),
    }
    result = {k: float(v) for k, v in base.items()}
    for k, v in base.items():
        result[k.replace("_mean", "_worst")] = float(np.clip(v * (1 + malignant_prob * 0.5), 0, 1))
        result[k.replace("_mean", "_se")]    = float(np.clip(v * 0.15 + abs(noise()), 0, 1))
    return result