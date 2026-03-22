"""
config.py — Global paths and constants
"""
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "mammogram_cnn.keras")
DB_PATH        = os.path.join(BASE_DIR, "mammocad.db")
DATA_DIR       = os.path.join(BASE_DIR, "data")
TRAIN_DIR      = os.path.join(DATA_DIR, "train")
TEST_DIR       = os.path.join(DATA_DIR, "test")
UPLOAD_DIR     = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
ASSETS_DIR     = os.path.join(BASE_DIR, "assets")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Model ───────────────────────────────────────────────
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS         = 30
CLASSES        = ["Benign", "Malignant"]

# ── BI-RADS thresholds ──────────────────────────────────
# Probability of malignancy → BI-RADS category
BIRADS_THRESHOLDS = [
    (0.02, "BI-RADS 0", "Incomplete — Need additional imaging"),
    (0.10, "BI-RADS 2", "Benign finding"),
    (0.35, "BI-RADS 3", "Probably benign (<2% malignancy)"),
    (0.65, "BI-RADS 4", "Suspicious (~15-30% malignancy)"),
    (0.95, "BI-RADS 5", "Highly suggestive of malignancy (>95%)"),
    (1.01, "BI-RADS 6", "Known biopsy-proven malignancy"),
]

# ── Synthetic feature names for radar/cell plot ─────────
CELL_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
]

CELL_FEATURES_WORST = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# ── App theme ────────────────────────────────────────────
APP_TITLE      = "MammoCAD"
APP_SUBTITLE   = "AI-Powered Mammogram Analysis"
PRIMARY_COLOR  = "#E50914"
BG_COLOR       = "#141414"
CARD_COLOR     = "#1f1f1f"
TEXT_COLOR     = "#FFFFFF"