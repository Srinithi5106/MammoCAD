"""
model_downloader.py — Downloads model from Google Drive on first startup.
Called automatically by predict.py if model is missing.
"""
import os
import sys

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mammogram_cnn.keras")

# ── Paste your Google Drive file ID here ──────────────────────
GDRIVE_FILE_ID = os.environ.get("MODEL_GDRIVE_ID", "1Zsyftw3xBbkQfCWNF1oSC0rZwfr0xRvc")


def download_model():
    if os.path.exists(MODEL_PATH):
        print("[MODEL] Already exists, skipping download.")
        return True

    if GDRIVE_FILE_ID == "1Zsyftw3xBbkQfCWNF1oSC0rZwfr0xRvc":
        print("[MODEL] ERROR: Set MODEL_GDRIVE_ID environment variable in Render dashboard.")
        return False

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[MODEL] Downloading from Google Drive (ID: {GDRIVE_FILE_ID})...")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"[MODEL] Downloaded successfully ({size_mb:.1f} MB)")
            return True
        else:
            print("[MODEL] Download failed — file not found after download.")
            return False

    except ImportError:
        print("[MODEL] gdown not installed, trying requests fallback...")
        return _download_with_requests()
    except Exception as e:
        print(f"[MODEL] gdown failed: {e}, trying requests fallback...")
        return _download_with_requests()


def _download_with_requests():
    """Fallback downloader using requests for large files."""
    import requests

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    session = requests.Session()
    url     = "https://drive.google.com/uc?export=download"
    params  = {"id": GDRIVE_FILE_ID}

    print("[MODEL] Attempting requests-based download...")
    response = session.get(url, params=params, stream=True)
    token    = get_confirm_token(response)

    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True)

    # Write in chunks to handle large files
    chunk_size = 32768
    downloaded = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024 * 10) == 0:
                    print(f"  ... {downloaded // (1024*1024)} MB downloaded")

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[MODEL] Downloaded successfully ({size_mb:.1f} MB)")
        return True
    else:
        print("[MODEL] Download appears to have failed (file too small or missing).")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)