"""
prepare_dataset.py — CBIS-DDSM organizer using dicom_info.csv as the bridge

HOW IT WORKS:
  dicom_info.csv has:
    image_path = CBIS-DDSM/jpeg/1.3.6.xxx/1-172.jpg
                                ^^^^^^^^^^
                                This UID matches your jpeg/ folder names exactly!

  mass/calc CSVs have:
    image_file_path = Mass-Training_P_00001_LEFT_CC/1.3.6.xxx_OUTER/1.3.6.xxx_INNER/...
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    We extract the OUTER UID to look up in dicom_info

  So: CSV outer UID  ->  dicom_info lookup  ->  actual jpeg path  ->  copy + resize

Usage (Windows CMD — paste whole block):
  python prepare_dataset.py ^
    --jpeg-root "C:\\Users\\Dell\\Downloads\\mammogram dataset\\jpeg" ^
    --csv-dir   "C:\\Users\\Dell\\Downloads\\mammogram dataset\\csv" ^
    --output    "data"
"""

import sys
import argparse
from pathlib import Path

try:
    import pandas as pd
    from PIL import Image
except ImportError:
    print("[ERROR] Run:  pip install pandas Pillow")
    sys.exit(1)

IMG_SIZE = (224, 224)


# ══════════════════════════════════════════════════════════════
# Build lookup from dicom_info.csv
# Returns: { "1.3.6.xxx" : Path_to_actual_jpg }
# ══════════════════════════════════════════════════════════════

def build_dicom_info_index(csv_dir: Path, jpeg_root: Path) -> dict:
    dicom_csv = csv_dir / "dicom_info.csv"
    if not dicom_csv.exists():
        print(f"[ERROR] dicom_info.csv not found in {csv_dir}")
        sys.exit(1)

    print(f"\n[INDEX] Reading dicom_info.csv ...")
    df = pd.read_csv(dicom_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # image_path column: CBIS-DDSM/jpeg/1.3.6.xxx/1-172.jpg
    if "image_path" not in df.columns:
        print(f"[ERROR] 'image_path' column not found in dicom_info.csv")
        print(f"  Columns: {list(df.columns)}")
        sys.exit(1)

    uid_to_jpg = {}
    missing    = 0

    for _, row in df.iterrows():
        img_path_str = str(row.get("image_path", "")).strip()
        if not img_path_str or img_path_str == "nan":
            continue

        # Extract UID from path: CBIS-DDSM/jpeg/UID/filename.jpg
        parts = Path(img_path_str.replace("\\", "/")).parts
        # UID is the part that starts with 1.3.
        uid = None
        for p in parts:
            if p.startswith("1.3.") or p.startswith("1.2."):
                uid = p
                break

        if not uid:
            continue

        # Build the actual local path: jpeg_root / uid / filename
        filename = parts[-1]                          # e.g. "1-172.jpg"
        local    = jpeg_root / uid / filename

        if local.exists():
            uid_to_jpg[uid] = local
        else:
            # Try any jpg inside the uid folder
            uid_dir = jpeg_root / uid
            if uid_dir.exists():
                jpgs = list(uid_dir.glob("*.jpg")) + list(uid_dir.glob("*.jpeg"))
                if jpgs:
                    # Pick smaller file = cropped ROI
                    uid_to_jpg[uid] = min(jpgs, key=lambda p: p.stat().st_size)
                    continue
            missing += 1

    print(f"  Mapped {len(uid_to_jpg)} UIDs to local jpg files")
    if missing > 0:
        print(f"  {missing} entries in dicom_info had no matching local file (normal)")

    if not uid_to_jpg:
        print("\n[ERROR] No UIDs could be mapped to local files.")
        print("  Check that --jpeg-root points to the correct folder.")
        sys.exit(1)

    return uid_to_jpg


# ══════════════════════════════════════════════════════════════
# Extract the FIRST 1.3.x UID from a CSV path string
# CSV path: Mass-Training_P_xxx/1.3.6.OUTER/1.3.6.INNER/file.dcm
#           We want OUTER (index 1 in parts, first 1.3. match)
# ══════════════════════════════════════════════════════════════

def get_first_uid(path_str: str) -> str:
    parts = Path(path_str.replace("\\", "/")).parts
    for p in parts:
        if p.startswith("1.3.") or p.startswith("1.2."):
            return p
    return ""


def get_all_uids(path_str: str) -> list:
    parts = Path(path_str.replace("\\", "/")).parts
    return [p for p in parts if p.startswith("1.3.") or p.startswith("1.2.")]


# ══════════════════════════════════════════════════════════════
# Main organizer
# ══════════════════════════════════════════════════════════════

def organize(jpeg_root: str, csv_dir: str, output_root: str):
    jpeg_root = Path(jpeg_root)
    csv_dir   = Path(csv_dir)
    out       = Path(output_root)

    if not jpeg_root.exists():
        print(f"[ERROR] Not found: {jpeg_root.resolve()}")
        sys.exit(1)
    if not csv_dir.exists():
        print(f"[ERROR] Not found: {csv_dir.resolve()}")
        sys.exit(1)

    # Build UID -> jpg map via dicom_info.csv
    uid_index = build_dicom_info_index(csv_dir, jpeg_root)

    csv_files = {
        "train": [
            csv_dir / "mass_case_description_train_set.csv",
            csv_dir / "calc_case_description_train_set.csv",
        ],
        "test": [
            csv_dir / "mass_case_description_test_set.csv",
            csv_dir / "calc_case_description_test_set.csv",
        ],
    }

    copied = skipped = errors = 0

    for split, csv_list in csv_files.items():
        for csv_path in csv_list:
            if not csv_path.exists():
                print(f"  [SKIP] {csv_path.name} not found")
                continue

            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            print(f"\n[CSV] {csv_path.name}  ({len(df)} rows) -> split={split}")

            for i, row in df.iterrows():
                pathology = str(row.get("pathology", "")).upper()
                label     = "malignant" if "MALIGNANT" in pathology else "benign"
                out_dir   = out / split / label
                out_dir.mkdir(parents=True, exist_ok=True)

                found_jpg = None

                # Try columns in order of preference:
                # cropped_image = actual lesion ROI (best for training)
                # image_file    = full mammogram (fallback)
                for col in [
                    "cropped_image_file_path",
                    "image_file_path",
                    "roi_mask_file_path",
                ]:
                    val = str(row.get(col, "")).strip()
                    if not val or val == "nan":
                        continue

                    # Try every UID in the path (outer and inner)
                    for uid in get_all_uids(val):
                        if uid in uid_index:
                            found_jpg = uid_index[uid]
                            break
                    if found_jpg:
                        break

                if not found_jpg:
                    skipped += 1
                    continue

                dst = out_dir / f"{split}_{label}_{i:05d}.jpg"
                if dst.exists():
                    copied += 1
                    continue

                try:
                    img = Image.open(found_jpg).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
                    img.save(dst, "JPEG", quality=95)
                    copied += 1
                    if copied % 200 == 0:
                        print(f"  ... {copied} images done")
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  [ERR] {found_jpg}: {e}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Copied   : {copied}")
    print(f"  Skipped  : {skipped}")
    print(f"  Errors   : {errors}")
    print(f"{'='*52}")

    total = 0
    for split in ["train", "test"]:
        for label in ["benign", "malignant"]:
            d = out / split / label
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            total += n
            print(f"  data/{split}/{label:10s}  ->  {n:4d} images")

    print(f"\n  TOTAL: {total} images")

    if total == 0:
        print("\n[STILL FAILING] Run these two inspect commands and share output:")
        print(f'  python prepare_dataset.py --inspect-jpeg "{jpeg_root}"')
        print(f'  python prepare_dataset.py --inspect-csv  "{csv_dir}"')
    else:
        print(f"\n[DONE] -> {out.resolve()}")
        print("\n  Next step:  python train_ai.py")


# ══════════════════════════════════════════════════════════════
# Inspect helpers
# ══════════════════════════════════════════════════════════════

def inspect_jpeg(root: str):
    root = Path(root)
    dirs = [d for d in root.iterdir() if d.is_dir()]
    print(f"\n[INSPECT jpeg] {root.resolve()}")
    print(f"  Total UID folders: {len(dirs)}\n")
    for d in dirs[:3]:
        files = list(d.iterdir())
        print(f"  FOLDER: {d.name}")
        for f in files[:5]:
            kb = f.stat().st_size // 1024 if f.is_file() else 0
            ftype = "DIR" if f.is_dir() else "FILE"
            print(f"    {ftype}: {f.name}  ({kb} KB)")
        print()


def inspect_csv(csv_dir: str):
    csv_dir = Path(csv_dir)
    print(f"\n[INSPECT csv] {csv_dir.resolve()}")
    csvs = list(csv_dir.glob("*.csv"))
    print(f"  CSV files found: {len(csvs)}")
    for csv_path in sorted(csvs):
        print(f"\n  FILE: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path, nrows=2)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            print(f"  Columns: {list(df.columns)}")
            for col in df.columns:
                if "path" in col or "file" in col:
                    val  = str(df[col].iloc[0]) if len(df) > 0 else "—"
                    uids = get_all_uids(val)
                    print(f"  [{col}]")
                    print(f"    value : {val[:120]}")
                    print(f"    UIDs  : {uids}")
        except Exception as e:
            print(f"  Could not read: {e}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM for MammoCAD")
    parser.add_argument("--jpeg-root",    dest="jpeg_root",    default="")
    parser.add_argument("--csv-dir",      dest="csv_dir",      default="")
    parser.add_argument("--output",       default="data")
    parser.add_argument("--inspect-jpeg", dest="inspect_jpeg", default="")
    parser.add_argument("--inspect-csv",  dest="inspect_csv",  default="")
    args = parser.parse_args()

    if args.inspect_jpeg:
        inspect_jpeg(args.inspect_jpeg)
    elif args.inspect_csv:
        inspect_csv(args.inspect_csv)
    elif args.jpeg_root and args.csv_dir:
        organize(args.jpeg_root, args.csv_dir, args.output)
    else:
        print("Usage (Windows):\n")
        print("  python prepare_dataset.py ^")
        print('    --jpeg-root "C:\\Users\\Dell\\Downloads\\mammogram dataset\\jpeg" ^')
        print('    --csv-dir   "C:\\Users\\Dell\\Downloads\\mammogram dataset\\csv" ^')
        print('    --output    "data"')