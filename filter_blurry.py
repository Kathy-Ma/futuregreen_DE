"""
filter_blurry.py
----------------
Scans data/processed/dataset_vers1 for blurry images using the Laplacian
variance method. Blurry images are moved to rejected/blurry/ and renamed
blurry_<N>.jpg, where N continues from whatever is already in that folder.

The `rejected` directory (and all its subdirectories) is always skipped.
"""

import os
import re
import shutil
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = os.path.join("trash")

BLURRY_DIR = os.path.join("data", "processed","rejected", "blurry")

# Per-class Laplacian variance thresholds.
# Lower threshold = more tolerant (only very blurry images rejected).
# Higher threshold = stricter (mildly blurry images also rejected).
THRESHOLDS: dict[str, float] = {
    "cardboard":    50.0,
    "glass":        80.0,
    "metal":        100.0,
    "organic":     500.0,
    "paper":        100.0,
    "plastic":      90.0,
    "trash":        100.0,
}

DEFAULT_THRESHOLD: float = 100.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_laplacian_variance(image_path: str) -> float | None:
    """Return the Laplacian variance for a grayscale image, or None on error."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_next_blurry_counter(blurry_dir: str) -> int:
    """
    Scan blurry_dir for files named blurry_<N>.jpg and return max(N) + 1.
    Returns 1 if the directory is empty or contains no matching files.
    """
    max_id = 0
    if not os.path.isdir(blurry_dir):
        return 1

    pattern = re.compile(r"^blurry_(\d+)\.", re.IGNORECASE)
    for fname in os.listdir(blurry_dir):
        m = pattern.match(fname)
        if m:
            max_id = max(max_id, int(m.group(1)))

    return max_id + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure the output directory exists.
    os.makedirs(BLURRY_DIR, exist_ok=True)

    # Start counter safely above any existing blurry files.
    counter = get_next_blurry_counter(BLURRY_DIR)

    total_checked = 0
    total_blurry  = 0
    total_errors  = 0

    # Walk only the immediate subdirectories of SOURCE_DIR so we never
    # accidentally descend into `rejected/` or its children.
    try:
        entries = os.scandir(SOURCE_DIR)
    except OSError as exc:
        print(f"[ERROR] Cannot scan source directory: {exc}")
        return

    with entries:
        class_dirs = [
            e for e in entries
            if e.is_dir() and e.name.lower() != "rejected"
        ]

    for class_entry in sorted(class_dirs, key=lambda e: e.name):
        class_name = class_entry.name
        threshold  = THRESHOLDS.get(class_name, DEFAULT_THRESHOLD)

        print(f"\n[INFO] Processing class '{class_name}'  (threshold={threshold})")

        try:
            image_files = [
                f for f in os.scandir(class_entry.path)
                if f.is_file()
                and os.path.splitext(f.name)[1].lower() in IMAGE_EXTENSIONS
            ]
        except OSError as exc:
            print(f"  [ERROR] Cannot scan directory {class_entry.path}: {exc}")
            total_errors += 1
            continue

        for img_entry in sorted(image_files, key=lambda f: f.name):
            img_path = img_entry.path
            total_checked += 1

            variance = compute_laplacian_variance(img_path)

            if variance is None:
                print(f"  [WARN ] Could not read image (skipping): {img_entry.name}")
                total_errors += 1
                continue

            if variance < threshold:
                dest_name = f"blurry_{counter}.jpg"
                dest_path = os.path.join(BLURRY_DIR, dest_name)

                try:
                    shutil.move(img_path, dest_path)
                    print(
                        f"  [BLURRY] {img_entry.name}  variance={variance:.2f} < {threshold}"
                        f"  →  {dest_name}"
                    )
                    counter       += 1
                    total_blurry  += 1
                except OSError as exc:
                    print(f"  [ERROR] Failed to move {img_entry.name}: {exc}")
                    total_errors += 1
            else:
                print(f"  [OK   ] {img_entry.name}  variance={variance:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Images checked : {total_checked}")
    print(f"Blurry (moved) : {total_blurry}")
    print(f"Errors         : {total_errors}")
    print(f"Blurry output  : {os.path.abspath(BLURRY_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
