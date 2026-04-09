"""
undersample.py
--------------
Applies random undersampling to data/processed/dataset_vers1 to produce a
balanced dataset across all classes.

How it works
------------
1. Scan each class subdirectory (ignoring `rejected/` and non-directories).
2. Count valid image files per class.
3. Set the target size = minimum count across all non-empty classes.
4. For every class that exceeds the target, randomly select the excess images
   and move them to  rejected/undersampled/<class_name>/.
   (Images are MOVED, not deleted, so nothing is lost.)
5. Renumber the surviving images in each class so they are consecutive
   starting from 1  (e.g. cardboard_1.jpg, cardboard_2.jpg, …).
6. Print a before/after summary.

Safe to re-run: images already moved to rejected/undersampled/ are not counted
in subsequent runs because that folder lives under `rejected/`, which is always
skipped.

Reproducibility: set RANDOM_SEED to a fixed integer (or None to disable).
"""

import os
import re
import random
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = os.path.join("data", "processed", "dataset_vers3")

# Destination for randomly removed images — lives inside the existing
# `rejected/` folder so it is automatically skipped on future runs.
UNDERSAMPLED_DIR = os.path.join(SOURCE_DIR, "rejected", "undersampled")

# Directories (case-insensitive) to always skip when scanning SOURCE_DIR.
IGNORED_DIRS = {"rejected"}

# Accepted image file extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Set to an integer for reproducible results, or None for a truly random run.
RANDOM_SEED: int | None = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_files(directory: str) -> list[str]:
    """Return a sorted list of absolute paths to image files in *directory*."""
    try:
        entries = os.scandir(directory)
    except OSError as exc:
        print(f"  [ERROR] Cannot scan {directory}: {exc}")
        return []

    with entries:
        files = sorted(
            e.path
            for e in entries
            if e.is_file()
            and os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS
        )
    return files


def collect_classes(source_dir: str) -> dict[str, list[str]]:
    """
    Scan *source_dir* for class subdirectories (ignoring IGNORED_DIRS).
    Returns {class_name: [image_path, ...]} for every discovered class.
    """
    classes: dict[str, list[str]] = {}

    try:
        entries = os.scandir(source_dir)
    except OSError as exc:
        print(f"[ERROR] Cannot scan source directory: {exc}")
        return classes

    with entries:
        class_dirs = [
            e for e in entries
            if e.is_dir() and e.name.lower() not in IGNORED_DIRS
        ]

    for entry in sorted(class_dirs, key=lambda e: e.name):
        images = get_image_files(entry.path)
        classes[entry.name] = images
        status = f"{len(images)} images"
        if not images:
            status += "  ⚠  (empty — will be skipped)"
        print(f"  {entry.name:<20} {status}")

    return classes


def natural_key(path: str) -> list:
    """Sort key that orders filenames with embedded numbers numerically."""
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def renumber_class(class_dir: str, class_name: str) -> tuple[int, int]:
    """
    Rename all image files in *class_dir* to  <class_name>_1.<ext>,
    <class_name>_2.<ext>, … in natural sort order.

    Uses a two-pass approach:
      Pass 1 – rename every file to a unique temp name (avoids collisions
               when e.g. cardboard_3 would overwrite cardboard_3 before it
               was itself renamed).
      Pass 2 – rename each temp file to its final consecutive name.

    Returns (renamed_count, error_count).
    """
    images = sorted(get_image_files(class_dir), key=natural_key)
    if not images:
        return 0, 0

    renamed = 0
    errors = 0

    # Pass 1: move every file to a guaranteed-unique temp name.
    temp_paths: list[tuple[str, str]] = []  # (temp_path, original_ext)
    for i, src_path in enumerate(images):
        ext = os.path.splitext(src_path)[1].lower()
        tmp_path = os.path.join(class_dir, f"__tmp_{i}{ext}")
        try:
            os.rename(src_path, tmp_path)
            temp_paths.append((tmp_path, ext))
        except OSError as exc:
            print(f"    [ERROR] Temp-rename failed for {os.path.basename(src_path)}: {exc}")
            errors += 1
            temp_paths.append((src_path, ext))  # keep original so pass 2 can skip it

    # Pass 2: rename each temp file to <class_name>_<n>.<ext>.
    for n, (tmp_path, ext) in enumerate(temp_paths, start=1):
        final_name = f"{class_name}_{n}{ext}"
        final_path = os.path.join(class_dir, final_name)
        if tmp_path == final_path:          # nothing changed
            renamed += 1
            continue
        try:
            os.rename(tmp_path, final_path)
            renamed += 1
        except OSError as exc:
            print(f"    [ERROR] Final-rename failed for {os.path.basename(tmp_path)}: {exc}")
            errors += 1

    return renamed, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Random Undersampling")
    print(f"Source : {os.path.abspath(SOURCE_DIR)}")
    print(f"Output : {os.path.abspath(UNDERSAMPLED_DIR)}")
    print(f"Seed   : {RANDOM_SEED}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Discover classes and count images
    # ------------------------------------------------------------------
    print("\n[INFO] Scanning class directories …\n")
    classes = collect_classes(SOURCE_DIR)

    if not classes:
        print("[ERROR] No class directories found. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2. Determine target sample size
    # ------------------------------------------------------------------
    non_empty = {cls: imgs for cls, imgs in classes.items() if imgs}
    empty_classes = [cls for cls, imgs in classes.items() if not imgs]

    if empty_classes:
        print(f"\n[WARN] Empty class(es) skipped when computing target: "
              f"{', '.join(empty_classes)}")

    if not non_empty:
        print("\n[ERROR] All class directories are empty. Nothing to do.")
        return

    target_size = min(len(imgs) for imgs in non_empty.values())
    print(f"\n[INFO] Target sample size (minimum across non-empty classes): "
          f"{target_size}")

    # ------------------------------------------------------------------
    # 3. Apply undersampling
    # ------------------------------------------------------------------
    print("\n[INFO] Applying undersampling …\n")

    total_moved = 0
    total_errors = 0

    for class_name, image_paths in sorted(classes.items()):
        count = len(image_paths)

        if class_name in empty_classes:
            print(f"  {class_name:<20} skipped (empty)")
            continue

        excess = count - target_size

        if excess <= 0:
            print(f"  {class_name:<20} OK  (already at or below target: "
                  f"{count} images)")
            continue

        # Destination folder for this class's excess images.
        dest_dir = os.path.join(UNDERSAMPLED_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        # Randomly choose which images to remove.
        to_move = random.sample(image_paths, excess)

        moved = 0
        for src_path in to_move:
            fname = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, fname)

            # Avoid name collision inside the destination folder.
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(fname)
                dest_path = os.path.join(dest_dir, f"{base}_dup{ext}")

            try:
                shutil.move(src_path, dest_path)
                moved += 1
            except OSError as exc:
                print(f"    [ERROR] Failed to move {fname}: {exc}")
                total_errors += 1

        print(f"  {class_name:<20} {count} → {target_size}  "
              f"(moved {moved} to rejected/undersampled/{class_name}/)")
        total_moved += moved

    # ------------------------------------------------------------------
    # 4. Renumber surviving images in every class
    # ------------------------------------------------------------------
    print("\n[INFO] Renumbering surviving images …\n")

    total_renamed = 0
    for class_name in sorted(classes):
        if class_name in empty_classes:
            continue
        class_dir = os.path.join(SOURCE_DIR, class_name)
        renamed, errs = renumber_class(class_dir, class_name)
        total_errors += errs
        total_renamed += renamed
        print(f"  {class_name:<20} {renamed} files renumbered consecutively")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Target size per class : {target_size}")
    print(f"Total images moved    : {total_moved}")
    print(f"Total images renamed  : {total_renamed}")
    print(f"Errors                : {total_errors}")
    print(f"Undersampled output   : {os.path.abspath(UNDERSAMPLED_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
