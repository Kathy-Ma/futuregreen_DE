"""
consolidate_paper.py
---------------------
Consolidates all paper sub-class images from data/raw into a single
directory with consistent naming: paper_1.jpg, paper_2.jpg, ...

Source datasets:
  - TrashBox/paper          (subfolders: news paper, paper, paper cups)
  - garbageClassification/paper
  - recyclableAndHouseholdWaste  (newspaper, office_paper, paper_cups)
  - trashNet/paper

Output:
  - data/processed/paper/paper_N.jpg
"""

import os
from pathlib import Path
from PIL import Image

def consolidate_paper():
    base_dir = Path(__file__).resolve().parents[2]
    data_raw = base_dir / "data" / "raw"

    source_dirs = [
        data_raw / "TrashBox" / "paper" / "news paper",
        data_raw / "TrashBox" / "paper" / "paper",
        data_raw / "TrashBox" / "paper" / "paper cups",

        data_raw / "garbageClassification" / "paper",

        data_raw / "recyclableAndHouseholdWaste" / "newspaper",
        data_raw / "recyclableAndHouseholdWaste" / "office_paper",
        data_raw / "recyclableAndHouseholdWaste" / "paper_cups",

        data_raw / "trashNet" / "paper",
    ]

    dest_dir = base_dir / "data" / "processed" / "paper"
    dest_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    global_counter = 1
    errors = 0
    skipped = 0

    for source in source_dirs:
        if not source.exists():
            print(f"[WARN] Source not found, skipping: {source}")
            continue

        files = sorted(
            p for p in source.rglob("*") if p.is_file()
        )
        print(f"[INFO] {source.relative_to(data_raw)}  →  {len(files)} files")

        for file_path in files:
            if file_path.suffix.lower() not in valid_extensions:
                skipped += 1
                continue

            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGBA").convert("RGB")
                    new_name = f"paper_{global_counter}.jpg"
                    img.save(dest_dir / new_name, "JPEG", quality=95)
                    global_counter += 1
            except Exception as exc:
                print(f"[ERROR] {file_path}: {exc}")
                errors += 1

    processed = global_counter - 1
    print("-" * 40)
    print(f"Done.  Processed: {processed}  |  Errors: {errors}  |  Skipped: {skipped}")
    print(f"Output: {dest_dir}")


if __name__ == "__main__":
    consolidate_paper()
