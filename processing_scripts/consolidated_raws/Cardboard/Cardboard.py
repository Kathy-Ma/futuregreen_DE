import os
from pathlib import Path
from PIL import Image

def consolidate_cardboard():
    base_dir = Path(__file__).resolve().parents[2] 
    data_raw = base_dir / "data" / "raw"
    
    source_dirs = [
        data_raw / "TrashBox" / "cardboard",

        data_raw / "trashNet" / "cardboard",

        data_raw / "garbageClassification" / "cardboard",

        data_raw / "recyclableAndHouseholdWaste" / "cardboard_boxes" / "default",
        data_raw / "recyclableAndHouseholdWaste" / "cardboard_boxes" / "real_world",

        data_raw / "recyclableAndHouseholdWaste" / "cardboard_packaging" / "default",
        data_raw / "recyclableAndHouseholdWaste" / "cardboard_packaging" / "real_world"
    ]

    dest_dir = base_dir / "data" / "processed" / "cardboard"
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
                    new_name = f"cardboard_{global_counter}.jpg"
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
    consolidate_cardboard()
