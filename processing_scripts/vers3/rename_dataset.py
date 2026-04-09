"""
Rename images in dataset_vers3.4 to follow the convention: class_number.jpg
Each class folder gets its images renamed with consecutive numbering starting from 1.
"""

import os
import shutil

# --- Configuration ---
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data", "processed", "dataset_vers3.4")

# Set to True to do a dry run (print renames without actually renaming)
DRY_RUN = False


def rename_images(dataset_dir, dry_run=False):
    """Rename all images in each class folder to class_number.jpg format."""
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])

    print(f"Dataset directory: {dataset_dir}")
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Dry run: {dry_run}\n")

    total_renamed = 0

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        
        # Get all image files, sorted for consistent ordering
        images = sorted([f for f in os.listdir(class_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"  {class_name}: {len(images)} images")

        # Rename to temporary names first to avoid conflicts
        # (e.g., cardboard_2.jpg -> cardboard_1.jpg when cardboard_1.jpg already exists)
        temp_names = []
        for i, old_name in enumerate(images):
            ext = os.path.splitext(old_name)[1]
            temp_name = f"__temp_{i}{ext}"
            old_path = os.path.join(class_dir, old_name)
            temp_path = os.path.join(class_dir, temp_name)

            if not dry_run:
                os.rename(old_path, temp_path)
            temp_names.append((temp_name, ext))

        # Now rename from temp to final names
        for i, (temp_name, ext) in enumerate(temp_names):
            new_name = f"{class_name}_{i + 1}{ext}"
            temp_path = os.path.join(class_dir, temp_name)
            new_path = os.path.join(class_dir, new_name)

            if dry_run:
                # Show original -> final mapping
                print(f"    {images[i]} -> {new_name}")
            else:
                os.rename(temp_path, new_path)

            total_renamed += 1

    print(f"\nTotal images renamed: {total_renamed}")
    if dry_run:
        print("(Dry run - no files were actually renamed)")


if __name__ == "__main__":
    rename_images(DATASET_DIR, dry_run=DRY_RUN)
