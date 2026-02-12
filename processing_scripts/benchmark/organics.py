import os
import cv2
import random
import numpy as np
from pathlib import Path

# Configuration
INPUT_DIR = r"data\raw\wasteClassificationData\DATASET\TRAIN\O"
OUTPUT_DIR = r"data\processed\benchmark_dataset\staging"
TARGET_COUNT = 600
RANDOM_SEED = 42

def process_organics():
    random.seed(RANDOM_SEED)
    
    # Resolve paths relative to CWD
    cwd = Path(os.getcwd())
    input_path = cwd / INPUT_DIR
    output_path = cwd / OUTPUT_DIR
    
    print(f"Current Working Directory: {cwd}")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Create output directories if they don't exist
    os.makedirs(output_path / "organics", exist_ok=True)

    # Collect images
    images = list(input_path.glob("*.jpg"))
    print(f"Found {len(images)} images in input.")
    
    if len(images) < TARGET_COUNT:
        print(f"Warning: Only found {len(images)} images, which is less than target {TARGET_COUNT}. Using all.")
        selected_images = images
    else:
        # Shuffle and select target count
        random.shuffle(images)
        selected_images = images[:TARGET_COUNT]
        print(f"Selected {len(selected_images)} images.")

    stats = {
        "total": 0
    }
    
    for img_path in selected_images:
        stats["total"] += 1
        
        # Read image
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Save to staging/organics/filename
        # We can rename here or in auto_sorter. Let's keep original names for now to avoid collision if run multiple times?
        # Or rename to known standard. "organics_raw_N.jpg" maybe?
        # Actually, let's just copy. auto_sorter will handle final naming.
        target_file = output_path / "organics" / img_path.name
        cv2.imwrite(str(target_file), img)

    print("\nProcessing Complete.")
    print(f"Total processed: {stats['total']}")
    print(f"Output directory: {output_path.resolve()}")

if __name__ == "__main__":
    process_organics()
