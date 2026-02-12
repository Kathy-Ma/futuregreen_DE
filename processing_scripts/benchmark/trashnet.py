
import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path



# Configuration
INPUT_DIR = r"d:\Files\CS\Future_Green\futuregreen_DE\data\raw\trashnet\dataset-resized"
OUTPUT_DIR = r"data\processed\benchmark_dataset\staging" # Changed to staging
RANDOM_SEED = 42

def process_dataset():
    random.seed(RANDOM_SEED)
    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Create staging directory
    os.makedirs(output_path, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Hardcoded classes
    classes = ["cardboard", "glass", "metal", "organics", "paper", "plastic", "rejected", "trash"]
    classes.sort() # Ensure consistent order
    print(f"Processing classes: {classes}")

    stats = {
        "total": 0
    }

    for class_name in classes:
        # Create class subdirectories in staging
        os.makedirs(output_path / class_name, exist_ok=True)

        class_dir = input_path / class_name
        if not class_dir.exists():
            print(f"Info: Input directory for class '{class_name}' does not exist. Created empty output directory.")
            continue
            
        images = list(class_dir.glob("*.jpg"))
        # No need to shuffle for split, but good for processing order randomisation if needed. 
        # Actually randomisation not needed for staging.
        
        print(f"Processing class: {class_name} ({len(images)} images)")
        
        for img_path in images:
            stats["total"] += 1
            
            # Read image
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Save to staging/class/filename
            # We keep original filename or normalize? 
            # Original filename is fine for staging.
            target_file = output_path / class_name / img_path.name
            cv2.imwrite(str(target_file), img)

    print("\nProcessing Complete.")
    print(f"Total images staged: {stats['total']}")
    print(f"Output directory: {output_path.resolve()}")
        


if __name__ == "__main__":
    process_dataset()
