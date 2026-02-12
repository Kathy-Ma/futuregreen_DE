import os
import shutil
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
STAGING_DIR = r"data\processed\benchmark_dataset\staging"
OUTPUT_DIR = r"data\processed\benchmark_dataset" # Final output
PERSON_CLASS_ID = 0 
SPLIT_RATIO = 0.8
RANDOM_SEED = 42

def process_dataset():
    random.seed(RANDOM_SEED)
    cwd = Path(os.getcwd())
    staging_path = cwd / STAGING_DIR
    output_path = cwd / OUTPUT_DIR
    
    if not staging_path.exists():
        print(f"Error: Staging {staging_path} not found.")
        return

    print(f"Processing from: {staging_path}")
    print(f"Output to: {output_path}")

    # Prepare outputs
    for split in ["train", "val"]:
        os.makedirs(output_path / split, exist_ok=True)
        
    model = YOLO("yolov8n.pt")
    
    # Counters
    stats = {"processed": 0, "kept": 0, "rejected": 0}
    class_stats = {}
    rejected_counter = 1
    
    # Iterate staging classes
    # Get all subdirectories in staging
    class_dirs = [d for d in staging_path.iterdir() if d.is_dir()]
    
    # Pool for rejected images (from all sources: YOLO failures + original 'rejected' class)
    rejected_pool = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # If the class is explicitly named "rejected", add all its images to the pool
        if class_name.lower() == "rejected":
            print(f"Pooling 'rejected' class images...")
            images = list(class_dir.glob("*.jpg"))
            rejected_pool.extend(images)
            continue
            
        print(f"Processing {class_name}...")
        class_stats[class_name] = {"total": 0, "kept": 0, "train": 0, "val": 0, "rejected": 0}
        
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        kept_buffer = []
        
        for img_path in images:
            stats["processed"] += 1
            class_stats[class_name]["total"] += 1
            
            # Filter
            try:
                # Agnostic NMS merges overlapping boxes of different classes
                results = model(str(img_path), verbose=False, conf=0.4, agnostic_nms=True)
                obj_count = 0
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if int(box.cls[0]) != PERSON_CLASS_ID:
                            obj_count += 1
                
                if obj_count <= 1:
                    kept_buffer.append(img_path)
                    stats["kept"] += 1
                    class_stats[class_name]["kept"] += 1
                else:
                    # Reject -> Add to pool
                    rejected_pool.append(img_path)
                    stats["rejected"] += 1
                    class_stats[class_name]["rejected"] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Split Kept
        split_idx = int(len(kept_buffer) * SPLIT_RATIO)
        train_imgs = kept_buffer[:split_idx]
        val_imgs = kept_buffer[split_idx:]
        
        # Save Train
        os.makedirs(output_path / "train" / class_name, exist_ok=True)
        for i, img_path in enumerate(train_imgs):
            new_name = f"{class_name}_{i+1}.jpg"
            shutil.copy(str(img_path), str(output_path / "train" / class_name / new_name))
            class_stats[class_name]["train"] += 1
            
        # Save Val
        os.makedirs(output_path / "val" / class_name, exist_ok=True)
        for i, img_path in enumerate(val_imgs):
            new_name = f"{class_name}_{i+1}.jpg"
            shutil.copy(str(img_path), str(output_path / "val" / class_name / new_name))
            class_stats[class_name]["val"] += 1

    # Process Rejected Pool
    print(f"Processing Rejected Pool ({len(rejected_pool)} images)...")
    random.shuffle(rejected_pool)
    
    split_idx = int(len(rejected_pool) * SPLIT_RATIO)
    train_rejected = rejected_pool[:split_idx]
    val_rejected = rejected_pool[split_idx:]
    
    os.makedirs(output_path / "train" / "rejected", exist_ok=True)
    for i, img_path in enumerate(train_rejected):
        new_name = f"rejected_{i+1}.jpg"
        shutil.copy(str(img_path), str(output_path / "train" / "rejected" / new_name))
        
    os.makedirs(output_path / "val" / "rejected", exist_ok=True)
    for i, img_path in enumerate(val_rejected):
        new_name = f"rejected_{i+1}.jpg"
        shutil.copy(str(img_path), str(output_path / "val" / "rejected" / new_name))
        
    print(f"Rejected Class -> Train: {len(train_rejected)}, Val: {len(val_rejected)}")

    # Cleanup Staging
    print("Cleaning up staging directory...")
    try:
        shutil.rmtree(staging_path)
        print("Staging directory deleted.")
    except Exception as e:
        print(f"Error deleting staging: {e}")

    print("\nProcessing Complete.")
    print(f"Total Processed: {stats['processed']}")
    print(f"Total Kept: {stats['kept']}")
    print(f"Total Rejected: {stats['rejected']}")
    
    # Add rejected class to stats for display
    class_stats["rejected"] = {
        "total": len(rejected_pool),
        "kept": 0,
        "rejected": 0, # They are all rejected, but kept in the rejected class
        "train": len(train_rejected),
        "val": len(val_rejected)
    }
    
    print("\nPer-class stats:")
    print(f"{'Class':<15} {'Total':<8} {'Kept':<8} {'Rej':<8} {'Train':<8} {'Val':<8}")
    print("-" * 65)
    for c, s in class_stats.items():
        print(f"{c:<15} {s['total']:<8} {s['kept']:<8} {s['rejected']:<8} {s['train']:<8} {s['val']:<8}")

if __name__ == "__main__":
    process_dataset()
