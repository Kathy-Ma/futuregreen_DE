import os
import shutil
import time
from pathlib import Path

# Import processing scripts
import trashnet
import organics
import auto_sorter

def main():
    print("==================================================")
    print("       STARTING BENCHMARK DATASET GENERATION       ")
    print("==================================================")
    start_time = time.time()

    # Define paths
    cwd = Path(os.getcwd())
    dataset_dir = cwd / r"data\processed\benchmark_dataset"

    # Step 0: Clean up existing dataset
    if dataset_dir.exists():
        print(f"\n[Step 0] Cleaning up existing dataset at {dataset_dir}...")
        try:
            shutil.rmtree(dataset_dir)
            print("Dataset directory removed.")
        except Exception as e:
            print(f"Error removing dataset directory: {e}")
            return

    # Step 1: Stage TrashNet
    print("\n[Step 1] Running TrashNet Staging...")
    try:
        trashnet.process_dataset()
    except Exception as e:
        print(f"TrashNet processing failed: {e}")
        return

    # Step 2: Stage Organics
    print("\n[Step 2] Running Organics Staging...")
    try:
        organics.process_organics()
    except Exception as e:
        print(f"Organics processing failed: {e}")
        return

    # Step 3: Auto-Sort (Filter, Split, Rejected Class, Cleanup)
    print("\n[Step 3] Running Auto-Sorter (Filter, Split, Cleanup)...")
    try:
        auto_sorter.process_dataset()
    except Exception as e:
        print(f"Auto-Sorter failed: {e}")
        return

    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f"       PIPELINE COMPLETE in {elapsed:.2f} seconds       ")
    print("==================================================")

if __name__ == "__main__":
    main()
