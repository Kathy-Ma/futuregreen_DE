import os
import random
import shutil

SOURCE_DIR = "data/processed/organic/organic1"
TARGET_DIR = "data/processed/organic_test"

os.makedirs(TARGET_DIR, exist_ok=True)

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg"))
]


sample_files = random.sample(files, 2500)


for i, fname in enumerate(sample_files, start=1):
    src = os.path.join(SOURCE_DIR, fname)
    dst = os.path.join(TARGET_DIR, f"organictest_{i}.jpg")
    shutil.copy2(src, dst)

print("Finished:", len(sample_files))