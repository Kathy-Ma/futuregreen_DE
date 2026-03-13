import os
import shutil
import cv2
import numpy as np
import re

SOURCE_DIR = os.path.join("data", "processed", "dataset_vers2", "organic")

BUSY_DIR = os.path.join("data", "processed", "dataset_vers2", "rejected", "busy")

os.makedirs(BUSY_DIR, exist_ok=True)

EDGE_RATIO_TH = 0.08

exts = (".jpg")

def edge_ratio(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    edges = cv2.Canny(g, 60, 160)
    return (edges > 0).mean()

busy_count = 0
accept_count = 0

def get_num(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

for f in sorted(os.listdir(SOURCE_DIR), key=get_num):

    if not f.lower().endswith(exts):
        continue

    path = os.path.join(SOURCE_DIR, f)
    img = cv2.imread(path)

    if img is None:
        continue

    r = edge_ratio(img)

    if r > EDGE_RATIO_TH:
        shutil.move(path, os.path.join(BUSY_DIR, f))
        busy_count += 1
        print("[BUSY ]", f, "edge_ratio=", round(r,4))
    else:
        accept_count += 1
        class_name = os.path.basename(SOURCE_DIR)
        new_name = f"{class_name}_{accept_count}.jpg"
        if f != new_name:
            shutil.move(path, os.path.join(SOURCE_DIR, new_name))

print("\nDone.")
print("busy:", busy_count)
print("accepted:", accept_count)