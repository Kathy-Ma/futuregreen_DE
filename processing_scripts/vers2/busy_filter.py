import os
import shutil
import cv2
import numpy as np

SOURCE_DIR = os.path.join("data","processed","organic","organic1")

BUSY_DIR = os.path.join("data", "processed","rejected", "busy")

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

for f in sorted(os.listdir(SOURCE_DIR)):

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

print("\nDone.")
print("busy:", busy_count)