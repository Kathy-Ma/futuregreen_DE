import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np

HAMMING_THRESHOLD = 10  # max bit difference to consider two images "similar"

def compute_phash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    median = np.median(dct_low)
    return (dct_low > median).flatten().astype(np.uint8)

def hash_file(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        return compute_phash(img)
    except Exception:
        return None

# Union-Find for grouping near-duplicates
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def find_and_count_duplicates(folder_path):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

    files = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            files.append(os.path.join(folder_path, filename))

    if not files:
        return 0

    # Parallel hashing
    with Pool(processes=min(cpu_count(), len(files))) as pool:
        hashes = pool.map(hash_file, files)

    # Filter out failed hashes
    valid_hashes = [h for h in hashes if h is not None]
    if len(valid_hashes) < 2:
        return 0

    n = len(valid_hashes)

    # Stack all hashes into a (n, 64) matrix for vectorized comparison
    hash_matrix = np.array(valid_hashes, dtype=np.uint8)  # shape (n, 64)

    uf = UnionFind(n)
    # Vectorized pairwise comparison: for each image i, compare against all j > i
    for i in range(n):
        diffs = np.sum(hash_matrix[i] != hash_matrix[i + 1:], axis=1)
        matches = np.where(diffs <= HAMMING_THRESHOLD)[0]
        for j_offset in matches:
            uf.union(i, i + 1 + j_offset)

    # Count duplicates: for each group of size k, k-1 are duplicates
    groups = defaultdict(int)
    for i in range(n):
        groups[uf.find(i)] += 1

    duplicate_count = sum(size - 1 for size in groups.values() if size > 1)
    return duplicate_count

if __name__ == "__main__":
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset_vers4")
    dataset_root = os.path.normpath(dataset_root)

    if not os.path.exists(dataset_root):
        print(f"Dataset not found: {dataset_root}")
    else:
        print(f"Hamming distance threshold: {HAMMING_THRESHOLD}")
        print(f"Using {cpu_count()} CPU cores for hashing\n")
        total_duplicates = 0
        for class_name in sorted(os.listdir(dataset_root)):
            class_path = os.path.join(dataset_root, class_name)
            if not os.path.isdir(class_path):
                continue
            dups = find_and_count_duplicates(class_path)
            total_duplicates += dups
            print(f"  {class_name}: {dups} near-duplicates")
        print(f"\nTotal near-duplicates found across all classes: {total_duplicates}")