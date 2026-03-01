import os
import re
import shutil
from pathlib import Path
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
SOURCE_DIR  = r"data\processed\consolidated_raws"
TARGET_DIR  = r"data\processed\dataset_vers1"

YOLO_MODEL  = "yolov8n.pt"
CONF        = 0.4
AGNOSTIC    = True
# ──────────────────────────────────────────────────────────────────────────────


def scan_existing_counts(target_path: Path) -> dict[str, int]:
    """
    If a previous run partially populated the target directory, find the
    highest existing index for each class so we don't overwrite anything.
    Returns {classname: next_index_to_use}.
    """
    counts: dict[str, int] = {}
    if not target_path.exists():
        return counts

    for class_dir in target_path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        max_num = 0
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$", re.IGNORECASE)
        for f in class_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        counts[class_name] = max_num  # next index will be max_num + 1
    return counts


def main():
    cwd         = Path(os.getcwd())
    source_path = cwd / SOURCE_DIR
    target_path = cwd / TARGET_DIR
    rejected_path = target_path / "rejected"

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return

    # ── Prepare output directories ──────────────────────────────────────────
    target_path.mkdir(parents=True, exist_ok=True)
    rejected_path.mkdir(parents=True, exist_ok=True)

    # Pre-scan so incremental runs don't collide with existing files
    existing = scan_existing_counts(target_path)

    # next index for every class (accepted images)
    accepted_counters: dict[str, int] = {}

    # rejected counter: find highest existing rejected_N.jpg
    rej_pattern = re.compile(r"^rejected_(\d+)\.jpg$", re.IGNORECASE)
    max_rej = 0
    for f in rejected_path.iterdir():
        m = rej_pattern.match(f.name)
        if m:
            max_rej = max(max_rej, int(m.group(1)))
    rejected_counter = max_rej + 1

    # ── Load YOLO model ──────────────────────────────────────────────────────
    print(f"Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    # ── Collect all source images, grouped by class subdir ──────────────────
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print("[ERROR] No class subdirectories found in source directory.")
        return

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0}
    class_stats: dict[str, dict] = {}

    # ── Process each class ───────────────────────────────────────────────────
    for class_dir in class_dirs:
        class_name = class_dir.name  # e.g. "plastic", "glass"
        images     = sorted(class_dir.glob("*.jpg"))

        if not images:
            print(f"[SKIP] {class_name}: no .jpg images found.")
            continue

        # Initialise per-class counter, accounting for any pre-existing files
        if class_name not in accepted_counters:
            accepted_counters[class_name] = existing.get(class_name, 0) + 1

        class_stats[class_name] = {"total": 0, "accepted": 0, "rejected": 0, "errors": 0}
        class_out_dir = target_path / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── Processing class: {class_name} ({len(images)} images) ──")

        for img_path in images:
            stats["total"]             += 1
            class_stats[class_name]["total"] += 1

            try:
                results  = model(str(img_path), verbose=False,
                                 conf=CONF, agnostic_nms=AGNOSTIC)
                obj_count = sum(len(r.boxes) for r in results)

                if obj_count <= 1:
                    # ── ACCEPT ───────────────────────────────────────────────
                    new_name = f"{class_name}_{accepted_counters[class_name]}.jpg"
                    dest     = class_out_dir / new_name
                    shutil.copy2(str(img_path), str(dest))

                    print(f"  [ACCEPT] {img_path.name} → {new_name}  "
                          f"(objects detected: {obj_count})")

                    accepted_counters[class_name]    += 1
                    stats["accepted"]                += 1
                    class_stats[class_name]["accepted"] += 1

                else:
                    # ── REJECT ───────────────────────────────────────────────
                    new_name = f"rejected_{rejected_counter}.jpg"
                    dest     = rejected_path / new_name
                    shutil.copy2(str(img_path), str(dest))

                    print(f"  [REJECT] {img_path.name} → {new_name}  "
                          f"(objects detected: {obj_count})")

                    rejected_counter                 += 1
                    stats["rejected"]                += 1
                    class_stats[class_name]["rejected"] += 1

            except Exception as exc:
                print(f"  [ERROR]  {img_path.name}: {exc}")
                stats["errors"]                     += 1
                class_stats[class_name]["errors"]   += 1

    # ── Final Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)
    print(f"{'Total processed':<25}: {stats['total']}")
    print(f"{'Total accepted':<25}: {stats['accepted']}")
    print(f"{'Total rejected':<25}: {stats['rejected']}")
    print(f"{'Total errors':<25}: {stats['errors']}")
    print()
    print(f"{'Class':<15} {'Total':<8} {'Accepted':<10} {'Rejected':<10} {'Errors':<8}")
    print("-" * 55)
    for cls, s in class_stats.items():
        print(f"{cls:<15} {s['total']:<8} {s['accepted']:<10} {s['rejected']:<10} {s['errors']:<8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
