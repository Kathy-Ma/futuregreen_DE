import os
import re
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
SOURCE_DIR     = Path(r"data\processed\consolidated_raws")
CROP_TARGET_DIR = Path(r"data\processed\dataset_vers4")

SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL = SCRIPT_DIR / "yolov8n-waste-12cls-best.pt"
IOU        = 0.35
AGNOSTIC   = False
DEVICE     = "cpu"

MIN_OBJECTS  = 1
MAX_OBJECTS  = 8
MIN_CROP_SIZE = 120
CROP_PADDING  = 5
IMAGE_EXTS    = {".jpg"}

# When YOLO detects more than 1 total object (any class), this value is added
# to each per-class confidence threshold, requiring higher confidence in busy images.
MULTI_OBJECT_CONF_BOOST = 0.15
NOISE_CONF_FLOOR        = 0.10   # Ignore detections below this for the total count
MIN_AREA_RATIO          = 0.25   # Drop boxes < 25% the area of the largest match

# Maps each dataset class to the YOLO model class names it corresponds to.
# This bridges the gap between the 7 user-defined classes and the 12 model classes.
CLASS_MAPPING = {
    "paper":     ["paper"],
    "cardboard": ["cardboard"],
    "metal":     ["metal"],
    "plastic":   ["plastic"],
    "organic":   ["biological"],
    "trash":     ["trash", "battery", "clothes", "shoes"],
    "glass":     ["brown-glass", "green-glass", "white-glass"],
}

CLASS_THRESHOLDS = {
    "cardboard":   0.25,
    "metal":       0.32,
    "paper":       0.25,
    "plastic":     0.30,
    "biological":  0.35,
    "trash":       0.20,
    "battery":     0.20,
    "clothes":     0.20,
    "shoes":       0.20,
    "brown-glass": 0.25,
    "green-glass": 0.25,
    "white-glass": 0.25,
}
# ──────────────────────────────────────────────────────────────────────────────


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def get_unique_target_path(target_dir: Path, filename: str) -> Path:
    """Return a collision-free save path within target_dir."""
    target_path = target_dir / filename
    if not target_path.exists():
        return target_path

    stem   = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_path = target_dir / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def count_all_detections(result, model_names: dict, conf_floor: float = 0.10) -> int:
    """
    Count ALL bounding boxes YOLO detected (any class) that are above
    conf_floor.  Used to decide whether to apply the confidence boost.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    return sum(1 for conf in result.boxes.conf.tolist() if conf >= conf_floor)


def get_all_target_boxes(
    result,
    model_names: dict,
    allowed_yolo_classes: list,
    class_thresholds: dict,
    conf_boost: float = 0.0,
    min_area_ratio: float = 1.0,
) -> list:
    """
    Return a list of xyxy bounding boxes that:
      - belong to one of `allowed_yolo_classes`
      - meet the per-class confidence threshold (+ optional conf_boost)
      - are at least `min_area_ratio` × the area of the largest matched box
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    allowed = {c.lower() for c in allowed_yolo_classes}
    matched_boxes = []

    for cls_id, conf, xyxy in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()
        if cls_name in allowed:
            threshold = class_thresholds.get(cls_name, 0.25) + conf_boost
            if conf >= threshold:
                matched_boxes.append(xyxy)

    # Drop boxes much smaller than the largest — filters out hands / body parts
    if len(matched_boxes) > 1:
        def _area(b):
            return (b[2] - b[0]) * (b[3] - b[1])
        matched_boxes.sort(key=_area, reverse=True)
        largest_area = _area(matched_boxes[0])
        matched_boxes = [b for b in matched_boxes
                         if _area(b) >= largest_area * min_area_ratio]

    return matched_boxes


def crop_and_save_image(img_path: Path, xyxy, save_path: Path, padding: int = 0) -> bool:
    """Crop img_path to xyxy (with padding, clamped to image bounds) and save.
    Returns True on success, False if the resulting crop is too small."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
        return False

    img.crop((x1, y1, x2, y2)).save(save_path)
    return True


def scan_existing_crop_counts(crop_target: Path) -> dict[str, int]:
    """
    For each class sub-directory already in the target, find the highest
    existing crop index so we can resume without collisions.
    Returns {class_name: highest_index_seen}.
    """
    counts: dict[str, int] = {}
    if not crop_target.exists():
        return counts

    for class_dir in crop_target.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)_obj\d+\.jpg$", re.IGNORECASE)
        max_num = 0
        for f in class_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        counts[class_name] = max_num
    return counts


def main():
    cwd = Path(os.getcwd())
    source_path      = cwd / SOURCE_DIR
    crop_target_path = cwd / CROP_TARGET_DIR

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return

    crop_target_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(str(YOLO_MODEL))
    print("Model names:", model.names)

    # Pre-scan so we can resume without overwriting
    existing = scan_existing_crop_counts(crop_target_path)

    # ── Collect and validate class subdirectories ─────────────────────────────
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print("[ERROR] No class subdirectories found in source directory.")
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total": 0,
        "crops": 0,
        "skipped": 0,
        "boosted": 0,
        "errors": 0,
    }
    class_stats: dict[str, dict] = {}

    # ── Process each class ────────────────────────────────────────────────────
    for class_dir in class_dirs:
        class_name = class_dir.name

        # Skip folders that are not in our class mapping (e.g. 'rejected', hidden dirs)
        if class_name not in CLASS_MAPPING:
            print(f"\n[SKIP DIR] '{class_name}' is not a known dataset class — ignoring.")
            continue

        allowed_yolo_classes = CLASS_MAPPING[class_name]
        class_out_dir = crop_target_path / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([p for p in class_dir.iterdir() if is_image_file(p)])
        if not images:
            print(f"\n[SKIP DIR] {class_name}: no .jpg images found.")
            continue

        # Image counter (for output naming), resuming from any existing files
        img_counter = existing.get(class_name, 0) + 1

        class_stats[class_name] = {
            "total": 0,
            "crops": 0,
            "skipped": 0,
            "boosted": 0,
            "errors": 0,
        }

        print(f"\n── Processing class: '{class_name}' → YOLO classes: {allowed_yolo_classes} ({len(images)} images) ──")

        for img_path in images:
            stats["total"] += 1
            class_stats[class_name]["total"] += 1
            try:
                results = model(
                    str(img_path),
                    verbose=False,
                    device=DEVICE,
                    iou=IOU,
                    agnostic_nms=AGNOSTIC,
                )
                result = results[0]

                # Count ALL detections (any class) to decide on conf boost
                total_detections = count_all_detections(
                    result, model.names, NOISE_CONF_FLOOR
                )
                boost = MULTI_OBJECT_CONF_BOOST if total_detections > 1 else 0.0

                # Only consider boxes that match THIS class's YOLO equivalents
                boxes = get_all_target_boxes(
                    result,
                    model.names,
                    allowed_yolo_classes,
                    CLASS_THRESHOLDS,
                    conf_boost=boost,
                    min_area_ratio=MIN_AREA_RATIO,
                )

                object_count = len(boxes)

                if not (MIN_OBJECTS <= object_count <= MAX_OBJECTS):
                    print(f"  [SKIP] {img_path.name} | matched_objects={object_count} (need {MIN_OBJECTS}–{MAX_OBJECTS})")
                    stats["skipped"] += 1
                    class_stats[class_name]["skipped"] += 1
                    continue

                saved_this_image = 0

                for i, box in enumerate(boxes, start=1):
                    save_name = f"{class_name}_{img_counter}_obj{i}.jpg"
                    save_path = get_unique_target_path(class_out_dir, save_name)

                    success = crop_and_save_image(
                        img_path=img_path,
                        xyxy=box,
                        save_path=save_path,
                        padding=CROP_PADDING,
                    )

                    if success:
                        saved_this_image += 1
                        stats["crops"] += 1
                        class_stats[class_name]["crops"] += 1
                        if boost > 0:
                            stats["boosted"] += 1
                            class_stats[class_name]["boosted"] += 1
                        boost_tag = " [boosted]" if boost > 0 else ""
                        print(f"  [CROP] {img_path.name} → {save_name}{boost_tag}")
                    else:
                        print(f"  [SKIP CROP] {img_path.name} → obj{i} smaller than {MIN_CROP_SIZE}px")

                img_counter += 1
                print(f"  [DONE] {img_path.name} | objects={object_count} | saved={saved_this_image}")

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                stats["errors"] += 1
                class_stats[class_name]["errors"] += 1

        cs = class_stats[class_name]
        print(f"  → Class '{class_name}' summary: crops={cs['crops']}, skipped={cs['skipped']}, boosted={cs['boosted']}, errors={cs['errors']}")

    # ── Final Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"{'Total images processed':<25}: {stats['total']}")
    print(f"{'Total crops saved':<25}: {stats['crops']}")
    print(f"{'Skipped (count OOB)':<25}: {stats['skipped']}")
    print(f"{'Crops w/ boosted conf.':<25}: {stats['boosted']}")
    print(f"{'Errors':<25}: {stats['errors']}")
    print()
    print(f"{'Class':<15} {'Total':<8} {'Crops':<8} {'Skipped':<10} {'Boosted':<10} {'Errors':<8}")
    print("-" * 80)
    for cls, s in class_stats.items():
        print(f"{cls:<15} {s['total']:<8} {s['crops']:<8} {s['skipped']:<10} {s['boosted']:<10} {s['errors']:<8}")
    print("=" * 80)


if __name__ == "__main__":
    main()