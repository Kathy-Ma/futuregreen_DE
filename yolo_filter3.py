import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


SOURCE_DIR = Path(r"TEST/paper_tt")
TARGET_DIR = Path(r"TEST/rejected/multiple")

YOLO_MODEL = "yolov8n-waste-12cls-best.pt"
IOU = 0.50
AGNOSTIC    = False

FILTER_CLASSES = ["paper"]   

CLASS_THRESHOLDS = {
    "cardboard": 0.25,
    "metal": 0.32,
    "paper": 0.25,
    "plastic": 0.30,
    "biological": 0.35,
    "trash": 0.20,
    "battery": 0.20,
    "clothes": 0.20,
    "shoes": 0.20,
    "brown-glass": 0.25,
    "green-glass": 0.25,
    "white-glass": 0.25
}

#MIN_OBJECTS_FOR_MULTIPLE = 2

IMAGE_EXTS = {".jpg"}

COPY_INSTEAD_OF_MOVE = False
CLEAR_TARGET_BEFORE_RUN = False

ENABLE_CROP = True
CROP_INPLACE = False
CROP_TARGET_DIR = Path(r"cropped/paper")
CROP_PADDING = 10
CLEAR_CROP_TARGET_BEFORE_RUN = False

def get_single_target_box(result, model_names: dict, test_classes: list, class_thresholds: dict):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    cls_ids = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    xyxys = result.boxes.xyxy.tolist()

    allowed_classes = {c.lower() for c in test_classes}
    matched_boxes = []

    for cls_id, conf, xyxy in zip(cls_ids, confs, xyxys):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()

        if cls_name in allowed_classes:
            threshold = class_thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                matched_boxes.append(xyxy)

    if len(matched_boxes) == 1:
        return matched_boxes[0]

    return None

def crop_and_save_image(img_path: Path, xyxy, save_path: Path, padding: int = 0):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = map(int, xyxy)

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(save_path)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def get_unique_target_path(target_dir: Path, original_name: str) -> Path:
    target_path = target_dir / original_name
    if not target_path.exists():
        return target_path

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    counter = 1

    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = target_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def count_target_objects(result, model_names: dict, test_classes: list, class_thresholds: dict) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    count = 0
    cls_ids = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()

    allowed_classes = {c.lower() for c in test_classes}

    for cls_id, conf in zip(cls_ids, confs):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()

        if cls_name in allowed_classes:
            threshold = class_thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                count += 1

    return count


def main():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source folder not found: {SOURCE_DIR}")
        return

    if CLEAR_TARGET_BEFORE_RUN and TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if ENABLE_CROP and not CROP_INPLACE:
        if CLEAR_CROP_TARGET_BEFORE_RUN and CROP_TARGET_DIR.exists():
            shutil.rmtree(CROP_TARGET_DIR)
        CROP_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = YOLO(YOLO_MODEL)

    print("\nModel task:", getattr(model, "task", "unknown"))
    print("Model names:", model.names)
    print(f"[INFO] Testing class: {FILTER_CLASSES}")
    print(f"[INFO] Class thresholds\n")

    image_paths = [p for p in SOURCE_DIR.iterdir() if is_image_file(p)]

    if not image_paths:
        print("[INFO] No image files found.")
        return

    moved_count = 0
    kept_count = 0
    error_count = 0

    for img_path in image_paths:
        try:
            results = model(
                str(img_path),
                iou=IOU,
                agnostic_nms=AGNOSTIC,
                verbose=False
            )

            result = results[0]
            object_count = count_target_objects(
                result,
                model.names,
                FILTER_CLASSES,
                CLASS_THRESHOLDS
            )

            if object_count !=1:
                target_path = get_unique_target_path(TARGET_DIR, img_path.name)

                if COPY_INSTEAD_OF_MOVE:
                    shutil.copy2(str(img_path), str(target_path))
                    print(f"[COPY] {img_path.name} | {FILTER_CLASSES} count={object_count}")
                else:
                    shutil.move(str(img_path), str(target_path))
                    print(f"[MOVE] {img_path.name} | {FILTER_CLASSES} count={object_count}")

                moved_count += 1
            else:
                kept_count += 1
                print(f"[KEEP] {img_path.name} | {FILTER_CLASSES} count={object_count}")

                if ENABLE_CROP:
                    single_box = get_single_target_box(
                        result,
                        model.names,
                        FILTER_CLASSES,
                        CLASS_THRESHOLDS
        )

                    if single_box is not None:
                        if CROP_INPLACE:
                            save_path = img_path
                        else:
                            save_path = get_unique_target_path(CROP_TARGET_DIR, img_path.name)

                        crop_and_save_image(
                            img_path=img_path,
                            xyxy=single_box,
                            save_path=save_path,
                            padding=CROP_PADDING
            )

                        print(f"[CROP] {img_path.name} -> {save_path}")
                    else:
                        print(f"[WARN] {img_path.name}: count=1 but no valid single box found")

        except Exception as e:
            error_count += 1
            print(f"[ERROR] {img_path.name}: {e}")

    print("\nDone.")
    print(f"Moved/Copied to multiple: {moved_count}")
    print(f"Kept: {kept_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()