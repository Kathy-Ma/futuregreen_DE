from pathlib import Path
import shutil
import cv2

SOURCE_DIR = Path("data/processed/organic_test")
REJECT_DIR = Path("data/processed/rejected/multiple")

IMG_EXTS = {".jpg"}


CANNY_LOW = 50
CANNY_HIGH = 150

MIN_AREA_RATIO = 0.02
MIN_OBJECTS = 2        


def count_objects_canny(img_path: Path) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    h, w = img.shape[:2]
    img_area = h * w
    min_area = img_area * MIN_AREA_RATIO

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = 0
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            valid += 1

    return valid


def safe_move(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
      
        dst = dst_dir / f"{src.stem}__dup{src.suffix}"
    shutil.move(str(src), str(dst))


def main():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source not found: {SOURCE_DIR.resolve()}")
        return

    files = [p for p in SOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()

    moved = 0
    total = len(files)

    for p in files:
        n = count_objects_canny(p)
        if n >= MIN_OBJECTS:
            safe_move(p, REJECT_DIR)
            moved += 1

    print(f"Done. Scanned={total}, moved_to_multiple={moved}, kept={total - moved}")
    print(f"Reject folder: {REJECT_DIR.resolve()}")


if __name__ == "__main__":
    main()