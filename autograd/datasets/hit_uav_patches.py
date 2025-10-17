from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


def read_image(path: str) -> np.array:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255


def resize_28x28(crop: np.ndarray) -> np.ndarray:
    img = Image.fromarray((crop * 255.0).astype(np.uint8))
    img = img.resize((28, 28), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def read_yolo_labels(txt_path: str):
    labs = []
    with open(txt_path) as f:
        for line in f:
            c, cx, cy, w, h = map(float, line.strip().split())
            labs.append((int(c), cx, cy, w, h))
    return labs


def yolo_to_xyxy(lbl, W, H):
    _, cx, cy, w, h = lbl
    x = cx * W
    y = cy * H
    bw = w * W
    bh = h * H
    x1 = int(round(x - bw / 2))
    y1 = int(round(y - bh / 2))
    x2 = int(round(x + bw / 2))
    y2 = int(round(y + bh / 2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)
    return x1, y1, x2, y2


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)

    union = max(1, area_a + area_b - inter)

    return inter / union


def sample_background_boxes(
    W: int,
    H: int,
    gt_boxes: List[Tuple[int, int, int, int]],
    n: int,
    rng: np.random.RandomState,
    min_iou: float = 0.1,
    box_side: int = 28,
) -> List[Tuple[int, int, int, int]]:
    boxes = []
    tries = 0
    while len(boxes) < n and tries < n * 50:
        tries += 1
        x1 = int(rng.randint(0, max(1, W - box_side)))
        y1 = int(rng.randint(0, max(1, H - box_side)))
        x2 = min(W - 1, x1 + box_side - 1)
        y2 = min(H - 1, y1 + box_side - 1)
        b = (x1, y1, x2, y2)
        if all(iou(b, g) < min_iou for g in gt_boxes):
            boxes.append(b)
    return boxes


def build_patches(
    root: str,
    split: str,
    n_neg_per_pos: int = 1,
    max_pos_per_image: int = 2,
    rng_seed: int = 0,
    max_images: int | None = None,
    max_total: int | None = None,
):
    """
    Build Person-vs-Background patches from YOLO labels.
    - root/images/{split}/*.jpg|png
    - root/labels/{split}/*.txt (YOLO: cls cx cy w h), keep cls==0 (Person)
    Returns:
      X: (N, 28, 28) float32 in [0,1]
      Y: (N, 2) one-hot [bg, person]
    """
    rng = np.random.RandomState(rng_seed)
    img_dir = Path(root) / "images" / split
    lbl_dir = Path(root) / "labels" / split
    X, Y = [], []
    img_count = 0

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if max_images is not None and img_count >= max_images:
            break

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = read_image(str(img_path))
        H, W = img.shape
        labels = read_yolo_labels(str(lbl_path))

        gt_person = [yolo_to_xyxy(l, W, H) for l in labels if l[0] == 0]
        if len(gt_person) == 0:
            continue

        gt_perm = gt_person.copy()
        rng.shuffle(gt_perm)
        pos_boxes = gt_perm[:max_pos_per_image]

        # Positives (capped)
        for x1, y1, x2, y2 in pos_boxes:
            crop = img[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size == 0:
                continue
            X.append(resize_28x28(crop))
            Y.append([0.0, 1.0])

        # Negatives scaled to used positives
        n_neg = len(pos_boxes) * n_neg_per_pos
        if n_neg > 0:
            neg_boxes = sample_background_boxes(W, H, gt_person, n=n_neg, rng=rng)
            for x1, y1, x2, y2 in neg_boxes:
                crop = img[y1 : y2 + 1, x1 : x2 + 1]
                if crop.size == 0:
                    continue
                X.append(resize_28x28(crop))
                Y.append([1.0, 0.0])

        img_count += 1
        if max_total is not None and len(X) >= max_total:
            break

    if not X:
        return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(
            (0, 2), dtype=np.float32
        )
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)

