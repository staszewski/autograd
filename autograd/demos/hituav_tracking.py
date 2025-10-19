import numpy as np
from pathlib import Path

from autograd.tensor import Tensor
from autograd.models.simple_cnn import SimpleCNN
from autograd.drone_problems.kalman import Kalman2D
from autograd.datasets.hit_uav_patches import read_image, iou
from PIL import Image, ImageDraw


def load_simplecnn(model, path: str):
    d = np.load(path, allow_pickle=True)
    for k, arr in zip(model.kernels, d["kernels"]): k._data = arr
    for w, arr in zip(model.W, d["W"]): w._data = arr
    for b, arr in zip(model.b, d["b"]): b._data = arr

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def nms(boxes, iou_threshold):
    # boxes: (x1,y1,x2,y2,score)
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    output = []
    while boxes:
        b = boxes.pop(0)
        output.append(b)
        bx = (b[0], b[1], b[2], b[3])
        boxes = [box for box in boxes if iou((box[0], box[1], box[2], box[3]), bx) < iou_threshold]
    return output

def sliding_window(image, stride = 12):
    H, W = image.shape
    for y in range(0, max(1, H - 28 + 1), stride):
        for x in range(0, max(1, W - 28 + 1), stride):
            yield x, y, image[y:y+28, x:x+28]

def associate(tracks, zs, gate = 15.0):
    matched, used_tracks, used_z = [], set(), set()
    for ti, tr in enumerate(tracks):
        # use predicted position if available, otherwise current
        if tr['kf'].hat_x_minus is not None:
            px, py = tr['kf'].hat_x_minus[:2]
        elif tr['kf'].hat_x is not None:
            px, py, *_ = tr['kf'].get_state()
        else:
            continue
        best = None
        best_d = gate
        for zi, z in enumerate(zs):
            if zi in used_z:
                continue
            d = float(np.hypot(px - z[0], py - z[1]))
            if d < best_d:
                best, best_d = (ti, zi), d
        
        if best is not None:
            matched.append(best)
            used_tracks.add(best[0])
            used_z.add(best[1])
        
    new_zs = [zs[i] for i in range(len(zs)) if i not in used_z]
    lost_tracks = [tracks[i] for i in range(len(tracks)) if i not in used_tracks]

    return matched, new_zs, lost_tracks

root = str((Path(__file__).resolve().parents[2] / "data" / "hit-uav"))

# Choose a specific short sequence by filename prefix, and which split to read from
# Example prefix: "0_60_30_0_016" will match 0_60_30_0_01611.jpg, 0_60_30_0_01623.jpg, ...
SEQ_SPLIT = "train"  # or "val"
SEQ_PREFIX = "0_60_30_0_016"  # set to "" to disable prefix filtering

seq_dir = Path(root) / "images" / SEQ_SPLIT
if SEQ_PREFIX:
    frames_all = list(seq_dir.glob(f"{SEQ_PREFIX}*.jpg"))
    def frame_key(p):
        tail = p.stem.split("_")[-1]
        try:
            return int(tail)
        except Exception:
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else 0
    frames = sorted(frames_all, key=frame_key)
else:
    frames = sorted(seq_dir.glob("*.jpg"))[:200]

# Detection/tracking parameters
PROB_THRESH = 0.9
STRIDE = 16
NMS_IOU = 0.2
GATE = 15.0
MISS_MAX = 3
MIN_HITS = 2  # only consider/report tracks that have been updated at least this many times

model = SimpleCNN(K=4, num_classes=2)
ckpt = str((Path(__file__).resolve().parents[2] / "data" / "hit-uav" / "hituav_person_simplecnn.npz"))
load_simplecnn(model, ckpt)

tracks = []
next_id = 0

def near_existing(z, tracks, gate=GATE):
    for tr in tracks:
        st = tr['kf'].get_state()
        if st is None:
            continue
        x, y, *_ = st
        if float(np.hypot(x - z[0], y - z[1])) < gate:
            return True
    return False

for frame_path in frames:
    image = read_image(frame_path)

    candidates = []
    for x, y, crop in sliding_window(image, stride = STRIDE):
        probs = softmax(model(Tensor(crop, False)).data[:,0])
        p_person = float(probs[1])

        if p_person > PROB_THRESH:
            candidates.append((x, y, x + 27, y + 27, p_person))

    dets = nms(candidates, iou_threshold = NMS_IOU)

    zs = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2, _ in dets]

    for tr in tracks:
        tr['kf'].predict()

    matched, new_zs, lost_tracks = associate(tracks, zs, gate=GATE)

    for ti, zi in matched:
        tracks[ti]['kf'].update(zs[zi])
        tracks[ti]['miss'] = 0
        tracks[ti]['hits'] = tracks[ti].get('hits', 0) + 1

    matched_ti = {ti for ti, _ in matched}
    for i, tr in enumerate(tracks):
        if i not in matched_ti:
            tr['kf'].commit_prediction()
            tr['miss'] += 1

    matched_zi = {zi for _, zi in matched}
    for idx, z in enumerate(zs):
        if idx in matched_zi:
            continue
        if near_existing(z, tracks, gate=GATE):
            continue
        kf = Kalman2D(dt=1.0, q=0.5, sigma_meas=3.0, x0=(z[0], z[1], 0.0, 0.0))
        tracks.append({'kf': kf, 'id': next_id, 'miss': 0, 'hits': 1}); next_id += 1

    # drop stale
    tracks = [tr for tr in tracks if tr['miss'] <= MISS_MAX]

    # (optional) print summary
    active = [tr for tr in tracks if tr.get('hits', 0) >= MIN_HITS and tr['miss'] == 0]
    print(f"{frame_path.name}: dets={len(dets)}, tracks={len(tracks)}, active={len(active)}")

    # Visualization: draw detections and tracks and save
    # Prepare RGB image
    img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis = Image.fromarray(img_uint8).convert("RGB")
    draw = ImageDraw.Draw(vis)

    # Draw detections (green)
    for (x1, y1, x2, y2, s) in dets:
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        draw.text((x1+2, y1+2), f"{s:.2f}", fill=(0, 255, 0))

    # Draw tracks: predicted/current positions with IDs
    for tr in tracks:
        st = tr['kf'].get_state()
        if st is None:
            continue
        cx, cy, vx, vy = st
        r = 4
        color = (0, 128, 255) if tr.get('hits', 0) >= MIN_HITS and tr['miss'] == 0 else (180, 180, 180)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=2)
        draw.text((cx + 6, cy - 6), f"ID{tr['id']} h{tr.get('hits',0)} m{tr.get('miss',0)}", fill=color)

    out_dir = Path(__file__).resolve().parents[2] / "media" / "output" / "hituav_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis.save(out_dir / frame_path.name)