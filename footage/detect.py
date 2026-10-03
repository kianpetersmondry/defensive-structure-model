"""Player / ball detection on broadcast frames with a COCO YOLOv8n ONNX model.

The model takes a fixed 640x640 input. Broadcast wide-shot players are only
~25-45 px tall at 720p, which is too small once a whole 1280x720 frame is
squeezed into 640 px, so each frame is split into overlapping 640x640 tiles at
native resolution and the detections are merged with NMS.
"""
import numpy as np
import cv2
import onnxruntime as ort

MODEL_PATH = '/home/claude/project_work/footage/models/yolov8n.onnx'
PERSON, BALL = 0, 32
_sess = None


def session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _sess = ort.InferenceSession(MODEL_PATH, so, providers=['CPUExecutionProvider'])
    return _sess


def _tiles(w, h, size=640, overlap=64):
    xs = list(range(0, max(w - size, 0) + 1, size - overlap))
    if xs[-1] + size < w:
        xs.append(w - size)
    ys = list(range(0, max(h - size, 0) + 1, size - overlap))
    if ys[-1] + size < h:
        ys.append(h - size)
    return [(x, y) for y in ys for x in xs]


def _run(img_rgb_640):
    x = img_rgb_640.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = session().run(None, {'images': x})[0][0]  # (84, 8400)
    return out.T  # (8400, 84): cx, cy, w, h, 80 class scores


def detect(frame_bgr, person_conf=0.25, ball_conf=0.15, tile=True):
    """Return (persons, balls): arrays of [x1, y1, x2, y2, score] in frame pixels."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, scores, classes = [], [], []
    offsets = _tiles(w, h) if tile else [(0, 0)]
    for ox, oy in offsets:
        if tile:
            crop = rgb[oy:oy + 640, ox:ox + 640]
            scale = 1.0
        else:
            scale = 640 / max(w, h)
            crop = np.zeros((640, 640, 3), np.uint8)
            rs = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            crop[:rs.shape[0], :rs.shape[1]] = rs
        pred = _run(crop)
        for cls, thr in ((PERSON, person_conf), (BALL, ball_conf)):
            s = pred[:, 4 + cls]
            keep = s > thr
            if not keep.any():
                continue
            p = pred[keep]
            cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
            b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / scale
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            boxes.append(b)
            scores.append(s[keep])
            classes.append(np.full(keep.sum(), cls))
    if not boxes:
        return np.zeros((0, 5)), np.zeros((0, 5))
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes = np.concatenate(classes)
    result = {}
    for cls in (PERSON, BALL):
        m = classes == cls
        if not m.any():
            result[cls] = np.zeros((0, 5))
            continue
        b, s = boxes[m], scores[m]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), 0.0, 0.45)
        idx = np.array(idx).reshape(-1)
        result[cls] = np.concatenate([b[idx], s[idx, None]], 1)
    return result[PERSON], result[BALL]


if __name__ == '__main__':
    import sys, time
    for path in sys.argv[1:]:
        f = cv2.imread(path)
        t = time.time()
        p, b = detect(f)
        dt = time.time() - t
        print(f'{path}: {len(p)} persons, {len(b)} balls, {dt:.2f}s')
        vis = f.copy()
        for x1, y1, x2, y2, s in p:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
        for x1, y1, x2, y2, s in b:
            cv2.rectangle(vis, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), (255, 0, 255), 2)
        cv2.imwrite(path.replace('.png', '_det.jpg'), vis)
