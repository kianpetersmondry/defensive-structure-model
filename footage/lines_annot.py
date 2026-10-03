"""Detect straight pitch lines in a frame, merge collinear Hough segments, and
number them on an image so they can be matched to pitch markings."""
import cv2, numpy as np, sys
from fit_camera import OV
from calib import line_mask
from detect import detect


def detect_lines(f, min_len=40):
    p, b = detect(f)
    m, g = line_mask(f, p, OV)
    segs = cv2.HoughLinesP(m, 1, np.pi / 720, 40, minLineLength=min_len, maxLineGap=25)
    segs = segs[:, 0, :].astype(float) if segs is not None else np.zeros((0, 4))

    def params(s):
        x1, y1, x2, y2 = s
        a = np.arctan2(y2 - y1, x2 - x1) % np.pi
        n = np.array([-np.sin(a), np.cos(a)])
        return a, n @ np.array([x1, y1])

    used = np.zeros(len(segs), bool)
    lines = []
    order = np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    for i in order:
        if used[i]:
            continue
        a, rho = params(segs[i])
        n = np.array([-np.sin(a), np.cos(a)])
        grp = []
        for j in range(len(segs)):
            if used[j]:
                continue
            a2, _ = params(segs[j])
            da = min(abs(a - a2), np.pi - abs(a - a2))
            d1, d2 = abs(n @ segs[j, :2] - rho), abs(n @ segs[j, 2:] - rho)
            if da < np.radians(3) and max(d1, d2) < 6:
                grp.append(j)
        used[grp] = True
        pts = np.concatenate([segs[grp, :2], segs[grp, 2:]])
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        d = vt[0]
        proj = (pts - c) @ d
        lines.append((c, d, proj.min(), proj.max(), len(grp)))
    return lines, m


if __name__ == '__main__':
    t = int(sys.argv[1])
    f = cv2.imread(f'frames/f_{t}.png')
    lines, m = detect_lines(f)
    vis = f.copy()
    for k, (c, d, lo, hi, n) in enumerate(lines):
        p1 = (c + d * lo).astype(int)
        p2 = (c + d * hi).astype(int)
        cv2.line(vis, tuple(p1), tuple(p2), (255, 0, 255), 2)
        mid = (p1 + p2) // 2
        cv2.putText(vis, str(k), tuple(mid + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        print(k, 'c', c.round(1), 'd', d.round(3), 'len', round(hi - lo), 'n', n)
    cv2.imwrite(f'frames/lines_{t}.jpg', vis)
    np.save(f'frames/lines_{t}.npy', np.array([np.concatenate([c, d, [lo, hi]]) for c, d, lo, hi, _ in lines]))
