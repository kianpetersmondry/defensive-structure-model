"""Calibrate the camera from hand-labelled line intersections in one frame.

Line ids come from lines_annot.py for that frame; each label maps a detected
image line to a known pitch line. Every pair of labelled lines whose pitch
lines intersect gives a pitch<->pixel correspondence; a homography is fitted
to those, then decomposed (square pixels, centred principal point) into
focal length, rotation and camera centre.
"""
import numpy as np, cv2, sys, json
from calib import HL, HW, CX0, CY0, rotation, project, draw_model

PA_W, GA_W = 40.32 / 2, 18.32 / 2
# pitch lines as (kind, value): kind 'x' = line X=value (parallel to goal line),
# kind 'y' = line Y=value (parallel to touchline)
PITCH_LINES = {
    'goal_left': ('x', -HL), 'goal_right': ('x', HL), 'halfway': ('x', 0.0),
    'pa_left_front': ('x', -HL + 16.5), 'pa_right_front': ('x', HL - 16.5),
    'ga_left_front': ('x', -HL + 5.5), 'ga_right_front': ('x', HL - 5.5),
    'touch_near': ('y', -HW), 'touch_far': ('y', HW),
    'pa_near': ('y', -PA_W), 'pa_far': ('y', PA_W),
    'ga_near': ('y', -GA_W), 'ga_far': ('y', GA_W),
}


def intersect(l1, l2):
    c1, d1 = l1[:2], l1[2:4]
    c2, d2 = l2[:2], l2[2:4]
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, c2 - c1)
    return c1 + s[0] * d1


def correspondences(lines, labels, max_extrap=None):
    img, world = [], []
    items = list(labels.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (k1, n1), (k2, n2) = items[i], items[j]
            a, b = PITCH_LINES[n1], PITCH_LINES[n2]
            if a[0] == b[0]:
                continue
            X = a[1] if a[0] == 'x' else b[1]
            Y = a[1] if a[0] == 'y' else b[1]
            # box side lines only exist between goal line and box front
            p = intersect(lines[k1], lines[k2])
            img.append(p)
            world.append((X, Y))
    return np.array(img), np.array(world)


def decompose(H):
    """H maps pitch (X,Y,1) -> pixels. Returns C, (pan, tilt, roll, f)."""
    T = np.array([[1, 0, -CX0], [0, 1, -CY0], [0, 0, 1.0]])
    Hc = T @ H
    h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]
    f2a = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    f2b = -((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / (h1[2] ** 2 - h2[2] ** 2)
    ests = [x for x in (f2a, f2b) if x > 0]
    f = np.sqrt(np.mean(ests))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1 = Kinv @ h1
    lam = 1 / np.linalg.norm(r1)
    r1 = r1 * lam
    r2 = Kinv @ h2 * lam
    t = Kinv @ h3 * lam
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)       # columns = world axes in camera frame
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    if C[2] < 0:                        # choose the solution above the pitch
        R[:, :2] *= -1
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
        t = -t
        C = -R.T @ t
    # rows of R (camera axes in world coords): right, down, forward
    right, down, fwd = R[0], R[1], R[2]
    pan = np.arctan2(fwd[0], fwd[1])
    tilt = -np.arcsin(np.clip(fwd[2], -1, 1))
    r0 = rotation(pan, tilt, 0.0)
    roll = np.arctan2(right @ r0[1], right @ r0[0])
    return C, np.array([pan, tilt, roll, f])


if __name__ == '__main__':
    t = int(sys.argv[1])
    labels = {int(k): v for k, v in json.loads(sys.argv[2]).items()}
    lines = np.load(f'frames/lines_{t}.npy')
    img, world = correspondences(lines, labels)
    H, inl = cv2.findHomography(world, img, 0)
    C, p = decompose(H)
    uv, _ = project(np.c_[world, np.zeros(len(world))], C, *p)
    err = np.linalg.norm(uv - img, axis=1)
    print('points', len(img), 'reproj err px: mean', err.mean().round(2), 'max', err.max().round(2))
    print('C', C.round(2), 'pan/tilt/roll deg', np.degrees(p[:3]).round(2), 'f', round(p[3]))
    f = cv2.imread(f'frames/f_{t}.png')
    cv2.imwrite(f'frames/solved_{t}.jpg', draw_model(f, C, p, (0, 255, 255)))
    np.save(f'frames/solved_{t}.npy', np.concatenate([C, p]))
