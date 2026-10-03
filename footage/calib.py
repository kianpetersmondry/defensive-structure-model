"""Broadcast camera calibration from pitch markings.

Camera model: a single fixed-position broadcast camera that only pans, tilts,
rolls slightly and zooms. The camera centre C=(Cx, Cy, Cz) is shared by every
frame; each frame has its own (pan, tilt, roll, f).

Pitch coordinates (metres): X along the length, -52.5 (left as seen from the
main camera) .. +52.5 (right); Y across the width, -34 (near touchline, camera
side) .. +34 (far touchline); Z up. Origin = centre spot.

Fitting: extract white line pixels on the grass, take their distance
transform, and minimise the (truncated) distance between the projected pitch
model and those pixels.
"""
import numpy as np
import cv2
from scipy.optimize import minimize

L, W = 105.0, 68.0
HL, HW = L / 2, W / 2
IMG_W, IMG_H = 1280, 720
CX0, CY0 = IMG_W / 2, IMG_H / 2


# --------------------------------------------------------------------- model
def _seg(a, b, step=0.5):
    a, b = np.array(a, float), np.array(b, float)
    n = max(int(np.linalg.norm(b - a) / step), 2)
    t = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * t


def _arc(c, r, a0, a1, step=0.5):
    n = max(int(abs(a1 - a0) * r / step), 8)
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def pitch_segments():
    """List of (name, Nx2 array) polylines of every standard marking."""
    segs = []
    segs.append(('touch_near', _seg((-HL, -HW), (HL, -HW))))
    segs.append(('touch_far', _seg((-HL, HW), (HL, HW))))
    segs.append(('goal_left', _seg((-HL, -HW), (-HL, HW))))
    segs.append(('goal_right', _seg((HL, -HW), (HL, HW))))
    segs.append(('halfway', _seg((0, -HW), (0, HW))))
    segs.append(('circle', _arc((0, 0), 9.15, 0, 2 * np.pi)))
    pa_d, pa_w = 16.5, 40.32 / 2
    ga_d, ga_w = 5.5, 18.32 / 2
    for s, name in ((-1, 'left'), (1, 'right')):
        gx = s * HL
        segs.append((f'pa_{name}_front', _seg((gx - s * pa_d, -pa_w), (gx - s * pa_d, pa_w))))
        segs.append((f'pa_{name}_near', _seg((gx, -pa_w), (gx - s * pa_d, -pa_w))))
        segs.append((f'pa_{name}_far', _seg((gx, pa_w), (gx - s * pa_d, pa_w))))
        segs.append((f'ga_{name}_front', _seg((gx - s * ga_d, -ga_w), (gx - s * ga_d, ga_w))))
        segs.append((f'ga_{name}_near', _seg((gx, -ga_w), (gx - s * ga_d, -ga_w))))
        segs.append((f'ga_{name}_far', _seg((gx, ga_w), (gx - s * ga_d, ga_w))))
        # penalty arc: part of r=9.15 circle round the spot (11 m) outside the box
        spot = (gx - s * 11.0, 0.0)
        ang = np.arccos((pa_d - 11.0) / 9.15)
        if s < 0:
            segs.append((f'arc_{name}', _arc(spot, 9.15, -ang, ang)))
        else:
            segs.append((f'arc_{name}', _arc(spot, 9.15, np.pi - ang, np.pi + ang)))
    return segs


SEGS = pitch_segments()
MODEL_PTS = np.concatenate([s for _, s in SEGS])
MODEL_PTS3 = np.concatenate([MODEL_PTS, np.zeros((len(MODEL_PTS), 1))], 1)


# -------------------------------------------------------------------- camera
def rotation(pan, tilt, roll):
    """Rows: camera right, down, forward (world coords). Angles in radians.
    pan=0 looks along +Y (towards the far touchline); tilt>0 looks downward."""
    d = np.array([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)])
    r = np.array([np.cos(pan), -np.sin(pan), 0.0])
    dn = np.cross(d, r)
    cr, sr = np.cos(roll), np.sin(roll)
    r2 = cr * r + sr * dn
    dn2 = -sr * r + cr * dn
    return np.stack([r2, dn2, d])


def project(P, C, pan, tilt, roll, f):
    """World points (N,3) -> pixel (N,2) and depth (N,)."""
    R = rotation(pan, tilt, roll)
    v = (P - C) @ R.T
    z = v[:, 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    u = CX0 + f * v[:, 0] / zs
    w = CY0 + f * v[:, 1] / zs
    return np.stack([u, w], 1), z


def homography(C, pan, tilt, roll, f):
    """3x3 homography mapping pitch (X, Y, 1) -> image pixels."""
    R = rotation(pan, tilt, roll)
    K = np.array([[f, 0, CX0], [0, f, CY0], [0, 0, 1.0]])
    t = -R @ C
    H = K @ np.stack([R[:, 0], R[:, 1], t], 1)
    return H / H[2, 2]


def image_to_pitch(pts, C, params):
    H = homography(C, *params)
    Hi = np.linalg.inv(H)
    p = np.concatenate([np.asarray(pts, float), np.ones((len(pts), 1))], 1) @ Hi.T
    return p[:, :2] / p[:, 2:3]


def pitch_to_image(pts, C, params):
    H = homography(C, *params)
    p = np.concatenate([np.asarray(pts, float), np.ones((len(pts), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:3]


# --------------------------------------------------------------- line mask
def grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (30, 60, 40), (85, 255, 255))
    # fill lines/players inside the grass
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    g = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return g


def line_mask(frame, person_boxes=None, overlay_boxes=()):
    """Thin, bright, low-saturation structures on the grass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    bright = (tophat > 18) & (hsv[:, :, 1] < 110) & (hsv[:, :, 2] > 120)
    g = grass_mask(frame) > 0
    m = (bright & g).astype(np.uint8) * 255
    if person_boxes is not None:
        for x1, y1, x2, y2 in person_boxes[:, :4].astype(int):
            pad = 3
            m[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
    for x1, y1, x2, y2 in overlay_boxes:
        m[y1:y2, x1:x2] = 0
    # drop tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    m = (keep[lab]).astype(np.uint8) * 255
    return m, g


def dist_transform(mask):
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


# ------------------------------------------------------------------ scoring
TRUNC = 12.0


class Obs:
    """Everything the scorer needs from one frame."""
    def __init__(self, mask, grass, ignore=None, max_line_pts=700, seed=0):
        self.dt = dist_transform(mask)
        self.grass = grass.astype(np.uint8)
        # ignore = pixels where a model line may legitimately be invisible
        # (players, graphics overlays); model points there cost nothing
        self.ignore = np.zeros_like(self.grass) if ignore is None else ignore.astype(np.uint8)
        ys, xs = np.nonzero(mask)
        rng = np.random.default_rng(seed)
        if len(xs) > max_line_pts:
            sel = rng.choice(len(xs), max_line_pts, replace=False)
            xs, ys = xs[sel], ys[sel]
        self.line_pts = np.stack([xs, ys], 1).astype(float)
        self.n_line = int(mask.sum() // 255)


def score(params, C, obs, pts3=MODEL_PTS3, need=40, w_back=1.0, trunc=None):
    TR = TRUNC if trunc is None else trunc
    """Symmetric truncated chamfer between projected model and observed lines.

    forward: every model point that lands inside the image must sit on a line
             (cost TR if it lands off the grass, i.e. in the stands);
    backward: every observed line pixel must be near some model point.
    """
    from scipy.spatial import cKDTree
    pan, tilt, roll, f = params
    uv, z = project(pts3, C, pan, tilt, roll, f)
    front = z > 1
    ok = front & (uv[:, 0] >= 1) & (uv[:, 0] < IMG_W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < IMG_H - 1)
    n_in = ok.sum()
    if n_in < need:
        return TR * 2
    ui = uv[ok].astype(int)
    ign = obs.ignore[ui[:, 1], ui[:, 0]] > 0
    on_grass = obs.grass[ui[:, 1], ui[:, 0]] > 0
    d = np.where(on_grass, np.minimum(obs.dt[ui[:, 1], ui[:, 0]], TR), TR)
    d = d[~ign]
    fwd = d.mean() if len(d) else TR
    if len(obs.line_pts) == 0:
        return fwd + TR
    tree = cKDTree(uv[front])
    bd, _ = tree.query(obs.line_pts, distance_upper_bound=TR)
    bwd = np.minimum(bd, TR).mean()
    return fwd + w_back * bwd


def refine(params, C, obs, iters=400, truncs=(40, 20, TRUNC)):
    p = np.array(params, float)
    for tr in truncs:
        res = minimize(lambda q: score(q, C, obs, trunc=tr), p, method='Powell',
                       options={'maxiter': iters, 'xtol': 1e-6, 'ftol': 1e-5})
        p = res.x
    return p, score(p, C, obs)


def _fwd_only(params, C, obs, pts3):
    uv, z = project(pts3, C, *params)
    ok = (z > 1) & (uv[:, 0] >= 1) & (uv[:, 0] < IMG_W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < IMG_H - 1)
    if ok.sum() < 40:
        return TRUNC * 2
    ui = uv[ok].astype(int)
    ign = obs.ignore[ui[:, 1], ui[:, 0]] > 0
    on_grass = obs.grass[ui[:, 1], ui[:, 0]] > 0
    d = np.where(on_grass, np.minimum(obs.dt[ui[:, 1], ui[:, 0]], TRUNC), TRUNC)[~ign]
    return d.mean() if len(d) else TRUNC


def rotations_batch(pan, tilt, roll=None):
    """Vectorised rotation(); pan, tilt arrays of shape (N,). Returns (N,3,3)."""
    n = len(pan)
    roll = np.zeros(n) if roll is None else roll
    d = np.stack([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)], 1)
    r = np.stack([np.cos(pan), -np.sin(pan), np.zeros(n)], 1)
    dn = np.cross(d, r)
    cr, sr = np.cos(roll)[:, None], np.sin(roll)[:, None]
    r2 = cr * r + sr * dn
    dn2 = -sr * r + cr * dn
    return np.stack([r2, dn2, d], 1)


def fwd_batch(P, C, obs, pan, tilt, f, trunc):
    """Forward chamfer for many (pan, tilt, f) at once (roll=0)."""
    R = rotations_batch(pan, tilt)
    v = np.einsum('mk,njk->nmj', P - C, R)          # (N, M, 3)
    z = v[:, :, 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    u = CX0 + f[:, None] * v[:, :, 0] / zs
    w = CY0 + f[:, None] * v[:, :, 1] / zs
    ok = (z > 1) & (u >= 1) & (u < IMG_W - 1) & (w >= 1) & (w < IMG_H - 1)
    ui = np.clip(u, 0, IMG_W - 1).astype(np.int32)
    wi = np.clip(w, 0, IMG_H - 1).astype(np.int32)
    d = np.minimum(obs.dt[wi, ui], trunc)
    d = np.where(obs.grass[wi, ui] > 0, d, trunc)
    use = ok & (obs.ignore[wi, ui] == 0)
    cnt = use.sum(1)
    s = (d * use).sum(1) / np.maximum(cnt, 1)
    s[ok.sum(1) < 40] = trunc * 2
    return s


def global_search(C, obs, f_values=None, pan_range=(-55, 55), tilt_range=(4, 30), px_step=35, top=60, return_all=False):
    """Coarse-to-fine search over (pan, tilt, f); step sizes scale with 1/f so that
    one step always moves the image by ~px_step pixels, whatever the zoom."""
    f_values = np.geomspace(900, 7000, 14) if f_values is None else f_values
    P = MODEL_PTS3[::4]
    obs_c = obs
    cands = []
    for f in f_values:
        step = px_step / f
        pans = np.arange(np.radians(pan_range[0]), np.radians(pan_range[1]), step)
        tilts = np.arange(np.radians(tilt_range[0]), np.radians(tilt_range[1]), step)
        pp, tt = np.meshgrid(pans, tilts)
        pp, tt = pp.ravel(), tt.ravel()
        for i in range(0, len(pp), 1500):
            sl = slice(i, i + 1500)
            s = fwd_batch(P, C, obs_c, pp[sl], tt[sl], np.full(len(pp[sl]), f), trunc=px_step * 1.5)
            order = np.argsort(s)[:top]
            cands += [(s[j], (pp[sl][j], tt[sl][j], 0.0, f)) for j in order]
    cands.sort(key=lambda c: c[0])
    # re-rank the best coarse candidates with the full symmetric score
    ranked = sorted(((score(p, C, obs, trunc=px_step), p) for _, p in cands[:top * 4]), key=lambda c: c[0])
    return ranked[0] if not return_all else ranked


def robust_init(C, obs, k=6, extra=()):
    """Refine the k best (and mutually distinct) coarse candidates plus any
    extra guesses (e.g. the last good camera state); keep the best result."""
    ranked = global_search(C, obs, return_all=True)
    picked = []
    for s, p in ranked:
        if all(abs(p[0] - q[0]) > 0.02 or abs(p[1] - q[1]) > 0.02 or abs(np.log(p[3] / q[3])) > 0.15 for q in picked):
            picked.append(p)
        if len(picked) >= k:
            break
    best = (1e9, None)
    for p in list(picked) + list(extra):
        q, s = refine(p, C, obs, iters=150)
        if s < best[0]:
            best = (s, q)
    return best


def grid_search(C, obs, pans=None, tilts=None, fs=None, top=150):
    pans = np.radians(np.arange(-60, 61, 2)) if pans is None else pans
    tilts = np.radians(np.arange(5, 45, 1.5)) if tilts is None else tilts
    fs = np.array([650, 800, 1000, 1250, 1550, 1950, 2450, 3100, 3900]) if fs is None else fs
    sub = MODEL_PTS3[::3]
    cands = []
    for f in fs:
        for tilt in tilts:
            for pan in pans:
                p = (pan, tilt, 0.0, f)
                cands.append((_fwd_only(p, C, obs, sub), p))
    cands.sort(key=lambda c: c[0])
    best = (1e9, None)
    for _, p in cands[:top]:
        s = score(p, C, obs)
        if s < best[0]:
            best = (s, p)
    return best


def draw_model(frame, C, params, color=(0, 255, 255)):
    out = frame.copy()
    for _, seg in SEGS:
        P = np.concatenate([seg, np.zeros((len(seg), 1))], 1)
        uv, z = project(P, C, *params)
        good = z > 1
        pts = uv.astype(np.int32)
        for i in range(len(pts) - 1):
            if good[i] and good[i + 1]:
                cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, 1, cv2.LINE_AA)
    return out


def inlier_stats(params, C, obs, tol=3.0, pts3=MODEL_PTS3):
    """Sharper fit-quality check than the truncated chamfer:
    fwd = share of visible, unoccluded, on-grass model points within `tol` px of
          a detected line pixel;
    bwd = share of detected line pixels within `tol` px of the projected model."""
    from scipy.spatial import cKDTree
    uv, z = project(pts3, C, *params)
    front = z > 1
    ok = front & (uv[:, 0] >= 1) & (uv[:, 0] < IMG_W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < IMG_H - 1)
    ui = uv[ok].astype(int)
    keep = (obs.ignore[ui[:, 1], ui[:, 0]] == 0) & (obs.grass[ui[:, 1], ui[:, 0]] > 0)
    d = obs.dt[ui[keep, 1], ui[keep, 0]]
    fwd = float((d <= tol).mean()) if len(d) else 0.0
    if len(obs.line_pts) == 0 or front.sum() == 0:
        return fwd, 0.0, int(keep.sum())
    bd, _ = cKDTree(uv[front]).query(obs.line_pts, distance_upper_bound=tol + 1)
    bwd = float((bd <= tol).mean())
    return fwd, bwd, int(keep.sum())
