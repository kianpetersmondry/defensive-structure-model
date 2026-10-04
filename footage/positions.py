"""Pass 3: detections + camera calibration -> tracked players and ball in pitch
coordinates, per sampled frame.

Steps per valid sampled frame:
  1. smooth the camera parameters over time within each unbroken run
  2. map each person's foot point (bottom-centre of the box) to the pitch
  3. reject people who can't be players (off the pitch, wrong size for where
     they stand), classify team by shirt colour
  4. track identities within each run (reset after every cut / invalid gap)
  5. smooth each track and differentiate for velocity
  6. pick the ball, track + gap-fill it, and assign possession
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
PIPELINE = _os.path.join(HERE, '..', 'pipeline')
import cv2, numpy as np, pickle
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from calib import homography, project, HL, HW
from fit_camera import OV

C = np.load(_os.path.join(HERE, 'camera_C.npy'))
MAX_GAP_SAMPLES = 3          # bridge up to 3 missing calibrations (~0.3 s) inside a run


# ------------------------------------------------------------ camera series
def camera_runs(calib):
    """Split sampled frames into runs of usable camera parameters; fill short
    gaps by interpolation and smooth each run. Returns {i: params}, {i: run_id}."""
    idx = [k for k, r in enumerate(calib)]
    runs, cur = [], []
    gap = 0
    for k in idx:
        r = calib[k]
        if r['cut'] and cur:
            runs.append(cur); cur = []; gap = 0
        if r['valid'] and r['live']:
            cur.append(k); gap = 0
        else:
            gap += 1
            if gap > MAX_GAP_SAMPLES and cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    params, run_of = {}, {}
    for rid, run in enumerate(runs):
        if len(run) < 5:
            continue
        ks = np.arange(run[0], run[-1] + 1)
        known = np.array([calib[k]['params'] for k in run])
        full = np.stack([np.interp(ks, run, known[:, j]) for j in range(4)], 1)
        if len(ks) >= 7:
            full = savgol_filter(full, 7, 2, axis=0)
        for k, p in zip(ks, full):
            if calib[k]['cut'] and k != run[0]:
                continue
            params[calib[k]['i']] = p
            run_of[calib[k]['i']] = rid
    return params, run_of


# ------------------------------------------------------------- team colour
def classify_team(hsv, box):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    h, w = y2 - y1, x2 - x1
    if h < 14 or w < 4:
        return None
    def med(ya, yb):
        patch = hsv[y1 + int(ya * h):y1 + int(yb * h), x1 + int(0.2 * w):x2 - int(0.2 * w)].reshape(-1, 3)
        green = (patch[:, 0] > 30) & (patch[:, 0] < 85) & (patch[:, 1] > 60)
        patch = patch[~green]
        return np.median(patch, 0) if len(patch) >= 6 else None
    t = med(0.18, 0.50)
    if t is None:
        return None
    hh, s, v = t
    if (hh <= 19 or hh >= 165) and s >= 90 and v >= 80:
        # Lloris is orange head to toe; Morocco wear green shorts. Look at the
        # raw shorts band (green pixels included) -- a Morocco player has plenty
        # of green there, Lloris has almost none and it is orange.
        sb = hsv[y1 + int(0.52 * h):y1 + int(0.70 * h), x1 + int(0.25 * w):x2 - int(0.25 * w)].reshape(-1, 3)
        if len(sb) >= 6 and 9 <= hh <= 22:
            green_share = ((sb[:, 0] > 30) & (sb[:, 0] < 90) & (sb[:, 1] > 50)).mean()
            orange_share = ((sb[:, 0] >= 6) & (sb[:, 0] <= 22) & (sb[:, 1] >= 120) & (sb[:, 2] >= 120)).mean()
            if green_share < 0.12 and orange_share > 0.45:
                return 'FRA_GK'
        return 'MAR'
    if 20 <= hh <= 34 and s >= 70 and v >= 90:
        return 'REF'
    if 95 <= hh <= 116 and s >= 110 and v >= 120:
        return 'MAR_GK'
    if v <= 140 and s <= 150 and (hh >= 95 or s <= 70):
        return 'FRA'
    return None


GROUP = {'FRA': 'FRA', 'FRA_GK': 'FRA_GK', 'MAR': 'MAR', 'MAR_GK': 'MAR_GK', 'REF': 'REF'}


def person_ok(box, X, Y, params):
    """Right place and right size for a person standing at (X, Y)?"""
    if not (-HL - 2 <= X <= HL + 2 and -HW - 2 <= Y <= HW + 2):
        return False
    foot, _ = project(np.array([[X, Y, 0.0]]), C, *params)
    head, _ = project(np.array([[X, Y, 1.8]]), C, *params)
    h_exp = np.linalg.norm(foot - head)
    h = box[3] - box[1]
    if not (0.55 * h_exp <= h <= 1.6 * h_exp):
        return False
    # feet hidden under a broadcast graphic -> foot point unreliable
    fx, fy = (box[0] + box[2]) / 2, box[3]
    for x1, y1, x2, y2 in OV:
        if x1 - 4 <= fx <= x2 + 4 and y1 - 4 <= fy <= y2 + 6:
            return False
    return True


def to_pitch(pts, params):
    H = homography(C, *params)
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.linalg.inv(H).T
    return p[:, :2] / p[:, 2:3]


# ----------------------------------------------------------------- tracker
class Tracker:
    def __init__(self, gate=2.5):
        self.tracks = {}
        self.next_id = 0
        self.gate = gate

    def reset(self):
        self.tracks = {}

    def step(self, t, dets):
        """dets: list of (x, y, group). Returns list of (track_id, x, y, group)."""
        out = []
        for g in set(d[2] for d in dets) | set(tr['g'] for tr in self.tracks.values()):
            tids = [k for k, tr in self.tracks.items() if tr['g'] == g]
            D = [d for d in dets if d[2] == g]
            pred = []
            for k in tids:
                tr = self.tracks[k]
                dt = t - tr['t']
                pred.append(tr['p'] + tr['v'] * min(dt, 0.5))
            matched_d, matched_t = set(), set()
            if tids and D:
                cost = np.array([[np.hypot(*(pr - np.array(d[:2]))) for d in D] for pr in pred])
                r, c = linear_sum_assignment(cost)
                for a, b in zip(r, c):
                    k = tids[a]
                    gate = self.gate + 1.0 * self.tracks[k]['miss']
                    if cost[a, b] <= gate:
                        tr = self.tracks[k]
                        newp = np.array(D[b][:2])
                        dt = max(t - tr['t'], 1e-3)
                        v = (newp - tr['p']) / dt
                        tr['v'] = 0.6 * tr['v'] + 0.4 * np.clip(v, -10, 10)
                        tr['p'], tr['t'], tr['miss'] = newp, t, 0
                        tr['hits'] += 1
                        matched_d.add(b); matched_t.add(k)
                        out.append((k, newp[0], newp[1], g))
            for k in tids:
                if k not in matched_t:
                    self.tracks[k]['miss'] += 1
                    if self.tracks[k]['miss'] > 10:
                        del self.tracks[k]
            for j, d in enumerate(D):
                if j not in matched_d:
                    k = self.next_id; self.next_id += 1
                    self.tracks[k] = {'p': np.array(d[:2]), 'v': np.zeros(2), 't': t, 'miss': 0, 'g': g, 'hits': 1}
                    out.append((k, d[0], d[1], g))
        return out


# ----------------------------------------------------------------- main
def build(calib_path='calib.pkl', det_path='detections.pkl', video='clip.mp4', limit_t=None):
    calib = pickle.load(open(calib_path, 'rb'))
    dets = {d['i']: d for d in pickle.load(open(det_path, 'rb'))}
    cam, run_of = camera_runs(calib)
    cap = cv2.VideoCapture(video)
    tracker = Tracker()
    ball_tracker = Tracker(gate=6.0)
    frames = []
    prev_run = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in dets and dets[i]['t'] > (limit_t or 1e9):
            break
        if i in dets:
            d = dets[i]
            rec = {'i': i, 't': d['t'], 'run': run_of.get(i), 'params': cam.get(i), 'people': [], 'ball': None}
            if i in cam:
                if run_of[i] != prev_run:
                    tracker.reset(); ball_tracker.reset()
                prev_run = run_of[i]
                p = cam[i]
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                boxes = d['persons']
                feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]] if len(boxes) else np.zeros((0, 2))
                P = to_pitch(feet, p) if len(feet) else np.zeros((0, 2))
                obs = []
                for b, (X, Y) in zip(boxes, P):
                    if not person_ok(b, X, Y, p):
                        continue
                    team = classify_team(hsv, b)
                    if team is None:
                        continue
                    obs.append((X, Y, GROUP[team]))
                for k, X, Y, g in tracker.step(d['t'], obs):
                    rec['people'].append({'track': k, 'x': X, 'y': Y, 'team': g})
                # ball: best in-pitch detection
                best = None
                for bx1, by1, bx2, by2, s in d['balls']:
                    Xb, Yb = to_pitch(np.array([[(bx1 + bx2) / 2, by2]]), p)[0]
                    if -HL - 1 <= Xb <= HL + 1 and -HW - 1 <= Yb <= HW + 1 and (best is None or s > best[2]):
                        best = (Xb, Yb, s)
                if best is not None:
                    rec['ball'] = (float(best[0]), float(best[1]))
            else:
                prev_run = None
                tracker.reset(); ball_tracker.reset()
            frames.append(rec)
        i += 1
    return frames


def smooth_tracks(frames):
    """Per-track Savitzky-Golay smoothing of positions + velocity, and the
    majority team vote per track."""
    from collections import defaultdict, Counter
    series = defaultdict(list)
    for n, fr in enumerate(frames):
        for k, pp in enumerate(fr['people']):
            series[pp['track']].append((n, k, fr['t'], pp['x'], pp['y'], pp['team']))
    for tid, s in series.items():
        t = np.array([a[2] for a in s]); x = np.array([a[3] for a in s]); y = np.array([a[4] for a in s])
        team = Counter(a[5] for a in s).most_common(1)[0][0]
        if len(s) >= 7:
            xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
        else:
            xs, ys = x, y
        vx = np.gradient(xs, t) if len(s) >= 3 else np.zeros_like(xs)
        vy = np.gradient(ys, t) if len(s) >= 3 else np.zeros_like(ys)
        vx, vy = np.clip(vx, -10, 10), np.clip(vy, -10, 10)
        for j, (n, k, *_rest) in enumerate(s):
            pp = frames[n]['people'][k]
            pp.update({'x': float(xs[j]), 'y': float(ys[j]), 'vx': float(vx[j]), 'vy': float(vy[j]),
                       'team': team, 'n_obs': len(s)})
    return frames


def fill_ball(frames, max_gap_s=1.0):
    """Linear gap-fill of the ball within a run for gaps up to max_gap_s."""
    known = [(n, fr['t'], fr['ball']) for n, fr in enumerate(frames) if fr['ball'] is not None]
    for (n0, t0, b0), (n1, t1, b1) in zip(known, known[1:]):
        if n1 - n0 < 2 or t1 - t0 > max_gap_s:
            continue
        if frames[n0]['run'] is None or frames[n0]['run'] != frames[n1]['run']:
            continue
        for n in range(n0 + 1, n1):
            a = (frames[n]['t'] - t0) / (t1 - t0)
            frames[n]['ball'] = (b0[0] + a * (b1[0] - b0[0]), b0[1] + a * (b1[1] - b0[1]))
            frames[n]['ball_filled'] = True
    return frames


if __name__ == '__main__':
    import sys
    lim = float(sys.argv[1]) if len(sys.argv) > 1 else None
    fr = build(limit_t=lim)
    fr = smooth_tracks(fr)
    fr = fill_ball(fr)
    pickle.dump(fr, open('positions.pkl', 'wb'))
    usable = [x for x in fr if x['params'] is not None]
    print('sampled', len(fr), 'usable', len(usable))
    if usable:
        from collections import Counter
        cnt = Counter(pp['team'] for x in usable for pp in x['people'])
        print('mean people per usable frame', np.mean([len(x['people']) for x in usable]).round(1), dict(cnt))
        print('ball seen in', round(np.mean([x['ball'] is not None for x in usable]) * 100, 1), '% of usable frames')
