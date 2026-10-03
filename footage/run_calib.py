"""Pass 2: per-frame camera calibration for every detection-sampled frame.

Tracks the main broadcast camera's (pan, tilt, roll, f) frame to frame from
the pitch markings, re-initialising after cuts, and marks every sampled frame
as tactical-read-OK or not:
  - not live (replay / pre-kickoff)          -> scoreboard-based live window
  - not the main camera / close-up / crowd   -> low grass fraction or the
                                                pitch model won't fit
"""
import cv2, numpy as np, pickle, time, sys
from calib import (Obs, line_mask, score, refine, robust_init, HW, inlier_stats)
import os
from fit_camera import OV

C = np.load('camera_C.npy')
VALID_T = 10.0          # symmetric chamfer score (px) under which a fit counts
MIN_INLIER = 0.5        # share of visible model markings within 3 px of a real line
LIVE_START, LIVE_END = 8.4, 307.7   # video seconds; see scoreboard scan
REINIT_EVERY_S = 1.0


def build_obs(f, persons):
    m, g = line_mask(f, persons, OV)
    ign = np.zeros(m.shape, np.uint8)
    for x1, y1, x2, y2 in persons[:, :4].astype(int):
        ign[max(y1 - 3, 0):y2 + 3, max(x1 - 3, 0):x2 + 3] = 1
    for x1, y1, x2, y2 in OV:
        ign[y1:y2, x1:x2] = 1
    gd = cv2.dilate(g.astype(np.uint8), np.ones((13, 13), np.uint8))
    return Obs(m, gd, ign), float(g.mean()), int(m.sum() // 255)


def frame_hist(f):
    small = cv2.resize(f, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def plausible(p):
    return abs(p[2]) < np.radians(4) and 700 < p[3] < 9000 and np.radians(2) < p[1] < np.radians(35)


def good(p, s, obs):
    if p is None or s > VALID_T or not plausible(p):
        return False, 0.0
    fwd, bwd, n = inlier_stats(p, C, obs)
    return (fwd >= MIN_INLIER and n >= 60), fwd


def main(limit_t=None, resume=None):
    dets = pickle.load(open('detections.pkl', 'rb'))
    by_i = {d['i']: d for d in dets}
    done = {r['i']: r for r in pickle.load(open(resume, 'rb'))} if resume else {}
    cap = cv2.VideoCapture('clip.mp4')
    out = []
    prev_valid = None      # params of the previous sample if it was valid
    prev2_valid = None
    last_valid = None
    last_init_fail_t = -1e9
    prev_hist = None
    i = 0
    t0 = time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        while i > max(by_i) and 'DONE' not in open('detect.log').read():
            time.sleep(20)
            dets = pickle.load(open('detections.pkl', 'rb'))
            by_i = {d['i']: d for d in dets}
        if i not in by_i:
            i += 1
            continue
        d = by_i[i]
        t = d['t']
        if limit_t is not None and t > limit_t:
            break
        if i in done:
            # reuse the earlier result and carry the tracking state forward
            rec = done[i]
            out.append(rec)
            prev_hist = frame_hist(f)
            if rec['valid']:
                prev2_valid, prev_valid, last_valid = prev_valid, np.array(rec['params']), np.array(rec['params'])
            else:
                prev_valid = prev2_valid = None
            i += 1
            continue
        live = LIVE_START <= t <= LIVE_END
        if not live and t > LIVE_END:
            # replays after the goal: no overlay is drawn, so skip the fitting
            out.append({'i': i, 't': t, 'live': False, 'cut': False, 'grass': 0.0, 'n_line': 0,
                        'params': None, 'score': None, 'valid': False, 'how': 'replay'})
            i += 1
            continue
        hist = frame_hist(f)
        cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.6
        prev_hist = hist
        obs, grass_frac, n_line = build_obs(f, d['persons'])
        rec = {'i': i, 't': t, 'live': live, 'cut': cut, 'grass': grass_frac, 'n_line': n_line,
               'params': None, 'score': None, 'valid': False, 'how': ''}
        if cut:
            prev_valid = prev2_valid = None
        if grass_frac < 0.30 or n_line < 150:
            rec['how'] = 'no-pitch'
            prev_valid = prev2_valid = None
        else:
            p, s, how = None, 1e9, ''
            if prev_valid is not None:
                guess = prev_valid if prev2_valid is None else prev_valid + (prev_valid - prev2_valid)
                p, s = refine(guess, C, obs, iters=60, truncs=(25, 12))
                how = 'track'
                if s > VALID_T and prev2_valid is not None:
                    p2, s2 = refine(prev_valid, C, obs, iters=80, truncs=(40, 20, 12))
                    if s2 < s:
                        p, s = p2, s2
            ok_track, _ = good(p, s, obs)
            if not ok_track and (t - last_init_fail_t) >= REINIT_EVERY_S:
                extra = [last_valid] if last_valid is not None else []
                s2, p2 = robust_init(C, obs, k=5, extra=extra)
                how = 'init'
                ok_init, _ = good(p2, s2, obs)
                if ok_init or s2 < s:
                    p, s = p2, s2
                if not ok_init:
                    last_init_fail_t = t
            if p is not None:
                rec['params'], rec['score'], rec['how'] = np.array(p), float(s), how
                v, fwd = good(p, s, obs)
                rec['valid'], rec['inlier'] = bool(v), fwd
            if rec['valid']:
                prev2_valid, prev_valid, last_valid = prev_valid, np.array(p), np.array(p)
            else:
                prev_valid = prev2_valid = None
        out.append(rec)
        if len(out) % 50 == 0:
            nv = sum(r['valid'] for r in out)
            print(f'{len(out)} samples, t={t:.1f}s, valid {nv}, {time.time() - t0:.0f}s elapsed', flush=True)
            pickle.dump(out, open('calib.pkl', 'wb'))
        i += 1
    pickle.dump(out, open('calib.pkl', 'wb'))
    print('DONE', len(out), 'samples,', sum(r['valid'] for r in out), 'valid,', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--resume':
        main(None, sys.argv[2])
    else:
        main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
