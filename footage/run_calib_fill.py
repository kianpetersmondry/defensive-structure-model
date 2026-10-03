"""Pass 2b: backward tracking to fill calibration gaps.

The forward tracker (run_calib.py) sometimes loses the camera mid-shot and only
re-acquires it a second or two later. Walking backwards from each re-acquired
frame through the gap (no cut in between) recovers most of those frames.
Frames of each gap are buffered JPEG-compressed while the video is read
sequentially (no seeking -- seeking in this file is not frame-accurate).
"""
import cv2, numpy as np, pickle, time
from calib import refine
from run_calib import build_obs, good, C

calib = pickle.load(open('calib.pkl', 'rb'))
dets = {d['i']: d for d in pickle.load(open('detections.pkl', 'rb'))}
by_i = {r['i']: n for n, r in enumerate(calib)}
MAX_BUF = 150

cap = cv2.VideoCapture('clip.mp4')
buf = []          # (sample index n, jpeg bytes) of the current invalid stretch
i = 0
filled = 0
t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i in by_i:
        n = by_i[i]
        r = calib[n]
        if r['cut']:
            buf = []
        if not r['valid']:
            if r['grass'] >= 0.30 and r['n_line'] >= 150:
                buf.append((n, cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]))
                buf = buf[-MAX_BUF:]
            else:
                buf = []          # a close-up breaks the chain
        elif buf:
            # walk backwards from this valid frame through the buffered gap
            prev = np.array(r['params'])
            prev2 = calib[n + 1]['params'] if n + 1 < len(calib) and calib[n + 1]['valid'] else None
            for m, jpg in reversed(buf):
                g = calib[m]
                fr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                obs, _, _ = build_obs(fr, dets[g['i']]['persons'])
                guess = prev if prev2 is None else prev + (prev - np.array(prev2))
                p, s = refine(guess, C, obs, iters=60, truncs=(25, 12))
                v, fwd = good(p, s, obs)
                if not v:
                    p, s = refine(prev, C, obs, iters=80, truncs=(40, 20, 12))
                    v, fwd = good(p, s, obs)
                if not v:
                    break
                g.update({'params': np.array(p), 'score': float(s), 'valid': True, 'inlier': fwd, 'how': 'back'})
                filled += 1
                prev2, prev = prev, np.array(p)
                if g['cut']:
                    break
            buf = []
    i += 1
pickle.dump(calib, open('calib.pkl', 'wb'))
live = [r for r in calib if r['live']]
print(f'DONE filled {filled} samples backwards in {time.time() - t0:.0f}s; live samples valid: '
      f'{sum(r["valid"] for r in live)}/{len(live)} ({100 * np.mean([r["valid"] for r in live]):.1f}%)', flush=True)
