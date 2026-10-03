"""Pass 1: person/ball detection on every 3rd frame (~10 fps) of the clip."""
import cv2, numpy as np, time, pickle
from detect import detect

cap = cv2.VideoCapture('clip.mp4')
out = []
i = 0
t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok:
        break
    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if i % 3 == 0:
        p, b = detect(f)
        out.append({'i': i, 't': ts, 'persons': p.astype(np.float32), 'balls': b.astype(np.float32)})
        if len(out) % 100 == 0:
            el = time.time() - t0
            print(f'{len(out)} frames, video t={ts:.1f}s, {el:.0f}s elapsed', flush=True)
            pickle.dump(out, open('detections.pkl', 'wb'))
    i += 1
pickle.dump(out, open('detections.pkl', 'wb'))
print('DONE', len(out), 'frames in', round(time.time() - t0), 's', flush=True)
open('detections.done', 'w').write('ok')
