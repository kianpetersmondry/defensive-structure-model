"""Scan the full match for 5-minute windows with much better ball-tracking
coverage than the current one (68%, with gaps up to 23s), so we can pick a
cleaner window before considering a much bigger rebuild."""
import bz2
import json
from config import TRACKING_PATH

FPS = 29.97
WINDOW_S = 300  # 5 minutes

# collect (period, frame, has_ball) for every frame in the match
records = []
with bz2.open(TRACKING_PATH, 'rt') as f:
    for line in f:
        d = json.loads(line)
        period = d['period']
        frame = d['frameNum']
        pet = d.get('periodElapsedTime')
        ball = d.get('ballsSmoothed')
        if isinstance(ball, list):
            ball = ball[0] if ball else None
        if ball is None or ball.get('x') is None:
            raw_ball = d.get('balls') or []
            b0 = raw_ball[0] if raw_ball else None
            ball = b0 if (b0 is not None and b0.get('x') is not None) else None
        records.append((period, frame, pet, ball is not None))

print(f"Total frames: {len(records)}")
by_period = {}
for r in records:
    by_period.setdefault(r[0], []).append(r)

for period, recs in sorted(by_period.items()):
    recs.sort(key=lambda r: r[1])
    print(f"Period {period}: {len(recs)} frames, "
          f"pet range {recs[0][2]}-{recs[-1][2]}")

import pickle
with open('/tmp/ball_coverage_records.pkl', 'wb') as f:
    pickle.dump(by_period, f)
