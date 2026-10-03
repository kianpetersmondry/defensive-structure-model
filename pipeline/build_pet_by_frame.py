"""
One-pass scan of the raw tracking file to build an accurate frame -> periodElapsedTime
mapping (the true match clock), since match_features_with_pressure.pkl's `video_t`
column is only frame_number / fps (see tracking_features.py), not the authoritative
broadcast clock. This mirrors the approach documented in full_match_export.py /
the project's methodology doc.

Match-aware as of round 33 (previously hardcoded to game 10508 only): reads
TRACKING_PATH from config.py and writes to json_path('pet_by_frame'), so
MATCH_ID=10515 produces pet_by_frame_10515.json alongside the original
pet_by_frame.json for game 10508, exactly like every other pipeline stage.

Writes pet_by_frame{suffix}.json: {frameNum: periodElapsedTime}
"""
import bz2
import json
import time
from config import TRACKING_PATH, json_path

OUT_PATH = json_path('pet_by_frame')

t0 = time.time()
pet_by_frame = {}
period_by_frame = {}
n = 0
missing_pet = 0

with bz2.open(TRACKING_PATH, 'rt') as f:
    for line in f:
        d = json.loads(line)
        frame = d['frameNum']
        period = d['period']
        pet = d.get('periodElapsedTime')
        if pet is None:
            missing_pet += 1
        pet_by_frame[frame] = pet
        period_by_frame[frame] = period
        n += 1

print(f'scanned {n} lines in {time.time()-t0:.1f}s, missing_pet={missing_pet}')

with open(OUT_PATH, 'w') as f:
    json.dump({'pet_by_frame': pet_by_frame, 'period_by_frame': period_by_frame}, f)

print('wrote', OUT_PATH)
