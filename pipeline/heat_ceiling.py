"""The danger heatmap's brightness ceiling for one match, computed from its own chapters.

    MATCH_ID=<id> python3 pipeline/heat_ceiling.py

full_match_heatmap.py maps log1p(mass / LOG_FLOOR) onto 0-255 with one fixed ceiling per match, so the same
danger looks equally bright in every chapter. The ceiling is the 99.5th percentile of the non-zero values
across the match (see config.py's HEATMAP_VMAX_LOG). This samples every chapter at 0.5 Hz (a tenth of the
heatmap's own 5 Hz) to estimate it and writes output/derived/heat_ceiling_<id>.json.

The registry's `heatmap_vmax_log` wins when it is set (the first six matches keep their published values);
config.py falls back to this file, then to the old shared constant.
"""
import json
import os
import time

import numpy as np

from config import CHAPTERS_DIR, MATCH_ID
from heatmap_export import raster_for_frame, LOG_FLOOR
from space_behind_line import build_velocity_lookup
from paths import out

SAMPLE_HZ = 0.5
PCT = 99.5


def main():
    manifest = json.load(open(f'{CHAPTERS_DIR}/manifest.json'))
    vals, t0, n = [], time.time(), 0
    for ch in manifest['chapters']:
        data = json.load(open(ch['animPath']))
        frames, meta = data['frames'], data['meta']
        vel = build_velocity_lookup(frames)
        step = max(1, round(meta['fps'] / SAMPLE_HZ))
        for i in range(0, len(frames), step):
            H = raster_for_frame(frames[i], vel[i], meta['homeDefendsPositiveX'], frame_id=i)
            if H is None:
                continue
            lv = np.log1p(np.clip(H, 0, None) / LOG_FLOOR)
            vals.append(lv[lv > 0].astype(np.float32))
            n += 1
        print(f"chapter {ch['chapterIndex']:02d}: {n} samples so far ({time.time() - t0:.0f}s)")
    ceiling = float(np.percentile(np.concatenate(vals), PCT))
    path = out('derived', f'heat_ceiling_{MATCH_ID}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({'match': MATCH_ID, 'heatmap_vmax_log': round(ceiling, 4), 'samples': n,
               'sample_hz': SAMPLE_HZ, 'percentile': PCT}, open(path, 'w'), indent=1)
    print(f'ceiling {ceiling:.4f} from {n} samples -> {path}')


if __name__ == '__main__':
    main()
