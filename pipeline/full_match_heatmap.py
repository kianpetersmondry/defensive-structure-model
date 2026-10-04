"""
Runs the danger-heatmap rasterization (see heatmap_export.py's module
docstring for the method itself -- unchanged here) over every chapter
produced by full_match_export.py, instead of the single curated window.

One deliberate difference from heatmap_export.py's per-window normalization:
here the log-scale ceiling (VMAX_LOG_FIXED) is a FIXED constant, held
constant across every chapter of ONE match, rather than recomputed per
chapter from that chapter's own 99.5th percentile. Reasoning: each chapter
is a separate page a viewer can move between, so if brightness were
independently re-normalized per chapter, the same underlying danger level
could look meaningfully brighter in one chapter than another for no
tactical reason -- purely an artifact of that chapter happening to have a
slightly different value distribution. A single fixed floor+ceiling keeps
"how bright is bright" meaning the same picture-to-picture across the whole
match.

**Round 37 fix: the ceiling is now per-match (`config.HEATMAP_VMAX_LOG`),
not one number shared by every match.** Through round 36 this used a single
literal constant (4.375829836329568) carried over unmodified from the very
first round-14 fix, which calibrated it against match 1's small hand-picked
5-minute highlight window only -- and that same number was then reused for
every match's full ~130-minute chapter export, including match 1's own.
The user flagged specific chapters across the recent matches looking
"messy"; direct inspection confirmed the pre-round-14 symptom (a hard-edged
plateau instead of a graded glow) was back in a meaningful fraction of
frames, worst in match 5 (up to ~27% of the pitch grid pinned to max
brightness in its hottest frames). Root cause: every match's real
full-match danger-density distribution runs hotter than that one curated
window did -- confirmed by recomputing each match's own 99.5th percentile
of log1p(mass/LOG_FLOOR) across a representative sample of its own
chapters: 5.19 (match 1) / 5.40 (match 2) / 5.64 (match 3) / 5.17 (match 4)
/ 6.21 (match 5), all higher than the shared 4.38 constant, worst for match
5. Fixed by computing that same 99.5th-percentile ceiling per match (see
`config.py`'s `HEATMAP_VMAX_LOG` and its own comment) instead of reusing
match 1's curated-window number everywhere -- still one fixed number per
match, preserving the original chapter-to-chapter consistency goal above,
just no longer borrowed from a different window's calibration.
"""
import json
import os
import time
import numpy as np
import pandas as pd
pd.set_option('future.infer_string', False)

from heatmap_export import area_elements, raster_for_frame, GRID_COLS, GRID_ROWS, LOG_FLOOR
from space_behind_line import build_velocity_lookup
from config import CHAPTERS_DIR, HEATMAP_VMAX_LOG

SAMPLE_HZ = 5.0
VMAX_LOG_FIXED = HEATMAP_VMAX_LOG  # per-match now -- see config.py and the module docstring above


def compute_chapter_heatmap(anim_path, out_path):
    with open(anim_path) as f:
        data = json.load(f)
    meta = data['meta']
    frames = data['frames']
    home_positive = meta['homeDefendsPositiveX']
    fps = meta['fps']

    vel_full = build_velocity_lookup(frames)
    step = max(1, round(fps / SAMPLE_HZ))
    sample_idxs = list(range(0, len(frames), step))

    rasters, times = [], []
    for i in sample_idxs:
        H = raster_for_frame(frames[i], vel_full[i], home_positive, frame_id=i)
        if H is not None:
            rasters.append(H)
            times.append(frames[i]['clockS'])

    if not rasters:
        # entire chapter is loose-ball / no possession resolved -- write an
        # empty-but-valid heatmap file rather than erroring out
        out = {'meta': {'cols': GRID_COLS, 'rows': GRID_ROWS, 'xMin': -52.5, 'xMax': 52.5,
                         'yMin': -34.0, 'yMax': 34.0, 'logFloor': LOG_FLOOR, 'vmaxLog': VMAX_LOG_FIXED},
               'frames': []}
        with open(out_path, 'w') as f:
            json.dump(out, f)
        return 0, os.path.getsize(out_path)

    stacked = np.clip(np.stack(rasters), 0, None)
    log_vals = np.log1p(stacked / LOG_FLOOR)
    quantized = np.clip(log_vals / VMAX_LOG_FIXED, 0, 1)
    quantized = np.round(quantized * 255).astype(np.uint8)

    import base64
    frame_records = []
    for t, grid in zip(times, quantized):
        b64 = base64.b64encode(grid.tobytes()).decode('ascii')
        frame_records.append({'t': round(t, 3), 'grid': b64})

    out = {
        'meta': {'cols': GRID_COLS, 'rows': GRID_ROWS, 'xMin': -52.5, 'xMax': 52.5,
                 'yMin': -34.0, 'yMax': 34.0, 'logFloor': LOG_FLOOR, 'vmaxLog': VMAX_LOG_FIXED},
        'frames': frame_records,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f)
    return len(frame_records), os.path.getsize(out_path)


def main(part=0, parts=1):
    """Compute every chapter's heatmap. A chapter whose heat file is newer than its animation file is kept
    (so an interrupted run resumes); part/parts splits the chapters across processes (part k takes every
    parts-th chapter starting at k)."""
    with open(f'{CHAPTERS_DIR}/manifest.json') as f:
        manifest = json.load(f)

    t0 = time.time()
    for i, ch in enumerate(manifest['chapters']):
        idx = ch['chapterIndex']
        anim_path = ch['animPath']
        out_path = f'{CHAPTERS_DIR}/heat_{idx:02d}.json'
        ch['heatPath'] = out_path
        if i % parts != part:
            continue
        if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(anim_path):
            print(f"chapter {idx:02d}: already done")
            continue
        n_frames, size_b = compute_chapter_heatmap(anim_path, out_path)
        print(f"chapter {idx:02d}: {n_frames} heatmap samples, {size_b/1e6:.2f}MB "
              f"({time.time()-t0:.1f}s elapsed)")

    if parts == 1:
        with open(f'{CHAPTERS_DIR}/manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        k, n = sys.argv[1].split('/')
        main(int(k), int(n))
    else:
        main()
