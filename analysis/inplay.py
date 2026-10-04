"""Ball-in-play mask from the event feed.

Play is live from an on-the-ball (OTB) event until the next one, provided the next starts within 5 s of the
last touch ending and no ball-out (OUT) event comes in between; otherwise it stays live for 1.5 s after the
last touch ends.
"""
import numpy as np

GAP, TAIL = 150, 45            # frames: 5 s and 1.5 s at 29.97 fps


def live_mask(ev, frames):
    """Boolean array: is play live at each of `frames` (frame numbers, sorted)."""
    g = ev['gev']
    otb = sorted((v['start'], v['end'] if v['end'] is not None else v['start'])
                 for v in g.values() if v['type'] == 'OTB' and v['start'] is not None)
    outs = np.array(sorted(v['start'] for v in g.values() if v['type'] == 'OUT' and v['start'] is not None))
    frames = np.asarray(frames)
    live = np.zeros(len(frames), bool)
    for i, (s, e) in enumerate(otb):
        nxt = otb[i + 1][0] if i + 1 < len(otb) else None
        ok = nxt is not None and nxt - e <= GAP and not ((outs > e) & (outs < nxt)).any()
        a, b = np.searchsorted(frames, s), np.searchsorted(frames, nxt if ok else e + TAIL)
        live[a:b] = True
    return live
