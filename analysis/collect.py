"""Player positions at ~5 Hz (every 6th tracking frame), with possession and live play per sample.

    python3 analysis/collect.py <match id> [...]       (needs collect_events.py and the core features first)

Writes output/analysis/pos_<id>.pkl:
- frames, period, clock_min (match minute, period offsets 0/45/90/105), ball (n, 3)
- home / away: {'jerseys': [...], 'xy': (n, J, 2) pitch metres (NaN when not on the pitch)}
- poss: team id (str) in possession per sample, or None; live: ball in play per sample
The 2022 final's extra-time keeper labels are swapped back here (registry quirk keeper_swap).
"""
import bz2, json, pickle, sys
import numpy as np
import pandas as pd
from common import R, out_path, load_ev
from inplay import live_mask

STEP = 6


def collect(mid):
    m = R.get(mid); swap = R.keeper_swap(mid)
    ev = load_ev(mid); allf = ev['frames']
    keep = allf[::STEP]                                    # every 6th unique frame
    want = set(keep[:, 0].astype(int))
    idx = {int(f): i for i, f in enumerate(keep[:, 0])}
    n = len(keep)
    pos = {'home': {}, 'away': {}}
    with bz2.open(R.raw(mid, 'tracking.jsonl.bz2'), 'rt') as f:
        done = set()
        for line in f:
            d = json.loads(line); fn = d.get('frameNum')
            if fn not in want or fn in done:
                continue
            done.add(fn); i = idx[fn]; per = d.get('period')
            for side in ('home', 'away'):
                pl = d.get(f'{side}PlayersSmoothed') or d.get(f'{side}Players') or []
                for p in pl:
                    if p.get('x') is None or p.get('jerseyNum') is None:
                        continue
                    j = str(p['jerseyNum']); s = side
                    if swap and per in swap['periods']:
                        if side == 'home' and j == str(swap['home']): s, j = 'away', str(swap['away'])
                        elif side == 'away' and j == str(swap['away']): s, j = 'home', str(swap['home'])
                    arr = pos[s].setdefault(j, np.full((n, 2), np.nan, np.float32))
                    if np.isnan(arr[i, 0]):
                        arr[i] = (p['x'], p['y'])
    out = {}
    for side in ('home', 'away'):
        js = sorted(pos[side], key=lambda x: int(x) if x.isdigit() else 999)
        out[side] = {'jerseys': js, 'xy': np.stack([pos[side][j] for j in js], 1) if js else np.zeros((n, 0, 2))}
    feat = pd.read_pickle(R.feat(mid))[['frameNum', 'possession_team_id']].drop_duplicates('frameNum').set_index('frameNum')['possession_team_id']
    pt = feat.reindex(keep[:, 0].astype(int)).to_numpy()
    poss = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(v)) for v in pt]
    per = keep[:, 1].astype(int)
    clock = keep[:, 2] / 60                               # periodGameClockTime is the running match clock (2nd half from 45:00)
    res = dict(frames=keep[:, 0].astype(int), period=per, clock_min=clock, ball=keep[:, 3:6],
               poss=poss, live=live_mask(ev, keep[:, 0]), **out)
    pickle.dump(res, open(out_path('analysis', f'pos_{mid}.pkl'), 'wb'))
    print(mid, f'{n:,} samples · home {len(out["home"]["jerseys"])} / away {len(out["away"]["jerseys"])} players · '
               f'live {res["live"].mean() * 100:.0f}% · possession known {np.mean([p is not None for p in poss]) * 100:.0f}%')


if __name__ == '__main__':
    for mid in sys.argv[1:]: collect(mid)
