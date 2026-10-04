"""Event feed, match clock and ball track from a match's raw tracking file (one pass).

    python3 analysis/collect_events.py <match id> [...]

Writes output/analysis/ev_<id>.pkl:
- gev: {game_event_id: {type, team, player, shirt, pos, home_ball, start, end}}  (frames; OTB = on the ball, OUT = ball out)
- pev: {possession_event_id: {type, gid, frame}}                               (PA pass, CR cross, SH shot, CH challenge, ...)
- frames: array (n, 6) for every tracking line: frameNum, period, clock_s, ball x, ball y, ball z (NaN when no ball)
"""
import bz2, json, os, pickle, sys
import numpy as np
from common import R, out_path

def collect(mid):
    gev, pev, rows = {}, {}, []
    with bz2.open(R.raw(mid, 'tracking.jsonl.bz2'), 'rt') as f:
        for line in f:
            d = json.loads(line)
            per = d.get('period')
            if per is None:
                continue
            b = d.get('ballsSmoothed') or {}
            if isinstance(b, list): b = b[0] if b else {}
            if b.get('x') is None:
                bl = d.get('balls') or []
                b = bl[0] if bl and bl[0].get('x') is not None else {}
            rows.append((d['frameNum'], per, d.get('periodGameClockTime') or np.nan,
                         b.get('x', np.nan) if b.get('x') is not None else np.nan,
                         b.get('y', np.nan) if b.get('y') is not None else np.nan,
                         b.get('z', np.nan) if b.get('z') is not None else np.nan))
            g = d.get('game_event'); gid = d.get('game_event_id')
            if g and gid is not None and gid not in gev:
                gev[gid] = dict(type=g.get('game_event_type'), team=None if g.get('team_id') is None else str(int(float(g['team_id']))),
                                player=g.get('player_name'), shirt=g.get('shirt_number'), pos=g.get('position_group_type'),
                                home_ball=g.get('home_ball'), start=g.get('start_frame'), end=g.get('end_frame'), period=per)
            p = d.get('possession_event'); pid = d.get('possession_event_id')
            if p and pid is not None and pid not in pev:
                pev[pid] = dict(type=p.get('possession_event_type'), gid=p.get('game_event_id'), frame=p.get('start_frame', d['frameNum']))
    fr = np.array(rows, dtype=float)
    fr = fr[np.argsort(fr[:, 0], kind='stable')]
    _, keep = np.unique(fr[:, 0], return_index=True)          # duplicate frame lines: keep the first
    fr = fr[np.sort(keep)]
    pickle.dump(dict(gev=gev, pev=pev, frames=fr), open(out_path('analysis', f'ev_{mid}.pkl'), 'wb'))
    print(mid, f'{len(fr):,} frames · {len(gev):,} game events · {len(pev):,} possession events')

if __name__ == '__main__':
    for m in sys.argv[1:]: collect(m)
