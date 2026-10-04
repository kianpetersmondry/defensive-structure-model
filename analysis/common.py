"""Shared helpers for the analysis views: paths, the registry, live-play mask, team frames."""
import os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
from registry import R            # noqa: E402
from paths import OUT, out        # noqa: E402

FPS = 29.97
L, W = 105.0, 68.0
OFF = {1: 0, 2: 45, 3: 90, 4: 105}          # minutes at the start of each period
PEND = {1: 45, 2: 90, 3: 105, 4: 120}       # minutes at the end of each period (normal time)


def out_path(*p):
    return out(*p)


def load_ev(mid):
    return pickle.load(open(out('analysis', f'ev_{mid}.pkl'), 'rb'))


def load_pos(mid):
    return pickle.load(open(out('analysis', f'pos_{mid}.pkl'), 'rb'))


def roster(mid):
    """{side: {jersey: dict(name, pos, started, gk)}} from the raw roster (keyed by shirt number)."""
    import json
    m = R.get(mid)
    rows = json.load(open(R.raw(mid, 'roster.json')))
    outd = {'home': {}, 'away': {}}
    for r in rows:
        tid = str(r['team']['id'])
        side = 'home' if tid == str(m['home']['id']) else 'away' if tid == str(m['away']['id']) else None
        if side:
            outd[side][str(r['shirtNumber'])] = dict(name=r['player'].get('nickname') or str(r['player'].get('id')),
                                                     pos=r.get('positionGroupType'), started=bool(r.get('started')),
                                                     gk=r.get('positionGroupType') == 'GK')
    return outd


def team_ids(mid):
    m = R.get(mid)
    return str(m['home']['id']), str(m['away']['id'])


def side_flip(mid, period, side):
    """Multiplier that turns raw pitch coordinates into `side`'s own frame: its own goal at -x, attacking +x
    (applied to both x and y, a 180-degree rotation, so the team's left stays at +y)."""
    d = R.dirs(mid)[int(period)]           # +1: home defends +x
    return -d if side == 'home' else d


def flips(mid, periods, side):
    """Vector of side_flip for an array of periods."""
    dirs = R.dirs(mid)
    return np.array([(-dirs[int(p)] if side == 'home' else dirs[int(p)]) for p in periods], dtype=float)


def outfield_mask(mid, side, jerseys, period_arr=None):
    """Boolean per jersey: not a goalkeeper (by roster position)."""
    ro = roster(mid)[side]
    return np.array([not ro.get(j, {}).get('gk', False) for j in jerseys])


def defending(P, mid):
    """Per sample: 'home'/'away' for the team WITHOUT the ball, or None."""
    hid, aid = team_ids(mid)
    return [None if p is None else ('away' if p == hid else 'home' if p == aid else None) for p in P['poss']]
