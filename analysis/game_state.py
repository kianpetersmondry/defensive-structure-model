"""Game state: how each team's defending changes when it is leading, level or trailing.

    python3 analysis/game_state.py build [<match id> ...]    # per-sample table (cached per match)
    python3 analysis/game_state.py stats                      # effects, uncertainty, JSON for the write-up

Score state
- Goals come from the chances view (every registry goal is found there, at its shot frame). A team's goal
  difference changes at the frame of the goal; penalty shoot-outs are not part of the tracking.
- State per team per 5 Hz sample: leading (goal difference > 0), level (0) or trailing (< 0).

Per-sample measures (the same definitions the site's views use)
- Possession: who has the ball (collect.py), live play only.
- Without the ball (live, the other team has it): back line, length and width (shape_time.team_measures);
  holes and free opponents inside the block (gaps.moment); the phase label (High Press / Mid Block /
  Low Block / Defensive Transition) from the frame classification.
- Presses (pressing view) and ball wins (turnovers view) are events, placed in the state at their frame.

Writes output/analysis/game_state/samples_<id>.pkl per match, and output/analysis/game_state/stats.json.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from common import R, defending, flips, load_pos, out_path, outfield_mask
from gaps import MIN_PLAYERS, moment
from shape_time import team_measures

DIR = out_path('analysis', 'game_state')


def goal_events(mid):
    """[(frame, scoring side)] in time order, from the chances view (keyed by the conceding team)."""
    m = R.get(mid)
    ch = json.load(open(out_path('analysis', 'chances', f"{m['slug']}.json"), encoding='utf-8'))
    out = []
    for conceding, v in ch.items():
        scorer = 'away' if conceding == m['home']['name'] else 'home'
        out += [(s['frame'], scorer) for s in v['shots'] if s['goal']]
    assert len(out) == len(m['goals']), f'{mid}: {len(out)} goals in the chances view, {len(m["goals"])} in the registry'
    return sorted(out)


def build(mid):
    m = R.get(mid)
    P = load_pos(mid)
    n = len(P['frames'])
    dfd = np.array(defending(P, mid), dtype=object)
    live = P['live'] & ~np.isnan(P['clock_min'])

    # home goal difference at every sample
    gd_home = np.zeros(n, int)
    for frame, side in goal_events(mid):
        i = np.searchsorted(P['frames'], frame)
        gd_home[i:] += 1 if side == 'home' else -1

    # phase labels by frame
    feat = 'match_features_classified.pkl' if mid == '10508' else f'match_features_classified_{mid}.pkl'
    F = pd.read_pickle(out_path(feat))[['frameNum', 'home_structure', 'away_structure']]
    F = F.drop_duplicates('frameNum').set_index('frameNum').sort_index()
    F = F.reindex(P['frames'], method='nearest', tolerance=3)

    rows = []
    for side in ('home', 'away'):
        opp = 'away' if side == 'home' else 'home'
        gd = gd_home if side == 'home' else -gd_home
        M = team_measures(P, mid, side)
        dmask = live & (dfd == side)
        # holes and free opponents, every live defending moment
        holes = np.full(n, np.nan); free = np.full(n, np.nan)
        idx = np.flatnonzero(dmask)
        fl = flips(mid, P['period'][idx], side)[:, None, None]
        D = P[side]['xy'][idx][:, outfield_mask(mid, side, np.array(P[side]['jerseys']))] * fl
        O = P[opp]['xy'][idx][:, outfield_mask(mid, opp, np.array(P[opp]['jerseys']))] * fl
        for k, i in enumerate(idx):
            on = ~np.isnan(D[k, :, 0])
            if on.sum() < MIN_PLAYERS:
                continue
            o = O[k][~np.isnan(O[k, :, 0])]
            _, h, _, fr = moment(D[k][on], o)
            holes[i], free[i] = h.sum(), len(fr)
        rows.append(pd.DataFrame({
            'mid': mid, 'slug': m['slug'], 'provider': m.get('provider', 'pff'), 'side': side,
            'team': m[side]['name'], 'opp': m[opp]['name'],
            'i': np.arange(n), 'frame': P['frames'], 'period': P['period'], 'clock': P['clock_min'],
            'live': live, 'has_ball': live & (dfd == opp), 'defending': dmask,
            'gd': gd, 'line': M['line'], 'length': M['length'], 'width': M['width'],
            'holes': holes, 'free': free, 'structure': F[f'{side}_structure'].to_numpy()}))
    df = pd.concat(rows, ignore_index=True)
    df['state'] = np.select([df.gd > 0, df.gd < 0], ['leading', 'trailing'], 'level')
    os.makedirs(DIR, exist_ok=True)
    df.to_pickle(os.path.join(DIR, f'samples_{mid}.pkl'))
    print(f"{m['title']}: {len(df)} rows, {int(df.defending.sum())} defending samples, "
          f"goals {len(m['goals'])}, states {df.groupby('side').state.unique().to_dict()}")


def events(mid):
    """Presses and ball wins as rows (side, frame, won/opp-half flags)."""
    m = R.get(mid)
    pr = json.load(open(out_path('analysis', 'press', f"{m['slug']}.json"), encoding='utf-8'))
    tu = json.load(open(out_path('analysis', 'turn', f"{m['slug']}.json"), encoding='utf-8'))
    side_of = {m['home']['name']: 'home', m['away']['name']: 'away'}
    rows = []
    for team, v in pr.items():
        rows += [dict(kind='press', side=side_of[team], frame=p['frame'], won=bool(p['won'])) for p in v['presses_list']]
    for team, v in tu.items():
        # x is in the winning team's frame (own goal at -x): opponents' half = x >= 0, as turnovers.py counts it
        rows += [dict(kind='win', side=side_of[team], frame=w['frame'], won=bool(w['x'] >= 0)) for w in v['won_events']]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
BIN = 5.0                      # minutes per time bin
MIN_BIN_S = 30                 # seconds of live defending a bin needs (rates and shape)
MIN_STATE_MIN = 3.0            # minutes of live defending a team needs in a state to be compared within itself
ORGANISED = ('High Press', 'Mid Block', 'Low Block')
METRICS = [  # key, label, unit, higher means
    ('line', 'Back-line height', 'm from own goal', 'defending further up'),
    ('length', 'Block length', 'm', 'more stretched'),
    ('width', 'Block width', 'm', 'wider'),
    ('holes', 'Holes inside the block', 'm²', 'more open space inside'),
    ('free', 'Opponents free inside the block', 'players', 'more free attackers'),
    ('high', 'Time in a high press', '% of organised defending', 'pressing higher more often'),
    ('low', 'Time in a low block', '% of organised defending', 'sitting deeper more often'),
    ('press_rate', 'Presses', 'per minute defending', 'pressing more often'),
    ('possession', 'Possession', '% of live play', 'more of the ball'),
]


def load_all():
    S = pd.concat([pd.read_pickle(os.path.join(DIR, f'samples_{mid}.pkl')) for mid in R.ids()], ignore_index=True)
    E = []
    for mid in R.ids():
        e = events(mid)
        e['mid'] = mid
        E.append(e)
    return S, pd.concat(E, ignore_index=True)


def bins(S, E):
    """One row per team-match x period x 5-minute bin x state: medians, shares and rates, with weights."""
    S = S.copy()
    S['tm'] = S.slug + '|' + S.team
    S['bin'] = (S.clock // BIN).astype('Int64')
    keys = ['mid', 'slug', 'team', 'tm', 'provider', 'period', 'bin', 'state']
    d = S[S.defending]
    org = d[d.structure.isin(ORGANISED)]
    g = d.groupby(keys)
    out = g.agg(n_def=('line', 'size'), line=('line', 'median'), length=('length', 'median'), width=('width', 'median'),
                holes=('holes', 'mean'), free=('free', 'mean'), clock=('clock', 'mean')).reset_index()
    o = org.groupby(keys).agg(n_org=('structure', 'size'),
                              high=('structure', lambda s: 100 * (s == 'High Press').mean()),
                              low=('structure', lambda s: 100 * (s == 'Low Block').mean())).reset_index()
    out = out.merge(o, on=keys, how='left')
    lv = S[S.live].groupby(keys).agg(n_live=('has_ball', 'size'), possession=('has_ball', lambda s: 100 * s.mean()),
                                     clock_live=('clock', 'mean')).reset_index()
    out = out.merge(lv, on=keys, how='outer')
    out['clock'] = out.clock.fillna(out.clock_live)
    # presses: place each in its team's state at that frame
    S_idx = {(mid, side): sub.set_index('frame')[['state', 'period', 'bin', 'slug', 'team', 'tm', 'provider']]
             for (mid, side), sub in S.groupby(['mid', 'side'])}
    pr = E[E.kind == 'press']
    rows = []
    for (mid, side), sub in pr.groupby(['mid', 'side']):
        ref = S_idx[(mid, side)]
        pos = np.clip(np.searchsorted(ref.index.values, sub.frame.values), 0, len(ref) - 1)
        r = ref.iloc[pos].reset_index(drop=True)
        r['mid'] = mid
        rows.append(r)
    pc = pd.concat(rows).groupby(keys).size().rename('presses').reset_index()
    out = out.merge(pc, on=keys, how='left')
    out['presses'] = out.presses.fillna(0)
    out['press_rate'] = np.where(out.n_def > 0, out.presses / (out.n_def * 0.2 / 60), np.nan)
    out['n_def'] = out.n_def.fillna(0)
    out['def_min'] = out.n_def * 0.2 / 60
    out['n_live'] = out.n_live.fillna(0)
    out['live_min'] = out.n_live * 0.2 / 60
    return out


def _demean(cols, groups, w):
    """Subtract each group's weighted mean from every column (the within transformation)."""
    out = np.empty_like(cols)
    for g in np.unique(groups):
        k = groups == g
        out[k] = cols[k] - np.average(cols[k], axis=0, weights=w[k])
    return out


def _fit(y, x, basis, groups, w):
    """Coefficient on x in weighted least squares of y on x + basis + group fixed effects."""
    Z = _demean(np.column_stack([y, x, basis]), groups, w)
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Z[:, 1:] * sw, Z[:, 0] * sw[:, 0], rcond=None)
    return coef[0]


def effect(B, key, compare, min_bin_s=MIN_BIN_S, reps=2000, seed=0):
    """State effect (compare vs level) on `key`, within team-match, controlling for match time.

    Weighted least squares on the bins: key ~ state + B-spline(clock, 4 df) + team-match fixed effects
    (fitted by the within transformation), using only team-matches with >= MIN_STATE_MIN minutes of live
    defending in both states (possession: live play). Weights: seconds of defending (organised defending for
    the phase shares, live play for possession). 95% interval: bootstrap over matches, each resampled whole."""
    from patsy import dmatrix
    wcol = 'n_live' if key == 'possession' else ('n_org' if key in ('high', 'low') else 'n_def')
    d = B[B.state.isin(['level', compare]) & B[key].notna() & (B[wcol].fillna(0) * 0.2 >= min_bin_s)].copy()
    tcol = 'live_min' if key == 'possession' else 'def_min'
    mins = d.groupby(['tm', 'state'])[tcol].sum().unstack(fill_value=0)
    if 'level' not in mins or compare not in mins:
        return None
    keep = mins[(mins['level'] >= MIN_STATE_MIN) & (mins[compare] >= MIN_STATE_MIN)].index
    d = d[d.tm.isin(keep)].reset_index(drop=True)
    if d.tm.nunique() < 3:
        return None
    y = d[key].to_numpy(float)
    x = (d.state == compare).to_numpy(float)
    w = d[wcol].to_numpy(float)
    basis = np.asarray(dmatrix('bs(clock, df=4) - 1', d))
    tm = d.tm.to_numpy()
    est = _fit(y, x, basis, tm, w)
    per = []
    for t in np.unique(tm):
        a = (tm == t) & (x == 1); b = (tm == t) & (x == 0)
        per.append(dict(tm=t, diff=float(np.average(y[a], weights=w[a]) - np.average(y[b], weights=w[b])),
                        min_compare=float(d[tcol][a].sum()), min_level=float(d[tcol][b].sum())))
    rng = np.random.default_rng(seed)
    mids = d.mid.to_numpy()
    matches = np.unique(mids)
    rows_of = {m: np.flatnonzero(mids == m) for m in matches}
    boots = []
    for _ in range(reps):
        pick = rng.choice(matches, len(matches), replace=True)
        ix = np.concatenate([rows_of[m] for m in pick])
        grp = np.concatenate([np.char.add(tm[rows_of[m]].astype(str), f'#{k}') for k, m in enumerate(pick)])
        if np.unique(x[ix]).size < 2:
            continue
        boots.append(_fit(y[ix], x[ix], basis[ix], grp, w[ix]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    same = sum(np.sign(p['diff']) == np.sign(est) for p in per)
    return dict(key=key, compare=compare, estimate=float(est), ci=[float(lo), float(hi)], n_teams=int(len(per)),
                n_matches=int(len(matches)), same_direction=int(same), per_team=per)


def stats():
    S, E = load_all()
    B = bins(S, E)
    B.to_pickle(os.path.join(DIR, 'bins.pkl'))
    res = {'metrics': [dict(key=k, label=l, unit=u, higher=h) for k, l, u, h in METRICS], 'effects': []}
    for compare in ('leading', 'trailing'):
        for key, label, unit, _ in METRICS:
            r = effect(B, key, compare)
            if r:
                res['effects'].append(r)
                print(f"{compare:8s} vs level  {label:34s} {r['estimate']:+7.2f} {unit:24s} "
                      f"95% [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  teams {r['n_teams']:2d}  same dir {r['same_direction']}")
    json.dump(res, open(os.path.join(DIR, 'stats.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'build':
        for mid in (sys.argv[2:] or R.ids()):
            build(mid)
    elif cmd == 'stats':
        stats()
