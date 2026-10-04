"""Benchmark footage-derived positions against PFF tracking (game 10515,
chapter 1 of the existing export = match clock 0:00-4:50).

PFF is itself broadcast-derived, so this is agreement with a professional
reference, not ground truth.

1. Alignment: search the axis signs and a small clock offset that minimise the
   median player position error (the scoreboard gives video_t - clock ~ 8.4 s).
2. Position error: per usable footage sample, match footage players to PFF
   players of the same team (Hungarian, <= 8 m gate) and record distances.
3. Coverage: of the PFF players standing inside the part of the pitch the
   camera shows, how many does the footage pipeline have?
4. Ball error and possession agreement.
"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
PIPELINE = _os.path.join(HERE, '..', 'pipeline')
import json, pickle, numpy as np, sys
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, HERE)
from calib import project, IMG_W, IMG_H
from model_footage import team_family

C = np.load(_os.path.join(HERE, 'camera_C.npy'))
pff = json.load(open(_os.path.join(_os.environ.get('DSM_OUT', _os.path.join(HERE, '..', 'output')), 'chapters_10515', 'anim_01.json')))['frames']
pff_clock = np.array([f['clockS'] for f in pff])


def pff_at(clock):
    k = int(np.clip(np.searchsorted(pff_clock, clock), 0, len(pff) - 1))
    if k > 0 and abs(pff_clock[k - 1] - clock) < abs(pff_clock[k] - clock):
        k -= 1
    return pff[k] if abs(pff_clock[k] - clock) < 0.2 else None


def match(foot, ref, gate=8.0):
    if not foot or not ref:
        return []
    A = np.array(foot); B = np.array(ref)
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    r, c = linear_sum_assignment(D)
    return [D[a, b] for a, b in zip(r, c) if D[a, b] <= gate]


def evaluate(positions, model, sx=1, sy=1, offset=8.4, sample_every=1, full=False):
    errs, cov_hit, cov_tot, ball_err, poss_agree = [], 0, 0, [], []
    mod = {r['i']: r for r in model}
    for n, fr in enumerate(positions):
        if fr['params'] is None or n % sample_every:
            continue
        pf = pff_at(fr['t'] - offset)
        if pf is None:
            continue
        foot = {'FRA': [], 'MAR': []}
        for pp in fr['people']:
            fam = team_family(pp['team'])
            if fam and pp.get('n_obs', 0) >= 3:
                foot[fam].append((sx * pp['x'], sy * pp['y']))
        ref = {'FRA': [], 'MAR': []}
        for pp in pf['players']:
            ref['FRA' if pp['id'][0] == 'H' else 'MAR'].append((pp['x'], pp['y']))
        for fam in ('FRA', 'MAR'):
            errs += match(foot[fam], ref[fam])
        if not full:
            continue
        # coverage: PFF players whose position projects inside the frame
        for fam in ('FRA', 'MAR'):
            if not ref[fam]:
                continue
            R = np.array(ref[fam])
            P = np.c_[sx * R[:, 0], sy * R[:, 1], np.zeros(len(R))]
            uv, z = project(P, C, *fr['params'])
            inside = (z > 1) & (uv[:, 0] > 5) & (uv[:, 0] < IMG_W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < IMG_H - 5)
            cov_tot += int(inside.sum())
            if foot[fam] and inside.any():
                A = np.array(foot[fam])
                Bm = R[inside]
                D = np.linalg.norm(Bm[:, None] - A[None], axis=2)
                r, c = linear_sum_assignment(D)
                cov_hit += int(sum(D[a, b] <= 3.0 for a, b in zip(r, c)))
        if fr['ball'] is not None and pf.get('ball') and pf['ball'].get('x') is not None:
            ball_err.append(np.hypot(sx * fr['ball'][0] - pf['ball']['x'], sy * fr['ball'][1] - pf['ball']['y']))
        m = mod.get(fr['i'], {})
        if m.get('poss') and pf.get('possessionTeam') in ('home', 'away'):
            poss_agree.append((m['poss'] == 'FRA') == (pf['possessionTeam'] == 'home'))
    return np.array(errs), cov_hit, cov_tot, np.array(ball_err), np.array(poss_agree)


if __name__ == '__main__':
    positions = pickle.load(open(_os.path.join(HERE, 'positions.pkl'), 'rb'))
    model = pickle.load(open(_os.path.join(HERE, 'model_out.pkl'), 'rb'))
    best = None
    for sx, sy in ((1, 1),):   # sign check done: (+1,+1) wins clearly (3.4 m vs 4.6-5.3 m)
        best = (0, sx, sy)
    _, sx, sy = best
    offs = np.round(np.arange(4.0, 9.01, 0.1), 2)
    meds = []
    for off in offs:
        e, *_ = evaluate(positions, model, sx, sy, off, sample_every=3)
        meds.append(np.median(e))
    off = offs[int(np.argmin(meds))]
    print('best clock offset', off, 's (median err by offset:', dict(zip(offs[::4], np.round(meds[::4], 2))), ')')
    e, hit, tot, be, pa = evaluate(positions, model, sx, sy, off, full=True)
    res = {'signs': (sx, sy), 'offset': float(off), 'n_matches': int(len(e)),
           'median_err_m': float(np.median(e)), 'p75_err_m': float(np.percentile(e, 75)),
           'p90_err_m': float(np.percentile(e, 90)), 'within_1m': float((e <= 1).mean()),
           'within_2m': float((e <= 2).mean()), 'coverage': hit / max(tot, 1), 'cov_tot': tot,
           'ball_median_err_m': float(np.median(be)) if len(be) else None,
           'ball_within_2m': float((be <= 2).mean()) if len(be) else None,
           'possession_agreement': float(pa.mean()) if len(pa) else None, 'n_poss': int(len(pa))}
    for k, v in res.items():
        print(f'  {k}: {v if not isinstance(v, float) else round(v, 3)}')
    json.dump(res, open(_os.path.join(HERE, 'benchmark.json'), 'w'), indent=1)
