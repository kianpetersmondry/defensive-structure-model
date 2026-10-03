"""Pass 4: the project's defensive-structure model on footage-derived positions.

Same definitions as the 2D pipeline, applied to what the camera can see:
  depth       mean distance-from-own-goal of the 4 deepest outfield defenders
  inter-line  deepest to most advanced outfield defender (team length)
  phase       High Press >= 40 m, Mid Block >= 32 m, Low Block < 32 m AND
              inter-line <= 20 m (else Mid Block); Transition via the same
              depth/inter-line settling idea as transitions.py
  pressure    Bekkers-style time-to-intercept (pressure.py), 3 s smoothed,
              ENGAGED at >= 0.67 (match 10515's own threshold)
  marking     TIGHT if the 5 defenders nearest the ball span <= 23 m
  heatmap     DAS raster (heatmap_export.raster_for_frame), 5 Hz, match
              10515's log ceiling

Footage-specific guard: a shape read only counts when the camera actually
shows the pitch behind the deepest defender (and beyond the most advanced one
for inter-line) -- otherwise a defender could be standing off-screen.
"""
import sys, math, pickle, numpy as np
sys.path.insert(0, '/home/claude/project_work')
from calib import project, HL, HW, IMG_W, IMG_H
from fit_camera import OV

C = np.load('/home/claude/project_work/footage/camera_C.npy')
HOME_POSITIVE = False             # France (home) defends -X in the 1st half (PFF + footage agree)
HIGH_MIN, MID_MIN, LOW_IL_MAX = 40.0, 32.0, 20.0
REACTION, T0, SCALE, VMAX = 0.7, 1.5, 0.5, 6.0
ENGAGED_T = 0.67
TIGHT_T = 23.0
POSS_TOUCH_M = 2.0
FULL_TEAM = 10     # user rule (round 38): a phase label, the inter-line band and the
                   # depth / team-length numbers need all 10 outfield defenders in shot
LOG_FLOOR, VMAX_LOG = 3e-4, 5.3972


def own_goal_x(team):
    home_def_pos = HOME_POSITIVE
    if team == 'FRA':
        return HL if home_def_pos else -HL
    return -HL if home_def_pos else HL


def visible_frac(x_lo, x_hi, params, y_step=2.0):
    xs = np.arange(min(x_lo, x_hi), max(x_lo, x_hi) + 1e-6, 1.0)
    ys = np.arange(-HW + 1, HW - 1 + 1e-6, y_step)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    uv, z = project(P, C, *params)
    ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    for x1, y1, x2, y2 in OV:
        ok &= ~((uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2))
    return float(ok.mean())


def pressure_on(ball, defenders):
    prod = 1.0
    for d in defenders:
        rx, ry = d['x'] + d['vx'] * REACTION, d['y'] + d['vy'] * REACTION
        ttc = REACTION + math.hypot(ball[0] - rx, ball[1] - ry) / VMAX
        c = 1.0 / (1.0 + math.exp((ttc - T0) / SCALE))
        prod *= (1 - c)
    return 1 - prod


def team_family(team):
    return 'FRA' if team.startswith('FRA') else ('MAR' if team.startswith('MAR') else None)


def compute(frames):
    poss, poss_since = None, None
    challenger, streak = None, 0          # debounce: a new team must be on the ball 3 samples running
    out = []
    for fr in frames:
        r = {'i': fr['i'], 't': fr['t'], 'usable': fr['params'] is not None}
        out.append(r)
        if not r['usable']:
            continue
        p = fr['params']
        people = [pp for pp in fr['people'] if pp.get('n_obs', 0) >= 3]
        ball = fr['ball']
        r['ball'] = ball
        # possession: changes only when a player of the other team is on the ball
        if ball is not None:
            cand = [(math.hypot(pp['x'] - ball[0], pp['y'] - ball[1]), pp) for pp in people if team_family(pp['team'])]
            if cand:
                dmin, pmin = min(cand, key=lambda c: c[0])
                if dmin <= POSS_TOUCH_M:
                    fam = team_family(pmin['team'])
                    if fam == poss:
                        challenger, streak = None, 0
                    elif poss is None:
                        poss, poss_since = fam, fr['t']
                    else:
                        streak = streak + 1 if fam == challenger else 1
                        challenger = fam
                        if streak >= 3:
                            poss, poss_since = fam, fr['t']
                            challenger, streak = None, 0
                    if fam == poss:
                        r['carrier'] = pmin['track']
        r['poss'] = poss
        r['poss_since'] = poss_since
        r['n_vis'] = {fam: sum(1 for pp in people if pp['team'] == fam) for fam in ('FRA', 'MAR')}
        if poss is None:
            continue
        dteam = 'MAR' if poss == 'FRA' else 'FRA'
        r['def'] = dteam
        gx = own_goal_x(dteam)
        sgn = 1 if gx > 0 else -1              # own goal at +X -> distance = gx - x
        outfield = [pp for pp in people if pp['team'] == dteam]
        dist = sorted((abs(gx - pp['x']), pp) for pp in outfield)
        r['n_def'] = len(outfield)
        if len(outfield) >= 4:
            d_min_x = dist[0][1]['x']
            d_max_x = dist[-1][1]['x']
            back_vis = visible_frac(d_min_x, d_min_x + sgn * 4.0, p)
            front_vis = visible_frac(d_max_x, d_max_x - sgn * 4.0, p)
            r['back_vis'], r['front_vis'] = back_vis, front_vis
            full = len(outfield) == FULL_TEAM      # exactly 10: more means a miscounted extra person
            if full and back_vis >= 0.6:
                r['depth'] = float(np.mean([d for d, _ in dist[:4]]))
                r['dmin_x'] = float(d_min_x)
            if full and back_vis >= 0.6 and front_vis >= 0.6:
                r['inter_line'] = float(dist[-1][0] - dist[0][0])
                r['dmax_x'] = float(d_max_x)
        # pressure + marking around the ball
        if ball is not None and outfield:
            r['pressure'] = float(pressure_on(ball, outfield))
            near = sorted(outfield, key=lambda pp: math.hypot(pp['x'] - ball[0], pp['y'] - ball[1]))[:5]
            if len(near) == 5:
                pts = np.array([[pp['x'], pp['y']] for pp in near])
                span = max(np.hypot(*(a - b)) for a in pts for b in pts)
                r['bpc'] = float(span)
                r['marking'] = 'TIGHT' if span <= TIGHT_T else 'LOOSE'
                r['hull'] = pts.tolist()
    add_smoothing_and_phase(out)
    return out


def add_smoothing_and_phase(out, win=3.0):
    """3 s trailing means (as transitions.py / engagement.py), phase labels,
    and a settling-based Transition window after each possession change."""
    ts = np.array([r['t'] for r in out])
    for key in ('depth', 'inter_line', 'pressure'):
        vals = np.array([r.get(key, np.nan) if r['usable'] else np.nan for r in out], float)
        for k, r in enumerate(out):
            if not r['usable'] or key not in r:
                continue
            m = (ts > r['t'] - win) & (ts <= r['t']) & ~np.isnan(vals)
            # a trailing mean must not mix defending teams across a turnover
            m &= np.array([o.get('def') == r.get('def') for o in out])
            r[key + '_s'] = float(vals[m].mean())
    for r in out:
        if not r['usable'] or 'def' not in r:
            continue
        if 'pressure_s' in r:
            r['engaged'] = r['pressure_s'] >= ENGAGED_T
        if 'depth_s' in r:
            d = r['depth_s']
            if d >= HIGH_MIN:
                r['phase'] = 'High Press'
            elif d >= MID_MIN:
                r['phase'] = 'Mid Block'
            elif 'inter_line_s' in r:
                r['phase'] = 'Low Block' if r['inter_line_s'] <= LOW_IL_MAX else 'Mid Block'
    # Transition: from a possession change until the defending team's depth
    # settles -- the readable samples in the next 3 s stay within 3 m of their
    # mean (inter-line within 8 m where known), as transitions.py. Broadcast
    # reads are sparse, so the window needs only 5 readable samples; a change
    # with no settling evidence is treated as settled after 6 s, capped at 20 s.
    changes = sorted(set(r['poss_since'] for r in out if r.get('poss_since') is not None))
    for c0 in changes:
        seg = [r for r in out if r.get('poss_since') == c0 and r['usable']]
        settle_t = None
        for k, r in enumerate(seg):
            if r['t'] - c0 >= 20:
                settle_t = r['t']; break
            if 'depth' not in r or r['t'] - c0 < 1.0:
                continue
            window = [s for s in seg[k:] if s['t'] <= r['t'] + 3.0 and 'depth' in s]
            if len(window) < 5:
                continue
            dep = np.array([s['depth'] for s in window])
            il = np.array([s['inter_line'] for s in window if 'inter_line' in s])
            if np.all(np.abs(dep - dep.mean()) <= 3.0) and (len(il) < 4 or np.all(np.abs(il - il.mean()) <= 8.0)):
                settle_t = r['t']; break
        if settle_t is None:
            settle_t = c0 + 6.0
        for r in seg:
            if r['t'] < settle_t and 'depth' in r:
                r['phase'] = 'Transition'
    # no phase label without a readable shape at that moment
    for r in out:
        if r.get('phase') and 'depth' not in r:
            del r['phase']
    bridge_short_dropouts(out)


def bridge_short_dropouts(out, max_gap_s=0.35):
    """A defender hidden behind another player for a frame or two drops the
    count from 10 to 9 and would make the phase label and band blink. Bridge
    gaps of <= 0.35 s, only when the full-team read is present on BOTH sides
    of the gap, for the same defending team and the same camera run.
    Anything longer still removes the read (the 10-player rule stands)."""
    idx = [k for k, r in enumerate(out) if r.get('phase') and 'depth' in r]
    for a, b in zip(idx, idx[1:]):
        ra, rb = out[a], out[b]
        if b - a < 2 or rb['t'] - ra['t'] > max_gap_s or ra.get('def') != rb.get('def'):
            continue
        gap = out[a + 1:b]
        if any(not g['usable'] or g.get('def') != ra['def'] for g in gap):
            continue
        for g in gap:
            w = (g['t'] - ra['t']) / (rb['t'] - ra['t'])
            g['phase'] = ra['phase'] if w < 0.5 else rb['phase']
            g['bridged'] = True
            for key in ('depth', 'depth_s', 'inter_line', 'inter_line_s', 'dmin_x', 'dmax_x'):
                if key in ra and key in rb:
                    g[key] = (1 - w) * ra[key] + w * rb[key]


def heatmaps(frames, out, hz=5.0):
    """DAS raster for usable frames with a known possession + ball, ~5 Hz."""
    import pandas as pd
    pd.set_option('future.infer_string', False)
    from heatmap_export import raster_for_frame
    last_t = -1e9
    by_i = {r['i']: r for r in out}
    for fr in frames:
        r = by_i[fr['i']]
        if not r['usable'] or r.get('poss') is None or fr['ball'] is None:
            continue
        # benchmark: the heatmap only goes wrong badly when possession is wrong,
        # so only draw it once possession has been stable for 1.5 s
        if r['t'] - r['poss_since'] < 1.5:
            continue
        if fr['t'] - last_t < 1.0 / hz - 1e-3:
            continue
        players, vel = [], {}
        for pp in fr['people']:
            fam = team_family(pp['team'])
            if fam is None or pp.get('n_obs', 0) < 3:
                continue
            pid = ('H' if fam == 'FRA' else 'A') + str(pp['track'])
            players.append({'id': pid, 'x': pp['x'], 'y': pp['y']})
            vel[pid] = (pp['vx'], pp['vy'])
        frame = {'possessionTeam': 'home' if r['poss'] == 'FRA' else 'away',
                 'ball': {'x': fr['ball'][0], 'y': fr['ball'][1]}, 'players': players}
        try:
            H = raster_for_frame(frame, vel, HOME_POSITIVE, frame_id=fr['i'])
        except Exception as e:
            H = None
        if H is None:
            continue
        v = np.log1p(np.clip(H, 0, None) / LOG_FLOOR) / VMAX_LOG
        r['heat'] = (np.clip(v, 0, 1) * 255).astype(np.uint8)   # (52 cols, 34 rows)
        last_t = fr['t']


if __name__ == '__main__':
    frames = pickle.load(open('positions.pkl', 'rb'))
    out = compute(frames)
    heatmaps(frames, out)
    pickle.dump(out, open('model_out.pkl', 'wb'))
    u = [r for r in out if r['usable']]
    from collections import Counter
    print('usable', len(u), 'of', len(out))
    print('possession known', round(np.mean([r.get('poss') is not None for r in u]) * 100, 1), '%')
    print('depth read', round(np.mean(['depth' in r for r in u]) * 100, 1), '%  inter-line read', round(np.mean(['inter_line' in r for r in u]) * 100, 1), '%')
    print('phases', Counter(r.get('phase') for r in u))
    print('pressure computed', round(np.mean(['pressure' in r for r in u]) * 100, 1), '%, heatmaps', sum('heat' in r for r in u))
