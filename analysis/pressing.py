"""Pressing: where each team got tight on the ball carrier, what set it off, and whether it won the ball.

    python3 analysis/pressing.py <match id> [...]

Definitions
- Possession is the core table's `possession_team_id` with flickers under 1 s absorbed: a spell shorter than
  1 s between two spells of the other team is given to that team (gaps with no possession carry the last
  team forward), as in the turnovers view.
- A press is a run of frames where `pressure_score` is above the match's engagement threshold
  (`R.press_threshold`, the same score and cut as the animations), with the pressing team (`pressing_team_id`)
  defending and the ball live. Runs less than 1 s apart are merged into one press, and a press must last at
  least 1/3 s (first to last frame).
- Won: the pressing team has the ball within 5 s of the press starting.
- In the opponents' half: the ball is in the opponents' half when the press starts.
- Trigger, the first that applies:
  - counter-press: the press starts within 3 s of the pressing team last having the ball;
  - otherwise the ball carrier's team's last pass (PA/CR possession event) in the 2.5 s before the press:
    back pass if the ball has gone 2 m+ back towards that team's own goal since the pass, pass out wide if
    the ball is outside the width of the penalty box (|y| > 20.16) when the press starts, else other pass;
  - carry or first touch: no pass in those 2.5 s.
- Who pressed: `primary_presser_jersey` when the press starts, named from the roster.
- Per minute of live defending: presses / minutes of 5 Hz samples with the team defending and the ball live.

The pitch is drawn in the pressing team's frame (own goal on the left): a dot at the ball when each press
began (white = won the ball within 5 s, hollow = didn't) over a glow of where the presses started.
Writes output/analysis/press/<slug>.png/.jpg/.json; the JSON is keyed by the pressing team's name.
A match without an engagement threshold yet is skipped with a message.
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter

from common import FPS, L, R, W, defending, load_ev, load_pos, out_path, roster, side_flip
from inplay import live_mask
from style import BG, DIM, FAINT, INK, MONO, MONO_MED, SEMI, TITLE, WIDTH_IN, Fig, pitch

FLICKER_S = 1.0          # possession spells shorter than this are absorbed
MERGE_S = 1.0            # presses closer than this are one press
MIN_PRESS_S = 1 / 3      # shortest press
WIN_S = 5.0              # won the ball within this long of the press starting
COUNTER_S = 3.0          # counter-press: press starts within this long of losing the ball
PASS_LOOKBACK_S = 2.5    # trigger window before the press
BACK_PASS_M = 2.0        # ball this far back towards the carrier's own goal since the pass = back pass
WIDE_Y = 20.16           # outside the penalty-box width = out wide
TOP_N = 5

TRIGGERS = ('counter-press', 'back pass', 'pass out wide', 'other pass', 'carry or first touch')
PASS_TYPES = ('PA', 'CR')

H_IN = 21.13             # figure height, inches
BLOCK_IN = 8.99          # height of one team's block
GLOW_SIGMA = 4.0         # m: smoothing of the press-start glow
GLOW_ALPHA = 0.72        # glow opacity at the densest spot
GLOW_GAMMA = 1.6         # >1 keeps thinly pressed areas closer to plain grass


# ---------------------------------------------------------------- possession and presses

def runs(mask):
    """(starts, ends) index pairs, inclusive, of the True runs in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def absorb_flickers(fn, poss):
    """Possession per frame with spells under FLICKER_S between two spells of the other team absorbed."""
    p = pd.Series(poss).ffill().to_numpy()
    idx = np.flatnonzero(~np.isnan(p))
    q, f = p[idx].copy(), fn[idx]
    changed = True
    while changed:
        changed = False
        cut = np.flatnonzero(np.diff(q) != 0) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, len(q)] - 1
        for k in range(1, len(starts) - 1):
            s, e = starts[k], ends[k]
            if (f[e] - f[s] + 1) / FPS < FLICKER_S and q[starts[k - 1]] == q[starts[k + 1]]:
                q[s:e + 1] = q[starts[k - 1]]
                changed = True
    out = np.full(len(p), np.nan)
    out[idx] = q
    return out


def find_presses(fn, pressing, score, live, poss, tid, oid, thr):
    """[start, end] frame-index pairs of the presses by team `tid`."""
    mask = (pressing == tid) & (score > thr) & live & (poss == oid)
    merged = []
    for a, b in zip(*runs(mask)):
        if merged and (fn[a] - fn[merged[-1][1]]) / FPS < MERGE_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (fn[b] - fn[a]) / FPS >= MIN_PRESS_S - 1e-9]


def pass_index(ev):
    """Sorted pass frames and the passing team per pass (PA/CR possession events)."""
    rows = sorted((p['frame'], ev['gev'].get(p['gid'], {}).get('team'))
                  for p in ev['pev'].values() if p['type'] in PASS_TYPES and p['frame'] is not None)
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=object)


def trigger(f0, a, fn, poss, tid, opp_code, passes, ball_at, carrier_flip):
    """What set off the press starting at frame f0 (index a)."""
    had = np.flatnonzero(poss[:a] == tid)
    if len(had) and (f0 - fn[had[-1]]) / FPS <= COUNTER_S:
        return 'counter-press'
    pf, pt = passes
    lo, hi = np.searchsorted(pf, f0 - PASS_LOOKBACK_S * FPS), np.searchsorted(pf, f0, 'right')
    own = [k for k in range(lo, hi) if pt[k] == opp_code]
    if not own:
        return 'carry or first touch'
    x_pass, _ = ball_at(pf[own[-1]])
    x_now, y_now = ball_at(f0)
    if (x_now - x_pass) * carrier_flip < -BACK_PASS_M:
        return 'back pass'
    return 'pass out wide' if abs(y_now) > WIDE_Y else 'other pass'


def defending_minutes(mid):
    """{side: minutes of 5 Hz samples with that side defending and the ball live}."""
    P = load_pos(mid)
    d = np.array([s or '' for s in defending(P, mid)])
    return {side: float(((d == side) & P['live']).sum()) / 5 / 60 for side in ('home', 'away')}


def analyse(mid, thr):
    m, ro = R.get(mid), roster(mid)
    df = pd.read_pickle(R.feat(mid)).drop_duplicates('frameNum').sort_values('frameNum').reset_index(drop=True)
    ev = load_ev(mid)
    fn = df['frameNum'].to_numpy()
    period = df['period'].to_numpy()
    bx, by = df['ball_x'].to_numpy(), df['ball_y'].to_numpy()
    live = live_mask(ev, fn)
    poss = absorb_flickers(fn, df['possession_team_id'].to_numpy())
    pressing, score = df['pressing_team_id'].to_numpy(), df['pressure_score'].to_numpy()
    jersey = df['primary_presser_jersey'].to_numpy()
    passes = pass_index(ev)
    def_min = defending_minutes(mid)

    def ball_at(f):
        i = min(np.searchsorted(fn, f), len(fn) - 1)
        return bx[i], by[i]

    def clock_min(f):
        i = min(np.searchsorted(ev['frames'][:, 0], f), len(ev['frames']) - 1)
        return float(ev['frames'][i, 2]) / 60

    res = {}
    for side, opp in (('home', 'away'), ('away', 'home')):
        tid, oid = float(m[side]['id']), float(m[opp]['id'])
        presses = []
        for a, b in find_presses(fn, pressing, score, live, poss, tid, oid, thr):
            f0, per = fn[a], int(period[a])
            fl = side_flip(mid, per, side)
            stop = np.searchsorted(fn, f0 + WIN_S * FPS, 'right')
            j = jersey[a]
            presses.append(dict(
                frame=int(f0), period=per, minute=round(clock_min(f0), 2),
                duration_s=round((fn[b] - f0) / FPS, 2),
                x=round(float(bx[a] * fl), 2), y=round(float(by[a] * fl), 2),
                won=bool((poss[a:stop] == tid).any()),
                trigger=trigger(f0, a, fn, poss, tid, str(m[opp]['id']), passes, ball_at, side_flip(mid, per, opp)),
                presser=ro[side].get(str(j), {}).get('name', f'#{j}') if pd.notna(j) else None))
        res[m[side]['name']] = summarise(presses, def_min[side])
    return res


def summarise(presses, def_min):
    n = len(presses)
    won = sum(p['won'] for p in presses)
    high = [p for p in presses if p['x'] > 0]
    trig = {t: [sum(1 for p in presses if p['trigger'] == t), sum(1 for p in presses if p['trigger'] == t and p['won'])]
            for t in TRIGGERS}
    by = Counter(p['presser'] for p in presses if p['presser'])
    top = sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]
    return dict(presses=n, won_5s=won, win_rate=round(100 * won / n) if n else 0,
                in_opp_half=len(high), opp_half_win_rate=round(100 * sum(p['won'] for p in high) / len(high)) if high else 0,
                per_defending_min=round(n / def_min, 2) if def_min else 0.0,
                defending_min=round(def_min, 2), triggers=trig,
                top_pressers=[[name, k, sum(1 for p in presses if p['presser'] == name and p['won'])] for name, k in top],
                presses_list=presses)


# ---------------------------------------------------------------- drawing

def glow(ax, presses, color):
    """Team-coloured glow of where the presses started, clipped to the pitch."""
    xs = np.array([p['x'] for p in presses])
    ys = np.array([p['y'] for p in presses])
    h, _, _ = np.histogram2d(xs, ys, bins=(int(L), int(W)), range=((-L / 2, L / 2), (-W / 2, W / 2)))
    h = gaussian_filter(h, GLOW_SIGMA, mode='constant')
    if h.max() <= 0:
        return
    rgba = np.zeros(h.T.shape + (4,))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = GLOW_ALPHA * (h.T / h.max()) ** GLOW_GAMMA
    ax.imshow(rgba, extent=(-L / 2, L / 2, -W / 2, W / 2), origin='lower', interpolation='bilinear', zorder=1)


def press_dots(ax, presses, color):
    for p in presses:
        if p['won']:
            ax.scatter(p['x'], p['y'], s=34, color=INK, edgecolor=BG, linewidth=0.6, zorder=6, clip_on=False)
        else:
            ax.scatter(p['x'], p['y'], s=22, facecolor='none', edgecolor=color, linewidth=0.9, alpha=0.55, zorder=5, clip_on=False)


def bar_panel(F, top, title, rows, color):
    """A titled list of rows (label, n, won): bar with a white won part, count, and % won in 5 s."""
    F.text(11.79, top, title, fontproperties=MONO_MED, fontsize=12.5, color=INK)
    F.text(16.91, top, 'won in 5 s', fontproperties=MONO, fontsize=10.5, color=FAINT, ha='right')
    peak = max([n for _, n, _ in rows] + [1])
    for i, (label, n, won) in enumerate(rows):
        y = top + 0.45 + 0.42 * i
        w = 1.43 * n / peak
        y0 = F.Y(y + 0.07)
        F.fig.patches.append(Rectangle((14.36 / WIDTH_IN, y0), w / WIDTH_IN, 0.14 / F.H, transform=F.fig.transFigure,
                                       facecolor=color, alpha=0.45, edgecolor='none'))
        if n:
            F.fig.patches.append(Rectangle((14.36 / WIDTH_IN, y0), w * won / n / WIDTH_IN, 0.14 / F.H,
                                           transform=F.fig.transFigure, facecolor=INK, edgecolor='none'))
        F.text(11.8, y, label, fontproperties=MONO, fontsize=11.3, color=DIM)
        F.text(14.36 + w + 0.11, y, str(n), fontproperties=MONO, fontsize=11.3, color=INK)
        F.text(16.92, y, f'{round(100 * won / n) if n else 0}%', fontproperties=MONO, fontsize=11.3, color=INK, ha='right')


def team_block(F, b, team, opp_name, d):
    F.fig.patches.append(Circle((1.175 / WIDTH_IN, F.Y(2.31 + b)), 0.095 / WIDTH_IN, transform=F.fig.transFigure,
                                facecolor=team['color'], edgecolor='none'))
    F.text(1.41, 2.25 + b, team['name'].upper(), fontproperties=SEMI, fontsize=28)
    F.text(1.08, 2.935 + b,
           f"{d['presses']} presses ({d['per_defending_min']:.1f} per minute of live defending) · "
           f"won the ball within 5 s {d['won_5s']} times ({d['win_rate']}%) · {d['in_opp_half']} in {opp_name}'s half, "
           f"{d['opp_half_win_rate']}% of those won it back", fontproperties=MONO, fontsize=12, color=DIM)
    F.text(1.27, 3.255 + b, 'own goal', fontproperties=MONO, fontsize=10, color=FAINT)
    ax = F.axes(1.12, 3.44 + b, 10.0, 6.5)
    pitch(ax, thirds=True, arcs=False)
    glow(ax, d['presses_list'], team['color'])
    press_dots(ax, d['presses_list'], team['color'])
    bar_panel(F, 3.485 + b, 'WHAT SET IT OFF', [(t, *d['triggers'][t]) for t in TRIGGERS], team['color'])
    bar_panel(F, 6.385 + b, 'WHO PRESSED MOST', d['top_pressers'], team['color'])


def legend(F, y):
    F.fig.patches.append(Circle((1.175 / WIDTH_IN, F.Y(y)), 0.035 / WIDTH_IN, transform=F.fig.transFigure,
                                facecolor=INK, edgecolor='none'))
    F.text(1.29, y, 'press that won the ball within 5 s', fontproperties=MONO, fontsize=10.5, color=FAINT)
    F.fig.patches.append(Circle((5.05 / WIDTH_IN, F.Y(y)), 0.03 / WIDTH_IN, transform=F.fig.transFigure,
                                facecolor='none', edgecolor=DIM, linewidth=1.0))
    F.text(5.17, y, "press that didn't", fontproperties=MONO, fontsize=10.5, color=FAINT)
    F.text(6.94, y, '·', fontproperties=MONO, fontsize=10.5, color=FAINT)
    F.text(7.29, y, 'bars: white part = won the ball', fontproperties=MONO, fontsize=10.5, color=FAINT)


def draw(mid, res, png, jpg):
    m = R.get(mid)
    F = Fig(H_IN)
    F.text(1.08, 0.55, 'PRESSING', fontproperties=TITLE, fontsize=46)
    for i, line in enumerate([f"{m['title']} · {m['sub']}",
                              'Where each team got tight on the ball carrier, what set it off, and whether it won the ball '
                              'within 5 s · own goal on the left']):
        F.text(1.08, 1.12 + 0.25 * i, line, fontproperties=MONO, fontsize=12, color=DIM)
    for i, (side, opp) in enumerate((('home', 'away'), ('away', 'home'))):
        team_block(F, BLOCK_IN * i, m[side], m[opp]['name'], res[m[side]['name']])
    legend(F, 20.175)
    F.footnotes([R.credit(mid, 'Press = pressure on the ball carrier above the engagement threshold calibrated for this match '
                               '(same score as the animations), in live play, lasting 1/3 s+.'),
                 R.credit(mid, "Dot = ball when the press began. Counter-press = within 3 s of losing the ball; other triggers = "
                               "the ball carrier's team's action in the 2.5 s before. Source: PFF FC tracking + events.")],
                20.625, color=FAINT, x=1.08)
    F.save(png, jpg)


# ---------------------------------------------------------------- run

def run(mid):
    try:
        thr = R.press_threshold(mid)
    except (OSError, KeyError, ValueError) as e:
        print(f'{mid}: skipped, no engagement threshold yet ({e.__class__.__name__}: {e})')
        return
    slug = R.get(mid)['slug']
    res = analyse(mid, thr)
    draw(mid, res, out_path('analysis', 'press', f'{slug}.png'), out_path('analysis', 'press', f'{slug}.jpg'))
    with open(out_path('analysis', 'press', f'{slug}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for team, d in res.items():
        print(f"{mid} {team}: {d['presses']} presses ({d['per_defending_min']}/min), won {d['won_5s']} ({d['win_rate']}%), "
              f"{d['in_opp_half']} in the opponents' half ({d['opp_half_win_rate']}%) · threshold {thr}")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
