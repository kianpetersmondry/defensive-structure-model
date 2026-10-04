"""WHERE THEY ATTACKED: which flank each team attacked down, from the ball track in the core features table.

Definitions (every tracking frame, every period including stoppage and extra time):
- Each team is shown in its own frame, attacking left to right; left and right are from the attacker's view.
- On-ball time: all frames with the team in possession (`possession_team_id`), live or not, / 29.97 fps.
- Lanes: thirds of the pitch width (|y| <= W/6 is the middle).
- Arrows: the share of the ball's forward movement in each lane. A frame-to-frame step counts when the team
  has the ball at both ends, it starts in the middle or final third (x > -L/6) and it moves the ball towards
  goal (dx > 0); the step's length goes to the lane the ball is in at the end of the step.
- Final-third entries: the ball crossing x = L/6 forwards between two frames of the team's possession, at
  most one per 4 s; the lane is where it crossed.
- Glow: where the ball was during the team's possession.
- Split line: the same shares per half (extra-time periods pooled).

    python3 analysis/lanes.py 3823 [3821 ...]
"""
import json
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrow

from common import FPS, L, W, R, out_path, team_ids, side_flip
from heat import PX, glow, mmss, pitch_ax, team_label
from style import DIM, FAINT, INK, BG, LINE, MONO, TITLE, Fig

LANES = ('left', 'middle', 'right')
ENTRY_GAP_S = 4.0
PERIOD_GROUPS = (('1st half', (1,)), ('2nd half', (2,)), ('extra time', (3, 4)))

# layout (px): one 1200 px block per team
H_PX, BLOCK, PITCH_TOP, PITCH_LEFT, PITCH_W = 2895, 1200, 358, 121, 1558
FOOTNOTES = ["Arrows: share of the ball's forward movement in each third of the pitch width, counted in the middle "
             "and final thirds while the team had the ball.",
             "Glow: where the ball was during their possession. Left and right are from the attacking team's view.",
             "Final-third entry: the ball carried or passed into the last third, at most one per 4 s."]
SOURCE = 'Source: PFF FC broadcast tracking behind the 2-D animation (match {mid}), every period including stoppage time.'


def lane_of(y):
    """0 = left, 1 = middle, 2 = right (attacker's view, +y is his left)."""
    return np.where(y > W / 6, 0, np.where(y < -W / 6, 2, 1))


def shares(dx, lane, mask):
    tot = np.array([dx[mask & (lane == k)].sum() for k in range(3)])
    return 100 * tot / tot.sum() if tot.sum() > 0 else np.zeros(3)


def team_lanes(f, mid, side, tid):
    """Lane shares, entries, on-ball time and the glow points for one team."""
    period = f['period'].to_numpy()
    own = (f['possession_team_id'].astype(float) == float(tid)).to_numpy()
    flip = np.array([side_flip(mid, p, side) for p in period])
    x, y = f['ball_x'].to_numpy() * flip, f['ball_y'].to_numpy() * flip
    lane = lane_of(y)

    step = own & np.r_[False, own[:-1]] & np.r_[False, period[1:] == period[:-1]]
    x_prev = np.r_[np.nan, x[:-1]]
    dx = np.nan_to_num(x - x_prev)
    fwd = step & (x_prev > -L / 6) & (dx > 0)

    per_period = {}
    for label, ps in PERIOD_GROUPS:
        m = np.isin(period, ps)
        if m.any():
            per_period[label] = dict(zip(LANES, np.round(shares(dx, lane, fwd & m), 2).tolist()))

    frame = f['frameNum'].to_numpy()
    cross = np.where(step & (x >= L / 6) & (x_prev < L / 6))[0]
    entries, last = [], -np.inf
    for i in cross:
        if frame[i] - last >= ENTRY_GAP_S * FPS:
            entries.append(i)
            last = frame[i]
    by_lane = np.bincount(lane[entries], minlength=3) if entries else np.zeros(3, int)

    tot = shares(dx, lane, fwd)
    return dict(**dict(zip(LANES, np.round(tot, 2).tolist())),
                entries=int(len(entries)), entries_by_lane=dict(zip(LANES, by_lane.tolist())),
                on_ball=mmss(own.sum() / FPS), per_period=per_period), (x[own], y[own])


def draw_arrow(ax, yc, share, n_entries):
    w = 1.1 + 0.1115 * share
    ax.add_patch(FancyArrow(-16, yc, 39.8, 0, width=w, head_width=2 * w, head_length=6.2, length_includes_head=False,
                            facecolor=INK, edgecolor=BG, lw=1.5, alpha=0.92, zorder=5))
    ax.text(33.5, yc + 0.5, f'{share:.0f}%', fontproperties=TITLE, fontsize=43.3, color=INK, va='baseline', zorder=6)
    ax.text(33.8, yc - 5.0, f'{n_entries} final-third entries', fontproperties=MONO, fontsize=12.7, color=INK,
            alpha=0.8, va='baseline', zorder=6)


def draw_pitch(F, top, color, res, pts):
    ax = pitch_ax(F, PITCH_LEFT, top, PITCH_W, thirds=True)
    for yy in (-W / 6, W / 6):
        ax.plot([-L / 2, L / 2], [yy, yy], color=LINE, lw=0.7, ls=(0, (3, 4)), alpha=0.6, zorder=3)
    glow(ax, *pts, color, sigma=4.5, pct=99, gamma=1.2)
    lab = dict(fontproperties=MONO, fontsize=14.2, color=INK, alpha=0.85, zorder=6)
    ax.text(-49.3, W / 2 - 1.25, 'LEFT WING', va='top', **lab)
    ax.text(-49.3, W / 6 - 1.25, 'THROUGH THE MIDDLE', va='top', **lab)
    ax.text(-49.3, -W / 2 + 1.85, 'RIGHT WING', va='baseline', **lab)
    for k, yc in enumerate((W / 3, 0, -W / 3)):
        draw_arrow(ax, yc, res[LANES[k]], res['entries_by_lane'][LANES[k]])


def split_line(F, base, per_period):
    for i, (label, s) in enumerate(per_period.items()):
        txt = f"{label}:  L {s['left']:.0f}% · M {s['middle']:.0f}% · R {s['right']:.0f}%"
        F.text((109 + 451 * i) * PX, base * PX, txt, fontproperties=MONO, fontsize=14.25, color=DIM, va='baseline')


def draw(mid, results, glows, png, jpg):
    m = R.get(mid)
    F = Fig(H_PX * PX)
    F.text(108 * PX, 126 * PX, 'WHERE THEY ATTACKED', fontproperties=TITLE, fontsize=60.7, va='baseline')
    for base, s in ((182, f"{m['title']} · {m['sub']}"),
                    (216, 'Full match, every period · both teams shown attacking left to right')):
        F.text(109 * PX, base * PX, s, fontproperties=MONO, fontsize=15.7, color=DIM, va='baseline')
    for i, side in enumerate(('home', 'away')):
        team, y0 = m[side], BLOCK * i
        res = results[team['name']]
        team_label(F, 108, 286 + y0, f"{team['name']} in possession", team['color'], size=31.5, dot=(11, 17.5),
                   gap=36, dot_dy=-11, right=f"{res['on_ball']} on the ball  ·  {res['entries']} final-third entries",
                   right_size=15.7, right_x=1690, right_dy=-5)
        draw_pitch(F, PITCH_TOP + y0, team['color'], res, glows[side])
        split_line(F, 1421 + y0, res['per_period'])
    for k, s in enumerate(FOOTNOTES):
        F.text(109 * PX, (2712 + 38 * k) * PX, s, fontproperties=MONO, fontsize=13.2, color=DIM, va='baseline')
    F.text(109 * PX, 2832 * PX, R.credit(mid, SOURCE.format(mid=mid)), fontproperties=MONO, fontsize=13.2,
           color=FAINT, va='baseline')
    F.save(png, jpg)


def run(mid):
    m = R.get(mid)
    f = pd.read_pickle(R.feat(mid))[['frameNum', 'period', 'ball_x', 'ball_y', 'possession_team_id']]
    results, glows = {}, {}
    for side, tid in zip(('home', 'away'), team_ids(mid)):
        results[m[side]['name']], glows[side] = team_lanes(f, mid, side, tid)
    path = out_path('analysis', 'attack', f"{m['slug']}.json")
    draw(mid, results, glows, path[:-5] + '.png', path[:-5] + '.jpg')
    json.dump(results, open(path, 'w'), indent=1, ensure_ascii=False)
    for name, r in results.items():
        print(f"{m['slug']} {name}: L {r['left']:.0f} / M {r['middle']:.0f} / R {r['right']:.0f}, "
              f"{r['entries']} entries {tuple(r['entries_by_lane'].values())}, {r['on_ball']} on the ball")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
