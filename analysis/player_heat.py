"""PLAYER HEATMAPS: one small pitch per player who played 5+ minutes, from the 5 Hz positions.

Definitions (every 5 Hz sample, every period, each team in its own frame attacking left to right):
- Minutes on the pitch: samples with a position / (29.97 / 6) / 60.
- Order: starters first, then subs, each by position (GK, RB, RWB, RCB, MCB, LCB, LB, LWB, DM, CM, AM, RW,
  LW, CF), then by minutes played, then in roster order.
- Glow: where the player was, on his own scale. White dot: his median position.

    python3 analysis/player_heat.py 3823 [3821 ...]
"""
import json
import sys

import numpy as np

from common import FPS, R, load_pos, out_path, roster
from heat import PX, glow, own_frame, pitch_ax, team_label
from style import BG, DIM, FAINT, INK, MONO, SEMI, TITLE, Fig

MIN_MINUTES = 5
ORDER = ['GK', 'RB', 'RWB', 'RCB', 'MCB', 'LCB', 'LB', 'LWB', 'DM', 'CM', 'AM', 'RW', 'LW', 'CF']
POS_NAME = {'GK': 'Goalkeeper', 'RB': 'Right back', 'RWB': 'Right wing-back', 'RCB': 'Right centre-back',
            'MCB': 'Centre-back', 'LCB': 'Left centre-back', 'LB': 'Left back', 'LWB': 'Left wing-back',
            'DM': 'Defensive midfield', 'CM': 'Central midfield', 'AM': 'Attacking midfield', 'RW': 'Right wing',
            'LW': 'Left wing', 'CF': 'Centre-forward'}
FOOTNOTE = ('Glow: share of the match each player spent in each area (his own scale). White dot: his median position. '
            'Source: PFF FC broadcast tracking behind the 2-D animations, every period.')

# layout (px)
COLS, COL_X, COL_STEP, PITCH_W, ROW_STEP = 4, 111, 396, 358, 313
TEAM_TOP, LABEL_TO_PITCH, PITCH_TO_LABEL = 229, 120, 55.5


def players(P, mid, side):
    """Players with 5+ minutes in display order, with minutes, median spot and their own-frame positions."""
    ro = roster(mid)[side]
    J = P[side]['jerseys']
    X, Y = own_frame(P, mid, side)
    order = list(ro)
    rows = []
    for k, j in enumerate(J):
        ok = ~np.isnan(X[:, k])
        minutes = ok.sum() / (FPS / 6) / 60
        if minutes < MIN_MINUTES:
            continue
        info = ro.get(j, {})
        rows.append(dict(jersey=j, name=info.get('name', j), pos=info.get('pos'), started=info.get('started', False),
                         minutes=minutes, median=[float(np.median(X[ok, k])), float(np.median(Y[ok, k]))],
                         xy=(X[ok, k], Y[ok, k])))
    rank = {p: i for i, p in enumerate(ORDER)}
    rows.sort(key=lambda r: (not r['started'], rank.get(r['pos'], len(ORDER)), -r['minutes'],
                             order.index(r['jersey']) if r['jersey'] in order else len(order)))
    return rows


def draw_player(F, left, top, r, color):
    F.text((left - 2) * PX, (top - 46) * PX, f"#{r['jersey']}  {r['name']}", fontproperties=SEMI, fontsize=15.5,
           va='baseline')
    info = f"{POS_NAME.get(r['pos'], r['pos'])} · {r['minutes']:.0f} min" + ('' if r['started'] else ' · sub')
    F.text((left - 2) * PX, (top - 19) * PX, info, fontproperties=MONO, fontsize=10, color=DIM, va='baseline')
    ax = pitch_ax(F, left, top, PITCH_W, lw=0.7)
    glow(ax, *r['xy'], color)
    ax.scatter(*r['median'], s=34, color=INK, edgecolor=BG, linewidth=1.2, zorder=6)


def layout(teams):
    """Baselines of the two team labels and the image height (px)."""
    rows = [-(-len(t) // COLS) for t in teams]
    second = TEAM_TOP + LABEL_TO_PITCH + rows[0] * ROW_STEP - (ROW_STEP - PITCH_W * 68 / 105) + PITCH_TO_LABEL
    last_bottom = second + LABEL_TO_PITCH + rows[1] * ROW_STEP - (ROW_STEP - PITCH_W * 68 / 105)
    return (TEAM_TOP, second), last_bottom


def draw(mid, teams, png, jpg):
    m = R.get(mid)
    (b1, b2), last_bottom = layout(teams)
    F = Fig(int(last_bottom + 76) * PX)
    F.text(109 * PX, 79 * PX, 'PLAYER HEATMAPS', fontproperties=TITLE, fontsize=46, va='baseline')
    for base, s in ((117, f"{m['title']} · {m['sub']}"),
                    (142, 'Where every player who played 5+ minutes spent the match · each team attacking left to right')):
        F.text(109 * PX, base * PX, s, fontproperties=MONO, fontsize=12, color=DIM, va='baseline')
    for side, base, rows in zip(('home', 'away'), (b1, b2), teams):
        team = m[side]
        n_st = sum(r['started'] for r in rows)
        team_label(F, 109, base, team['name'], team['color'], size=26, dot=9, gap=33, dot_dy=-8,
                   right=f'{n_st} starters · {len(rows) - n_st} subs', right_size=11.5, right_dy=-5)
        for i, r in enumerate(rows):
            draw_player(F, COL_X + COL_STEP * (i % COLS), base + LABEL_TO_PITCH + ROW_STEP * (i // COLS), r,
                        team['color'])
    F.text(109 * PX, (last_bottom + 35) * PX, R.credit(mid, FOOTNOTE), fontproperties=MONO, fontsize=10.5,
           color=FAINT, va='baseline')
    F.save(png, jpg)


def run(mid):
    m = R.get(mid)
    P = load_pos(mid)
    teams = [players(P, mid, side) for side in ('home', 'away')]
    path = out_path('analysis', 'players', f"{m['slug']}.json")
    draw(mid, teams, path[:-5] + '.png', path[:-5] + '.jpg')
    out = {m[side]['name']: {r['jersey']: dict(name=r['name'], minutes=round(r['minutes'], 1),
                                               median=[round(v, 1) for v in r['median']]) for r in rows}
           for side, rows in zip(('home', 'away'), teams)}
    json.dump(out, open(path, 'w'), indent=1, ensure_ascii=False)
    for name, rows in out.items():
        print(f"{m['slug']} {name}: " + ', '.join(f"#{j} {r['minutes']:.0f}" for j, r in rows.items()))


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
