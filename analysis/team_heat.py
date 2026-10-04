"""Team heatmaps: where each team's outfield players spent the match, from the 5 Hz positions.

Definitions (every 5 Hz sample, every period, each team in its own frame attacking left to right):
- Glow: all outfield players' positions pooled (keepers left out so they don't swamp the scale).
- Circles: each starter's median position (keeper included).
- Team centre: median x of the pooled outfield positions, in metres from halfway (+ = opponents' half).
- Usual width: 10th to 90th percentile of the pooled outfield y.

    python3 analysis/team_heat.py 3823 [3821 ...]
"""
import json
import sys

import numpy as np

from common import R, load_pos, out_path
from heat import PX, glow, marker, outfield, own_frame, pitch_ax, starters, team_label
from style import DIM, FAINT, MONO, TITLE, Fig

H_PX, BLOCK, PITCH_TOP, PITCH_LEFT, PITCH_W = 2642, 1182, 284, 123, 1555
FOOTNOTES = ['Glow: all outfield players’ positions over the match, every period (keepers left out so they don’t '
             'swamp the scale).',
             'Circles: each starter’s median position. Source: PFF FC broadcast tracking behind the 2-D animations.']


def centre_text(c):
    m = abs(c)
    if round(m) == 0:
        return 'team centre on the halfway line'
    return f'team centre {m:.0f} m ' + ('into the opponents’ half' if c > 0 else 'inside its own half')


def team_shape(P, mid, side):
    """Centre, usual width, starters' median spots and the pooled outfield points for one team."""
    J = P[side]['jerseys']
    X, Y = own_frame(P, mid, side)
    of = outfield(mid, side, J)
    x, y = X[:, of].ravel(), Y[:, of].ravel()
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    spots = {j: [round(float(np.nanmedian(X[:, J.index(j)])), 1), round(float(np.nanmedian(Y[:, J.index(j)])), 1)]
             for j in starters(mid, side, J)}
    return dict(centre=round(float(np.median(x)), 2),
                width=round(float(np.percentile(y, 90) - np.percentile(y, 10)), 2),
                starters=spots), (x, y)


def draw(mid, results, points, png, jpg):
    m = R.get(mid)
    F = Fig(H_PX * PX)
    F.text(109 * PX, 79 * PX, 'TEAM HEATMAPS', fontproperties=TITLE, fontsize=46, va='baseline')
    for base, s in ((117, f"{m['title']} · {m['sub']}"),
                    (142, 'Where each team\'s outfield players spent the match · numbers: starters\' average positions'
                          ' · each team attacking left to right')):
        F.text(109 * PX, base * PX, s, fontproperties=MONO, fontsize=12, color=DIM, va='baseline')
    for i, side in enumerate(('home', 'away')):
        team, y0 = m[side], BLOCK * i
        res = results[team['name']]
        team_label(F, 108, 234 + y0, team["name"], team["color"], size=28, dot=10, gap=33, dot_dy=-8,
                   right=f"{centre_text(res['centre'])} · usual width {res['width']:.0f} m", right_dy=-5, right_x=1692)
        ax = pitch_ax(F, PITCH_LEFT, PITCH_TOP + y0, PITCH_W, lw=0.7)
        glow(ax, *points[side], team['color'])
        for j, (x, y) in res['starters'].items():
            marker(ax, x, y, j, r=1.15, size=14)
    for k, s in enumerate(FOOTNOTES):
        F.text(109 * PX, (2592 + 25 * k) * PX, R.credit(mid, s), fontproperties=MONO, fontsize=10.5, color=FAINT,
               va='baseline')
    F.save(png, jpg)


def run(mid):
    m = R.get(mid)
    P = load_pos(mid)
    results, points = {}, {}
    for side in ('home', 'away'):
        results[m[side]['name']], points[side] = team_shape(P, mid, side)
    path = out_path('analysis', 'teams', f"{m['slug']}.json")
    draw(mid, results, points, path[:-5] + '.png', path[:-5] + '.jpg')
    json.dump(results, open(path, 'w'), indent=1, ensure_ascii=False)
    for name, r in results.items():
        print(f"{m['slug']} {name}: {centre_text(r['centre'])}, usual width {r['width']:.0f} m")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
