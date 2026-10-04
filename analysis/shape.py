"""In and out of possession: how each team's shape changes when it loses the ball, from the 5 Hz positions.

Definitions (each team in its own frame attacking left to right; every 5 Hz sample, every period, live or
dead ball: that reproduces the published figures, live play alone does not):
- Phase: with the ball when `P['poss']` is the team, without it when it is the opponent (no team: left out).
  "% of the match" is the phase's share of the samples with a team in possession.
- Per moment, from the outfield players on the pitch (keepers left out; moments with fewer than 8 skipped):
  centre = mean x (m from halfway, + = opponents' half), back line = mean distance from own goal of the
  deepest four, width = max - min y, length = max - min x. Each is the median over the phase's moments.
- Circles: starters' median positions in the phase. Glow: the phase's outfield positions.
- Sentence: change from with to without the ball in whole metres; under 1 m counts as no change.

    python3 analysis/shape.py 3823 [3821 ...]
"""
import json
import sys

import numpy as np

from common import L, W, R, load_pos, out_path, team_ids
from heat import PX, glow, marker, outfield, own_frame, pitch_ax, starters, team_label
from style import DIM, FAINT, INK, MONO, MONO_MED, TITLE, Fig

MIN_OUTFIELD = 8
METRICS = ('centre', 'back_line', 'width', 'length')
WORDS = {'back_line': ('back line', 'steps up', 'drops', 'holds its height'),
         'width': ('block', 'widens', 'narrows', 'keeps its width'),
         'length': (None, 'stretches', 'shortens', 'keeps its length')}

# layout (px)
H_PX, BLOCK, PITCH_TOP, PITCH_LEFTS, PITCH_W = 1757, 706, 337, (115, 925), 761
FOOTNOTES = ['Glow: outfield players’ positions in that phase (keepers left out). Circles: starters’ median positions. '
             'Dashed line: team centre.',
             'Small dots + lines on the right: where each starter sat with the ball. Centre: metres from halfway '
             '(+ = opponents’ half). Back line: deepest four outfielders, metres from own goal.',
             'Width / length: outfield spread at each moment, median. Source: PFF FC tracking, ~5 samples a second, '
             'possession from the event feed.']


def moment_metrics(X, Y):
    """Per-sample centre, back line, width and length of the outfield block (NaN when too few players)."""
    n = (~np.isnan(X)).sum(1)
    with np.errstate(all='ignore'):
        deepest = np.sort(X, axis=1)[:, :4]          # NaNs sort last
        m = dict(centre=np.nanmean(X, 1), back_line=deepest.mean(1) + L / 2,
                 width=np.nanmax(Y, 1) - np.nanmin(Y, 1), length=np.nanmax(X, 1) - np.nanmin(X, 1))
    for v in m.values():
        v[n < MIN_OUTFIELD] = np.nan
    return m


def change(key, d):
    noun, up, down, same = WORDS[key]
    verb = same if abs(d) < 1 else f'{up if d > 0 else down} {abs(d):.0f} m'
    return f'{noun} {verb}' if noun else verb


def sentence(w, wo):
    d = {k: wo[k] - w[k] for k in ('back_line', 'width', 'length')}
    return (f"Without the ball: {change('back_line', d['back_line'])}, {change('width', d['width'])} and "
            f"{change('length', d['length'])}")


def team_phases(P, mid, side, tid):
    J = P[side]['jerseys']
    X, Y = own_frame(P, mid, side)
    of = outfield(mid, side, J)
    mm = moment_metrics(X[:, of], Y[:, of])
    poss = np.array([p or '' for p in P['poss']])
    phases = {'with': poss == tid, 'without': (poss != tid) & (poss != '')}
    out, draw = {}, {}
    for ph, mask in phases.items():
        out[ph] = {k: float(np.nanmedian(mm[k][mask])) for k in METRICS}
        out[ph]['share'] = float(mask.sum() / (poss != '').sum() * 100)
        spots = {j: (float(np.nanmedian(X[mask, J.index(j)])), float(np.nanmedian(Y[mask, J.index(j)])))
                 for j in starters(mid, side, J)}
        xo, yo = X[mask][:, of].ravel(), Y[mask][:, of].ravel()
        draw[ph] = dict(spots=spots, pts=(xo, yo))
    keepers = {j for j in starters(mid, side, J) if not of[J.index(j)]}
    return out, draw, keepers


def stats_line(s):
    return (f"centre {s['centre']:+.0f} m  ·  back line {s['back_line']:.0f} m  ·  width {s['width']:.0f} m  ·  "
            f"length {s['length']:.0f} m")


def draw_phase(F, left, top, color, stats, d, keepers, ref=None):
    ax = pitch_ax(F, left, top, PITCH_W, lw=0.7)
    glow(ax, *d['pts'], color)
    ax.plot([stats['centre']] * 2, [-W / 2, W / 2], color=INK, lw=1.2, ls=(0, (4, 3)), alpha=0.6, zorder=4)
    for j, (x, y) in d['spots'].items():
        if ref:
            rx, ry = ref[j]
            ax.plot([rx, x], [ry, y], color=INK, lw=0.8, alpha=0.45, zorder=5)
            ax.scatter([rx], [ry], s=9, color=INK, alpha=0.55, linewidths=0, zorder=5)
        gk = j in keepers
        marker(ax, x, y, j, r=2.0, size=11.5, edge=DIM if gk else INK, text=DIM if gk else INK, alpha=0.85 if gk else 1)


def draw(mid, results, drawing, keepers, png, jpg):
    m = R.get(mid)
    F = Fig(H_PX * PX)
    F.text(109 * PX, 79 * PX, 'IN & OUT OF POSSESSION', fontproperties=TITLE, fontsize=46, va='baseline')
    for base, s in ((117, f"{m['title']} · {m['sub']}"),
                    (142, 'How each team\'s shape changes when it loses the ball · each team attacking left to right')):
        F.text(109 * PX, base * PX, s, fontproperties=MONO, fontsize=12, color=DIM, va='baseline')
    for i, side in enumerate(('home', 'away')):
        team, y0 = m[side], BLOCK * i
        res = results[team['name']]
        team_label(F, 108, 239 + y0, team['name'], team['color'], size=28, dot=10, gap=33, dot_dy=-8,
                   right=res['line'] + '.', right_dy=-5, right_x=1692)
        for left, ph, title in zip(PITCH_LEFTS, ('with', 'without'), ('WITH THE BALL', 'WITHOUT THE BALL')):
            s = res[ph]
            F.text((left - 7) * PX, (309 + y0) * PX, title, fontproperties=MONO_MED, fontsize=12.4, va='baseline')
            F.text((left + 767) * PX, (308 + y0) * PX, f"{s['share']:.0f}% of the match", fontproperties=MONO,
                   fontsize=11, color=DIM, ha='right', va='baseline')
            ref = drawing[side]['with']['spots'] if ph == 'without' else None
            draw_phase(F, left, PITCH_TOP + y0, team['color'], s, drawing[side][ph], keepers[side], ref)
            F.text((left - 6) * PX, (867 + y0) * PX, stats_line(s), fontproperties=MONO, fontsize=10.9, color=DIM,
                   va='baseline')
    for k, s in enumerate(FOOTNOTES):
        F.text(109 * PX, (1672 + 28 * k) * PX, R.credit(mid, s), fontproperties=MONO, fontsize=10.5, color=FAINT,
               va='baseline')
    F.save(png, jpg)


def run(mid):
    m = R.get(mid)
    P = load_pos(mid)
    results, drawing, keepers = {}, {}, {}
    for side, tid in zip(('home', 'away'), team_ids(mid)):
        res, drawing[side], keepers[side] = team_phases(P, mid, side, tid)
        res['line'] = sentence(res['with'], res['without'])
        results[m[side]['name']] = res
    path = out_path('analysis', 'shape', f"{m['slug']}.json")
    draw(mid, results, drawing, keepers, path[:-5] + '.png', path[:-5] + '.jpg')
    out = {name: {**{ph: {k: round(v, 2) for k, v in r[ph].items()} for ph in ('with', 'without')},
                  'line': r['line'],
                  'starters': {ph: {j: [round(v, 1) for v in xy] for j, xy in drawing[side][ph]['spots'].items()}
                               for ph in ('with', 'without')}}
           for (name, r), side in zip(results.items(), ('home', 'away'))}
    json.dump(out, open(path, 'w'), indent=1, ensure_ascii=False)
    for name, r in results.items():
        print(f"{m['slug']} {name}: {r['line']}")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
