"""Shape over time: each team's block without the ball, minute by minute.

    python3 analysis/shape_time.py <match id> [...]

Definitions (live play only, the team without the ball, in its own frame with its own goal at -x):
- back line = mean x of the deepest four outfielders, as metres from the team's own goal line (+ L/2)
- length    = deepest to highest outfielder (x)
- width     = widest to widest outfielder (y)
Samples are the ~5 Hz positions from collect.py; the clock is the running match clock.

Lines: at every half minute, the median of every defending moment within 5 minutes either side (a pooled
10-minute window, cut at the period's edges). A point needs at least 40 s of defending (200 samples) in its
window, otherwise the line breaks. Each period has its own panel, stoppage time is shaded, goals (from the
registry) are dashed lines.

Header: median of every defending moment in the first 15 minutes vs the last 15 of normal time (from 75',
including second-half stoppage time). The JSON also carries whole-match medians for the comparison board.

Writes output/analysis/time/<slug>.png/.jpg/.json, keyed by team name.
"""
import json
import sys

import numpy as np
from matplotlib.transforms import blended_transform_factory

from common import R, L, OFF, PEND, defending, flips, load_pos, out_path, outfield_mask
from style import BG, DIM, FAINT, INK, MED, MONO, MONO_MED, SEMI, Fig

HALF_WINDOW = 5.0           # minutes either side of each point
STEP = 0.5                  # minutes between points
MIN_SAMPLES = 200           # ~40 s of defending in the window
FIRST, LAST = 15, 75        # header: first 15 minutes vs 75' to the end of the second half

MEASURES = [  # key, chart title, description
    ('line', 'BACK-LINE HEIGHT', 'deepest four outfielders, metres from own goal · higher = defending further up'),
    ('length', 'BLOCK LENGTH', 'deepest to highest outfielder, metres · lower = more compact'),
    ('width', 'BLOCK WIDTH', 'widest to widest outfielder, metres · lower = narrower'),
]
PERIOD_NAMES = {1: '1ST HALF', 2: '2ND HALF', 3: 'ET 1', 4: 'ET 2'}

# layout, inches from the top-left (the figure is 18 in wide)
LEFT, RIGHT, PANEL_GAP = 1.08, 16.92, 0.22
CHART_TOP, CHART_H, CHART_STEP = 4.00, 2.70, 3.65
SHADE = (*[int(INK[i:i + 2], 16) / 255 for i in (1, 3, 5)], 0.035)
GRID = (*SHADE[:3], 0.08)


def team_measures(P, mid, side):
    """Per-sample back line, length and width of `side`'s outfielders in its own frame."""
    jerseys = P[side]['jerseys']
    xy = P[side]['xy'][:, outfield_mask(mid, side, jerseys), :].astype(float)
    f = flips(mid, P['period'], side)[:, None]
    x, y = xy[..., 0] * f, xy[..., 1] * f
    n = np.sum(~np.isnan(x), 1)
    deepest = np.sort(np.where(np.isnan(x), np.inf, x), 1)[:, :4]
    with np.errstate(invalid='ignore'):
        line = np.where(n >= 4, deepest.mean(1) + L / 2, np.nan)
        ok = n >= 2
        length = np.where(ok, np.nanmax(np.where(ok[:, None], x, 0), 1) - np.nanmin(np.where(ok[:, None], x, 0), 1), np.nan)
        width = np.where(ok, np.nanmax(np.where(ok[:, None], y, 0), 1) - np.nanmin(np.where(ok[:, None], y, 0), 1), np.nan)
    return {'line': line, 'length': length, 'width': width}


def rolling(clock, values, start, end):
    """Points every STEP minutes from start to end: median of samples within HALF_WINDOW, NaN if too few."""
    grid = np.arange(start, end + 1e-9, STEP)
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        v = values[(clock >= g - HALF_WINDOW) & (clock <= g + HALF_WINDOW)]
        v = v[~np.isnan(v)]
        if len(v) >= MIN_SAMPLES:
            out[i] = np.median(v)
    return grid, out


def periods(P):
    """[(period, start minute, end minute)] from the samples' clock."""
    c, per = P['clock_min'], P['period']
    return [(int(p), float(OFF[int(p)]), float(np.nanmax(c[per == p]))) for p in np.unique(per)]


def nanmed(v):
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else None


def analyse(P, mid):
    """{side: dict(first15, last15, match, series)} for both teams."""
    D = np.array(defending(P, mid), dtype=object)
    c, per = P['clock_min'], P['period']
    valid = P['live'] & ~np.isnan(c)
    res = {}
    for side in ('home', 'away'):
        M = team_measures(P, mid, side)
        sel = valid & (D == side)
        first = sel & (per == 1) & (c < FIRST)
        last = sel & (per == 2) & (c >= LAST)
        series = {}
        for p, start, end in periods(P):
            s = sel & (per == p)
            for k in M:
                grid, line = rolling(c[s], M[k][s], start, end)
                series.setdefault(p, {'minute': grid})[k] = line
        res[side] = dict(
            first15={k: nanmed(M[k][first]) for k in M},
            last15={k: nanmed(M[k][last]) for k in M},
            match={'back_line': nanmed(M['line'][sel]), 'length': nanmed(M['length'][sel]), 'width': nanmed(M['width'][sel])},
            series=series)
    return res, periods(P)


# ---------------------------------------------------------------- drawing

def change_text(a, b):
    d = round(b) - round(a)
    return 'no change' if d == 0 else f'{d:+d}'


def header_row(r):
    parts = [f"{name} {round(r['first15'][k])} to {round(r['last15'][k])} m ({change_text(r['first15'][k], r['last15'][k])})"
             for k, name in (('line', 'back line'), ('length', 'length'), ('width', 'width'))]
    return 'first 15 min vs last 15 of normal time:   ' + '   ·   '.join(parts)


def goal_groups(mid):
    """Goals as (team, minute, kind), with same-team goals within 3 minutes grouped under one label."""
    groups = []
    for team, minute, kind in sorted(R.goals(mid), key=lambda g: g[1]):
        if groups and groups[-1][0] == team and minute - groups[-1][1][-1][0] <= 3:
            groups[-1][1].append((minute, kind))
        else:
            groups.append((team, [(minute, kind)]))
    return groups


def goal_period(minute, spans):
    """The period a goal minute falls in: the last one starting before it."""
    return [p for p, start, _ in spans if start < minute][-1]


def y_range(res, key):
    vals = np.concatenate([s[key] for r in res.values() for s in r['series'].values()])
    lo, hi = np.floor(np.nanmin(vals) / 5) * 5, np.ceil(np.nanmax(vals) / 5) * 5
    return lo, hi


def spread_labels(ys, gap):
    """Push label positions apart (sorted order kept) so neighbours are at least `gap` apart."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = (np.mean(np.array(ys, float)[order]) - np.mean(pos))
    out = np.empty_like(pos)
    out[order] = pos + shift
    return out


def draw_chart(fig, top, key, title, desc, res, spans, mid, colors, names, label_goals):
    m = R.get(mid)
    fig.text(LEFT, top - 0.27, title, fontproperties=MONO_MED, fontsize=13, color=INK)
    fig.text(3.60, top - 0.27, desc, fontproperties=MONO, fontsize=11, color=DIM)
    lo, hi = y_range(res, key)
    total = sum(e - s for _, s, e in spans)
    width = RIGHT - LEFT - PANEL_GAP * (len(spans) - 1)
    x0 = LEFT
    goals = goal_groups(mid)
    for i, (p, start, end) in enumerate(spans):
        w = width * (end - start) / total
        ax = fig.axes(x0, top, w, CHART_H)
        x0 += w + PANEL_GAP
        ax.set_xlim(start, end)
        ax.set_ylim(lo - 2, hi + 2)
        ax.axis('off')
        ticks = np.arange(lo, hi + 0.1, 5)
        for t in ticks:
            ax.axhline(t, color=GRID, lw=0.7, zorder=1)
        if end > PEND[p]:
            ax.axvspan(PEND[p], end, color=SHADE, lw=0, zorder=0)
        for side in ('home', 'away'):
            s = res[side]['series'][p]
            ax.plot(s['minute'], s[key], color=colors[side], lw=1.8, zorder=4, solid_capstyle='round')
        tb = blended_transform_factory(ax.transData, ax.transAxes)
        step = 15 if p <= 2 else 5
        for t in np.arange(np.ceil(start / step) * step, end + 1e-9, step):
            if t == start and p > 1:
                continue
            ax.text(t, -0.04, f"{t:.0f}'", transform=tb, ha='center', va='center', fontproperties=MONO, fontsize=9.5, color=FAINT)
        if i == 0:
            for t in ticks:
                ax.text(-0.07 / w, t, f'{t:.0f}', transform=blended_transform_factory(ax.transAxes, ax.transData),
                        ha='right', va='center', fontproperties=MONO, fontsize=10.5, color=FAINT)
        ax.annotate(PERIOD_NAMES[p], (0, 0), xycoords='axes fraction', xytext=(5.5, 6.5), textcoords='offset points',
                    ha='left', va='center', fontproperties=MONO, fontsize=10, color=FAINT)
        for team, items in goals:
            col = R.team(mid, team)['color']
            here = [(mi, kind) for mi, kind in items if goal_period(mi, spans) == p]
            for mi, _ in here:
                ax.axvline(mi - 0.5, color=col, lw=1.2, ls=(0, (2.5, 2.5)), alpha=0.85, zorder=3)
            if label_goals and here:
                label = R.code(mid, team) + ' ' + ', '.join(f"{mi}'" + (f' {kind}' if kind else '') for mi, kind in here)
                ax.text(here[0][0] - 0.5 + 0.08 / w * (end - start), 0.98, label, transform=tb,
                        ha='left', va='center', fontproperties=MONO, fontsize=9.5, color=col, zorder=5,
                        bbox=dict(facecolor=BG, edgecolor='none', pad=3))
    # team names at the end of the last panel
    last = spans[-1][0]
    ends = []
    for side in ('home', 'away'):
        v = res[side]['series'][last][key]
        ok = ~np.isnan(v)
        ends.append(v[ok][-1] if ok.any() else np.nan)
    gap = 0.25 * (hi - lo + 4) / CHART_H
    ys = spread_labels(ends, gap)
    for side, yv in zip(('home', 'away'), ys):
        ax.text(1.006, yv, names[side], transform=blended_transform_factory(ax.transAxes, ax.transData),
                ha='left', va='center', fontproperties=SEMI, fontsize=13.5, color=colors[side], clip_on=False)


def draw(res, spans, mid, png, jpg):
    m = R.get(mid)
    fig = Fig(15.65)
    fig.header('Shape over time', [f"{m['title']} · {m['sub']}",
                                   "Each team's block without the ball, minute by minute (10-minute rolling window) · dashed lines: goals"],
               x=LEFT, top=0.55, size=12, step=0.25, title_size=46, gap=0.57)
    colors = {s: m[s]['color'] for s in ('home', 'away')}
    names = {s: m[s]['name'] for s in ('home', 'away')}
    for i, side in enumerate(('home', 'away')):
        y = 2.29 + 0.62 * i
        fig.team_chip(1.15, y - 0.02, names[side], colors[side], size=21.5, dy=0.055)
        fig.text(3.60, y, header_row(res[side]), fontproperties=MONO, fontsize=12, color=DIM)
    for j, (key, title, desc) in enumerate(MEASURES):
        draw_chart(fig, CHART_TOP + j * CHART_STEP, key, title, desc, res, spans, mid, colors, names, j == 0)
    fig.footnotes(['Only live play without the ball: stoppages, set-piece set-ups and celebrations removed. Shaded strip: stoppage time.',
                   'Lines: median of every defending moment in a 10-minute window, ~5 samples a second.',
                   R.credit(mid, 'Source: PFF FC tracking behind the 2-D animations; possession and ball-in-play from the event feed; '
                                 'goal minutes from the match record.')],
                  14.86, color=FAINT, x=LEFT)
    fig.save(png, jpg)


def rounded(d):
    return {k: None if v is None else round(v, 1) for k, v in d.items()}


def to_json(res, mid):
    m = R.get(mid)
    out = {}
    for side in ('home', 'away'):
        r = res[side]
        out[m[side]['name']] = dict(
            first15=rounded(r['first15']), last15=rounded(r['last15']), match=rounded(r['match']),
            series={PERIOD_NAMES[p]: {k: [None if np.isnan(v) else round(float(v), 1) for v in s[k]] if k != 'minute'
                                      else [round(float(v), 2) for v in s[k]] for k in s}
                    for p, s in r['series'].items()})
    return out


def run(mid):
    m = R.get(mid)
    res, spans = analyse(load_pos(mid), mid)
    base = out_path('analysis', 'time', m['slug'])
    draw(res, spans, mid, base + '.png', base + '.jpg')
    data = to_json(res, mid)
    json.dump(data, open(base + '.json', 'w'), ensure_ascii=False, indent=1)
    for name, r in data.items():
        print(f"{mid} {name}: first 15 {r['first15']} · last 15 {r['last15']} · match {r['match']}")


if __name__ == '__main__':
    for a in sys.argv[1:]:
        run(a)
