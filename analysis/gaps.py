"""Where the gaps open: each team's defensive block seen from behind its own back line.

    python3 analysis/gaps.py <match id> [...]

Definitions
- A defending moment is a 5 Hz sample in live play when the team does not have the ball.
- Every moment is lined up on the team's back line (the mean x of its deepest four outfielders, in the team's
  own frame) and viewed from behind the defence: the team's left is on the left and play goes up.
- The block is the convex hull of the team's outfielders. A hole is a 1 m cell inside it with no defending
  outfielder within 8 m; holes (m²) is the average hole area per moment.
- A free opponent is an opposing outfielder inside the block with no defending outfielder within 5 m;
  free is the average number per moment.
- Lanes by y in the team's own frame (its left = +y): R wing < -20.16 < R half-space < -9.16 < centre < 9.16
  < L half-space < 20.16 < L wing. Depth bands are 8 m deep, measured from the back line.
- The most frequent hole is the lane and depth band holding the most hole area across the match; opponents
  free most often is the zone holding the most free-opponent moments.
- Drawn on both panels: starters' median spots relative to the back line, and the block's usual outline
  (cells inside the hull in at least half of the moments).
- Left panel: the share of moments each cell was a hole, on one amber scale for both teams (0 to the highest
  cell share, capped at 30%). Right panel: where free opponents stood, on that panel's own scale.

The per-moment work is cached in output/analysis/gaps_<id>.pkl (rebuilt when pos_<id>.pkl is newer).
Writes output/analysis/gaps/<slug>.png/.jpg/.json; the JSON is keyed by the defending team's name.
"""
import json
import os
import pickle
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from common import R, defending, flips, load_pos, out_path, outfield_mask, roster
from style import AMBER, BG, DIM, FAINT, INK, MONO, MONO_MED, PITCH, TITLE, Fig

HOLE_R, FREE_R = 8.0, 5.0              # m: hole = no defender within HOLE_R; free = no defender within FREE_R
OUTLINE_SHARE = 0.5                    # block outline: inside the hull in at least this share of moments
BAND = 8                               # m: depth bands ahead of the back line
LANE_EDGES = (-20.16, -9.16, 9.16, 20.16)
LANE_NAMES = ('R wing', 'R half-space', 'centre', 'L half-space', 'L wing')   # from -y to +y
VMAX_CAP = 0.30                        # hole scale tops out at the highest cell share, capped at 30%
MIN_PLAYERS = 3                        # outfielders needed to form a block

X0, X1 = -20, 70                       # grid extent, m relative to the back line
Y0, Y1 = -34, 34
XC = np.arange(X0, X1) + 0.5           # 1 m cell centres
YC = np.arange(Y0, Y1) + 0.5
CELLS = np.stack(np.meshgrid(XC, YC, indexing='ij'), -1).reshape(-1, 2)

VIEW_X = (-10, 38)                     # drawn depth range
SIGMA_HOLE, SIGMA_FREE = 1.0, 2.0      # m: display smoothing of the two maps
GAMMA = 1.3                            # opacity = (share / top of scale) ** GAMMA


# ---------------------------------------------------------------- per-moment geometry

def inside_hull(pts, hull):
    """Boolean per point: inside (or on) the convex hull."""
    return (pts @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9).all(1)


def moment(d, o):
    """One defending moment. d: defending outfielders, o: opposing outfielders, both in the defending team's
    frame (NaN rows dropped). Returns (back-line x, hole mask over CELLS, inside mask, free opponents rel.)."""
    bx = np.sort(d[:, 0])[:4].mean()
    d = d - (bx, 0)
    o = o - (bx, 0)
    hull = ConvexHull(d)
    ins = inside_hull(CELLS, hull)
    hole = np.zeros(len(CELLS), bool)
    k = np.flatnonzero(ins)
    hole[k[cdist(CELLS[k], d).min(1) > HOLE_R]] = True
    oi = o[inside_hull(o, hull)]
    free = oi[cdist(oi, d).min(1) > FREE_R] if len(oi) else oi
    return bx, hole, ins, free


def cell_index(p):
    """Flat CELLS index for points p (n, 2), -1 outside the grid."""
    ix = np.floor(p[:, 0] - X0).astype(int)
    iy = np.floor(p[:, 1] - Y0).astype(int)
    ok = (ix >= 0) & (ix < len(XC)) & (iy >= 0) & (iy < len(YC))
    return np.where(ok, ix * len(YC) + iy, -1)


def compute_side(mid, P, side, dfd):
    """Accumulate every live defending moment of `side`."""
    opp = 'away' if side == 'home' else 'home'
    idx = np.array([i for i, (lv, d) in enumerate(zip(P['live'], dfd)) if lv and d == side])
    fl = flips(mid, P['period'][idx], side)[:, None, None]
    jd = np.array(P[side]['jerseys'])
    om = outfield_mask(mid, side, jd)
    D = P[side]['xy'][idx][:, om] * fl
    O = P[opp]['xy'][idx][:, outfield_mask(mid, opp, P[opp]['jerseys'])] * fl
    ro = roster(mid)[side]
    rel = {j: [] for j in jd[om] if ro[j]['started']}
    hole = np.zeros(len(CELLS)); inside = np.zeros(len(CELLS)); free_map = np.zeros(len(CELLS))
    area, free_n = [], []
    for k in range(len(idx)):
        on = ~np.isnan(D[k, :, 0])
        if on.sum() < MIN_PLAYERS:
            continue
        o = O[k][~np.isnan(O[k, :, 0])]
        bx, h, ins, free = moment(D[k][on], o)
        hole += h; inside += ins
        c = cell_index(free); np.add.at(free_map, c[c >= 0], 1)
        area.append(int(h.sum())); free_n.append(len(free))
        for j, p in zip(jd[om][on], D[k][on]):
            if j in rel:
                rel[j].append((p[0] - bx, p[1]))
    return dict(n=len(area), hole=hole, inside=inside, free_map=free_map, area=np.array(area), free=np.array(free_n),
                starters={j: np.median(np.array(v), 0).tolist() for j, v in rel.items() if v})


def compute(mid):
    """{side: accumulated grids}, from the cache when it is newer than the position file."""
    cache, src = out_path('analysis', f'gaps_{mid}.pkl'), out_path('analysis', f'pos_{mid}.pkl')
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(src):
        return pickle.load(open(cache, 'rb'))
    P = load_pos(mid)
    dfd = defending(P, mid)
    res = {side: compute_side(mid, P, side, dfd) for side in ('home', 'away')}
    pickle.dump(res, open(cache, 'wb'))
    return res


# ---------------------------------------------------------------- summary numbers

def zone_name(lane, band):
    """'centre, 8-16 m ahead of the back line' from a lane index and a depth-band index."""
    a, b = band * BAND, (band + 1) * BAND
    depth = f'{a}-{b} m ahead of' if a >= 0 else f'{-b}-{-a} m behind'
    return f'{LANE_NAMES[lane]}, {depth} the back line'


def busiest_zone(grid):
    """Zone (lane x depth band) holding the largest total of a per-cell grid."""
    lanes = np.searchsorted(LANE_EDGES, CELLS[:, 1])
    bands = np.floor(CELLS[:, 0] / BAND).astype(int) - X0 // BAND        # shifted to start at 0
    tot = np.zeros((len(LANE_NAMES), bands.max() + 1))
    np.add.at(tot, (lanes, bands), grid)
    lane, band = np.unravel_index(np.argmax(tot), tot.shape)
    return zone_name(int(lane), int(band) + X0 // BAND)


def summarise(acc):
    return dict(holes=int(round(acc['area'].mean())), free=round(float(acc['free'].mean()), 2),
                biggest_pocket=busiest_zone(acc['hole']), opponents_free_most=busiest_zone(acc['free_map']),
                starters={j: [round(v, 1) for v in p] for j, p in acc['starters'].items()},
                moments=int(acc['n']))


# ---------------------------------------------------------------- drawing

PANEL_W, PANEL_H = 6.8, 4.8            # in: 10 px per metre
PANEL_X = (1.46, 9.74)                 # left edges of the two panels
BLOCK_TOP, BLOCK_STEP = 2.31, 7.80     # team label of the first block, distance between blocks
LEGEND_Y, HEIGHT = 17.83, 18.95
AMBER_CMAP = LinearSegmentedColormap.from_list('hole', ['#fffaf3', AMBER])
LINE_C = (*to_rgb(INK), 0.45)
SOFT = '#a3b4ab'                       # starters' rings on the free-opponent panel


def as_grid(v):
    return v.reshape(len(XC), len(YC))


def glow(ax, share, color, vmax, sigma):
    """A smoothed per-cell share drawn as `color`, fully opaque at vmax."""
    sm = gaussian_filter(as_grid(share), sigma)
    rgba = np.zeros(sm.shape + (4,))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.clip(sm / vmax, 0, 1) ** GAMMA
    ax.imshow(rgba, extent=(Y0, Y1, X0, X1), origin='lower', interpolation='bilinear', zorder=1, aspect='auto')


def panel(fig, x_in, top_in, acc, starters_big):
    """Pitch-coloured panel in the back-line frame, with lanes, back line, outline and starters' spots."""
    ax = fig.axes(x_in, top_in, PANEL_W, PANEL_H)
    ax.set_xlim(Y1, Y0)                                  # team's left (+y) on the left
    ax.set_ylim(*VIEW_X)
    ax.axis('off')
    ax.add_patch(Rectangle((Y0, VIEW_X[0]), Y1 - Y0, VIEW_X[1] - VIEW_X[0], facecolor=PITCH, edgecolor='none', zorder=0))
    for e in LANE_EDGES:
        ax.plot([e, e], VIEW_X, color=INK, alpha=0.14, lw=0.8, ls=(0, (1.5, 2.5)), zorder=2)
    for y in (Y0, Y1):
        ax.plot([y, y], VIEW_X, color=LINE_C, lw=1.0, zorder=4)
    ax.plot([Y0, Y1], [0, 0], color=LINE_C, lw=1.6, zorder=4)
    for x, s, alpha, va in ((0.9, 'BACK LINE', 0.6, 'bottom'), (VIEW_X[1] - 0.8, 'opponents’ goal this way', 0.3, 'top'),
                            (VIEW_X[0] + 0.8, 'own goal this way', 0.3, 'bottom')):
        ax.text(Y0 + 0.9, x, s, color=INK, alpha=alpha, fontproperties=MONO, fontsize=9.5, ha='right', va=va, zorder=5)
    share = gaussian_filter(as_grid(acc['inside'] / acc['n']), 1.0)
    ax.contour(YC, XC, share, levels=[OUTLINE_SHARE], colors=[INK], alpha=0.7, linewidths=1.2, linestyles=[(0, (4, 3))], zorder=5)
    r, lw, fs, ec, fc = (1.3, 1.8, 13, INK, BG) if starters_big else (0.95, 1.4, 9.5, SOFT, (*to_rgb(BG), 0.85))
    for j, (x, y) in acc['starters'].items():
        ax.add_patch(Circle((y, x), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=6))
        ax.text(y, x - 0.05, j, color=ec, fontproperties=TITLE, fontsize=fs, ha='center', va='center', zorder=7)
    return ax


def axis_labels(fig, x_in, top_in):
    """Depth ticks on the left and lane names underneath a panel."""
    for d in (10, 20, 30):
        fig.text(x_in - 0.09, top_in + (VIEW_X[1] - d) / 10, f'{d} m', fontproperties=MONO, fontsize=9, color=FAINT, ha='right')
    edges = (Y1,) + LANE_EDGES[::-1] + (Y0,)
    for k, name in enumerate(LANE_NAMES[::-1]):
        mid_y = (edges[k] + edges[k + 1]) / 2
        fig.text(x_in + (Y1 - mid_y) / 10, top_in + PANEL_H + 0.19, name, fontproperties=MONO, fontsize=9.5, color=FAINT, ha='center')


def team_block(fig, mid, side, acc, s, top, vmax):
    m = R.get(mid)
    team, opp = m[side], m['away' if side == 'home' else 'home']
    fig.team_label(1.17, top - 0.06, f"{team['name']} defending", team['color'], size=28, dot=0.095, gap=0.23, dot_dy=0.065)
    fig.text(1.08, top + 0.63, f"Open holes inside the block: {s['holes']} m² on average  ·  {s['free']:.1f} {opp['name']} players "
                               f"free inside it at any moment  ·  most frequent hole: {s['biggest_pocket']}",
             fontproperties=MONO, fontsize=12, color=DIM)
    for x, title, r in ((PANEL_X[0], 'HOLES IN THE BLOCK', HOLE_R), (PANEL_X[1], f"{opp['name'].upper()} PLAYERS FREE INSIDE IT", FREE_R)):
        fig.text(x - 0.2, top + 1.23, title, fontproperties=MONO_MED, fontsize=12.5)
        fig.text(x + PANEL_W + 0.2, top + 1.23, f"no {team['name']} outfielder within {r:g} m",
                 fontproperties=MONO, fontsize=11, color=DIM, ha='right')
    ptop = top + 1.54
    ax = panel(fig, PANEL_X[0], ptop, acc, True)
    glow(ax, acc['hole'] / acc['n'], AMBER, vmax, SIGMA_HOLE)
    fm = acc['free_map'] / acc['n']
    ax = panel(fig, PANEL_X[1], ptop, acc, False)
    glow(ax, fm, opp['color'], gaussian_filter(as_grid(fm), SIGMA_FREE).max(), SIGMA_FREE)
    for x in PANEL_X:
        axis_labels(fig, x, ptop)


def legend(fig, mid, vmax):
    ax = fig.axes(1.08, LEGEND_Y - 0.07, 2.52, 0.14)
    ax.imshow(np.linspace(0, 1, 256)[None], cmap=AMBER_CMAP, aspect='auto', extent=(0, 1, 0, 1))
    ax.axis('off')
    fig.text(3.69, LEGEND_Y, f'hole open 0 to {int(vmax * 100)}% of defending moments (same scale for both teams)  ·  '
                             'circles: starters’ usual spots  ·  dashed: the block’s usual outline',
             fontproperties=MONO, fontsize=10.5, color=FAINT)
    notes = ['Every moment is lined up on that team’s back line (the deepest four outfielders). Block = the shape around the ten '
             'outfielders. Hole = a spot inside it more than 8 m',
             'from every defender. Free = an opponent inside it with no defender within 5 m. Live play only, while defending, '
             '~5 samples a second. ' + R.credit(mid, 'Source: PFF FC tracking + event feed.')]
    for i, s in enumerate(notes):
        fig.text(1.08, LEGEND_Y + 0.49 + 0.27 * i, s, fontproperties=MONO, fontsize=10.5, color=FAINT)


def draw(mid, res, summ):
    m = R.get(mid)
    vmax = min(VMAX_CAP, max((res[s]['hole'] / res[s]['n']).max() for s in res))
    fig = Fig(HEIGHT)
    fig.header('Where the gaps open', [f"{m['title']} · {m['sub']}",
                                       'Each team\'s block seen from behind its own back line: the holes that keep opening, '
                                       'and where opponents stood free inside it'], x=1.08, size=12, step=0.25, title_size=46, gap=0.57)
    for k, side in enumerate(('home', 'away')):
        team_block(fig, mid, side, res[side], summ[side], BLOCK_TOP + k * BLOCK_STEP, vmax)
    legend(fig, mid, vmax)
    fig.save(out_path('analysis', 'gaps', f"{m['slug']}.png"), out_path('analysis', 'gaps', f"{m['slug']}.jpg"))


def run(mid):
    res = compute(mid)
    summ = {side: summarise(res[side]) for side in res}
    draw(mid, res, summ)
    m = R.get(mid)
    data = {m[side]['name']: summ[side] for side in ('home', 'away')}
    json.dump(data, open(out_path('analysis', 'gaps', f"{m['slug']}.json"), 'w'), ensure_ascii=False, indent=1)
    for name, s in data.items():
        print(mid, f"{name} defending: holes {s['holes']} m², free {s['free']}, pocket {s['biggest_pocket']}, "
                   f"opponents free most {s['opponents_free_most']}")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)
