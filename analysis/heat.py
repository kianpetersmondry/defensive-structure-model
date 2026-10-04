"""Drawing helpers shared by the heatmap-style views (attack lanes, team / player heatmaps, in & out of
possession): team-frame coordinates, the glow layer, pixel-placed pitches and labels, starter markers.

Layouts in these views are given in pixels of the 1800 px-wide image (1 px = 0.01 in at 100 dpi).
"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse
from scipy.ndimage import gaussian_filter

from common import L, W, flips, roster
from style import BG, INK, DIM, MONO, SEMI, WIDTH_IN, pitch

PX = 0.01                       # inches per pixel


def own_frame(P, mid, side):
    """(x, y) arrays (samples x jerseys) of one side's players in its own frame: own goal at -x, attacking +x."""
    f = flips(mid, P['period'], side)[:, None]
    xy = P[side]['xy']
    return xy[..., 0] * f, xy[..., 1] * f


def outfield(mid, side, jerseys):
    """Boolean per jersey: True for outfield players (not a goalkeeper by roster position)."""
    ro = roster(mid)[side]
    return np.array([not ro.get(j, {}).get('gk', False) for j in jerseys])


def starters(mid, side, jerseys):
    ro = roster(mid)[side]
    return [j for j in jerseys if ro.get(j, {}).get('started')]


def mmss(seconds):
    s = int(round(seconds))
    return f'{s // 60}:{s % 60:02d}'


def pitch_ax(F, left, top, width, lw=1.0, color=None, alpha=None, **kw):
    """A full pitch whose grass spans exactly `width` px from (left, top) px, drawn without the penalty arcs."""
    height = width * W / L
    ax = F.axes(left * PX, top * PX, width * PX, height * PX)
    args = dict(lw=lw, arcs=False, **kw)
    if color is not None:
        args['color'] = color
    pitch(ax, **args)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-W / 2, W / 2)
    for a in ax.patches + ax.lines:
        a.set_clip_on(False)
        if alpha is not None and a.get_zorder() >= 3:
            a.set_alpha(alpha)
    return ax


def glow(ax, x, y, color, sigma=3.0, res=0.5, pct=99.5, gamma=1.2, alpha=0.9):
    """Density of points (x, y) in metres as a glow of `color` over the grass.

    Points are binned on a `res`-metre grid, smoothed with a Gaussian of `sigma` metres and scaled so the
    `pct` percentile of the smoothed density reaches full strength (`alpha` opacity); `gamma` > 1 fades the
    thin edges."""
    ok = ~(np.isnan(x) | np.isnan(y))
    nx, ny = int(L / res), int(W / res)
    H, _, _ = np.histogram2d(np.clip(x[ok], -L / 2, L / 2 - 1e-6), np.clip(y[ok], -W / 2, W / 2 - 1e-6),
                             bins=(nx, ny), range=[[-L / 2, L / 2], [-W / 2, W / 2]])
    D = gaussian_filter(H.T, sigma / res, mode='constant')
    top = np.percentile(D, pct)
    a = np.clip(D / top, 0, 1) ** gamma if top > 0 else np.zeros_like(D)
    rgba = np.zeros(D.shape + (4,))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = alpha * a
    ax.imshow(rgba, extent=(-L / 2, L / 2, -W / 2, W / 2), origin='lower', interpolation='bicubic', zorder=1)


def team_label(F, x, base, name, color, size=28, dot=10, gap=35, dot_dy=-12, right=None, right_size=12, right_x=1690,
               right_dy=0):
    """Colour dot plus the team name in caps; `base` is the text baseline (px from top). `dot` is the dot's
    radius in px, or (rx, ry) for an upright oval. Optional grey text right-aligned at `right_x`, its baseline
    `right_dy` px below the name's."""
    rx, ry = dot if isinstance(dot, tuple) else (dot, dot)
    cx, cy = (x + rx) / WIDTH_IN * PX, F.Y((base + dot_dy) * PX)
    F.fig.patches.append(Ellipse((cx, cy), 2 * rx * PX / WIDTH_IN, 2 * ry * PX / F.H, transform=F.fig.transFigure,
                                 facecolor=color, edgecolor='none'))
    F.text((x + gap) * PX, base * PX, name.upper(), fontproperties=SEMI, fontsize=size, va='baseline')
    if right:
        F.text(right_x * PX, (base + right_dy) * PX, right, fontproperties=MONO, fontsize=right_size, color=DIM,
               ha='right', va='baseline')


def marker(ax, x, y, label, r=1.2, size=13, edge=INK, text=INK, lw=1.8, alpha=1.0):
    """Numbered circle (a starter's position) at (x, y) metres."""
    ax.add_patch(Circle((x, y), r, facecolor=BG, edgecolor=edge, lw=lw, alpha=alpha, zorder=6))
    ax.text(x, y - 0.05 * r, label, ha='center', va='center', color=text, fontproperties=SEMI, fontsize=size,
            alpha=alpha, zorder=7)
